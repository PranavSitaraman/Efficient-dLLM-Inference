"""
Frozen LLaDA base-model wrapper.

Wraps a HuggingFace LLaDA model and exposes:
  - forward(): full logits over vocabulary at every position.
  - forward_with_hidden(): logits + last-layer hidden states (for PRISM).
  - get_embedding_weight(): token embedding matrix E (for soft-masked state).

Supports four backends selected via config ``base_model.backend``:
  - ``hf``       — HuggingFace AutoModelForCausalLM (default for dense LLaDA models).
  - ``dkv``      — dKV-Cache patched model (external/dKV-Cache).
  - ``dinfer``   — dInfer framework (external/dInfer; for LLaDA2.X MoE).
  - ``soft_moe`` — dInfer MoE with hard-top-k-preserving tail-expanded routing
    (paper §3.7).
  - ``auto``     — auto-detect from model name.

All backends fail hard on missing dependencies (no silent fallbacks).
"""

import gc
import inspect
import os
import sys
import torch
import torch.nn as nn
from typing import Any, List, Optional, Tuple
from transformers import AutoTokenizer
from ..runtime_checks import ensure_vllm_moe_runtime, is_global_rank_zero


_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DKV_DIR = os.path.join(_CODE_DIR, "external", "dKV-Cache")
_DINFER_DIR = os.path.join(_CODE_DIR, "external", "dInfer", "python")

if os.path.isdir(_DKV_DIR) and _DKV_DIR not in sys.path:
    sys.path.insert(0, _DKV_DIR)
if os.path.isdir(_DINFER_DIR) and _DINFER_DIR not in sys.path:
    sys.path.insert(0, _DINFER_DIR)


def _get_dinfer_runtime(cfg: dict) -> str:
    runtime = str(cfg.get("base_model", {}).get("dinfer_runtime", "vllm")).strip().lower()
    if runtime not in {"vllm", "sglang"}:
        raise ValueError(
            f"Unsupported base_model.dinfer_runtime={runtime!r}. "
            "Choose 'vllm' or 'sglang'."
        )
    return runtime


def _detect_backend(name_or_path: str, cfg: dict) -> str:
    """Choose backend based on model name and config."""
    backend = cfg["base_model"].get("backend", "auto")
    if backend != "auto":
        return backend
    lower = name_or_path.lower()
    if any(tag in lower for tag in ("llada2.0", "llada2.1", "llada2")):
        return "dinfer"
    if "llada" in lower:
        return "hf"
    return "hf"


def _patch_vllm_uuid_device_ids() -> None:
    """Allow vLLM helpers to tolerate UUID-style CUDA_VISIBLE_DEVICES entries.

    On MIG-enabled clusters, Slurm often exposes devices as UUIDs like
    ``MIG-...`` instead of integer ordinals. Some vLLM utility code assumes the
    env var contains integers and crashes during import-time capability checks.
    For those UUID tokens, fall back to the local visible-device ordinal.
    """
    try:
        from vllm.platforms import current_platform
        from vllm.platforms.interface import Platform
    except ImportError:
        return

    patched = getattr(Platform.device_id_to_physical_device_id, "__func__", None)
    if getattr(patched, "_aoae_uuid_patch", False):
        return

    def _device_id_to_physical_device_id(cls, device_id: int):
        env_var = getattr(cls, "device_control_env_var", None)
        raw = os.environ.get(env_var, "") if env_var else ""
        if raw:
            device_ids = raw.split(",")
            physical_device_id = device_ids[device_id]
            try:
                return int(physical_device_id)
            except (TypeError, ValueError):
                # UUID tokens such as GPU-... or MIG-... are valid CUDA device
                # selectors but not valid integers. Use the local ordinal that
                # CUDA exposes inside the restricted visible-device set.
                return device_id
        return device_id

    _device_id_to_physical_device_id._aoae_uuid_patch = True
    class_method = classmethod(_device_id_to_physical_device_id)
    Platform.device_id_to_physical_device_id = class_method
    current_platform.__class__.device_id_to_physical_device_id = class_method


def _ensure_hf_rope_compatibility() -> None:
    """Add the legacy ``rope_type='default'`` handler if transformers dropped it.

    Some remote-code LLaDA2 model implementations still index
    ``transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS['default']``.
    Newer transformers releases removed that key, which breaks HF loading at
    model construction time. Restore the default RoPE initializer in-place so
    those remote modules keep working.
    """
    from transformers import modeling_rope_utils

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(
        config=None,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
        **rope_kwargs,
    ):
        del seq_len  # Default RoPE does not depend on the runtime sequence length.
        if config is not None and rope_kwargs:
            raise ValueError("Pass either a config or explicit RoPE kwargs, not both.")

        if config is None:
            base = rope_kwargs["base"]
            dim = int(rope_kwargs["dim"])
        else:
            base = float(getattr(config, "rope_theta", 10000.0))
            partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = getattr(config, "hidden_size") // getattr(config, "num_attention_heads")
            dim = int(head_dim * partial_rotary_factor)

        if dim <= 0:
            raise ValueError(f"RoPE dimension must be positive, got {dim}.")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        return inv_freq, 1.0

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _init_vllm_distributed(tp_size: int = 1):
    """Initialize vLLM distributed state for dInfer MoE models.

    Follows the pattern from dInfer/tests/test_llada_moe.py:
      1. init_distributed_environment(world_size, rank, ...)
      2. initialize_model_parallel(tp_size)
    """
    from vllm import distributed as vllm_dist
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    def _vllm_world_initialized() -> bool:
        get_world_group = getattr(vllm_dist, "get_world_group", None)
        if not callable(get_world_group):
            # Older/fake vLLM shims in tests do not expose get_world_group. If
            # PyTorch distributed is already initialized, treat that as enough
            # for the shim; real vLLM exposes get_world_group and is checked
            # above before model-parallel initialization.
            return torch.distributed.is_initialized()
        try:
            get_world_group()
            return True
        except AssertionError:
            return False
        except Exception:
            return False

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
                "Check torchrun / Slurm GPU binding and CUDA_VISIBLE_DEVICES."
            )
        torch.cuda.set_device(local_rank)

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if not _vllm_world_initialized():
            vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")
    else:
        if tp_size > 1 and "RANK" not in os.environ:
            raise RuntimeError(
                f"tp_size={tp_size} requires multi-process launch (e.g., torchrun). "
                "Run with torchrun or use a single-GPU config."
            )

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        if os.environ.get("MASTER_ADDR") == "localhost":
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
        world_size = int(os.environ.get("WORLD_SIZE", str(tp_size)))
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

    if not vllm_dist.model_parallel_is_initialized():
        vllm_dist.initialize_model_parallel(tp_size, backend="nccl")


def _cleanup_vllm_distributed() -> None:
    """Best-effort teardown for in-process vLLM model-parallel state."""
    try:
        from vllm import distributed as vllm_dist
    except ImportError:
        vllm_dist = None

    if vllm_dist is not None:
        destroy_model_parallel = getattr(vllm_dist, "destroy_model_parallel", None)
        if callable(destroy_model_parallel):
            try:
                destroy_model_parallel()
            except Exception:
                pass
        destroy_distributed_environment = getattr(vllm_dist, "destroy_distributed_environment", None)
        if callable(destroy_distributed_environment):
            try:
                destroy_distributed_environment()
            except Exception:
                pass

    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


def _init_sglang_distributed(tp_size: int = 1, ep_size: int = 1):
    """Initialize SGLang distributed/model-parallel state for dInfer MoE models."""
    from sglang.srt import distributed as sglang_dist

    if not torch.distributed.is_initialized():
        if tp_size > 1 and "RANK" not in os.environ:
            raise RuntimeError(
                f"tp_size={tp_size} requires multi-process launch (e.g., torchrun). "
                "Run with torchrun or use a single-GPU config."
            )

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        if os.environ.get("MASTER_ADDR") == "localhost":
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        sglang_dist.init_distributed_environment(tp_size, rank, "env://", rank, "nccl")

    sglang_dist.initialize_model_parallel(tp_size, ep_size, 1, backend="nccl")


def _patch_missing_moe_align_block_size(cfg: Optional[dict] = None) -> None:
    """Back-compat wrapper: ensure required vLLM MoE runtime capability."""
    cfg = cfg or {}
    allow_python_fallback = bool(cfg.get("base_model", {}).get("allow_python_fallback_ops", True))
    report = ensure_vllm_moe_runtime(
        strict=True,
        verbose=is_global_rank_zero(),
        allow_python_fallback=allow_python_fallback,
    )
    if report.get("patched_fallback_ops") and is_global_rank_zero():
        print(f"[Model] Runtime patched fallback ops: {report['patched_fallback_ops']}")


def _extract_to_device(args, kwargs) -> Optional[torch.device]:
    target = kwargs.get("device")
    if target is None and args:
        first = args[0]
        if isinstance(first, torch.Tensor):
            target = first.device
        elif not isinstance(first, torch.dtype):
            try:
                target = torch.device(first)
            except (TypeError, ValueError, RuntimeError):
                target = None
    if target is None:
        return None
    return torch.device(target)


def _devices_match(current: Optional[torch.device], target: Optional[torch.device]) -> bool:
    if current is None or target is None or current.type != target.type:
        return False
    if current.type != "cuda":
        return True
    if current.index is None or target.index is None:
        return True
    return current.index == target.index


def _kspec_find_clusters(non_agreed: torch.BoolTensor) -> list:
    """Return [(start, end), ...] for each contiguous run of True in non_agreed [L].

    Runs on CPU to avoid a CUDA sync in the cluster-detection loop.
    """
    na = non_agreed.cpu().tolist()
    clusters: list = []
    L = len(na)
    i = 0
    while i < L:
        if na[i]:
            j = i + 1
            while j < L and na[j]:
                j += 1
            clusters.append((i, j))
            i = j
        else:
            i += 1
    return clusters


def _nonfinite_tensor_summary(name: str, tensor: torch.Tensor) -> str:
    """Return a compact summary of NaN/Inf counts and finite range."""
    data = tensor.detach()
    nan_count = int(torch.isnan(data).sum().item())
    posinf_count = int(torch.isposinf(data).sum().item())
    neginf_count = int(torch.isneginf(data).sum().item())
    finite = data[torch.isfinite(data)]
    if finite.numel() > 0:
        finite_min = float(finite.min().item())
        finite_max = float(finite.max().item())
    else:
        finite_min = float("nan")
        finite_max = float("nan")
    return (
        f"{name}: nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, "
        f"finite_min={finite_min}, finite_max={finite_max}"
    )


class LLaDABaseModel(nn.Module):
    """Frozen LLaDA wrapper — no gradients flow through this module."""

    def __init__(self, cfg: dict):
        super().__init__()
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(
            cfg["base_model"].get("dtype", cfg["base_model"].get("torch_dtype", "bfloat16")),
            torch.bfloat16,
        )
        self.cfg = cfg
        # LLaDA2 was trained with block-causal attention (bidirectional within
        # blocks, causal across blocks).  ALL forward passes must use this mask
        # pattern; a fully-bidirectional (all-zeros) mask is out-of-distribution
        # and produces empty / garbage outputs.
        self._block_length = cfg.get("inference", {}).get("block_length", 32)

        name_or_path = cfg["base_model"]["name_or_path"]
        self._backend = _detect_backend(name_or_path, cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True,
        )

        # Auto-detect mask_token_id from tokenizer; fall back to config
        cfg_mask_id = cfg["base_model"].get("mask_token_id")
        tok_mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if tok_mask_id is not None:
            if cfg_mask_id is not None and cfg_mask_id != tok_mask_id:
                print(f"[Model] WARNING: config mask_token_id={cfg_mask_id} != "
                      f"tokenizer mask_token_id={tok_mask_id} "
                      f"({self.tokenizer.mask_token!r}). Using tokenizer value.")
            self.mask_id = tok_mask_id
        elif cfg_mask_id is not None:
            self.mask_id = cfg_mask_id
        else:
            raise ValueError(
                "No mask_token_id found in config or tokenizer. "
                "Set base_model.mask_token_id in the config file."
            )
        # Sync back to cfg so downstream code sees the correct value
        cfg["base_model"]["mask_token_id"] = self.mask_id
        self._dinfer_runtime = _get_dinfer_runtime(cfg)

        # vLLM config context manager (kept alive for dInfer forward passes)
        self._vllm_config_ctx = None
        self._vllm_config = None
        self._attention_mask_cache = {}
        self._query_attention_mask_cache = {}

        _INIT = {
            "hf": self._init_hf,
            "dkv": self._init_dkv,
            "dinfer": self._init_dinfer,
            "soft_moe": self._init_soft_moe,
        }
        if self._backend not in _INIT:
            raise ValueError(f"Unknown backend: {self._backend!r}. Choose from {list(_INIT)}")
        _INIT[self._backend](name_or_path)

        self._embedding_weight = self._resolve_embedding_weight()
        self.vocab_size = self._embedding_weight.shape[0]
        self.hidden_dim = self._resolve_representation_dim()
        self._closed = False

    def to(self, *args, **kwargs):
        """Move the wrapper when safe, and no-op for runtime-managed backends.

        dInfer/vLLM-backed MoE models already allocate onto their runtime-owned
        CUDA device in some environments, but in others they still need a single
        initial promotion from CPU to CUDA after weight loading. Calling
        ``nn.Module.to()`` repeatedly on the outer wrapper can therefore either
        be necessary once or catastrophically duplicate tens of GB on the next
        call.

        For ``dinfer`` and ``soft_moe`` we therefore:
          - allow the first real move when the model is still on CPU, and
          - no-op if the requested device already matches the current device.

        Standard HF and dKV backends still honor ``.to(...)`` normally.
        """
        target_device = _extract_to_device(args, kwargs)
        if self._backend in {"dinfer", "soft_moe"} and _devices_match(self.device, target_device):
            return self

        super().to(*args, **kwargs)
        self._embedding_weight = self._resolve_embedding_weight()
        return self

    # ------------------------------------------------------------------
    # Backend initializers (fail hard on missing deps)
    # ------------------------------------------------------------------
    def _init_hf(self, name_or_path: str):
        from transformers import AutoModelForCausalLM

        _ensure_hf_rope_compatibility()

        # LLaDA HF remote-code models currently require eager attention in
        # Transformers (SDPA dispatch is not implemented for this architecture).
        attn_impl = self.cfg.get("base_model", {}).get("attn_implementation", "eager")
        load_kwargs = dict(
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation=attn_impl,
        )

        # Optional: load with quantization if configured
        quant_method = self.cfg.get("base_model", {}).get("quantization")
        if quant_method == "int8":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs.pop("dtype", None)
                print("[Model] Loading with INT8 quantization (bitsandbytes)")
            except ImportError:
                print("[Model] bitsandbytes not available, skipping INT8 quantization")
        elif quant_method == "int4":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs.pop("dtype", None)
                print("[Model] Loading with INT4 quantization (bitsandbytes)")
            except ImportError:
                print("[Model] bitsandbytes not available, skipping INT4 quantization")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, **load_kwargs)
        except ValueError as exc:
            msg = str(exc)
            if ("scaled_dot_product_attention" in msg) and (load_kwargs.get("attn_implementation") != "eager"):
                print("[Model] WARNING: HF backend SDPA is unsupported for this architecture; retrying with eager attention.")
                load_kwargs["attn_implementation"] = "eager"
                self.model = AutoModelForCausalLM.from_pretrained(name_or_path, **load_kwargs)
            else:
                raise
        print(f"[Model] Loaded {type(self.model).__name__} (backend=hf)")
        self._freeze()

        # Optional: torch.compile for faster repeated forward passes
        if self.cfg.get("base_model", {}).get("compile", False):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[Model] Applied torch.compile (reduce-overhead mode)")
            except Exception as e:
                print(f"[Model] torch.compile failed, using eager mode: {e}")

    def _init_dkv(self, name_or_path: str):
        try:
            from models.modeling_llada_dkv_cache_decode import LLaDAModelLM
        except ImportError:
            raise ImportError(
                "dKV-Cache backend requires external/dKV-Cache. "
                "Run: git clone https://github.com/horseee/dKV-Cache.git external/dKV-Cache"
            )
        self.model = LLaDAModelLM.from_pretrained(
            name_or_path, trust_remote_code=True, dtype=self.dtype,
        )
        self._freeze()

    def _init_dinfer(self, name_or_path: str):
        runtime = self._dinfer_runtime
        if runtime == "vllm":
            self._init_dinfer_vllm(name_or_path)
            return
        if runtime == "sglang":
            self._init_dinfer_sglang(name_or_path)
            return
        raise RuntimeError(f"Unhandled dInfer runtime: {runtime!r}")

    def _init_dinfer_vllm(self, name_or_path: str):
        """Initialize dInfer MoE model with vLLM tensor parallelism."""
        # Must patch before any vLLM import: on MIG clusters CUDA_VISIBLE_DEVICES
        # contains UUID strings and vLLM calls int() on them at module import time.
        _patch_vllm_uuid_device_ids()
        try:
            from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
            from transformers import AutoConfig
            from dinfer.model import LLaDA2MoeModelLM
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                f"dInfer backend requires vllm and dinfer packages: {e}. "
                "Run setup.sh or: pip install vllm && pip install -e external/dInfer"
            )

        tp_size = self.cfg.get("hardware", {}).get("tp_size", 1)

        _init_vllm_distributed(tp_size)
        _patch_missing_moe_align_block_size(self.cfg)

        parallel_config = ParallelConfig(enable_expert_parallel=True)
        self._vllm_config = VllmConfig(parallel_config=parallel_config)
        self._vllm_config_ctx = set_current_vllm_config(self._vllm_config)
        self._vllm_config_ctx.__enter__()

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        load_device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

        try:
            local_model_path = snapshot_download(
                name_or_path,
                allow_patterns=["*.json", "*.safetensors", "*.model"],
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )

            model_config = AutoConfig.from_pretrained(
                name_or_path, trust_remote_code=True,
            )
            self.model = LLaDA2MoeModelLM(config=model_config).eval()
            self.model.load_weights(local_model_path, torch_dtype=self.dtype, device=load_device)
            # Move any remaining buffers (e.g., expert_map created by vLLM's
            # FusedMoE layer for EP routing) to the target device. load_weights
            # only touches tensors present in the checkpoint.
            self.model.to(load_device)
            self._freeze()
        except Exception:
            self._vllm_config_ctx.__exit__(None, None, None)
            self._vllm_config_ctx = None
            raise

    def _init_dinfer_sglang(self, name_or_path: str):
        """Initialize dInfer MoE model with the SGLang runtime."""
        _patch_vllm_uuid_device_ids()
        try:
            from transformers import AutoConfig
            from huggingface_hub import snapshot_download
            # Import the SGLang model module directly so we do not depend on
            # ``dinfer.model`` re-exporting LLaDA2SGLangLM in every env.
            from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
            from sglang.srt.layers.dp_attention import initialize_dp_attention
            from sglang.srt.layers.moe import initialize_moe_config
            from sglang.srt.server_args import ServerArgs
        except ImportError as e:
            raise ImportError(
                f"dInfer SGLang runtime requires sglang and dinfer packages: {e}. "
                "Install sglang and pip install -e external/dInfer."
            )

        tp_size = int(self.cfg.get("hardware", {}).get("tp_size", 1))
        ep_size = int(self.cfg.get("base_model", {}).get("dinfer_ep_size", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        _init_sglang_distributed(tp_size, ep_size=ep_size)

        model_config = AutoConfig.from_pretrained(
            name_or_path, trust_remote_code=True,
        )
        server_args = ServerArgs(
            model_path=name_or_path,
            enable_dp_attention=True,
            trust_remote_code=True,
            tp_size=tp_size,
            dp_size=1,
            pp_size=1,
        )
        try:
            from sglang.srt.server_args import set_global_server_args_for_scheduler
        except ImportError:
            pass
        else:
            set_global_server_args_for_scheduler(server_args)
        initialize_dp_attention(
            server_args=server_args,
            model_config=model_config,
        )
        initialize_moe_config(server_args)

        old_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.dtype)
            self.model = LLaDA2SGLangLM(
                config=model_config,
                quant_config=getattr(model_config, "quant_config", None),
                expert_map_path=self.cfg.get("base_model", {}).get("dinfer_expert_map_path", "."),
            ).eval()
        finally:
            torch.set_default_dtype(old_default_dtype)

        local_model_path = snapshot_download(
            name_or_path,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        self.model.load_weights(local_model_path, torch_dtype=self.dtype, device=device)
        self.model = self.model.to(device)
        if hasattr(self.model, "after_processing"):
            self.model.after_processing()
        self._freeze()

    def _init_soft_moe(self, name_or_path: str):
        """Initialize dInfer MoE model with soft routing (paper §3.7).

        Same as _init_dinfer but replaces hard top-k routing with
        hard-top-k-preserving tail-expanded routing using top-K_soft pruning.
        """
        self._init_dinfer(name_or_path)

        from .soft_moe import patch_model_with_soft_routing
        tau_r = self.cfg.get("base_model", {}).get("routing_temperature", 0.01)
        soft_topk = self.cfg.get("base_model", {}).get("soft_topk", None)
        patch_model_with_soft_routing(self.model, tau_r=tau_r, soft_topk=soft_topk)
        self._backend = "soft_moe"

    def _freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def close(self) -> None:
        """Release heavyweight runtime state for sequential evals in one process."""
        if getattr(self, "_closed", False):
            return

        if getattr(self, "_vllm_config_ctx", None) is not None:
            try:
                self._vllm_config_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._vllm_config_ctx = None

        backend = getattr(self, "_backend", None)
        runtime = getattr(self, "_dinfer_runtime", None)
        self.model = None
        self._embedding_weight = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if backend in {"dinfer", "soft_moe"}:
            if runtime == "vllm":
                _cleanup_vllm_distributed()
            elif torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception:
                    pass

        self._closed = True

    # ------------------------------------------------------------------
    def _resolve_embedding_weight(self) -> torch.Tensor:
        """Return the [V, D] token embedding weight tensor.

        Uses the standard HF ``get_input_embeddings()`` API.  Falls back to
        scanning ``named_modules()`` for an ``nn.Embedding`` only if the
        model is a non-standard custom class without that API.
        """
        if hasattr(self.model, "get_input_embeddings"):
            emb_module = self.model.get_input_embeddings()
            if emb_module is not None and hasattr(emb_module, "weight") and emb_module.weight is not None:
                return emb_module.weight

        if hasattr(self.model, "get_embed_and_head"):
            embed_w, _ = self.model.get_embed_and_head()
            if embed_w is not None:
                return embed_w

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and module.weight is not None:
                return module.weight

        raise RuntimeError(
            "Cannot locate embedding weight in the loaded model. "
            f"Model type: {type(self.model).__name__}. "
            "Ensure the model exposes get_input_embeddings() or contains an nn.Embedding."
        )

    def _resolve_representation_dim(self) -> int:
        """Return the width of token hidden states used by downstream adapters."""
        for obj in (
            getattr(self, "model", None),
            getattr(getattr(self, "model", None), "model", None),
            getattr(getattr(self, "model", None), "base_model", None),
        ):
            config = getattr(obj, "config", None)
            if config is None:
                continue
            for attr in ("hidden_size", "d_model", "n_embd", "model_dim"):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
        return int(self._embedding_weight.shape[1])

    # ------------------------------------------------------------------
    @property
    def device(self):
        embedding_weight = getattr(self, "_embedding_weight", None)
        if embedding_weight is not None:
            return embedding_weight.device

        model = getattr(self, "model", None)
        if model is not None:
            try:
                return next(model.parameters()).device
            except (StopIteration, AttributeError, TypeError):
                pass
            try:
                return next(model.buffers()).device
            except (StopIteration, AttributeError, TypeError):
                pass

        return torch.device("cpu")

    @property
    def backend(self) -> str:
        return self._backend

    def get_embedding_weight(self) -> torch.Tensor:
        """Return [V, D] token embedding matrix (frozen)."""
        return self._embedding_weight

    def set_routing_temperature(self, tau_r: float) -> None:
        """Update soft-routing temperature in-place for patched MoE models."""
        self.cfg.setdefault("base_model", {})["routing_temperature"] = float(tau_r)
        if self._backend != "soft_moe":
            return
        from .soft_moe import set_routing_temperature

        set_routing_temperature(self.model, float(tau_r))

    def set_soft_topk(self, soft_topk: int) -> None:
        """Update soft-routing expert budget in-place for patched MoE models."""
        self.cfg.setdefault("base_model", {})["soft_topk"] = int(soft_topk)
        if self._backend != "soft_moe":
            return
        from .soft_moe import set_soft_topk

        set_soft_topk(self.model, int(soft_topk))

    # ------------------------------------------------------------------
    def _make_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        block_length: int = 0,
    ) -> torch.Tensor:
        """Build 4D attention mask (B, 1, L, L) for LLaDA2.

        LLaDA2 is a masked diffusion model requiring bidirectional attention.
        - block_length=0: full attention (every position sees every other)
        - block_length>0: block-causal (full within block, causal across blocks)

        Returns a float mask for HF backends (0.0=attend, -inf=block) or a
        boolean mask for dInfer/SDPA backends (True=attend, False=block).
        """
        tp_size = int(self.cfg.get("hardware", {}).get("tp_size", 1) or 1)
        use_cache = not (
            self._backend in ("dinfer", "soft_moe")
            and tp_size > 1
        )

        cache = getattr(self, "_attention_mask_cache", None)
        if cache is None:
            cache = {}
            self._attention_mask_cache = cache
        key = (
            str(self._backend),
            batch_size,
            seq_len,
            int(block_length),
            device.type,
            device.index,
        )
        if use_cache:
            cached = cache.get(key)
            if cached is not None:
                # Clone so vLLM's TP kernels can't corrupt the cached base tensor
                # via in-place writes to the returned view.
                return cached.expand(batch_size, -1, -1, -1).clone()

        if block_length > 0:
            pos = torch.arange(seq_len, device=device)
            block_ends = (pos // block_length + 1) * block_length  # [L]
            col = pos.unsqueeze(0)       # [1, L]
            ends = block_ends.unsqueeze(1)  # [L, 1]
            if self._backend in ("dinfer", "soft_moe"):
                # dInfer SDPA converts to .bool() — True=attend, False=block
                base = (col < ends).unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
                if use_cache:
                    cache[key] = base
                return base.expand(batch_size, -1, -1, -1).clone()
            else:
                base = torch.where(col < ends, 0.0, float("-inf")).to(dtype=self.dtype)
                base = base.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
                if use_cache:
                    cache[key] = base
                return base.expand(batch_size, -1, -1, -1).clone()
        else:
            if self._backend in ("dinfer", "soft_moe"):
                mask = torch.ones(
                    (batch_size, 1, seq_len, seq_len),
                    dtype=torch.bool,
                    device=device,
                )
                if use_cache:
                    cache[key] = mask
                return mask.clone()
            else:
                mask = torch.zeros(
                    (batch_size, 1, seq_len, seq_len),
                    dtype=self.dtype,
                    device=device,
                )
                if use_cache:
                    cache[key] = mask
                return mask.clone()

    def _make_query_attention_mask(
        self,
        batch_size: int,
        full_seq_len: int,
        query_start: int,
        query_end: int,
        device: torch.device,
        block_length: int = 0,
    ) -> torch.Tensor:
        """Build attention mask for a query span against a cached full prefix."""
        if not (0 <= query_start < query_end <= full_seq_len):
            raise ValueError(
                f"Invalid query range [{query_start}, {query_end}) for full_seq_len={full_seq_len}."
            )

        tp_size = int(self.cfg.get("hardware", {}).get("tp_size", 1) or 1)
        use_cache = not (
            self._backend in ("dinfer", "soft_moe")
            and tp_size > 1
        )

        cache = getattr(self, "_query_attention_mask_cache", None)
        if cache is None:
            cache = {}
            self._query_attention_mask_cache = cache
        key = (
            str(self._backend),
            batch_size,
            full_seq_len,
            int(query_start),
            int(query_end),
            int(block_length),
            device.type,
            device.index,
        )
        if use_cache:
            cached = cache.get(key)
            if cached is not None:
                return cached.expand(batch_size, -1, -1, -1).clone()

        q_len = query_end - query_start
        if block_length > 0:
            query_pos = torch.arange(query_start, query_end, device=device)
            block_ends = (query_pos // block_length + 1) * block_length
            cols = torch.arange(full_seq_len, device=device).unsqueeze(0)
            ends = block_ends.unsqueeze(1)
            if self._backend in ("dinfer", "soft_moe"):
                base = (cols < ends).unsqueeze(0).unsqueeze(0)  # [1, 1, q, L]
                if use_cache:
                    cache[key] = base
                return base.expand(batch_size, -1, -1, -1).clone()
            base = torch.where(cols < ends, 0.0, float("-inf")).to(dtype=self.dtype)
            base = base.unsqueeze(0).unsqueeze(0)  # [1, 1, q, L]
            if use_cache:
                cache[key] = base
            return base.expand(batch_size, -1, -1, -1).clone()

        if self._backend in ("dinfer", "soft_moe"):
            mask = torch.ones(
                (batch_size, 1, q_len, full_seq_len),
                dtype=torch.bool,
                device=device,
            )
            if use_cache:
                cache[key] = mask
            return mask.clone()
        mask = torch.zeros(
            (batch_size, 1, q_len, full_seq_len),
            dtype=self.dtype,
            device=device,
        )
        if use_cache:
            cache[key] = mask
        return mask.clone()

    def _forward_dinfer_outputs(self, **kwargs):
        """Direct dInfer forward preserving the active runtime context."""
        if self._dinfer_runtime == "vllm":
            if self._vllm_config is None:
                from vllm.config import get_current_vllm_config
                self._vllm_config = get_current_vllm_config()
            if not hasattr(self, "_set_forward_context"):
                from vllm.forward_context import set_forward_context
                self._set_forward_context = set_forward_context
            with self._set_forward_context(None, self._vllm_config):
                return self.model(**kwargs)
        return self.model(**kwargs)

    def _forward_hf_outputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        use_cache: bool = False,
        backbone=None,
    ):
        """HF/remote-code forward using 4D block masks as attention_bias."""
        model = self.model if backbone is None else backbone
        kwargs = {}
        if attention_mask is not None:
            kwargs.update(self._resolve_hf_attention_kwargs(model, attention_mask))
        if output_hidden_states:
            kwargs["output_hidden_states"] = True
        if output_attentions:
            kwargs["output_attentions"] = True
        if use_cache:
            kwargs["use_cache"] = True
        return model(input_ids, **kwargs)

    def _resolve_hf_attention_kwargs(
        self,
        model: Any,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pass the block mask under the keyword the current HF backend expects.

        Older remote-code LLaDA variants consumed ``attention_bias`` while newer
        releases expect ``attention_mask`` and directly dereference it. Resolve
        the best keyword from the callable signature and fall back to supplying
        both when the target is too dynamic to inspect safely.
        """
        style = self._infer_hf_attention_style(model)
        if style == "attention_mask":
            # Models that only have attention_mask (2D HF convention) cannot
            # accept a 4D float block mask: they reshape it via .view(B, -1)
            # which flattens [B,1,L,L] → [B,L²] and then expands to [B,1,1,L²],
            # causing a broadcast mismatch against the internal [B,H,L,L] bias.
            if attention_mask.dim() == 4:
                return {}
            return {"attention_mask": attention_mask}
        if style == "attention_bias":
            return {"attention_bias": attention_mask}
        # "both" style: model has attention_mask (2D HF convention, internally
        # preprocessed via .view(B,-1)[:, None, None, :]) AND attention_bias
        # (additive 4D float, used as-is).  For a 4D float block mask, pass it
        # only as attention_bias to bypass the destructive 2D preprocessing;
        # the model adds bidirectional bias only when attention_bias is None,
        # so our mask replaces it correctly at full precision.
        if attention_mask.dim() == 4:
            return {"attention_bias": attention_mask}
        return {
            "attention_mask": attention_mask,
            "attention_bias": attention_mask,
        }

    def _infer_hf_attention_style(self, model: Any) -> str:
        cache = getattr(self, "_hf_attention_style_cache", None)
        if cache is None:
            cache = {}
            self._hf_attention_style_cache = cache

        cache_key = id(model)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        target = getattr(model, "forward", None)
        if not callable(target):
            target = getattr(model, "__call__", None)

        style = "both"
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            params = signature.parameters
            has_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            )
            has_attention_mask = "attention_mask" in params
            has_attention_bias = "attention_bias" in params

            if has_attention_mask and has_attention_bias:
                style = "both"
            elif has_attention_mask:
                style = "attention_mask"
            elif has_attention_bias:
                style = "attention_bias"
            elif has_var_kwargs:
                style = "both"
            else:
                style = "attention_mask"

        cache[cache_key] = style
        return style

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Full forward pass → [B, L, V] logits.

        Uses block-causal attention (the pattern LLaDA2 was trained with).
        """
        return self.forward_block_causal(input_ids, block_length=self._block_length)

    @torch.no_grad()
    def forward_block_causal(
        self, input_ids: torch.LongTensor, block_length: int = 32,
    ) -> torch.Tensor:
        """Forward pass with block-causal attention for S-mode decoding.

        Full attention within each block, causal across blocks.
        This allows KV caching of completed blocks.
        """
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=block_length,
        )
        if self._backend in ("dinfer", "soft_moe"):
            return self._forward_dinfer(input_ids, attention_mask=attention_mask)
        outputs = self._forward_hf_outputs(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        try:
            return outputs[0]
        except (TypeError, IndexError):
            raise RuntimeError(f"Unexpected model output type: {type(outputs)}")

    def _forward_dinfer(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through dInfer MoE model with the configured runtime.

        Args:
            input_ids: [B, L] token ids.
            attention_mask: [B, 1, L, L] block-causal attention mask.
                For dInfer SDPA, this should be a boolean tensor where
                True = attend, False = block.
        """
        outputs = self._forward_dinfer_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        raise RuntimeError(f"Unexpected dInfer model output type: {type(outputs)}")

    @torch.no_grad()
    def forward_with_cache(
        self,
        input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, object]:
        """Forward pass returning logits and a dInfer KV cache object."""
        if self._backend not in ("dinfer", "soft_moe"):
            raise NotImplementedError("forward_with_cache is only supported for dInfer-backed models.")
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        input_ids = input_ids.contiguous()
        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=self._block_length,
        )
        outputs = self._forward_dinfer_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("dInfer cached forward did not return past_key_values.")
        if hasattr(past_key_values, "consolidate"):
            past_key_values.consolidate()
        return logits, past_key_values

    @torch.no_grad()
    def forward_replace_with_cache(
        self,
        full_input_ids: torch.LongTensor,
        replace_slice: slice,
        past_key_values: object,
    ) -> Tuple[torch.Tensor, object]:
        """Recompute a contiguous query span against cached prefix state."""
        if self._backend not in ("dinfer", "soft_moe"):
            raise NotImplementedError("forward_replace_with_cache is only supported for dInfer-backed models.")
        if past_key_values is None:
            raise ValueError("past_key_values must be provided for replace-position forward.")
        if hasattr(past_key_values, "consolidate"):
            past_key_values.consolidate()
        if full_input_ids.device != self.device:
            full_input_ids = full_input_ids.to(self.device)
        full_input_ids = full_input_ids.contiguous()
        self._aoae_last_cache_replace_fell_back = False
        self._aoae_last_full_recompute_logits = None

        start = int(replace_slice.start)
        end = int(replace_slice.stop)
        batch_size, full_seq_len = full_input_ids.shape
        query_ids = full_input_ids[:, start:end].contiguous()
        attention_mask = self._make_query_attention_mask(
            batch_size,
            full_seq_len,
            start,
            end,
            full_input_ids.device,
            block_length=self._block_length,
        )
        try:
            outputs = self._forward_dinfer_outputs(
                input_ids=query_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=(start, end),
            )
        except (FloatingPointError, RuntimeError) as exc:
            message = str(exc).lower()
            if "[aoae][nonfinite" not in message and "incoherent cache" not in message:
                raise
            return self._fallback_from_incoherent_cache(
                full_input_ids=full_input_ids,
                replace_slice=replace_slice,
                reason=str(exc),
            )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if not torch.isfinite(logits).all():
            return self._fallback_from_incoherent_cache(
                full_input_ids=full_input_ids,
                replace_slice=replace_slice,
                reason=_nonfinite_tensor_summary("replace_logits", logits),
            )
        next_cache = getattr(outputs, "past_key_values", None)
        if next_cache is None:
            raise RuntimeError("dInfer replace-position forward did not return updated past_key_values.")
        if hasattr(next_cache, "consolidate"):
            next_cache.consolidate()
        return logits, next_cache

    @torch.no_grad()
    def forward_with_kspec_cache(
        self,
        full_input_ids: torch.LongTensor,
        resp_slice: slice,
        aux_past_kv: object,
        k_spec_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, object]:
        """Legacy accepted-reuse primary forward.

        Reuses aux_past_kv (K/V from the auxiliary model) at positions marked by
        k_spec_mask. For each contiguous cluster of non-skipped response positions
        the model runs forward_replace_with_cache, so only those tokens incur
        primary Q/K/V projection cost.

        IMPORTANT: canonical AOAE does not use this function for K_spec
        validation. It is retained only as a disabled legacy ablation because
        skipped primary computation cannot validate unverified drafted tokens.

        Args:
            full_input_ids: [B, L_total] — full prompt + response sequence.
            resp_slice:     slice(P, P+L_gen) — response region in full sequence.
            aux_past_kv:    KV cache from the auxiliary forward this step.
                            Used as the starting cache; aux K/V already injected at
                            ALL positions (prompt + agreed response) by construction.
            k_spec_mask:    [B, L_gen] bool — legacy accepted-reuse skip mask.
                            True = reuse aux K/V and skip primary there.

        Returns:
            logits_fresh:  [B, L_gen, V] — primary logits at non-agreed positions;
                           zeros at agreed positions (caller substitutes aux_logits there).
            kv_updated:    KV cache updated at non-agreed response positions.
        """
        if self._backend not in ("dinfer", "soft_moe"):
            raise NotImplementedError(
                "forward_with_kspec_cache requires a dInfer/soft_moe backend."
            )

        B = full_input_ids.shape[0]
        P = resp_slice.start
        L_gen = resp_slice.stop - P
        dev = full_input_ids.device

        # All positions skipped → caller uses aux_logits everywhere.
        # caller uses aux_logits everywhere.
        if k_spec_mask.all():
            return (
                torch.zeros(B, L_gen, self.vocab_size, dtype=self.dtype, device=dev),
                aux_past_kv,
            )

        # Conservative union across batch: position is non-agreed if ANY sample disagrees.
        # Exact when B=1 (the common training/eval case).
        non_agreed_pos = ~k_spec_mask.all(dim=0)  # [L_gen]
        clusters = _kspec_find_clusters(non_agreed_pos)

        logits_out = torch.zeros(B, L_gen, self.vocab_size, dtype=self.dtype, device=dev)
        current_kv = aux_past_kv

        for c_start, c_end in clusters:
            span_logits, current_kv = self.forward_replace_with_cache(
                full_input_ids,
                slice(P + c_start, P + c_end),
                current_kv,
            )
            if getattr(self, "_aoae_last_cache_replace_fell_back", False):
                full_logits = getattr(self, "_aoae_last_full_recompute_logits", None)
                if full_logits is None:
                    raise RuntimeError(
                        "[AOAE][CACHE FALLBACK BUG] Missing full recompute logits after cached "
                        "replace fallback."
                    )
                return full_logits[:, resp_slice, :], current_kv
            logits_out[:, c_start:c_end, :] = span_logits

        return logits_out, current_kv

    def _fallback_from_incoherent_cache(
        self,
        *,
        full_input_ids: torch.LongTensor,
        replace_slice: slice,
        reason: str,
    ) -> Tuple[torch.Tensor, object]:
        """Discard hybrid cached state and fully recompute after a non-finite replace pass."""
        start = int(replace_slice.start)
        end = int(replace_slice.stop)
        rank = (
            os.environ.get("RANK")
            or os.environ.get("SLURM_PROCID")
            or os.environ.get("LOCAL_RANK")
            or "unknown"
        )
        fallback_count = int(getattr(self, "_aoae_cache_fallback_count", 0)) + 1
        self._aoae_cache_fallback_count = fallback_count

        banner = "!" * 108
        print(banner, file=sys.stderr, flush=True)
        print(
            (
                f"[AOAE][CACHE FALLBACK][rank={rank}][count={fallback_count}] "
                f"NON-FINITE OUTPUT FROM CACHED replace_position=({start}, {end})"
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            (
                "[AOAE][CACHE FALLBACK] This implies the cached/stale hybrid KV state became "
                "incoherent. Discarding cached KV and recomputing the FULL sequence."
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[AOAE][CACHE FALLBACK] Trigger summary: {reason}",
            file=sys.stderr,
            flush=True,
        )
        print(banner, file=sys.stderr, flush=True)

        full_logits, fresh_cache = self.forward_with_cache(full_input_ids)
        if not torch.isfinite(full_logits).all():
            raise RuntimeError(
                "[AOAE][CACHE FALLBACK FAILED] Fresh full recompute also returned non-finite logits. "
                f"{_nonfinite_tensor_summary('full_logits', full_logits)}"
            )
        self._aoae_last_cache_replace_fell_back = True
        self._aoae_last_full_recompute_logits = full_logits.detach()
        return full_logits[:, start:end, :], fresh_cache

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_with_hidden(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass → (logits [B,L,V], last_hidden [B,L,D])."""
        logits, hidden_states = self.forward_with_all_hidden(input_ids)
        return logits, hidden_states[-1]

    @torch.no_grad()
    def forward_with_all_hidden(
        self, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass → (logits [B,L,V], hidden_states [num_layers][B,L,D])."""
        logits, hidden_states, _, _ = self.forward_with_diagnostics(
            input_ids,
            output_attentions=False,
            output_kv=False,
        )
        return logits, hidden_states

    @torch.no_grad()
    def forward_with_diagnostics(
        self,
        input_ids: torch.LongTensor,
        *,
        output_attentions: bool = False,
        output_kv: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass with optional hidden-state, attention, and KV diagnostics."""
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=self._block_length,
        )
        if self._backend in ("dinfer", "soft_moe"):
            outputs = self._forward_outputs_with_fallback(
                backend="dinfer",
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_kv=output_kv,
            )
            return self._extract_diagnostics(outputs, source="dinfer")

        outputs = self._forward_outputs_with_fallback(
            backend="hf",
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_kv=output_kv,
        )
        return self._extract_diagnostics(outputs, source="hf")

    @torch.no_grad()
    def forward_hidden_only(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass → hidden_states [B, L, D] only, skipping the LM head.

        Use when logits over the full vocabulary are not needed (e.g. PRISM
        training).  Avoids allocating the [B, L, vocab_size] logit tensor
        (~7-10 GiB for LLaDA2 with vocab_size=160 k).

        Uses block-causal attention (the pattern LLaDA2 was trained with).
        """
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if self._backend in ("dinfer", "soft_moe"):
            return self.forward_with_hidden(input_ids)[1]

        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=self._block_length,
        )
        # Call the backbone (LLaDA2MoeModel) directly, bypassing the LM head.
        # self.model is LLaDA2MoeModelLM; .model is LLaDA2MoeModel.
        backbone = getattr(self.model, "model", None)
        if backbone is None:
            # Fallback for other HF architectures that don't expose .model
            return self.forward_with_hidden(input_ids)[1]

        outputs = self._forward_hf_hidden_outputs(
            input_ids,
            attention_mask=attention_mask,
            backbone=backbone,
        )
        return self._extract_hidden_only_tensor(
            outputs,
            batch_size=batch_size,
            seq_len=seq_len,
            source="hf_backbone",
        )

    def _forward_hf_hidden_outputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        *,
        backbone=None,
    ):
        """HF backbone forward with best-effort hidden-state support."""
        try:
            return self._forward_hf_outputs(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                backbone=backbone,
            )
        except TypeError:
            return self._forward_hf_outputs(
                input_ids,
                attention_mask=attention_mask,
                backbone=backbone,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            unsupported = (
                "unexpected keyword" in message
                or "output_hidden_states" in message
            )
            if not unsupported:
                raise
            return self._forward_hf_outputs(
                input_ids,
                attention_mask=attention_mask,
                backbone=backbone,
            )

    def _extract_hidden_only_tensor(
        self,
        outputs: Any,
        *,
        batch_size: int,
        seq_len: int,
        source: str,
    ) -> torch.Tensor:
        """Extract [B, L, D] hidden states from heterogeneous HF-style outputs."""

        def _is_seq_tensor(value: Any) -> bool:
            return (
                isinstance(value, torch.Tensor)
                and value.ndim == 3
                and value.shape[0] == batch_size
                and value.shape[1] == seq_len
            )

        expected_dim = int(getattr(self, "hidden_dim", 0) or self._embedding_weight.shape[1])
        vocab_size = int(getattr(self, "vocab_size", 0) or self._embedding_weight.shape[0])

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if _is_seq_tensor(last_hidden):
            return last_hidden

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            if isinstance(hidden_states, torch.Tensor):
                if _is_seq_tensor(hidden_states):
                    return hidden_states
            else:
                candidates = [tensor for tensor in hidden_states if _is_seq_tensor(tensor)]
                for tensor in reversed(candidates):
                    if tensor.shape[-1] == expected_dim:
                        return tensor
                if candidates:
                    return candidates[-1]

        tuple_candidates = []
        if isinstance(outputs, (tuple, list)):
            tuple_candidates = [tensor for tensor in outputs if _is_seq_tensor(tensor)]

        for tensor in tuple_candidates:
            if tensor.shape[-1] == expected_dim:
                return tensor

        non_vocab = [tensor for tensor in tuple_candidates if tensor.shape[-1] != vocab_size]
        if len(non_vocab) == 1:
            return non_vocab[0]
        if non_vocab:
            return min(non_vocab, key=lambda tensor: abs(int(tensor.shape[-1]) - expected_dim))

        if tuple_candidates:
            return tuple_candidates[0]

        raise RuntimeError(
            f"{source} model did not return usable hidden states with shape [B, L, D]. "
            f"Expected batch={batch_size}, seq_len={seq_len}, hidden_dim={expected_dim}."
        )

    def _extract_logits_and_hidden_states(
        self, outputs, source: str = "hf",
    ) -> Tuple[torch.Tensor, list]:
        logits, hidden_states, _, _ = self._extract_diagnostics(outputs, source=source)
        return logits, hidden_states

    def _extract_diagnostics(
        self, outputs, source: str = "hf",
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            try:
                logits = outputs[0]
            except (TypeError, IndexError):
                raise RuntimeError(f"Unexpected {source} output type: {type(outputs)}")

        hidden_states = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            if isinstance(outputs.hidden_states, torch.Tensor):
                hidden_states = [outputs.hidden_states]
            else:
                hidden_states = list(outputs.hidden_states)
                # HF convention usually includes embedding output at index 0.
                if len(hidden_states) > 1:
                    hidden_states = hidden_states[1:]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = [outputs.last_hidden_state]

        if not hidden_states:
            raise RuntimeError(f"{source} model did not return hidden states.")

        attentions = None
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            if isinstance(outputs.attentions, torch.Tensor):
                attentions = [outputs.attentions]
            else:
                attentions = list(outputs.attentions)

        layer_kv = self._extract_layer_kv(outputs)
        return logits, hidden_states, attentions, layer_kv

    def _extract_layer_kv(
        self, outputs: Any,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            return None

        try:
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()
        except Exception:
            return None

        if not isinstance(past_key_values, (list, tuple)):
            return None

        layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_key_values:
            if not isinstance(layer, (list, tuple)) or len(layer) < 2:
                return None
            key, value = layer[0], layer[1]
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                return None
            layers.append((key, value))
        return layers or None

    def _forward_outputs_with_fallback(
        self,
        *,
        backend: str,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
        output_kv: bool,
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }
        if output_attentions:
            kwargs["output_attentions"] = True
        if output_kv:
            kwargs["use_cache"] = True

        try:
            if backend == "hf":
                return self._forward_hf_outputs(**kwargs)
            return self._forward_dinfer_outputs(**kwargs)
        except TypeError:
            if not (output_attentions or output_kv):
                raise
        except RuntimeError as exc:
            message = str(exc).lower()
            unsupported = (
                "unexpected keyword" in message
                or "output_attentions" in message
                or "use_cache" in message
            )
            if not unsupported:
                raise

        fallback_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }
        if backend == "hf":
            return self._forward_hf_outputs(**fallback_kwargs)
        return self._forward_dinfer_outputs(**fallback_kwargs)

    def _forward_with_all_hidden_dinfer(
        self, input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Forward with hidden states through dInfer MoE model."""
        outputs = self._forward_dinfer_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(self._dinfer_runtime == "vllm"),
        )
        return self._extract_logits_and_hidden_states(outputs, source="dinfer")

    def _forward_with_hidden_dinfer(
        self, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, hidden_states = self._forward_with_all_hidden_dinfer(input_ids)
        return logits, hidden_states[-1]
