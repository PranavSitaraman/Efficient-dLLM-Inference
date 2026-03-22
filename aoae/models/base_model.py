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
  - ``soft_moe`` — dInfer MoE with soft routing (all experts active; paper §3.7).
  - ``auto``     — auto-detect from model name.

All backends fail hard on missing dependencies (no silent fallbacks).
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Tuple, Optional
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

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
                "Check torchrun / Slurm GPU binding and CUDA_VISIBLE_DEVICES."
            )
        torch.cuda.set_device(local_rank)

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
        
        torch.distributed.init_process_group(
            backend="nccl", 
            world_size=tp_size, 
            rank=int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))),
        )

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    vllm_dist.init_distributed_environment(world_size, rank, "env://", rank, "nccl")
    vllm_dist.initialize_model_parallel(tp_size, backend="nccl")


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
        self.hidden_dim = self._embedding_weight.shape[1]

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
            self.model.load_weights(local_model_path, torch_dtype=self.dtype)
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
        temperature-controlled soft routing using top-K_soft pruning.
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

    # ------------------------------------------------------------------
    @property
    def device(self):
        return self._embedding_weight.device

    @property
    def backend(self) -> str:
        return self._backend

    def get_embedding_weight(self) -> torch.Tensor:
        """Return [V, D] token embedding matrix (frozen)."""
        return self._embedding_weight

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
        if block_length > 0:
            pos = torch.arange(seq_len, device=device)
            block_ends = (pos // block_length + 1) * block_length  # [L]
            col = pos.unsqueeze(0)       # [1, L]
            ends = block_ends.unsqueeze(1)  # [L, 1]
            if self._backend in ("dinfer", "soft_moe"):
                # dInfer SDPA converts to .bool() — True=attend, False=block
                mask = (col < ends)  # [L, L] bool
                return mask.unsqueeze(0).unsqueeze(0).expand(
                    batch_size, -1, -1, -1,
                )
            else:
                mask = torch.where(col < ends, 0.0, float("-inf"))  # [L, L]
                return mask.to(dtype=self.dtype).unsqueeze(0).unsqueeze(0).expand(
                    batch_size, -1, -1, -1,
                )
        else:
            if self._backend in ("dinfer", "soft_moe"):
                return torch.ones(
                    (batch_size, 1, seq_len, seq_len),
                    dtype=torch.bool,
                    device=device,
                )
            else:
                return torch.zeros(
                    (batch_size, 1, seq_len, seq_len),
                    dtype=self.dtype,
                    device=device,
                )

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
        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=block_length,
        )
        if self._backend in ("dinfer", "soft_moe"):
            return self._forward_dinfer(input_ids, attention_mask=attention_mask)
        outputs = self.model(input_ids, attention_mask=attention_mask)
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
        if self._dinfer_runtime == "vllm":
            if self._vllm_config is None:
                from vllm.config import get_current_vllm_config
                self._vllm_config = get_current_vllm_config()
            if not hasattr(self, '_set_forward_context'):
                from vllm.forward_context import set_forward_context
                self._set_forward_context = set_forward_context
            with self._set_forward_context(None, self._vllm_config):
                outputs = self.model(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        raise RuntimeError(f"Unexpected dInfer model output type: {type(outputs)}")

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
        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=self._block_length,
        )
        if self._backend in ("dinfer", "soft_moe"):
            return self._forward_with_all_hidden_dinfer(input_ids, attention_mask=attention_mask)

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return self._extract_logits_and_hidden_states(outputs, source="hf")

    @torch.no_grad()
    def forward_hidden_only(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass → hidden_states [B, L, D] only, skipping the LM head.

        Use when logits over the full vocabulary are not needed (e.g. PRISM
        training).  Avoids allocating the [B, L, vocab_size] logit tensor
        (~7-10 GiB for LLaDA2 with vocab_size=160 k).

        Uses block-causal attention (the pattern LLaDA2 was trained with).
        """
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

        outputs = backbone(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        raise RuntimeError(f"Unexpected backbone output type: {type(outputs)}")

    def _extract_logits_and_hidden_states(
        self, outputs, source: str = "hf",
    ) -> Tuple[torch.Tensor, list]:
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

        return logits, hidden_states

    def _forward_with_all_hidden_dinfer(
        self, input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Forward with hidden states through dInfer MoE model."""
        if self._dinfer_runtime == "vllm":
            if self._vllm_config is None:
                from vllm.config import get_current_vllm_config
                self._vllm_config = get_current_vllm_config()
            if not hasattr(self, '_set_forward_context'):
                from vllm.forward_context import set_forward_context
                self._set_forward_context = set_forward_context
            with self._set_forward_context(None, self._vllm_config):
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True,
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return self._extract_logits_and_hidden_states(outputs, source="dinfer")

    def _forward_with_hidden_dinfer(
        self, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, hidden_states = self._forward_with_all_hidden_dinfer(input_ids)
        return logits, hidden_states[-1]
