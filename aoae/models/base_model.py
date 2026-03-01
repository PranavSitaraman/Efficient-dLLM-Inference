"""
Frozen LLaDA base-model wrapper.

Wraps a HuggingFace LLaDA model and exposes:
  - forward(): full logits over vocabulary at every position.
  - forward_with_hidden(): logits + last-layer hidden states (for PRISM).
  - get_embedding_weight(): token embedding matrix E (for soft-masked state).

Supports four backends selected via config ``base_model.backend``:
  - ``hf``       — HuggingFace AutoModelForCausalLM (default; inclusionAI/LLaDA2.1-mini).
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


_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DKV_DIR = os.path.join(_CODE_DIR, "external", "dKV-Cache")
_DINFER_DIR = os.path.join(_CODE_DIR, "external", "dInfer", "python")

if os.path.isdir(_DKV_DIR) and _DKV_DIR not in sys.path:
    sys.path.insert(0, _DKV_DIR)
if os.path.isdir(_DINFER_DIR) and _DINFER_DIR not in sys.path:
    sys.path.insert(0, _DINFER_DIR)


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


def _init_vllm_distributed(tp_size: int = 1):
    """Initialize vLLM distributed state for dInfer MoE models.

    Follows the pattern from dInfer/tests/test_llada_moe.py:
      1. init_distributed_environment(world_size, rank, ...)
      2. initialize_model_parallel(tp_size)
    """
    from vllm import distributed as vllm_dist

    if not torch.distributed.is_initialized():
        if tp_size > 1 and "RANK" not in os.environ:
            raise RuntimeError(
                f"tp_size={tp_size} requires multi-process launch (e.g., torchrun). "
                "Run with torchrun or use a single-GPU config."
            )

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
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

        load_kwargs = dict(
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="sdpa",  # use PyTorch scaled dot-product attention
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

        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, **load_kwargs)
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
        """Initialize dInfer MoE model with vLLM tensor parallelism.

        Follows the exact pattern from dInfer/tests/test_llada_moe.py:
        1. Initialize vLLM distributed environment
        2. Set vLLM config with parallel settings
        3. Load model config + instantiate + load weights inside config context
        """
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

        # Step 1: Initialize vLLM distributed state
        _init_vllm_distributed(tp_size)

        # Step 2: Set vLLM config context with expert parallelism
        parallel_config = ParallelConfig(enable_expert_parallel=True)
        self._vllm_config = VllmConfig(parallel_config=parallel_config)
        self._vllm_config_ctx = set_current_vllm_config(self._vllm_config)
        self._vllm_config_ctx.__enter__()

        # Step 3: Download model from HuggingFace if needed
        # dInfer's load_weights expects a local path to model.safetensors.index.json
        local_model_path = snapshot_download(
            name_or_path,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )

        # Step 4: Load model inside vLLM config context
        model_config = AutoConfig.from_pretrained(
            name_or_path, trust_remote_code=True,
        )
        self.model = LLaDA2MoeModelLM(config=model_config).eval()
        self.model.load_weights(local_model_path, torch_dtype=self.dtype)
        self._freeze()

    def _init_soft_moe(self, name_or_path: str):
        """Initialize dInfer MoE model with soft routing (paper §3.7).

        Same as _init_dinfer but replaces hard top-k routing with
        temperature-controlled soft routing so all experts are active.
        """
        # First, initialize exactly like dinfer
        self._init_dinfer(name_or_path)

        # Then patch with soft routing
        from .soft_moe import patch_model_with_soft_routing
        tau_r = self.cfg.get("base_model", {}).get("routing_temperature", 0.01)
        patch_model_with_soft_routing(self.model, tau_r=tau_r)
        self._backend = "soft_moe"  # restore after _init_dinfer sets it

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
        """
        if block_length > 0:
            # Block-causal: full attention within each block,
            # causal across blocks (can see all previous blocks).
            # SDPA additive mask: 0.0 = attend, -inf = don't attend.
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), dtype=self.dtype, device=device,
            )
            for i in range(seq_len):
                block_i = i // block_length
                see_end = (block_i + 1) * block_length
                mask[i, :min(see_end, seq_len)] = 0.0
            return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            # Full bidirectional attention (standard diffusion).
            # SDPA additive mask: all-zeros = no masking = full attention.
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
        if self._backend in ("dinfer", "soft_moe"):
            return self._forward_dinfer(input_ids)
        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=block_length,
        )
        outputs = self.model(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        try:
            return outputs[0]
        except (TypeError, IndexError):
            raise RuntimeError(f"Unexpected model output type: {type(outputs)}")

    def _forward_dinfer(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass through dInfer MoE model with vLLM context."""
        from vllm.config import get_current_vllm_config
        from vllm.forward_context import set_forward_context
        vllm_config = get_current_vllm_config()
        with set_forward_context(None, vllm_config):
            outputs = self.model(input_ids)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        raise RuntimeError(f"Unexpected dInfer model output type: {type(outputs)}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_with_hidden(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass → (logits [B,L,V], hidden_states [B,L,D]).

        Uses block-causal attention (the pattern LLaDA2 was trained with).
        """
        if self._backend in ("dinfer", "soft_moe"):
            return self._forward_with_hidden_dinfer(input_ids)

        batch_size, seq_len = input_ids.shape
        attention_mask = self._make_attention_mask(
            batch_size, seq_len, input_ids.device, block_length=self._block_length,
        )
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Extract logits from various output types
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            # Handle MoeModelOutputWithPast and other custom output classes
            # Try to get the first element if it's a dataclass-like object
            try:
                logits = outputs[0]
            except (TypeError, IndexError):
                raise RuntimeError(
                    f"Unexpected model output type: {type(outputs)}. "
                    f"Available attributes: {dir(outputs)}"
                )

        # Extract hidden states from various output types
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            raise RuntimeError(
                "Model did not return hidden_states. "
                "Ensure the model supports output_hidden_states=True."
            )

        return logits, hidden

    @torch.no_grad()
    def forward_hidden_only(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass → hidden_states [B, L, D] only, skipping the LM head.

        Use when logits over the full vocabulary are not needed (e.g. PRISM
        training).  Avoids allocating the [B, L, vocab_size] logit tensor
        (~7-10 GiB for LLaDA2 with vocab_size=160 k).

        Uses block-causal attention (the pattern LLaDA2 was trained with).
        """
        if self._backend in ("dinfer", "soft_moe"):
            return self._forward_with_hidden_dinfer(input_ids)[1]

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

    def _forward_with_hidden_dinfer(
        self, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with hidden states through dInfer MoE model."""
        from vllm.config import get_current_vllm_config
        from vllm.forward_context import set_forward_context
        vllm_config = get_current_vllm_config()
        with set_forward_context(None, vllm_config):
            outputs = self.model(input_ids, output_hidden_states=True)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            raise RuntimeError(f"Unexpected dInfer output type: {type(outputs)}")

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            raise RuntimeError("dInfer model did not return hidden_states.")

        return logits, hidden
