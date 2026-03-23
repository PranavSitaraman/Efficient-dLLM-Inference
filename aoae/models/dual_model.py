"""
Dual-Model MoE Wrapper for Speculative Diffusion (paper §3.7).

Wraps a SINGLE LLaDA2.1-mini (16B MoE) model and toggles its routing:
  - Hard routing (auxiliary mode): top-k experts, ~1.4B active, fast
  - Soft routing (primary mode):  all experts, ~16B active, slow but expressive

The auxiliary pass produces fast draft predictions whose KV states are
pre-cached; the primary pass verifies and refines, reusing cached KV states
where the two routing modes agree on the same token prediction.

Only ONE copy of the 16B model is loaded; routing is toggled in-place.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .base_model import LLaDABaseModel, _detect_backend
from .soft_moe import (
    patch_model_with_soft_routing,
    set_hard_routing,
    set_soft_routing,
    set_routing_temperature,
    set_soft_topk,
)


@dataclass
class DualModelOutput:
    """Output from a dual-model forward pass."""
    primary_logits: torch.Tensor      # [B, L, V] from soft-routed primary
    auxiliary_logits: torch.Tensor     # [B, L, V] from hard-routed auxiliary
    agreement: torch.Tensor           # [B, L] bool: argmax match
    agreement_rate: float             # scalar: mean agreement across batch
    primary_hidden: Optional[torch.Tensor] = None  # [B, L, D] if requested
    primary_hidden_states: Optional[List[torch.Tensor]] = None  # [N][B, L, D]


def _select_dual_base_backend(cfg: dict) -> str:
    """Resolve the concrete backend used to load the single shared model.

    Dual mode toggles MoE routing in-place, so it only works with LLaDA2-style
    MoE backbones. Use the normal backend auto-detection rules unless the user
    explicitly sets ``base_model.dual_backend``.
    """
    model_cfg = _deep_copy_cfg(cfg)["base_model"]
    requested = model_cfg.get("dual_backend")
    if requested is None:
        auto_cfg = {"base_model": dict(model_cfg)}
        auto_cfg["base_model"]["backend"] = "auto"
        requested = _detect_backend(model_cfg["name_or_path"], auto_cfg)

    if requested == "soft_moe":
        # Dual mode starts from hard routing, then patches soft routing itself.
        return "dinfer"
    if requested != "dinfer":
        raise ValueError(
            "DualModelWrapper requires an MoE-capable backend because it toggles "
            f"expert routing in-place, but resolved backend {requested!r}. "
            "Use an inclusionAI/LLaDA2.X model or set base_model.dual_backend='dinfer'."
        )
    return "dinfer"


class DualModelWrapper(nn.Module):
    """Single-model wrapper that toggles between hard and soft MoE routing.

    Loads ONE LLaDA2.X MoE model via the MoE-capable backend, patches its gates with
    SoftMoERouter, then switches routing mode per forward pass:
      - auxiliary_forward(): hard routing (~1.4B active, fast)
      - primary_forward():  soft routing (~16B active, slow)

    This halves GPU memory compared to loading two separate model copies.

    Args:
        cfg: full config dict with base_model.routing_temperature for τ_r.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        self._cfg = cfg
        self.tau_r = cfg["base_model"].get("routing_temperature", 0.01)
        self._soft_topk = cfg["base_model"].get("soft_topk", None)

        base_cfg = _deep_copy_cfg(cfg)
        base_cfg["base_model"]["backend"] = _select_dual_base_backend(cfg)
        self._model = LLaDABaseModel(base_cfg)

        patch_model_with_soft_routing(
            self._model.model, tau_r=self.tau_r, soft_topk=self._soft_topk,
        )
        set_hard_routing(self._model.model)

        # Expose shared attributes
        self.tokenizer = self._model.tokenizer
        self.mask_id = self._model.mask_id
        self.vocab_size = self._model.vocab_size
        self.hidden_dim = self._model.hidden_dim

    def get_embedding_weight(self) -> torch.Tensor:
        """Return [V, D] token embedding matrix."""
        return self._model.get_embedding_weight()

    @property
    def device(self):
        return self._model.device

    @property
    def dtype(self):
        return self._model.dtype

    def to(self, device):
        self._model = self._model.to(device)
        return self

    def set_tau_r(self, tau_r: float) -> None:
        """Update routing temperature on the fly (for sweeps)."""
        self.tau_r = tau_r
        set_routing_temperature(self._model.model, tau_r)

    def set_soft_topk(self, soft_topk: int) -> None:
        """Update the number of active experts for soft routing."""
        self._soft_topk = soft_topk
        set_soft_topk(self._model.model, soft_topk)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def auxiliary_forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Fast forward: hard-routed MoE (~1.4B active) → [B, L, V] logits."""
        set_hard_routing(self._model.model)
        return self._model.forward(input_ids)

    @torch.no_grad()
    def auxiliary_forward_resp(
        self, input_ids: torch.LongTensor, resp_slice: slice,
    ) -> torch.Tensor:
        """Fast forward returning logits only for the requested response span."""
        return self.auxiliary_forward(input_ids)[:, resp_slice, :]

    @torch.no_grad()
    def primary_forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Slow forward: soft-routed MoE (~16B active) → [B, L, V] logits."""
        set_soft_routing(self._model.model)
        try:
            return self._model.forward(input_ids)
        finally:
            set_hard_routing(self._model.model)

    @torch.no_grad()
    def primary_forward_with_cache(
        self, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, object]:
        """Soft-routed forward returning logits and a reusable KV cache."""
        set_soft_routing(self._model.model)
        try:
            return self._model.forward_with_cache(input_ids)
        finally:
            set_hard_routing(self._model.model)

    @torch.no_grad()
    def primary_forward_replace_with_cache(
        self,
        full_input_ids: torch.LongTensor,
        replace_slice: slice,
        past_key_values: object,
    ) -> Tuple[torch.Tensor, object]:
        """Soft-routed partial recompute against an existing KV cache."""
        set_soft_routing(self._model.model)
        try:
            return self._model.forward_replace_with_cache(
                full_input_ids,
                replace_slice,
                past_key_values,
            )
        finally:
            set_hard_routing(self._model.model)

    @torch.no_grad()
    def primary_forward_with_hidden(
        self, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft-routed forward → (logits [B,L,V], hidden [B,L,D])."""
        set_soft_routing(self._model.model)
        try:
            return self._model.forward_with_hidden(input_ids)
        finally:
            set_hard_routing(self._model.model)

    @torch.no_grad()
    def primary_forward_with_all_hidden(
        self, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Soft-routed forward → (logits [B,L,V], all hidden states)."""
        set_soft_routing(self._model.model)
        try:
            return self._model.forward_with_all_hidden(input_ids)
        finally:
            set_hard_routing(self._model.model)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def dual_forward(
        self,
        input_ids: torch.LongTensor,
        need_hidden: bool = False,
        need_all_hidden: bool = False,
    ) -> DualModelOutput:
        """Run hard auxiliary then soft primary, compute agreement.

        Args:
            input_ids: [B, L] token ids.
            need_hidden: if True, also return primary hidden states.

        Returns:
            DualModelOutput with logits from both routing modes and agreement.
        """
        # Phase 0: Auxiliary draft (hard routing, fast)
        aux_logits = self.auxiliary_forward(input_ids)

        # Phase 1: Primary verification (soft routing, slow)
        if need_all_hidden:
            pri_logits, pri_hidden_states = self.primary_forward_with_all_hidden(input_ids)
            pri_hidden = pri_hidden_states[-1]
        elif need_hidden:
            pri_logits, pri_hidden = self.primary_forward_with_hidden(input_ids)
            pri_hidden_states = None
        else:
            pri_logits = self.primary_forward(input_ids)
            pri_hidden = None
            pri_hidden_states = None

        # Compute agreement: positions where argmax tokens match
        aux_tokens = aux_logits.argmax(dim=-1)  # [B, L]
        pri_tokens = pri_logits.argmax(dim=-1)  # [B, L]
        agreement = (aux_tokens == pri_tokens)   # [B, L] bool
        agreement_rate = agreement.float().mean().item()

        return DualModelOutput(
            primary_logits=pri_logits,
            auxiliary_logits=aux_logits,
            agreement=agreement,
            agreement_rate=agreement_rate,
            primary_hidden=pri_hidden,
            primary_hidden_states=pri_hidden_states,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def dual_forward_resp(
        self,
        input_ids: torch.LongTensor,
        resp_slice: slice,
        need_hidden: bool = False,
        need_all_hidden: bool = False,
    ) -> DualModelOutput:
        """Dual forward, returning logits only for the response region.

        This is a convenience for inference loops where we only need
        logits for positions [P:P+L_gen].
        """
        out = self.dual_forward(
            input_ids,
            need_hidden=need_hidden,
            need_all_hidden=need_all_hidden,
        )
        out.primary_logits = out.primary_logits[:, resp_slice, :]
        out.auxiliary_logits = out.auxiliary_logits[:, resp_slice, :]
        out.agreement = out.agreement[:, resp_slice]
        out.agreement_rate = out.agreement.float().mean().item()
        if out.primary_hidden is not None:
            out.primary_hidden = out.primary_hidden[:, resp_slice, :]
        if out.primary_hidden_states is not None:
            out.primary_hidden_states = [h[:, resp_slice, :] for h in out.primary_hidden_states]
        return out


def _deep_copy_cfg(cfg: dict) -> dict:
    """Deep copy a config dict."""
    import copy
    return copy.deepcopy(cfg)
