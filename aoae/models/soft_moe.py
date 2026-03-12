"""
Soft-Routed MoE Wrapper (paper §3.7).

Replaces hard top-k expert routing in LLaDA2.1-mini's MoE layers with
temperature-controlled soft routing. Uses top-K_soft pruning after the
temperature-scaled softmax to keep only the most relevant experts active,
dramatically reducing FusedMoE compute while preserving soft routing
characteristics. At K_soft=num_experts this is dense-equivalent; at
K_soft=top_k this is equivalent to hard routing with soft weights.
"""

import math
import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMoERouter(nn.Module):
    """Drop-in replacement for LLaDA2MoeGate with top-K pruned soft routing.

    Computes a temperature-controlled softmax over all experts, then keeps
    only the top ``soft_topk`` experts per token. This gives the distributional
    benefit of soft routing (smooth weight gradients, better agreement signal)
    while running the FusedMoE kernel on K_soft << num_experts.

    Args:
        original_gate: The original LLaDA2MoeGate module (weights are shared).
        tau_r: Routing temperature. Lower = closer to hard routing.
        soft_topk: Number of experts to keep after pruning. None = all experts.
    """

    def __init__(
        self,
        original_gate: nn.Module,
        tau_r: float = 0.01,
        soft_topk: Optional[int] = None,
    ):
        super().__init__()
        self.original_gate = original_gate
        self._tau_r = tau_r
        self.num_experts = original_gate.num_experts
        self.top_k = original_gate.top_k
        self.routed_scaling_factor = original_gate.routed_scaling_factor
        self._soft_topk = soft_topk if soft_topk is not None else self.num_experts
        self._last_weights: Optional[torch.Tensor] = None

    @property
    def tau_r(self) -> float:
        return self._tau_r

    @tau_r.setter
    def tau_r(self, value: float):
        if value <= 0:
            raise ValueError(f"tau_r must be positive, got {value}")
        self._tau_r = value

    @property
    def soft_topk(self) -> int:
        return self._soft_topk

    @soft_topk.setter
    def soft_topk(self, value: int):
        if value < 1 or value > self.num_experts:
            raise ValueError(
                f"soft_topk must be in [1, {self.num_experts}], got {value}"
            )
        self._soft_topk = value

    def _compute_soft_topk(
        self, logits: torch.Tensor, hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute temperature-softmax weights, then prune to top-K_soft."""
        full_weights = F.softmax(logits / self._tau_r, dim=-1).type_as(hidden_states)
        if self.training:
            self._last_weights = full_weights.detach()

        k = min(self._soft_topk, self.num_experts)
        if k >= self.num_experts:
            indices = torch.arange(
                self.num_experts, device=hidden_states.device,
            ).unsqueeze(0).expand(hidden_states.shape[0], -1).contiguous()
            return full_weights * self.routed_scaling_factor, indices

        topk_weights, topk_indices = full_weights.topk(k, dim=-1, sorted=False)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.contiguous(), topk_indices.contiguous()

    def forward(self, hidden_states: torch.Tensor):
        """Compute soft routing weights with top-K_soft pruning.

        Returns:
            topk_idx: [num_tokens, K_soft] expert indices
            topk_weight: [num_tokens, K_soft] soft routing weights
            logits: [num_tokens, num_experts] raw router logits
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.original_gate.weight.type(torch.float32),
        )
        weights, indices = self._compute_soft_topk(logits, hidden_states)
        return indices, weights, logits

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute raw router logits (same as original gate)."""
        return self.original_gate.get_logits(hidden_states)

    def routing(self, hidden_states, gating_output, topk, renormalize):
        """Soft routing for FusedMoE compatibility with top-K_soft pruning."""
        weights, indices = self._compute_soft_topk(gating_output, hidden_states)
        return weights, indices


_PATCHED_BLOCKS: Dict[int, List[Tuple[nn.Module, nn.Module, SoftMoERouter]]] = {}
_PATCHED_REFS: Dict[int, weakref.ref] = {}
_ROUTING_STATE: Dict[int, str] = {}  # "hard" or "soft"


def _cleanup_stale(model_id: int) -> None:
    ref = _PATCHED_REFS.get(model_id)
    if ref is not None and ref() is None:
        _PATCHED_BLOCKS.pop(model_id, None)
        _PATCHED_REFS.pop(model_id, None)
        _ROUTING_STATE.pop(model_id, None)


def patch_model_with_soft_routing(
    model: nn.Module,
    tau_r: float = 0.01,
    soft_topk: Optional[int] = None,
) -> nn.Module:
    """Patch all MoE gates in a LLaDA2 MoE model with soft routing.

    This modifies the model in-place, replacing each LLaDA2MoeGate with
    a SoftMoERouter wrapper. The original gate weights are preserved
    (shared by reference) so that hard routing can be restored later
    via :func:`set_hard_routing` / :func:`set_soft_routing`.

    Works with both HF-loaded and dInfer-loaded LLaDA2 MoE models.

    Args:
        model: A LLaDA2 MoE model with LLaDA2MoeSparseMoeBlock layers.
        tau_r: Routing temperature for soft routing.
        soft_topk: Number of experts to keep after pruning (None = 2 * top_k).

    Returns:
        The modified model (same object, modified in-place).
    """
    entries: List[Tuple[nn.Module, nn.Module, SoftMoERouter]] = []

    for name, module in model.named_modules():
        if hasattr(module, 'gate') and hasattr(module, 'experts'):
            gate = module.gate
            if hasattr(gate, 'group_limited_topk'):
                effective_topk = soft_topk
                if effective_topk is None:
                    effective_topk = min(gate.top_k * 2, gate.num_experts)
                soft_router = SoftMoERouter(
                    gate, tau_r=tau_r, soft_topk=effective_topk,
                )
                entries.append((module, gate, soft_router))
                module.gate = soft_router

    if not entries:
        raise RuntimeError(
            "No MoE gates found to patch. Ensure the model has "
            "LLaDA2MoeSparseMoeBlock layers with LLaDA2MoeGate."
        )

    mid = id(model)
    _PATCHED_BLOCKS[mid] = entries
    _PATCHED_REFS[mid] = weakref.ref(model)
    _ROUTING_STATE[mid] = "soft"
    print(f"[SoftMoE] Patched {len(entries)} MoE gates with "
          f"tau_r={tau_r}, soft_topk={entries[0][2].soft_topk}")
    return model


def set_hard_routing(model: nn.Module) -> None:
    """Restore original hard top-k routing on a previously-patched model."""
    mid = id(model)
    if _ROUTING_STATE.get(mid) == "hard":
        return
    _cleanup_stale(mid)
    entries = _PATCHED_BLOCKS.get(mid)
    if entries is None:
        return
    for block, original_gate, _ in entries:
        block.gate = original_gate
    _ROUTING_STATE[mid] = "hard"


def set_soft_routing(model: nn.Module) -> None:
    """Re-activate soft routing on a previously-patched model."""
    mid = id(model)
    if _ROUTING_STATE.get(mid) == "soft":
        return
    _cleanup_stale(mid)
    entries = _PATCHED_BLOCKS.get(mid)
    if entries is None:
        raise RuntimeError(
            "Model was never patched with soft routing. "
            "Call patch_model_with_soft_routing first."
        )
    for block, _, soft_router in entries:
        block.gate = soft_router
    _ROUTING_STATE[mid] = "soft"


def set_routing_temperature(model: nn.Module, tau_r: float) -> None:
    """Change routing temperature on all patched SoftMoERouters in-place."""
    _cleanup_stale(id(model))
    entries = _PATCHED_BLOCKS.get(id(model))
    if entries is None:
        raise RuntimeError(
            "Model was never patched with soft routing. "
            "Call patch_model_with_soft_routing first."
        )
    for _, _, soft_router in entries:
        soft_router.tau_r = tau_r


def set_soft_topk(model: nn.Module, soft_topk: int) -> None:
    """Change the number of active experts on all patched SoftMoERouters."""
    _cleanup_stale(id(model))
    entries = _PATCHED_BLOCKS.get(id(model))
    if entries is None:
        raise RuntimeError(
            "Model was never patched with soft routing. "
            "Call patch_model_with_soft_routing first."
        )
    for _, _, soft_router in entries:
        soft_router.soft_topk = soft_topk


@contextmanager
def soft_routing_context(model: nn.Module):
    """Context manager: temporarily enable soft routing, restore hard on exit."""
    set_soft_routing(model)
    try:
        yield
    finally:
        set_hard_routing(model)


def compute_routing_entropy(model: nn.Module) -> dict:
    """Compute routing entropy from the last forward pass of each SoftMoERouter.

    Requires a forward pass to have been run first so that ``_last_weights``
    is populated. Returns the *actual* per-token routing entropy averaged
    across layers, not the theoretical maximum.

    Returns:
        dict with 'mean_entropy', 'max_possible_entropy', 'num_layers',
        'tau_r', and per-layer 'layer_entropies'.
    """
    layer_entropies: List[float] = []
    last_router: Optional[SoftMoERouter] = None
    for _name, module in model.named_modules():
        if isinstance(module, SoftMoERouter):
            last_router = module
            w = module._last_weights  # [num_tokens, num_experts] from last fwd
            if w is None:
                layer_entropies.append(float("nan"))
                continue
            # Renormalize in case routed_scaling_factor was applied before storage
            p = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            ent = -(p * (p + 1e-12).log()).sum(dim=-1)  # [num_tokens]
            layer_entropies.append(ent.mean().item())

    if not layer_entropies:
        return {"mean_entropy": 0.0, "num_layers": 0}

    valid = [e for e in layer_entropies if not math.isnan(e)]
    mean_ent = sum(valid) / len(valid) if valid else float("nan")
    return {
        "mean_entropy": mean_ent,
        "max_possible_entropy": math.log(last_router.num_experts),
        "num_layers": len(layer_entropies),
        "tau_r": last_router.tau_r,
        "layer_entropies": layer_entropies,
    }
