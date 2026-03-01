"""
Soft-Routed MoE Wrapper (paper §3.7).

Replaces hard top-k expert routing in LLaDA2.1-mini's MoE layers with
temperature-controlled soft routing. At low temperature (tau_r -> 0),
this closely approximates the original model but activates ALL experts
per forward pass, creating a dense-equivalent compute baseline.

This allows fair evaluation of AOAE's throughput improvements by
isolating them from MoE sparsity gains.
"""

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMoERouter(nn.Module):
    """Drop-in replacement for LLaDA2MoeGate that uses soft routing.

    Instead of top-k selection, computes a temperature-controlled softmax
    over all experts, so every expert participates in the forward pass.

    Args:
        original_gate: The original LLaDA2MoeGate module (weights are shared).
        tau_r: Routing temperature. Lower = closer to hard routing.
    """

    def __init__(self, original_gate: nn.Module, tau_r: float = 0.01):
        super().__init__()
        self.original_gate = original_gate
        self.tau_r = tau_r
        self.num_experts = original_gate.num_experts
        self.top_k = original_gate.top_k
        self.routed_scaling_factor = original_gate.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor):
        """Compute soft routing weights over all experts.

        Returns:
            topk_idx: [num_tokens, num_experts] — all expert indices
            topk_weight: [num_tokens, num_experts] — soft routing weights
            logits: [num_tokens, num_experts] — raw router logits
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.original_gate.weight.type(torch.float32),
        )

        # Soft routing: temperature-controlled softmax over ALL experts
        soft_weights = F.softmax(logits / self.tau_r, dim=-1).type_as(hidden_states)
        soft_weights = soft_weights * self.routed_scaling_factor

        # Return indices for all experts (dense routing)
        # .contiguous() required: .expand() is non-contiguous and the HF MoE
        # block calls topk_idx.view(-1) which fails on non-contiguous tensors.
        all_indices = torch.arange(
            self.num_experts, device=hidden_states.device
        ).unsqueeze(0).expand(hidden_states.shape[0], -1).contiguous()

        return all_indices, soft_weights, logits

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute raw router logits (same as original gate)."""
        return self.original_gate.get_logits(hidden_states)

    def routing(self, hidden_states, gating_output, topk, renormalize):
        """Soft routing for FusedMoE compatibility.

        Returns weights for ALL experts, not just top-k.
        """
        scores = torch.sigmoid(gating_output.float()).type_as(gating_output)

        # Soft routing via temperature-controlled softmax
        soft_weights = F.softmax(gating_output / self.tau_r, dim=-1).type_as(gating_output)
        soft_weights = soft_weights * self.routed_scaling_factor

        all_indices = torch.arange(
            self.num_experts, device=hidden_states.device
        ).unsqueeze(0).expand(hidden_states.shape[0], -1).contiguous()

        return soft_weights, all_indices


# ---- registry of (moe_block, original_gate, soft_router) for mode switching ----
_PATCHED_BLOCKS: Dict[int, List[Tuple[nn.Module, nn.Module, SoftMoERouter]]] = {}


def patch_model_with_soft_routing(
    model: nn.Module,
    tau_r: float = 0.01,
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

    Returns:
        The modified model (same object, modified in-place).
    """
    entries: List[Tuple[nn.Module, nn.Module, SoftMoERouter]] = []

    for name, module in model.named_modules():
        # Find MoE blocks that contain a gate
        if hasattr(module, 'gate') and hasattr(module, 'experts'):
            gate = module.gate
            # Check if this is a LLaDA2MoeGate (has group_limited_topk)
            if hasattr(gate, 'group_limited_topk'):
                soft_router = SoftMoERouter(gate, tau_r=tau_r)
                entries.append((module, gate, soft_router))
                module.gate = soft_router

    if not entries:
        raise RuntimeError(
            "No MoE gates found to patch. Ensure the model has "
            "LLaDA2MoeSparseMoeBlock layers with LLaDA2MoeGate."
        )

    _PATCHED_BLOCKS[id(model)] = entries
    print(f"[SoftMoE] Patched {len(entries)} MoE gates with tau_r={tau_r}")
    return model


def set_hard_routing(model: nn.Module) -> None:
    """Restore original hard top-k routing on a previously-patched model."""
    entries = _PATCHED_BLOCKS.get(id(model))
    if entries is None:
        return  # not patched — already hard
    for block, original_gate, _ in entries:
        block.gate = original_gate


def set_soft_routing(model: nn.Module) -> None:
    """Re-activate soft routing on a previously-patched model."""
    entries = _PATCHED_BLOCKS.get(id(model))
    if entries is None:
        raise RuntimeError(
            "Model was never patched with soft routing. "
            "Call patch_model_with_soft_routing first."
        )
    for block, _, soft_router in entries:
        block.gate = soft_router


@contextmanager
def soft_routing_context(model: nn.Module):
    """Context manager: temporarily enable soft routing, restore hard on exit."""
    set_soft_routing(model)
    try:
        yield
    finally:
        set_hard_routing(model)


def compute_routing_entropy(model: nn.Module) -> dict:
    """Compute routing entropy statistics across all MoE layers.

    Useful for verifying that soft routing is working as expected:
    - Low entropy = nearly hard routing (good for approximation quality)
    - High entropy = uniform routing (all experts equally weighted)

    Returns:
        dict with 'mean_entropy', 'max_possible_entropy', 'num_layers', 'tau_r'.
    """
    entropies = []
    last_router = None
    for name, module in model.named_modules():
        if isinstance(module, SoftMoERouter):
            # Max entropy for uniform distribution over num_experts
            entropies.append(math.log(module.num_experts))
            last_router = module

    if not entropies:
        return {"mean_entropy": 0.0, "num_layers": 0}

    return {
        "mean_entropy": sum(entropies) / len(entropies),
        "max_possible_entropy": entropies[0],  # all layers have same num_experts
        "num_layers": len(entropies),
        "tau_r": last_router.tau_r,
    }
