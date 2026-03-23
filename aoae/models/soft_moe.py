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
import types
import weakref
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

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

    def _group_limited_topk(self, scores_for_routing: torch.Tensor, k: int) -> torch.Tensor:
        original_top_k = getattr(self.original_gate, "top_k", None)
        if original_top_k is None or not hasattr(self.original_gate, "group_limited_topk"):
            return scores_for_routing.topk(k, dim=-1, sorted=False).indices

        self.original_gate.top_k = k
        try:
            _, topk_indices = self.original_gate.group_limited_topk(scores_for_routing)
        finally:
            self.original_gate.top_k = original_top_k
        return topk_indices

    def _compute_soft_topk(
        self, logits: torch.Tensor, hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a temperature-softened variant of the original gate.

        We preserve the original LLaDA gate semantics:
        - logits -> sigmoid scores
        - expert_bias only affects routing selection
        - group_limited_topk controls which experts are reachable

        Soft routing is introduced by temperature-scaling the logits before the
        sigmoid and by allowing a wider ``soft_topk`` selection set than the
        original hard gate.
        """
        del hidden_states
        scaled_logits = logits / self._tau_r
        scores = torch.sigmoid(scaled_logits.float()).type_as(logits)

        # Keep a normalized full routing distribution for diagnostics such as
        # routing entropy without affecting the execution-time router math.
        self._last_weights = (
            scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        ).detach()

        k = min(self._soft_topk, self.num_experts)
        if k >= self.num_experts:
            indices = torch.arange(
                self.num_experts, device=logits.device,
            ).unsqueeze(0).expand(logits.shape[0], -1).contiguous()
            full_weights = self._last_weights.type_as(logits) * self.routed_scaling_factor
            return full_weights, indices

        expert_bias = getattr(self.original_gate, "expert_bias", None)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
        else:
            scores_for_routing = scores
        topk_indices = self._group_limited_topk(scores_for_routing, k)
        topk_weights = torch.gather(scores, dim=1, index=topk_indices).type_as(logits)
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


class SGLangSoftTopKRouter(nn.Module):
    """Wrapper around SGLang's TopK selector that applies temperature scaling."""

    def __init__(
        self,
        original_topk: nn.Module,
        *,
        num_experts: int,
        tau_r: float = 0.01,
        soft_topk: Optional[int] = None,
        score_function: Optional[str] = None,
        top_k_override: Optional[int] = None,
    ):
        super().__init__()
        self.original_topk = original_topk
        self.num_experts = int(num_experts)
        _topk_val = getattr(original_topk, "top_k", None)
        if _topk_val is None:
            _topk_val = top_k_override
        if _topk_val is None:
            raise RuntimeError(
                "SGLangSoftTopKRouter: cannot determine top_k — "
                "TopK module has no 'top_k' attribute and no top_k_override given."
            )
        self.top_k = int(_topk_val)
        self.routed_scaling_factor = float(
            getattr(original_topk, "routed_scaling_factor", 1.0)
        )
        self.score_function = score_function
        self._tau_r = tau_r
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

    def _record_weights(self, scaled_logits: torch.Tensor) -> None:
        logits = scaled_logits.float()
        if self.score_function == "sigmoid":
            weights = torch.sigmoid(logits)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        else:
            weights = F.softmax(logits, dim=-1)
        self._last_weights = weights.detach()

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, *args, **kwargs):
        scaled_logits = router_logits / self._tau_r
        self._record_weights(scaled_logits)

        # Try to override top_k on the TopK module for soft_topk pruning.
        # Some SGLang TopK implementations don't expose top_k as a settable
        # attribute — in that case we still apply temperature scaling but skip
        # the top_k override (soft_topk pruning is a secondary optimisation).
        original_top_k = getattr(self.original_topk, "top_k", None)
        if original_top_k is not None:
            self.original_topk.top_k = min(self._soft_topk, self.num_experts)
        try:
            return self.original_topk(hidden_states, scaled_logits, *args, **kwargs)
        finally:
            if original_top_k is not None:
                self.original_topk.top_k = original_top_k


_PATCHED_BLOCKS: Dict[int, List[Dict[str, Any]]] = {}
_PATCHED_REFS: Dict[int, weakref.ref] = {}
_ROUTING_STATE: Dict[int, str] = {}  # "hard" or "soft"


def _cleanup_stale(model_id: int) -> None:
    ref = _PATCHED_REFS.get(model_id)
    if ref is not None and ref() is None:
        _PATCHED_BLOCKS.pop(model_id, None)
        _PATCHED_REFS.pop(model_id, None)
        _ROUTING_STATE.pop(model_id, None)


def _iter_known_moe_blocks(model: nn.Module):
    """Yield MoE blocks from explicit model layouts not always visible to named_modules()."""
    candidates = []

    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        layers = getattr(root, "layers", None)
        if layers is None:
            continue
        try:
            layer_iter = list(layers)
        except TypeError:
            continue
        for idx, layer in enumerate(layer_iter):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                candidates.append((f"{type(root).__name__}.layers[{idx}].mlp", mlp))

    seen = set()
    for name, block in candidates:
        block_id = id(block)
        if block_id in seen:
            continue
        seen.add(block_id)
        yield name, block


def _block_routing_function(
    block: nn.Module,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Route through the block's current gate.

    dInfer binds ``custom_routing_function`` to the original gate object when the
    MoE block is constructed. If we only swap ``block.gate``, the fused runtime
    continues to call the stale hard gate. Binding the callback to the block
    instead makes routing follow whichever gate is currently active.
    """
    return block.gate.routing(hidden_states, gating_output, topk=topk, renormalize=renormalize)


def _apply_gate_runtime_state(entry: Dict[str, Any]) -> None:
    block = entry["block"]
    experts = entry["experts"]
    soft_topk = entry["soft_topk"]

    if entry["original_block_top_k"] is not None:
        block.top_k = soft_topk

    if experts is None:
        return

    if entry["original_experts_top_k"] is not None:
        experts.top_k = soft_topk

    routing_fn = partial(_block_routing_function, block)
    if entry["original_custom_routing_function"] is not None:
        experts.custom_routing_function = routing_fn

    router = getattr(experts, "router", None)
    if router is not None:
        if entry["original_router_top_k"] is not None:
            router.top_k = soft_topk
        if entry["original_router_custom_routing_function"] is not None:
            router.custom_routing_function = routing_fn

    moe_config = getattr(experts, "moe_config", None)
    if moe_config is not None and entry["original_moe_config_experts_per_token"] is not None:
        moe_config.experts_per_token = soft_topk


def _restore_gate_runtime_state(entry: Dict[str, Any]) -> None:
    block = entry["block"]
    experts = entry["experts"]

    if entry["original_block_top_k"] is not None:
        block.top_k = entry["original_block_top_k"]

    if experts is None:
        return

    if entry["original_experts_top_k"] is not None:
        experts.top_k = entry["original_experts_top_k"]

    if entry["original_custom_routing_function"] is not None:
        experts.custom_routing_function = entry["original_custom_routing_function"]

    router = getattr(experts, "router", None)
    if router is not None:
        if entry["original_router_top_k"] is not None:
            router.top_k = entry["original_router_top_k"]
        if entry["original_router_custom_routing_function"] is not None:
            router.custom_routing_function = entry["original_router_custom_routing_function"]

    moe_config = getattr(experts, "moe_config", None)
    if moe_config is not None and entry["original_moe_config_experts_per_token"] is not None:
        moe_config.experts_per_token = entry["original_moe_config_experts_per_token"]


def _maybe_build_patch_entry(
    module: nn.Module,
    tau_r: float,
    soft_topk: Optional[int],
) -> Optional[Dict[str, Any]]:
    gate = getattr(module, "gate", None)
    if gate is not None and hasattr(gate, "group_limited_topk"):
        effective_topk = soft_topk
        if effective_topk is None:
            effective_topk = min(gate.top_k * 2, gate.num_experts)
        soft_router = SoftMoERouter(
            gate, tau_r=tau_r, soft_topk=effective_topk,
        )
        experts = getattr(module, "experts", None)
        return {
            "kind": "gate",
            "block": module,
            "experts": experts,
            "original_gate": gate,
            "soft_router": soft_router,
            "soft_topk": effective_topk,
            "original_block_top_k": getattr(module, "top_k", None),
            "original_experts_top_k": getattr(experts, "top_k", None) if experts is not None else None,
            "original_custom_routing_function": (
                getattr(experts, "custom_routing_function", None)
                if experts is not None else None
            ),
            "original_router_top_k": (
                getattr(getattr(experts, "router", None), "top_k", None)
                if experts is not None else None
            ),
            "original_router_custom_routing_function": (
                getattr(getattr(experts, "router", None), "custom_routing_function", None)
                if experts is not None else None
            ),
            "original_moe_config_experts_per_token": (
                getattr(getattr(experts, "moe_config", None), "experts_per_token", None)
                if experts is not None else None
            ),
        }

    topk = getattr(module, "topk", None)
    if gate is not None and topk is not None:
        # Try to get top_k from the TopK module itself; fall back to the parent
        # block's self.top_k (LLaDA2SparseMoeBlock always stores it).
        _topk_on_topk = getattr(topk, "top_k", None)
        hard_topk_val = _topk_on_topk if _topk_on_topk is not None else getattr(module, "top_k", None)
        if hard_topk_val is not None:
            effective_topk = soft_topk
            num_experts = int(getattr(module, "num_experts", getattr(gate, "num_experts", 0)))
            hard_topk = int(hard_topk_val)
            if effective_topk is None:
                effective_topk = min(hard_topk * 2, num_experts)
            soft_router = SGLangSoftTopKRouter(
                topk,
                num_experts=num_experts,
                tau_r=tau_r,
                soft_topk=effective_topk,
                score_function=getattr(module, "score_function", None),
                top_k_override=hard_topk,
            )
            return {
                "kind": "topk",
                "block": module,
                "original_topk": module.topk,
                "soft_router": soft_router,
            }

    return None


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
    entries: List[Dict[str, Any]] = []
    patched_block_ids = set()

    def _append_entry(entry: Dict[str, Any]) -> None:
        block_id = id(entry["block"])
        if block_id in patched_block_ids:
            return
        patched_block_ids.add(block_id)
        entries.append(entry)
        if entry["kind"] == "gate":
            entry["block"].gate = entry["soft_router"]
            _apply_gate_runtime_state(entry)
        elif entry["kind"] == "topk":
            entry["block"].topk = entry["soft_router"]

    for _, module in model.named_modules():
        entry = _maybe_build_patch_entry(module, tau_r=tau_r, soft_topk=soft_topk)
        if entry is not None:
            _append_entry(entry)

    for _, module in _iter_known_moe_blocks(model):
        entry = _maybe_build_patch_entry(module, tau_r=tau_r, soft_topk=soft_topk)
        if entry is not None:
            _append_entry(entry)

    if not entries:
        known_block_types = []
        for _, module in _iter_known_moe_blocks(model):
            known_block_types.append(type(module).__name__)
        raise RuntimeError(
            "No MoE gates found to patch. Ensure the model has "
            "LLaDA2/LLaDA-MoE sparse blocks with gate/topk modules. "
            f"Known explicit blocks seen: {known_block_types[:8]}"
        )

    mid = id(model)
    _PATCHED_BLOCKS[mid] = entries
    _PATCHED_REFS[mid] = weakref.ref(model)
    _ROUTING_STATE[mid] = "soft"
    print(f"[SoftMoE] Patched {len(entries)} MoE gates with "
          f"tau_r={tau_r}, soft_topk={entries[0]['soft_router'].soft_topk}")
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
    for entry in entries:
        if entry["kind"] == "gate":
            entry["block"].gate = entry["original_gate"]
            _restore_gate_runtime_state(entry)
        elif entry["kind"] == "topk":
            entry["block"].topk = entry["original_topk"]
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
    for entry in entries:
        if entry["kind"] == "gate":
            entry["block"].gate = entry["soft_router"]
            _apply_gate_runtime_state(entry)
        elif entry["kind"] == "topk":
            entry["block"].topk = entry["soft_router"]
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
    for entry in entries:
        entry["soft_router"].tau_r = tau_r


def set_soft_topk(model: nn.Module, soft_topk: int) -> None:
    """Change the number of active experts on all patched SoftMoERouters."""
    _cleanup_stale(id(model))
    entries = _PATCHED_BLOCKS.get(id(model))
    if entries is None:
        raise RuntimeError(
            "Model was never patched with soft routing. "
            "Call patch_model_with_soft_routing first."
        )
    for entry in entries:
        entry["soft_router"].soft_topk = soft_topk
        if entry["kind"] == "gate":
            entry["soft_topk"] = soft_topk
            if _ROUTING_STATE.get(id(model)) == "soft":
                _apply_gate_runtime_state(entry)


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
    last_router: Optional[nn.Module] = None
    _cleanup_stale(id(model))
    entries = _PATCHED_BLOCKS.get(id(model), [])
    for entry in entries:
        module = entry["soft_router"]
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
    mean_ent = sum(valid) / len(valid) if valid else 0.0
    return {
        "mean_entropy": mean_ent,
        "max_possible_entropy": math.log(last_router.num_experts),
        "num_layers": len(layer_entropies),
        "num_layers_with_data": len(valid),
        "tau_r": last_router.tau_r,
        "layer_entropies": layer_entropies,
    }
