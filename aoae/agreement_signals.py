"""
Training-free draft/primary agreement signals for speculative KV reuse.

These signals can be used as "safe-to-reuse" gates in place of strict
argmax-match, enabling POC2 sweeps without policy retraining.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def _js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    eps = 1e-8
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = m.clamp(min=eps)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _topk_overlap(
    primary_logits: torch.Tensor,
    auxiliary_logits: torch.Tensor,
    top_k: int,
    min_overlap: int,
) -> torch.Tensor:
    pri_idx = torch.topk(primary_logits, k=top_k, dim=-1).indices
    aux_idx = torch.topk(auxiliary_logits, k=top_k, dim=-1).indices
    overlap = (pri_idx.unsqueeze(-1) == aux_idx.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    return overlap >= min_overlap


def compute_reuse_signal(
    primary_logits: torch.Tensor,
    auxiliary_logits: torch.Tensor,
    cfg: dict,
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Compute per-position safe-to-reuse signal.

    Returns:
        safe_reuse: [B, L] float in {0,1}
        new_state: state dict for temporal signals (currently streak)
        diagnostics: aggregate scalar diagnostics for logging
    """
    ic = cfg.get("inference", {})
    sc = ic.get("reuse_signal", {})
    method = sc.get("method", "argmax_match")
    threshold = float(sc.get("threshold", 0.0))
    top_k = int(sc.get("top_k", 4))
    min_overlap = int(sc.get("min_overlap", 1))
    min_streak = int(sc.get("min_streak", 2))

    pri_probs = F.softmax(primary_logits, dim=-1)
    aux_probs = F.softmax(auxiliary_logits, dim=-1)
    pri_conf, pri_tok = pri_probs.max(dim=-1)
    aux_conf, aux_tok = aux_probs.max(dim=-1)
    match = (pri_tok == aux_tok)
    pri_margin = _top2_margin(primary_logits)
    aux_margin = _top2_margin(auxiliary_logits)
    js = _js_divergence(primary_logits, auxiliary_logits)

    if state is None or "match_streak" not in state:
        match_streak = torch.zeros_like(pri_conf, dtype=torch.long)
    else:
        match_streak = state["match_streak"]
    match_streak = torch.where(match, match_streak + 1, torch.zeros_like(match_streak))

    if method == "argmax_match":
        safe = match
    elif method == "topk_overlap":
        safe = _topk_overlap(primary_logits, auxiliary_logits, top_k=top_k, min_overlap=min_overlap)
    elif method == "min_confidence":
        safe = match & (torch.minimum(pri_conf, aux_conf) >= threshold)
    elif method == "min_margin":
        safe = match & (torch.minimum(pri_margin, aux_margin) >= threshold)
    elif method == "js_divergence":
        safe = js <= threshold
    elif method == "temporal_confidence":
        stable = match_streak >= min_streak
        confident = torch.minimum(pri_conf, aux_conf) >= threshold
        safe = match & stable & confident
    else:
        raise ValueError(
            f"Unknown reuse_signal.method='{method}'. "
            "Choose from: argmax_match, topk_overlap, min_confidence, "
            "min_margin, js_divergence, temporal_confidence."
        )

    diagnostics = {
        "mean_match": float(match.float().mean().item()),
        "mean_safe_reuse": float(safe.float().mean().item()),
        "mean_js_divergence": float(js.mean().item()),
        "mean_min_conf": float(torch.minimum(pri_conf, aux_conf).mean().item()),
        "mean_min_margin": float(torch.minimum(pri_margin, aux_margin).mean().item()),
        "mean_streak": float(match_streak.float().mean().item()),
    }
    return safe.float(), {"match_streak": match_streak}, diagnostics

