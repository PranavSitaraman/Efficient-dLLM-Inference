"""
Utilities for positional speculative caching (next-H access prediction).

This module provides:
  - Per-position temporal state (age, last action).
  - Access-set construction with mandatory inclusion of unmask/remask edits.
  - Refresh-budget enforcement for speculative access positions.
  - Next-H access quality metrics (precision/recall/F1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch


def _pcfg(cfg: dict) -> Dict[str, object]:
    ic = cfg.get("inference", {})
    pc = ic.get("positional_cache", {})
    return {
        "enabled": bool(pc.get("enabled", False)),
        "horizon": int(pc.get("horizon", 4)),
        "refresh_budget": int(pc.get("refresh_budget", 0)),
        "use_topb_from_probs": bool(pc.get("use_topb_from_probs", True)),
        "force_mandatory": bool(pc.get("force_mandatory", True)),
        "age_cap": max(1, int(pc.get("age_cap", 64))),
        "candidate_policy": str(pc.get("candidate_policy", "learned_topb")),
        "window_radius": max(1, int(pc.get("window_radius", 8))),
    }


def init_positional_state(batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "age": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        "last_action": torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device),
    }


def get_policy_positional_features(state: Dict[str, torch.Tensor], cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    pc = _pcfg(cfg)
    age = state["age"].float() / float(pc["age_cap"])
    age = age.clamp(0.0, 1.0)
    last_action = state["last_action"].float()
    return age, last_action


def _topb_non_mandatory(
    scores: torch.Tensor,
    mandatory: torch.BoolTensor,
    budget: Union[int, torch.Tensor],
    candidate: Optional[torch.BoolTensor] = None,
) -> torch.BoolTensor:
    """
    Select up to B non-mandatory positions per sample from highest scores.
    """
    B, L = scores.shape
    if isinstance(budget, torch.Tensor):
        per_sample_budget = budget.to(device=scores.device).long().view(-1)
        if per_sample_budget.numel() != B:
            raise ValueError(
                f"Per-sample budget must have shape [B], got {tuple(per_sample_budget.shape)} for B={B}."
            )
    else:
        per_sample_budget = torch.full((B,), int(budget), dtype=torch.long, device=scores.device)
    selected = torch.zeros_like(mandatory)
    if int(per_sample_budget.max().item()) <= 0:
        return selected

    for b in range(B):
        budget_b = int(per_sample_budget[b].item())
        if budget_b <= 0:
            continue
        eligible_mask = ~mandatory[b]
        if candidate is not None:
            eligible_mask = eligible_mask & candidate[b]
        eligible = eligible_mask.nonzero(as_tuple=True)[0]
        if eligible.numel() == 0:
            continue
        k = min(budget_b, int(eligible.numel()))
        vals = scores[b, eligible]
        topk_idx = vals.topk(k=k).indices
        chosen = eligible[topk_idx]
        selected[b, chosen] = True
    return selected


def build_access_set(
    actions: Dict[str, torch.Tensor],
    policy_out: Dict[str, torch.Tensor],
    cfg: dict,
    confidence: Optional[torch.Tensor] = None,
    boundary_action: Optional[torch.Tensor] = None,
    boundary_num_bins: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Build executed access set Q_t from sampled actions and policy access scores.

    Returns:
        q_exec: [B, L] float in {0,1}
        mandatory: [B, L] float in {0,1}
        diagnostics: scalar metrics for logging
    """
    pc = _pcfg(cfg)

    u_t = actions.get("u_t")
    r_t = actions.get("r_t")
    q_t = actions.get("q_t")
    if q_t is None:
        q_t = torch.zeros_like(u_t)
    q_raw = q_t.bool()

    mandatory = torch.zeros_like(q_raw)
    if pc["force_mandatory"]:
        mandatory = u_t.bool() | r_t.bool()

    if not pc["enabled"]:
        # Disabled mode should preserve legacy behavior:
        # no positional gating on refresh/commit decisions.
        q_exec = torch.ones_like(q_raw)
        # Mark all positions mandatory so q_t does not affect policy log-prob.
        mandatory = torch.ones_like(q_raw)
        return q_exec.float(), mandatory.float(), {
            "access_rate": 0.0,
            "access_mandatory_rate": 0.0,
            "access_optional_rate": 0.0,
            "access_budget_utilization": 0.0,
            "access_effective_budget": 0.0,
        }
    else:
        budget = int(pc["refresh_budget"])
        per_sample_budget = torch.full((q_raw.shape[0],), budget, dtype=torch.long, device=q_raw.device)
        if budget > 0 and boundary_action is not None:
            bsz = q_raw.shape[0]
            ell = boundary_action.long().view(-1)
            if ell.numel() != bsz:
                raise ValueError(
                    f"boundary_action must have shape [B], got {tuple(boundary_action.shape)} for B={bsz}."
                )
            bins = int(boundary_num_bins) if boundary_num_bins is not None else int(torch.max(ell).item() + 1)
            bins = max(2, bins)
            # Higher boundary depth allows larger optional refresh budget.
            frac = (ell.float() + 1.0) / float(bins)
            per_sample_budget = torch.round(frac * float(budget)).long().clamp(min=0, max=budget)
        scores = policy_out.get("access_probs", q_t.float())
        candidate_policy = str(pc["candidate_policy"]).lower()

        if candidate_policy == "learned_topb":
            if pc["use_topb_from_probs"] and budget > 0:
                speculative = _topb_non_mandatory(scores, mandatory, per_sample_budget, candidate=q_raw)
                q_exec = mandatory | speculative
            else:
                q_exec = q_raw | mandatory
                if budget > 0:
                    keep = _topb_non_mandatory(scores, mandatory, per_sample_budget, candidate=q_exec)
                    q_exec = mandatory | keep
        elif candidate_policy == "confidence_topb":
            conf = confidence if confidence is not None else scores
            if budget > 0:
                speculative = _topb_non_mandatory(conf, mandatory, per_sample_budget, candidate=None)
                q_exec = mandatory | speculative
            else:
                q_exec = mandatory
        elif candidate_policy == "sliding_window":
            B, L = q_raw.shape
            radius = int(pc["window_radius"])
            candidate = torch.zeros_like(q_raw)
            mandatory_idx = mandatory.nonzero(as_tuple=False)
            if mandatory_idx.numel() > 0:
                for b, pos in mandatory_idx:
                    lo = max(0, int(pos.item()) - radius)
                    hi = min(L, int(pos.item()) + radius + 1)
                    candidate[int(b.item()), lo:hi] = True
            else:
                candidate[:] = True
            if budget > 0:
                speculative = _topb_non_mandatory(scores, mandatory, per_sample_budget, candidate=candidate)
                q_exec = mandatory | speculative
            else:
                q_exec = mandatory
        else:
            raise ValueError(
                f"Unknown positional_cache.candidate_policy={candidate_policy!r}. "
                "Choose from: learned_topb, sliding_window, confidence_topb."
            )

    optional = q_exec & ~mandatory
    total_budget = float(max(int(per_sample_budget.sum().item()), 1))
    diagnostics = {
        "access_rate": float(q_exec.float().mean().item()),
        "access_mandatory_rate": float(mandatory.float().mean().item()),
        "access_optional_rate": float(optional.float().mean().item()),
        "access_budget_utilization": float(optional.sum().item()) / total_budget if int(per_sample_budget.max().item()) > 0 else 0.0,
        "access_effective_budget": float(per_sample_budget.float().mean().item()),
    }
    return q_exec.float(), mandatory.float(), diagnostics


def update_positional_state(
    state: Dict[str, torch.Tensor],
    q_exec: torch.Tensor,
    changed: torch.Tensor,
    cfg: dict,
) -> None:
    pc = _pcfg(cfg)
    age_cap = int(pc["age_cap"])
    access_mask = (q_exec.bool() | changed.bool())
    state["age"] = (state["age"] + 1).clamp(max=age_cap)
    state["age"][access_mask] = 0
    state["last_action"] = q_exec.float()


def _prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    f1 = 2.0 * p * r / max(p + r, 1e-8)
    return p, r, f1


def compute_next_h_access_metrics(
    access_exec_steps: List[torch.Tensor],
    changed_steps: List[torch.Tensor],
    mandatory_steps: Optional[List[torch.Tensor]],
    horizon: int,
) -> Dict[str, float]:
    """
    Evaluate access prediction quality against realized edits in next H steps.
    """
    if not access_exec_steps or not changed_steps:
        return {
            "access_next_h_precision": 0.0,
            "access_next_h_recall": 0.0,
            "access_next_h_f1": 0.0,
            "access_next_h_spec_precision": 0.0,
            "access_next_h_spec_recall": 0.0,
            "access_next_h_spec_f1": 0.0,
        }

    n = min(len(access_exec_steps), len(changed_steps))
    h = max(1, int(horizon))
    overall_tp = overall_fp = overall_fn = 0.0
    spec_tp = spec_fp = spec_fn = 0.0

    for t in range(n):
        end = min(n, t + h)
        future = torch.zeros_like(changed_steps[t], dtype=torch.bool)
        for j in range(t, end):
            future = future | changed_steps[j].bool()

        pred = access_exec_steps[t].bool()
        mand = torch.zeros_like(pred)
        if mandatory_steps is not None and t < len(mandatory_steps):
            mand = mandatory_steps[t].bool()

        # Overall access quality
        tp = (pred & future).sum().item()
        fp = (pred & ~future).sum().item()
        fn = (~pred & future).sum().item()
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        # Speculative-only (exclude mandatory edits)
        pred_s = pred & ~mand
        future_s = future & ~mand
        tp_s = (pred_s & future_s).sum().item()
        fp_s = (pred_s & ~future_s).sum().item()
        fn_s = (~pred_s & future_s).sum().item()
        spec_tp += tp_s
        spec_fp += fp_s
        spec_fn += fn_s

    p, r, f1 = _prf(overall_tp, overall_fp, overall_fn)
    ps, rs, f1s = _prf(spec_tp, spec_fp, spec_fn)
    return {
        "access_next_h_precision": float(p),
        "access_next_h_recall": float(r),
        "access_next_h_f1": float(f1),
        "access_next_h_spec_precision": float(ps),
        "access_next_h_spec_recall": float(rs),
        "access_next_h_spec_f1": float(f1s),
    }
