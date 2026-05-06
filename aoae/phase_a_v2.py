"""Phase A V2 position policy utilities.

This module is intentionally narrow: it implements the V4 u_t/r_t path without
cache, access, boundary, or stable-KV decisions.  The policy scores sequence
positions only; it never predicts vocabulary logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


UNLABELED = -1.0


def logit_from_rate(rate: float) -> float:
    rate = min(max(float(rate), 1e-6), 1.0 - 1e-6)
    return math.log(rate / (1.0 - rate))


def extract_unmask_candidates(mask_indicator: torch.BoolTensor) -> torch.BoolTensor:
    """u_t candidates are exactly currently masked positions."""
    return mask_indicator.bool()


def extract_remask_candidates(
    mask_indicator: torch.BoolTensor,
    verifier_eligible: Optional[torch.BoolTensor] = None,
) -> torch.BoolTensor:
    """r_t candidates are verifier-eligible positions that are currently unmasked."""
    candidates = ~mask_indicator.bool()
    if verifier_eligible is not None:
        candidates = candidates & verifier_eligible.bool()
    return candidates


def apply_forced_rejections(
    response_tokens: torch.Tensor,
    mask_token_id: int,
    forced_rejects: torch.BoolTensor,
) -> torch.Tensor:
    """Deterministically remask forced verifier rejects before learned r_t acts."""
    out = response_tokens.clone()
    out[forced_rejects.bool()] = int(mask_token_id)
    return out


def apply_safe_remask(
    response_tokens: torch.Tensor,
    mask_token_id: int,
    forced_rejects: torch.BoolTensor,
    learned_remask: torch.BoolTensor,
    remask_exec_candidates: torch.BoolTensor,
) -> torch.Tensor:
    """Apply safe verifier semantics.

    Forced rejects always win.  Learned r_t can only add remasks inside the
    execution candidate domain, which should already exclude forced rejects
    after they have been remasked.
    """
    out = apply_forced_rejections(response_tokens, mask_token_id, forced_rejects)
    learned = learned_remask.bool() & remask_exec_candidates.bool()
    out[learned] = int(mask_token_id)
    return out


def _clean_feature(value: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(like, dtype=torch.float32)
    return torch.nan_to_num(
        value.to(device=like.device, dtype=torch.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


def build_phase_a_features(
    *,
    head: str,
    confidence: torch.Tensor,
    step_frac: float,
    agreement: Optional[torch.Tensor] = None,
    age: Optional[torch.Tensor] = None,
    frontier_membership: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, bool]]:
    """Build fixed-width scalar features for u_t or r_t candidates.

    Feature order:
      confidence, agreement, step_frac, age, frontier_membership

    For drafter u_t states, fresh verifier agreement/frontier membership are
    not treated as meaningful.  They are zero-filled and marked unavailable.
    """
    h = str(head).lower()
    if h not in {"u", "u_t", "unmask", "r", "r_t", "remask"}:
        raise ValueError(f"Unsupported Phase A head: {head!r}")
    is_u = h in {"u", "u_t", "unmask"}
    base = confidence.to(dtype=torch.float32)
    B, L = base.shape
    step = torch.full((B, L), float(step_frac), device=base.device, dtype=torch.float32)
    conf = _clean_feature(confidence, base).clamp(0.0, 1.0)
    age_f = _clean_feature(age, base)
    if is_u:
        agree = torch.zeros_like(base, dtype=torch.float32)
        frontier = torch.zeros_like(base, dtype=torch.float32)
        availability = {
            "confidence": True,
            "step_frac": True,
            "age": age is not None,
            "agreement": False,
            "frontier_membership": False,
        }
    else:
        agree = _clean_feature(agreement, base).clamp(0.0, 1.0)
        frontier = _clean_feature(frontier_membership, base).clamp(0.0, 1.0)
        availability = {
            "confidence": True,
            "step_frac": True,
            "age": age is not None,
            "agreement": agreement is not None,
            "frontier_membership": frontier_membership is not None,
        }
    feats = torch.stack([conf, agree, step, age_f, frontier], dim=-1)
    return feats, availability


@dataclass
class WarmStartLabels:
    labels: torch.Tensor
    weights: torch.Tensor


def build_unmask_labels(
    *,
    candidate_mask: torch.BoolTensor,
    heuristic_selected: torch.BoolTensor,
    verifier_accepted: torch.BoolTensor,
    soon_remasked_or_changed: Optional[torch.BoolTensor] = None,
    confidence: Optional[torch.Tensor] = None,
    low_confidence_threshold: float = 0.15,
    accepted_weight: float = 3.0,
    accepted_soon_changed_weight: float = 1.0,
    rejected_weight: float = 1.0,
    low_conf_unselected_weight: float = 3.0,
    soft_changed_positive: bool = True,
) -> WarmStartLabels:
    """Construct partial u_t labels.

    Unselected medium/high-confidence candidates remain unlabeled by default.

    Weight rationale:
      - low_conf_unselected (label=0, w=3.0): strong negative — the heuristic
        would never unmask these (conf << tau=0.7); policy must not either.
      - rejected (label=0, w=1.0): weak negative — heuristic made a reasonable
        choice (conf > tau) that happened to be wrong downstream; don't
        over-penalise the same reasonable mistake.
      - stable_accept (label=1, w=3.0): strong positive — heuristic correct
        and verifier confirmed.
      - changed_accept (label=1, w=1.0): kept as soft positive; accepted but
        later remasked so signal is mixed.
    """
    labels = torch.full(candidate_mask.shape, UNLABELED, device=candidate_mask.device)
    weights = torch.zeros(candidate_mask.shape, device=candidate_mask.device, dtype=torch.float32)
    selected = candidate_mask.bool() & heuristic_selected.bool()
    accepted = selected & verifier_accepted.bool()
    rejected = selected & ~verifier_accepted.bool()
    changed = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if soon_remasked_or_changed is not None:
        changed = soon_remasked_or_changed.bool()

    stable_accept = accepted & ~changed
    changed_accept = accepted & changed
    labels[stable_accept] = 1.0
    weights[stable_accept] = accepted_weight
    labels[changed_accept] = 1.0 if soft_changed_positive else 0.5
    weights[changed_accept] = accepted_soon_changed_weight
    labels[rejected] = 0.0
    weights[rejected] = rejected_weight

    if confidence is not None:
        low_conf = confidence.to(device=candidate_mask.device).float() < float(low_confidence_threshold)
        low_unselected = candidate_mask.bool() & ~heuristic_selected.bool() & low_conf
        labels[low_unselected] = 0.0
        weights[low_unselected] = low_conf_unselected_weight

    return WarmStartLabels(labels=labels, weights=weights)


def build_remask_labels(
    *,
    candidate_mask: torch.BoolTensor,
    forced_rejects: torch.BoolTensor,
    accepted_frontier: torch.BoolTensor,
    stable_kept: Optional[torch.BoolTensor] = None,
    later_remasked_or_changed: Optional[torch.BoolTensor] = None,
    forced_weight: float = 4.0,
    accepted_weight: float = 3.0,
    stable_weight: float = 1.0,
    later_changed_weight: float = 1.5,
) -> WarmStartLabels:
    """Construct partial r_t labels for the training rollback domain.

    Weight rationale:
      - forced (label=1, w=4.0): strongest positive — verifier oracle says remask.
      - accepted (label=0, w=3.0): strong negative — verifier confirmed good token.
      - stable_kept (label=0, w=1.0): moderate negative — old unmasked tokens the
        verifier didn't touch this step. Remasking these causes unnecessary thrashing
        and slower decoding, so the policy should learn to leave them alone. Kept
        below accepted_weight so forced-reject signal still dominates.
    """
    labels = torch.full(candidate_mask.shape, UNLABELED, device=candidate_mask.device)
    weights = torch.zeros(candidate_mask.shape, device=candidate_mask.device, dtype=torch.float32)
    candidates = candidate_mask.bool()

    forced = candidates & forced_rejects.bool()
    accepted = candidates & accepted_frontier.bool()
    labels[forced] = 1.0
    weights[forced] = forced_weight
    labels[accepted] = 0.0
    weights[accepted] = accepted_weight

    if stable_kept is not None:
        stable = candidates & stable_kept.bool() & ~forced & ~accepted
        labels[stable] = 0.0
        weights[stable] = stable_weight
    if later_remasked_or_changed is not None:
        changed = candidates & later_remasked_or_changed.bool() & ~forced & ~accepted
        labels[changed] = 1.0
        weights[changed] = later_changed_weight

    return WarmStartLabels(labels=labels, weights=weights)


def weighted_bce_ignore_unlabeled(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    labeled = labels >= 0.0
    if valid_mask is not None:
        labeled = labeled & valid_mask.bool()
    if not bool(labeled.any()):
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        logits[labeled],
        labels[labeled].to(dtype=logits.dtype),
        reduction="none",
    )
    w = weights[labeled].to(dtype=logits.dtype)
    return (loss * w).sum() / w.sum().clamp(min=1e-8)


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: Optional[torch.BoolTensor] = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """Within-row logistic ranking loss for positive labels over negatives."""
    active = (labels >= 0.0) & (weights > 0.0)
    if valid_mask is not None:
        active = active & valid_mask.bool()
    losses: List[torch.Tensor] = []
    for b in range(logits.shape[0]):
        pos = active[b] & (labels[b] > 0.5)
        neg = active[b] & (labels[b] < 0.5)
        if not bool(pos.any() and neg.any()):
            continue
        diff = logits[b, pos].unsqueeze(-1) - logits[b, neg].unsqueeze(0)
        losses.append(F.softplus(float(margin) - diff).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def rate_prior_loss(
    probs: torch.Tensor,
    valid_mask: torch.BoolTensor,
    target_rate: float,
) -> torch.Tensor:
    valid = valid_mask.bool()
    if not bool(valid.any()):
        return probs.sum() * 0.0
    rate = probs[valid].mean()
    target = torch.tensor(float(target_rate), device=probs.device, dtype=probs.dtype)
    return (rate - target).pow(2)


def phase_a_supervised_loss(
    policy_out: Dict[str, torch.Tensor],
    u_labels: WarmStartLabels,
    r_labels: WarmStartLabels,
    *,
    lambda_rank: float = 0.0,
    lambda_rate: float = 0.0,
    lambda_hidden_gate: float = 0.0,
    target_u_rate: float = 0.10,
    target_r_rate: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    u_valid = policy_out["unmask_valid_mask"].bool()
    r_valid = policy_out["remask_valid_mask"].bool()
    u_bce = weighted_bce_ignore_unlabeled(
        policy_out["unmask_logits"], u_labels.labels, u_labels.weights, u_valid
    )
    r_bce = weighted_bce_ignore_unlabeled(
        policy_out["remask_logits"], r_labels.labels, r_labels.weights, r_valid
    )
    rank = pairwise_ranking_loss(
        policy_out["unmask_logits"], u_labels.labels, u_labels.weights, u_valid
    )
    rank = rank + pairwise_ranking_loss(
        policy_out["remask_logits"], r_labels.labels, r_labels.weights, r_valid
    )
    rate = rate_prior_loss(policy_out["unmask_probs"], u_valid, target_u_rate)
    rate = rate + rate_prior_loss(policy_out["remask_probs"], r_valid, target_r_rate)
    hidden_gate = policy_out.get("hidden_gate_regularizer")
    if hidden_gate is None:
        hidden_gate = u_bce.sum() * 0.0
    total = (
        u_bce
        + r_bce
        + float(lambda_rank) * rank
        + float(lambda_rate) * rate
        + float(lambda_hidden_gate) * hidden_gate
    )
    metrics = {
        "warmstart/u_bce": u_bce.detach(),
        "warmstart/r_bce": r_bce.detach(),
        "warmstart/rank_loss": rank.detach(),
        "warmstart/rate_loss": rate.detach(),
        "warmstart/u_pos_rate": policy_out["unmask_probs"][u_valid].mean().detach()
        if bool(u_valid.any()) else torch.tensor(0.0, device=u_bce.device),
        "warmstart/r_pos_rate": policy_out["remask_probs"][r_valid].mean().detach()
        if bool(r_valid.any()) else torch.tensor(0.0, device=u_bce.device),
    }
    return total, metrics


def bernoulli_log_prob_scoped(
    probs: torch.Tensor,
    actions: torch.Tensor,
    valid_mask: torch.BoolTensor,
) -> torch.Tensor:
    p = probs.clamp(1e-7, 1.0 - 1e-7)
    a = actions.to(dtype=p.dtype)
    lp = a * torch.log(p) + (1.0 - a) * torch.log1p(-p)
    return (lp * valid_mask.to(dtype=p.dtype)).sum(dim=-1)


def action_stats(
    policy_out: Dict[str, torch.Tensor],
    actions: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    stats: Dict[str, torch.Tensor] = {}
    for prefix, action_key, prob_key, valid_key in (
        ("u", "u_t", "unmask_probs", "unmask_valid_mask"),
        ("r", "r_t", "remask_probs", "remask_valid_mask"),
    ):
        valid = policy_out[valid_key].bool()
        probs = policy_out[prob_key]
        acts = actions.get(action_key, torch.zeros_like(probs))
        denom = valid.float().sum().clamp(min=1.0)
        p_valid = probs[valid] if bool(valid.any()) else probs.new_zeros((1,))
        stats[f"{prefix}/action_rate"] = (acts.float() * valid.float()).sum() / denom
        stats[f"{prefix}/mean_prob"] = probs[valid].mean() if bool(valid.any()) else probs.sum() * 0.0
        entropy = -(p_valid.clamp(1e-7, 1 - 1e-7) * torch.log(p_valid.clamp(1e-7, 1 - 1e-7))
                    + (1 - p_valid).clamp(1e-7, 1 - 1e-7) * torch.log((1 - p_valid).clamp(1e-7, 1 - 1e-7)))
        stats[f"{prefix}/entropy"] = entropy.mean()
        stats[f"{prefix}/num_candidates"] = valid.float().sum()
        invalid = acts.float() * (~valid).float()
        stats[f"{prefix}/invalid_action_rate"] = invalid.sum() / acts.numel()
    return stats


class PhaseAV2Policy(nn.Module):
    """Two-head Phase A position-space policy for u_t/r_t."""

    scalar_dim = 5

    def __init__(self, cfg: dict, input_dim: int):
        super().__init__()
        pc = cfg.get("policy", {})
        raw_v2c = cfg.get("phase_a_v2_config", {}) or {}
        v2c = raw_v2c if isinstance(raw_v2c, dict) else {}
        d_model = int(pc.get("d_model", 128))
        n_heads = int(pc.get("n_heads", 4))
        n_layers = int(pc.get("n_layers", 1))
        dropout = float(pc.get("dropout", 0.0))
        self.feature_mode = str(v2c.get("feature_mode", pc.get("feature_mode", "scalar_only")))
        if self.feature_mode not in {"scalar_only", "hidden_residual", "v5_hybrid"}:
            raise ValueError("feature_mode must be scalar_only, hidden_residual, or v5_hybrid")
        self.target_u_rate = float(v2c.get("target_u_rate", pc.get("target_u_rate", 0.10)))
        self.target_r_rate = float(v2c.get("target_r_rate", pc.get("target_r_rate", 0.02)))

        self.input_proj = nn.Linear(self.scalar_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head_unmask = nn.Linear(d_model, 1)
        self.head_remask = nn.Linear(d_model, 1)

        # --- hidden_residual mode (legacy E_t / emb_t path) ---
        self.hidden_norm = nn.LayerNorm(input_dim)
        self.hidden_proj = nn.Linear(input_dim, d_model)
        self.hidden_delta_unmask = nn.Linear(d_model, 1)
        self.hidden_delta_remask = nn.Linear(d_model, 1)
        self.gate_u = nn.Parameter(torch.tensor(-8.0))
        self.gate_r = nn.Parameter(torch.tensor(-8.0))

        # --- v5_hybrid mode ---
        # u_t: aux_h_final (drafter's last hidden state) gated residual
        # r_t: pri_h_final (primary's last hidden state) gated residual
        # hidden_dim defaults to input_dim if not specified (same for LLaDA-mini).
        v5_hidden_dim = int(v2c.get("hidden_dim", pc.get("hidden_dim", input_dim)))
        # u_t branch: drafter hidden state
        self.aux_hidden_norm = nn.LayerNorm(v5_hidden_dim)
        self.aux_hidden_proj = nn.Linear(v5_hidden_dim, d_model)
        self.aux_hidden_delta = nn.Linear(d_model, 1)
        self.gate_aux_u = nn.Parameter(torch.tensor(-8.0))
        # r_t branch: primary hidden state
        self.pri_hidden_norm = nn.LayerNorm(v5_hidden_dim)
        self.pri_hidden_proj = nn.Linear(v5_hidden_dim, d_model)
        self.pri_hidden_delta = nn.Linear(d_model, 1)
        self.gate_pri_r = nn.Parameter(torch.tensor(-8.0))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.head_unmask.bias, logit_from_rate(self.target_u_rate))
        nn.init.constant_(self.head_remask.bias, logit_from_rate(self.target_r_rate))
        # hidden_residual: zero-init delta heads so mode starts as scalar_only
        nn.init.zeros_(self.hidden_delta_unmask.weight)
        nn.init.zeros_(self.hidden_delta_unmask.bias)
        nn.init.zeros_(self.hidden_delta_remask.weight)
        nn.init.zeros_(self.hidden_delta_remask.bias)
        # v5_hybrid: zero-init projection heads so mode starts as scalar_only
        nn.init.zeros_(self.aux_hidden_proj.weight)
        nn.init.zeros_(self.aux_hidden_proj.bias)
        nn.init.zeros_(self.aux_hidden_delta.weight)
        nn.init.zeros_(self.aux_hidden_delta.bias)
        nn.init.zeros_(self.pri_hidden_proj.weight)
        nn.init.zeros_(self.pri_hidden_proj.bias)
        nn.init.zeros_(self.pri_hidden_delta.weight)
        nn.init.zeros_(self.pri_hidden_delta.bias)

    def forward(
        self,
        H_t: torch.Tensor,
        mask_indicator: torch.BoolTensor,
        step_frac: float,
        temperature: float = 1.0,
        confidence: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        age_feature: Optional[torch.Tensor] = None,
        frontier_membership: Optional[torch.Tensor] = None,
        remask_candidate_mask: Optional[torch.BoolTensor] = None,
        quality_scores: Optional[torch.Tensor] = None,
        last_action_feature: Optional[torch.Tensor] = None,
        aux_h_final: Optional[torch.Tensor] = None,
        pri_h_final: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del quality_scores, last_action_feature
        B, L, _ = H_t.shape
        device = H_t.device
        if confidence is None:
            confidence = torch.zeros(B, L, device=device)
        u_valid = extract_unmask_candidates(mask_indicator)
        r_valid = (
            remask_candidate_mask.bool()
            if remask_candidate_mask is not None
            else extract_remask_candidates(mask_indicator)
        )
        u_feats, u_avail = build_phase_a_features(
            head="u",
            confidence=confidence,
            step_frac=step_frac,
            age=age_feature,
        )
        r_feats, r_avail = build_phase_a_features(
            head="r",
            confidence=confidence,
            step_frac=step_frac,
            agreement=agreement,
            age=age_feature,
            frontier_membership=frontier_membership,
        )
        scalar_in = torch.cat([u_feats, r_feats], dim=1)
        x_scalar = self.backbone(self.input_proj(scalar_in))
        x_u, x_r = x_scalar[:, :L], x_scalar[:, L:]
        z_u_scalar = self.head_unmask(x_u).squeeze(-1)
        z_r_scalar = self.head_remask(x_r).squeeze(-1)

        delta_u = torch.zeros_like(z_u_scalar)
        delta_r = torch.zeros_like(z_r_scalar)
        gate_u = torch.sigmoid(self.gate_u)
        gate_r = torch.sigmoid(self.gate_r)
        
        # --- hidden_residual mode (legacy E_t / emb_t path) ---
        if self.feature_mode == "hidden_residual":
            h = self.hidden_proj(self.hidden_norm(H_t))
            x_hidden = self.backbone(torch.cat([self.input_proj(u_feats) + h, self.input_proj(r_feats) + h], dim=1))
            h_u, h_r = x_hidden[:, :L], x_hidden[:, L:]
            delta_u = self.hidden_delta_unmask(h_u).squeeze(-1)
            delta_r = self.hidden_delta_remask(h_r).squeeze(-1)
        
        # --- v5_hybrid mode ---
        elif self.feature_mode == "v5_hybrid":
            # u_t: aux_h_final (drafter's last hidden state) for u_t
            delta_u_v5 = torch.zeros_like(z_u_scalar)
            if aux_h_final is not None:
                # aux_h_final is the drafter's last hidden state (transformer hidden space)
                u_proj = self.aux_hidden_proj(self.aux_hidden_norm(aux_h_final))
                delta_u_v5 = self.aux_hidden_delta(u_proj).squeeze(-1)
            gate_aux_u = torch.sigmoid(self.gate_aux_u)
            delta_u = gate_aux_u * delta_u_v5
            
            # r_t: pri_h_final (primary's last hidden state) for r_t
            delta_r_v5 = torch.zeros_like(z_r_scalar)
            if pri_h_final is not None:
                # pri_h_final is the primary's last hidden state (transformer hidden space)
                r_proj = self.pri_hidden_proj(self.pri_hidden_norm(pri_h_final))
                delta_r_v5 = self.pri_hidden_delta(r_proj).squeeze(-1)
            gate_pri_r = torch.sigmoid(self.gate_pri_r)
            delta_r = gate_pri_r * delta_r_v5
        
        z_u = z_u_scalar + gate_u * delta_u
        z_r = z_r_scalar + gate_r * delta_r
        z_u = z_u.masked_fill(~u_valid, -1e9)
        z_r = z_r.masked_fill(~r_valid, -1e9)
        temp = max(float(temperature), 1e-6)
        p_u = torch.sigmoid(z_u / temp)
        p_r = torch.sigmoid(z_r / temp)
        zeros = torch.zeros(B, L, device=device, dtype=p_u.dtype)
        out = {
            "unmask_logits": z_u,
            "remask_logits": z_r,
            "cache_logits": torch.full_like(z_u, -1e9),
            "access_logits": torch.full_like(z_u, -1e9),
            "unmask_probs": p_u,
            "remask_probs": p_r,
            "cache_probs": zeros,
            "access_probs": zeros,
            "unmask_valid_mask": u_valid,
            "remask_valid_mask": r_valid,
            "feature_availability_u": u_avail,
            "feature_availability_r": r_avail,
            "hidden/gate_u": gate_u.detach(),
            "hidden/gate_r": gate_r.detach(),
            "hidden/delta_logit_norm_u": delta_u.detach().norm(),
            "hidden/delta_logit_norm_r": delta_r.detach().norm(),
            "hidden_gate_regularizer": gate_u.pow(2) + gate_r.pow(2),
            "v5/gate_aux_u": gate_aux_u.detach() if self.feature_mode == "v5_hybrid" else torch.tensor(0.0),
            "v5/gate_pri_r": gate_pri_r.detach() if self.feature_mode == "v5_hybrid" else torch.tensor(0.0),
        }
        return out

    def sample_actions(
        self,
        policy_out: Dict[str, torch.Tensor],
        mask_indicator: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        del mask_indicator
        u = torch.bernoulli(policy_out["unmask_probs"]) * policy_out["unmask_valid_mask"].float()
        r = torch.bernoulli(policy_out["remask_probs"]) * policy_out["remask_valid_mask"].float()
        zeros = torch.zeros_like(u)
        return {"u_t": u, "r_t": r, "kappa_t": zeros, "q_t": zeros}

    def log_prob(
        self,
        policy_out: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        include_heads: Optional[set] = None,
    ) -> torch.Tensor:
        include = {str(h) for h in include_heads} if include_heads is not None else None
        total = torch.zeros(
            actions["u_t"].shape[0],
            device=actions["u_t"].device,
            dtype=policy_out["unmask_probs"].dtype,
        )
        if include is None or include.intersection({"u", "u_t", "unmask"}):
            total = total + bernoulli_log_prob_scoped(
                policy_out["unmask_probs"], actions["u_t"], policy_out["unmask_valid_mask"]
            )
        if include is None or include.intersection({"r", "r_t", "remask"}):
            total = total + bernoulli_log_prob_scoped(
                policy_out["remask_probs"], actions["r_t"], policy_out["remask_valid_mask"]
            )
        return total


class PhaseAHeuristicExpert:
    """Canonical threshold expert over the same u_t/r_t domains."""

    def __init__(self, draft_threshold: float = 0.7, remask_threshold: float = 0.5):
        self.draft_threshold = float(draft_threshold)
        self.remask_threshold = float(remask_threshold)

    def actions(
        self,
        *,
        confidence: torch.Tensor,
        mask_indicator: torch.BoolTensor,
        remask_candidate_mask: Optional[torch.BoolTensor] = None,
        forced_rejects: Optional[torch.BoolTensor] = None,
        agreement: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        u_valid = extract_unmask_candidates(mask_indicator)
        r_valid = (
            remask_candidate_mask.bool()
            if remask_candidate_mask is not None
            else extract_remask_candidates(mask_indicator)
        )
        u = (confidence >= self.draft_threshold).float() * u_valid.float()
        r = torch.zeros_like(u)
        if forced_rejects is not None:
            r = torch.maximum(r, forced_rejects.bool().float() * r_valid.float())
        if agreement is not None:
            r = torch.maximum(
                r,
                ((agreement.float() < self.remask_threshold) & r_valid).float(),
            )
        zeros = torch.zeros_like(u)
        return {"u_t": u, "r_t": r, "kappa_t": zeros, "q_t": zeros}
