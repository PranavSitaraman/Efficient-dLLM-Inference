"""
AOAE Policy Network (paper §3.2, §3.4).

A lightweight 1-layer bidirectional transformer with four independent
Bernoulli output heads (unmask, remask, cache, access).  Validity constraints are
enforced via logit masking before the sigmoid.

Architecture follows Jazbec et al. (2025) "Learning Unmasking Policies
for Diffusion Language Models" — extended from 1 head to 4.

Key change from earlier AOAE formulation: the "edit" head (T2T replacement)
is replaced by a "remask" head that simply reverts positions to [M],
preserving the any-order property of masked diffusion models.
"""

import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def _safe_policy_temperature(temperature: float) -> float:
    """Return a strictly positive finite temperature for probability heads."""
    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        return 1e-6
    if not math.isfinite(temp) or temp <= 0.0:
        return 1e-6
    return temp


def _validate_bernoulli_probs(name: str, probs: torch.Tensor) -> None:
    """Raise a clear error before CUDA Bernoulli kernels trip a device assert."""
    finite_mask = torch.isfinite(probs)
    in_range_mask = (probs >= 0.0) & (probs <= 1.0)
    valid_mask = finite_mask & in_range_mask
    if bool(valid_mask.all()):
        return

    finite_vals = probs[finite_mask]
    if finite_vals.numel() > 0:
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
    nan_count = int(torch.isnan(probs).sum().item())
    posinf_count = int(torch.isposinf(probs).sum().item())
    neginf_count = int(torch.isneginf(probs).sum().item())
    out_of_range_count = int((finite_mask & ~in_range_mask).sum().item())
    raise RuntimeError(
        f"Invalid Bernoulli probabilities in {name}: "
        f"nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, "
        f"out_of_range={out_of_range_count}, finite_min={min_val}, finite_max={max_val}"
    )


def call_policy(
    policy,
    H_t: torch.Tensor,
    mask_indicator: torch.BoolTensor,
    step_frac: float,
    **kwargs,
):
    """Call a policy while keeping backward compatibility with older call signatures."""
    target = policy.module if hasattr(policy, "module") else policy
    forward_fn = getattr(target, "forward", None)
    candidate = forward_fn if callable(forward_fn) else getattr(target, "__call__", None)
    try:
        params = inspect.signature(candidate).parameters if candidate is not None else {}
    except (TypeError, ValueError):
        params = {}
    if "confidence" not in params:
        kwargs.pop("confidence", None)
    return policy(H_t, mask_indicator, step_frac, **kwargs)


class AOAEPolicy(nn.Module):
    """
    Policy pi_phi(a_t | s_t) with factorized Bernoulli likelihood.

    Input per position:
      (h_t^k [D], m_t^k [1], q_t^k [1], alpha_t^k [1], t/T [1], age_t^k [1], last_q_t^k [1])
      → projected to d_model.
    Backbone:            N-layer bidirectional transformer.
    Output:              4 scalar logits per position (unmask, remask, cache, access).
    """

    def __init__(self, cfg, input_dim: int):
        """
        Args:
            cfg:       full config dict.
            input_dim: dimension D of soft-masked embeddings h_t^k.
        """
        super().__init__()
        pc = cfg["policy"]
        d = pc["d_model"]
        self.d_model = d

        self.use_positional_features = bool(pc.get("use_positional_features", False))
        self.use_agreement_feature = bool(pc.get("use_agreement_feature", True))
        self.use_age_feature = bool(pc.get("use_age_feature", self.use_positional_features))
        self.use_last_action_feature = bool(pc.get("use_last_action_feature", self.use_positional_features))
        self.boundary_cfg = pc.get("boundary_head", {})
        self.boundary_enabled = bool(self.boundary_cfg.get("enabled", False))
        self.boundary_num_bins = max(2, int(self.boundary_cfg.get("num_bins", 8)))
        extra_feats = 3 + int(self.use_agreement_feature) + int(self.use_age_feature) + int(self.use_last_action_feature)
        # --- Input projection: base + optional positional features ---
        self.input_proj = nn.Linear(input_dim + extra_feats, d)

        # --- Transformer backbone (bidirectional) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=pc["n_heads"],
            dim_feedforward=d * 4,
            dropout=pc["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer, num_layers=pc["n_layers"]
        )

        # --- Four independent output heads → scalar logit per position ---
        self.head_unmask = nn.Linear(d, 1)
        self.head_remask = nn.Linear(d, 1)
        self.head_cache = nn.Linear(d, 1)
        self.head_access = nn.Linear(d, 1)  # next-H positional access head
        self.head_boundary = nn.Linear(d, self.boundary_num_bins) if self.boundary_enabled else None

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Small init so early policy is roughly uniform (logit ≈ 0)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        H_t: torch.Tensor,
        mask_indicator: torch.BoolTensor,
        step_frac: float,
        temperature: float = 1.0,
        confidence: Optional[torch.Tensor] = None,
        quality_scores: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        age_feature: Optional[torch.Tensor] = None,
        last_action_feature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action logits with validity constraints.

        Args:
            H_t:            [B, L, D]  soft-masked embeddings.
            mask_indicator: [B, L]     True where token is [M].
            step_frac:      scalar     t / T.
            temperature:    policy temperature tau_pi.
            confidence:     [B, L]     optional per-position primary confidence.
                            Reserved for heuristic fallback policies; ignored here.
            quality_scores: [B, L]     PRISM quality scores (0=bad, 1=good).
                            If None, defaults to zeros.
            agreement:      [B, L]     auxiliary-primary agreement (0/1 float).
                            If None, defaults to zeros.
            age_feature:    [B, L]     normalized positional age feature.
            last_action_feature: [B, L] previous-step access action.

        Returns:
            dict with keys:
                "unmask_logits":  [B, L]  (masked to -inf for unmasked positions)
                "remask_logits":  [B, L]  (masked to -inf for masked positions)
                "cache_logits":   [B, L]
                "access_logits":  [B, L]
                "unmask_probs":   [B, L]  sigmoid(logit / tau_pi)
                "remask_probs":   [B, L]
                "cache_probs":    [B, L]
                "access_probs":   [B, L]
        """
        B, L, D = H_t.shape
        device = H_t.device
        del confidence  # Kept for interface parity with heuristic fallback policies.
        temp = _safe_policy_temperature(temperature)

        # --- Build per-position input features ---
        m_feat = mask_indicator.float().unsqueeze(-1)              # [B, L, 1]
        if quality_scores is not None:
            q_feat = quality_scores.unsqueeze(-1)                  # [B, L, 1]
        else:
            q_feat = torch.zeros(B, L, 1, device=device)          # [B, L, 1]
        t_feat = torch.full((B, L, 1), step_frac, device=device)  # [B, L, 1]
        feats = [H_t, m_feat, q_feat, t_feat]
        if self.use_agreement_feature:
            if agreement is not None:
                a_feat = agreement.unsqueeze(-1)                   # [B, L, 1]
            else:
                a_feat = torch.zeros(B, L, 1, device=device)      # [B, L, 1]
            feats.append(a_feat)
        if self.use_age_feature:
            if age_feature is not None:
                age_feat = age_feature.unsqueeze(-1)
            else:
                age_feat = torch.zeros(B, L, 1, device=device)
            feats.append(age_feat)
        if self.use_last_action_feature:
            if last_action_feature is not None:
                last_feat = last_action_feature.unsqueeze(-1)
            else:
                last_feat = torch.zeros(B, L, 1, device=device)
            feats.append(last_feat)
        x = torch.cat(feats, dim=-1)
        x = self.input_proj(x)                                     # [B, L, d]

        # --- Transformer backbone ---
        x = self.backbone(x)                                       # [B, L, d]

        # --- Head logits ---
        unmask_logits = self.head_unmask(x).squeeze(-1)  # [B, L]
        remask_logits = self.head_remask(x).squeeze(-1)  # [B, L]
        cache_logits = self.head_cache(x).squeeze(-1)    # [B, L]
        access_logits = self.head_access(x).squeeze(-1)  # [B, L]
        boundary_logits = None
        boundary_probs = None
        if self.boundary_enabled and self.head_boundary is not None:
            pooled = x.mean(dim=1)  # [B, d]
            boundary_logits = self.head_boundary(pooled)  # [B, num_bins]
            boundary_probs = F.softmax(boundary_logits / temp, dim=-1)

        # --- Validity constraints via logit masking ---
        # Unmask only on masked positions
        unmask_logits = unmask_logits.masked_fill(~mask_indicator, -1e9)
        # Remask only on unmasked positions
        remask_logits = remask_logits.masked_fill(mask_indicator, -1e9)
        # Cache-remask exclusion is enforced at sampling time (see sample_actions)

        # --- Tempered probabilities ---
        unmask_probs = torch.sigmoid(unmask_logits / temp)
        remask_probs = torch.sigmoid(remask_logits / temp)
        cache_probs = torch.sigmoid(cache_logits / temp)
        access_probs = torch.sigmoid(access_logits / temp)

        out = {
            "unmask_logits": unmask_logits,
            "remask_logits": remask_logits,
            "cache_logits": cache_logits,
            "access_logits": access_logits,
            "unmask_probs": unmask_probs,
            "remask_probs": remask_probs,
            "cache_probs": cache_probs,
            "access_probs": access_probs,
        }
        if boundary_logits is not None and boundary_probs is not None:
            out["boundary_logits"] = boundary_logits
            out["boundary_probs"] = boundary_probs
        return out

    # ------------------------------------------------------------------
    def sample_actions(
        self,
        policy_out: Dict[str, torch.Tensor],
        mask_indicator: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample binary actions from Bernoulli distributions.

        Enforces cache-remask exclusion: kappa_t^k * r_t^k = 0.

        Returns:
            dict with "u_t", "r_t", "kappa_t", "q_t" — each [B, L] binary.
        """
        _validate_bernoulli_probs("unmask_probs", policy_out["unmask_probs"])
        _validate_bernoulli_probs("remask_probs", policy_out["remask_probs"])
        _validate_bernoulli_probs("cache_probs", policy_out["cache_probs"])
        _validate_bernoulli_probs("access_probs", policy_out["access_probs"])
        u_t = torch.bernoulli(policy_out["unmask_probs"])    # [B, L]
        r_t = torch.bernoulli(policy_out["remask_probs"])    # [B, L]
        kappa_t = torch.bernoulli(policy_out["cache_probs"])  # [B, L]
        q_t = torch.bernoulli(policy_out["access_probs"])     # [B, L]
        ell_t = None
        if "boundary_probs" in policy_out:
            ell_t = torch.multinomial(policy_out["boundary_probs"], num_samples=1).squeeze(-1)

        # Enforce cache-remask exclusion: if remask is 1, cache must be 0
        kappa_t = kappa_t * (1.0 - r_t)

        out = {"u_t": u_t, "r_t": r_t, "kappa_t": kappa_t, "q_t": q_t}
        if ell_t is not None:
            out["ell_t"] = ell_t
        return out

    # ------------------------------------------------------------------
    def log_prob(
        self,
        policy_out: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log pi_phi(a_t | s_t) = sum over positions and heads of
        Bernoulli log-likelihoods.

        Args:
            policy_out: dict from forward().
            actions:    dict from sample_actions().

        Returns:
            log_prob: [B] scalar per sample.
        """
        total = torch.zeros(actions["u_t"].shape[0], device=actions["u_t"].device)

        heads = [
            ("u_t", "unmask_probs"),
            ("r_t", "remask_probs"),
            ("kappa_t", "cache_probs"),
            ("q_t", "access_probs"),
        ]
        for key, prob_key in heads:
            if key not in actions or prob_key not in policy_out:
                continue
            a = actions[key]        # [B, L]
            p = policy_out[prob_key].clamp(1e-7, 1.0 - 1e-7)  # [B, L]
            lp = a * torch.log(p) + (1.0 - a) * torch.log(1.0 - p)  # [B, L]
            # Mandatory q_t positions are deterministic (forced include), so skip
            # their Bernoulli contribution when provided by the caller.
            if key == "q_t" and "q_t_mandatory" in actions:
                lp = lp * (1.0 - actions["q_t_mandatory"].float())
            total = total + lp.sum(dim=-1)  # [B]

        if "ell_t" in actions and "boundary_probs" in policy_out:
            probs = policy_out["boundary_probs"].clamp(1e-7, 1.0)
            idx = actions["ell_t"].long().unsqueeze(-1)
            lp_b = torch.log(torch.gather(probs, dim=-1, index=idx).squeeze(-1))
            total = total + lp_b

        return total


class DefaultPolicy(nn.Module):
    """Heuristic policy for speculative eval without GRPO training.

    Uses primary-model confidence and draft/primary agreement as a deterministic,
    training-free fallback when no learned AOAE checkpoint is available.

    Behavior:
      - Unmask: masked positions where confidence > tau_mask
      - Remask: never (all zeros)
      - Cache:  positions where auxiliary and primary agree
      - Access: same as cache when positional caching is enabled

    Args:
        tau_mask: confidence threshold for unmasking (default 0.7 = S-mode).
    """

    def __init__(self, tau_mask: float = 0.7, num_steps: int = 8):
        super().__init__()
        self.tau_mask = tau_mask
        self._num_steps = num_steps

    def forward(
        self,
        H_t: torch.Tensor,
        mask_indicator: torch.BoolTensor,
        step_frac: float,
        temperature: float = 1.0,
        confidence: Optional[torch.Tensor] = None,
        quality_scores: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        age_feature: Optional[torch.Tensor] = None,
        last_action_feature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L, D = H_t.shape
        device = H_t.device
        del step_frac, temperature, quality_scores, age_feature, last_action_feature, D

        if confidence is None:
            confidence = torch.zeros(B, L, device=device)
        else:
            confidence = confidence.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if agreement is None:
            cache_probs = torch.zeros(B, L, device=device)
        else:
            cache_probs = agreement.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        access_probs = cache_probs

        unmask_probs = mask_indicator.float() * (confidence > self.tau_mask).float()

        # Never remask
        remask_probs = torch.zeros(B, L, device=device)

        return {
            "unmask_logits": torch.zeros(B, L, device=device),
            "remask_logits": torch.full((B, L), -1e9, device=device),
            "cache_logits": torch.zeros(B, L, device=device),
            "access_logits": torch.zeros(B, L, device=device),
            "unmask_probs": unmask_probs,
            "remask_probs": remask_probs,
            "cache_probs": cache_probs,
            "access_probs": access_probs,
            "boundary_logits": torch.zeros(B, 2, device=device),
            "boundary_probs": torch.full((B, 2), 0.5, device=device),
        }

    def sample_actions(
        self,
        policy_out: Dict[str, torch.Tensor],
        mask_indicator: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        unmask_probs = policy_out["unmask_probs"]
        B = unmask_probs.shape[0]
        u_t = (unmask_probs > 0.5).float() * mask_indicator.float()
        r_t = torch.zeros_like(u_t)
        kappa_t = (policy_out["cache_probs"].clamp(0.0, 1.0) > 0.5).float()
        q_t = (policy_out["access_probs"].clamp(0.0, 1.0) > 0.5).float()
        ell_t = torch.zeros(B, dtype=torch.long, device=u_t.device)
        return {"u_t": u_t, "r_t": r_t, "kappa_t": kappa_t, "q_t": q_t, "ell_t": ell_t}

    def log_prob(
        self,
        policy_out: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.zeros(actions["u_t"].shape[0], device=actions["u_t"].device)
