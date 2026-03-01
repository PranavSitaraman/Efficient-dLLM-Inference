"""
AOAE Policy Network (paper §3.2, §3.4).

A lightweight 1-layer bidirectional transformer with three independent
Bernoulli output heads (unmask, remask, cache).  Validity constraints are
enforced via logit masking before the sigmoid.

Architecture follows Jazbec et al. (2025) "Learning Unmasking Policies
for Diffusion Language Models" — extended from 1 head to 3.

Key change from earlier AOAE formulation: the "edit" head (T2T replacement)
is replaced by a "remask" head that simply reverts positions to [M],
preserving the any-order property of masked diffusion models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AOAEPolicy(nn.Module):
    """
    Policy pi_phi(a_t | s_t) with factorized Bernoulli likelihood.

    Input per position:  (h_t^k [D], m_t^k [1], q_t^k [1], alpha_t^k [1], t/T [1])  →  projected to d_model.
    Backbone:            N-layer bidirectional transformer.
    Output:              3 scalar logits per position (unmask, remask, cache).
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

        # --- Input projection: (h_t^k, m_t^k, q_t^k, alpha_t^k, t/T) → d_model ---
        self.input_proj = nn.Linear(input_dim + 4, d)

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

        # --- Three independent output heads → scalar logit per position ---
        self.head_unmask = nn.Linear(d, 1)
        self.head_remask = nn.Linear(d, 1)
        self.head_cache = nn.Linear(d, 1)

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
        quality_scores: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action logits with validity constraints.

        Args:
            H_t:            [B, L, D]  soft-masked embeddings.
            mask_indicator: [B, L]     True where token is [M].
            step_frac:      scalar     t / T.
            temperature:    policy temperature tau_pi.
            quality_scores: [B, L]     PRISM quality scores (0=bad, 1=good).
                            If None, defaults to zeros.
            agreement:      [B, L]     auxiliary-primary agreement (0/1 float).
                            If None, defaults to zeros.

        Returns:
            dict with keys:
                "unmask_logits":  [B, L]  (masked to -inf for unmasked positions)
                "remask_logits":  [B, L]  (masked to -inf for masked positions)
                "cache_logits":   [B, L]
                "unmask_probs":   [B, L]  sigmoid(logit / tau_pi)
                "remask_probs":   [B, L]
                "cache_probs":    [B, L]
        """
        B, L, D = H_t.shape
        device = H_t.device

        # --- Build per-position input features ---
        m_feat = mask_indicator.float().unsqueeze(-1)              # [B, L, 1]
        if quality_scores is not None:
            q_feat = quality_scores.unsqueeze(-1)                  # [B, L, 1]
        else:
            q_feat = torch.zeros(B, L, 1, device=device)          # [B, L, 1]
        if agreement is not None:
            a_feat = agreement.unsqueeze(-1)                       # [B, L, 1]
        else:
            a_feat = torch.zeros(B, L, 1, device=device)          # [B, L, 1]
        t_feat = torch.full((B, L, 1), step_frac, device=device)  # [B, L, 1]
        x = torch.cat([H_t, m_feat, q_feat, a_feat, t_feat], dim=-1)  # [B, L, D+4]
        x = self.input_proj(x)                                     # [B, L, d]

        # --- Transformer backbone ---
        x = self.backbone(x)                                       # [B, L, d]

        # --- Head logits ---
        unmask_logits = self.head_unmask(x).squeeze(-1)  # [B, L]
        remask_logits = self.head_remask(x).squeeze(-1)  # [B, L]
        cache_logits = self.head_cache(x).squeeze(-1)    # [B, L]

        # --- Validity constraints via logit masking ---
        # Unmask only on masked positions
        unmask_logits = unmask_logits.masked_fill(~mask_indicator, -1e9)
        # Remask only on unmasked positions
        remask_logits = remask_logits.masked_fill(mask_indicator, -1e9)
        # Cache-remask exclusion is enforced at sampling time (see sample_actions)

        # --- Tempered probabilities ---
        unmask_probs = torch.sigmoid(unmask_logits / temperature)
        remask_probs = torch.sigmoid(remask_logits / temperature)
        cache_probs = torch.sigmoid(cache_logits / temperature)

        return {
            "unmask_logits": unmask_logits,
            "remask_logits": remask_logits,
            "cache_logits": cache_logits,
            "unmask_probs": unmask_probs,
            "remask_probs": remask_probs,
            "cache_probs": cache_probs,
        }

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
            dict with "u_t", "r_t", "kappa_t" — each [B, L] binary.
        """
        u_t = torch.bernoulli(policy_out["unmask_probs"])    # [B, L]
        r_t = torch.bernoulli(policy_out["remask_probs"])    # [B, L]
        kappa_t = torch.bernoulli(policy_out["cache_probs"])  # [B, L]

        # Enforce cache-remask exclusion: if remask is 1, cache must be 0
        kappa_t = kappa_t * (1.0 - r_t)

        return {"u_t": u_t, "r_t": r_t, "kappa_t": kappa_t}

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

        for key, prob_key in [("u_t", "unmask_probs"), ("r_t", "remask_probs"), ("kappa_t", "cache_probs")]:
            a = actions[key]        # [B, L]
            p = policy_out[prob_key].clamp(1e-7, 1.0 - 1e-7)  # [B, L]
            lp = a * torch.log(p) + (1.0 - a) * torch.log(1.0 - p)  # [B, L]
            total = total + lp.sum(dim=-1)  # [B]

        return total


class DefaultPolicy(nn.Module):
    """Heuristic policy for speculative eval without GRPO training.

    Uses the auxiliary model's confidence as the basis for unmask/cache
    decisions, mimicking AOAEPolicy's interface so it can be dropped
    into the speculative inference loop.

    Behavior:
      - Unmask: masked positions where confidence > tau_mask
      - Remask: never (all zeros)
      - Cache:  positions where auxiliary and primary agree

    Args:
        tau_mask: confidence threshold for unmasking (default 0.7 = S-mode).
    """

    def __init__(self, tau_mask: float = 0.7):
        super().__init__()
        self.tau_mask = tau_mask

    def forward(
        self,
        H_t: torch.Tensor,
        mask_indicator: torch.BoolTensor,
        step_frac: float,
        temperature: float = 1.0,
        quality_scores: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L, D = H_t.shape
        device = H_t.device

        # Use agreement signal directly as cache probability
        cache_probs = agreement if agreement is not None else torch.zeros(B, L, device=device)

        # Gradual unmask schedule: at diffusion step t (step_frac = t/T),
        # unmask ~1/t of the remaining masked positions per step.
        # This mimics the uniform diffusion schedule without needing T.
        # step_frac goes from 1.0 (first step) to ~0 (last step).
        unmask_rate = min(1.0 / max(step_frac * 200.0, 1.0), 1.0)
        unmask_probs = mask_indicator.float() * unmask_rate

        # Never remask
        remask_probs = torch.zeros(B, L, device=device)

        return {
            "unmask_logits": torch.zeros(B, L, device=device),
            "remask_logits": torch.full((B, L), -1e9, device=device),
            "cache_logits": torch.zeros(B, L, device=device),
            "unmask_probs": unmask_probs,
            "remask_probs": remask_probs,
            "cache_probs": cache_probs,
        }

    def sample_actions(
        self,
        policy_out: Dict[str, torch.Tensor],
        mask_indicator: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        unmask_probs = policy_out["unmask_probs"]
        u_t = (torch.rand_like(unmask_probs) < unmask_probs).float() * mask_indicator.float()
        r_t = torch.zeros_like(u_t)
        kappa_t = policy_out["cache_probs"]
        return {"u_t": u_t, "r_t": r_t, "kappa_t": kappa_t}

    def log_prob(
        self,
        policy_out: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.zeros(actions["u_t"].shape[0], device=actions["u_t"].device)
