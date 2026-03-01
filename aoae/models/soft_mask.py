"""
Soft-Masked State Construction (Eq. softmask + gating from paper §3.1).

Converts base-model logits into per-position feature vectors h_t^k that
blend the mask embedding with top-K predicted token embeddings, gated by
a confidence-scaled sigmoid of negative entropy.

Reference: Hersche et al. "Soft-Masked Diffusion Language Models" (2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SoftMaskedState(nn.Module):
    """
    Builds the soft-masked state s_t = (H_t, m_t, t) from base-model outputs.

    Learnable parameters: omega = (omega_s, omega_a, omega_b) for the gating
    function lambda(p_t^k).
    """

    def __init__(self, cfg, embedding_weight: torch.Tensor):
        """
        Args:
            cfg: full config dict.
            embedding_weight: [V, D] token embedding matrix from base model.
        """
        super().__init__()
        sm = cfg["soft_mask"]
        self.top_k = sm["top_k"]

        # Gating parameters (Eq. gating): lambda = omega_s * sigmoid(omega_a * (-H - omega_b))
        self.omega_s = nn.Parameter(torch.tensor(sm["omega_s_init"], dtype=torch.float32))
        self.omega_a = nn.Parameter(torch.tensor(sm["omega_a_init"], dtype=torch.float32))
        self.omega_b = nn.Parameter(torch.tensor(sm["omega_b_init"], dtype=torch.float32))

        # Store embedding weight as buffer (frozen, from base model)
        self.register_buffer("embedding_weight", embedding_weight.float())
        self.embed_dim = embedding_weight.shape[1]

        # Dedicated mask embedding buffer (set via set_mask_embedding)
        self.register_buffer("mask_embed", torch.zeros(1, embedding_weight.shape[1]))

    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        mask_indicator: torch.BoolTensor,
        step_frac: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct the soft-masked state.

        Args:
            logits:         [B, L, V] base-model logits.
            mask_indicator: [B, L]    True where y_t^k == [M].
            step_frac:      scalar    t / T ∈ [0, 1].

        Returns:
            H_t:        [B, L, D]   soft-masked embeddings.
            confidence: [B, L]      max-prob per position (for policy input).
            entropy:    [B, L]      per-position entropy (for diagnostics).
        """
        B, L, V = logits.shape

        # --- Per-position distributions and statistics ---
        probs = F.softmax(logits.float(), dim=-1)                   # [B, L, V]
        confidence = probs.max(dim=-1).values                       # [B, L]
        log_probs = F.log_softmax(logits.float(), dim=-1)           # [B, L, V]
        entropy = -(probs * log_probs).sum(dim=-1)                  # [B, L]

        # --- Top-K tokens and renormalized probabilities ---
        topk_probs, topk_ids = probs.topk(self.top_k, dim=-1)      # [B, L, K]
        topk_probs_norm = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # --- Token embeddings for top-K predictions ---
        # [B, L, K, D]
        topk_embeds = F.embedding(topk_ids, self.embedding_weight)
        # Weighted sum: [B, L, D]
        weighted_embeds = (topk_probs_norm.unsqueeze(-1) * topk_embeds).sum(dim=2)

        # --- Mask embedding: [1, D] dedicated buffer ---
        mask_embed = self.mask_embed  # [1, D]

        # --- Gating function lambda (Eq. gating) ---
        # lambda(p_t^k) = omega_s * sigmoid(omega_a * (-H(p_t^k) - omega_b))
        lam = self.omega_s * torch.sigmoid(
            self.omega_a * (-entropy - self.omega_b)
        )  # [B, L]

        # --- Soft-masked embedding (Eq. softmask) ---
        # h_t^k = lambda * E_mask + (1 - lambda) * weighted_embeds
        lam_exp = lam.unsqueeze(-1)  # [B, L, 1]
        H_t = lam_exp * mask_embed + (1.0 - lam_exp) * weighted_embeds  # [B, L, D]

        return H_t, confidence, entropy

    # ------------------------------------------------------------------
    def set_mask_embedding(self, mask_token_id: int):
        """
        Copy the mask token's embedding into the dedicated buffer.
        Call this once after loading the base model.
        """
        self.mask_embed.copy_(self.embedding_weight[mask_token_id].unsqueeze(0))
