"""
Soft-Masked State Construction (Eq. softmask + gating from paper section 3.1).

Converts base-model logits into per-position feature vectors that blend the
mask embedding with top-K predicted token embeddings, gated by a
confidence-scaled sigmoid of negative entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SoftMaskedState(nn.Module):
    """Build the soft-masked state from base-model outputs."""

    def __init__(self, cfg, embedding_weight: torch.Tensor):
        super().__init__()
        sm = cfg["soft_mask"]
        self.top_k = sm["top_k"]

        self.omega_s = nn.Parameter(torch.tensor(sm["omega_s_init"], dtype=torch.float32))
        self.omega_a = nn.Parameter(torch.tensor(sm["omega_a_init"], dtype=torch.float32))
        self.omega_b = nn.Parameter(torch.tensor(sm["omega_b_init"], dtype=torch.float32))

        # The embedding matrix belongs to the frozen base model. Keep it as a
        # non-persistent buffer so AOAE checkpoints do not serialize gigabytes
        # of static embeddings on every save.
        self.register_buffer("embedding_weight", embedding_weight.float(), persistent=False)
        self.embed_dim = embedding_weight.shape[1]

        self.register_buffer("mask_embed", torch.zeros(1, embedding_weight.shape[1]))
        self._mask_embed_set = False

    def forward(
        self,
        logits: torch.Tensor,
        mask_indicator: torch.BoolTensor,
        step_frac: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mask_indicator, step_frac
        bsz, seq_len, vocab_size = logits.shape

        logits_f = torch.nan_to_num(logits.float(), nan=0.0)
        log_probs = F.log_softmax(logits_f, dim=-1)
        probs = log_probs.exp()
        confidence = probs.max(dim=-1).values
        entropy = -(torch.nan_to_num(probs * log_probs, nan=0.0)).sum(dim=-1)

        topk_probs, topk_ids = probs.topk(self.top_k, dim=-1)
        topk_probs_norm = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        topk_embeds = F.embedding(topk_ids, self.embedding_weight)
        weighted_embeds = (topk_probs_norm.unsqueeze(-1) * topk_embeds).sum(dim=2)

        if not self._mask_embed_set:
            raise RuntimeError(
                "SoftMaskedState.set_mask_embedding() was never called. "
                "Call it once after constructing the base model."
            )
        mask_embed = self.mask_embed

        lam = self.omega_s * torch.sigmoid(
            self.omega_a * (-entropy - self.omega_b)
        )

        lam_exp = lam.unsqueeze(-1)
        h_t = lam_exp * mask_embed + (1.0 - lam_exp) * weighted_embeds
        assert h_t.shape[:2] == (bsz, seq_len)
        assert logits.shape[-1] == vocab_size
        return h_t, confidence, entropy, weighted_embeds

    def recompute_h_t(
        self,
        weighted_embeds: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute h_t from stored rollout intermediates, with grad through ω scalars.

        During GRPO training, ``weighted_embeds`` and ``entropy`` are detached
        tensors stored from the rollout.  Recomputing h_t here (rather than
        using the stored H_t directly) allows autograd to flow through
        ω_s / ω_a / ω_b, making them genuinely trainable.

        Args:
            weighted_embeds: [B, L, D] float32 — top-K weighted token embeddings
                             stored from the forward pass (detached).
            entropy:         [B, L] float32 — per-position entropy (detached).

        Returns:
            h_t: [B, L, D] with grad w.r.t. omega parameters.
        """
        lam = self.omega_s * torch.sigmoid(
            self.omega_a * (-entropy - self.omega_b)
        )
        return lam.unsqueeze(-1) * self.mask_embed + (1.0 - lam.unsqueeze(-1)) * weighted_embeds

    def set_mask_embedding(self, mask_token_id: int):
        self.mask_embed.copy_(self.embedding_weight[mask_token_id].unsqueeze(0))
        self._mask_embed_set = True
