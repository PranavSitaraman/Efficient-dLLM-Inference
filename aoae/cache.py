"""
Policy-Controlled dKV-Cache Manager (paper §2.3, §3.3).

Unlike the original dKV-Cache (Ma et al., 2025) which uses heuristic
confidence thresholds to decide when to cache, AOAE makes caching a
policy-controlled action (the kappa_t head).  The cache manager tracks:
  - Which positions are currently cached.
  - Invalidation when a position is edited (Phase 1 of Algorithm 1).
  - Commit when the policy's cache action fires (Phase 3).
  - Cache thrashing counts for the multiplicative reward.
"""

import torch


class DKVCacheManager:
    """
    Manages the set of cached positions K across diffusion steps.

    Tracks which positions are cached so the inference loop can:
      1. Skip recomputation for cached positions (speedup proxy).
      2. Invalidate on edit.
      3. Count cache thrashing for the reward.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.B = batch_size
        self.L = seq_len
        self.device = device
        self.cached: torch.BoolTensor = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

    def reset(self):
        """Clear all cached positions (start of new generation)."""
        self.cached.zero_()

    def invalidate(self, edit_mask: torch.Tensor):
        """Invalidate cache for edited positions.

        Args:
            edit_mask: [B, L] float/bool — 1 where position was edited.
        """
        self.cached = self.cached & ~(edit_mask.bool())

    def commit(self, cache_mask: torch.Tensor):
        """Commit new positions to the cache.

        Args:
            cache_mask: [B, L] float/bool — 1 where policy says to cache.
        """
        self.cached = self.cached | cache_mask.bool()

    def get_cached_mask(self) -> torch.BoolTensor:
        """Return [B, L] bool mask of currently cached positions."""
        return self.cached

    def count_thrash(self, edit_mask: torch.Tensor) -> torch.Tensor:
        """Count cache thrashing: positions that are both cached AND edited.

        Thrash(t) = sum_k I[k in K_{t-1} AND e_t^k = 1]

        Args:
            edit_mask: [B, L] float — 1 where position is being edited.
        Returns:
            thrash_count: [B] per-sample count.
        """
        thrash = self.cached.float() * edit_mask.float()
        return thrash.sum(dim=-1)

    def cached_fraction(self) -> torch.Tensor:
        """Return [B] fraction of positions currently cached."""
        return self.cached.float().mean(dim=-1)
