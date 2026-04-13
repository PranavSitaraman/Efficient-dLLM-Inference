"""
Policy-Controlled KV-Cache Managers (paper §3.3, §3.6).

Two distinct caches are maintained (per kv_cache_staleness.md):

  K_spec  (SpeculativeKVCache) — transient, one-step validity.
    Positions where the auxiliary drafter and primary verifier have the same
    argmax token at step t.  Valid for ONE forward pass only; replaced each
    step by the new agreement mask.

  K_stable (StableKVCache) — persistent, multi-step validity.
    Positions where the policy's κ_t head predicts stable KV across future
    steps.  Persists across steps; evicted only by an explicit remask (r_t=1).

  TwoCacheManager combines both into a single interface for inference loops.

  DKVCacheManager (legacy) — the original single-cache design that conflated
    the two concepts.  Kept for backward compatibility with non-speculative
    inference paths (aoae/inference.py) and dinfer_integration.py.
"""

import torch


# ---------------------------------------------------------------------------
# New two-cache system
# ---------------------------------------------------------------------------

class SpeculativeKVCache:
    """K_spec: agreement-based transient cache (one-step validity).

    Replaced entirely each step.  Provides a skip signal for positions where
    the auxiliary drafter KV vectors are known to agree with the primary, so
    re-computation can be skipped for ONE subsequent forward pass.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.B = batch_size
        self.L = seq_len
        self.device = device
        self.cached: torch.BoolTensor = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

    def accept(self, agreement: torch.Tensor):
        """Replace K_spec with the current agreement mask (not a union)."""
        self.cached = agreement.bool()

    def get_cached_mask(self) -> torch.BoolTensor:
        return self.cached

    def cached_fraction(self) -> torch.Tensor:
        """[B] fraction of positions currently in K_spec."""
        return self.cached.float().mean(dim=-1)

    def reset(self):
        self.cached.zero_()


class StableKVCache:
    """K_stable: κ_t-driven persistent cache (multi-step validity).

    Accumulates positions the policy predicts will remain stable.
    Eviction is triggered by remask actions (r_t=1), which corresponds
    to using remask as an eviction policy (§kv_cache_staleness.md).

    Tracks per-position age (steps in cache) to support age-decaying thrash
    penalties: remasking a freshly-committed token is a bad commit (high penalty),
    but remasking a long-lived stable token is a justified correction (low penalty).
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.B = batch_size
        self.L = seq_len
        self.device = device
        self.cached: torch.BoolTensor = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        # Per-position age: steps since this position was committed to K_stable.
        # Reset to 0 on commit or eviction.  Incremented by step_age() each step.
        self.age: torch.Tensor = torch.zeros(
            batch_size, seq_len, dtype=torch.float32, device=device
        )

    def commit(self, kappa_t: torch.Tensor, r_t: torch.Tensor):
        """Add κ_t=1 positions (not currently being remasked) to K_stable.
        Newly admitted positions start with age=0.
        """
        to_add = kappa_t.bool() & ~r_t.bool()
        newly_added = to_add & ~self.cached
        self.cached = self.cached | to_add
        self.age[newly_added] = 0.0

    def evict(self, r_t: torch.Tensor):
        """Evict remasked positions from K_stable and reset their age."""
        evicted = self.cached & r_t.bool()
        self.cached = self.cached & ~r_t.bool()
        self.age[evicted] = 0.0

    def step_age(self):
        """Increment age by 1 for all positions currently in K_stable.
        Call once per step AFTER commit/evict so newly admitted positions age
        from 1 at the start of the next step.
        """
        self.age += self.cached.float()

    def count_thrash(self, r_t: torch.Tensor, age_decay: float = 0.0) -> torch.Tensor:
        """[B] (age-weighted) count of stable-cached positions being remasked.

        With age_decay > 0, penalty weight = exp(-age_decay * age):
          - Freshly committed (age~0) → weight~1.0  (bad commit, full penalty)
          - Long-lived (age large)   → weight→0     (justified correction, near-zero penalty)
        """
        thrash_mask = self.cached.float() * r_t.float()   # [B, L]
        if age_decay > 0.0:
            weights = torch.exp(-age_decay * self.age)    # [B, L]
            return (thrash_mask * weights).sum(dim=-1)    # [B]
        return thrash_mask.sum(dim=-1)                    # [B]

    def get_cached_mask(self) -> torch.BoolTensor:
        return self.cached

    def cached_fraction(self) -> torch.Tensor:
        """[B] fraction of positions currently in K_stable."""
        return self.cached.float().mean(dim=-1)

    def reset(self):
        self.cached.zero_()
        self.age.zero_()


class SpeculativeCacheBookkeeper:
    """Bookkeeper for K_spec + K_stable set membership (§3.6 two-cache system).

    IMPORTANT: This class tracks BOOLEAN MASKS only — it does NOT store KV
    tensors and does NOT skip any actual computation.  Its sole purpose is to
    record which positions would be in each logical cache pool so that:
      (a) the reward can compute mean_cached_fraction / effective_flops, and
      (b) thrash counts and cache-quality F1 can be measured during training.

    Actual KV-computation skipping is handled separately:
      - Prompt-prefix skip (real): auxiliary_forward_with_cache /
        primary_forward_with_cache → replace_position=resp_slice each step.
        Controlled by cache.sim_log_only in config (false = real skip enabled).
      - Per-position response skip (pending): requires token packing or a
        non-contiguous update extension to dInfer's KVCache.

    Exposes a backward-compatible interface so existing callers that use
    count_thrash / invalidate / get_cached_mask continue to work, while
    new callers can access spec vs. stable fractions separately.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        thrash_age_decay: float = 0.0,
    ):
        self.spec = SpeculativeKVCache(batch_size, seq_len, device)
        self.stable = StableKVCache(batch_size, seq_len, device)
        # Decay rate for age-weighted thrash penalty (0 = no decay, uniform penalty).
        self.thrash_age_decay = thrash_age_decay

    # --- New two-cache API (called from speculative_inference.py Phase 3) ---

    def step_spec(self, agreement: torch.Tensor):
        """Phase 3a: Update K_spec with current step's agreement mask."""
        self.spec.accept(agreement)

    def step_stable(self, kappa_t: torch.Tensor, r_t: torch.Tensor):
        """Phase 3b: Commit κ_t positions to K_stable; evict r_t positions; tick age."""
        self.stable.commit(kappa_t, r_t)
        self.stable.evict(r_t)
        self.stable.step_age()   # age incremented AFTER commit/evict this step

    # --- Backward-compatible API (drop-in for DKVCacheManager) ---

    def invalidate(self, edit_mask: torch.Tensor):
        """Evict stable-cached positions that are being edited."""
        self.stable.evict(edit_mask)
        # K_spec is replaced each step — no persistent eviction needed.

    def count_thrash(self, r_t: torch.Tensor) -> torch.Tensor:
        """[B] age-weighted count of stable-cached positions that are remasked."""
        return self.stable.count_thrash(r_t, age_decay=self.thrash_age_decay)

    def combined_cached_mask(self) -> torch.BoolTensor:
        """[B, L] union of K_spec and K_stable."""
        return self.spec.cached | self.stable.cached

    def get_cached_mask(self) -> torch.BoolTensor:
        """Backward-compatible alias for combined_cached_mask()."""
        return self.combined_cached_mask()

    def spec_cached_fraction(self) -> torch.Tensor:
        """[B] fraction of positions in K_spec."""
        return self.spec.cached_fraction()

    def stable_cached_fraction(self) -> torch.Tensor:
        """[B] fraction of positions in K_stable."""
        return self.stable.cached_fraction()

    def cached_fraction(self) -> torch.Tensor:
        """[B] fraction of positions in K_spec ∪ K_stable."""
        return self.combined_cached_mask().float().mean(dim=-1)

    def reset(self):
        self.spec.reset()
        self.stable.reset()


# ---------------------------------------------------------------------------
# Legacy single-cache (kept for backward compat with non-speculative paths)
# ---------------------------------------------------------------------------

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
