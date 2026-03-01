"""
dInfer Integration: Hook AOAE policy into dInfer's high-performance caching.

This module provides a bridge between the AOAE policy's per-position
cache/unmask/remask decisions and dInfer's native KV-cache infrastructure,
enabling fair evaluation of the policy on frontier hardware.

The key integration points are:
  1. PolicyGuidedCacheManager: wraps dInfer's cache with policy-driven
     commit/invalidate decisions instead of confidence thresholds.
  2. PolicyGuidedDecoder: runs the full AOAE inference loop using dInfer's
     sparse attention and FusedMoE kernels for actual speedup measurement.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from .cache import DKVCacheManager
from .models.composed_prediction import compose_prediction


@dataclass
class CacheStats:
    """Statistics for a single inference run."""
    total_commits: int = 0
    total_invalidations: int = 0
    total_remasks: int = 0
    total_unmasks: int = 0
    cache_hit_rate: float = 0.0
    steps_used: int = 0


class PolicyGuidedCacheManager:
    """Wraps dKV-Cache with AOAE policy-driven commit/invalidate.

    Instead of using confidence thresholds to decide which positions
    to cache, this manager uses the policy's kappa_t predictions.
    The policy's remask predictions drive cache invalidation.

    This provides a fair comparison: the same underlying cache mechanism
    (dKV-Cache sparse attention) is used, but steering is learned vs heuristic.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.cache_mgr = DKVCacheManager(batch_size, seq_len, device)
        self.stats = CacheStats()

    def step(
        self,
        r_t: torch.Tensor,
        kappa_t: torch.Tensor,
        u_t: torch.Tensor,
    ):
        """Process one inference step's worth of policy actions.

        Args:
            r_t: [B, L] remask decisions (1 = remask)
            kappa_t: [B, L] cache commit decisions (1 = commit)
            u_t: [B, L] unmask decisions (1 = unmask)
        """
        # Phase 1: Invalidate cached positions that are being remasked
        if r_t.any():
            self.cache_mgr.invalidate(r_t)
            self.stats.total_invalidations += int(r_t.sum().item())
            self.stats.total_remasks += int(r_t.sum().item())

        # Phase 2: Count unmasks
        self.stats.total_unmasks += int(u_t.sum().item())

        # Phase 3: Commit new positions to cache
        if kappa_t.any():
            self.cache_mgr.commit(kappa_t)
            self.stats.total_commits += int(kappa_t.sum().item())

        self.stats.steps_used += 1

    @property
    def cached_mask(self) -> torch.Tensor:
        """Return [B, L] bool mask of currently cached positions."""
        return self.cache_mgr.get_cached_mask()

    def get_stats(self) -> CacheStats:
        """Return accumulated statistics."""
        total_ops = self.stats.total_commits + self.stats.total_invalidations
        if total_ops > 0:
            self.stats.cache_hit_rate = (
                self.stats.total_commits - self.stats.total_invalidations
            ) / max(total_ops, 1)
        return self.stats

    def count_thrash(self, r_t: torch.Tensor) -> torch.Tensor:
        """Count positions that are cached AND being remasked (thrashing)."""
        return self.cache_mgr.count_thrash(r_t)


class SpeculativeCacheManager:
    """Cache manager for dual-model speculative diffusion.

    Extends PolicyGuidedCacheManager with agreement-gated caching:
    positions are only committed to the persistent cache when the
    hard-routed auxiliary and soft-routed primary agree on the token.

    Tracks additional metrics: agreement rate, draft acceptance rate,
    and effective speedup from speculative caching.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.cache_mgr = DKVCacheManager(batch_size, seq_len, device)
        self.stats = CacheStats()
        self._agreement_sum = 0.0
        self._agreement_count = 0
        self._draft_accepts = 0
        self._draft_rejects = 0

    def step(
        self,
        r_t: torch.Tensor,
        kappa_t: torch.Tensor,
        u_t: torch.Tensor,
        agreement: torch.Tensor,
    ):
        """Process one step with agreement-gated caching.

        Args:
            r_t: [B, L] remask decisions
            kappa_t: [B, L] cache commit decisions
            u_t: [B, L] unmask decisions
            agreement: [B, L] bool/float auxiliary-primary agreement
        """
        # Phase 1: Invalidate
        if r_t.any():
            self.cache_mgr.invalidate(r_t)
            self.stats.total_invalidations += int(r_t.sum().item())
            self.stats.total_remasks += int(r_t.sum().item())

        # Phase 2: Unmasks
        self.stats.total_unmasks += int(u_t.sum().item())

        # Phase 3: Agreement-gated cache commit
        agreement_f = agreement.float()
        accepted = kappa_t * agreement_f        # cache only where models agree
        rejected = kappa_t * (1.0 - agreement_f)  # wanted to cache but disagreed

        if accepted.any():
            self.cache_mgr.commit(accepted)
            self.stats.total_commits += int(accepted.sum().item())

        self._draft_accepts += int(accepted.sum().item())
        self._draft_rejects += int(rejected.sum().item())
        self._agreement_sum += agreement_f.sum().item()
        self._agreement_count += agreement_f.numel()
        self.stats.steps_used += 1

    @property
    def cached_mask(self) -> torch.Tensor:
        return self.cache_mgr.get_cached_mask()

    def get_stats(self) -> dict:
        """Return stats dict with speculative caching metrics."""
        base = self.stats
        total_ops = base.total_commits + base.total_invalidations
        cache_hit_rate = (
            (base.total_commits - base.total_invalidations) / max(total_ops, 1)
        )
        total_drafts = self._draft_accepts + self._draft_rejects
        return {
            "total_commits": base.total_commits,
            "total_invalidations": base.total_invalidations,
            "total_remasks": base.total_remasks,
            "total_unmasks": base.total_unmasks,
            "cache_hit_rate": cache_hit_rate,
            "steps_used": base.steps_used,
            "draft_accept_rate": self._draft_accepts / max(total_drafts, 1),
            "draft_accepts": self._draft_accepts,
            "draft_rejects": self._draft_rejects,
            "mean_agreement": self._agreement_sum / max(self._agreement_count, 1),
        }

    def count_thrash(self, r_t: torch.Tensor) -> torch.Tensor:
        return self.cache_mgr.count_thrash(r_t)


def run_speculative_inference(
    dual_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    policy_temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Run speculative diffusion with dInfer integration for benchmarking.

    Uses SpeculativeCacheManager for accurate dual-model cache statistics.

    Args:
        dual_model: DualModelWrapper (soft primary + hard auxiliary).
        policy: AOAE steering policy.
        soft_mask_module: soft-masked state builder.
        prism_adapter: PRISM quality head (or None).
        prompt_ids: [B, P] prompt token ids.
        cfg: config dict.
        policy_temperature: tau_pi.

    Returns:
        output_ids: [B, P + L_gen] full sequence.
        stats: dict with detailed speculative cache metrics.
    """
    from .models.composed_prediction import compose_prediction_dual

    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = ic["gen_length"]
    mask_id = cfg["base_model"]["mask_token_id"]
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)
    use_fallback = ic["fallback_unmask"]

    B = prompt_ids.shape[0]
    P = prompt_ids.shape[1]
    device = prompt_ids.device

    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    resp_slice = slice(P, P + L_gen)
    cache_mgr = SpeculativeCacheManager(B, L_gen, device)

    for t in range(T, 0, -1):
        step_frac = t / T
        resp_tokens = y[:, resp_slice]
        mask_ind = (resp_tokens == mask_id)

        if not mask_ind.any():
            break

        # Dual-model forward
        need_hidden = (prism_adapter is not None)
        dual_out = dual_model.dual_forward_resp(y, resp_slice, need_hidden=need_hidden)

        resp_logits = dual_out.primary_logits
        aux_logits = dual_out.auxiliary_logits
        agreement = dual_out.agreement

        # PRISM quality scores
        q_scores = None
        if prism_adapter is not None and dual_out.primary_hidden is not None:
            with torch.no_grad():
                q_scores = prism_adapter(dual_out.primary_hidden.float())

        # Soft-masked state
        H_t, confidence, entropy = soft_mask_module(resp_logits, mask_ind, step_frac)

        # Policy forward
        policy_out = policy(
            H_t, mask_ind, step_frac,
            temperature=policy_temperature,
            quality_scores=q_scores,
            agreement=agreement.float(),
        )
        pol_inner = policy.module if hasattr(policy, "module") else policy
        actions = pol_inner.sample_actions(policy_out, mask_ind)

        u_t = actions["u_t"]
        r_t = actions["r_t"]
        kappa_t = actions["kappa_t"]

        resp_tokens = resp_tokens.clone()

        # Phase 1: Remask
        remask_positions = r_t.bool() & ~mask_ind
        if remask_positions.any():
            resp_tokens[remask_positions] = mask_id

        # Phase 2: Unmask with composed prediction
        unmask_positions = u_t.bool() & mask_ind
        if unmask_positions.any():
            if gamma > 0:
                composed_logits = compose_prediction_dual(
                    resp_logits, aux_logits, agreement, gamma=gamma,
                )
            else:
                composed_logits = resp_logits

            if base_temp > 0:
                probs = F.softmax(composed_logits / base_temp, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, L_gen)
            else:
                sampled = composed_logits.argmax(dim=-1)
            resp_tokens[unmask_positions] = sampled[unmask_positions]

        # Fallback
        if use_fallback:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)
            if no_unmasks.any():
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    masked_pos = still_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        best_pos = masked_pos[confidence[b_idx, masked_pos].argmax()]
                        resp_tokens[b_idx, best_pos] = resp_logits[b_idx, best_pos].argmax()

        # Phase 3: Agreement-gated cache commit
        cache_mgr.step(r_t, kappa_t, u_t, agreement)

        y = y.clone()
        y[:, resp_slice] = resp_tokens

    return y, cache_mgr.get_stats()


def run_policy_guided_inference(
    base_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    policy_temperature: float = 1.0,
) -> Tuple[torch.Tensor, CacheStats]:
    """Run AOAE inference with dInfer integration for fair benchmarking.

    This is a simplified version of aoae_inference that:
    1. Uses PolicyGuidedCacheManager for accurate cache statistics
    2. Integrates composed prediction for cache-aligned token selection
    3. Reports detailed cache hit/miss statistics for evaluation

    Args:
        base_model: frozen LLaDA wrapper (dinfer or soft_moe backend).
        policy: AOAE policy network.
        soft_mask_module: soft-masked state builder.
        prism_adapter: PRISM quality head (or None).
        prompt_ids: [B, P] prompt token ids.
        cfg: config dict.
        policy_temperature: tau_pi for Bernoulli tempering.

    Returns:
        output_ids: [B, P + L_gen] full sequence.
        stats: CacheStats with detailed cache performance metrics.
    """
    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = ic["gen_length"]
    mask_id = cfg["base_model"]["mask_token_id"]
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)
    use_fallback = ic["fallback_unmask"]

    B = prompt_ids.shape[0]
    P = prompt_ids.shape[1]
    device = prompt_ids.device

    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    resp_slice = slice(P, P + L_gen)
    cache_mgr = PolicyGuidedCacheManager(B, L_gen, device)

    for t in range(T, 0, -1):
        step_frac = t / T
        resp_tokens = y[:, resp_slice]
        mask_ind = (resp_tokens == mask_id)

        if not mask_ind.any():
            break

        # Base model forward
        if prism_adapter is not None:
            logits, hidden_states = base_model.forward_with_hidden(y)
            resp_hidden = hidden_states[:, resp_slice, :]
        else:
            logits = base_model.forward(y)
            resp_hidden = None
        resp_logits = logits[:, resp_slice, :]

        # PRISM quality scores
        q_scores = None
        if prism_adapter is not None and resp_hidden is not None:
            with torch.no_grad():
                q_scores = prism_adapter(resp_hidden.float())

        # Soft-masked state
        H_t, confidence, entropy = soft_mask_module(
            resp_logits, mask_ind, step_frac
        )

        # Policy forward
        policy_out = policy(
            H_t, mask_ind, step_frac,
            temperature=policy_temperature,
            quality_scores=q_scores,
        )
        pol_inner = policy.module if hasattr(policy, "module") else policy
        actions = pol_inner.sample_actions(policy_out, mask_ind)

        u_t = actions["u_t"]
        r_t = actions["r_t"]
        kappa_t = actions["kappa_t"]

        resp_tokens = resp_tokens.clone()

        # Phase 1: Remask
        remask_positions = r_t.bool() & ~mask_ind
        if remask_positions.any():
            resp_tokens[remask_positions] = mask_id

        # Phase 2: Unmask with composed prediction
        unmask_positions = u_t.bool() & mask_ind
        if unmask_positions.any():
            if gamma > 0 and "cache_probs" in policy_out:
                composed_logits = compose_prediction(
                    resp_logits, policy_out["cache_probs"], gamma=gamma,
                )
            else:
                composed_logits = resp_logits

            if base_temp > 0:
                probs = F.softmax(composed_logits / base_temp, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, L_gen)
            else:
                sampled = composed_logits.argmax(dim=-1)
            resp_tokens[unmask_positions] = sampled[unmask_positions]

        # Fallback
        if use_fallback:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)
            if no_unmasks.any():
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    masked_pos = still_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        best_pos = masked_pos[confidence[b_idx, masked_pos].argmax()]
                        resp_tokens[b_idx, best_pos] = resp_logits[b_idx, best_pos].argmax()

        # Phase 3: Cache commit + stats tracking
        cache_mgr.step(r_t, kappa_t, u_t)

        y = y.clone()
        y[:, resp_slice] = resp_tokens

    return y, cache_mgr.get_stats()
