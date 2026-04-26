"""
Tests for AOAE cache semantics and fingerprint stability.

Covers:
  1. K_spec frontier accumulation and post-verifier clearing
  2. K_stable admission conditions: κ_t=1 AND r_t=0
  3. K_stable eviction on remask (r_t=1)
  4. SpeculativeCacheBookkeeper two-pool combined view
  5. GRPO reward computation: correctness * speed, thrash penalty, unresolved penalty
  6. Fingerprint stability: llada21_official does NOT change the fingerprint

Run with:
    python -m pytest tests/test_cache_semantics.py -v
"""

import sys
import os
import types

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aoae.cache import (
    SpeculativeKVCache,
    StableKVCache,
    SpeculativeCacheBookkeeper,
    DKVCacheManager,
)
from aoae.checkpoints import build_grpo_config_fingerprint
from aoae.speculative_inference import DraftFrontier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool(lst):
    """Convert nested list to BoolTensor."""
    return torch.tensor(lst, dtype=torch.bool)


def _float(lst):
    return torch.tensor(lst, dtype=torch.float32)


# ===========================================================================
# 1. K_spec (SpeculativeKVCache) — frontier accumulation and clearing
# ===========================================================================

class TestKspecFrontierSemantics:
    """Unit tests for SpeculativeKVCache (the transient K_spec mirror)."""

    def test_accept_sets_frontier_mask(self):
        """accept() replaces the internal mask with the provided frontier."""
        kspec = SpeculativeKVCache(batch_size=1, seq_len=4, device=torch.device("cpu"))
        mask = _bool([[True, False, True, False]])
        kspec.accept(mask)
        assert kspec.cached.tolist() == [[True, False, True, False]]

    def test_frontier_replaced_not_accumulated(self):
        """K_spec is a mirror: accept() replaces, it does NOT OR-accumulate."""
        kspec = SpeculativeKVCache(batch_size=1, seq_len=4, device=torch.device("cpu"))
        kspec.accept(_bool([[True, False, False, False]]))
        # Second accept with a different mask — previous positions lost.
        kspec.accept(_bool([[False, False, True, False]]))
        assert kspec.cached.tolist() == [[False, False, True, False]]

    def test_reset_clears_frontier(self):
        """reset() returns the K_spec mirror to all-False."""
        kspec = SpeculativeKVCache(batch_size=1, seq_len=6, device=torch.device("cpu"))
        kspec.accept(torch.ones(1, 6, dtype=torch.bool))
        assert kspec.cached.any()
        kspec.reset()
        assert not kspec.cached.any()

    def test_draft_frontier_cleared_after_verifier_event(self):
        """DraftFrontier.clear() after a verifier event zeroes mask, tokens, scores, age."""
        frontier = DraftFrontier(batch_size=1, seq_len=4, device=torch.device("cpu"))
        tokens = torch.tensor([[10, 11, 12, 13]])
        frontier.add(torch.ones(1, 4, dtype=torch.bool), tokens)
        assert frontier.mask.any()

        frontier.clear()  # simulates post-verifier-event clearing

        assert not frontier.mask.any(), "Frontier mask must be all-False after clear"
        assert frontier.token_ids.tolist() == [[-1, -1, -1, -1]]
        assert frontier.scores.sum().item() == 0.0
        assert frontier.age.sum().item() == 0.0

    def test_kspec_frontier_accumulates_across_aux_steps(self):
        """DraftFrontier accumulates positions across multiple aux microsteps."""
        frontier = DraftFrontier(batch_size=1, seq_len=6, device=torch.device("cpu"))
        tokens = torch.arange(6).unsqueeze(0)

        # Step A: draft positions 0, 2
        frontier.add(_bool([[True, False, True, False, False, False]]), tokens)
        assert frontier.mask[0, 0].item() and frontier.mask[0, 2].item()
        assert not frontier.mask[0, 1].item()

        # Step B: draft position 4 (union)
        frontier.add(_bool([[False, False, False, False, True, False]]), tokens)
        assert frontier.mask[0, 0].item(), "pos 0 should still be in frontier"
        assert frontier.mask[0, 2].item(), "pos 2 should still be in frontier"
        assert frontier.mask[0, 4].item(), "pos 4 newly added"
        assert frontier.numel_per_batch().item() == 3

    def test_kspec_bookkeeper_step_spec_mirrors_mask(self):
        """SpeculativeCacheBookkeeper.step_spec() mirrors the agreement mask in K_spec."""
        bk = SpeculativeCacheBookkeeper(batch_size=1, seq_len=4, device=torch.device("cpu"))
        agreement = _bool([[True, False, True, False]])
        bk.step_spec(agreement)
        assert bk.spec.cached.tolist() == [[True, False, True, False]]


# ===========================================================================
# 2. K_stable (StableKVCache) — admission requires κ_t=1, q_t=1, r_t=0
# ===========================================================================

class TestKstableAdmission:
    """K_stable only admits positions where κ_t=1 AND r_t=0."""

    def test_kappa1_r0_enters_stable(self):
        """Position with κ_t=1, r_t=0 is committed to K_stable."""
        stable = StableKVCache(batch_size=1, seq_len=4, device=torch.device("cpu"))
        kappa_t = _bool([[False, True, True, False]])   # positions 1 and 2
        r_t     = _bool([[False, False, False, False]])  # no remask
        stable.commit(kappa_t, r_t)
        assert stable.cached.tolist() == [[False, True, True, False]]

    def test_kappa1_r1_does_not_enter_stable(self):
        """A position simultaneously remasked (r_t=1) is NOT admitted even if κ_t=1."""
        stable = StableKVCache(batch_size=1, seq_len=4, device=torch.device("cpu"))
        kappa_t = _bool([[True, True, True, True]])   # all predicted stable
        r_t     = _bool([[False, True, False, False]])  # position 1 is remasked
        stable.commit(kappa_t, r_t)
        # pos 1: κ_t=1 AND r_t=1 → excluded by commit; remaining 0,2,3 admitted
        assert stable.cached[0, 0].item()
        assert not stable.cached[0, 1].item(), "κ_t=1 AND r_t=1 must NOT enter K_stable"
        assert stable.cached[0, 2].item()
        assert stable.cached[0, 3].item()

    def test_kappa0_r0_does_not_enter_stable(self):
        """Position with κ_t=0 is never admitted regardless of r_t."""
        stable = StableKVCache(batch_size=1, seq_len=4, device=torch.device("cpu"))
        kappa_t = _bool([[False, False, False, False]])
        r_t     = _bool([[False, False, False, False]])
        stable.commit(kappa_t, r_t)
        assert not stable.cached.any(), "κ_t=0 positions must never enter K_stable"

    def test_bookkeeper_step_stable_admits_only_kappa1_r0(self):
        """SpeculativeCacheBookkeeper.step_stable() honours the κ_t=1, r_t=0 gate."""
        bk = SpeculativeCacheBookkeeper(batch_size=1, seq_len=4, device=torch.device("cpu"))
        kappa_t = _bool([[True, True, False, False]])
        r_t     = _bool([[False, True, False, False]])
        bk.step_stable(kappa_t, r_t)
        # pos 0: κ=1, r=0 → in stable; pos 1: κ=1, r=1 → evicted/not added; pos 2,3: κ=0
        assert bk.stable.cached[0, 0].item()
        assert not bk.stable.cached[0, 1].item()
        assert not bk.stable.cached[0, 2].item()


# ===========================================================================
# 3. K_stable eviction on remask (r_t=1)
# ===========================================================================

class TestKstableEviction:
    """Remask action (r_t=1) evicts the position from K_stable."""

    def test_evict_removes_remasked_positions(self):
        """evict() removes r_t=1 positions from K_stable and resets their age."""
        stable = StableKVCache(batch_size=1, seq_len=4, device=torch.device("cpu"))
        # Commit positions 0,1,2 to stable.
        stable.commit(
            _bool([[True, True, True, False]]),
            _bool([[False, False, False, False]]),
        )
        stable.step_age()   # all cached positions get age = 1
        assert stable.cached[0, 0].item()
        assert stable.cached[0, 1].item()

        # Remask position 1.
        stable.evict(_bool([[False, True, False, False]]))

        assert stable.cached[0, 0].item(), "pos 0 should remain in K_stable"
        assert not stable.cached[0, 1].item(), "pos 1 should be evicted"
        assert stable.cached[0, 2].item(), "pos 2 should remain in K_stable"
        assert stable.age[0, 1].item() == 0.0, "evicted position age must be reset to 0"

    def test_invalidate_via_bookkeeper_evicts_stable(self):
        """SpeculativeCacheBookkeeper.invalidate() (the backward-compat API) evicts stable."""
        bk = SpeculativeCacheBookkeeper(batch_size=1, seq_len=4, device=torch.device("cpu"))
        kappa_t = _bool([[True, True, False, False]])
        r_t     = _bool([[False, False, False, False]])
        bk.step_stable(kappa_t, r_t)
        assert bk.stable.cached[0, 0].item()
        assert bk.stable.cached[0, 1].item()

        # Invalidate (remask) position 0.
        bk.invalidate(_bool([[True, False, False, False]]))
        assert not bk.stable.cached[0, 0].item(), "pos 0 must be evicted after invalidate"
        assert bk.stable.cached[0, 1].item(), "pos 1 must remain"

    def test_remask_via_step_stable_evicts_existing_entry(self):
        """step_stable with r_t=1 for an already-cached position evicts it."""
        bk = SpeculativeCacheBookkeeper(batch_size=1, seq_len=4, device=torch.device("cpu"))
        # First step: commit position 2.
        bk.step_stable(_bool([[False, False, True, False]]), _bool([[False, False, False, False]]))
        assert bk.stable.cached[0, 2].item()

        # Second step: remask position 2.
        bk.step_stable(_bool([[False, False, False, False]]), _bool([[False, False, True, False]]))
        assert not bk.stable.cached[0, 2].item(), "pos 2 must be evicted after r_t=1"

    def test_dkvcache_invalidate_removes_cached_positions(self):
        """DKVCacheManager.invalidate() removes positions from the legacy single cache."""
        mgr = DKVCacheManager(batch_size=1, seq_len=6, device=torch.device("cpu"))
        mgr.commit(_bool([[True, True, True, False, False, False]]))
        mgr.invalidate(_bool([[False, True, False, False, False, False]]))
        mask = mgr.get_cached_mask()
        assert mask[0, 0].item()
        assert not mask[0, 1].item(), "pos 1 should be invalidated"
        assert mask[0, 2].item()


# ===========================================================================
# 4. SpeculativeCacheBookkeeper combined view
# ===========================================================================

class TestBookkeeperCombinedView:
    """combined_cached_mask() is the union of K_spec and K_stable."""

    def test_combined_is_union(self):
        bk = SpeculativeCacheBookkeeper(batch_size=1, seq_len=6, device=torch.device("cpu"))
        bk.step_spec(_bool([[True, False, False, False, False, False]]))
        bk.step_stable(
            _bool([[False, False, True, False, False, False]]),
            _bool([[False, False, False, False, False, False]]),
        )
        combined = bk.combined_cached_mask()
        expected = [True, False, True, False, False, False]
        assert combined[0].tolist() == expected

    def test_get_cached_mask_is_alias_for_combined(self):
        bk = SpeculativeCacheBookkeeper(batch_size=1, seq_len=4, device=torch.device("cpu"))
        bk.step_spec(_bool([[True, False, True, False]]))
        bk.step_stable(
            _bool([[False, True, False, True]]),
            _bool([[False, False, False, False]]),
        )
        assert (bk.get_cached_mask() == bk.combined_cached_mask()).all()

    def test_reset_clears_both_pools(self):
        bk = SpeculativeCacheBookkeeper(batch_size=1, seq_len=4, device=torch.device("cpu"))
        bk.step_spec(torch.ones(1, 4, dtype=torch.bool))
        bk.step_stable(torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool))
        bk.reset()
        assert not bk.spec.cached.any()
        assert not bk.stable.cached.any()


# ===========================================================================
# 5. GRPO reward computation
# ===========================================================================

class TestGRPORewardComputation:
    """compute_reward component tests using minimal stub tokenizer/trajectories."""

    def _make_tokenizer(self, decoded_text: str):
        """Stub tokenizer that always decodes to the given text."""
        tok = types.SimpleNamespace()
        tok.decode = lambda ids, skip_special_tokens=True: decoded_text
        return tok

    def _make_traj(self, B: int, L_gen: int = 4, T: int = 8, *, thrash: float = 0.0):
        """Minimal stub trajectory compatible with compute_reward."""
        traj = types.SimpleNamespace()
        device = torch.device("cpu")
        # One step of zero thrash / zero cache
        traj.thrash_counts = [torch.full((B,), thrash, device=device)]
        traj.cached_fractions = [torch.zeros(B, device=device)]
        traj.spec_cached_fractions = [torch.zeros(B, device=device)]
        traj.stable_cached_fractions = [torch.zeros(B, device=device)]
        traj.actions = [{"u_t": torch.ones(B, L_gen)}]  # at least one active step
        traj.final_tokens = None
        traj.completion_step = None
        traj.effective_flops = None
        traj.aux_compute_units = None
        traj.verifier_compute_units = None
        traj.baseline_compute_units = None
        traj.access_metrics = {}
        traj.cache_quality_f1 = []
        return traj

    def _base_cfg(self, alpha=1.0, beta=0.0, upw=0.0):
        return {
            "base_model": {"mask_token_id": 99},
            "grpo": {
                "alpha": alpha,
                "beta": beta,
                "unresolved_penalty_weight": upw,
                "evaluator": "gsm8k",
                "train_dataset": "gsm8k",
            },
            "data": {
                "train_dataset": "gsm8k",
                "train_split": "train",
            },
        }

    def test_correct_answer_gives_positive_reward(self):
        """A correct answer with zero thrash and alpha=0 (no speed penalty) gives reward=1."""
        from aoae.train_grpo import compute_reward

        B, L_gen, T = 1, 4, 8
        # Tokens that decode to something containing "#### 42"
        generated = torch.zeros(B, L_gen, dtype=torch.long)
        # alpha=0 disables the speed exponent: speed_factor = (1 - eff_flops)^0 = 1.0
        cfg = self._base_cfg(alpha=0.0, beta=0.0)
        tok = self._make_tokenizer("The answer is #### 42")
        traj = self._make_traj(B, L_gen, T)
        traj.completion_step = torch.tensor([T])

        rewards, components = compute_reward(
            generated, ["42"], tok, traj, cfg, T
        )

        assert components["correctness"][0].item() == 1.0, "Should be correct"
        assert rewards[0].item() > 0.0, "Correct answer with alpha=0 must yield R=correctness>0"

    def test_wrong_answer_gives_zero_reward(self):
        """An incorrect answer gives correctness=0 → reward=0 (no thrash, no speed credit)."""
        from aoae.train_grpo import compute_reward

        B, L_gen, T = 1, 4, 8
        generated = torch.zeros(B, L_gen, dtype=torch.long)
        cfg = self._base_cfg(alpha=1.0, beta=0.0)
        tok = self._make_tokenizer("The answer is #### 7")
        traj = self._make_traj(B, L_gen, T)
        traj.completion_step = torch.tensor([T])

        rewards, components = compute_reward(
            generated, ["42"], tok, traj, cfg, T
        )

        assert components["correctness"][0].item() == 0.0
        assert rewards[0].item() == 0.0, "Wrong answer with no thrash → reward must be 0"

    def test_thrash_penalty_reduces_reward(self):
        """Non-zero thrash count with beta>0 reduces reward below correctness*speed.

        alpha=0 ensures speed_factor=1 so the baseline reward is exactly correctness=1,
        making the reduction from thrash clearly observable.
        """
        from aoae.train_grpo import compute_reward

        B, L_gen, T = 1, 4, 8
        generated = torch.zeros(B, L_gen, dtype=torch.long)
        cfg = self._base_cfg(alpha=0.0, beta=1.0)
        tok = self._make_tokenizer("The answer is #### 42")

        traj_clean = self._make_traj(B, L_gen, T, thrash=0.0)
        traj_clean.completion_step = torch.tensor([T])
        rewards_clean, _ = compute_reward(generated, ["42"], tok, traj_clean, cfg, T)

        traj_thrash = self._make_traj(B, L_gen, T, thrash=2.0)
        traj_thrash.completion_step = torch.tensor([T])
        rewards_thrash, components = compute_reward(generated, ["42"], tok, traj_thrash, cfg, T)

        assert components["total_thrash"][0].item() == 2.0
        assert components["thrash_penalty"][0].item() > 0.0
        assert rewards_thrash[0].item() < rewards_clean[0].item(), (
            "Thrash should reduce reward"
        )

    def test_unresolved_mask_penalty_reduces_reward(self):
        """Unresolved mask tokens at end of generation penalize the reward.

        alpha=0 ensures speed_factor=1 so correctness=1 is the clean baseline.
        """
        from aoae.train_grpo import compute_reward

        MASK_ID = 99
        B, L_gen, T = 1, 4, 8
        # Half the generated positions are still masks.
        generated = torch.tensor([[MASK_ID, MASK_ID, 5, 6]], dtype=torch.long)
        cfg = self._base_cfg(alpha=0.0, beta=0.0, upw=1.0)
        tok = self._make_tokenizer("The answer is #### 42")

        traj_no_mask = self._make_traj(B, L_gen, T)
        traj_no_mask.final_tokens = torch.tensor([[5, 6, 5, 6]])
        traj_no_mask.completion_step = torch.tensor([T])
        rewards_clean, _ = compute_reward(generated, ["42"], tok, traj_no_mask, cfg, T)

        traj_masked = self._make_traj(B, L_gen, T)
        traj_masked.final_tokens = torch.tensor([[MASK_ID, MASK_ID, 5, 6]])
        traj_masked.completion_step = torch.tensor([T])
        rewards_masked, components = compute_reward(generated, ["42"], tok, traj_masked, cfg, T)

        assert components["unresolved_penalty"][0].item() > 0.0, (
            "Should have a non-zero unresolved penalty"
        )
        assert rewards_masked[0].item() < rewards_clean[0].item(), (
            "Unresolved masks must reduce reward"
        )

    def test_reward_correctness_times_speed_baseline(self):
        """reward = correctness * speed_factor when beta=0, no unresolved penalty.

        We use alpha=0 so speed_factor=1 regardless of effective_flops, making the
        identity reward = correctness * 1 = correctness easy to verify numerically.
        """
        from aoae.train_grpo import compute_reward

        B, L_gen, T = 1, 4, 8
        generated = torch.zeros(B, L_gen, dtype=torch.long)
        cfg = self._base_cfg(alpha=0.0, beta=0.0)
        tok = self._make_tokenizer("#### 42")

        traj = self._make_traj(B, L_gen, T)
        traj.completion_step = torch.tensor([T])
        rewards, components = compute_reward(generated, ["42"], tok, traj, cfg, T)

        c = components["correctness"][0].item()
        s = components["speed_factor"][0].item()
        assert s == pytest.approx(1.0), "alpha=0 must give speed_factor=1"
        assert abs(rewards[0].item() - c * s) < 1e-5, (
            f"reward={rewards[0].item():.6f} should equal correctness({c})*speed({s})={c*s:.6f}"
        )


# ===========================================================================
# 6. Fingerprint stability — llada21_official is excluded
# ===========================================================================

class TestFingerprintStability:
    """build_grpo_config_fingerprint excludes inference.llada21_official."""

    def _base_cfg(self):
        return {
            "base_model": {"name": "llada-8b-instruct"},
            "soft_mask": {"top_k": 3},
            "policy": {"d_model": 128},
            "prism": {"hidden_dim": 64},
            "grpo": {"alpha": 2.0, "beta": 0.1, "max_steps": 500},
            "inference": {"steps": 32, "gen_length": 128},
            "data": {
                "train_dataset": "gsm8k",
                "train_split": "train",
                "train_max_samples": None,
                "max_prompt_len": 512,
                "max_answer_len": 256,
            },
        }

    def test_fingerprint_stable_under_llada21_official_addition(self):
        """Adding inference.llada21_official to a config must NOT change the fingerprint."""
        cfg_without = self._base_cfg()
        fp_without = build_grpo_config_fingerprint(cfg_without)

        cfg_with = self._base_cfg()
        cfg_with["inference"]["llada21_official"] = True
        fp_with = build_grpo_config_fingerprint(cfg_with)

        assert fp_without == fp_with, (
            f"Fingerprint changed when adding llada21_official:\n"
            f"  without: {fp_without}\n"
            f"  with:    {fp_with}"
        )

    def test_fingerprint_stable_under_llada21_official_removal(self):
        """Removing inference.llada21_official also must NOT change the fingerprint."""
        cfg_with = self._base_cfg()
        cfg_with["inference"]["llada21_official"] = False
        fp_with = build_grpo_config_fingerprint(cfg_with)

        cfg_without = self._base_cfg()
        fp_without = build_grpo_config_fingerprint(cfg_without)

        assert fp_with == fp_without

    def test_fingerprint_changes_on_grpo_hyperparameter_change(self):
        """Sanity: changing a real hyperparameter (alpha) DOES change the fingerprint."""
        cfg_a = self._base_cfg()
        cfg_b = self._base_cfg()
        cfg_b["grpo"]["alpha"] = 99.9

        assert build_grpo_config_fingerprint(cfg_a) != build_grpo_config_fingerprint(cfg_b), (
            "Changing grpo.alpha must change the fingerprint"
        )

    def test_fingerprint_changes_on_inference_steps_change(self):
        """Sanity: changing inference.steps (not eval-only) changes the fingerprint."""
        cfg_a = self._base_cfg()
        cfg_b = self._base_cfg()
        cfg_b["inference"]["steps"] = 999

        assert build_grpo_config_fingerprint(cfg_a) != build_grpo_config_fingerprint(cfg_b)

    def test_fingerprint_changes_on_tp_size_change(self):
        """Changing vLLM TP size changes the MoE execution path and invalidates GRPO."""
        cfg_a = self._base_cfg()
        cfg_b = self._base_cfg()
        cfg_a["hardware"] = {"tp_size": 1}
        cfg_b["hardware"] = {"tp_size": 2}

        assert build_grpo_config_fingerprint(cfg_a) != build_grpo_config_fingerprint(cfg_b)

    def test_fingerprint_stable_under_min_checkpoint_reward_change(self):
        """min_checkpoint_reward is a quality gate, not a hyperparameter — excluded."""
        cfg_a = self._base_cfg()
        cfg_a["grpo"]["min_checkpoint_reward"] = 0.5
        cfg_b = self._base_cfg()
        cfg_b["grpo"]["min_checkpoint_reward"] = 0.9

        assert build_grpo_config_fingerprint(cfg_a) == build_grpo_config_fingerprint(cfg_b), (
            "min_checkpoint_reward must be excluded from the fingerprint"
        )
