"""Phase 3: Comprehensive test suite covering gaps identified in the audit.

Tests:
  1. Positional cache: build_access_set, compute_next_h_access_metrics, budget enforcement
  2. End-to-end reward: full compute_reward with trajectory
  3. Composed prediction: single-model and dual-model numeric correctness
  4. Soft MoE: real routing entropy and runtime tau_r
  5. Cache invalidation consistency
  6. GRPO training step: one step with mock model, verify loss + gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


VOCAB, DIM, MASK_ID = 100, 64, 99


# ======================================================================
# 1. Positional Cache
# ======================================================================


class TestPositionalCache:
    def test_build_access_set_mandatory_positions(self):
        from aoae.positional_cache import build_access_set

        B, L = 1, 16
        actions = {
            "u_t": torch.zeros(B, L),
            "r_t": torch.zeros(B, L),
            "kappa_t": torch.zeros(B, L),
            "q_t": torch.zeros(B, L),
        }
        actions["u_t"][0, 3] = 1.0
        actions["r_t"][0, 7] = 1.0

        policy_out = {
            "access_probs": torch.ones(B, L) * 0.5,
            "access_logits": torch.zeros(B, L),
        }
        cfg = {
            "inference": {
                "positional_cache": {
                    "enabled": True,
                    "horizon": 4,
                    "refresh_budget": 4,
                    "force_mandatory": True,
                    "candidate_policy": "learned_topb",
                    "age_cap": 64,
                    "use_topb_from_probs": True,
                    "window_radius": 8,
                }
            }
        }

        q_exec, q_mandatory, stats = build_access_set(actions, policy_out, cfg)
        assert q_exec.shape == (B, L)
        assert q_mandatory.shape == (B, L)
        assert q_mandatory[0, 3].item() == 1.0
        assert q_mandatory[0, 7].item() == 1.0
        assert q_exec[0, 3].item() == 1.0
        assert q_exec[0, 7].item() == 1.0

    def test_build_access_set_budget_limits_optional(self):
        from aoae.positional_cache import build_access_set

        B, L = 1, 32
        actions = {
            "u_t": torch.zeros(B, L),
            "r_t": torch.zeros(B, L),
            "kappa_t": torch.zeros(B, L),
            "q_t": torch.ones(B, L),
        }
        policy_out = {
            "access_probs": torch.ones(B, L),
            "access_logits": torch.randn(B, L),
        }
        cfg = {
            "inference": {
                "positional_cache": {
                    "enabled": True,
                    "horizon": 4,
                    "refresh_budget": 4,
                    "force_mandatory": True,
                    "candidate_policy": "learned_topb",
                    "age_cap": 64,
                    "use_topb_from_probs": True,
                    "window_radius": 8,
                }
            }
        }
        q_exec, _, stats = build_access_set(actions, policy_out, cfg)
        total_accessed = q_exec.sum().item()
        assert total_accessed <= 4 + 1  # budget + possible mandatory overlap

    def test_next_h_access_metrics(self):
        from aoae.positional_cache import compute_next_h_access_metrics

        B, L = 1, 10
        # Simulate 3 steps of access sets and corresponding edits
        access_steps = [
            torch.zeros(B, L),
            torch.zeros(B, L),
            torch.zeros(B, L),
        ]
        access_steps[0][0, :5] = 1.0
        access_steps[1][0, 2:7] = 1.0
        access_steps[2][0, :3] = 1.0

        changed_steps = [
            torch.zeros(B, L),
            torch.zeros(B, L),
            torch.zeros(B, L),
        ]
        changed_steps[0][0, :3] = 1.0
        changed_steps[1][0, 3:6] = 1.0
        changed_steps[2][0, 1:4] = 1.0

        mandatory_steps = [torch.zeros(B, L) for _ in range(3)]

        metrics = compute_next_h_access_metrics(access_steps, changed_steps, mandatory_steps, horizon=2)
        assert "access_next_h_precision" in metrics
        assert "access_next_h_recall" in metrics

    def test_access_diagnostics_are_averaged_for_reporting(self):
        from aoae.positional_cache import summarize_access_diagnostics

        metrics = summarize_access_diagnostics([
            {
                "access_rate": 0.25,
                "access_mandatory_rate": 0.10,
                "access_optional_rate": 0.15,
                "access_budget_utilization": 0.50,
                "access_effective_budget": 4.0,
            },
            {
                "access_rate": 0.75,
                "access_mandatory_rate": 0.30,
                "access_optional_rate": 0.45,
                "access_budget_utilization": 1.00,
                "access_effective_budget": 8.0,
            },
        ])

        assert metrics["access_rate"] == pytest.approx(0.50)
        assert metrics["access_mandatory_rate"] == pytest.approx(0.20)
        assert metrics["access_optional_rate"] == pytest.approx(0.30)
        assert metrics["access_budget_utilization"] == pytest.approx(0.75)
        assert metrics["access_effective_budget"] == pytest.approx(6.0)

    def test_next_h_access_metrics_per_sample_separates_group_members(self):
        from aoae.positional_cache import compute_next_h_access_metrics_per_sample

        access_steps = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        changed_steps = [torch.tensor([[1.0, 0.0], [1.0, 0.0]])]
        mandatory_steps = [torch.zeros(2, 2)]

        metrics = compute_next_h_access_metrics_per_sample(
            access_steps,
            changed_steps,
            mandatory_steps,
            horizon=1,
        )

        assert metrics["access_next_h_spec_f1"].tolist() == pytest.approx([1.0, 0.0])


# ======================================================================
# 2. End-to-end reward
# ======================================================================


class TestEndToEndReward:
    def test_correct_fast_generation_high_reward(self):
        from aoae.train_grpo import compute_reward, AOAETrajectory
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "The answer is \\boxed{42}."
        reference = ["42"]

        traj = AOAETrajectory()
        traj.completion_step = torch.tensor([5.0])
        traj.thrash_counts = [torch.tensor([0.0])]
        traj.actions = [{"u_t": torch.ones(1, 16)}]

        gen_tokens = torch.randint(0, 50, (1, 16))
        cfg = {"grpo": {"alpha": 1.0, "beta": 0.1, "access_reward_weight": 0.0}}
        reward, _ = compute_reward(gen_tokens, reference, tokenizer, traj, cfg, T=64)
        assert reward.shape == (1,)
        assert reward[0].item() > 0

    def test_incorrect_generation_zero_correctness(self):
        from aoae.train_grpo import compute_reward, AOAETrajectory
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "I don't know."
        reference = ["42"]

        traj = AOAETrajectory()
        traj.completion_step = torch.tensor([5.0])
        traj.thrash_counts = [torch.tensor([0.0])]
        traj.actions = [{"u_t": torch.ones(1, 16)}]

        gen_tokens = torch.randint(0, 50, (1, 16))
        cfg = {"grpo": {"alpha": 1.0, "beta": 0.0, "access_reward_weight": 0.0}}
        reward = compute_reward(gen_tokens, reference, tokenizer, traj, cfg, T=64)
        assert reward[0].item() == 0.0

    def test_thrashing_penalty(self):
        from aoae.train_grpo import compute_reward, AOAETrajectory
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "\\boxed{42}"
        reference = ["42"]

        traj = AOAETrajectory()
        traj.completion_step = torch.tensor([5.0])
        traj.thrash_counts = [torch.tensor([10.0])]
        traj.actions = [{"u_t": torch.ones(1, 16)}]

        gen_tokens = torch.randint(0, 50, (1, 16))
        cfg_no_thrash = {"grpo": {"alpha": 1.0, "beta": 0.0, "access_reward_weight": 0.0}}
        cfg_thrash = {"grpo": {"alpha": 1.0, "beta": 0.5, "access_reward_weight": 0.0}}

        r_no = compute_reward(gen_tokens, reference, tokenizer, traj, cfg_no_thrash, T=64)
        r_yes = compute_reward(gen_tokens, reference, tokenizer, traj, cfg_thrash, T=64)
        assert r_yes[0].item() < r_no[0].item()


# ======================================================================
# 3. Composed prediction numeric correctness
# ======================================================================


class TestComposedPredictionNumerics:
    def test_single_model_no_gamma_passthrough(self):
        from aoae.models.composed_prediction import compose_prediction
        logits = torch.randn(2, 4, 10)
        cache_probs = torch.ones(2, 4) * 0.5
        out = compose_prediction(logits, cache_probs, gamma=0.0)
        assert torch.allclose(out, logits)

    def test_single_model_sharpening_increases_max_prob(self):
        from aoae.models.composed_prediction import compose_prediction
        logits = torch.randn(1, 4, 20)
        cache_probs = torch.ones(1, 4)
        out = compose_prediction(logits, cache_probs, gamma=1.0)
        base_max = F.softmax(logits, dim=-1).max(dim=-1).values
        composed_max = F.softmax(out, dim=-1).max(dim=-1).values
        assert (composed_max >= base_max - 1e-5).all()

    def test_dual_model_agreement_concentrates(self):
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = torch.randn(1, 4, 20)
        aux = pri.clone()
        agreement = torch.ones(1, 4)
        out = compose_prediction_dual(pri, aux, agreement, gamma=0.5)
        base_ent = -(F.softmax(pri, -1) * F.log_softmax(pri, -1)).sum(-1)
        comp_ent = -(F.softmax(out, -1) * F.log_softmax(out, -1)).sum(-1)
        assert (comp_ent <= base_ent + 1e-5).all()

    def test_dual_model_disagreement_passthrough(self):
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = torch.randn(1, 4, 20)
        aux = torch.randn(1, 4, 20)
        agreement = torch.zeros(1, 4)
        out = compose_prediction_dual(pri, aux, agreement, gamma=0.5)
        assert torch.allclose(out, pri)


# ======================================================================
# 4. Soft MoE: real routing entropy and runtime tau_r
# ======================================================================


class TestSoftMoERoutingEntropy:
    def _make_mock_moe_model(self, num_experts=8, top_k=2, tau_r=0.01):
        from aoae.models.soft_moe import SoftMoERouter, patch_model_with_soft_routing

        class FakeGate(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                self.routed_scaling_factor = 1.0
                self.group_limited_topk = True
                self.weight = nn.Parameter(torch.randn(num_experts, DIM))

        class FakeMoEBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = FakeGate()
                self.experts = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(num_experts)])

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = FakeMoEBlock()
                self.block1 = FakeMoEBlock()

        model = FakeModel()
        patch_model_with_soft_routing(model, tau_r=tau_r)
        return model

    def test_entropy_after_forward_is_not_nan(self):
        from aoae.models.soft_moe import compute_routing_entropy
        model = self._make_mock_moe_model(tau_r=0.01)
        hidden = torch.randn(2, 10, DIM)
        for name, mod in model.named_modules():
            if hasattr(mod, 'forward') and hasattr(mod, '_tau_r'):
                mod(hidden)

        info = compute_routing_entropy(model)
        assert info["num_layers"] == 2
        import math
        assert not math.isnan(info["mean_entropy"])
        assert info["mean_entropy"] > 0
        assert info["mean_entropy"] <= info["max_possible_entropy"]

    def test_runtime_tau_r_change(self):
        from aoae.models.soft_moe import set_routing_temperature, compute_routing_entropy
        model = self._make_mock_moe_model(tau_r=0.01)
        hidden = torch.randn(2, 10, DIM)
        for name, mod in model.named_modules():
            if hasattr(mod, '_tau_r'):
                mod(hidden)
        ent_low = compute_routing_entropy(model)["mean_entropy"]

        set_routing_temperature(model, 1.0)
        for name, mod in model.named_modules():
            if hasattr(mod, '_tau_r'):
                mod(hidden)
        ent_high = compute_routing_entropy(model)["mean_entropy"]

        assert ent_high > ent_low

    def test_invalid_tau_r_raises(self):
        from aoae.models.soft_moe import set_routing_temperature
        model = self._make_mock_moe_model(tau_r=0.1)
        with pytest.raises(ValueError):
            set_routing_temperature(model, -0.1)


# ======================================================================
# 5. Cache invalidation consistency
# ======================================================================


class TestCacheConsistency:
    def test_commit_then_invalidate_then_query(self):
        from aoae.cache import DKVCacheManager
        cm = DKVCacheManager(1, 16, torch.device("cpu"))
        commit_mask = torch.zeros(1, 16, dtype=torch.bool)
        commit_mask[0, 0] = True
        commit_mask[0, 2] = True
        cm.commit(commit_mask)
        assert cm.cached[0, 0].item() is True
        assert cm.cached[0, 2].item() is True

        inv_mask = torch.zeros(1, 16, dtype=torch.bool)
        inv_mask[0, 2] = True
        thrash = cm.count_thrash(inv_mask.float())
        assert thrash[0].item() == 1.0
        cm.invalidate(inv_mask)
        assert cm.cached[0, 0].item() is True
        assert cm.cached[0, 2].item() is False

    def test_double_commit_no_thrash(self):
        from aoae.cache import DKVCacheManager
        cm = DKVCacheManager(1, 8, torch.device("cpu"))
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, :2] = True
        cm.commit(mask)
        cm.commit(mask)
        # No edit, so thrash with non-cached positions should be 0
        no_edit = torch.zeros(1, 8)
        assert cm.count_thrash(no_edit)[0].item() == 0.0

    def test_reset_clears_everything(self):
        from aoae.cache import DKVCacheManager
        cm = DKVCacheManager(1, 8, torch.device("cpu"))
        cm.commit(torch.ones(1, 8, dtype=torch.bool))
        cm.invalidate(torch.ones(1, 8, dtype=torch.bool))
        cm.reset()
        assert not cm.cached.any()
        assert cm.count_thrash(torch.ones(1, 8))[0].item() == 0.0


# ======================================================================
# 6. GRPO training step: gradient flow
# ======================================================================


class TestGRPOTrainingStep:
    def test_loss_computes_and_backprops(self):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.train_grpo import compute_grpo_loss

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "soft_mask": {"top_k": 3, "omega_s_init": 0.8, "omega_a_init": 1.0, "omega_b_init": 2.0},
            "policy": {"d_model": 32, "n_layers": 1, "n_heads": 4, "dropout": 0.0},
            "grpo": {"clip_eps": 0.2, "alpha": 1.0, "beta": 0.1, "policy_temperature": 1.0},
        }

        embed_w = torch.randn(VOCAB, DIM)
        sm = SoftMaskedState(cfg, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(cfg, input_dim=DIM)

        def make_traj_dict(n_steps=3):
            """Build trajectory dict matching compute_grpo_loss expected format."""
            H_t_list, weighted_embeds_list, entropy_list = [], [], []
            mask_ind_list, step_fracs = [], []
            actions_list, old_lps = [], []
            for t in range(n_steps):
                logits = torch.randn(1, 10, VOCAB)
                mask_ind = torch.randint(0, 2, (1, 10)).bool()
                H, _, entropy, weighted_embeds = sm(
                    logits, mask_ind, t / n_steps, return_weighted=True
                )
                out = pol(H, mask_ind, t / n_steps)
                actions = pol.sample_actions(out, mask_ind)
                old_lp = pol.log_prob(out, actions)
                H_t_list.append(H.detach())
                weighted_embeds_list.append(weighted_embeds.detach())
                entropy_list.append(entropy.detach())
                mask_ind_list.append(mask_ind)
                step_fracs.append(t / n_steps)
                actions_list.append({k: v.detach() for k, v in actions.items()})
                old_lps.append(old_lp.detach())
            return {
                "H_t_list": H_t_list,
                "weighted_embeds_list": weighted_embeds_list,
                "entropy_list": entropy_list,
                "mask_ind_list": mask_ind_list,
                "step_fracs": step_fracs,
                "actions_list": actions_list,
                "old_log_probs": old_lps,
            }

        trajectories = [make_traj_dict(), make_traj_dict()]
        advantages = torch.tensor([1.0, -0.5])

        loss = compute_grpo_loss(pol, sm, trajectories, advantages, clip_eps=0.2)
        assert torch.isfinite(loss)

        loss.backward()
        grad_norms = []
        for p in pol.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        assert any(g > 0 for g in grad_norms)
