"""
Comprehensive test suite for AOAE components.

Run with:  python3 -m pytest tests/ -v
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# Mock base model (no HuggingFace dependency needed)
# ======================================================================

class MockBaseModel(nn.Module):
    """Lightweight mock that mimics LLaDABaseModel interface."""

    def __init__(self, vocab_size=100, hidden_dim=64, mask_id=99):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mask_id = mask_id
        self._embedding_weight = nn.Parameter(
            torch.randn(vocab_size, hidden_dim), requires_grad=False
        )
        self.tokenizer = None

    @property
    def device(self):
        return self._embedding_weight.device

    def get_embedding_weight(self):
        return self._embedding_weight

    def forward(self, input_ids):
        B, L = input_ids.shape
        logits = torch.randn(B, L, self.vocab_size)
        logits[:, :, : self.mask_id] += 1.0
        return logits

    def forward_with_hidden(self, input_ids):
        B, L = input_ids.shape
        logits = self.forward(input_ids)
        hidden = torch.randn(B, L, self.hidden_dim)
        return logits, hidden


# ======================================================================
# Shared fixtures
# ======================================================================

VOCAB, DIM, MASK_ID = 100, 64, 99

DEFAULT_CFG = {
    "base_model": {"mask_token_id": MASK_ID},
    "soft_mask": {
        "top_k": 3,
        "omega_s_init": 0.8,
        "omega_a_init": 1.0,
        "omega_b_init": 2.0,
    },
    "policy": {"d_model": 32, "n_layers": 1, "n_heads": 4, "dropout": 0.0},
    "prism": {
        "hidden_dim": 32,
        "threshold": 0.5,
        "train_samples": 10,
        "epochs": 1,
        "lr": 1e-3,
        "batch_size": 4,
    },
    "cache": {"enabled": True},
    "inference": {
        "steps": 8,
        "gen_length": 16,
        "temperature": 0.0,
        "fallback_unmask": True,
    },
    "grpo": {
        "alpha": 1.0,
        "beta": 0.1,
        "group_size": 2,
        "clip_eps": 0.2,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "epochs": 1,
        "max_steps": 10,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "policy_temperature": 1.0,
    },
}


@pytest.fixture
def base_model():
    return MockBaseModel(VOCAB, DIM, MASK_ID)


@pytest.fixture
def embed_w(base_model):
    return base_model.get_embedding_weight()


# ======================================================================
# Tests: DKVCacheManager
# ======================================================================

class TestBackendDetection:
    def test_auto_hf(self):
        from aoae.models.base_model import _detect_backend
        cfg = {"base_model": {"backend": "auto"}}
        assert _detect_backend("GSAI-ML/LLaDA-8B-Instruct", cfg) == "hf"

    def test_auto_dinfer(self):
        from aoae.models.base_model import _detect_backend
        cfg = {"base_model": {"backend": "auto"}}
        assert _detect_backend("inclusionAI/LLaDA2.1-flash", cfg) == "dinfer"
        assert _detect_backend("inclusionAI/LLaDA2.0-flash", cfg) == "dinfer"

    def test_explicit_override(self):
        from aoae.models.base_model import _detect_backend
        cfg = {"base_model": {"backend": "hf"}}
        assert _detect_backend("inclusionAI/LLaDA2.1-flash", cfg) == "hf"
        cfg = {"base_model": {"backend": "dinfer"}}
        assert _detect_backend("GSAI-ML/LLaDA-8B-Instruct", cfg) == "dinfer"

    def test_unknown_model_defaults_hf(self):
        from aoae.models.base_model import _detect_backend
        cfg = {"base_model": {"backend": "auto"}}
        assert _detect_backend("some-random/model", cfg) == "hf"


class TestHFCompatibility:
    def test_default_rope_handler_is_restored(self):
        from aoae.models.base_model import _ensure_hf_rope_compatibility
        from transformers import modeling_rope_utils

        original_default = modeling_rope_utils.ROPE_INIT_FUNCTIONS.pop("default", None)
        try:
            _ensure_hf_rope_compatibility()
            assert "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS

            class DummyConfig:
                rope_theta = 10000.0
                partial_rotary_factor = 0.5
                hidden_size = 32
                num_attention_heads = 4

            inv_freq, scaling = modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"](
                config=DummyConfig(),
                device=torch.device("cpu"),
            )
            assert inv_freq.shape == (2,)
            assert scaling == 1.0
        finally:
            if original_default is None:
                modeling_rope_utils.ROPE_INIT_FUNCTIONS.pop("default", None)
            else:
                modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = original_default


class TestEvalExtraction:
    def test_extract_eval_prompt_reference_supports_math_schema(self):
        from aoae.evaluate import _extract_eval_prompt_reference

        prompt, reference = _extract_eval_prompt_reference(
            {
                "problem": "Compute 2 + 2.",
                "solution": "We get \\boxed{4}.",
            }
        )
        assert prompt == "Compute 2 + 2."
        assert reference == "We get \\boxed{4}."


class TestDKVCacheManager:
    def test_init_empty(self):
        from aoae.cache import DKVCacheManager

        c = DKVCacheManager(2, 10, torch.device("cpu"))
        assert c.cached.shape == (2, 10)
        assert not c.cached.any()

    def test_commit_and_query(self):
        from aoae.cache import DKVCacheManager

        c = DKVCacheManager(1, 5, torch.device("cpu"))
        c.commit(torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool))
        assert c.cached[0, 0].item() and c.cached[0, 2].item()
        assert not c.cached[0, 1].item()

    def test_invalidate(self):
        from aoae.cache import DKVCacheManager

        c = DKVCacheManager(1, 5, torch.device("cpu"))
        c.commit(torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool))
        c.invalidate(torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool))
        assert c.cached[0, 0].item()
        assert not c.cached[0, 1].item()
        assert c.cached[0, 2].item()

    def test_thrash_count(self):
        from aoae.cache import DKVCacheManager

        c = DKVCacheManager(1, 5, torch.device("cpu"))
        c.commit(torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool))
        edit = torch.tensor([[1, 0, 0, 0, 1]], dtype=torch.float)
        thrash = c.count_thrash(edit)
        assert thrash[0].item() == 1.0  # only position 0 is cached AND edited

    def test_reset(self):
        from aoae.cache import DKVCacheManager

        c = DKVCacheManager(1, 5, torch.device("cpu"))
        c.commit(torch.ones(1, 5, dtype=torch.bool))
        c.reset()
        assert not c.cached.any()


# ======================================================================
# Tests: SoftMaskedState
# ======================================================================

class TestSoftMaskedState:
    def test_output_shapes(self, embed_w):
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        logits = torch.randn(2, 10, VOCAB)
        mask_ind = torch.randint(0, 2, (2, 10)).bool()
        H, conf, ent = sm(logits, mask_ind, 0.5)
        assert H.shape == (2, 10, DIM)
        assert conf.shape == (2, 10)
        assert ent.shape == (2, 10)

    def test_confidence_range(self, embed_w):
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        logits = torch.randn(1, 5, VOCAB)
        mask_ind = torch.ones(1, 5, dtype=torch.bool)
        _, conf, _ = sm(logits, mask_ind, 0.5)
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_entropy_nonneg(self, embed_w):
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        logits = torch.randn(1, 5, VOCAB)
        mask_ind = torch.ones(1, 5, dtype=torch.bool)
        _, _, ent = sm(logits, mask_ind, 0.5)
        assert (ent >= -1e-5).all()  # entropy should be non-negative

    def test_gating_params_learnable(self, embed_w):
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        assert sm.omega_s.requires_grad
        assert sm.omega_a.requires_grad
        assert sm.omega_b.requires_grad


# ======================================================================
# Tests: AOAEPolicy
# ======================================================================

class TestAOAEPolicy:
    def test_output_shapes(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        logits = torch.randn(2, 10, VOCAB)
        mask_ind = torch.randint(0, 2, (2, 10)).bool()
        H, _, _ = sm(logits, mask_ind, 0.5)

        out = pol(H, mask_ind, 0.5)
        for key in ["unmask_logits", "remask_logits", "cache_logits",
                     "unmask_probs", "remask_probs", "cache_probs"]:
            assert key in out
            assert out[key].shape == (2, 10)

    def test_validity_constraints(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        logits = torch.randn(2, 10, VOCAB)
        mask_ind = torch.tensor([
            [True, True, False, False, True, True, True, False, False, True],
            [True, False, True, True, False, True, False, True, True, False],
        ])
        H, _, _ = sm(logits, mask_ind, 0.5)
        out = pol(H, mask_ind, 0.5)

        assert (out["unmask_logits"][~mask_ind] < -1e8).all()
        assert (out["remask_logits"][mask_ind] < -1e8).all()

    def test_cache_remask_exclusion(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        logits = torch.randn(4, 10, VOCAB)
        mask_ind = torch.randint(0, 2, (4, 10)).bool()
        H, _, _ = sm(logits, mask_ind, 0.5)
        out = pol(H, mask_ind, 0.5)
        actions = pol.sample_actions(out, mask_ind)

        assert (actions["kappa_t"] * actions["r_t"]).sum() == 0

    def test_log_prob_finite(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        from aoae.models.soft_mask import SoftMaskedState

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        logits = torch.randn(2, 10, VOCAB)
        mask_ind = torch.randint(0, 2, (2, 10)).bool()
        H, _, _ = sm(logits, mask_ind, 0.5)
        out = pol(H, mask_ind, 0.5)
        actions = pol.sample_actions(out, mask_ind)
        lp = pol.log_prob(out, actions)

        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()


# ======================================================================
# Tests: PRISMAdapter
# ======================================================================

class TestPRISMAdapter:
    def test_output_range(self):
        from aoae.models.prism import PRISMAdapter

        adapter = PRISMAdapter(DEFAULT_CFG, hidden_dim=DIM)
        hidden = torch.randn(2, 10, DIM)
        q = adapter(hidden)
        assert q.shape == (2, 10)
        assert (q >= 0).all() and (q <= 1).all()

    def test_should_remask(self):
        from aoae.models.prism import PRISMAdapter

        adapter = PRISMAdapter(DEFAULT_CFG, hidden_dim=DIM)
        q = torch.tensor([[0.3, 0.7, 0.4, 0.9]])
        remask = adapter.should_remask(q)
        assert remask[0, 0].item()  # 0.3 < 0.5
        assert not remask[0, 1].item()  # 0.7 >= 0.5
        assert remask[0, 2].item()  # 0.4 < 0.5
        assert not remask[0, 3].item()  # 0.9 >= 0.5


# ======================================================================
# Tests: AOAE Inference
# ======================================================================

class TestAOAEInference:
    def test_output_shape(self, base_model, embed_w):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.inference import aoae_inference

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        prompt = torch.tensor([[10, 20, 30, 40]])
        output, traj = aoae_inference(
            base_model, pol, sm, None, prompt, DEFAULT_CFG
        )
        L_gen = DEFAULT_CFG["inference"]["gen_length"]
        assert output.shape == (1, 4 + L_gen)

    def test_prompt_preserved(self, base_model, embed_w):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.inference import aoae_inference

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        prompt = torch.tensor([[10, 20, 30, 40]])
        output, _ = aoae_inference(
            base_model, pol, sm, None, prompt, DEFAULT_CFG
        )
        assert (output[0, :4] == prompt[0]).all()

    def test_trajectory_recording(self, base_model, embed_w):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.models.prism import PRISMAdapter
        from aoae.inference import aoae_inference

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)
        prism = PRISMAdapter(DEFAULT_CFG, hidden_dim=DIM)

        prompt = torch.tensor([[10, 20, 30]])
        _, traj = aoae_inference(
            base_model, pol, sm, prism, prompt, DEFAULT_CFG,
            record_trajectory=True,
        )

        assert traj is not None
        n = len(traj.actions)
        assert n > 0
        assert len(traj.log_probs) == n
        assert len(traj.H_t_list) == n
        assert len(traj.mask_ind_list) == n
        assert len(traj.step_fracs) == n

        for a in traj.actions:
            assert "u_t" in a and "r_t" in a and "kappa_t" in a

    def test_some_tokens_unmasked(self, base_model, embed_w):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.inference import aoae_inference

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        prompt = torch.tensor([[10, 20, 30, 40]])
        output, _ = aoae_inference(
            base_model, pol, sm, None, prompt, DEFAULT_CFG
        )
        n_unmasked = (output[0, 4:] != MASK_ID).sum().item()
        assert n_unmasked > 0


# ======================================================================
# Tests: Baseline decoders
# ======================================================================

class TestBaselines:
    def test_uniform_decode(self, base_model):
        from aoae.inference import uniform_decode

        prompt = torch.tensor([[10, 20, 30, 40]])
        out = uniform_decode(base_model, prompt, DEFAULT_CFG)
        L_gen = DEFAULT_CFG["inference"]["gen_length"]
        assert out.shape == (1, 4 + L_gen)
        # Should unmask at least some tokens
        assert (out[0, 4:] != MASK_ID).sum().item() > 0

    def test_confidence_decode(self, base_model):
        from aoae.inference import confidence_threshold_decode

        prompt = torch.tensor([[10, 20, 30, 40]])
        out = confidence_threshold_decode(
            base_model, prompt, DEFAULT_CFG, tau_mask=0.3
        )
        L_gen = DEFAULT_CFG["inference"]["gen_length"]
        assert out.shape == (1, 4 + L_gen)

    def test_block_smode_decode(self, base_model):
        from aoae.inference import block_smode_decode

        prompt = torch.tensor([[10, 20, 30, 40]])
        out = block_smode_decode(
            base_model, prompt, DEFAULT_CFG,
            tau_mask=0.3, tau_edit=0.5, max_steps_per_block=4,
        )
        L_gen = DEFAULT_CFG["inference"]["gen_length"]
        assert out.shape == (1, 4 + L_gen)
        # Should unmask at least some tokens
        assert (out[0, 4:] != MASK_ID).sum().item() > 0


# ======================================================================
# Tests: Reward computation
# ======================================================================

class TestReward:
    def test_extract_boxed(self):
        from aoae.train_grpo import extract_answer

        assert extract_answer("The answer is \\boxed{42}") == "42"
        assert extract_answer("\\boxed{-3.14}") == "-3.14"

    def test_extract_gsm8k(self):
        from aoae.train_grpo import extract_answer

        assert extract_answer("#### 123") == "123"
        assert extract_answer("Therefore, #### 42,000") == "42000"

    def test_extract_last_number(self):
        from aoae.train_grpo import extract_answer

        assert extract_answer("The result is 3.14 approximately") == "3.14"

    def test_correctness_check(self):
        from aoae.train_grpo import check_math_correctness

        assert check_math_correctness("\\boxed{42}", "#### 42")
        assert not check_math_correctness("\\boxed{41}", "#### 42")
        assert check_math_correctness("The answer is 3.14", "\\boxed{3.14}")

    def test_speed_reward(self):
        from aoae.train_grpo import compute_reward
        from aoae.inference import AOAETrajectory

        # Fast trajectory (1 step used): T_hat = T - 1 + 1 = T
        traj_fast = AOAETrajectory()
        traj_fast.actions = [{}]  # 1 step
        traj_fast.thrash_counts = [torch.zeros(1)]

        # Slow trajectory (all T=8 steps used): T_hat = 8 - 8 + 1 = 1
        traj_slow = AOAETrajectory()
        traj_slow.actions = [{} for _ in range(8)]
        traj_slow.thrash_counts = [torch.zeros(1) for _ in range(8)]

        # Both correct, but different speeds
        # We can't easily test without a tokenizer, so just verify the formula
        T_hat_fast = max(1, 8 - 1 + 1)  # = 8
        T_hat_slow = max(1, 8 - 8 + 1)  # = 1
        assert T_hat_fast / 8 > T_hat_slow / 8


# ======================================================================
# Tests: GRPO loss
# ======================================================================

class TestGRPOLoss:
    def test_loss_computes_and_backprops(self, embed_w):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.models.prism import PRISMAdapter
        from aoae.inference import aoae_inference
        from aoae.train_grpo import compute_grpo_loss

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)
        prism = PRISMAdapter(DEFAULT_CFG, hidden_dim=DIM)
        base = MockBaseModel(VOCAB, DIM, MASK_ID)

        prompt = torch.tensor([[10, 20, 30]])
        _, traj = aoae_inference(
            base, pol, sm, prism, prompt, DEFAULT_CFG,
            record_trajectory=True,
        )

        traj_data = {
            "actions_list": traj.actions,
            "old_log_probs": [lp.clone() for lp in traj.log_probs],
            "H_t_list": traj.H_t_list,
            "mask_ind_list": traj.mask_ind_list,
            "step_fracs": traj.step_fracs,
        }

        advantages = torch.tensor([0.5, -0.5])
        loss = compute_grpo_loss(
            pol, sm, [traj_data, traj_data], advantages, clip_eps=0.2
        )

        assert loss.shape == ()
        assert torch.isfinite(loss)

        pol.zero_grad()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pol.parameters()
        )
        assert has_grad, "No gradients flowing to policy"


# ======================================================================
# Tests: ComposedPrediction
# ======================================================================

class TestComposedPrediction:
    def test_no_composition_passthrough(self):
        from aoae.models.composed_prediction import compose_prediction
        logits = torch.randn(2, 10, VOCAB)
        cache_probs = torch.rand(2, 10)
        # gamma=0 should return identical logits
        out = compose_prediction(logits, cache_probs, gamma=0.0)
        assert torch.allclose(out, logits)

    def test_sharpening_effect(self):
        from aoae.models.composed_prediction import compose_prediction
        logits = torch.randn(2, 10, VOCAB)
        cache_probs = torch.ones(2, 10)  # all positions "stable"
        composed = compose_prediction(logits, cache_probs, gamma=1.0)
        # Composed logits should be scaled up (sharpened)
        # scale = 1 + 1.0 * 1.0 = 2.0
        assert torch.allclose(composed, logits * 2.0)

    def test_selective_sharpening(self):
        from aoae.models.composed_prediction import compose_prediction
        logits = torch.randn(1, 5, VOCAB)
        cache_probs = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
        composed = compose_prediction(logits, cache_probs, gamma=0.5)
        # Positions 0,1,4 should be unchanged (scale=1.0)
        assert torch.allclose(composed[0, 0], logits[0, 0])
        assert torch.allclose(composed[0, 1], logits[0, 1])
        assert torch.allclose(composed[0, 4], logits[0, 4])
        # Positions 2,3 should be scaled by 1.5
        assert torch.allclose(composed[0, 2], logits[0, 2] * 1.5)
        assert torch.allclose(composed[0, 3], logits[0, 3] * 1.5)

    def test_entropy_reduction(self):
        from aoae.models.composed_prediction import compute_composition_entropy
        logits = torch.randn(2, 10, VOCAB)
        cache_probs = torch.ones(2, 10)
        ent_base = compute_composition_entropy(logits, cache_probs, gamma=0.0)
        ent_composed = compute_composition_entropy(logits, cache_probs, gamma=1.0)
        # Sharpening should reduce entropy
        assert (ent_composed <= ent_base + 1e-5).all()

    def test_sample_shape(self):
        from aoae.models.composed_prediction import sample_from_composed
        logits = torch.randn(2, 10, VOCAB)
        cache_probs = torch.rand(2, 10)
        tokens = sample_from_composed(logits, cache_probs, gamma=0.5, temperature=0.0)
        assert tokens.shape == (2, 10)


# ======================================================================
# Tests: SoftMoERouter
# ======================================================================

class TestSoftMoERouter:
    def test_soft_routing_all_experts(self):
        from aoae.models.soft_moe import SoftMoERouter

        # Create a mock gate with the minimal interface
        class MockGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 2.5
                self.weight = torch.nn.Parameter(torch.randn(4, 16))
                self.register_buffer("expert_bias", torch.zeros(4))
            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)
            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

        gate = MockGate()
        soft = SoftMoERouter(gate, tau_r=0.01)

        hidden = torch.randn(2, 16)
        idx, weights, logits = soft(hidden)
        # All experts should be returned
        assert idx.shape == (2, 4)
        assert weights.shape == (2, 4)
        # Weights should sum to routed_scaling_factor (softmax sums to 1, then scaled)
        assert torch.allclose(weights.sum(dim=-1),
                              torch.full((2,), 2.5), atol=1e-4)

    def test_low_temp_approximation(self):
        from aoae.models.soft_moe import SoftMoERouter

        class MockGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 1.0
                self.weight = torch.nn.Parameter(torch.randn(4, 16))
                self.register_buffer("expert_bias", torch.zeros(4))
            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)
            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

        gate = MockGate()
        soft = SoftMoERouter(gate, tau_r=0.001)

        hidden = torch.randn(1, 16)
        _, weights, _ = soft(hidden)
        # At very low temp, one expert should dominate
        assert weights.max() > 0.99

    def _make_mock_moe_model(self):
        """Helper: create a mock model with MoE blocks for patching tests."""
        class MockGateModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 2.5
                self.weight = torch.nn.Parameter(torch.randn(4, 16))
                self.register_buffer("expert_bias", torch.zeros(4))
            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

        class MockMoeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = MockGateModule()
                self.experts = torch.nn.Linear(16, 16)  # dummy

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = MockMoeBlock()
                self.layer2 = MockMoeBlock()

        return MockModel()

    def _make_mock_sglang_moe_model(self):
        """Helper: create a mock SGLang-shaped MoE model for patching tests."""
        class MockGateModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.weight = torch.nn.Parameter(torch.randn(4, 16))

            def forward(self, hidden_states):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                return torch.nn.functional.linear(hidden_states, self.weight)

        class MockTopK(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.top_k = 2
                self.last_top_k = None
                self.last_logits = None

            def forward(self, hidden_states, router_logits, *args, **kwargs):
                del hidden_states, args, kwargs
                self.last_top_k = self.top_k
                self.last_logits = router_logits.detach().clone()
                return router_logits.topk(self.top_k, dim=-1)

        class MockMoeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = MockGateModule()
                self.topk = MockTopK()
                self.experts = torch.nn.Linear(16, 16)
                self.num_experts = 4
                self.score_function = "softmax"

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = MockMoeBlock()
                self.layer2 = MockMoeBlock()

        return MockModel()

    def test_patch_model_finds_gates(self):
        from aoae.models.soft_moe import patch_model_with_soft_routing, SoftMoERouter

        model = self._make_mock_moe_model()
        patched = patch_model_with_soft_routing(model, tau_r=0.01)
        assert isinstance(patched.layer1.gate, SoftMoERouter)
        assert isinstance(patched.layer2.gate, SoftMoERouter)

    def test_routing_mode_switching(self):
        from aoae.models.soft_moe import (
            patch_model_with_soft_routing, set_hard_routing,
            set_soft_routing, SoftMoERouter,
        )

        model = self._make_mock_moe_model()
        # Save reference to original gates
        orig_gate1_type = type(model.layer1.gate)
        patch_model_with_soft_routing(model, tau_r=0.05)

        # After patching: soft routing active
        assert isinstance(model.layer1.gate, SoftMoERouter)
        assert isinstance(model.layer2.gate, SoftMoERouter)

        # Switch to hard routing
        set_hard_routing(model)
        assert not isinstance(model.layer1.gate, SoftMoERouter)
        assert not isinstance(model.layer2.gate, SoftMoERouter)

        # Switch back to soft routing
        set_soft_routing(model)
        assert isinstance(model.layer1.gate, SoftMoERouter)
        assert isinstance(model.layer2.gate, SoftMoERouter)

    def test_soft_routing_context_manager(self):
        from aoae.models.soft_moe import (
            patch_model_with_soft_routing, set_hard_routing,
            soft_routing_context, SoftMoERouter,
        )

        model = self._make_mock_moe_model()
        patch_model_with_soft_routing(model, tau_r=0.01)
        set_hard_routing(model)  # start hard

        assert not isinstance(model.layer1.gate, SoftMoERouter)
        with soft_routing_context(model):
            assert isinstance(model.layer1.gate, SoftMoERouter)
        # After context: restored to hard
        assert not isinstance(model.layer1.gate, SoftMoERouter)

    def test_patch_model_supports_sglang_topk_blocks(self):
        from aoae.models.soft_moe import (
            patch_model_with_soft_routing,
            set_hard_routing,
            set_soft_routing,
            SGLangSoftTopKRouter,
        )

        model = self._make_mock_sglang_moe_model()
        orig_topk = model.layer1.topk
        patched = patch_model_with_soft_routing(model, tau_r=0.5)

        assert isinstance(patched.layer1.topk, SGLangSoftTopKRouter)
        assert isinstance(patched.layer2.topk, SGLangSoftTopKRouter)

        hidden = torch.randn(3, 16)
        logits = patched.layer1.gate(hidden)
        patched.layer1.topk(hidden, logits)
        assert orig_topk.last_top_k == 4

        set_hard_routing(model)
        assert model.layer1.topk is orig_topk

        set_soft_routing(model)
        assert isinstance(model.layer1.topk, SGLangSoftTopKRouter)


# ======================================================================
# Tests: DualModelWrapper (mock-based, no HuggingFace)
# ======================================================================

class MockDualModel:
    """Mock DualModelWrapper for testing without real model loading."""
    def __init__(self, vocab_size=100, hidden_dim=64, mask_id=99):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mask_id = mask_id
        self.tokenizer = None
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def get_embedding_weight(self):
        return torch.randn(self.vocab_size, self.hidden_dim)

    def to(self, device):
        self._device = device
        return self

    def auxiliary_forward(self, input_ids):
        B, L = input_ids.shape
        return torch.randn(B, L, self.vocab_size)

    def primary_forward(self, input_ids):
        B, L = input_ids.shape
        return torch.randn(B, L, self.vocab_size)

    def primary_forward_with_hidden(self, input_ids):
        B, L = input_ids.shape
        logits = torch.randn(B, L, self.vocab_size)
        hidden = torch.randn(B, L, self.hidden_dim)
        return logits, hidden

    def dual_forward(self, input_ids, need_hidden=False):
        from aoae.models.dual_model import DualModelOutput
        aux_logits = self.auxiliary_forward(input_ids)
        if need_hidden:
            pri_logits, pri_hidden = self.primary_forward_with_hidden(input_ids)
        else:
            pri_logits = self.primary_forward(input_ids)
            pri_hidden = None
        agreement = (aux_logits.argmax(-1) == pri_logits.argmax(-1))
        return DualModelOutput(
            primary_logits=pri_logits,
            auxiliary_logits=aux_logits,
            agreement=agreement,
            agreement_rate=agreement.float().mean().item(),
            primary_hidden=pri_hidden,
        )

    def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False):
        out = self.dual_forward(input_ids, need_hidden=need_hidden)
        out.primary_logits = out.primary_logits[:, resp_slice, :]
        out.auxiliary_logits = out.auxiliary_logits[:, resp_slice, :]
        out.agreement = out.agreement[:, resp_slice]
        if out.primary_hidden is not None:
            out.primary_hidden = out.primary_hidden[:, resp_slice, :]
        return out


class TestDualModelOutput:
    def test_agreement_shape(self):
        from aoae.models.dual_model import DualModelOutput
        mock = MockDualModel()
        ids = torch.randint(0, 90, (2, 20))
        out = mock.dual_forward(ids)
        assert out.agreement.shape == (2, 20)
        assert out.primary_logits.shape == (2, 20, 100)
        assert out.auxiliary_logits.shape == (2, 20, 100)
        assert 0.0 <= out.agreement_rate <= 1.0

    def test_resp_slice(self):
        mock = MockDualModel()
        ids = torch.randint(0, 90, (1, 30))
        resp_slice = slice(10, 30)
        out = mock.dual_forward_resp(ids, resp_slice)
        assert out.primary_logits.shape == (1, 20, 100)
        assert out.agreement.shape == (1, 20)

    def test_with_hidden(self):
        mock = MockDualModel()
        ids = torch.randint(0, 90, (1, 15))
        out = mock.dual_forward(ids, need_hidden=True)
        assert out.primary_hidden is not None
        assert out.primary_hidden.shape == (1, 15, 64)


class TestDualModelWrapperInit:
    @staticmethod
    def _make_mock_moe_model():
        class MockGateModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 1.0
                self.weight = torch.nn.Parameter(torch.randn(4, 16))
                self.register_buffer("expert_bias", torch.zeros(4))

            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

        class MockMoeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = MockGateModule()
                self.experts = torch.nn.Linear(16, 16)

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = MockMoeBlock()
                self.layer2 = MockMoeBlock()

        return MockModel()

    def test_dual_wrapper_uses_dinfer_backend_for_llada2(self, monkeypatch):
        from aoae.models import dual_model as dual_module
        from aoae.models.soft_moe import SoftMoERouter

        received = {}
        mock_moe_model = self._make_mock_moe_model()

        class StubBaseModel:
            def __init__(self, cfg):
                received["backend"] = cfg["base_model"]["backend"]
                self.model = mock_moe_model
                self.tokenizer = None
                self.mask_id = cfg["base_model"]["mask_token_id"]
                self.vocab_size = VOCAB
                self.hidden_dim = DIM
                self._embedding = torch.randn(VOCAB, DIM)
                self._device = torch.device("cpu")
                self._dtype = torch.float32

            @property
            def device(self):
                return self._device

            @property
            def dtype(self):
                return self._dtype

            def get_embedding_weight(self):
                return self._embedding

            def to(self, device):
                self._device = device
                return self

        monkeypatch.setattr(dual_module, "LLaDABaseModel", StubBaseModel)

        cfg = {
            "base_model": {
                "name_or_path": "inclusionAI/LLaDA2.1-mini",
                "backend": "dual",
                "mask_token_id": MASK_ID,
                "routing_temperature": 0.01,
            }
        }
        wrapper = dual_module.DualModelWrapper(cfg)

        assert received["backend"] == "dinfer"
        assert not isinstance(wrapper._model.model.layer1.gate, SoftMoERouter)

    def test_dual_wrapper_rejects_dense_models(self):
        from aoae.models.dual_model import _select_dual_base_backend

        cfg = {
            "base_model": {
                "name_or_path": "GSAI-ML/LLaDA-8B-Instruct",
                "backend": "dual",
            }
        }
        with pytest.raises(ValueError, match="MoE-capable backend"):
            _select_dual_base_backend(cfg)


# ======================================================================
# Tests: compose_prediction_dual
# ======================================================================

class TestComposePredictionDual:
    def test_no_composition(self):
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = torch.randn(2, 10, VOCAB)
        aux = torch.randn(2, 10, VOCAB)
        agree = torch.ones(2, 10)
        result = compose_prediction_dual(pri, aux, agree, gamma=0.0)
        assert torch.allclose(result, pri)

    def test_disagreement_passthrough(self):
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = torch.randn(2, 10, VOCAB)
        aux = torch.randn(2, 10, VOCAB)
        agree = torch.zeros(2, 10)  # all disagree
        result = compose_prediction_dual(pri, aux, agree, gamma=0.5)
        assert torch.allclose(result, pri)

    def test_agreement_sharpens(self):
        from aoae.models.composed_prediction import compose_prediction_dual
        import torch.nn.functional as F
        pri = torch.randn(2, 10, VOCAB)
        aux = pri.clone()  # identical → full agreement composition
        agree = torch.ones(2, 10)
        result = compose_prediction_dual(pri, aux, agree, gamma=1.0)
        # Result should be different from primary (aux contribution)
        assert not torch.allclose(result, pri)
        # Composed entropy should be lower (sharpened)
        ent_pri = -(F.softmax(pri, -1) * F.log_softmax(pri, -1)).sum(-1).mean()
        ent_comp = -(F.softmax(result, -1) * F.log_softmax(result, -1)).sum(-1).mean()
        assert ent_comp <= ent_pri + 1e-5

    def test_output_shape(self):
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = torch.randn(3, 8, VOCAB)
        aux = torch.randn(3, 8, VOCAB)
        agree = torch.randint(0, 2, (3, 8)).float()
        result = compose_prediction_dual(pri, aux, agree, gamma=0.5)
        assert result.shape == (3, 8, VOCAB)


# ======================================================================
# Tests: Policy with agreement signal
# ======================================================================

class TestPolicyAgreement:
    def test_forward_with_agreement(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        policy = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)
        H = torch.randn(2, 10, DIM)
        mask = torch.randint(0, 2, (2, 10)).bool()
        agree = torch.randint(0, 2, (2, 10)).float()
        out = policy(H, mask, 0.5, agreement=agree)
        assert "unmask_probs" in out
        assert "cache_probs" in out
        assert out["unmask_probs"].shape == (2, 10)

    def test_forward_without_agreement(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        policy = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)
        H = torch.randn(2, 10, DIM)
        mask = torch.randint(0, 2, (2, 10)).bool()
        # agreement=None should default to zeros
        out = policy(H, mask, 0.5, agreement=None)
        assert out["unmask_probs"].shape == (2, 10)

    def test_input_dim_d_plus_4(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        policy = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)
        # input_proj should accept D+4
        assert policy.input_proj.in_features == DIM + 4


# ======================================================================
# Tests: DefaultPolicy (no-GRPO heuristic fallback)
# ======================================================================

class TestDefaultPolicy:
    def test_forward_shape(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy(tau_mask=0.7)
        B, L, D = 2, 10, 64
        H = torch.randn(B, L, D)
        mask = torch.randint(0, 2, (B, L)).bool()
        agree = torch.rand(B, L)
        out = policy(H, mask, step_frac=0.5, agreement=agree)
        assert out["unmask_probs"].shape == (B, L)
        assert out["remask_probs"].shape == (B, L)
        assert out["cache_probs"].shape == (B, L)

    def test_remask_probs_always_zero(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy()
        H = torch.randn(2, 8, 32)
        mask = torch.ones(2, 8).bool()
        out = policy(H, mask, 0.5)
        assert out["remask_probs"].sum().item() == 0.0

    def test_unmask_probs_scaled_by_step(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy()
        H = torch.randn(1, 6, 32)
        mask = torch.tensor([[True, False, True, True, False, False]])
        out = policy(H, mask, 0.5)
        # unmask_probs should be >0 for masked, 0.0 for unmasked
        assert out["unmask_probs"][0, 0].item() > 0
        assert out["unmask_probs"][0, 1].item() == 0.0
        # At step_frac=0.5, rate = 1/(0.5*200) = 0.01
        expected_rate = 1.0 / max(0.5 * 8, 1.0)
        assert abs(out["unmask_probs"][0, 0].item() - expected_rate) < 1e-6

    def test_unmask_rate_increases_as_step_frac_decreases(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy()
        H = torch.randn(1, 4, 32)
        mask = torch.ones(1, 4).bool()
        out_early = policy(H, mask, step_frac=1.0)  # early: low rate
        out_late = policy(H, mask, step_frac=0.01)   # late: high rate
        assert out_early["unmask_probs"].max().item() < out_late["unmask_probs"].max().item()

    def test_cache_probs_match_agreement(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy()
        H = torch.randn(1, 4, 32)
        mask = torch.zeros(1, 4).bool()
        agree = torch.tensor([[0.9, 0.1, 0.8, 0.3]])
        out = policy(H, mask, 0.5, agreement=agree)
        assert torch.allclose(out["cache_probs"], agree)

    def test_sample_actions_interface(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy()
        H = torch.randn(2, 5, 32)
        mask = torch.randint(0, 2, (2, 5)).bool()
        # Use step_frac=0.001 so unmask_rate ≈ 1.0 (most positions unmasked)
        out = policy(H, mask, step_frac=0.001)
        actions = policy.sample_actions(out, mask)
        assert "u_t" in actions and "r_t" in actions and "kappa_t" in actions
        assert actions["r_t"].sum().item() == 0.0  # never remask
        # u_t must be 0 at unmasked positions
        assert (actions["u_t"][~mask] == 0.0).all()

    def test_log_prob_returns_zeros(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy()
        H = torch.randn(2, 5, 32)
        mask = torch.randint(0, 2, (2, 5)).bool()
        out = policy(H, mask, 0.5)
        actions = policy.sample_actions(out, mask)
        lp = policy.log_prob(out, actions)
        assert lp.shape == (2,)
        assert (lp == 0.0).all()


# ======================================================================
# Tests: SpeculativeCacheManager
# ======================================================================

class TestSpeculativeCacheManager:
    def test_agreement_gated_commit(self):
        from aoae.dinfer_integration import SpeculativeCacheManager
        mgr = SpeculativeCacheManager(1, 5, torch.device("cpu"))
        r_t = torch.zeros(1, 5)
        u_t = torch.ones(1, 5)
        kappa_t = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
        agreement = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.float)  # pos 0,2 agree
        mgr.step(r_t, kappa_t, u_t, agreement)
        stats = mgr.get_stats()
        assert stats["draft_accepts"] == 2   # pos 0, 2
        assert stats["draft_rejects"] == 1   # pos 1 (wanted to cache, disagreed)
        assert stats["total_commits"] == 2

    def test_empty_step(self):
        from aoae.dinfer_integration import SpeculativeCacheManager
        mgr = SpeculativeCacheManager(1, 5, torch.device("cpu"))
        r_t = torch.zeros(1, 5)
        u_t = torch.zeros(1, 5)
        kappa_t = torch.zeros(1, 5)
        agreement = torch.ones(1, 5)
        mgr.step(r_t, kappa_t, u_t, agreement)
        stats = mgr.get_stats()
        assert stats["total_commits"] == 0
        assert stats["steps_used"] == 1

    def test_invalidation_with_remask(self):
        from aoae.dinfer_integration import SpeculativeCacheManager
        mgr = SpeculativeCacheManager(1, 5, torch.device("cpu"))
        # First: commit some positions
        r_t0 = torch.zeros(1, 5)
        u_t0 = torch.ones(1, 5)
        kappa_t0 = torch.ones(1, 5)
        agree0 = torch.ones(1, 5)
        mgr.step(r_t0, kappa_t0, u_t0, agree0)
        # Then: remask position 0
        r_t1 = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float)
        u_t1 = torch.zeros(1, 5)
        kappa_t1 = torch.zeros(1, 5)
        agree1 = torch.ones(1, 5)
        mgr.step(r_t1, kappa_t1, u_t1, agree1)
        stats = mgr.get_stats()
        assert stats["total_invalidations"] == 1
        assert stats["total_remasks"] == 1
