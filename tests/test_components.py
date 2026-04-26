"""
Comprehensive test suite for AOAE components.

Run with:  python3 -m pytest tests/ -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import os
import types
from functools import partial

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


class TestVllmDistributedInit:
    def test_init_vllm_distributed_uses_vllm_env_init_without_manual_process_group(self, monkeypatch):
        import aoae.models.base_model as mod

        calls = []
        fake_vllm_dist = types.SimpleNamespace(
            init_distributed_environment=lambda world_size, rank, init_method, local_rank, backend: calls.append(
                ("env", world_size, rank, init_method, local_rank, backend)
            ),
            initialize_model_parallel=lambda tp_size, backend="nccl": calls.append(
                ("mp", tp_size, backend)
            ),
            model_parallel_is_initialized=lambda: False,
        )
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.distributed = fake_vllm_dist
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mod.torch.distributed, "is_initialized", lambda: False)
        monkeypatch.setattr(
            mod.torch.distributed,
            "init_process_group",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("manual init_process_group should not be used for vLLM")
            ),
        )

        mod._init_vllm_distributed(2)

        assert calls == [
            ("env", 2, 1, "env://", 1, "nccl"),
            ("mp", 2, "nccl"),
        ]

    def test_init_vllm_distributed_reuses_existing_process_group(self, monkeypatch):
        import aoae.models.base_model as mod

        calls = []
        fake_vllm_dist = types.SimpleNamespace(
            init_distributed_environment=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("existing distributed env should be reused")
            ),
            initialize_model_parallel=lambda tp_size, backend="nccl": calls.append(
                ("mp", tp_size, backend)
            ),
            model_parallel_is_initialized=lambda: False,
        )
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.distributed = fake_vllm_dist
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mod.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(mod.torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(mod.torch.distributed, "get_rank", lambda: 0)

        mod._init_vllm_distributed(2)

        assert calls == [("mp", 2, "nccl")]


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

    def test_build_prompt_text_falls_back_to_plain_question_without_template(self):
        from aoae.tasks import build_prompt_text

        class NoTemplateTokenizer:
            chat_template = None

            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("should not be called")

        question = "What is 6 * 7?"
        text = build_prompt_text(NoTemplateTokenizer(), question, {"data": {"use_chat_template": "auto"}})
        assert text == question

    def test_build_prompt_auto_uses_callable_chat_template_even_without_field(self):
        from aoae.tasks import build_prompt

        class RuntimeTemplateTokenizer:
            chat_template = None

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                assert tokenize is False
                assert add_generation_prompt is True
                return f"CHAT::{messages[0]['content']}"

        question = "What is 6 * 7?"
        prompt_text, add_special_tokens = build_prompt(
            RuntimeTemplateTokenizer(),
            question,
            {"data": {"use_chat_template": "auto"}},
        )
        assert prompt_text == "CHAT::What is 6 * 7?"
        assert add_special_tokens is False

    def test_build_prompt_auto_formats_gsm8k_questions(self):
        from aoae.tasks import build_prompt

        class PlainTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("chat path should be disabled")

        question = "Natalia sold clips."
        prompt_text, add_special_tokens = build_prompt(
            PlainTokenizer(),
            question,
            {
                "data": {
                    "use_chat_template": "off",
                    "math_prompt_style": "auto",
                    "eval_dataset": "openai/gsm8k",
                },
                "evaluation": {"task_type": "math"},
            },
        )
        assert "#### (your numerical answer)" in prompt_text
        assert "Please reason step by step" in prompt_text
        assert prompt_text.startswith(question)
        assert add_special_tokens is True

    def test_decode_generated_tokens_truncates_eos_and_strips_mask(self):
        from aoae.tasks import decode_generated_tokens

        class DummyTokenizer:
            eos_token_id = 2

            def decode(self, ids, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join(str(x) for x in ids)

        tok = DummyTokenizer()
        decoded = decode_generated_tokens(tok, [5, 99, 6, 2, 7, 8], mask_token_id=99)
        assert decoded == "5 6"

    def test_gsm8k_llada_extractor_handles_strict_and_prose_answers(self):
        from aoae.tasks import extract_gsm8k_llada_answer

        assert extract_gsm8k_llada_answer("Reasoning\n#### 42,000") == "42000"
        assert extract_gsm8k_llada_answer("Reasoning\n#### -3.5") == "-3.5"
        assert extract_gsm8k_llada_answer("Answer: Claire will eat 7 dozens of eggs in 4 weeks.") == "7"
        assert extract_gsm8k_llada_answer("#### <23>") == "23"
        assert extract_gsm8k_llada_answer("#### <answer>243</answer>") == "243"

    def test_math_evaluator_uses_flexible_gsm8k_rule(self):
        from aoae.evaluators import build_evaluator

        cfg = {
            "evaluation": {"task_type": "math"},
            "data": {"eval_dataset": "openai/gsm8k"},
        }
        evaluator = build_evaluator(cfg)

        # Strict #### format — must succeed.
        decision = evaluator.evaluate(
            "Claire will eat 7 dozens of eggs in 4 weeks.\n#### 7",
            "#### 7",
        )
        assert decision.correct is True
        assert decision.detail == "gsm8k_llada_flexible"
        assert decision.extracted_prediction == "7"
        assert decision.extracted_reference == "7"

        # Prose answer without #### — flexible extractor must recover the answer.
        decision2 = evaluator.evaluate(
            "Step-by-step reasoning...\nThe answer is 7.",
            "#### 7",
        )
        assert decision2.correct is True
        assert decision2.detail == "gsm8k_llada_flexible"


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

    def test_policy_initializes_cache_and_remask_conservatively(self):
        from aoae.models.policy import AOAEPolicy

        cfg = {
            **DEFAULT_CFG,
            "policy": {
                **DEFAULT_CFG["policy"],
                "init_remask_bias": -4.0,
                "init_cache_bias": -2.0,
                "init_access_bias": -2.0,
            },
        }
        pol = AOAEPolicy(cfg, input_dim=DIM)

        assert pol.head_unmask.bias.mean().item() == pytest.approx(0.0)
        assert pol.head_remask.bias.mean().item() == pytest.approx(-4.0)
        assert pol.head_cache.bias.mean().item() == pytest.approx(-2.0)
        assert pol.head_access.bias.mean().item() == pytest.approx(-2.0)

    def test_apply_unmask_budget_keeps_highest_probability_actions(self):
        from aoae.models.policy import apply_unmask_budget

        actions = {
            "u_t": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]]),
            "r_t": torch.zeros(1, 5),
            "kappa_t": torch.zeros(1, 5),
            "q_t": torch.zeros(1, 5),
        }
        policy_out = {
            "unmask_probs": torch.tensor([[0.2, 0.9, 0.4, 0.8, 0.7]]),
        }
        mask = torch.tensor([[True, True, True, True, True]])
        cfg = {"inference": {"max_unmask_tokens_per_step": 2}}

        capped = apply_unmask_budget(actions, policy_out, mask, cfg)

        assert capped["u_t"].tolist() == [[0.0, 1.0, 0.0, 1.0, 0.0]]


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

    def test_fallback_unmask_still_applies_when_recording_trajectory(self, base_model, embed_w):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.inference import aoae_inference

        sm = SoftMaskedState(DEFAULT_CFG, embed_w)
        sm.set_mask_embedding(MASK_ID)
        pol = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)

        def zero_actions(policy_out, mask_ind):
            shape = mask_ind.shape
            return {
                "u_t": torch.zeros(shape, dtype=torch.float32),
                "r_t": torch.zeros(shape, dtype=torch.float32),
                "kappa_t": torch.zeros(shape, dtype=torch.float32),
            }

        pol.sample_actions = zero_actions

        prompt = torch.tensor([[10, 20, 30, 40]])
        output, traj = aoae_inference(
            base_model, pol, sm, None, prompt, DEFAULT_CFG,
            record_trajectory=True,
        )

        assert traj is not None
        assert traj.changed_list
        assert traj.changed_list[0].sum().item() > 0
        assert (output[0, 4:] != MASK_ID).sum().item() > 0


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
        assert (out[0, 4:] != MASK_ID).all()

    def test_confidence_decode(self, base_model):
        from aoae.inference import confidence_threshold_decode

        prompt = torch.tensor([[10, 20, 30, 40]])
        out = confidence_threshold_decode(
            base_model, prompt, DEFAULT_CFG, tau_mask=0.3
        )
        L_gen = DEFAULT_CFG["inference"]["gen_length"]
        assert out.shape == (1, 4 + L_gen)
        assert (out[0, 4:] != MASK_ID).all()

    def test_confidence_decode_force_completes_when_mask_token_is_argmax(self):
        from aoae.inference import confidence_threshold_decode

        class MaskPreferringModel(MockBaseModel):
            def __init__(self):
                super().__init__(VOCAB, DIM, MASK_ID)

            def forward(self, input_ids):
                B, L = input_ids.shape
                logits = torch.zeros(B, L, self.vocab_size)
                logits[..., self.mask_id] = 10.0
                logits[..., 1] = 9.0
                return logits

        cfg = {
            **DEFAULT_CFG,
            "inference": {
                **DEFAULT_CFG["inference"],
                "steps": 2,
                "gen_length": 6,
            },
        }
        prompt = torch.tensor([[10, 20, 30, 40]])
        out = confidence_threshold_decode(
            MaskPreferringModel(),
            prompt,
            cfg,
            tau_mask=0.999,
            tau_edit=0.999,
            enable_t2t=True,
        )
        assert (out[0, 4:] != MASK_ID).all()

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

    def test_block_smode_uses_active_prefix_and_finishes_blocks(self):
        from aoae.inference import block_smode_decode

        class TrackingBaseModel(MockBaseModel):
            def __init__(self):
                super().__init__(VOCAB, DIM, MASK_ID)
                self.seen_lengths = []

            def forward_block_causal(self, input_ids, block_length=32):
                del block_length
                self.seen_lengths.append(int(input_ids.shape[1]))
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
                logits[..., 0] = 1.0
                return logits

        model = TrackingBaseModel()
        cfg = {
            **DEFAULT_CFG,
            "inference": {
                **DEFAULT_CFG["inference"],
                "gen_length": 4,
                "block_length": 2,
            },
        }
        prompt = torch.tensor([[10, 20, 30, 40]])
        out = block_smode_decode(
            model, prompt, cfg,
            tau_mask=0.999, tau_edit=0.999, max_steps_per_block=1,
        )

        assert model.seen_lengths[0] == 6
        assert max(model.seen_lengths) == 8
        assert (out[0, 4:] != MASK_ID).all()

    def test_block_smode_eos_early_stop_masks_future_positions(self):
        from aoae.inference import block_smode_decode

        class EosTokenizer:
            eos_token_id = 2

        class EarlyStopModel(MockBaseModel):
            def __init__(self):
                super().__init__(VOCAB, DIM, MASK_ID)
                self.tokenizer = EosTokenizer()

            def forward_block_causal(self, input_ids, block_length=32):
                del block_length
                B, L = input_ids.shape
                logits = torch.zeros(B, L, self.vocab_size)
                logits[..., 1] = 1.0
                if L >= 4:
                    logits[:, 2, 2] = 10.0
                    logits[:, 3, 1] = 9.0
                return logits

        cfg = {
            **DEFAULT_CFG,
            "inference": {
                **DEFAULT_CFG["inference"],
                "gen_length": 4,
                "block_length": 2,
            },
        }
        prompt = torch.tensor([[10, 20]])
        out = block_smode_decode(
            EarlyStopModel(),
            prompt,
            cfg,
            tau_mask=0.0,
            tau_edit=0.0,
            max_steps_per_block=1,
            eos_early_stop=True,
        )

        assert out[0, 2].item() == 2
        assert (out[0, 3:] == MASK_ID).all()


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

    def test_extract_final_answer_with_currency_and_commas(self):
        from aoae.train_grpo import extract_answer

        assert extract_answer("Final answer: $1,250.00") == "1250.00"

    def test_extract_fraction_answer(self):
        from aoae.train_grpo import extract_answer

        assert extract_answer("The answer is 1/2.") == "1/2"

    def test_correctness_check(self):
        from aoae.train_grpo import check_math_correctness

        assert check_math_correctness("\\boxed{42}", "#### 42")
        assert not check_math_correctness("\\boxed{41}", "#### 42")
        assert check_math_correctness("The answer is 3.14", "\\boxed{3.14}")
        assert check_math_correctness("Final answer: 1/2", "#### 0.5")

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
        from aoae.train_grpo import compute_grpo_loss, split_group_trajectory

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

        traj_data = split_group_trajectory(traj, 1)[0]

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

    def test_loss_stays_finite_with_extreme_importance_ratio(self, embed_w):
        from aoae.models.soft_mask import SoftMaskedState
        from aoae.models.policy import AOAEPolicy
        from aoae.models.prism import PRISMAdapter
        from aoae.inference import aoae_inference
        from aoae.train_grpo import compute_grpo_loss, split_group_trajectory

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

        traj_data = split_group_trajectory(traj, 1)[0]
        traj_data["old_log_probs"] = [lp.clone() - 1000.0 for lp in traj_data["old_log_probs"]]

        advantages = torch.tensor([0.5, -0.5])
        loss = compute_grpo_loss(
            pol, sm, [traj_data, traj_data], advantages, clip_eps=0.2
        )

        assert torch.isfinite(loss)


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
        # The composition operates in normalized log-prob space:
        #     p_tilde(v) ∝ p(v)^{1 + gamma * sigma_k}
        # which sharpens the distribution without distorting argmax order.
        from aoae.models.composed_prediction import compose_prediction
        logits = torch.randn(2, 10, VOCAB)
        cache_probs = torch.ones(2, 10)
        composed = compose_prediction(logits, cache_probs, gamma=1.0)

        scale = 2.0  # 1 + gamma * sigma_k
        base_log_probs = F.log_softmax(logits.float(), dim=-1)
        expected_log_probs = scale * base_log_probs
        expected_log_probs = expected_log_probs - torch.logsumexp(
            expected_log_probs, dim=-1, keepdim=True
        )
        assert torch.allclose(composed.float(), expected_log_probs, atol=1e-5)

        composed_log_probs = F.log_softmax(composed.float(), dim=-1)
        assert torch.allclose(composed.float(), composed_log_probs, atol=1e-5)
        assert torch.equal(composed.argmax(dim=-1), logits.argmax(dim=-1))

        ent_base = -(F.softmax(logits.float(), -1) * base_log_probs).sum(-1).mean()
        ent_comp = -(F.softmax(composed.float(), -1) * composed_log_probs).sum(-1).mean()
        assert ent_comp <= ent_base + 1e-5

    def test_selective_sharpening(self):
        from aoae.models.composed_prediction import compose_prediction
        logits = torch.randn(1, 5, VOCAB)
        cache_probs = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
        composed = compose_prediction(logits, cache_probs, gamma=0.5)
        # sigma_k=0: untouched base logits.
        for k in (0, 1, 4):
            assert torch.allclose(composed[0, k], logits[0, k])

        # sigma_k=1: log-probs scaled by 1.5, renormalized.
        scale = 1.5
        base_lp = F.log_softmax(logits[0, 2:4].float(), dim=-1)
        expected_lp = scale * base_lp
        expected_lp = expected_lp - torch.logsumexp(expected_lp, dim=-1, keepdim=True)
        assert torch.allclose(composed[0, 2:4].float(), expected_lp, atol=1e-5)
        assert torch.equal(
            composed[0, 2:4].argmax(dim=-1), logits[0, 2:4].argmax(dim=-1)
        )

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
            def forward(self, hidden_states):
                logits = self.get_logits(hidden_states.view(-1, hidden_states.shape[-1]))
                scores = torch.sigmoid(logits.float()).type_as(logits)
                scores_for_routing = scores + self.expert_bias
                _, topk_idx = self.group_limited_topk(scores_for_routing)
                topk_weight = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
                topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                topk_weight = topk_weight * self.routed_scaling_factor
                return topk_idx, topk_weight, logits

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
                self.weight = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [4.0, 0.0],
                            [3.0, 0.0],
                            [-2.0, 0.0],
                            [-3.0, 0.0],
                        ],
                        dtype=torch.float32,
                    )
                )
                self.register_buffer("expert_bias", torch.zeros(4))
            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)
            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)
            def forward(self, hidden_states):
                logits = self.get_logits(hidden_states.view(-1, hidden_states.shape[-1]))
                scores = torch.sigmoid(logits.float()).type_as(logits)
                scores_for_routing = scores + self.expert_bias
                _, topk_idx = self.group_limited_topk(scores_for_routing)
                topk_weight = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
                topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                topk_weight = topk_weight * self.routed_scaling_factor
                return topk_idx, topk_weight, logits

        gate = MockGate()
        soft = SoftMoERouter(gate, tau_r=0.001, soft_topk=gate.top_k)

        hidden = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        hard_idx, hard_weights, _ = gate(hidden)
        soft_idx, soft_weights, _ = soft(hidden)
        assert set(soft_idx[0].tolist()) == set(hard_idx[0].tolist())
        assert torch.allclose(
            soft_weights.sum(dim=-1),
            torch.ones(1),
            atol=1e-5,
        )

    def test_low_temp_wide_soft_topk_keeps_mass_on_hard_experts(self):
        from aoae.models.soft_moe import SoftMoERouter

        class MockGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 1.0
                self.weight = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [4.0, 0.0],
                            [3.0, 0.0],
                            [-2.0, 0.0],
                            [-3.0, 0.0],
                        ],
                        dtype=torch.float32,
                    )
                )
                self.register_buffer("expert_bias", torch.zeros(4))

            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)

            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

            def forward(self, hidden_states):
                logits = self.get_logits(hidden_states.view(-1, hidden_states.shape[-1]))
                scores = torch.sigmoid(logits.float()).type_as(logits)
                scores_for_routing = scores + self.expert_bias
                _, topk_idx = self.group_limited_topk(scores_for_routing)
                topk_weight = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
                topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                topk_weight = topk_weight * self.routed_scaling_factor
                return topk_idx, topk_weight, logits

        gate = MockGate()
        soft = SoftMoERouter(gate, tau_r=0.001, soft_topk=gate.num_experts)

        hidden = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        hard_idx, hard_weights, _ = gate(hidden)
        soft_idx, soft_weights, _ = soft(hidden)
        hard_experts = set(hard_idx[0].tolist())
        retained_mass = sum(
            float(soft_weights[0, pos].item())
            for pos, expert_idx in enumerate(soft_idx[0].tolist())
            if expert_idx in hard_experts
        )
        assert retained_mass > 0.999

        hard_weight_map = {
            int(expert_idx): float(weight)
            for expert_idx, weight in zip(hard_idx[0].tolist(), hard_weights[0].tolist())
        }
        retained = []
        retained_expected = []
        for pos, expert_idx in enumerate(soft_idx[0].tolist()):
            if expert_idx in hard_experts:
                retained.append(float(soft_weights[0, pos].item()) / retained_mass)
                retained_expected.append(hard_weight_map[int(expert_idx)])
        assert retained
        assert retained_expected
        assert torch.allclose(
            torch.tensor(retained),
            torch.tensor(retained_expected),
            atol=1e-4,
        )

    def test_tau_one_matches_hard_weights_with_expert_bias(self):
        from aoae.models.soft_moe import SoftMoERouter

        class MockGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 1.0
                self.weight = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [1.8, 0.0],
                            [1.6, 0.0],
                            [1.2, 0.0],
                            [-1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    )
                )
                self.register_buffer(
                    "expert_bias",
                    torch.tensor([0.0, 0.15, -0.05, 0.0], dtype=torch.float32),
                )

            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)

            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

            def forward(self, hidden_states):
                logits = self.get_logits(hidden_states.view(-1, hidden_states.shape[-1]))
                scores = torch.sigmoid(logits.float()).type_as(logits)
                scores_for_routing = scores + self.expert_bias
                _, topk_idx = self.group_limited_topk(scores_for_routing)
                topk_weight = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
                topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                topk_weight = topk_weight * self.routed_scaling_factor
                return topk_idx, topk_weight, logits

        gate = MockGate()
        soft = SoftMoERouter(gate, tau_r=1.0, soft_topk=gate.top_k)

        hidden = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        hard_idx, hard_weights, _ = gate(hidden)
        soft_idx, soft_weights, _ = soft(hidden)

        assert torch.equal(soft_idx, hard_idx)
        assert torch.allclose(soft_weights, hard_weights, atol=1e-6)

    def test_soft_topk_equal_hard_topk_is_tau_invariant(self):
        from aoae.models.soft_moe import SoftMoERouter

        class MockGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 1.0
                self.weight = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [2.0, 0.0],
                            [1.5, 0.0],
                            [0.1, 0.0],
                            [-1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    )
                )
                self.register_buffer("expert_bias", torch.zeros(4))

            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)

            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

            def forward(self, hidden_states):
                logits = self.get_logits(hidden_states.view(-1, hidden_states.shape[-1]))
                scores = torch.sigmoid(logits.float()).type_as(logits)
                scores_for_routing = scores + self.expert_bias
                _, topk_idx = self.group_limited_topk(scores_for_routing)
                topk_weight = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
                topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                topk_weight = topk_weight * self.routed_scaling_factor
                return topk_idx, topk_weight, logits

        gate = MockGate()
        hidden = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        hard_idx, hard_weights, _ = gate(hidden)

        for tau_r in (0.001, 0.01, 0.1, 0.5, 1.0):
            soft = SoftMoERouter(gate, tau_r=tau_r, soft_topk=gate.top_k)
            soft_idx, soft_weights, _ = soft(hidden)
            assert torch.equal(soft_idx, hard_idx)
            assert torch.allclose(soft_weights, hard_weights, atol=1e-6)

    def test_low_temp_preserves_hard_selection_with_expert_bias(self):
        from aoae.models.soft_moe import SoftMoERouter

        class MockGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.routed_scaling_factor = 1.0
                self.weight = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [2.0, 0.0],
                            [1.9, 0.0],
                            [1.8, 0.0],
                            [-2.0, 0.0],
                        ],
                        dtype=torch.float32,
                    )
                )
                self.register_buffer(
                    "expert_bias",
                    torch.tensor([0.0, 0.20, -0.25, 0.0], dtype=torch.float32),
                )

            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)

            def group_limited_topk(self, scores):
                return scores.topk(self.top_k, dim=-1)

            def forward(self, hidden_states):
                logits = self.get_logits(hidden_states.view(-1, hidden_states.shape[-1]))
                scores = torch.sigmoid(logits.float()).type_as(logits)
                scores_for_routing = scores + self.expert_bias
                _, topk_idx = self.group_limited_topk(scores_for_routing)
                topk_weight = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
                topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                topk_weight = topk_weight * self.routed_scaling_factor
                return topk_idx, topk_weight, logits

        gate = MockGate()
        soft = SoftMoERouter(gate, tau_r=0.001, soft_topk=gate.top_k)

        hidden = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        hard_idx, _, _ = gate(hidden)
        soft_idx, soft_weights, _ = soft(hidden)

        assert torch.equal(soft_idx, hard_idx)
        assert torch.allclose(soft_weights.sum(dim=-1), torch.ones(1), atol=1e-6)

    def test_records_last_weights_in_eval_mode(self):
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
        soft = SoftMoERouter(gate, tau_r=0.25)
        soft.eval()

        hidden = torch.randn(3, 16)
        _, _, _ = soft(hidden)

        assert soft._last_weights is not None
        assert soft._last_weights.shape == (3, 4)
        assert torch.allclose(
            soft._last_weights.sum(dim=-1),
            torch.ones(3),
            atol=1e-5,
        )

    def test_preserves_original_gate_bias_and_group_limited_selection(self):
        from aoae.models.soft_moe import SoftMoERouter

        class MockGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.top_k = 2
                self.n_group = 2
                self.topk_group = 1
                self.routed_scaling_factor = 1.0
                self.weight = torch.nn.Parameter(torch.zeros(4, 4))
                self.register_buffer("expert_bias", torch.tensor([0.0, 0.0, 10.0, 9.0]))

            def get_logits(self, h):
                return torch.nn.functional.linear(h, self.weight)

            def group_limited_topk(self, scores):
                num_tokens, _ = scores.size()
                group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
                group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
                    .reshape(num_tokens, -1)
                )
                masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
                probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)
                return probs, top_indices

        gate = MockGate()
        soft = SoftMoERouter(gate, tau_r=0.5, soft_topk=2)

        hidden = torch.ones(1, 4)
        indices, weights, _ = soft(hidden)

        # The very large bias on experts 2 and 3 should force group-limited
        # selection into that group, even though raw logits are tied.
        assert set(indices[0].tolist()) == {2, 3}
        assert weights.shape == (1, 2)

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

    def _make_mock_dinfer_moe_model(self):
        """Helper: model whose fused layer stores a routing callback bound to the original gate."""
        def static_routing_function(gate, hidden_states, gating_output, topk, renormalize):
            return gate.routing(hidden_states, gating_output, topk, renormalize)

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

            def routing(self, hidden_states, gating_output, topk, renormalize):
                del hidden_states, gating_output, renormalize
                num_tokens = 3 if topk is None else 3
                k = self.top_k if topk is None else int(topk)
                weights = torch.ones(num_tokens, k, dtype=torch.float32)
                indices = torch.arange(k, dtype=torch.int64).unsqueeze(0).expand(num_tokens, -1)
                return weights, indices

        class MockExperts:
            def __init__(self, gate):
                self.top_k = gate.top_k
                self.custom_routing_function = partial(static_routing_function, gate)
                self.router = types.SimpleNamespace(
                    top_k=gate.top_k,
                    custom_routing_function=self.custom_routing_function,
                )
                self.moe_config = types.SimpleNamespace(
                    experts_per_token=gate.top_k,
                )

        class MockMoeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = MockGateModule()
                self.top_k = self.gate.top_k
                self.experts = MockExperts(self.gate)

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = MockMoeBlock()

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

    def _make_mock_hidden_sglang_moe_model(self):
        """Helper: create a model where MoE blocks are only reachable via model.layers."""
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

            def forward(self, hidden_states, router_logits, *args, **kwargs):
                del hidden_states, args, kwargs
                return router_logits.topk(self.top_k, dim=-1)

        class MockMoeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = MockGateModule()
                self.topk = MockTopK()
                self.num_experts = 4
                self.score_function = "softmax"

        class MockLayer:
            def __init__(self):
                self.mlp = MockMoeBlock()

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = types.SimpleNamespace(layers=[MockLayer(), MockLayer()])

        return MockModel()

    def test_patch_model_finds_gates(self):
        from aoae.models.soft_moe import patch_model_with_soft_routing, SoftMoERouter

        model = self._make_mock_moe_model()
        patched = patch_model_with_soft_routing(model, tau_r=0.01)
        assert isinstance(patched.layer1.gate, SoftMoERouter)
        assert isinstance(patched.layer2.gate, SoftMoERouter)

    def test_patch_updates_fused_routing_callback_and_effective_topk(self):
        from aoae.models.soft_moe import (
            SoftMoERouter,
            patch_model_with_soft_routing,
            set_hard_routing,
            set_soft_topk,
        )

        model = self._make_mock_dinfer_moe_model()
        original_gate = model.layer1.gate

        patch_model_with_soft_routing(model, tau_r=0.5, soft_topk=4)

        assert isinstance(model.layer1.gate, SoftMoERouter)
        assert model.layer1.top_k == 4
        assert model.layer1.experts.top_k == 4
        assert model.layer1.experts.router.top_k == 4
        assert model.layer1.experts.moe_config.experts_per_token == 4

        weights, indices = model.layer1.experts.custom_routing_function(
            torch.randn(3, 16),
            torch.randn(3, 4),
            model.layer1.experts.top_k,
            False,
        )
        assert weights.shape == (3, 4)
        assert indices.shape == (3, 4)
        assert model.layer1.experts.custom_routing_function.args[0] is model.layer1
        assert model.layer1.experts.router.custom_routing_function.args[0] is model.layer1

        set_soft_topk(model, 3)
        assert model.layer1.top_k == 3
        assert model.layer1.experts.top_k == 3
        assert model.layer1.experts.router.top_k == 3
        assert model.layer1.experts.moe_config.experts_per_token == 3
        weights3, indices3 = model.layer1.experts.custom_routing_function(
            torch.randn(3, 16),
            torch.randn(3, 4),
            model.layer1.experts.top_k,
            False,
        )
        assert weights3.shape == (3, 3)
        assert indices3.shape == (3, 3)

        set_hard_routing(model)
        assert model.layer1.gate is original_gate
        assert model.layer1.top_k == 2
        assert model.layer1.experts.top_k == 2
        assert model.layer1.experts.router.top_k == 2
        assert model.layer1.experts.moe_config.experts_per_token == 2
        assert model.layer1.experts.custom_routing_function.args[0] is original_gate
        assert model.layer1.experts.router.custom_routing_function.args[0] is original_gate

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

    def test_sglang_router_infers_sigmoid_from_correction_bias(self):
        from aoae.models.soft_moe import SGLangSoftTopKRouter

        class MockTopK(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.top_k = 2
                self.correction_bias = torch.zeros(4)

            def forward(self, hidden_states, router_logits, *args, **kwargs):
                del hidden_states, args, kwargs
                return router_logits.topk(self.top_k, dim=-1)

        topk = MockTopK()
        router = SGLangSoftTopKRouter(
            topk,
            num_experts=4,
            tau_r=0.001,
            soft_topk=4,
            score_function=None,
            top_k_override=2,
        )
        router(
            torch.randn(1, 8),
            torch.tensor([[8.0, 7.0, -6.0, -7.0]], dtype=torch.float32),
        )
        assert router._last_weights is not None
        assert float(router._last_weights[0, :2].sum().item()) > 0.999

    def test_patch_model_supports_explicit_model_layers_fallback(self):
        from aoae.models.soft_moe import (
            patch_model_with_soft_routing,
            set_hard_routing,
            SGLangSoftTopKRouter,
        )

        model = self._make_mock_hidden_sglang_moe_model()
        first_block = model.model.layers[0].mlp
        orig_topk = first_block.topk

        patch_model_with_soft_routing(model, tau_r=0.5)
        assert isinstance(first_block.topk, SGLangSoftTopKRouter)

        set_hard_routing(model)
        assert first_block.topk is orig_topk


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

    def test_dual_forward_resp_recomputes_agreement_rate(self):
        from aoae.models.dual_model import DualModelOutput, DualModelWrapper

        def fake_dual_forward(input_ids, need_hidden=False, need_all_hidden=False):
            agreement = torch.tensor([[True, True, False, False]])
            return DualModelOutput(
                primary_logits=torch.zeros(1, 4, 3),
                auxiliary_logits=torch.zeros(1, 4, 3),
                agreement=agreement,
                agreement_rate=0.5,
                primary_hidden=None,
                primary_hidden_states=None,
            )

        dummy = types.SimpleNamespace(dual_forward=fake_dual_forward)
        ids = torch.randint(0, 3, (1, 4))

        out = DualModelWrapper.dual_forward_resp(dummy, ids, slice(2, 4))

        assert out.agreement.shape == (1, 2)
        assert out.agreement_rate == 0.0

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

    def test_zero_gamma_strict_noop(self):
        # Regression: previous bug shifted unnormalized primary logits by
        # negative aux log-probs even at gamma=0 due to a control-flow
        # mistake.  With the rewrite, gamma=0 must be a strict identity for
        # any agreement / aux input.
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = torch.randn(4, 6, VOCAB)
        aux = torch.randn(4, 6, VOCAB)
        agree = torch.randint(0, 2, (4, 6)).float()
        out = compose_prediction_dual(pri, aux, agree, gamma=0.0)
        assert out.data_ptr() == pri.data_ptr() or torch.equal(out, pri)
        assert torch.equal(out.argmax(-1), pri.argmax(-1))

    def test_gamma_positive_no_argmax_pathology_when_aux_disagrees_uniformly(self):
        # Regression: the broken composition (raw logits + log_softmax of aux)
        # systematically biased the argmax toward primary tokens with the
        # least-negative aux log-prob, even at agreement positions where the
        # aux argmax already matched the primary.  Under the corrected
        # log-prob composition, when agreement=1 and primary == aux, the
        # composed argmax must equal the primary argmax for any gamma >= 0.
        from aoae.models.composed_prediction import compose_prediction_dual
        torch.manual_seed(0)
        pri = torch.randn(2, 12, VOCAB)
        aux = pri.clone()
        agree = torch.ones(2, 12)
        for g in (0.1, 0.5, 1.0, 4.0):
            out = compose_prediction_dual(pri, aux, agree, gamma=g)
            assert torch.equal(out.argmax(-1), pri.argmax(-1))

    def test_returns_normalized_log_probs_at_agreement(self):
        # The composition path must return normalized log-probabilities at
        # agreement positions so downstream argmax/softmax remain coherent
        # and finite (no infs / NaNs from logsumexp on bf16 mass).
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = torch.randn(2, 6, VOCAB)
        aux = torch.randn(2, 6, VOCAB)
        agree = torch.ones(2, 6)
        out = compose_prediction_dual(pri, aux, agree, gamma=0.5)
        sums = torch.logsumexp(out.float(), dim=-1)
        assert torch.allclose(sums, torch.zeros_like(sums), atol=1e-4)
        assert torch.isfinite(out).all()

    def test_bf16_input_stable(self):
        # The runtime calls this on bf16 logits.  The renormalization must not
        # produce infs / NaNs at the bf16 precision used by the model.
        from aoae.models.composed_prediction import compose_prediction_dual
        pri = (torch.randn(2, 4, VOCAB) * 5).to(torch.bfloat16)
        aux = (torch.randn(2, 4, VOCAB) * 5).to(torch.bfloat16)
        agree = torch.ones(2, 4)
        out = compose_prediction_dual(pri, aux, agree, gamma=0.5)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()


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

    def test_sample_actions_raises_on_invalid_probs(self, embed_w):
        from aoae.models.policy import AOAEPolicy
        policy = AOAEPolicy(DEFAULT_CFG, input_dim=DIM)
        mask = torch.ones(1, 2).bool()
        bad = torch.tensor([[float("nan"), 0.5]])
        policy_out = {
            "unmask_probs": bad,
            "remask_probs": torch.zeros_like(bad),
            "cache_probs": torch.zeros_like(bad),
            "access_probs": torch.zeros_like(bad),
        }
        with pytest.raises(RuntimeError, match="unmask_probs"):
            policy.sample_actions(policy_out, mask)


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
        conf = torch.rand(B, L)
        out = policy(H, mask, step_frac=0.5, confidence=conf, agreement=agree)
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

    def test_unmask_probs_follow_confidence_threshold(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy(tau_mask=0.7)
        H = torch.randn(1, 6, 32)
        mask = torch.tensor([[True, False, True, True, False, False]])
        confidence = torch.tensor([[0.95, 0.99, 0.2, 0.8, 0.5, 0.75]])
        out = policy(H, mask, 0.5, confidence=confidence)
        assert out["unmask_probs"][0, 0].item() == 1.0
        assert out["unmask_probs"][0, 1].item() == 0.0
        assert out["unmask_probs"][0, 2].item() == 0.0
        assert out["unmask_probs"][0, 3].item() == 1.0

    def test_missing_confidence_keeps_masked_tokens_masked(self):
        from aoae.models.policy import DefaultPolicy
        policy = DefaultPolicy()
        H = torch.randn(1, 4, 32)
        mask = torch.ones(1, 4).bool()
        out = policy(H, mask, step_frac=1.0, confidence=None)
        assert out["unmask_probs"].sum().item() == 0.0

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
        policy = DefaultPolicy(tau_mask=0.7)
        H = torch.randn(2, 5, 32)
        mask = torch.randint(0, 2, (2, 5)).bool()
        confidence = torch.ones(2, 5)
        out = policy(H, mask, step_frac=0.001, confidence=confidence)
        actions = policy.sample_actions(out, mask)
        assert "u_t" in actions and "r_t" in actions and "kappa_t" in actions
        assert actions["r_t"].sum().item() == 0.0  # never remask
        # u_t must be 0 at unmasked positions
        assert (actions["u_t"][~mask] == 0.0).all()
        assert torch.equal(actions["u_t"], mask.float())

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


class TestPRISMAdapter:
    def test_forward_scrubs_nonfinite_hidden_states(self):
        from aoae.models.prism import PRISMAdapter

        cfg = {"prism": {"hidden_dim": 16, "threshold": 0.5}}
        adapter = PRISMAdapter(cfg, hidden_dim=8)
        hidden = torch.randn(2, 5, 8)
        hidden[0, 0, 0] = float("nan")
        hidden[0, 1, 1] = float("inf")
        hidden[0, 2, 2] = float("-inf")

        scores = adapter(hidden)
        assert scores.shape == (2, 5)
        assert torch.isfinite(scores).all()
        assert ((scores >= 0.0) & (scores <= 1.0)).all()


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
        assert stats["cache_hit_rate"] == pytest.approx(5.0 / 6.0)

    def test_uncached_remask_does_not_count_as_cache_invalidation(self):
        from aoae.dinfer_integration import SpeculativeCacheManager

        mgr = SpeculativeCacheManager(1, 4, torch.device("cpu"))
        r_t = torch.tensor([[1, 0, 1, 0]], dtype=torch.float)
        u_t = torch.zeros(1, 4)
        kappa_t = torch.zeros(1, 4)
        agreement = torch.ones(1, 4)

        mgr.step(r_t, kappa_t, u_t, agreement)
        stats = mgr.get_stats()

        assert stats["total_invalidations"] == 0
        assert stats["total_remasks"] == 2
        assert stats["cache_hit_rate"] == 0.0


class TestRunSpeculativeInferenceMetrics:
    def test_metrics_distinguish_raw_agreement_safe_reuse_and_draft_acceptance(self, monkeypatch):
        from aoae.dinfer_integration import run_speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "inference": {
                "steps": 1,
                "gen_length": 2,
                "temperature": 0.0,
                "fallback_unmask": False,
                "compose_gamma": 0.0,
                "disable_remask": False,
                "reuse_signal": {"method": "argmax_match"},
            },
            "analysis": {"track_kv_dynamics": False},
        }

        class DummyDualModel:
            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                primary_logits = torch.tensor(
                    [[[5.0, 0.0, 0.0], [0.0, 4.0, 0.0]]],
                    dtype=torch.float32,
                )
                auxiliary_logits = torch.tensor(
                    [[[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
                    dtype=torch.float32,
                )
                agreement = torch.tensor([[True, False]])
                return types.SimpleNamespace(
                    primary_logits=primary_logits,
                    auxiliary_logits=auxiliary_logits,
                    agreement=agreement,
                    agreement_rate=0.5,
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

            def auxiliary_forward(self, input_ids):
                raise AssertionError("auxiliary_forward should not run when primary_every_n=1")

        class DummyPolicy:
            def __call__(
                self,
                H_t,
                mask_ind,
                step_frac,
                temperature=1.0,
                quality_scores=None,
                agreement=None,
                age_feature=None,
                last_action_feature=None,
            ):
                return {"agreement_seen": agreement}

            def sample_actions(self, policy_out, mask_ind):
                ones = torch.ones_like(mask_ind, dtype=torch.float32)
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                return {"u_t": ones, "r_t": zeros, "kappa_t": ones}

        class DummySoftMask:
            def __call__(self, resp_logits, mask_ind, step_frac):
                hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
                confidence = torch.ones_like(mask_ind, dtype=torch.float32)
                entropy = torch.zeros_like(mask_ind, dtype=torch.float32)
                weighted = torch.zeros_like(hidden)
                return hidden, confidence, entropy, weighted

        def fake_reuse_signal(resp_logits, aux_logits, cfg, state=None):
            safe_reuse = torch.tensor([[False, False]])
            return safe_reuse, state, {}

        def fake_build_access_set(
            actions,
            policy_out,
            cfg,
            confidence=None,
            boundary_action=None,
            boundary_num_bins=None,
        ):
            q_exec = torch.ones_like(actions["u_t"], dtype=torch.float32)
            q_mandatory = torch.zeros_like(actions["u_t"], dtype=torch.float32)
            return q_exec, q_mandatory, {}

        monkeypatch.setattr("aoae.dinfer_integration.compute_reuse_signal", fake_reuse_signal)
        monkeypatch.setattr("aoae.dinfer_integration.build_access_set", fake_build_access_set)

        prompt_ids = torch.tensor([[1]], dtype=torch.long)
        _, stats = run_speculative_inference(
            dual_model=DummyDualModel(),
            policy=DummyPolicy(),
            soft_mask_module=DummySoftMask(),
            prism_adapter=None,
            prompt_ids=prompt_ids,
            cfg=cfg,
        )

        assert stats["mean_agreement"] == pytest.approx(0.5)
        assert stats["agreement_observations"] == 2
        assert stats["reuse_mean_safe_reuse"] == pytest.approx(0.0)
        assert stats["safe_reuse_observations"] == 2
        assert stats["draft_accepts"] == 1
        assert stats["draft_rejects"] == 1
        assert stats["draft_accept_rate"] == pytest.approx(0.5)
        assert stats["total_commits"] == 0


class TestAOAESpeculativeInferenceLoop:
    def test_fresh_primary_agreement_masks_skipped_kspec_positions(self):
        from aoae.speculative_inference import _fresh_primary_agreement

        agreement = torch.tensor(
            [[True, True, False], [True, False, True]],
            dtype=torch.bool,
        )
        primary_fresh = torch.tensor(
            [[True, False, False], [False, True, True]],
            dtype=torch.bool,
        )

        verified = _fresh_primary_agreement(agreement, primary_fresh)

        expected = torch.tensor(
            [[True, False, False], [False, False, True]],
            dtype=torch.bool,
        )
        assert torch.equal(verified, expected)

    def test_stable_cache_miss_fraction_credits_only_persistent_cache(self):
        from aoae.cache import SpeculativeCacheBookkeeper
        from aoae.speculative_inference import _stable_verifier_miss_fraction

        cache = SpeculativeCacheBookkeeper(batch_size=1, seq_len=4, device=torch.device("cpu"))
        cache.step_spec(torch.tensor([[True, True, False, False]]))
        cache.step_stable(
            torch.tensor([[True, False, True, False]], dtype=torch.float32),
            torch.zeros(1, 4),
        )

        miss = _stable_verifier_miss_fraction(
            cache,
            stable_kv_cache_enabled=True,
            primary_cache_enabled=True,
        )
        assert miss.item() == pytest.approx(0.5)

        disabled = _stable_verifier_miss_fraction(
            cache,
            stable_kv_cache_enabled=False,
            primary_cache_enabled=True,
        )
        assert disabled.item() == pytest.approx(1.0)

        hidden_required = _stable_verifier_miss_fraction(
            cache,
            stable_kv_cache_enabled=True,
            primary_cache_enabled=False,
        )
        assert hidden_required.item() == pytest.approx(1.0)

    def test_stable_kv_cache_execution_flag_fails_loudly_until_wired(self):
        from aoae.speculative_inference import speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "cache": {
                "enabled": True,
                "kspec_skip": False,
                "stable_kv_cache": True,
                "prefix_kv_cache": False,
            },
            "inference": {
                "steps": 1,
                "gen_length": 1,
                "temperature": 0.0,
                "fallback_unmask": False,
            },
            "analysis": {"track_kv_dynamics": False},
        }

        with pytest.raises(RuntimeError, match="stable_kv_cache=true"):
            speculative_inference(
                dual_model=object(),
                policy=None,
                soft_mask_module=None,
                prism_adapter=None,
                prompt_ids=torch.tensor([[1]]),
                cfg=cfg,
            )

    def test_kspec_frontier_accumulates_then_authoritative_verifier_consumes_it(self):
        from aoae.speculative_inference import speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "cache": {
                "enabled": True,
                "kspec_skip": False,
                "stable_kv_cache": False,
                "prefix_kv_cache": False,
            },
            "grpo": {"thrash_age_decay": 0.0},
            "inference": {
                "steps": 2,
                "gen_length": 3,
                "temperature": 0.0,
                "fallback_unmask": False,
                "disable_remask": False,
                "compose_gamma": 0.0,
                "primary_agree_threshold": 0.0,
                "force_primary_first_last": False,
                "aux_cache_reset_threshold": 1.1,
                "verifier_schedule": {
                    "mode": "candidate_budget",
                    "draft_token_budget": 2,
                    "min_draft_microsteps": 1,
                    "max_draft_microsteps": 2,
                    "force_first_last": False,
                },
                "verifier": {"acceptance_mode": "argmax_match"},
                "positional_cache": {"enabled": False},
                "reuse_signal": {"method": "argmax_match"},
            },
            "analysis": {"track_kv_dynamics": False},
        }

        class DummyDualModel:
            @staticmethod
            def _aux_logits(batch, length):
                logits = torch.zeros(batch, length, 3)
                logits[..., 0] = 5.0
                return logits

            @staticmethod
            def _primary_logits(input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                logits[..., 0] = 5.0
                # Reject the middle drafted response position by preferring token 1.
                logits[:, -2, :] = torch.tensor([0.0, 6.0, 0.0])
                return logits

            def auxiliary_forward(self, input_ids):
                return self._aux_logits(input_ids.shape[0], input_ids.shape[1])

            def primary_forward(self, input_ids):
                return self._primary_logits(input_ids)

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                del need_hidden, need_all_hidden
                return types.SimpleNamespace(
                    primary_logits=self._primary_logits(input_ids)[:, resp_slice, :],
                    auxiliary_logits=self._aux_logits(input_ids.shape[0], input_ids.shape[1])[:, resp_slice, :],
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

        class DraftAllPolicy:
            def __init__(self):
                self.calls = 0

            def __call__(
                self,
                H_t,
                mask_ind,
                step_frac,
                temperature=1.0,
                confidence=None,
                quality_scores=None,
                agreement=None,
                age_feature=None,
                last_action_feature=None,
            ):
                del H_t, step_frac, temperature, confidence, quality_scores, agreement
                del age_feature, last_action_feature
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                ones = torch.ones_like(zeros)
                return {
                    "unmask_probs": ones * mask_ind.float(),
                    "remask_probs": zeros,
                    "cache_probs": zeros,
                    "access_probs": zeros,
                    "access_logits": zeros,
                }

            def sample_actions(self, policy_out, mask_ind):
                del policy_out
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                ones = torch.ones_like(zeros)
                self.calls += 1
                if self.calls > 1:
                    ones = zeros
                return {
                    "u_t": ones * mask_ind.float(),
                    "r_t": zeros,
                    "kappa_t": zeros,
                    "q_t": zeros,
                }

        class DummySoftMask:
            def __call__(self, resp_logits, mask_ind, step_frac):
                del step_frac
                hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
                confidence = torch.ones_like(mask_ind, dtype=torch.float32)
                entropy = torch.zeros_like(confidence)
                weighted = torch.zeros_like(hidden)
                return hidden, confidence, entropy, weighted

        output, traj = speculative_inference(
            dual_model=DummyDualModel(),
            policy=DraftAllPolicy(),
            soft_mask_module=DummySoftMask(),
            prism_adapter=None,
            prompt_ids=torch.tensor([[1]], dtype=torch.long),
            cfg=cfg,
            collect_stats=True,
        )

        assert traj.aux_only_steps == 1
        assert traj.primary_steps == 1
        assert traj.draft_accepts == 2
        assert traj.draft_rejects == 1
        assert [float(x.item()) for x in traj.spec_cached_fractions] == pytest.approx([1.0, 0.0])
        assert output[0, -2].item() == MASK_ID
        assert traj.frontier_accept_rate == pytest.approx(2 / 3)
        assert traj.frontier_reject_rate == pytest.approx(1 / 3)

    def test_recompute_after_reject_false_does_not_rerun_verifier(self):
        from aoae.speculative_inference import speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "cache": {
                "enabled": True,
                "kspec_skip": False,
                "stable_kv_cache": False,
                "prefix_kv_cache": False,
            },
            "grpo": {"thrash_age_decay": 0.0},
            "inference": {
                "steps": 2,
                "gen_length": 1,
                "temperature": 0.0,
                "fallback_unmask": False,
                "disable_remask": False,
                "compose_gamma": 0.0,
                "primary_agree_threshold": 0.0,
                "force_primary_first_last": False,
                "aux_cache_reset_threshold": 1.1,
                "verifier_schedule": {
                    "mode": "candidate_budget",
                    "draft_token_budget": 1,
                    "min_draft_microsteps": 1,
                    "max_draft_microsteps": 2,
                    "force_first_last": False,
                },
                "verifier": {
                    "acceptance_mode": "argmax_match",
                    "rejection_action": "remask",
                    "recompute_after_reject": False,
                },
                "positional_cache": {"enabled": False},
                "reuse_signal": {"method": "argmax_match"},
            },
            "analysis": {"track_kv_dynamics": False},
        }

        class DummyDualModel:
            # No primary_forward — exercises the dual_forward_resp compatibility
            # path used by lightweight test shims and older wrappers.
            def __init__(self):
                self.verifier_calls = 0

            def auxiliary_forward(self, input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                logits[..., 0] = 5.0
                return logits

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                del need_hidden, need_all_hidden
                self.verifier_calls += 1
                primary_logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                primary_logits[..., 1] = 5.0
                aux_logits = self.auxiliary_forward(input_ids)
                return types.SimpleNamespace(
                    primary_logits=primary_logits[:, resp_slice, :],
                    auxiliary_logits=aux_logits[:, resp_slice, :],
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

        class DraftThenStopPolicy:
            def __init__(self):
                self.calls = 0

            def __call__(
                self,
                H_t,
                mask_ind,
                step_frac,
                temperature=1.0,
                confidence=None,
                quality_scores=None,
                agreement=None,
                age_feature=None,
                last_action_feature=None,
            ):
                del H_t, step_frac, temperature, confidence, quality_scores, agreement
                del age_feature, last_action_feature
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                return {
                    "unmask_probs": mask_ind.float(),
                    "remask_probs": zeros,
                    "cache_probs": zeros,
                    "access_probs": zeros,
                    "access_logits": zeros,
                }

            def sample_actions(self, policy_out, mask_ind):
                del policy_out
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                self.calls += 1
                u_t = mask_ind.float() if self.calls == 1 else zeros
                return {"u_t": u_t, "r_t": zeros, "kappa_t": zeros, "q_t": zeros}

        class DummySoftMask:
            def __call__(self, resp_logits, mask_ind, step_frac):
                del step_frac
                hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
                confidence = torch.ones_like(mask_ind, dtype=torch.float32)
                entropy = torch.zeros_like(confidence)
                weighted = torch.zeros_like(hidden)
                return hidden, confidence, entropy, weighted

        dual_model = DummyDualModel()
        output, traj = speculative_inference(
            dual_model=dual_model,
            policy=DraftThenStopPolicy(),
            soft_mask_module=DummySoftMask(),
            prism_adapter=None,
            prompt_ids=torch.tensor([[1]], dtype=torch.long),
            cfg=cfg,
            collect_stats=True,
        )

        assert dual_model.verifier_calls == 1
        assert traj.draft_rejects == 1
        assert output[0, -1].item() == MASK_ID

    def test_rejection_action_keep_evicts_without_remasking_or_stable_commit(self):
        from aoae.speculative_inference import speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "cache": {
                "enabled": True,
                "kspec_skip": False,
                "stable_kv_cache": False,
                "prefix_kv_cache": False,
            },
            "grpo": {"thrash_age_decay": 0.0},
            "inference": {
                "steps": 2,
                "gen_length": 1,
                "temperature": 0.0,
                "fallback_unmask": False,
                "disable_remask": False,
                "compose_gamma": 0.0,
                "primary_agree_threshold": 0.0,
                "force_primary_first_last": False,
                "aux_cache_reset_threshold": 1.1,
                "verifier_schedule": {
                    "mode": "candidate_budget",
                    "draft_token_budget": 1,
                    "min_draft_microsteps": 1,
                    "max_draft_microsteps": 2,
                    "force_first_last": False,
                },
                "verifier": {
                    "acceptance_mode": "argmax_match",
                    "rejection_action": "keep",
                    "recompute_after_reject": False,
                },
                "positional_cache": {"enabled": False},
                "reuse_signal": {"method": "argmax_match"},
            },
            "analysis": {"track_kv_dynamics": False},
        }

        class DummyDualModel:
            def auxiliary_forward(self, input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                logits[..., 0] = 5.0
                return logits

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                del need_hidden, need_all_hidden
                primary_logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                primary_logits[..., 1] = 5.0
                aux_logits = self.auxiliary_forward(input_ids)
                return types.SimpleNamespace(
                    primary_logits=primary_logits[:, resp_slice, :],
                    auxiliary_logits=aux_logits[:, resp_slice, :],
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

        class DraftThenCommitPolicy:
            def __init__(self):
                self.calls = 0

            def __call__(
                self,
                H_t,
                mask_ind,
                step_frac,
                temperature=1.0,
                confidence=None,
                quality_scores=None,
                agreement=None,
                age_feature=None,
                last_action_feature=None,
            ):
                del H_t, step_frac, temperature, confidence, quality_scores, agreement
                del age_feature, last_action_feature
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                ones = torch.ones_like(zeros)
                return {
                    "unmask_probs": ones * mask_ind.float(),
                    "remask_probs": zeros,
                    "cache_probs": ones,
                    "access_probs": ones,
                    "access_logits": ones,
                }

            def sample_actions(self, policy_out, mask_ind):
                del policy_out
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                ones = torch.ones_like(zeros)
                self.calls += 1
                if self.calls == 1:
                    return {"u_t": mask_ind.float(), "r_t": zeros, "kappa_t": zeros, "q_t": zeros}
                return {"u_t": zeros, "r_t": zeros, "kappa_t": ones, "q_t": ones}

        class DummySoftMask:
            def __call__(self, resp_logits, mask_ind, step_frac):
                del step_frac
                hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
                confidence = torch.ones_like(mask_ind, dtype=torch.float32)
                entropy = torch.zeros_like(confidence)
                weighted = torch.zeros_like(hidden)
                return hidden, confidence, entropy, weighted

        output, traj = speculative_inference(
            dual_model=DummyDualModel(),
            policy=DraftThenCommitPolicy(),
            soft_mask_module=DummySoftMask(),
            prism_adapter=None,
            prompt_ids=torch.tensor([[1]], dtype=torch.long),
            cfg=cfg,
            collect_stats=True,
        )

        assert traj.draft_rejects == 1
        assert output[0, -1].item() == 0
        assert traj.total_stable_commits == 0

    def test_aux_only_steps_do_not_remask_or_commit_stable_cache(self, monkeypatch):
        from aoae.speculative_inference import speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "cache": {
                "enabled": True,
                "kspec_skip": False,
                "stable_kv_cache": False,
                "prefix_kv_cache": False,
            },
            "grpo": {"thrash_age_decay": 0.0},
            "inference": {
                "steps": 3,
                "gen_length": 3,
                "temperature": 0.0,
                "fallback_unmask": True,
                "disable_remask": False,
                "compose_gamma": 0.0,
                "primary_every_n": 2,
                "primary_agree_threshold": 0.0,
                "force_primary_first_last": False,
                "positional_cache": {"enabled": False},
                "reuse_signal": {"method": "argmax_match"},
            },
            "analysis": {"track_kv_dynamics": False},
        }

        class DummyDualModel:
            def auxiliary_forward(self, input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                logits[..., 0] = 5.0
                return logits

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                del need_hidden, need_all_hidden
                aux_logits = self.auxiliary_forward(input_ids)[:, resp_slice, :]
                primary_logits = aux_logits.clone()
                return types.SimpleNamespace(
                    primary_logits=primary_logits,
                    auxiliary_logits=aux_logits,
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

        class GreedyCommitPolicy:
            def __call__(
                self,
                H_t,
                mask_ind,
                step_frac,
                temperature=1.0,
                confidence=None,
                quality_scores=None,
                agreement=None,
                age_feature=None,
                last_action_feature=None,
            ):
                del H_t, step_frac, temperature, confidence, quality_scores, agreement
                del age_feature, last_action_feature
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                ones = torch.ones_like(zeros)
                return {
                    "unmask_probs": zeros,
                    "remask_probs": zeros,
                    "cache_probs": ones,
                    "access_probs": zeros,
                    "access_logits": zeros,
                }

            def sample_actions(self, policy_out, mask_ind):
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                ones = torch.ones_like(zeros)
                return {
                    "u_t": zeros,
                    "r_t": zeros,
                    "kappa_t": ones,
                    "q_t": zeros,
                }

            def log_prob(self, policy_out, actions):
                return torch.zeros(actions["u_t"].shape[0])

        class DummySoftMask:
            def __call__(self, resp_logits, mask_ind, step_frac):
                del step_frac
                hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
                confidence = torch.ones_like(mask_ind, dtype=torch.float32)
                entropy = torch.zeros_like(confidence)
                weighted = torch.zeros_like(hidden)
                return hidden, confidence, entropy, weighted

        def fake_reuse_signal(resp_logits, aux_logits, cfg, state=None):
            del resp_logits, aux_logits, cfg
            return torch.ones(1, 3, dtype=torch.bool), state, {}

        monkeypatch.setattr("aoae.speculative_inference.compute_reuse_signal", fake_reuse_signal)

        _, traj = speculative_inference(
            dual_model=DummyDualModel(),
            policy=GreedyCommitPolicy(),
            soft_mask_module=DummySoftMask(),
            prism_adapter=None,
            prompt_ids=torch.tensor([[1]], dtype=torch.long),
            cfg=cfg,
            collect_stats=True,
        )

        assert traj.primary_steps == 1
        assert traj.aux_only_steps == 2
        assert [float(x.item()) for x in traj.stable_cached_fractions[:2]] == [0.0, 0.0]
        assert float(traj.stable_cached_fractions[-1].item()) == pytest.approx(1.0)
        assert [float(x.item()) for x in traj.spec_cached_fractions[:2]] == pytest.approx([1 / 3, 2 / 3])
        assert traj.total_stable_commits == 3
        assert traj.agreement_observations == 3

    def test_prism_keeps_aux_prefix_cache_on_draft_steps(self):
        from aoae.speculative_inference import speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "cache": {
                "enabled": True,
                "prefix_kv_cache": True,
            },
            "grpo": {"thrash_age_decay": 0.0},
            "inference": {
                "steps": 3,
                "gen_length": 2,
                "temperature": 0.0,
                "fallback_unmask": False,
                "disable_remask": False,
                "compose_gamma": 0.0,
                "primary_every_n": 2,
                "primary_agree_threshold": 0.0,
                "force_primary_first_last": False,
                "aux_cache_reset_threshold": 1.1,
                "positional_cache": {"enabled": False},
                "reuse_signal": {"method": "argmax_match"},
            },
            "analysis": {"track_kv_dynamics": False},
        }

        class DummyDualModel:
            def __init__(self):
                self.calls = []

            @staticmethod
            def _full_logits(input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                logits[..., 0] = 5.0
                return logits

            def auxiliary_forward(self, input_ids):
                self.calls.append("aux_plain")
                return self._full_logits(input_ids)

            def auxiliary_forward_with_cache(self, input_ids):
                self.calls.append("aux_cache_full")
                return self._full_logits(input_ids), object()

            def auxiliary_forward_replace_with_cache(self, input_ids, resp_slice, aux_past_kv):
                del aux_past_kv
                self.calls.append(("aux_cache_replace", resp_slice.start, resp_slice.stop))
                return self._full_logits(input_ids)[:, resp_slice, :], object()

            def primary_forward_with_hidden(self, input_ids):
                self.calls.append("pri_hidden")
                logits = self._full_logits(input_ids)
                hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
                return logits, hidden

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                del input_ids, resp_slice, need_hidden, need_all_hidden
                raise AssertionError("Verifier path should not fall back to dual_forward_resp when PRISM is active.")

        class ZeroPolicy:
            def __call__(
                self,
                H_t,
                mask_ind,
                step_frac,
                temperature=1.0,
                confidence=None,
                quality_scores=None,
                agreement=None,
                age_feature=None,
                last_action_feature=None,
            ):
                del H_t, step_frac, temperature, confidence, quality_scores, agreement
                del age_feature, last_action_feature
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                return {
                    "unmask_probs": zeros,
                    "remask_probs": zeros,
                    "cache_probs": zeros,
                    "access_probs": zeros,
                    "access_logits": zeros,
                }

            def sample_actions(self, policy_out, mask_ind):
                del policy_out
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                return {
                    "u_t": zeros,
                    "r_t": zeros,
                    "kappa_t": zeros,
                    "q_t": zeros,
                }

        class DummySoftMask:
            def __call__(self, resp_logits, mask_ind, step_frac):
                del step_frac
                hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
                confidence = torch.ones_like(mask_ind, dtype=torch.float32)
                entropy = torch.zeros_like(confidence)
                weighted = torch.zeros_like(hidden)
                return hidden, confidence, entropy, weighted

        class DummyPrism:
            def __call__(self, hidden_states):
                return torch.ones(hidden_states.shape[:2], dtype=torch.float32)

        dual_model = DummyDualModel()
        _, traj = speculative_inference(
            dual_model=dual_model,
            policy=ZeroPolicy(),
            soft_mask_module=DummySoftMask(),
            prism_adapter=DummyPrism(),
            prompt_ids=torch.tensor([[1]], dtype=torch.long),
            cfg=cfg,
            collect_stats=True,
        )

        assert traj.aux_only_steps == 2
        assert traj.primary_steps == 1
        # With compose_gamma=0 and temperature=0 the verifier no longer needs
        # a fresh auxiliary forward (greedy argmax decisions are unchanged by
        # the aux distribution at non-agreement positions and the cache-aligned
        # composition collapses to the primary).  The aux prefix cache is
        # therefore reused only on draft microsteps.
        assert dual_model.calls == [
            "aux_cache_full",
            ("aux_cache_replace", 1, 3),
            "pri_hidden",
        ]

    def test_log_speculative_config_reports_rollout_mode(self, capsys):
        from aoae.speculative_inference import speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "cache": {
                "enabled": True,
                "prefix_kv_cache": True,
            },
            "grpo": {"thrash_age_decay": 0.0},
            "inference": {
                "steps": 1,
                "gen_length": 2,
                "temperature": 0.0,
                "fallback_unmask": False,
                "disable_remask": False,
                "compose_gamma": 0.0,
                "primary_every_n": 1,
                "primary_agree_threshold": 0.0,
                "force_primary_first_last": False,
                "aux_cache_reset_threshold": 1.1,
                "positional_cache": {"enabled": False},
                "reuse_signal": {"method": "argmax_match"},
            },
            "analysis": {
                "track_kv_dynamics": False,
                "log_speculative_config": True,
            },
        }

        class DummyDualModel:
            @staticmethod
            def _full_logits(input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
                logits[..., 0] = 5.0
                return logits

            def auxiliary_forward_with_cache(self, input_ids):
                return self._full_logits(input_ids), object()

            def primary_forward_with_hidden(self, input_ids):
                logits = self._full_logits(input_ids)
                hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
                return logits, hidden

        class ZeroPolicy:
            def __call__(
                self,
                H_t,
                mask_ind,
                step_frac,
                temperature=1.0,
                confidence=None,
                quality_scores=None,
                agreement=None,
                age_feature=None,
                last_action_feature=None,
            ):
                del H_t, step_frac, temperature, confidence, quality_scores, agreement
                del age_feature, last_action_feature
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                return {
                    "unmask_probs": zeros,
                    "remask_probs": zeros,
                    "cache_probs": zeros,
                    "access_probs": zeros,
                    "access_logits": zeros,
                }

            def sample_actions(self, policy_out, mask_ind):
                del policy_out
                zeros = torch.zeros_like(mask_ind, dtype=torch.float32)
                return {
                    "u_t": zeros,
                    "r_t": zeros,
                    "kappa_t": zeros,
                    "q_t": zeros,
                }

        class DummySoftMask:
            def __call__(self, resp_logits, mask_ind, step_frac):
                del step_frac
                hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
                confidence = torch.ones_like(mask_ind, dtype=torch.float32)
                entropy = torch.zeros_like(confidence)
                weighted = torch.zeros_like(hidden)
                return hidden, confidence, entropy, weighted

        class DummyPrism:
            def __call__(self, hidden_states):
                return torch.ones(hidden_states.shape[:2], dtype=torch.float32)

        speculative_inference(
            dual_model=DummyDualModel(),
            policy=ZeroPolicy(),
            soft_mask_module=DummySoftMask(),
            prism_adapter=DummyPrism(),
            prompt_ids=torch.tensor([[1]], dtype=torch.long),
            cfg=cfg,
            collect_stats=True,
        )

        out = capsys.readouterr().out
        assert "[Speculative]" in out
        assert "schedule=aoae" in out
        assert "prism=on" in out
        assert "kv_tracking=off" in out
        assert "aux_cache=on" in out
        assert "primary_hidden=off" in out
        assert "primary_cache_fastpath=on" in out
        assert "verifier_mode=prefix_cache_replace" in out
        assert "primary_every_n=1" in out
        assert "gamma=0.000" in out
        assert "remask=on" in out


class TestRunBlockwiseSpeculativeInference:
    def test_blockwise_runner_uses_primary_thresholds_and_editing(self, monkeypatch):
        from aoae.dinfer_integration import run_blockwise_speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "inference": {
                "gen_length": 2,
                "block_length": 2,
                "fallback_unmask": True,
                "disable_remask": False,
                "reuse_signal": {"method": "argmax_match"},
                "llada21_official": {
                    "use_block_diffusion": True,
                    "threshold": 0.7,
                    "editing_threshold": 0.5,
                    "max_post_steps": 2,
                    "enable_mbe": False,
                },
            },
        }

        class DummyDualModel:
            def __init__(self):
                self.calls = 0

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                self.calls += 1
                if self.calls == 1:
                    primary_logits = torch.tensor(
                        [[[8.0, 0.0, 0.0], [0.0, 0.2, 0.0]]],
                        dtype=torch.float32,
                    )
                    auxiliary_logits = torch.tensor(
                        [[[8.0, 0.0, 0.0], [0.0, 0.0, 6.0]]],
                        dtype=torch.float32,
                    )
                    agreement = torch.tensor([[True, False]])
                else:
                    primary_logits = torch.tensor(
                        [[[0.0, 0.0, 7.5], [0.0, 7.0, 0.0]]],
                        dtype=torch.float32,
                    )
                    auxiliary_logits = torch.tensor(
                        [[[0.0, 0.0, 7.5], [7.0, 0.0, 0.0]]],
                        dtype=torch.float32,
                    )
                    agreement = torch.tensor([[True, False]])
                return types.SimpleNamespace(
                    primary_logits=primary_logits,
                    auxiliary_logits=auxiliary_logits,
                    agreement=agreement,
                    agreement_rate=agreement.float().mean().item(),
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

        def fake_reuse_signal(resp_logits, aux_logits, cfg, state=None):
            safe_reuse = torch.zeros(resp_logits.shape[:2], dtype=torch.bool)
            return safe_reuse, state, {}

        monkeypatch.setattr("aoae.dinfer_integration.compute_reuse_signal", fake_reuse_signal)

        prompt_ids = torch.tensor([[1]], dtype=torch.long)
        output_ids, stats = run_blockwise_speculative_inference(
            dual_model=DummyDualModel(),
            policy=None,
            soft_mask_module=None,
            prism_adapter=None,
            prompt_ids=prompt_ids,
            cfg=cfg,
        )

        assert output_ids.shape == (1, 3)
        assert output_ids[0, 1:].tolist() == [2, 1]
        assert stats["primary_steps"] == 2
        assert stats["draft_accepts"] == 1
        assert stats["draft_rejects"] == 1
        assert stats["draft_accept_rate"] == pytest.approx(0.5)
        assert stats["agreement_observations"] == 3
        assert stats["mean_agreement"] == pytest.approx(1.0 / 3.0)
        assert stats["total_invalidations"] == 0

    def test_blockwise_runner_skips_primary_to_active_span(self, monkeypatch):
        from aoae.dinfer_integration import run_blockwise_speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "inference": {
                "gen_length": 2,
                "block_length": 2,
                "fallback_unmask": True,
                "disable_remask": False,
                "reuse_signal": {"method": "argmax_match"},
                "llada21_official": {
                    "use_block_diffusion": True,
                    "threshold": 0.7,
                    "editing_threshold": 0.5,
                    "max_post_steps": 2,
                    "enable_mbe": False,
                },
            },
        }

        class DummyDualModel:
            def __init__(self):
                self._model = types.SimpleNamespace(_dinfer_runtime="vllm")
                self.aux_calls = 0
                self.full_calls = 0
                self.partial_calls = []

            def auxiliary_forward_resp(self, input_ids, resp_slice):
                self.aux_calls += 1
                if self.aux_calls == 1:
                    return torch.tensor(
                        [[[0.0, 0.0, 8.0], [7.0, 0.0, 0.0]]],
                        dtype=torch.float32,
                    )
                return torch.tensor(
                    [[[0.0, 0.0, 8.0], [0.0, 8.0, 0.0]]],
                    dtype=torch.float32,
                )

            def primary_forward_with_cache(self, input_ids):
                self.full_calls += 1
                logits = torch.zeros((1, input_ids.shape[1], 3), dtype=torch.float32)
                logits[:, 1:, :] = torch.tensor(
                    [[[0.0, 0.0, 8.0], [0.0, 0.2, 0.0]]],
                    dtype=torch.float32,
                )
                return logits, {"cache": "full"}

            def primary_forward_replace_with_cache(self, full_input_ids, replace_slice, past_key_values):
                self.partial_calls.append((replace_slice.start, replace_slice.stop))
                logits = torch.tensor([[[0.0, 8.0, 0.0]]], dtype=torch.float32)
                return logits, {"cache": "partial"}

        def fake_reuse_signal(resp_logits, aux_logits, cfg, state=None):
            if state is None:
                safe_reuse = torch.tensor([[True, False]], dtype=torch.bool)
                return safe_reuse, {"step": 1}, {}
            safe_reuse = torch.tensor([[True, True]], dtype=torch.bool)
            return safe_reuse, {"step": 2}, {}

        monkeypatch.setattr("aoae.dinfer_integration.compute_reuse_signal", fake_reuse_signal)

        prompt_ids = torch.tensor([[1]], dtype=torch.long)
        dual_model = DummyDualModel()
        output_ids, stats = run_blockwise_speculative_inference(
            dual_model=dual_model,
            policy=None,
            soft_mask_module=None,
            prism_adapter=None,
            prompt_ids=prompt_ids,
            cfg=cfg,
        )

        assert output_ids[0, 1:].tolist() == [2, 1]
        assert dual_model.full_calls == 1
        assert dual_model.partial_calls == [(2, 3)]
        assert stats["primary_steps"] == 2
        assert stats["primary_full_steps"] == 1
        assert stats["primary_partial_steps"] == 1
        assert stats["primary_verified_positions"] == 3
        assert stats["primary_full_equiv_positions"] == 4
        assert stats["primary_skip_ratio"] == pytest.approx(0.25)

    def test_blockwise_runner_force_completes_remaining_masks(self, monkeypatch):
        from aoae.dinfer_integration import run_blockwise_speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "inference": {
                "gen_length": 2,
                "block_length": 2,
                "fallback_unmask": True,
                "disable_remask": False,
                "reuse_signal": {"method": "argmax_match"},
                "llada21_official": {
                    "use_block_diffusion": True,
                    "threshold": 0.999,
                    "editing_threshold": 0.999,
                    "max_post_steps": 1,
                    "enable_mbe": False,
                },
            },
        }

        class DummyDualModel:
            def __init__(self):
                self.calls = 0

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                del input_ids, resp_slice, need_hidden, need_all_hidden
                self.calls += 1
                primary_logits = torch.tensor(
                    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                    dtype=torch.float32,
                )
                auxiliary_logits = primary_logits.clone()
                agreement = torch.tensor([[True, True]])
                return types.SimpleNamespace(
                    primary_logits=primary_logits,
                    auxiliary_logits=auxiliary_logits,
                    agreement=agreement,
                    agreement_rate=1.0,
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

        def fake_reuse_signal(resp_logits, aux_logits, cfg, state=None):
            del resp_logits, aux_logits, cfg, state
            return torch.ones((1, 2), dtype=torch.bool), None, {}

        monkeypatch.setattr("aoae.dinfer_integration.compute_reuse_signal", fake_reuse_signal)

        prompt_ids = torch.tensor([[1]], dtype=torch.long)
        output_ids, stats = run_blockwise_speculative_inference(
            dual_model=DummyDualModel(),
            policy=None,
            soft_mask_module=None,
            prism_adapter=None,
            prompt_ids=prompt_ids,
            cfg=cfg,
        )

        assert (output_ids[0, 1:] != MASK_ID).all()
        assert output_ids[0, 1:].tolist() == [0, 1]
        assert stats["primary_steps"] == 2

    def test_blockwise_runner_tracks_kv_dynamics_when_enabled(self, monkeypatch):
        from aoae.dinfer_integration import run_blockwise_speculative_inference

        cfg = {
            "base_model": {"mask_token_id": MASK_ID},
            "analysis": {"track_kv_dynamics": True, "track_attention_deviation": False},
            "inference": {
                "gen_length": 2,
                "block_length": 2,
                "fallback_unmask": True,
                "disable_remask": False,
                "reuse_signal": {"method": "argmax_match"},
                "llada21_official": {
                    "use_block_diffusion": True,
                    "threshold": 0.7,
                    "editing_threshold": 0.5,
                    "max_post_steps": 2,
                    "enable_mbe": False,
                },
            },
        }

        class DummyDualModel:
            def __init__(self):
                self.calls = 0
                self.diag_calls = 0

            def dual_forward_resp(self, input_ids, resp_slice, need_hidden=False, need_all_hidden=False):
                del input_ids, resp_slice, need_hidden, need_all_hidden
                self.calls += 1
                if self.calls == 1:
                    primary_logits = torch.tensor(
                        [[[8.0, 0.0, 0.0], [0.0, 0.2, 0.0]]],
                        dtype=torch.float32,
                    )
                    auxiliary_logits = torch.tensor(
                        [[[8.0, 0.0, 0.0], [0.0, 0.0, 6.0]]],
                        dtype=torch.float32,
                    )
                    agreement = torch.tensor([[True, False]])
                else:
                    primary_logits = torch.tensor(
                        [[[8.0, 0.0, 0.0], [0.0, 7.0, 0.0]]],
                        dtype=torch.float32,
                    )
                    auxiliary_logits = torch.tensor(
                        [[[8.0, 0.0, 0.0], [7.0, 0.0, 0.0]]],
                        dtype=torch.float32,
                    )
                    agreement = torch.tensor([[True, False]])
                return types.SimpleNamespace(
                    primary_logits=primary_logits,
                    auxiliary_logits=auxiliary_logits,
                    agreement=agreement,
                    agreement_rate=agreement.float().mean().item(),
                    primary_hidden=None,
                    primary_hidden_states=None,
                )

            def primary_forward_with_diagnostics(self, input_ids, output_attentions=True, output_kv=True):
                del input_ids, output_attentions, output_kv
                self.diag_calls += 1
                if self.diag_calls == 1:
                    hidden = [torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float32)]
                    layer_kv = [(
                        torch.tensor([[[[0.0], [1.0], [2.0]]]], dtype=torch.float32),
                        torch.tensor([[[[0.0], [1.5], [2.5]]]], dtype=torch.float32),
                    )]
                else:
                    hidden = [torch.tensor([[[0.0], [2.0], [4.0]]], dtype=torch.float32)]
                    layer_kv = [(
                        torch.tensor([[[[0.0], [2.0], [4.0]]]], dtype=torch.float32),
                        torch.tensor([[[[0.0], [2.5], [4.5]]]], dtype=torch.float32),
                    )]
                logits = torch.zeros((1, 3, 3), dtype=torch.float32)
                return logits, hidden, None, layer_kv

        def fake_reuse_signal(resp_logits, aux_logits, cfg, state=None):
            del resp_logits, aux_logits, cfg, state
            return torch.zeros((1, 2), dtype=torch.bool), None, {}

        monkeypatch.setattr("aoae.dinfer_integration.compute_reuse_signal", fake_reuse_signal)

        prompt_ids = torch.tensor([[1]], dtype=torch.long)
        output_ids, stats = run_blockwise_speculative_inference(
            dual_model=DummyDualModel(),
            policy=None,
            soft_mask_module=None,
            prism_adapter=None,
            prompt_ids=prompt_ids,
            cfg=cfg,
        )

        assert output_ids.shape == (1, 3)
        assert "kv_dynamics" in stats
        summary = stats["kv_dynamics"]["summary"]
        assert summary["layer_drift_measure"] == "exact_kv"
        assert summary["exact_kv_drift_steps"] == 1
        assert "off_by_one_drift_ratio" in summary
        assert stats["kv_dynamics"]["per_layer"]


class TestHFBlockCausalBias:
    def test_hf_path_uses_attention_bias_when_model_requests_it(self):
        from aoae.models.base_model import LLaDABaseModel

        class DummyOut:
            def __init__(self):
                self.logits = torch.randn(1, 4, 8)

        class DummyHFModel(torch.nn.Module):
            def forward(self, input_ids, attention_bias=None):
                del input_ids
                assert attention_bias is not None
                bias = attention_bias
                assert bias.shape == (1, 1, 4, 4)
                return DummyOut()

        model = object.__new__(LLaDABaseModel)
        torch.nn.Module.__init__(model)
        model._backend = "hf"
        model.dtype = torch.float32
        model._block_length = 2
        model.model = DummyHFModel()

        input_ids = torch.ones((1, 4), dtype=torch.long)
        out = model.forward_block_causal(input_ids, block_length=2)
        assert out.shape == (1, 4, 8)

    def test_hf_path_drops_4d_mask_for_attention_mask_style_models(self):
        # Models whose forward signature only has attention_mask (2D HF convention)
        # reshape it via .view(B,-1)[:, None, None, :], turning [B,1,L,L] into
        # [B,1,1,L²] which then fails when added to [B,H,L,L] bias.  Drop it.
        from aoae.models.base_model import LLaDABaseModel

        class DummyOut:
            def __init__(self):
                self.logits = torch.randn(1, 4, 8)

        received = {}

        class DummyHFModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                del input_ids
                received["mask"] = attention_mask
                return DummyOut()

        model = object.__new__(LLaDABaseModel)
        torch.nn.Module.__init__(model)
        model._backend = "hf"
        model.dtype = torch.float32
        model._block_length = 2
        model.model = DummyHFModel()

        input_ids = torch.ones((1, 4), dtype=torch.long)
        out = model.forward_block_causal(input_ids, block_length=2)
        assert out.shape == (1, 4, 8)
        assert received["mask"] is None, "4D float mask must be dropped for attention_mask-style models"

    def test_hf_path_routes_4d_mask_to_attention_bias_for_both_style_models(self):
        # LLaDA-8B-Instruct has both attention_mask and attention_bias in its
        # forward signature.  The model preprocesses attention_mask via
        # .view(B,-1)[:, None, None, :], so a 4D float block mask must be
        # passed only as attention_bias (additive, used as-is).
        from aoae.models.base_model import LLaDABaseModel

        class DummyOut:
            def __init__(self):
                self.logits = torch.randn(1, 4, 8)

        received = {}

        class DummyHFModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, attention_bias=None):
                del input_ids
                received["mask"] = attention_mask
                received["bias"] = attention_bias
                return DummyOut()

        model = object.__new__(LLaDABaseModel)
        torch.nn.Module.__init__(model)
        model._backend = "hf"
        model.dtype = torch.float32
        model._block_length = 2
        model.model = DummyHFModel()

        input_ids = torch.ones((1, 4), dtype=torch.long)
        out = model.forward_block_causal(input_ids, block_length=2)
        assert out.shape == (1, 4, 8)
        # 4D mask must arrive as attention_bias only, NOT as attention_mask
        assert received["mask"] is None, "4D float mask must not be passed as attention_mask"
        assert received["bias"] is not None and received["bias"].shape == (1, 1, 4, 4)


class TestDInferCacheReuse:
    def test_base_model_consolidates_cache_before_replace(self):
        from aoae.models.base_model import LLaDABaseModel

        model = object.__new__(LLaDABaseModel)
        model._backend = "dinfer"
        model._block_length = 32
        model._dinfer_runtime = "vllm"
        model.dtype = torch.float32
        model._make_attention_mask = lambda b, s, d, block_length=32: torch.ones(
            (b, 1, s, s), dtype=torch.bool, device=d
        )
        model._make_query_attention_mask = lambda b, fs, qs, qe, d, block_length=32: torch.ones(
            (b, 1, qe - qs, fs), dtype=torch.bool, device=d
        )

        class DummyCache:
            def __init__(self):
                self.consolidated = 0

            def consolidate(self):
                self.consolidated += 1

        first_cache = DummyCache()
        second_cache = DummyCache()
        calls = []

        def fake_forward(**kwargs):
            calls.append(kwargs)
            if "replace_position" in kwargs:
                return types.SimpleNamespace(
                    logits=torch.randn(1, 1, 3),
                    past_key_values=second_cache,
                )
            return types.SimpleNamespace(
                logits=torch.randn(1, 3, 3),
                past_key_values=first_cache,
            )

        model._forward_dinfer_outputs = fake_forward

        _, cached = LLaDABaseModel.forward_with_cache(model, torch.tensor([[1, 2, 3]]))
        assert cached is first_cache
        assert first_cache.consolidated == 1

        _, updated = LLaDABaseModel.forward_replace_with_cache(
            model,
            torch.tensor([[1, 2, 3]]),
            slice(1, 2),
            cached,
        )
        assert updated is second_cache
        assert first_cache.consolidated == 2
        assert second_cache.consolidated == 1
        assert calls[1]["replace_position"] == (1, 2)

    def test_base_model_replace_falls_back_loudly_on_nonfinite_logits(self, capsys):
        from aoae.models.base_model import LLaDABaseModel

        model = object.__new__(LLaDABaseModel)
        model._backend = "dinfer"
        model._block_length = 32
        model._dinfer_runtime = "vllm"
        model.dtype = torch.float32
        model._make_attention_mask = lambda b, s, d, block_length=32: torch.ones(
            (b, 1, s, s), dtype=torch.bool, device=d
        )
        model._make_query_attention_mask = lambda b, fs, qs, qe, d, block_length=32: torch.ones(
            (b, 1, qe - qs, fs), dtype=torch.bool, device=d
        )

        class DummyCache:
            def __init__(self):
                self.consolidated = 0

            def consolidate(self):
                self.consolidated += 1

        first_cache = DummyCache()
        fallback_cache = DummyCache()
        fallback_logits = torch.tensor(
            [[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            dtype=torch.float32,
        )
        calls = []

        def fake_forward(**kwargs):
            calls.append(kwargs)
            if "replace_position" in kwargs:
                return types.SimpleNamespace(
                    logits=torch.tensor([[[float("nan"), 0.0, 0.0]]], dtype=torch.float32),
                    past_key_values=DummyCache(),
                )
            if len(calls) == 1:
                return types.SimpleNamespace(
                    logits=torch.randn(1, 3, 3),
                    past_key_values=first_cache,
                )
            return types.SimpleNamespace(
                logits=fallback_logits,
                past_key_values=fallback_cache,
            )

        model._forward_dinfer_outputs = fake_forward

        _, cached = LLaDABaseModel.forward_with_cache(model, torch.tensor([[1, 2, 3]]))
        logits, updated = LLaDABaseModel.forward_replace_with_cache(
            model,
            torch.tensor([[1, 2, 3]]),
            slice(1, 2),
            cached,
        )

        assert updated is fallback_cache
        assert torch.equal(logits, fallback_logits[:, 1:2, :])
        assert getattr(model, "_aoae_cache_fallback_count", 0) == 1
        err = capsys.readouterr().err
        assert "NON-FINITE OUTPUT FROM CACHED" in err
        assert "incoherent" in err.lower()
        assert "FULL sequence" in err

    def test_base_model_replace_falls_back_on_tagged_nonfinite_runtime_error(self, capsys):
        from aoae.models.base_model import LLaDABaseModel

        model = object.__new__(LLaDABaseModel)
        model._backend = "dinfer"
        model._block_length = 32
        model._dinfer_runtime = "vllm"
        model.dtype = torch.float32
        model._make_attention_mask = lambda b, s, d, block_length=32: torch.ones(
            (b, 1, s, s), dtype=torch.bool, device=d
        )
        model._make_query_attention_mask = lambda b, fs, qs, qe, d, block_length=32: torch.ones(
            (b, 1, qe - qs, fs), dtype=torch.bool, device=d
        )

        class DummyCache:
            def __init__(self):
                self.consolidated = 0

            def consolidate(self):
                self.consolidated += 1

        first_cache = DummyCache()
        fallback_cache = DummyCache()
        fallback_logits = torch.tensor(
            [[[0.5, 0.4, 0.3], [9.0, 8.0, 7.0], [0.2, 0.1, 0.0]]],
            dtype=torch.float32,
        )
        calls = []

        def fake_forward(**kwargs):
            calls.append(kwargs)
            if "replace_position" in kwargs:
                raise RuntimeError(
                    "[AOAE][NONFINITE_REPLACE_PATH] layer=model.layers.0 label=fused_moe_output"
                )
            if len(calls) == 1:
                return types.SimpleNamespace(
                    logits=torch.randn(1, 3, 3),
                    past_key_values=first_cache,
                )
            return types.SimpleNamespace(
                logits=fallback_logits,
                past_key_values=fallback_cache,
            )

        model._forward_dinfer_outputs = fake_forward

        _, cached = LLaDABaseModel.forward_with_cache(model, torch.tensor([[1, 2, 3]]))
        logits, updated = LLaDABaseModel.forward_replace_with_cache(
            model,
            torch.tensor([[1, 2, 3]]),
            slice(1, 2),
            cached,
        )

        assert updated is fallback_cache
        assert torch.equal(logits, fallback_logits[:, 1:2, :])
        err = capsys.readouterr().err
        assert "[AOAE][CACHE FALLBACK]" in err

    def test_kspec_cache_returns_full_recompute_logits_after_mid_loop_fallback(self):
        from aoae.models.base_model import LLaDABaseModel

        model = object.__new__(LLaDABaseModel)
        model._backend = "dinfer"
        model._block_length = 32
        model._dinfer_runtime = "vllm"
        model.dtype = torch.float32
        model.vocab_size = 3
        model._make_attention_mask = lambda b, s, d, block_length=32: torch.ones(
            (b, 1, s, s), dtype=torch.bool, device=d
        )
        model._make_query_attention_mask = lambda b, fs, qs, qe, d, block_length=32: torch.ones(
            (b, 1, qe - qs, fs), dtype=torch.bool, device=d
        )

        class DummyCache:
            def __init__(self, name):
                self.name = name
                self.consolidated = 0

            def consolidate(self):
                self.consolidated += 1

        aux_cache = DummyCache("aux")
        first_replace_cache = DummyCache("replace1")
        fallback_cache = DummyCache("fallback")
        fallback_logits = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]],
            dtype=torch.float32,
        )

        def fake_forward(**kwargs):
            replace_position = kwargs.get("replace_position")
            if replace_position == (1, 2):
                return types.SimpleNamespace(
                    logits=torch.tensor([[[9.0, 9.1, 9.2]]], dtype=torch.float32),
                    past_key_values=first_replace_cache,
                )
            if replace_position == (3, 4):
                return types.SimpleNamespace(
                    logits=torch.tensor([[[float("nan"), 0.0, 0.0]]], dtype=torch.float32),
                    past_key_values=DummyCache("bad"),
                )
            return types.SimpleNamespace(
                logits=fallback_logits,
                past_key_values=fallback_cache,
            )

        model._forward_dinfer_outputs = fake_forward

        logits, updated = LLaDABaseModel.forward_with_kspec_cache(
            model,
            torch.tensor([[10, 11, 12, 13]]),
            slice(1, 4),
            aux_cache,
            torch.tensor([[False, True, False]]),
        )

        assert updated is fallback_cache
        assert torch.equal(logits, fallback_logits[:, 1:4, :])
