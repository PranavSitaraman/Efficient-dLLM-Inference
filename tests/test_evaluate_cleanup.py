from types import SimpleNamespace

import torch

from aoae.evaluate import EvalResult
from aoae.checkpoints import GRPO_TRAIN_CONTRACT_VERSION, build_grpo_config_fingerprint


def _base_cfg(tmp_path):
    return {
        "base_model": {
            "name_or_path": "inclusionAI/LLaDA2.1-mini",
            "backend": "hf",
            "mask_token_id": 123,
        },
        "data": {
            "eval_dataset": "openai/gsm8k",
            "eval_split": "test",
            "eval_max_samples": 1,
        },
        "evaluation": {"task_type": "math"},
        "inference": {"steps": 4},
        "logging": {"output_dir": str(tmp_path / "out")},
        "hardware": {"seed": 0, "deterministic": False},
    }


def test_main_closes_owned_standard_base_model(monkeypatch, tmp_path):
    import aoae.evaluate as mod

    closed = {"value": False}

    class FakeBaseModel:
        def __init__(self, cfg):
            del cfg
            self.tokenizer = object()

        def to(self, device):
            del device
            return self

        def close(self):
            closed["value"] = True

    monkeypatch.setattr(mod, "_load_eval_dataset", lambda dc: [{"question": "q", "answer": "a"}])
    monkeypatch.setattr(mod, "_run_selected_baselines", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_append_manifest", lambda metadata, results: "manifest.jsonl")
    monkeypatch.setattr(mod, "_save_kv_dynamics_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_save_eval_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "LLaDABaseModel", FakeBaseModel)

    mod.main(_base_cfg(tmp_path), mode="standard", skip_baselines=True)

    assert closed["value"] is True


def test_main_closes_owned_standard_base_model_on_exception(monkeypatch, tmp_path):
    import aoae.evaluate as mod

    closed = {"value": False}

    class FakeBaseModel:
        def __init__(self, cfg):
            del cfg
            self.tokenizer = object()

        def to(self, device):
            del device
            return self

        def close(self):
            closed["value"] = True

    monkeypatch.setattr(mod, "_load_eval_dataset", lambda dc: [{"question": "q", "answer": "a"}])
    monkeypatch.setattr(mod, "_run_selected_baselines", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(mod, "LLaDABaseModel", FakeBaseModel)

    try:
        mod.main(_base_cfg(tmp_path), mode="standard", skip_baselines=False)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected evaluation failure")

    assert closed["value"] is True


def test_main_does_not_close_preloaded_dual_model(monkeypatch, tmp_path):
    import aoae.evaluate as mod

    closed = {"value": False}

    class FakeInnerModel:
        pass

    class FakePreloadedDualModel:
        def __init__(self):
            self.tokenizer = object()
            self._model = FakeInnerModel()

        def set_tau_r(self, tau_r):
            del tau_r

        def get_embedding_weight(self):
            raise AssertionError("should not be called for llada21_block")

        def close(self):
            closed["value"] = True

    fake_result = EvalResult(
        method="Speculative-AOAE",
        accuracy=0.1,
        total_samples=1,
        correct_samples=0,
        avg_nfe=1.0,
        avg_tokens_per_sec=1.0,
        avg_gen_time_sec=1.0,
        config_note="test",
    )

    cfg = _base_cfg(tmp_path)
    cfg["base_model"]["backend"] = "dual"
    cfg["inference"]["speculative_schedule"] = "llada21_block"

    monkeypatch.setattr(mod, "_load_eval_dataset", lambda dc: [{"question": "q", "answer": "a"}])
    monkeypatch.setattr(mod, "_run_selected_baselines", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "evaluate_speculative", lambda *args, **kwargs: fake_result)
    monkeypatch.setattr(mod, "_append_manifest", lambda metadata, results: "manifest.jsonl")
    monkeypatch.setattr(mod, "_save_kv_dynamics_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_save_eval_plots", lambda *args, **kwargs: None)

    mod.main(
        cfg,
        mode="speculative",
        skip_baselines=True,
        preloaded_dual_model=FakePreloadedDualModel(),
    )

    assert closed["value"] is False


def test_speculative_eval_points_come_from_config_sweep(tmp_path):
    import aoae.evaluate as mod

    cfg = _base_cfg(tmp_path)
    cfg["grpo"] = {"policy_temperature": 1.0}
    cfg["evaluation"]["speculative_sweep"] = {
        "enabled": True,
        "points": [
            {
                "name": "verified",
                "policy_temperature": 0.7,
                "overrides": {
                    "inference.speculative_schedule": "aoae",
                    "inference.verifier_schedule.draft_token_budget": 4,
                    "inference.primary_agree_threshold": 0.98,
                    "inference.max_unmask_fraction_per_step": 0.0625,
                },
            },
            {
                "name": "fast",
                "tau_pi": 1.5,
                "draft_token_budget": 16,
                "disable_remask": True,
            },
        ],
    }

    points = mod._build_speculative_eval_points(cfg, explicit_policy_temperatures=None)

    assert [point["name"] for point in points] == ["verified", "fast"]
    assert points[0]["policy_temperature"] == 0.7
    assert points[0]["overrides"]["inference.speculative_schedule"] == "aoae"
    assert points[0]["overrides"]["inference.verifier_schedule.draft_token_budget"] == 4
    assert points[1]["policy_temperature"] == 1.5
    assert points[1]["overrides"]["inference.verifier_schedule.draft_token_budget"] == 16
    assert points[1]["overrides"]["inference.disable_remask"] is True

    point_cfg = mod._apply_speculative_eval_point(cfg, points[0])
    assert point_cfg["_active_speculative_eval_point"] == "verified"
    assert point_cfg["inference"]["speculative_schedule"] == "aoae"
    assert point_cfg["inference"]["verifier_schedule"]["draft_token_budget"] == 4
    assert point_cfg["inference"]["primary_agree_threshold"] == 0.98


def test_block_policy_sweep_points_are_explicitly_isolated_by_generation_filter(tmp_path):
    import aoae.evaluate as mod

    cfg = _base_cfg(tmp_path)
    cfg["grpo"] = {"policy_temperature": 1.0}
    cfg["evaluation"]["generation_mode_filter"] = "block"
    cfg["evaluation"]["speculative_sweep"] = {
        "enabled": True,
        "points": [
            {
                "name": "disabled_bad_blockhead",
                "enabled": False,
                "generation_mode": "block",
                "policy_temperature": 1.0,
                "overrides": {"inference.speculative_schedule": "aoae_block_policy"},
            },
            {
                "name": "stable_block_frontier",
                "policy_temperature": 1.0,
                "overrides": {"inference.speculative_schedule": "aoae_block"},
            },
            {
                "name": "experimental_block_policy",
                "generation_mode": "any_order",
                "policy_temperature": 1.0,
                "overrides": {"inference.speculative_schedule": "aoae_block_policy"},
            },
            {
                "name": "implicit_experimental_block_policy",
                "policy_temperature": 1.0,
                "overrides": {"inference.speculative_schedule": "aoae_block_policy"},
            },
        ],
    }

    block_points = mod._build_speculative_eval_points(cfg, explicit_policy_temperatures=None)
    assert [p["name"] for p in block_points] == ["stable_block_frontier"]
    assert block_points[0]["overrides"]["inference.speculative_schedule"] == "aoae_block"

    cfg["evaluation"]["generation_mode_filter"] = "any_order"
    any_order_points = mod._build_speculative_eval_points(cfg, explicit_policy_temperatures=None)
    assert [p["name"] for p in any_order_points] == [
        "experimental_block_policy",
        "implicit_experimental_block_policy",
    ]
    assert any_order_points[0]["generation_mode"] == "any_order"
    assert any_order_points[1]["generation_mode"] == "any_order"

    point_cfg = mod._apply_speculative_eval_point(cfg, any_order_points[0])
    assert point_cfg["_active_speculative_eval_generation_mode"] == "any_order"
    assert point_cfg["inference"]["speculative_schedule"] == "aoae_block_policy"


def test_aoae_block_schedule_uses_heuristic_runner_even_with_block_policy_enabled(monkeypatch):
    import aoae.evaluate as mod
    from aoae.evaluators import EvalDecision

    calls = {"heuristic": 0}

    class FakeDual:
        device = torch.device("cpu")

        def auxiliary_forward(self, input_ids):
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)

        def primary_forward(self, input_ids):
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)

    class FakeTokenizer:
        def encode(self, *args, **kwargs):
            return torch.tensor([[7]], dtype=torch.long)

    class FakeEvaluator:
        evaluator_name = "fake"

        def evaluate(self, generated, reference, sample=None):
            return EvalDecision(True, "ok", generated, reference)

    def fake_heuristic_runner(**kwargs):
        calls["heuristic"] += 1
        prompt_ids = kwargs["prompt_ids"]
        return (
            torch.cat([prompt_ids, torch.tensor([[2]], dtype=torch.long)], dim=1),
            {
                "effective_flops": 0.25,
                "agreement_observations": 1,
                "safe_reuse_observations": 1,
            },
        )

    monkeypatch.setattr(mod, "build_prompt", lambda tokenizer, question, cfg: ("prompt", False))
    monkeypatch.setattr(mod, "build_evaluator", lambda cfg: FakeEvaluator())
    monkeypatch.setattr(
        mod,
        "_summarize_generation",
        lambda *args, **kwargs: {
            "generated_tokens": 1,
            "mask_tokens_remaining": 0,
            "truncated_generation": False,
            "generated_text": "42",
            "generation_cap": 1,
            "has_eos": False,
        },
    )
    monkeypatch.setattr(mod, "run_block_frontier_speculative_inference", fake_heuristic_runner)
    monkeypatch.setattr(
        mod,
        "_aoae_block_inference",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("trained block path hijacked aoae_block")),
    )

    cfg = {
        "base_model": {"mask_token_id": 0, "routing_temperature": 0.1},
        "data": {"max_prompt_len": 4},
        "policy": {"block_wise": {"enabled": True}},
        "inference": {
            "steps": 4,
            "gen_length": 1,
            "speculative_schedule": "aoae_block",
            "reuse_signal": {"method": "argmax_match"},
            "positional_cache": {"enabled": False},
        },
        "evaluation": {"task_type": "math"},
    }

    result = mod.evaluate_speculative(
        dual_model=FakeDual(),
        policy=object(),
        soft_mask=None,
        prism=None,
        dataset=[{"question": "q", "answer": "#### 42"}],
        tokenizer=FakeTokenizer(),
        cfg=cfg,
        max_samples=1,
    )

    assert calls["heuristic"] == 1
    assert result.accuracy == 1.0
    assert result.generation_mode == "block"


def test_lossless_verification_override_flows_through_per_point(tmp_path):
    # The Pareto frontier mixes soft and lossless operating points; the
    # per-point lossless override must mutate the dual model wrapper's
    # _lossless flag so primary_forward switches to hard routing in place.
    import aoae.evaluate as mod

    cfg = _base_cfg(tmp_path)
    cfg["base_model"]["lossless_verification"] = False
    cfg["evaluation"]["speculative_sweep"] = {
        "enabled": True,
        "points": [
            {
                "name": "soft_point",
                "policy_temperature": 1.0,
                "overrides": {"base_model.lossless_verification": False},
            },
            {
                "name": "lossless_point",
                "policy_temperature": 1.0,
                "overrides": {"base_model.lossless_verification": True},
            },
        ],
    }

    points = mod._build_speculative_eval_points(cfg, explicit_policy_temperatures=None)
    soft_cfg = mod._apply_speculative_eval_point(cfg, points[0])
    lossless_cfg = mod._apply_speculative_eval_point(cfg, points[1])
    assert soft_cfg["base_model"]["lossless_verification"] is False
    assert lossless_cfg["base_model"]["lossless_verification"] is True

    class StubDual:
        def __init__(self):
            self._lossless = False

        def set_tau_r(self, value):
            self.tau_r = value

        def set_soft_topk(self, value):
            self.soft_topk = value

    dual = StubDual()
    mod._configure_dual_model_for_eval_cfg(dual, soft_cfg)
    assert dual._lossless is False
    mod._configure_dual_model_for_eval_cfg(dual, lossless_cfg)
    assert dual._lossless is True


def test_eval_auto_checkpoint_allows_low_shaped_reward(tmp_path):
    import json
    import aoae.evaluate as mod

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ckpt = out_dir / "policy_final.pt"
    ckpt.write_text("checkpoint")
    cfg = _base_cfg(tmp_path)
    cfg["base_model"]["backend"] = "dual"
    cfg["logging"]["output_dir"] = str(out_dir)
    cfg["soft_mask"] = {"top_k": 5}
    cfg["policy"] = {"d_model": 128}
    cfg["prism"] = {"hidden_dim": 256}
    cfg["grpo"] = {"min_checkpoint_reward": 0.0}
    cfg["data"]["train_dataset"] = "demo"
    cfg["data"]["train_split"] = "train"

    metadata = {
        "stage": "grpo",
        "train_contract_version": GRPO_TRAIN_CONTRACT_VERSION,
        "config_fingerprint": build_grpo_config_fingerprint(cfg),
        "best_reward": -0.25,
    }
    (out_dir / "grpo_training_metadata.json").write_text(json.dumps(metadata))

    assert mod._resolve_valid_auto_policy_checkpoint(None, cfg) == str(ckpt)


def test_explicit_policy_temperatures_override_config_sweep(tmp_path):
    import aoae.evaluate as mod

    cfg = _base_cfg(tmp_path)
    cfg["evaluation"]["speculative_sweep"] = {
        "enabled": True,
        "points": [{"name": "quality", "policy_temperature": 0.7, "draft_token_budget": 4}],
    }

    points = mod._build_speculative_eval_points(cfg, explicit_policy_temperatures=[0.4, 1.2])

    assert [point["name"] for point in points] == ["tau_pi_0.4", "tau_pi_1.2"]
    assert [point["policy_temperature"] for point in points] == [0.4, 1.2]
    assert all(point["overrides"] == {} for point in points)


def test_mean_fraction_series_averages_step_fractions():
    import aoae.evaluate as mod

    value = mod._mean_fraction_series([
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ])

    assert value == 0.75


def test_main_reuses_preloaded_base_model_and_updates_soft_routing(monkeypatch, tmp_path):
    import aoae.evaluate as mod

    updates = {"tau_r": None, "soft_topk": None, "closed": False}

    class FakePreloadedBaseModel:
        def __init__(self):
            self.tokenizer = object()
            self.device = "cpu"

        def set_routing_temperature(self, tau_r):
            updates["tau_r"] = tau_r

        def set_soft_topk(self, soft_topk):
            updates["soft_topk"] = soft_topk

        def close(self):
            updates["closed"] = True

    monkeypatch.setattr(mod, "_load_eval_dataset", lambda dc: [{"question": "q", "answer": "a"}])
    monkeypatch.setattr(mod, "_run_selected_baselines", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_append_manifest", lambda metadata, results: "manifest.jsonl")
    monkeypatch.setattr(mod, "_save_kv_dynamics_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_save_eval_plots", lambda *args, **kwargs: None)

    cfg = _base_cfg(tmp_path)
    cfg["base_model"]["backend"] = "soft_moe"
    cfg["base_model"]["routing_temperature"] = 0.05
    cfg["base_model"]["soft_topk"] = 16

    mod.main(
        cfg,
        mode="standard",
        skip_baselines=True,
        preloaded_base_model=FakePreloadedBaseModel(),
    )

    assert updates["tau_r"] == 0.05
    assert updates["soft_topk"] == 16
    assert updates["closed"] is False


def test_runtime_managed_base_model_to_is_noop_when_target_matches(monkeypatch):
    import aoae.models.base_model as mod

    base_model = object.__new__(mod.LLaDABaseModel)
    torch.nn.Module.__init__(base_model)
    base_model._backend = "soft_moe"
    base_model._embedding_weight = torch.zeros(1, device="cpu")
    base_model.model = torch.nn.Linear(1, 1)
    base_model._resolve_embedding_weight = lambda: base_model._embedding_weight

    called = {"value": False}

    def fail_super_to(self, *args, **kwargs):
        del self, args, kwargs
        called["value"] = True
        raise AssertionError("runtime-managed backends should not recurse into nn.Module.to()")

    monkeypatch.setattr(torch.nn.Module, "to", fail_super_to)

    returned = mod.LLaDABaseModel.to(base_model, torch.device("cpu"))

    assert returned is base_model
    assert called["value"] is False


def test_runtime_managed_base_model_to_delegates_for_initial_device_move(monkeypatch):
    import aoae.models.base_model as mod

    base_model = object.__new__(mod.LLaDABaseModel)
    torch.nn.Module.__init__(base_model)
    base_model._backend = "soft_moe"
    base_model._embedding_weight = torch.zeros(1, device="cpu")
    base_model.model = torch.nn.Linear(1, 1)
    base_model._resolve_embedding_weight = lambda: base_model._embedding_weight

    called = {"value": False}

    def fake_super_to(self, *args, **kwargs):
        del args, kwargs
        called["value"] = True
        return self

    monkeypatch.setattr(torch.nn.Module, "to", fake_super_to)

    returned = mod.LLaDABaseModel.to(base_model, torch.device("cuda"))

    assert returned is base_model
    assert called["value"] is True


def test_semi_any_order_block_length_returns_moderate_default_and_honors_override():
    # The LLaDA2.1 *semi-any-order* baseline uses a moderate block size:
    # wider than the canonical block-mode default (32) so the response gets
    # multi-token bidirectional context, narrower than the full sequence so
    # the mask stays inside LLaDA2.1's training distribution.
    from aoae.evaluate import _semi_any_order_block_length

    # Default — 128 is the documented semi-any-order block size.
    assert _semi_any_order_block_length({}) == 128
    assert _semi_any_order_block_length({"data": {}}) == 128
    # Configurable via data.any_order_block_length (e.g., 64 for tighter
    # alignment with the 32-token block-trained distribution).
    assert _semi_any_order_block_length({"data": {"any_order_block_length": 64}}) == 64
    assert _semi_any_order_block_length({"data": {"any_order_block_length": 256}}) == 256
    # Returns at least 1 even with absurd input (the helper must always feed
    # ``block_smode_decode`` a usable block_length).
    assert _semi_any_order_block_length({"data": {"any_order_block_length": 0}}) >= 1
    assert _semi_any_order_block_length({"data": {"any_order_block_length": None}}) >= 1


def test_block_smode_decode_suppress_eos_keeps_position_zero_visible():
    # Regression: LLaDA2.1 any-order mode places EOS at response[0] under
    # full bidirectional attention, which then truncates the visible output
    # to length zero in the eval summariser.  ``suppress_eos=True`` masks
    # EOS out of the per-step logits before the M2T threshold check, so the
    # first response position never gets unmasked to EOS.
    import torch
    from aoae.inference import block_smode_decode

    EOS_ID = 2
    MASK_ID = 0
    NON_EOS_ID = 1

    class EOSPreferringModel:
        vocab_size = 4

        def __init__(self, eos_logit: float, non_eos_logit: float):
            self._eos_logit = eos_logit
            self._non_eos_logit = non_eos_logit
            self.tokenizer = type("T", (), {"eos_token_id": EOS_ID})()

        def forward(self, input_ids):
            B, L = input_ids.shape
            logits = torch.full((B, L, self.vocab_size), -10.0)
            logits[..., EOS_ID] = self._eos_logit
            logits[..., NON_EOS_ID] = self._non_eos_logit
            return logits

    cfg = {
        "inference": {
            "gen_length": 3,
            "block_length": 16,  # > P + L_gen so single block + full attention
            "temperature": 0.0,
        },
        "base_model": {"mask_token_id": MASK_ID},
    }
    prompt = torch.tensor([[10, 20]], dtype=torch.long)

    # Without suppression: model picks EOS, response[0] becomes EOS, the
    # visible-token count would collapse to zero in the eval summariser.
    out_no_suppress = block_smode_decode(
        EOSPreferringModel(eos_logit=10.0, non_eos_logit=2.0),
        prompt, cfg,
        tau_mask=0.5, tau_edit=0.5, max_steps_per_block=4, enable_mbe=False,
        gen_length=3,
    )
    response_no_suppress = out_no_suppress[0, prompt.shape[1]:]
    assert int(response_no_suppress[0].item()) == EOS_ID

    # With suppression: EOS is masked out of the per-step logits and the
    # next-best token wins instead.  response[0] is therefore never EOS.
    out_suppress = block_smode_decode(
        EOSPreferringModel(eos_logit=10.0, non_eos_logit=2.0),
        prompt, cfg,
        tau_mask=0.5, tau_edit=0.5, max_steps_per_block=4, enable_mbe=False,
        gen_length=3,
        suppress_eos=True,
    )
    response_suppress = out_suppress[0, prompt.shape[1]:]
    assert int(response_suppress[0].item()) != EOS_ID
    assert int(response_suppress[0].item()) == NON_EOS_ID


def test_aoae_block_inference_populates_trajectory_effective_flops():
    # ``evaluate_speculative`` raises when a trajectory is missing
    # ``effective_flops``.  This regression test asserts the block-mode
    # speculative path always reports the cost-unit fields the eval harness
    # consumes (effective_flops, primary_steps, aux_only_steps).
    import torch
    import types
    from aoae.speculative_inference import aoae_block_inference

    cfg = {
        "base_model": {"mask_token_id": 0},
        "cache": {"enabled": True, "stable_kv_cache": False, "prefix_kv_cache": False},
        "grpo": {"thrash_age_decay": 0.0},
        "inference": {
            "steps": 4,
            "gen_length": 4,
            "temperature": 0.0,
            "fallback_unmask": False,
            "disable_remask": False,
            "compose_gamma": 0.0,
            "primary_agree_threshold": 0.0,
            "block_length": 2,
            "verifier_schedule": {
                "mode": "candidate_budget",
                "draft_token_budget": 4,
                "min_draft_microsteps": 1,
                "max_draft_microsteps": 1,
                "force_first_last": True,
            },
            "verifier": {"acceptance_mode": "argmax_match"},
            "drafter": {"confidence_threshold": 0.1, "aux_compute_ratio": 0.35},
            "positional_cache": {"enabled": False},
            "reuse_signal": {"method": "argmax_match"},
        },
        "analysis": {"track_kv_dynamics": False},
    }

    class StubDual:
        @staticmethod
        def _logits(B, L):
            x = torch.zeros(B, L, 3)
            x[..., 1] = 8.0
            return x

        def auxiliary_forward_resp(self, prefix_ids, blk_slice):
            return self._logits(prefix_ids.shape[0], blk_slice.stop - blk_slice.start)

        def primary_forward_resp(self, prefix_ids, blk_slice):
            return self._logits(prefix_ids.shape[0], blk_slice.stop - blk_slice.start)

    class StubPolicy:
        def __call__(self, *args, **kwargs):
            mask_ind = args[1]
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
            return {"u_t": zeros, "r_t": zeros, "kappa_t": zeros, "q_t": zeros}

    class StubSoftMask:
        def __call__(self, resp_logits, mask_ind, step_frac, return_weighted=False):
            del step_frac
            hidden = torch.zeros(resp_logits.shape[0], resp_logits.shape[1], 1)
            confidence = torch.ones_like(mask_ind, dtype=torch.float32)
            entropy = torch.zeros_like(confidence)
            weighted = torch.zeros_like(hidden)
            if return_weighted:
                return hidden, confidence, entropy, weighted
            return hidden, confidence, entropy

    _, traj = aoae_block_inference(
        dual_model=StubDual(),
        policy=StubPolicy(),
        soft_mask_module=StubSoftMask(),
        prism_adapter=None,
        prompt_ids=torch.tensor([[1]], dtype=torch.long),
        cfg=cfg,
        collect_stats=True,
    )

    assert traj is not None
    assert traj.effective_flops is not None
    assert float(traj.effective_flops.mean().item()) >= 0.0
    assert traj.aux_compute_units is not None
    assert traj.verifier_compute_units is not None
    assert traj.baseline_compute_units is not None
    # The eval harness reads these directly into the per-row stats blob.
    assert isinstance(traj.primary_steps, int)
    assert isinstance(traj.aux_only_steps, int)
