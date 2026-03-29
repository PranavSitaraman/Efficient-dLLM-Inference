from types import SimpleNamespace

import torch

from aoae.evaluate import EvalResult


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
