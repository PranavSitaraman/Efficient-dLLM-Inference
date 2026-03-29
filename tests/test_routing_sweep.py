import json
import sys
from pathlib import Path

import yaml

from aoae.evaluate import EvalResult


def _make_cfg(path: Path, *, backend: str = "hf") -> Path:
    cfg = {
        "base_model": {
            "name_or_path": "inclusionAI/LLaDA2.1-mini",
            "backend": backend,
            "routing_temperature": 0.01,
        },
        "inference": {"disable_remask": False},
        "data": {"eval_dataset": "openai/gsm8k", "eval_split": "test"},
        "logging": {"run_name": "routing_test", "output_dir": str(path.parent / "run")},
    }
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_routing_sweep_default_summary_method_matches_llada21_configs(monkeypatch, tmp_path):
    import aoae.paper as mod

    hard_cfg = _make_cfg(tmp_path / "hard.yaml")
    soft_cfg = _make_cfg(tmp_path / "soft.yaml")

    def fake_eval_main(cfg, **kwargs):
        del kwargs
        tau_r = float(cfg.get("base_model", {}).get("routing_temperature", 0.0))
        if tau_r == 0.0:
            return [
                EvalResult(
                    method="llada21_speed_mode",
                    accuracy=0.48,
                    total_samples=10,
                    correct_samples=5,
                    avg_nfe=64,
                    avg_tokens_per_sec=147.1,
                    avg_gen_time_sec=1.0,
                    config_note="llada21_speed_mode,remask=off",
                ),
                EvalResult(
                    method="llada21_quality_mode",
                    accuracy=0.55,
                    total_samples=10,
                    correct_samples=6,
                    avg_nfe=64,
                    avg_tokens_per_sec=135.1,
                    avg_gen_time_sec=1.0,
                    config_note="llada21_quality_mode,remask=off",
                ),
            ]
        return [
            EvalResult(
                method="llada21_quality_mode",
                accuracy=0.40,
                total_samples=10,
                correct_samples=4,
                avg_nfe=64,
                avg_tokens_per_sec=120.0,
                avg_gen_time_sec=1.0,
                config_note=f"llada21_quality_mode,tau_r={tau_r},remask=off",
            )
        ]

    monkeypatch.setattr(mod, "eval_main", fake_eval_main)
    monkeypatch.setattr(mod, "_plot_routing_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_load_eval_dataset", lambda dc: [])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "routing-sweep",
            "--hard_config",
            str(hard_cfg),
            "--soft_config",
            str(soft_cfg),
            "--tau_r_values",
            "0.01",
            "--output_root",
            str(tmp_path / "out"),
        ],
    )

    mod.routing_sweep_main()

    with (tmp_path / "out" / "routing_sweep_summary.json").open() as f:
        rows = json.load(f)

    assert rows[0]["method"] == "llada21_quality_mode"
    assert rows[1]["method"] == "llada21_quality_mode"


def test_routing_sweep_copies_hard_baseline_methods_to_soft_config(monkeypatch, tmp_path):
    import aoae.paper as mod

    hard_cfg = _make_cfg(tmp_path / "hard.yaml")
    soft_cfg = _make_cfg(tmp_path / "soft.yaml", backend="soft_moe")
    hard_payload = yaml.safe_load(hard_cfg.read_text())
    hard_payload.setdefault("evaluation", {})["baseline_methods"] = [
        "llada21_speed_mode",
        "llada21_quality_mode",
    ]
    hard_cfg.write_text(yaml.safe_dump(hard_payload))

    captured_soft_baseline_methods = []

    def fake_eval_main(cfg, **kwargs):
        del kwargs
        if cfg["base_model"]["backend"] == "soft_moe":
            captured_soft_baseline_methods.append(
                list(cfg.get("evaluation", {}).get("baseline_methods", []))
            )
        tau_r = float(cfg.get("base_model", {}).get("routing_temperature", 0.0))
        method = "llada21_quality_mode"
        return [
            EvalResult(
                method=method,
                accuracy=0.5,
                total_samples=10,
                correct_samples=5,
                avg_nfe=64,
                avg_tokens_per_sec=120.0 + tau_r,
                avg_gen_time_sec=1.0,
                config_note=f"{method},tau_r={tau_r},remask=off",
            )
        ]

    class FakeSoftBaseModel:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tokenizer = object()

        def to(self, device):
            del device
            return self

        def close(self):
            return None

    monkeypatch.setattr(mod, "eval_main", fake_eval_main)
    monkeypatch.setattr(mod, "_plot_routing_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_load_eval_dataset", lambda dc: [])
    monkeypatch.setattr(mod, "LLaDABaseModel", FakeSoftBaseModel, raising=False)
    monkeypatch.setattr(
        __import__("aoae.models.base_model", fromlist=["LLaDABaseModel"]),
        "LLaDABaseModel",
        FakeSoftBaseModel,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "routing-sweep",
            "--hard_config",
            str(hard_cfg),
            "--soft_config",
            str(soft_cfg),
            "--tau_r_values",
            "0.01",
            "--output_root",
            str(tmp_path / "out"),
        ],
    )

    mod.routing_sweep_main()

    assert captured_soft_baseline_methods == [[
        "llada21_speed_mode",
        "llada21_quality_mode",
    ]]


def test_routing_sweep_closes_failed_soft_preload_before_fallback(monkeypatch, tmp_path):
    import aoae.paper as mod
    import aoae.models.base_model as base_model_mod

    hard_cfg = _make_cfg(tmp_path / "hard.yaml")
    soft_cfg = _make_cfg(tmp_path / "soft.yaml", backend="soft_moe")
    closed = {"count": 0}

    def fake_eval_main(cfg, **kwargs):
        del kwargs
        tau_r = float(cfg.get("base_model", {}).get("routing_temperature", 0.0))
        method = "llada21_quality_mode"
        if tau_r == 0.0:
            return [
                EvalResult(
                    method=method,
                    accuracy=0.6,
                    total_samples=10,
                    correct_samples=6,
                    avg_nfe=64,
                    avg_tokens_per_sec=120.0,
                    avg_gen_time_sec=1.0,
                    config_note=f"{method},remask=off",
                )
            ]
        return [
            EvalResult(
                method=method,
                accuracy=0.5,
                total_samples=10,
                correct_samples=5,
                avg_nfe=64,
                avg_tokens_per_sec=110.0,
                avg_gen_time_sec=1.0,
                config_note=f"{method},tau_r={tau_r},remask=off",
            )
        ]

    class FailingSoftBaseModel:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tokenizer = object()

        def to(self, device):
            del device
            raise RuntimeError("synthetic preload OOM")

        def close(self):
            closed["count"] += 1

    monkeypatch.setattr(mod, "eval_main", fake_eval_main)
    monkeypatch.setattr(mod, "_plot_routing_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_load_eval_dataset", lambda dc: [])
    monkeypatch.setattr(base_model_mod, "LLaDABaseModel", FailingSoftBaseModel)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "routing-sweep",
            "--hard_config",
            str(hard_cfg),
            "--soft_config",
            str(soft_cfg),
            "--tau_r_values",
            "0.01",
            "--output_root",
            str(tmp_path / "out"),
        ],
    )

    mod.routing_sweep_main()

    assert closed["count"] == 1
