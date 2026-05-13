"""Integration tests for the POC 1 tau_r sweep pipeline.

Validates that the sweep script:
  - Iterates over tau_r values correctly
  - Produces CSV, JSON, and markdown summary files
  - Includes routing entropy in outputs
  - Handles the hard-routing baseline config
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

from aoae.evaluate import EvalResult


def _fake_speculative_result(tau_r: float = 0.1) -> EvalResult:
    """Produce a deterministic fake result that varies with tau_r."""
    acc = max(0.3, 0.8 - tau_r)
    tps = max(10.0, 200.0 * (1.0 - tau_r))
    return EvalResult(
        method="Speculative-AOAE",
        accuracy=round(acc, 4),
        total_samples=10,
        correct_samples=int(round(acc * 10)),
        avg_nfe=128.0,
        avg_tokens_per_sec=round(tps, 2),
        avg_gen_time_sec=1.0,
        config_note=f"tau_r={tau_r},tau_pi=1.0,reuse=argmax_match,pc=off,remask=off",
        cache_hit_rate=max(0.1, 1.0 - tau_r * 2),
        agreement_rate=max(0.2, 1.0 - tau_r),
        draft_accept_rate=0.7,
        reuse_mean_safe=0.6,
        reuse_mean_js=0.02,
        access_next_h_f1=0.4,
        access_next_h_spec_f1=0.35,
        routing_entropy=tau_r * 4.0,
        max_routing_entropy=4.85,
    )


def _make_base_cfg(tmp_path: Path) -> Path:
    cfg = {
        "base_model": {
            "name_or_path": "inclusionAI/LLaDA2.1-mini",
            "backend": "dual",
            "routing_temperature": 0.01,
            "mask_token_id": 156895,
        },
        "soft_mask": {"top_k": 5, "omega_s_init": 0.8, "omega_a_init": 1.0, "omega_b_init": 2.0},
        "policy": {"d_model": 128, "n_layers": 1, "n_heads": 4, "dropout": 0.0},
        "prism": {"hidden_dim": 256, "threshold": 0.5, "train_samples": 10, "epochs": 1, "lr": 1e-3, "batch_size": 4},
        "cache": {"enabled": True},
        "grpo": {"enabled": False, "group_size": 2, "clip_eps": 0.2, "alpha": 1.0, "beta": 0.1,
                 "access_reward_weight": 0.0, "lr": 3e-4, "weight_decay": 0.01, "epochs": 1,
                 "max_steps": 10, "batch_size": 1, "grad_accum_steps": 1, "warmup_steps": 0,
                 "max_grad_norm": 1.0, "policy_temperature": 1.0},
        "inference": {
            "steps": 8, "gen_length": 16, "temperature": 0.0, "fallback_unmask": True,
            "disable_remask": False,
            "reuse_signal": {"method": "argmax_match", "threshold": 0.0, "top_k": 4,
                             "min_overlap": 1, "min_streak": 2},
            "block_length": 32, "compose_gamma": 0.5,
        },
        "data": {"eval_dataset": "openai/gsm8k", "eval_split": "test", "eval_max_samples": 10},
        "evaluation": {"task_type": "math"},
        "logging": {"run_name": "test_sweep", "output_dir": str(tmp_path / "run"), "use_wandb": False},
        "hardware": {"seed": 42, "bf16": True, "tp_size": 1},
    }
    cfg_path = tmp_path / "base_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


class TestTauSweep:
    def test_sweep_produces_all_outputs(self, monkeypatch, tmp_path):
        """Full sweep with 3 tau_r values produces JSON, CSV, and markdown."""
        import aoae.paper as mod

        captured_tau_values = []

        def mock_eval_main(cfg, **kwargs):
            tau_r = cfg["base_model"]["routing_temperature"]
            captured_tau_values.append(tau_r)
            return [_fake_speculative_result(tau_r)]

        monkeypatch.setattr(mod, "eval_main", mock_eval_main)
        cfg_path = _make_base_cfg(tmp_path)
        out_dir = tmp_path / "sweep_out"

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.01,0.1,0.5",
            "--output_root", str(out_dir),
        ])
        mod.tau_sweep_main()

        assert captured_tau_values == [0.01, 0.1, 0.5]
        assert (out_dir / "tau_sweep_summary.json").exists()
        assert (out_dir / "tau_sweep_summary.csv").exists()
        assert (out_dir / "tau_sweep_summary.md").exists()

        with (out_dir / "tau_sweep_summary.json").open() as f:
            rows = json.load(f)
        assert len(rows) == 3
        assert float(rows[0]["tau_r"]) == pytest.approx(0.01)
        assert float(rows[2]["tau_r"]) == pytest.approx(0.5)

    def test_sweep_includes_routing_entropy(self, monkeypatch, tmp_path):
        """Routing entropy fields are present in sweep output."""
        import aoae.paper as mod

        monkeypatch.setattr(mod, "eval_main",
                            lambda cfg, **kw: [_fake_speculative_result(cfg["base_model"]["routing_temperature"])])
        cfg_path = _make_base_cfg(tmp_path)
        out_dir = tmp_path / "ent_out"

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.1",
            "--output_root", str(out_dir),
        ])
        mod.tau_sweep_main()

        with (out_dir / "tau_sweep_summary.json").open() as f:
            rows = json.load(f)
        assert "routing_entropy" in rows[0]
        assert "max_routing_entropy" in rows[0]
        assert float(rows[0]["routing_entropy"]) > 0

    def test_remask_disabled_by_default(self, monkeypatch, tmp_path):
        """Explicit AOAE sweeps should disable remasking unless requested."""
        import aoae.paper as mod

        captured_cfgs = []

        def mock_eval(cfg, **kw):
            captured_cfgs.append(cfg)
            return [_fake_speculative_result(cfg["base_model"]["routing_temperature"])]

        monkeypatch.setattr(mod, "eval_main", mock_eval)
        cfg_path = _make_base_cfg(tmp_path)
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["inference"]["speculative_schedule"] = "aoae"
        cfg_path.write_text(yaml.safe_dump(cfg))

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.1",
            "--output_root", str(tmp_path / "out"),
        ])
        mod.tau_sweep_main()
        assert captured_cfgs[0]["inference"]["disable_remask"] is True

    def test_remask_enabled_flag(self, monkeypatch, tmp_path):
        """With --enable_remask, AOAE sweeps should keep remasking on."""
        import aoae.paper as mod

        captured_cfgs = []

        def mock_eval(cfg, **kw):
            captured_cfgs.append(cfg)
            return [_fake_speculative_result(cfg["base_model"]["routing_temperature"])]

        monkeypatch.setattr(mod, "eval_main", mock_eval)
        cfg_path = _make_base_cfg(tmp_path)
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["inference"]["speculative_schedule"] = "aoae"
        cfg_path.write_text(yaml.safe_dump(cfg))

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.1",
            "--output_root", str(tmp_path / "out"),
            "--enable_remask",
        ])
        mod.tau_sweep_main()
        assert captured_cfgs[0]["inference"]["disable_remask"] is False

    def test_blockwise_schedule_respects_config_remask_default(self, monkeypatch, tmp_path):
        """The blockwise fidelity schedule should keep its config remask setting by default."""
        import aoae.paper as mod

        captured_cfgs = []

        def mock_eval(cfg, **kw):
            captured_cfgs.append(cfg)
            return [_fake_speculative_result(cfg["base_model"]["routing_temperature"])]

        monkeypatch.setattr(mod, "eval_main", mock_eval)
        cfg_path = _make_base_cfg(tmp_path)
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["inference"]["speculative_schedule"] = "llada21_block"
        cfg["inference"]["disable_remask"] = False
        cfg_path.write_text(yaml.safe_dump(cfg))

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.1",
            "--output_root", str(tmp_path / "out"),
        ])
        mod.tau_sweep_main()
        assert captured_cfgs[0]["inference"]["disable_remask"] is False

    def test_sweep_can_enable_prediction_saving(self, monkeypatch, tmp_path):
        import aoae.paper as mod

        captured_cfgs = []

        def mock_eval(cfg, **kw):
            captured_cfgs.append(cfg)
            return [_fake_speculative_result(cfg["base_model"]["routing_temperature"])]

        monkeypatch.setattr(mod, "eval_main", mock_eval)
        cfg_path = _make_base_cfg(tmp_path)

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.1",
            "--output_root", str(tmp_path / "out"),
            "--save_predictions",
            "--max_saved_predictions", "99",
        ])
        mod.tau_sweep_main()

        ev_cfg = captured_cfgs[0]["evaluation"]
        assert ev_cfg["save_predictions"] is True
        assert ev_cfg["max_saved_predictions"] == 50

    def test_missing_schedule_defaults_to_blockwise_without_checkpoint(self, monkeypatch, tmp_path):
        """Training-free tau sweeps should default to the official blockwise schedule."""
        import aoae.paper as mod

        captured_cfgs = []

        def mock_eval(cfg, **kw):
            captured_cfgs.append(cfg)
            return [_fake_speculative_result(cfg["base_model"]["routing_temperature"])]

        monkeypatch.setattr(mod, "eval_main", mock_eval)
        cfg_path = _make_base_cfg(tmp_path)

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.1",
            "--output_root", str(tmp_path / "out"),
        ])
        mod.tau_sweep_main()

        inf_cfg = captured_cfgs[0]["inference"]
        assert inf_cfg["speculative_schedule"] == "llada21_block"
        assert inf_cfg["llada21_official"]["use_block_diffusion"] is True

    def test_pareto_data_monotonicity(self, monkeypatch, tmp_path):
        """With our fake data, accuracy should decrease and TPS should decrease as tau_r grows."""
        import aoae.paper as mod

        monkeypatch.setattr(mod, "eval_main",
                            lambda cfg, **kw: [_fake_speculative_result(cfg["base_model"]["routing_temperature"])])
        cfg_path = _make_base_cfg(tmp_path)
        out_dir = tmp_path / "mono_out"

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.01,0.1,0.5",
            "--output_root", str(out_dir),
        ])
        mod.tau_sweep_main()

        with (out_dir / "tau_sweep_summary.json").open() as f:
            rows = json.load(f)
        accuracies = [float(r["accuracy"]) for r in rows]
        tps_vals = [float(r["tps"]) for r in rows]
        assert accuracies == sorted(accuracies, reverse=True)
        assert tps_vals == sorted(tps_vals, reverse=True)

    def test_failed_dual_preload_is_closed_before_fallback(self, monkeypatch, tmp_path):
        import aoae.paper as mod

        closed = {"count": 0}

        class FailingDualModel:
            def __init__(self, cfg):
                del cfg

            def to(self, device):
                del device
                raise RuntimeError("synthetic preload OOM")

            def close(self):
                closed["count"] += 1

        monkeypatch.setattr(mod, "eval_main",
                            lambda cfg, **kw: [_fake_speculative_result(cfg["base_model"]["routing_temperature"])])
        monkeypatch.setattr(
            __import__("aoae.models.dual_model", fromlist=["DualModelWrapper"]),
            "DualModelWrapper",
            FailingDualModel,
        )
        cfg_path = _make_base_cfg(tmp_path)
        out_dir = tmp_path / "fallback_out"

        monkeypatch.setattr(sys, "argv", [
            "tau-sweep",
            "--config", str(cfg_path),
            "--tau_r_values", "0.01",
            "--output_root", str(out_dir),
        ])
        mod.tau_sweep_main()

        assert closed["count"] == 1
