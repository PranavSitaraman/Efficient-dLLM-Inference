"""POC 2 tests: agreement signal correctness, edge cases, multi-batch,
and reuse signal sweep integration.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from aoae.agreement_signals import compute_reuse_signal
from aoae.evaluate import EvalResult


def _cfg(method="argmax_match", threshold=0.0, top_k=4, min_overlap=1, min_streak=2):
    return {
        "inference": {
            "reuse_signal": {
                "method": method,
                "threshold": threshold,
                "top_k": top_k,
                "min_overlap": min_overlap,
                "min_streak": min_streak,
            }
        }
    }


# ======================================================================
# Unit tests: agreement signal edge cases
# ======================================================================


class TestArgmaxMatchEdgeCases:
    def test_identical_logits_full_agreement(self):
        logits = torch.randn(2, 8, 50)
        safe, _, diag = compute_reuse_signal(logits, logits.clone(), _cfg("argmax_match"))
        assert (safe > 0).all()
        assert safe.shape == (2, 8)

    def test_opposite_logits_no_agreement(self):
        aux = torch.zeros(1, 4, 10)
        pri = torch.zeros(1, 4, 10)
        aux[0, :, 0] = 10.0
        pri[0, :, 1] = 10.0
        safe, _, _ = compute_reuse_signal(pri, aux, _cfg("argmax_match"))
        assert (safe == 0).all()

    def test_zero_logits_agree(self):
        logits = torch.zeros(1, 4, 10)
        safe, _, _ = compute_reuse_signal(logits, logits, _cfg("argmax_match"))
        assert (safe > 0).all()

    def test_large_batch(self):
        aux = torch.randn(16, 32, 100)
        pri = aux.clone()
        pri[:8] = torch.randn(8, 32, 100)
        safe, _, _ = compute_reuse_signal(pri, aux, _cfg("argmax_match"))
        assert safe.shape == (16, 32)
        assert (safe[8:] > 0).all()


class TestTopkOverlapEdgeCases:
    def test_identical_logits_full_overlap(self):
        logits = torch.randn(2, 6, 50)
        safe, _, _ = compute_reuse_signal(
            logits, logits.clone(),
            _cfg("topk_overlap", top_k=5, min_overlap=3),
        )
        assert (safe > 0).all()

    def test_disjoint_topk_no_overlap(self):
        aux = torch.zeros(1, 2, 20)
        pri = torch.zeros(1, 2, 20)
        aux[0, :, :5] = torch.arange(5, 0, -1).float()
        pri[0, :, 10:15] = torch.arange(5, 0, -1).float()
        safe, _, _ = compute_reuse_signal(
            pri, aux,
            _cfg("topk_overlap", top_k=5, min_overlap=1),
        )
        assert (safe == 0).all()

    def test_multi_batch(self):
        aux = torch.randn(4, 8, 30)
        pri = torch.randn(4, 8, 30)
        safe, _, diag = compute_reuse_signal(
            pri, aux,
            _cfg("topk_overlap", top_k=5, min_overlap=1),
        )
        assert safe.shape == (4, 8)


class TestMinConfidenceEdgeCases:
    def test_high_confidence_agreement(self):
        aux = torch.zeros(1, 3, 10)
        pri = torch.zeros(1, 3, 10)
        aux[0, :, 0] = 100.0
        pri[0, :, 0] = 100.0
        safe, _, _ = compute_reuse_signal(
            pri, aux, _cfg("min_confidence", threshold=0.9),
        )
        assert (safe > 0).all()

    def test_low_confidence_rejected(self):
        aux = torch.ones(1, 3, 10) * 0.1
        pri = torch.ones(1, 3, 10) * 0.1
        safe, _, _ = compute_reuse_signal(
            pri, aux, _cfg("min_confidence", threshold=0.9),
        )
        assert (safe == 0).all()


class TestMinMarginEdgeCases:
    def test_large_margin_accepted(self):
        aux = torch.zeros(1, 2, 10)
        pri = torch.zeros(1, 2, 10)
        aux[0, :, 0] = 10.0
        aux[0, :, 1] = 0.0
        pri[0, :, 0] = 10.0
        pri[0, :, 1] = 0.0
        safe, _, _ = compute_reuse_signal(
            pri, aux, _cfg("min_margin", threshold=0.1),
        )
        assert (safe > 0).all()

    def test_tiny_margin_rejected(self):
        aux = torch.zeros(1, 2, 10)
        pri = torch.zeros(1, 2, 10)
        aux[0, :, 0] = 1.01
        aux[0, :, 1] = 1.0
        pri[0, :, 0] = 1.01
        pri[0, :, 1] = 1.0
        safe, _, _ = compute_reuse_signal(
            pri, aux, _cfg("min_margin", threshold=0.5),
        )
        assert (safe == 0).all()


class TestJSDivergenceEdgeCases:
    def test_identical_distributions_zero_jsd(self):
        logits = torch.randn(2, 4, 20)
        safe, _, diag = compute_reuse_signal(
            logits, logits.clone(), _cfg("js_divergence", threshold=0.01),
        )
        assert (safe > 0).all()
        assert diag["mean_js_divergence"] < 1e-5

    def test_very_different_distributions(self):
        aux = torch.zeros(1, 2, 10)
        pri = torch.zeros(1, 2, 10)
        aux[0, :, 0] = 100.0
        pri[0, :, 9] = 100.0
        safe, _, diag = compute_reuse_signal(
            pri, aux, _cfg("js_divergence", threshold=0.01),
        )
        assert (safe == 0).all()

    def test_multi_batch_jsd(self):
        aux = torch.randn(8, 16, 50)
        safe, _, _ = compute_reuse_signal(
            aux, aux.clone(), _cfg("js_divergence", threshold=0.1),
        )
        assert safe.shape == (8, 16)
        assert (safe > 0).all()


class TestTemporalConfidenceEdgeCases:
    def test_first_call_no_streak(self):
        aux = torch.zeros(1, 3, 10)
        pri = torch.zeros(1, 3, 10)
        aux[0, :, 0] = 10.0
        pri[0, :, 0] = 10.0
        state = {}
        safe, state, _ = compute_reuse_signal(
            pri, aux, _cfg("temporal_confidence", threshold=0.5, min_streak=2),
            state=state,
        )
        assert (safe == 0).all()

    def test_streak_builds_over_calls(self):
        aux = torch.zeros(1, 3, 10)
        pri = torch.zeros(1, 3, 10)
        aux[0, :, 0] = 10.0
        pri[0, :, 0] = 10.0
        state = {}
        _, state, _ = compute_reuse_signal(
            pri, aux, _cfg("temporal_confidence", threshold=0.5, min_streak=2),
            state=state,
        )
        safe, state, _ = compute_reuse_signal(
            pri, aux, _cfg("temporal_confidence", threshold=0.5, min_streak=2),
            state=state,
        )
        assert (safe > 0).all()

    def test_streak_resets_on_disagreement(self):
        state = {}
        aux1 = torch.zeros(1, 2, 10)
        pri1 = torch.zeros(1, 2, 10)
        aux1[0, :, 0] = 10.0
        pri1[0, :, 0] = 10.0
        cfg_tc = _cfg("temporal_confidence", threshold=0.5, min_streak=2)
        _, state, _ = compute_reuse_signal(pri1, aux1, cfg_tc, state=state)
        _, state, _ = compute_reuse_signal(pri1, aux1, cfg_tc, state=state)
        # Break agreement
        aux2 = torch.zeros(1, 2, 10)
        pri2 = torch.zeros(1, 2, 10)
        aux2[0, :, 0] = 10.0
        pri2[0, :, 5] = 10.0
        safe, state, _ = compute_reuse_signal(pri2, aux2, cfg_tc, state=state)
        assert (safe == 0).all()


# ======================================================================
# Integration test: reuse signal sweep
# ======================================================================


def _fake_reuse_result(method="argmax_match", threshold=0.0):
    hit_rate = max(0.1, 0.8 - abs(threshold) * 0.3)
    return EvalResult(
        method="Speculative-AOAE",
        accuracy=max(0.3, 0.75 - abs(threshold) * 0.1),
        total_samples=10,
        correct_samples=7,
        avg_nfe=128.0,
        avg_tokens_per_sec=100.0 + hit_rate * 50,
        avg_gen_time_sec=1.0,
        config_note=f"tau_r=0.1,tau_pi=1.0,reuse={method},pc=off",
        cache_hit_rate=hit_rate,
        agreement_rate=0.8,
        draft_accept_rate=0.7,
        reuse_mean_safe=0.6,
        reuse_mean_js=max(threshold, 0.0),
        access_next_h_f1=0.4,
        access_next_h_spec_f1=0.35,
    )


def _write_fake_kv_summary(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kv_dynamics_summary.json").write_text(
        json.dumps(
            {
                "layer_drift_measure": "exact_kv",
                "exact_kv_drift_steps": 3,
                "hidden_state_proxy_steps": 0,
                "mean_layer_drift_slope": 0.125,
                "mean_off_by_one_drift_ratio": 0.25,
                "mean_age_drift": {
                    "age0": 0.8,
                    "age1": 0.5,
                    "age2p": 2.0,
                },
                "per_layer_drift": [
                    {"layer_idx": 0, "mean_drift": 0.1},
                    {"layer_idx": 1, "mean_drift": 0.2},
                ],
            }
        )
    )


class TestReuseSweepIntegration:
    def test_sweep_produces_outputs_and_baselines(self, monkeypatch, tmp_path):
        import aoae.paper as mod

        def mock_eval(cfg, **kwargs):
            _write_fake_kv_summary(Path(cfg["logging"]["output_dir"]))
            return [_fake_reuse_result()]

        monkeypatch.setattr(mod, "eval_main", mock_eval)

        cfg = {
            "logging": {"run_name": "poc2_smoke", "output_dir": str(tmp_path / "run")},
            "inference": {"reuse_signal": {"grid": {"argmax_match": [0.0], "js_divergence": [0.05]}}},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        monkeypatch.setattr(sys, "argv", [
            "reuse-sweep",
            "--config", str(cfg_path),
            "--output_root", str(tmp_path / "out"),
        ])
        mod.reuse_signal_sweep_main()

        assert (tmp_path / "out" / "reuse_signal_sweep_full.json").exists()
        assert (tmp_path / "out" / "best_method_by_constraint.json").exists()
        assert (tmp_path / "out" / "reuse_signal_kv_variant_summary.json").exists()

        with (tmp_path / "out" / "reuse_signal_sweep_full.json").open() as f:
            rows = json.load(f)
        with (tmp_path / "out" / "reuse_signal_kv_variant_summary.json").open() as f:
            kv_rows = json.load(f)
        methods = [r["reuse_signal_method"] for r in rows]
        assert "no_reuse" in methods
        assert "oracle_reuse" in methods
        assert "argmax_match" in methods
        assert "js_divergence" in methods
        assert all(r["kv_drift_measure"] == "exact_kv" for r in rows)
        assert all("L0:0.1000" in r["per_layer_drift_preview"] for r in rows)
        assert all(float(r["mean_off_by_one_drift_ratio"]) == pytest.approx(0.25) for r in rows)
        assert any(r["variant"] == "argmax_match" for r in kv_rows)
        assert all(float(r["drift_delta"]) >= 0.0 for r in kv_rows)

    def test_missing_schedule_defaults_to_blockwise_without_checkpoint(self, monkeypatch, tmp_path):
        import aoae.paper as mod

        captured_cfgs = []

        def mock_eval(cfg, **kwargs):
            captured_cfgs.append(cfg)
            return [_fake_reuse_result()]

        monkeypatch.setattr(mod, "eval_main", mock_eval)

        cfg = {
            "logging": {"run_name": "poc2_sched", "output_dir": str(tmp_path / "run")},
            "inference": {"reuse_signal": {"grid": {"argmax_match": [0.0]}}},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        monkeypatch.setattr(sys, "argv", [
            "reuse-sweep",
            "--config", str(cfg_path),
            "--output_root", str(tmp_path / "out"),
        ])
        mod.reuse_signal_sweep_main()

        inf_cfg = captured_cfgs[0]["inference"]
        assert inf_cfg["speculative_schedule"] == "llada21_block"
        assert inf_cfg["llada21_official"]["use_block_diffusion"] is True

    def test_decision_table_has_constraints(self, monkeypatch, tmp_path):
        import aoae.paper as mod

        monkeypatch.setattr(mod, "eval_main",
                            lambda cfg, **kw: [_fake_reuse_result()])
        cfg = {
            "logging": {"run_name": "dt_test", "output_dir": str(tmp_path / "run")},
            "inference": {"reuse_signal": {"grid": {"argmax_match": [0.0]}}},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        monkeypatch.setattr(sys, "argv", [
            "reuse-sweep",
            "--config", str(cfg_path),
            "--output_root", str(tmp_path / "out"),
        ])
        mod.reuse_signal_sweep_main()

        with (tmp_path / "out" / "best_method_by_constraint.json").open() as f:
            decisions = json.load(f)
        constraints = [d["constraint"] for d in decisions]
        assert "acc_drop<=0.01" in constraints
        assert "acc_drop<=0.02" in constraints
        assert "acc_drop<=0.05" in constraints

    def test_reuse_sweep_can_enable_prediction_saving(self, monkeypatch, tmp_path):
        import aoae.paper as mod

        captured_cfgs = []

        def mock_eval(cfg, **kwargs):
            captured_cfgs.append(cfg)
            return [_fake_reuse_result()]

        monkeypatch.setattr(mod, "eval_main", mock_eval)

        cfg = {
            "logging": {"run_name": "preds", "output_dir": str(tmp_path / "run")},
            "inference": {"reuse_signal": {"grid": {"argmax_match": [0.0]}}},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        monkeypatch.setattr(sys, "argv", [
            "reuse-sweep",
            "--config", str(cfg_path),
            "--output_root", str(tmp_path / "out"),
            "--save_predictions",
            "--max_saved_predictions", "75",
        ])
        mod.reuse_signal_sweep_main()

        ev_cfg = captured_cfgs[0]["evaluation"]
        assert ev_cfg["save_predictions"] is True
        assert ev_cfg["max_saved_predictions"] == 50
