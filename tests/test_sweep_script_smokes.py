import sys
from pathlib import Path

import yaml

from aoae.evaluate import EvalResult


def _fake_result() -> EvalResult:
    return EvalResult(
        method="Speculative-AOAE",
        accuracy=0.6,
        total_samples=10,
        correct_samples=6,
        avg_nfe=128.0,
        avg_tokens_per_sec=100.0,
        avg_gen_time_sec=1.0,
        config_note="tau_pi=1.0,remask=off",
        cache_hit_rate=0.5,
        cache_commits=10,
        cache_invalidations=2,
        agreement_rate=0.8,
        draft_accept_rate=0.7,
        reuse_mean_safe=0.75,
        reuse_mean_js=0.02,
        access_next_h_f1=0.4,
        access_next_h_spec_f1=0.35,
    )


def test_reuse_signal_sweep_smoke(monkeypatch, tmp_path):
    import scripts.run_reuse_signal_sweep as mod

    cfg = {
        "logging": {"run_name": "smoke", "output_dir": str(tmp_path / "run")},
        "inference": {"reuse_signal": {"grid": {"argmax_match": [0.0], "js_divergence": [0.05]}}},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(mod, "eval_main", lambda *args, **kwargs: [_fake_result()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_reuse_signal_sweep.py",
            "--config",
            str(cfg_path),
            "--output_root",
            str(tmp_path / "out"),
        ],
    )
    mod.main()
    assert (tmp_path / "out" / "reuse_signal_sweep_full.json").exists()
    assert (tmp_path / "out" / "best_method_by_constraint.json").exists()


def test_ablation_matrix_smoke(monkeypatch, tmp_path):
    import scripts.run_ablation_matrix as mod

    cfg = {
        "logging": {"run_name": "smoke", "output_dir": str(tmp_path / "run")},
        "inference": {},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(mod, "eval_main", lambda *args, **kwargs: [_fake_result()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation_matrix.py",
            "--config",
            str(cfg_path),
            "--output_root",
            str(tmp_path / "abl"),
            "--matrix_json",
            str(_write_matrix(tmp_path)),
        ],
    )
    mod.main()
    assert (tmp_path / "abl" / "ablation_matrix_results.json").exists()


def _write_matrix(tmp_path: Path) -> Path:
    p = tmp_path / "matrix.json"
    p.write_text('[{"name":"base","overrides":{}},{"name":"noremask","overrides":{"inference.disable_remask":true}}]')
    return p

