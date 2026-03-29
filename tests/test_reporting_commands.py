import json
from pathlib import Path


def _write_eval_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_results.json").write_text(
        json.dumps(
            [
                {
                    "method": "Speculative-AOAE",
                    "accuracy": 0.6,
                    "avg_tokens_per_sec": 123.4,
                    "avg_nfe": 64,
                    "config_note": "tau_r=0.1,remask=off",
                    "cache_hit_rate": 0.5,
                    "agreement_rate": 0.7,
                    "draft_accept_rate": 0.6,
                    "reuse_mean_safe": 0.55,
                    "reuse_mean_js": 0.02,
                    "access_rate": 0.1,
                    "access_mandatory_rate": 0.05,
                    "access_optional_rate": 0.05,
                    "access_budget_utilization": 0.8,
                    "access_effective_budget": 4.0,
                    "access_next_h_precision": 0.4,
                    "access_next_h_recall": 0.5,
                    "access_next_h_f1": 0.44,
                    "access_next_h_spec_precision": 0.3,
                    "access_next_h_spec_recall": 0.6,
                    "access_next_h_spec_f1": 0.4,
                    "mean_boundary_depth": 0.0,
                    "boundary_distribution": "{}",
                    "total_samples": 10,
                    "cache_commits": 8,
                    "cache_invalidations": 2,
                }
            ]
        )
    )
    (run_dir / "eval_metadata.json").write_text(
        json.dumps(
            {
                "run_name": "demo",
                "backend": "dual",
                "mode": "speculative",
                "config_path": "configs/paper.yaml",
                "output_dir": str(run_dir),
                "eval_dataset": "openai/gsm8k",
                "routing_temperature": 0.1,
                "reuse_signal_method": "argmax_match",
                "task_type": "math",
            }
        )
    )


def test_comparison_table_main(tmp_path):
    from aoae.reporting import comparison_table_main

    run_dir = tmp_path / "outputs" / "demo"
    _write_eval_artifacts(run_dir)

    csv_path = tmp_path / "results" / "comparison.csv"
    md_path = tmp_path / "results" / "comparison.md"
    comparison_table_main(
        [
            "--glob",
            str(tmp_path / "outputs" / "**" / "eval_results.json"),
            "--csv",
            str(csv_path),
            "--md",
            str(md_path),
        ]
    )

    assert csv_path.exists()
    assert md_path.exists()
    assert "Speculative-AOAE" in csv_path.read_text()


def test_kv_summary_main(tmp_path):
    from aoae.reporting import kv_summary_main

    run_dir = tmp_path / "outputs" / "demo"
    _write_eval_artifacts(run_dir)
    (run_dir / "kv_dynamics_summary.json").write_text(
        json.dumps(
            {
                "num_records": 3,
                "mean_agreement": 0.7,
                "mean_access": 0.2,
                "mean_layer_drift_slope": 0.1,
                "mean_off_by_one_drift_ratio": 0.05,
                "mean_confident_token_drift_ratio": 0.04,
                "mean_thrash_rate_given_cached": 0.02,
            }
        )
    )

    csv_path = tmp_path / "results" / "kv.csv"
    md_path = tmp_path / "results" / "kv.md"
    kv_summary_main(
        [
            "--glob",
            str(tmp_path / "outputs" / "**" / "kv_dynamics_summary.json"),
            "--csv",
            str(csv_path),
            "--md",
            str(md_path),
        ]
    )

    assert csv_path.exists()
    assert md_path.exists()
    assert "mean_thrash_rate_given_cached" in csv_path.read_text()
