from pathlib import Path


def test_paper_suite_runs_final_dataset_loop(monkeypatch, tmp_path):
    import aoae.paper as mod

    eval_calls = []
    invoke_calls = []

    def fake_run_eval_subprocess(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_calls.append(
            {
                "output_dir": str(out_dir),
                "config_path": kwargs["config_path"],
                "checkpoint_path": kwargs["checkpoint_path"],
                "max_samples": kwargs["max_samples"],
                "skip_baselines": kwargs["skip_baselines"],
                "save_predictions": kwargs["save_predictions"],
                "max_saved_predictions": kwargs["max_saved_predictions"],
            }
        )
        return []

    def fake_invoke(main_fn, prog, argv):
        del main_fn
        invoke_calls.append((prog, list(argv)))

    ckpt = tmp_path / "policy_best.pt"
    ckpt.write_text("checkpoint")
    monkeypatch.setattr(mod, "_run_eval_subprocess", fake_run_eval_subprocess)
    monkeypatch.setattr(mod, "_invoke_main", fake_invoke)
    monkeypatch.setattr(mod, "DEFAULT_FINAL_POLICY_CHECKPOINT", ckpt)

    out_root = tmp_path / "paper_final"
    mod.paper_suite_main(
        [
            "--config",
            "configs/paper.yaml",
            "--output_root",
            str(out_root),
            "--datasets",
            "gsm8k,humaneval",
            "--max_samples",
            "8",
            "--checkpoint",
            "auto",
            "--skip_baselines",
            "--save_predictions",
            "--max_saved_predictions",
            "12",
        ]
    )

    assert [Path(call["output_dir"]).relative_to(out_root).as_posix() for call in eval_calls] == [
        "gsm8k/trained",
        "gsm8k/notrain",
        "humaneval/trained",
        "humaneval/notrain",
        "ablations/trained",
        "ablations/notrain",
    ]
    assert [call["checkpoint_path"] for call in eval_calls] == [
        str(ckpt),
        None,
        str(ckpt),
        None,
        str(ckpt),
        None,
    ]
    assert all(call["max_samples"] == 8 for call in eval_calls)
    assert all(call["skip_baselines"] is True for call in eval_calls)
    assert all(call["save_predictions"] is True for call in eval_calls)
    assert all(call["max_saved_predictions"] == 12 for call in eval_calls)
    assert invoke_calls and invoke_calls[0][0] == "comparison-table"
    assert (out_root / "paper_suite_summary.json").exists()
    assert (out_root / "run_manifest.json").exists()
