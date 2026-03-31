import sys


def test_paper_suite_invokes_all_stages(monkeypatch, tmp_path):
    import aoae.paper as mod

    calls = []

    def fake_invoke(main_fn, prog, argv):
        del main_fn
        calls.append((prog, list(argv)))

    monkeypatch.setattr(mod, "_invoke_main", fake_invoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper-suite",
            "--config",
            "configs/paper.yaml",
            "--output_root",
            str(tmp_path / "suite"),
            "--max_samples",
            "8",
            "--poc2_disable_remask",
            "--save_predictions",
            "--max_saved_predictions",
            "12",
        ],
    )

    mod.paper_suite_main()

    progs = [prog for prog, _argv in calls]
    assert progs == [
        "tau-sweep",
        "routing-sweep",
        "reuse-sweep",
        "ablations",
        "comparison-table",
        "kv-summary",
    ]

    reuse_call = next(argv for prog, argv in calls if prog == "reuse-sweep")
    tau_call = next(argv for prog, argv in calls if prog == "tau-sweep")
    routing_call = next(argv for prog, argv in calls if prog == "routing-sweep")
    assert "--disable_remask" in reuse_call
    assert "--save_predictions" in tau_call
    assert "--max_saved_predictions" in tau_call
    assert "--hard_config" in routing_call
    assert "--soft_config" in routing_call
    assert (tmp_path / "suite" / "paper_suite_summary.json").exists()
