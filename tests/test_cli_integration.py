import importlib
import subprocess
import sys
import types
from types import SimpleNamespace

import yaml

from aoae.checkpoints import resolve_policy_checkpoint
from aoae.cli import apply_eval_overrides, main


def test_apply_eval_overrides_updates_config():
    cfg = {
        "base_model": {"backend": "auto"},
        "inference": {},
        "data": {},
        "logging": {},
    }
    args = type(
        "Args",
        (),
        {
            "reuse_signal_method": "js_divergence",
            "reuse_signal_threshold": 0.05,
            "disable_remask": True,
            "track_kv_dynamics": True,
            "enable_positional_cache": True,
            "positional_cache_horizon": 4,
            "positional_cache_refresh_budget": 16,
            "eval_dataset": "openai/gsm8k",
            "eval_dataset_config": "",
            "eval_split": "test",
            "backend": "hf",
            "routing_temperature": 0.1,
            "soft_topk": 8,
            "task_type": "code",
            "code_timeout_sec": 5.0,
            "code_cpu_time_limit_sec": 3,
            "code_memory_limit_mb": 256,
            "save_predictions": True,
            "max_saved_predictions": 99,
            "run_name": "demo",
            "output_dir": "outputs/demo",
        },
    )()

    out = apply_eval_overrides(cfg, args)
    assert out["inference"]["reuse_signal"]["method"] == "js_divergence"
    assert out["inference"]["reuse_signal"]["threshold"] == 0.05
    assert out["inference"]["disable_remask"] is True
    assert out["analysis"]["track_kv_dynamics"] is True
    assert out["inference"]["positional_cache"]["enabled"] is True
    assert out["evaluation"]["task_type"] == "code"
    assert out["evaluation"]["code"]["memory_limit_mb"] == 256
    assert out["evaluation"]["max_saved_predictions"] == 50
    assert out["logging"]["output_dir"] == "outputs/demo"


def test_resolve_policy_checkpoint_prefers_final_then_steps(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    step1 = out_dir / "policy_step1.pt"
    step2 = out_dir / "policy_step12.pt"
    step1.write_text("x")
    step2.write_text("x")
    assert resolve_policy_checkpoint(None, str(out_dir)) == str(step2)

    final = out_dir / "policy_final.pt"
    final.write_text("x")
    assert resolve_policy_checkpoint(None, str(out_dir)) == str(final)


def test_cli_eval_dry_run_creates_output_dir(tmp_path):
    cfg = {
        "base_model": {"backend": "hf"},
        "data": {"eval_dataset": "openai/gsm8k", "eval_split": "test"},
        "inference": {},
        "logging": {"output_dir": str(tmp_path / "dry")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    main(["eval", "--config", str(cfg_path), "--dry_run"])
    assert (tmp_path / "dry").exists()


def test_cli_pipeline_skip_all_is_noop(tmp_path):
    cfg = {"logging": {"output_dir": str(tmp_path / "run")}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    main(
        [
            "pipeline",
            "--config",
            str(cfg_path),
            "--skip_preflight",
            "--skip_prism",
            "--skip_grpo",
            "--skip_eval",
        ]
    )


def test_cli_tau_sweep_passthrough(monkeypatch):
    calls = []

    def fake_main():
        calls.append(list(sys.argv))

    def fake_import_module(name):
        assert name == "aoae.paper"
        return types.SimpleNamespace(tau_sweep_main=lambda argv=None: fake_main())

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    main(["tau-sweep", "--config", "cfg.yaml", "--max_samples", "4"])
    assert calls == [["tau-sweep", "--config", "cfg.yaml", "--max_samples", "4"]]


def test_cli_main_cleans_up_initialized_process_group(monkeypatch):
    calls = []

    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def destroy_process_group():
            calls.append("destroy")

    fake_torch = SimpleNamespace(distributed=FakeDist)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", FakeDist)

    def fake_import_module(name):
        assert name == "aoae.paper"
        return types.SimpleNamespace(tau_sweep_main=lambda argv=None: None)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    main(["tau-sweep", "--config", "cfg.yaml"])
    assert calls == ["destroy"]


def test_cli_paper_suite_passthrough(monkeypatch):
    calls = []

    def fake_main():
        calls.append(list(sys.argv))

    def fake_import_module(name):
        assert name == "aoae.paper"
        return types.SimpleNamespace(paper_suite_main=lambda argv=None: fake_main())

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    main(["paper-suite", "--config", "cfg.yaml", "--skip_table"])
    assert calls == [["paper-suite", "--config", "cfg.yaml", "--skip_table"]]


def test_cli_auto_torchrun_for_multi_gpu_config(monkeypatch, tmp_path):
    cfg = {"hardware": {"tp_size": 2}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    calls = []

    def fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(importlib, "import_module", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    rc = main(["tau-sweep", "--config", str(cfg_path), "--max_samples", "4"])

    assert rc == 0
    assert len(calls) == 1
    cmd, env = calls[0]
    assert cmd[0].endswith("torchrun") or cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "-m" in cmd
    assert "aoae.cli" in cmd
    assert "tau-sweep" in cmd
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["NCCL_SOCKET_FAMILY"] == "AF_INET"


def test_cli_skips_auto_torchrun_inside_distributed_env(monkeypatch, tmp_path):
    cfg = {"hardware": {"tp_size": 2}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    calls = []

    def fake_main():
        calls.append(list(sys.argv))

    def fake_import_module(name):
        assert name == "aoae.paper"
        return types.SimpleNamespace(tau_sweep_main=lambda argv=None: fake_main())

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    main(["tau-sweep", "--config", str(cfg_path)])
    assert calls == [["tau-sweep", "--config", str(cfg_path)]]
