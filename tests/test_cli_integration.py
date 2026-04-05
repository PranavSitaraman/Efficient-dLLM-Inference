import importlib
import json
import subprocess
import sys
import types
from types import SimpleNamespace

import yaml

from aoae.checkpoints import (
    GRPO_TRAIN_CONTRACT_VERSION,
    build_grpo_config_fingerprint,
    inspect_grpo_artifacts,
    inspect_grpo_resume_candidate,
    resolve_policy_checkpoint,
)
from aoae.cli import _normalize_legacy_cli_argv, apply_eval_overrides, main


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


def test_resolve_policy_checkpoint_prefers_latest_when_no_best_or_final(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    latest = out_dir / "policy_latest.pt"
    latest.write_text("x")
    step = out_dir / "policy_step25.pt"
    step.write_text("x")

    assert resolve_policy_checkpoint(None, str(out_dir)) == str(latest)


def test_inspect_grpo_artifacts_rejects_missing_metadata(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "policy_final.pt").write_text("x")

    cfg = {"grpo": {"min_checkpoint_reward": 0.0}}
    status = inspect_grpo_artifacts(str(out_dir), cfg)

    assert status["valid"] is False
    assert status["reason"] == "missing_metadata"


def test_inspect_grpo_artifacts_accepts_matching_metadata(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "policy_final.pt").write_text("x")
    cfg = {
        "base_model": {"backend": "hf"},
        "soft_mask": {"top_k": 5},
        "policy": {"d_model": 128},
        "prism": {"hidden_dim": 256},
        "grpo": {"min_checkpoint_reward": 0.0},
        "inference": {"steps": 16},
        "data": {"train_dataset": "demo", "train_split": "train"},
    }
    metadata = {
        "stage": "grpo",
        "train_contract_version": GRPO_TRAIN_CONTRACT_VERSION,
        "config_fingerprint": build_grpo_config_fingerprint(cfg),
        "best_reward": 0.1,
    }
    (out_dir / "grpo_training_metadata.json").write_text(json.dumps(metadata))

    status = inspect_grpo_artifacts(str(out_dir), cfg)

    assert status["valid"] is True
    assert status["reason"] == "ok"


def test_inspect_grpo_resume_candidate_rejects_completed_low_reward_run(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "policy_latest.pt").write_text("x")
    cfg = {
        "base_model": {"backend": "dual"},
        "soft_mask": {"top_k": 5},
        "policy": {"d_model": 128},
        "prism": {"hidden_dim": 256},
        "grpo": {"min_checkpoint_reward": 0.0, "max_steps": 500},
        "inference": {"steps": 16},
        "data": {"train_dataset": "demo", "train_split": "train"},
    }
    metadata = {
        "stage": "grpo",
        "train_contract_version": GRPO_TRAIN_CONTRACT_VERSION,
        "config_fingerprint": build_grpo_config_fingerprint(cfg),
        "best_reward": -0.25,
        "completed_steps": 500,
        "max_steps": 500,
    }
    (out_dir / "grpo_training_metadata.json").write_text(json.dumps(metadata))

    status = inspect_grpo_resume_candidate(str(out_dir), cfg)

    assert status["valid"] is False
    assert status["reason"] == "reward_below_threshold"


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


def test_cli_pipeline_does_not_auto_torchrun_for_multi_gpu_config(monkeypatch, tmp_path):
    cfg = {"hardware": {"tp_size": 2}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    calls = []

    def fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    rc = main(
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

    assert rc is None
    assert calls == []


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


def test_cli_pipeline_delegates_to_canonical_subcommands(monkeypatch, tmp_path):
    import aoae.cli as mod

    cfg = {
        "hardware": {"tp_size": 1},
        "logging": {"output_dir": str(tmp_path / "out")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    calls = []

    def fake_run_preflight(config_path, strict_moe=False):
        calls.append(("preflight", config_path, strict_moe))
        return {"ok": True}

    def fake_main(argv=None):
        calls.append(tuple(argv))
        return 0

    status_iter = iter(
        [
            {"valid": False, "reason": "missing_checkpoint", "checkpoint_path": None},
            {"valid": True, "reason": "ok", "checkpoint_path": f"{tmp_path / 'out'}/policy_final.pt"},
        ]
    )

    monkeypatch.setattr(mod, "run_preflight", fake_run_preflight)
    monkeypatch.setattr(mod, "inspect_grpo_artifacts", lambda *args, **kwargs: next(status_iter))
    monkeypatch.setattr(mod, "inspect_grpo_resume_candidate", lambda *args, **kwargs: {"valid": True, "reason": "ok"})
    monkeypatch.setattr(mod, "resolve_policy_checkpoint", lambda checkpoint, output_dir: checkpoint or f"{output_dir}/policy_final.pt")
    monkeypatch.setattr(mod, "main", fake_main)

    mod.run_pipeline_command(
        SimpleNamespace(
            config=str(cfg_path),
            resume="auto",
            checkpoint=None,
            max_samples=7,
            mode="speculative",
            policy_temperatures="0.8,1.0",
            skip_preflight=False,
            skip_prism=False,
            skip_grpo=False,
            skip_eval=False,
            skip_baselines=True,
            strict_moe=True,
        )
    )

    assert calls[0] == ("preflight", str(cfg_path), True)
    assert calls[1] == ("train", "--config", str(cfg_path), "--stage", "prism")
    assert calls[2] == ("train", "--config", str(cfg_path), "--stage", "grpo", "--resume", "auto")
    assert calls[3] == (
        "eval",
        "--config",
        str(cfg_path),
        "--mode",
        "speculative",
        "--checkpoint",
        f"{tmp_path / 'out'}/policy_final.pt",
        "--max_samples",
        "7",
        "--policy_temperatures",
        "0.8,1.0",
        "--skip_baselines",
    )


def test_cli_pipeline_skips_completed_prism_and_resumes_grpo(monkeypatch, tmp_path):
    import aoae.cli as mod

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "prism_adapter.pt").write_text("ready")
    (out_dir / "policy_step25.pt").write_text("ckpt")

    cfg = {"logging": {"output_dir": str(out_dir)}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    calls = []

    monkeypatch.setattr(mod, "run_preflight", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "main", lambda argv=None: calls.append(tuple(argv)) or 0)
    monkeypatch.setattr(mod, "resolve_policy_checkpoint", lambda checkpoint, output_dir: checkpoint or f"{output_dir}/policy_step25.pt")

    mod.run_pipeline_command(
        SimpleNamespace(
            config=str(cfg_path),
            resume=None,
            checkpoint=None,
            max_samples=None,
            mode="standard",
            policy_temperatures=None,
            skip_preflight=False,
            skip_prism=False,
            skip_grpo=False,
            skip_eval=True,
            skip_baselines=False,
            strict_moe=False,
        )
    )

    assert calls == [
        ("train", "--config", str(cfg_path), "--stage", "grpo", "--resume", "auto"),
    ]


def test_cli_pipeline_skips_completed_grpo_when_final_checkpoint_exists(monkeypatch, tmp_path):
    import aoae.cli as mod

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "prism_adapter.pt").write_text("ready")
    (out_dir / "policy_final.pt").write_text("done")

    cfg = {
        "base_model": {"backend": "hf"},
        "logging": {"output_dir": str(out_dir)},
        "grpo": {"min_checkpoint_reward": 0.0},
        "soft_mask": {"top_k": 5},
        "policy": {"d_model": 128},
        "prism": {"hidden_dim": 256},
        "inference": {"steps": 16},
        "data": {"train_dataset": "demo", "train_split": "train"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    metadata = {
        "stage": "grpo",
        "train_contract_version": GRPO_TRAIN_CONTRACT_VERSION,
        "config_fingerprint": build_grpo_config_fingerprint(cfg),
        "best_reward": 0.1,
    }
    (out_dir / "grpo_training_metadata.json").write_text(json.dumps(metadata))

    calls = []

    monkeypatch.setattr(mod, "run_preflight", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "main", lambda argv=None: calls.append(tuple(argv)) or 0)
    monkeypatch.setattr(mod, "resolve_policy_checkpoint", lambda checkpoint, output_dir: checkpoint or f"{output_dir}/policy_final.pt")

    mod.run_pipeline_command(
        SimpleNamespace(
            config=str(cfg_path),
            resume=None,
            checkpoint=None,
            max_samples=None,
            mode="standard",
            policy_temperatures=None,
            skip_preflight=False,
            skip_prism=False,
            skip_grpo=False,
            skip_eval=True,
            skip_baselines=False,
            strict_moe=False,
        )
    )

    assert calls == []


def test_cli_pipeline_retrains_when_final_checkpoint_is_stale(monkeypatch, tmp_path):
    import aoae.cli as mod

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "prism_adapter.pt").write_text("ready")
    (out_dir / "policy_final.pt").write_text("done")
    (out_dir / "policy_latest.pt").write_text("resume")

    cfg = {"logging": {"output_dir": str(out_dir)}, "grpo": {"min_checkpoint_reward": 0.0}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    calls = []

    monkeypatch.setattr(mod, "run_preflight", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "main", lambda argv=None: calls.append(tuple(argv)) or 0)
    monkeypatch.setattr(mod, "resolve_policy_checkpoint", lambda checkpoint, output_dir: checkpoint or f"{output_dir}/policy_latest.pt")

    mod.run_pipeline_command(
        SimpleNamespace(
            config=str(cfg_path),
            resume=None,
            checkpoint=None,
            max_samples=None,
            mode="standard",
            policy_temperatures=None,
            skip_preflight=False,
            skip_prism=False,
            skip_grpo=False,
            skip_eval=True,
            skip_baselines=False,
            strict_moe=False,
        )
    )

    assert calls == [
        ("train", "--config", str(cfg_path), "--stage", "grpo", "--resume", "auto"),
    ]


def test_cli_pipeline_does_not_resume_failed_completed_grpo_run(monkeypatch, tmp_path):
    import aoae.cli as mod

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "prism_adapter.pt").write_text("ready")
    (out_dir / "policy_final.pt").write_text("done")
    (out_dir / "policy_latest.pt").write_text("resume")

    cfg = {
        "base_model": {"backend": "dual"},
        "logging": {"output_dir": str(out_dir)},
        "grpo": {"min_checkpoint_reward": 0.0, "max_steps": 500},
        "soft_mask": {"top_k": 5},
        "policy": {"d_model": 128},
        "prism": {"hidden_dim": 256},
        "inference": {"steps": 16},
        "data": {"train_dataset": "demo", "train_split": "train"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    metadata = {
        "stage": "grpo",
        "train_contract_version": GRPO_TRAIN_CONTRACT_VERSION,
        "config_fingerprint": build_grpo_config_fingerprint(cfg),
        "best_reward": -0.1,
        "completed_steps": 500,
        "max_steps": 500,
    }
    (out_dir / "grpo_training_metadata.json").write_text(json.dumps(metadata))

    calls = []

    monkeypatch.setattr(mod, "run_preflight", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "main", lambda argv=None: calls.append(tuple(argv)) or 0)
    monkeypatch.setattr(mod, "resolve_policy_checkpoint", lambda checkpoint, output_dir: checkpoint or f"{output_dir}/policy_latest.pt")

    mod.run_pipeline_command(
        SimpleNamespace(
            config=str(cfg_path),
            resume=None,
            checkpoint=None,
            max_samples=None,
            mode="standard",
            policy_temperatures=None,
            skip_preflight=False,
            skip_prism=False,
            skip_grpo=False,
            skip_eval=True,
            skip_baselines=False,
            strict_moe=False,
        )
    )

    assert calls == [
        ("train", "--config", str(cfg_path), "--stage", "grpo"),
    ]


def test_cli_normalizes_legacy_prism_train_invocation():
    assert _normalize_legacy_cli_argv(["train", "prism", "configs/default.yaml"]) == [
        "train",
        "--stage",
        "prism",
        "--config",
        "configs/default.yaml",
    ]


def test_cli_normalizes_legacy_grpo_train_invocation_with_resume():
    assert _normalize_legacy_cli_argv(["train", "grpo", "configs/default.yaml", "auto"]) == [
        "train",
        "--stage",
        "grpo",
        "--config",
        "configs/default.yaml",
        "--resume",
        "auto",
    ]
