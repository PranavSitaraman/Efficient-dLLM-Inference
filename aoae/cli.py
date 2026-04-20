"""Canonical AOAE command-line interface."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from typing import Iterable, List, Optional

import yaml

from .checkpoints import resolve_policy_checkpoint
from .checkpoints import inspect_grpo_artifacts, inspect_grpo_resume_candidate
from .checkpoints import find_latest_checkpoint
from .experiment_utils import parse_float_list
from .preflight import run_preflight


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_runtime_env_defaults(env: Optional[dict] = None) -> dict:
    target = os.environ if env is None else env
    target.setdefault("HF_HUB_DISABLE_XET", "1")
    target.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    target.setdefault("MASTER_ADDR", "127.0.0.1")
    target.setdefault("MASTER_PORT", "29500")
    target.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    target.setdefault("GLOO_SOCKET_FAMILY", "AF_INET")
    if target.get("MASTER_ADDR") == "localhost":
        target["MASTER_ADDR"] = "127.0.0.1"
    return target


def _extract_flag_value(argv: List[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    prefix = f"{flag}="
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix):]
    return default


def _normalize_legacy_cli_argv(argv_list: List[str]) -> List[str]:
    """Rewrite legacy positional train invocations to the canonical flag form.

    Older wrappers used:
      - ``train prism [config]``
      - ``train grpo [config] [resume]``

    The canonical interface is now:
      - ``train --config <cfg> --stage prism``
      - ``train --config <cfg> --stage grpo --resume <resume>``
    """
    if not argv_list or argv_list[0] != "train" or len(argv_list) < 2:
        return argv_list

    stage = argv_list[1]
    if stage not in {"prism", "grpo"}:
        return argv_list

    remainder = list(argv_list[2:])
    if any(token.startswith("-") for token in remainder):
        return argv_list

    rewritten = ["train", "--stage", stage]
    if remainder:
        rewritten.extend(["--config", remainder[0]])
        remainder = remainder[1:]
    if stage == "grpo" and remainder:
        rewritten.extend(["--resume", remainder[0]])
        remainder = remainder[1:]
    rewritten.extend(remainder)
    return rewritten


def _config_tp_size(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 1
    try:
        cfg = _load_config(path)
    except Exception:
        return 1
    return int(cfg.get("hardware", {}).get("tp_size", 1) or 1)


def _required_world_size(argv_list: List[str]) -> int:
    if not argv_list:
        return 1
    command = argv_list[0]
    args = argv_list[1:]

    if command in {"test", "preflight", "comparison-table", "kv-summary"}:
        return 1

    if command in {"train", "eval"}:
        config_path = _extract_flag_value(args, "--config", "configs/llada21_hard.yaml")
        return _config_tp_size(config_path)

    if command == "pipeline":
        # The pipeline acts as a single-process coordinator and delegates
        # train/eval stages through the canonical subcommands below. Those
        # child subcommands own any required torchrun relaunch.
        return 1

    if command in {"tau-sweep", "reuse-sweep", "ablations", "paper-suite"}:
        default_config = {
            "tau-sweep": "configs/poc1.yaml",
            "reuse-sweep": "configs/poc2.yaml",
            "ablations": "configs/paper.yaml",
            "paper-suite": "configs/paper.yaml",
        }[command]
        config_path = _extract_flag_value(args, "--config", default_config)
        return _config_tp_size(config_path)

    if command == "routing-sweep":
        hard_config = _extract_flag_value(args, "--hard_config", "configs/llada21_hard.yaml")
        soft_config = _extract_flag_value(args, "--soft_config", "configs/llada21_soft.yaml")
        return max(_config_tp_size(hard_config), _config_tp_size(soft_config))

    return 1


def _maybe_relaunch_with_torchrun(argv_list: List[str]) -> Optional[int]:
    if os.environ.get("AOAE_DISABLE_AUTO_TORCHRUN") == "1":
        return None
    if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return None

    world_size = _required_world_size(argv_list)
    if world_size <= 1:
        return None

    env = _apply_runtime_env_defaults(dict(os.environ))
    torchrun_bin = shutil.which("torchrun")
    if torchrun_bin:
        cmd = [
            torchrun_bin,
            "--nproc_per_node",
            str(world_size),
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            "--master_addr",
            env["MASTER_ADDR"],
            "--master_port",
            env["MASTER_PORT"],
            "-m",
            "aoae.cli",
            *argv_list,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(world_size),
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            "--master_addr",
            env["MASTER_ADDR"],
            "--master_port",
            env["MASTER_PORT"],
            "-m",
            "aoae.cli",
            *argv_list,
        ]
    env["AOAE_DISABLE_AUTO_TORCHRUN"] = "1"
    print(
        f"[Launcher] Detected tp_size={world_size}; relaunching under torchrun "
        f"with MASTER_ADDR={env['MASTER_ADDR']} MASTER_PORT={env['MASTER_PORT']}."
    )
    return subprocess.call(cmd, env=env)


def _setup_distributed():
    _apply_runtime_env_defaults()
    if "RANK" not in os.environ:
        return None
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _cleanup_distributed(dist_info) -> None:
    if dist_info is None:
        return
    try:
        import torch.distributed as dist
    except Exception:
        return

    if not dist.is_available() or not dist.is_initialized():
        return

    try:
        dist.destroy_process_group()
    except (AssertionError, RuntimeError):
        # vLLM/NCCL teardown can clear the default process group before the
        # CLI finally-block runs. Cleanup should be best-effort; the training
        # stage has already completed by the time we reach this path.
        pass


def _cleanup_process_group_if_initialized() -> None:
    """Best-effort distributed teardown for eval and sweep commands."""
    try:
        import torch.distributed as dist
    except Exception:
        return

    if not dist.is_available() or not dist.is_initialized():
        return

    try:
        dist.destroy_process_group()
    except Exception:
        pass

def _invoke_python_entrypoint(module_name: str, prog: str, argv: List[str], entry_name: str = "main"):
    module = importlib.import_module(module_name)
    main_fn = getattr(module, entry_name)
    prev_argv = sys.argv[:]
    try:
        sys.argv = [prog, *argv]
        if len(inspect.signature(main_fn).parameters) == 0:
            return main_fn()
        return main_fn(argv)
    finally:
        sys.argv = prev_argv


def _run_script_command(module_name: str, entry_name: str, prog: str, script_args: Optional[List[str]]):
    return _invoke_python_entrypoint(module_name, prog, list(script_args or []), entry_name=entry_name)


def _add_passthrough_command(
    subparsers: argparse._SubParsersAction,
    command: str,
    help_text: str,
    module_name: str,
    entry_name: str,
    prog: str,
) -> None:
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying research script.",
    )
    parser.set_defaults(
        func=lambda args, _module=module_name, _entry=entry_name, _prog=prog: _run_script_command(
            _module,
            _entry,
            _prog,
            args.script_args,
        )
    )


PASSTHROUGH_COMMANDS = [
    ("tau-sweep", "Run the PoC 1 routing-temperature sweep.", "aoae.paper", "tau_sweep_main", "tau-sweep"),
    ("routing-sweep", "Run the hard-vs-soft routing comparison sweep.", "aoae.paper", "routing_sweep_main", "routing-sweep"),
    ("reuse-sweep", "Run the PoC 2 KV-reuse signal sweep.", "aoae.paper", "reuse_signal_sweep_main", "reuse-sweep"),
    ("reuse-posthoc", "Rebuild PoC 2 KV/reuse summaries from saved sweep artifacts.", "aoae.paper", "reuse_signal_posthoc_main", "reuse-posthoc"),
    ("ablations", "Run the AOAE ablation matrix.", "aoae.paper", "ablation_matrix_main", "ablations"),
    ("paper-suite", "Run the paper-aligned end-to-end experiment suite.", "aoae.paper", "paper_suite_main", "paper-suite"),
    ("comparison-table", "Build a comparison table from eval artifacts.", "aoae.reporting", "comparison_table_main", "comparison-table"),
    ("kv-summary", "Summarize KV-dynamics artifacts across runs.", "aoae.reporting", "kv_summary_main", "kv-summary"),
]


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/llada21_hard.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained AOAE policy checkpoint (.pt).")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (None = full dataset).")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "speculative"], help="'standard' for single-model, 'speculative' for dual-model.")
    parser.add_argument("--reuse_signal_method", type=str, default=None, choices=["argmax_match", "topk_overlap", "min_confidence", "min_margin", "js_divergence", "temporal_confidence"], help="Override inference.reuse_signal.method for speculative runs.")
    parser.add_argument("--reuse_signal_threshold", type=float, default=None, help="Override inference.reuse_signal.threshold.")
    parser.add_argument("--track_kv_dynamics", action="store_true", help="Enable analysis.track_kv_dynamics.")
    parser.add_argument("--disable_remask", action="store_true", help="Set inference.disable_remask=true for remask-off ablation.")
    parser.add_argument("--enable_positional_cache", action="store_true", help="Enable inference.positional_cache for next-H access experiments.")
    parser.add_argument("--positional_cache_horizon", type=int, default=None, help="Override inference.positional_cache.horizon.")
    parser.add_argument("--positional_cache_refresh_budget", type=int, default=None, help="Override inference.positional_cache.refresh_budget.")
    parser.add_argument("--policy_temperatures", type=str, default=None, help="Comma-separated tau_pi values for speculative runs (e.g. 0.5,1.0,1.5).")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline decoding methods and only evaluate the target policy/method.")
    parser.add_argument("--task_type", type=str, default=None, choices=["math", "code"], help="Override evaluation.task_type.")
    parser.add_argument("--code_timeout_sec", type=float, default=None, help="Override evaluation.code.timeout_sec for task_type=code.")
    parser.add_argument("--code_cpu_time_limit_sec", type=int, default=None, help="Override evaluation.code.cpu_time_limit_sec.")
    parser.add_argument("--code_memory_limit_mb", type=int, default=None, help="Override evaluation.code.memory_limit_mb.")
    parser.add_argument("--save_predictions", action="store_true", help="Save a bounded set of per-sample generated responses.")
    parser.add_argument("--max_saved_predictions", type=int, default=None, help="Maximum number of responses to save (hard-capped at 50).")
    parser.add_argument("--preflight", action="store_true", help="Run environment+runtime+config preflight and exit.")
    parser.add_argument("--strict_moe", action="store_true", help="When used with --preflight, fail if required MoE ops are unavailable.")
    parser.add_argument("--dry_run", action="store_true", help="Validate config + create output dir, then exit without model loading.")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Override data.eval_dataset.")
    parser.add_argument("--eval_dataset_config", type=str, default=None, help="Override data.eval_dataset_config (use empty string for no config).")
    parser.add_argument("--eval_split", type=str, default=None, help="Override data.eval_split.")
    parser.add_argument("--backend", type=str, default=None, choices=["auto", "hf", "dkv", "dinfer", "soft_moe"], help="Override base_model.backend.")
    parser.add_argument("--routing_temperature", type=float, default=None, help="Override base_model.routing_temperature (tau_r).")
    parser.add_argument("--soft_topk", type=int, default=None, help="Override base_model.soft_topk. Use the full expert count for all-experts soft gating.")
    parser.add_argument("--run_name", type=str, default=None, help="Override logging.run_name.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override logging.output_dir.")


def apply_eval_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    ic = cfg.setdefault("inference", {})
    dc = cfg.setdefault("data", {})
    bc = cfg.setdefault("base_model", {})
    lc = cfg.setdefault("logging", {})

    if args.reuse_signal_method is not None:
        ic.setdefault("reuse_signal", {})["method"] = args.reuse_signal_method
    if args.reuse_signal_threshold is not None:
        ic.setdefault("reuse_signal", {})["threshold"] = float(args.reuse_signal_threshold)
    if args.disable_remask:
        ic["disable_remask"] = True
    if args.track_kv_dynamics:
        cfg.setdefault("analysis", {})["track_kv_dynamics"] = True
    if args.enable_positional_cache:
        ic.setdefault("positional_cache", {})["enabled"] = True
    if args.positional_cache_horizon is not None:
        ic.setdefault("positional_cache", {})["horizon"] = int(args.positional_cache_horizon)
    if args.positional_cache_refresh_budget is not None:
        ic.setdefault("positional_cache", {})["refresh_budget"] = int(args.positional_cache_refresh_budget)
    if args.eval_dataset is not None:
        dc["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        dc["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        dc["eval_split"] = args.eval_split
    if args.backend is not None:
        bc["backend"] = args.backend
    if args.routing_temperature is not None:
        bc["routing_temperature"] = float(args.routing_temperature)
    if args.soft_topk is not None:
        bc["soft_topk"] = int(args.soft_topk)
    if args.task_type is not None:
        cfg.setdefault("evaluation", {})["task_type"] = args.task_type
    if args.code_timeout_sec is not None:
        cfg.setdefault("evaluation", {}).setdefault("code", {})["timeout_sec"] = float(args.code_timeout_sec)
    if args.code_cpu_time_limit_sec is not None:
        cfg.setdefault("evaluation", {}).setdefault("code", {})["cpu_time_limit_sec"] = int(args.code_cpu_time_limit_sec)
    if args.code_memory_limit_mb is not None:
        cfg.setdefault("evaluation", {}).setdefault("code", {})["memory_limit_mb"] = int(args.code_memory_limit_mb)
    if args.save_predictions:
        cfg.setdefault("evaluation", {})["save_predictions"] = True
    if args.max_saved_predictions is not None:
        cfg.setdefault("evaluation", {})["save_predictions"] = True
        cfg.setdefault("evaluation", {})["max_saved_predictions"] = min(int(args.max_saved_predictions), 50)
    if args.run_name is not None:
        lc["run_name"] = args.run_name
    if args.output_dir is not None:
        lc["output_dir"] = args.output_dir

    return cfg


def run_eval_command(args: argparse.Namespace):
    cfg = _load_config(args.config)

    if args.preflight:
        report = run_preflight(args.config, strict_moe=args.strict_moe)
        print(json.dumps(report, indent=2))
        return report

    cfg = apply_eval_overrides(cfg, args)

    if args.dry_run:
        out_dir = cfg.setdefault("logging", {}).get("output_dir", "outputs/llada21_hard/")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[DryRun] Config parsed OK. Output dir ready: {out_dir}")
        return None

    policy_temperatures = None
    if args.policy_temperatures is not None:
        policy_temperatures = parse_float_list(args.policy_temperatures, label="policy temperature")

    from .evaluate import main as eval_main

    return eval_main(
        cfg,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        mode=args.mode,
        config_path=args.config,
        skip_baselines=args.skip_baselines,
        speculative_policy_temperatures=policy_temperatures,
    )


def run_train_command(args: argparse.Namespace):
    cfg = _load_config(args.config)
    dist_info = _setup_distributed()
    try:
        cfg["_dist"] = dist_info
        if dist_info is not None and dist_info["rank"] == 0:
            print(f"Distributed training: {dist_info['world_size']} GPUs")

        if args.stage == "prism":
            from .train_prism import main as train_prism

            return train_prism(cfg)

        from .train_grpo import train

        return train(cfg, resume_from=args.resume)
    finally:
        _cleanup_distributed(dist_info)


def run_pipeline_command(args: argparse.Namespace):
    cfg = _load_config(args.config)
    output_dir = cfg.get("logging", {}).get("output_dir", "")
    prism_path = os.path.join(output_dir, "prism_adapter.pt") if output_dir else ""
    final_policy_path = os.path.join(output_dir, "policy_final.pt") if output_dir else ""
    latest_policy_ckpt = find_latest_checkpoint(output_dir)
    grpo_status = inspect_grpo_artifacts(output_dir, cfg, enforce_min_reward=False) if output_dir else {
        "valid": False,
        "reason": "missing_output_dir",
        "checkpoint_path": None,
    }
    grpo_quality_status = inspect_grpo_artifacts(output_dir, cfg, enforce_min_reward=True) if output_dir else grpo_status
    resume_status = inspect_grpo_resume_candidate(output_dir, cfg) if output_dir else {
        "valid": False,
        "reason": "missing_output_dir",
        "checkpoint_path": None,
    }

    if not args.skip_preflight:
        report = run_preflight(args.config, strict_moe=args.strict_moe)
        print(json.dumps(report, indent=2))

    is_dual_backend = cfg.get("base_model", {}).get("backend") == "dual"
    if not is_dual_backend:
        print(
            "[Pipeline] Using a single-model dense/HF config. "
            "This path is a reference/dev pipeline, not the canonical paper speculative AOAE setup."
        )

    def _run_nested_cli(argv: List[str]):
        rc = main(argv)
        if isinstance(rc, int) and rc != 0:
            raise SystemExit(rc)
        return rc

    if not args.skip_prism and prism_path and os.path.exists(prism_path):
        print(f"[Pipeline] Found existing PRISM adapter at {prism_path}; skipping PRISM training.")
    elif not args.skip_prism:
        _run_nested_cli(["train", "--config", args.config, "--stage", "prism"])

    ran_grpo = False
    if (
        not args.skip_grpo
        and final_policy_path
        and os.path.exists(final_policy_path)
        and bool(grpo_status.get("valid"))
    ):
        print(f"[Pipeline] Found existing final GRPO checkpoint at {final_policy_path}; skipping GRPO training.")
        if not bool(grpo_status.get("quality_ok", True)):
            print(
                "[Pipeline] Note: checkpoint matches the current config but is below "
                f"min_checkpoint_reward={grpo_status.get('min_checkpoint_reward')}; "
                "evaluation will still load it instead of falling back to the default policy."
            )
    elif not args.skip_grpo:
        if final_policy_path and os.path.exists(final_policy_path):
            print(
                "[Pipeline] Existing GRPO artifacts do not match the current run contract "
                f"({grpo_status.get('reason')}); retraining."
            )
        nested_grpo = ["train", "--config", args.config, "--stage", "grpo"]
        resume_value = args.resume
        if resume_value is None and latest_policy_ckpt is not None and bool(resume_status.get("valid")):
            resume_value = "auto"
            print(f"[Pipeline] Found existing GRPO checkpoint at {latest_policy_ckpt}; resuming training.")
        elif resume_value is None and latest_policy_ckpt is not None:
            print(
                "[Pipeline] Ignoring existing GRPO resume checkpoint "
                f"({resume_status.get('reason')}); starting GRPO from scratch."
            )
        if resume_value is not None:
            nested_grpo.extend(["--resume", resume_value])
        _run_nested_cli(nested_grpo)
        ran_grpo = True
        grpo_status = inspect_grpo_artifacts(output_dir, cfg, enforce_min_reward=False) if output_dir else grpo_status
        grpo_quality_status = inspect_grpo_artifacts(output_dir, cfg, enforce_min_reward=True) if output_dir else grpo_quality_status
        resume_status = inspect_grpo_resume_candidate(output_dir, cfg) if output_dir else resume_status
        if bool(grpo_status.get("valid")) and not bool(grpo_status.get("quality_ok", True)):
            print(
                "[Pipeline] Completed GRPO checkpoint is below the configured reward threshold "
                f"({grpo_quality_status.get('reason')}); continuing to evaluation with the trained checkpoint."
            )
    elif not bool(grpo_status.get("valid")) and args.checkpoint is None:
        print(
            "[Pipeline] No valid GRPO checkpoint is available for evaluation "
            f"({grpo_status.get('reason')}); AOAE policy eval will be skipped."
        )

    if args.skip_eval:
        return None

    checkpoint = args.checkpoint
    if checkpoint is None and bool(grpo_status.get("valid")):
        checkpoint = resolve_policy_checkpoint(None, cfg.get("logging", {}).get("output_dir", ""))
    if checkpoint is None and ran_grpo and output_dir and bool(grpo_status.get("valid")):
        checkpoint = resolve_policy_checkpoint(None, output_dir)
    # Auto-select evaluation mode: speculative for dual backends, standard for all others.
    # An explicit --mode flag from the user always wins.
    if args.mode is not None:
        eval_mode = args.mode
    else:
        eval_mode = "speculative" if is_dual_backend else "standard"
    eval_args = argparse.Namespace(
        config=args.config,
        checkpoint=checkpoint,
        max_samples=args.max_samples,
        mode=eval_mode,
        reuse_signal_method=None,
        reuse_signal_threshold=None,
        track_kv_dynamics=False,
        disable_remask=False,
        enable_positional_cache=False,
        positional_cache_horizon=None,
        positional_cache_refresh_budget=None,
        policy_temperatures=args.policy_temperatures,
        skip_baselines=args.skip_baselines,
        task_type=None,
        code_timeout_sec=None,
        code_cpu_time_limit_sec=None,
        code_memory_limit_mb=None,
        save_predictions=False,
        max_saved_predictions=None,
        preflight=False,
        strict_moe=False,
        dry_run=False,
        eval_dataset=None,
        eval_dataset_config=None,
        eval_split=None,
        backend=None,
        routing_temperature=None,
        soft_topk=None,
        run_name=None,
        output_dir=None,
    )
    nested_eval = [
        "eval",
        "--config",
        args.config,
        "--mode",
        eval_mode,
    ]
    if checkpoint is not None:
        nested_eval.extend(["--checkpoint", checkpoint])
    if args.max_samples is not None:
        nested_eval.extend(["--max_samples", str(args.max_samples)])
    if args.policy_temperatures is not None:
        nested_eval.extend(["--policy_temperatures", args.policy_temperatures])
    if args.skip_baselines:
        nested_eval.append("--skip_baselines")
    return _run_nested_cli(nested_eval)


def run_test_command(args: argparse.Namespace) -> int:
    pytest_args = list(args.pytest_args or ["tests", "-v"])
    return subprocess.call([sys.executable, "-m", "pytest", *pytest_args])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AOAE integrated training, evaluation, and testing CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run PRISM or GRPO training.")
    train_parser.add_argument("--config", type=str, default="configs/llada21_hard.yaml", help="Path to YAML config file.")
    train_parser.add_argument("--stage", type=str, choices=["prism", "grpo"], required=True, help="Training stage to run.")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume GRPO training from checkpoint or use 'auto'.")
    train_parser.set_defaults(func=run_train_command)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation/inference.")
    add_eval_args(eval_parser)
    eval_parser.set_defaults(func=run_eval_command)

    preflight_parser = subparsers.add_parser("preflight", help="Run environment and runtime checks.")
    preflight_parser.add_argument("--config", default="configs/llada21_hard.yaml", help="YAML config path.")
    preflight_parser.add_argument("--strict_moe", action="store_true", help="Fail if required MoE ops are missing.")
    preflight_parser.set_defaults(
        func=lambda args: print(json.dumps(run_preflight(args.config, strict_moe=args.strict_moe), indent=2))
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run preflight, training, and evaluation end to end.")
    pipeline_parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to YAML config file.")
    pipeline_parser.add_argument("--resume", type=str, default=None, help="Resume GRPO training from checkpoint or use 'auto'.")
    pipeline_parser.add_argument("--checkpoint", type=str, default=None, help="Optional explicit checkpoint for the eval step.")
    pipeline_parser.add_argument("--max_samples", type=int, default=None, help="Optional evaluation cap.")
    pipeline_parser.add_argument("--mode", type=str, default=None, choices=["standard", "speculative"], help="Evaluation mode for the final stage. Defaults to 'speculative' for dual backends and 'standard' otherwise.")
    pipeline_parser.add_argument("--policy_temperatures", type=str, default=None, help="Comma-separated tau_pi values for speculative evaluation.")
    pipeline_parser.add_argument("--skip_preflight", action="store_true", help="Skip preflight checks.")
    pipeline_parser.add_argument("--skip_prism", action="store_true", help="Skip PRISM training.")
    pipeline_parser.add_argument("--skip_grpo", action="store_true", help="Skip GRPO training.")
    pipeline_parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation.")
    pipeline_parser.add_argument("--skip_baselines", action="store_true", help="Skip baselines during the eval stage.")
    pipeline_parser.add_argument("--strict_moe", action="store_true", help="Fail preflight if required MoE ops are missing.")
    pipeline_parser.set_defaults(func=run_pipeline_command)

    test_parser = subparsers.add_parser("test", help="Run the repository test suite.")
    test_parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Additional pytest arguments.")
    test_parser.set_defaults(func=run_test_command)

    for command, help_text, module_name, entry_name, prog in PASSTHROUGH_COMMANDS:
        _add_passthrough_command(subparsers, command, help_text, module_name, entry_name, prog)

    return parser


def main(argv: Optional[Iterable[str]] = None):
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    argv_list = _normalize_legacy_cli_argv(argv_list)
    _apply_runtime_env_defaults()
    relaunch_code = _maybe_relaunch_with_torchrun(argv_list)
    if relaunch_code is not None:
        return relaunch_code
    try:
        passthrough_map = {
            command: (module_name, entry_name, prog)
            for command, _help, module_name, entry_name, prog in PASSTHROUGH_COMMANDS
        }
        if argv_list and argv_list[0] in passthrough_map:
            module_name, entry_name, prog = passthrough_map[argv_list[0]]
            return _run_script_command(module_name, entry_name, prog, argv_list[1:])

        parser = build_parser()
        args = parser.parse_args(argv_list)
        return args.func(args)
    finally:
        _cleanup_process_group_if_initialized()


if __name__ == "__main__":
    main()
