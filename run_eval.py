#!/usr/bin/env python3
"""
Main evaluation entry point for AOAE.

Usage:
    # Evaluate baselines only (no trained policy needed)
    python3 run_eval.py --config configs/default.yaml --max_samples 50

    # Evaluate baselines + AOAE with Pareto sweep
    python3 run_eval.py --config configs/default.yaml --checkpoint outputs/default/policy_final.pt
"""

import argparse
import yaml

from aoae.preflight import run_preflight

def _parse_float_list(raw: str):
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def main():
    parser = argparse.ArgumentParser(description="AOAE Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained AOAE policy checkpoint (.pt).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None = full dataset).")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "speculative"],
                        help="'standard' for single-model, 'speculative' for dual-model.")
    parser.add_argument("--reuse_signal_method", type=str, default=None,
                        choices=[
                            "argmax_match", "topk_overlap", "min_confidence",
                            "min_margin", "js_divergence", "temporal_confidence",
                        ],
                        help="Override inference.reuse_signal.method for speculative runs.")
    parser.add_argument("--reuse_signal_threshold", type=float, default=None,
                        help="Override inference.reuse_signal.threshold.")
    parser.add_argument("--track_kv_dynamics", action="store_true",
                        help="Enable analysis.track_kv_dynamics.")
    parser.add_argument("--disable_remask", action="store_true",
                        help="Set inference.disable_remask=true for remask-off ablation.")
    parser.add_argument("--enable_positional_cache", action="store_true",
                        help="Enable inference.positional_cache for next-H access experiments.")
    parser.add_argument("--positional_cache_horizon", type=int, default=None,
                        help="Override inference.positional_cache.horizon.")
    parser.add_argument("--positional_cache_refresh_budget", type=int, default=None,
                        help="Override inference.positional_cache.refresh_budget.")
    parser.add_argument("--policy_temperatures", type=str, default=None,
                        help="Comma-separated tau_pi values for speculative runs (e.g. 0.5,1.0,1.5).")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip baseline decoding methods and only evaluate the target policy/method.")
    parser.add_argument("--task_type", type=str, default=None, choices=["math", "code"],
                        help="Override evaluation.task_type.")
    parser.add_argument("--code_timeout_sec", type=float, default=None,
                        help="Override evaluation.code.timeout_sec for task_type=code.")
    parser.add_argument("--code_cpu_time_limit_sec", type=int, default=None,
                        help="Override evaluation.code.cpu_time_limit_sec.")
    parser.add_argument("--code_memory_limit_mb", type=int, default=None,
                        help="Override evaluation.code.memory_limit_mb.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save a bounded set of per-sample generated responses.")
    parser.add_argument("--max_saved_predictions", type=int, default=None,
                        help="Maximum number of responses to save (hard-capped at 50).")
    parser.add_argument("--preflight", action="store_true",
                        help="Run environment+runtime+config preflight and exit.")
    parser.add_argument("--strict_moe", action="store_true",
                        help="When used with --preflight, fail if required MoE ops are unavailable.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate config + create output dir, then exit without model loading.")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Override data.eval_dataset.")
    parser.add_argument("--eval_dataset_config", type=str, default=None,
                        help="Override data.eval_dataset_config (use empty string for no config).")
    parser.add_argument("--eval_split", type=str, default=None,
                        help="Override data.eval_split.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.preflight:
        report = run_preflight(args.config, strict_moe=args.strict_moe)
        import json as _json
        print(_json.dumps(report, indent=2))
        return

    ic = cfg.setdefault("inference", {})
    dc = cfg.setdefault("data", {})
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
        cfg.setdefault("evaluation", {})["max_saved_predictions"] = min(
            int(args.max_saved_predictions), 50
        )

    if args.dry_run:
        out_dir = cfg.setdefault("logging", {}).get("output_dir", "outputs/default/")
        import os as _os
        _os.makedirs(out_dir, exist_ok=True)
        print(f"[DryRun] Config parsed OK. Output dir ready: {out_dir}")
        return

    policy_temperatures = None
    if args.policy_temperatures is not None:
        policy_temperatures = _parse_float_list(args.policy_temperatures)

    from aoae.evaluate import main as eval_main
    eval_main(
        cfg,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        mode=args.mode,
        config_path=args.config,
        skip_baselines=args.skip_baselines,
        speculative_policy_temperatures=policy_temperatures,
    )


if __name__ == "__main__":
    main()
