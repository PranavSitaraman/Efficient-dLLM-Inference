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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ic = cfg.setdefault("inference", {})
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

    from aoae.evaluate import main as eval_main
    eval_main(
        cfg,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        mode=args.mode,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
