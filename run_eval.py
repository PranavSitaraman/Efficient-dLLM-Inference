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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
