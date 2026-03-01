#!/usr/bin/env python3
"""
Main training entry point for AOAE.

Usage:
    # Step 1: Train PRISM adapter (single GPU)
    python3 run_train.py --config configs/default.yaml --stage prism

    # Step 2: Train AOAE policy via GRPO (single GPU)
    python3 run_train.py --config configs/default.yaml --stage grpo

    # Resume GRPO from latest checkpoint (auto-detect):
    python3 run_train.py --config configs/default.yaml --stage grpo --resume auto

    # Resume GRPO from a specific checkpoint:
    python3 run_train.py --config configs/default.yaml --stage grpo --resume outputs/policy_step1000.pt

    # Multi-GPU via torchrun:
    torchrun --nproc_per_node 4 run_train.py --config configs/default.yaml --stage grpo
"""

import os
import argparse
import yaml


def setup_distributed():
    """Initialize distributed training if launched via torchrun."""
    if "RANK" in os.environ:
        import torch
        import torch.distributed as dist
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return {"rank": rank, "local_rank": local_rank, "world_size": world_size}
    return None


def main():
    parser = argparse.ArgumentParser(description="AOAE Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--stage", type=str, choices=["prism", "grpo"], required=True,
                        help="Training stage: 'prism' for PRISM adapter, 'grpo' for policy.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume GRPO training from checkpoint. "
                             "Pass 'auto' to auto-detect latest, or a path to a .pt file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Inject distributed info into config
    dist_info = setup_distributed()
    if dist_info is not None:
        cfg["_dist"] = dist_info
        if dist_info["rank"] == 0:
            print(f"Distributed training: {dist_info['world_size']} GPUs")
    else:
        cfg["_dist"] = None

    if args.stage == "prism":
        from aoae.train_prism import main as train_prism
        train_prism(cfg)
    elif args.stage == "grpo":
        from aoae.train_grpo import train
        train(cfg, resume_from=args.resume)

    # Cleanup
    if dist_info is not None:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
