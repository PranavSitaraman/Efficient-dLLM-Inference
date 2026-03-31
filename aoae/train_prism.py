"""
PRISM Adapter Training Script (paper §2.4).

Trains the lightweight quality head on (corrupted, clean) pairs from
the training dataset.  The base model is frozen; only the adapter MLP
is updated with binary cross-entropy loss.

Usage:
    python3 -m aoae.train_prism --config configs/default.yaml
"""

import os
import json
import copy
import yaml
import torch
import random
import numpy as np
from datasets import load_dataset

from .models.base_model import LLaDABaseModel
from .models.prism import PRISMAdapter, create_prism_training_data, train_prism_adapter
from .runtime_checks import collect_runtime_info


def _setup_distributed():
    """Initialize DDP from torchrun env vars when invoked as a module.

    Mirrors run_train.setup_distributed() so that invoking this script
    directly via ``torchrun -m aoae.train_prism`` works correctly.
    Returns (rank, local_rank, world_size).
    """
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def main(cfg: dict):
    """Train and save the PRISM quality adapter."""
    pc = cfg["prism"]
    dc = cfg["data"]
    lc = cfg["logging"]

    # Support both direct torchrun invocation and aoae.cli train dispatch.
    # The canonical CLI injects cfg["_dist"]; direct torchrun invocation does not.
    if cfg.get("_dist") is not None:
        rank = cfg["_dist"]["rank"]
        local_rank = cfg["_dist"]["local_rank"]
    else:
        rank, local_rank, _ = _setup_distributed()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(cfg["hardware"]["seed"] + rank)
    random.seed(cfg["hardware"]["seed"] + rank)
    np.random.seed(cfg["hardware"]["seed"] + rank)

    if rank == 0:
        print(f"Device: {device}")

    # --- Load base model ---
    # For PRISM training, we only need hidden states, so override to HF backend
    # to avoid vLLM MoE kernel requirements (soft_moe/dinfer backends)
    cfg_prism = copy.deepcopy(cfg)
    if cfg_prism["base_model"]["backend"] in ["soft_moe", "dinfer", "dual"]:
        if rank == 0:
            print(f"Overriding backend from '{cfg_prism['base_model']['backend']}' to 'hf' for PRISM training")
        cfg_prism["base_model"]["backend"] = "hf"
    
    base_model = None
    training_data = []
    try:
        if rank == 0:
            print("Loading base model...")
        base_model = LLaDABaseModel(cfg_prism)
        base_model = base_model.to(device)
        tokenizer = base_model.tokenizer
        mask_id = cfg["base_model"]["mask_token_id"]

        # --- Load training data ---
        if rank == 0:
            print("Loading training data for PRISM...")
        ds = load_dataset(dc["train_dataset"], split=dc["train_split"])

        if rank == 0:
            print(f"Creating {pc['train_samples']} corrupted/clean pairs...")
        training_data = create_prism_training_data(
            tokenizer=tokenizer,
            dataset=ds,
            mask_token_id=mask_id,
            max_samples=pc["train_samples"],
            max_length=dc["max_prompt_len"] + dc["max_answer_len"],
        )
        if rank == 0:
            print(f"  Created {len(training_data)} training pairs.")

        # --- Initialize and train adapter ---
        embed_dim = base_model.hidden_dim
        adapter = PRISMAdapter(cfg, hidden_dim=embed_dim)

        if rank == 0:
            print("Training PRISM adapter...")
        adapter = train_prism_adapter(
            adapter=adapter,
            base_model=base_model,
            training_data=training_data,
            cfg=cfg,
            device=device,
        )

        # --- Save ---
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if rank == 0:
            os.makedirs(lc["output_dir"], exist_ok=True)
            save_path = os.path.join(lc["output_dir"], "prism_adapter.pt")
            torch.save(adapter.state_dict(), save_path)
            metadata_path = os.path.join(lc["output_dir"], "prism_training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "stage": "prism",
                        "output_dir": lc["output_dir"],
                        "artifact_path": save_path,
                        "train_dataset": dc["train_dataset"],
                        "train_split": dc["train_split"],
                        "requested_train_samples": int(pc["train_samples"]),
                        "materialized_train_pairs": int(len(training_data)),
                        "seed": int(cfg["hardware"]["seed"]),
                        "runtime": collect_runtime_info(),
                    },
                    f,
                    indent=2,
                )
            print(f"PRISM adapter saved to {save_path}")
            print(f"PRISM metadata saved to {metadata_path}")
    finally:
        if base_model is not None:
            base_model.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
