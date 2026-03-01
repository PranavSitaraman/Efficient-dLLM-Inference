#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_mini.out
#SBATCH -e logs/%j_mini.err
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --job-name=aoae_mini
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

# ============================================================
# AOAE: LLaDA2.1-mini training (soft-routed or hard-routed)
# NOTE: This script requests 2 GPUs for LLaDA2.1-mini (tp_size=2).
# Usage:
#   sbatch slurm/train_mini.sh                                    # soft-routed (default)
#   sbatch slurm/train_mini.sh configs/llada21_mini_hard.yaml     # hard-routed baseline
# ============================================================

set -euo pipefail
mkdir -p logs

export HF_HUB_DISABLE_XET=1
unset NCCL_BLOCKING_WAIT 2>/dev/null || true

CONFIG="${1:-configs/llada21_mini_soft.yaml}"
echo "=== AOAE LLaDA2.1-mini Training ==="
echo "Config: $CONFIG"
echo "GPUs:   $SLURM_GPUS_ON_NODE"

# Step 1: Train PRISM adapter
echo "--- Step 1: PRISM adapter training ---"
python3 -m aoae.train_prism --config "$CONFIG"

# Step 2: GRPO policy training
echo "--- Step 2: GRPO policy training ---"
python3 -m aoae.train_grpo --config "$CONFIG"

echo "=== Training complete ==="
