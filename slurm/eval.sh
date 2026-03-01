#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_eval.out
#SBATCH -e logs/%j_eval.err
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --job-name=aoae_eval
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

# ============================================================
# AOAE — Evaluation (Step 3)
# 
# NOTE: This script requests 4 GPUs for default_large.yaml (tp_size=4).
# For default.yaml (8B, single GPU), use: sbatch --gres=gpu:1 slurm/eval.sh configs/default.yaml
# ============================================================
set -e

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
unset NCCL_BLOCKING_WAIT

CONFIG="${1:-configs/default_large.yaml}"
CHECKPOINT="${2:-outputs/default_large/policy_final.pt}"

mkdir -p logs outputs

echo "=== AOAE Evaluation ==="
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"

python3 run_eval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"

echo "=== Evaluation complete ==="
echo "Results: outputs/eval_results.json"
