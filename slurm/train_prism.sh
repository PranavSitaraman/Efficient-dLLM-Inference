#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_prism.out
#SBATCH -e logs/%j_prism.err
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --job-name=aoae_prism
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

# ============================================================
# AOAE — PRISM Adapter Training (Step 1)
# 
# NOTE: This script requests 4 GPUs for default_large.yaml (tp_size=4).
# For default.yaml (8B, single GPU), use: sbatch --gres=gpu:1 slurm/train_prism.sh configs/default.yaml
# ============================================================
set -e

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
CONFIG="${1:-configs/llada21_mini_soft.yaml}"
mkdir -p logs outputs

export GPUS_PER_NODE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29505
export NCCL_SOCKET_FAMILY=AF_INET

echo "=== PRISM Training ==="
echo "Config: $CONFIG"
echo "Node:   $(hostname)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    run_train.py --config "$CONFIG" --stage prism

echo "=== PRISM training complete ==="
