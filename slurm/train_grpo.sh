#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_grpo.out
#SBATCH -e logs/%j_grpo.err
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --job-name=aoae_grpo
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

# ============================================================
# AOAE — GRPO Training (Step 2)
# 
# NOTE: This script requests 4 GPUs for default_large.yaml (tp_size=4).
# For default.yaml (8B, single GPU), use: sbatch --gres=gpu:1 slurm/train_grpo.sh configs/default.yaml
# ============================================================
set -e

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
unset NCCL_BLOCKING_WAIT

CONFIG="${1:-configs/llada21_mini_soft.yaml}"
RESUME="${2:-auto}"

export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_DISABLE_XET=1
export GPUS_PER_NODE=4
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_DISTRIBUTED_DETAIL=DEBUG

mkdir -p logs outputs

echo "=== AOAE GRPO Training ==="
echo "Config:     $CONFIG"
echo "Resume:     $RESUME"
echo "Nodes:      $SLURM_NNODES"
echo "GPUs/node:  $GPUS_PER_NODE"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"

srun bash -c 'torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    run_train.py \
    --config '"$CONFIG"' \
    --stage grpo \
    --resume '"$RESUME"''

echo "=== GRPO training complete ==="
