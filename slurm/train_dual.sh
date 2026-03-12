#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_dual_train.out
#SBATCH -e logs/%j_dual_train.err
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --job-name=aoae_dual
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

# ============================================================
# AOAE — Dual-Model Speculative Diffusion Pipeline
#
# Single LLaDA2.1-mini (16B MoE) with routing mode toggling:
#   - Hard routing (auxiliary): ~1.4B active params, fast draft
#   - Soft routing (primary):  all 16B active, slow verification
#
# Steps:
#   1. PRISM adapter training (always runs)
#   2. GRPO policy training  (optional, controlled by grpo.enabled)
#   3. Evaluation sweep       (inference throughput comparison)
#
# Usage:
#   sbatch slurm/train_dual.sh                              # default τ_r=0.01
#   sbatch slurm/train_dual.sh configs/dual_mini_tau01.yaml  # τ_r=0.1
# ============================================================
set -e

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP16=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0"
export GPUS_PER_NODE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29506
export NCCL_SOCKET_FAMILY=AF_INET

CONFIG="${1:-configs/dual_mini_tau001.yaml}"
RESUME="${2:-auto}"

mkdir -p logs outputs

echo "=== AOAE Dual-Model Speculative Training ==="
echo "Config:  $CONFIG"
echo "Resume:  $RESUME"
echo "Node:    $(hostname)"
echo "GPUs:    $GPUS_PER_NODE"

# Step 1: Train PRISM adapter (single-model, HF backend)
# Route through run_train.py so DDP init + torch.cuda.set_device(local_rank)
# are called before model loading, preventing all ranks from landing on cuda:0.
echo ""
echo "--- Step 1: PRISM Adapter Training ---"
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    run_train.py --stage prism --config "$CONFIG"

# Step 2: GRPO training (optional — skipped if grpo.enabled=false in config)
echo ""
echo "--- Step 2: GRPO Training (optional) ---"
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    run_train.py --stage grpo --config "$CONFIG" --resume "$RESUME"

# Step 3: Evaluation (inference throughput comparison)
echo ""
echo "--- Step 3: Evaluation (hard aux vs soft primary, τ_r sweep) ---"
python3 run_eval.py --config "$CONFIG" --mode speculative --max_samples 10


echo ""
echo "=== Pipeline complete ==="
