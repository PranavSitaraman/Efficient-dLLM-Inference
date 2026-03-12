#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_benchmark_tps.out
#SBATCH -e logs/%j_benchmark_tps.err
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=aoae_tps_bench
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

# ============================================================
# AOAE TPS Benchmark — Measure end-to-end inference speed
#
# Runs speculative diffusion inference on a small eval set
# and reports tokens per second (TPS).
#
# Usage:
#   sbatch slurm/benchmark_tps.sh
#   sbatch slurm/benchmark_tps.sh configs/dual_mini_tau01.yaml 20
# ============================================================
set -e

module load python/3.12.5-fasrc01 cuda/12.4.1-fasrc01 cudnn/9.1.1.17_cuda12-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

# CUDA 12.x overrides MUST come AFTER conda activate — conda's PyTorch bundles
# CUDA 11.8 nvcc, which cannot compile for H100 (compute_90a).
export CUDA_HOME=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

# Clear stale FlashInfer JIT cache that was compiled with wrong nvcc
rm -rf ~/.cache/flashinfer/*/90a/

export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Disable FlashInfer CUTLASS MoE — CCCL header conflict with CUDA 12.4
# H100 still benefits from Triton fused_moe which works without CUTLASS
export VLLM_USE_FLASHINFER_MOE_FP16=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export GPUS_PER_NODE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29507
export NCCL_SOCKET_FAMILY=AF_INET

CONFIG="${1:-configs/dual_mini_tau01.yaml}"
MAX_SAMPLES="${2:-10}"

mkdir -p logs outputs

echo "=== AOAE TPS Benchmark ==="
echo "Config:      $CONFIG"
echo "Max samples: $MAX_SAMPLES"
echo "Node:        $(hostname)"
echo "GPUs:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    run_eval.py \
    --config "$CONFIG" \
    --mode speculative \
    --max_samples "$MAX_SAMPLES" \
    --skip_baselines

echo ""
echo "=== Benchmark complete ==="
