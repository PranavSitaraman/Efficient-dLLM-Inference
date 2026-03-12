#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_benchmark_tps_a100.out
#SBATCH -e logs/%j_benchmark_tps_a100.err
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=aoae_tps_a100
#SBATCH --partition=kempner
#SBATCH --account=kempner_sham_lab

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
export MASTER_PORT=29508
export NCCL_SOCKET_FAMILY=AF_INET

CONFIG="${1:-configs/dual_mini_tau01.yaml}"
MAX_SAMPLES="${2:-10}"

mkdir -p logs outputs

echo "=== AOAE TPS Benchmark (A100) ==="
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
