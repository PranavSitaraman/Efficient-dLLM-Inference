#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_poc2_reuse_signal.out
#SBATCH -e logs/%j_poc2_reuse_signal.err
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=poc2_reuse
#SBATCH --partition=kempner
#SBATCH --account=kempner_sham_lab

# ============================================================
# POC 2 — Training-free Agreement Signal for KV Reuse
#
# Fixed τ_r=0.1, sweeps 6 signal types × multiple thresholds.
# primary_every_n=1 ensures both logit distributions are
# available for accurate signal computation at every step.
# KV dynamics tracking enabled for thrash-rate diagnostics.
#
# Outputs: outputs/sweeps/poc2_reuse_signal_sweep/
#   - reuse_signal_sweep_full.{json,csv,md}
#   - best_method_by_constraint.{json,csv,md}
#   - reuse_signal_pareto.png, reuse_signal_cache_vs_acc.png
#   - Per-signal subdirectories with full eval artifacts
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
export MASTER_PORT=29511
export NCCL_SOCKET_FAMILY=AF_INET

CONFIG="configs/poc2_reuse_signal.yaml"
MAX_SAMPLES="${1:-100}"

mkdir -p logs outputs

echo "============================================================"
echo "POC 2 — Agreement Signal Sweep"
echo "============================================================"
echo "Config:      $CONFIG"
echo "Max samples: $MAX_SAMPLES"
echo "Node:        $(hostname)"
echo "GPUs:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started:     $(date)"
echo ""

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    scripts/run_reuse_signal_sweep.py \
    --config "$CONFIG" \
    --max_samples "$MAX_SAMPLES" \
    --mode speculative \
    --sweep_name "poc2_reuse_signal_sweep"

echo ""
echo "============================================================"
echo "POC 2 complete — $(date)"
echo "Results: outputs/sweeps/poc2_reuse_signal_sweep/"
echo "============================================================"
