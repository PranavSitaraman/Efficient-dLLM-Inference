#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_poc1_tau_sweep.out
#SBATCH -e logs/%j_poc1_tau_sweep.err
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=poc1_tau
#SBATCH --partition=kempner
#SBATCH --account=kempner_sham_lab

# ============================================================
# POC 1 — Soft Routing Trade-off
#
# Sweeps τ_r ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5}
# with primary_every_n=1 to isolate the pure routing effect.
# Baselines (block_smode, etc.) run on the first sweep point.
#
# Outputs: outputs/sweeps/poc1_tau_sweep/
#   - tau_sweep_summary.{json,csv,md}
#   - tau_sweep_vs_tau.png, tau_sweep_pareto.png
#   - Per-τ_r subdirectories with full eval artifacts
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
export MASTER_PORT=29510
export NCCL_SOCKET_FAMILY=AF_INET

CONFIG="configs/poc1_tau_sweep.yaml"
MAX_SAMPLES="${1:-200}"
TAU_VALUES="${2:-0.001,0.005,0.01,0.05,0.1,0.2,0.5}"

mkdir -p logs outputs

echo "============================================================"
echo "POC 1 — Soft Routing Trade-off Sweep"
echo "============================================================"
echo "Config:      $CONFIG"
echo "Max samples: $MAX_SAMPLES"
echo "τ_r values:  $TAU_VALUES"
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
    scripts/run_tau_sweep.py \
    --config "$CONFIG" \
    --tau_r_values "$TAU_VALUES" \
    --max_samples "$MAX_SAMPLES" \
    --mode speculative \
    --sweep_name "poc1_tau_sweep"

echo ""
echo "============================================================"
echo "POC 1 complete — $(date)"
echo "Results: outputs/sweeps/poc1_tau_sweep/"
echo "============================================================"
