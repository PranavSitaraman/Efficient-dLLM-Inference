#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_poc1_tau_sweep_blockwise_seas_gpu.out
#SBATCH -e logs/%j_poc1_tau_sweep_blockwise_seas_gpu.err
#SBATCH --partition=seas_gpu
#SBATCH --account=sitanc_lab
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --cpus-per-task=8
#SBATCH --time=09:00:00
#SBATCH --job-name=poc1_tau_swp_blk

set -euo pipefail

module load cuda
eval "$(conda shell.bash hook)"
conda activate rtx

export GPUS_PER_NODE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29540
export NCCL_SOCKET_FAMILY=AF_INET

CONFIG="configs/poc1_tau_sweep_blockwise.yaml"
MAX_SAMPLES="${1:-200}"
TAU_VALUES="${2:-0.001,0.005,0.01,0.05,0.1,0.2,0.5}"

mkdir -p logs outputs

echo "============================================================"
echo "POC 1 — Soft Routing Trade-off (blockwise, seas_gpu)"
echo "============================================================"
echo "Config:      $CONFIG"
echo "Max samples: $MAX_SAMPLES"
echo "tau_r values: $TAU_VALUES"
echo "Node:        $(hostname)"
echo "Started:     $(date)"
echo ""

nohup torchrun \
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
    --sweep_name "poc1_tau_sweep_blockwise" \
    --enable_remask \
    --no_resume   > log3_23.txt 2>&1

echo ""
echo "============================================================"
echo "POC 1 blockwise sweep complete — $(date)"
echo "Results: outputs/sweeps/poc1_tau_sweep_blockwise/"
echo "============================================================"

