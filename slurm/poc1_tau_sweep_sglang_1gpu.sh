#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_poc1_tau_sweep_sglang_1gpu.out
#SBATCH -e logs/%j_poc1_tau_sweep_sglang_1gpu.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=poc1_tau_sg1
#SBATCH --partition=kempner
#SBATCH --account=kempner_sham_lab

set -e

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

# Keep SGLang/sgl-kernel pointed at the cluster CUDA toolkit after conda activation.
if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
    CUDA_HOME="/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/cuda"
fi
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HUB_DISABLE_XET=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0"
export GPUS_PER_NODE=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29521
export NCCL_SOCKET_FAMILY=AF_INET

CONFIG="configs/poc1_tau_sweep_sglang_1gpu.yaml"
MAX_SAMPLES="${1:-200}"
TAU_VALUES="${2:-0.001,0.005,0.01,0.05,0.1,0.2,0.5}"

mkdir -p logs outputs

echo "============================================================"
echo "POC 1 — Soft Routing Trade-off Sweep (SGLang, 1 GPU)"
echo "============================================================"
echo "Config:      $CONFIG"
echo "Max samples: $MAX_SAMPLES"
echo "τ_r values:  $TAU_VALUES"
echo "Node:        $(hostname)"
echo "GPUs:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started:     $(date)"
echo ""

python scripts/run_tau_sweep.py \
    --config "$CONFIG" \
    --tau_r_values "$TAU_VALUES" \
    --max_samples "$MAX_SAMPLES" \
    --mode speculative \
    --sweep_name "poc1_tau_sweep_sglang_1gpu"

echo ""
echo "============================================================"
echo "POC 1 complete — $(date)"
echo "Results: outputs/sweeps/poc1_tau_sweep_sglang_1gpu/"
echo "============================================================"
