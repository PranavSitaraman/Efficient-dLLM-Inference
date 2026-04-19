#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_eval_.out
#SBATCH -e logs/%j_eval_.err
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=aoae_eval
#SBATCH --partition=seas_gpu
#SBATCH --account=sitanc_lab

# 1-GPU A100 baseline-eval variant of slurm/eval.sh.
#
# Usage:
#   sbatch slurm/eval_a100.sh [config] [extra eval args...]
#
# Examples:
# sbatch slurm/eval_h200.sh configs/llada21_hard.yaml \
#   --max_samples 50 \
#   --save_predictions \
#   --max_saved_predictions 50
#
#   sbatch --partition=gpu_a100 --account=kempner_sham_lab \
#     slurm/eval_a100.sh configs/llada21_soft.yaml --max_samples 50

set -euo pipefail

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
unset NCCL_BLOCKING_WAIT

CONFIG="${1:-configs/llada21_hard.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

mkdir -p logs outputs

echo "=== AOAE Baseline Evaluation (1x GPU) ==="
echo "Config:     $CONFIG"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "Account:    ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Node:       $(hostname)"
echo "Command:    python3 -m aoae.cli eval --config $CONFIG $*"

CMD=(python3 -m aoae.cli eval --config "$CONFIG")
CMD+=("$@")

"${CMD[@]}"

echo "=== Baseline evaluation complete ==="
