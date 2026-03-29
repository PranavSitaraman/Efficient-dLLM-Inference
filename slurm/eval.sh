#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_eval.out
#SBATCH -e logs/%j_eval.err
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=aoae_eval
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

set -euo pipefail

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
unset NCCL_BLOCKING_WAIT

CONFIG="${1:-configs/default.yaml}"
CHECKPOINT="${2:-}"
shift 2 || true

mkdir -p logs outputs

echo "=== AOAE Evaluation ==="
echo "Config:     $CONFIG"
echo "Checkpoint: ${CHECKPOINT:-<auto>}"
echo "Node:       $(hostname)"

CMD=(python3 -m aoae.cli eval --config "$CONFIG")
if [ -n "$CHECKPOINT" ]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi
CMD+=("$@")

"${CMD[@]}"

echo "=== Evaluation complete ==="
