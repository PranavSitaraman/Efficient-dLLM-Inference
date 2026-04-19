#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_eval_tp2.out
#SBATCH -e logs/%j_eval_tp2.err
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --job-name=aoae_eval_tp2
#SBATCH --partition=seas_gpu
#SBATCH --account=sitanc_lab

# 2-GPU tensor-parallel baseline eval wrapper.
#
# This script follows the public dInfer multi-GPU launch style more closely
# than torchrun: it spawns one worker process per local rank with
# RANK/LOCAL_RANK/WORLD_SIZE exported explicitly. AOAE's auto-torchrun
# relaunch is disabled so distributed is initialized only once in-process.
#
# Usage:
#   sbatch slurm/eval_h200_tp2.sh [config] [extra eval args...]
#
# Example:
#   sbatch slurm/eval_h200_tp2.sh configs/llada21_hard.yaml \
#     --max_samples 2 \
#     --save_predictions \
#     --max_saved_predictions 2

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

TMP_CONFIG="$(mktemp /tmp/aoae_eval_tp2.XXXXXX.yaml)"
cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

python3 - "$CONFIG" "$TMP_CONFIG" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("hardware", {})["tp_size"] = 2

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + (${SLURM_JOB_ID:-0} % 1000)))}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-AF_INET}"
export AOAE_DISABLE_AUTO_TORCHRUN=1

echo "=== AOAE Baseline Evaluation (2x GPU torchrun) ==="
echo "Config:         $CONFIG"
echo "Temp config:    $TMP_CONFIG"
echo "Partition:      ${SLURM_JOB_PARTITION:-unknown}"
echo "Account:        ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Node:           $(hostname)"
echo "MASTER_ADDR:    $MASTER_ADDR"
echo "MASTER_PORT:    $MASTER_PORT"
echo "Command:        python3 -m aoae.cli eval --config $TMP_CONFIG $*"

pids=()
statuses=()

for local_rank in 0 1; do
  (
    export RANK="$local_rank"
    export LOCAL_RANK="$local_rank"
    export WORLD_SIZE=2
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    python3 -m aoae.cli eval --config "$TMP_CONFIG" "$@"
  ) &
  pids+=("$!")
done

for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "$pid"; then
    statuses+=(0)
  else
    statuses+=($?)
  fi
done

for status in "${statuses[@]}"; do
  if [[ "$status" -ne 0 ]]; then
    echo "A rank exited with status $status"
    exit "$status"
  fi
done

echo "=== 2-GPU baseline evaluation complete ==="
