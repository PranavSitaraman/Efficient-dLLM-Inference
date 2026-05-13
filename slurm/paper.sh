#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_paper.out
#SBATCH -e logs/%j_paper.err
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=aoae_paper
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

set -euo pipefail

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

MODE="${1:-suite}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ "$MODE" != "suite" ]]; then
  echo "Unknown paper mode: $MODE" >&2
  echo "Expected: suite" >&2
  exit 2
fi

CONFIG="${1:-configs/paper.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

mkdir -p logs outputs results

echo "=== AOAE Final Paper Suite ==="
echo "Config: $CONFIG"
echo "Node:   $(hostname)"

python3 -m aoae.cli paper-suite --config "$CONFIG" "$@"

echo "=== Paper suite complete ==="
