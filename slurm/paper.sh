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

MODE="${1:-suite}"
if [ $# -gt 0 ]; then
  shift
fi

mkdir -p logs outputs results

echo "=== AOAE Paper / POC Workflow ==="
echo "Mode: $MODE"
echo "Node: $(hostname)"

case "$MODE" in
  suite)
    CONFIG="${1:-configs/paper.yaml}"
    if [ $# -gt 0 ]; then
      shift
    fi
    python3 -m aoae.cli paper-suite --config "$CONFIG" "$@"
    ;;
  poc1)
    CONFIG="${1:-configs/poc1.yaml}"
    if [ $# -gt 0 ]; then
      shift
    fi
    python3 -m aoae.cli tau-sweep --config "$CONFIG" "$@"
    ;;
  poc2)
    CONFIG="${1:-configs/poc2.yaml}"
    if [ $# -gt 0 ]; then
      shift
    fi
    python3 -m aoae.cli reuse-sweep --config "$CONFIG" "$@"
    ;;
  routing)
    HARD_CONFIG="${1:-configs/llada21_hard.yaml}"
    SOFT_CONFIG="${2:-configs/llada21_soft.yaml}"
    if [ $# -gt 0 ]; then
      shift
    fi
    if [ $# -gt 0 ]; then
      shift
    fi
    python3 -m aoae.cli routing-sweep --hard_config "$HARD_CONFIG" --soft_config "$SOFT_CONFIG" "$@"
    ;;
  ablations)
    CONFIG="${1:-configs/paper.yaml}"
    if [ $# -gt 0 ]; then
      shift
    fi
    python3 -m aoae.cli ablations --config "$CONFIG" "$@"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected one of: suite, poc1, poc2, routing, ablations" >&2
    exit 1
    ;;
esac

echo "=== Paper / POC workflow complete ==="
