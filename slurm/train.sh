#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_train.out
#SBATCH -e logs/%j_train.err
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=aoae_train
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

STAGE="${1:-prism}"
CONFIG="${2:-configs/default.yaml}"
RESUME="${3:-auto}"
shift 3 || true

DEFAULT_GPUS="$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(int(cfg.get("hardware", {}).get("tp_size", 1) or 1))
PY
)"
export GPUS_PER_NODE="${GPUS_PER_NODE:-$DEFAULT_GPUS}"

mkdir -p logs outputs

echo "=== AOAE Training ==="
echo "Stage:      $STAGE"
echo "Config:     $CONFIG"
echo "Resume:     $RESUME"
echo "GPUs/node:  $GPUS_PER_NODE"
echo "Node:       $(hostname)"

case "$STAGE" in
  prism)
    torchrun \
      --nproc_per_node "$GPUS_PER_NODE" \
      --nnodes 1 \
      --node_rank 0 \
      --master_addr "$MASTER_ADDR" \
      --master_port "$MASTER_PORT" \
      -m aoae.cli train --config "$CONFIG" --stage prism "$@"
    ;;
  grpo)
    torchrun \
      --nproc_per_node "$GPUS_PER_NODE" \
      --nnodes 1 \
      --node_rank 0 \
      --master_addr "$MASTER_ADDR" \
      --master_port "$MASTER_PORT" \
      -m aoae.cli train --config "$CONFIG" --stage grpo --resume "$RESUME" "$@"
    ;;
  pipeline)
    python3 -m aoae.cli pipeline --config "$CONFIG" --resume "$RESUME" "$@"
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Expected one of: prism, grpo, pipeline" >&2
    exit 1
    ;;
esac

echo "=== Training stage complete ==="
