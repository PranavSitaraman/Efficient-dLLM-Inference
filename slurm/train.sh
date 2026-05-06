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

# PATH fix: ~/.local/bin/torchrun (Py3.12) shadows the rtx env's torchrun
# (Py3.10) on this cluster. Force conda env bin to the front so torch versions
# match. Without this the launcher fails to import torch (libcusparseLt.so.0).
export PATH="$CONDA_PREFIX/bin:$PATH"

# WandB defaults (override in shell as needed). To run without wandb,
# set logging.use_wandb=false in the config; the WANDB_* env vars then
# become no-ops.
export WANDB_ENTITY="${WANDB_ENTITY:-codeblock}"
export WANDB_PROJECT="${WANDB_PROJECT:-spec-dlm-grpo}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_SOCKET_FAMILY=AF_INET
if [[ "$MASTER_ADDR" == "localhost" ]]; then
  export MASTER_ADDR="127.0.0.1"
fi

STAGE="${1:-prism}"
CONFIG="${2:-configs/llada21_hard.yaml}"
RESUME="${3:-auto}"
if [[ $# -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

DEFAULT_GPUS="$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
hw = cfg.get("hardware", {})
# dp_size: total data-parallel GPUs (for GRPO DP runs); tp_size: tensor-parallel GPUs per group.
# Use dp_size if set, otherwise fall back to tp_size (legacy TP-only configs).
dp = int(hw.get("dp_size", 0) or 0)
tp = int(hw.get("tp_size", 1) or 1)
print(dp if dp > 0 else tp)
PY
)"
export GPUS_PER_NODE="${GPUS_PER_NODE:-$DEFAULT_GPUS}"

# Hard-fail if logging.use_wandb=true but WANDB_API_KEY is unset.
USE_WANDB="$(python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
v = cfg.get("logging", {}).get("use_wandb", False)
print(str(v).strip().lower())
PY
)"
if [[ "$USE_WANDB" == "true" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: ${CONFIG} has logging.use_wandb=true but WANDB_API_KEY is unset." >&2
    echo "Either:" >&2
    echo "  (a) export WANDB_API_KEY=<your-key> before launching, or" >&2
    echo "  (b) set logging.use_wandb: false in ${CONFIG}." >&2
    exit 2
fi

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
  warmstart)
    torchrun \
      --nproc_per_node "$GPUS_PER_NODE" \
      --nnodes 1 \
      --node_rank 0 \
      --master_addr "$MASTER_ADDR" \
      --master_port "$MASTER_PORT" \
      -m aoae.cli train --config "$CONFIG" --stage warmstart "$@"
    ;;
  pipeline)
    python3 -m aoae.cli pipeline --config "$CONFIG" --resume "$RESUME" "$@"
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Expected one of: prism, warmstart, grpo, pipeline" >&2
    exit 1
    ;;
esac

echo "=== Training stage complete ==="
