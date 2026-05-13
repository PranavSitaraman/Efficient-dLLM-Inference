#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_paper.out
#SBATCH -e logs/%j_paper.err
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --job-name=aoae_repro
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

set -euo pipefail

WORKFLOW="smoke"
STAGE="all"
DATASET="gsm8k"
CONFIG=""
CHECKPOINT="auto"
MAX_SAMPLES=""
USE_SLURM=false
SITANC=false
STRICT_MOE=false
DRY_RUN="${AOAE_REPRO_DRY_RUN:-0}"
FORWARD_ARGS=()

DEFAULT_FINAL_CKPT="outputs/v4_grpo_quality_balanced/policy_best.pt"

usage() {
    cat <<'EOF'
AOAE reproduction wrapper

Supported workflows:
  bash reproduce.sh --workflow paper [--max_samples N] [--checkpoint auto|none|PATH]
  bash reproduce.sh --workflow train --stage warmstart|grpo|all
  bash reproduce.sh --workflow eval --dataset gsm8k|math500|humaneval [--checkpoint auto|none|PATH]
  bash reproduce.sh --workflow smoke

Common flags:
  --config PATH        Override the workflow default config.
  --slurm             Submit through the SLURM helper scripts.
  --sitanc            Use the sitanc/seas_gpu GRPO submission defaults.
  --dry_run           Print commands without executing them.
  --                  Forward remaining arguments to aoae.cli.
EOF
}

setup_env() {
    if command -v module >/dev/null 2>&1; then
        module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01 || true
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate rtx || true
    fi
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
    export GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-AF_INET}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

config_for_eval_dataset() {
    case "$1" in
        gsm8k) echo "configs/eval_gsm8k.yaml" ;;
        math500) echo "configs/eval_math500.yaml" ;;
        humaneval) echo "configs/eval_humaneval.yaml" ;;
        *)
            echo "Unknown eval dataset: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
}

default_config() {
    case "$WORKFLOW" in
        paper|train) echo "configs/paper.yaml" ;;
        eval) config_for_eval_dataset "$DATASET" ;;
        smoke) echo "configs/paper_smoke.yaml" ;;
        *)
            echo "Unknown workflow: $WORKFLOW" >&2
            usage >&2
            exit 2
            ;;
    esac
}

checkpoint_args() {
    local value="$1"
    if [[ "$value" == "none" || "$value" == "null" || -z "$value" ]]; then
        return 0
    fi
    if [[ "$value" == "auto" ]]; then
        if [[ -f "$DEFAULT_FINAL_CKPT" ]]; then
            printf '%s\n' --checkpoint "$DEFAULT_FINAL_CKPT"
        fi
        return 0
    fi
    printf '%s\n' --checkpoint "$value"
}

checkpoint_value_for_slurm() {
    local value="$1"
    if [[ "$value" == "none" || "$value" == "null" || -z "$value" ]]; then
        printf ''
        return 0
    fi
    if [[ "$value" == "auto" ]]; then
        if [[ -f "$DEFAULT_FINAL_CKPT" ]]; then
            printf '%s' "$DEFAULT_FINAL_CKPT"
        fi
        return 0
    fi
    printf '%s' "$value"
}

run_cmd() {
    echo "+ $*"
    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workflow) WORKFLOW="$2"; shift 2 ;;
        --workflow=*) WORKFLOW="${1#*=}"; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        --stage=*) STAGE="${1#*=}"; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset=*) DATASET="${1#*=}"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --max_samples=*) MAX_SAMPLES="${1#*=}"; shift ;;
        --slurm) USE_SLURM=true; shift ;;
        --sitanc) SITANC=true; shift ;;
        --strict_moe) STRICT_MOE=true; shift ;;
        --dry_run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            FORWARD_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$STAGE" != "warmstart" && "$STAGE" != "grpo" && "$STAGE" != "all" ]]; then
    echo "Unknown train stage: $STAGE" >&2
    usage >&2
    exit 2
fi

if [[ -z "$CONFIG" ]]; then
    CONFIG="$(default_config)"
fi

MAX_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_ARGS=(--max_samples "$MAX_SAMPLES")
fi

STRICT_ARGS=()
if $STRICT_MOE; then
    STRICT_ARGS=(--strict_moe)
fi

mkdir -p logs outputs results
setup_env

OUTPUT_DIR="$(python3 - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg.get("logging", {}).get("output_dir", str(Path("outputs") / Path("$CONFIG").stem)))
PY
)"

echo "=========================================="
echo " AOAE Reproduction Wrapper"
echo " Workflow:   $WORKFLOW"
echo " Config:     $CONFIG"
echo " Output dir: $OUTPUT_DIR"
echo " Checkpoint: $CHECKPOINT"
echo " SLURM:      $USE_SLURM"
echo "=========================================="

echo ""
echo "[Preflight] Checking environment..."
run_cmd python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
run_cmd python3 -m aoae.cli preflight --config "$CONFIG" "${STRICT_ARGS[@]}"

if $USE_SLURM; then
    case "$WORKFLOW" in
        paper)
            CMD=(sbatch --parsable slurm/paper.sh suite "$CONFIG" --checkpoint "$CHECKPOINT" "${MAX_ARGS[@]}" "${FORWARD_ARGS[@]}")
            ;;
        train)
            if [[ "$STAGE" == "all" ]]; then
                echo "+ sbatch --parsable slurm/train.sh warmstart $CONFIG fresh ${FORWARD_ARGS[*]-}"
                if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
                    WARM_JOB="DRYRUN"
                else
                    WARM_JOB="$(sbatch --parsable slurm/train.sh warmstart "$CONFIG" fresh "${FORWARD_ARGS[@]}")"
                fi
                CMD=(sbatch --parsable --dependency=afterok:"$WARM_JOB" slurm/train.sh grpo "$CONFIG" auto "${FORWARD_ARGS[@]}")
            else
                CMD=(sbatch --parsable slurm/train.sh "$STAGE" "$CONFIG" auto "${FORWARD_ARGS[@]}")
            fi
            ;;
        eval)
            CKPT="$(checkpoint_value_for_slurm "$CHECKPOINT")"
            CMD=(sbatch --parsable slurm/eval.sh "$CONFIG" "$CKPT" --mode speculative "${MAX_ARGS[@]}" "${FORWARD_ARGS[@]}")
            ;;
        smoke)
            CMD=(sbatch --parsable slurm/paper.sh suite "$CONFIG" --max_samples 0 --skip_eval --skip_table "${FORWARD_ARGS[@]}")
            ;;
        *)
            echo "Unknown workflow: $WORKFLOW" >&2
            exit 2
            ;;
    esac
    run_cmd "${CMD[@]}"
    echo "Submitted. Monitor with: squeue -u \$USER"
    exit 0
fi

case "$WORKFLOW" in
    smoke)
        run_cmd python3 -m aoae.cli eval --config "$CONFIG" --mode speculative --dry_run
        run_cmd python3 -m aoae.cli paper-suite --config "$CONFIG" --output_root outputs/paper_final_smoke --max_samples 0 --skip_eval --skip_table
        ;;
    paper)
        run_cmd python3 -m aoae.cli paper-suite --config "$CONFIG" --checkpoint "$CHECKPOINT" "${MAX_ARGS[@]}" "${FORWARD_ARGS[@]}"
        ;;
    train)
        if [[ "$STAGE" == "warmstart" || "$STAGE" == "all" ]]; then
            run_cmd python3 -m aoae.cli train --config "$CONFIG" --stage warmstart "${FORWARD_ARGS[@]}"
        fi
        if [[ "$STAGE" == "grpo" || "$STAGE" == "all" ]]; then
            run_cmd python3 -m aoae.cli train --config "$CONFIG" --stage grpo --resume auto "${FORWARD_ARGS[@]}"
        fi
        ;;
    eval)
        mapfile -t CKPT_ARGS < <(checkpoint_args "$CHECKPOINT")
        run_cmd python3 -m aoae.cli eval --config "$CONFIG" --mode speculative "${CKPT_ARGS[@]}" "${MAX_ARGS[@]}" "${FORWARD_ARGS[@]}"
        ;;
    *)
        echo "Unknown workflow: $WORKFLOW" >&2
        usage >&2
        exit 2
        ;;
esac

echo ""
echo "Workflow complete."
