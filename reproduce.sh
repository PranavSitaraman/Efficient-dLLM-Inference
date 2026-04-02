#!/bin/bash
set -euo pipefail

# Usage:
#   bash reproduce.sh
#   bash reproduce.sh --workflow paper --max_samples 100
#   bash reproduce.sh --slurm --workflow poc1 --max_samples 100
#   bash reproduce.sh --workflow routing -- --tau_r_values 0.01,0.05,0.1

WORKFLOW="pipeline"
USE_SLURM=false
CONFIG=""
CHECKPOINT=""
MAX_SAMPLES=""
STRICT_MOE=false
FORWARD_ARGS=()

resolve_default_config() {
    case "$1" in
        pipeline)  echo "configs/paper.yaml" ;;
        paper)     echo "configs/paper.yaml" ;;
        poc1)      echo "configs/poc1.yaml" ;;
        poc2)      echo "configs/poc2.yaml" ;;
        ablations) echo "configs/paper.yaml" ;;
        routing)   echo "configs/llada21_hard.yaml" ;;
        *)
            echo "Unknown workflow: $1" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)
            USE_SLURM=true
            shift
            ;;
        --workflow)
            WORKFLOW="$2"
            shift 2
            ;;
        --workflow=*)
            WORKFLOW="${1#*=}"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG="${1#*=}"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT="${1#*=}"
            shift
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --max_samples=*)
            MAX_SAMPLES="${1#*=}"
            shift
            ;;
        --strict_moe)
            STRICT_MOE=true
            shift
            ;;
        --)
            shift
            FORWARD_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use -- to forward extra CLI flags." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    CONFIG="$(resolve_default_config "$WORKFLOW")"
fi

STRICT_ARGS=()
if $STRICT_MOE; then
    STRICT_ARGS+=(--strict_moe)
fi

COMMON_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
    COMMON_ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
    COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
COMMON_ARGS+=("${FORWARD_ARGS[@]}")

PIPELINE_EVAL_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    PIPELINE_EVAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
PIPELINE_EVAL_ARGS+=("${FORWARD_ARGS[@]}")

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
echo " SLURM:      $USE_SLURM"
echo "=========================================="

mkdir -p logs outputs results

echo ""
echo "[Preflight] Checking environment..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"
python3 -m aoae.cli preflight --config "$CONFIG" "${STRICT_ARGS[@]}" || true

if $USE_SLURM; then
    case "$WORKFLOW" in
        pipeline)
            echo ""
            echo "[Submit] PRISM -> GRPO -> Eval"
            PRISM_JOB=$(sbatch --parsable slurm/train.sh prism "$CONFIG")
            GRPO_JOB=$(sbatch --parsable --dependency=afterok:$PRISM_JOB slurm/train.sh grpo "$CONFIG" auto)
            EVAL_JOB=$(sbatch --parsable --dependency=afterok:$GRPO_JOB slurm/eval.sh "$CONFIG" "$CHECKPOINT" "${PIPELINE_EVAL_ARGS[@]}")
            echo "PRISM job: $PRISM_JOB"
            echo "GRPO job:  $GRPO_JOB"
            echo "Eval job:  $EVAL_JOB"
            ;;
        paper)
            JOB=$(sbatch --parsable slurm/paper.sh suite "$CONFIG" "${COMMON_ARGS[@]}")
            echo "Paper-suite job: $JOB"
            ;;
        poc1)
            JOB=$(sbatch --parsable slurm/paper.sh poc1 "$CONFIG" "${COMMON_ARGS[@]}")
            echo "PoC1 job: $JOB"
            ;;
        poc2)
            JOB=$(sbatch --parsable slurm/paper.sh poc2 "$CONFIG" "${COMMON_ARGS[@]}")
            echo "PoC2 job: $JOB"
            ;;
        ablations)
            JOB=$(sbatch --parsable slurm/paper.sh ablations "$CONFIG" "${COMMON_ARGS[@]}")
            echo "Ablations job: $JOB"
            ;;
        routing)
            JOB=$(sbatch --parsable slurm/paper.sh routing configs/llada21_hard.yaml configs/llada21_soft.yaml "${FORWARD_ARGS[@]}")
            echo "Routing-sweep job: $JOB"
            ;;
        *)
            echo "Unknown workflow: $WORKFLOW" >&2
            exit 1
            ;;
    esac

    echo ""
    echo "Submitted. Monitor with: squeue -u \$USER"
    exit 0
fi

case "$WORKFLOW" in
    pipeline)
        python3 -m aoae.cli pipeline --config "$CONFIG" --skip_preflight "${COMMON_ARGS[@]}"
        ;;
    paper)
        python3 -m aoae.cli paper-suite --config "$CONFIG" "${COMMON_ARGS[@]}"
        ;;
    poc1)
        python3 -m aoae.cli tau-sweep --config "$CONFIG" "${COMMON_ARGS[@]}"
        ;;
    poc2)
        python3 -m aoae.cli reuse-sweep --config "$CONFIG" "${COMMON_ARGS[@]}"
        ;;
    ablations)
        python3 -m aoae.cli ablations --config "$CONFIG" "${COMMON_ARGS[@]}"
        ;;
    routing)
        python3 -m aoae.cli routing-sweep \
            --hard_config configs/llada21_hard.yaml \
            --soft_config configs/llada21_soft.yaml \
            "${FORWARD_ARGS[@]}"
        ;;
    *)
        echo "Unknown workflow: $WORKFLOW" >&2
        exit 1
        ;;
esac

echo ""
echo "Workflow complete."
