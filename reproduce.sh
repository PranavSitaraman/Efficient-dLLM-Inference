#!/bin/bash
# ============================================================
# AOAE — Full Reproduction Script
#
# Usage:
#   Local (single GPU):   bash reproduce.sh
#   SLURM (multi-GPU):    bash reproduce.sh --slurm
#   Custom config:        bash reproduce.sh --config configs/llada21_mini.yaml
#
# Expected wall-clock:
#   LLaDA-8B on 1x A100:          ~4-8 hours
#   LLaDA2.1-mini on 2x H100:     ~3-6 hours
#   LLaDA2.1-flash on 8x H100:    ~6-12 hours
# ============================================================
set -e

# --- Parse arguments ---
CONFIG="configs/default.yaml"
USE_SLURM=false
for arg in "$@"; do
    case $arg in
        --slurm)      USE_SLURM=true; shift ;;
        --config)     shift; CONFIG="$1"; shift ;;
        --config=*)   CONFIG="${arg#*=}"; shift ;;
    esac
done

# Extract output_dir from config YAML
OUTPUT_DIR=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['logging']['output_dir'])" 2>/dev/null || echo "outputs/default/")

echo "=========================================="
echo " AOAE Reproduction Pipeline"
echo " Config:     $CONFIG"
echo " Output dir: $OUTPUT_DIR"
echo " SLURM:      $USE_SLURM"
echo "=========================================="

mkdir -p logs "$OUTPUT_DIR"

# --- Step 0: Environment check ---
echo ""
echo "[Step 0] Checking environment..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"

if $USE_SLURM; then
    # ==============================
    # SLURM submission mode
    # ==============================
    echo ""
    echo "[Step 1] Submitting PRISM training to SLURM..."
    PRISM_JOB=$(sbatch --parsable slurm/train_prism.sh "$CONFIG")
    echo "  PRISM job: $PRISM_JOB"

    echo ""
    echo "[Step 2] Submitting GRPO training (depends on PRISM)..."
    GRPO_JOB=$(sbatch --parsable --dependency=afterok:$PRISM_JOB slurm/train_grpo.sh "$CONFIG")
    echo "  GRPO job: $GRPO_JOB"

    echo ""
    echo "[Step 3] Submitting evaluation (depends on GRPO)..."
    EVAL_JOB=$(sbatch --parsable --dependency=afterok:$GRPO_JOB slurm/eval.sh "$CONFIG")
    echo "  Eval job: $EVAL_JOB"

    echo ""
    echo "=========================================="
    echo " All jobs submitted!"
    echo " Monitor: squeue -u \$USER"
    echo " PRISM:   $PRISM_JOB"
    echo " GRPO:    $GRPO_JOB"
    echo " Eval:    $EVAL_JOB"
    echo "=========================================="

else
    # ==============================
    # Local execution mode
    # ==============================

    # --- Step 1: Train PRISM adapter ---
    echo ""
    echo "[Step 1] Training PRISM quality adapter (10k samples, ~10 min)..."
    python3 run_train.py --config $CONFIG --stage prism
    echo "  PRISM adapter saved to $OUTPUT_DIR/prism_adapter.pt"

    # --- Step 2: Train AOAE policy via GRPO ---
    echo ""
    echo "[Step 2] Training AOAE policy via GRPO (~2-6 hours)..."
    python3 run_train.py --config $CONFIG --stage grpo
    echo "  Policy saved to $OUTPUT_DIR/policy_final.pt"

    # --- Step 3: Evaluate on GSM8K ---
    echo ""
    echo "[Step 3] Evaluating on GSM8K (baselines + AOAE Pareto sweep)..."
    python3 run_eval.py \
        --config $CONFIG \
        --checkpoint $OUTPUT_DIR/policy_final.pt

    # --- Step 4: Plot Pareto curves ---
    echo ""
    echo "[Step 4] Generating Pareto curves..."
    python3 -m aoae.plot_pareto \
        --results $OUTPUT_DIR/eval_results.json \
        --output $OUTPUT_DIR/pareto.png 2>/dev/null || echo "  (matplotlib not installed, skipping plot)"

    echo ""
    echo "=========================================="
    echo " Reproduction complete!"
    echo " Results:     $OUTPUT_DIR/eval_results.json"
    echo " Pareto plot: $OUTPUT_DIR/pareto.png"
    echo "=========================================="
fi
