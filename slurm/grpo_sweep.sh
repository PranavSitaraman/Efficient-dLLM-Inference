#!/bin/bash
# GRPO hyperparameter sweep launcher.
#
# Submits one SLURM job per trial. Each trial gets a unique config YAML
# (generated from configs/llada21_hard.yaml with overrides) and a unique output_dir.
#
# Usage:
#   bash slurm/grpo_sweep.sh [--partition <p>] [--account <a>] [--dry-run]
#
# Default sweep: alpha × cache_quality_weight (3×3 = 9 trials).
# Edit the ALPHA_VALUES and CACHE_Q_VALUES arrays below to change the grid.
# Add more nested loops to sweep additional axes.
#
# Results land in:  outputs/sweeps/grpo_sweep_<timestamp>/<trial_name>/
# Logs land in:     logs/<jobid>_train.{out,err}
#
# Hardware:
#   Default partition: gpu_a100 (A100 nodes, 1 GPU per trial, 8h)
#   For H100: pass --partition kempner_h100
#
# Each trial retrains from scratch (resume=fresh). The existing checkpoint
# must NOT be reused — it was trained without positional cache features or the
# updated reward function.

set -euo pipefail

# ---- Sweep grid -------------------------------------------------------
ALPHA_VALUES=(0.5 1.0 2.0)
CACHE_Q_VALUES=(0.0 0.05 0.1)

# ---- Defaults ---------------------------------------------------------
BASE_CONFIG="configs/llada21_hard.yaml"
PARTITION="${GRPO_PARTITION:-gpu_a100}"
ACCOUNT="${GRPO_ACCOUNT:-kempner_sham_lab}"
DRY_RUN=0
SWEEP_DIR="outputs/sweeps/grpo_sweep_$(date +%Y%m%d_%H%M%S)"

# ---- Parse flags ------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --account)   ACCOUNT="$2";   shift 2 ;;
    --dry-run)   DRY_RUN=1;      shift   ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$SWEEP_DIR" logs

echo "=== GRPO Sweep ==="
echo "Base config:  $BASE_CONFIG"
echo "Sweep dir:    $SWEEP_DIR"
echo "Partition:    $PARTITION"
echo "Account:      $ACCOUNT"
echo "Grid:         alpha=${ALPHA_VALUES[*]}  cache_quality=${CACHE_Q_VALUES[*]}"
echo ""

# ---- Generate configs and submit jobs ---------------------------------
TRIAL_COUNT=0
for ALPHA in "${ALPHA_VALUES[@]}"; do
  for CACHE_Q in "${CACHE_Q_VALUES[@]}"; do

    TRIAL_NAME="alpha${ALPHA}_cacheq${CACHE_Q}"
    TRIAL_DIR="${SWEEP_DIR}/${TRIAL_NAME}"
    TRIAL_CONFIG="${TRIAL_DIR}/config.yaml"

    mkdir -p "$TRIAL_DIR"

    # Generate trial config by overlaying overrides onto base config
    python3 - <<PY
import yaml, copy

with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)

# Apply hyperparameter overrides
cfg.setdefault("grpo", {})["alpha"] = float("$ALPHA")
cfg.setdefault("grpo", {})["cache_quality_weight"] = float("$CACHE_Q")

# Point output to trial-specific directory
cfg.setdefault("logging", {})["output_dir"] = "$TRIAL_DIR/"
cfg.setdefault("logging", {})["run_name"] = "$TRIAL_NAME"

with open("$TRIAL_CONFIG", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"[Config] Written: $TRIAL_CONFIG")
PY

    echo "--- Trial: $TRIAL_NAME ---"
    echo "  alpha=$ALPHA  cache_quality_weight=$CACHE_Q"
    echo "  config: $TRIAL_CONFIG"

    SBATCH_CMD=(
      sbatch
      --partition="$PARTITION"
      --account="$ACCOUNT"
      --job-name="grpo_${TRIAL_NAME}"
      slurm/train_a100.sh grpo "$TRIAL_CONFIG" fresh
    )

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "  [DRY RUN] Would submit: ${SBATCH_CMD[*]}"
    else
      JOB_ID=$("${SBATCH_CMD[@]}" | awk '{print $NF}')
      echo "  Submitted job $JOB_ID"
    fi

    TRIAL_COUNT=$((TRIAL_COUNT + 1))
  done
done

echo ""
echo "=== Sweep submitted: $TRIAL_COUNT trials ==="
echo "Results: $SWEEP_DIR"
echo "Monitor: squeue -u \$USER"
