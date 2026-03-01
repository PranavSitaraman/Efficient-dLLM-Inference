#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j_sweep.out
#SBATCH -e logs/%j_sweep.err
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=aoae_sweep
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab

# ============================================================
# AOAE — τ_r Sweep Evaluation for Speculative Diffusion
#
# Evaluates the dual-model architecture across routing temperatures
# to characterize the quality-throughput Pareto frontier.
#
# Each config automatically runs:
#   - Block S-Mode baseline (LLaDA 2.1 semi-AR decoding)
#   - Block S-Mode + MBE baseline
#   - S-Mode (full-seq threshold) baseline
#   - Speculative AOAE at multiple τ_pi values
#
# Usage:
#   sbatch slurm/eval_sweep.sh                    # all τ_r configs
#   sbatch slurm/eval_sweep.sh configs/dual_mini_tau001.yaml  # single config
# Quick test: MAX_SAMPLES=10 sbatch slurm/eval_sweep.sh
# ============================================================
set -e

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
eval "$(conda shell.bash hook)"
conda activate rtx

export HF_HUB_DISABLE_XET=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_SAMPLES="${MAX_SAMPLES:-}"
mkdir -p logs outputs results

if [ -n "$1" ]; then
    CONFIGS=("$1")
else
    CONFIGS=(
        configs/dual_mini_tau001.yaml
        configs/dual_mini_tau01.yaml
        configs/dual_mini_tau05.yaml
    )
fi

echo "=== AOAE τ_r Sweep Evaluation ==="
echo "Configs: ${CONFIGS[*]}"
echo "Node:    $(hostname)"
[ -n "$MAX_SAMPLES" ] && echo "Max samples: $MAX_SAMPLES"

SAMPLES_FLAG=""
[ -n "$MAX_SAMPLES" ] && SAMPLES_FLAG="--max_samples $MAX_SAMPLES"

for CONFIG in "${CONFIGS[@]}"; do
    echo ""
    echo "--- Evaluating: $CONFIG ---"
    
    python3 run_eval.py --config "$CONFIG" --mode speculative $SAMPLES_FLAG
    
    echo "--- Done: $CONFIG ---"
done

echo ""
echo "=== τ_r sweep complete ==="
echo "Results saved to results/"
