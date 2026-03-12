#!/bin/bash
#SBATCH -J diagnose_gen
#SBATCH -o logs/%j_diagnose.out
#SBATCH -e logs/%j_diagnose.err
#SBATCH -p kempner
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 01:00:00

set -euo pipefail

source ~/.bashrc
conda activate rtx

cd /n/holylabs/ydu_lab/Lab/pranavsitaraman/Efficient-dLLM-Inference

export TORCH_CUDA_ARCH_LIST="8.0"
export VLLM_USE_FLASHINFER_MOE_FP16=1
export HF_HOME=/n/holylabs/LABS/kempner_dev/Everyone/hf_cache

echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"

torchrun --nproc_per_node=2 scripts/diagnose_generation.py
