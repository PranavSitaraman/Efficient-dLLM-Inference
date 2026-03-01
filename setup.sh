#!/bin/bash
# ============================================================
# AOAE Setup Script
# Installs all dependencies for Any-Order Adaptive Editing.
#
# Usage:
#   bash setup.sh              # Full install (core + dInfer + vLLM)
#   bash setup.sh --minimal    # Core deps only (HF backend, no dInfer)
# ============================================================
set -e

MINIMAL=false
for arg in "$@"; do
    case $arg in
        --minimal) MINIMAL=true ;;
    esac
done

echo "=== AOAE Setup ==="

# --- Create conda environment (if not exists) ---
if ! conda info --envs | grep -q "rtx"; then
    echo "Creating conda environment 'rtx' with Python 3.10..."
    conda create -n rtx python=3.10 -y
fi

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate rtx

# --- Install PyTorch (CUDA 12.1) ---
echo "Installing PyTorch..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Install core dependencies ---
echo "Installing core requirements..."
python3 -m pip install -r requirements.txt

# --- Clone external repos (if not already present) ---
echo ""
echo "Setting up external repos..."

if [ ! -d "external/dInfer/.git" ]; then
    echo "  Cloning dInfer..."
    rm -rf external/dInfer
    git clone https://github.com/inclusionAI/dInfer.git external/dInfer
else
    echo "  dInfer already cloned."
fi

if [ ! -d "external/dKV-Cache/.git" ]; then
    echo "  Cloning dKV-Cache..."
    rm -rf external/dKV-Cache
    git clone https://github.com/horseee/dKV-Cache.git external/dKV-Cache
else
    echo "  dKV-Cache already cloned."
fi

if ! $MINIMAL; then
    # --- Install vLLM (required for dInfer MoE models) ---
    echo ""
    echo "Installing vLLM (for dInfer/LLaDA2.X MoE backend)..."
    python3 -m pip install vllm 2>/dev/null || echo "  WARNING: vLLM install failed (needed only for 100B MoE model)"

    # --- Install dInfer (setup.py is in the root, python package in python/) ---
    echo ""
    echo "Installing dInfer..."
    python3 -m pip install -e external/dInfer 2>/dev/null || echo "  WARNING: dInfer install failed (needed only for 100B MoE model)"
fi

# --- Sanity check ---
echo ""
echo "Sanity check..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'  transformers {transformers.__version__}')"
python3 -c "import datasets; print(f'  datasets {datasets.__version__}')"
python3 -c "import yaml; print(f'  PyYAML OK')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Quick start (8B model — default config):"
echo "  1. Train PRISM adapter (single GPU, ~10 min):"
echo "     python3 run_train.py --config configs/default.yaml --stage prism"
echo ""
echo "  2. Train AOAE policy via GRPO (multi-GPU):"
echo "     torchrun --nproc_per_node 4 run_train.py --config configs/default.yaml --stage grpo"
echo ""
echo "  3. Evaluate:"
echo "     python3 run_eval.py --config configs/default.yaml --checkpoint outputs/default/policy_final.pt"
echo ""
echo "SLURM cluster:"
echo "  sbatch slurm/train_prism.sh"
echo "  sbatch slurm/train_grpo.sh"
echo "  sbatch slurm/eval.sh"
echo ""
echo "For 100B MoE model (requires vLLM + dInfer + 4 GPUs for TP):"
echo "  Use configs/default_large.yaml instead."
