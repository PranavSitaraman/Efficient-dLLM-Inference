#!/bin/bash
# ============================================================
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
echo "Quick start:"
echo "  1. Preflight:"
echo "     python3 -m aoae.cli preflight --config configs/paper.yaml"
echo ""
echo "  2. Smoke reproduction:"
echo "     bash reproduce.sh --workflow smoke"
echo ""
echo "  3. Train scalar AOAE policy:"
echo "     bash reproduce.sh --workflow train --stage all"
echo ""
echo "  4. Evaluate final datasets:"
echo "     bash reproduce.sh --workflow paper --max_samples 50"
echo ""
echo "Multi-GPU note:"
echo "  Commands using configs with hardware.tp_size > 1 auto-relaunch under torchrun."
echo ""
echo "Paper workflow:"
echo "  python3 -m aoae.cli paper-suite --config configs/paper.yaml --max_samples 50"
echo ""
echo "SLURM cluster:"
echo "  bash reproduce.sh --slurm --workflow train --stage all"
echo "  bash reproduce.sh --slurm --workflow eval --dataset gsm8k --checkpoint auto"
echo "  bash reproduce.sh --slurm --workflow paper --max_samples 50"
