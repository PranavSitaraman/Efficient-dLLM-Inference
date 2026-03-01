# Setup Guide — AOAE on Kempner H100 Cluster

## Quick Start (< 15 minutes)

```bash
# 1. Load modules
module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

# 2. Create environment
conda create -n rtx python=3.10 -y
conda activate rtx

# 3. Install everything (PyTorch + deps + dInfer + SGLang)
bash setup.sh

# 4. Authenticate with HuggingFace (needed for gated models)
huggingface-cli login
```

## Verify Installation

```bash
# Check core deps
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"

# Run unit tests (no GPU needed, uses mock model)
python3 -m pytest tests/ -v
```

## Model Access

| Model | Size | GPUs needed | Backend |
|-------|------|-------------|---------|
| `GSAI-ML/LLaDA-8B-Instruct` | 8B | 1 | HuggingFace |
| `inclusionAI/LLaDA2.1-mini` | 16B | 2 | dInfer/HF |
| `inclusionAI/LLaDA2.1-flash` | 100B MoE | 4 | dInfer+SGLang |
| `inclusionAI/LLaDA2.0-flash` | 100B MoE | 4 | dInfer+SGLang |
| `inclusionAI/LLaDA2.0-flash-CAP` | 100B MoE | 4 | dInfer+SGLang |

## Running on SLURM

```bash
# Full reproduction pipeline (PRISM → GRPO → Eval, chained jobs)
bash reproduce.sh --slurm

# With a specific config
bash reproduce.sh --slurm --config configs/llada21_mini.yaml

# Individual jobs
sbatch slurm/train_prism.sh configs/default.yaml
sbatch slurm/train_grpo.sh configs/default.yaml
sbatch slurm/eval.sh configs/default.yaml outputs/default/policy_final.pt
```

## Running Locally (single GPU)

```bash
# Full pipeline
bash reproduce.sh

# Or step by step:
python3 run_train.py --config configs/default.yaml --stage prism
python3 run_train.py --config configs/default.yaml --stage grpo
python3 run_eval.py --config configs/default.yaml --checkpoint outputs/default/policy_final.pt
```

## Troubleshooting

- **OOM on LLaDA-8B**: Reduce `inference.gen_length` or `grpo.group_size` in config.
- **dInfer import error**: Run `pip install -e external/dInfer` after `pip install sglang`.
- **Tokenizer error**: Make sure you ran `huggingface-cli login` with a valid token.
- **NCCL timeout**: Increase `NCCL_BLOCKING_WAIT` or check inter-node connectivity.