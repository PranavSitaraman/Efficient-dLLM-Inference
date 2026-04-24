# Setup Guide

## Environment

```bash
module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
conda create -n rtx python=3.10 -y
conda activate rtx

bash setup.sh
pip install -e .
```

For dense / HF-only work:

```bash
bash setup.sh --minimal
pip install -e .
```

## Verify

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python3 -c "import transformers; print(transformers.__version__)"
aoae preflight --config configs/llada21_hard.yaml
aoae test
```

## Recommended configs

| Config | Use case |
| --- | --- |
| `configs/llada21_hard.yaml` | Main 8B training/eval path |
| `configs/paper.yaml` | Main paper suite |
| `configs/poc1.yaml` | PoC 1 tau sweep |
| `configs/poc2.yaml` | PoC 2 reuse sweep |
| `configs/llada21_hard.yaml` | Routing sweep hard baseline |
| `configs/llada21_soft.yaml` | Routing sweep soft config |

## Local usage

```bash
aoae pipeline --config configs/llada21_hard.yaml

aoae tau-sweep --config configs/poc1.yaml --max_samples 50
aoae reuse-sweep --config configs/poc2.yaml --max_samples 50
aoae paper-suite --config configs/paper.yaml --max_samples 50
```

If a config sets `hardware.tp_size > 1`, the local `aoae` CLI will relaunch itself under `torchrun` automatically.

## SLURM usage

Training / eval:

```bash
sbatch slurm/train.sh prism configs/llada21_hard.yaml
sbatch slurm/train.sh grpo configs/llada21_hard.yaml auto
sbatch slurm/eval.sh configs/llada21_hard.yaml outputs/llada21_hard/policy_final.pt
```

Paper / POCs:

```bash
sbatch slurm/paper.sh suite configs/paper.yaml --max_samples 50
sbatch slurm/paper.sh poc1 configs/poc1.yaml --max_samples 50
sbatch slurm/paper.sh poc2 configs/poc2.yaml --max_samples 50
sbatch slurm/paper.sh ablations configs/paper.yaml --max_samples 50
sbatch slurm/paper.sh routing configs/llada21_hard.yaml configs/llada21_soft.yaml --max_samples 50
```

Workflow wrapper:

```bash
bash reproduce.sh --slurm
bash reproduce.sh --slurm --workflow paper --max_samples 50
```
