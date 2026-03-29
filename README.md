# AOAE

Reference implementation for Any-Order Adaptive Editing (AOAE) on masked diffusion LLMs, with a simplified repo surface centered on one package, one CLI, a small canonical config set, and three generic SLURM wrappers.

## What is in the repo

```text
Efficient-dLLM-Inference/
├── aoae/
│   ├── cli.py            # Canonical entrypoint: train, eval, pipeline, paper-suite, sweeps
│   ├── paper.py          # PoC 1, PoC 2, routing sweep, ablations, paper suite
│   ├── reporting.py      # Comparison-table and KV-summary artifact aggregation
│   ├── checkpoints.py    # Shared checkpoint resolution helpers
│   ├── tasks.py          # Shared prompt / answer parsing helpers
│   ├── evaluate.py       # Evaluation loop and artifact writing
│   ├── train_prism.py    # PRISM stage
│   ├── train_grpo.py     # AOAE policy training
│   └── models/           # Base model wrappers, policy, PRISM, soft routing, dual model
├── configs/
│   ├── default.yaml      # Main 8B dense training/eval config
│   ├── paper.yaml        # Main paper config
│   ├── poc1.yaml         # PoC 1 soft-routing tradeoff sweep
│   ├── poc2.yaml         # PoC 2 reuse-signal sweep
│   ├── llada21_hard.yaml # Hard-routing routing-sweep baseline
│   ├── llada21_soft.yaml # Soft-routing routing-sweep config
│   └── llada21_flash.yaml# Large MoE / dInfer config
├── slurm/
│   ├── train.sh          # Generic PRISM / GRPO / pipeline job
│   ├── eval.sh           # Generic eval job
│   └── paper.sh          # Generic paper / POC job
├── paper/                # Paper source
├── tests/                # Unit + integration tests
├── reproduce.sh          # Local or SLURM orchestration wrapper
└── setup.sh              # Environment bootstrap
```

Generated artifacts live in `outputs/`, `logs/`, and `results/` and are gitignored.

## Canonical configs

| Config | Purpose |
| --- | --- |
| `configs/default.yaml` | Main single-run training and eval path on `GSAI-ML/LLaDA-8B-Instruct` |
| `configs/paper.yaml` | Main paper-oriented config for integrated experiments |
| `configs/poc1.yaml` | PoC 1: soft-routing speed/quality tradeoff |
| `configs/poc2.yaml` | PoC 2: training-free KV-reuse agreement signal study |
| `configs/llada21_hard.yaml` | Routing sweep hard-routing reference |
| `configs/llada21_soft.yaml` | Routing sweep soft-routing counterpart |
| `configs/llada21_flash.yaml` | Large dInfer / MoE runtime config |

## Install

```bash
bash setup.sh
pip install -e .
```

Minimal install without dInfer / large-MoE extras:

```bash
bash setup.sh --minimal
pip install -e .
```

## Quick start

Environment and runtime check:

```bash
aoae preflight --config configs/default.yaml
```

Run baseline eval only:

```bash
aoae eval --config configs/default.yaml --max_samples 100
```

Train end to end:

```bash
aoae train --config configs/default.yaml --stage prism
aoae train --config configs/default.yaml --stage grpo
aoae eval --config configs/default.yaml --checkpoint outputs/default/policy_final.pt
```

Run the integrated local pipeline:

```bash
aoae pipeline --config configs/default.yaml
```

For configs with `hardware.tp_size > 1`, `aoae` now auto-relaunches itself under `torchrun` with the same local environment defaults used by the SLURM wrappers.

## Paper and POC workflows

PoC 1 soft-routing tradeoff:

```bash
aoae tau-sweep --config configs/poc1.yaml --max_samples 100
```

PoC 2 reuse-signal study:

```bash
aoae reuse-sweep --config configs/poc2.yaml --max_samples 100
```

Routing-only hard vs soft comparison:

```bash
aoae routing-sweep \
  --hard_config configs/llada21_hard.yaml \
  --soft_config configs/llada21_soft.yaml \
  --max_samples 100
```

Ablation matrix:

```bash
aoae ablations --config configs/paper.yaml --max_samples 100
```

Run the full paper suite:

```bash
aoae paper-suite --config configs/paper.yaml --max_samples 100
```

Aggregate saved artifacts:

```bash
aoae comparison-table
aoae kv-summary
```

## Reproduction wrapper

`reproduce.sh` is now workflow-oriented instead of hardwired to one legacy path.

Local:

```bash
bash reproduce.sh
bash reproduce.sh --workflow paper --max_samples 100
bash reproduce.sh --workflow poc1 --max_samples 100
bash reproduce.sh --workflow poc2 --max_samples 100
```

SLURM:

```bash
bash reproduce.sh --slurm
bash reproduce.sh --slurm --workflow paper --max_samples 100
bash reproduce.sh --slurm --workflow ablations --max_samples 100
```

Supported workflows:

| Workflow | Local command | SLURM wrapper |
| --- | --- | --- |
| `pipeline` | `aoae pipeline` | `slurm/train.sh` + `slurm/eval.sh` |
| `paper` | `aoae paper-suite` | `slurm/paper.sh suite` |
| `poc1` | `aoae tau-sweep` | `slurm/paper.sh poc1` |
| `poc2` | `aoae reuse-sweep` | `slurm/paper.sh poc2` |
| `routing` | `aoae routing-sweep` | `slurm/paper.sh routing` |
| `ablations` | `aoae ablations` | `slurm/paper.sh ablations` |

The local CLI will automatically use `torchrun` for these workflows when the selected config requires multi-process tensor parallelism.

## Important commands

```bash
aoae train --config configs/default.yaml --stage prism
aoae train --config configs/default.yaml --stage grpo --resume auto
aoae eval --config configs/default.yaml --checkpoint outputs/default/policy_final.pt
aoae pipeline --config configs/default.yaml

aoae tau-sweep --config configs/poc1.yaml
aoae reuse-sweep --config configs/poc2.yaml
aoae routing-sweep --hard_config configs/llada21_hard.yaml --soft_config configs/llada21_soft.yaml
aoae ablations --config configs/paper.yaml
aoae paper-suite --config configs/paper.yaml
aoae comparison-table
aoae kv-summary
```

## Artifact layout

Single eval runs write:

- `outputs/<run>/eval_results.json`
- `outputs/<run>/eval_metadata.json`
- `outputs/<run>/eval_tps_vs_accuracy.png`
- `outputs/<run>/eval_predictions.json` when prediction saving is enabled
- `outputs/<run>/kv_dynamics_*.json|png` when KV tracking is enabled

Paper/POC workflows additionally write sweep summaries under:

- `outputs/sweeps/...`
- `outputs/ablations/...`
- `outputs/paper_suite/...`

Aggregated tables write to:

- `results/comparison_table.csv`
- `results/comparison_table.md`
- `results/kv_dynamics_table.csv`
- `results/kv_dynamics_table.md`

## Testing

Run the full suite:

```bash
pytest -q
```

Run a focused subset:

```bash
pytest tests/test_cli_integration.py tests/test_poc1_tau_sweep.py tests/test_poc2_reuse_signals.py -q
```

## Notes

- Legacy `run_*.py`, sweep-specific shell wrappers, and redundant YAML variants have been consolidated or removed.
- The supported command surface is the `aoae` CLI plus the three generic scripts in `slurm/`.
- If you need custom sweep parameters, pass them directly to the CLI instead of creating new one-off scripts.
