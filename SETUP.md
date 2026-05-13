# AOAE Setup

## Environment

On the Kempner cluster:

```bash
module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
conda activate rtx
```

For a fresh local environment:

```bash
bash setup.sh --minimal
```

Then validate:

```bash
python3 -m aoae.cli preflight --config configs/paper.yaml
bash reproduce.sh --workflow smoke
```

## Maintained Configs

The final deliverable keeps exactly six configs:

- `configs/paper.yaml`: canonical warmstart -> GRPO training.
- `configs/paper_smoke.yaml`: tiny smoke path.
- `configs/eval_gsm8k.yaml`: in-distribution GSM8K eval.
- `configs/eval_math500.yaml`: out-of-distribution MATH-500 eval.
- `configs/eval_humaneval.yaml`: out-of-distribution HumanEval code eval.
- `configs/ablation.yaml`: compact final ablations.

`paper.yaml` trains scalar-only Phase-A/V2 `u/r` heads and disables stable-cache,
cache reward, and access reward terms in the canonical GRPO run.

## Reproduction

```bash
bash reproduce.sh --workflow paper --max_samples 50
bash reproduce.sh --workflow train --stage warmstart
bash reproduce.sh --workflow train --stage grpo
bash reproduce.sh --workflow eval --dataset gsm8k --checkpoint auto
bash reproduce.sh --workflow eval --dataset math500 --checkpoint auto
bash reproduce.sh --workflow eval --dataset humaneval --checkpoint auto
```

`--checkpoint auto` uses `outputs/v4_grpo_quality_balanced/policy_best.pt` when
available. Use `--checkpoint none` for the no-train heuristic policy.

## SLURM

```bash
bash reproduce.sh --slurm --workflow paper --max_samples 50
bash reproduce.sh --slurm --workflow train --stage all
bash reproduce.sh --slurm --workflow eval --dataset gsm8k --checkpoint auto
```

The wrapper delegates to `slurm/paper.sh`, `slurm/train.sh`, and `slurm/eval.sh`.

## Verification

```bash
pytest tests/test_config_contracts.py tests/test_cli_integration.py tests/test_code_eval.py -q
pytest tests/test_paper_suite_smoke.py tests/test_reporting_commands.py -q
pytest -q
```
