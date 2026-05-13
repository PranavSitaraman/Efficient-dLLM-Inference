# Efficient dLLM Inference: AOAE

This repository contains the submission-ready AOAE reproduction pipeline for
speculative inference on LLaDA2.1. The final paper story is intentionally small:

- Speculative AOAE creates a controllable speed/accuracy tradeoff.
- A semi-any-order policy is more expressive than block-only decoding, with a
  speed cost.
- GRPO on GSM8K pushes the built frontier mostly upward in accuracy and modestly
  rightward in throughput.

The default final checkpoint is the empirically supported V4 quality-balanced
scalar-only policy at `outputs/v4_grpo_quality_balanced/policy_best.pt` when that
artifact is present. V5 experiments remain non-default until full evaluations
beat the V4 frontier.

## Repository Surface

`configs/` is deliberately limited to six maintained YAML files:

| Config | Purpose |
| --- | --- |
| `configs/paper.yaml` | Canonical warmstart -> GRPO training config, scalar-only `u/r`, cache/access rewards disabled |
| `configs/paper_smoke.yaml` | Tiny CI/smoke variant of `paper.yaml` |
| `configs/eval_gsm8k.yaml` | Full GSM8K eval with the final shared AOAE sweep |
| `configs/eval_math500.yaml` | Full MATH-500 out-of-distribution eval |
| `configs/eval_humaneval.yaml` | Full HumanEval code eval using `CodeEvaluator` |
| `configs/ablation.yaml` | Final compact ablation matrix: hard/soft verifier, trained/no-train, block vs semi-any-order |

Historical config settings and preliminary numbers are summarized in
`docs/final_results.md`; the old config files are not part of the final public
surface.

## Quick Start

```bash
bash setup.sh --minimal
bash reproduce.sh --workflow smoke
```

The smoke workflow runs preflight, validates the smoke config, and exercises the
paper-suite manifest path without loading the full model.

## Final Reproduction Commands

```bash
# Final paper loop: GSM8K, MATH-500, HumanEval, ablation, aggregate table.
bash reproduce.sh --workflow paper --max_samples 50

# Train from scratch into outputs/paper_final/train/.
bash reproduce.sh --workflow train --stage warmstart
bash reproduce.sh --workflow train --stage grpo
bash reproduce.sh --workflow train --stage all

# Dataset-specific evals. checkpoint=auto uses the V4 default if present.
bash reproduce.sh --workflow eval --dataset gsm8k --checkpoint auto
bash reproduce.sh --workflow eval --dataset math500 --checkpoint auto
bash reproduce.sh --workflow eval --dataset humaneval --checkpoint auto
```

Use `--checkpoint none` to run the training-free heuristic AOAE policy, or pass
an explicit checkpoint path to evaluate another policy. Use `--slurm` with the
same wrapper commands to submit through `slurm/train.sh`, `slurm/eval.sh`, and
`slurm/paper.sh`.

## Direct CLI

The wrapper is the preferred interface, but the underlying CLI remains available:

```bash
python3 -m aoae.cli preflight --config configs/paper.yaml
python3 -m aoae.cli train --config configs/paper.yaml --stage warmstart
python3 -m aoae.cli train --config configs/paper.yaml --stage grpo --resume auto
python3 -m aoae.cli eval --config configs/eval_gsm8k.yaml --mode speculative --checkpoint auto
python3 -m aoae.cli paper-suite --config configs/paper.yaml --max_samples 50
```

## Outputs

Final artifacts are written under `outputs/paper_final/`:

- `gsm8k/trained`, `gsm8k/notrain`, and matching `math500/` and
  `humaneval/` folders: per-dataset `eval_results.json`,
  `eval_metadata.json`, optional predictions, and Pareto plots.
- `ablations/trained` and `ablations/notrain`: compact ablation results.
- `aggregate_comparison_table.csv` and `.md`: final table inputs for the paper.
- `run_manifest.json`: exact configs, checkpoint, sample cap, and output roots.

The global append-only `outputs/experiment_manifest.jsonl` records individual
evaluation rows across runs.

## Tests

```bash
pytest tests/test_config_contracts.py tests/test_cli_integration.py tests/test_code_eval.py -q
pytest tests/test_paper_suite_smoke.py tests/test_reporting_commands.py -q
pytest -q
```

The contract tests assert that only the six canonical configs exist and that all
load through the public config loader. CLI smoke tests run the reproduction
wrapper in dry-run mode so heavyweight model stages are skipped.

## Current Training Contract

The canonical GRPO run trains only the scalar `unmask/remask` heads. Stable-cache
execution, cache/access reward terms, and cache/access policy-head training are
disabled in the final run. This keeps the paper focused on the measured
speculative AOAE frontier rather than deferred cache-reuse work.

HumanEval uses `evaluation.task_type: code` with the existing `CodeEvaluator`.
The `openai/openai_humaneval` schema is consumed via `prompt`,
`canonical_solution`, `test`, and `entry_point`.
