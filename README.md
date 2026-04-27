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
│   ├── llada21_hard.yaml      # Main 8B dense training/eval config
│   ├── paper.yaml        # Main paper config
│   ├── poc1.yaml         # PoC 1 soft-routing tradeoff sweep
│   ├── poc2.yaml         # PoC 2 reuse-signal sweep
│   ├── llada21_hard.yaml # Hard-routing routing-sweep baseline
│   └── llada21_soft.yaml # Soft-routing routing-sweep config
├── slurm/
│   ├── train.sh          # Generic PRISM / GRPO / pipeline job
│   ├── eval.sh           # Generic eval job
│   └── paper.sh          # Generic paper / POC job
├── paper/                # Paper source
├── tests/                # Unit + integration tests
├── reproduce.sh          # Local or SLURM orchestration wrapper
└── setup.sh              # Environment bootstrap
```

Generated artifacts live in `outputs/` and are gitignored.

## Canonical configs

| Config | Purpose |
| --- | --- |
| `configs/llada21_hard.yaml` | Main single-run training and eval path on `GSAI-ML/LLaDA-8B-Instruct` |
| `configs/paper.yaml` | Main paper-oriented speculative AOAE config for integrated experiments and end-to-end training |
| `configs/poc1.yaml` | PoC 1: soft-routing speed/quality tradeoff |
| `configs/poc2.yaml` | PoC 2: training-free KV-reuse agreement signal study |
| `configs/llada21_hard.yaml` | Routing sweep hard-routing reference |
| `configs/llada21_soft.yaml` | Routing sweep soft-routing counterpart |

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
aoae preflight --config configs/llada21_hard.yaml
```

Run baseline eval only:

```bash
aoae eval --config configs/llada21_hard.yaml --max_samples 50
```

Train end to end:

```bash
aoae train --config configs/llada21_hard.yaml --stage prism
aoae train --config configs/llada21_hard.yaml --stage grpo
aoae eval --config configs/llada21_hard.yaml --checkpoint outputs/llada21_hard/policy_final.pt
```

Run the integrated local pipeline:

```bash
aoae pipeline --config configs/llada21_hard.yaml
```

For configs with `hardware.tp_size > 1`, `aoae` now auto-relaunches itself under `torchrun` with the same local environment defaults used by the SLURM wrappers.

## Paper and POC workflows

PoC 1 soft-routing tradeoff:

```bash
aoae tau-sweep --config configs/poc1.yaml --max_samples 50
```

PoC 2 reuse-signal study:

```bash
aoae reuse-sweep --config configs/poc2.yaml --max_samples 50
```

Routing-only hard vs soft comparison:

```bash
aoae routing-sweep \
  --hard_config configs/llada21_hard.yaml \
  --soft_config configs/llada21_soft.yaml \
  --max_samples 50
```

Ablation matrix:

```bash
aoae ablations --config configs/paper.yaml --max_samples 50
```

Run the full paper suite:

```bash
aoae paper-suite --config configs/paper.yaml --max_samples 50
```

The paper suite runs the routing sweep, PoC 1, PoC 2, the ablation matrix, and
aggregated reporting artifacts in one workflow.

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
bash reproduce.sh --workflow paper --max_samples 50
bash reproduce.sh --workflow poc1 --max_samples 50
bash reproduce.sh --workflow poc2 --max_samples 50
```

SLURM:

```bash
bash reproduce.sh --slurm
bash reproduce.sh --slurm --workflow paper --max_samples 50
bash reproduce.sh --slurm --workflow ablations --max_samples 50
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
aoae train --config configs/llada21_hard.yaml --stage prism
aoae train --config configs/llada21_hard.yaml --stage grpo --resume auto
aoae eval --config configs/llada21_hard.yaml --checkpoint outputs/llada21_hard/policy_final.pt
aoae pipeline --config configs/llada21_hard.yaml

aoae pipeline --config configs/paper.yaml
aoae eval --config configs/paper.yaml --checkpoint outputs/paper/policy_best.pt --mode speculative
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

For speculative runs, `configs/paper.yaml` defines `evaluation.speculative_sweep.points`: four named AOAE operating points that span the speed/quality Pareto frontier:

| Point | Routing | tau_pi | Draft budget | Goal |
| --- | --- | --- | --- | --- |
| `quality_max` | soft | 0.7 | 4 / 1 microstep | Top accuracy (extends the high-quality side of the frontier) |
| `quality_balanced` | soft | 1.0 | 8 / 3 microsteps | Sweet spot — strong accuracy, moderate TPS |
| `speed_balanced` | lossless | 1.0 | 12 / 4 microsteps | Match LLaDA Quality accuracy at higher TPS |
| `speed_max` | lossless | 1.25 | 16 / 4 microsteps | Match/beat LLaDA Speed TPS at competitive accuracy |

The two soft-routed points use the verifier's softened tail routing for higher accuracy at ~2x per-NFE wall time. The two lossless points route the verifier through the same hard top-k as the auxiliary, restoring native LLaDA throughput while keeping the trained K_stable / access policy active. Together the four points dominate or extend the LLaDA Speed / Quality / Fast-dLLM frontier on both axes. Passing `--policy_temperatures 0.5,1.0,1.5` overrides this sweep with a temperature-only ablation. `cache_hit_rate` in eval artifacts is the stable-cache commit survival rate; `stable_cache_fraction`, `spec_cache_fraction`, and `combined_cache_fraction` report actual occupancy of the persistent stable cache, transient speculative frontier, and their union.
Prompt construction now uses a robust fallback: `data.use_chat_template=auto` attempts tokenizer chat formatting whenever the tokenizer supports it (including runtime/default templates), and falls back cleanly to plain text when unavailable. Non-chat prompts now encode with special tokens enabled, and `data.math_prompt_style=auto` adds a GSM8K-targeted final-answer format instruction when evaluating GSM8K. For confidence-style baselines, terminal unresolved masks are force-completed before scoring so accuracy is computed on complete responses.
GRPO checkpoints are resolved by training-contract and config fingerprint. Eval loads a contract-valid trained checkpoint even when its shaped reward is negative; low scalar reward is reported as an outcome, not silently replaced by the default heuristic policy.

Paper/POC workflows additionally write sweep summaries under:

- `outputs/sweeps/...`
- `outputs/ablations/...`
- `outputs/paper_suite/...`

Aggregated tables write to:

- `outputs/comparison_table.csv`
- `outputs/comparison_table.md`
- `outputs/kv_dynamics_table.csv`
- `outputs/kv_dynamics_table.md`

## GRPO Training System

### Overview

AOAE uses **Group Relative Policy Optimization (GRPO)** to train a lightweight steering policy that controls four per-position actions at each diffusion step:

| Head | Action | Purpose |
| --- | --- | --- |
| **u_t** | Unmask | Reveal a masked position using the base model's prediction |
| **r_t** | Remask | Revert a previously-unmasked position back to `[M]` (self-correction) |
| **κ_t** | Cache | Commit a position's KV state to the dKV-Cache for reuse |
| **q_t** | Access | Predict which positions will need recomputation in the next H steps |

The policy is a 1-layer bidirectional transformer (`d_model=128`) that takes the soft-masked embedding H_t as input and outputs independent Bernoulli logits for each head.  It is trained via GRPO: for each prompt, G rollouts are collected, per-rollout rewards are computed, and group-relative advantages drive a clipped surrogate loss.

### Workflow

```
 Prompt → [G rollouts via aoae_inference] → compute_reward per rollout
        → normalize_group_advantages (mean-center)
        → compute_grpo_loss (clipped surrogate)
        → gradient update on policy parameters
```

1. **Rollout collection** — The frozen base model runs G parallel diffusion trajectories; the policy controls unmask/remask/cache/access at each step.
2. **Reward computation** — A composite scalar reward (detailed below) is computed for each rollout.
3. **Advantage normalization** — Rewards are mean-centered within the group so that GRPO learns *relative* quality.
4. **Policy update** — The clipped surrogate loss `L = -1/G Σ_g min(ρ·A, clip(ρ)·A)` is backpropagated through the policy only; the base model stays frozen.

### Reward Formulation

The per-rollout reward is:

```
R = correctness · speed_factor − β · thrash − w_unresolved · unresolved_frac
    + w_cache · mean_cache_F1 + w_access · access_F1
```

#### 1. Correctness (sparse, terminal)

Binary: 1 if the final decoded text matches the reference answer, 0 otherwise.  Evaluated via numeric/string matching for math, execution-based pass/fail for code.

**Justification:** This is the ground truth signal.  Everything else is a shaping reward to provide denser gradients.

#### 2. Speed Factor (a.k.a effective_flops) (dense, per-rollout)

```
effective_flops = (used_steps / T) · (1 − mean_cached_fraction)
speed_factor    = (1 − effective_flops)^α
```

- **used_steps / T** captures early termination (fewer diffusion steps).
- **1 − mean_cached_fraction** captures within-step savings: cached positions skip KV recomputation at deployment.
- **α** (default 1.0) controls how aggressively speed is rewarded.

**Justification:** A policy that caches 80% of positions but uses all T steps deserves credit for the FLOP reduction.  Multiplying step fraction by cache miss fraction gives a hardware-independent proxy ∈ [0, 1] for total compute cost.

#### 3. Cache Thrashing Penalty (dense, per-step)

```
thrash(t) = Σ_k 𝟙[k ∈ K_{t−1} ∧ e_t^k = 1]
penalty   = β · Σ_t thrash(t)
```

Penalizes caching a position and then immediately editing it (wasteful invalidation).

**Justification:** Without this, the policy can game the effective flops by caching everything indiscriminately.  β = 0.1 is small enough to not dominate but large enough to discourage pathological thrashing.

#### 4. Unresolved Mask Penalty (dense, terminal)

```
unresolved_frac = fraction of positions still [M] at the end
penalty = w_unresolved · unresolved_frac
```

**Justification:** Prevents the policy from learning a "do nothing" strategy where it never unmasks and achieves zero thrashing.  Keeps zero-progress rollouts firmly negative.

#### 5. Cache Quality F1 (dense, per-step) — *new*

This is a soft precision-recall signal over the cache set, replacing the earlier drift-only penalty:

```
For each position k:
  stability(k) = exp(−λ · rel_drift_k)           ∈ (0, 1]
where:
  rel_drift_k = ‖H_t^k − H_{t−1}^k‖₂ / ‖H_{t−1}^k‖₂

Over the cache set K_t:
  precision = mean_{k ∈ K_t}(stability(k))
  recall    = Σ_{k ∈ K_t} stability(k) / Σ_all stability(k)
  cache_F1  = 2 · precision · recall / (precision + recall)

reward += w_cache · mean_t(cache_F1_t)
```

**Justification:** The old drift penalty was precision-only (penalize caching unstable positions) and had no recall gradient (no reward for caching stable ones).  The F1 formulation provides a two-sided signal:
- **Precision** → "don't cache tokens whose embeddings are still changing" (avoids stale KV entries).
- **Recall** → "do cache tokens whose embeddings have converged" (encourages actually using the cache).

This is a *reward* (higher is better), not a penalty, so it is added to the total.

**Why rel_drift as a proxy?**  Extracting per-layer KV vectors at every step is expensive and couples the reward to model internals.  The soft-masked embedding H_t is already computed; its relative L2 drift is a lightweight, scale-invariant proxy that correlates with actual KV staleness.

#### 6. Access Prediction F1 (dense, per-rollout)

```
reward += w_access · speculative_next_H_F1
```

Measures how well q_t predicted which positions would actually change in the next H steps.

**Justification:** The terminal correctness reward is too sparse to credit individual per-step access decisions.  This dense F1 signal teaches the q_t head to be forward-looking.

### Justification for λ = 10

The stability function `exp(−λ · d)` maps relative drift d ∈ [0, ~2] to a soft score in (0, 1]:

| d (rel drift) | λ = 5 | λ = 10 | λ = 20 |
| --- | --- | --- | --- |
| 0.00 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.78 | 0.61 | 0.37 |
| 0.10 | 0.61 | 0.37 | 0.14 |
| 0.20 | 0.37 | 0.14 | 0.02 |
| 0.50 | 0.08 | 0.007 | ~0 |

**λ = 10** is chosen because:
- **d < 0.05** (converged tokens): stability ≈ 0.6, still gets meaningful credit.
- **d > 0.20** (actively changing tokens): stability < 0.14, effectively excluded.
- **Gradient quality:** λ = 5 is too flat (poor discrimination between d = 0.05 and d = 0.20). λ = 20 is too sharp (collapses almost everything to 0 or 1, losing gradient signal). λ = 10 preserves a useful gradient in the [0.05, 0.20] range where most token dynamics live.
- **Empirically:** d ≈ 0.1 is a typical transition point between "converged" and "still refining" in LLaDA diffusion trajectories.  λ = 10 places the sigmoid-like inflection right at this transition.

Tunable via `grpo.stability_lambda` in the config; good range is 5–20.

### Training Commands

```bash
# Stage 1: Train PRISM quality adapter (optional, improves policy input)
aoae train --config configs/llada21_hard.yaml --stage prism

# Stage 2: Train AOAE policy via GRPO
aoae train --config configs/llada21_hard.yaml --stage grpo

# Resume from latest checkpoint
aoae train --config configs/llada21_hard.yaml --stage grpo --resume auto

# End-to-end pipeline (PRISM → GRPO → Eval)
aoae pipeline --config configs/llada21_hard.yaml

# Evaluate a trained policy
aoae eval --config configs/llada21_hard.yaml --checkpoint outputs/llada21_hard/policy_final.pt
```

### Key Tuning Knobs

| Parameter | Config path | Default | Effect |
| --- | --- | --- | --- |
| **α** (speed exponent) | `grpo.alpha` | 1.0 | Higher = more aggressive speed reward |
| **β** (thrash penalty) | `grpo.beta` | 0.1 | Higher = stronger penalty for cache invalidation |
| **w_cache** (cache F1 weight) | `grpo.cache_quality_weight` | 0.02 | Higher = stronger gradient for cache decisions |
| **λ** (stability sharpness) | `grpo.stability_lambda` | 10.0 | Higher = sharper stable/unstable distinction |
| **w_access** (access F1 weight) | `grpo.access_reward_weight` | 0.0 | Higher = stronger gradient for access prediction |
| **w_unresolved** | `grpo.unresolved_penalty_weight` | 0.25 | Higher = harsher penalty for leftover masks |
| **G** (group size) | `grpo.group_size` | 8 | More rollouts = lower variance advantages |
| **ε** (clip range) | `grpo.clip_eps` | 0.2 | Standard PPO/GRPO clip range |
| **τ_π** (policy temp) | `grpo.policy_temperature` | 1.0 | < 1 = more deterministic; > 1 = more exploration |
| **T** (rollout steps) | `grpo.rollout_steps` | 16 | Training-time diffusion horizon (eval uses `inference.steps`) |
| **L_gen** (rollout length) | `grpo.rollout_gen_length` | 128 | Training-time decode budget (eval uses `inference.gen_length`) |

Eval-time Pareto points vary inference controls: verifier schedule budget, `inference.primary_agree_threshold`, `inference.max_unmask_fraction_per_step` or `inference.max_unmask_tokens_per_step`, `inference.disable_remask`, positional-cache settings, and `policy_temperature` per point under `evaluation.speculative_sweep.points`. The legacy accepted-reuse shortcut is disabled for canonical AOAE; `K_spec` is a transient draft frontier, while the baseline rows provide verifier-quality blockwise anchors.

### Hyperparameter Sweep

The GRPO sweep script (`slurm/grpo_sweep.sh`) searches over α and cache quality weight:

```bash
sbatch slurm/grpo_sweep.sh configs/llada21_hard.yaml
```

This generates a grid of `alpha ∈ {0.5, 1.0, 2.0} × cache_quality_weight ∈ {0.0, 0.05, 0.1}` and trains each configuration independently.

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
