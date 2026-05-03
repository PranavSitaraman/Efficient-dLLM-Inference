# SYNC 2026-05-03 — GRPO v7 handoff

This document captures the state of the codebase as of the end of the
2026-05-03 working session. It is intended as a complete cold-start brief
for the teammate who picks up the next leg of the work.

If you are an LLM (Claude, etc.) reading this to help the teammate: trust
the doc as the most recent ground truth, but verify file/line citations
against the live code before recommending any further changes.

---

## TL;DR — what to do right now

1. **Launch the full GRPO run** (v7 design, ~4–5 hours on 2× A100, 500 steps).
   First, export your WandB key in the shell (the repo does NOT carry it —
   `slurm/train.sh` will refuse to start if the config wants WandB and the
   key isn't set):

   ```bash
   export WANDB_API_KEY="<your-codeblock-team-key>"
   bash reproduce.sh --slurm --workflow grpo --sitanc
   ```

   That delegates to `slurm/train.sh grpo configs/paper.yaml fresh` with the
   right partition / account / `--gres=gpu:nvidia_a100-sxm4-80gb:2`. WandB
   logging lands at `https://wandb.ai/codeblock/spec-dlm-grpo` (run name
   `grpo_uvscope_qbal_hardver`).

2. While it runs, read this doc top to bottom plus the two referenced
   research notes:
   - `paper/grpo_loop_iteration.md` — full walkthrough of one GRPO step
     compared to the proposal pseudocode.
   - `paper/5_3_soft_gating_hurts_verifier-performance.md` — the surprising
     ablation finding that drove the choice of training target.

3. After training completes, eval the trained checkpoint at multiple sweep
   points (see *Evaluation plan* below).

---

## Session context

- **Goal of the session**: design and validate a GRPO training setup that
  produces a non-trivially-trained AOAE policy head, train it on the
  `quality_balanced_hardver` operating point, and have it ready to ship.
- **Major discoveries** (in the order they happened):
  1. The previous trained checkpoint (`outputs/paper/policy_best.pt`) was
     trained with `train_heads=["cache","access"]`. The `u_t` and `r_t`
     heads were frozen at initialization — they never received gradient.
     All accuracy in `quality_balanced` eval came from the speculative
     loop's argmax_match self-correction, NOT from the policy.
  2. At `quality_balanced`, the soft top-16 verifier *hurts* both accuracy
     and TPS vs hard top-8 (`quality_balanced_hardver`). 50-sample
     ablation: 86 / 55 vs 82 / 51.
  3. At `quality_max` (more conservative), the routing decision flips:
     soft wins accuracy by 4.7 points but loses TPS by 3. So the
     soft-vs-hard choice is operating-point dependent, not global.
  4. LLaDA-style s/q drafter dynamics (lower threshold + larger budget)
     consistently regress accuracy at both operating points.
  5. Multi-GPU GRPO had a latent bug: the code did per-rank prompt
     sharding and per-rank seed offsets (DP-style) but the LLaDA model
     was loaded EP-split (one model across ranks). Two ranks calling
     forward on different prompts deadlocked NCCL collectives. Fixed.

---

## Files changed in this session

### Code patches

| File | Change |
|---|---|
| `aoae/checkpoints.py` | `GRPO_TRAIN_CONTRACT_VERSION: 6 → 7`. Old checkpoints incompatible (different trained heads). |
| `aoae/speculative_inference.py` | Added `_scope_active(scope, run_primary)` helper. Extended `_apply_frozen_action_heads` with `run_primary` param + `unmask_scope` / `remask_scope` config-driven gates. Both call sites (any-order at `:1151`, block at `:1615`) updated. |
| `aoae/train_grpo.py` | Imported `set_nested`. Extended `build_rollout_cfg` to apply `grpo.rollout_overrides` via `set_nested`. Added `sync_ranks` mode at distributed setup: when `hardware.tp_size > 1` AND distributed, all ranks use the same seed and skip prompt sharding (otherwise EP collectives deadlock). |
| `slurm/train.sh` | Prepended `$CONDA_PREFIX/bin` to `PATH` so `torchrun` resolves to the rtx env's Py3.10 binary, not `~/.local/bin/torchrun` (Py3.12, broken cusparseLt). Added WandB env defaults (`WANDB_ENTITY=codeblock`, `WANDB_PROJECT=spec-dlm-grpo`) and a hard-fail check if `logging.use_wandb=true` but `WANDB_API_KEY` is not set in the launching shell. |
| `reproduce.sh` | Updated usage docs to flag the v7 GRPO command. |
| `configs/paper.yaml` | Full v7 GRPO config diff — see "Final GRPO design" below. |
| `tests/test_config_contracts.py` | Updated expected sweep-point list (added 8 new ablation points: 4× qbal, 4× qmax). |

### New configs / scripts

| Path | Purpose |
|---|---|
| `configs/paper_smoke.yaml` | 10-step smoke at `tp_size=2`, generated from `paper.yaml` with smoke overrides. Regenerate with the helper Python in `run_grpo_smoke.sh`. |
| `run_grpo_smoke.sh` | SLURM smoke (we used this to validate multi-GPU GRPO before the production run). |
| `run_eval_qb_ablations.sh` | 50-sample qbal 4-variant ablation eval. |
| `run_eval_qmax_ablations.sh` | 64-sample qmax 4-variant ablation eval. |
| `check_heads.py` | Diagnostic: walks AdamW state per head to verify which heads received gradients. Use after a training run if you suspect head gating regressed. |

### New documentation

| Path | Purpose |
|---|---|
| `paper/grpo_loop_iteration.md` | End-to-end walkthrough of one GRPO step vs proposal pseudocode. |
| `paper/5_3_soft_gating_hurts_verifier-performance.md` | The qbal+qmax ablation results, candidate explanations, implications. |
| `SYNC_5.3.md` | This doc. |

---

## Final GRPO design (v7)

### Heads / scopes

```yaml
grpo:
  train_heads: ["unmask", "remask"]          # was ["cache", "access"]
  include_heads_in_logprob: ["unmask", "remask"]
  unmask_scope: "drafter"                    # NEW — u_t fires only on aux microsteps
  remask_scope: "verifier"                   # NEW — r_t fires only on primary microsteps
```

Rationale: under speculative decoding, the drafter learns *what to propose*
(u_t on aux microsteps) and the verifier learns *what to roll back* (r_t on
primary microsteps). Verifier microsteps still use the canonical threshold
rule for u_t (preserves accuracy floor; argmax_match auto-correction is
unaffected). κ_t and q_t are deliberately unused for now (stability cache
deferred — proposal Phase C).

The mechanism is in `aoae/speculative_inference.py:_apply_frozen_action_heads`
(lines 305–355 after the patch). Verified by `check_heads.py`: smoke v6's
checkpoint shows AdamW `exp_avg` is nonzero only for `head_unmask.*` and
`head_remask.*`; `head_cache` and `head_access` have no optimizer state at
all (they never received gradient).

### Reward

```
R = correctness × (1 − effective_flops)^α  −  w_unresolved × unresolved_frac
  with α = 1.0,  w_unresolved = 0.25
```

`β × thrash_rate`, `cache_F1`, `access_F1` all evaluate to zero under our
config:

```yaml
grpo:
  alpha: 1.0
  beta: 0.1                                  # inert: thrash counts cache invalidations,
                                             # cached set is empty when κ_t = 0
  unresolved_penalty_weight: 0.25
  cache_quality_weight: 0.0                  # κ_t not trained
  access_reward_weight: 0.0                  # q_t not trained
  cache_speed_credit_cap: 0.0
```

### Rollout target = `quality_balanced_hardver`

```yaml
grpo:
  rollout_steps: 16                          # T (full-quality verifier baseline reference)
  rollout_gen_length: 512                    # match eval (was 128)
  rollout_overrides:
    "base_model.lossless_verification": true                # hardver: primary == auxiliary forward
    "inference.primary_agree_threshold": 0.92               # match eval
    "inference.drafter.run_on_verifier": "never"            # under hardver, aux on verifier microsteps
                                                            # is a wasted second forward
    "inference.drafter.aux_compute_ratio": 0.0              # under hardver, only reject-induced
                                                            # recomputes cost (TPS-equivalent reward)
```

The two key hardver-specific tweaks (`run_on_verifier=never` and
`aux_compute_ratio=0.0`) are explained at length in
`paper/grpo_loop_iteration.md` and in the rollout_overrides comments in
`configs/paper.yaml`. **TL;DR**: under hardver, drafter and verifier are
the same forward function (same routing). The default
`aux_compute_ratio=0.35` is calibrated for soft verifier (top-16) and
over-credits aux microsteps as cheap; under hardver they cost the same as
primary. Setting `aux_compute_ratio=0.0` reframes the speed reward as
*tokens-committed per forward call*, which is the right target under
hardver where the only speed lever is "fewer rejections".

### Schedule / training

```yaml
grpo:
  max_steps: 500                             # first run; extend on convergence telemetry
  warmup_steps: 50
  epochs: 3
  group_size: 8                              # G in GRPO advantage
  lr: 3.0e-4
  weight_decay: 0.01
  batch_size: 1
  grad_accum_steps: 1
  policy_temperature: 1.0
  normalize_advantage_std: true
  clip_eps: 0.2                              # PPO clip
  warm_start_from: null                      # backbone biased toward κ/q in prior run; start fresh
  warm_start_strict: false
```

### Distributed mode (multi-GPU)

```yaml
hardware:
  tp_size: 2                                 # 2× A100 with EP-split MoE
  bf16: true
  seed: 42
```

The new `EP/TP-shared` mode (auto-activated when `tp_size > 1` AND
distributed) makes both ranks process the *same* prompt with synchronized
RNG state. The G rollouts in a group are still independent across the
batch dim (different bernoulli draws per group element), so the advantage
signal is preserved. Without this fix, rank 0 and rank 1 would call EP
collectives on different prompts and hang.

You will see this on startup:
```
[GRPO] Distributed mode: EP/TP-shared (synchronized prompts + RNG)
       (tp_size=2, world_size=2)
```

If you ever need pure data-parallel GRPO (one model per GPU, different
prompts per rank, gradient sync via DDP), set `hardware.tp_size: 1` and
the original DP behavior is preserved.

### Logging

```yaml
logging:
  project: "spec-dlm-grpo"
  run_name: "grpo_uvscope_qbal_hardver"
  output_dir: "outputs/grpo_uvscope/"
  use_wandb: true
  log_every: 10
  eval_every: 200
  save_every: 100
```

**WandB credentials**: the API key is **NOT** stored in the repo. Before
launching, export it in your shell:

```bash
export WANDB_API_KEY="<your-codeblock-team-key>"
bash reproduce.sh --slurm --workflow grpo --sitanc
```

`slurm/train.sh` will hard-fail at job startup if `logging.use_wandb=true`
in the config and `WANDB_API_KEY` is unset, so you cannot accidentally run
without WandB silently. The `WANDB_ENTITY` (defaults to `codeblock`) and
`WANDB_PROJECT` (defaults to `spec-dlm-grpo`) env vars are set in
`slurm/train.sh` and can be overridden the same way.

To run without wandb, set `logging.use_wandb: false` in the config.

---

## Pre-launch checklist

1. **Branch / git state**: we did not commit during the session. `git status`
   will show edits to `aoae/{checkpoints.py, speculative_inference.py, train_grpo.py}`,
   `configs/paper.yaml`, `slurm/train.sh`, `reproduce.sh`, and several new
   files in `paper/`, `outputs/qb_ablations/`, `outputs/qmax_ablations/`,
   `outputs/smoke_uvscope/`. Decide whether to commit before launching.
2. **PRISM adapter**: GRPO config has `verifier.use_prism_score: false`,
   so the existing `outputs/paper/prism_adapter.pt` is loaded but its
   output is not consumed. Safe to leave as-is.
3. **Old policy checkpoints**: `outputs/paper/policy_best.pt` is v6
   (cache+access). The v7 contract version bump means the GRPO trainer
   will refuse to resume it. With `warm_start_from: null` we start from
   fresh init — intended.
4. **Disk**: each saved checkpoint is ~5.5 MB; `save_every: 100` means
   ~5 checkpoints over a 500-step run. Trivial.

---

## Launching the run

### One-line command

```bash
bash reproduce.sh --slurm --workflow grpo --sitanc
```

### What that resolves to

`reproduce.sh` invokes `sbatch` with these parameters:

```bash
sbatch \
  --gres=gpu:nvidia_a100-sxm4-80gb:2 \
  --partition=seas_gpu \
  --account=sitanc_lab \
  slurm/train.sh grpo configs/paper.yaml fresh
```

`slurm/train.sh` then:
1. Activates the `rtx` conda env, fixes `PATH`, exports WandB env vars.
2. Reads `hardware.tp_size=2` from the config → sets `GPUS_PER_NODE=2`.
3. Launches `torchrun --nproc_per_node 2 -m aoae.cli train --stage grpo --resume fresh`.

### Expected timing

| Phase | Wall time |
|---|---|
| Module load + conda activate | ~30 s |
| Dual-model load + EP setup | ~3 min |
| GRPO rollouts at gen=512, T=16, G=8 | ~30–40 s/step |
| 500 steps | ~4–5 hours |
| Post-training eval (eval_every=200) | ~10 min each |

The job's SLURM time limit is set by `slurm/train.sh` to `12:00:00`, plenty
of buffer.

### Live monitoring

- WandB: `https://wandb.ai/codeblock/spec-dlm-grpo`
- SLURM: `squeue -u $USER` and `tail -f logs/<jobid>_train.{out,err}`
- Local JSONL: `outputs/grpo_uvscope/training_log.jsonl`

### What "working" looks like in the first ~50 steps

- `Distributed mode: EP/TP-shared` line at startup.
- Per-step trace lines like:
  ```
  step=    1  reward=...  loss=...  lr=...
    correct=...  speed=0.X  eff_flops=0.Y  steps_frac=0.Z
    thrash_rate=0.000  unresolved_pen=...
  ```
  with `eff_flops < 1.0` (otherwise speed gradient is dead — see notes).
- `correct` and `frac_pos` should fluctuate (some rollouts correct, some
  not) — this is what gives the advantage signal.
- `loss` magnitudes should be in the 1e−4 to 1e−2 range; if they spike
  past 0.1 something is wrong (clip should bound at ε=0.2, but big spikes
  signal divergent rollouts).
- `thrash_rate` should be 0 throughout (cache machinery is off; if it's
  nonzero, something has changed).

### What "broken" looks like

- NCCL timeout (`Watchdog caught collective operation timeout`) → the EP
  sync regressed. Check `aoae/train_grpo.py:828-845` `sync_ranks` logic
  and prompt-sharding gate at `:1130`.
- `eff_flops > 1.0` → the `aux_compute_ratio` / `run_on_verifier`
  overrides didn't apply. Verify `build_rollout_cfg(cfg)` returns the
  expected nested values.
- `head_cache.weight.grad != 0` (use `check_heads.py` after first save) →
  the `_apply_frozen_action_heads` scope gating regressed.
- WandB falls back to file-only with `Personal entities are disabled` →
  the `WANDB_ENTITY` env var didn't make it into the launched torchrun
  child processes.

---

## Pinned design decisions (deliberately deferred)

These are real research ideas we considered and chose to defer rather than
fold into this run. Each is small to implement (1–30 lines) and would be a
clean follow-up sweep point. See `paper/grpo_loop_iteration.md` §5 for
fuller treatment.

### 1. Verifier-corrects-instead-of-remasks (option A1)

**Idea**: at `aoae/speculative_inference.py:988`, replace
`resp_tokens[reject_mask] = mask_id` with
`resp_tokens[reject_mask] = pri_logits.argmax(-1)[reject_mask]`. The
verifier commits its own argmax instead of remasking the rejected draft.
Analogous to the LLaDA q-mode edit rule.

**Why deferred**: it changes loop semantics and would invalidate the eval
baseline we just collected (50-sample qbal ablation). Should be a sibling
sweep point + a matching GRPO run, not a default. Under hardver
specifically, the standard remask-then-redraft loop converges to the
verifier's argmax in 1–2 microsteps anyway (same model), so A1 is mostly
an efficiency tweak rather than a semantic improvement.

### 2. Separate drafter / verifier rewards (option B)

**Idea**: per-microstep credit assignment in the GRPO surrogate.

```
R_drafter  = correctness × speed_factor − w_reject × reject_rate − w_unres × unresolved
R_verifier = correctness × speed_factor − w_unres × unresolved
                                          (no reject penalty — rejection IS the verifier's job)
```

`u_t` log-probs at aux microsteps weighted by `R_drafter`'s advantage;
`r_t` log-probs at primary microsteps weighted by `R_verifier`'s
advantage.

**Why deferred**: the user's concern was that an explicit reject penalty
on the drafter could create a "be conservative" attractor that loses the
speculative-decoding speed benefit (which depends on aggressive drafting).
The single-scalar reward already rewards aggression conditional on accept
rate via the `speed_factor`-correctness coupling, so the dense reject
signal is redundant in expectation. Revisit if first-run telemetry shows
the drafter under-utilizing aggression or the verifier remasking
spuriously. Implementation cost: ~30 lines around `compute_reward` and
the loss in `aoae/train_grpo.py`.

### 3. Per-position rejection-history feature

**Idea**: add an explicit per-position counter of how many times the
verifier has rejected drafts at that position. Pass to the policy as a
new input feature so it can learn to back off positions that repeatedly
fail.

**Why deferred / deprioritized**: the existing `agreement` and `age`
features already encode a one-step lookback (was-this-position-accepted-
last-time, how-long-since-last-state-change). The user prefers the
verifier-side trained `r_t` to handle the same role, which is more
aligned with the speculative-decoding responsibility split.

### 4. Direct remask-invalidation penalty

**Idea**: add a counter `(r_t.bool() & ~mask_ind).sum()` per step as a
fallback penalty if the trained r_t collapses to "remask everything".
The current reward has `β × thrash_rate` but `count_thrash` is keyed on
the cached set, which is empty under κ=0, so β·thrash is inert.

**Why deferred**: don't preempt. The terminal `unresolved_frac` penalty
should already catch the "remask everything → leftover masks" attractor.
Add only if telemetry shows otherwise.

### 5. Stability cache (κ_t / q_t / K_stable)

The proposal's Phase C. Out of scope for this run. When you re-enable it,
expect to:
- Re-add `cache` and `access` to `train_heads`.
- Set `cache.stable_kv_cache: true`.
- Restore `cache_quality_weight`, `access_reward_weight`, `cache_speed_credit_cap`.
- Bump `GRPO_TRAIN_CONTRACT_VERSION` again.

### 6. Soft verifier rehabilitation

The 50-sample qbal ablation showed soft top-16 hurts. The 64-sample qmax
ablation showed soft top-16 wins on accuracy but loses on TPS — the
crossover is operating-point dependent. A future ablation
(`primary_agree_threshold ∈ {0.85, 0.92, 0.96, 0.98}` × {soft, hard})
would map where the crossover sits along the Pareto curve.

---

## Eval ablation results (recap)

### Quality-balanced track (50 samples, GSM8K, any-order, against `policy_best.pt`)

| Sweep point | Acc | TPS |
|---|---|---|
| `quality_balanced` (soft, q-drafter) | 0.82 | 51.1 |
| **`quality_balanced_hardver`** (hard, q-drafter) | **0.86** | **54.7** |
| `quality_balanced_sq` (soft, s-drafter) | 0.72 | 48.3 |
| `quality_balanced_sq_hardver` (hard, s-drafter) | 0.70 | 50.5 |

Hard verifier Pareto-dominates at qbal. SQ drafter dynamics regress.

### Quality-max track (64 samples)

| Sweep point | Acc | TPS |
|---|---|---|
| **`quality_max`** (soft, q-drafter) | **0.859** | 42.84 |
| `quality_max_hardver` (hard, q-drafter) | 0.812 | **45.87** |
| `quality_max_sq` (soft, s-drafter) | 0.750 | 42.56 |
| `quality_max_sq_hardver` (hard, s-drafter) | 0.734 | 42.79 |

At qmax, soft wins accuracy (+4.7 pts), hard wins TPS (+3). No Pareto
domination; the crossover is real and operating-point dependent. SQ
drafter regression replicates.

### Reference baselines (any-order, same eval runs)

| Method | Acc | TPS |
|---|---|---|
| `llada21_speed_anyorder` | 0.30–0.44 | 160 |
| `llada21_quality_anyorder` | 0.10–0.22 | 156 |
| `fast_dllm` | 0.62–0.70 | 46–62 |

Note: pure LLaDA any-order baselines collapse on these tiny GSM8K samples
(10–22% acc). The speculative-AOAE pipeline's argmax_match self-
correction is what produces the 70%+ numbers, NOT the trained policy
heads (verified — see `policy_best.pt` discussion above).

---

## Evaluation plan after the GRPO run completes

The trained checkpoint will land at `outputs/grpo_uvscope/policy_best.pt`
(and `policy_final.pt`, plus interrupt/latest variants). To evaluate:

```bash
# Full any-order Pareto: includes quality_max, quality_balanced, all 8
# new ablation points, and aoae_llada_sq_anyorder (50 samples).
sbatch run_eval_qb_ablations.sh                                     # qbal-family
sbatch run_eval_qmax_ablations.sh                                   # qmax-family

# Headline: in-distribution (matches GRPO rollout target):
python3 -m aoae.cli eval \
    --config configs/paper.yaml \
    --checkpoint outputs/grpo_uvscope/policy_best.pt \
    --mode speculative \
    --max_samples 50 \
    --generation_mode_filter any_order \
    --sweep_points quality_balanced_hardver \
    --output_dir outputs/grpo_uvscope_eval
```

Compare against the v6 (cache+access trained) baseline at the same point
(ours got `quality_balanced_hardver=0.86 / 54.7` from the 50-sample qbal
ablation). The trained v7 policy should at minimum match this — the
unmask/remask heads learning anything useful adds *on top of* the
loop's argmax_match floor.

---

## What the proposal pseudocode says vs what we implemented

The proposal (Algorithm `alg:aoae` in `paper/proposal.tex`, lines 277–316)
specifies:
- 4 trained Bernoulli heads (u, r, κ, q) on a shared backbone
- Multiplicative reward with cache F1 and access F1 terms
- Stability cache K_stable evicted by remask
- Composed prediction p̃ ∝ p · d^(γα)
- PRISM-based acceptance via threshold δ_acc

Our v7 run is the proposal's **Phase A**: speculative validation only.
- Only u, r heads trained (κ, q deferred).
- Reward = `correctness × speed_factor − w_unres × unresolved_frac`
  (cache F1 / access F1 zeroed).
- K_stable disabled.
- Composed prediction off (`compose_gamma: 0`; CLAUDE.md notes prior
  runs found γ > 0 degraded accuracy).
- Acceptance via `argmax_match` (Phase A simplification; PRISM head is
  loaded but its output isn't consumed since `verifier.use_prism_score:
  false`).

Where we deviate: hardver target instead of soft top-16. Justified by
the 50-sample qbal ablation. Documented end-to-end in
`paper/grpo_loop_iteration.md`.

---

## Quick reference: which files matter

```
configs/paper.yaml                    # production config (v7 GRPO, hardver target)
configs/paper_smoke.yaml              # 10-step smoke (regenerate from paper.yaml)
aoae/speculative_inference.py         # _apply_frozen_action_heads, _scope_active
aoae/train_grpo.py                    # build_rollout_cfg (rollout_overrides), sync_ranks
aoae/checkpoints.py                   # GRPO_TRAIN_CONTRACT_VERSION = 7
slurm/train.sh                        # PATH fix + WandB env vars + torchrun launcher
reproduce.sh                          # top-level wrapper; --workflow grpo for v7

paper/grpo_loop_iteration.md          # full v7 walkthrough vs proposal
paper/5_3_soft_gating_hurts_verifier-performance.md
                                      # the ablation that drove the hardver choice

run_grpo_smoke.sh                     # 2-GPU smoke (validates plumbing)
run_eval_qb_ablations.sh              # qbal 4-variant ablation
run_eval_qmax_ablations.sh            # qmax 4-variant ablation
check_heads.py                        # diagnostic: which heads got gradient

outputs/qb_ablations/                 # 50-sample qbal results
outputs/qmax_ablations/               # 64-sample qmax results
outputs/smoke_uvscope/                # smoke v6 artifacts (validates v7 design)
```

---

## Open questions for whoever picks this up

1. **Does the trained v7 policy beat its untrained baseline?** The
   untrained-but-gated baseline at `quality_balanced_hardver` is 0.86 /
   54.7 (50 samples). The trained v7 policy needs to match or improve.
2. **Does the trained policy generalize?** Eval at `quality_balanced` (off-
   target, soft verifier) and `quality_max` (different operating point).
3. **Is 500 steps enough?** Watch for plateau in WandB. If reward is still
   trending up at step 400, extend `max_steps` and resume.
4. **When to enable κ/q (Phase C)?** The proposal layers stability cache
   on top of a working speculative loop. Once v7 has converged and we're
   confident in the u/r training, this is the next deferred lever.
5. **Should we revisit the soft verifier?** The qmax ablation hints that
   along the Pareto curve, soft routing might be useful at conservative
   operating points. A targeted `primary_agree_threshold` × routing sweep
   would map the crossover.

When you take any of these on, please add a `SYNC_<date>.md` for the
next handoff.
