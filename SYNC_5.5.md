# SYNC 2026-05-05 — GRPO V3 + V3.5 handoff

This document captures the state of the codebase as of the end of the
2026-05-04 working session. Cold-start brief for the next leg.

If you are an LLM (Claude, etc.) reading this to help the teammate: trust
the doc as the most recent ground truth, but verify file/line citations
against the live code before recommending any further changes.

---

## TL;DR — what is running, what to do next

1. **V3 GRPO is running** as SLURM job `10032928` on 2× H200, ~9 h wall-clock,
   1000 optimizer steps. It implements Path C feature set + Expert Steering
   + PRISM-as-input + paper-aligned hyperparameters.
   - WandB run: `codeblock/spec-dlm-grpo` (run name `grpo_v3_pathC_es_prism`).
   - Output dir: `outputs/grpo_v3/`.
   - Code committed on `main` as `d38486c`.

2. **V3.5 is staged on branch `v3.5-paper-aligned`** as a speculative
   parallel run for the case where V3 plateaus near heuristic baseline.
   V3.5 tightens to fully paper-aligned hyperparameters AND increases
   effective batch via gradient accumulation. Implementation details below;
   smoke-tested and ready to launch in parallel without disturbing V3.

3. **Watch curves** at the V2 plateau point (~step 250-300). Decisions:
   - If `reward/correctness` is climbing (windowed batch > 0.30 sustained,
     trending up), V3 is working — let it complete, then eval at
     `quality_balanced_hardver`.
   - If reward stays in the 0.00-0.10 plateau band for 100+ steps
     consecutive, **launch V3.5 in parallel** (`git checkout v3.5-paper-aligned`,
     submit using same `slurm/train.sh grpo` invocation). V3.5 does NOT
     conflict with V3 — separate output dir, separate wandb run.
   - The most informative early signal once we add the `es/*` curves:
     `es/expert_advantage` should start large positive (heuristic >> policy),
     trend down, and ideally cross zero (policy = heuristic) within ~500
     steps. Crossing-zero by step ~500 means the policy has learned to
     match the heuristic; surpassing it requires the remaining ~500 steps.

4. **`es/*` logging is NOT yet implemented**. We discussed it at the end
   of the session but didn't land. See *Open follow-ups* below.

---

## Session context

- **Goal**: V2 GRPO plateaued at correctness ≈ 0.10-0.30 (windowed batch),
  far below the heuristic baseline (0.86 at `quality_balanced_hardver`).
  Diagnosis: cold-start RL on a clean-slate policy head needs structured
  guidance — the GRPO reward signal alone is too weak vs the strong
  threshold-rule heuristic baseline.
- **Solution**: implement Expert Steering (Jazbec et al. 2026,
  arXiv:2512.09106 Appendix F) — augment each GRPO group with one
  deterministic heuristic rollout. The mixture policy
  `π^ES = G/(G+E)·π_φ + E/(G+E)·δ_expert` provides a bounded importance
  ratio and a reliable gradient direction.
- **Secondary changes**: drop H_t (hidden-state input — paper found it
  hurts), add c_t (max-confidence input), add q_feat (PRISM as input
  feature, NOT teacher), drop `age_feat` and `last_action_feat`.
- **Hyperparameters**: paper-aligned where it matters most for stability:
  `lr=3e-5`, `max_grad_norm=0.5` (between V2's 2.0 and paper's 0.2),
  `max_steps=1000`, `normalize_advantage_std=false`.

---

## V3 design (currently running, branch `main`)

### Code changes (committed `d38486c`)

| File | Change |
|---|---|
| `aoae/models/policy.py` | New flags `use_hidden_state` (default True), `use_max_confidence_feature` (default False), `use_quality_score_feature` (default True). `forward()` conditionally includes `H_t`, `c_t`, `q_feat`. Input dim accounting fixed in `__init__`. The `confidence` parameter is now actually used (no longer `del`'d). |
| `aoae/train_grpo.py` | Pulled `_run_speculative_group` helper. `collect_rollout_group` runs G policy + E expert rollouts (when ES enabled), tags trajectories with `is_expert`, concatenates trajectories/rewards/components, computes group-mean advantage over G+E. New `_expert_steering_mixture_log_prob(...)` helper computes `log π^ES`. `compute_grpo_loss` takes `expert_steering_G/E` args; per-rollout, applies mixture log-prob ratio when `is_expert=True`, standard ratio otherwise. Outer loop iterates over `len(trajectories)` (G+E) and threads ES counts through. |
| `aoae/speculative_inference.py` | `fallback_unmask` now gated to test-time only: `use_fallback = config_value AND (not record_trajectory)`. Training rollouts (record_trajectory=True) get no force-unmask, matching paper §3.2 explicit. |
| `aoae/checkpoints.py` | `GRPO_TRAIN_CONTRACT_VERSION: 7 → 8`. Old V7 checkpoints have a different `input_proj` input dim (D + 6 vs current 5) and cannot be resumed. |
| `configs/paper.yaml` | `policy.use_hidden_state: false`, `policy.use_max_confidence_feature: true`, `use_age_feature: false`, `use_last_action_feature: false`. `grpo.lr: 3e-5`, `max_grad_norm: 0.5`, `max_steps: 1000`, `normalize_advantage_std: false`. New `grpo.expert_steering: { enabled: true, expert_count: 1 }`. New rollout override `inference.verifier.use_prism_score: true`. `logging.run_name: grpo_v3_pathC_es_prism`, `output_dir: outputs/grpo_v3/`. |

### Path C feature set (per-position policy input)

| feature | enabled | shape | notes |
|---|---|---|---|
| `H_t` | ❌ disabled | `[B, L, D=2048]` | paper's §4.4 ablation: hurts |
| `m_t` (mask indicator) | ✅ | `[B, L, 1]` | always-on |
| `t_feat` (t/T) | ✅ | `[B, L, 1]` | always-on |
| `c_feat` (max conf) | ✅ NEW | `[B, L, 1]` | paper's primary input |
| `q_feat` (PRISM score) | ✅ | `[B, L, 1]` | only nonzero on verifier microsteps; otherwise zero |
| `agreement` | ✅ | `[B, L, 1]` | drafter↔verifier match indicator |
| `age_feat` | ❌ disabled | `[B, L, 1]` | dropped |
| `last_action_feat` | ❌ disabled | `[B, L, 1]` | dropped |

Total enabled per-position dim = **5 scalars**. `input_proj = Linear(5, 128)`.
Policy params drop from 461K (V2) → **199K**.

### Expert Steering plumbing

- `grpo.expert_steering.enabled: true`, `expert_count: 1`. Group becomes G+E = 9.
- Expert rollout: re-runs `speculative_inference` with `cfg.grpo.train_heads = []`
  (deepcopy of rollout_cfg). With empty `train_heads`,
  `_apply_frozen_action_heads` overwrites every action head with its
  deterministic fallback (threshold-rule u_t at confidence ≥ 0.7, r_t = 0,
  κ_t = 0, q_t = 0). The policy network is still evaluated; its logits
  go into `policy_outputs` so the loss can compute log π_φ(expert | state).
- Mixture log-prob (Jazbec App F):
  `log π^ES(a | s) = logsumexp([log π_φ(a) + log(G/(G+E)),  log(E/(G+E))])`.
  - For policy samples: Dirac contributes 0 (policy actions ≠ expert
    actions in continuous-action limit); mixture term cancels in the ratio
    new/old (φ-independent constant). Standard ratio applied.
  - For expert sample (`is_expert=True`): mixture ratio computed via
    `_expert_steering_mixture_log_prob`. Bounded below by `log(E/(G+E))
    = log(1/9) ≈ -2.2` regardless of how small `π_φ(expert)` becomes —
    prevents importance-ratio blow-up when policy diverges from expert.
- The expert is the *current* heuristic (threshold-rule + verifier
  argmax_match + verifier-driven remask), i.e., behaviorally identical to
  the `quality_balanced_hardver` baseline (~0.86 acc on 50-sample GSM8K).

### V3 hyperparameters (vs V1, V2, paper)

| | V1 | V2 | **V3** | Paper (Jazbec 2026) |
|---|---|---|---|---|
| lr | 3e-4 | 1e-4 | **3e-5** | 3e-5 ✓ |
| max_grad_norm | 1.0 | 2.0 | **0.5** | 0.2 |
| max_steps | 500 | 500 | **1000** | ~938 ✓ |
| warmup_steps | 50 | 100 | **100** | 100 ✓ |
| weight_decay | 0.01 | 0.01 | **0.01** | 0.1 |
| effective batch | 1 | 1 | **1** | 16 |
| group_size G | 8 | 8 | **8** ✓ | 8 ✓ |
| ε (clip) | 0.2 | 0.2 | **0.2** ✓ | 0.2 (with ES) ✓ |
| KL β | 0.0 | 0.0 | **0.0** ✓ | 0.0 ✓ |
| normalize_advantage_std | true | true | **false** ✓ | false ✓ |
| Expert Steering | — | — | **E=1** ✓ | E=1 ✓ |

`max_grad_norm=0.5` is the midpoint between V2's 2.0 and paper's 0.2.
Justification: with 1-prompt effective batch (vs paper's 16), our per-step
gradient is noisier; tightening all the way to 0.2 might over-restrict
useful signal.  `weight_decay=0.01` left at V2 value — for our 199K-param
policy the regularization difference vs paper's 0.1 is small.

### V3 launch

- SLURM job `10032928` on 2× nvidia_h200 (`seas_gpu`, `sitanc_lab`),
  submitted at ~22:55 EDT 2026-05-04.
- Submission: `sbatch --gres=gpu:nvidia_h200:2 --partition=seas_gpu
  --account=sitanc_lab slurm/train.sh grpo configs/paper.yaml fresh`.
- Config fingerprint at submit time: `paper.yaml` with the V3 settings.
- Smoke `10030558` passed all 10 steps cleanly: distributed mode active,
  Path C policy compiled (5-dim input), ES augmentation working
  (G+E rollouts complete without rank deadlock or mixture-ratio explosion),
  PRISM input feeds correctly, gradient flow active (loss values −0.0010
  to +0.0004, head_unmask + head_remask gradients sustained).

### Where to monitor V3

- WandB: `https://wandb.ai/codeblock/spec-dlm-grpo` — find run by name
  `grpo_v3_pathC_es_prism`.
- JSONL: `outputs/grpo_v3/training_log.jsonl` — one record per
  `log_every=10` step.
- Console: `logs/10032928_train.{out,err}`.
- Checkpoints saved every 50 steps (overwrites `policy_latest.pt`),
  plus `policy_best.pt` at end of (truncated) epoch and `policy_final.pt`
  at end of run.

### Decision tree at V3 completion

1. **If trained checkpoint correctness ≥ heuristic baseline** (i.e., eval
   at `quality_balanced_hardver` with `policy_final.pt` ≥ 0.86):
   - V3 worked — ship it.
   - Run full sweep eval at multiple operating points to map the Pareto
     contribution.
2. **If trained checkpoint correctness < heuristic baseline by < 10 points**:
   - Tighten hyperparameters via V3.5 (see below).
   - Or: eval `policy_best.pt` (mid-training checkpoint) — sometimes the
     best mid-training point beats the final due to over-fitting.
3. **If trained checkpoint correctness substantially below baseline**
   (gap > 15 pts):
   - The Expert Steering signal isn't propagating effectively. Think
     beyond V3.5 — possibilities listed in *Open follow-ups*.

---

## V3.5 design (branch `v3.5-paper-aligned`, smoke-tested, ready to launch)

V3.5 is the speculative parallel-track run for if V3 plateaus. It tightens
to fully paper-aligned hyperparameters *and* doubles effective batch via
gradient accumulation. Implemented on a separate branch so it doesn't
disturb V3.

### V3.5 incremental changes (vs V3)

| Knob | V3 | **V3.5** | Rationale |
|---|---|---|---|
| `max_grad_norm` | 0.5 | **0.2** | Match paper exactly; per-step direction better preserved when accumulated over more samples. |
| `weight_decay` | 0.01 | **0.1** | Match paper; tighter regularization on small policy. |
| `grad_accum_steps` | 1 | **2** | Effective batch = 2 prompts/step (vs V3's 1, paper's 16). Halves the cross-prompt gradient noise. |
| `max_steps` | 1000 | **500** | Compensates for 2× per-step wall-clock from accumulation. |
| `output_dir` | `outputs/grpo_v3/` | **`outputs/grpo_v3_5/`** | No collision with V3. |
| `run_name` | `grpo_v3_pathC_es_prism` | **`grpo_v3_5_paperalign`** | New wandb run. |

Net effect: same wall-clock as V3 (~9 h), same total prompts visited
(1000), but **fewer, cleaner gradient updates** (500 updates each averaged
over 2 prompts). Closer to paper's training dynamics in spirit.

### Why this is the right next step if V3 plateaus

V3 already addresses the *signal* problem (Expert Steering provides
reliable gradient direction).  If it still plateaus, the bottleneck is
likely *noise* — per-step gradient is noisy because we have batch=1.
Three responses, in order of sophistication:
1. **Reduce per-step noise via grad_accum** (V3.5) — implemented.
2. Add proper data parallelism (TP=2, DP=2 on 4 GPUs) — non-trivial code
   refactor, deferred.
3. Tighten policy via more aggressive decay + clipping (V3.5 includes the
   single-knob aspects of this).

### V3.5 launch (when ready)

```bash
git checkout v3.5-paper-aligned
mkdir -p outputs/grpo_v3_5 && \
  ln -sf /n/holylabs/sitanc_lab/Users/gye/outputs/paper/prism_adapter.pt \
    outputs/grpo_v3_5/prism_adapter.pt
sbatch --gres=gpu:nvidia_h200:2 \
  --partition=seas_gpu --account=sitanc_lab \
  slurm/train.sh grpo configs/paper.yaml fresh
```

Output: `outputs/grpo_v3_5/`. WandB run: `grpo_v3_5_paperalign` in
`codeblock/spec-dlm-grpo`. Same V3 design, just the hyperparameter
tightening + grad_accum=2.

---

## Open follow-ups

### Pending (small)

1. **Add `es/*` logging panels** — was discussed and agreed but not landed.
   Add to `train_grpo.py` log block:
   - `es/expert_reward`, `es/policy_reward_mean`, `es/expert_advantage`
     (= R[expert] − mean(R[full group])), `es/expert_correctness`,
     `es/policy_correctness_mean`, `es/expert_completion_step`.
   - `es/expert_mix_ratio_min/max/mean` from per-step `_expert_steering_mixture_log_prob`
     output.
   - Trajectory expectation: `es/expert_advantage` starts large positive,
     declines, ideally crosses zero around the V2-plateau-equivalent point.

2. **Eval setup bug**: The 50-sample PRISM-validation eval (`9970584`,
   then `9973057`) showed `quality_balanced_hardver` accuracy collapsing
   from 0.86 (prior) → 0.30. Root cause: `paper.yaml` train_heads change
   means the eval reads
   `train_heads=["unmask","remask"]` and lets the *untrained* head_unmask
   / head_remask drive eval decisions. Need to either (a) freeze
   `train_heads=["cache","access"]` semantics for old-checkpoint evals
   (eval-only override), or (b) ship a v2-baseline-comparison that uses
   a separate eval-only config file. Out of scope for V3 launch but is the
   next eval-engineering task.

3. **PRISM-validation: PRISM as input is reliable.** With same broken-baseline
   (untrained u/r), PRISM gating still raised acc 0.30 → 0.50 (+0.20).
   Confirms the PRISM checkpoint at `outputs/paper/prism_adapter.pt` is
   producing useful per-position quality scores; safe to use as `q_feat`
   input in V3.

### Pending (larger)

4. **Reward-shaping evolution** — once V3 lands, the reward function may
   need refinement:
   - Currently `R = correctness × (1 − used_steps/T)^α − w_unres × unresolved_frac`.
   - Speed signal could be reformulated as TPS (wall-clock-based) for
     evaluation alignment, but training-time wall-clock is noisy across
     SLURM nodes.
   - The α=1 choice was Jazbec-validated; Jazbec also notes higher α can
     introduce training instability.

5. **Verifier-corrects-instead-of-remasks** (deferred from earlier
   sessions). Documented in
   `~/.claude/projects/-n-holylabs-sitanc-lab-Users-gye/memory/project_grpo_design_pinned.md`.
   At `aoae/speculative_inference.py:988`, swap `resp_tokens[reject] = mask_id`
   for `resp_tokens[reject] = primary.argmax(...)`. Avoids the
   "drafter keeps proposing same wrong token" attractor.

6. **Separate drafter vs verifier rewards** (deferred). Per-microstep
   credit assignment in GRPO surrogate. Drafter rewarded for accept rate;
   verifier doesn't pay for over-rejection.

7. **DP=2 architecture refactor** — Decouple `world_size` from
   `hardware.tp_size` in `aoae/cli.py` so we can run TP=2 × DP=2 on 4
   GPUs (closer to paper's effective batch). Non-trivial: prompt
   sharding logic must distinguish "TP group" from "DP rank", DDP-wrap
   the policy correctly under mixed-rank topology, vLLM TP-init must use
   a sub-group not the full world. Estimate: 100-200 line patch.

---

## Files modified in this session (V3 + handoff doc)

V3 changes (committed on `main` as `d38486c`):
- `aoae/checkpoints.py` (contract version 7→8 + comment)
- `aoae/models/policy.py` (Path C flags + forward)
- `aoae/speculative_inference.py` (`fallback_unmask` gate)
- `aoae/train_grpo.py` (Expert Steering plumbing)
- `configs/paper.yaml` (V3 hyperparameters + flags)
- `configs/paper_smoke.yaml` (regenerated from V3 paper.yaml)
- `tests/test_config_contracts.py` (added new sweep-point names from earlier sessions)

V3.5 changes (on branch `v3.5-paper-aligned`):
- `configs/paper.yaml` only — incremental hyperparameter changes listed above.

Handoff:
- `SYNC_5.5.md` — this document.

---

## Reading order for the next person picking this up

1. **This document** (top to bottom).
2. `paper/grpo_loop_iteration.md` — what one GRPO step does end-to-end.
3. `paper/5_3_soft_gating_hurts_verifier-performance.md` — why the
   training target is `quality_balanced_hardver`.
4. Skim the Jazbec et al. paper at https://arxiv.org/pdf/2512.09106
   (especially §3, §4.1-4.4, Appendix F, Appendix H).
5. Check the live V3 wandb run + `logs/10032928_train.out` for current
   state.

---

## Risks / things that might surprise the next person

- **Old policy checkpoints are not loadable** under V3's contract version
  8. The `input_proj` shape changed (D+6 → 5) when H_t was dropped. The
  contract check in `aoae/checkpoints.py` will refuse v6/v7 checkpoints.
  This is intentional; warm-starting from a v7 backbone wouldn't transfer
  cleanly anyway.
- **`paper.yaml` change affects eval too**. Setting `train_heads=
  ["unmask", "remask"]` is what the rollout uses, but eval also reads it
  (via `_apply_frozen_action_heads` which is called unconditionally from
  `speculative_inference`). Eval'ing an old checkpoint with the new
  `paper.yaml` will produce nonsense (untrained heads driving decisions).
  Workaround: use a separate eval config or override `train_heads` per
  sweep point.
- **`use_prism_score: true` in rollout overrides** triggers both PRISM
  score computation (good — feeds q_feat) AND PRISM gating in the
  verifier acceptance path *if* `acceptance_mode` is also set to
  `prism_gate`. We did NOT change `acceptance_mode` (stays `argmax_match`),
  so verifier acceptance is unchanged — only the policy q_feat input
  uses PRISM scores. Don't change `acceptance_mode` without thinking.
- **Smoke test config disables wandb** (`logging.use_wandb: false`).
  Production config has `use_wandb: true`. The launcher script
  `slurm/train.sh` hard-fails if `use_wandb=true` and `WANDB_API_KEY`
  is unset; the API key is hard-defaulted in the script itself.
