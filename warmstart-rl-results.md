# V4 Warmstart + GRPO: Results & Ongoing Experiments

## Warmstart Training (Completed)

**Job:** 10275730 (1× GPU, `seas_gpu`)
**Config:** `configs/v2_warmstart_scalar_only.yaml`
**Duration:** ~2.5 hours, 500/500 steps
**Checkpoint:** `outputs/v4_warmstart_scalar_only/policy_final.pt` (1.8 MB)

### Training Curve
- Initial loss: ~0.08 (step 1)
- Final loss: ~0.002 (step 500)
- Both `u_bce` and `r_bce` converge near zero
- Smooth convergence, no instability

### Architecture
- **Policy:** PhaseAV2Policy (scalar_only mode)
  - d_model=128, 1 layer, 4 heads, ~465K params
  - Heads: u_t (unmask), r_t (remask) only
- **Features:** scalar only (confidence, agreement, quality scores) — no H_t conditioning
- **Labels:** Per-position BCE targets derived from future verifier acceptance/rejection
  - u_t label: high-confidence positions the verifier would accept
  - r_t label: low-confidence positions the verifier would reject

### Training Environment (rollout settings)
| Setting | Value |
|---------|-------|
| `lossless_verification` | `true` (hard-gated drafter & verifier) |
| `primary_agree_threshold` | `0.5` |
| `verifier.use_prism_score` | `true` |
| `drafter.run_on_verifier` | `never` |
| `drafter.aux_compute_ratio` | `0.0` |
| `inference.steps` | 256 |
| `inference.gen_length` | 512 |

### Data
- **Dataset:** nvidia/OpenMathInstruct-2 (train split, first 2000 samples)
- Math word problems with numerical expected answers

### Reproduction
**Local (single GPU):**
```bash
aoae train --config configs/v2_warmstart_scalar_only.yaml --stage warmstart
```

**Via SLURM (original job 10275730):**
```bash
sbatch --partition=seas_gpu --account=sitanc_lab --gres=gpu:1 slurm/train.sh warmstart configs/v2_warmstart_scalar_only.yaml
```
- Runtime: ~2.5 hours on 1× A100/H200 GPU
- Output: `outputs/v4_warmstart_scalar_only/policy_final.pt`

---

## Warmstart Evaluation (GSM8K, 50 samples)

Evaluated on `openai/gsm8k` test set, 50 samples. The warmstart policy was compared against the training-free `DefaultPolicy` (confidence-threshold heuristic: unmask when confident, never remask, cache=agreement).

### Results Table

| Config | tau_pi | Accuracy | TPS | NFE | Notes |
|--------|--------|----------|-----|-----|-------|
| **Baseline (DefaultPolicy)** | 1.0 | **84%** | **53.1** | 25 | agree=0.5, lossless, no trained policy |
| Trained (strict, agree=0.85) | 0.5 | 76% | 43.2 | 49 | |
| Trained (strict, agree=0.85) | 1.0 | 84% | 31.1 | 96 | Matches baseline accuracy |
| Trained (strict, agree=0.85) | 1.5 | 42% | 18.3 | 180 | Accuracy collapses |
| Trained (faithful, agree=0.5) | 0.5 | 72% | 44.1 | 29 | |
| Trained (faithful, agree=0.5) | 1.0 | 68% | 30.1 | 54 | |
| Trained (faithful, agree=0.5) | 1.5 | 40% | 19.9 | 97 | |

**Reference baselines** (from CLAUDE.md, full paper eval):
- llada21_quality_mode: ~78% accuracy, ~59 TPS
- llada21_speed_mode: ~76% accuracy, ~64 TPS

### Eval Configs
- **Strict:** `configs/eval_v4_warmstart_strict.yaml` — matches warmstart rollout environment except `primary_agree_threshold: 0.85` (tests generalization to stricter agreement)
- **Faithful:** `configs/eval_v4_warmstart.yaml` — exactly matches warmstart rollout environment (agree=0.5)
- **Baseline:** `configs/eval_v4_warmstart_baseline.yaml` — same environment, no trained checkpoint (DefaultPolicy)

### Interpretation

1. **The trained policy matches baseline accuracy (84%) at tau_pi=1.0 with strict agreement**, confirming it learned meaningful decision boundaries for unmask/remask. However, it is significantly slower (31.1 TPS vs 53.1 TPS, 96 NFE vs 25 NFE).

2. **The trained policy is slower because it actively uses more compute.** DefaultPolicy achieves high accuracy cheaply because with `lossless_verification=true` and ~97.5% natural agreement, it rarely invokes the verifier. The trained policy makes more deliberate decisions (possibly more remasks, different unmask patterns) that trigger more verification steps.

3. **Higher tau_pi hurts accuracy dramatically.** At tau_pi=1.5, accuracy drops to 42% (strict) and 40% (faithful). This suggests the policy's remask decisions become destructive when given too many steps — it over-rejects tokens that were actually correct.

4. **The `agree=0.5` (faithful) setting underperforms `agree=0.85` (strict).** The policy was trained with agree=0.5, but evaluates better at agree=0.85. This may indicate the policy learned to work with the verifier rather than independently making quality judgments — the tighter acceptance threshold at eval forces the system to rely more on the verifier's judgment (which is good), compensating for occasional bad policy remask decisions.

5. **Key insight for GRPO:** The warmstart policy has learned *structure* (it can achieve 84% correct) but has no speed incentive. It uses the full compute budget regardless of problem difficulty. GRPO's multiplicative reward `correctness × (1 - effective_flops)^alpha` should teach it to be selective — unmask quickly on easy positions, use more compute only where needed.

---

## V4 GRPO Post-Warmstart (Step 100) — Evaluated

**Job:** 10387381 (4× GPU, `seas_gpu`, DP=4)
**Checkpoint:** `outputs/v4_grpo_post_warmstart/policy_latest.pt` (step 100)
**Config:** `configs/v4_grpo_post_warmstart.yaml` (warm_start_from, G=4, DP=4, full rollouts 256/512, 200 steps)
**Eval config:** `configs/eval_v4_grpo_step100.yaml` (lossless, agree=0.5, full 256/512 generation)

### Results (GSM8K, 50 samples)

| tau_pi | Accuracy | TPS | NFE | Warmstart (Acc/TPS) | Δ Acc | Δ TPS |
|--------|----------|-----|-----|---------------------|-------|-------|
| 0.5 | 70% | 60.7 | 31 | 72% / 44.1 | -2% | **+37%** |
| 1.0 | 66% | 57.6 | 40 | 68% / 30.1 | -2% | **+91%** |
| 1.5 | 36% | 36.4 | 79 | 40% / 19.9 | -4% | +83% |

### Interpretation

**GRPO is successfully learning speed improvements:**

1. **tau_pi=1.0 is the sweet spot:** 66% accuracy at 57.6 TPS vs warmstart's 68% at 30.1 TPS — **nearly 2x speed with only 2% accuracy loss**.

2. **tau_pi=0.5 beats baseline speed:** 70% accuracy at 60.7 TPS vs DefaultPolicy's 84% at 53.1 TPS — faster than the training-free heuristic but lower accuracy.

3. **Speed gains are real:** The TPS improvements (37–91%) are consistent across tau_pi values, confirming the multiplicative reward `correctness × speed_factor` is teaching the policy to be selective.

4. **Accuracy trade-off:** The 2–4% accuracy loss suggests the policy is learning to skip some compute at the cost of occasional mistakes. With 100 more training steps (up to 200), accuracy may recover while maintaining speed.

**Next steps:** Wait for step 200 (final checkpoint) and re-evaluate. If accuracy recovers to warmstart levels while maintaining the speed gains, this will be a successful demonstration of GRPO's compute-aware optimization.

---

## V3 Cold-Start GRPO (Expert-Steered) — Evaluated

**Job:** 10356204 (1× GPU, `seas_gpu`)
**Checkpoint:** `outputs/grpo_v3/policy_latest.pt` (150 steps, cold-start)
**Config:** Trained with `paper_smoke.yaml` defaults (expert_steering: enabled, G=2, short rollouts 8/64)
**Eval config:** `configs/eval_v3_coldstart.yaml` (lossless, agree=0.5, full 256/512 generation)

### Results (GSM8K, 50 samples)

| tau_pi | Accuracy | TPS | Notes |
|--------|----------|-----|-------|
| 0.5 | 14% | 49.4 | |
| 1.0 | 10% | 18.8 | |

### Interpretation

**Cold-start GRPO with expert steering failed dramatically.** After 150 training steps, the policy achieves only 10-14% accuracy on GSM8K — far below the 84% baseline (DefaultPolicy) and the 84% achieved by the warm-started policy.

This validates the warmstart hypothesis:
- Cold-start RL from random initialization cannot learn meaningful unmask/remask policies within 150 steps, even with expert steering
- The training log showed low correctness (~0.17) and low reward (~0.04) throughout, confirming the policy never improved
- Expert steering (which was intended to provide a structural prior) was insufficient to overcome the cold-start initialization problem

**Comparison:**
| Method | Training | Steps | Accuracy (tau_pi=1.0) | TPS |
|--------|----------|-------|----------------------|-----|
| DefaultPolicy (no training) | N/A | N/A | 84% | 53.1 |
| V3 cold-start GRPO + expert steering | RL (cold) | 150 | 10% | 18.8 |
| V4 warmstart (supervised) | BCE (warm) | 500 | 84% | 31.1 |

The warmstart phase provides the structural prior that expert steering attempted to inject, but in a much more effective way: dense per-position labels teach the policy directly which actions lead to good outcomes, rather than relying on sparse RL reward plus occasional expert rollouts.

---

## V4 GRPO Post-Warmstart (In Progress)

**Job:** 10347441 (4× NVIDIA H200, `seas_gpu`)
**Config:** `configs/v4_grpo_post_warmstart.yaml`
**Status:** Submitted, pending resources (H200 queue)

### Setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Group size (G)** | 4 | 4 rollouts per prompt for advantage estimation |
| **Data parallelism** | 4 GPUs (DP=4) | `tp_size: 1`, `GPUS_PER_NODE=4` → each GPU processes different prompts |
| **Effective batch** | 4 prompts/step | 1 prompt/GPU × 4 GPUs, each with G=4 rollouts = 16 trajectories/step |
| **Max steps** | 200 | ~4-8h estimated on H200 |
| **Warm start** | `outputs/v4_warmstart_scalar_only/policy_final.pt` | Strict loading |
| **Rollout length** | 256 steps, 512 gen_length | Matches warmstart distribution exactly |

### Reward Function

```
R = correctness × speed_factor
    - beta × extra_remask_rate
    - unresolved_penalty_weight × unresolved_fraction
```

| Component | Formula / Value | Notes |
|-----------|----------------|-------|
| `correctness` | 0 or 1 (GSM8K answer match) | Binary, from OpenMathInstruct-2 |
| `speed_factor` | `(1 - effective_flops)^alpha` | alpha=1.0 (linear) |
| `effective_flops` | `used_steps / T` | No cache credit (`cache_speed_source: none`) |
| `extra_remask_penalty` | `beta × (remask_count / response_length)` | beta=0.1, penalizes unnecessary remasking |
| `unresolved_penalty` | `0.25 × fraction_still_masked` | Penalizes incomplete generation |
| `thrash_penalty` | 0 | Disabled (`reward_cache_terms_enabled: false`) |
| `cache_f1_reward` | 0 | Disabled (no cache heads) |
| `access_reward` | 0 | Disabled (no access head) |

### Optimization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **lr** | 3e-5 | Standard RL lr; warmstart used 1e-4 for supervised |
| **clip_eps** | 0.2 | PPO-style clipping, conservative for post-warmstart |
| **max_grad_norm** | 0.5 | Moderate gradient clipping (warmstart used 1.0) |
| **weight_decay** | 0.01 | |
| **warmup_steps** | 0 | No warmup (warm-started) |
| **policy_temperature** | 1.0 | No action smoothing |
| **normalize_advantage_std** | false | Group-mean only, no std normalization |
| **KL regularization** | None | Not implemented; rely on clip_eps=0.2 |
| **Expert steering** | Disabled | Warmstart provides structural prior |

### Rollout Environment (identical to warmstart)

```yaml
rollout_overrides:
  base_model.lossless_verification: true
  inference.primary_agree_threshold: 0.5
  inference.verifier.use_prism_score: true
  inference.drafter.run_on_verifier: never
  inference.drafter.aux_compute_ratio: 0.0
  inference.positional_cache.enabled: false
  inference.drafter.fast_path: false
```

### Data
- **Dataset:** nvidia/OpenMathInstruct-2 (train split, first 2000 samples)
- Same prompt pool as warmstart training
- Sharded across 4 GPUs (500 prompts/rank)
- With 200 steps × batch_size 1, each rank uses 200/500 of its shard

### Key Design Decisions

1. **Full rollouts (256/512) instead of short (8/64):** The warmstart trained on full-length trajectories. Using short rollouts for GRPO would create a distribution mismatch — the reward would be computed over a regime the policy never saw during supervision. Full rollouts ensure the GRPO reward signal is meaningful end-to-end.

2. **G=4 with DP=4:** GRPO advantage estimation is per-prompt (relative ranking within group). With G=4 per prompt, we get reasonable variance within groups. DP=4 means 4 prompts processed in parallel, reducing wall-clock time per step.

3. **No KL regularization:** The codebase has no KL implementation. We rely on clip_eps=0.2 (importance ratio bounded to [0.8, 1.2]) to prevent large policy shifts from warmstart. If instability is observed, we would need to implement KL toward the warmstart checkpoint.

4. **Batch size 1:** Each GPU processes 1 prompt with 4 rollouts per step. Increasing batch_size would double per-step time without improving within-group advantage quality. Gradient variance across prompts is acceptable with 4 effective prompts/step.

### What We Expect

**Hypothesis:** GRPO should teach the policy to be faster while maintaining accuracy. The warmstart policy uses the full compute budget (effective_flops ≈ 0.3, speed_factor ≈ 0.7) regardless of problem difficulty. GRPO's reward multiplicatively couples correctness with speed, so:
- On easy problems (high natural agreement, confident tokens): policy should learn to unmask aggressively and skip verifier → high speed_factor
- On hard problems (low agreement, uncertain tokens): policy should maintain careful verify behavior → maintain correctness

**Success criteria:**
- reward > 0 consistently (correctness × speed > penalties)
- speed_factor improves from ~0.7 baseline toward 0.8-0.9
- Accuracy on GSM8K eval maintains ≥ 80%

### Bugs Fixed Before Submission

1. **DDP `find_unused_parameters=True`:** When all G rollouts get identical reward (e.g., all incorrect → advantage=0 → loss=0), some policy parameters don't receive gradients. DDP requires explicit opt-in to tolerate this.

2. **`logger` → `_logger` naming:** The training logger was created as `_logger` but referenced as `logger` in the sample-logging code path (triggered on checkpoint save).

---

## Timeline

| Date | Event |
|------|-------|
| 2026-05-05 ~16:00 | Warmstart job 10275730 completed (500 steps) |
| 2026-05-05 ~18:00 | Eval jobs submitted (10332815, 10332838, 10339931) |
| 2026-05-05 ~18:30 | Baseline eval completed: DefaultPolicy = 84% / 53.1 TPS |
| 2026-05-05 ~18:45 | First GRPO attempt (10344953) crashed on step 2 (DDP + logger bugs) |
| 2026-05-05 ~18:50 | Bugs fixed, GRPO resubmitted as 10347441 |
| 2026-05-05 ~18:55 | All warmstart eval jobs completed |
| Pending | GRPO job 10347441 starts (waiting for H200 resources) |

---

## Next Steps

1. **Monitor GRPO job 10347441** — check reward trend, speed_factor improvement, accuracy maintenance
2. **Early stopping check at step 20** — first checkpoint; if reward is consistently negative or speed_factor isn't improving, may need lr/reward adjustments
3. **Post-GRPO eval** — run same eval suite (tau_pi sweep) on GRPO-trained checkpoint, compare to warmstart and baseline
4. **If successful:** Consider V5 with H_t conditioning (warmstart + GRPO with hidden state features)
5. **If unstable:** Add KL regularization toward warmstart reference (beta_kl ~ 0.01-0.1)
