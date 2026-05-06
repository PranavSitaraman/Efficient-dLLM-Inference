# Verifier Design and Proposal Mapping

## Purpose

This note maps the proposal's "plug-and-play verifier" idea to the actual codebase implementation.

The core split is:

- The **drafter** is the auxiliary hard-routed model inside `aoae/models/dual_model.py`.
- The **verifier** is a separate scoring module that produces per-position `quality_scores`.
- The **stability cache** (`K_stable`) is separate from both of the above.

So the verifier is not the cache and is not the drafter; it is the quality-scoring component that conditions the policy.

## How the proposal is realized

The proposal text describes "PRISM or a PRISM-like head." In code, that is now realized in `aoae/models/verifier.py` with a common interface:

- `PRISMVerifier`
  - Wraps the existing `PRISMAdapter`
  - Uses primary-model hidden states
  - Can be frozen or trainable under GRPO
- `LearnedVerificationHead`
  - Fresh MLP verifier head
  - Uses primary-model hidden states
  - Trainable under GRPO
- `ConfidenceVerifier`
  - Logits-only heuristic verifier
  - Supports `max_prob`, `margin`, and `one_minus_entropy`
  - Stateless, useful for ablations that should not depend on hidden-state extraction

## Config surface

Verifier behavior is controlled by the `verifier:` section in config:

```yaml
verifier:
  enabled: true
  kind: "prism"        # prism | learned_head | confidence | none
  trainable: false
  artifact_name: "prism_adapter.pt"
  score_mode: "max_prob"
  hidden_dim: 256
  threshold: 0.5
  dropout: 0.0
```

Important conventions:

- `kind: prism`, `trainable: false` means frozen PRISM
- `kind: prism`, `trainable: true` means unfrozen PRISM
- `kind: learned_head` means a fresh verifier head saved as `verifier_head.pt`
- `kind: confidence` means no learned artifact is required

## Where the verifier is used

### Single-model AOAE path

`aoae/inference.py`

- Runs the verifier on primary hidden states or logits
- Feeds the resulting `quality_scores` into `call_policy(...)`
- Stores verifier inputs in the trajectory so trainable verifier backends can be recomputed inside GRPO loss

### Speculative path

`aoae/speculative_inference.py`

- Runs the drafter (auxiliary hard-routed model) when `cache.kspec_skip` is enabled
- Runs the primary verifier path and computes agreement
- Runs the configured verifier backend to produce `quality_scores`
- Feeds `quality_scores` plus `agreement` into the policy

Implementation note:

- If PRISM or KV-dynamics tracking needs primary hidden states, the primary takes a
  full verifier forward on those verifier steps.
- That no longer disables the auxiliary prefix cache globally: aux-only draft
  steps and the auxiliary side of verifier steps can still reuse `C_aux`.

This means the verifier currently influences remasking **through the policy**, not by directly hard-thresholding tokens itself.

## What "trainable verifier" means in GRPO

For `verifier.trainable: true`, GRPO now:

- includes verifier parameters in the optimizer
- stores verifier inputs (`verifier_hidden_list`, `verifier_logits_list`) in each rollout trajectory
- recomputes verifier scores inside `compute_grpo_loss(...)`

This is necessary because otherwise the verifier would sit in the optimizer without any gradient path.

## Speculation vs. KV-stability cache

These are intentionally separable:

- `cache.kspec_skip`
  - enables the drafter/verifier speculative path
  - `K_spec` is the transient drafted frontier awaiting the next verifier pass
- `cache.stable_kv_cache`
  - enables the persistent primary-owned stability cache
  - `K_stable` is the multi-step cache managed by the policy's `kappa_t` head

If both are enabled:

- the aux drafter / verifier agreement path stays active
- the stable-primary KV cache remains active on verifier steps
- speculative wall-time gains come from fewer primary verifier passes
- stable-primary KV reuse compounds that by making the verifier steps themselves cheaper

The speculative cadence is controlled by `inference.primary_every_n`:

- `1` means verify every step
- `>1` means the auxiliary drafts for multiple steps before the primary/verifier runs again

That is the main implementation caveat to remember.

## Why `K_spec` is still needed even though accepted tokens already live in `y`

Accepted tokens do stay directly in the sequence state `y`. `K_spec` is still
useful because it marks which positions are still *tentative* since the last
verifier pass. The sequence alone does not tell us that.

So `K_spec` provides:

- the transient frontier the verifier still needs to validate
- the set of positions whose drafter-side progress is still provisional
- a training/eval signal about where the drafter and verifier have not yet reconciled

Once the verifier runs, accepted positions simply remain in `y` and are removed
from `K_spec`. New drafts then populate a fresh frontier for the next burst.

## Current recommendation for final verification design

The cleanest ablation ladder is:

1. Frozen PRISM
2. Unfrozen PRISM
3. Learned verification head
4. Confidence-based verifier

This keeps the design progression aligned with the proposal:

- start from the literature-grounded verifier
- test whether GRPO benefits from tuning it
- test whether a fresh verifier head can replace it
- test whether a much cheaper heuristic verifier is already sufficient
