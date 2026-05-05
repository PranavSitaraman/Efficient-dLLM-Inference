# Prompt: Implement Unified Policy Head v2

Implement the **Phase A** v2 unified policy head and training pipeline for AOAE
speculative inference in this repository.

You must follow the design in:

- [unified_policy_head_v2.md](/Users/gavinye/Efficient-dLLM-Inference/unified_policy_head_v2.md:1)

Do not reintroduce deferred Phase C ideas such as cache/access head training or
stable-KV execution.

Current project status:

- the **2-step training idea is relatively agreed**
- the **exact feature set is still under review**

Implement the training structure precisely, but treat feature choices as a
review-sensitive area that should remain modular and easy to adjust.


## Objective

Build a rigorous Phase A policy that learns:

- `u_t`: which masked positions to draft on drafter microsteps
- `r_t`: which unmasked positions to remask on verifier microsteps

The policy must remain a **position-space control policy**. It must not predict
vocabulary logits or replace the base model token head.


## Design Requirements

### 1. Architecture

Implement a shared lightweight contextual trunk plus two scoped heads:

- shared input projection
- shared 1-layer bidirectional transformer encoder
- `head_unmask`
- `head_remask`

Remove Phase A dependence on:

- cache head training
- access head training
- boundary head training

You may leave legacy modules in place for backward compatibility, but the v2
training and inference path must treat Phase A as a two-head design.

Important:

- the shared trunk is part of the learned policy
- warm-start supervision must update the trunk **and** both heads
- GRPO fine-tuning must continue updating the trunk **and** both heads

Do not implement warm start as "only initialize the final linear heads."


### 2. Scoped execution domains

Do **not** rely on a full-sequence feature `m_t` as a learned input if you can
avoid it.

Instead:

- evaluate `u_t` only on masked positions `k in M_t`
- evaluate `r_t` only on verifier-eligible unmasked positions

Mask status should be handled primarily by indexing / validity logic, not by
asking the model to relearn it as a feature.


### 3. Feature sets

Use head-specific features.

This section is **not fully locked**. Implement it in a modular way so the
feature inventory can be revised without rewriting the whole training stack.

For `u_t`, the preferred feature set is:

- `H_t^k`
- explicit `confidence_t^k`
- `step_t`
- optional `age_t^k`

Do **not** rely on `agreement_t` as a core `u_t` feature, because fresh
agreement is unavailable on drafter-only microsteps in the current speculative
loop.

For `r_t`, the preferred feature set is:

- `H_t^k`
- `confidence_t^k`
- `agreement_t^k`
- `step_t`
- optional `age_t^k`
- optional `frontier_membership_t^k`

This is the current best proposal, but the verifier feature set should be easy
to revise during review.

Important:

- `step_t` must be included explicitly
- `SoftMaskedState` currently does not encode `mask_indicator` or `step_frac`
  into `H_t`, so do not assume the hidden state already contains them


### 4. Action parameterization

During training:

- parameterize each head as Bernoulli logits
- `u_t^k ~ Ber(sigmoid(z_u^k))`
- `r_t^k ~ Ber(sigmoid(z_r^k))`

This is for tractable GRPO likelihoods.

At inference:

- interpret the scores as **budgeted ranking signals**
- `u_t`: rank candidates and apply the existing per-step unmask budget
- `r_t`: rank verifier-side rollback candidates and remask conservatively

Do not treat v2 as "independent coin flips everywhere" in the execution path.


## Training Pipeline Requirements

### 5. Warm-start stage for `u_t`

Add a supervised or imitation stage before GRPO.

This stage trains:

- the shared trunk
- `head_unmask`
- `head_remask` only insofar as it participates in the joint module forward
  path, unless the implementation cleanly isolates losses per head

At minimum, the shared trunk and `head_unmask` must be updated by this stage.

The `u_t` teacher should not simply be the raw heuristic draft set.

Instead, construct labels from the existing speculative loop:

- positive: positions selected by the heuristic drafter that were later
  accepted by the verifier
- negative: positions selected by the heuristic drafter that were later
  rejected by the verifier
- unlabeled or weakly weighted: positions never selected by the heuristic

Goal:

- learn the **accepted subset** of heuristic draft proposals
- avoid copying teacher false positives

Use a loss that supports partial or weighted supervision if needed.


### 6. Warm-start stage for `r_t`

Add a supervised stage for `r_t`.

This stage also trains the shared trunk jointly with `head_remask`.

Primary supervision:

- positive: positions explicitly rejected by verifier frontier validation
- negative: accepted frontier positions and stable kept tokens

Optional auxiliary supervision:

- PRISM-style low-quality labels or soft scores

PRISM should be treated as:

- auxiliary teacher
- distillation signal
- or pseudo-label source

not as the sole definition of rollback correctness.


### 7. GRPO fine-tuning stage

After warm start, fine-tune `u_t` and `r_t` with GRPO.

Requirements:

- keep the speculative loop structure intact
- retain `u_t` on drafter microsteps and `r_t` on verifier microsteps
- preserve the current strong heuristic baseline behavior outside the learned
  scopes
- continue to use position-space policy likelihoods

GRPO should improve on the heuristic teacher rather than rediscover the entire
speculative policy from scratch.


## Required 2-Step Narrative

Make the implementation and documentation reflect this exact Phase A training
story:

### Step 1: supervised warm start

Train the whole v2 policy module so it starts near the current strong
speculative loop:

- `u_t` learns the useful subset of heuristic draft proposals
- `r_t` learns verifier-side rollback behavior, optionally aided by PRISM
- the shared trunk learns contextual features for both tasks

### Step 2: GRPO fine-tuning

Start from the warm-started policy and fine-tune with end-to-end speculative
reward so the learned policy can outperform its heuristic teachers and improve
speculation dynamics.

This 2-step story should be explicit in:

- code structure
- configs
- training entrypoints
- documentation/comments where relevant


## Concrete Implementation Tasks

1. Refactor the policy module so Phase A can use head-specific feature inputs
   cleanly.
2. Add explicit confidence and step features to the v2 path.
3. Add verifier-frontier membership as an optional `r_t` feature.
4. Remove dependence on `m_t` as a learned feature in the v2 scoped-head path.
5. Build dataset / trajectory extraction utilities for:
   - `u_t` accepted-vs-rejected heuristic proposals
   - `r_t` verifier rejection targets
6. Add a warm-start training entrypoint or stage configuration for the v2 head
   that trains the shared trunk jointly with the supervised heads.
7. Wire the warm-started checkpoint into the existing GRPO pipeline.
8. Add config flags that clearly separate:
   - legacy path
   - v2 warm start
   - v2 GRPO fine-tuning
9. Add focused tests for:
   - scoped candidate extraction
   - feature construction
   - label construction
   - compatibility with speculative inference rollout recording


## Constraints

- Do not add vocab prediction to the policy head.
- Do not enable stable-KV execution.
- Do not re-enable cache/access heads in Phase A.
- Do not silently rely on stale exploratory design notes over
  `unified_policy_head_v2.md`.
- Prefer minimal, justified features over feature sprawl.


## Deliverable

Return:

- the code changes
- the new configs / training stages
- a short explanation of how the shared trunk is trained in step 1 and step 2
- a short explanation of how warm-start targets are constructed
- any residual assumptions or open risks
