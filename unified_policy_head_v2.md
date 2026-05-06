# Unified Policy Head v2

## Purpose

This document finalizes a rigorous **Phase A** design for the AOAE unified
policy head used in speculative inference with GRPO.

The goal is to learn **better speculation dynamics** while preserving the
strong hand-designed speculative loop that already performs well. In Phase A,
the policy controls **where to act in position space**, while the base model
continues to control **which token to write in vocabulary space**.

This document intentionally excludes deferred ideas unless they are explicitly
marked as future work.


## Review Status

The current state of this proposal is:

- **agreed / relatively stable**: the 2-step training pipeline
- **agreed / relatively stable**: shared trunk + two position heads (`u_t`,
  `r_t`)
- **still under review**: the exact finalized verifier/drafter feature sets

So this document should be read as:

- a near-final Phase A training/design proposal
- with a feature section that is still open to refinement before implementation


## Scope

Phase A covers only:

- `u_t`: which masked positions to unmask on drafter microsteps
- `r_t`: which unmasked positions to remask on verifier microsteps

Phase A does **not** include:

- `kappa_t` / stable-cache commit decisions
- `q_t` / access prediction
- stable-KV execution
- boundary head training
- LM-head replacement or vocab prediction by the policy

Those are deferred to later phases.


## Core Position

The unified policy head should be a **position-selection policy**, not a token
prediction policy.

Formally:

- The base model outputs token distributions over vocabulary `V`
- The policy outputs action scores over positions `k in {1, ..., L}`

So the policy lives in **position space**, not vocabulary space.


## Relation to PRISM

PRISM and `r_t` are mathematically analogous in one narrow sense:

- both can be represented by a scalar `s_t^k in [0, 1]` per position
- both can induce a binary remask decision in position space

For example, one may interpret:

`r_t^k ~ Ber(1 - q_t^k)`

where `q_t^k` is a quality/correctness score.

However, they are **not the same training object**.

PRISM:

- is a supervised correctness estimator
- predicts whether the current token is likely correct
- is trained with BCE on corrupted/clean pairs

Phase A `r_t`:

- is a control policy for speculative decoding
- should learn whether remasking this position improves end-to-end
  speed-quality behavior
- is trained through rollout reward, not only token correctness labels

So PRISM is best viewed as a possible **feature**, **teacher**, or
**initialization target** for `r_t`, not as a full replacement for it.


## Final Action Parameterization

We keep the action space factorized at the head level:

- drafter head: `u_t`
- verifier head: `r_t`

Each head produces one scalar logit per position.

### Training parameterization

For GRPO, define:

- `p_u^k = sigmoid(z_u^k)`
- `p_r^k = sigmoid(z_r^k)`

and treat:

- `u_t^k ~ Ber(p_u^k)` on drafter microsteps
- `r_t^k ~ Ber(p_r^k)` on verifier microsteps

This is justified because GRPO needs a tractable policy likelihood, and
factorized Bernoulli heads provide that cleanly.

### Execution parameterization

At inference, we should not think of the policy as "many independent coin
flips." The real control problem is budgeted subset selection.

Therefore, the execution semantics should be:

- `u_t`: rank masked positions by `z_u^k` or `p_u^k`, then apply the per-step
  unmask budget
- `r_t`: rank eligible verifier-side positions by `z_r^k` or `p_r^k`, then
  threshold or budget remasks conservatively

So:

- **Bernoulli is the training parameterization**
- **priority ranking is the decoding interpretation**

This is the cleanest way to reconcile GRPO tractability with the actual
structure of the inference loop.


## Final Architecture

Use one shared contextual trunk plus two scoped action heads.

### Trunk

- input projection
- 1-layer bidirectional transformer encoder

Purpose:

- allow each position decision to depend on the global response state
- couple decisions across positions through contextualization
- remain lightweight enough for rollout training

### Heads

- `head_unmask`: scalar logit per position
- `head_remask`: scalar logit per position

No cache head and no access head in Phase A.

The shared trunk is part of the learned policy and should be updated jointly
with both heads during warm start and GRPO. The warm-start stage is not just
"head initialization"; it is supervised initialization of the whole v2 policy
module.


## Scope Semantics

The responsibility split is fixed:

- `u_t` is active only on **drafter** microsteps
- `r_t` is active only on **verifier** microsteps

Outside scope:

- verifier-side unmasking falls back to the canonical deterministic rule
- drafter-side remasking is disabled

This preserves the current strong speculative loop while allowing RL to learn:

- what to draft
- what to roll back

without destabilizing the verifier baseline.


## Proposed Feature Set (Under Review)

The design should use a **small, justified** feature set.

This section is intentionally marked **under review**. The training structure
below is more stable than the exact feature inventory.

### Required features

1. `H_t`

- soft-masked hidden representation
- primary state summary for each position
- the most important feature

Important implementation fact:

- in the current code, `SoftMaskedState.forward(...)` explicitly discards both
  `mask_indicator` and `step_frac`
- therefore `H_t` does **not** contain exact information about whether a
  position is masked or how many steps remain

2. `t / T`

- diffusion progress
- needed because the speculative control problem is finite-horizon and
  non-stationary

Rigorous justification:

- the same local hidden state can arise at different remaining horizons
- but the optimal action can differ because the value-to-go differs

For `u_t`:

- early in the rollout, a medium-confidence position can be deferred because
  future verifier opportunities still exist
- late in the rollout, the same position should often be drafted now to avoid
  unresolved-mask penalty or fallback

For `r_t`:

- early in the rollout, remasking a borderline token is relatively cheap
  because repair opportunities remain
- late in the rollout, the same rollback can be too expensive if it risks
  leaving the token unresolved before termination

So `step_t` is not an arbitrary extra feature. It is the simplest way to let a
shared policy represent the correct finite-horizon policy.

PRISM does not need `step_t` for the same reason that a token-correctness
estimator does not need a planning horizon: PRISM predicts local correctness,
not control value.

### Strongly recommended features

3. `confidence_t`

- explicit primary confidence or margin feature
- important because the current strong heuristic frontier is confidence-driven
- should be fed explicitly rather than expected to be fully recoverable from
  `H_t`

4. `agreement_t`

- auxiliary-primary agreement signal
- especially useful for verifier-side `r_t`
- can also help diagnose fragile draft positions

Important scope note:

- fresh agreement is only observed on verifier microsteps
- on drafter-only microsteps, the current code zero-fills agreement
- therefore agreement should not be treated as a core drafter feature in
  Phase A

### Optional Phase A features

5. `age_t`

- steps since the position last changed
- useful as a stability proxy
- modest complexity, reasonable to keep

6. `frontier_membership_t`

- whether the token is part of the currently verified speculative frontier
- especially useful for `r_t`
- semantically cleaner than several indirect proxy features

### Excluded from finalized Phase A

7. PRISM score as a required input

- not needed for the first rigorous Phase A design
- may be used later as an ablation, teacher, or warm-start target for `r_t`
- should not be mandatory in the base design

8. `last_action_feature`

- weakly justified on its own
- can be dropped unless ablations show value beyond `age_t`

9. cache/access-related features

- out of scope in Phase A


## Head-Specific State

The final Phase A design should be **head-specific**, even if a shared trunk is
retained.

### Drafter state for `u_t`

Apply only on masked positions `k in M_t`.

Preferred feature set:

- `H_t^k`
- `confidence_t^k`
- `step_t`
- optional `age_t^k`

Not recommended as a core feature:

- `agreement_t^k`, because fresh agreement is unavailable on drafter-only
  microsteps in the current speculative loop

### Verifier state for `r_t`

Apply only on verifier-eligible unmasked positions.

Preferred feature set:

- `H_t^k`
- `confidence_t^k`
- `agreement_t^k`
- `step_t`
- optional `age_t^k`
- optional `frontier_membership_t^k`

This means the unified policy is "unified" at the trunk level, but not forced
to use exactly the same semantic feature set in both heads.


## Is `m_t` Necessary?

`m_t` is necessary only under one implementation style, and unnecessary under a
better-scoped one.

### If heads run on the full sequence

Then `m_t` is useful, because:

- `H_t` does not encode mask status exactly
- the same-looking hidden state could correspond to a masked or already
  committed token
- action admissibility depends on mask status

### If heads run only on their valid domains

- `u_t` runs only on masked positions
- `r_t` runs only on verifier-side unmasked positions

then `m_t` becomes redundant as an explicit feature, because domain membership
is already known by construction.

### Final recommendation on `m_t`

For Phase A, prefer the scoped implementation:

- evaluate `u_t` only over masked positions
- evaluate `r_t` only over eligible verifier-side unmasked positions

Under that design, **drop `m_t` from the learned feature set**. Keep mask
status only as execution-time indexing / validity logic.


## What the Heads Should Mean

### Unmask head `u_t`

`u_t` should estimate:

"If I spend draft budget on this masked position now, how likely is that to
produce a useful speculative proposal?"

This is not "is the token correct?" It is closer to:

- likely to be accepted soon
- likely to reduce future verifier work
- unlikely to cause wasted draft effort

### Remask head `r_t`

`r_t` should estimate:

"Given the verifier-side state, is it better to roll this position back and
re-denoise it than to keep the current token?"

This is related to token quality, but broader than PRISM:

- local token correctness matters
- verifier disagreement matters
- speculation dynamics matter
- rollback cost matters


## Token Generation Is Not Learned Here

The policy head does **not** output token logits.

Token identity still comes from the model:

- drafter path: auxiliary logits
- verifier path: primary logits

This separation is desirable because:

- the base model already knows how to score tokens
- the speculative control problem is primarily about **where** to spend compute
- mixing vocab prediction into the policy would greatly increase variance and
  complexity without clear need in Phase A


## Two-Step Training Strategy

The policy should not be trained from scratch against the full end-to-end
reward alone.

The v2 training plan has two explicit stages:

1. **Warm start / supervised imitation**
2. **GRPO fine-tuning**

Both stages train the full v2 policy module:

- shared input projection
- shared transformer trunk
- `head_unmask`
- `head_remask`

So warm-start supervision updates:

- the individual heads
- the shared contextual backbone / extra attention layer

This matters because the trunk must learn contextual features useful for:

- selecting promising draft positions
- selecting verifier-side rollback positions

and should not remain a random or frozen feature extractor.

### Recommended training recipe

1. Behavioral warm start for `u_t`

- imitate the current strong drafter heuristic, but only where that heuristic
  was actually useful

Recommended label construction:

- positive: positions selected by the heuristic drafter that were later
  accepted by the verifier
- negative: positions selected by the heuristic drafter that were later
  rejected by the verifier
- unlabeled or weakly weighted: positions the heuristic never selected

This is better than treating the full heuristic selection set as positive,
because it teaches `u_t` to draft the **accepted subset** of heuristic
proposals rather than blindly copying the teacher's false positives.

2. Behavioral warm start for `r_t`

- imitate verifier rejection/remask behavior on verifier microsteps
- optionally augment with PRISM-style low-quality targets

Recommended label construction:

- positive: positions explicitly rejected by verifier frontier validation
- positive: optionally, positions whose PRISM score falls below threshold
- negative: accepted frontier positions and stable kept tokens

3. GRPO fine-tuning

- optimize end-to-end speculation reward
- allow deviation from the teacher behaviors where reward improves

This gives the policy a strong initial operating point near the current
successful speculative loop instead of forcing RL to rediscover it.


## How the Speculative Loop Uses v2

Phase A keeps the current speculative loop structure intact. The learned policy
plugs into that loop at the position-selection level.

### Drafter microsteps

On drafter-only microsteps:

- the auxiliary model produces draft logits
- `H_t` and scalar features are constructed
- `u_t` scores masked positions
- the loop applies the existing unmask budget and drafts top-scoring masked
  positions
- drafted positions enter the speculative frontier

`r_t` is inactive on drafter microsteps.

### Verifier microsteps

On verifier microsteps:

- the verifier explicitly validates the current drafting frontier
- frontier acceptance/rejection remains part of the loop
- `r_t` then scores verifier-side rollback candidates
- the loop remasks positions selected by `r_t`
- verifier-side unmasking outside the learned scope may remain on the
  canonical deterministic path

So v2 does not replace the speculative loop. It supplies:

- a learned drafter-side position selector (`u_t`)
- a learned verifier-side rollback selector (`r_t`)


## Warm-Start Recommendation

### Do not warm-start `u_t` from the LM head

Reason:

- the LM head maps hidden state to vocabulary logits
- `u_t` maps hidden state to a position-action score

These are different objects with different geometry and supervision.

### PRISM-based warm start for `r_t` is plausible

This is feasible only as a **teacher-style** or **score-transfer** warm start,
not as literal parameter reuse unless dimensions and representation path are
made compatible.

Preferred interpretation:

- PRISM provides target scores or pseudo-labels for rollback propensity
- `r_t` is initialized to mimic that behavior
- GRPO then refines `r_t` toward speculation-aware rollback decisions

PRISM can be interpreted as a reconstructability-style token-quality signal.
That makes it a natural auxiliary supervision source for verifier-side rollback
warm start, even though the final `r_t` objective remains broader than PRISM's
local correctness estimate.


## Reward Alignment

Phase A policy quality should be judged by whether it improves:

- correctness
- verifier acceptance of useful drafts
- committed-token-per-forward efficiency
- reduced wasted redrafting / rollback

not merely by whether it matches token-level correctness labels.

So:

- PRISM-style supervision is useful for initialization
- GRPO remains necessary for the final objective


## Why This Design Is Justified

This design is justified because the real speculative-decoding control problem
is:

- choose promising masked positions to draft
- roll back bad positions when verifier evidence says to do so
- do this under hard budget and cadence constraints

That is naturally a **position-action** problem.

A position-action policy with scalar per-position scores is therefore the
correct abstraction. Bernoulli logits are an appropriate optimization-friendly
parameterization of that abstraction, as long as we interpret them as
budgeted ranking signals at inference time rather than as fully independent
coin flips.


## Final Recommendation

For Phase A, the unified policy head should be:

- a lightweight shared contextual trunk
- two scalar position heads: `u_t` and `r_t`
- trained first by supervised warm start, then by GRPO
- trained in Bernoulli policy form for GRPO
- executed as budgeted ranking decisions
- using head-specific scoped feature sets
- initialized by imitation of the current good speculative loop
- optionally aided by PRISM-style rollback supervision
- explicitly fed confidence, and agreement only where it is semantically fresh
- kept free of cache/access complexity until `u_t/r_t` clearly beat the
  heuristic baseline

This is the Phase A proposal to review. The 2-step training structure is the
part to treat as most stable; the exact feature set is still open for
discussion and refinement.
