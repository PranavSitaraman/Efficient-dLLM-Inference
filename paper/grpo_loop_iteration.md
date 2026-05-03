# GRPO loop iteration: end-to-end walkthrough

This document explains exactly what one GRPO step does in our codebase, with
explicit comparison to the proposal pseudocode (Algorithm `alg:aoae` in
`paper/proposal.tex`). It nails down where the proposal and the implementation
agree, where they diverge, and what we are committing to for the full run on
the `quality_balanced_hardver` operating point.

---

## 1. Proposal pseudocode (reference)

From `paper/proposal.tex` Algorithm 1 (`alg:aoae`), simplified to the
draft–validate loop. For each verifier event $t = T, T-1, \dots, 1$:

```
for j = 1 .. K_draft:                              # Phase 0 (draft microsteps)
    d_{t,j}  ← p_θ^hard( · | x, ŷ_t ; C_aux )      # cheap drafter forward
    u_{t,j} ~ π_φ^unmask( · | ŷ_t )                # policy decides which to draft
    for k : u_{t,j}^k == 1 and ŷ_t^k == [M]:
        ŷ_t^k ~ d_{t,j}^k                          # write a drafted token
        K_spec ← K_spec ∪ {(k, ŷ_t^k)}             # add to transient frontier

p_t , ρ_t ← p_θ^soft( · | x, ŷ_t ) , h_ψ(ŷ_t)      # Phase 0b (one verifier pass)
α_t^k = 1[ρ_t^k ≥ δ_acc]                          # accept gate
r_t^k = 1 - α_t^k                                 # reject = remask
s_t  ← softmask(p_t, α_t)                          # build policy state
(κ_t, z_t) ~ π_φ( · | s_t )                       # cache + access actions

if mean(1 - α_t) > η:
    C_aux ← ∅                                      # cache realignment

for k : r_t^k == 1 and ŷ_t^k != [M]:               # Phase 1 (remask)
    ŷ_t^k ← [M]
    K_stable ← K_stable \ {k}

# Phase 2 (accept): keep accepted drafted tokens
# Phase 3b: K_stable ← K_stable ∪ {k : κ_t^k == 1, r_t^k == 0}

ŷ_{t-1} ← ŷ_t
```

with reward

$$R = r(\hat{y}_1, y^*) \cdot (1 - f_{\text{eff}})^\alpha
      - \beta \cdot f_{\text{thrash}}
      - w_u \cdot f_{\text{unresolved}}
      + w_{F1} \cdot \overline{F1}_{\text{cache}}
      + w_q \cdot F1_{\text{access}}$$

and standard PPO-clipped surrogate

$$\mathcal{J}(\phi) = \mathbb{E}\Big[\min\big(\rho_t^g A^g,\; \text{clip}(\rho_t^g)\, A^g\big)\Big]$$

---

## 2. What our run actually does

Operating point: `quality_balanced_hardver`. Heads trained: `u_t`, `r_t`.
Cache/access machinery (κ, q, K_stable, F1_cache, F1_access) deliberately
disabled. The proposal's "stage A" framing matches: speculative-only loop
with the unmasking head trained, no stability cache.

### 2.1 One training step

```
# ---------- 1. Prompt + group ----------
x ← sample 1 prompt from OpenMathInstruct-2 train split
prompts ← x.repeat(G)                              # G = group_size = 8

# ---------- 2. Build rollout config ----------
rollout_cfg ← deepcopy(cfg)
rollout_cfg.inference.steps      ← 16              # rollout_steps
rollout_cfg.inference.gen_length ← 512             # rollout_gen_length
apply rollout_overrides:
  base_model.lossless_verification ← true          # hardver target
  inference.primary_agree_threshold ← 0.92         # match eval

# ---------- 3. Collect G rollouts (one per group element) ----------
trajectories, rewards, advantages, components ← collect_rollout_group(...)
# (each rollout runs the speculative loop below)

# ---------- 4. Compute log-probs under current vs old policy ----------
for g in 1..G:
    log_prob_unmask_new[g] = Σ_t log π_φ(u_t^g | s_t^g)   # aux microsteps only
    log_prob_remask_new[g] = Σ_t log π_φ(r_t^g | s_t^g)   # primary microsteps only
    (similarly for π_φ_old, captured at rollout time)

ratio_u[g] = exp(clip( log_prob_unmask_new[g] - log_prob_unmask_old[g], −20, 20))
ratio_r[g] = exp(clip( log_prob_remask_new[g] - log_prob_remask_old[g], −20, 20))

# ---------- 5. Surrogate loss with separated drafter/verifier rewards ----------
A_drafter[g]  = (R_drafter[g]  - mean(R_drafter))  / std(R_drafter)
A_verifier[g] = (R_verifier[g] - mean(R_verifier)) / std(R_verifier)

L_drafter  = − E_g[ min( ratio_u[g] · A_drafter[g],  clip(ratio_u[g], 1−ε, 1+ε) · A_drafter[g]  ) ]
L_verifier = − E_g[ min( ratio_r[g] · A_verifier[g], clip(ratio_r[g], 1−ε, 1+ε) · A_verifier[g] ) ]

L = L_drafter + L_verifier

# ---------- 6. Backprop + step ----------
L.backward()
clip_grad_norm_(policy.parameters(), max_grad_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

### 2.2 Inside one rollout: the speculative loop (`speculative_inference`)

Per microstep `t = T, T-1, ..., 1`:

```
# Decide microstep type: drafter (aux) or verifier (primary)
run_primary ← _should_run_verifier(
    schedule, step_idx, frontier_size, microsteps_since_verify, ...
)
# Triggered when:
#   — frontier ≥ draft_token_budget (12 in rollout cfg), OR
#   — microsteps_since_verify ≥ max_draft_microsteps (4 in rollout cfg), OR
#   — first / last step (force_first_last=true), OR
#   — _ema_agreement < primary_agree_threshold (0.92)

# ============= AUX MICROSTEP =============
if not run_primary:
    aux_logits ← dual_model.auxiliary_forward(y)        # hard top-8 (drafter)
    H_t, confidence ← soft_mask(aux_logits, mask, t/T)
    policy_out ← AOAEPolicy(H_t, confidence, agreement, age, ...)
    actions ← sample_actions(policy_out, mask)
    actions ← _apply_frozen_action_heads(actions, run_primary=False)
    # Under unmask_scope=drafter, remask_scope=verifier:
    #   u_t ← policy.head_unmask    (kept — drafter scope active)
    #   r_t ← 0                     (zeroed — verifier scope; r is OFF on aux)
    #   κ_t ← 0, q_t ← 0            (off entirely)

    actions ← apply_unmask_budget(actions, mask, cfg)   # cap at 12.5%/step
    u_t ← actions["u_t"]
    sampled ← aux_logits.argmax(-1)
    resp_tokens[u_t & mask] ← sampled[u_t & mask]
    draft_frontier.add(drafted_positions, resp_tokens, aux_logits)

# ============= VERIFIER MICROSTEP =============
else:
    aux_logits ← (cached from previous aux pass; reused)
    pri_logits ← dual_model.primary_forward(y)           # hardver: same as aux
    # NB: with lossless_verification=true, primary_forward → auxiliary_forward
    # in dual_model.py:154-160. So drafter == verifier in routing; the diff
    # is only that this pass sees the cumulative drafted state.

    # Validate previously-drafted frontier via argmax_match
    accept_mask, reject_mask ← draft_frontier.validate(pri_logits, drafter_argmax)
    # If pri_argmax != drafter_argmax at a frontier position → reject

    if rejection_action == "remask":
        resp_tokens[reject_mask] ← [M]                   # remask rejected drafts
        if recompute_after_reject:
            pri_logits ← primary_forward(y_after_remask) # rescore on corrected state

    H_t, confidence ← soft_mask(pri_logits, mask, t/T)
    policy_out ← AOAEPolicy(H_t, confidence, agreement, age, ...)
    actions ← sample_actions(policy_out, mask)
    actions ← _apply_frozen_action_heads(actions, run_primary=True)
    # Under unmask_scope=drafter, remask_scope=verifier:
    #   u_t ← (confidence ≥ 0.7) & mask  (heuristic — drafter scope inactive)
    #   r_t ← policy.head_remask         (kept — verifier scope active)
    #   κ_t ← 0, q_t ← 0                 (off entirely)

    # Phase 1: remask via policy r_t (in addition to verifier auto-rejection)
    remask_positions ← r_t & ¬mask
    resp_tokens[remask_positions] ← [M]

    # Phase 2: unmask via threshold u_t at verifier microstep
    unmask_positions ← u_t & mask
    resp_tokens[unmask_positions] ← pri_logits.argmax(-1)[unmask_positions]

    draft_frontier.clear()                               # frontier consumed
```

Loop terminates when all positions are filled (`mask_ind.any() == False`)
or `t == 0`.

### 2.3 Reward computation per rollout

Currently in `aoae/train_grpo.py:compute_reward`. With separated drafter /
verifier rewards (the change we're making for this run):

```
correctness   = 1 if extracted_answer == reference_answer else 0
used_steps    = trajectory.completion_step  (≤ T = 16)
speed_factor  = (1 − used_steps / T)^α    with α = 1.0
unresolved    = mean(final_tokens == [M])
reject_rate   = total_rejects / total_drafts   # NEW — needed for R_drafter

R_drafter  = correctness · speed_factor − w_reject · reject_rate − w_unres · unresolved
R_verifier = correctness · speed_factor                          − w_unres · unresolved
                          # no reject penalty — rejection is the verifier's job

# disabled in this run (set their weights to 0):
#   β · thrash_rate          (cache_quality_weight=0 ⇒ thrash_count = 0 anyway,
#                             since cache=κ_t=0)
#   w_F1 · cache_F1          (cache_quality_weight=0)
#   w_q · access_F1          (access_reward_weight=0)
```

### 2.4 GRPO surrogate loss

For each head k ∈ {unmask, remask}, with G = 8:

$$A_k^g = \frac{R_k^g - \bar R_k}{\sigma(R_k)}, \qquad
  \rho_k^g = \exp\!\Big(\text{clip}\big(\log \pi_\phi(a_k^g | s) - \log \pi_{\phi_{old}}(a_k^g | s),\; -20,\; 20\big)\Big)$$

$$\mathcal{L}_k = -\, \mathbb{E}_g\!\left[\min\!\Big(\rho_k^g A_k^g,\; \text{clip}(\rho_k^g, 1-\epsilon, 1+\epsilon)\, A_k^g\Big)\right]$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{unmask}} + \mathcal{L}_{\text{remask}}$$

with `ε = 0.2`, `normalize_advantage_std = true`. KL term omitted (training
from initialization for the u/r heads).

Critical implementation notes that match the proposal:
1. **Importance ratio is clipped at the log level before exp** to prevent
   NaN gradient explosions at G ≥ 8 (`ρ = exp(clip(log ρ, −20, 20))`).
2. **Per-head log-prob isolation**: `include_heads_in_logprob = ["unmask",
   "remask"]` ensures only the relevant heads' Bernoulli log-probs enter
   each ratio. The cache/access heads' (zeroed) outputs do not contribute.
3. **Length normalization**: surrogate is averaged over `T - completion_step_g`
   policy events per rollout, matching the proposal Eq. `eq:grpo`.

---

## 3. Comparison with the proposal pseudocode

| Element | Proposal | Our run | Comment |
|---|---|---|---|
| Drafter forward | hard-routed $p_\theta^{\text{hard}}$ | `auxiliary_forward` (hard top-8) | match |
| Verifier forward | soft-routed $p_\theta^{\text{soft}}$ | `auxiliary_forward` via `lossless_verification=true` (hardver) | **differs from proposal** — but ablation shows hardver Pareto-dominates |
| Drafter cache reset | `if mean(1-α) > η: C_aux ← ∅` | `aux_cache_reset_threshold = 0.35` (same mechanism) | match |
| Acceptance gate | $α_t^k = 1[ρ_t^k ≥ δ_{\text{acc}}]$ via PRISM | `argmax_match`: $α_t^k = 1[\text{drafter}.\text{argmax}(k) = \text{verifier}.\text{argmax}(k)]$ | **differs** — argmax_match is simpler and PRISM-free, in keeping with proposal Phase A |
| `u_t` head trained | yes (`u_{t,j} ~ π_φ^unmask`) | yes — **drafter scope** (aux microsteps); verifier microsteps use threshold `(conf ≥ 0.7) & mask` | match (drafter side) |
| `r_t` head trained | yes (`r_t = 1 - α_t`) | yes — **verifier scope** (primary microsteps); aux microsteps force r_t = 0 | match |
| `κ_t` head trained | yes (with `w_F1`) | **off** (κ_t ≡ 0; stability cache deferred) | proposal's "Phase A" framing |
| `q_t` head trained | yes (with `w_q`) | **off** | proposal's "Phase A" framing |
| Composed prediction | $\tilde{p} ∝ p \cdot d^{γ α}$ | **off** (`compose_gamma = 0`) | CLAUDE.md notes γ > 0 degraded accuracy in prior runs |
| Reward base | $r \cdot (1 - f_{\text{eff}})^α - β f_{\text{thrash}} - w_u f_{\text{unres}} + w_{F1} \overline{F1}_c + w_q F1_a$ | $r \cdot (1 - f_{\text{eff}})^α - w_u f_{\text{unres}}$ (β·thrash inert under κ=0; F1 terms zeroed) | proposal special-cased to Phase A |
| Reward split | single scalar $R$ | **two rewards** $R_{\text{drafter}}, R_{\text{verifier}}$; surrogate sums per-head losses | **NEW vs proposal** — enables direct gradient on drafter accept rate |
| Group baseline | $A^g = R^g - \bar R$ (advantage) | $A_k^g = (R_k^g - \bar R_k) / σ(R_k)$ per-head | proposal-compatible plus advantage-normalization (Eq. `eq:grpo` paragraph below) |
| PPO clip ε | 0.2 | 0.2 | match |
| Group size G | 8 | 8 | match |
| KL term | omitted | omitted | match |
| `K_draft` | `{1, 2, 4, 8}` (sweep) | `max_draft_microsteps = 4` (rollout cfg) | match (single point in the sweep) |
| Verifier event count $T$ | sweep | `rollout_steps = 16` | match (single point) |

### 3.1 Things that are **stricter** in our run vs the proposal

- **`u_t` only on drafter microsteps** (proposal trains it on every step).
  Rationale: speculative-decoding theory wants the verifier canonical and
  trusted; learning u_t on verifier microsteps would compete with the
  threshold rule and risk regressing accuracy below the threshold floor.
  We get this by setting `unmask_scope: "drafter"`.
- **`r_t` only on verifier microsteps** (proposal allows aux microsteps to
  remask, line `for k : r_t^k = 1`). The proposal's r_t fires whenever
  `r_t = 1 - α_t`, which only meaningful at verifier events. So our
  `remask_scope: "verifier"` is a no-op tightening — implementation just
  matches the proposal's actual semantics.

### 3.2 Things that are **looser / different**

- **Verifier routing**: hardver (top-8) instead of soft (top-16). Justified
  by the 50-sample ablation in `5_3_soft_gating_hurts_verifier-performance.md`:
  hardver Pareto-dominates at quality_balanced. We're running the qmax
  replication (`outputs/qmax_ablations/`) before final commit; if qmax
  contradicts, revert to soft for the full run.
- **Acceptance: argmax_match instead of PRISM threshold $δ_{\text{acc}}$.**
  This is the proposal's Phase A simplification (no learned PRISM-like head).
  Replacing with a learned correction head is listed as a follow-up in
  proposal §`sec:experiments`.
- **Composed prediction off** (`γ = 0`). Empirical decision from prior
  runs. Re-enable as a separate ablation if needed.
- **Reward splits per-head**. The proposal uses a single scalar reward.
  Splitting is novel here; rationale is the user's explicit ask to put a
  direct gradient on drafter accept rate.

### 3.3 Why the loop is rigorous despite the simplifications

- **The accuracy floor is the verifier-driven argmax_match self-correction.**
  Even with all four policy heads at random init, the speculative loop
  reaches ≥ 50% accuracy on GSM8K (CLAUDE.md baseline numbers). The
  trained heads can only *improve* beyond that floor — the loop's safety
  net is independent of the policy.
- **The drafter and verifier scopes are mutually exclusive on each
  microstep type.** No two heads compete for the same decision; aux
  microsteps are the drafter's domain (u_t), verifier microsteps are the
  verifier's domain (r_t + threshold u_t). Clean credit assignment.
- **The reward decomposition has a clear per-head responsibility.** R_drafter
  penalizes reject rate (the drafter's own metric); R_verifier ignores
  rejection (the verifier owns when to reject). Each head's gradient comes
  only from its scope's actions and its scope's reward.
- **All four "frozen" action sources are deterministic at runtime.**
  `_apply_frozen_action_heads` overwrites with deterministic functions
  (threshold rule for u, zeros for r/κ/q in their inactive scopes). No
  silent stochasticity outside the trained heads.

---

## 4. Concrete config diff for this run

```yaml
grpo:
  train_heads:                ["unmask", "remask"]
  include_heads_in_logprob:   ["unmask", "remask"]
  unmask_scope:               "drafter"
  remask_scope:               "verifier"

  # Reward shape — split, with reject-rate penalty on drafter only
  alpha:                       1.0
  reward_split:                true        # NEW — turns on R_drafter / R_verifier
  reject_rate_weight:          0.1         # NEW — w_reject in R_drafter
  unresolved_penalty_weight:   0.25        # in both R_drafter and R_verifier

  # Disabled cache/access machinery
  cache_quality_weight:        0.0
  access_reward_weight:        0.0
  cache_speed_credit_cap:      0.0
  beta:                        0.1         # inert under κ=0 (cached=0 ⇒ thrash=0)

  # Rollout target = quality_balanced_hardver
  rollout_steps:               16
  rollout_gen_length:          512
  rollout_overrides:                       # NEW — applied via build_rollout_cfg patch
    "base_model.lossless_verification":                  true
    "inference.primary_agree_threshold":                 0.92

  warm_start_from:             null
  max_steps:                   500
  warmup_steps:                50
  epochs:                      3
  group_size:                  8
  lr:                          3.0e-4
  policy_temperature:          1.0
  normalize_advantage_std:     true
  clip_eps:                    0.2

logging:
  project:                     "aoae"
  run_name:                    "grpo_uvscope_qbal_hardver"
  output_dir:                  "outputs/grpo_uvscope/"
  use_wandb:                   true
```

## 5. Things deferred to a follow-up sweep

These are **explicitly not** part of this run, to keep the experiment
attributable to a single change-set:

1. **Verifier-corrects-instead-of-remasks** (option (a) in the design
   discussion). Changes `aoae/speculative_inference.py:988` from
   `resp_tokens[reject_mask] = [M]` to
   `resp_tokens[reject_mask] = pri_logits.argmax(-1)[reject_mask]`.
   Avoids the "drafter keeps proposing the same wrong token" attractor.
2. **Composed prediction γ > 0** (Eq. `eq:composed`).
3. **Learned PRISM-like correction head** replacing argmax_match.
4. **κ_t / q_t / stability cache** (proposal's Phase C).
5. **Soft verifier (top-16)** if a future ablation rehabilitates it.

Each of these is a clean ~1 to ~30-line diff on top of this run's
implementation, and any of them could be a follow-up sweep point.
