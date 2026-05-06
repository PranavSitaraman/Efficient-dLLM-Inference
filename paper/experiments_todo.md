# Experiments TODO

Open questions / ablations still to run. Each block lists the hypothesis, the
exact config knobs to flip, and the variants the experiment applies to.

## Quick reference: which variants use what

| Variant | Schedule | Uses AOAE policy | Uses soft-mask H_t |
|---|---|---|---|
| `quality_max` | `aoae` (any-order) | YES | YES |
| `quality_balanced` | `aoae` | YES | YES |
| `aoae_llada_sq_anyorder` | `aoae` | YES | YES |
| `quality_max_block` | `aoae_block` | no | no |
| `quality_balanced_block` | `aoae_block` | no | no |
| `speed_balanced` | `aoae_block` | no | no |
| `speed_max` | `aoae_block` | no | no |
| `speed_extreme` | `aoae_block` | no | no |
| `aoae_llada_sq` | `aoae_block` | no | no |

The block-frontier runner (`run_block_frontier_speculative_inference` in
`aoae/dinfer_integration.py`) starts with `del policy, soft_mask_module, ...`
(line 290), so block-mode variants never consume policy outputs or H_t even
when a checkpoint is loaded.

---

## 1. Soft-routing on/off on the verifier (dT)

**Hypothesis.** The `aoae_llada_sq*` variants currently route the verifier
through hard top-8 (auxiliary) routing — drafter and verifier are the same
forward, just with q-mode thresholds. Does enabling soft (widened top-k)
routing on the verifier improve quality enough to justify the extra MoE
cost?

**Variants to compare:**
- `aoae_llada_sq` (block, dM=s-mode, dT=q-mode) — soft routing OFF vs ON
- `aoae_llada_sq_anyorder` (any-order, dM=s-mode, dT=q-mode) — soft routing OFF vs ON

**Config diff (turn ON):**
- `base_model.lossless_verification`: `false`
- `inference.block_speculative.verifier_routing`: `"primary"`  *(only for the block variant — any-order has no analogous knob and routes through `primary_forward` whenever `lossless_verification=false`)*

**Config diff (turn OFF — current sq state):**
- `base_model.lossless_verification`: `true`
- `inference.block_speculative.verifier_routing`: `"auxiliary"`  *(block only)*

**Notes.**
- Cost ratio matters: `verifier_compute_ratio` defaults to 1.0 when
  `lossless_verification=true`, 2.0 otherwise. Re-tune unmask budget /
  draft schedule if the verifier becomes 2× more expensive.
- Compare on the same Pareto plot as quality_* and speed_* points.

**Results (TODO):**

| Variant | Soft routing | Accuracy | TPS | effective_flops | accept_rate | Notes |
|---|---|---|---|---|---|---|
| `aoae_llada_sq` | OFF (current) | TODO | TODO | TODO | TODO | |
| `aoae_llada_sq` | ON | TODO | TODO | TODO | TODO | |
| `aoae_llada_sq_anyorder` | OFF (current) | TODO | TODO | TODO | TODO | |
| `aoae_llada_sq_anyorder` | ON | TODO | TODO | TODO | TODO | |

---

## 2. Soft-masking H_t on/off (only meaningful for `aoae` schedule)

**Hypothesis.** H_t blends the mask embedding with the top-k predicted token
embeddings (Mahdavi et al. 2025, "Soft-Masked Diffusion Language Models").
We use it as a feature for the policy net — does the blend actually help vs.
just feeding the argmax embedding?

**Variants:** `quality_max`, `quality_balanced`, `aoae_llada_sq_anyorder`.

**Config diff (turn OFF — argmax-only):**
- `soft_mask.top_k`: `1`  *(reduces convex combination to a single token)*

**Config diff (turn fully OFF — feed raw mask embedding):**
- This requires either `soft_mask.top_k=0` (need to verify it's accepted) or
  setting `omega_a_init`, `omega_b_init` to large positive values so the
  gate outputs ≈0 weight on the predicted tokens. Cleanest is a small code
  patch: `SoftMaskedState.forward` returns the raw mask embed when a flag is
  set. **Skip unless top_k=1 already shows a meaningful gap.**

**Adjacent ablation — replace policy with `DefaultPolicy`** (heuristic, no
training):
- Run eval without `--checkpoint`, OR delete `policy_final.pt`. Code path
  at `aoae/evaluate.py:2017–2024` instantiates `DefaultPolicy` with
  `tau_mask=0.7`, no learned u_t/r_t/κ_t/q_t.
- This separates "soft-masking helps" from "trained policy helps."

**Results (TODO):**

| Variant | Policy | top_k | Accuracy | TPS | accept_rate | Notes |
|---|---|---|---|---|---|---|
| `quality_max` | trained | 8 (current) | TODO | TODO | TODO | baseline |
| `quality_max` | trained | 1 | TODO | TODO | TODO | argmax-only H_t |
| `quality_max` | DefaultPolicy | 8 | TODO | TODO | TODO | no training |
| `quality_balanced` | trained | 8 (current) | TODO | TODO | TODO | baseline |
| `quality_balanced` | trained | 1 | TODO | TODO | TODO | argmax-only H_t |
| `quality_balanced` | DefaultPolicy | 8 | TODO | TODO | TODO | no training |
| `aoae_llada_sq_anyorder` | trained | 8 (current) | TODO | TODO | TODO | baseline |
| `aoae_llada_sq_anyorder` | trained | 1 | TODO | TODO | TODO | argmax-only H_t |
| `aoae_llada_sq_anyorder` | DefaultPolicy | 8 | TODO | TODO | TODO | no training |

---

## 3. Routing temperature sweep (soft-MoE only)

**Hypothesis.** `base_model.routing_temperature` (`τ_r`, default 0.01)
controls how sharp the widened top-k routing is. Lower = closer to hard
top-k; higher = more uniform mixing across all 16 selected experts.

**Sweep:** `τ_r ∈ {0.005, 0.01, 0.05, 0.1, 0.5}`

**Variants to test on:** any soft-routed variant — easiest is one of the
`quality_*` points (since they keep `lossless_verification=false`). Also
the proposed soft-routing-ON sq variants from Experiment 1.

**How to set:** `--set base_model.routing_temperature=<value>` on the eval
command, or add a per-point override in the sweep list.

**What to look for.** A sweet spot in the speed/quality trade where soft
routing buys quality without spending too much on diluted experts. If the
curve is flat across τ_r, soft routing is doing nothing useful and we should
just use hard routing everywhere.

**Results (TODO):**

| Variant | τ_r | Accuracy | TPS | effective_flops | Notes |
|---|---|---|---|---|---|
| `quality_balanced` | 0.005 | TODO | TODO | TODO | |
| `quality_balanced` | 0.01 (current) | TODO | TODO | TODO | baseline |
| `quality_balanced` | 0.05 | TODO | TODO | TODO | |
| `quality_balanced` | 0.1 | TODO | TODO | TODO | |
| `quality_balanced` | 0.5 | TODO | TODO | TODO | |
| `aoae_llada_sq` (soft ON) | 0.005 | TODO | TODO | TODO | only if Exp 1 wins |
| `aoae_llada_sq` (soft ON) | 0.01 | TODO | TODO | TODO | only if Exp 1 wins |
| `aoae_llada_sq` (soft ON) | 0.05 | TODO | TODO | TODO | only if Exp 1 wins |
| `aoae_llada_sq` (soft ON) | 0.1 | TODO | TODO | TODO | only if Exp 1 wins |
| `aoae_llada_sq` (soft ON) | 0.5 | TODO | TODO | TODO | only if Exp 1 wins |

---

## Pinned baselines (from earlier discussion)

### dKV-Cache / Elastic-Cache baselines

dInfer ships block-KV cache decoders (`generate_cache.py`, `generate_hierarchy.py`,
`generate_fastdllm.py`). dKV-Cache is essentially the dInfer `fix_iter`
schedule. Elastic-Cache requires a small port of the layer-adaptive refresh
heuristic.

**To run:**
- Wire dInfer's `generate_cache.cache_update_tag(strategy="fix_iter", iter=N)`
  as a baseline alongside `llada21_official_decode`. Sweep `N ∈ {2, 4, 8}`.
- Elastic: defer until dKV results are in.

These are the natural KV-cache baselines for the drafter-cache (K_stable)
mechanism — same idea, different staleness signal (clock vs. self-similarity
vs. our verifier-driven gate).

**Results (TODO):**

| Method | Refresh policy | Accuracy | TPS | Cache-keep ratio | Notes |
|---|---|---|---|---|---|
| dKV-Cache (`fix_iter`) | every 2 steps | TODO | TODO | TODO | |
| dKV-Cache (`fix_iter`) | every 4 steps | TODO | TODO | TODO | |
| dKV-Cache (`fix_iter`) | every 8 steps | TODO | TODO | TODO | |
| Elastic-Cache | layer-adaptive | TODO | TODO | TODO | requires port |
| AOAE K_stable (ours) | verifier-gated | TODO | TODO | TODO | from `quality_balanced` |

---

## Reminders

- **Any-order eval needs the filter explicitly:**
  `--set evaluation.generation_mode_filter=any_order` (block runs separately
  with `=block`).
- **LLaDA 2.1 attention is locked to block-causal-32:** "any-order" means
  the unmask schedule is any-order, not the attention pattern. Don't widen
  the bidirectional window.

---

## Suggested run order

1. **Experiment 2 (cheapest, biggest interpretive payoff):** `top_k=1` on
   `quality_balanced` and `aoae_llada_sq_anyorder`. ~30 min compute each.
2. **Experiment 1 (block sq + soft routing):** the high-value comparison
   for the paper's sq story. ~1 h.
3. **Experiment 3 (routing temperature):** only worth running if
   Experiment 1 shows soft routing helps in sq mode. ~2 h for full sweep.
4. **dKV baseline:** independent of 1–3, can run in parallel.
