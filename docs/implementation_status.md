# AOAE: Proposal → Implementation Status

**Date:** 2026-04-12
**Source doc:** `docs/aoae_system_overview.tex`
**Key files:** `aoae/models/dual_model.py`, `aoae/models/policy.py`, `aoae/speculative_inference.py`, `aoae/train_grpo.py`

---

## Plan Update (2026-04-13)

The current implementation plan has pivoted toward a **lossy speculative path** with `K_draft > 1`:

- The **drafter** is the hard-gated model running with a plug-and-play cache/freezing policy, so it is intentionally faster and lower quality.
- The **verifier** is now a real plug-and-play module: it can be a frozen PRISM sidecar, an unfrozen PRISM sidecar, a fresh learned verifier head, or a logits-only heuristic verifier.
- `K_spec` should be treated as a **transient frontier of drafted-but-not-yet-verified candidates**, not as a persistent cache and not as a store of already accepted tokens.
- The learned **stability cache** (`K_stable`, `κ_t`) is now a **separate later contribution** and may be disabled while we first get the speculative path working.
- The speculation path and the persistent stability cache are now independently configurable: `cache.kspec_skip` toggles the drafter/verifier speculative path, and `cache.stable_kv_cache` toggles the primary-owned stability cache. When both are enabled, the aux drafter / agreement path stays active while `K_stable` delivers the actual primary-side KV skipping.

### Verifier backend matrix (implemented)

| Verifier config | Meaning | Trainable in GRPO | Artifact |
|---|---|---|---|
| `verifier.kind: prism`, `trainable: false` | Frozen PRISM verifier (legacy default) | No | `prism_adapter.pt` |
| `verifier.kind: prism`, `trainable: true` | Unfrozen PRISM verifier | Yes | `prism_adapter.pt` |
| `verifier.kind: learned_head` | Fresh PRISM-like learned verifier head | Yes | `verifier_head.pt` |
| `verifier.kind: confidence` | Logits-only heuristic verifier (`max_prob`, `margin`, `one_minus_entropy`) | Stateless | none |

### Runtime composition matrix (implemented)

| `cache.kspec_skip` | `cache.stable_kv_cache` | Runtime behavior |
|---|---|---|
| `false` | `false` | Full verifier-primary forward each step; no KV skipping |
| `true` | `false` | k>1 lossy speculative draft--validate schedule; wall-time gain comes from fewer primary verifier passes |
| `false` | `true` | Stable primary-owned KV skipping (`K_stable`) |
| `true` | `true` | Drafter/verifier speculation remains active, and stable-primary KV skipping remains active; current runtime combines fewer primary passes with stable-primary reuse on verifier steps |

---

## §3.1 Dual-Model MoE Architecture

**Status: Realized** — `aoae/models/dual_model.py:DualModelWrapper`

The soft routing gate from the proposal:

$$w_i^{\text{soft}} = \frac{\exp(z_i / \tau_r)}{\sum_{j=1}^E \exp(z_j / \tau_r)}$$

is implemented in `aoae/models/soft_moe.py:SoftMoERouter.forward()` (line 241–256).
`auxiliary_forward()` uses hard top-$k$; `primary_forward()` uses soft routing at temperature `tau_r`.

**Correction vs. proposal:** The proposal states "2× model memory" — this is wrong. Only **one** model copy is loaded; routing is toggled in-place via `set_hard_routing` / `set_soft_routing`. Memory overhead is negligible (router parameter masks only).

---

## §3.2 Soft-Masked State

**Status: Realized** — `aoae/models/soft_mask.py:SoftMaskedState`

The soft-masked embedding:

$$\mathbf{h}_t^k = \lambda(p_t^k)\,\mathbf{E}_{[\mathrm{M}]} + \bigl(1 - \lambda(p_t^k)\bigr) \sum_{j \in \mathrm{top\text{-}}K(p_t^k)} \pi_j^k\,\mathbf{E}_{v_j}$$

with gating function:

$$\lambda(p_t^k) = \omega_s \cdot \sigma\!\bigl(\omega_a(-H(p_t^k) - \omega_b)\bigr)$$

**is implemented exactly. The three learnable scalars $(\omega_s, \omega_a, \omega_b)$ are `nn.Parameter` in `SoftMaskedState.__init__` (lines 23–25).**

**Grad-flow fix (2026-04-12):** Previously these received zero gradient because $\mathbf{H}_t$ was `.detach()`-ed before trajectory storage. Fixed by:
1. `SoftMaskedState.forward()` now returns a 4-tuple `(h_t, confidence, entropy, weighted_embeds)`.
2. `AOAETrajectory` / `SpeculativeTrajectory` store `weighted_embeds` [B, L, D] and `entropy` [B, L] (detached) instead of only `H_t`.
3. `SoftMaskedState.recompute_h_t(weighted_embeds, entropy)` recomputes `h_t = λ · E_[M] + (1−λ) · weighted_embeds` using the current live ω parameters, allowing autograd to flow through ω without re-running the base model.
4. `compute_grpo_loss` calls `recompute_h_t()` instead of using the stored `H_t`.

Memory overhead: ~200 MB per rollout group (vs. ~16 GB if storing full vocab logits).

---

## §3.3 Action Space

**Status: Realized with extension** — `aoae/models/policy.py:AOAEPolicy`

The proposal defines three per-position Bernoulli actions:

$$u_t^k \sim \mathrm{Ber}\!\left(\sigma(f_{\phi^{\mathrm{unmask}}}(s_t)^k)\right), \quad r_t^k \sim \mathrm{Ber}\!\left(\sigma(f_{\phi^{\mathrm{remask}}}(s_t)^k)\right), \quad \kappa_t^k \sim \mathrm{Ber}\!\left(\sigma(f_{\phi^{\mathrm{cache}}}(s_t)^k)\right)$$

The code implements **four** heads: `head_unmask`, `head_remask`, `head_cache`, **`head_access`**. The fourth head ($q_t^k$) was added for positional speculative caching (§3.6 of the proposal) and integrated directly into the unified action space rather than kept as a separate module.

Validity constraints via logit masking (policy.py lines 237–240):
- Unmask masked to $-\infty$ for $m_t^k = 0$
- Remask masked to $-\infty$ for $m_t^k = 1$
- Cache–remask exclusion enforced at sampling time: $\kappa_t^k \leftarrow \kappa_t^k \cdot (1 - r_t^k)$

---

## §3.4 Speculative Inference Loop

**Status: Realized** — `aoae/speculative_inference.py`

Algorithm 1 maps line-by-line:

| Algorithm step | Code location |
|---|---|
| Phase 0 (Draft, hard routing) | `dual_model.auxiliary_forward` via `dual_forward_resp` |
| Phase 0b (Verify, soft routing) | `dual_model.primary_forward` via `dual_forward_resp` |
| Agreement $\alpha_t^k$ | `dual_model.py:dual_forward` lines 267–269 |
| Soft-masked state $s_t$ | `soft_mask_module(resp_logits, mask_ind, step_frac)` |
| Policy sample | `call_policy` → `pol_inner.sample_actions` |
| Phase 1: Remask | lines 321–325 |
| Phase 2: Unmask + compose | lines 327–344 |
| Phase 3a: K_spec frontier update | `cache_mgr.step_spec(agreement)` |
| Phase 3b: K_stable update | `cache_mgr.step_stable(kappa_t, r_t)` |

**Current realized behavior (2026-04-18):** `primary_every_n` is read by the canonical `speculative_inference.py` path. When `cache.kspec_skip=true`, the auxiliary drafts every step while the primary/verifier runs every `primary_every_n` steps (plus bootstrap/final steps). This is intentionally a lossy approximation to the original per-step verifier algorithm. When PRISM or KV-dynamics tracking needs primary hidden states, the primary falls back to a full verifier forward, but the auxiliary can still reuse its prefix cache on draft steps and before verifier events.

Runtime note: setting `analysis.log_speculative_config: true` prints one rollout-level summary of the speculative mode selection, including whether PRISM is active, whether KV tracking is active, whether the auxiliary prefix cache is enabled, whether a primary cache fast path is available, and the key verifier cadence / remask / composition knobs.

Implementation note:

- On auxiliary-only steps, drafted tokens may be written directly into `y`, but `K_spec` itself remains empty because no verifier observation happened yet.
- On verifier steps, the current one-step `K_spec` reuse mask is consumed by the primary skip path when hidden states are not needed; after validation, `K_spec` is cleared/replaced with the freshly verified reusable positions from that verifier event.
- Accepted positions are **not** kept in `K_spec`; they simply remain in the sequence state `y`.
---

## §3.5 Composed Prediction

**Status: Realized** — `aoae/models/composed_prediction.py:compose_prediction_dual`

The composed distribution:

$$\tilde{p}_t^k(v) \propto p_t^k(v) \cdot d_t^k(v)^{\gamma \cdot \alpha_t^k}$$

is implemented in log-space as:

$$\log \tilde{p}_t^k(v) = \log p_t^k(v) + \gamma \cdot \alpha_t^k \cdot \log d_t^k(v)$$

via `composed = primary_logits + gamma * alpha_k * F.log_softmax(auxiliary_logits)`.

**Bugs fixed (IEEE 754 `0 × −∞ = NaN`):** In bf16, `log_softmax` / `log_probs` produce $-\infty$ for near-zero-probability tokens. Three locations had unguarded products:
1. `composed_prediction.py:122` — `alpha_k * aux_log_probs` at disagreement positions ($\alpha_t^k = 0$)
2. `soft_mask.py:49` — `probs * log_probs` in per-position entropy (primary NaN source: corrupts `h_t` → policy input → `unmask_probs`)
3. `composed_prediction.py:148` — `probs * log_probs` in composed-entropy utility

All three fixed via `torch.nan_to_num(..., nan=0.0)` guard.

---

## §3.6 Positional Speculative Caching

**Status: Partially realized** — `aoae/positional_cache.py`, `aoae/cache.py`

### Two-cache split (2026-04-12) ✅ Realized

The proposal's Algorithm 1 now distinguishes a transient speculative frontier
from the persistent stable cache:

| Pool | Class | Validity | Meaning | Eviction |
|---|---|---|---|---|
| `K_spec` | `SpeculativeKVCache` (legacy name) | One verifier call / next-step reuse hint | Transient speculative frontier / reusable verifier state | Cleared / replaced on verifier pass |
| `K_stable` | `StableKVCache` | Multi-step | Persistent KV cache predicted by `κ_t` | `r_t=1` (remask) |

**`aoae/cache.py`** now exports:
- `SpeculativeKVCache`: legacy class name for the transient `K_spec` frontier; in the current k=1 runtime it stores a one-step verified/reusable mask and is replaced each step
- `StableKVCache`: accumulates κ_t positions; evicted by remask via `commit(kappa_t, r_t)`
- `SpeculativeCacheBookkeeper`: combined interface used by `speculative_inference.py`
- `DKVCacheManager`: legacy single-cache kept for `inference.py` and `dinfer_integration.py`

**`aoae/speculative_inference.py`** Phase 3 is now split:
- Phase 3a: `cache_mgr.step_spec(...)` → updates the transient speculative frontier / one-step reuse mask
- Phase 3b: `cache_mgr.step_stable(kappa_t, r_t)` → updates K_stable
- `SpeculativeTrajectory` tracks `spec_cached_fractions` (legacy metric name for K_spec frontier occupancy) and `cached_fractions` (K_spec ∪ K_stable)

**`aoae/train_grpo.py`**: `compute_reward` exposes `mean_spec_cached` in reward components for logging. This is a legacy metric name for transient K_spec occupancy, not a persistent cache fraction.

**Important semantic note:** `K_spec` is not a persistent cache. In the current
`k=1` runtime it is a transient one-step verifier frontier / reuse mask; already
accepted tokens are represented directly in `y`, not stored in `K_spec`.

### Verifier realization (2026-04-13) ✅ Realized

The proposal language "PRISM or a PRISM-like head" is now implemented explicitly in
`aoae/models/verifier.py`:

- `PRISMVerifier`: wraps the legacy `PRISMAdapter`
- `LearnedVerificationHead`: a fresh MLP verifier head on top of primary hidden states
- `ConfidenceVerifier`: a logits-only heuristic verifier for ablations that should not require hidden-state extraction

In both `aoae/inference.py` and `aoae/speculative_inference.py`, the verifier score enters the policy as the `quality_scores` feature. This means the verifier does **not** directly perform the remask action in the current implementation; instead it provides the quality signal that conditions the policy's remask/cache/unmask decisions. This is the main implementation note to keep in mind when comparing the realized method against the proposal wording.

For trainable verifier backends, GRPO now stores the verifier inputs in the rollout trajectory and recomputes verifier scores inside `compute_grpo_loss`, so unfrozen PRISM / learned verifier ablations are actually optimized rather than merely included in the optimizer by name.

### Access head (Top-B refresh)

The refresh set:

$$\mathcal{Q}_t = \mathcal{C}_t^{(H)} \cup \{k : u_t^k = 1 \lor r_t^k = 1\}$$

where $\mathcal{C}_t^{(H)} = \mathrm{Top\text{-}}B(\{g_\phi^{(H)}(s_t)^k\}_{k=1}^L)$, is implemented in `build_access_set()`. The access head scores (`access_probs` from `head_access`) drive top-$B$ candidate selection under `candidate_policy: "learned_topb"`. Age features $a_t^k$ and last-action features are fed to the policy when `use_age_feature: true`.

**Not yet realized:** The **boundary layer head**:

$$\ell_t^\star \sim \mathrm{Cat}\!\left(\mathrm{softmax}(h_\phi(s_t))\right)$$

is present in `policy.py` as `head_boundary` / `boundary_head`, but **disabled** (`boundary_head.enabled: false` in all configs). The warm-start from attention-triggered boundaries (Elastic-Cache style) is unimplemented.

---

## §3.7 GRPO Training — Reward Redesign

**Status: Realized with significant departure from proposal**

### Proposal (Eq. reward):

$$R(\mathbf{y}^*, \mathbf{y}_{\hat{T}}, \hat{T}) = r(\mathbf{y}^*, \mathbf{y}_{\hat{T}}) \cdot \left(\frac{\hat{T}}{T}\right)^\alpha - \beta \sum_{t=1}^{T} \mathrm{Thrash}(t)$$

The speed bonus only rewards finishing in fewer **steps**.

### Code (`aoae/train_grpo.py:compute_reward`):

$$\mathrm{effective\_flops} = \frac{\mathrm{used\_steps}}{T} \cdot \left(1 - \bar{\kappa}\right)$$

$$\mathrm{speed\_factor} = \left(1 - \mathrm{effective\_flops}\right)^\alpha$$

$$R = r(\mathbf{y}^*, \hat{\mathbf{y}}) \cdot \mathrm{speed\_factor} - \beta \sum_t \mathrm{Thrash}(t) - w_u \cdot f_{\mathrm{unresolved}} + w_{\mathrm{F1}} \cdot \overline{\mathrm{cacheF1}} + w_q \cdot \mathrm{accessF1}$$

where $\bar{\kappa} = \frac{1}{T}\sum_t \frac{|\mathcal{K}_t|}{L}$ is the mean cached fraction per step.

**Key change:** The speed term is now **compute-aware** — a policy that caches 80% of positions per step but uses all $T$ steps still gets substantial credit for the 80% FLOP reduction. The proposal's formulation would give that policy zero speed bonus. This better reflects actual wall-time savings. Added terms:

- **Cache quality F1** (precision-recall over stable positions): rewards caching positions that are actually stable (low $\mathbf{H}_t$ drift), penalizes caching unstable ones
- **Access prediction F1**: dense signal for the $q_t$ head
- **Unresolved mask penalty**: penalizes sequences with $[\mathrm{M}]$ tokens still present at rollout end

### GRPO objective

Implemented as in the proposal (Eq. grpo), with:
- Clipped surrogate, $\epsilon = 0.2$
- Importance ratio **clamped**: $\rho = \exp(\mathrm{clip}(\log\rho, -20, 20))$ — the unclamped version caused NaN gradient explosion with `group_size=8`
- `normalize_advantage_std: true` (added for `group_size ≥ 8` stability)
- No KL regularization (as stated)

---

## §3.8 Implicit Behavioral Distillation

**Status: Not implemented / unverified**

Described as an emergent property of the GRPO reward structure. No dedicated experiment or metric tracks this. Requires a completed training run to evaluate.

---

## Summary Table

| Proposal section | Status | Code location |
|---|---|---|
| §3.1 Dual-model MoE | ✅ Realized | `models/dual_model.py`, `models/soft_moe.py` |
| §3.2 Soft-masked state | ✅ Realized | `models/soft_mask.py` |
| §3.3 Action space (3 heads) | ✅ + extended to 4 | `models/policy.py` |
| §3.4 Speculative inference loop | ✅ Realized as lossy k>1 draft--validate schedule | `speculative_inference.py` |
| §3.5 Composed prediction | ✅ Realized (NaN bug fixed) | `models/composed_prediction.py` |
| §3.6 Two-cache split (K_spec + K_stable) | ✅ Realized (2026-04-12) | `cache.py` → `SpeculativeCacheBookkeeper` |
| §3.6 Positional caching (top-B) | ✅ Realized | `positional_cache.py` |
| §3.6 Boundary layer head | ❌ Disabled | `models/policy.py` (head exists) |
| §3.6 Prefix KV cache (prompt prefill + response replace) | ✅ Realized (2026-04-12) | `speculative_inference.py` + `dual_model.py` |
| §3.6 Actual KV-vector skip in forward | ✅ Partial | `K_stable` primary skip is real; `K_spec` primary skip is real for logits-only verifier steps; when PRISM / KV tracking needs primary hidden states the auxiliary cache still runs but the primary falls back to a full verifier pass |
| §3.7 GRPO training | ✅ Realized, reward redesigned | `train_grpo.py` |
| §3.8 Behavioral distillation | ❌ Unverified | — |
| §4 τ_r sweep | ✅ Realized | `scripts/run_routing_sweep.py` |
| §4 Full ablation suite | ❌ Not automated | — |
| §4 Attention mechanism ablations | ❌ Not implemented | — |
| SGLang integration | ❌ Not visible | — |
| HumanEval integration | ⚠️ Partial | `aoae/code_eval.py` (not in eval loop) |



--                                                    
  Session Summary
                                                                                                    
  What was built this session:           
                                                                                                    
  1. K_spec wall-clock analysis (updated 2026-04-13)
  The old k=1 accepted-mask interpretation was not a compelling throughput path because the
  auxiliary tax was paid every step. The canonical speculative path now instead realizes a lossy
  k>1 draft--validate schedule: throughput gain comes from fewer primary verifier passes, while
  `K_spec` tracks which drafted positions are still awaiting validation.

  2. Stable KV Cache (cache.stable_kv_cache: true, cache.kspec_skip: false)
  Primary-only approach. Maintains stable_primary_kv (dInfer KV cache object) across steps. Each
  step: cluster-wise forward_replace_with_cache at active positions only. Active = positions not in
  K_stable, or currently [MASK]. Stable positions (K_stable & ~[MASK]) reuse cached KV exactly.
  K_stable is policy-driven (κ_t commits, remask evicts) — savings track K_stable size, not total
  unmasked count.

  3. Age-decaying thrash penalty (grpo.thrash_age_decay: 0.1)
  StableKVCache.age tracks per-position steps-in-cache. count_thrash weights by exp(-0.1·age).
  Freshly committed tokens incur full penalty on remask; long-lived stable tokens incur near-zero
  penalty — discourages premature commits, not justified corrections.

  4. Logits cache fix (_stable_logits_cache)
  Stable positions have zero logits in resp_logits (computation was skipped). Before
  soft_mask_module, cached primary logits are substituted at stable positions. This ensures H_t,
  policy κ_t/r_t inputs, ω gradient (weighted_embeds), and cache_quality_f1 all see correct values.
  ~78MB overhead.
