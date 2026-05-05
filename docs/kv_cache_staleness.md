# Speculative Frontier vs. Positional Stability

**Date:** 2026-04-12
**Status:** Historical design note — several concerns below have since been addressed in code; keep this as background context, not current implementation status.

---

## The Core Distinction

There are two separate per-position properties that the current system conflates:

| Property | Defined at | Meaning |
|---|---|---|
| **Speculative frontier / acceptance** (`K_spec`, `agreement`) | Step $t$ only | Short-lived drafter-verifier state used to carry speculative candidates / one-step accepts into the next verifier pass |
| **KV stability** (`κ_t`) | Multi-step | This position's key/value vectors will remain accurate across future steps as more tokens are unmasked |

These are orthogonal. A position entering `K_spec` at step $t$ is only part of a
transient speculative frontier: it says something about the drafter/verifier
interaction for the immediate next verification event, not about long-term KV
stability. As subsequent steps reveal more tokens and the global attention
context shifts, that position's KV vectors may drift significantly — the
speculation result says nothing about steps $t+1, t+2, \ldots$

**<User feedback>**: [User](): this is the key realization. This suggests our KV Cache manager needs to keep a cache of stable KV tokens vs. speculative KV (temporary) cache of tokens. For the stable KV-cache to work, we may need a *eviction policy as well*, or we could consider remasking as a kind of eviction policy, perhaps. On the other hand, the temporary specualtive acceptance would quickly be resolved by the verifier accepting or rejecting the token, with accepting meaning to reuse drafter KV-vectors in the next step, skipping computation that step but the tokens are not cached for a extended periods of time (contrasting to the stable KV cache). 

But the most important thing is that we should first test out the gains we get from **speculative acceptance** (agreement) and **positional stability** (kappa) separately. This shall be configurable, and our immediate next goal is to test out the speculative acceptance part first and then the positional stability part. To test out whether speculative acceptance path actually produce any gains, we will need to wire **actual KV-vector computation skipping** when the verifer accepts the drafter's tokens at some step. 

This means that (1) Speculative tokens committed to long-term cache must be fixed: KVCacheManager has a sperata way to deal with sepculative acceptance at step t that is separate from the stability cache. (We may still need some data strcture to store these draft candidates because the drafter may move faster than the verifier (e.g. a deque)) and (3) to actually test whether specualtiv acceptance is meaningful, we must fix the " KV skip not actually implemented in the forward pass" problem. We will also test the stability cache part immediately after.** The overall ultimate system aims to combine both gains: just as normal AR models also use both speculative decoding and KV-caching.

 (2) Thrash metric under-counts staleness can be fixed later.
**</User feedback>** 

---

## How the Current Code Handles This

The cache commit in `speculative_inference.py` (Phase 3) is:

```python
agreement_cache = kappa_t * agreement.float() * q_exec
cache_mgr.commit(agreement_cache)
```

The `agreement.float()` factor is a **quality gate** — don't commit positions we're uncertain about — which is reasonable but is not a stability guarantee. The right mechanism for training $\kappa_t$ to predict stability is `cache_quality_f1` in the reward: it measures whether committed positions had low $H_t$ drift (soft-masked state didn't shift between steps). That is the correct multi-step signal.

---

## Three Layered Concerns

### 1. Speculative tokens committed to long-term cache

**Severity:** Resolved in code (2026-04-12). The two caches are now split.

**Resolution:** `aoae/cache.py` now has `SpeculativeKVCache` (legacy class name
for `K_spec`) and `StableKVCache` (K_stable):
- `K_spec` is a transient speculative frontier / one-step speculative-accept state and is **replaced each step**. It no longer accumulates.
- `K_stable` is populated by `κ_t=1` only (no agreement gating). It accumulates and is evicted by `r_t=1`.
- `speculative_inference.py` Phase 3 is split into 3a (K_spec update) and 3b (K_stable update).
- The reward now tracks `spec_cached_fractions` and `cached_fractions` (K_spec ∪ K_stable) separately. `spec_cached_fractions` is a legacy metric name for transient K_spec occupancy.

**Remaining gap:** `cache_quality_f1` still uses $H_t$ drift as a proxy for KV drift (concern §3). Whether $H_t$ drift faithfully tracks actual KV-layer drift is unverified.

---

### 2. Thrash metric under-counts staleness

**Severity:** Moderate. The thrash reward term underestimates the real cost of stale cache entries.

**Mechanism:** `DKVCacheManager.count_thrash(r_t)` counts positions that are cached AND explicitly remasked in the same step:

```python
thrash = self.cached.float() * edit_mask.float()
```

A position can become *contextually stale* (its KV should be recomputed because the surrounding context changed) without ever being explicitly remasked. This form of staleness goes unpenalized by the thrash term.

**Potential fix:** At each step, compute $\|H_t^k - H_{t-1}^k\|$ for all cached positions. Positions exceeding a drift threshold should incur a staleness penalty, even without an explicit remask action. This is essentially what `cache_quality_f1` does from the recall side, but a direct penalty would give a stronger and more interpretable signal.

---

### 3. KV skip not actually implemented in the forward pass

**Severity:** Makes all "speedup" a simulation for now.

**Mechanism:** `DKVCacheManager` is a logical bookkeeper — it tracks *which* positions would be cached but does not actually skip KV recomputation in the model forward pass. The `cached_fraction` metric feeds into `effective_flops` in the reward as a proxy for real compute savings:

```python
effective_flops = (used_steps / T) * (1 - mean_cached_fraction)
speed_factor    = (1 - effective_flops) ** alpha
```

This proxy trains the policy toward high cached fractions, which is the right direction, but the actual latency benefit is zero until KV skipping is wired into the underlying model (dInfer/vLLM/SGLang).

**Implication:** The staleness concern (§1–2) has no *runtime* consequence until §3 is resolved. But training the policy now with a correct stability signal ensures that when real KV skipping is implemented, the policy already knows which positions are safe to cache.

---

## Summary Table

| Concern | Severity | Currently addressed by | Gap |
|---|---|---|---|
| Speculative tokens committed to long-term cache | Real (reward proxy only) | `cache_quality_f1` trains κ_t vs. $H_t$ drift | H_t drift ≠ KV drift (unverified proxy) |
| Thrash under-counts context-drift staleness | Moderate | `cache_quality_f1` (recall side) | No direct staleness penalty for cached positions |
| KV skip not implemented in model forward | High (all speedup is simulated) | Out of scope — needs SGLang/vLLM integration | Entire §3.6 real-compute path missing |

---

## Potential Fixes (To Be Prioritized)

1. **Drift-based staleness penalty:** At each step, penalize cached positions whose $H_t$ drift exceeds a threshold $\delta$, regardless of whether they were explicitly remasked. Strengthens the signal $\kappa_t$ learns from and makes thrash more complete.

2. **Decouple `agreement` from cache commit:** The `agreement.float()` factor in `commit()` conflates token-correctness with KV-stability. Consider removing it from the commit condition and relying solely on $\kappa_t$ for stability decisions. The quality signal for token correctness is already embedded in the correctness reward term.

3. **Validate $H_t$ drift as a KV proxy:** Before wiring real KV skipping, run a diagnostic comparing per-position $H_t$ drift (current proxy) against actual per-position KV-vector drift (from stored hidden states or `kv_dynamics_summary`). If the proxy is poor, a better intermediate feature should replace it.

4. **Real KV skip integration:** Long-term — integrate with SGLang or vLLM's paged attention to actually skip recomputation for cached positions. Until this is done, all efficiency gains are reward-level incentives only.
