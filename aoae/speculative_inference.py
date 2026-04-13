"""
Speculative Diffusion Inference (Algorithm 1 from paper §3.3).

Dual-model loop:
  Phase 0:  Auxiliary draft (hard-routed, fast ~1.4B active)
  Phase 0b: Primary verification (soft-routed, slow all 16B active)
  Phase 1:  Remask uncertain positions
  Phase 2:  Unmask with composed prediction (auxiliary + primary)
  Phase 3:  Cache commit (only at agreement positions)

The key insight: where auxiliary and primary argmax tokens agree,
the auxiliary's pre-cached KV states are valid → speedup.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from .cache import SpeculativeCacheBookkeeper
from .models.composed_prediction import compose_prediction_dual
from .models.dual_model import DualModelWrapper, DualModelOutput
from .models.policy import call_policy
from .agreement_signals import compute_reuse_signal
from .positional_cache import (
    init_positional_state,
    get_policy_positional_features,
    build_access_set,
    update_positional_state,
    compute_next_h_access_metrics,
)
import json


@dataclass
class SpeculativeTrajectory:
    """Stores a full speculative inference trajectory for GRPO training."""
    actions: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    policy_outputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    thrash_counts: List[torch.Tensor] = field(default_factory=list)
    H_t_list: List[torch.Tensor] = field(default_factory=list)
    # Soft-mask intermediates for differentiable ω re-computation in GRPO loss.
    weighted_embeds_list: List[torch.Tensor] = field(default_factory=list)
    entropy_list: List[torch.Tensor] = field(default_factory=list)
    mask_ind_list: List[torch.BoolTensor] = field(default_factory=list)
    quality_scores_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    agreement_list: List[torch.Tensor] = field(default_factory=list)
    age_feature_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    last_action_feature_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    access_exec_list: List[torch.Tensor] = field(default_factory=list)
    access_mandatory_list: List[torch.Tensor] = field(default_factory=list)
    changed_list: List[torch.Tensor] = field(default_factory=list)
    boundary_actions: List[torch.Tensor] = field(default_factory=list)
    step_fracs: List[float] = field(default_factory=list)
    final_tokens: Optional[torch.Tensor] = None
    completion_step: Optional[torch.Tensor] = None
    # Aggregate stats
    mean_agreement_rate: float = 0.0
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    access_metrics: Dict[str, float] = field(default_factory=dict)
    mean_boundary_depth: float = 0.0
    boundary_distribution: str = "{}"
    # ---- compute-aware speed bonus ----
    # K_spec fraction: per-step fraction of positions with drafter-verifier agreement
    # (speculative acceptance, one-step validity).  Updated each step by agreement mask.
    spec_cached_fractions: List[torch.Tensor] = field(default_factory=list)
    # K_stable fraction: per-step fraction of positions in the persistent κ_t cache.
    # Combined, effective_flops = (used_steps/T)*(1 - mean(K_spec ∪ K_stable)).
    cached_fractions: List[torch.Tensor] = field(default_factory=list)
    # ---- Cache quality F1 (soft precision-recall training signal) ----
    # Per-step F1 measuring whether the cache set contains the *stable*
    # positions (low H_t drift) and excludes the *unstable* ones.
    #
    # For every position k, we compute:
    #   stability(k) = exp(-λ * rel_drift_k)   ∈ (0, 1]
    # where rel_drift_k = ||H_t^k - H_{t-1}^k||₂ / ||H_{t-1}^k||₂.
    #
    # Then soft precision/recall over the cache set K_t:
    #   precision = mean_{k ∈ K_t}(stability(k))
    #   recall    = Σ_{k ∈ K_t} stability(k) / Σ_all_k stability(k)
    #   cache_F1  = 2 * precision * recall / (precision + recall)
    #
    # Subsumes the old drift_penalty (precision-only) and adds the "commit
    # stable tokens" recall gradient.  Computed using primary model H_t.
    cache_quality_f1: List[torch.Tensor] = field(default_factory=list)
    # ---- KV dynamics tracker summary (eval only, None during training) ----
    # Populated by SpeculativeDynamicsTracker when track_kv_dynamics=True.
    # Mirrors the same field in AOAETrajectory so evaluate.py can treat both
    # trajectory types uniformly when extracting KV dynamics for analysis.
    kv_dynamics_summary: Optional[Dict] = None


def speculative_inference(
    dual_model: DualModelWrapper,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    record_trajectory: bool = False,
    policy_temperature: float = 1.0,
    track_kv_dynamics: bool = False,
) -> Tuple[torch.Tensor, Optional[SpeculativeTrajectory]]:
    """
    Run speculative diffusion inference (Algorithm 1).

    Args:
        dual_model:       DualModelWrapper (soft primary + hard auxiliary).
        policy:           AOAE steering policy (with agreement input).
        soft_mask_module:  soft-masked state builder.
        prism_adapter:     PRISM quality head (or None).
        prompt_ids:        [B, P] prompt token ids.
        cfg:               config dict.
        record_trajectory: if True, store actions/log_probs for GRPO.
        policy_temperature: tau_pi for Bernoulli tempering.
        track_kv_dynamics: if True (and analysis.track_kv_dynamics=True in cfg),
            create a SpeculativeDynamicsTracker and populate
            trajectory.kv_dynamics_summary using the PRIMARY model's all-layer
            hidden states as a hidden-state proxy for KV drift.  Off by default
            during GRPO training to avoid the extra forward-pass overhead.

    Returns:
        output_ids: [B, P + L_gen] full sequence with generated tokens.
        trajectory: SpeculativeTrajectory (if record_trajectory, else None).
    """
    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = ic["gen_length"]
    mask_id = cfg["base_model"]["mask_token_id"]
    use_cache = cfg["cache"]["enabled"]
    use_fallback = ic["fallback_unmask"]
    disable_remask = ic.get("disable_remask", False)
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)
    use_positional_cache = bool(ic.get("positional_cache", {}).get("enabled", False))

    B = prompt_ids.shape[0]
    P = prompt_ids.shape[1]
    L_total = P + L_gen
    device = prompt_ids.device

    # --- Initialize: prompt + fully masked response ---
    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)  # [B, L_total]

    resp_slice = slice(P, L_total)

    # --- Two-cache system (K_spec + K_stable) ---
    _thrash_age_decay = float(cfg.get("grpo", {}).get("thrash_age_decay", 0.0))
    cache_mgr = (
        SpeculativeCacheBookkeeper(B, L_gen, device, thrash_age_decay=_thrash_age_decay)
        if use_cache else None
    )

    trajectory = SpeculativeTrajectory() if record_trajectory else None
    agreement_rates = []
    reuse_state = None
    pos_state = init_positional_state(B, L_gen, device)

    # H_t from the previous step: used to compute per-position drift for the
    # cache quality F1 signal.  Mirrors the same variable in aoae_inference().
    _prev_H_t: Optional[torch.Tensor] = None

    # --- KV cache state ---
    # K_spec skip (cache.kspec_skip): real wall-clock saving for the primary model.
    #   Auxiliary forward always runs with use_cache=True.  The returned aux_past_kv
    #   is used as the starting KV state for the primary; the primary only runs
    #   forward_replace_with_cache over non-agreed contiguous clusters, reusing
    #   aux K/V at agreed positions under the K_spec hypothesis (aux K/V ≈ pri K/V).
    #   Saving per step ≈ (agreement rate) × (primary response FLOP).
    #
    # Prefix KV cache (cache.prefix_kv_cache): additionally caches the prompt
    #   prefix for the auxiliary (skips prompt recompute each step).  The primary
    #   always starts from aux_past_kv so it also reuses prompt K/V implicitly.
    #
    # Both are disabled when hidden states are needed (PRISM, KV dynamics tracker).
    use_kspec_skip = cfg["cache"].get("kspec_skip", True) and use_cache
    use_prefix_kv_cache = cfg["cache"].get("prefix_kv_cache", False) and use_cache
    aux_past_kv = None
    pri_past_kv = None   # only used when kspec_skip=False and prefix_kv_cache=True
    _prefix_cache_initialized = False

    # Stable KV cache path (no drafter): maintain primary KV across steps and only
    # recompute at positions that changed (newly unmasked / remasked) or are still [MASK].
    # Active when cache.stable_kv_cache=true and kspec_skip=false.
    use_stable_kv_skip = (
        cfg["cache"].get("stable_kv_cache", False) and use_cache
        and not use_kspec_skip
    )
    stable_primary_kv = None      # persistent primary KV cache (initialized on step 1)
    _stable_logits_cache = None   # [B, L_gen, V] most recent real primary logits at each
                                  # position; updated at active positions each step and
                                  # substituted at stable positions before soft_mask_module
                                  # so H_t / policy / cache_quality_f1 see correct values.
    # Note: no prev_y_resp needed — K_stable mask (from cache_mgr.stable) is the
    # ground-truth for what's genuinely stable; changed/remasked positions are
    # evicted from K_stable automatically via invalidate()/step_stable().

    # --- KV dynamics tracker (eval diagnostic, off during GRPO rollouts) ---
    # Uses primary model all-layer hidden states as a hidden-state proxy for KV
    # drift (no actual K/V extraction needed).  When active, dual_forward_resp
    # is called with need_all_hidden=True so primary_hidden_states is populated;
    # this also covers PRISM's need_hidden=True (last hidden state is set too).
    _track_kv = track_kv_dynamics and bool(
        cfg.get("analysis", {}).get("track_kv_dynamics", False)
    )
    _dynamics_tracker = None
    if _track_kv:
        from .kv_dynamics import SpeculativeDynamicsTracker
        _dynamics_tracker = SpeculativeDynamicsTracker(cfg)

    # --- Main speculative diffusion loop ---
    for t in range(T, 0, -1):
        step_frac = t / T

        resp_tokens = y[:, resp_slice]             # [B, L_gen]
        mask_ind = (resp_tokens == mask_id)        # [B, L_gen]

        if not mask_ind.any():
            if trajectory is not None:
                trajectory.completion_step = torch.full((B,), T - t, device=device)
            break

        # === Phase 0 + 0b: Dual-model forward ===
        # When KV dynamics tracking is enabled, request all-layer hidden states
        # so the tracker can compute hidden-state-proxy drift per layer.
        # need_all_hidden=True also satisfies PRISM's need for the last hidden
        # state (dual_forward sets primary_hidden = primary_hidden_states[-1]).
        _need_all_hidden = _track_kv
        _need_hidden = (prism_adapter is not None) and not _need_all_hidden
        _can_use_cache = not _need_hidden and not _need_all_hidden

        if use_stable_kv_skip and _can_use_cache:
            # --- Stable KV path: primary only, skip positions in K_stable ---
            # K_stable = positions the policy's κ_t head committed as stable.
            # Eviction: remask (r_t=1) evicts from K_stable via invalidate().
            # Skip mask: K_stable positions that are NOT currently [MASK].
            #   - [MASK] positions always need fresh logits for decoding.
            #   - Non-K_stable unmasked positions: not yet committed or evicted;
            #     recomputed so their KV stays current until κ_t commits them.
            # Step 1: K_stable is empty → full primary forward (seeds stable_primary_kv).
            if stable_primary_kv is None:
                _full_logits, stable_primary_kv = dual_model.primary_forward_with_cache(y)
                resp_logits = _full_logits[:, resp_slice, :]
            else:
                # Use K_stable mask from end of previous step (post eviction/commit).
                _k_stable_mask = cache_mgr.stable.get_cached_mask()       # [B, L_gen]
                _stable_skip = _k_stable_mask & ~mask_ind                  # [B, L_gen]
                _fresh, stable_primary_kv = dual_model.primary_forward_with_stable_cache(
                    y, resp_slice, stable_primary_kv, _stable_skip,
                )
                # _fresh: primary logits at active positions; zeros at skipped positions.
                # Zeros are safe: skipped positions are unmasked and won't be sampled.
                resp_logits = _fresh
            # No drafter in stable path.
            aux_logits = resp_logits
            agreement = torch.zeros(B, L_gen, dtype=torch.bool, device=device)
            _pri_hidden_for_prism = None
            _pri_hidden_states_for_tracker = None

        elif use_kspec_skip and _can_use_cache:
            # --- K_spec path: real KV skip for agreed positions ---
            # Auxiliary runs with cache (prompt prefix reused when configured).
            if use_prefix_kv_cache and _prefix_cache_initialized:
                aux_logits, aux_past_kv = dual_model.auxiliary_forward_replace_with_cache(
                    y, resp_slice, aux_past_kv,
                )
            else:
                _aux_full, aux_past_kv = dual_model.auxiliary_forward_with_cache(y)
                aux_logits = _aux_full[:, resp_slice, :]
                _prefix_cache_initialized = True

            # Primary: run only over non-agreed response clusters (K_spec skip).
            # k_spec_mask = previous step's agreement; all-False on step 1 → full forward.
            _k_spec = cache_mgr.spec.get_cached_mask()  # [B, L_gen] prev-step agreement
            _pri_fresh, _ = dual_model.primary_forward_with_kspec(
                y, resp_slice, aux_past_kv, _k_spec,
            )
            # Merge: agreed positions → aux_logits (no primary recompute there);
            #        non-agreed positions → fresh primary logits.
            resp_logits = torch.where(
                _k_spec.unsqueeze(-1).expand_as(_pri_fresh),
                aux_logits,
                _pri_fresh,
            )
            _pri_hidden_for_prism = None
            _pri_hidden_states_for_tracker = None

        elif use_prefix_kv_cache and _can_use_cache:
            # --- Prefix-only path: no K_spec skip, but prompt KVs cached ---
            if not _prefix_cache_initialized:
                _aux_full, aux_past_kv = dual_model.auxiliary_forward_with_cache(y)
                _pri_full, pri_past_kv = dual_model.primary_forward_with_cache(y)
                aux_logits = _aux_full[:, resp_slice, :]
                resp_logits = _pri_full[:, resp_slice, :]
                _prefix_cache_initialized = True
            else:
                aux_logits, aux_past_kv = dual_model.auxiliary_forward_replace_with_cache(
                    y, resp_slice, aux_past_kv,
                )
                resp_logits, pri_past_kv = dual_model.primary_forward_replace_with_cache(
                    y, resp_slice, pri_past_kv,
                )
            _pri_hidden_for_prism = None
            _pri_hidden_states_for_tracker = None

        else:
            # --- Full forward path: needed for hidden states (PRISM, KV tracker) ---
            dual_out = dual_model.dual_forward_resp(
                y, resp_slice, need_hidden=_need_hidden, need_all_hidden=_need_all_hidden,
            )
            resp_logits = dual_out.primary_logits      # [B, L_gen, V]
            aux_logits = dual_out.auxiliary_logits      # [B, L_gen, V]
            _pri_hidden_for_prism = dual_out.primary_hidden
            _pri_hidden_states_for_tracker = dual_out.primary_hidden_states

        if not (use_stable_kv_skip and _can_use_cache):
            # Stable path sets agreement=zeros directly (no drafter to compare against).
            agreement, reuse_state, _ = compute_reuse_signal(
                resp_logits, aux_logits, cfg, state=reuse_state
            )
            agreement = agreement.bool()
        agreement_rates.append(float(agreement.float().mean().item()))

        # --- PRISM quality scores ---
        q_scores = None
        if prism_adapter is not None and _pri_hidden_for_prism is not None:
            with torch.no_grad():
                q_scores = prism_adapter(_pri_hidden_for_prism.float())

        # --- Correct resp_logits at stable positions (stable KV path only) ---
        # In the stable path, resp_logits is zeros at K_stable positions (we skipped
        # computing them).  Substituting cached logits there ensures that:
        #   (a) soft_mask_module produces correct H_t / confidence / weighted_embeds
        #       at stable positions (needed for policy κ_t/r_t and ω gradient);
        #   (b) cache_quality_f1 measures real drift, not the artifact of zero logits;
        #   (c) _prev_H_t stored below reflects true hidden-state proxy at stable positions.
        # The substitution is exact in expectation: stable KV → same primary attention
        # output → same logits → same H_t as the last step those positions were active.
        if use_stable_kv_skip and _can_use_cache:
            _stable_skip = cache_mgr.stable.get_cached_mask() & ~mask_ind  # [B, L_gen]
            if _stable_logits_cache is None:
                # Step 1: all positions were active, resp_logits is fully real.
                _stable_logits_cache = resp_logits.detach().clone()
            else:
                # Update cache at active positions with this step's fresh logits.
                _active_lc = ~_stable_skip   # [B, L_gen]
                _stable_logits_cache = torch.where(
                    _active_lc.unsqueeze(-1).expand_as(_stable_logits_cache),
                    resp_logits.detach(),
                    _stable_logits_cache,
                )
                # Substitute cached logits at stable positions.
                resp_logits = torch.where(
                    _stable_skip.unsqueeze(-1).expand_as(resp_logits),
                    _stable_logits_cache,
                    resp_logits,
                )

        # --- Construct soft-masked state from PRIMARY logits ---
        H_t, confidence, entropy, weighted_embeds = soft_mask_module(
            resp_logits, mask_ind, step_frac
        )

        # --- Cache quality F1 (soft precision-recall of cache set) ---
        # Measured BEFORE Phase 1 invalidation so we capture the quality of
        # the cache set as it stood at the start of this step.
        #
        # Identical computation to aoae_inference (inference.py lines 205-244),
        # using the PRIMARY model's H_t as the hidden-state proxy for drift.
        # This was previously absent from the speculative path entirely, causing
        # cache_quality_f1 to remain empty and the reward to silently drop the
        # entire cache quality term.
        if trajectory is not None and _prev_H_t is not None and cache_mgr is not None:
            _cached_mask = cache_mgr.get_cached_mask()           # [B, L_gen] bool
            _h_delta = (H_t.detach() - _prev_H_t).norm(dim=-1)  # [B, L_gen]
            _h_norm  = _prev_H_t.norm(dim=-1).clamp(min=1e-8)   # [B, L_gen]
            _rel_drift = _h_delta / _h_norm                      # [B, L_gen] ∈ [0, ~2]

            # Soft stability: threshold-free, scale-invariant
            _stab_lambda = float(cfg.get("grpo", {}).get("stability_lambda", 10.0))
            _all_stability = torch.exp(-_stab_lambda * _rel_drift)  # [B, L_gen]

            _cached_f = _cached_mask.float()                         # [B, L_gen]
            _n_cached = _cached_f.sum(-1).clamp(min=1.0)             # [B]

            # Precision: mean stability of cached positions
            _cached_prec = (_all_stability * _cached_f).sum(-1) / _n_cached  # [B]

            # Recall: fraction of total stability budget captured by cache
            _total_stab = _all_stability.sum(-1).clamp(min=1e-8)    # [B]
            _cached_stab = (_all_stability * _cached_f).sum(-1)     # [B]
            _recall = _cached_stab / _total_stab                    # [B]

            # Harmonic mean (F1)
            _cache_f1 = 2.0 * _cached_prec * _recall / (_cached_prec + _recall + 1e-8)  # [B]
            trajectory.cache_quality_f1.append(_cache_f1.detach())
        _prev_H_t = H_t.detach()

        age_feat = None
        last_action_feat = None
        if use_positional_cache:
            age_feat, last_action_feat = get_policy_positional_features(pos_state, cfg)

        # --- Policy forward (with agreement signal) ---
        policy_out = call_policy(
            policy,
            H_t, mask_ind, step_frac,
            temperature=policy_temperature,
            confidence=confidence,
            quality_scores=q_scores,
            agreement=agreement.float(),
            age_feature=age_feat,
            last_action_feature=last_action_feat,
        )
        pol_inner = policy.module if hasattr(policy, "module") else policy

        # --- Sample actions ---
        actions = pol_inner.sample_actions(policy_out, mask_ind)
        u_t = actions["u_t"]
        r_t = actions["r_t"]
        kappa_t = actions["kappa_t"]
        if disable_remask:
            r_t = torch.zeros_like(r_t)
            actions = {**actions, "r_t": r_t}
        q_exec, q_mandatory, _access_diag = build_access_set(
            actions,
            policy_out,
            cfg,
            confidence=confidence,
            boundary_action=actions.get("ell_t"),
            boundary_num_bins=(
                int(policy_out["boundary_probs"].shape[-1])
                if "boundary_probs" in policy_out
                else None
            ),
        )
        actions = {**actions, "q_t_mandatory": q_mandatory}

        # --- Record trajectory ---
        if trajectory is not None:
            lp = pol_inner.log_prob(policy_out, actions)
            trajectory.actions.append({k: v.detach() for k, v in actions.items()})
            trajectory.log_probs.append(lp.detach())
            trajectory.policy_outputs.append(
                {k: v.detach() for k, v in policy_out.items()}
            )
            trajectory.H_t_list.append(H_t.detach())
            trajectory.weighted_embeds_list.append(weighted_embeds.detach())
            trajectory.entropy_list.append(entropy.detach())
            trajectory.mask_ind_list.append(mask_ind.detach())
            trajectory.quality_scores_list.append(
                q_scores.detach() if q_scores is not None else None
            )
            trajectory.agreement_list.append(agreement.detach())
            trajectory.age_feature_list.append(age_feat.detach() if age_feat is not None else None)
            trajectory.last_action_feature_list.append(
                last_action_feat.detach() if last_action_feat is not None else None
            )
            trajectory.access_exec_list.append(q_exec.detach())
            trajectory.access_mandatory_list.append(q_mandatory.detach())
            if "ell_t" in actions:
                trajectory.boundary_actions.append(actions["ell_t"].detach())
            trajectory.step_fracs.append(step_frac)

        # --- Count cache thrashing ---
        if cache_mgr is not None and trajectory is not None:
            thrash = cache_mgr.count_thrash(r_t)
            trajectory.thrash_counts.append(thrash.detach())

        resp_tokens = resp_tokens.clone()
        fallback_positions = torch.zeros_like(mask_ind)

        # ====== Phase 1: Remask ======
        remask_positions = r_t.bool() & ~mask_ind
        if remask_positions.any():
            resp_tokens[remask_positions] = mask_id
            if cache_mgr is not None:
                cache_mgr.invalidate(r_t)

        # ====== Phase 2: Unmask with Composed Prediction ======
        unmask_positions = u_t.bool() & mask_ind
        if unmask_positions.any():
            if gamma > 0:
                composed_logits = compose_prediction_dual(
                    resp_logits, aux_logits, agreement, gamma=gamma,
                )
            else:
                composed_logits = resp_logits

            if base_temp > 0:
                probs = F.softmax(composed_logits / base_temp, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, L_gen)
            else:
                sampled = composed_logits.argmax(dim=-1)
            resp_tokens[unmask_positions] = sampled[unmask_positions]

        # ====== Fallback ======
        if use_fallback:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)
            if no_unmasks.any():
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    masked_pos = still_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        best_pos = masked_pos[confidence[b_idx, masked_pos].argmax()]
                        resp_tokens[b_idx, best_pos] = resp_logits[b_idx, best_pos].argmax()
                        fallback_positions[b_idx, best_pos] = True

        # ====== Phase 3a: Speculative Accept (K_spec) ======
        # Positions where drafter and verifier agree: KV valid for one step.
        # K_spec is replaced each step (not accumulated).
        if cache_mgr is not None:
            cache_mgr.step_spec(agreement)

        # ====== Phase 3b: Stable Cache (K_stable) ======
        # Positions the κ_t head predicts will remain stable across future steps.
        # Accumulated persistently; evicted by r_t (Phase 1 already called invalidate).
        if cache_mgr is not None:
            cache_mgr.step_stable(kappa_t, r_t)

            if trajectory is not None:
                trajectory.total_cache_hits += int(
                    (agreement.float() + kappa_t.float()).clamp(0, 1).sum().item()
                )
                trajectory.total_cache_misses += int(
                    ((~agreement).float() * (1 - kappa_t.float())).sum().item()
                )

        # --- Record cached fractions after commit (compute-aware speed bonus) ---
        # spec_cached_fractions: K_spec (agreement-based, one-step, replaced each step).
        # cached_fractions:      K_spec ∪ K_stable (combined, for reward computation).
        # effective_flops = (used_steps/T) * (1 - mean_combined_cached_fraction).
        if trajectory is not None and cache_mgr is not None:
            trajectory.spec_cached_fractions.append(cache_mgr.spec_cached_fraction().detach())
            trajectory.cached_fractions.append(cache_mgr.cached_fraction().detach())

        # --- KV dynamics tracker observation (eval diagnostic only) ---
        # Uses primary model all-layer hidden states as a hidden-state proxy for
        # KV drift.  The real agreement signal is directly available here (unlike
        # the single-model path which uses a zeros proxy).
        if _dynamics_tracker is not None:
            _layer_hiddens_for_tracker = (
                [h.detach() for h in _pri_hidden_states_for_tracker]
                if _pri_hidden_states_for_tracker is not None
                else []
            )
            _q_t_tracked = actions.get("q_t", torch.zeros_like(u_t))
            _dynamics_tracker.observe_step(
                layer_hiddens=_layer_hiddens_for_tracker,
                max_prob=confidence,
                mask_ind=mask_ind,
                agreement=agreement.float(),  # real agreement, not proxy
                u_t=u_t,
                r_t=r_t,
                kappa_t=kappa_t,
                q_t=_q_t_tracked,
            )

        changed = (u_t.bool() | r_t.bool() | fallback_positions).float()
        if trajectory is not None:
            trajectory.changed_list.append(changed.detach())
        if use_positional_cache:
            update_positional_state(pos_state, q_exec=q_exec, changed=changed, cfg=cfg)

        if trajectory is not None:
            y = y.clone()
        y[:, resp_slice] = resp_tokens

    # --- Record final state ---
    if trajectory is not None:
        trajectory.final_tokens = y[:, resp_slice].detach()
        if trajectory.completion_step is None:
            trajectory.completion_step = torch.full((B,), T, device=device)
        if agreement_rates:
            trajectory.mean_agreement_rate = sum(agreement_rates) / len(agreement_rates)
        pc_cfg = cfg.get("inference", {}).get("positional_cache", {})
        if pc_cfg.get("enabled", False):
            horizon = int(pc_cfg.get("horizon", 4))
            trajectory.access_metrics = compute_next_h_access_metrics(
                access_exec_steps=trajectory.access_exec_list,
                changed_steps=trajectory.changed_list,
                mandatory_steps=trajectory.access_mandatory_list,
                horizon=horizon,
            )
        else:
            trajectory.access_metrics = compute_next_h_access_metrics([], [], None, 1)
        if trajectory.boundary_actions:
            all_boundary = torch.cat([x.reshape(-1) for x in trajectory.boundary_actions], dim=0)
            max_bin = int(all_boundary.max().item()) if all_boundary.numel() > 0 else 0
            denom = max(max_bin, 1)
            trajectory.mean_boundary_depth = float((all_boundary.float() / denom).mean().item())
            counts = torch.bincount(all_boundary, minlength=max_bin + 1).tolist()
            trajectory.boundary_distribution = json.dumps({str(i): int(v) for i, v in enumerate(counts)})
        else:
            trajectory.mean_boundary_depth = 0.0
            trajectory.boundary_distribution = "{}"

    # --- Finalize KV dynamics tracker ---
    # If the tracker ran, store its summary in the trajectory so evaluate.py
    # and the training logger can collect it (same pattern as aoae_inference).
    if _dynamics_tracker is not None:
        _dyn_summary = _dynamics_tracker.summarize()
        if trajectory is None:
            trajectory = SpeculativeTrajectory()
        trajectory.kv_dynamics_summary = _dyn_summary

    return y, trajectory
