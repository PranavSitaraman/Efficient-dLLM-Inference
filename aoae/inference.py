"""
AOAE Inference Loop (Algorithm 1 from paper §3.3).

Implements the full three-phase per-step procedure:
  Phase 1: Remask (revert uncertain positions to [M])
  Phase 2: Unmask (M2T) with composed prediction
  Phase 3: Cache commit

Also implements baseline decoders for comparison:
  - Uniform unmasking (standard MDLM)
  - Confidence-threshold (LLaDA 2.1 S-Mode / Q-Mode style)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import json


from .cache import DKVCacheManager
from .experiment_utils import parse_head_set
from .models.composed_prediction import compose_prediction
from .models.policy import (
    active_block_window,
    apply_unmask_budget,
    call_policy,
    call_policy_block,
)
from .models.soft_mask import call_soft_mask
from .positional_cache import (
    init_positional_state,
    get_policy_positional_features,
    build_access_set,
    update_positional_state,
    compute_next_h_access_metrics,
    compute_next_h_access_metrics_per_sample,
    summarize_access_diagnostics,
)


def _max_prob_and_argmax(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return max softmax probability and argmax token without allocating full probs."""
    logits_f = logits.float()
    max_logits, max_tok = logits_f.max(dim=-1)
    max_prob = torch.exp(max_logits - torch.logsumexp(logits_f, dim=-1))
    return max_prob.type_as(logits), max_tok


def _resolve_eos_token_id(base_model) -> Optional[int]:
    tokenizer = getattr(base_model, "tokenizer", None)
    if tokenizer is None:
        return None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return None
    try:
        return int(eos_token_id)
    except (TypeError, ValueError):
        return None


def _mask_after_first_eos(
    tokens: torch.Tensor,
    *,
    eos_token_id: Optional[int],
    mask_id: int,
) -> torch.Tensor:
    if eos_token_id is None or tokens.numel() == 0:
        return tokens
    out = tokens.clone()
    for row_idx in range(out.shape[0]):
        eos_pos = (out[row_idx] == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_pos.numel() == 0:
            continue
        first = int(eos_pos[0].item())
        if first + 1 < out.shape[1]:
            out[row_idx, first + 1 :] = int(mask_id)
    return out


def resolve_llada21_official_settings(cfg: dict, mode: str = "speed") -> Dict[str, Any]:
    """Resolve the official LLaDA2.1 decode settings for speed or quality mode."""
    mode = str(mode).lower()
    if mode not in {"speed", "quality"}:
        raise ValueError(f"Unknown LLaDA2.1 decode mode: {mode!r}")

    defaults = {
        "speed": {"threshold": 0.5, "editing_threshold": 0.0},
        "quality": {"threshold": 0.7, "editing_threshold": 0.5},
    }
    inf_cfg = cfg.get("inference", {})
    off_cfg = inf_cfg.get("llada21_official", {})
    mode_cfg = off_cfg.get(mode, {}) if isinstance(off_cfg.get(mode), dict) else {}
    legacy_threshold = off_cfg.get("threshold")
    legacy_editing_threshold = off_cfg.get("editing_threshold")

    threshold = mode_cfg.get("threshold")
    if threshold is None:
        threshold = legacy_threshold if legacy_threshold is not None else defaults[mode]["threshold"]

    editing_threshold = mode_cfg.get("editing_threshold")
    if editing_threshold is None:
        editing_threshold = (
            legacy_editing_threshold
            if legacy_editing_threshold is not None
            else defaults[mode]["editing_threshold"]
        )

    return {
        "mode": mode,
        "threshold": float(threshold),
        "editing_threshold": float(editing_threshold),
        "use_block_diffusion": bool(off_cfg.get("use_block_diffusion", True)),
        "max_post_steps": int(off_cfg.get("max_post_steps", 16)),
        "enable_mbe": bool(off_cfg.get("enable_mbe", False)),
        "gen_length": int(off_cfg.get("gen_length", max(512, int(inf_cfg.get("gen_length", 512))))),
        "eos_early_stop": bool(off_cfg.get("eos_early_stop", True)),
        "block_length": int(inf_cfg.get("block_length", 32)),
        "temperature": float(inf_cfg.get("temperature", 0.0)),
    }


@dataclass
class AOAETrajectory:
    """Stores a full inference trajectory for GRPO training."""
    actions: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    policy_outputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    thrash_counts: List[torch.Tensor] = field(default_factory=list)
    H_t_list: List[torch.Tensor] = field(default_factory=list)
    # Soft-mask intermediates stored for differentiable ω re-computation in GRPO loss.
    # Storing these (instead of just H_t) lets autograd flow through ω_s/ω_a/ω_b
    # without re-running the base model or storing full vocab-size logits.
    weighted_embeds_list: List[torch.Tensor] = field(default_factory=list)
    entropy_list: List[torch.Tensor] = field(default_factory=list)
    mask_ind_list: List[torch.BoolTensor] = field(default_factory=list)
    quality_scores_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    age_feature_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    last_action_feature_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    access_exec_list: List[torch.Tensor] = field(default_factory=list)
    access_mandatory_list: List[torch.Tensor] = field(default_factory=list)
    access_diag_list: List[Dict[str, float]] = field(default_factory=list)
    changed_list: List[torch.Tensor] = field(default_factory=list)
    boundary_actions: List[torch.Tensor] = field(default_factory=list)
    step_fracs: List[float] = field(default_factory=list)
    final_tokens: Optional[torch.Tensor] = None
    completion_step: Optional[torch.Tensor] = None
    access_metrics: Dict[str, float] = field(default_factory=dict)
    access_metric_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    mean_boundary_depth: float = 0.0
    boundary_distribution: str = "{}"
    # ---- compute-aware speed bonus ----
    # Fraction of positions currently in the dKV-Cache at each step (after Phase 3
    # commit).  Used to compute effective_flops = (used_steps/T)*(1-mean_cached),
    # which captures BOTH fewer forward passes AND cheaper passes due to caching.
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
    # This subsumes the old drift_penalty (which was precision-only, no recall)
    # and adds the missing "go commit those stable tokens" gradient.
    cache_quality_f1: List[torch.Tensor] = field(default_factory=list)
    # ---- KV dynamics tracker summary (eval only, None during training) ----
    kv_dynamics_summary: Optional[Dict] = None


def aoae_inference(
    base_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    record_trajectory: bool = False,
    policy_temperature: float = 1.0,
    track_kv_dynamics: bool = False,
) -> Tuple[torch.Tensor, Optional[AOAETrajectory]]:
    """
    Run AOAE inference (Algorithm 1).

    Args:
        base_model:       frozen LLaDA wrapper.
        policy:           AOAE policy network.
        soft_mask_module:  soft-masked state builder.
        prism_adapter:     PRISM quality head (or None to skip edit/remask).
        prompt_ids:        [B, P] prompt token ids.
        cfg:               config dict.
        record_trajectory: if True, store actions/log_probs for GRPO.
        policy_temperature: tau_pi for Bernoulli tempering.

    Returns:
        output_ids: [B, P + L_gen] full sequence with generated tokens.
        trajectory: AOAETrajectory (if record_trajectory=True, else None).
    """
    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = ic["gen_length"]
    mask_id = cfg["base_model"]["mask_token_id"]
    use_cache = cfg["cache"]["enabled"]
    use_fallback = ic["fallback_unmask"]
    disable_remask = ic.get("disable_remask", False)
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)  # Composed prediction strength
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

    # Only operate on the response region [P:]
    resp_slice = slice(P, L_total)

    # --- dKV-Cache ---
    cache_mgr = DKVCacheManager(B, L_gen, device) if use_cache else None

    trajectory = AOAETrajectory() if record_trajectory else None
    pos_state = init_positional_state(B, L_gen, device)

    # --- KV dynamics tracker (eval diagnostic, off during GRPO rollouts) ---
    # Uses hidden-state proxy (forward_with_all_hidden) when enabled.
    # Requires track_kv_dynamics=True AND analysis.track_kv_dynamics=True in cfg.
    _track_kv = track_kv_dynamics and bool(
        cfg.get("analysis", {}).get("track_kv_dynamics", False)
    )
    _dynamics_tracker = None
    if _track_kv:
        from .kv_dynamics import SpeculativeDynamicsTracker
        _dynamics_tracker = SpeculativeDynamicsTracker(cfg)

    # H_t from the previous step: used to compute per-cached-position drift.
    # drift_k = ||H_t^k - H_{t-1}^k|| / ||H_{t-1}^k||  for k in K (cache set)
    # This is a proxy for actual KV-vector drift without requiring K/V extraction.
    _prev_H_t: Optional[torch.Tensor] = None

    # --- Main diffusion loop: t = T, T-1, ..., 1 ---
    for t in range(T, 0, -1):
        step_frac = t / T

        # Mask indicator for response region
        resp_tokens = y[:, resp_slice]                      # [B, L_gen]
        mask_ind = (resp_tokens == mask_id)                 # [B, L_gen]

        # Check if all masks are resolved
        if not mask_ind.any():
            if trajectory is not None:
                trajectory.completion_step = torch.full((B,), T - t, device=device)
            break

        # --- Base model forward ---
        # When KV dynamics tracking is on we need all layer hidden states for the
        # SpeculativeDynamicsTracker (hidden-state-proxy drift mode).  We reuse the
        # same forward_with_all_hidden call for PRISM if both are active, avoiding
        # a redundant pass.
        _layer_hiddens_for_tracker: List[torch.Tensor] = []
        if _track_kv:
            logits, _all_hidden = base_model.forward_with_all_hidden(y)
            resp_hidden = _all_hidden[-1][:, resp_slice, :] if prism_adapter is not None else None
            _layer_hiddens_for_tracker = [h[:, resp_slice, :].detach() for h in _all_hidden]
        elif prism_adapter is not None:
            logits, hidden_states = base_model.forward_with_hidden(y)
            resp_hidden = hidden_states[:, resp_slice, :]
        else:
            logits = base_model.forward(y)
            resp_hidden = None
        resp_logits = logits[:, resp_slice, :]

        # --- PRISM quality scores ---
        q_scores = None
        if prism_adapter is not None and resp_hidden is not None:
            with torch.no_grad():
                q_scores = prism_adapter(resp_hidden.float())  # [B, L_gen]

        # --- Construct soft-masked state ---
        H_t, confidence, entropy, weighted_embeds = call_soft_mask(
            soft_mask_module,
            resp_logits, mask_ind, step_frac, return_weighted=True
        )  # H_t: [B, L_gen, D]

        # --- Cache quality F1 (soft precision-recall of cache set) ---
        # Measured BEFORE Phase 1 invalidation so we capture the quality of
        # the cache set as it stood at the start of this step.
        #
        # For every position k we compute a soft stability score:
        #   stability(k) = exp(-λ * rel_drift_k)   ∈ (0, 1]
        # where rel_drift_k = ||H_t^k - H_{t-1}^k||₂ / ||H_{t-1}^k||₂.
        #
        # Soft precision/recall over cache set K_t:
        #   precision = mean_{k ∈ K_t}(stability(k))
        #   recall    = Σ_{k ∈ K_t} stability(k) / Σ_all stability(k)
        #   F1        = 2 * precision * recall / (precision + recall)
        #
        # This subsumes the old drift penalty (precision-only) and adds
        # the missing recall gradient: "commit those stable tokens."
        if trajectory is not None and _prev_H_t is not None and cache_mgr is not None:
            _cached_mask = cache_mgr.get_cached_mask()          # [B, L_gen] bool
            _h_delta = (H_t.detach() - _prev_H_t).norm(dim=-1) # [B, L_gen]
            _h_norm  = _prev_H_t.norm(dim=-1).clamp(min=1e-8)  # [B, L_gen]
            _rel_drift = _h_delta / _h_norm                     # [B, L_gen] ∈ [0, ~2]

            # Soft stability: threshold-free, scale-invariant
            _stab_lambda = float(cfg.get("grpo", {}).get("stability_lambda", 10.0))
            _all_stability = torch.exp(-_stab_lambda * _rel_drift)  # [B, L_gen]

            _cached_f = _cached_mask.float()                        # [B, L_gen]
            _n_cached = _cached_f.sum(-1).clamp(min=1.0)            # [B]

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

        # --- Policy forward (with PRISM quality scores) ---
        # Block-wise policy (Option A) — see aoae/models/policy.py for design.
        _blockwise_cfg = (cfg.get("policy", {}) or {}).get("block_wise", {}) or {}
        if bool(_blockwise_cfg.get("enabled", False)):
            _blk_window = active_block_window(
                mask_ind,
                max(1, int(ic.get("block_length", 32))),
                context_left=max(0, int(_blockwise_cfg.get("context_left", 0))),
            )
            policy_out = call_policy_block(
                policy,
                H_t, mask_ind, step_frac,
                _blk_window,
                temperature=policy_temperature,
                confidence=confidence,
                quality_scores=q_scores,
                age_feature=age_feat,
                last_action_feature=last_action_feat,
            )
        else:
            policy_out = call_policy(
                policy,
                H_t, mask_ind, step_frac,
                temperature=policy_temperature,
                confidence=confidence,
                quality_scores=q_scores,
                age_feature=age_feat,
                last_action_feature=last_action_feat,
            )
        pol_inner = policy.module if hasattr(policy, "module") else policy

        # --- Sample actions ---
        actions = pol_inner.sample_actions(policy_out, mask_ind)
        actions = apply_unmask_budget(actions, policy_out, mask_ind, cfg)
        u_t = actions["u_t"]        # [B, L_gen] unmask
        r_t = actions["r_t"]        # [B, L_gen] remask
        kappa_t = actions["kappa_t"]  # [B, L_gen] cache
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

        # --- Record trajectory for GRPO ---
        if trajectory is not None:
            include_heads = parse_head_set(
                cfg.get("grpo", {}).get(
                    "include_heads_in_logprob",
                    cfg.get("grpo", {}).get("train_heads"),
                )
            )
            lp = pol_inner.log_prob(policy_out, actions, include_heads=include_heads)
            trajectory.actions.append({k: v.detach() for k, v in actions.items()})
            trajectory.log_probs.append(lp.detach())
            trajectory.policy_outputs.append(
                {k: v.detach() for k, v in policy_out.items()}
            )
            # Store states for off-policy importance sampling.
            # weighted_embeds and entropy are the soft-mask intermediates that
            # allow ω_s/ω_a/ω_b to receive gradients in compute_grpo_loss
            # via soft_mask_module.recompute_h_t() (no base model re-run needed).
            trajectory.H_t_list.append(H_t.detach())
            trajectory.weighted_embeds_list.append(weighted_embeds.detach())
            trajectory.entropy_list.append(entropy.detach())
            trajectory.mask_ind_list.append(mask_ind.detach())
            trajectory.quality_scores_list.append(
                q_scores.detach() if q_scores is not None else None
            )
            trajectory.age_feature_list.append(age_feat.detach() if age_feat is not None else None)
            trajectory.last_action_feature_list.append(
                last_action_feat.detach() if last_action_feat is not None else None
            )
            trajectory.access_exec_list.append(q_exec.detach())
            trajectory.access_mandatory_list.append(q_mandatory.detach())
            trajectory.access_diag_list.append(dict(_access_diag))
            if "ell_t" in actions:
                trajectory.boundary_actions.append(actions["ell_t"].detach())
            trajectory.step_fracs.append(step_frac)

        # --- Count cache thrashing BEFORE invalidation ---
        if cache_mgr is not None and trajectory is not None:
            thrash = cache_mgr.count_thrash(r_t)
            trajectory.thrash_counts.append(thrash.detach())

        # Clone once for all mutations this step
        resp_tokens = resp_tokens.clone()
        fallback_positions = torch.zeros_like(mask_ind)

        # ====== Phase 1: Remask ======
        remask_positions = r_t.bool() & ~mask_ind  # only unmasked positions
        if remask_positions.any():
            resp_tokens[remask_positions] = mask_id
            if cache_mgr is not None:
                cache_mgr.invalidate(r_t)

        # ====== Phase 2: Unmask (M2T) with Composed Prediction ======
        unmask_positions = u_t.bool() & mask_ind
        if unmask_positions.any():
            # Apply composed prediction: sharpen distribution at cache-likely positions
            if gamma > 0 and "cache_probs" in policy_out:
                composed_logits = compose_prediction(
                    resp_logits, policy_out["cache_probs"], gamma=gamma,
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

        # ====== Fallback: unmask highest-confidence if no unmasks ======
        if use_fallback:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)  # [B]
            if no_unmasks.any():
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    masked_pos = still_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        best_pos = masked_pos[confidence[b_idx, masked_pos].argmax()]
                        resp_tokens[b_idx, best_pos] = resp_logits[b_idx, best_pos].argmax()
                        fallback_positions[b_idx, best_pos] = True

        # ====== Phase 3: Cache commit ======
        if cache_mgr is not None:
            cache_mgr.commit(kappa_t * q_exec)

        # --- Record cached fraction after commit (for compute-aware speed bonus) ---
        # Stored per-step so compute_reward() can compute mean_cached_fraction and
        # use it in: effective_flops = (used_steps/T) * (1 - mean_cached_fraction)
        if trajectory is not None and cache_mgr is not None:
            trajectory.cached_fractions.append(cache_mgr.cached_fraction().detach())

        changed = (u_t.bool() | r_t.bool() | fallback_positions).float()
        if trajectory is not None:
            trajectory.changed_list.append(changed.detach())
        if use_positional_cache:
            update_positional_state(pos_state, q_exec=q_exec, changed=changed, cfg=cfg)

        # --- KV dynamics tracker observation (eval diagnostic only) ---
        # Uses all-layer hidden states (hidden-state proxy drift, not actual K/V).
        # agreement is unavailable in single-model mode → zeros (no auxiliary).
        if _dynamics_tracker is not None and _layer_hiddens_for_tracker:
            _agreement_proxy = torch.zeros(B, L_gen, device=device)
            _q_t_tracked = actions.get("q_t", torch.zeros_like(u_t))
            _dynamics_tracker.observe_step(
                layer_hiddens=_layer_hiddens_for_tracker,
                max_prob=confidence,
                mask_ind=mask_ind,
                agreement=_agreement_proxy,
                u_t=u_t,
                r_t=r_t,
                kappa_t=kappa_t,
                q_t=_q_t_tracked,
            )

        # --- Write back response tokens ---
        if trajectory is not None:
            y = y.clone()
        y[:, resp_slice] = resp_tokens

    # --- Record final state ---
    if trajectory is not None:
        trajectory.final_tokens = y[:, resp_slice].detach()
        if trajectory.completion_step is None:
            # Did not break early — used all T steps
            trajectory.completion_step = torch.full((B,), T, device=device)
        pc_cfg = cfg.get("inference", {}).get("positional_cache", {})
        if pc_cfg.get("enabled", False):
            horizon = int(pc_cfg.get("horizon", 4))
            trajectory.access_metrics = summarize_access_diagnostics(trajectory.access_diag_list)
            trajectory.access_metrics.update(compute_next_h_access_metrics(
                access_exec_steps=trajectory.access_exec_list,
                changed_steps=trajectory.changed_list,
                mandatory_steps=trajectory.access_mandatory_list,
                horizon=horizon,
            ))
            trajectory.access_metric_tensors = compute_next_h_access_metrics_per_sample(
                access_exec_steps=trajectory.access_exec_list,
                changed_steps=trajectory.changed_list,
                mandatory_steps=trajectory.access_mandatory_list,
                horizon=horizon,
            )
        else:
            trajectory.access_metrics = summarize_access_diagnostics([])
            trajectory.access_metrics.update(compute_next_h_access_metrics([], [], None, 1))
            trajectory.access_metric_tensors = {}
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
    # If the tracker ran, store its summary so evaluate_aoae() can collect it.
    # When record_trajectory=False (normal eval path), we create a minimal
    # trajectory shell just to carry the summary — avoids changing the return type.
    if _dynamics_tracker is not None:
        _dyn_summary = _dynamics_tracker.summarize()
        if trajectory is None:
            trajectory = AOAETrajectory()
        trajectory.kv_dynamics_summary = _dyn_summary

    return y, trajectory


# ======================================================================
# Baseline decoders for comparison
# ======================================================================


def _force_complete_masked_positions(
    base_model,
    y: torch.Tensor,
    resp_slice: slice,
    mask_id: int,
    max_passes: int = 4,
    skip_rows: Optional[torch.BoolTensor] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Fill any remaining [MASK] positions so eval compares complete outputs.

    Each forward pass is counted into ``stats['iterations']`` if a ``stats``
    dict is provided; this keeps NFE reporting honest across baselines whose
    decoders run a variable number of forwards.
    """
    extra = 0
    for _ in range(max(1, int(max_passes))):
        resp = y[:, resp_slice]
        masked = resp.eq(mask_id)
        if skip_rows is not None:
            masked = masked & (~skip_rows).unsqueeze(-1)
        if not masked.any():
            break

        logits = base_model.forward(y)[:, resp_slice, :]
        extra += 1
        if 0 <= int(mask_id) < int(logits.shape[-1]):
            logits = logits.clone()
            logits[..., int(mask_id)] = torch.finfo(logits.dtype).min
        fill_tokens = logits.argmax(dim=-1)

        resp = resp.clone()
        resp[masked] = fill_tokens[masked]
        y[:, resp_slice] = resp

    if stats is not None:
        stats["iterations"] = int(stats.get("iterations", 0)) + extra
        stats["force_complete_passes"] = extra
    return y

def uniform_decode(
    base_model,
    prompt_ids: torch.LongTensor,
    cfg: dict,
) -> torch.Tensor:
    """Standard uniform unmasking baseline: unmask L/T tokens per step."""
    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = ic["gen_length"]
    mask_id = cfg["base_model"]["mask_token_id"]

    B, P = prompt_ids.shape
    device = prompt_ids.device

    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    resp_slice = slice(P, P + L_gen)
    tokens_per_step = max(1, L_gen // T)

    for t in range(T, 0, -1):
        resp = y[:, resp_slice]
        mask_ind = (resp == mask_id)

        if not mask_ind.any():
            break

        logits = base_model.forward(y)[:, resp_slice, :]

        for b in range(B):
            masked_pos = mask_ind[b].nonzero(as_tuple=True)[0]
            if len(masked_pos) == 0:
                continue
            n_unmask = min(tokens_per_step, len(masked_pos))
            # Random selection
            perm = torch.randperm(len(masked_pos), device=device)[:n_unmask]
            sel = masked_pos[perm]
            y[b, P + sel] = logits[b, sel].argmax(dim=-1)

    return _force_complete_masked_positions(base_model, y, resp_slice, mask_id)


def confidence_threshold_decode(
    base_model,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    tau_mask: float = 0.9,
    tau_edit: float = 0.95,
    enable_t2t: bool = True,
    gen_length: Optional[int] = None,
    eos_early_stop: bool = False,
    stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    LLaDA 2.1-style confidence threshold decoding.

    Implements S-Mode (aggressive thresholds) or Q-Mode (conservative)
    depending on tau_mask and tau_edit.

    Pass a ``stats`` dict to receive the *actual* number of model forwards
    (loop iterations + force-complete passes); the loop terminates early when
    every position has been unmasked, so NFE is typically much smaller than T.
    """
    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = int(gen_length if gen_length is not None else ic["gen_length"])
    mask_id = cfg["base_model"]["mask_token_id"]

    B, P = prompt_ids.shape
    device = prompt_ids.device
    eos_token_id = _resolve_eos_token_id(base_model) if eos_early_stop else None
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    resp_slice = slice(P, P + L_gen)
    iterations = 0

    for t in range(T, 0, -1):
        resp = y[:, resp_slice]
        active_rows = (~finished).unsqueeze(-1)
        mask_ind = (resp == mask_id) & active_rows

        if not mask_ind.any():
            break

        logits = base_model.forward(y)[:, resp_slice, :]
        iterations += 1
        max_prob, max_tok = _max_prob_and_argmax(logits)

        # M2T: unmask positions above tau_mask
        unmask = mask_ind & (max_prob > tau_mask)
        resp = resp.clone()
        resp[unmask] = max_tok[unmask]

        # T2T: edit unmasked positions where model disagrees and confidence > tau_edit
        if enable_t2t:
            unmasked = (resp != mask_id) & active_rows
            disagree = (max_tok != resp) & unmasked
            confident = max_prob > tau_edit
            edit = disagree & confident
            resp[edit] = max_tok[edit]

        # Fallback: if nothing was unmasked, unmask the most confident
        still_masked = (resp == mask_id) & active_rows
        nothing_happened = mask_ind.any(dim=-1) & ~unmask.any(dim=-1)
        for b in nothing_happened.nonzero(as_tuple=True)[0]:
            masked_pos = still_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_pos) > 0:
                best = masked_pos[max_prob[b, masked_pos].argmax()]
                resp[b, best] = max_tok[b, best]

        if eos_early_stop and eos_token_id is not None:
            resp = _mask_after_first_eos(
                resp,
                eos_token_id=eos_token_id,
                mask_id=mask_id,
            )
            finished = finished | (resp == eos_token_id).any(dim=-1)

        y[:, resp_slice] = resp
        if eos_early_stop and finished.all():
            break

    if stats is not None:
        stats["iterations"] = int(stats.get("iterations", 0)) + iterations
    return _force_complete_masked_positions(
        base_model,
        y,
        resp_slice,
        mask_id,
        skip_rows=finished if eos_early_stop else None,
        stats=stats,
    )


def block_smode_decode(
    base_model,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    tau_mask: float = 0.7,
    tau_edit: float = 0.9,
    max_steps_per_block: int = 16,
    enable_mbe: bool = False,
    gen_length: Optional[int] = None,
    eos_early_stop: bool = False,
    stats: Optional[Dict[str, Any]] = None,
    suppress_eos: bool = False,
    eos_steady_passes: int = 0,
) -> torch.Tensor:
    """
    Block-wise Semi-Autoregressive S-Mode Decoding (LLaDA 2.1 paper §2).

    Generates text block-by-block (left-to-right). Within each block,
    parallel threshold decoding unmasks many tokens simultaneously.

    Args:
        base_model: frozen LLaDA model.
        prompt_ids: [B, P] prompt token ids.
        cfg: config dict with inference.block_length, inference.gen_length.
        tau_mask: confidence threshold for M2T unmasking.
        tau_edit: confidence threshold for T2T editing.
        max_steps_per_block: max diffusion steps per block.
        enable_mbe: if True, enable Multiple Block Editing (revisit prev blocks).
        suppress_eos: if True, mask EOS out of the per-step logits before the
            M2T threshold check.
        eos_steady_passes: when > 0, enables steady-state EOS stopping instead
            of the immediate ``_mask_after_first_eos`` behaviour.  After EOS
            appears anywhere in the response the loop continues; if EOS is
            still present (not overwritten by T2T editing) for
            ``eos_steady_passes`` consecutive diffusion steps, the sequence is
            marked finished.  A step that overwrites EOS resets the counter.
            Intended for any-order baselines where the model needs a few passes
            to reach a stable fixed point at the EOS position.  Requires
            ``eos_early_stop=True``.

    Returns:
        output_ids: [B, P + L_gen] generated sequence.
    """
    ic = cfg["inference"]
    L_gen = int(gen_length if gen_length is not None else ic["gen_length"])
    block_len = ic.get("block_length", 32)
    mask_id = cfg["base_model"]["mask_token_id"]

    B, P = prompt_ids.shape
    device = prompt_ids.device
    n_blocks = (L_gen + block_len - 1) // block_len
    eos_token_id = _resolve_eos_token_id(base_model) if (eos_early_stop or suppress_eos) else None
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    iterations = 0
    _suppress_eos_id = eos_token_id if suppress_eos else None
    # Steady-state EOS stopping: counts consecutive passes where EOS is present.
    _use_steady = eos_early_stop and eos_steady_passes > 0 and eos_token_id is not None
    eos_stable_count = torch.zeros(B, dtype=torch.long, device=device) if _use_steady else None

    # Start with prompt + all masks
    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    for blk_idx in range(n_blocks):
        if eos_early_stop and finished.all():
            break
        blk_start = P + blk_idx * block_len
        blk_end = min(P + (blk_idx + 1) * block_len, P + L_gen)
        blk_slice = slice(blk_start, blk_end)

        for step in range(max_steps_per_block):
            blk_tokens = y[:, blk_slice]
            active_rows = (~finished).unsqueeze(-1)
            mask_ind = (blk_tokens == mask_id) & active_rows

            if not mask_ind.any():
                break

            prefix_ids = y[:, :blk_end]
            # Only score the active prefix for this block; later masked blocks
            # should not bloat the diffusion frontier.
            if hasattr(base_model, 'forward_block_causal'):
                logits = base_model.forward_block_causal(
                    prefix_ids, block_length=block_len,
                )[:, blk_slice, :]
            else:
                logits = base_model.forward(prefix_ids)[:, blk_slice, :]
            iterations += 1
            if _suppress_eos_id is not None and 0 <= int(_suppress_eos_id) < int(logits.shape[-1]):
                # Forbid EOS as the M2T argmax under semi-any-order: under
                # wider-than-trained block-causal attention LLaDA2.1 places
                # EOS at response[0], which would truncate the visible output
                # to length zero in the eval summariser.
                logits = logits.clone()
                logits[..., int(_suppress_eos_id)] = torch.finfo(logits.dtype).min
            max_prob, max_tok = _max_prob_and_argmax(logits)

            # M2T: unmask confident positions
            unmask = mask_ind & (max_prob > tau_mask)
            blk_tokens = blk_tokens.clone()
            blk_tokens[unmask] = max_tok[unmask]

            # T2T: edit unmasked positions where model disagrees
            unmasked = (blk_tokens != mask_id) & active_rows
            disagree = (max_tok != blk_tokens) & unmasked
            confident = max_prob > tau_edit
            edit = disagree & confident
            blk_tokens[edit] = max_tok[edit]

            # Fallback: unmask most confident if nothing changed
            still_masked = (blk_tokens == mask_id) & active_rows
            nothing_happened = mask_ind.any(dim=-1) & ~unmask.any(dim=-1)
            for b in nothing_happened.nonzero(as_tuple=True)[0]:
                masked_pos = still_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_pos) > 0:
                    best = masked_pos[max_prob[b, masked_pos].argmax()]
                    blk_tokens[b, best] = max_tok[b, best]

            y[:, blk_slice] = blk_tokens

            if eos_early_stop and eos_token_id is not None:
                if _use_steady:
                    has_eos = (y[:, P:] == eos_token_id).any(dim=-1)
                    active = ~finished
                    eos_stable_count[active & has_eos] += 1
                    eos_stable_count[active & ~has_eos] = 0
                    finished = finished | (eos_stable_count >= eos_steady_passes)
                else:
                    blk_tokens = _mask_after_first_eos(
                        blk_tokens,
                        eos_token_id=eos_token_id,
                        mask_id=mask_id,
                    )
                    finished = finished | (blk_tokens == eos_token_id).any(dim=-1)
                    y[:, blk_slice] = blk_tokens

        remaining_mask = y[:, blk_slice] == mask_id
        if eos_early_stop:
            remaining_mask = remaining_mask & (~finished).unsqueeze(-1)
        if remaining_mask.any():
            prefix_ids = y[:, :blk_end]
            if hasattr(base_model, 'forward_block_causal'):
                final_logits = base_model.forward_block_causal(
                    prefix_ids, block_length=block_len,
                )[:, blk_slice, :]
            else:
                final_logits = base_model.forward(prefix_ids)[:, blk_slice, :]
            iterations += 1
            if _suppress_eos_id is not None and 0 <= int(_suppress_eos_id) < int(final_logits.shape[-1]):
                final_logits = final_logits.clone()
                final_logits[..., int(_suppress_eos_id)] = torch.finfo(final_logits.dtype).min
            _, final_tok = _max_prob_and_argmax(final_logits)
            completed = y[:, blk_slice].clone()
            completed[remaining_mask] = final_tok[remaining_mask]
            y[:, blk_slice] = completed

            if eos_early_stop and eos_token_id is not None:
                if _use_steady:
                    has_eos = (y[:, P:] == eos_token_id).any(dim=-1)
                    active = ~finished
                    eos_stable_count[active & has_eos] += 1
                    eos_stable_count[active & ~has_eos] = 0
                    finished = finished | (eos_stable_count >= eos_steady_passes)
                else:
                    completed = _mask_after_first_eos(
                        completed,
                        eos_token_id=eos_token_id,
                        mask_id=mask_id,
                    )
                    finished = finished | (completed == eos_token_id).any(dim=-1)
                    y[:, blk_slice] = completed

        # Optional: Multiple Block Editing — revisit previous blocks
        if enable_mbe and blk_idx > 0:
            prev_start = P
            prev_end = blk_start
            prev_slice = slice(prev_start, prev_end)

            prefix_ids = y[:, :blk_end]
            if hasattr(base_model, 'forward_block_causal'):
                logits = base_model.forward_block_causal(
                    prefix_ids, block_length=block_len,
                )[:, prev_slice, :]
            else:
                logits = base_model.forward(prefix_ids)[:, prev_slice, :]
            iterations += 1
            max_prob, max_tok = _max_prob_and_argmax(logits)

            prev_tokens = y[:, prev_slice].clone()
            disagree = (max_tok != prev_tokens) & (max_prob > tau_edit) & (~finished).unsqueeze(-1)
            prev_tokens[disagree] = max_tok[disagree]
            y[:, prev_slice] = prev_tokens

    if stats is not None:
        stats["iterations"] = int(stats.get("iterations", 0)) + iterations
    return y


def llada21_official_decode(
    base_model,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    mode: str = "speed",
    stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    LLaDA2.1 paper/model-card style threshold decoding.

    This path keeps the official semantics separate from AOAE-specific
    ablations such as ``disable_remask``:
      - Speed mode: threshold=0.5, editing_threshold=0.0
      - Quality mode: threshold=0.7, editing_threshold=0.5
      - block diffusion enabled by default with max_post_steps=16
    """
    settings = resolve_llada21_official_settings(cfg, mode=mode)

    if settings["use_block_diffusion"]:
        return block_smode_decode(
            base_model,
            prompt_ids,
            cfg,
            tau_mask=settings["threshold"],
            tau_edit=settings["editing_threshold"],
            max_steps_per_block=settings["max_post_steps"],
            enable_mbe=settings["enable_mbe"],
            gen_length=settings["gen_length"],
            eos_early_stop=settings["eos_early_stop"],
            stats=stats,
        )

    return confidence_threshold_decode(
        base_model,
        prompt_ids,
        cfg,
        tau_mask=settings["threshold"],
        tau_edit=settings["editing_threshold"],
        enable_t2t=True,
        stats=stats,
        gen_length=settings["gen_length"],
        eos_early_stop=settings["eos_early_stop"],
    )
