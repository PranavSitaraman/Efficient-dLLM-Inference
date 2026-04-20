"""
Speculative diffusion inference for AOAE.

The runtime separates two ideas that are easy to conflate:
  - K_spec: a transient drafter/verifier acceptance frontier. It is refreshed
    at verifier events and never treated as a persistent stability guarantee.
  - K_stable: a persistent policy-controlled stability cache. It survives
    across steps until an explicit remask/eviction.

Between verifier events the hard-routed auxiliary may take cheap draft steps.
Verifier events compare auxiliary and primary predictions, compose or correct
the draft, and optionally reset stale drafter cache state.
"""

import torch
import torch.nn.functional as F
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from .cache import SpeculativeCacheBookkeeper
from .models.composed_prediction import compose_prediction_dual
from .models.dual_model import DualModelWrapper, DualModelOutput
from .models.policy import apply_unmask_budget, call_policy
from .models.soft_mask import call_soft_mask
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
    total_stable_commits: int = 0
    total_stable_invalidations: int = 0
    draft_accepts: int = 0
    draft_rejects: int = 0
    agreement_observations: int = 0
    primary_steps: int = 0
    aux_only_steps: int = 0
    drafter_cache_resets: int = 0
    frontier_sizes: List[torch.Tensor] = field(default_factory=list)
    frontier_accept_counts: List[torch.Tensor] = field(default_factory=list)
    frontier_reject_counts: List[torch.Tensor] = field(default_factory=list)
    frontier_accept_rate: float = 0.0
    frontier_reject_rate: float = 0.0
    mean_frontier_size: float = 0.0
    aux_compute_units: Optional[torch.Tensor] = None
    verifier_compute_units: Optional[torch.Tensor] = None
    baseline_compute_units: Optional[torch.Tensor] = None
    effective_flops: Optional[torch.Tensor] = None
    access_metrics: Dict[str, float] = field(default_factory=dict)
    mean_boundary_depth: float = 0.0
    boundary_distribution: str = "{}"
    # ---- compute-aware speed bonus ----
    # Per-step fraction of positions in the transient draft frontier awaiting
    # verifier validation.
    spec_cached_fractions: List[torch.Tensor] = field(default_factory=list)
    # K_stable fraction: per-step fraction of positions in the persistent κ_t cache.
    stable_cached_fractions: List[torch.Tensor] = field(default_factory=list)
    # Combined K_spec ∪ K_stable diagnostic occupancy.
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


class DraftFrontier:
    """Transient K_spec frontier of drafted-but-not-yet-verified tokens.

    The frontier stores the actual token proposal made by the drafter. It may
    accumulate over several cheap auxiliary microsteps and is consumed exactly
    when the primary verifier runs. This is deliberately separate from
    K_stable: K_spec is about *verification debt*, not persistent KV reuse.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.B = batch_size
        self.L = seq_len
        self.device = device
        self.mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        self.token_ids = torch.full(
            (batch_size, seq_len), -1, dtype=torch.long, device=device
        )
        self.scores = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        self.age = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

    def add(
        self,
        drafted_mask: torch.Tensor,
        response_tokens: torch.Tensor,
        draft_logits: Optional[torch.Tensor] = None,
    ) -> None:
        drafted = drafted_mask.bool()
        if not drafted.any():
            return
        self.mask = self.mask | drafted
        self.token_ids = torch.where(drafted, response_tokens.long(), self.token_ids)
        if draft_logits is not None:
            probs = F.softmax(draft_logits.float(), dim=-1)
            safe_ids = response_tokens.long().clamp(min=0, max=probs.shape[-1] - 1)
            token_scores = torch.gather(probs, dim=-1, index=safe_ids.unsqueeze(-1)).squeeze(-1)
            self.scores = torch.where(drafted, token_scores.detach(), self.scores)
        self.age = torch.where(drafted, torch.zeros_like(self.age), self.age)

    def step_age(self) -> None:
        self.age = self.age + self.mask.float()

    def clear(self) -> None:
        self.mask.zero_()
        self.token_ids.fill_(-1)
        self.scores.zero_()
        self.age.zero_()

    def numel_per_batch(self) -> torch.Tensor:
        return self.mask.float().sum(dim=-1)

    def fraction_per_batch(self) -> torch.Tensor:
        return self.mask.float().mean(dim=-1)

    def validate(
        self,
        primary_logits: torch.Tensor,
        cfg: dict,
        *,
        prism_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (accepted, rejected) masks for the current frontier."""
        frontier = self.mask
        accepted = torch.zeros_like(frontier)
        rejected = torch.zeros_like(frontier)
        if not frontier.any():
            return accepted, rejected

        verifier_cfg = cfg.get("inference", {}).get("verifier", {})
        mode = str(verifier_cfg.get("acceptance_mode", "argmax_match")).lower()
        threshold = float(verifier_cfg.get("primary_prob_threshold", 0.5))
        prism_threshold = float(verifier_cfg.get("prism_accept_threshold", 0.5))

        draft_tokens = self.token_ids.clamp(min=0, max=primary_logits.shape[-1] - 1)
        primary_argmax = primary_logits.argmax(dim=-1)
        argmax_accept = primary_argmax.eq(draft_tokens)

        if mode in ("argmax_match", "argmax", "token_match"):
            accepted = frontier & argmax_accept
        elif mode in ("prob_threshold", "primary_prob", "token_probability"):
            probs = F.softmax(primary_logits.float(), dim=-1)
            draft_prob = torch.gather(probs, dim=-1, index=draft_tokens.unsqueeze(-1)).squeeze(-1)
            accepted = frontier & (draft_prob >= threshold)
        elif mode in ("argmax_and_prob", "argmax_prob"):
            probs = F.softmax(primary_logits.float(), dim=-1)
            draft_prob = torch.gather(probs, dim=-1, index=draft_tokens.unsqueeze(-1)).squeeze(-1)
            accepted = frontier & argmax_accept & (draft_prob >= threshold)
        elif mode in ("prism_gate", "prism_gated"):
            if prism_scores is None:
                accepted = frontier & argmax_accept
            else:
                accepted = frontier & argmax_accept & (prism_scores >= prism_threshold)
        else:
            raise ValueError(
                f"Unsupported verifier.acceptance_mode={mode!r}. "
                "Use argmax_match, prob_threshold, argmax_and_prob, or prism_gate."
            )

        rejected = frontier & ~accepted
        return accepted, rejected


def _verifier_schedule(ic: dict) -> Dict[str, Any]:
    """Resolve the verifier scheduler, keeping primary_every_n as an alias."""
    raw = dict(ic.get("verifier_schedule", {}) or {})
    if "mode" not in raw:
        if "primary_every_n" in ic:
            raw["mode"] = "step_interval"
            raw["step_interval"] = int(ic.get("primary_every_n", 1))
        else:
            raw["mode"] = "candidate_budget"
    mode = str(raw.get("mode", "candidate_budget")).lower()
    if mode in ("primary_every_n", "interval"):
        mode = "step_interval"
    raw["mode"] = mode
    raw["draft_token_budget"] = int(raw.get("draft_token_budget", 12))
    raw["min_draft_microsteps"] = int(raw.get("min_draft_microsteps", 1))
    raw["max_draft_microsteps"] = int(raw.get("max_draft_microsteps", 4))
    raw["force_first_last"] = bool(raw.get("force_first_last", ic.get("force_primary_first_last", True)))
    raw["step_interval"] = max(1, int(raw.get("step_interval", ic.get("primary_every_n", 1))))
    return raw


def _should_run_verifier(
    *,
    schedule: Dict[str, Any],
    step_idx: int,
    t: int,
    frontier: DraftFrontier,
    draft_microsteps_since_verify: int,
    force_next: bool,
) -> bool:
    if force_next:
        return True
    if schedule.get("force_first_last", True) and (step_idx == 0 or t == 1):
        return True
    mode = schedule.get("mode", "candidate_budget")
    if mode == "step_interval":
        every_n = max(1, int(schedule.get("step_interval", 1)))
        return every_n <= 1 or (step_idx > 0 and step_idx % every_n == 0)
    if mode != "candidate_budget":
        raise ValueError(f"Unsupported verifier_schedule.mode={mode!r}")

    if draft_microsteps_since_verify < int(schedule.get("min_draft_microsteps", 1)):
        return False
    if draft_microsteps_since_verify >= int(schedule.get("max_draft_microsteps", 4)):
        return True
    budget = int(schedule.get("draft_token_budget", 12))
    if budget <= 0:
        return True
    return bool((frontier.numel_per_batch() >= budget).any().item())


def _as_head_set(value: Any) -> Optional[set]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in value if str(p).strip()]
    return {p for p in parts}


def _apply_frozen_action_heads(
    actions: Dict[str, torch.Tensor],
    *,
    confidence: torch.Tensor,
    mask_ind: torch.Tensor,
    cfg: dict,
) -> Dict[str, torch.Tensor]:
    """Replace frozen u/r heads with deterministic runtime decisions.

    Canonical GRPO isolates the new cache/access policy by excluding u/r from
    train_heads. In that setting unmasking follows the drafter confidence
    schedule and remasking is reserved for the authoritative verifier.
    """
    gc = cfg.get("grpo", {})
    train_heads = _as_head_set(gc.get("train_heads"))
    if train_heads is None:
        return actions

    out = dict(actions)
    drafter_cfg = cfg.get("inference", {}).get("drafter", {})
    threshold = float(drafter_cfg.get("confidence_threshold", 0.7))
    if "unmask" not in train_heads and "u" not in train_heads and "u_t" not in train_heads:
        out["u_t"] = ((confidence >= threshold) & mask_ind.bool()).float()
    if "remask" not in train_heads and "r" not in train_heads and "r_t" not in train_heads:
        out["r_t"] = torch.zeros_like(out["r_t"])
    if "cache" not in train_heads and "kappa" not in train_heads and "kappa_t" not in train_heads:
        out["kappa_t"] = torch.zeros_like(out["kappa_t"])
    if "access" not in train_heads and "q" not in train_heads and "q_t" not in train_heads:
        out["q_t"] = torch.zeros_like(out["q_t"])
    return out


def _fresh_primary_agreement(
    agreement: torch.Tensor,
    primary_fresh_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep only agreement positions observed by a fresh primary verifier pass."""
    if agreement.shape != primary_fresh_mask.shape:
        raise ValueError(
            "agreement and primary_fresh_mask must have identical shapes, "
            f"got {tuple(agreement.shape)} and {tuple(primary_fresh_mask.shape)}"
        )
    return agreement.bool() & primary_fresh_mask.bool()


def _on_off(flag: bool) -> str:
    return "on" if bool(flag) else "off"



def _maybe_log_speculative_rollout_config(
    *,
    cfg: dict,
    prism_adapter,
    track_kv_enabled: bool,
    use_cache: bool,
    use_fallback: bool,
    disable_remask: bool,
    use_prefix_kv_cache: bool,
    aux_cache_enabled: bool,
    primary_cache_enabled: bool,
    need_hidden: bool,
    need_all_hidden: bool,
    primary_every_n: int,
    primary_agree_threshold: float,
    force_primary_endpoints: bool,
    aux_cache_reset_threshold: float,
    gamma: float,
) -> None:
    if not bool(cfg.get("analysis", {}).get("log_speculative_config", False)):
        return

    schedule = cfg.get("inference", {}).get("speculative_schedule", "aoae")
    verifier_schedule = _verifier_schedule(cfg.get("inference", {}))
    if need_hidden or need_all_hidden:
        verifier_mode = "full_hidden_with_aux_cache" if aux_cache_enabled else "full_hidden"
    elif use_prefix_kv_cache and primary_cache_enabled:
        verifier_mode = "prefix_cache_replace"
    else:
        verifier_mode = "full_dual_no_cache"
    print(
        "[Speculative] "
        f"schedule={schedule} "
        f"prism={_on_off(prism_adapter is not None)} "
        f"kv_tracking={_on_off(track_kv_enabled)} "
        f"cache={_on_off(use_cache)} "
        f"aux_cache={_on_off(aux_cache_enabled)} "
        f"primary_hidden={_on_off(need_hidden)} "
        f"primary_all_hidden={_on_off(need_all_hidden)} "
        f"primary_cache_fastpath={_on_off(primary_cache_enabled and use_prefix_kv_cache)} "
        f"verifier_mode={verifier_mode} "
        f"verifier_schedule={verifier_schedule.get('mode')} "
        f"draft_token_budget={int(verifier_schedule.get('draft_token_budget', 0))} "
        f"max_draft_microsteps={int(verifier_schedule.get('max_draft_microsteps', 0))} "
        f"primary_every_n={primary_every_n} "
        f"primary_agree_threshold={primary_agree_threshold:.3f} "
        f"force_primary_endpoints={_on_off(force_primary_endpoints)} "
        f"aux_cache_reset_threshold={aux_cache_reset_threshold:.3f} "
        f"gamma={float(gamma):.3f} "
        f"remask={_on_off(not disable_remask)} "
        f"fallback={_on_off(use_fallback)}"
    )


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
    collect_stats: bool = False,
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
        collect_stats: if True, return lightweight trajectory metrics during
            evaluation without storing full GRPO rollout tensors.

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

    trajectory = SpeculativeTrajectory() if (record_trajectory or collect_stats) else None
    agreement_rates = []
    reuse_state = None
    pos_state = init_positional_state(B, L_gen, device)
    draft_frontier = DraftFrontier(B, L_gen, device)

    # H_t from the previous step: used to compute per-position drift for the
    # cache quality F1 signal.  Mirrors the same variable in aoae_inference().
    _prev_H_t: Optional[torch.Tensor] = None

    # --- KV cache state ---
    schedule_cfg = _verifier_schedule(ic)
    use_prefix_kv_cache = cfg["cache"].get("prefix_kv_cache", False) and use_cache
    aux_past_kv = None
    pri_past_kv = None
    _prefix_cache_initialized = False

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
        if trajectory is None:
            trajectory = SpeculativeTrajectory()

    # Hidden-state requirements are rollout-global: they depend on the verifier
    # wiring for this call, not on the current diffusion step.
    _need_all_hidden = _track_kv
    _need_hidden = (prism_adapter is not None) and not _need_all_hidden
    _primary_cache_enabled = not (_need_hidden or _need_all_hidden)
    _aux_cache_enabled = use_prefix_kv_cache

    def _run_auxiliary_resp(current_y: torch.Tensor) -> torch.Tensor:
        nonlocal aux_past_kv, _prefix_cache_initialized

        if _aux_cache_enabled and _prefix_cache_initialized:
            aux_logits, aux_past_kv = dual_model.auxiliary_forward_replace_with_cache(
                current_y, resp_slice, aux_past_kv,
            )
            return aux_logits

        if _aux_cache_enabled:
            aux_full, aux_past_kv = dual_model.auxiliary_forward_with_cache(current_y)
            _prefix_cache_initialized = True
            return aux_full[:, resp_slice, :]

        return dual_model.auxiliary_forward(current_y)[:, resp_slice, :]

    def _run_primary_full_resp(
        current_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        if _need_all_hidden:
            pri_full, pri_hidden_states, _, _ = dual_model.primary_forward_with_diagnostics(current_y)
            pri_hidden = pri_hidden_states[-1]
            return (
                pri_full[:, resp_slice, :],
                pri_hidden[:, resp_slice, :],
                [h[:, resp_slice, :] for h in pri_hidden_states],
            )

        if _need_hidden:
            pri_full, pri_hidden = dual_model.primary_forward_with_hidden(current_y)
            return pri_full[:, resp_slice, :], pri_hidden[:, resp_slice, :], None

        pri_full = dual_model.primary_forward(current_y)
        return pri_full[:, resp_slice, :], None, None

    primary_every_n = int(schedule_cfg.get("step_interval", ic.get("primary_every_n", 1)))
    primary_agree_threshold = float(ic.get("primary_agree_threshold", 0.0))
    force_primary_endpoints = bool(schedule_cfg.get("force_first_last", True))
    aux_cache_reset_threshold = float(ic.get("aux_cache_reset_threshold", 1.1))
    drafter_cfg = ic.get("drafter", {})
    aux_compute_ratio = float(drafter_cfg.get("aux_compute_ratio", ic.get("aux_compute_ratio", 0.35)))
    _maybe_log_speculative_rollout_config(
        cfg=cfg,
        prism_adapter=prism_adapter,
        track_kv_enabled=_track_kv,
        use_cache=use_cache,
        use_fallback=use_fallback,
        disable_remask=disable_remask,
        use_prefix_kv_cache=use_prefix_kv_cache,
        aux_cache_enabled=_aux_cache_enabled,
        primary_cache_enabled=_primary_cache_enabled,
        need_hidden=_need_hidden,
        need_all_hidden=_need_all_hidden,
        primary_every_n=primary_every_n,
        primary_agree_threshold=primary_agree_threshold,
        force_primary_endpoints=force_primary_endpoints,
        aux_cache_reset_threshold=aux_cache_reset_threshold,
        gamma=gamma,
    )
    _ema_agreement = 1.0
    _primary_steps = 0
    _aux_only_steps = 0
    _agreement_sum = 0.0
    _agreement_obs = 0
    _safe_reuse_sum = 0.0
    _safe_reuse_obs = 0
    _draft_accepts = 0
    _draft_rejects = 0
    _stable_commits = 0
    _stable_invalidations = 0
    _drafter_cache_resets = 0
    _draft_microsteps_since_verify = 0
    _force_next_verifier = False
    _aux_units = torch.zeros(B, device=device)
    _verifier_units = torch.zeros(B, device=device)
    _baseline_units = torch.zeros(B, device=device)

    # --- Main speculative diffusion loop ---
    for t in range(T, 0, -1):
        step_frac = t / T

        resp_tokens = y[:, resp_slice]             # [B, L_gen]
        mask_ind = (resp_tokens == mask_id)        # [B, L_gen]

        if not mask_ind.any() and not draft_frontier.mask.any():
            if trajectory is not None:
                trajectory.completion_step = torch.full((B,), T - t, device=device)
            break

        # === Phase 0 + 0b: Drafter or verifier forward ===
        step_idx = T - t
        run_primary = _should_run_verifier(
            schedule=schedule_cfg,
            step_idx=step_idx,
            t=t,
            frontier=draft_frontier,
            draft_microsteps_since_verify=_draft_microsteps_since_verify,
            force_next=(_force_next_verifier or _ema_agreement < primary_agree_threshold),
        )
        _force_next_verifier = False
        _baseline_units += 1.0
        primary_fresh_mask = torch.zeros(B, L_gen, dtype=torch.bool, device=device)
        frontier_accept_mask = torch.zeros(B, L_gen, dtype=torch.bool, device=device)
        frontier_reject_mask = torch.zeros(B, L_gen, dtype=torch.bool, device=device)

        if not run_primary:
            aux_logits = _run_auxiliary_resp(y)
            resp_logits = aux_logits
            # No verifier observation happened on this cheap draft step, so no
            # position is considered accepted for composition or K_spec.
            agreement = torch.zeros(B, L_gen, dtype=torch.bool, device=device)
            _pri_hidden_for_prism = None
            _pri_hidden_states_for_tracker = None
            _aux_only_steps += 1

        elif _need_hidden or _need_all_hidden:
            # --- Mixed verifier path: cached auxiliary + full hidden-state primary ---
            # PRISM / diagnostics need hidden states from a full primary pass, but
            # the auxiliary can still reuse its prefix cache to keep draft steps and
            # cache-reset accounting aligned with the speculative path.
            aux_logits = _run_auxiliary_resp(y)
            resp_logits, _pri_hidden_for_prism, _pri_hidden_states_for_tracker = _run_primary_full_resp(y)
            primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
            _primary_steps += 1

        elif use_prefix_kv_cache and _primary_cache_enabled:
            # --- Prefix-only path: no K_spec skip, but prompt KVs cached ---
            if not hasattr(dual_model, "primary_forward_with_cache") or not hasattr(dual_model, "primary_forward_replace_with_cache"):
                aux_logits = _run_auxiliary_resp(y)
                resp_logits, _pri_hidden_for_prism, _pri_hidden_states_for_tracker = _run_primary_full_resp(y)
                primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
                _primary_steps += 1
            elif not _prefix_cache_initialized:
                _aux_full, aux_past_kv = dual_model.auxiliary_forward_with_cache(y)
                _pri_full, pri_past_kv = dual_model.primary_forward_with_cache(y)
                aux_logits = _aux_full[:, resp_slice, :]
                resp_logits = _pri_full[:, resp_slice, :]
                _prefix_cache_initialized = True
                primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
                _pri_hidden_for_prism = None
                _pri_hidden_states_for_tracker = None
                _primary_steps += 1
            else:
                aux_logits, aux_past_kv = dual_model.auxiliary_forward_replace_with_cache(
                    y, resp_slice, aux_past_kv,
                )
                resp_logits, pri_past_kv = dual_model.primary_forward_replace_with_cache(
                    y, resp_slice, pri_past_kv,
                )
                primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
                _pri_hidden_for_prism = None
                _pri_hidden_states_for_tracker = None
                _primary_steps += 1
        else:
            # --- Full dual-model forward path: logits only, no cache reuse ---
            dual_out = dual_model.dual_forward_resp(
                y, resp_slice, need_hidden=_need_hidden, need_all_hidden=_need_all_hidden,
            )
            resp_logits = dual_out.primary_logits      # [B, L_gen, V]
            aux_logits = dual_out.auxiliary_logits      # [B, L_gen, V]
            _pri_hidden_for_prism = dual_out.primary_hidden
            _pri_hidden_states_for_tracker = dual_out.primary_hidden_states
            primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
            _primary_steps += 1

        if run_primary:
            _verifier_units += 1.0
            _aux_units += aux_compute_ratio
        else:
            _aux_units += aux_compute_ratio

        # --- PRISM quality scores ---
        q_scores = None
        verifier_cfg = ic.get("verifier", {})
        _use_prism_score = bool(verifier_cfg.get("use_prism_score", prism_adapter is not None))
        if _use_prism_score and prism_adapter is not None and _pri_hidden_for_prism is not None:
            with torch.no_grad():
                q_scores = prism_adapter(_pri_hidden_for_prism.float())

        if run_primary:
            # The primary verifier consumes the accumulated frontier before the
            # policy acts. Acceptance is authoritative and local to the stored
            # drafted token, not an advisory feature that the policy may ignore.
            frontier_before = draft_frontier.mask.clone()
            if frontier_before.any():
                frontier_accept_mask, frontier_reject_mask = draft_frontier.validate(
                    resp_logits,
                    cfg,
                    prism_scores=q_scores,
                )
                accept_counts = frontier_accept_mask.float().sum(dim=-1)
                reject_counts = frontier_reject_mask.float().sum(dim=-1)
                _draft_accepts += int(accept_counts.sum().item())
                _draft_rejects += int(reject_counts.sum().item())
                if trajectory is not None:
                    trajectory.frontier_sizes.append(frontier_before.float().sum(dim=-1).detach())
                    trajectory.frontier_accept_counts.append(accept_counts.detach())
                    trajectory.frontier_reject_counts.append(reject_counts.detach())

                if frontier_reject_mask.any():
                    resp_tokens = y[:, resp_slice].clone()
                    resp_tokens[frontier_reject_mask] = mask_id
                    y = y.clone()
                    y[:, resp_slice] = resp_tokens
                    mask_ind = (resp_tokens == mask_id)
                    if cache_mgr is not None:
                        _stable_invalidations += int(
                            (cache_mgr.stable.get_cached_mask() & frontier_reject_mask).sum().item()
                        )
                        cache_mgr.invalidate(frontier_reject_mask.float())

                    # Policy state must be computed on the corrected sequence.
                    # A rejected token has been removed, so rerun the verifier
                    # on the remasked state rather than using logits conditioned
                    # on the stale draft.
                    aux_logits = _run_auxiliary_resp(y)
                    _aux_units += aux_compute_ratio
                    resp_logits, _pri_hidden_for_prism, _pri_hidden_states_for_tracker = _run_primary_full_resp(y)
                    _verifier_units += 1.0
                    primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
                    if _use_prism_score and prism_adapter is not None and _pri_hidden_for_prism is not None:
                        with torch.no_grad():
                            q_scores = prism_adapter(_pri_hidden_for_prism.float())

                rejected_total = float(frontier_reject_mask.float().sum().item())
                frontier_total = float(frontier_before.float().sum().item())
                rejection_rate = rejected_total / max(frontier_total, 1.0)
                accept_rate = 1.0 - rejection_rate
                _ema_agreement = 0.8 * _ema_agreement + 0.2 * accept_rate
                if rejection_rate > aux_cache_reset_threshold:
                    aux_past_kv = None
                    pri_past_kv = None
                    _prefix_cache_initialized = False
                    _drafter_cache_resets += 1
                    _force_next_verifier = True

            # Keep a separate raw agreement diagnostic for safe-reuse analysis.
            raw_agreement, reuse_state, _ = compute_reuse_signal(
                resp_logits, aux_logits, cfg, state=reuse_state
            )
            raw_agreement = _fresh_primary_agreement(raw_agreement, primary_fresh_mask)
            active_for_agreement = (mask_ind.bool() | frontier_before.bool()) & primary_fresh_mask
            if active_for_agreement.any():
                _agreement_sum += float(raw_agreement[active_for_agreement].float().sum().item())
                _agreement_obs += int(active_for_agreement.sum().item())
                _safe_reuse_sum += float(raw_agreement[active_for_agreement].float().sum().item())
                _safe_reuse_obs += int(active_for_agreement.sum().item())
            if not frontier_before.any() and primary_fresh_mask.any():
                verifier_agreement = float(raw_agreement[primary_fresh_mask].float().mean().item())
                _ema_agreement = 0.8 * _ema_agreement + 0.2 * verifier_agreement

            agreement = frontier_accept_mask
            draft_frontier.clear()
            _draft_microsteps_since_verify = 0
        else:
            _draft_microsteps_since_verify += 1
        agreement_rates.append(float(agreement.float().mean().item()))

        # --- Construct soft-masked state from PRIMARY logits ---
        H_t, confidence, entropy, weighted_embeds = call_soft_mask(
            soft_mask_module,
            resp_logits, mask_ind, step_frac, return_weighted=True
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
        if record_trajectory and trajectory is not None and _prev_H_t is not None and cache_mgr is not None:
            # Evaluate the learned persistent stability cache, not the
            # transient K_spec agreement frontier.
            _cached_mask = cache_mgr.stable.get_cached_mask()    # [B, L_gen] bool
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
        actions = _apply_frozen_action_heads(
            actions,
            confidence=confidence,
            mask_ind=mask_ind,
            cfg=cfg,
        )
        actions = apply_unmask_budget(actions, policy_out, mask_ind, cfg)
        u_t = actions["u_t"]
        r_t = actions["r_t"]
        kappa_t = actions["kappa_t"]
        if not run_primary:
            # Aux-only draft microsteps are intentionally unverified.  They may
            # propose tokens, but they cannot safely remask or make persistent
            # K_stable commitments until the verifier observes the state.
            r_t = torch.zeros_like(r_t)
            kappa_t = torch.zeros_like(kappa_t)
            actions = {**actions, "r_t": r_t, "kappa_t": kappa_t}
        elif disable_remask:
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
        if record_trajectory and trajectory is not None:
            include_heads = _as_head_set(
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
        elif trajectory is not None:
            # Lightweight eval stats: enough for access-pattern summaries
            # without storing logits, H_t, log-probs, or policy outputs.
            trajectory.access_exec_list.append(q_exec.detach())
            trajectory.access_mandatory_list.append(q_mandatory.detach())
            if "ell_t" in actions:
                trajectory.boundary_actions.append(actions["ell_t"].detach())

        # --- Count cache thrashing ---
        thrash = None
        if cache_mgr is not None:
            thrash = cache_mgr.count_thrash(r_t)
            _stable_invalidations += int(thrash.sum().item())
        if record_trajectory and trajectory is not None and thrash is not None:
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

        drafted_positions = unmask_positions | fallback_positions
        if not run_primary and drafted_positions.any():
            draft_frontier.add(drafted_positions, resp_tokens, aux_logits)

        # ====== Phase 3a: Speculative Frontier / Accept State (K_spec) ======
        # K_spec is the transient frontier of unverified drafted positions.  It
        # accumulates across auxiliary microsteps and is cleared only above when
        # the primary verifier consumes it.
        if cache_mgr is not None:
            cache_mgr.step_spec(draft_frontier.mask)

        # ====== Phase 3b: Stable Cache (K_stable) ======
        # Positions the κ_t head predicts will remain stable across future steps.
        # Accumulated persistently; evicted by r_t (Phase 1 already called invalidate).
        if cache_mgr is not None:
            stable_commit_mask = (
                kappa_t
                * (1.0 - r_t)
                * q_exec
                * (resp_tokens != mask_id).float()
                * (1.0 - draft_frontier.mask.float())
            )
            stable_before = cache_mgr.stable.get_cached_mask().clone()
            cache_mgr.step_stable(stable_commit_mask, r_t)
            newly_stable = stable_commit_mask.bool() & ~stable_before & ~r_t.bool()
            _stable_commits += int(newly_stable.sum().item())

        # --- Record cached fractions after commit (compute-aware speed bonus) ---
        # spec_cached_fractions: legacy metric name for K_spec frontier occupancy.
        # stable_cached_fractions: learned persistent cache occupancy.
        # cached_fractions:        K_spec ∪ K_stable diagnostic occupancy.
        if trajectory is not None and cache_mgr is not None:
            trajectory.spec_cached_fractions.append(cache_mgr.spec_cached_fraction().detach())
            trajectory.stable_cached_fractions.append(cache_mgr.stable_cached_fraction().detach())
            trajectory.cached_fractions.append(cache_mgr.cached_fraction().detach())

        draft_frontier.step_age()

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
        if _agreement_obs > 0:
            trajectory.mean_agreement_rate = _agreement_sum / max(_agreement_obs, 1)
        elif agreement_rates:
            trajectory.mean_agreement_rate = sum(agreement_rates) / len(agreement_rates)
        trajectory.total_stable_commits = _stable_commits
        trajectory.total_stable_invalidations = _stable_invalidations
        trajectory.total_cache_hits = _stable_commits
        trajectory.total_cache_misses = _stable_invalidations
        trajectory.draft_accepts = _draft_accepts
        trajectory.draft_rejects = _draft_rejects
        trajectory.agreement_observations = _agreement_obs
        trajectory.primary_steps = _primary_steps
        trajectory.aux_only_steps = _aux_only_steps
        trajectory.drafter_cache_resets = _drafter_cache_resets
        frontier_total = float(_draft_accepts + _draft_rejects)
        trajectory.frontier_accept_rate = float(_draft_accepts / max(frontier_total, 1.0))
        trajectory.frontier_reject_rate = float(_draft_rejects / max(frontier_total, 1.0))
        if trajectory.frontier_sizes:
            trajectory.mean_frontier_size = float(
                torch.stack([x.float().mean() for x in trajectory.frontier_sizes]).mean().item()
            )
        else:
            trajectory.mean_frontier_size = 0.0
        trajectory.aux_compute_units = _aux_units.detach()
        trajectory.verifier_compute_units = _verifier_units.detach()
        trajectory.baseline_compute_units = _baseline_units.detach().clamp(min=1.0)
        trajectory.effective_flops = (
            (_aux_units + _verifier_units) / _baseline_units.clamp(min=1.0)
        ).detach()
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
