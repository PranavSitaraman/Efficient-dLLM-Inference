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

import json

from .cache import SpeculativeCacheBookkeeper
from .experiment_utils import parse_head_set
from .models.composed_prediction import compose_prediction_dual
from .models.dual_model import DualModelWrapper, DualModelOutput
from .models.policy import (
    active_block_window,
    apply_unmask_budget,
    call_policy,
    call_policy_block,
)
from .models.soft_mask import call_soft_mask
from .agreement_signals import compute_reuse_signal
from .positional_cache import (
    init_positional_state,
    get_policy_positional_features,
    build_access_set,
    update_positional_state,
    compute_next_h_access_metrics,
    compute_next_h_access_metrics_per_sample,
    summarize_access_diagnostics,
)


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
    confidence_list: List[torch.Tensor] = field(default_factory=list)
    run_primary_list: List[bool] = field(default_factory=list)
    frontier_before_list: List[torch.BoolTensor] = field(default_factory=list)
    frontier_accept_mask_list: List[torch.BoolTensor] = field(default_factory=list)
    frontier_reject_mask_list: List[torch.BoolTensor] = field(default_factory=list)
    quality_scores_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    agreement_list: List[torch.Tensor] = field(default_factory=list)
    age_feature_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    last_action_feature_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    access_exec_list: List[torch.Tensor] = field(default_factory=list)
    access_mandatory_list: List[torch.Tensor] = field(default_factory=list)
    access_diag_list: List[Dict[str, float]] = field(default_factory=list)
    changed_list: List[torch.Tensor] = field(default_factory=list)
    boundary_actions: List[torch.Tensor] = field(default_factory=list)
    step_fracs: List[float] = field(default_factory=list)
    # Block window per step (start, end) in response coords; populated only
    # when policy.block_wise.enabled. Used by the per-block GRPO reward to
    # attribute thrash/cache_F1/access_F1 to the active block.
    block_windows: List[Tuple[int, int]] = field(default_factory=list)
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
    access_metric_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
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
        # Python-side maximum count over the batch.  Tracked on CPU so the
        # per-step verifier scheduler does not need a GPU->CPU sync.  Updated
        # on add()/clear() and is a strict upper bound on the live max(); the
        # verifier still re-counts authoritatively when it fires.
        self._max_count = 0

    def add(
        self,
        drafted_mask: torch.Tensor,
        response_tokens: torch.Tensor,
        draft_logits: Optional[torch.Tensor] = None,
    ) -> None:
        drafted = drafted_mask.bool()
        if not drafted.any():
            return
        new_mask = self.mask | drafted
        # Best-effort upper bound on the per-batch max count without forcing a
        # CPU sync: if any new positions were added, increment by the maximum
        # number of newly drafted positions in this batch (an overestimate is
        # safe for scheduling because it only triggers earlier verifier calls).
        added_per_row = (drafted & ~self.mask).sum(dim=-1)
        self._max_count = self._max_count + int(added_per_row.max().item())
        self.mask = new_mask
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
        self._max_count = 0

    def numel_per_batch(self) -> torch.Tensor:
        return self.mask.float().sum(dim=-1)

    def fraction_per_batch(self) -> torch.Tensor:
        return self.mask.float().mean(dim=-1)

    def max_count_cpu(self) -> int:
        """Python-side upper bound on max(numel_per_batch); avoids GPU sync."""
        return int(self._max_count)

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
    """Resolve the verifier scheduler."""
    if "primary_every_n" in ic:
        raise ValueError(
            "inference.primary_every_n was removed. Use "
            "inference.verifier_schedule.{mode: step_interval, step_interval: N} instead."
        )
    raw = dict(ic.get("verifier_schedule", {}) or {})
    raw.setdefault("mode", "candidate_budget")
    mode = str(raw["mode"]).lower()
    if mode == "interval":
        mode = "step_interval"
    if mode not in {"candidate_budget", "step_interval"}:
        raise ValueError(
            f"Unsupported verifier_schedule.mode={mode!r}. "
            "Use 'candidate_budget' or 'step_interval'."
        )
    raw["mode"] = mode
    raw["draft_token_budget"] = int(raw.get("draft_token_budget", 12))
    raw["min_draft_microsteps"] = int(raw.get("min_draft_microsteps", 1))
    raw["max_draft_microsteps"] = int(raw.get("max_draft_microsteps", 4))
    raw["force_first_last"] = bool(raw.get("force_first_last", True))
    raw["step_interval"] = max(1, int(raw.get("step_interval", 1)))
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
    # Python-side counter tracked on add()/clear() — avoids per-step GPU->CPU sync.
    return frontier.max_count_cpu() >= budget


def _scope_active(scope: str, run_primary: bool) -> bool:
    """Return True if a head trained under ``scope`` should be active on this microstep.

    ``scope`` is one of {"drafter", "verifier", "both"}. ``run_primary`` is True
    on verifier microsteps, False on aux microsteps. ``"drafter"`` activates
    only on aux microsteps; ``"verifier"`` only on primary microsteps; ``"both"``
    always active. Used to keep the trained policy responsibilities aligned
    with the speculative-decoding role split (u_t learned for drafter, r_t for
    verifier) without changing the loop's microstep cadence.
    """
    s = str(scope).strip().lower()
    if s == "both":
        return True
    if s == "drafter":
        return not run_primary
    if s == "verifier":
        return run_primary
    raise ValueError(f"Unsupported policy scope {scope!r}; expected drafter|verifier|both.")


def _apply_frozen_action_heads(
    actions: Dict[str, torch.Tensor],
    *,
    confidence: torch.Tensor,
    mask_ind: torch.Tensor,
    cfg: dict,
    run_primary: bool,
) -> Dict[str, torch.Tensor]:
    """Replace frozen u/r heads with deterministic runtime decisions.

    Two layers of gating:
    1. ``train_heads``: which heads are trained at all. Untrained heads are
       overwritten with deterministic fallbacks (threshold rule for u_t,
       zeros for r_t / kappa_t / q_t).
    2. ``unmask_scope`` / ``remask_scope``: which microstep type the trained
       head is active on. A trained head outside its scope is also overwritten
       with the deterministic fallback. Defaults: unmask=drafter, remask=verifier.

    This lets us train u_t for the drafter (aux microsteps) and r_t for the
    verifier (primary microsteps) while leaving the verifier's u_t and the
    drafter's r_t at their canonical deterministic values.
    """
    gc = cfg.get("grpo", {})
    train_heads = parse_head_set(gc.get("train_heads"))

    out = dict(actions)
    drafter_cfg = cfg.get("inference", {}).get("drafter", {})
    threshold = float(drafter_cfg.get("confidence_threshold", 0.7))

    if train_heads is None:
        # Eval-time fallthrough: no GRPO config → keep all sampled actions.
        return out

    u_trained = any(h in train_heads for h in ("unmask", "u", "u_t"))
    r_trained = any(h in train_heads for h in ("remask", "r", "r_t"))
    kappa_trained = any(h in train_heads for h in ("cache", "kappa", "kappa_t"))
    q_trained = any(h in train_heads for h in ("access", "q", "q_t"))

    u_scope = gc.get("unmask_scope", "drafter")
    r_scope = gc.get("remask_scope", "verifier")

    u_policy_active = u_trained and _scope_active(u_scope, run_primary)
    r_policy_active = r_trained and _scope_active(r_scope, run_primary)

    if not u_policy_active:
        out["u_t"] = ((confidence >= threshold) & mask_ind.bool()).float()
    if not r_policy_active:
        out["r_t"] = torch.zeros_like(out["r_t"])
    if not kappa_trained:
        out["kappa_t"] = torch.zeros_like(out["kappa_t"])
    if not q_trained:
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


def _confidence_and_argmax_fast(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cheap confidence + argmax without materialising a [B, L, V] softmax.

    The full ``soft_mask`` path runs ``log_softmax`` over the [B, L, V] tensor
    (~1 GB of fp32 traffic at V≈156895 / L=512), which dominates per-step
    Python overhead on aux microsteps where the policy network's output is
    fully ignored under frozen ``u/r`` heads.  When all the drafter needs is
    ``max softmax(logits)`` to threshold and ``argmax(logits)`` for the
    sampled token, we can compute both with a single ``logsumexp`` reduction:

        max_logit  = logits.max(-1)
        log_p_max  = max_logit - logsumexp(logits, -1)
        confidence = exp(log_p_max)
    """
    max_logit, max_tok = logits.max(dim=-1)
    lse = torch.logsumexp(logits.float(), dim=-1)
    confidence = (max_logit.float() - lse).exp().to(logits.dtype)
    return confidence, max_tok


def _drafter_confidence_threshold(cfg: dict) -> float:
    return float(cfg.get("inference", {}).get("drafter", {}).get("confidence_threshold", 0.7))


def _resolve_unmask_budget(cfg: dict, L: int) -> Optional[int]:
    """Convert ``inference.max_unmask_*`` to an integer budget; ``None`` if unbounded."""
    import math
    ic = cfg.get("inference", {})
    max_tokens = ic.get("max_unmask_tokens_per_step")
    max_frac = ic.get("max_unmask_fraction_per_step")
    if max_tokens is None and max_frac is None:
        return None
    if max_tokens is not None:
        budget = int(max_tokens)
    else:
        try:
            frac = float(max_frac)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(frac) or frac <= 0.0:
            return None
        budget = int(math.ceil(frac * max(L, 1)))
    if budget <= 0 or budget >= L:
        return None
    return budget


def _select_topb_by_score(
    candidate_mask: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Vectorised top-``budget`` selection among ``candidate_mask``-positive cells."""
    L = candidate_mask.shape[-1]
    masked_scores = torch.where(
        candidate_mask, scores.float(), torch.full_like(scores, float("-inf"), dtype=torch.float32)
    )
    top_idx = masked_scores.topk(min(budget, L), dim=-1).indices
    keep = torch.zeros_like(candidate_mask)
    keep.scatter_(-1, top_idx, True)
    return keep & candidate_mask


def _stable_verifier_miss_fraction(
    cache_mgr: SpeculativeCacheBookkeeper,
    *,
    stable_kv_cache_enabled: bool,
    primary_cache_enabled: bool,
) -> torch.Tensor:
    """Per-sample verifier compute fraction after persistent K_stable reuse.

    K_spec is intentionally excluded: the transient draft frontier is
    verification debt, not a persistent cache. When stable-cache execution is
    unavailable for the current run, a verifier event costs one full verifier
    pass.
    """
    stable_mask = cache_mgr.stable.get_cached_mask()
    if not stable_kv_cache_enabled or not primary_cache_enabled:
        return torch.ones(stable_mask.shape[0], device=stable_mask.device)
    return (1.0 - stable_mask.float()).mean(dim=-1).clamp(0.0, 1.0)


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
    run_aux_on_verifier: bool,
    primary_cache_enabled: bool,
    need_hidden: bool,
    need_all_hidden: bool,
    step_interval: int,
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
        f"aux_on_verifier={_on_off(run_aux_on_verifier)} "
        f"primary_hidden={_on_off(need_hidden)} "
        f"primary_all_hidden={_on_off(need_all_hidden)} "
        f"primary_cache_fastpath={_on_off(primary_cache_enabled and use_prefix_kv_cache)} "
        f"verifier_mode={verifier_mode} "
        f"verifier_schedule={verifier_schedule.get('mode')} "
        f"draft_token_budget={int(verifier_schedule.get('draft_token_budget', 0))} "
        f"max_draft_microsteps={int(verifier_schedule.get('max_draft_microsteps', 0))} "
        f"step_interval={step_interval} "
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
    # V3: per Jazbec et al. 2026 §3.2, the argmax-fallback (force-unmask the
    # highest-prob masked position when the policy returns u_t = 0
    # everywhere) is applied at TEST TIME ONLY. Forcing unmasking during
    # training was found to be prone to reward hacking; instead, rollouts
    # that fail to fill all positions terminate naturally and are penalised
    # via the unresolved-fraction term in the reward. We use record_trajectory
    # as the proxy for "this is a training rollout" — eval calls pass
    # record_trajectory=False, training rollouts pass True.
    _config_fallback = bool(ic["fallback_unmask"])
    use_fallback = _config_fallback and (not record_trajectory)
    disable_remask = ic.get("disable_remask", False)
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)
    use_positional_cache = bool(ic.get("positional_cache", {}).get("enabled", False))
    verifier_cfg = ic.get("verifier", {})

    # Block-wise policy (Option A): crop policy inputs to the active block.
    # See aoae/models/policy.py::call_policy_block for the design note and
    # the deferred Option B (BlockAOAEPolicy with global summary token).
    _blockwise_cfg = (cfg.get("policy", {}) or {}).get("block_wise", {}) or {}
    _blockwise_policy = bool(_blockwise_cfg.get("enabled", False))
    _block_length = max(1, int(ic.get("block_length", 32)))
    _block_context_left = max(0, int(_blockwise_cfg.get("context_left", 0)))

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
    reuse_state = None
    pos_state = init_positional_state(B, L_gen, device)
    draft_frontier = DraftFrontier(B, L_gen, device)

    # H_t from the previous step: used to compute per-position drift for the
    # cache quality F1 signal.  Mirrors the same variable in aoae_inference().
    _prev_H_t: Optional[torch.Tensor] = None

    # --- KV cache state ---
    schedule_cfg = _verifier_schedule(ic)
    cache_cfg = cfg.get("cache", {}) or {}
    if "kspec_skip" in cache_cfg:
        raise ValueError(
            "cache.kspec_skip was removed. K_spec is the transient draft frontier; "
            "it is not a persistent cache and cannot be 'skipped'.  Drop the key."
        )
    use_prefix_kv_cache = bool(cache_cfg.get("prefix_kv_cache", False)) and use_cache
    requested_stable_kv_cache = bool(cache_cfg.get("stable_kv_cache", False)) and use_cache
    if requested_stable_kv_cache:
        raise RuntimeError(
            "cache.stable_kv_cache=true requests persistent K_stable KV execution, "
            "but the current AOAE verifier path does not safely merge skipped-position "
            "primary logits/H_t back into policy state. Keep cache.stable_kv_cache=false "
            "for the canonical paper path; K_stable is still trained and reported."
        )
    use_stable_kv_cache = False
    aux_past_kv = None
    pri_past_kv = None
    _aux_cache_initialized = False
    _primary_cache_initialized = False

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
    # wiring for this call, not on the current diffusion step. Loading a PRISM
    # sidecar must not by itself disable the primary cache fast path; hidden
    # states are required only when PRISM scores are actually enabled.
    _need_all_hidden = _track_kv
    _use_prism_score_for_hidden = bool(verifier_cfg.get("use_prism_score", False))
    _need_hidden = (prism_adapter is not None) and _use_prism_score_for_hidden and not _need_all_hidden
    _primary_cache_enabled = not (_need_hidden or _need_all_hidden)
    _aux_cache_enabled = use_prefix_kv_cache

    drafter_cfg = ic.get("drafter", {})
    # Fast drafter path: when frozen u/r heads make the draft action
    # deterministic (u_t = (confidence >= threshold) & mask, r_t=0,
    # kappa_t=0 forced on aux microsteps anyway), the policy network and the
    # full ``log_softmax`` over V≈156895 inside soft-mask are pure overhead.
    # Skipping them on aux microsteps cuts ~10 ms per draft microstep on
    # H100 — the single biggest win for speculative AOAE wall time.
    _fast_drafter_path = bool(drafter_cfg.get("fast_path", False)) and not record_trajectory
    if _fast_drafter_path:
        # The fast path is only safe when u/r are frozen out of the trained
        # head set (the canonical paper config trains only ``cache`` and
        # ``access``).  Validate up-front to avoid silently changing rollout
        # semantics when a future config trains u/r.
        _train_heads_check = parse_head_set(cfg.get("grpo", {}).get("train_heads"))
        if _train_heads_check is not None and any(
            h in _train_heads_check for h in ("unmask", "u", "u_t", "remask", "r", "r_t")
        ):
            _fast_drafter_path = False
    _drafter_threshold = _drafter_confidence_threshold(cfg)
    _unmask_budget = _resolve_unmask_budget(cfg, L_gen)
    _run_aux_on_verifier_raw = drafter_cfg.get("run_on_verifier", "auto")
    if isinstance(_run_aux_on_verifier_raw, str):
        mode = _run_aux_on_verifier_raw.strip().lower()
        if mode in {"always", "true", "yes", "on"}:
            _run_aux_on_verifier = True
        elif mode in {"never", "false", "no", "off"}:
            _run_aux_on_verifier = False
        elif mode == "auto":
            # Greedy AOAE validates stored draft tokens.  A fresh auxiliary pass
            # at verifier time is only needed for explicit verifier-step
            # composition/diagnostics, and otherwise doubles verifier wall time.
            _run_aux_on_verifier = bool(gamma > 0.0 and base_temp > 0.0)
        else:
            raise ValueError(
                "inference.drafter.run_on_verifier must be one of "
                "auto/always/never or a boolean."
            )
    else:
        _run_aux_on_verifier = bool(_run_aux_on_verifier_raw)

    def _run_auxiliary_resp(current_y: torch.Tensor) -> torch.Tensor:
        nonlocal aux_past_kv, _aux_cache_initialized

        if _aux_cache_enabled and _aux_cache_initialized:
            aux_logits, aux_past_kv = dual_model.auxiliary_forward_replace_with_cache(
                current_y, resp_slice, aux_past_kv,
            )
            return aux_logits

        if _aux_cache_enabled:
            aux_full, aux_past_kv = dual_model.auxiliary_forward_with_cache(current_y)
            _aux_cache_initialized = True
            return aux_full[:, resp_slice, :]

        return dual_model.auxiliary_forward(current_y)[:, resp_slice, :]

    def _run_primary_cached_resp(
        current_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        nonlocal pri_past_kv, _primary_cache_initialized

        if (
            use_prefix_kv_cache
            and _primary_cache_enabled
            and hasattr(dual_model, "primary_forward_with_cache")
            and hasattr(dual_model, "primary_forward_replace_with_cache")
        ):
            if _primary_cache_initialized:
                logits, pri_past_kv = dual_model.primary_forward_replace_with_cache(
                    current_y, resp_slice, pri_past_kv,
                )
                return logits, None, None

            pri_full, pri_past_kv = dual_model.primary_forward_with_cache(current_y)
            _primary_cache_initialized = True
            return pri_full[:, resp_slice, :], None, None

        return _run_primary_full_resp(current_y)

    def _reset_prefix_caches(*, aux: bool = True, primary: bool = True) -> None:
        nonlocal aux_past_kv, pri_past_kv, _aux_cache_initialized, _primary_cache_initialized
        if aux:
            aux_past_kv = None
            _aux_cache_initialized = False
        if primary:
            pri_past_kv = None
            _primary_cache_initialized = False

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

        if hasattr(dual_model, "primary_forward"):
            pri_full = dual_model.primary_forward(current_y)
        elif hasattr(dual_model, "primary_forward_with_hidden"):
            # Lightweight test shims and a few older wrappers expose only the
            # hidden-state variant. Use it as a compatibility fallback while
            # still treating PRISM as disabled when use_prism_score=false.
            pri_full, _ = dual_model.primary_forward_with_hidden(current_y)
        else:
            raise AttributeError(
                "dual_model must expose primary_forward or primary_forward_with_hidden"
            )
        return pri_full[:, resp_slice, :], None, None

    step_interval = int(schedule_cfg.get("step_interval", 1))
    primary_agree_threshold = float(ic.get("primary_agree_threshold", 0.0))
    force_primary_endpoints = bool(schedule_cfg.get("force_first_last", True))
    aux_cache_reset_threshold = float(ic.get("aux_cache_reset_threshold", 1.1))
    rejection_action = str(verifier_cfg.get("rejection_action", "remask")).lower()
    recompute_after_reject = bool(verifier_cfg.get("recompute_after_reject", True))
    if rejection_action not in ("remask", "mask", "keep", "none", "evict_only"):
        raise ValueError(
            f"Unsupported verifier.rejection_action={rejection_action!r}. "
            "Use remask or keep/evict_only."
        )
    if "aux_compute_ratio" in ic:
        raise ValueError(
            "inference.aux_compute_ratio was moved under inference.drafter.aux_compute_ratio."
        )
    aux_compute_ratio = float(drafter_cfg.get("aux_compute_ratio", 0.35))
    _maybe_log_speculative_rollout_config(
        cfg=cfg,
        prism_adapter=prism_adapter,
        track_kv_enabled=_track_kv,
        use_cache=use_cache,
        use_fallback=use_fallback,
        disable_remask=disable_remask,
        use_prefix_kv_cache=use_prefix_kv_cache,
        aux_cache_enabled=_aux_cache_enabled,
        run_aux_on_verifier=_run_aux_on_verifier,
        primary_cache_enabled=_primary_cache_enabled,
        need_hidden=_need_hidden,
        need_all_hidden=_need_all_hidden,
        step_interval=step_interval,
        primary_agree_threshold=primary_agree_threshold,
        force_primary_endpoints=force_primary_endpoints,
        aux_cache_reset_threshold=aux_cache_reset_threshold,
        gamma=gamma,
    )
    _ema_agreement = 1.0
    _primary_steps = 0
    _aux_only_steps = 0
    # Counters live on the GPU and accumulate across steps; we sync to Python
    # ints exactly once at the end of the rollout.  This eliminates ~10 per-step
    # GPU->CPU syncs (each ~0.1-0.5 ms on H100), which dominates the per-NFE
    # Python overhead for the speculative loop.
    _agreement_sum_t = torch.zeros((), dtype=torch.float64, device=device)
    _agreement_obs_t = torch.zeros((), dtype=torch.long, device=device)
    _safe_reuse_sum_t = torch.zeros((), dtype=torch.float64, device=device)
    _safe_reuse_obs_t = torch.zeros((), dtype=torch.long, device=device)
    _draft_accepts_t = torch.zeros((), dtype=torch.long, device=device)
    _draft_rejects_t = torch.zeros((), dtype=torch.long, device=device)
    _stable_commits_t = torch.zeros((), dtype=torch.long, device=device)
    _stable_invalidations_t = torch.zeros((), dtype=torch.long, device=device)
    _drafter_cache_resets = 0
    _draft_microsteps_since_verify = 0
    _force_next_verifier = False
    _aux_units = torch.zeros(B, device=device)
    _verifier_units = torch.zeros(B, device=device)
    # Full-quality baseline: run the verifier in normal mode at every planned
    # diffusion microstep. Keeping the denominator fixed at T gives credit both
    # for cheaper verifier passes and for early completion.
    _baseline_units = torch.full((B,), float(T), device=device)

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
        primary_fresh_mask = torch.zeros(B, L_gen, dtype=torch.bool, device=device)
        frontier_accept_mask = torch.zeros(B, L_gen, dtype=torch.bool, device=device)
        frontier_reject_mask = torch.zeros(B, L_gen, dtype=torch.bool, device=device)

        aux_logits: Optional[torch.Tensor] = None
        composition_agreement = torch.zeros(B, L_gen, dtype=torch.bool, device=device)

        if not run_primary:
            aux_logits = _run_auxiliary_resp(y)
            resp_logits = aux_logits
            # No verifier observation happened on this cheap draft step, so no
            # position is considered accepted for composition or K_spec.
            agreement = torch.zeros(B, L_gen, dtype=torch.bool, device=device)
            _pri_hidden_for_prism = None
            _pri_hidden_states_for_tracker = None
            _aux_only_steps += 1

            if _fast_drafter_path:
                # ---- Fast aux drafter (no policy / no soft-mask H_t) ----
                # Skip the soft_mask + policy + access-set chain entirely when
                # the trained heads do not influence aux-only decisions.  The
                # only thing the rest of the loop body would do here is:
                #   r_t = 0, kappa_t = 0 (forced on aux microsteps regardless),
                #   u_t = (confidence >= threshold) & mask_ind & budget,
                #   resp_tokens[u_t] = aux_argmax, draft_frontier.add(...),
                #   pos_state / cache_mgr bookkeeping.
                # All of that is reproduced cheaply below with no per-step
                # GPU->CPU sync — even ``draft_frontier.add`` is replaced by an
                # inline GPU update so the only sync ever performed in this
                # branch is the verifier scheduler reading the Python counter.
                fast_conf, fast_argmax = _confidence_and_argmax_fast(aux_logits)
                fast_u = (fast_conf >= _drafter_threshold) & mask_ind.bool()
                if _unmask_budget is not None:
                    fast_u = _select_topb_by_score(fast_u, fast_conf, _unmask_budget)

                # Inline ``draft_frontier.add`` with a Python-side upper-bound
                # counter — keeps the verifier scheduler sync-free and lets it
                # always trigger after ``max_draft_microsteps`` regardless of
                # the actual frontier population.  A conservative upper bound
                # is fine because (a) the scheduler also caps at
                # ``max_draft_microsteps`` and (b) firing the verifier slightly
                # earlier than strictly necessary never affects correctness.
                draft_frontier._max_count += (
                    _unmask_budget if _unmask_budget is not None else L_gen
                )
                draft_frontier.mask = draft_frontier.mask | fast_u
                resp_tokens = y[:, resp_slice].clone()
                resp_tokens = torch.where(fast_u, fast_argmax, resp_tokens)
                draft_frontier.token_ids = torch.where(
                    fast_u, resp_tokens.long(), draft_frontier.token_ids
                )
                draft_frontier.age = torch.where(
                    fast_u, torch.zeros_like(draft_frontier.age), draft_frontier.age
                )
                if record_trajectory:
                    y = y.clone()
                y[:, resp_slice] = resp_tokens

                # Phase 3a / 3b on the fast path: K_spec mirror tracks the
                # frontier, K_stable cannot grow because kappa_t is forced to 0.
                if cache_mgr is not None:
                    cache_mgr.step_spec(draft_frontier.mask)
                    cache_mgr.stable.step_age()

                if trajectory is not None and cache_mgr is not None:
                    trajectory.spec_cached_fractions.append(
                        cache_mgr.spec_cached_fraction().detach()
                    )
                    trajectory.stable_cached_fractions.append(
                        cache_mgr.stable_cached_fraction().detach()
                    )
                    trajectory.cached_fractions.append(
                        cache_mgr.cached_fraction().detach()
                    )

                draft_frontier.step_age()
                _aux_units += aux_compute_ratio
                _draft_microsteps_since_verify += 1
                continue

        elif (
            not (
                hasattr(dual_model, "primary_forward")
                or hasattr(dual_model, "primary_forward_with_hidden")
                or hasattr(dual_model, "primary_forward_with_cache")
            )
        ) and hasattr(dual_model, "dual_forward_resp"):
            # Compatibility / ablation path: run both modes at verifier time.
            dual_out = dual_model.dual_forward_resp(
                y, resp_slice, need_hidden=_need_hidden, need_all_hidden=_need_all_hidden,
            )
            resp_logits = dual_out.primary_logits
            aux_logits = dual_out.auxiliary_logits
            _pri_hidden_for_prism = dual_out.primary_hidden
            _pri_hidden_states_for_tracker = dual_out.primary_hidden_states
            primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
            _primary_steps += 1
        else:
            # Canonical verifier path: the primary validates stored frontier
            # tokens directly.  A fresh auxiliary pass here is optional and off
            # in the paper config because it doubles verifier wall time without
            # changing greedy argmax decisions under accepted-frontier validation.
            if _run_aux_on_verifier:
                aux_logits = _run_auxiliary_resp(y)
            resp_logits, _pri_hidden_for_prism, _pri_hidden_states_for_tracker = _run_primary_cached_resp(y)
            primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
            _primary_steps += 1

        if run_primary:
            verifier_miss = (
                _stable_verifier_miss_fraction(
                    cache_mgr,
                    stable_kv_cache_enabled=use_stable_kv_cache,
                    primary_cache_enabled=_primary_cache_enabled,
                )
                if cache_mgr is not None
                else torch.ones(B, device=device)
            )
            _verifier_units += verifier_miss.to(device)
            if aux_logits is not None:
                _aux_units += aux_compute_ratio
        else:
            _aux_units += aux_compute_ratio

        # --- PRISM quality scores ---
        q_scores = None
        _use_prism_score = bool(verifier_cfg.get("use_prism_score", False))
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
                accept_counts = frontier_accept_mask.sum(dim=-1)
                reject_counts = frontier_reject_mask.sum(dim=-1)
                _draft_accepts_t = _draft_accepts_t + accept_counts.sum()
                _draft_rejects_t = _draft_rejects_t + reject_counts.sum()
                if trajectory is not None:
                    trajectory.frontier_sizes.append(frontier_before.sum(dim=-1).detach())
                    trajectory.frontier_accept_counts.append(accept_counts.detach())
                    trajectory.frontier_reject_counts.append(reject_counts.detach())

                if frontier_reject_mask.any():
                    resp_tokens = y[:, resp_slice].clone()
                    if cache_mgr is not None:
                        _stable_invalidations_t = _stable_invalidations_t + (
                            cache_mgr.stable.get_cached_mask() & frontier_reject_mask
                        ).sum()
                        cache_mgr.invalidate(frontier_reject_mask.float())

                    if rejection_action in ("remask", "mask"):
                        resp_tokens[frontier_reject_mask] = mask_id
                        y = y.clone()
                        y[:, resp_slice] = resp_tokens
                        mask_ind = (resp_tokens == mask_id)
                        _reset_prefix_caches(aux=True, primary=True)

                        if recompute_after_reject:
                            # Policy state should normally be computed on the
                            # corrected sequence. A rejected token has been
                            # removed, so rerun the verifier on the remasked
                            # state instead of using logits conditioned on the
                            # stale draft.
                            if _run_aux_on_verifier:
                                aux_logits = _run_auxiliary_resp(y)
                                _aux_units += aux_compute_ratio
                            else:
                                aux_logits = None
                            resp_logits, _pri_hidden_for_prism, _pri_hidden_states_for_tracker = _run_primary_cached_resp(y)
                            verifier_miss = (
                                _stable_verifier_miss_fraction(
                                    cache_mgr,
                                    stable_kv_cache_enabled=use_stable_kv_cache,
                                    primary_cache_enabled=_primary_cache_enabled,
                                )
                                if cache_mgr is not None
                                else torch.ones(B, device=device)
                            )
                            _verifier_units += verifier_miss.to(device)
                            primary_fresh_mask = torch.ones(B, L_gen, dtype=torch.bool, device=device)
                            if _use_prism_score and prism_adapter is not None and _pri_hidden_for_prism is not None:
                                with torch.no_grad():
                                    q_scores = prism_adapter(_pri_hidden_for_prism.float())
                    else:
                        # Ablation mode: leave rejected draft tokens in the
                        # sequence but still mark their KV as unusable. They are
                        # excluded from K_stable commits below.
                        y = y.clone()
                        y[:, resp_slice] = resp_tokens

                rejected_total = float(frontier_reject_mask.float().sum().item())
                frontier_total = float(frontier_before.float().sum().item())
                rejection_rate = rejected_total / max(frontier_total, 1.0)
                accept_rate = 1.0 - rejection_rate
                _ema_agreement = 0.8 * _ema_agreement + 0.2 * accept_rate
                if rejection_rate > aux_cache_reset_threshold:
                    _reset_prefix_caches(aux=True, primary=True)
                    _drafter_cache_resets += 1
                    _force_next_verifier = True

            # Keep a separate raw agreement diagnostic for safe-reuse analysis.
            if aux_logits is not None:
                raw_agreement, reuse_state, _ = compute_reuse_signal(
                    resp_logits, aux_logits, cfg, state=reuse_state
                )
                raw_agreement = _fresh_primary_agreement(raw_agreement, primary_fresh_mask)
                composition_agreement = raw_agreement
                active_for_agreement = (mask_ind.bool() | frontier_before.bool()) & primary_fresh_mask
            else:
                raw_agreement = frontier_accept_mask & primary_fresh_mask
                active_for_agreement = frontier_before & primary_fresh_mask
            if active_for_agreement.any():
                _active_count = active_for_agreement.sum()
                _hit_count = (raw_agreement & active_for_agreement).sum()
                _agreement_sum_t = _agreement_sum_t + _hit_count.to(torch.float64)
                _agreement_obs_t = _agreement_obs_t + _active_count
                _safe_reuse_sum_t = _safe_reuse_sum_t + _hit_count.to(torch.float64)
                _safe_reuse_obs_t = _safe_reuse_obs_t + _active_count
            if aux_logits is not None and not frontier_before.any() and primary_fresh_mask.any():
                verifier_agreement = float(raw_agreement[primary_fresh_mask].float().mean().item())
                _ema_agreement = 0.8 * _ema_agreement + 0.2 * verifier_agreement

            agreement = frontier_accept_mask
            draft_frontier.clear()
            _draft_microsteps_since_verify = 0
        else:
            _draft_microsteps_since_verify += 1
        # ``agreement_rates`` is only used as a fallback for
        # trajectory.mean_agreement_rate when the authoritative observation
        # counter is empty.  Skipping the per-step sync saves a CPU/GPU round
        # trip every iteration for both eval and GRPO rollouts.

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
        if _blockwise_policy:
            _blk_window = active_block_window(
                mask_ind,
                _block_length,
                context_left=_block_context_left,
            )
            policy_out = call_policy_block(
                policy,
                H_t, mask_ind, step_frac,
                _blk_window,
                temperature=policy_temperature,
                confidence=confidence,
                quality_scores=q_scores,
                agreement=agreement.float(),
                age_feature=age_feat,
                last_action_feature=last_action_feat,
            )
        else:
            _blk_window = None
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
            run_primary=run_primary,
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
            include_heads = parse_head_set(
                cfg.get("grpo", {}).get(
                    "include_heads_in_logprob",
                    cfg.get("grpo", {}).get("train_heads"),
                )
            )
            if bool(cfg.get("phase_a_v2", False)):
                include_heads = {"remask"} if run_primary else {"unmask"}
            lp = pol_inner.log_prob(policy_out, actions, include_heads=include_heads)
            trajectory.actions.append(
                {k: (v.detach() if torch.is_tensor(v) else v) for k, v in actions.items()}
            )
            trajectory.log_probs.append(lp.detach())
            trajectory.policy_outputs.append(
                {k: (v.detach() if torch.is_tensor(v) else v) for k, v in policy_out.items()}
            )
            trajectory.H_t_list.append(H_t.detach())
            trajectory.weighted_embeds_list.append(weighted_embeds.detach())
            trajectory.entropy_list.append(entropy.detach())
            trajectory.mask_ind_list.append(mask_ind.detach())
            trajectory.confidence_list.append(confidence.detach())
            trajectory.run_primary_list.append(bool(run_primary))
            trajectory.frontier_before_list.append(
                (frontier_before.detach() if run_primary else torch.zeros_like(mask_ind))
            )
            trajectory.frontier_accept_mask_list.append(frontier_accept_mask.detach())
            trajectory.frontier_reject_mask_list.append(frontier_reject_mask.detach())
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
            trajectory.access_diag_list.append(dict(_access_diag))
            if "ell_t" in actions:
                trajectory.boundary_actions.append(actions["ell_t"].detach())
            trajectory.step_fracs.append(step_frac)
            if _blk_window is not None:
                trajectory.block_windows.append(_blk_window)
        elif trajectory is not None:
            # Lightweight eval stats: enough for access-pattern summaries
            # without storing logits, H_t, log-probs, or policy outputs.
            trajectory.access_exec_list.append(q_exec.detach())
            trajectory.access_mandatory_list.append(q_mandatory.detach())
            trajectory.access_diag_list.append(dict(_access_diag))
            if "ell_t" in actions:
                trajectory.boundary_actions.append(actions["ell_t"].detach())

        # --- Count cache thrashing ---
        thrash = None
        if cache_mgr is not None:
            thrash = cache_mgr.count_thrash(r_t)
            _stable_invalidations_t = _stable_invalidations_t + thrash.sum().to(_stable_invalidations_t.dtype)
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
            if gamma > 0 and aux_logits is not None:
                composed_logits = compose_prediction_dual(
                    resp_logits, aux_logits, composition_agreement, gamma=gamma,
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
                * (1.0 - frontier_reject_mask.float())
            )
            stable_before = cache_mgr.stable.get_cached_mask().clone()
            cache_mgr.step_stable(stable_commit_mask, r_t)
            newly_stable = stable_commit_mask.bool() & ~stable_before & ~r_t.bool()
            _stable_commits_t = _stable_commits_t + newly_stable.sum()

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

        # The full-rollout GRPO path keeps tensor identity stable for
        # autograd-aware trajectory storage; eval / lightweight stat collection
        # mutates y in place to skip a [B, P+L_gen] clone every step.
        if record_trajectory:
            y = y.clone()
        y[:, resp_slice] = resp_tokens

    # --- Record final state ---
    if trajectory is not None:
        # Sync GPU-resident counters once now that the loop is over.
        _agreement_sum = float(_agreement_sum_t.item())
        _agreement_obs = int(_agreement_obs_t.item())
        _safe_reuse_sum = float(_safe_reuse_sum_t.item())
        _safe_reuse_obs = int(_safe_reuse_obs_t.item())
        _draft_accepts = int(_draft_accepts_t.item())
        _draft_rejects = int(_draft_rejects_t.item())
        _stable_commits = int(_stable_commits_t.item())
        _stable_invalidations = int(_stable_invalidations_t.item())

        trajectory.final_tokens = y[:, resp_slice].detach()
        if trajectory.completion_step is None:
            trajectory.completion_step = torch.full((B,), T, device=device)
        if _agreement_obs > 0:
            trajectory.mean_agreement_rate = _agreement_sum / max(_agreement_obs, 1)
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
    # If the tracker ran, store its summary in the trajectory so evaluate.py
    # and the training logger can collect it (same pattern as aoae_inference).
    if _dynamics_tracker is not None:
        _dyn_summary = _dynamics_tracker.summarize()
        if trajectory is None:
            trajectory = SpeculativeTrajectory()
        trajectory.kv_dynamics_summary = _dyn_summary

    return y, trajectory


# ============================================================================
# Block-structured AOAE inference (paper-faithful, no KV cache).
# ============================================================================
# Mirrors block_smode_decode's structure (block-by-block left-to-right,
# prefix_ids = y[:, :blk_end] truncated each microstep, NO KV cache) so it sits
# at TPS parity with the LLaDA 2.1 baseline.  Differences vs the baseline:
#   - Within each block, the auxiliary (hard-routed) drafter takes K_draft
#     cheap microsteps proposing tokens; the primary (soft-routed) verifier
#     runs periodically per inference.verifier_schedule.
#   - Per-microstep decisions come from the AOAE policy operating on a cropped
#     soft-mask H_t for the active block (~32 positions, ~1ms vs ~10ms full).
#   - On draft microsteps: only u_t (unmask) is applied; r_t/kappa_t are
#     forced to zero (parallel of speculative_inference's run_primary=False
#     gating — drafts must not make persistent commitments before verification).
#   - On verifier microsteps: full AOAE phases — frontier validate/reject,
#     policy r_t (remask), kappa_t (cache commit), q_t (access).
#
# OPTION B note (deferred): the block-cropped policy could additionally
# receive a small global summary token (committed-prefix mean H, blk_idx /
# n_blocks, prev-block accept rate).  See aoae/models/policy.py for the
# design note.
def aoae_block_inference(
    dual_model: DualModelWrapper,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    record_trajectory: bool = False,
    policy_temperature: float = 1.0,
    collect_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[SpeculativeTrajectory]]:
    """Block-structured AOAE speculative inference (no KV cache, paper-faithful)."""
    from .inference import _max_prob_and_argmax

    ic = cfg["inference"]
    L_gen = int(ic["gen_length"])
    block_len = max(1, int(ic.get("block_length", 32)))
    mask_id = int(cfg["base_model"]["mask_token_id"])
    use_cache = bool(cfg["cache"]["enabled"])
    disable_remask = bool(ic.get("disable_remask", False))
    use_positional_cache = bool(ic.get("positional_cache", {}).get("enabled", False))
    schedule_cfg = _verifier_schedule(ic)
    max_draft_microsteps = max(1, int(schedule_cfg.get("max_draft_microsteps", 4)))

    B, P = prompt_ids.shape
    L_total = P + L_gen
    device = prompt_ids.device

    y = torch.cat(
        [
            prompt_ids,
            torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    resp_slice = slice(P, L_total)

    n_blocks = (L_gen + block_len - 1) // block_len

    _thrash_age_decay = float(cfg.get("grpo", {}).get("thrash_age_decay", 0.0))
    cache_mgr = (
        SpeculativeCacheBookkeeper(B, L_gen, device, thrash_age_decay=_thrash_age_decay)
        if use_cache else None
    )
    pos_state = init_positional_state(B, L_gen, device)
    draft_frontier = DraftFrontier(B, L_gen, device)
    trajectory = SpeculativeTrajectory() if (record_trajectory or collect_stats) else None
    pol_inner = policy.module if hasattr(policy, "module") else policy

    # Approximate t/T fraction.  Block_smode_decode does not use a global step
    # counter; for AOAE the policy's step_frac feature is informative, so we
    # approximate it by (1 - n_blocks_done / n_blocks) decreasing each block,
    # plus a within-block term for finer granularity.
    total_microsteps = 0
    primary_steps = 0
    aux_steps = 0
    draft_accepts = 0
    draft_rejects = 0

    for blk_idx in range(n_blocks):
        blk_start = P + blk_idx * block_len
        blk_end = min(P + (blk_idx + 1) * block_len, P + L_total)
        # Response-relative window for trajectory.block_windows.
        resp_b_s = blk_start - P
        resp_b_e = blk_end - P
        blk_slice = slice(blk_start, blk_end)
        blk_w = blk_end - blk_start

        # Most recent verifier output for this block (for stale agreement).
        last_pri_argmax_blk: Optional[torch.Tensor] = None
        # Verifier scheduling state.
        draft_microsteps_since_verify = 0
        force_next_verifier = False
        # Per-block step counter for step_frac approximation.
        blk_step_idx = 0
        max_blk_steps = max(1, int(schedule_cfg.get("max_draft_microsteps", 4))
                            * 4)  # generous cap to avoid infinite loop
        while blk_step_idx < max_blk_steps:
            blk_tokens = y[:, blk_slice]
            mask_ind_blk = blk_tokens == mask_id
            # Done with this block when no masks remain AND no pending frontier.
            if (not mask_ind_blk.any()) and (not draft_frontier.mask[:, resp_b_s:resp_b_e].any()):
                break

            run_verifier = _should_run_verifier(
                schedule=schedule_cfg,
                step_idx=blk_step_idx,
                t=max(1, max_blk_steps - blk_step_idx),
                frontier=draft_frontier,
                draft_microsteps_since_verify=draft_microsteps_since_verify,
                force_next=force_next_verifier,
            )
            force_next_verifier = False

            prefix_ids = y[:, :blk_end]

            # ---- Auxiliary forward (drafter), cropped to block ----
            aux_logits_blk = dual_model.auxiliary_forward_resp(prefix_ids, blk_slice)
            aux_steps += 1
            aux_conf_blk, aux_tok_blk = _max_prob_and_argmax(aux_logits_blk)

            pri_logits_blk: Optional[torch.Tensor] = None
            if run_verifier:
                pri_logits_blk = dual_model.primary_forward_resp(prefix_ids, blk_slice)
                primary_steps += 1

            # Source logits for soft-mask: prefer fresh primary on verifier
            # microsteps (matches speculative_inference), else aux on draft
            # microsteps.  This is the smallest distributional shift relative
            # to the full-seq policy training.
            sm_logits_blk = pri_logits_blk if pri_logits_blk is not None else aux_logits_blk
            H_blk, conf_blk, ent_blk, weighted_blk = call_soft_mask(
                soft_mask_module, sm_logits_blk, mask_ind_blk,
                step_frac=(1.0 - blk_idx / max(1, n_blocks)),
                return_weighted=True,
            )

            # PRISM scores cropped to block (disabled by default; quality_scores=None).
            q_scores_blk = None
            if prism_adapter is not None:
                # Only built when verifier ran (PRISM consumes hidden states; on
                # draft microsteps we have aux logits but typically no hidden state).
                # Keep None on draft microsteps; the policy treats None as zeros.
                q_scores_blk = None

            # Agreement between drafter argmax and primary argmax for this block.
            # On verifier microsteps: fresh agreement.  On draft microsteps:
            # stale agreement vs the most recent verifier argmax (or zeros if
            # the verifier hasn't run yet for this block).
            if pri_logits_blk is not None:
                _, pri_argmax_blk = _max_prob_and_argmax(pri_logits_blk)
                last_pri_argmax_blk = pri_argmax_blk
                agreement_blk = (aux_tok_blk == pri_argmax_blk).float()
            elif last_pri_argmax_blk is not None:
                agreement_blk = (aux_tok_blk == last_pri_argmax_blk).float()
            else:
                agreement_blk = torch.zeros_like(aux_conf_blk)

            # ---- Cropped policy forward (block-local tensors only) ----
            step_frac = max(0.0, 1.0 - (blk_idx + blk_step_idx / max_blk_steps) / max(1, n_blocks))
            age_feat_blk = None
            last_action_feat_blk = None
            if use_positional_cache:
                age_feat, last_action_feat = get_policy_positional_features(pos_state, cfg)
                age_feat_blk = age_feat[:, resp_b_s:resp_b_e].contiguous()
                last_action_feat_blk = last_action_feat[:, resp_b_s:resp_b_e].contiguous()

            policy_out_blk = call_policy(
                policy,
                H_blk, mask_ind_blk, step_frac,
                temperature=policy_temperature,
                confidence=conf_blk,
                quality_scores=q_scores_blk,
                agreement=agreement_blk,
                age_feature=age_feat_blk,
                last_action_feature=last_action_feat_blk,
            )
            actions_blk = pol_inner.sample_actions(policy_out_blk, mask_ind_blk)
            actions_blk = _apply_frozen_action_heads(
                actions_blk,
                confidence=conf_blk,
                mask_ind=mask_ind_blk,
                cfg=cfg,
                run_primary=run_verifier,
            )
            actions_blk = apply_unmask_budget(
                actions_blk, policy_out_blk, mask_ind_blk, cfg,
            )

            u_t_blk = actions_blk["u_t"]
            r_t_blk = actions_blk["r_t"]
            kappa_t_blk = actions_blk["kappa_t"]

            # Draft microsteps must not make persistent commitments.
            if not run_verifier:
                r_t_blk = torch.zeros_like(r_t_blk)
                kappa_t_blk = torch.zeros_like(kappa_t_blk)
                actions_blk = {**actions_blk, "r_t": r_t_blk, "kappa_t": kappa_t_blk}
            elif disable_remask:
                r_t_blk = torch.zeros_like(r_t_blk)
                actions_blk = {**actions_blk, "r_t": r_t_blk}

            # ---- Phase 1: Remask (verifier microsteps only) ----
            blk_tokens = blk_tokens.clone()
            remask_positions_blk = r_t_blk.bool() & ~mask_ind_blk
            if remask_positions_blk.any():
                blk_tokens[remask_positions_blk] = mask_id

            # ---- Phase 2: Unmask via aux argmax (draft frontier) or via
            # verifier argmax (when verifier ran) ----
            unmask_positions_blk = u_t_blk.bool() & mask_ind_blk
            if pri_logits_blk is not None:
                # Verifier microstep: validate the existing frontier first.
                pri_conf_blk, pri_tok_blk = _max_prob_and_argmax(pri_logits_blk)
                f_mask_blk = draft_frontier.mask[:, resp_b_s:resp_b_e]
                f_tok_blk = draft_frontier.token_ids[:, resp_b_s:resp_b_e]
                if f_mask_blk.any():
                    accept_blk = f_mask_blk & pri_tok_blk.eq(f_tok_blk.clamp(min=0))
                    reject_blk = f_mask_blk & ~accept_blk
                    draft_accepts += int(accept_blk.sum().item())
                    draft_rejects += int(reject_blk.sum().item())
                    if reject_blk.any():
                        # Replace rejected drafts with verifier argmax.
                        blk_tokens[reject_blk] = pri_tok_blk[reject_blk]
                    # Clear the block's slice of the frontier.
                    draft_frontier.mask[:, resp_b_s:resp_b_e] = False
                    draft_frontier.token_ids[:, resp_b_s:resp_b_e] = -1
                    draft_frontier.scores[:, resp_b_s:resp_b_e] = 0.0
                # Apply policy unmask using verifier argmax.
                if unmask_positions_blk.any():
                    blk_tokens[unmask_positions_blk] = pri_tok_blk[unmask_positions_blk]
            else:
                # Draft microstep: stage drafted tokens in the frontier.
                if unmask_positions_blk.any():
                    blk_tokens[unmask_positions_blk] = aux_tok_blk[unmask_positions_blk]
                    full_mask = torch.zeros_like(draft_frontier.mask)
                    full_tok = torch.zeros_like(draft_frontier.token_ids)
                    full_mask[:, resp_b_s:resp_b_e] = unmask_positions_blk
                    full_tok[:, resp_b_s:resp_b_e] = aux_tok_blk
                    draft_frontier.add(full_mask, full_tok)
                draft_microsteps_since_verify += 1

            # ---- Phase 3: Cache commit (verifier microsteps only) ----
            thrash_blk = None
            if cache_mgr is not None:
                # Build full-L tensors for cache_mgr (it tracks K_stable on
                # the response slice).
                full_r = torch.zeros((B, L_gen), device=device)
                full_r[:, resp_b_s:resp_b_e] = r_t_blk
                thrash_blk = cache_mgr.count_thrash(full_r)

            y[:, blk_slice] = blk_tokens

            if pri_logits_blk is not None:
                draft_microsteps_since_verify = 0

            # ---- Trajectory recording (scatter back to full-seq for GRPO) ----
            if record_trajectory and trajectory is not None:
                full_H = torch.zeros((B, L_gen, H_blk.shape[-1]), device=device, dtype=H_blk.dtype)
                full_H[:, resp_b_s:resp_b_e, :] = H_blk
                full_W = torch.zeros((B, L_gen, weighted_blk.shape[-1]), device=device, dtype=weighted_blk.dtype)
                full_W[:, resp_b_s:resp_b_e, :] = weighted_blk
                full_E = torch.zeros((B, L_gen), device=device, dtype=ent_blk.dtype)
                full_E[:, resp_b_s:resp_b_e] = ent_blk
                full_M = torch.zeros((B, L_gen), device=device, dtype=torch.bool)
                full_M[:, resp_b_s:resp_b_e] = mask_ind_blk
                full_A = torch.zeros((B, L_gen), device=device)
                full_A[:, resp_b_s:resp_b_e] = agreement_blk

                # Scatter actions and policy_out back to full-L.
                full_actions: Dict[str, torch.Tensor] = {}
                for k, v in actions_blk.items():
                    if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == blk_w:
                        full_v = torch.zeros((B, L_gen) + tuple(v.shape[2:]), device=device, dtype=v.dtype)
                        full_v[:, resp_b_s:resp_b_e] = v
                        full_actions[k] = full_v
                    else:
                        full_actions[k] = v
                full_policy_out: Dict[str, torch.Tensor] = {}
                for k, v in policy_out_blk.items():
                    if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == blk_w:
                        if k.endswith("_logits"):
                            full_v = torch.full((B, L_gen) + tuple(v.shape[2:]), -1e9,
                                                device=device, dtype=v.dtype)
                        else:
                            full_v = torch.zeros((B, L_gen) + tuple(v.shape[2:]),
                                                 device=device, dtype=v.dtype)
                        full_v[:, resp_b_s:resp_b_e] = v
                        full_policy_out[k] = full_v
                    else:
                        full_policy_out[k] = v

                include_heads = parse_head_set(
                    cfg.get("grpo", {}).get(
                        "include_heads_in_logprob",
                        cfg.get("grpo", {}).get("train_heads"),
                    )
                )
                lp = pol_inner.log_prob(policy_out_blk, actions_blk, include_heads=include_heads)

                trajectory.actions.append(
                    {k: (v.detach() if torch.is_tensor(v) else v) for k, v in full_actions.items()}
                )
                trajectory.log_probs.append(lp.detach())
                trajectory.policy_outputs.append(
                    {k: (v.detach() if torch.is_tensor(v) else v) for k, v in full_policy_out.items()}
                )
                trajectory.H_t_list.append(full_H.detach())
                trajectory.weighted_embeds_list.append(full_W.detach())
                trajectory.entropy_list.append(full_E.detach())
                trajectory.mask_ind_list.append(full_M.detach())
                trajectory.quality_scores_list.append(None)
                trajectory.agreement_list.append(full_A.detach())
                trajectory.age_feature_list.append(None)
                trajectory.last_action_feature_list.append(None)
                trajectory.step_fracs.append(step_frac)
                trajectory.block_windows.append((resp_b_s, resp_b_e))
                if thrash_blk is not None:
                    trajectory.thrash_counts.append(thrash_blk.detach())

            blk_step_idx += 1
            total_microsteps += 1

        # Final commit: any remaining masks in this block get verifier argmax.
        remaining = (y[:, blk_slice] == mask_id)
        if remaining.any():
            pri_logits_final = dual_model.primary_forward_resp(y[:, :blk_end], blk_slice)
            primary_steps += 1
            _, pri_tok_final = _max_prob_and_argmax(pri_logits_final)
            blk_tokens = y[:, blk_slice].clone()
            blk_tokens[remaining] = pri_tok_final[remaining]
            y[:, blk_slice] = blk_tokens

    if trajectory is not None:
        trajectory.final_tokens = y[:, resp_slice].detach()
        # ---- Compute-unit accounting (mirrors speculative_inference) ----
        # The full-quality baseline cost is one verifier pass per planned
        # diffusion microstep, so we use the schedule cap ``n_blocks ×
        # max_blk_steps`` as the denominator; ``aoae_block_inference`` is
        # always invoked with that cap as its iteration ceiling.  Aux drafts
        # cost ``aux_compute_ratio`` units each, verifier passes cost 1
        # (this path does not implement K_stable backend skipping).
        aux_compute_ratio = float(
            cfg.get("inference", {}).get("drafter", {}).get("aux_compute_ratio", 0.35)
        )
        baseline_units = float(max(n_blocks * max(1, max_blk_steps), 1))
        aux_units_t = torch.full(
            (B,), float(aux_steps) * aux_compute_ratio,
            device=device, dtype=torch.float32,
        )
        verifier_units_t = torch.full(
            (B,), float(primary_steps),
            device=device, dtype=torch.float32,
        )
        baseline_units_t = torch.full(
            (B,), baseline_units, device=device, dtype=torch.float32,
        )
        trajectory.aux_compute_units = aux_units_t.detach()
        trajectory.verifier_compute_units = verifier_units_t.detach()
        trajectory.baseline_compute_units = baseline_units_t.detach()
        trajectory.effective_flops = (
            (aux_units_t + verifier_units_t) / baseline_units_t.clamp(min=1.0)
        ).detach()
        trajectory.primary_steps = int(primary_steps)
        trajectory.aux_only_steps = int(aux_steps)
        trajectory.draft_accepts = int(draft_accepts)
        trajectory.draft_rejects = int(draft_rejects)
        _frontier_total = float(draft_accepts + draft_rejects)
        trajectory.frontier_accept_rate = float(
            draft_accepts / max(_frontier_total, 1.0)
        )
        trajectory.frontier_reject_rate = float(
            draft_rejects / max(_frontier_total, 1.0)
        )
        # Block path does not (yet) integrate with cache_mgr accumulators,
        # so report zero K_stable / K_spec occupancy and leave the access /
        # KV-dynamics fields at their dataclass defaults.
        trajectory.total_stable_commits = 0
        trajectory.total_stable_invalidations = 0
        trajectory.total_cache_hits = 0
        trajectory.total_cache_misses = 0
        trajectory.agreement_observations = int(draft_accepts + draft_rejects)
        trajectory.mean_agreement_rate = trajectory.frontier_accept_rate
        if trajectory.completion_step is None:
            trajectory.completion_step = torch.full(
                (B,), float(total_microsteps), device=device,
            )

    return y, trajectory
