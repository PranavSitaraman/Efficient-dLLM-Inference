"""
GRPO Training for the AOAE Policy (paper §3.5).

Implements Group Relative Policy Optimization following Jazbec et al. (2025)
and Shao et al. (2024, DeepSeekMath), extended to the unified action space.

Key design choices:
  - Multiplicative reward: correctness * speed_penalty - beta * normalized thrashing
  - Group-mean advantage normalization (no std normalization)
  - Clipped surrogate objective with importance sampling
  - No KL regularization (policy trained from scratch)
  - Terminal reward propagated to all preceding steps
"""

import os
import json
import math
import random
import copy
import tempfile
import collections
import torch
import torch.nn as nn
from torch.optim import AdamW
# LambdaLR imported at usage site below (to keep scheduler logic together)
from tqdm import tqdm
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Union

from .checkpoints import (
    GRPO_TRAIN_CONTRACT_VERSION,
    build_grpo_config_fingerprint,
    find_latest_checkpoint,
    load_state_dict_flexible,
    read_grpo_training_metadata,
)
from .inference import aoae_inference, AOAETrajectory
from .speculative_inference import speculative_inference, SpeculativeTrajectory
from .tasks import (
    build_prompt,
    check_math_correctness,
    decode_generated_tokens,
    extract_answer,
    extract_prompt_and_reference,
)
from .runtime_checks import collect_runtime_info

# Both trajectory types share the same reward-relevant attributes; Union lets
# type checkers verify this without forcing a common base class.
AnyTrajectory = Union[AOAETrajectory, SpeculativeTrajectory]
_MAX_IMPORTANCE_LOG_RATIO = 20.0


def _head_set(value: Any) -> Optional[set]:
    if value is None:
        return None
    if isinstance(value, str):
        return {p.strip() for p in value.split(",") if p.strip()}
    return {str(p).strip() for p in value if str(p).strip()}


def _include_heads_for_logprob(cfg: dict) -> Optional[set]:
    gc = cfg.get("grpo", {})
    return _head_set(gc.get("include_heads_in_logprob", gc.get("train_heads")))


# ======================================================================
# Training logger (WandB + JSONL + enriched console output)
# ======================================================================

class _TrainingLogger:
    """
    Unified training logger: optional WandB, mandatory JSONL file, console.

    Activated by ``logging.use_wandb=true`` in config.  All scalar metrics
    logged via ``log_step()`` are:
      - printed to stdout at every ``log_every`` step
      - appended as a JSON line to ``<output_dir>/training_log.jsonl``
      - forwarded to WandB (if enabled)

    Design: the logger is constructed once before the training loop, holds
    the file handle open for efficiency, and is closed in a ``close()`` call.
    """

    def __init__(self, cfg: dict):
        lc = cfg["logging"]
        self._output_dir = str(lc["output_dir"])
        _wandb_val = lc.get("use_wandb", False)
        # Accept: false/True/true (bool) or "offline" / "online" (string)
        if isinstance(_wandb_val, str):
            _wandb_mode = _wandb_val  # "offline", "online", "disabled", etc.
            self._use_wandb = _wandb_val.lower() not in ("false", "disabled", "")
        else:
            _wandb_mode = "online"
            self._use_wandb = bool(_wandb_val)
        os.makedirs(self._output_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self._output_dir, "training_log.jsonl")
        self._jsonl_fh = open(self._jsonl_path, "a", buffering=1)  # line-buffered

        if self._use_wandb:
            try:
                import wandb
                wandb.init(
                    project=str(lc.get("project", "aoae")),
                    name=str(lc.get("run_name", "grpo")),
                    config=cfg,
                    resume="allow",
                    mode=_wandb_mode,
                )
                self._wandb = wandb
                print("[Logger] WandB run initialized.")
            except Exception as e:
                print(f"[Logger] WandB init failed ({e}); falling back to file-only logging.")
                self._wandb = None
                self._use_wandb = False
        else:
            self._wandb = None

    def log_step(self, metrics: Dict[str, Any], step: int) -> None:
        """Write one log entry (dict of scalar metrics + step number)."""
        record = {"global_step": step, **metrics}
        # JSONL
        try:
            self._jsonl_fh.write(json.dumps(record) + "\n")
        except Exception:
            pass
        # WandB
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._jsonl_fh.close()
        except Exception:
            pass
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


# ======================================================================
# Reward computation (Eq. reward from paper)
# ======================================================================

def compute_reward(
    generated_tokens: torch.Tensor,
    reference_answer: List[str],
    tokenizer,
    trajectory: "AnyTrajectory",
    cfg: dict,
    T: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multiplicative reward (Eq. reward) and return a component breakdown.

      R = r(y*, y_hat) * speed_factor
          - beta * normalized_thrash
          - unresolved_penalty_weight * unresolved_fraction
          + cache_quality_weight * mean_cache_F1
          + access_reward_weight * access_F1

    Speed factor (compute-aware):
      effective_flops = (used_steps / T) * (1 - mean_cache_for_speed)
      speed_factor    = (1 - effective_flops)^alpha

    By default, transient K_spec frontier occupancy is logged but not credited
    as free persistent cache. Speculative trajectories instead provide explicit
    aux/verifier compute accounting.

    Args:
        generated_tokens: [B, L_gen] generated response tokens.
        reference_answer: list of B reference answer strings.
        tokenizer:        for decoding generated tokens.
        trajectory:       AOAETrajectory or SpeculativeTrajectory with thrash counts.
        cfg:              config dict.
        T:                total diffusion steps.

    Returns:
        rewards:    [B] per-sample total scalar reward.
        components: dict of [B] per-sample component tensors, containing:
                    correctness, speed_factor, effective_flops, used_steps_frac,
                    mean_cached_fraction, mean_combined_cached_fraction,
                    mean_stable_cached_fraction, mean_spec_cached,
                    thrash_penalty, total_thrash, thrash_rate, thrash_denominator,
                    unresolved_penalty, cache_f1_reward, access_reward.
                    All on the same device as generated_tokens.
    """
    gc = cfg["grpo"]
    alpha = gc["alpha"]
    beta = gc["beta"]
    unresolved_penalty_weight = float(gc.get("unresolved_penalty_weight", 0.0))
    B = generated_tokens.shape[0]
    device = generated_tokens.device

    # --- Correctness term r(y*, y) ---
    correctness = torch.zeros(B, device=device)
    for b in range(B):
        gen_text = decode_generated_tokens(
            tokenizer,
            generated_tokens[b],
            mask_token_id=cfg.get("base_model", {}).get("mask_token_id"),
        )
        correct = check_math_correctness(gen_text, reference_answer[b])
        correctness[b] = 1.0 if correct else 0.0

    if getattr(trajectory, "completion_step", None) is not None:
        used_steps = trajectory.completion_step.to(device).float().clamp(min=1.0, max=float(T))
    else:
        n_active_steps = sum(
            1 for a in trajectory.actions
            if "u_t" in a and a["u_t"].sum() > 0
        )
        used_steps = torch.full((B,), float(max(n_active_steps, 1)), device=device)

    def _mean_fraction(name: str) -> torch.Tensor:
        values = getattr(trajectory, name, [])
        if values:
            return torch.stack([v.to(device) for v in values]).mean(dim=0)
        return torch.zeros(B, device=device)

    mean_combined_cached = _mean_fraction("cached_fractions")
    mean_spec_cached = _mean_fraction("spec_cached_fractions")
    mean_stable_cached = _mean_fraction("stable_cached_fractions")
    if not getattr(trajectory, "stable_cached_fractions", []):
        # Single-cache trajectories predate the two-cache split; their cached
        # fraction corresponds to the only operational cache-like set they have.
        mean_stable_cached = mean_combined_cached

    trajectory_eff = getattr(trajectory, "effective_flops", None)
    if trajectory_eff is not None:
        effective_flops = trajectory_eff.to(device).float().clamp(min=0.0)
        mean_cached = mean_stable_cached
    else:
        # Fallback for legacy/single-model trajectories. K_spec is transient and
        # never credited by default; stable cache credit remains source-gated.
        cache_speed_source = str(gc.get("cache_speed_source", "none")).lower()
        if cache_speed_source in ("stable", "k_stable", "persistent"):
            mean_cached = mean_stable_cached
        elif cache_speed_source in ("spec", "k_spec", "transient"):
            mean_cached = mean_spec_cached
        elif cache_speed_source in ("combined", "union", "all"):
            mean_cached = mean_combined_cached
        else:
            mean_cached = torch.zeros(B, device=device)

        cache_speed_credit_cap = float(gc.get("cache_speed_credit_cap", 1.0))
        mean_cached = mean_cached.clamp(0.0, max(0.0, min(cache_speed_credit_cap, 1.0)))
        used_steps_frac = used_steps / float(T)                               # [B] in [0,1]
        effective_flops = used_steps_frac * (1.0 - mean_cached)               # [B] in [0,1]
    used_steps_frac = used_steps / float(T)
    speed_factor = (1.0 - effective_flops.clamp(0.0, 1.0)).pow(alpha)         # [B] in [0,1]

    # --- Cache thrashing penalty ---
    total_thrash = torch.zeros(B, device=device)
    for thrash_t in trajectory.thrash_counts:
        total_thrash += thrash_t.to(device)
    # Normalize raw invalidation counts before mixing them with bounded terms
    # such as correctness and speed_factor. The raw count remains logged as
    # total_thrash for diagnostics; the reward uses a per-response-token rate by
    # default, so beta has a stable interpretation across sequence lengths.
    norm_mode = gc.get("thrash_normalization", "response_length")
    response_len = max(int(generated_tokens.shape[-1]), 1)
    if isinstance(norm_mode, (int, float)):
        thrash_denominator = torch.full(
            (B,), max(float(norm_mode), 1.0), device=device
        )
    elif str(norm_mode).lower() in ("none", "raw", "count"):
        thrash_denominator = torch.ones(B, device=device)
    elif str(norm_mode).lower() in ("token_steps", "step_tokens", "tl"):
        thrash_denominator = torch.full(
            (B,), max(float(T * response_len), 1.0), device=device
        )
    elif str(norm_mode).lower() in ("used_token_steps", "used_tl"):
        thrash_denominator = (used_steps * float(response_len)).clamp(min=1.0)
    else:
        thrash_denominator = torch.full(
            (B,), float(response_len), device=device
        )
    thrash_rate = total_thrash / thrash_denominator
    thrash_penalty = beta * thrash_rate                                        # [B]

    # --- Reward (base) ---
    reward = correctness * speed_factor - thrash_penalty

    # --- Unresolved-mask penalty ---
    unresolved_penalty = torch.zeros(B, device=device)
    if unresolved_penalty_weight > 0.0:
        final_tokens = getattr(trajectory, "final_tokens", None)
        if final_tokens is not None:
            mask_id = int(cfg["base_model"]["mask_token_id"])
            unresolved_fraction = (final_tokens.to(device) == mask_id).float().mean(dim=-1)
            unresolved_penalty = unresolved_penalty_weight * unresolved_fraction
            reward = reward - unresolved_penalty

    # --- Cache quality F1 reward ---
    #
    # Replaces the old drift_penalty (precision-only) with a two-sided F1 that
    # also rewards committing stable tokens (recall).  Computed in inference loop:
    #   stability(k) = exp(-λ * rel_drift_k)
    #   precision     = mean_{k ∈ K_t}(stability(k))
    #   recall        = Σ_{k ∈ K_t} stability(k) / Σ_all stability(k)
    #   cache_F1      = 2 * precision * recall / (precision + recall)
    cache_q_w = float(gc.get("cache_quality_weight", 0.0))
    cache_f1_reward = torch.zeros(B, device=device)
    if cache_q_w > 0.0:
        cache_f1_steps = getattr(trajectory, "cache_quality_f1", [])
        if cache_f1_steps:
            mean_cache_f1 = torch.stack([f.to(device) for f in cache_f1_steps]).mean(dim=0)
            cache_f1_reward = cache_q_w * mean_cache_f1
            reward = reward + cache_f1_reward

    # --- Dense access-prediction reward ---
    #
    # Provides a dense gradient for the q_t head (access prediction) that is
    # otherwise too sparse to learn from terminal correctness alone.
    access_w = float(gc.get("access_reward_weight", 0.0))
    access_reward = torch.zeros(B, device=device)
    if access_w > 0.0 and hasattr(trajectory, "access_metrics"):
        spec_f1 = float(trajectory.access_metrics.get("access_next_h_spec_f1", 0.0))
        access_reward = torch.full((B,), access_w * spec_f1, device=device)
        reward = reward + access_reward

    components: Dict[str, torch.Tensor] = {
        "correctness":         correctness,
        "speed_factor":        speed_factor,
        "effective_flops":     effective_flops,
        "aux_compute_units":    (
            getattr(trajectory, "aux_compute_units", torch.zeros(B, device=device))
            .to(device).float()
            if getattr(trajectory, "aux_compute_units", None) is not None
            else torch.zeros(B, device=device)
        ),
        "verifier_compute_units": (
            getattr(trajectory, "verifier_compute_units", torch.zeros(B, device=device))
            .to(device).float()
            if getattr(trajectory, "verifier_compute_units", None) is not None
            else torch.zeros(B, device=device)
        ),
        "baseline_compute_units": (
            getattr(trajectory, "baseline_compute_units", torch.zeros(B, device=device))
            .to(device).float()
            if getattr(trajectory, "baseline_compute_units", None) is not None
            else torch.zeros(B, device=device)
        ),
        "used_steps_frac":     used_steps_frac,
        "mean_cached_fraction": mean_cached,
        "mean_combined_cached_fraction": mean_combined_cached,
        "mean_stable_cached_fraction": mean_stable_cached,
        "thrash_penalty":       thrash_penalty,
        "total_thrash":         total_thrash,
        "thrash_rate":          thrash_rate,
        "thrash_denominator":   thrash_denominator,
        "unresolved_penalty":   unresolved_penalty,
        "cache_f1_reward":      cache_f1_reward,
        "access_reward":        access_reward,
        "mean_spec_cached":     mean_spec_cached,   # transient draft frontier
    }

    return reward, components


# ======================================================================
# GRPO objective (Eq. grpo from paper)
# ======================================================================

def compute_grpo_loss(
    policy,
    soft_mask_module,
    trajectories: List[Dict],
    advantages: torch.Tensor,
    clip_eps: float,
    include_heads_in_logprob: Optional[set] = None,
) -> torch.Tensor:
    """
    Compute clipped GRPO surrogate loss.

    For each trajectory g in the group, recompute log pi_phi(a_t | s_t)
    under the current policy, then compute:

      L = -1/G sum_g 1/(T-T_hat_g) sum_t min(rho * A, clip(rho) * A)

    Args:
        policy:          current policy (parameters being updated).
        soft_mask_module: soft-masked state builder.
        trajectories:    list of G trajectory dicts, each with:
                         "H_t_list", "mask_ind_list", "step_fracs",
                         "actions_list", "old_log_probs"
        advantages:      [G] per-trajectory advantage.
        clip_eps:        clipping threshold epsilon.

    Returns:
        loss: scalar (to be minimized).
    """
    G = len(trajectories)
    total_loss = torch.tensor(0.0, device=advantages.device)

    for g in range(G):
        traj = trajectories[g]
        n_steps = len(traj["actions_list"])
        if n_steps == 0:
            continue

        step_loss = torch.tensor(0.0, device=advantages.device)

        sm_inner = soft_mask_module.module if hasattr(soft_mask_module, "module") else soft_mask_module

        for t_idx in range(n_steps):
            # Recompute h_t from stored intermediates so that ω_s/ω_a/ω_b receive
            # gradients.  weighted_embeds and entropy are detached rollout data;
            # autograd flows through the live ω parameters in recompute_h_t().
            # Fall back to H_t_list for legacy/manual trajectory views that
            # predate the lightweight omega-gradient storage path.
            weighted_list = traj.get("weighted_embeds_list") or []
            entropy_list = traj.get("entropy_list") or []
            if t_idx < len(weighted_list) and t_idx < len(entropy_list):
                weighted_embeds = weighted_list[t_idx]  # [1, L, D]
                entropy_t = entropy_list[t_idx]         # [1, L]
                H_t = sm_inner.recompute_h_t(weighted_embeds, entropy_t)
            else:
                H_t = traj["H_t_list"][t_idx]
            mask_ind = traj["mask_ind_list"][t_idx]  # [1, L]
            step_frac = traj["step_fracs"][t_idx]
            actions = traj["actions_list"][t_idx]     # dict of [1, L]
            old_lp = traj["old_log_probs"][t_idx]    # [1]

            # Recompute log prob under current policy
            # policy() goes through DDP forward hook for gradient sync;
            # log_prob is a non-forward method, access via .module if DDP-wrapped
            q_scores = traj.get("quality_scores_list", [None] * n_steps)[t_idx]
            age_list = traj.get("age_feature_list", None)
            age_feat = age_list[t_idx] if age_list else None
            last_action_list = traj.get("last_action_feature_list", None)
            last_action_feat = last_action_list[t_idx] if last_action_list else None
            agreement_list = traj.get("agreement_list", None)
            agreement = agreement_list[t_idx].float() if agreement_list else None
            policy_out = policy(
                H_t, mask_ind, step_frac,
                quality_scores=q_scores,
                agreement=agreement,
                age_feature=age_feat,
                last_action_feature=last_action_feat,
            )
            pol_inner = policy.module if hasattr(policy, 'module') else policy
            new_lp = pol_inner.log_prob(
                policy_out,
                actions,
                include_heads=include_heads_in_logprob,
            )  # [1]

            # Importance ratio
            log_ratio = new_lp - old_lp
            if not torch.isfinite(log_ratio).all():
                # NaN/inf log_ratio means the policy has NaN parameters (from a
                # previous gradient explosion).  Skip this step's contribution
                # rather than crashing — the NaN grad guard above will zero out
                # the offending gradients before the next optimizer step.
                continue
            rho = torch.exp(log_ratio.clamp(
                min=-_MAX_IMPORTANCE_LOG_RATIO,
                max=_MAX_IMPORTANCE_LOG_RATIO,
            ))  # [1]

            # Clipped surrogate
            A_g = advantages[g]
            surr1 = rho * A_g
            surr2 = torch.clamp(rho, 1.0 - clip_eps, 1.0 + clip_eps) * A_g
            step_loss = step_loss + torch.min(surr1, surr2).squeeze()

        # Normalize by number of steps
        step_loss = step_loss / max(n_steps, 1)
        total_loss = total_loss + step_loss

    # Average over group, negate for minimization
    return -total_loss / max(G, 1)


def normalize_group_advantages(
    rewards: torch.Tensor,
    *,
    normalize_std: bool = False,
) -> torch.Tensor:
    """Center rewards within the rollout group, with optional std scaling.

    The paper objective uses ``A^g = R^g - mean(R)``. Standard-deviation
    normalization is kept as an opt-in compatibility flag for experiments that
    want the older behavior.
    """
    advantages = rewards - rewards.mean()
    if not normalize_std:
        return advantages

    std = rewards.std(unbiased=False)
    if torch.isfinite(std) and std > 1e-8:
        advantages = advantages / std
    return advantages


def build_rollout_cfg(cfg: dict) -> dict:
    """Return the training-time rollout config with optional GRPO overrides."""
    rollout_cfg = copy.deepcopy(cfg)
    inference_cfg = rollout_cfg.setdefault("inference", {})
    grpo_cfg = rollout_cfg.get("grpo", {})

    rollout_steps = grpo_cfg.get("rollout_steps")
    if rollout_steps is not None:
        inference_cfg["steps"] = int(rollout_steps)

    rollout_gen_length = grpo_cfg.get("rollout_gen_length")
    if rollout_gen_length is not None:
        inference_cfg["gen_length"] = int(rollout_gen_length)

    return rollout_cfg


def configure_grpo_trainability(policy, soft_mask_module, cfg: dict) -> List[nn.Parameter]:
    """Freeze deterministic heads and return parameters for the optimizer.

    The canonical draft-frontier setup trains the cache/access policy while
    unmasking follows a fixed drafter-confidence schedule and rejection is
    handled by the verifier. Full four-head training remains available by
    setting ``grpo.train_heads`` to include ``unmask`` and ``remask``.
    """
    gc = cfg.get("grpo", {})
    train_heads = _head_set(gc.get("train_heads"))
    pol_inner = policy.module if hasattr(policy, "module") else policy
    if train_heads is not None:
        head_modules = {
            "unmask": getattr(pol_inner, "head_unmask", None),
            "u_t": getattr(pol_inner, "head_unmask", None),
            "remask": getattr(pol_inner, "head_remask", None),
            "r_t": getattr(pol_inner, "head_remask", None),
            "cache": getattr(pol_inner, "head_cache", None),
            "kappa_t": getattr(pol_inner, "head_cache", None),
            "access": getattr(pol_inner, "head_access", None),
            "q_t": getattr(pol_inner, "head_access", None),
            "boundary": getattr(pol_inner, "head_boundary", None),
            "ell_t": getattr(pol_inner, "head_boundary", None),
        }
        for module in {m for m in head_modules.values() if m is not None}:
            for p in module.parameters():
                p.requires_grad_(False)
        for name, module in head_modules.items():
            if module is not None and name in train_heads:
                for p in module.parameters():
                    p.requires_grad_(True)

    if not bool(gc.get("train_soft_mask", True)):
        sm_inner = soft_mask_module.module if hasattr(soft_mask_module, "module") else soft_mask_module
        for p in sm_inner.parameters():
            p.requires_grad_(False)

    return [p for p in list(policy.parameters()) + list(soft_mask_module.parameters()) if p.requires_grad]


def split_group_trajectory(trajectory: Any, group_size: int) -> List[Dict[str, Any]]:
    """Split a batched AOAE/speculative trajectory into per-sample GRPO views."""
    trajectories: List[Dict[str, Any]] = []

    def _slice_tensor(value: Optional[torch.Tensor], index: int):
        if value is None:
            return None
        return value[index:index + 1].detach().clone()

    for g in range(group_size):
        traj_data = {
            "actions_list": [
                {key: _slice_tensor(value, g) for key, value in step.items()}
                for step in trajectory.actions
            ],
            "old_log_probs": [_slice_tensor(lp, g) for lp in trajectory.log_probs],
            "H_t_list": [_slice_tensor(h_t, g) for h_t in trajectory.H_t_list],
            "weighted_embeds_list": [
                _slice_tensor(we, g) for we in getattr(trajectory, "weighted_embeds_list", [])
            ],
            "entropy_list": [
                _slice_tensor(ent, g) for ent in getattr(trajectory, "entropy_list", [])
            ],
            "mask_ind_list": [_slice_tensor(mask_ind, g) for mask_ind in trajectory.mask_ind_list],
            "quality_scores_list": [
                _slice_tensor(q_scores, g) for q_scores in trajectory.quality_scores_list
            ],
            "age_feature_list": [
                _slice_tensor(age_feat, g) for age_feat in getattr(trajectory, "age_feature_list", [])
            ],
            "last_action_feature_list": [
                _slice_tensor(last_action_feat, g)
                for last_action_feat in getattr(trajectory, "last_action_feature_list", [])
            ],
            "step_fracs": list(trajectory.step_fracs),
            "access_metrics": dict(getattr(trajectory, "access_metrics", {})),
        }
        agreement_list = getattr(trajectory, "agreement_list", None)
        if agreement_list:
            traj_data["agreement_list"] = [
                _slice_tensor(agreement, g) for agreement in agreement_list
            ]
        trajectories.append(traj_data)

    return trajectories


# ======================================================================
# Rollout collection
# ======================================================================

@torch.no_grad()
def collect_rollout_group(
    base_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    reference_answers: List[str],
    cfg: dict,
    tokenizer,
    dual_model=None,
    rollout_cfg: Optional[dict] = None,
) -> Tuple[List[Dict], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Collect G rollout trajectories for a single prompt batch.

    Args:
        base_model: LLaDABaseModel (used when dual_model is None).
        dual_model: DualModelWrapper (used for speculative mode).
        (other args as before)

    Returns:
        trajectories:      list of G trajectory dicts (for GRPO loss).
        rewards:           [G] per-trajectory total rewards.
        advantages:        [G] group-mean normalized advantages.
        reward_components: dict of [G] per-trajectory component tensors.
    """
    gc = cfg["grpo"]
    rollout_cfg = rollout_cfg if rollout_cfg is not None else build_rollout_cfg(cfg)
    G = gc["group_size"]
    T = rollout_cfg["inference"]["steps"]

    B = prompt_ids.shape[0]
    assert B == 1, "Collect rollouts one prompt at a time, then group."
    use_speculative = (dual_model is not None)
    repeated_prompt_ids = prompt_ids.repeat(G, 1)
    repeated_references = [reference_answers[0] for _ in range(G)]

    if use_speculative:
        output_ids, trajectory = speculative_inference(
            dual_model=dual_model,
            policy=policy,
            soft_mask_module=soft_mask_module,
            prism_adapter=prism_adapter,
            prompt_ids=repeated_prompt_ids,
            cfg=rollout_cfg,
            record_trajectory=True,
            policy_temperature=gc["policy_temperature"],
        )
    else:
        output_ids, trajectory = aoae_inference(
            base_model=base_model,
            policy=policy,
            soft_mask_module=soft_mask_module,
            prism_adapter=prism_adapter,
            prompt_ids=repeated_prompt_ids,
            cfg=rollout_cfg,
            record_trajectory=True,
            policy_temperature=gc["policy_temperature"],
        )

    gen_tokens = output_ids[:, prompt_ids.shape[1]:]
    rewards_t, reward_components_t = compute_reward(
        gen_tokens,
        repeated_references,
        tokenizer,
        trajectory,
        rollout_cfg,
        T,
    )
    rewards_t = rewards_t.detach().cpu()
    reward_components_t = {k: v.detach().cpu() for k, v in reward_components_t.items()}

    trajectories = split_group_trajectory(trajectory, G)
    for g, traj_data in enumerate(trajectories):
        traj_data["reward"] = float(rewards_t[g].item())

    advantages = normalize_group_advantages(
        rewards_t,
        normalize_std=bool(gc.get("normalize_advantage_std", False)),
    )

    return trajectories, rewards_t, advantages, reward_components_t


# ======================================================================
# Main training loop
# ======================================================================
def train(cfg: dict, resume_from: Optional[str] = None):
    """
    Main GRPO training entrypoint.

    1. Load base model (frozen), tokenizer, policy, soft-mask, PRISM.
    2. Load training data (prompts + reference answers).
    3. For each epoch:
       a. For each prompt batch:
          - Collect G rollout trajectories.
          - Compute rewards and advantages.
          - Run GRPO policy gradient update.
       b. Periodically evaluate on held-out set.
    """
    from .models.base_model import LLaDABaseModel
    from .models.soft_mask import SoftMaskedState
    from .models.policy import AOAEPolicy
    from .models.prism import PRISMAdapter
    from datasets import load_dataset

    gc = cfg["grpo"]
    dc = cfg["data"]
    lc = cfg["logging"]

    # GRPO is optional — skip if disabled in config
    if not gc.get("enabled", True):
        rank = cfg.get("_dist", {}).get("rank", 0) if cfg.get("_dist") else 0
        if rank == 0:
            print("[GRPO] grpo.enabled=false — skipping GRPO training.")
        return

    # --- Setup ---
    dist_info = cfg.get("_dist", None)
    is_distributed = dist_info is not None
    rank = dist_info["rank"] if is_distributed else 0
    local_rank = dist_info["local_rank"] if is_distributed else 0
    world_size = dist_info["world_size"] if is_distributed else 1
    is_main = rank == 0

    torch.manual_seed(cfg["hardware"]["seed"] + rank)
    random.seed(cfg["hardware"]["seed"] + rank)
    np.random.seed(cfg["hardware"]["seed"] + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"Device: {device}  (world_size={world_size})")

    # --- Load base model (or dual model for speculative training) ---
    use_dual = cfg["base_model"].get("backend") == "dual"
    dual_model = None
    base_model = None

    prism = None
    policy = None
    soft_mask = None
    optimizer = None
    scheduler = None
    _logger = None
    global_step = 0
    accum_step = 0
    start_epoch = 0
    best_reward = -float("inf")
    current_epoch = 0

    def _save_checkpoint(path: str, *, epoch_idx: int, step_value: int, accum_value: int, best_value=None):
        if policy is None or soft_mask is None or optimizer is None or scheduler is None:
            return
        pol_sd = policy.module.state_dict() if hasattr(policy, 'module') else policy.state_dict()
        sm_sd = soft_mask.module.state_dict() if hasattr(soft_mask, 'module') else soft_mask.state_dict()
        payload = {
            "policy": pol_sd,
            "soft_mask": sm_sd,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step_value,
            "accum_step": accum_value,
            "epoch": epoch_idx,
        }
        if best_value is not None:
            payload["best_reward"] = best_value
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".",
            suffix=".tmp",
            dir=os.path.dirname(path) or ".",
        )
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    try:
        if use_dual:
            from .models.dual_model import DualModelWrapper
            if is_main:
                print("Loading dual model (hard aux + soft primary, single-copy)...\n")
            dual_model = DualModelWrapper(cfg)
            dual_model = dual_model.to(device)
            tokenizer = dual_model.tokenizer
            mask_id = cfg["base_model"]["mask_token_id"]
            embed_w = dual_model.get_embedding_weight()
            embed_dim = embed_w.shape[1]
        else:
            # Single-model mode — override backend to HF if vLLM unavailable
            cfg_grpo = cfg.copy()
            if cfg_grpo["base_model"]["backend"] in ("soft_moe", "dinfer"):
                try:
                    _ = torch.ops._moe_C.moe_align_block_size
                except AttributeError:
                    if is_main:
                        print(
                            f"[WARN] vLLM MoE kernels unavailable; overriding backend "
                            f"'{cfg_grpo['base_model']['backend']}' → 'hf' for GRPO training.\n"
                        )
                    cfg_grpo["base_model"] = dict(cfg_grpo["base_model"])
                    cfg_grpo["base_model"]["backend"] = "hf"

            if is_main:
                print("Loading base model...\n")
            base_model = LLaDABaseModel(cfg_grpo)
            base_model = base_model.to(device)
            tokenizer = base_model.tokenizer
            mask_id = cfg["base_model"]["mask_token_id"]
            embed_w = base_model.get_embedding_weight()
            embed_dim = embed_w.shape[1]

        # --- Initialize modules ---
        soft_mask = SoftMaskedState(cfg, embed_w).to(device)
        soft_mask.set_mask_embedding(mask_id)

        policy = AOAEPolicy(cfg, input_dim=embed_dim).to(device)
        n_params = sum(p.numel() for p in policy.parameters())
        if is_main:
            print(f"Policy parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

        # Freeze deterministic/non-canonical heads before DDP construction.
        # DDP snapshots the trainable parameter set when it is built; freezing
        # heads afterward leaves reducer buckets expecting gradients for heads
        # intentionally excluded from the GRPO likelihood.
        configure_grpo_trainability(policy, soft_mask, cfg)

        # Wrap policy in DDP for multi-GPU gradient sync
        if is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            policy = DDP(policy, device_ids=[local_rank])
            # Only wrap soft_mask in DDP if it has parameters that require gradients.
            # When train_soft_mask=false, configure_grpo_trainability freezes all its
            # params, and DDP raises RuntimeError on a fully-frozen module.
            if any(p.requires_grad for p in soft_mask.parameters()):
                soft_mask = DDP(soft_mask, device_ids=[local_rank])

        # PRISM adapter (optional — load if checkpoint exists, else skip)
        prism_path = os.path.join(lc["output_dir"], "prism_adapter.pt")
        if os.path.exists(prism_path):
            if is_main:
                print(f"Loading PRISM adapter from {prism_path}")
            prism = PRISMAdapter(cfg, embed_dim).to(device)
            prism.load_state_dict(torch.load(prism_path, map_location=device))
            prism.eval()
        else:
            if is_main:
                print("No PRISM adapter found — policy will not receive quality scores.")
            prism = None

        # --- Optimizer (over configured trainable params including DDP wrappers) ---
        trainable_params = [
            p for p in list(policy.parameters()) + list(soft_mask.parameters())
            if p.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError(
                "No trainable GRPO parameters. Check grpo.train_heads and grpo.train_soft_mask."
            )
        if is_main and gc.get("train_heads") is not None:
            print(
                "[GRPO] train_heads="
                f"{list(_head_set(gc.get('train_heads')) or [])}; "
                f"include_logprob={list(_include_heads_for_logprob(cfg) or [])}; "
                f"train_soft_mask={bool(gc.get('train_soft_mask', True))}"
            )
        optimizer = AdamW(
            trainable_params,
            lr=gc["lr"],
            weight_decay=gc["weight_decay"],
        )
        # Linear warmup + cosine decay schedule
        from torch.optim.lr_scheduler import LambdaLR

        def _lr_lambda(step):
            if step < gc["warmup_steps"]:
                return step / max(gc["warmup_steps"], 1)
            progress = (step - gc["warmup_steps"]) / max(gc["max_steps"] - gc["warmup_steps"], 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, _lr_lambda)

        # --- Load training data ---
        if is_main:
            print("Loading training data...")
        train_ds = load_dataset(
            dc["train_dataset"], split=dc["train_split"]
        )
        if dc["train_max_samples"]:
            train_ds = train_ds.select(range(min(dc["train_max_samples"], len(train_ds))))
        rollout_cfg = build_rollout_cfg(cfg)
        if is_main:
            prompt_budget = len(train_ds) * gc["epochs"]
            print(
                f"GRPO rollout budget: group_size={gc['group_size']}  "
                f"steps={rollout_cfg['inference']['steps']}  "
                f"gen_length={rollout_cfg['inference']['gen_length']}  "
                f"max_steps={gc['max_steps']}  "
                f"prompt_budget≈{prompt_budget}"
            )

        # --- Training loop ---
        os.makedirs(lc["output_dir"], exist_ok=True)
        grad_accum = gc["grad_accum_steps"]
        best_path = os.path.join(lc["output_dir"], "policy_best.pt")

        # Initialize logger (rank 0 only to avoid duplicate writes in DDP)
        _logger = _TrainingLogger(cfg) if is_main else None

        # --- Resume from checkpoint ---
        # Normalize sentinel strings to None so callers can pass "fresh"/"none"/""
        if resume_from in ("fresh", "none", ""):
            resume_from = None
        if resume_from == "auto":
            resume_from = find_latest_checkpoint(lc["output_dir"])
            if resume_from and is_main:
                print(f"Auto-detected checkpoint: {resume_from}")

        if resume_from and os.path.isfile(resume_from):
            if is_main:
                print(f"Resuming from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=device)

            # Restore model weights
            pol_inner = policy.module if hasattr(policy, 'module') else policy
            load_state_dict_flexible(pol_inner, ckpt["policy"], "policy")
            sm_inner = soft_mask.module if hasattr(soft_mask, 'module') else soft_mask
            load_state_dict_flexible(sm_inner, ckpt["soft_mask"], "soft_mask")

            # Restore optimizer
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])

            # Restore scheduler
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

            # Restore counters
            if "step" in ckpt:
                global_step = ckpt["step"]
            if "accum_step" in ckpt:
                accum_step = ckpt["accum_step"]
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
            if "best_reward" in ckpt and ckpt["best_reward"] is not None:
                best_reward = float(ckpt["best_reward"])
            else:
                metadata = read_grpo_training_metadata(lc["output_dir"])
                if isinstance(metadata, dict):
                    try:
                        best_from_metadata = metadata.get("best_reward")
                        if best_from_metadata is not None:
                            best_reward = float(best_from_metadata)
                    except (TypeError, ValueError):
                        pass

            if is_main:
                print(f"  Resumed at global_step={global_step}, epoch={start_epoch}, "
                      f"accum_step={accum_step}")
            del ckpt
        elif resume_from:
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_from}. "
                f"Use --resume auto to auto-detect, or pass a valid path."
            )

        should_optimize = global_step < gc["max_steps"]
        if is_main and not should_optimize:
            print(
                "Resume checkpoint already reached max_steps; skipping further GRPO "
                "optimizer steps and materializing final artifacts only."
            )

        for epoch in range(start_epoch, gc["epochs"]):
            if not should_optimize:
                break
            current_epoch = epoch
            if is_main:
                print(f"\n=== Epoch {epoch + 1}/{gc['epochs']} ===")
            epoch_rewards = []
            valid_samples_this_epoch = 0
            optimizer_steps_this_epoch = 0
            # Running buffers for reward component logging (reset each log window)
            _component_bufs: Dict[str, List[float]] = collections.defaultdict(list)
            _last_grpo_loss: float = 0.0

            indices = list(range(len(train_ds)))
            random.shuffle(indices)

            # Shard data across ranks for distributed training
            if is_distributed:
                indices = indices[rank::world_size]

            pbar = tqdm(range(0, len(indices), gc["batch_size"]), desc="Training", disable=not is_main)
            for i in pbar:
                if global_step >= gc["max_steps"]:
                    break
                batch_indices = indices[i : i + gc["batch_size"]]

                for idx in batch_indices:
                    sample = train_ds[idx]

                    # Prepare prompt
                    question, reference = extract_prompt_and_reference(sample)
                    if not question or not reference:
                        continue
                    valid_samples_this_epoch += 1

                    prompt_text, add_special_tokens = build_prompt(tokenizer, question, cfg)
                    prompt_ids = tokenizer.encode(
                        prompt_text,
                        add_special_tokens=add_special_tokens,
                        max_length=dc["max_prompt_len"],
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    if prompt_ids.dim() == 1:
                        prompt_ids = prompt_ids.unsqueeze(0)

                    # Collect G rollouts (returns reward component breakdown too)
                    trajectories, rewards, advantages, reward_components = collect_rollout_group(
                        base_model=base_model,
                        policy=policy,
                        soft_mask_module=soft_mask,
                        prism_adapter=prism,
                        prompt_ids=prompt_ids,
                        reference_answers=[reference],
                        cfg=cfg,
                        tokenizer=tokenizer,
                        dual_model=dual_model,
                        rollout_cfg=rollout_cfg,
                    )

                    epoch_rewards.append(rewards.mean().item())

                    # Accumulate per-rollout component values for logging
                    G = gc["group_size"]
                    for comp_key, comp_val in reward_components.items():
                        for g in range(min(G, comp_val.shape[0])):
                            _component_bufs[comp_key].append(float(comp_val[g].item()))

                    # Clipped GRPO surrogate with importance sampling
                    # Pass DDP-wrapped modules so .backward() syncs gradients
                    grpo_loss = compute_grpo_loss(
                        policy=policy,
                        soft_mask_module=soft_mask,
                        trajectories=trajectories,
                        advantages=advantages.to(device),
                        clip_eps=gc["clip_eps"],
                        include_heads_in_logprob=_include_heads_for_logprob(cfg),
                    )
                    _last_grpo_loss = float(grpo_loss.item())
                    # Scale loss for gradient accumulation
                    scaled_loss = grpo_loss / (len(batch_indices) * grad_accum)
                    scaled_loss.backward()

                accum_step += 1

                # --- Optimizer step after grad_accum mini-batches ---
                if accum_step % grad_accum == 0:
                    # Detect NaN/inf gradients before clipping; zero them out so a
                    # single bad sample cannot infect the policy parameters with NaN.
                    # (Root cause: unclamped importance ratio exp(new_lp - old_lp) can
                    # be inf when the policy has changed significantly since rollout
                    # collection, producing inf loss → inf grad → NaN after clip.)
                    _nan_grad_params = []
                    for _n, _p in policy.named_parameters():
                        if _p.grad is not None and not torch.isfinite(_p.grad).all():
                            _nan_grad_params.append(_n)
                            _p.grad = None
                    if _nan_grad_params and is_main:
                        print(
                            f"[WARN] step={global_step}: NaN/inf gradients zeroed in "
                            f"{len(_nan_grad_params)} policy params: "
                            f"{_nan_grad_params[:5]}"
                            + (" ..." if len(_nan_grad_params) > 5 else "")
                        )
                    nn.utils.clip_grad_norm_(trainable_params, gc["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    optimizer_steps_this_epoch += 1

                    # --- Logging ---
                    if global_step % lc["log_every"] == 0 and is_main:
                        log_window = lc["log_every"]
                        recent_r = epoch_rewards[-log_window:] if epoch_rewards else [0.0]
                        avg_r = float(np.mean(recent_r))
                        std_r = float(np.std(recent_r)) if len(recent_r) > 1 else 0.0
                        frac_pos = float(np.mean([1.0 if r > 0 else 0.0 for r in recent_r]))
                        cur_lr = float(scheduler.get_last_lr()[0])

                        def _buf_mean(k: str) -> float:
                            vs = _component_bufs.get(k, [])
                            return float(np.mean(vs)) if vs else 0.0

                        # Compact console line + secondary detail line
                        print(
                            f"  step={global_step:5d}  "
                            f"reward={avg_r:+.4f}±{std_r:.4f}  "
                            f"frac_pos={frac_pos:.2f}  "
                            f"loss={_last_grpo_loss:+.4f}  "
                            f"lr={cur_lr:.2e}"
                        )
                        print(
                            f"    correct={_buf_mean('correctness'):.3f}  "
                            f"speed={_buf_mean('speed_factor'):.3f}  "
                            f"eff_flops={_buf_mean('effective_flops'):.3f}  "
                            f"steps_frac={_buf_mean('used_steps_frac'):.3f}  "
                            f"speed_cache={_buf_mean('mean_cached_fraction'):.3f}  "
                            f"stable={_buf_mean('mean_stable_cached_fraction'):.3f}  "
                            f"spec={_buf_mean('mean_spec_cached'):.3f}"
                        )
                        print(
                            f"    thrash_pen={_buf_mean('thrash_penalty'):.4f}  "
                            f"thrash_rate={_buf_mean('thrash_rate'):.3f}  "
                            f"thrash_cnt={_buf_mean('total_thrash'):.1f}  "
                            f"cacheF1_rew={_buf_mean('cache_f1_reward'):.4f}  "
                            f"unresolved_pen={_buf_mean('unresolved_penalty'):.4f}"
                        )

                        # Structured log entry for WandB + JSONL
                        log_metrics: Dict[str, Any] = {
                            "epoch": epoch,
                            "lr": cur_lr,
                            "grpo_loss": _last_grpo_loss,
                            # Reward summary
                            "reward/total_mean": avg_r,
                            "reward/total_std": std_r,
                            "reward/frac_positive": frac_pos,
                            # Reward component means
                            "reward/correctness": _buf_mean("correctness"),
                            "reward/speed_factor": _buf_mean("speed_factor"),
                            "reward/effective_flops": _buf_mean("effective_flops"),
                            "reward/used_steps_frac": _buf_mean("used_steps_frac"),
                            "reward/thrash_penalty": _buf_mean("thrash_penalty"),
                            "reward/thrash_rate": _buf_mean("thrash_rate"),
                            "reward/cache_f1_reward": _buf_mean("cache_f1_reward"),
                            "reward/unresolved_penalty": _buf_mean("unresolved_penalty"),
                            "reward/access_reward": _buf_mean("access_reward"),
                            # Cache metrics (useful for WandB curves)
                            "cache/speed_credit_fraction": _buf_mean("mean_cached_fraction"),
                            "cache/mean_stable_fraction": _buf_mean("mean_stable_cached_fraction"),
                            "cache/mean_spec_fraction": _buf_mean("mean_spec_cached"),
                            "cache/mean_combined_fraction": _buf_mean("mean_combined_cached_fraction"),
                            "cache/total_thrash_count": _buf_mean("total_thrash"),
                            "cache/cache_f1": (
                                _buf_mean("cache_f1_reward") / max(float(gc.get("cache_quality_weight", 1e-8)), 1e-8)
                            ),
                        }
                        if _logger is not None:
                            _logger.log_step(log_metrics, step=global_step)

                        # Reset component accumulators for next window
                        _component_bufs.clear()

                    # --- Save checkpoint (rank 0 only) ---
                    if is_main and (global_step == 1 or global_step % lc["save_every"] == 0):
                        ckpt_path = os.path.join(lc["output_dir"], "policy_latest.pt")
                        _save_checkpoint(
                            ckpt_path,
                            epoch_idx=epoch,
                            step_value=global_step,
                            accum_value=accum_step,
                            best_value=None if best_reward == -float("inf") else best_reward,
                        )
                        print(f"  Saved checkpoint: {ckpt_path}")

                    if global_step >= gc["max_steps"]:
                        break

            if valid_samples_this_epoch == 0:
                sample0 = train_ds[0] if len(train_ds) > 0 else {}
                sample_keys = sorted(list(sample0.keys())) if isinstance(sample0, dict) else []
                sample_preview = str(sample0)[:500]
                raise RuntimeError(
                    "No valid (prompt, reference) samples were found for this epoch. "
                    f"Dataset schema does not match expected fields. "
                    f"Sample keys={sample_keys}. Sample preview={sample_preview}"
                )

            if is_main and epoch_rewards and optimizer_steps_this_epoch > 0:
                epoch_mean_reward = float(np.mean(epoch_rewards))
                if epoch_mean_reward > best_reward:
                    best_reward = epoch_mean_reward
                    _save_checkpoint(
                        best_path,
                        epoch_idx=epoch,
                        step_value=global_step,
                        accum_value=accum_step,
                        best_value=best_reward,
                    )
                    print(f"  Saved best checkpoint: {best_path} (epoch_mean_reward={best_reward:.4f})")

            if global_step >= gc["max_steps"]:
                break

        # --- Save final model (rank 0 only) ---
        if is_main:
            pol_sd = policy.module.state_dict() if hasattr(policy, 'module') else policy.state_dict()
            sm_sd = soft_mask.module.state_dict() if hasattr(soft_mask, 'module') else soft_mask.state_dict()
            final_path = os.path.join(lc["output_dir"], "policy_final.pt")
            torch.save({
                "policy": pol_sd,
                "soft_mask": sm_sd,
            }, final_path)
            metadata_path = os.path.join(lc["output_dir"], "grpo_training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "stage": "grpo",
                        "output_dir": lc["output_dir"],
                        "backend": cfg["base_model"].get("backend", ""),
                        "used_dual_model": bool(use_dual),
                        "train_dataset": dc["train_dataset"],
                        "train_split": dc["train_split"],
                        "train_max_samples": dc.get("train_max_samples"),
                        "epochs": int(gc["epochs"]),
                        "max_steps": int(gc["max_steps"]),
                        "completed_steps": int(global_step),
                        "grad_accum_steps": int(gc["grad_accum_steps"]),
                        "group_size": int(gc["group_size"]),
                        "rollout_steps": int(rollout_cfg["inference"]["steps"]),
                        "rollout_gen_length": int(rollout_cfg["inference"]["gen_length"]),
                        "normalize_advantage_std": bool(gc.get("normalize_advantage_std", False)),
                        "best_reward": None if best_reward == -float("inf") else float(best_reward),
                        "best_checkpoint": best_path if os.path.exists(best_path) else None,
                        "final_checkpoint": final_path,
                        "resume_from": resume_from,
                        "seed": int(cfg["hardware"]["seed"]),
                        "train_contract_version": int(GRPO_TRAIN_CONTRACT_VERSION),
                        "config_fingerprint": build_grpo_config_fingerprint(cfg),
                        "runtime": collect_runtime_info(),
                    },
                    f,
                    indent=2,
                )
            print(f"\nTraining complete. Final model saved to {final_path}")
            print(f"GRPO metadata saved to {metadata_path}")
            if _logger is not None:
                print(f"Training log (JSONL) saved to {_logger._jsonl_path}")
    except KeyboardInterrupt:
        if is_main and policy is not None and soft_mask is not None:
            interrupt_path = os.path.join(lc["output_dir"], "policy_interrupt.pt")
            _save_checkpoint(
                interrupt_path,
                epoch_idx=current_epoch,
                step_value=global_step,
                accum_value=accum_step,
                best_value=None if best_reward == -float("inf") else best_reward,
            )
            print(f"\nTraining interrupted. Recovery checkpoint saved to {interrupt_path}")
        raise
    finally:
        if _logger is not None:
            _logger.close()
        if dual_model is not None:
            dual_model.close()
        if base_model is not None:
            base_model.close()
