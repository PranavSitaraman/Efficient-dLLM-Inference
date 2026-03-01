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

from .cache import DKVCacheManager
from .models.composed_prediction import compose_prediction_dual
from .models.dual_model import DualModelWrapper, DualModelOutput


@dataclass
class SpeculativeTrajectory:
    """Stores a full speculative inference trajectory for GRPO training."""
    actions: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    policy_outputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    thrash_counts: List[torch.Tensor] = field(default_factory=list)
    H_t_list: List[torch.Tensor] = field(default_factory=list)
    mask_ind_list: List[torch.BoolTensor] = field(default_factory=list)
    quality_scores_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    agreement_list: List[torch.Tensor] = field(default_factory=list)
    step_fracs: List[float] = field(default_factory=list)
    final_tokens: Optional[torch.Tensor] = None
    completion_step: Optional[torch.Tensor] = None
    # Aggregate stats
    mean_agreement_rate: float = 0.0
    total_cache_hits: int = 0
    total_cache_misses: int = 0


def speculative_inference(
    dual_model: DualModelWrapper,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    record_trajectory: bool = False,
    policy_temperature: float = 1.0,
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
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)

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

    # --- dKV-Cache ---
    cache_mgr = DKVCacheManager(B, L_gen, device) if use_cache else None

    trajectory = SpeculativeTrajectory() if record_trajectory else None
    agreement_rates = []

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
        need_hidden = (prism_adapter is not None)
        dual_out = dual_model.dual_forward_resp(
            y, resp_slice, need_hidden=need_hidden,
        )
        agreement_rates.append(dual_out.agreement_rate)

        resp_logits = dual_out.primary_logits      # [B, L_gen, V]
        aux_logits = dual_out.auxiliary_logits      # [B, L_gen, V]
        agreement = dual_out.agreement             # [B, L_gen] bool

        # --- PRISM quality scores ---
        q_scores = None
        if prism_adapter is not None and dual_out.primary_hidden is not None:
            with torch.no_grad():
                q_scores = prism_adapter(dual_out.primary_hidden.float())

        # --- Construct soft-masked state from PRIMARY logits ---
        H_t, confidence, entropy = soft_mask_module(
            resp_logits, mask_ind, step_frac
        )

        # --- Policy forward (with agreement signal) ---
        policy_out = policy(
            H_t, mask_ind, step_frac,
            temperature=policy_temperature,
            quality_scores=q_scores,
            agreement=agreement.float(),
        )
        pol_inner = policy.module if hasattr(policy, "module") else policy

        # --- Sample actions ---
        actions = pol_inner.sample_actions(policy_out, mask_ind)
        u_t = actions["u_t"]
        r_t = actions["r_t"]
        kappa_t = actions["kappa_t"]

        # --- Record trajectory ---
        if trajectory is not None:
            lp = pol_inner.log_prob(policy_out, actions)
            trajectory.actions.append({k: v.detach() for k, v in actions.items()})
            trajectory.log_probs.append(lp.detach())
            trajectory.policy_outputs.append(
                {k: v.detach() for k, v in policy_out.items()}
            )
            trajectory.H_t_list.append(H_t.detach())
            trajectory.mask_ind_list.append(mask_ind.detach())
            trajectory.quality_scores_list.append(
                q_scores.detach() if q_scores is not None else None
            )
            trajectory.agreement_list.append(agreement.detach())
            trajectory.step_fracs.append(step_frac)

        # --- Count cache thrashing ---
        if cache_mgr is not None and trajectory is not None:
            thrash = cache_mgr.count_thrash(r_t)
            trajectory.thrash_counts.append(thrash.detach())

        resp_tokens = resp_tokens.clone()

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
        if use_fallback and not record_trajectory:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)
            if no_unmasks.any():
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    masked_pos = still_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        best_pos = masked_pos[confidence[b_idx, masked_pos].argmax()]
                        resp_tokens[b_idx, best_pos] = resp_logits[b_idx, best_pos].argmax()

        # ====== Phase 3: Cache (agreement positions only) ======
        if cache_mgr is not None:
            # Only cache positions where auxiliary and primary agree
            agreement_cache = kappa_t * agreement.float()
            cache_mgr.commit(agreement_cache)

            if trajectory is not None:
                trajectory.total_cache_hits += int(agreement_cache.sum().item())
                trajectory.total_cache_misses += int(
                    (kappa_t * (~agreement).float()).sum().item()
                )

        y = y.clone()
        y[:, resp_slice] = resp_tokens

    # --- Record final state ---
    if trajectory is not None:
        trajectory.final_tokens = y[:, resp_slice].detach()
        if trajectory.completion_step is None:
            trajectory.completion_step = torch.full((B,), T, device=device)
        if agreement_rates:
            trajectory.mean_agreement_rate = sum(agreement_rates) / len(agreement_rates)

    return y, trajectory
