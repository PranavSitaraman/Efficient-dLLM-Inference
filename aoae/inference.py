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
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from .cache import DKVCacheManager
from .models.composed_prediction import compose_prediction


@dataclass
class AOAETrajectory:
    """Stores a full inference trajectory for GRPO training."""
    actions: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    policy_outputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    thrash_counts: List[torch.Tensor] = field(default_factory=list)
    H_t_list: List[torch.Tensor] = field(default_factory=list)
    mask_ind_list: List[torch.BoolTensor] = field(default_factory=list)
    quality_scores_list: List[Optional[torch.Tensor]] = field(default_factory=list)
    step_fracs: List[float] = field(default_factory=list)
    final_tokens: Optional[torch.Tensor] = None
    completion_step: Optional[torch.Tensor] = None


def aoae_inference(
    base_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    record_trajectory: bool = False,
    policy_temperature: float = 1.0,
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
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)  # Composed prediction strength

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
        if prism_adapter is not None:
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
        H_t, confidence, entropy = soft_mask_module(
            resp_logits, mask_ind, step_frac
        )  # H_t: [B, L_gen, D]

        # --- Policy forward (with PRISM quality scores) ---
        policy_out = policy(
            H_t, mask_ind, step_frac,
            temperature=policy_temperature,
            quality_scores=q_scores,
        )
        pol_inner = policy.module if hasattr(policy, "module") else policy

        # --- Sample actions ---
        actions = pol_inner.sample_actions(policy_out, mask_ind)
        u_t = actions["u_t"]        # [B, L_gen] unmask
        r_t = actions["r_t"]        # [B, L_gen] remask
        kappa_t = actions["kappa_t"]  # [B, L_gen] cache

        # --- Record trajectory for GRPO ---
        if trajectory is not None:
            lp = pol_inner.log_prob(policy_out, actions)
            trajectory.actions.append({k: v.detach() for k, v in actions.items()})
            trajectory.log_probs.append(lp.detach())
            trajectory.policy_outputs.append(
                {k: v.detach() for k, v in policy_out.items()}
            )
            # Store states for off-policy importance sampling
            trajectory.H_t_list.append(H_t.detach())
            trajectory.mask_ind_list.append(mask_ind.detach())
            trajectory.quality_scores_list.append(
                q_scores.detach() if q_scores is not None else None
            )
            trajectory.step_fracs.append(step_frac)

        # --- Count cache thrashing BEFORE invalidation ---
        if cache_mgr is not None and trajectory is not None:
            thrash = cache_mgr.count_thrash(r_t)
            trajectory.thrash_counts.append(thrash.detach())

        # Clone once for all mutations this step
        resp_tokens = resp_tokens.clone()

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
        if use_fallback and not record_trajectory:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)  # [B]
            if no_unmasks.any():
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    masked_pos = still_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        best_pos = masked_pos[confidence[b_idx, masked_pos].argmax()]
                        resp_tokens[b_idx, best_pos] = resp_logits[b_idx, best_pos].argmax()

        # ====== Phase 3: Cache commit ======
        if cache_mgr is not None:
            cache_mgr.commit(kappa_t)

        # --- Write back response tokens ---
        y = y.clone()
        y[:, resp_slice] = resp_tokens

    # --- Record final state ---
    if trajectory is not None:
        trajectory.final_tokens = y[:, resp_slice].detach()
        if trajectory.completion_step is None:
            # Did not break early — used all T steps
            trajectory.completion_step = torch.full((B,), T, device=device)

    return y, trajectory


# ======================================================================
# Baseline decoders for comparison
# ======================================================================

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

    return y


def confidence_threshold_decode(
    base_model,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    tau_mask: float = 0.9,
    tau_edit: float = 0.95,
    enable_t2t: bool = True,
) -> torch.Tensor:
    """
    LLaDA 2.1-style confidence threshold decoding.

    Implements S-Mode (aggressive thresholds) or Q-Mode (conservative)
    depending on tau_mask and tau_edit.
    """
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

    for t in range(T, 0, -1):
        resp = y[:, resp_slice]
        mask_ind = (resp == mask_id)

        if not mask_ind.any():
            break

        logits = base_model.forward(y)[:, resp_slice, :]
        probs = F.softmax(logits, dim=-1)
        max_prob, max_tok = probs.max(dim=-1)  # [B, L_gen]

        # M2T: unmask positions above tau_mask
        unmask = mask_ind & (max_prob > tau_mask)
        resp = resp.clone()
        resp[unmask] = max_tok[unmask]

        # T2T: edit unmasked positions where model disagrees and confidence > tau_edit
        if enable_t2t:
            unmasked = ~mask_ind
            disagree = (max_tok != resp) & unmasked
            confident = max_prob > tau_edit
            edit = disagree & confident
            resp[edit] = max_tok[edit]

        # Fallback: if nothing was unmasked, unmask the most confident
        still_masked = (resp == mask_id)
        nothing_happened = mask_ind.any(dim=-1) & ~unmask.any(dim=-1)
        for b in nothing_happened.nonzero(as_tuple=True)[0]:
            masked_pos = still_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_pos) > 0:
                best = masked_pos[max_prob[b, masked_pos].argmax()]
                resp[b, best] = max_tok[b, best]

        y = y.clone()
        y[:, resp_slice] = resp

    return y


def block_smode_decode(
    base_model,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    tau_mask: float = 0.7,
    tau_edit: float = 0.9,
    max_steps_per_block: int = 16,
    enable_mbe: bool = False,
) -> torch.Tensor:
    """
    Block-wise Semi-Autoregressive S-Mode Decoding (LLaDA 2.1 paper §2).

    Generates text block-by-block (left-to-right). Within each block,
    parallel threshold decoding unmasks many tokens simultaneously.

    This is the key technique for high TPS:
      - Only the current block is masked → shorter effective seq for diffusion
      - Threshold decoding unmasks many tokens per forward pass → fewer steps
      - Blocks processed sequentially → maintains left-to-right coherence

    Args:
        base_model: frozen LLaDA model.
        prompt_ids: [B, P] prompt token ids.
        cfg: config dict with inference.block_length, inference.gen_length.
        tau_mask: confidence threshold for M2T unmasking.
        tau_edit: confidence threshold for T2T editing.
        max_steps_per_block: max diffusion steps per block.
        enable_mbe: if True, enable Multiple Block Editing (revisit prev blocks).

    Returns:
        output_ids: [B, P + L_gen] generated sequence.
    """
    ic = cfg["inference"]
    L_gen = ic["gen_length"]
    block_len = ic.get("block_length", 32)
    mask_id = cfg["base_model"]["mask_token_id"]

    B, P = prompt_ids.shape
    device = prompt_ids.device
    n_blocks = (L_gen + block_len - 1) // block_len

    # Start with prompt + all masks
    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    for blk_idx in range(n_blocks):
        blk_start = P + blk_idx * block_len
        blk_end = min(P + (blk_idx + 1) * block_len, P + L_gen)
        blk_slice = slice(blk_start, blk_end)
        blk_len_actual = blk_end - blk_start

        for step in range(max_steps_per_block):
            blk_tokens = y[:, blk_slice]
            mask_ind = (blk_tokens == mask_id)

            if not mask_ind.any():
                break

            # Forward pass with block-causal attention if available
            if hasattr(base_model, 'forward_block_causal'):
                logits = base_model.forward_block_causal(
                    y, block_length=block_len,
                )[:, blk_slice, :]
            else:
                logits = base_model.forward(y)[:, blk_slice, :]
            probs = F.softmax(logits, dim=-1)
            max_prob, max_tok = probs.max(dim=-1)

            # M2T: unmask confident positions
            unmask = mask_ind & (max_prob > tau_mask)
            blk_tokens = blk_tokens.clone()
            blk_tokens[unmask] = max_tok[unmask]

            # T2T: edit unmasked positions where model disagrees
            unmasked = ~mask_ind
            disagree = (max_tok != blk_tokens) & unmasked
            confident = max_prob > tau_edit
            edit = disagree & confident
            blk_tokens[edit] = max_tok[edit]

            # Fallback: unmask most confident if nothing changed
            still_masked = (blk_tokens == mask_id)
            nothing_happened = mask_ind.any(dim=-1) & ~unmask.any(dim=-1)
            for b in nothing_happened.nonzero(as_tuple=True)[0]:
                masked_pos = still_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_pos) > 0:
                    best = masked_pos[max_prob[b, masked_pos].argmax()]
                    blk_tokens[b, best] = max_tok[b, best]

            y = y.clone()
            y[:, blk_slice] = blk_tokens

        # Optional: Multiple Block Editing — revisit previous blocks
        if enable_mbe and blk_idx > 0:
            prev_start = P
            prev_end = blk_start
            prev_slice = slice(prev_start, prev_end)

            logits = base_model.forward(y)[:, prev_slice, :]
            probs = F.softmax(logits, dim=-1)
            max_prob, max_tok = probs.max(dim=-1)

            prev_tokens = y[:, prev_slice].clone()
            disagree = (max_tok != prev_tokens) & (max_prob > tau_edit)
            prev_tokens[disagree] = max_tok[disagree]
            y = y.clone()
            y[:, prev_slice] = prev_tokens

    return y
