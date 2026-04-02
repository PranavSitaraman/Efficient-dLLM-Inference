"""
GRPO Training for the AOAE Policy (paper §3.5).

Implements Group Relative Policy Optimization following Jazbec et al. (2025)
and Shao et al. (2024, DeepSeekMath), extended to the unified action space.

Key design choices:
  - Multiplicative reward: correctness * speed_penalty - beta * thrashing
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
import torch
import torch.nn as nn
from torch.optim import AdamW
# LambdaLR imported at usage site below (to keep scheduler logic together)
from tqdm import tqdm
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from .checkpoints import (
    GRPO_TRAIN_CONTRACT_VERSION,
    build_grpo_config_fingerprint,
    find_latest_checkpoint,
    load_state_dict_flexible,
)
from .inference import aoae_inference, AOAETrajectory
from .speculative_inference import speculative_inference, SpeculativeTrajectory
from .tasks import check_math_correctness, extract_answer, extract_prompt_and_reference
from .runtime_checks import collect_runtime_info


# ======================================================================
# Reward computation (Eq. reward from paper)
# ======================================================================

def compute_reward(
    generated_tokens: torch.Tensor,
    reference_answer: List[str],
    tokenizer,
    trajectory: AOAETrajectory,
    cfg: dict,
    T: int,
) -> torch.Tensor:
    """
    Compute multiplicative reward (Eq. reward):

      R = r(y*, y_hat) * (T_hat/T)^alpha - beta * sum Thrash(t)
          - lambda_unresolved * unresolved_fraction

    Here T_hat is completion in the paper's reverse-time convention
    (larger is faster). Internally we track used forward steps, then map
    to reverse-time completion before applying the speed factor.

    Args:
        generated_tokens: [B, L_gen] generated response tokens.
        reference_answer: list of B reference answer strings.
        tokenizer:        for decoding generated tokens.
        trajectory:       AOAETrajectory with thrash counts.
        cfg:              config dict.
        T:                total diffusion steps.

    Returns:
        rewards: [B] per-sample scalar reward.
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
        gen_text = tokenizer.decode(generated_tokens[b], skip_special_tokens=True)
        correct = check_math_correctness(gen_text, reference_answer[b])
        correctness[b] = 1.0 if correct else 0.0

    # --- Speed factor ---
    # used_steps: forward-time steps consumed (smaller is faster).
    # Map to reverse-time completion index T_hat in [1, T]:
    #   T_hat = T - used_steps + 1
    if getattr(trajectory, "completion_step", None) is not None:
        used_steps = trajectory.completion_step.to(device).float().clamp(min=1.0, max=float(T))
    else:
        n_active_steps = sum(
            1 for a in trajectory.actions
            if "u_t" in a and a["u_t"].sum() > 0
        )
        used_steps = torch.full((B,), float(max(n_active_steps, 1)), device=device)
    t_hat = (float(T) - used_steps + 1.0).clamp(min=1.0, max=float(T))
    speed_factor = (t_hat / float(T)).pow(alpha)

    # --- Cache thrashing penalty ---
    total_thrash = torch.zeros(B, device=device)
    for thrash_t in trajectory.thrash_counts:
        total_thrash += thrash_t.to(device)

    # --- Reward ---
    reward = correctness * speed_factor - beta * total_thrash

    # Penalize trajectories that terminate with unresolved masks. This keeps
    # "do nothing / let masks linger" from becoming a neutral zero-reward mode.
    if unresolved_penalty_weight > 0.0:
        final_tokens = getattr(trajectory, "final_tokens", None)
        if final_tokens is not None:
            mask_id = int(cfg["base_model"]["mask_token_id"])
            unresolved_fraction = (final_tokens.to(device) == mask_id).float().mean(dim=-1)
            reward = reward - unresolved_penalty_weight * unresolved_fraction

    # Optional dense signal for next-H positional access quality.
    access_w = float(gc.get("access_reward_weight", 0.0))
    if access_w > 0.0 and hasattr(trajectory, "access_metrics"):
        spec_f1 = float(trajectory.access_metrics.get("access_next_h_spec_f1", 0.0))
        reward = reward + access_w * torch.full_like(reward, spec_f1)

    return reward


# ======================================================================
# GRPO objective (Eq. grpo from paper)
# ======================================================================

def compute_grpo_loss(
    policy,
    soft_mask_module,
    trajectories: List[Dict],
    advantages: torch.Tensor,
    clip_eps: float,
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

        for t_idx in range(n_steps):
            H_t = traj["H_t_list"][t_idx]           # [1, L, D]
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
            new_lp = pol_inner.log_prob(policy_out, actions)  # [1]

            # Importance ratio
            rho = torch.exp(new_lp - old_lp)  # [1]

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
) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
    """
    Collect G rollout trajectories for a single prompt batch.

    Args:
        base_model: LLaDABaseModel (used when dual_model is None).
        dual_model: DualModelWrapper (used for speculative mode).
        (other args as before)

    Returns:
        trajectories: list of G trajectory dicts (for GRPO loss).
        rewards:       [G] per-trajectory rewards.
        advantages:    [G] group-mean normalized advantages.
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
    rewards_t = compute_reward(
        gen_tokens,
        repeated_references,
        tokenizer,
        trajectory,
        rollout_cfg,
        T,
    ).detach().cpu()
    trajectories = split_group_trajectory(trajectory, G)
    for g, traj_data in enumerate(trajectories):
        traj_data["reward"] = float(rewards_t[g].item())

    advantages = normalize_group_advantages(
        rewards_t,
        normalize_std=bool(gc.get("normalize_advantage_std", False)),
    )

    return trajectories, rewards_t, advantages


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

        # Wrap policy in DDP for multi-GPU gradient sync
        if is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            policy = DDP(policy, device_ids=[local_rank])
            # soft_mask has learnable gating params too
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

        # --- Optimizer (over all trainable params including DDP wrappers) ---
        trainable_params = list(policy.parameters()) + list(soft_mask.parameters())
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

        # --- Resume from checkpoint ---
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

            if is_main:
                print(f"  Resumed at global_step={global_step}, epoch={start_epoch}, "
                      f"accum_step={accum_step}")
            del ckpt
        elif resume_from:
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_from}. "
                f"Use --resume auto to auto-detect, or pass a valid path."
            )

        for epoch in range(start_epoch, gc["epochs"]):
            current_epoch = epoch
            if is_main:
                print(f"\n=== Epoch {epoch + 1}/{gc['epochs']} ===")
            epoch_rewards = []
            valid_samples_this_epoch = 0

            indices = list(range(len(train_ds)))
            random.shuffle(indices)

            # Shard data across ranks for distributed training
            if is_distributed:
                indices = indices[rank::world_size]

            pbar = tqdm(range(0, len(indices), gc["batch_size"]), desc="Training", disable=not is_main)
            for i in pbar:
                batch_indices = indices[i : i + gc["batch_size"]]

                for idx in batch_indices:
                    sample = train_ds[idx]

                    # Prepare prompt
                    question, reference = extract_prompt_and_reference(sample)
                    if not question or not reference:
                        continue
                    valid_samples_this_epoch += 1

                    messages = [{"role": "user", "content": question}]
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt_ids = tokenizer.encode(
                        prompt_text,
                        add_special_tokens=False,
                        max_length=dc["max_prompt_len"],
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    if prompt_ids.dim() == 1:
                        prompt_ids = prompt_ids.unsqueeze(0)

                    # Collect G rollouts
                    trajectories, rewards, advantages = collect_rollout_group(
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

                    # Clipped GRPO surrogate with importance sampling
                    # Pass DDP-wrapped modules so .backward() syncs gradients
                    grpo_loss = compute_grpo_loss(
                        policy=policy,
                        soft_mask_module=soft_mask,
                        trajectories=trajectories,
                        advantages=advantages.to(device),
                        clip_eps=gc["clip_eps"],
                    )
                    # Scale loss for gradient accumulation
                    scaled_loss = grpo_loss / (len(batch_indices) * grad_accum)
                    scaled_loss.backward()

                accum_step += 1

                # --- Optimizer step after grad_accum mini-batches ---
                if accum_step % grad_accum == 0:
                    nn.utils.clip_grad_norm_(trainable_params, gc["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # --- Logging ---
                    if global_step % lc["log_every"] == 0 and is_main:
                        recent = epoch_rewards[-lc["log_every"]:] if epoch_rewards else [0]
                        avg_r = np.mean(recent)
                        std_r = np.std(recent) if len(recent) > 1 else 0.0
                        frac_pos = np.mean([1.0 if r > 0 else 0.0 for r in recent])
                        print(f"  step={global_step}  avg_reward={avg_r:.4f}  "
                              f"std_reward={std_r:.4f}  frac_positive={frac_pos:.2f}  "
                              f"lr={scheduler.get_last_lr()[0]:.2e}")

                    # --- Save checkpoint (rank 0 only) ---
                    if is_main and (global_step == 1 or global_step % lc["save_every"] == 0):
                        ckpt_path = os.path.join(lc["output_dir"], "policy_latest.pt")
                        _save_checkpoint(
                            ckpt_path,
                            epoch_idx=epoch,
                            step_value=global_step,
                            accum_value=accum_step,
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

            if is_main and epoch_rewards:
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
        if dual_model is not None:
            dual_model.close()
        if base_model is not None:
            base_model.close()
