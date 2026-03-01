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
import math
import re
import time
import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
# LambdaLR imported at usage site below (to keep scheduler logic together)
from tqdm import tqdm
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from .inference import aoae_inference, AOAETrajectory
from .speculative_inference import speculative_inference, SpeculativeTrajectory
import glob


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

      R = r(y*, y_T_hat) * (1 - (T - T_hat)/T)^alpha - beta * sum Thrash(t)

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
    B = generated_tokens.shape[0]
    device = generated_tokens.device

    # --- Correctness term r(y*, y) ---
    correctness = torch.zeros(B, device=device)
    for b in range(B):
        gen_text = tokenizer.decode(generated_tokens[b], skip_special_tokens=True)
        correct = check_math_correctness(gen_text, reference_answer[b])
        correctness[b] = 1.0 if correct else 0.0

    # --- Speed bonus ---
    # Count steps where the policy actively drafted (unmasked) tokens.
    # Fewer active steps → faster completion → higher bonus.
    n_active_steps = sum(
        1 for a in trajectory.actions
        if "u_t" in a and a["u_t"].sum() > 0
    )
    n_active_steps = max(n_active_steps, 1)
    speed_bonus = torch.tensor(
        max(0.0, 1.0 - n_active_steps / T),
        dtype=torch.float32, device=device,
    )

    # --- Cache thrashing penalty ---
    total_thrash = torch.zeros(B, device=device)
    for thrash_t in trajectory.thrash_counts:
        total_thrash += thrash_t.to(device)

    # --- Reward: correct answers always get ≥ 1.0, with additive speed bonus ---
    # R = r(y*, y) * (1 + alpha * speed_bonus) - beta * thrash
    reward = correctness * (1.0 + alpha * speed_bonus) - beta * total_thrash

    return reward


def check_math_correctness(generated: str, reference: str) -> bool:
    """
    Check if the generated answer matches the reference for math problems.

    Extracts the final numerical answer from \\boxed{} or after "####".
    """
    gen_answer = extract_answer(generated)
    ref_answer = extract_answer(reference)

    if gen_answer is None or ref_answer is None:
        return False

    try:
        return abs(float(gen_answer) - float(ref_answer)) < 1e-3
    except (ValueError, TypeError):
        return gen_answer.strip() == ref_answer.strip()


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text (supports \\boxed{} and #### formats)."""

    # Try \boxed{...}
    match = re.findall(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match[-1].strip()

    # Try #### answer format (GSM8K)
    match = re.findall(r'####\s*(.+)', text)
    if match:
        return match[-1].strip().replace(",", "")

    # Try last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return None


def extract_prompt_and_reference(sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract (prompt, reference_answer) from heterogeneous math dataset schemas."""

    def _collect_text(value: Any) -> List[str]:
        """Recursively collect textual fragments from nested json-like values."""
        out: List[str] = []
        if isinstance(value, str):
            s = value.strip()
            if s:
                out.append(s)
            return out
        if isinstance(value, dict):
            # Prefer common content carriers first for stable ordering.
            for k in ("content", "text", "value"):
                if k in value:
                    out.extend(_collect_text(value[k]))
            for k, v in value.items():
                if k not in ("content", "text", "value"):
                    out.extend(_collect_text(v))
            return out
        if isinstance(value, list):
            for item in value:
                out.extend(_collect_text(item))
            return out
        return out

    prompt: Optional[str] = None
    reference: Optional[str] = None

    # Common prompt fields
    for key in ("question", "problem", "prompt", "instruction", "input"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break

    # Common target/reference fields
    for key in ("answer", "solution", "output", "response", "completion", "final_answer"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            reference = value.strip()
            break

    # Chat-style fallback
    messages = sample.get("messages") or sample.get("conversations")
    if isinstance(messages, list):
        user_chunks = []
        assistant_chunks = []
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            content_chunks = _collect_text(turn.get("content"))
            if not content_chunks:
                continue
            content = "\n".join(content_chunks)
            if role == "user":
                user_chunks.append(content)
            elif role == "assistant":
                assistant_chunks.append(content)
        if prompt is None and user_chunks:
            prompt = "\n".join(user_chunks)
        if reference is None and assistant_chunks:
            reference = "\n".join(assistant_chunks)

    # Last-resort fallback: choose two longest textual fields from the sample.
    if prompt is None or reference is None:
        field_texts: List[Tuple[str, str]] = []
        for key, value in sample.items():
            chunks = _collect_text(value)
            if chunks:
                merged = "\n".join(chunks).strip()
                if merged:
                    field_texts.append((key, merged))

        # Prefer not to use metadata-like fields when alternatives exist.
        preferred = [
            kv for kv in field_texts
            if kv[0].lower() not in {"id", "source", "split", "dataset", "metadata"}
        ]
        pool = preferred if preferred else field_texts
        pool.sort(key=lambda kv: len(kv[1]), reverse=True)

        if pool:
            if prompt is None:
                prompt = pool[0][1]
            if reference is None:
                reference = pool[1][1] if len(pool) > 1 else pool[0][1]

    return prompt, reference


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
            agreement_list = traj.get("agreement_list", None)
            agreement = agreement_list[t_idx].float() if agreement_list else None
            policy_out = policy(H_t, mask_ind, step_frac, quality_scores=q_scores, agreement=agreement)
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
    G = gc["group_size"]
    T = cfg["inference"]["steps"]
    mask_id = cfg["base_model"]["mask_token_id"]

    B = prompt_ids.shape[0]
    assert B == 1, "Collect rollouts one prompt at a time, then group."

    trajectories = []
    rewards = []
    use_speculative = (dual_model is not None)

    for g in range(G):
        if use_speculative:
            output_ids, trajectory = speculative_inference(
                dual_model=dual_model,
                policy=policy,
                soft_mask_module=soft_mask_module,
                prism_adapter=prism_adapter,
                prompt_ids=prompt_ids,
                cfg=cfg,
                record_trajectory=True,
                policy_temperature=gc["policy_temperature"],
            )
        else:
            output_ids, trajectory = aoae_inference(
                base_model=base_model,
                policy=policy,
                soft_mask_module=soft_mask_module,
                prism_adapter=prism_adapter,
                prompt_ids=prompt_ids,
                cfg=cfg,
                record_trajectory=True,
                policy_temperature=gc["policy_temperature"],
            )

        # Compute reward
        gen_tokens = output_ids[:, prompt_ids.shape[1]:]
        reward = compute_reward(
            gen_tokens, reference_answers, tokenizer, trajectory, cfg, T
        )

        # Store trajectory data for GRPO recomputation
        traj_data = {
            "actions_list": trajectory.actions,
            "old_log_probs": [lp.clone() for lp in trajectory.log_probs],
            "H_t_list": trajectory.H_t_list,
            "mask_ind_list": trajectory.mask_ind_list,
            "quality_scores_list": trajectory.quality_scores_list,
            "step_fracs": trajectory.step_fracs,
            "reward": reward.item(),
        }
        # Store agreement for speculative trajectories
        if use_speculative and hasattr(trajectory, "agreement_list"):
            traj_data["agreement_list"] = trajectory.agreement_list
        trajectories.append(traj_data)
        rewards.append(reward.item())

    # --- Compute advantages: A^g = (R^g - mean(R)) / std(R) ---
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = rewards_t - rewards_t.mean()
    std = rewards_t.std()
    if std > 1e-8:
        advantages = advantages / std

    return trajectories, rewards_t, advantages


# ======================================================================
# Main training loop
# ======================================================================

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest policy_step*.pt checkpoint in output_dir by step number."""
    pattern = os.path.join(output_dir, "policy_step*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    # Extract step number and sort
    def _step_num(path: str) -> int:
        base = os.path.basename(path)
        # policy_step1000.pt -> 1000
        num_str = base.replace("policy_step", "").replace(".pt", "")
        try:
            return int(num_str)
        except ValueError:
            return -1
    ckpts.sort(key=_step_num)
    return ckpts[-1]


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

    # --- Training loop ---
    os.makedirs(lc["output_dir"], exist_ok=True)
    global_step = 0
    accum_step = 0
    start_epoch = 0
    grad_accum = gc["grad_accum_steps"]
    best_reward = -float("inf")

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
        pol_inner.load_state_dict(ckpt["policy"])
        sm_inner = soft_mask.module if hasattr(soft_mask, 'module') else soft_mask
        sm_inner.load_state_dict(ckpt["soft_mask"])

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
                if global_step % lc["save_every"] == 0 and is_main:
                    pol_sd = policy.module.state_dict() if hasattr(policy, 'module') else policy.state_dict()
                    sm_sd = soft_mask.module.state_dict() if hasattr(soft_mask, 'module') else soft_mask.state_dict()
                    ckpt_path = os.path.join(lc["output_dir"], f"policy_step{global_step}.pt")
                    torch.save({
                        "policy": pol_sd,
                        "soft_mask": sm_sd,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": global_step,
                        "accum_step": accum_step,
                        "epoch": epoch,
                    }, ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")

                if global_step >= gc["max_steps"]:
                    break

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

    # --- Save final model (rank 0 only) ---
    if is_main:
        pol_sd = policy.module.state_dict() if hasattr(policy, 'module') else policy.state_dict()
        sm_sd = soft_mask.module.state_dict() if hasattr(soft_mask, 'module') else soft_mask.state_dict()
        final_path = os.path.join(lc["output_dir"], "policy_final.pt")
        torch.save({
            "policy": pol_sd,
            "soft_mask": sm_sd,
        }, final_path)
        print(f"\nTraining complete. Final model saved to {final_path}")
