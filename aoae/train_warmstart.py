"""Supervised Phase A V2 warm-start training."""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from .checkpoints import load_state_dict_flexible
from .speculative_inference import speculative_inference
from .tasks import build_prompt, decode_generated_tokens, extract_prompt_and_reference
from .train_grpo import _TrainingLogger
from .phase_a_v2 import (
    PhaseAV2Policy,
    WarmStartLabels,
    build_remask_labels,
    build_unmask_labels,
    extract_remask_candidates,
    extract_unmask_candidates,
    phase_a_supervised_loss,
)


def _phase_a_cfg(cfg: dict) -> dict:
    raw = cfg.get("phase_a_v2_config", {}) or {}
    return raw if isinstance(raw, dict) else {}


def _copy_with_overrides(cfg: dict, overrides: Dict[str, Any]) -> dict:
    from .experiment_utils import set_nested

    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        set_nested(out, str(key), value)
    return out


def _future_accept_reject_for_unmask(
    *,
    step_idx: int,
    selected: torch.Tensor,
    accept_masks: List[torch.Tensor],
    reject_masks: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return future verifier outcomes for positions selected at one step."""
    future_accept = torch.zeros_like(selected, dtype=torch.bool)
    future_reject = torch.zeros_like(selected, dtype=torch.bool)
    undecided = selected.bool().clone()
    for j in range(step_idx + 1, len(accept_masks)):
        acc = accept_masks[j].bool()
        rej = reject_masks[j].bool()
        future_accept = future_accept | (undecided & acc)
        future_reject = future_reject | (undecided & rej)
        undecided = undecided & ~(acc | rej)
        if not bool(undecided.any()):
            break
    return future_accept, future_reject


def _empty_labels_like(mask: torch.Tensor) -> WarmStartLabels:
    return WarmStartLabels(
        labels=torch.full(mask.shape, -1.0, device=mask.device),
        weights=torch.zeros(mask.shape, device=mask.device, dtype=torch.float32),
    )


def _make_step_labels(
    trajectory: Any,
    step_idx: int,
    *,
    low_confidence_threshold: float,
) -> Tuple[WarmStartLabels, WarmStartLabels, torch.Tensor, torch.Tensor]:
    mask_ind = trajectory.mask_ind_list[step_idx].bool()
    confidence = trajectory.confidence_list[step_idx].float()
    actions = trajectory.actions[step_idx]
    run_primary = bool(trajectory.run_primary_list[step_idx])
    accept_masks = list(getattr(trajectory, "frontier_accept_mask_list", []))
    reject_masks = list(getattr(trajectory, "frontier_reject_mask_list", []))

    u_candidates = extract_unmask_candidates(mask_ind)
    r_exec_candidates = extract_remask_candidates(mask_ind)
    forced_rejects = reject_masks[step_idx].bool() if step_idx < len(reject_masks) else torch.zeros_like(mask_ind)
    accepted_frontier = accept_masks[step_idx].bool() if step_idx < len(accept_masks) else torch.zeros_like(mask_ind)
    r_train_candidates = r_exec_candidates | forced_rejects

    u_labels = _empty_labels_like(mask_ind)
    r_labels = _empty_labels_like(mask_ind)
    if not run_primary:
        selected = actions.get("u_t", torch.zeros_like(mask_ind, dtype=torch.float32)).bool()
        future_accept, future_reject = _future_accept_reject_for_unmask(
            step_idx=step_idx,
            selected=selected,
            accept_masks=accept_masks,
            reject_masks=reject_masks,
        )
        verifier_accepted = future_accept & ~future_reject
        u_labels = build_unmask_labels(
            candidate_mask=u_candidates,
            heuristic_selected=selected,
            verifier_accepted=verifier_accepted,
            confidence=confidence,
            low_confidence_threshold=low_confidence_threshold,
        )
        # If a selected position was never verified inside the recorded window,
        # keep it unlabeled rather than teaching a false rejection.
        unverified_selected = selected & ~(future_accept | future_reject)
        u_labels.labels[unverified_selected] = -1.0
        u_labels.weights[unverified_selected] = 0.0
    else:
        stable_kept = r_exec_candidates & ~accepted_frontier & ~forced_rejects
        r_labels = build_remask_labels(
            candidate_mask=r_train_candidates,
            forced_rejects=forced_rejects,
            accepted_frontier=accepted_frontier,
            stable_kept=stable_kept,
        )

    return u_labels, r_labels, u_candidates, r_train_candidates


def _mean_metric(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _save_checkpoint(path: str, policy, soft_mask, metadata: dict) -> None:
    pol_sd = policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()
    sm_sd = (
        soft_mask.module.state_dict()
        if hasattr(soft_mask, "module")
        else (soft_mask.state_dict() if soft_mask is not None else {})
    )
    payload = {
        "policy": pol_sd,
        "soft_mask": sm_sd,
        "metadata": metadata,
    }
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


def train(cfg: dict) -> str:
    """Run supervised V2 warm-start and return the final checkpoint path."""
    from datasets import load_dataset

    from .models.dual_model import DualModelWrapper
    from .models.soft_mask import SoftMaskedState

    if not bool(cfg.get("phase_a_v2", False)):
        raise ValueError("Warm-start is only implemented for phase_a_v2=true.")
    if cfg.get("base_model", {}).get("backend") != "dual":
        raise ValueError("Phase A V2 warm-start requires base_model.backend=dual.")

    hw = cfg.get("hardware", {})
    dist_info = cfg.get("_dist", None)
    is_distributed = dist_info is not None
    rank = int(dist_info.get("rank", 0)) if is_distributed else 0
    local_rank = int(dist_info.get("local_rank", 0)) if is_distributed else 0
    world_size = int(dist_info.get("world_size", 1)) if is_distributed else 1
    is_main = rank == 0

    # Mirror train_grpo.py distributed semantics:
    #   tp_size > 1  → all ranks form one TP group; rollouts must use the SAME
    #                  prompts and synchronized RNG (LLaDA EP all-to-all would
    #                  otherwise deadlock). No effective DP.
    #   tp_size == 1 → pure data-parallel: each rank loads its own model copy,
    #                  prompts are sharded, gradients synced via DDP.
    hw_tp_size = int(hw.get("tp_size", 1) or 1)
    sync_ranks = is_distributed and hw_tp_size > 1
    rank_offset = 0 if sync_ranks else rank
    seed = int(hw.get("seed", 42))
    torch.manual_seed(seed + rank_offset)
    random.seed(seed + rank_offset)
    np.random.seed(seed + rank_offset)
    if is_main and is_distributed:
        mode = "TP-shared (synchronized prompts + RNG)" if sync_ranks else "data-parallel (sharded prompts)"
        print(f"[Warm-start] Distributed mode: {mode}  (tp_size={hw_tp_size}, world_size={world_size})")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    wc = cfg.get("warmstart", {}) or {}
    v2c = _phase_a_cfg(cfg)
    output_dir = str(cfg.get("logging", {}).get("output_dir", "outputs/v4_warmstart"))
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, "policy_final.pt")
    latest_path = os.path.join(output_dir, "policy_latest.pt")

    if is_main:
        print(f"[Warm-start] device={device}")
    dual_model = DualModelWrapper(cfg).to(device)
    tokenizer = dual_model.tokenizer
    embed_w = dual_model.get_embedding_weight()
    mask_id = int(cfg["base_model"]["mask_token_id"])
    soft_mask = SoftMaskedState(cfg, embed_w).to(device)
    soft_mask.set_mask_embedding(mask_id)
    for p in soft_mask.parameters():
        p.requires_grad_(False)

    policy = PhaseAV2Policy(cfg, input_dim=embed_w.shape[1]).to(device)
    warm_start_path = wc.get("resume_from") or v2c.get("warmstart_checkpoint")
    if warm_start_path:
        ckpt = torch.load(warm_start_path, map_location=device)
        load_state_dict_flexible(policy, ckpt["policy"], "policy(warmstart_resume)")

    # Wrap policy in DDP only for pure-DP (tp_size=1) multi-rank runs.
    # Under sync_ranks (TP > 1), all ranks already see identical inputs/grads;
    # DDP would do redundant all-reduce on identical tensors.
    # find_unused_parameters=True is required because the hidden_residual
    # parameters (hidden_proj, hidden_norm, hidden_delta_*, gates) receive no
    # gradients in scalar_only mode.
    if is_distributed and not sync_ranks:
        from torch.nn.parallel import DistributedDataParallel as DDP

        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(wc.get("lr", 1e-4)),
        weight_decay=float(wc.get("weight_decay", 0.01)),
    )

    data_cfg = cfg.get("data", {})
    train_ds = load_dataset(data_cfg["train_dataset"], split=data_cfg.get("train_split", "train"))
    max_samples = wc.get("max_samples", data_cfg.get("train_max_samples"))
    if max_samples:
        train_ds = train_ds.select(range(min(int(max_samples), len(train_ds))))

    extract_cfg = _copy_with_overrides(
        cfg,
        {
            "grpo.train_heads": [],
            "grpo.include_heads_in_logprob": ["unmask", "remask"],
            "grpo.reward_cache_terms_enabled": False,
            "cache.enabled": False,
            "cache.stable_kv_cache": False,
            "inference.positional_cache.enabled": False,
            "inference.drafter.fast_path": False,
        },
    )
    for key, value in (wc.get("rollout_overrides", {}) or {}).items():
        from .experiment_utils import set_nested

        set_nested(extract_cfg, str(key), value)

    epochs = int(wc.get("epochs", 1))
    max_steps = int(wc.get("max_steps", 100))
    grad_clip = float(wc.get("max_grad_norm", 1.0))
    lambda_rank = float(wc.get("lambda_rank", 0.1))
    lambda_rate = float(wc.get("lambda_rate", 0.01))
    lambda_hidden_gate = float(wc.get("lambda_hidden_gate", 0.0))
    low_confidence_threshold = float(wc.get("low_confidence_threshold", 0.15))
    log_every = int(cfg.get("logging", {}).get("log_every", 10))
    save_every = int(cfg.get("logging", {}).get("save_every", 100))
    global_step = 0

    logger = _TrainingLogger(cfg) if is_main else None
    try:
        for epoch in range(epochs):
            indices = list(range(len(train_ds)))
            random.shuffle(indices)
            # Pure-DP mode: shard prompts across ranks. Under sync_ranks all ranks
            # process the same shuffled order (RNG seeded identically above).
            if is_distributed and not sync_ranks:
                indices = indices[rank::world_size]
            pbar = tqdm(indices, desc=f"Warm-start epoch {epoch + 1}/{epochs}", disable=not is_main)
            metric_buf: Dict[str, List[float]] = {}
            for idx in pbar:
                if global_step >= max_steps:
                    break
                sample = train_ds[idx]
                question, reference = extract_prompt_and_reference(sample)
                if not question or not reference:
                    continue
                prompt_text, add_special_tokens = build_prompt(tokenizer, question, cfg)
                prompt_ids = tokenizer.encode(
                    prompt_text,
                    add_special_tokens=add_special_tokens,
                    max_length=int(data_cfg.get("max_prompt_len", 512)),
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                if prompt_ids.dim() == 1:
                    prompt_ids = prompt_ids.unsqueeze(0)

                with torch.no_grad():
                    generated_ids, trajectory = speculative_inference(
                        dual_model=dual_model,
                        policy=policy,
                        soft_mask_module=soft_mask,
                        prism_adapter=None,
                        prompt_ids=prompt_ids,
                        cfg=extract_cfg,
                        record_trajectory=True,
                        policy_temperature=1.0,
                    )

                step_losses: List[torch.Tensor] = []
                for step_idx, H_t in enumerate(trajectory.H_t_list):
                    u_labels, r_labels, _u_candidates, r_train_candidates = _make_step_labels(
                        trajectory,
                        step_idx,
                        low_confidence_threshold=low_confidence_threshold,
                    )
                    confidence = trajectory.confidence_list[step_idx]
                    agreement = trajectory.agreement_list[step_idx].float()
                    frontier = trajectory.frontier_before_list[step_idx].float()
                    # V5 hybrid: pass aux_h_final and pri_h_final to policy
                    aux_h_final = trajectory.aux_h_final_list[step_idx] if hasattr(trajectory, 'aux_h_final_list') else None
                    pri_h_final = trajectory.pri_h_final_list[step_idx] if hasattr(trajectory, 'pri_h_final_list') else None
                    out = policy(
                        H_t,
                        trajectory.mask_ind_list[step_idx],
                        trajectory.step_fracs[step_idx],
                        confidence=confidence,
                        agreement=agreement,
                        frontier_membership=frontier,
                        remask_candidate_mask=r_train_candidates,
                        aux_h_final=aux_h_final,
                        pri_h_final=pri_h_final,
                    )
                    loss, metrics = phase_a_supervised_loss(
                        out,
                        u_labels,
                        r_labels,
                        lambda_rank=lambda_rank,
                        lambda_rate=lambda_rate,
                        lambda_hidden_gate=lambda_hidden_gate,
                        target_u_rate=float(v2c.get("target_u_rate", cfg.get("target_u_rate", 0.10))),
                        target_r_rate=float(v2c.get("target_r_rate", cfg.get("target_r_rate", 0.02))),
                    )
                    if torch.isfinite(loss):
                        step_losses.append(loss)
                        for k, v in metrics.items():
                            metric_buf.setdefault(k, []).append(float(v.item()))

                if not step_losses:
                    continue
                total_loss = torch.stack(step_losses).mean()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                optimizer.step()
                global_step += 1

                metric_buf.setdefault("warmstart/loss", []).append(float(total_loss.item()))
                if global_step % log_every == 0:
                    record = {
                        "epoch": epoch,
                        **{k: _mean_metric(v) for k, v in metric_buf.items()},
                    }
                    if is_main:
                        print(
                            f"[Warm-start] step={global_step} "
                            f"loss={record.get('warmstart/loss', 0.0):.4f} "
                            f"u_bce={record.get('warmstart/u_bce', 0.0):.4f} "
                            f"r_bce={record.get('warmstart/r_bce', 0.0):.4f}"
                        )
                        if logger is not None:
                            logger.log_step(record, step=global_step)
                    metric_buf.clear()

                if global_step == 1 or global_step % save_every == 0:
                    if is_main:
                        _save_checkpoint(
                            latest_path,
                            policy,
                            soft_mask,
                            {
                                "stage": "warmstart",
                                "global_step": global_step,
                                "feature_mode": v2c.get("feature_mode", cfg.get("feature_mode", "scalar_only")),
                            },
                        )
                        # Log a sample generation for inspection
                        if logger is not None:
                            try:
                                resp_ids = generated_ids[0, prompt_ids.shape[-1]:]
                                gen_text = decode_generated_tokens(
                                    tokenizer, resp_ids,
                                    mask_token_id=dual_model.mask_id,
                                )
                                n_masks = int((resp_ids == dual_model.mask_id).sum().item())
                                sample_record = {
                                    "sample/question": question[:200],
                                    "sample/reference": reference[:200],
                                    "sample/generation": gen_text[:500],
                                    "sample/n_masks_remaining": n_masks,
                                    "sample/gen_length": int(resp_ids.shape[0]),
                                }
                                logger.log_step(sample_record, step=global_step)
                                print(
                                    f"  [Sample] Q: {question[:80]}...\n"
                                    f"  [Sample] A: {gen_text[:120]}...\n"
                                    f"  [Sample] masks_remaining={n_masks}"
                                )
                            except Exception:
                                pass  # don't crash training on logging failure
            if global_step >= max_steps:
                break
    finally:
        dual_model.close()
        if logger is not None:
            logger.close()

    metadata = {
        "stage": "warmstart",
        "completed_steps": global_step,
        "train_dataset": data_cfg.get("train_dataset"),
        "train_split": data_cfg.get("train_split"),
        "feature_mode": v2c.get("feature_mode", cfg.get("feature_mode", "scalar_only")),
        "target_u_rate": float(v2c.get("target_u_rate", cfg.get("target_u_rate", 0.10))),
        "target_r_rate": float(v2c.get("target_r_rate", cfg.get("target_r_rate", 0.02))),
    }
    if is_main:
        _save_checkpoint(final_path, policy, soft_mask, metadata)
        with open(os.path.join(output_dir, "warmstart_training_metadata.json"), "w") as f:
            json.dump({**metadata, "final_checkpoint": final_path}, f, indent=2)
        print(f"[Warm-start] complete. Final checkpoint: {final_path}")
    return final_path


def main(cfg: dict) -> str:
    return train(cfg)
