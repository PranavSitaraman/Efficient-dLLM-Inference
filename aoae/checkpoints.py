"""Shared checkpoint and sidecar artifact helpers."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Dict, Optional

import torch


GRPO_TRAIN_CONTRACT_VERSION = 1


def load_state_dict_flexible(
    module,
    state_dict: Dict[str, torch.Tensor],
    label: str,
) -> None:
    """Load only compatible checkpoint tensors into a module."""
    own = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if (key in own) and (own[key].shape == value.shape)
    }
    skipped = [
        key
        for key, value in state_dict.items()
        if (key not in own) or (key in own and own[key].shape != value.shape)
    ]
    module.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"[Checkpoint] {label}: skipped {len(skipped)} incompatible keys.")


def find_latest_checkpoint(output_dir: str, pattern: str = "policy_step*.pt") -> Optional[str]:
    """Return the most recent resumable training checkpoint in an output directory."""
    if not output_dir:
        return None
    explicit_candidates = [
        os.path.join(output_dir, "policy_latest.pt"),
        os.path.join(output_dir, "policy_interrupt.pt"),
    ]
    ckpts = [path for path in explicit_candidates if os.path.exists(path)]
    ckpts.extend(glob.glob(os.path.join(output_dir, pattern)))
    if not ckpts:
        return None

    def _step_num(path: str) -> int:
        base = os.path.basename(path)
        digits = "".join(ch for ch in base if ch.isdigit())
        try:
            return int(digits)
        except ValueError:
            return -1

    ckpts.sort(key=lambda path: (os.path.getmtime(path), _step_num(path)))
    return ckpts[-1]


def resolve_policy_checkpoint(
    explicit: Optional[str],
    output_dir: str,
) -> Optional[str]:
    """Resolve an explicit checkpoint path or auto-detect the best available one."""
    if explicit:
        return explicit
    for name in ("policy_best.pt", "policy_final.pt", "policy_latest.pt"):
        candidate = os.path.join(output_dir, name)
        if os.path.exists(candidate):
            return candidate
    return find_latest_checkpoint(output_dir)


def resolve_sidecar_artifact(
    checkpoint_path: Optional[str],
    output_dir: str,
    filename: str,
) -> Optional[str]:
    """Locate an artifact stored next to a checkpoint or in the run output dir."""
    candidates = []
    if checkpoint_path:
        candidates.append(os.path.join(os.path.dirname(checkpoint_path), filename))
    candidates.append(os.path.join(output_dir, filename))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def build_grpo_config_fingerprint(cfg: Dict[str, object]) -> str:
    """Build a stable fingerprint for GRPO-relevant config sections."""
    tracked = {
        "base_model": cfg.get("base_model", {}),
        "soft_mask": cfg.get("soft_mask", {}),
        "policy": cfg.get("policy", {}),
        "prism": cfg.get("prism", {}),
        "grpo": cfg.get("grpo", {}),
        "inference": cfg.get("inference", {}),
        "data": {
            "train_dataset": cfg.get("data", {}).get("train_dataset"),
            "train_split": cfg.get("data", {}).get("train_split"),
            "train_max_samples": cfg.get("data", {}).get("train_max_samples"),
            "max_prompt_len": cfg.get("data", {}).get("max_prompt_len"),
            "max_answer_len": cfg.get("data", {}).get("max_answer_len"),
        },
    }
    payload = json.dumps(tracked, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_grpo_training_metadata(output_dir: str) -> Optional[Dict[str, object]]:
    """Load the sidecar GRPO metadata if it exists and is valid JSON."""
    if not output_dir:
        return None
    path = os.path.join(output_dir, "grpo_training_metadata.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def inspect_grpo_artifacts(output_dir: str, cfg: Dict[str, object]) -> Dict[str, object]:
    """Determine whether existing GRPO artifacts are safe to reuse."""
    status: Dict[str, object] = {
        "valid": False,
        "reason": "missing_output_dir",
        "checkpoint_path": None,
        "metadata": None,
    }
    if not output_dir:
        return status

    metadata = read_grpo_training_metadata(output_dir)
    checkpoint_path = resolve_policy_checkpoint(None, output_dir)
    status["checkpoint_path"] = checkpoint_path
    status["metadata"] = metadata

    if checkpoint_path is None:
        status["reason"] = "missing_checkpoint"
        return status
    if metadata is None:
        status["reason"] = "missing_metadata"
        return status
    if metadata.get("stage") != "grpo":
        status["reason"] = "wrong_stage"
        return status
    if int(metadata.get("train_contract_version", -1)) != GRPO_TRAIN_CONTRACT_VERSION:
        status["reason"] = "stale_contract"
        return status

    expected_fingerprint = build_grpo_config_fingerprint(cfg)
    if metadata.get("config_fingerprint") != expected_fingerprint:
        status["reason"] = "config_mismatch"
        return status

    min_reward = float(cfg.get("grpo", {}).get("min_checkpoint_reward", 0.0))
    best_reward = metadata.get("best_reward")
    if best_reward is None or float(best_reward) < min_reward:
        status["reason"] = "reward_below_threshold"
        return status

    status["valid"] = True
    status["reason"] = "ok"
    return status


def inspect_grpo_resume_candidate(output_dir: str, cfg: Dict[str, object]) -> Dict[str, object]:
    """Determine whether an existing GRPO checkpoint is safe to resume from.

    Resume eligibility is intentionally stricter than merely "a checkpoint file
    exists": completed runs that failed the configured quality gate should not be
    resurrected into fresh training/eval cycles.
    """
    status: Dict[str, object] = {
        "valid": False,
        "reason": "missing_output_dir",
        "checkpoint_path": None,
        "metadata": None,
    }
    if not output_dir:
        return status

    checkpoint_path = find_latest_checkpoint(output_dir)
    metadata = read_grpo_training_metadata(output_dir)
    status["checkpoint_path"] = checkpoint_path
    status["metadata"] = metadata

    if checkpoint_path is None:
        status["reason"] = "missing_checkpoint"
        return status

    if metadata is None:
        status["valid"] = True
        status["reason"] = "checkpoint_only"
        return status

    if metadata.get("stage") != "grpo":
        status["reason"] = "wrong_stage"
        return status
    if int(metadata.get("train_contract_version", -1)) != GRPO_TRAIN_CONTRACT_VERSION:
        status["reason"] = "stale_contract"
        return status

    expected_fingerprint = build_grpo_config_fingerprint(cfg)
    if metadata.get("config_fingerprint") != expected_fingerprint:
        status["reason"] = "config_mismatch"
        return status

    min_reward = float(cfg.get("grpo", {}).get("min_checkpoint_reward", 0.0))
    best_reward = metadata.get("best_reward")
    if best_reward is None or float(best_reward) < min_reward:
        status["reason"] = "reward_below_threshold"
        return status

    max_steps = int(metadata.get("max_steps", cfg.get("grpo", {}).get("max_steps", -1)) or -1)
    completed_steps = int(metadata.get("completed_steps", -1) or -1)
    if max_steps >= 0 and completed_steps >= max_steps:
        status["reason"] = "already_complete"
        return status

    status["valid"] = True
    status["reason"] = "ok"
    return status
