"""Shared checkpoint and sidecar artifact helpers."""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import torch


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
    """Return the numerically latest step checkpoint in an output directory."""
    if not output_dir:
        return None
    ckpts = glob.glob(os.path.join(output_dir, pattern))
    if not ckpts:
        return None

    def _step_num(path: str) -> int:
        base = os.path.basename(path)
        digits = "".join(ch for ch in base if ch.isdigit())
        try:
            return int(digits)
        except ValueError:
            return -1

    ckpts.sort(key=_step_num)
    return ckpts[-1]


def resolve_policy_checkpoint(
    explicit: Optional[str],
    output_dir: str,
) -> Optional[str]:
    """Resolve an explicit checkpoint path or auto-detect the best available one."""
    if explicit:
        return explicit
    for name in ("policy_best.pt", "policy_final.pt"):
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
