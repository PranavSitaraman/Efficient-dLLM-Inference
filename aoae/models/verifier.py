"""
Plug-and-play verifier modules for speculative AOAE.

The verifier is conceptually separate from:
  1. the drafter / auxiliary model used for speculative proposals, and
  2. the persistent KV-stability cache (K_stable).

This module exposes a small interface so the speculation loop can use:
  - a frozen PRISM adapter,
  - an unfrozen / GRPO-tunable PRISM adapter,
  - a separate learned verification head, or
  - a heuristic logits-only verifier such as confidence / entropy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prism import PRISMAdapter


def _verifier_cfg(cfg: dict) -> dict:
    vc = cfg.get("verifier")
    if isinstance(vc, dict):
        return vc
    # Backward-compatible default: legacy PRISM sidecar behavior.
    return {
        "enabled": True,
        "kind": "prism",
        "trainable": False,
    }


def verifier_enabled(cfg: dict) -> bool:
    vc = _verifier_cfg(cfg)
    return bool(vc.get("enabled", True)) and str(vc.get("kind", "prism")).lower() != "none"


def verifier_kind(cfg: dict) -> str:
    return str(_verifier_cfg(cfg).get("kind", "prism")).strip().lower()


def verifier_trainable(cfg: dict) -> bool:
    return bool(_verifier_cfg(cfg).get("trainable", False))


def verifier_artifact_name(cfg: dict) -> Optional[str]:
    vc = _verifier_cfg(cfg)
    if "artifact_name" in vc and vc["artifact_name"]:
        return str(vc["artifact_name"])

    kind = verifier_kind(cfg)
    if kind == "prism":
        return "prism_adapter.pt"
    if kind in {"learned_head", "quality_head", "mlp_head"}:
        return "verifier_head.pt"
    return None


@dataclass
class VerifierBuildInfo:
    enabled: bool
    kind: str
    trainable: bool
    artifact_name: Optional[str]
    loaded_from: Optional[str] = None
    initialized_fresh: bool = False


class BaseVerifier(nn.Module):
    """Small interface for interchangeable verifier modules."""

    needs_hidden_states: bool = False
    needs_logits: bool = False
    artifact_name: Optional[str] = None

    def score(
        self,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        mask_indicator: Optional[torch.Tensor] = None,
        step_frac: Optional[float] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        mask_indicator: Optional[torch.Tensor] = None,
        step_frac: Optional[float] = None,
    ) -> torch.Tensor:
        return self.score(
            hidden_states=hidden_states,
            logits=logits,
            agreement=agreement,
            mask_indicator=mask_indicator,
            step_frac=step_frac,
        )


class PRISMVerifier(BaseVerifier):
    """Wrapper around the legacy PRISMAdapter under the generic verifier interface."""

    needs_hidden_states = True
    needs_logits = False
    artifact_name = "prism_adapter.pt"

    def __init__(self, cfg: dict, hidden_dim: int):
        super().__init__()
        self.adapter = PRISMAdapter(cfg, hidden_dim)

    @property
    def threshold(self) -> float:
        return float(self.adapter.threshold)

    def score(
        self,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        mask_indicator: Optional[torch.Tensor] = None,
        step_frac: Optional[float] = None,
    ) -> torch.Tensor:
        del logits, agreement, mask_indicator, step_frac
        if hidden_states is None:
            raise RuntimeError("PRISMVerifier requires hidden_states, but none were provided.")
        return self.adapter(hidden_states)

    def should_remask(self, quality_scores: torch.Tensor) -> torch.BoolTensor:
        return self.adapter.should_remask(quality_scores)


class LearnedVerificationHead(BaseVerifier):
    """Trainable PRISM-like verifier head initialized from scratch or a sidecar."""

    needs_hidden_states = True
    needs_logits = False
    artifact_name = "verifier_head.pt"

    def __init__(self, cfg: dict, hidden_dim: int):
        super().__init__()
        vc = _verifier_cfg(cfg)
        prism_cfg = cfg.get("prism", {})
        mid = int(vc.get("hidden_dim", prism_cfg.get("hidden_dim", 256)))
        self.threshold = float(vc.get("threshold", prism_cfg.get("threshold", 0.5)))
        dropout = float(vc.get("dropout", 0.0))

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def score(
        self,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        mask_indicator: Optional[torch.Tensor] = None,
        step_frac: Optional[float] = None,
    ) -> torch.Tensor:
        del logits, agreement, mask_indicator, step_frac
        if hidden_states is None:
            raise RuntimeError("LearnedVerificationHead requires hidden_states, but none were provided.")
        hidden_states = torch.nan_to_num(
            hidden_states.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        scores = torch.sigmoid(self.net(hidden_states).squeeze(-1))
        return torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def should_remask(self, quality_scores: torch.Tensor) -> torch.BoolTensor:
        return quality_scores < self.threshold


class ConfidenceVerifier(BaseVerifier):
    """Logits-only verifier that turns confidence / entropy into a quality score."""

    needs_hidden_states = False
    needs_logits = True
    artifact_name = None

    def __init__(self, cfg: dict):
        super().__init__()
        vc = _verifier_cfg(cfg)
        self.mode = str(vc.get("score_mode", "max_prob")).strip().lower()

    def score(
        self,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        agreement: Optional[torch.Tensor] = None,
        mask_indicator: Optional[torch.Tensor] = None,
        step_frac: Optional[float] = None,
    ) -> torch.Tensor:
        del hidden_states, agreement, mask_indicator, step_frac
        if logits is None:
            raise RuntimeError("ConfidenceVerifier requires logits, but none were provided.")

        logits_f = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.mode == "margin":
            top2 = torch.topk(logits_f, k=min(2, logits_f.shape[-1]), dim=-1).values
            if top2.shape[-1] == 1:
                score = torch.ones_like(top2[..., 0])
            else:
                margin = top2[..., 0] - top2[..., 1]
                score = torch.sigmoid(margin)
        elif self.mode in {"one_minus_entropy", "entropy"}:
            probs = F.softmax(logits_f, dim=-1)
            log_probs = F.log_softmax(logits_f, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(logits_f.shape[-1], device=logits_f.device, dtype=logits_f.dtype))
            score = 1.0 - (entropy / max_entropy.clamp(min=1e-8))
        else:
            max_logits = logits_f.max(dim=-1).values
            score = torch.exp(max_logits - torch.logsumexp(logits_f, dim=-1))

        return torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def build_verifier(cfg: dict, hidden_dim: int) -> Optional[BaseVerifier]:
    """Instantiate the verifier requested by config without loading weights."""
    if not verifier_enabled(cfg):
        return None

    kind = verifier_kind(cfg)
    if kind == "prism":
        return PRISMVerifier(cfg, hidden_dim)
    if kind in {"learned_head", "quality_head", "mlp_head"}:
        return LearnedVerificationHead(cfg, hidden_dim)
    if kind in {"confidence", "logits_confidence", "heuristic"}:
        return ConfidenceVerifier(cfg)
    if kind == "none":
        return None
    raise ValueError(
        f"Unsupported verifier.kind={kind!r}. "
        "Choose from {'prism', 'learned_head', 'confidence', 'none'}."
    )


def verifier_requires_hidden_states(verifier: Optional[nn.Module]) -> bool:
    verifier = verifier.module if hasattr(verifier, "module") else verifier
    if isinstance(verifier, PRISMAdapter):
        return True
    return bool(getattr(verifier, "needs_hidden_states", False))


def verifier_requires_logits(verifier: Optional[nn.Module]) -> bool:
    verifier = verifier.module if hasattr(verifier, "module") else verifier
    if isinstance(verifier, PRISMAdapter):
        return False
    return bool(getattr(verifier, "needs_logits", False))


def run_verifier(
    verifier: Optional[nn.Module],
    *,
    hidden_states: Optional[torch.Tensor] = None,
    logits: Optional[torch.Tensor] = None,
    agreement: Optional[torch.Tensor] = None,
    mask_indicator: Optional[torch.Tensor] = None,
    step_frac: Optional[float] = None,
) -> Optional[torch.Tensor]:
    if verifier is None:
        return None
    inner = verifier.module if hasattr(verifier, "module") else verifier
    if isinstance(inner, PRISMAdapter):
        if hidden_states is None:
            raise RuntimeError("Legacy PRISMAdapter verifier requires hidden_states.")
        return inner(hidden_states)
    with torch.set_grad_enabled(any(p.requires_grad for p in verifier.parameters())):
        return verifier(
            hidden_states=hidden_states,
            logits=logits,
            agreement=agreement,
            mask_indicator=mask_indicator,
            step_frac=step_frac,
        )


def _load_state_flexible(module: nn.Module, state_dict: Dict[str, torch.Tensor], label: str) -> None:
    from ..checkpoints import load_state_dict_flexible

    target = module.module if hasattr(module, "module") else module
    load_state_dict_flexible(target, state_dict, label)


def _normalize_prism_state_keys(
    verifier: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Accept both legacy PRISMAdapter keys and wrapped PRISMVerifier keys."""
    own = verifier.state_dict()
    if not state_dict:
        return state_dict
    if any(key in own for key in state_dict):
        return state_dict
    prefixed = {f"adapter.{key}": value for key, value in state_dict.items()}
    if any(key in own for key in prefixed):
        return prefixed
    return state_dict


def export_verifier_state(
    verifier: Optional[nn.Module],
    *,
    for_artifact: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Return a checkpointable verifier state dict.

    For PRISMVerifier artifacts we preserve legacy compatibility by saving the
    inner adapter weights under the original PRISMAdapter key names.
    """
    if verifier is None:
        return None
    target = verifier.module if hasattr(verifier, "module") else verifier
    if for_artifact and isinstance(target, PRISMVerifier):
        return target.adapter.state_dict()
    return target.state_dict()


def create_or_load_verifier(
    cfg: dict,
    hidden_dim: int,
    device: torch.device,
    *,
    artifact_path: Optional[str] = None,
    checkpoint_state: Optional[Dict[str, torch.Tensor]] = None,
    allow_fresh_init: bool = False,
    verbose: bool = False,
) -> Tuple[Optional[BaseVerifier], VerifierBuildInfo]:
    """Instantiate, optionally load, and freeze/unfreeze the configured verifier."""
    enabled = verifier_enabled(cfg)
    kind = verifier_kind(cfg)
    trainable = verifier_trainable(cfg)
    artifact_name = verifier_artifact_name(cfg)

    info = VerifierBuildInfo(
        enabled=enabled,
        kind=kind,
        trainable=trainable,
        artifact_name=artifact_name,
    )
    if not enabled:
        return None, info

    verifier = build_verifier(cfg, hidden_dim)
    if verifier is None:
        return None, info
    verifier = verifier.to(device)

    loaded = False
    if checkpoint_state is not None:
        if kind == "prism":
            checkpoint_state = _normalize_prism_state_keys(verifier, checkpoint_state)
        _load_state_flexible(verifier, checkpoint_state, "verifier")
        info.loaded_from = "checkpoint"
        loaded = True
    elif artifact_path and os.path.exists(artifact_path):
        state = torch.load(artifact_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if isinstance(state, dict):
            if kind == "prism":
                state = _normalize_prism_state_keys(verifier, state)
            _load_state_flexible(verifier, state, "verifier")
            info.loaded_from = artifact_path
            loaded = True

    if not loaded and trainable and allow_fresh_init:
        info.initialized_fresh = True
    elif not loaded and not any(p.numel() for p in verifier.parameters()):
        # Heuristic verifiers have no learned state to load.
        info.loaded_from = "stateless"
    elif not loaded and artifact_name is None:
        info.loaded_from = "stateless"
    elif not loaded and not allow_fresh_init:
        if verbose:
            print(
                f"[Verifier] No artifact found for verifier.kind={kind!r}; "
                "disabling verifier for this run."
            )
        return None, info
    elif not loaded and verbose:
        print(
            f"[Verifier] No artifact found for verifier.kind={kind!r}. "
            f"{'Initializing fresh trainable verifier.' if trainable and allow_fresh_init else 'Falling back to random/no verifier weights.'}"
        )

    if not trainable:
        verifier.eval()
        for param in verifier.parameters():
            param.requires_grad_(False)
    return verifier, info
