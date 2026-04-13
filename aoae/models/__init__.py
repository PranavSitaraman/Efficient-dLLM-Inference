"""Model components for AOAE.

Modules:
  - base_model: Frozen LLaDA wrapper (hf, dkv, dinfer, soft_moe backends).
  - dual_model: DualModelWrapper for speculative diffusion (hard aux + soft primary).
  - policy: Lightweight steering policy with unmask/remask/cache heads (D+4 input).
  - soft_mask: Soft-masked state construction from model logits.
  - soft_moe: SoftMoERouter for hard-top-k-preserving widened routing.
  - composed_prediction: Dual-model composed prediction for cache-aligned tokens.
  - prism: PRISM quality adapter for self-correction signals.
  - verifier: Plug-and-play verifier backends (PRISM, learned head, confidence).
"""

from .policy import AOAEPolicy, DefaultPolicy
from .soft_mask import SoftMaskedState
from .prism import PRISMAdapter
from .verifier import (
    BaseVerifier,
    PRISMVerifier,
    LearnedVerificationHead,
    ConfidenceVerifier,
    build_verifier,
    create_or_load_verifier,
    export_verifier_state,
    run_verifier,
    verifier_artifact_name,
    verifier_enabled,
    verifier_kind,
    verifier_requires_hidden_states,
    verifier_requires_logits,
    verifier_trainable,
)
from .composed_prediction import (
    compose_prediction,
    compose_prediction_dual,
    sample_from_composed,
)

__all__ = [
    "AOAEPolicy",
    "DefaultPolicy",
    "SoftMaskedState",
    "PRISMAdapter",
    "BaseVerifier",
    "PRISMVerifier",
    "LearnedVerificationHead",
    "ConfidenceVerifier",
    "build_verifier",
    "create_or_load_verifier",
    "export_verifier_state",
    "run_verifier",
    "verifier_artifact_name",
    "verifier_enabled",
    "verifier_kind",
    "verifier_requires_hidden_states",
    "verifier_requires_logits",
    "verifier_trainable",
    "compose_prediction",
    "compose_prediction_dual",
    "sample_from_composed",
]
