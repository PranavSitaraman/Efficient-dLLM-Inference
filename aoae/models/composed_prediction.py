"""
Composed Prediction: Cache-Aligned Token Selection (paper §3.6).

Composes the auxiliary policy's stability estimates with the base model's
token distribution to bias generation toward cache-aligned orderings.

The key idea: positions the policy marks as "stable enough to cache" have
their token distributions sharpened, concentrating mass on the most probable
token (which is most likely to remain stable). This increases KV-cache hit
rates without sacrificing quality at uncertain positions.

    p_tilde(v) ∝ p(v)^{1 + gamma * sigma_k}

where sigma_k is the policy's cache probability for position k and
gamma >= 0 is the composition strength.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compose_prediction(
    base_logits: torch.Tensor,
    cache_probs: torch.Tensor,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Compose base model logits with policy's cache stability signal.

    Single-model variant: sharpens the distribution at positions the
    policy considers stable (high cache probability).

        p_tilde(v) proportional to p(v)^{1 + gamma * sigma_k}

    In log-space: scales logits by (1 + gamma * sigma_k).

    Args:
        base_logits: [B, L, V] raw logits from the base model.
        cache_probs: [B, L] policy's per-position cache probability sigma_k.
        gamma: Composition strength. 0 = no composition, higher = more sharpening.

    Returns:
        composed_logits: [B, L, V] logits after composition.
    """
    if gamma <= 0.0:
        return base_logits

    sigma_k = cache_probs.unsqueeze(-1)  # [B, L, 1]
    scale = 1.0 + gamma * sigma_k  # [B, L, 1]
    return base_logits * scale


def sample_from_composed(
    base_logits: torch.Tensor,
    cache_probs: torch.Tensor,
    gamma: float = 0.5,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Sample tokens from the composed distribution.

    Args:
        base_logits: [B, L, V] raw logits from the base model.
        cache_probs: [B, L] policy's per-position cache probability.
        gamma: Composition strength.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        tokens: [B, L] sampled token ids.
    """
    composed_logits = compose_prediction(base_logits, cache_probs, gamma)

    if temperature <= 0:
        return composed_logits.argmax(dim=-1)

    probs = F.softmax(composed_logits / temperature, dim=-1)
    B, L, V = probs.shape
    flat_probs = probs.view(-1, V)
    tokens = torch.multinomial(flat_probs, 1).view(B, L)
    return tokens


def compose_prediction_dual(
    primary_logits: torch.Tensor,
    auxiliary_logits: torch.Tensor,
    agreement: torch.Tensor,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Compose primary and auxiliary logits for speculative diffusion (paper §3.6).

    Implements Eq. (composed) from the paper:
        p_tilde(v) ∝ p_primary(v) * p_auxiliary(v)^{gamma * alpha_k}

    In log-space:
        log p_tilde(v) = log p_primary(v) + gamma * alpha_k * log p_auxiliary(v)

    At agreement positions (alpha_k=1), the composed distribution concentrates
    mass on the shared high-probability token, increasing cache hit rate.
    At disagreement positions (alpha_k=0), reduces to the primary alone.

    Args:
        primary_logits:   [B, L, V] logits from soft-routed primary.
        auxiliary_logits:  [B, L, V] logits from hard-routed auxiliary.
        agreement:         [B, L] bool/float: 1 where argmax tokens match.
        gamma: Composition strength. 0 = no composition.

    Returns:
        composed_logits: [B, L, V] logits after dual-model composition.
    """
    if gamma <= 0.0:
        return primary_logits

    # Convert auxiliary logits to log-probs for stable composition
    aux_log_probs = F.log_softmax(auxiliary_logits, dim=-1)  # [B, L, V]

    # alpha_k: [B, L, 1] for broadcasting
    alpha_k = agreement.float().unsqueeze(-1)  # [B, L, 1]

    # Compose: primary_logits + gamma * alpha_k * aux_log_probs
    # nan_to_num guards against 0 * -inf = NaN (IEEE 754) in bf16,
    # where log_softmax produces -inf for near-zero-probability tokens and
    # alpha_k=0 at disagreement positions.
    composed = primary_logits + torch.nan_to_num(gamma * alpha_k * aux_log_probs, nan=0.0)

    return composed


def compute_composition_entropy(
    base_logits: torch.Tensor,
    cache_probs: torch.Tensor,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Compute per-position entropy of the composed distribution.

    Useful for diagnostics: verifying that entropy is preserved at
    uncertain positions and reduced at stable ones.

    Args:
        base_logits: [B, L, V] raw logits.
        cache_probs: [B, L] cache probabilities.
        gamma: Composition strength.

    Returns:
        entropy: [B, L] per-position entropy in nats.
    """
    composed_logits = compose_prediction(base_logits, cache_probs, gamma)
    probs = F.softmax(composed_logits, dim=-1)
    log_probs = F.log_softmax(composed_logits, dim=-1)
    entropy = -(torch.nan_to_num(probs * log_probs, nan=0.0)).sum(dim=-1)
    return entropy
