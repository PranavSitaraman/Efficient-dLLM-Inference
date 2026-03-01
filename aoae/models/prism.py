"""
PRISM Quality Adapter (paper §2.4, §3.2).

A lightweight MLP that estimates per-token quality scores q_t^k — the
probability that a generated token matches the ground truth.  Trained with
binary cross-entropy on (corrupted input, clean target) pairs.

At inference time, the edit subroutine uses q_t^k < delta to decide
whether to remask (PRISM-style self-correction) or T2T-edit a position.

Reference: Kim et al. "PRISM: Plug-in Remasking for Inference-time
Self-correction of Masked Diffusions" (2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Any


class PRISMAdapter(nn.Module):
    """
    Lightweight quality head: hidden states → per-token quality score ∈ [0, 1].

    Architecture: 2-layer MLP on top of base-model hidden states.
    """

    def __init__(self, cfg, hidden_dim: int):
        """
        Args:
            cfg:        full config dict.
            hidden_dim: dimension of base-model hidden states.
        """
        super().__init__()
        pc = cfg["prism"]
        mid = pc["hidden_dim"]
        self.threshold = pc["threshold"]

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D] from base model last hidden layer.
        Returns:
            quality_scores: [B, L] in (0, 1).
        """
        return torch.sigmoid(self.net(hidden_states).squeeze(-1))

    def should_remask(self, quality_scores: torch.Tensor) -> torch.BoolTensor:
        """Return True where quality is below threshold → remask."""
        return quality_scores < self.threshold


# ======================================================================
# Training utilities for the PRISM adapter
# ======================================================================

def create_prism_training_data(
    tokenizer,
    dataset,
    mask_token_id: int,
    max_samples: int = 10000,
    mask_ratio_range: Tuple[float, float] = (0.2, 0.8),
    max_length: int = 512,
) -> List[Dict]:
    """
    Create (corrupted, clean, labels) pairs for PRISM adapter training.

    For each sample, we:
      1. Tokenize the text.
      2. Sample a mask ratio uniformly from mask_ratio_range.
      3. Corrupt by replacing that fraction of tokens with [M].
      4. Label = 1 where the corrupted token matches clean, 0 otherwise.
         (For masked positions, label is always 0.)

    Returns:
        List of dicts with keys: "input_ids", "clean_ids", "labels".
    """
    import random

    def _extract_text(sample: Dict[str, Any]) -> Optional[str]:
        # Common math dataset fields
        for key in (
            "answer",
            "solution",
            "generated_solution",
            "rationale",
            "text",
            "output",
            "response",
            "completion",
        ):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Chat-style schema: messages/conversations list[{content: ...}]
        for key in ("messages", "conversations"):
            value = sample.get(key)
            if isinstance(value, list):
                chunks = []
                for turn in value:
                    if isinstance(turn, dict):
                        content = turn.get("content")
                        if isinstance(content, str) and content.strip():
                            chunks.append(content.strip())
                if chunks:
                    return "\n".join(chunks)

        # Last fallback: pick the longest non-empty string field
        best = ""
        for v in sample.values():
            if isinstance(v, str) and len(v.strip()) > len(best):
                best = v.strip()
        return best if best else None

    pairs = []
    skipped_empty = 0
    skipped_short = 0
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        text = _extract_text(sample)
        if not text:
            skipped_empty += 1
            continue

        ids = tokenizer.encode(text, max_length=max_length, truncation=True)
        if len(ids) < 10:
            skipped_short += 1
            continue

        clean = torch.tensor(ids, dtype=torch.long)
        L = clean.shape[0]

        ratio = random.uniform(*mask_ratio_range)
        n_mask = max(1, int(L * ratio))
        mask_pos = torch.randperm(L)[:n_mask]

        corrupted = clean.clone()
        corrupted[mask_pos] = mask_token_id

        # Label: 1 where token is correct (uncorrupted and matches clean), 0 otherwise
        labels = (corrupted == clean).float()

        pairs.append({
            "input_ids": corrupted,
            "clean_ids": clean,
            "labels": labels,
        })

    if not pairs:
        raise RuntimeError(
            "PRISM training data creation produced 0 pairs. "
            f"Checked up to {max_samples} samples; skipped_empty={skipped_empty}, "
            f"skipped_short={skipped_short}. "
            "Dataset schema likely mismatches expected text fields."
        )

    return pairs


def train_prism_adapter(
    adapter: PRISMAdapter,
    base_model,
    training_data: List[Dict],
    cfg: dict,
    device: torch.device,
) -> PRISMAdapter:
    """
    Train the PRISM adapter with BCE loss on (corrupted, labels) pairs.

    The base model is run in eval/no_grad mode to produce hidden states;
    only the adapter parameters are updated.

    When torch.distributed is initialized (DDP / torchrun), the adapter is
    wrapped in DistributedDataParallel and each rank processes its own shard
    of the training data so gradients sync correctly.
    """
    import torch.distributed as dist
    pc = cfg["prism"]
    batch_size = pc["batch_size"]
    epochs = pc["epochs"]

    # --- Shard data across ranks when using DDP ---
    ddp_active = dist.is_initialized() and dist.get_world_size() > 1
    if ddp_active:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        shard_size = (len(training_data) + world_size - 1) // world_size
        start = rank * shard_size
        end = min(start + shard_size, len(training_data))
        training_data = training_data[start:end]

    adapter = adapter.to(device).train()

    # --- Wrap in DDP for gradient synchronisation ---
    if ddp_active:
        from torch.nn.parallel import DistributedDataParallel as DDP
        adapter_ddp = DDP(adapter, device_ids=[device.index])
    else:
        adapter_ddp = adapter

    optimizer = torch.optim.AdamW(adapter_ddp.parameters(), lr=pc["lr"])

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i : i + batch_size]

            # Pad to same length
            max_len = max(d["input_ids"].shape[0] for d in batch)
            input_ids = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            labels = torch.zeros(len(batch), max_len, device=device)

            for j, d in enumerate(batch):
                L = d["input_ids"].shape[0]
                input_ids[j, :L] = d["input_ids"]
                labels[j, :L] = d["labels"]

            # Get hidden states from frozen base model (no LM head — saves ~7 GiB)
            with torch.no_grad():
                hidden = base_model.forward_hidden_only(input_ids)
                hidden = hidden.float()

            # Forward through adapter (DDP-wrapped or bare)
            q_scores = adapter_ddp(hidden)  # [B, L]
            loss = F.binary_cross_entropy(q_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if not ddp_active or dist.get_rank() == 0:
            print(f"  PRISM epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    # Unwrap DDP to get the bare module for saving / returning
    final_adapter = adapter_ddp.module if ddp_active else adapter_ddp
    final_adapter.eval()
    for p in final_adapter.parameters():
        p.requires_grad_(False)

    return final_adapter
