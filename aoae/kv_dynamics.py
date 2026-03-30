"""
KV-dynamics tracking utilities for speculative diffusion.

The tracker prefers exact per-layer KV drift when the backend exposes K/V
states. When exact caches are unavailable, it falls back to the existing
hidden-state drift proxy and labels the resulting artifacts accordingly.
It can also record per-layer attention deviation when attention tensors are
available from the model backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class DynamicsConfig:
    locality_windows: List[int]
    confidence_threshold: float
    keep_step_traces: bool
    attention_proxy_top_frac: float
    track_attention_deviation: bool


def _default_cfg(cfg: dict) -> DynamicsConfig:
    ac = cfg.get("analysis", {})
    return DynamicsConfig(
        locality_windows=list(ac.get("locality_windows", [8, 16, 32])),
        confidence_threshold=float(ac.get("confidence_threshold", 0.9)),
        keep_step_traces=bool(ac.get("keep_step_traces", True)),
        attention_proxy_top_frac=float(ac.get("attention_proxy_top_frac", 0.1)),
        track_attention_deviation=bool(ac.get("track_attention_deviation", True)),
    )


class SpeculativeDynamicsTracker:
    """Tracks per-step and per-layer dynamics for one sample/run."""

    def __init__(self, cfg: dict):
        self.cfg = _default_cfg(cfg)
        self.prev_layers: Optional[List[torch.Tensor]] = None
        self.prev_layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.prev_attentions: Optional[List[torch.Tensor]] = None
        self.prev_changed: Optional[torch.Tensor] = None
        self.age: Optional[torch.Tensor] = None
        self.cached: Optional[torch.Tensor] = None

        self.layer_sum: List[float] = []
        self.layer_count: List[int] = []
        self.attn_dev_sum: List[float] = []
        self.attn_dev_count: List[int] = []

        self.age_drift_sum = {"age0": 0.0, "age1": 0.0, "age2p": 0.0}
        self.age_drift_count = {"age0": 0, "age1": 0, "age2p": 0}

        self.locality_hits = {w: 0 for w in self.cfg.locality_windows}
        self.locality_total = {w: 0 for w in self.cfg.locality_windows}

        self.sum_conf_masked = 0.0
        self.sum_conf_unmasked = 0.0
        self.count_conf_masked = 0
        self.count_conf_unmasked = 0

        self.sum_agreement = 0.0
        self.count_agreement = 0
        self.sum_access = 0.0
        self.count_access = 0

        self.thrash_events = 0
        self.cached_opportunities = 0

        self.sum_confident_drift_ratio = 0.0
        self.count_confident_drift_ratio = 0

        self.layer_drift_kind_counts = {"exact_kv": 0, "hidden_state_proxy": 0}
        self.attention_steps = 0

        self.step_records: List[Dict] = []
        self.steps = 0

    @staticmethod
    def _mean_or_zero(t: torch.Tensor) -> float:
        return float(t.mean().item()) if t.numel() > 0 else 0.0

    @staticmethod
    def _clone_layers(layers: List[torch.Tensor]) -> List[torch.Tensor]:
        return [h.detach().clone() for h in layers]

    @staticmethod
    def _clone_layer_kv(
        layer_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [(k.detach().clone(), v.detach().clone()) for k, v in layer_kv]

    @staticmethod
    def _reduce_attention(attn: torch.Tensor) -> torch.Tensor:
        # Standard HF attention tensors are [B, H, Q, K]. Average over heads so
        # per-layer deviation is comparable across architectures.
        if attn.ndim == 4:
            return attn.mean(dim=1)
        if attn.ndim == 3:
            return attn
        if attn.ndim == 2:
            return attn.unsqueeze(0)
        raise ValueError(f"Unsupported attention tensor rank: {attn.ndim}")

    @staticmethod
    def _token_l2_drift(cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        delta = cur - prev
        if delta.ndim == 4:
            return torch.norm(delta, dim=-1).mean(dim=1)
        if delta.ndim == 3:
            return torch.norm(delta, dim=-1)
        if delta.ndim == 2:
            return torch.norm(delta, dim=-1, keepdim=True).transpose(0, 1)
        raise ValueError(f"Unsupported drift tensor rank: {delta.ndim}")

    def _compute_exact_kv_drift(
        self,
        cur_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        prev_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[List[torch.Tensor]]:
        if len(cur_kv) != len(prev_kv):
            return None

        layer_drifts: List[torch.Tensor] = []
        for (cur_k, cur_v), (prev_k, prev_v) in zip(cur_kv, prev_kv):
            try:
                k_drift = self._token_l2_drift(cur_k, prev_k)
                v_drift = self._token_l2_drift(cur_v, prev_v)
            except ValueError:
                return None
            layer_drifts.append(k_drift + v_drift)
        return layer_drifts

    def _ensure_buffers(self, mask_ind: torch.Tensor):
        if self.age is None:
            self.age = torch.zeros_like(mask_ind, dtype=torch.long)
        if self.prev_changed is None:
            self.prev_changed = torch.zeros_like(mask_ind, dtype=torch.bool)
        if self.cached is None:
            self.cached = torch.zeros_like(mask_ind, dtype=torch.bool)

    def _accumulate_layer_drift(
        self,
        layers: List[torch.Tensor],
        changed: torch.Tensor,
        layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> str:
        drift_by_layer: Optional[List[torch.Tensor]] = None
        drift_kind = "hidden_state_proxy"

        if layer_kv is not None and self.prev_layer_kv is not None:
            drift_by_layer = self._compute_exact_kv_drift(layer_kv, self.prev_layer_kv)
            if drift_by_layer is not None:
                drift_kind = "exact_kv"

        if drift_by_layer is None:
            if self.prev_layers is None:
                return drift_kind
            drift_by_layer = [
                self._token_l2_drift(cur, prev)
                for cur, prev in zip(layers, self.prev_layers)
            ]

        self.layer_drift_kind_counts[drift_kind] += 1
        for i, drift in enumerate(drift_by_layer):
            drift_for_stats = drift
            if valid_mask is not None:
                drift_for_stats = drift[valid_mask]
                if drift_for_stats.numel() == 0:
                    continue
            if len(self.layer_sum) <= i:
                self.layer_sum.append(0.0)
                self.layer_count.append(0)
            self.layer_sum[i] += float(drift_for_stats.mean().item())
            self.layer_count[i] += 1

            if self.age is not None:
                age0 = self.age == 0
                age1 = self.age == 1
                age2p = self.age >= 2
                for name, mask in (("age0", age0), ("age1", age1), ("age2p", age2p)):
                    if valid_mask is not None:
                        mask = mask & valid_mask
                    vals = drift[mask]
                    if vals.numel() > 0:
                        self.age_drift_sum[name] += float(vals.mean().item())
                        self.age_drift_count[name] += 1
        return drift_kind

    def _accumulate_attention_deviation(self, layer_attentions: Optional[List[torch.Tensor]]) -> bool:
        if not self.cfg.track_attention_deviation or not layer_attentions:
            return False
        if self.prev_attentions is None or len(layer_attentions) != len(self.prev_attentions):
            return False

        for i, (cur, prev) in enumerate(zip(layer_attentions, self.prev_attentions)):
            cur_reduced = self._reduce_attention(cur)
            prev_reduced = self._reduce_attention(prev)
            if cur_reduced.shape != prev_reduced.shape:
                continue
            if len(self.attn_dev_sum) <= i:
                self.attn_dev_sum.append(0.0)
                self.attn_dev_count.append(0)
            delta = cur_reduced - prev_reduced
            deviation = torch.norm(delta, dim=-1)
            self.attn_dev_sum[i] += float(deviation.mean().item())
            self.attn_dev_count[i] += 1

        self.attention_steps += 1
        return True

    def _accumulate_locality(self, changed: torch.Tensor):
        if self.prev_changed is None:
            return
        bsz = changed.shape[0]
        for b in range(bsz):
            cur_idx = changed[b].nonzero(as_tuple=True)[0]
            prev_idx = self.prev_changed[b].nonzero(as_tuple=True)[0]
            if cur_idx.numel() == 0:
                continue
            for w in self.cfg.locality_windows:
                self.locality_total[w] += int(cur_idx.numel())
                if prev_idx.numel() == 0:
                    continue
                # distance from each current changed token to nearest previous changed token
                d = (cur_idx[:, None] - prev_idx[None, :]).abs().min(dim=1).values
                self.locality_hits[w] += int((d <= w).sum().item())

    def observe_step(
        self,
        layer_hiddens: List[torch.Tensor],
        max_prob: torch.Tensor,
        mask_ind: torch.Tensor,
        agreement: torch.Tensor,
        u_t: torch.Tensor,
        r_t: torch.Tensor,
        kappa_t: torch.Tensor,
        q_t: Optional[torch.Tensor] = None,
        layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        layer_attentions: Optional[List[torch.Tensor]] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """Observe one diffusion step (response-region tensors only)."""
        self._ensure_buffers(mask_ind)
        changed = (u_t.bool() | r_t.bool())
        drift_kind = self._accumulate_layer_drift(
            layer_hiddens, changed=changed, layer_kv=layer_kv, valid_mask=valid_mask,
        )
        self._accumulate_locality(changed)
        has_attention_deviation = self._accumulate_attention_deviation(layer_attentions)

        if self.prev_layers is not None and layer_hiddens:
            # Proxy for "most-attended tokens drift less":
            # use top-confidence positions as a lightweight attention proxy.
            drift_last = torch.norm(layer_hiddens[-1] - self.prev_layers[-1], dim=-1)
            if valid_mask is not None:
                flat_prob = max_prob[valid_mask]
                flat_drift = drift_last[valid_mask]
            else:
                flat_prob = max_prob.reshape(-1)
                flat_drift = drift_last.reshape(-1)
            if flat_prob.numel() > 0:
                frac = max(min(self.cfg.attention_proxy_top_frac, 1.0), 1e-3)
                k = max(1, int(frac * flat_prob.numel()))
                top_idx = flat_prob.topk(k=k).indices
                top_mean = flat_drift[top_idx].mean()
                global_mean = flat_drift.mean()
                ratio = float((top_mean / (global_mean + 1e-8)).item())
                self.sum_confident_drift_ratio += ratio
                self.count_confident_drift_ratio += 1

        if valid_mask is not None:
            masked_vals = max_prob[mask_ind & valid_mask]
            unmasked_vals = max_prob[(~mask_ind) & valid_mask]
            agreement_vals = agreement[valid_mask]
            access_vals = q_t[valid_mask] if q_t is not None else None
        else:
            masked_vals = max_prob[mask_ind]
            unmasked_vals = max_prob[~mask_ind]
            agreement_vals = agreement
            access_vals = q_t if q_t is not None else None
        if masked_vals.numel() > 0:
            self.sum_conf_masked += float(masked_vals.mean().item())
            self.count_conf_masked += 1
        if unmasked_vals.numel() > 0:
            self.sum_conf_unmasked += float(unmasked_vals.mean().item())
            self.count_conf_unmasked += 1

        if agreement_vals.numel() > 0:
            self.sum_agreement += float(agreement_vals.float().mean().item())
            self.count_agreement += 1
        if access_vals is not None and access_vals.numel() > 0:
            self.sum_access += float(access_vals.float().mean().item())
            self.count_access += 1

        if self.cached is not None:
            self.cached_opportunities += int(self.cached.sum().item())
            self.thrash_events += int((self.cached & r_t.bool()).sum().item())

        if self.cfg.keep_step_traces:
            self.step_records.append(
                {
                    "step_index": self.steps,
                    "masked_ratio": float(mask_ind.float().mean().item()),
                    "unmask_ratio": float(u_t.float().mean().item()),
                    "remask_ratio": float(r_t.float().mean().item()),
                    "changed_ratio": float(changed.float().mean().item()),
                    "cache_commit_ratio": float(kappa_t.float().mean().item()),
                    "access_ratio": float(q_t.float().mean().item()) if q_t is not None else 0.0,
                    "agreement_ratio": float(agreement.float().mean().item()),
                    "mean_confidence_masked": self._mean_or_zero(masked_vals),
                    "mean_confidence_unmasked": self._mean_or_zero(unmasked_vals),
                    "layer_drift_kind": drift_kind,
                    "has_attention_deviation": has_attention_deviation,
                }
            )

        # Update temporal buffers for next step
        self.age = self.age + 1
        self.age[changed] = 0
        self.cached = (self.cached | kappa_t.bool()) & (~r_t.bool())
        self.prev_changed = changed.clone()
        self.prev_layers = self._clone_layers(layer_hiddens)
        self.prev_layer_kv = self._clone_layer_kv(layer_kv) if layer_kv else None
        self.prev_attentions = self._clone_layers(layer_attentions) if layer_attentions else None
        self.steps += 1

    def summarize(self) -> Dict:
        layer_means = []
        for s, c in zip(self.layer_sum, self.layer_count):
            layer_means.append(s / max(c, 1))

        if len(layer_means) >= 2:
            x = np.arange(len(layer_means), dtype=np.float32)
            y = np.array(layer_means, dtype=np.float32)
            slope = float(np.polyfit(x, y, deg=1)[0])
        else:
            slope = 0.0

        attn_layer_means = []
        for s, c in zip(self.attn_dev_sum, self.attn_dev_count):
            attn_layer_means.append(s / max(c, 1))
        if len(attn_layer_means) >= 2:
            x = np.arange(len(attn_layer_means), dtype=np.float32)
            y = np.array(attn_layer_means, dtype=np.float32)
            attn_slope = float(np.polyfit(x, y, deg=1)[0])
        else:
            attn_slope = 0.0

        age_means = {}
        for k in self.age_drift_sum:
            age_means[k] = self.age_drift_sum[k] / max(self.age_drift_count[k], 1)
        off_by_one_ratio = age_means["age1"] / max(age_means["age2p"], 1e-8)

        locality = {}
        for w in self.cfg.locality_windows:
            locality[f"w{w}_hit_ratio"] = self.locality_hits[w] / max(self.locality_total[w], 1)

        per_layer = [
            {
                "layer_idx": i,
                "mean_drift": float(v),
                "mean_hidden_drift": float(v),
            }
            for i, v in enumerate(layer_means)
        ]
        per_layer_attn = [
            {
                "layer_idx": i,
                "mean_attention_deviation": float(v),
                "deviation_measure": "mean_l2_delta",
            }
            for i, v in enumerate(attn_layer_means)
        ]

        drift_kind = "hidden_state_proxy"
        if self.layer_drift_kind_counts["exact_kv"] > 0 and self.layer_drift_kind_counts["hidden_state_proxy"] == 0:
            drift_kind = "exact_kv"
        elif self.layer_drift_kind_counts["exact_kv"] > 0 and self.layer_drift_kind_counts["hidden_state_proxy"] > 0:
            drift_kind = "mixed"

        summary = {
            "num_steps": self.steps,
            "mean_agreement": self.sum_agreement / max(self.count_agreement, 1),
            "mean_access": self.sum_access / max(self.count_access, 1),
            "mean_confidence_masked": self.sum_conf_masked / max(self.count_conf_masked, 1),
            "mean_confidence_unmasked": self.sum_conf_unmasked / max(self.count_conf_unmasked, 1),
            "layer_drift_measure": drift_kind,
            "exact_kv_drift_steps": self.layer_drift_kind_counts["exact_kv"],
            "hidden_state_proxy_steps": self.layer_drift_kind_counts["hidden_state_proxy"],
            "layer_drift_slope": slope,
            "off_by_one_drift_ratio": off_by_one_ratio,
            "confident_token_drift_ratio": self.sum_confident_drift_ratio / max(self.count_confident_drift_ratio, 1),
            "thrash_rate_given_cached": self.thrash_events / max(self.cached_opportunities, 1),
            "locality": locality,
            "age_drift_means": age_means,
            "attention_deviation_available": bool(attn_layer_means),
            "attention_deviation_measure": "mean_l2_delta" if attn_layer_means else "unavailable",
            "attention_deviation_steps": self.attention_steps,
            "attention_deviation_slope": attn_slope,
            "top_token_proxy_kind": "top_confidence_positions",
        }

        return {
            "summary": summary,
            "per_layer": per_layer,
            "per_layer_attention_deviation": per_layer_attn,
            "per_step": self.step_records,
        }
