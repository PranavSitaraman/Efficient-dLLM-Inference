"""
KV-dynamics proxy tracking utilities for speculative diffusion.

This tracker records structured metrics inspired by Elastic-Cache/dKV-Cache:
layer drift profiles, off-by-1 stabilization proxies, temporal locality, and
confidence/agreement dynamics across diffusion steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class DynamicsConfig:
    locality_windows: List[int]
    confidence_threshold: float
    keep_step_traces: bool
    attention_proxy_top_frac: float


def _default_cfg(cfg: dict) -> DynamicsConfig:
    ac = cfg.get("analysis", {})
    return DynamicsConfig(
        locality_windows=list(ac.get("locality_windows", [8, 16, 32])),
        confidence_threshold=float(ac.get("confidence_threshold", 0.9)),
        keep_step_traces=bool(ac.get("keep_step_traces", True)),
        attention_proxy_top_frac=float(ac.get("attention_proxy_top_frac", 0.1)),
    )


class SpeculativeDynamicsTracker:
    """Tracks per-step and per-layer dynamics for one sample/run."""

    def __init__(self, cfg: dict):
        self.cfg = _default_cfg(cfg)
        self.prev_layers: Optional[List[torch.Tensor]] = None
        self.prev_changed: Optional[torch.Tensor] = None
        self.age: Optional[torch.Tensor] = None
        self.cached: Optional[torch.Tensor] = None

        self.layer_sum: List[float] = []
        self.layer_count: List[int] = []

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

        self.step_records: List[Dict] = []
        self.steps = 0

    @staticmethod
    def _mean_or_zero(t: torch.Tensor) -> float:
        return float(t.mean().item()) if t.numel() > 0 else 0.0

    def _ensure_buffers(self, mask_ind: torch.Tensor):
        if self.age is None:
            self.age = torch.zeros_like(mask_ind, dtype=torch.long)
        if self.prev_changed is None:
            self.prev_changed = torch.zeros_like(mask_ind, dtype=torch.bool)
        if self.cached is None:
            self.cached = torch.zeros_like(mask_ind, dtype=torch.bool)

    def _accumulate_layer_drift(self, layers: List[torch.Tensor], changed: torch.Tensor):
        if self.prev_layers is None:
            return

        for i, (cur, prev) in enumerate(zip(layers, self.prev_layers)):
            if len(self.layer_sum) <= i:
                self.layer_sum.append(0.0)
                self.layer_count.append(0)
            drift = torch.norm(cur - prev, dim=-1)  # [B, L]
            self.layer_sum[i] += float(drift.mean().item())
            self.layer_count[i] += 1

            if self.age is not None:
                age0 = self.age == 0
                age1 = self.age == 1
                age2p = self.age >= 2
                for name, mask in (("age0", age0), ("age1", age1), ("age2p", age2p)):
                    vals = drift[mask]
                    if vals.numel() > 0:
                        self.age_drift_sum[name] += float(vals.mean().item())
                        self.age_drift_count[name] += 1

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
    ):
        """Observe one diffusion step (response-region tensors only)."""
        self._ensure_buffers(mask_ind)
        changed = (u_t.bool() | r_t.bool())
        self._accumulate_layer_drift(layer_hiddens, changed=changed)
        self._accumulate_locality(changed)

        if self.prev_layers is not None and layer_hiddens:
            # Proxy for "most-attended tokens drift less":
            # use top-confidence positions as a lightweight attention proxy.
            drift_last = torch.norm(layer_hiddens[-1] - self.prev_layers[-1], dim=-1)
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

        masked_vals = max_prob[mask_ind]
        unmasked_vals = max_prob[~mask_ind]
        if masked_vals.numel() > 0:
            self.sum_conf_masked += float(masked_vals.mean().item())
            self.count_conf_masked += 1
        if unmasked_vals.numel() > 0:
            self.sum_conf_unmasked += float(unmasked_vals.mean().item())
            self.count_conf_unmasked += 1

        self.sum_agreement += float(agreement.float().mean().item())
        self.count_agreement += 1
        if q_t is not None:
            self.sum_access += float(q_t.float().mean().item())
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
                }
            )

        # Update temporal buffers for next step
        self.age = self.age + 1
        self.age[changed] = 0
        self.cached = (self.cached | kappa_t.bool()) & (~r_t.bool())
        self.prev_changed = changed.clone()
        self.prev_layers = [h.detach().clone() for h in layer_hiddens]
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

        age_means = {}
        for k in self.age_drift_sum:
            age_means[k] = self.age_drift_sum[k] / max(self.age_drift_count[k], 1)
        off_by_one_ratio = age_means["age1"] / max(age_means["age2p"], 1e-8)

        locality = {}
        for w in self.cfg.locality_windows:
            locality[f"w{w}_hit_ratio"] = self.locality_hits[w] / max(self.locality_total[w], 1)

        per_layer = [
            {"layer_idx": i, "mean_hidden_drift": float(v)}
            for i, v in enumerate(layer_means)
        ]

        summary = {
            "num_steps": self.steps,
            "mean_agreement": self.sum_agreement / max(self.count_agreement, 1),
            "mean_access": self.sum_access / max(self.count_access, 1),
            "mean_confidence_masked": self.sum_conf_masked / max(self.count_conf_masked, 1),
            "mean_confidence_unmasked": self.sum_conf_unmasked / max(self.count_conf_unmasked, 1),
            "layer_drift_slope": slope,
            "off_by_one_drift_ratio": off_by_one_ratio,
            "confident_token_drift_ratio": self.sum_confident_drift_ratio / max(self.count_confident_drift_ratio, 1),
            "thrash_rate_given_cached": self.thrash_events / max(self.cached_opportunities, 1),
            "locality": locality,
            "age_drift_means": age_means,
        }

        return {
            "summary": summary,
            "per_layer": per_layer,
            "per_step": self.step_records,
        }
