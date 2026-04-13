"""
dInfer Integration: Hook AOAE policy into dInfer's high-performance caching.

This module provides a bridge between the AOAE policy's per-position
cache/unmask/remask decisions and dInfer's native KV-cache infrastructure,
enabling fair evaluation of the policy on frontier hardware.

The key integration points are:
  1. PolicyGuidedCacheManager: wraps dInfer's cache with policy-driven
     commit/invalidate decisions instead of confidence thresholds.
  2. PolicyGuidedDecoder: runs the full AOAE inference loop using dInfer's
     sparse attention and FusedMoE kernels for actual speedup measurement.
"""

import torch
import torch.nn.functional as F
import json
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from .cache import DKVCacheManager
from .models.composed_prediction import compose_prediction
from .models.policy import call_policy
from .agreement_signals import compute_reuse_signal
from .kv_dynamics import SpeculativeDynamicsTracker
from .positional_cache import (
    init_positional_state,
    get_policy_positional_features,
    build_access_set,
    update_positional_state,
    compute_next_h_access_metrics,
)


def _max_prob_and_argmax(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return max softmax probability and argmax token without full probs."""
    logits_f = logits.float()
    max_logits, max_tok = logits_f.max(dim=-1)
    max_prob = torch.exp(max_logits - torch.logsumexp(logits_f, dim=-1))
    return max_prob.type_as(logits), max_tok


def _pad_response_hidden_states(
    hidden_states: List[torch.Tensor],
    *,
    prompt_len: int,
    response_len: int,
    total_response_len: int,
) -> List[torch.Tensor]:
    padded: List[torch.Tensor] = []
    for layer in hidden_states:
        resp = layer[:, prompt_len:prompt_len + response_len, :]
        full = layer.new_zeros((layer.shape[0], total_response_len, layer.shape[-1]))
        full[:, :response_len, :] = resp
        padded.append(full)
    return padded


def _pad_response_layer_kv(
    layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    *,
    prompt_len: int,
    response_len: int,
    total_response_len: int,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    if layer_kv is None:
        return None

    padded: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in layer_kv:
        if key.ndim == 4:
            resp_key = key[:, :, prompt_len:prompt_len + response_len, :]
            resp_value = value[:, :, prompt_len:prompt_len + response_len, :]
            full_key = key.new_zeros((key.shape[0], key.shape[1], total_response_len, key.shape[-1]))
            full_value = value.new_zeros((value.shape[0], value.shape[1], total_response_len, value.shape[-1]))
            full_key[:, :, :response_len, :] = resp_key
            full_value[:, :, :response_len, :] = resp_value
        elif key.ndim == 3:
            resp_key = key[:, prompt_len:prompt_len + response_len, :]
            resp_value = value[:, prompt_len:prompt_len + response_len, :]
            full_key = key.new_zeros((key.shape[0], total_response_len, key.shape[-1]))
            full_value = value.new_zeros((value.shape[0], total_response_len, value.shape[-1]))
            full_key[:, :response_len, :] = resp_key
            full_value[:, :response_len, :] = resp_value
        else:
            full_key = key
            full_value = value
        padded.append((full_key, full_value))
    return padded


def _primary_forward_with_blockwise_diagnostics(
    dual_model,
    input_ids: torch.LongTensor,
):
    """Call primary diagnostics compatibly across older/newer wrappers."""
    diagnostics_fn = getattr(dual_model, "primary_forward_with_diagnostics", None)
    if diagnostics_fn is None:
        raise AttributeError("dual_model is missing primary_forward_with_diagnostics")
    try:
        return diagnostics_fn(
            input_ids,
            output_attentions=False,
            output_kv=True,
        )
    except TypeError as exc:
        message = str(exc)
        if "output_attentions" not in message and "output_kv" not in message:
            raise
        return diagnostics_fn(input_ids)


def _observe_blockwise_kv_dynamics(
    dynamics_tracker: SpeculativeDynamicsTracker,
    *,
    layer_hiddens: List[torch.Tensor],
    max_prob: torch.Tensor,
    mask_ind: torch.Tensor,
    agreement: torch.Tensor,
    u_t: torch.Tensor,
    r_t: torch.Tensor,
    kappa_t: torch.Tensor,
    q_t: torch.Tensor,
    layer_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    valid_mask: torch.Tensor,
) -> None:
    """Compat wrapper for newer/older tracker.observe_step signatures."""
    try:
        dynamics_tracker.observe_step(
            layer_hiddens=layer_hiddens,
            max_prob=max_prob,
            mask_ind=mask_ind,
            agreement=agreement,
            u_t=u_t,
            r_t=r_t,
            kappa_t=kappa_t,
            q_t=q_t,
            layer_kv=layer_kv,
            layer_attentions=None,
            valid_mask=valid_mask,
        )
    except TypeError as exc:
        message = str(exc)
        if "valid_mask" not in message:
            raise
        dynamics_tracker.observe_step(
            layer_hiddens=layer_hiddens,
            max_prob=max_prob,
            mask_ind=mask_ind,
            agreement=agreement,
            u_t=u_t,
            r_t=r_t,
            kappa_t=kappa_t,
            q_t=q_t,
            layer_kv=layer_kv,
            layer_attentions=None,
        )


@dataclass
class CacheStats:
    """Statistics for a single inference run."""
    total_commits: int = 0
    total_invalidations: int = 0
    total_remasks: int = 0
    total_unmasks: int = 0
    # Historical name kept for backwards compatibility with saved result schemas.
    # Semantically this is a bounded cache-keep ratio, not a literal runtime
    # lookup-hit metric.
    cache_hit_rate: float = 0.0
    steps_used: int = 0


class PolicyGuidedCacheManager:
    """Wraps dKV-Cache with AOAE policy-driven commit/invalidate.

    Instead of using confidence thresholds to decide which positions
    to cache, this manager uses the policy's kappa_t predictions.
    The policy's remask predictions drive cache invalidation.

    This provides a fair comparison: the same underlying cache mechanism
    (dKV-Cache sparse attention) is used, but steering is learned vs heuristic.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.cache_mgr = DKVCacheManager(batch_size, seq_len, device)
        self.stats = CacheStats()

    def step(
        self,
        r_t: torch.Tensor,
        kappa_t: torch.Tensor,
        u_t: torch.Tensor,
    ):
        """Process one inference step's worth of policy actions.

        Args:
            r_t: [B, L] remask decisions (1 = remask)
            kappa_t: [B, L] cache commit decisions (1 = commit)
            u_t: [B, L] unmask decisions (1 = unmask)
        """
        # Phase 1: Invalidate cached positions that are being remasked
        if r_t.any():
            invalidated = self.cache_mgr.count_thrash(r_t)
            self.cache_mgr.invalidate(r_t)
            self.stats.total_invalidations += int(invalidated.sum().item())
            self.stats.total_remasks += int(r_t.sum().item())

        # Phase 2: Count unmasks
        self.stats.total_unmasks += int(u_t.sum().item())

        # Phase 3: Commit new positions to cache
        if kappa_t.any():
            self.cache_mgr.commit(kappa_t)
            self.stats.total_commits += int(kappa_t.sum().item())

        self.stats.steps_used += 1

    @property
    def cached_mask(self) -> torch.Tensor:
        """Return [B, L] bool mask of currently cached positions."""
        return self.cache_mgr.get_cached_mask()

    def get_stats(self) -> CacheStats:
        """Return accumulated statistics."""
        total_ops = self.stats.total_commits + self.stats.total_invalidations
        if total_ops > 0:
            # Keep this bounded in [0, 1] so the reporting layer can interpret
            # it as the fraction of cache operations that ended in a retained
            # commit rather than an invalidation.
            self.stats.cache_hit_rate = self.stats.total_commits / max(total_ops, 1)
        return self.stats

    def count_thrash(self, r_t: torch.Tensor) -> torch.Tensor:
        """Count positions that are cached AND being remasked (thrashing)."""
        return self.cache_mgr.count_thrash(r_t)


class SpeculativeCacheManager:
    """Cache manager for dual-model speculative diffusion.

    Extends PolicyGuidedCacheManager with agreement-gated caching:
    positions are only committed to the persistent cache when the
    hard-routed auxiliary and soft-routed primary agree on the token.

    Tracks additional metrics: agreement rate, draft acceptance rate,
    and effective speedup from speculative caching.
    """

    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.cache_mgr = DKVCacheManager(batch_size, seq_len, device)
        self.stats = CacheStats()
        self._agreement_sum = 0.0
        self._agreement_count = 0
        self._draft_accepts = 0
        self._draft_rejects = 0

    def step(
        self,
        r_t: torch.Tensor,
        kappa_t: torch.Tensor,
        u_t: torch.Tensor,
        agreement: torch.Tensor,
    ):
        """Process one step with agreement-gated caching.

        Args:
            r_t: [B, L] remask decisions
            kappa_t: [B, L] cache commit decisions
            u_t: [B, L] unmask decisions
            agreement: [B, L] bool/float auxiliary-primary agreement
        """
        # Phase 1: Invalidate
        if r_t.any():
            invalidated = self.cache_mgr.count_thrash(r_t)
            self.cache_mgr.invalidate(r_t)
            self.stats.total_invalidations += int(invalidated.sum().item())
            self.stats.total_remasks += int(r_t.sum().item())

        # Phase 2: Unmasks
        self.stats.total_unmasks += int(u_t.sum().item())

        # Phase 3: Agreement-gated cache commit
        agreement_f = agreement.float()
        accepted = kappa_t * agreement_f        # cache only where models agree
        rejected = kappa_t * (1.0 - agreement_f)  # wanted to cache but disagreed

        if accepted.any():
            self.cache_mgr.commit(accepted)
            self.stats.total_commits += int(accepted.sum().item())

        self._draft_accepts += int(accepted.sum().item())
        self._draft_rejects += int(rejected.sum().item())
        self._agreement_sum += agreement_f.sum().item()
        self._agreement_count += agreement_f.numel()
        self.stats.steps_used += 1

    @property
    def cached_mask(self) -> torch.Tensor:
        return self.cache_mgr.get_cached_mask()

    def get_stats(self) -> dict:
        """Return stats dict with speculative caching metrics."""
        base = self.stats
        total_ops = base.total_commits + base.total_invalidations
        # This is a bounded cache-keep ratio, not a literal runtime lookup-hit
        # metric.
        cache_hit_rate = base.total_commits / max(total_ops, 1)
        total_drafts = self._draft_accepts + self._draft_rejects
        return {
            "total_commits": base.total_commits,
            "total_invalidations": base.total_invalidations,
            "total_remasks": base.total_remasks,
            "total_unmasks": base.total_unmasks,
            "cache_hit_rate": cache_hit_rate,
            "steps_used": base.steps_used,
            "draft_accept_rate": self._draft_accepts / max(total_drafts, 1),
            "draft_accepts": self._draft_accepts,
            "draft_rejects": self._draft_rejects,
            "mean_agreement": self._agreement_sum / max(self._agreement_count, 1),
        }

    def count_thrash(self, r_t: torch.Tensor) -> torch.Tensor:
        return self.cache_mgr.count_thrash(r_t)


def _active_span(mask: torch.Tensor) -> Optional[Tuple[int, int]]:
    """Return the minimal contiguous span covering any active positions."""
    if mask.numel() == 0 or not mask.any():
        return None
    cols = mask.any(dim=0).nonzero(as_tuple=True)[0]
    return int(cols[0].item()), int(cols[-1].item()) + 1


def run_blockwise_speculative_inference(
    dual_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    policy_temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Run speculative decoding inside the official LLaDA2.1 block schedule.

    This path preserves the paper/model-card decode order:
      - generate one block at a time, left-to-right
      - apply threshold-based M2T unmasking within the active block
      - allow T2T editing within that same block

    The hard-routed auxiliary stays in the loop for agreement/reuse tracking,
    but the actual token updates follow the primary soft-routed logits so that
    fidelity stays close to the official LLaDA2.1 decode semantics.
    """
    del policy, soft_mask_module, prism_adapter, policy_temperature  # Unused in this scheduler.

    ic = cfg["inference"]
    off_cfg = ic.get("llada21_official", {})
    if not bool(off_cfg.get("use_block_diffusion", True)):
        raise ValueError(
            "run_blockwise_speculative_inference requires "
            "inference.llada21_official.use_block_diffusion=true."
        )
    if bool(off_cfg.get("enable_mbe", False)):
        raise NotImplementedError(
            "Speculative blockwise PoC1 does not yet support enable_mbe=true."
        )

    block_len = int(ic.get("block_length", 32))
    max_post_steps = int(off_cfg.get("max_post_steps", 16))
    threshold = float(off_cfg.get("threshold", 0.7))
    editing_threshold = float(off_cfg.get("editing_threshold", 0.5))
    L_gen = int(ic["gen_length"])
    mask_id = int(cfg["base_model"]["mask_token_id"])
    use_fallback = bool(ic.get("fallback_unmask", True))
    disable_remask = bool(ic.get("disable_remask", False))
    track_kv_dynamics = bool(cfg.get("analysis", {}).get("track_kv_dynamics", False))

    B, P = prompt_ids.shape
    device = prompt_ids.device
    n_blocks = (L_gen + block_len - 1) // block_len

    y = torch.cat(
        [
            prompt_ids,
            torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
        ],
        dim=1,
    )

    cache_mgr = SpeculativeCacheManager(B, L_gen, device)
    dynamics_tracker = SpeculativeDynamicsTracker(cfg) if track_kv_dynamics else None
    can_skip_primary = (
        hasattr(dual_model, "primary_forward_with_cache")
        and getattr(getattr(dual_model, "_model", None), "_dinfer_runtime", None) == "vllm"
    )
    reuse_state = None
    reuse_diag_sum: Dict[str, float] = {}
    reuse_diag_steps = 0
    _primary_steps = 0
    _primary_full_steps = 0
    _primary_partial_steps = 0
    _primary_verified_positions = 0
    _primary_full_equiv_positions = 0
    _raw_agreement_sum = 0.0
    _raw_agreement_count = 0
    _safe_reuse_sum = 0.0
    _safe_reuse_count = 0
    _draft_accepts = 0
    _draft_rejects = 0
    tracked_prefix_len = 0
    tracked_max_prob = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
    tracked_agreement = torch.zeros((B, L_gen), dtype=torch.float32, device=device)

    for blk_idx in range(n_blocks):
        blk_start = P + blk_idx * block_len
        blk_end = min(P + (blk_idx + 1) * block_len, P + L_gen)
        blk_slice = slice(blk_start, blk_end)
        rel_slice = slice(blk_start - P, blk_end - P)
        blk_width = blk_end - blk_start
        reuse_state = None
        verifier_active = torch.ones((B, blk_width), dtype=torch.bool, device=device)
        pri_logits = None
        primary_cache = None

        for _ in range(max_post_steps):
            blk_tokens = y[:, blk_slice]
            mask_ind = blk_tokens == mask_id
            if not mask_ind.any():
                break

            prefix_ids = y[:, :blk_end]
            prefix_blk_slice = slice(blk_start, blk_end)
            active_mask = verifier_active | mask_ind
            if active_mask.any() and can_skip_primary:
                aux_logits = dual_model.auxiliary_forward_resp(prefix_ids, prefix_blk_slice)
                span = _active_span(active_mask)
                verified_positions = 0
                if primary_cache is None or pri_logits is None or span == (0, blk_width):
                    pri_prefix_logits, primary_cache = dual_model.primary_forward_with_cache(prefix_ids)
                    pri_logits = pri_prefix_logits[:, prefix_blk_slice, :]
                    _primary_full_steps += 1
                    verified_positions = B * blk_width
                elif span is not None:
                    span_start, span_end = span
                    pri_span_logits, primary_cache = dual_model.primary_forward_replace_with_cache(
                        prefix_ids,
                        slice(blk_start + span_start, blk_start + span_end),
                        primary_cache,
                    )
                    pri_logits[:, span_start:span_end, :] = pri_span_logits
                    _primary_partial_steps += 1
                    verified_positions = B * (span_end - span_start)
                _primary_steps += 1
                _primary_verified_positions += verified_positions
                _primary_full_equiv_positions += B * blk_width
            else:
                dual_out = dual_model.dual_forward_resp(prefix_ids, prefix_blk_slice)
                pri_logits = dual_out.primary_logits
                aux_logits = dual_out.auxiliary_logits
                primary_cache = None
                _primary_steps += 1
                _primary_full_steps += 1
                _primary_verified_positions += B * blk_width
                _primary_full_equiv_positions += B * blk_width

            aux_tokens = aux_logits.argmax(dim=-1)
            pri_tokens = pri_logits.argmax(dim=-1)
            raw_agreement = (aux_tokens == pri_tokens)
            safe_reuse, reuse_state, reuse_diag = compute_reuse_signal(
                pri_logits, aux_logits, cfg, state=reuse_state,
            )
            safe_reuse = safe_reuse.bool()

            # Agreement / reuse metrics are only meaningful on positions that
            # are still drafted from masked state in this blockwise scheduler.
            metric_mask = mask_ind.bool()
            if metric_mask.any():
                _raw_agreement_sum += float(raw_agreement[metric_mask].float().sum().item())
                _raw_agreement_count += int(metric_mask.sum().item())
                _safe_reuse_sum += float(safe_reuse[metric_mask].float().sum().item())
                _safe_reuse_count += int(metric_mask.sum().item())
            for key, value in reuse_diag.items():
                reuse_diag_sum[key] = reuse_diag_sum.get(key, 0.0) + float(value)
            reuse_diag_steps += 1

            pri_max_prob, pri_max_tok = _max_prob_and_argmax(pri_logits)

            prev_tokens = blk_tokens.clone()
            next_tokens = prev_tokens.clone()

            unmask_positions = mask_ind & (pri_max_prob > threshold)
            drafted_positions = unmask_positions.clone()
            if unmask_positions.any():
                next_tokens[unmask_positions] = pri_max_tok[unmask_positions]

            if use_fallback:
                still_masked = next_tokens == mask_id
                no_unmasks = mask_ind.any(dim=-1) & ~unmask_positions.any(dim=-1)
                if no_unmasks.any():
                    fallback_conf = pri_max_prob.clone()
                    fallback_conf[~still_masked] = -1.0
                    best_pos = fallback_conf.argmax(dim=-1)
                    best_tok = pri_max_tok
                    for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                        next_tokens[b_idx, best_pos[b_idx]] = best_tok[b_idx, best_pos[b_idx]]
                        drafted_positions[b_idx, best_pos[b_idx]] = True

            if disable_remask:
                edit_positions = torch.zeros_like(mask_ind)
            else:
                previously_unmasked = (~mask_ind) & active_mask
                disagree = (pri_max_tok != prev_tokens) & previously_unmasked
                confident = pri_max_prob > editing_threshold
                edit_positions = disagree & confident
                if edit_positions.any():
                    next_tokens[edit_positions] = pri_max_tok[edit_positions]

            if drafted_positions.any():
                drafted_agreement = raw_agreement[drafted_positions]
                _draft_accepts += int(drafted_agreement.sum().item())
                _draft_rejects += int(drafted_positions.sum().item() - drafted_agreement.sum().item())

            u_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
            r_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
            kappa_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
            safe_reuse_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)

            cache_commit_positions = (drafted_positions | edit_positions) & safe_reuse
            u_full[:, rel_slice] = drafted_positions.float()
            r_full[:, rel_slice] = edit_positions.float()
            kappa_full[:, rel_slice] = cache_commit_positions.float()
            safe_reuse_full[:, rel_slice] = safe_reuse.float()

            if dynamics_tracker is not None:
                diag_logits, diag_hidden_states, _, diag_layer_kv = _primary_forward_with_blockwise_diagnostics(
                    dual_model,
                    prefix_ids,
                )
                del diag_logits
                prefix_resp_len = blk_end - P
                full_layer_hiddens = _pad_response_hidden_states(
                    diag_hidden_states,
                    prompt_len=P,
                    response_len=prefix_resp_len,
                    total_response_len=L_gen,
                )
                full_layer_kv = _pad_response_layer_kv(
                    diag_layer_kv,
                    prompt_len=P,
                    response_len=prefix_resp_len,
                    total_response_len=L_gen,
                )
                full_mask_ind = y[:, P:P + L_gen] == mask_id
                tracked_max_prob[:, rel_slice] = pri_max_prob.float()
                tracked_agreement[:, rel_slice] = raw_agreement.float()
                valid_mask = torch.zeros((B, L_gen), dtype=torch.bool, device=device)
                valid_mask[:, :min(tracked_prefix_len, prefix_resp_len)] = True
                _observe_blockwise_kv_dynamics(
                    dynamics_tracker,
                    layer_hiddens=full_layer_hiddens,
                    max_prob=tracked_max_prob,
                    mask_ind=full_mask_ind,
                    agreement=tracked_agreement,
                    u_t=u_full,
                    r_t=r_full,
                    kappa_t=kappa_full,
                    q_t=torch.zeros_like(u_full),
                    layer_kv=full_layer_kv,
                    valid_mask=valid_mask,
                )
                tracked_prefix_len = max(tracked_prefix_len, prefix_resp_len)
            cache_mgr.step(r_full, kappa_full, u_full, safe_reuse_full)

            y[:, blk_slice] = next_tokens
            next_mask_ind = next_tokens == mask_id
            verifier_active = next_mask_ind | edit_positions | ((~next_mask_ind) & ~safe_reuse)

        remaining_mask = y[:, blk_slice] == mask_id
        if remaining_mask.any():
            prefix_ids = y[:, :blk_end]
            prefix_blk_slice = slice(blk_start, blk_end)
            if can_skip_primary:
                aux_logits = dual_model.auxiliary_forward_resp(prefix_ids, prefix_blk_slice)
                pri_prefix_logits, primary_cache = dual_model.primary_forward_with_cache(prefix_ids)
                pri_logits = pri_prefix_logits[:, prefix_blk_slice, :]
                _primary_steps += 1
                _primary_full_steps += 1
                _primary_verified_positions += B * blk_width
                _primary_full_equiv_positions += B * blk_width
            else:
                dual_out = dual_model.dual_forward_resp(prefix_ids, prefix_blk_slice)
                pri_logits = dual_out.primary_logits
                aux_logits = dual_out.auxiliary_logits
                primary_cache = None
                _primary_steps += 1
                _primary_full_steps += 1
                _primary_verified_positions += B * blk_width
                _primary_full_equiv_positions += B * blk_width

            raw_agreement = aux_logits.argmax(dim=-1) == pri_logits.argmax(dim=-1)
            safe_reuse, reuse_state, reuse_diag = compute_reuse_signal(
                pri_logits, aux_logits, cfg, state=reuse_state,
            )
            safe_reuse = safe_reuse.bool()
            if remaining_mask.any():
                _raw_agreement_sum += float(raw_agreement[remaining_mask].float().sum().item())
                _raw_agreement_count += int(remaining_mask.sum().item())
                _safe_reuse_sum += float(safe_reuse[remaining_mask].float().sum().item())
                _safe_reuse_count += int(remaining_mask.sum().item())
            for key, value in reuse_diag.items():
                reuse_diag_sum[key] = reuse_diag_sum.get(key, 0.0) + float(value)
            reuse_diag_steps += 1

            _, pri_max_tok = _max_prob_and_argmax(pri_logits)
            completed = y[:, blk_slice].clone()
            completed[remaining_mask] = pri_max_tok[remaining_mask]
            y[:, blk_slice] = completed

            drafted_agreement = raw_agreement[remaining_mask]
            _draft_accepts += int(drafted_agreement.sum().item())
            _draft_rejects += int(remaining_mask.sum().item() - drafted_agreement.sum().item())

            cache_commit_positions = remaining_mask & safe_reuse
            u_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
            r_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
            kappa_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
            safe_reuse_full = torch.zeros((B, L_gen), dtype=torch.float32, device=device)
            u_full[:, rel_slice] = remaining_mask.float()
            kappa_full[:, rel_slice] = cache_commit_positions.float()
            safe_reuse_full[:, rel_slice] = safe_reuse.float()

            if dynamics_tracker is not None:
                diag_logits, diag_hidden_states, _, diag_layer_kv = _primary_forward_with_blockwise_diagnostics(
                    dual_model,
                    prefix_ids,
                )
                del diag_logits
                prefix_resp_len = blk_end - P
                full_layer_hiddens = _pad_response_hidden_states(
                    diag_hidden_states,
                    prompt_len=P,
                    response_len=prefix_resp_len,
                    total_response_len=L_gen,
                )
                full_layer_kv = _pad_response_layer_kv(
                    diag_layer_kv,
                    prompt_len=P,
                    response_len=prefix_resp_len,
                    total_response_len=L_gen,
                )
                full_mask_ind = y[:, P:P + L_gen] == mask_id
                tracked_max_prob[:, rel_slice] = _max_prob_and_argmax(pri_logits)[0].float()
                tracked_agreement[:, rel_slice] = raw_agreement.float()
                valid_mask = torch.zeros((B, L_gen), dtype=torch.bool, device=device)
                valid_mask[:, :min(tracked_prefix_len, prefix_resp_len)] = True
                _observe_blockwise_kv_dynamics(
                    dynamics_tracker,
                    layer_hiddens=full_layer_hiddens,
                    max_prob=tracked_max_prob,
                    mask_ind=full_mask_ind,
                    agreement=tracked_agreement,
                    u_t=u_full,
                    r_t=r_full,
                    kappa_t=kappa_full,
                    q_t=torch.zeros_like(u_full),
                    layer_kv=full_layer_kv,
                    valid_mask=valid_mask,
                )
                tracked_prefix_len = max(tracked_prefix_len, prefix_resp_len)
            cache_mgr.step(r_full, kappa_full, u_full, safe_reuse_full)

    stats = cache_mgr.get_stats()
    stats["mean_agreement"] = _raw_agreement_sum / max(_raw_agreement_count, 1)
    stats["agreement_observations"] = _raw_agreement_count
    stats["draft_accepts"] = _draft_accepts
    stats["draft_rejects"] = _draft_rejects
    stats["draft_accept_rate"] = _draft_accepts / max(_draft_accepts + _draft_rejects, 1)
    stats["reuse_mean_safe_reuse"] = _safe_reuse_sum / max(_safe_reuse_count, 1)
    stats["safe_reuse_observations"] = _safe_reuse_count
    stats["primary_steps"] = _primary_steps
    stats["aux_only_steps"] = 0
    stats["primary_full_steps"] = _primary_full_steps
    stats["primary_partial_steps"] = _primary_partial_steps
    stats["primary_verified_positions"] = _primary_verified_positions
    stats["primary_full_equiv_positions"] = _primary_full_equiv_positions
    stats["primary_skip_ratio"] = 1.0 - (
        _primary_verified_positions / max(_primary_full_equiv_positions, 1)
    )
    stats["reuse_signal_method"] = cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")
    for key, value in reuse_diag_sum.items():
        stats[f"reuse_{key}"] = value / max(reuse_diag_steps, 1)
    stats.update(compute_next_h_access_metrics([], [], None, 1))
    stats["mean_boundary_depth"] = 0.0
    stats["boundary_distribution"] = "{}"
    if dynamics_tracker is not None:
        stats["kv_dynamics"] = dynamics_tracker.summarize()
    return y, stats


def run_speculative_inference(
    dual_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    policy_temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Run speculative diffusion with dInfer integration for benchmarking.

    Uses SpeculativeCacheManager for accurate dual-model cache statistics.

    Args:
        dual_model: DualModelWrapper (soft primary + hard auxiliary).
        policy: AOAE steering policy.
        soft_mask_module: soft-masked state builder.
        prism_adapter: PRISM quality head (or None).
        prompt_ids: [B, P] prompt token ids.
        cfg: config dict.
        policy_temperature: tau_pi.

    Returns:
        output_ids: [B, P + L_gen] full sequence.
        stats: dict with detailed speculative cache metrics.
    """
    from .models.composed_prediction import compose_prediction_dual

    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = ic["gen_length"]
    mask_id = cfg["base_model"]["mask_token_id"]
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)
    use_fallback = ic["fallback_unmask"]
    disable_remask = ic.get("disable_remask", False)
    track_kv_dynamics = bool(cfg.get("analysis", {}).get("track_kv_dynamics", False))
    use_positional_cache = bool(ic.get("positional_cache", {}).get("enabled", False))

    from .models.policy import DefaultPolicy
    _skip_soft_mask = isinstance(policy, DefaultPolicy) and not use_positional_cache

    B = prompt_ids.shape[0]
    P = prompt_ids.shape[1]
    device = prompt_ids.device

    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    resp_slice = slice(P, P + L_gen)
    cache_mgr = SpeculativeCacheManager(B, L_gen, device)
    dynamics_tracker = SpeculativeDynamicsTracker(cfg) if track_kv_dynamics else None
    reuse_state = None
    reuse_diag_sum: Dict[str, float] = {}
    reuse_diag_steps = 0
    access_diag_sum: Dict[str, float] = {}
    access_diag_steps = 0
    access_exec_steps: List[torch.Tensor] = [] if use_positional_cache else None
    changed_steps: List[torch.Tensor] = [] if use_positional_cache else None
    mandatory_steps: List[torch.Tensor] = [] if use_positional_cache else None
    boundary_actions: List[torch.Tensor] = []
    pos_state = init_positional_state(B, L_gen, device) if use_positional_cache else None

    primary_every_n = max(1, int(ic.get("primary_every_n", 1)))
    primary_agree_threshold = float(ic.get("primary_agree_threshold", 0.0))
    force_primary_endpoints = bool(ic.get("force_primary_first_last", True))
    _ema_agreement = 1.0  # optimistic init: first step is always primary, will update
    _primary_steps = 0
    _aux_only_steps = 0
    _raw_agreement_sum = 0.0
    _raw_agreement_count = 0
    _safe_reuse_sum = 0.0
    _safe_reuse_count = 0
    _draft_accepts = 0
    _draft_rejects = 0

    for t in range(T, 0, -1):
        step_frac = t / T
        resp_tokens = y[:, resp_slice]
        mask_ind = (resp_tokens == mask_id)

        if not mask_ind.any():
            break

        need_hidden = (prism_adapter is not None) or track_kv_dynamics
        need_all_hidden = track_kv_dynamics

        step_idx = T - t
        run_primary = (
            primary_every_n <= 1
            or (step_idx > 0 and step_idx % primary_every_n == 0)
            or (force_primary_endpoints and (t == T or t == 1))
            or _ema_agreement < primary_agree_threshold
        )

        if run_primary:
            dual_out = dual_model.dual_forward_resp(
                y, resp_slice, need_hidden=need_hidden,
                need_all_hidden=need_all_hidden,
            )
            resp_logits = dual_out.primary_logits
            aux_logits = dual_out.auxiliary_logits
            raw_agreement = dual_out.agreement.bool()
            safe_reuse, reuse_state, reuse_diag = compute_reuse_signal(
                resp_logits, aux_logits, cfg, state=reuse_state,
            )
            safe_reuse = safe_reuse.bool()
            active_mask = mask_ind.bool()
            if active_mask.any():
                _raw_agreement_sum += float(raw_agreement[active_mask].float().sum().item())
                _raw_agreement_count += int(active_mask.sum().item())
                _safe_reuse_sum += float(safe_reuse[active_mask].float().sum().item())
                _safe_reuse_count += int(active_mask.sum().item())
            _ema_agreement = 0.8 * _ema_agreement + 0.2 * dual_out.agreement_rate
            _primary_steps += 1
            for k, v in reuse_diag.items():
                reuse_diag_sum[k] = reuse_diag_sum.get(k, 0.0) + float(v)
            reuse_diag_steps += 1
        else:
            aux_logits = dual_model.auxiliary_forward(y)[:, resp_slice, :]
            resp_logits = aux_logits
            raw_agreement = torch.ones(B, L_gen, dtype=torch.bool, device=device)
            safe_reuse = raw_agreement
            _aux_only_steps += 1

        q_scores = None
        if run_primary and prism_adapter is not None:
            pri_hidden = getattr(dual_out, "primary_hidden", None)
            if pri_hidden is not None:
                with torch.no_grad():
                    q_scores = prism_adapter(pri_hidden.float())

        if _skip_soft_mask:
            confidence, _ = _max_prob_and_argmax(resp_logits)
            H_t_dummy = resp_logits[:, :, :1].expand(-1, -1, 1)
            policy_out = call_policy(
                policy,
                H_t_dummy, mask_ind, step_frac,
                temperature=policy_temperature,
                confidence=confidence,
                quality_scores=q_scores,
                agreement=safe_reuse.float(),
            )
        else:
            H_t, confidence, entropy, _ = soft_mask_module(resp_logits, mask_ind, step_frac)
            age_feat = None
            last_action_feat = None
            if use_positional_cache:
                age_feat, last_action_feat = get_policy_positional_features(pos_state, cfg)
            policy_out = call_policy(
                policy,
                H_t, mask_ind, step_frac,
                temperature=policy_temperature,
                confidence=confidence,
                quality_scores=q_scores,
                agreement=safe_reuse.float(),
                age_feature=age_feat,
                last_action_feature=last_action_feat,
            )
        pol_inner = policy.module if hasattr(policy, "module") else policy
        actions = pol_inner.sample_actions(policy_out, mask_ind)

        u_t = actions["u_t"]
        r_t = actions["r_t"]
        kappa_t = actions["kappa_t"]
        if disable_remask:
            r_t = torch.zeros_like(r_t)
            actions = {**actions, "r_t": r_t}
        q_exec, q_mandatory, access_diag = build_access_set(
            actions,
            policy_out,
            cfg,
            confidence=confidence,
            boundary_action=actions.get("ell_t"),
            boundary_num_bins=(
                int(policy_out["boundary_probs"].shape[-1])
                if "boundary_probs" in policy_out
                else None
            ),
        )
        if "ell_t" in actions:
            boundary_actions.append(actions["ell_t"].detach())
        for k, v in access_diag.items():
            access_diag_sum[k] = access_diag_sum.get(k, 0.0) + float(v)
        access_diag_steps += 1

        # Phase 1: Remask (modify y via view for zero-copy)
        remask_positions = r_t.bool() & ~mask_ind
        if remask_positions.any():
            resp_tokens[remask_positions] = mask_id

        # Phase 2: Unmask with composed prediction
        unmask_positions = u_t.bool() & mask_ind
        drafted_positions = unmask_positions.clone()
        if unmask_positions.any():
            use_composition = gamma > 0 and run_primary and resp_logits is not aux_logits
            if use_composition:
                composed_logits = compose_prediction_dual(
                    resp_logits, aux_logits, safe_reuse, gamma=gamma,
                )
            else:
                composed_logits = resp_logits

            if base_temp > 0:
                probs = F.softmax(composed_logits / base_temp, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, L_gen)
            else:
                sampled = composed_logits.argmax(dim=-1)
            resp_tokens[unmask_positions] = sampled[unmask_positions]

        # Fallback: if policy produced no unmask actions, force-unmask most confident
        if use_fallback:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)
            if no_unmasks.any():
                if confidence is None:
                    confidence, _ = _max_prob_and_argmax(resp_logits)
                fallback_conf = confidence.clone()
                fallback_conf[~still_masked] = -1.0
                best_pos = fallback_conf.argmax(dim=-1)
                best_tok = resp_logits.argmax(dim=-1)
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    resp_tokens[b_idx, best_pos[b_idx]] = best_tok[b_idx, best_pos[b_idx]]
                    drafted_positions[b_idx, best_pos[b_idx]] = True

        if run_primary and drafted_positions.any():
            drafted_agreement = raw_agreement[drafted_positions]
            _draft_accepts += int(drafted_agreement.sum().item())
            _draft_rejects += int(drafted_positions.sum().item() - drafted_agreement.sum().item())

        if dynamics_tracker is not None and run_primary:
            layer_hiddens = dual_out.primary_hidden_states
            if not layer_hiddens:
                if dual_out.primary_hidden is not None:
                    layer_hiddens = [dual_out.primary_hidden]
                else:
                    layer_hiddens = []
            if layer_hiddens:
                max_prob, _ = _max_prob_and_argmax(resp_logits)
                dynamics_tracker.observe_step(
                    layer_hiddens=layer_hiddens,
                    max_prob=max_prob,
                    mask_ind=mask_ind,
                    agreement=raw_agreement.float(),
                    u_t=u_t,
                    r_t=r_t,
                    kappa_t=kappa_t,
                    q_t=(
                        q_exec
                        if cfg.get("inference", {}).get("positional_cache", {}).get("enabled", False)
                        else torch.zeros_like(q_exec)
                    ),
                    layer_kv=dual_out.primary_layer_kv,
                    layer_attentions=dual_out.primary_attentions,
                )

        # Phase 3: Agreement-gated cache commit
        cache_mgr.step(r_t, kappa_t * q_exec, u_t, safe_reuse.float())

        changed = (u_t.bool() | r_t.bool()).float()
        if use_positional_cache:
            access_exec_steps.append(q_exec.detach())
            changed_steps.append(changed.detach())
            mandatory_steps.append(q_mandatory.detach())
            update_positional_state(pos_state, q_exec=q_exec, changed=changed, cfg=cfg)

        y[:, resp_slice] = resp_tokens

    stats = cache_mgr.get_stats()
    stats["mean_agreement"] = _raw_agreement_sum / max(_raw_agreement_count, 1)
    stats["agreement_observations"] = _raw_agreement_count
    stats["draft_accepts"] = _draft_accepts
    stats["draft_rejects"] = _draft_rejects
    stats["draft_accept_rate"] = _draft_accepts / max(_draft_accepts + _draft_rejects, 1)
    stats["reuse_mean_safe_reuse"] = _safe_reuse_sum / max(_safe_reuse_count, 1)
    stats["safe_reuse_observations"] = _safe_reuse_count
    stats["primary_steps"] = _primary_steps
    stats["aux_only_steps"] = _aux_only_steps
    stats["primary_skip_ratio"] = _aux_only_steps / max(_primary_steps + _aux_only_steps, 1)
    reuse_method = cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")
    stats["reuse_signal_method"] = reuse_method
    for k, v in reuse_diag_sum.items():
        stats[f"reuse_{k}"] = v / max(reuse_diag_steps, 1)
    for k, v in access_diag_sum.items():
        stats[f"access_{k}"] = v / max(access_diag_steps, 1)
    if use_positional_cache and access_exec_steps:
        horizon = int(ic.get("positional_cache", {}).get("horizon", 4))
        stats.update(
            compute_next_h_access_metrics(
                access_exec_steps=access_exec_steps,
                changed_steps=changed_steps,
                mandatory_steps=mandatory_steps,
                horizon=horizon,
            )
        )
    else:
        stats.update(compute_next_h_access_metrics([], [], None, 1))
    if boundary_actions:
        all_boundary = torch.cat([x.reshape(-1) for x in boundary_actions], dim=0)
        max_bin = int(all_boundary.max().item()) if all_boundary.numel() > 0 else 0
        denom = max(max_bin, 1)
        stats["mean_boundary_depth"] = float((all_boundary.float() / denom).mean().item())
        counts = torch.bincount(all_boundary, minlength=max_bin + 1).tolist()
        stats["boundary_distribution"] = json.dumps({str(i): int(v) for i, v in enumerate(counts)})
    else:
        stats["mean_boundary_depth"] = 0.0
        stats["boundary_distribution"] = "{}"
    if dynamics_tracker is not None:
        stats["kv_dynamics"] = dynamics_tracker.summarize()
    return y, stats


def run_policy_guided_inference(
    base_model,
    policy,
    soft_mask_module,
    prism_adapter,
    prompt_ids: torch.LongTensor,
    cfg: dict,
    policy_temperature: float = 1.0,
) -> Tuple[torch.Tensor, CacheStats]:
    """Run AOAE inference with dInfer integration for fair benchmarking.

    This is a simplified version of aoae_inference that:
    1. Uses PolicyGuidedCacheManager for accurate cache statistics
    2. Integrates composed prediction for cache-aligned token selection
    3. Reports detailed cache hit/miss statistics for evaluation

    Args:
        base_model: frozen LLaDA wrapper (dinfer or soft_moe backend).
        policy: AOAE policy network.
        soft_mask_module: soft-masked state builder.
        prism_adapter: PRISM quality head (or None).
        prompt_ids: [B, P] prompt token ids.
        cfg: config dict.
        policy_temperature: tau_pi for Bernoulli tempering.

    Returns:
        output_ids: [B, P + L_gen] full sequence.
        stats: CacheStats with detailed cache performance metrics.
    """
    ic = cfg["inference"]
    T = ic["steps"]
    L_gen = ic["gen_length"]
    mask_id = cfg["base_model"]["mask_token_id"]
    base_temp = ic["temperature"]
    gamma = ic.get("compose_gamma", 0.0)
    use_fallback = ic["fallback_unmask"]
    disable_remask = ic.get("disable_remask", False)
    use_positional_cache = bool(ic.get("positional_cache", {}).get("enabled", False))

    B = prompt_ids.shape[0]
    P = prompt_ids.shape[1]
    device = prompt_ids.device

    y = torch.cat([
        prompt_ids,
        torch.full((B, L_gen), mask_id, dtype=torch.long, device=device),
    ], dim=1)

    resp_slice = slice(P, P + L_gen)
    cache_mgr = PolicyGuidedCacheManager(B, L_gen, device)
    pos_state = init_positional_state(B, L_gen, device)

    for t in range(T, 0, -1):
        step_frac = t / T
        resp_tokens = y[:, resp_slice]
        mask_ind = (resp_tokens == mask_id)

        if not mask_ind.any():
            break

        # Base model forward
        if prism_adapter is not None:
            logits, hidden_states = base_model.forward_with_hidden(y)
            resp_hidden = hidden_states[:, resp_slice, :]
        else:
            logits = base_model.forward(y)
            resp_hidden = None
        resp_logits = logits[:, resp_slice, :]

        # PRISM quality scores
        q_scores = None
        if prism_adapter is not None and resp_hidden is not None:
            with torch.no_grad():
                q_scores = prism_adapter(resp_hidden.float())

        # Soft-masked state
        H_t, confidence, entropy, _ = soft_mask_module(
            resp_logits, mask_ind, step_frac
        )
        age_feat = None
        last_action_feat = None
        if use_positional_cache:
            age_feat, last_action_feat = get_policy_positional_features(pos_state, cfg)

        # Policy forward
        policy_out = call_policy(
            policy,
            H_t, mask_ind, step_frac,
            temperature=policy_temperature,
            confidence=confidence,
            quality_scores=q_scores,
            age_feature=age_feat,
            last_action_feature=last_action_feat,
        )
        pol_inner = policy.module if hasattr(policy, "module") else policy
        actions = pol_inner.sample_actions(policy_out, mask_ind)

        u_t = actions["u_t"]
        r_t = actions["r_t"]
        kappa_t = actions["kappa_t"]
        if disable_remask:
            r_t = torch.zeros_like(r_t)
            actions = {**actions, "r_t": r_t}
        q_exec, _q_mandatory, _ = build_access_set(
            actions,
            policy_out,
            cfg,
            confidence=confidence,
            boundary_action=actions.get("ell_t"),
            boundary_num_bins=(
                int(policy_out["boundary_probs"].shape[-1])
                if "boundary_probs" in policy_out
                else None
            ),
        )

        resp_tokens = resp_tokens.clone()

        # Phase 1: Remask
        remask_positions = r_t.bool() & ~mask_ind
        if remask_positions.any():
            resp_tokens[remask_positions] = mask_id

        # Phase 2: Unmask with composed prediction
        unmask_positions = u_t.bool() & mask_ind
        if unmask_positions.any():
            if gamma > 0 and "cache_probs" in policy_out:
                composed_logits = compose_prediction(
                    resp_logits, policy_out["cache_probs"], gamma=gamma,
                )
            else:
                composed_logits = resp_logits

            if base_temp > 0:
                probs = F.softmax(composed_logits / base_temp, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, L_gen)
            else:
                sampled = composed_logits.argmax(dim=-1)
            resp_tokens[unmask_positions] = sampled[unmask_positions]

        # Fallback
        if use_fallback:
            still_masked = (resp_tokens == mask_id)
            no_unmasks = (u_t.sum(dim=-1) == 0) & still_masked.any(dim=-1)
            if no_unmasks.any():
                for b_idx in no_unmasks.nonzero(as_tuple=True)[0]:
                    masked_pos = still_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(masked_pos) > 0:
                        best_pos = masked_pos[confidence[b_idx, masked_pos].argmax()]
                        resp_tokens[b_idx, best_pos] = resp_logits[b_idx, best_pos].argmax()

        # Phase 3: Cache commit + stats tracking
        cache_mgr.step(r_t, kappa_t * q_exec, u_t)

        changed = (u_t.bool() | r_t.bool()).float()
        if use_positional_cache:
            update_positional_state(pos_state, q_exec=q_exec, changed=changed, cfg=cfg)

        y[:, resp_slice] = resp_tokens

    return y, cache_mgr.get_stats()
