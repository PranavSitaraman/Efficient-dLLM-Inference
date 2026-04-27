"""Blockwise speculative diffusion under the official LLaDA2.1 schedule.

The runtime AOAE path lives in ``aoae.speculative_inference``.  This module
implements the ``llada21_block`` baseline: a left-to-right block scheduler that
preserves the LLaDA2.1 model card decode order while still exposing the
hard-routed auxiliary for agreement / cache-reuse diagnostics.
"""

import torch
import torch.nn.functional as F
import json
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from .cache import DKVCacheManager
from .inference import _max_prob_and_argmax
from .agreement_signals import compute_reuse_signal
from .kv_dynamics import SpeculativeDynamicsTracker
from .positional_cache import compute_next_h_access_metrics


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
    diagnostics_fn = getattr(dual_model, "primary_forward_with_diagnostics", None)
    if diagnostics_fn is None:
        raise AttributeError("dual_model is missing primary_forward_with_diagnostics")
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


class SpeculativeCacheManager:
    """Legacy cache manager for older dual-model speculative runners.

    WARNING: this older path still conflates speculative acceptance with a
    persistent cache by committing agreement-gated positions into a
    DKVCacheManager. The newer AOAE path in speculative_inference.py instead
    treats K_spec as transient one-step speculative state and K_stable as the
    only persistent cache.

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
        """Process one legacy step with agreement-gated persistent caching.

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
