"""
Runtime capability checks and fallbacks for AOAE deployments.

This module centralizes:
  - Environment/runtime introspection (torch/CUDA/vLLM availability).
  - vLLM MoE custom-op capability detection.
  - Safe Python fallbacks for missing _moe_C ops used in AOAE paths.
"""

from __future__ import annotations

import os
import socket
import subprocess
import random
import sys
import importlib
import importlib.util
from typing import Any, Dict, List, Optional

import torch
import numpy as np


# Ops used by the unquantized fused-MoE code paths exercised in this repo.
_MOE_REQUIRED_OPS = ("moe_align_block_size", "moe_sum")

# ---------------------------------------------------------------------------
# Triton moe_align_block_size — inlined from dInfer/tools/fuse_moe.py.
# Only depends on torch + triton; no vLLM side-effects at import time.
# Based on: https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0
# ---------------------------------------------------------------------------
_triton_moe_align_fn = None  # cached after first successful build


def _build_triton_moe_align() -> Optional[object]:
    """Build and return the Triton moe_align_block_size_triton function.

    Returns None if triton is not importable.  Result is cached in
    _triton_moe_align_fn so JIT compilation only happens once.
    """
    global _triton_moe_align_fn
    if _triton_moe_align_fn is not None:
        return _triton_moe_align_fn
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    def _ceil_div(a, b):
        return (a + b - 1) // b

    @triton.jit
    def _stage1(topk_ids_ptr, tokens_cnts_ptr, num_experts: tl.constexpr,
                numel: tl.constexpr, tokens_per_thread: tl.constexpr):
        pid = tl.program_id(0)
        start_idx = pid * tokens_per_thread
        off_c = (pid + 1) * num_experts
        for i in range(tokens_per_thread):
            if start_idx + i < numel:
                idx = tl.load(topk_ids_ptr + start_idx + i)
                cnt = tl.load(tokens_cnts_ptr + off_c + idx)
                tl.store(tokens_cnts_ptr + off_c + idx, cnt + 1)

    @triton.jit
    def _stage2(tokens_cnts_ptr, num_experts: tl.constexpr):
        pid = tl.program_id(0)
        last = 0
        for i in range(1, num_experts + 1):
            cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
            last = last + cnt
            tl.store(tokens_cnts_ptr + i * num_experts + pid, last)

    @triton.jit
    def _stage3(total_ptr, tokens_cnts_ptr, cumsum_ptr,
                num_experts: tl.constexpr, block_size: tl.constexpr):
        last = 0
        off_cnt = num_experts * num_experts
        for i in range(1, num_experts + 1):
            cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
            last = last + tl.cdiv(cnt, block_size) * block_size
            tl.store(cumsum_ptr + i, last)
        tl.store(total_ptr, last)

    @triton.jit
    def _stage4(topk_ids_ptr, sorted_ids_ptr, expert_ids_ptr,
                tokens_cnts_ptr, cumsum_ptr,
                num_experts: tl.constexpr, block_size: tl.constexpr,
                numel: tl.constexpr, tokens_per_thread: tl.constexpr):
        pid = tl.program_id(0)
        start_idx = tl.load(cumsum_ptr + pid)
        end_idx = tl.load(cumsum_ptr + pid + 1)
        for i in range(start_idx, end_idx, block_size):
            tl.store(expert_ids_ptr + i // block_size, pid)
        start_idx = pid * tokens_per_thread
        off_t = pid * num_experts
        for i in range(start_idx,
                       tl.minimum(start_idx + tokens_per_thread, numel)):
            eid = tl.load(topk_ids_ptr + i)
            cnt = tl.load(tokens_cnts_ptr + off_t + eid)
            tl.store(sorted_ids_ptr + cnt + tl.load(cumsum_ptr + eid), i)
            tl.store(tokens_cnts_ptr + off_t + eid, cnt + 1)

    def _moe_align_block_size_triton(
        topk_ids: torch.Tensor,
        num_experts: int,
        block_size: int,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
    ) -> None:
        numel = topk_ids.numel()
        # Pre-fill padding sentinels before Triton writes the live positions.
        # sorted_token_ids padding positions must hold `numel` (the sentinel
        # that tells the fused-MoE kernel to skip that slot).
        # expert_ids padding positions beyond num_tokens_post_pad//block_size
        # are never read by the fused-MoE Triton kernel, but vLLM's Python
        # wrapper applies expert_map[expert_ids] to the FULL tensor before
        # the kernel runs — uninitialized values here cause IndexKernel OOB.
        sorted_token_ids.fill_(numel)
        expert_ids.zero_()
        grid = (num_experts,)
        tokens_cnts = torch.zeros(
            (num_experts + 1, num_experts), dtype=torch.int32,
            device=topk_ids.device)
        cumsum = torch.zeros(
            (num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
        tpt = _ceil_div(numel, num_experts)
        _stage1[grid](topk_ids, tokens_cnts, num_experts, numel, tpt)
        _stage2[grid](tokens_cnts, num_experts)
        _stage3[(1,)](num_tokens_post_pad, tokens_cnts, cumsum,
                      num_experts, block_size)
        _stage4[grid](topk_ids, sorted_token_ids, expert_ids,
                      tokens_cnts, cumsum, num_experts, block_size,
                      numel, tpt)

    _triton_moe_align_fn = _moe_align_block_size_triton
    return _triton_moe_align_fn


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def collect_runtime_info() -> Dict[str, Any]:
    """Collect a stable runtime snapshot for metadata/preflight output."""
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices: List[str] = []
    for idx in range(gpu_count):
        try:
            devices.append(torch.cuda.get_device_name(idx))
        except Exception:
            devices.append(f"cuda:{idx}")

    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python_version": "{}.{}.{}".format(*tuple(__import__("sys").version_info[:3])),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": gpu_count,
        "cuda_devices": devices,
        "vllm_version": _pkg_version("vllm"),
        "transformers_version": _pkg_version("transformers"),
        "git_commit": get_git_commit(),
    }


def get_git_commit() -> Optional[str]:
    """Best-effort current git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or None
    except Exception:
        return None


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set process-wide RNG seeds for reproducible evaluation/training."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return int(torch.distributed.get_rank())
        except Exception:
            pass
    for key in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def is_global_rank_zero() -> bool:
    return get_global_rank() == 0


def _fallback_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Vectorized fallback — no per-expert Python loop, minimal CUDA syncs."""
    if topk_ids.numel() == 0:
        num_tokens_post_pad.zero_()
        sorted_token_ids.zero_()
        experts_ids.fill_(-1)
        return

    device = topk_ids.device
    flat_topk = topk_ids.reshape(-1).to(torch.int64)
    num_tokens = flat_topk.numel()
    pad_token_id = num_tokens

    sorted_expert_vals, sort_order = torch.sort(flat_topk, stable=True)

    counts = torch.bincount(flat_topk, minlength=num_experts).to(torch.int64)
    padded_counts = ((counts + block_size - 1) // block_size) * block_size

    cum_padded = padded_counts.cumsum(0)
    dst_starts = torch.cat([cum_padded.new_zeros(1), cum_padded[:-1]])

    cum_counts = counts.cumsum(0)
    src_starts = torch.cat([counts.new_zeros(1), cum_counts[:-1]])

    total_output = int(cum_padded[-1].item())

    sorted_token_ids[:].fill_(pad_token_id)

    token_expert = sorted_expert_vals
    token_idx = torch.arange(num_tokens, device=device, dtype=torch.int64)
    within_expert = token_idx - src_starts[token_expert]
    output_pos = dst_starts[token_expert] + within_expert

    sorted_token_ids.scatter_(
        0, output_pos.clamp(max=sorted_token_ids.numel() - 1),
        sort_order.to(sorted_token_ids.dtype),
    )

    num_blocks_per_expert = padded_counts // block_size
    total_blocks = int(num_blocks_per_expert.sum().item())
    if total_blocks > 0:
        expert_indices = torch.arange(num_experts, device=device, dtype=experts_ids.dtype)
        experts_ids[:total_blocks] = torch.repeat_interleave(
            expert_indices, num_blocks_per_expert.to(torch.int64),
        )
    if total_blocks < experts_ids.numel():
        experts_ids[total_blocks:] = -1

    num_tokens_post_pad[0] = total_output


def _fallback_moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    # vLLM's output tensor is [M, H] while input is usually [M, top_k, H].
    if input.shape == output.shape:
        output.copy_(input)
    else:
        output.copy_(input.sum(dim=1))


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_external_dinfer_fuse_moe():
    cache_name = "_aoae_external_dinfer_fuse_moe"
    cached = sys.modules.get(cache_name)
    if cached is not None:
        return cached

    module_path = os.path.join(_repo_root(), "external", "dInfer", "tools", "fuse_moe.py")
    if not os.path.isfile(module_path):
        return None

    try:
        spec = importlib.util.spec_from_file_location(cache_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[cache_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(cache_name, None)
        return None
    return module


def _load_vllm_triton_moe_align():
    try:
        vllm_moe_align = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.moe_align_block_size"
        )
    except Exception:
        return None
    return getattr(vllm_moe_align, "moe_align_block_size_triton", None)


def _external_triton_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    triton_impl = _load_vllm_triton_moe_align()
    if triton_impl is None:
        fuse_moe = _load_external_dinfer_fuse_moe()
        triton_impl = getattr(fuse_moe, "moe_align_block_size_triton", None) if fuse_moe is not None else None
    if triton_impl is None:
        raise RuntimeError("no Triton moe_align_block_size implementation is available")
    triton_impl(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
    )


def inspect_vllm_moe_ops() -> Dict[str, Any]:
    """Inspect availability of required vLLM _moe_C symbols."""
    info: Dict[str, Any] = {
        "namespace_present": False,
        "available_ops": [],
        "missing_required_ops": list(_MOE_REQUIRED_OPS),
    }

    try:
        ns = torch.ops._moe_C
    except Exception:
        return info

    info["namespace_present"] = True
    available = [name for name in _MOE_REQUIRED_OPS if hasattr(ns, name)]
    missing = [name for name in _MOE_REQUIRED_OPS if name not in available]
    info["available_ops"] = available
    info["missing_required_ops"] = missing
    return info


def ensure_vllm_moe_runtime(
    *,
    strict: bool = True,
    verbose: bool = True,
    allow_python_fallback: bool = True,
) -> Dict[str, Any]:
    """Ensure required vLLM MoE ops are available (or patched).

    Returns a report with detected and patched symbols.
    Raises RuntimeError when strict=True and unsupported ops remain missing.
    """
    report = inspect_vllm_moe_ops()
    report["patched_fallback_ops"] = []
    report["patched_fast_fallback_ops"] = []

    if not report["missing_required_ops"]:
        return report

    try:
        import vllm._custom_ops as vllm_ops
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "Missing required vLLM MoE ops and failed to import vllm._custom_ops. "
                f"Missing: {report['missing_required_ops']}. Import error: {exc!r}"
            ) from exc
        return report

    if "moe_align_block_size" in report["missing_required_ops"]:
        patched = False
        if getattr(vllm_ops, "_aoae_moe_align_external_fallback", False):
            report["patched_fast_fallback_ops"].append("moe_align_block_size")
            patched = True
        elif not getattr(vllm_ops, "_aoae_moe_align_external_fallback", False):
            try:
                # Prefer a Triton implementation over the Python fallback.
                # Try in order: (1) inlined Triton kernels (no vLLM deps),
                # (2) vLLM's own Triton module, (3) dInfer fuse_moe.py.
                triton_fn = _build_triton_moe_align()
                if triton_fn is None:
                    triton_fn = _load_vllm_triton_moe_align()
                if triton_fn is None:
                    fuse_moe_mod = _load_external_dinfer_fuse_moe()
                    if fuse_moe_mod is not None:
                        triton_fn = getattr(fuse_moe_mod, "moe_align_block_size_triton", None)
                if triton_fn is None:
                    raise RuntimeError("no Triton moe_align_block_size implementation importable")
                vllm_ops.moe_align_block_size = triton_fn
                vllm_ops._aoae_moe_align_external_fallback = True
                report["patched_fast_fallback_ops"].append("moe_align_block_size")
                patched = True
                if verbose:
                    print(
                        "[Runtime] WARNING: torch.ops._moe_C.moe_align_block_size missing; "
                        "using Triton fallback."
                    )
            except Exception as _triton_exc:
                if verbose:
                    print(
                        f"[Runtime] WARNING: Triton moe_align_block_size fallback failed "
                        f"({_triton_exc!r}); will try Python fallback."
                    )
                patched = False
        if not patched:
            if getattr(vllm_ops, "_aoae_moe_align_fallback", False):
                report["patched_fallback_ops"].append("moe_align_block_size")
            else:
                if not allow_python_fallback:
                    raise RuntimeError(
                        "Compiled vLLM MoE custom ops are required for fast inference, and no Triton fallback "
                        "was available for moe_align_block_size. Rebuild/install a compatible vLLM with _moe_C ops."
                    )
                vllm_ops.moe_align_block_size = _fallback_moe_align_block_size
                vllm_ops._aoae_moe_align_fallback = True
                report["patched_fallback_ops"].append("moe_align_block_size")
                if verbose:
                    print(
                        "[Runtime] WARNING: torch.ops._moe_C.moe_align_block_size missing; "
                        "using Python fallback."
                    )

    if "moe_sum" in report["missing_required_ops"]:
        if getattr(vllm_ops, "_aoae_moe_sum_fallback", False):
            report["patched_fallback_ops"].append("moe_sum")
        else:
            vllm_ops.moe_sum = _fallback_moe_sum
            vllm_ops._aoae_moe_sum_fallback = True
            report["patched_fallback_ops"].append("moe_sum")
            if verbose:
                print(
                    "[Runtime] WARNING: torch.ops._moe_C.moe_sum missing; "
                    "using Python fallback."
                )

    post = inspect_vllm_moe_ops()
    # If patched, report missing ops as resolved for AOAE runtime purposes.
    still_missing = [
        op for op in post["missing_required_ops"]
        if op not in report["patched_fallback_ops"] and op not in report["patched_fast_fallback_ops"]
    ]
    report["missing_required_ops"] = still_missing
    report["available_ops"] = sorted(
        set(post["available_ops"])
        | set(report["patched_fallback_ops"])
        | set(report["patched_fast_fallback_ops"])
    )

    if strict and still_missing:
        raise RuntimeError(
            "Unsupported vLLM MoE runtime: missing required _moe_C ops "
            f"with no AOAE fallback available: {still_missing}. "
            "Use a compatible vLLM build or switch to backend='hf'."
        )
    return report
