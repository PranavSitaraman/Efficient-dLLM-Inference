"""
Evaluation Script for AOAE and Baselines (paper §3.7).

Runs inference on GSM8K / MATH benchmarks and computes:
  - Accuracy (exact match)
  - Throughput (tokens/sec, NFEs)
  - Pareto curves by sweeping policy temperature tau_pi

Usage:
    python3 -m aoae.evaluate --config configs/default.yaml --checkpoint outputs/default/policy_final.pt
"""

import os
import json
import time
import collections
import yaml
import torch
import numpy as np
from datetime import datetime, timezone
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datasets import load_dataset
from typing import Optional, List, Dict, Any

from .checkpoints import (
    load_state_dict_flexible,
    resolve_policy_checkpoint,
    resolve_sidecar_artifact,
)
from .models.base_model import LLaDABaseModel
from .models.soft_mask import SoftMaskedState
from .models.policy import AOAEPolicy, DefaultPolicy
from .models.prism import PRISMAdapter
from .inference import (
    aoae_inference,
    uniform_decode,
    confidence_threshold_decode,
    block_smode_decode,
    llada21_official_decode,
)
from .dinfer_integration import (
    run_blockwise_speculative_inference,
    run_speculative_inference,
)
from .runtime_checks import collect_runtime_info, set_global_seed, is_global_rank_zero
from .evaluators import build_evaluator
from .tasks import extract_answer, extract_prompt_and_reference


_MAX_SAVED_PREDICTIONS = 50


@dataclass
class EvalResult:
    method: str
    accuracy: float
    total_samples: int
    correct_samples: int
    avg_nfe: float          # average network function evaluations per sample
    avg_tokens_per_sec: float
    avg_gen_time_sec: float
    config_note: str = ""   # e.g., "tau_pi=0.5" or "tau_mask=0.9"
    cache_hit_rate: float = 0.0
    cache_commits: int = 0
    cache_invalidations: int = 0
    agreement_rate: float = 0.0
    draft_accept_rate: float = 0.0
    reuse_mean_safe: float = 0.0
    reuse_mean_js: float = 0.0
    access_rate: float = 0.0
    access_mandatory_rate: float = 0.0
    access_optional_rate: float = 0.0
    access_budget_utilization: float = 0.0
    access_effective_budget: float = 0.0
    access_next_h_precision: float = 0.0
    access_next_h_recall: float = 0.0
    access_next_h_f1: float = 0.0
    access_next_h_spec_precision: float = 0.0
    access_next_h_spec_recall: float = 0.0
    access_next_h_spec_f1: float = 0.0
    mean_boundary_depth: float = 0.0
    boundary_distribution: str = "{}"
    routing_entropy: float = 0.0
    max_routing_entropy: float = 0.0
    primary_skip_ratio: float = 0.0
    primary_full_steps: float = 0.0
    primary_partial_steps: float = 0.0
    primary_verified_positions: float = 0.0
    primary_full_equiv_positions: float = 0.0


_DEFAULT_BASELINE_METHODS = [
    "block_smode",
    "block_smode_mbe",
    "confidence_s_mode",
    "confidence_q_mode",
    "fast_dllm",
]


_BASELINE_TITLES = {
    "block_smode": "Baseline: Block S-Mode (tau=0.7, block=32)",
    "block_smode_mbe": "Baseline: Block S-Mode + MBE",
    "confidence_s_mode": "Baseline: S-Mode (tau=0.7, speed)",
    "confidence_q_mode": "Baseline: Q-Mode (tau=0.95, quality)",
    "fast_dllm": "Baseline: Fast-dLLM (Wu et al. 2025)",
    "llada21_speed_mode": "Baseline: LLaDA2.1 Speed Mode (threshold=0.5, edit=0.0, block diffusion)",
    "llada21_quality_mode": "Baseline: LLaDA2.1 Quality Mode (threshold=0.7, edit=0.5, block diffusion)",
}


def _get_prediction_save_limit(cfg: Dict[str, Any]) -> int:
    eval_cfg = cfg.get("evaluation", {})
    if not bool(eval_cfg.get("save_predictions", False)):
        return 0
    requested = int(eval_cfg.get("max_saved_predictions", _MAX_SAVED_PREDICTIONS))
    if requested <= 0:
        return 0
    return min(requested, _MAX_SAVED_PREDICTIONS)


def _maybe_record_prediction(
    predictions_sink: Optional[List[Dict[str, Any]]],
    prediction_limit: int,
    *,
    method: str,
    sample_index: int,
    question: str,
    reference: str,
    generated_text: str,
    is_correct: bool,
    generated_tokens: int,
    note: str = "",
) -> None:
    if predictions_sink is None or prediction_limit <= 0:
        return
    if len(predictions_sink) >= prediction_limit:
        return
    predictions_sink.append(
        {
            "method": method,
            "sample_index": int(sample_index),
            "correct": bool(is_correct),
            "question": question,
            "reference": reference,
            "generated_text": generated_text,
            "extracted_prediction": extract_answer(generated_text),
            "extracted_reference": extract_answer(reference),
            "generated_tokens": int(generated_tokens),
            "config_note": note,
        }
    )


def _save_prediction_artifact(
    predictions: List[Dict[str, Any]],
    output_dir: str,
    prediction_limit: int,
) -> str:
    payload = {
        "saved_predictions": len(predictions),
        "max_saved_predictions": int(prediction_limit),
        "truncated": len(predictions) >= prediction_limit,
        "predictions": predictions,
    }
    predictions_path = os.path.join(output_dir, "eval_predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(payload, f, indent=2)
    return predictions_path


def _get_baseline_methods(cfg: Dict[str, Any]) -> List[str]:
    methods = cfg.get("evaluation", {}).get("baseline_methods")
    if methods is None:
        return list(_DEFAULT_BASELINE_METHODS)
    return [str(method) for method in methods]


def _run_selected_baselines(
    base_model,
    eval_ds,
    tokenizer,
    cfg: dict,
    max_samples: Optional[int],
    all_results: List["EvalResult"],
    predictions_sink: Optional[List[Dict[str, Any]]] = None,
    prediction_limit: int = 0,
) -> None:
    for method in _get_baseline_methods(cfg):
        if is_global_rank_zero():
            print(f"\n====== {_BASELINE_TITLES.get(method, f'Baseline: {method}')} ======")
        r = evaluate_baseline(
            base_model,
            eval_ds,
            tokenizer,
            cfg,
            method,
            max_samples=max_samples,
            predictions_sink=predictions_sink,
            prediction_limit=prediction_limit,
        )
        all_results.append(r)
        if is_global_rank_zero():
            print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")


def _load_eval_dataset(dc: Dict[str, Any]):
    """Load the configured evaluation dataset, optionally with a builder config."""
    dataset_name = dc["eval_dataset"]
    dataset_config = dc.get("eval_dataset_config")
    split = dc["eval_split"]
    if dataset_config in (None, "", "null") and dataset_name == "openai/gsm8k":
        dataset_config = "main"
    if dataset_config in (None, "", "null"):
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, dataset_config, split=split)


def _extract_eval_prompt_reference(sample: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Normalize heterogeneous benchmark samples into a prompt/reference pair."""
    prompt, reference = extract_prompt_and_reference(sample)
    if prompt is not None:
        prompt = prompt.strip()
    if reference is not None:
        reference = reference.strip()
    return prompt, reference


def _remask_note(cfg: dict) -> str:
    return "remask=off" if cfg.get("inference", {}).get("disable_remask", False) else "remask=on"


def _append_note(note: str, extra: str) -> str:
    return f"{note},{extra}" if note else extra


def evaluate_aoae(
    base_model,
    policy,
    soft_mask,
    prism,
    dataset,
    tokenizer,
    cfg: dict,
    policy_temperature: float = 1.0,
    max_samples: Optional[int] = None,
    predictions_sink: Optional[List[Dict[str, Any]]] = None,
    prediction_limit: int = 0,
) -> EvalResult:
    """Evaluate AOAE on a dataset."""
    mask_id = cfg["base_model"]["mask_token_id"]
    device = next(policy.parameters()).device
    T = cfg["inference"]["steps"]

    correct = 0
    total = 0
    total_time = 0.0
    total_nfe = 0
    total_gen_tokens = 0
    evaluator = build_evaluator(cfg)

    n_eval = min(len(dataset), max_samples) if max_samples else len(dataset)

    for i in tqdm(range(n_eval), desc=f"AOAE (tau_pi={policy_temperature})", disable=not is_global_rank_zero()):
        sample = dataset[i]
        question, reference = _extract_eval_prompt_reference(sample)
        if not question or not reference:
            continue

        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            max_length=cfg["data"]["max_prompt_len"],
            truncation=True,
            return_tensors="pt",
        ).to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids, trajectory = aoae_inference(
                base_model=base_model,
                policy=policy,
                soft_mask_module=soft_mask,
                prism_adapter=prism,
                prompt_ids=prompt_ids,
                cfg=cfg,
                record_trajectory=False,
                policy_temperature=policy_temperature,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_tokens = output_ids[0, prompt_ids.shape[1]:]
        n_gen = int((gen_tokens != mask_id).sum().item())
        total_gen_tokens += n_gen
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        decision = evaluator.evaluate(gen_text, reference, sample=sample)
        is_correct = decision.correct
        if is_correct:
            correct += 1
        total += 1

        _maybe_record_prediction(
            predictions_sink,
            prediction_limit,
            method="AOAE",
            sample_index=i,
            question=question,
            reference=reference,
            generated_text=gen_text,
            is_correct=is_correct,
            generated_tokens=n_gen,
            note=_append_note(f"tau_pi={policy_temperature}", _remask_note(cfg)),
        )

        elapsed = t1 - t0
        total_time += elapsed
        total_nfe += T

    accuracy = correct / max(total, 1)
    avg_time = total_time / max(total, 1)
    avg_nfe = total_nfe / max(total, 1)
    avg_tps = total_gen_tokens / max(total_time, 1e-6)
    pc_cfg = cfg.get("inference", {}).get("positional_cache", {})
    pc_note = (
        f"pc=on(H={int(pc_cfg.get('horizon', 4))},B={int(pc_cfg.get('refresh_budget', 0))})"
        if pc_cfg.get("enabled", False)
        else "pc=off"
    )

    return EvalResult(
        method="AOAE",
        accuracy=accuracy,
        total_samples=total,
        correct_samples=correct,
        avg_nfe=avg_nfe,
        avg_tokens_per_sec=avg_tps,
        avg_gen_time_sec=avg_time,
        config_note=_append_note(f"tau_pi={policy_temperature},{pc_note}", _remask_note(cfg)),
    )


def evaluate_baseline(
    base_model,
    dataset,
    tokenizer,
    cfg: dict,
    method: str = "uniform",
    tau_mask: float = 0.9,
    tau_edit: float = 0.95,
    max_samples: Optional[int] = None,
    predictions_sink: Optional[List[Dict[str, Any]]] = None,
    prediction_limit: int = 0,
) -> EvalResult:
    """Evaluate a baseline decoder on a dataset."""
    mask_id = cfg["base_model"]["mask_token_id"]
    device = base_model.device
    T = cfg["inference"]["steps"]

    correct = 0
    total = 0
    total_time = 0.0
    total_gen_tokens = 0
    evaluator = build_evaluator(cfg)
    debug_eval = bool(cfg.get("evaluation", {}).get("debug_logging", False))
    rank0 = is_global_rank_zero()

    n_eval = min(len(dataset), max_samples) if max_samples else len(dataset)

    # Warm-up with realistic size to trigger Triton kernel compilation
    warmup_len = cfg["data"].get("max_prompt_len", 512) + cfg["inference"]["gen_length"]
    warmup_ids = torch.full((1, warmup_len), mask_id, dtype=torch.long, device=device)
    with torch.no_grad():
        base_model.forward(warmup_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del warmup_ids

    # Debug: show prompt format for first sample
    if debug_eval and rank0 and n_eval > 0:
        s0 = dataset[0]
        q0 = s0.get("question", s0.get("problem", ""))
        try:
            pt0 = tokenizer.apply_chat_template(
                [{"role": "user", "content": q0}],
                tokenize=False, add_generation_prompt=True,
            )
            print(f"  [DEBUG] Prompt format (first 200 chars): {pt0[:200]}")
        except Exception as e:
            print(f"  [DEBUG] apply_chat_template FAILED: {e}")
            print(f"  [DEBUG] Falling back to raw question: {q0[:100]}")

    progress = tqdm(range(n_eval), desc=f"Baseline ({method})", disable=not rank0)
    for i in progress:
        sample = dataset[i]
        question, reference = _extract_eval_prompt_reference(sample)
        if not question or not reference:
            continue

        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            max_length=cfg["data"]["max_prompt_len"],
            truncation=True,
            return_tensors="pt",
        ).to(device)
        # Ensure 2D shape [1, seq_len]
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if method == "uniform":
                output_ids = uniform_decode(base_model, prompt_ids, cfg)
            elif method == "confidence":
                output_ids = confidence_threshold_decode(
                    base_model, prompt_ids, cfg,
                    tau_mask=tau_mask, tau_edit=tau_edit,
                )
            elif method == "confidence_s_mode":
                output_ids = confidence_threshold_decode(
                    base_model, prompt_ids, cfg,
                    tau_mask=0.7, tau_edit=0.9, enable_t2t=True,
                )
            elif method == "confidence_q_mode":
                output_ids = confidence_threshold_decode(
                    base_model, prompt_ids, cfg,
                    tau_mask=0.95, tau_edit=0.99, enable_t2t=True,
                )
            elif method == "fast_dllm":
                output_ids = confidence_threshold_decode(
                    base_model, prompt_ids, cfg,
                    tau_mask=0.5, tau_edit=1.0, enable_t2t=False,
                )
            elif method == "llada21_speed_mode":
                output_ids = llada21_official_decode(
                    base_model, prompt_ids, cfg, mode="speed",
                )
            elif method == "llada21_quality_mode":
                output_ids = llada21_official_decode(
                    base_model, prompt_ids, cfg, mode="quality",
                )
            elif method == "block_smode":
                output_ids = block_smode_decode(
                    base_model, prompt_ids, cfg,
                    tau_mask=0.7, tau_edit=0.9, enable_mbe=False,
                )
            elif method == "block_smode_mbe":
                output_ids = block_smode_decode(
                    base_model, prompt_ids, cfg,
                    tau_mask=0.7, tau_edit=0.9, enable_mbe=True,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_tokens = output_ids[0, prompt_ids.shape[1]:]
        n_gen = int((gen_tokens != mask_id).sum().item())
        n_mask_remaining = int((gen_tokens == mask_id).sum().item())
        total_gen_tokens += n_gen
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        decision = evaluator.evaluate(gen_text, reference, sample=sample)
        is_correct = decision.correct
        if is_correct:
            correct += 1
        total += 1
        total_time += (t1 - t0)

        note = method
        if method == "confidence":
            note = f"tau_mask={tau_mask},tau_edit={tau_edit}"
        note = _append_note(note, _remask_note(cfg))

        _maybe_record_prediction(
            predictions_sink,
            prediction_limit,
            method=method,
            sample_index=i,
            question=question,
            reference=reference,
            generated_text=gen_text,
            is_correct=is_correct,
            generated_tokens=n_gen,
            note=note,
        )

        # Debug output for first 3 samples
        if debug_eval and rank0 and i < 3:
            gen_answer = extract_answer(gen_text)
            ref_answer = extract_answer(reference)
            print(f"\n  [DEBUG sample {i}] method={method}")
            print(f"    prompt_len={prompt_ids.shape[1]}, gen_len={len(gen_tokens)}, "
                  f"unmasked={n_gen}, masks_remaining={n_mask_remaining}")
            print(f"    reference_answer='{ref_answer}' (from: {reference[:80]}...)")
            print(f"    extracted_answer='{gen_answer}'")
            print(f"    correct={is_correct}")
            print(f"    generated_text (first 300 chars): {gen_text[:300]}")

    accuracy = correct / max(total, 1)
    avg_time = total_time / max(total, 1)
    avg_nfe = T
    # TPS = actual generated tokens / wall time
    avg_tps = total_gen_tokens / max(total_time, 1e-6)

    note = method
    if method == "confidence":
        note = f"tau_mask={tau_mask},tau_edit={tau_edit}"
    note = _append_note(note, _remask_note(cfg))

    return EvalResult(
        method=method,
        accuracy=accuracy,
        total_samples=total,
        correct_samples=correct,
        avg_nfe=avg_nfe,
        avg_tokens_per_sec=avg_tps,
        avg_gen_time_sec=avg_time,
        config_note=note,
    )


def evaluate_speculative(
    dual_model,
    policy,
    soft_mask,
    prism,
    dataset,
    tokenizer,
    cfg: dict,
    policy_temperature: float = 1.0,
    max_samples: Optional[int] = None,
    dynamics_sink: Optional[List[Dict[str, Any]]] = None,
    predictions_sink: Optional[List[Dict[str, Any]]] = None,
    prediction_limit: int = 0,
) -> EvalResult:
    """Evaluate speculative diffusion (dual-model) on a dataset."""
    mask_id = cfg["base_model"]["mask_token_id"]
    device = dual_model.device
    T = cfg["inference"]["steps"]
    schedule = str(cfg.get("inference", {}).get("speculative_schedule", "aoae")).strip().lower()

    correct = 0
    total = 0
    total_time = 0.0
    total_gen_tokens = 0
    total_nfe = 0
    total_agreement = 0.0
    total_agreement_obs = 0
    total_cache_commits = 0
    total_cache_invalidations = 0
    total_draft_accepts = 0
    total_draft_rejects = 0
    total_reuse_safe = 0.0
    total_reuse_safe_obs = 0
    total_reuse_js = 0.0
    total_access_rate = 0.0
    total_access_mandatory = 0.0
    total_access_optional = 0.0
    total_access_budget_util = 0.0
    total_access_effective_budget = 0.0
    total_access_next_h_precision = 0.0
    total_access_next_h_recall = 0.0
    total_access_next_h_f1 = 0.0
    total_access_next_h_spec_precision = 0.0
    total_access_next_h_spec_recall = 0.0
    total_access_next_h_spec_f1 = 0.0
    total_boundary_depth = 0.0
    total_routing_entropy = 0.0
    total_max_routing_entropy = 0.0
    routing_entropy_samples = 0
    total_primary_skip_ratio = 0.0
    total_primary_full_steps = 0.0
    total_primary_partial_steps = 0.0
    total_primary_verified_positions = 0.0
    total_primary_full_equiv_positions = 0.0
    boundary_dist_counter: collections.Counter[str] = collections.Counter()
    evaluator = build_evaluator(cfg)

    n_eval = min(len(dataset), max_samples) if max_samples else len(dataset)
    tau_r = cfg["base_model"].get("routing_temperature", 0.01)
    reuse_method = cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")
    pc_cfg = cfg.get("inference", {}).get("positional_cache", {})
    pc_note = (
        f"pc=on(H={int(pc_cfg.get('horizon', 4))},B={int(pc_cfg.get('refresh_budget', 0))})"
        if pc_cfg.get("enabled", False)
        else "pc=off"
    )

    # Warm-up with realistic input size to trigger Triton kernel compilation
    warmup_len = cfg["data"].get("max_prompt_len", 512) + cfg["inference"]["gen_length"]
    warmup_ids = torch.full((1, warmup_len), mask_id, dtype=torch.long, device=device)
    with torch.no_grad():
        dual_model.auxiliary_forward(warmup_ids)
        dual_model.primary_forward(warmup_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del warmup_ids

    for i in tqdm(range(n_eval), desc=f"Speculative (tau_r={tau_r})", disable=not is_global_rank_zero()):
        sample = dataset[i]
        question, reference = _extract_eval_prompt_reference(sample)
        if not question or not reference:
            continue

        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            max_length=cfg["data"]["max_prompt_len"],
            truncation=True,
            return_tensors="pt",
        ).to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if schedule == "llada21_block":
                output_ids, stats = run_blockwise_speculative_inference(
                    dual_model=dual_model,
                    policy=policy,
                    soft_mask_module=soft_mask,
                    prism_adapter=prism,
                    prompt_ids=prompt_ids,
                    cfg=cfg,
                    policy_temperature=policy_temperature,
                )
            else:
                output_ids, stats = run_speculative_inference(
                    dual_model=dual_model,
                    policy=policy,
                    soft_mask_module=soft_mask,
                    prism_adapter=prism,
                    prompt_ids=prompt_ids,
                    cfg=cfg,
                    policy_temperature=policy_temperature,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_tokens = output_ids[0, prompt_ids.shape[1]:]
        # Count actual generated tokens (non-mask)
        n_gen = int((gen_tokens != mask_id).sum().item())
        total_gen_tokens += n_gen
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        decision = evaluator.evaluate(gen_text, reference, sample=sample)
        is_correct = decision.correct
        if is_correct:
            correct += 1
        total += 1

        _maybe_record_prediction(
            predictions_sink,
            prediction_limit,
            method="Speculative-AOAE",
            sample_index=i,
            question=question,
            reference=reference,
            generated_text=gen_text,
            is_correct=is_correct,
            generated_tokens=n_gen,
            note=_append_note(
                f"tau_r={tau_r},tau_pi={policy_temperature},reuse={reuse_method},{pc_note},sched={schedule}",
                _remask_note(cfg),
            ),
        )

        elapsed = t1 - t0
        total_time += elapsed
        pri_steps = int(stats.get("primary_steps", T))
        aux_steps = int(stats.get("aux_only_steps", 0))
        total_nfe += pri_steps * 2 + aux_steps  # 2 fwd per primary step, 1 per aux-only
        total_cache_commits += int(stats.get("total_commits", 0))
        total_cache_invalidations += int(stats.get("total_invalidations", 0))
        total_draft_accepts += int(stats.get("draft_accepts", 0))
        total_draft_rejects += int(stats.get("draft_rejects", 0))
        agreement_obs = int(stats.get("agreement_observations", 0))
        total_agreement += float(stats.get("mean_agreement", 0.0)) * agreement_obs
        total_agreement_obs += agreement_obs
        safe_reuse_obs = int(stats.get("safe_reuse_observations", 0))
        total_reuse_safe += float(stats.get("reuse_mean_safe_reuse", 0.0)) * safe_reuse_obs
        total_reuse_safe_obs += safe_reuse_obs
        total_reuse_js += float(stats.get("reuse_mean_js_divergence", 0.0))
        total_access_rate += float(stats.get("access_access_rate", 0.0))
        total_access_mandatory += float(stats.get("access_access_mandatory_rate", 0.0))
        total_access_optional += float(stats.get("access_access_optional_rate", 0.0))
        total_access_budget_util += float(stats.get("access_access_budget_utilization", 0.0))
        total_access_effective_budget += float(stats.get("access_access_effective_budget", 0.0))
        total_access_next_h_precision += float(stats.get("access_next_h_precision", 0.0))
        total_access_next_h_recall += float(stats.get("access_next_h_recall", 0.0))
        total_access_next_h_f1 += float(stats.get("access_next_h_f1", 0.0))
        total_access_next_h_spec_precision += float(stats.get("access_next_h_spec_precision", 0.0))
        total_access_next_h_spec_recall += float(stats.get("access_next_h_spec_recall", 0.0))
        total_access_next_h_spec_f1 += float(stats.get("access_next_h_spec_f1", 0.0))
        total_boundary_depth += float(stats.get("mean_boundary_depth", 0.0))
        total_primary_skip_ratio += float(stats.get("primary_skip_ratio", 0.0))
        total_primary_full_steps += float(stats.get("primary_full_steps", 0.0))
        total_primary_partial_steps += float(stats.get("primary_partial_steps", 0.0))
        total_primary_verified_positions += float(stats.get("primary_verified_positions", 0.0))
        total_primary_full_equiv_positions += float(stats.get("primary_full_equiv_positions", 0.0))
        try:
            bd = json.loads(stats.get("boundary_distribution", "{}"))
            for k, v in bd.items():
                boundary_dist_counter[str(k)] += int(v)
        except Exception:
            pass
        try:
            from aoae.models.soft_moe import compute_routing_entropy
            ent_info = compute_routing_entropy(dual_model._model.model)
            if ent_info.get("num_layers_with_data", 0) > 0:
                total_routing_entropy += float(ent_info.get("mean_entropy", 0.0))
                total_max_routing_entropy += float(ent_info.get("max_possible_entropy", 0.0))
                routing_entropy_samples += 1
        except Exception:
            pass
        if dynamics_sink is not None and "kv_dynamics" in stats:
            dynamics_sink.append(
                {
                    "tau_r": float(tau_r),
                    "tau_pi": float(policy_temperature),
                    "reuse_signal_method": reuse_method,
                    "sample_index": i,
                    "sample_correct": bool(is_correct),
                    "kv_dynamics": stats["kv_dynamics"],
                }
            )

    accuracy = correct / max(total, 1)
    avg_time = total_time / max(total, 1)
    avg_nfe = total_nfe / max(total, 1)
    # TPS = actual generated tokens / wall time
    avg_tps = total_gen_tokens / max(total_time, 1e-6)
    total_cache_ops = total_cache_commits + total_cache_invalidations
    cache_hit_rate = (
        (total_cache_commits - total_cache_invalidations) / max(total_cache_ops, 1)
    )
    draft_accept_rate = total_draft_accepts / max(total_draft_accepts + total_draft_rejects, 1)
    agreement_rate = total_agreement / max(total_agreement_obs, 1)
    reuse_mean_safe = total_reuse_safe / max(total_reuse_safe_obs, 1)
    reuse_mean_js = total_reuse_js / max(total, 1)
    access_rate = total_access_rate / max(total, 1)
    access_mandatory_rate = total_access_mandatory / max(total, 1)
    access_optional_rate = total_access_optional / max(total, 1)
    access_budget_utilization = total_access_budget_util / max(total, 1)
    access_effective_budget = total_access_effective_budget / max(total, 1)
    access_next_h_precision = total_access_next_h_precision / max(total, 1)
    access_next_h_recall = total_access_next_h_recall / max(total, 1)
    access_next_h_f1 = total_access_next_h_f1 / max(total, 1)
    access_next_h_spec_precision = total_access_next_h_spec_precision / max(total, 1)
    access_next_h_spec_recall = total_access_next_h_spec_recall / max(total, 1)
    access_next_h_spec_f1 = total_access_next_h_spec_f1 / max(total, 1)
    mean_boundary_depth = total_boundary_depth / max(total, 1)
    boundary_distribution = json.dumps(dict(sorted(boundary_dist_counter.items(), key=lambda kv: int(kv[0]))))

    routing_ent = total_routing_entropy / max(routing_entropy_samples, 1)
    max_routing_ent = total_max_routing_entropy / max(routing_entropy_samples, 1)
    primary_skip_ratio = total_primary_skip_ratio / max(total, 1)
    primary_full_steps = total_primary_full_steps / max(total, 1)
    primary_partial_steps = total_primary_partial_steps / max(total, 1)
    primary_verified_positions = total_primary_verified_positions / max(total, 1)
    primary_full_equiv_positions = total_primary_full_equiv_positions / max(total, 1)

    return EvalResult(
        method="Speculative-AOAE",
        accuracy=accuracy,
        total_samples=total,
        correct_samples=correct,
        avg_nfe=avg_nfe,
        avg_tokens_per_sec=avg_tps,
        avg_gen_time_sec=avg_time,
        config_note=_append_note(
            f"tau_r={tau_r},tau_pi={policy_temperature},reuse={reuse_method},{pc_note},sched={schedule}",
            _remask_note(cfg),
        ),
        cache_hit_rate=cache_hit_rate,
        cache_commits=total_cache_commits,
        cache_invalidations=total_cache_invalidations,
        agreement_rate=agreement_rate,
        draft_accept_rate=draft_accept_rate,
        reuse_mean_safe=reuse_mean_safe,
        reuse_mean_js=reuse_mean_js,
        access_rate=access_rate,
        access_mandatory_rate=access_mandatory_rate,
        access_optional_rate=access_optional_rate,
        access_budget_utilization=access_budget_utilization,
        access_effective_budget=access_effective_budget,
        access_next_h_precision=access_next_h_precision,
        access_next_h_recall=access_next_h_recall,
        access_next_h_f1=access_next_h_f1,
        access_next_h_spec_precision=access_next_h_spec_precision,
        access_next_h_spec_recall=access_next_h_spec_recall,
        access_next_h_spec_f1=access_next_h_spec_f1,
        mean_boundary_depth=mean_boundary_depth,
        boundary_distribution=boundary_distribution,
        routing_entropy=routing_ent,
        max_routing_entropy=max_routing_ent,
        primary_skip_ratio=primary_skip_ratio,
        primary_full_steps=primary_full_steps,
        primary_partial_steps=primary_partial_steps,
        primary_verified_positions=primary_verified_positions,
        primary_full_equiv_positions=primary_full_equiv_positions,
    )


def run_pareto_sweep(
    base_model,
    policy,
    soft_mask,
    prism,
    dataset,
    tokenizer,
    cfg,
    max_samples,
    predictions_sink: Optional[List[Dict[str, Any]]] = None,
    prediction_limit: int = 0,
) -> List[EvalResult]:
    """Sweep policy temperature to generate Pareto curve points."""
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    results = []

    for tau_pi in temperatures:
        print(f"\n--- AOAE with tau_pi = {tau_pi} ---")
        r = evaluate_aoae(
            base_model, policy, soft_mask, prism, dataset, tokenizer, cfg,
            policy_temperature=tau_pi,
            max_samples=max_samples,
            predictions_sink=predictions_sink,
            prediction_limit=prediction_limit,
        )
        results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

    return results


def _aggregate_kv_dynamics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    summaries = [r.get("kv_dynamics", {}).get("summary", {}) for r in records]
    summaries = [s for s in summaries if s]
    if not summaries:
        return {}

    def _mean(name: str) -> float:
        vals = [float(s.get(name, 0.0)) for s in summaries if name in s]
        return float(np.mean(vals)) if vals else 0.0

    def _majority_label(name: str, default: str = "unknown") -> str:
        counts: Dict[str, int] = {}
        for s in summaries:
            label = str(s.get(name, default))
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return default
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # Locality metrics (windowed hit ratios) and age-bucket drifts.
    locality_vals: Dict[str, List[float]] = {}
    age_vals: Dict[str, List[float]] = {}
    for s in summaries:
        for k, v in s.get("locality", {}).items():
            locality_vals.setdefault(k, []).append(float(v))
        for k, v in s.get("age_drift_means", {}).items():
            age_vals.setdefault(k, []).append(float(v))

    # Aggregate per-layer drift by averaging across samples.
    layer_map: Dict[int, List[float]] = {}
    for r in records:
        for row in r.get("kv_dynamics", {}).get("per_layer", []):
            idx = int(row.get("layer_idx", 0))
            val = float(row.get("mean_drift", row.get("mean_hidden_drift", 0.0)))
            layer_map.setdefault(idx, []).append(val)
    per_layer = [
        {
            "layer_idx": idx,
            "mean_drift": float(np.mean(vals)),
            "mean_hidden_drift": float(np.mean(vals)),
        }
        for idx, vals in sorted(layer_map.items())
    ]

    attn_map: Dict[int, List[float]] = {}
    for r in records:
        for row in r.get("kv_dynamics", {}).get("per_layer_attention_deviation", []):
            idx = int(row.get("layer_idx", 0))
            val = float(row.get("mean_attention_deviation", 0.0))
            attn_map.setdefault(idx, []).append(val)
    per_layer_attention_deviation = [
        {
            "layer_idx": idx,
            "mean_attention_deviation": float(np.mean(vals)),
            "deviation_measure": "mean_l2_delta",
        }
        for idx, vals in sorted(attn_map.items())
    ]

    return {
        "num_records": len(records),
        "mean_agreement": _mean("mean_agreement"),
        "mean_access": _mean("mean_access"),
        "mean_confidence_masked": _mean("mean_confidence_masked"),
        "mean_confidence_unmasked": _mean("mean_confidence_unmasked"),
        "layer_drift_measure": _majority_label("layer_drift_measure", default="hidden_state_proxy"),
        "exact_kv_drift_steps": _mean("exact_kv_drift_steps"),
        "hidden_state_proxy_steps": _mean("hidden_state_proxy_steps"),
        "mean_layer_drift_slope": _mean("layer_drift_slope"),
        "mean_off_by_one_drift_ratio": _mean("off_by_one_drift_ratio"),
        "mean_confident_token_drift_ratio": _mean("confident_token_drift_ratio"),
        "mean_thrash_rate_given_cached": _mean("thrash_rate_given_cached"),
        "mean_locality": {
            k: float(np.mean(vs)) for k, vs in sorted(locality_vals.items())
        },
        "mean_age_drift": {
            k: float(np.mean(vs)) for k, vs in sorted(age_vals.items())
        },
        "per_layer_drift": per_layer,
        "attention_deviation_available": any(
            bool(s.get("attention_deviation_available", False)) for s in summaries
        ),
        "attention_deviation_measure": _majority_label(
            "attention_deviation_measure", default="unavailable",
        ),
        "mean_attention_deviation_slope": _mean("attention_deviation_slope"),
        "per_layer_attention_deviation": per_layer_attention_deviation,
    }


def _save_kv_dynamics_artifacts(records: List[Dict[str, Any]], output_dir: str) -> None:
    if not records:
        return

    raw_path = os.path.join(output_dir, "kv_dynamics_records.json")
    with open(raw_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"KV dynamics records saved to {raw_path}")

    summary = _aggregate_kv_dynamics(records)
    summary_path = os.path.join(output_dir, "kv_dynamics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"KV dynamics summary saved to {summary_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"KV dynamics plotting skipped (matplotlib unavailable): {e}")
        return

    per_layer = summary.get("per_layer_drift", [])
    if per_layer:
        x = [int(r["layer_idx"]) for r in per_layer]
        y = [float(r.get("mean_drift", r.get("mean_hidden_drift", 0.0))) for r in per_layer]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(x, y, "o-", linewidth=2)
        ax.set_xlabel("Layer index")
        drift_measure = str(summary.get("layer_drift_measure", "hidden_state_proxy"))
        if drift_measure == "exact_kv":
            ax.set_ylabel("Mean KV drift")
            ax.set_title("Layer-wise KV Drift")
        elif drift_measure == "mixed":
            ax.set_ylabel("Mean drift")
            ax.set_title("Layer-wise Drift (Mixed KV/Proxy)")
        else:
            ax.set_ylabel("Mean hidden-state drift")
            ax.set_title("Layer-wise Drift (Hidden-State Proxy)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        layer_plot = os.path.join(output_dir, "kv_dynamics_layer_drift.png")
        fig.savefig(layer_plot, dpi=150)
        plt.close(fig)
        print(f"KV dynamics plot saved to {layer_plot}")

    per_layer_attn = summary.get("per_layer_attention_deviation", [])
    if per_layer_attn:
        x = [int(r["layer_idx"]) for r in per_layer_attn]
        y = [float(r["mean_attention_deviation"]) for r in per_layer_attn]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(x, y, "o-", linewidth=2, color="#c65d00")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Mean attention deviation")
        ax.set_title("Layer-wise Attention Deviation")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        attn_plot = os.path.join(output_dir, "kv_dynamics_attention_deviation.png")
        fig.savefig(attn_plot, dpi=150)
        plt.close(fig)
        print(f"Attention deviation plot saved to {attn_plot}")


def _save_eval_plots(all_results: List[EvalResult], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Eval plotting skipped (matplotlib unavailable): {e}")
        return

    if not all_results:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for r in all_results:
        label = r.method
        ax.scatter(r.avg_tokens_per_sec, r.accuracy, s=80, alpha=0.8, label=label)
    ax.set_xlabel("Tokens / sec")
    ax.set_ylabel("Accuracy")
    ax.set_title("Eval: Accuracy vs Throughput")
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup_h = []
    dedup_l = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        dedup_h.append(h)
        dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "eval_tps_vs_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Eval plot saved to {path}")

def _build_run_metadata(
    cfg: dict,
    mode: str,
    config_path: Optional[str],
    checkpoint_path: Optional[str],
    max_samples: Optional[int],
    results_path: str,
    num_results: int,
    predictions_path: Optional[str] = None,
    saved_predictions: int = 0,
    prediction_limit: int = 0,
) -> Dict[str, Any]:
    runtime = collect_runtime_info()
    eval_cfg = cfg.get("evaluation", {})
    return {
        "schema_version": "aoae_eval_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": cfg.get("logging", {}).get("run_name", ""),
        "output_dir": cfg.get("logging", {}).get("output_dir", ""),
        "results_path": results_path,
        "predictions_path": predictions_path,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "mode": mode,
        "model_name_or_path": cfg.get("base_model", {}).get("name_or_path", ""),
        "backend": cfg.get("base_model", {}).get("backend", "auto"),
        "speculative_schedule": cfg.get("inference", {}).get("speculative_schedule", "aoae"),
        "routing_temperature": cfg.get("base_model", {}).get("routing_temperature"),
        "seed": int(cfg.get("hardware", {}).get("seed", 42)),
        "deterministic": bool(cfg.get("hardware", {}).get("deterministic", False)),
        "task_type": eval_cfg.get("task_type", "math"),
        "code_timeout_sec": eval_cfg.get("code", {}).get("timeout_sec"),
        "code_cpu_time_limit_sec": eval_cfg.get("code", {}).get("cpu_time_limit_sec"),
        "code_memory_limit_mb": eval_cfg.get("code", {}).get("memory_limit_mb"),
        "disable_remask": bool(cfg.get("inference", {}).get("disable_remask", False)),
        "reuse_signal_method": cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match"),
        "reuse_signal_threshold": cfg.get("inference", {}).get("reuse_signal", {}).get("threshold", 0.0),
        "compose_gamma": cfg.get("inference", {}).get("compose_gamma", 0.0),
        "llada21_use_block_diffusion": bool(cfg.get("inference", {}).get("llada21_official", {}).get("use_block_diffusion", False)),
        "llada21_threshold": cfg.get("inference", {}).get("llada21_official", {}).get("threshold"),
        "llada21_editing_threshold": cfg.get("inference", {}).get("llada21_official", {}).get("editing_threshold"),
        "llada21_max_post_steps": cfg.get("inference", {}).get("llada21_official", {}).get("max_post_steps"),
        "llada21_enable_mbe": bool(cfg.get("inference", {}).get("llada21_official", {}).get("enable_mbe", False)),
        "candidate_policy": cfg.get("inference", {}).get("positional_cache", {}).get("candidate_policy", "learned_topb"),
        "positional_cache_enabled": bool(cfg.get("inference", {}).get("positional_cache", {}).get("enabled", False)),
        "positional_cache_horizon": int(cfg.get("inference", {}).get("positional_cache", {}).get("horizon", 4)),
        "positional_cache_refresh_budget": int(cfg.get("inference", {}).get("positional_cache", {}).get("refresh_budget", 0)),
        "boundary_head_enabled": bool(cfg.get("policy", {}).get("boundary_head", {}).get("enabled", False)),
        "boundary_num_bins": int(cfg.get("policy", {}).get("boundary_head", {}).get("num_bins", 0)),
        "track_kv_dynamics": bool(cfg.get("analysis", {}).get("track_kv_dynamics", False)),
        "kv_locality_windows": cfg.get("analysis", {}).get("locality_windows", [8, 16, 32]),
        "kv_confidence_threshold": cfg.get("analysis", {}).get("confidence_threshold", 0.9),
        "kv_attention_proxy_top_frac": cfg.get("analysis", {}).get("attention_proxy_top_frac", 0.1),
        "eval_dataset": cfg.get("data", {}).get("eval_dataset", ""),
        "eval_dataset_config": cfg.get("data", {}).get("eval_dataset_config"),
        "eval_split": cfg.get("data", {}).get("eval_split", ""),
        "eval_max_samples": max_samples,
        "save_predictions": bool(cfg.get("evaluation", {}).get("save_predictions", False)),
        "saved_predictions": int(saved_predictions),
        "max_saved_predictions": int(prediction_limit),
        "num_results": num_results,
        "host": runtime.get("host"),
        "git_commit": runtime.get("git_commit"),
        "torch_version": runtime.get("torch_version"),
        "vllm_version": runtime.get("vllm_version"),
        "transformers_version": runtime.get("transformers_version"),
        "cuda_available": runtime.get("cuda_available"),
        "cuda_device_count": runtime.get("cuda_device_count"),
        "cuda_devices": runtime.get("cuda_devices"),
    }


def _append_manifest(metadata: Dict[str, Any], all_results: List[EvalResult]) -> str:
    manifest_path = os.path.join("results", "experiment_manifest.jsonl")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    run_id = (
        f"{metadata.get('run_name', 'run')}:"
        f"{metadata.get('created_at_utc', '')}:"
        f"{metadata.get('mode', '')}"
    )
    with open(manifest_path, "a") as f:
        for row in all_results:
            entry = dict(metadata)
            entry["run_id"] = run_id
            entry.update(asdict(row))
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    return manifest_path


def main(
    cfg: dict,
    checkpoint_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    mode: str = "standard",
    config_path: Optional[str] = None,
    skip_baselines: bool = False,
    speculative_policy_temperatures: Optional[List[float]] = None,
    preloaded_dual_model=None,
    preloaded_eval_ds=None,
    preloaded_base_model=None,
):
    """Run full evaluation suite.

    Args:
        cfg: config dict.
        checkpoint_path: path to policy checkpoint.
        max_samples: max samples to evaluate.
        mode: 'standard' for single-model, 'speculative' for dual-model.
        config_path: path to config YAML (for metadata/manifest tracking).
        skip_baselines: skip baseline decoding methods (useful for sweeps).
        speculative_policy_temperatures: optional tau_pi list for speculative runs.
        preloaded_dual_model: reuse a previously loaded DualModelWrapper (avoids reload).
        preloaded_eval_ds: reuse a previously loaded eval dataset.
        preloaded_base_model: reuse a previously loaded LLaDABaseModel for standard evals.
    """
    seed = int(cfg.get("hardware", {}).get("seed", 42))
    deterministic = bool(cfg.get("hardware", {}).get("deterministic", False))
    set_global_seed(seed, deterministic=deterministic)
    if not deterministic:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dc = cfg["data"]
    if preloaded_eval_ds is not None:
        eval_ds = preloaded_eval_ds
    else:
        if is_global_rank_zero():
            print(f"Loading eval dataset: {dc['eval_dataset']}...")
        eval_ds = _load_eval_dataset(dc)
    if max_samples is None:
        max_samples = dc.get("eval_max_samples")
    if speculative_policy_temperatures is None:
        speculative_policy_temperatures = [0.5, 1.0, 1.5]
    prediction_limit = _get_prediction_save_limit(cfg)

    all_results: List[EvalResult] = []
    kv_dynamics_records: List[Dict[str, Any]] = []
    saved_predictions: List[Dict[str, Any]] = []
    owned_model = None
    try:
        if mode == "speculative" or cfg["base_model"].get("backend") == "dual":
            from .models.dual_model import DualModelWrapper
            schedule = str(cfg.get("inference", {}).get("speculative_schedule", "aoae")).strip().lower()
            uses_aoae_policy = schedule != "llada21_block"

            if preloaded_dual_model is not None:
                dual_model = preloaded_dual_model
                tau_r = cfg["base_model"].get("routing_temperature", 0.01)
                dual_model.set_tau_r(tau_r)
                soft_topk = cfg["base_model"].get("soft_topk")
                if soft_topk is not None:
                    dual_model.set_soft_topk(soft_topk)
            else:
                if is_global_rank_zero():
                    print("Loading dual model (hard auxiliary + soft primary)...")
                dual_model = DualModelWrapper(cfg)
                dual_model = dual_model.to(device)
                owned_model = dual_model
            tokenizer = dual_model.tokenizer
            mask_id = cfg["base_model"]["mask_token_id"]
            embed_w = dual_model.get_embedding_weight() if uses_aoae_policy else None
            embed_dim = embed_w.shape[1] if embed_w is not None else None

            base_model = dual_model._model

            if not skip_baselines:
                _run_selected_baselines(
                    base_model,
                    eval_ds,
                    tokenizer,
                    cfg,
                    max_samples,
                    all_results,
                    predictions_sink=saved_predictions,
                    prediction_limit=prediction_limit,
                )

            if not checkpoint_path:
                checkpoint_path = resolve_policy_checkpoint(
                    checkpoint_path,
                    cfg["logging"]["output_dir"],
                )

            has_trained_policy = checkpoint_path and os.path.exists(checkpoint_path) and uses_aoae_policy

            soft_mask = None
            if uses_aoae_policy:
                soft_mask = SoftMaskedState(cfg, embed_w).to(device)
                soft_mask.set_mask_embedding(mask_id)
                soft_mask.eval()
            elif checkpoint_path and os.path.exists(checkpoint_path) and is_global_rank_zero():
                print(
                    "Checkpoint provided, but inference.speculative_schedule=llada21_block "
                    "does not use the AOAE policy. Ignoring checkpoint for this run."
                )

            if has_trained_policy:
                policy = AOAEPolicy(cfg, input_dim=embed_dim).to(device)
                if is_global_rank_zero():
                    print(f"\nLoading trained policy from {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location=device)
                load_state_dict_flexible(policy, ckpt["policy"], "policy")
                if "soft_mask" in ckpt:
                    load_state_dict_flexible(soft_mask, ckpt["soft_mask"], "soft_mask")
                policy.eval()

                prism_path = resolve_sidecar_artifact(
                    checkpoint_path,
                    cfg["logging"]["output_dir"],
                    "prism_adapter.pt",
                )
                prism = None
                if prism_path is not None:
                    prism = PRISMAdapter(cfg, embed_dim).to(device)
                    prism.load_state_dict(torch.load(prism_path, map_location=device))
                    prism.eval()
                    if is_global_rank_zero():
                        print(f"  Loaded PRISM adapter from {prism_path}")

                tau_r = cfg["base_model"].get("routing_temperature", 0.01)
                if is_global_rank_zero():
                    print(f"\n====== Speculative AOAE — Trained Policy (tau_r={tau_r}) ======")
                for tau_pi in speculative_policy_temperatures:
                    if is_global_rank_zero():
                        print(f"\n--- tau_pi = {tau_pi} ---")
                    r = evaluate_speculative(
                        dual_model, policy, soft_mask, prism, eval_ds, tokenizer, cfg,
                        policy_temperature=tau_pi,
                        max_samples=max_samples,
                        dynamics_sink=kv_dynamics_records,
                        predictions_sink=saved_predictions,
                        prediction_limit=prediction_limit,
                    )
                    all_results.append(r)
                    if is_global_rank_zero():
                        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")
            else:
                tau_r = cfg["base_model"].get("routing_temperature", 0.01)
                if uses_aoae_policy:
                    if is_global_rank_zero():
                        print(f"\n====== Speculative AOAE — Default Policy (tau_r={tau_r}) ======")
                        print("  (No trained checkpoint found; using a confidence-guided training-free heuristic policy)")
                    policy = DefaultPolicy(
                        tau_mask=0.7, num_steps=cfg["inference"]["steps"],
                    ).to(device)
                    policy.eval()
                else:
                    if is_global_rank_zero():
                        print(f"\n====== Speculative PoC1 — Blockwise LLaDA2.1 Schedule (tau_r={tau_r}) ======")
                        print("  (Using hard auxiliary for agreement/reuse only; token updates follow the soft primary block schedule)")
                    policy = None

                r = evaluate_speculative(
                    dual_model, policy, soft_mask, None, eval_ds, tokenizer, cfg,
                    policy_temperature=1.0,
                    max_samples=max_samples,
                    dynamics_sink=kv_dynamics_records,
                    predictions_sink=saved_predictions,
                    prediction_limit=prediction_limit,
                )
                all_results.append(r)
                if is_global_rank_zero():
                    print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

        else:
            if preloaded_base_model is not None:
                base_model = preloaded_base_model
                tau_r = cfg.get("base_model", {}).get("routing_temperature")
                if tau_r is not None:
                    set_tau = getattr(base_model, "set_routing_temperature", None)
                    if callable(set_tau):
                        set_tau(float(tau_r))
                soft_topk = cfg.get("base_model", {}).get("soft_topk")
                if soft_topk is not None:
                    set_topk = getattr(base_model, "set_soft_topk", None)
                    if callable(set_topk):
                        set_topk(int(soft_topk))
            else:
                if is_global_rank_zero():
                    print("Loading base model...")
                base_model = LLaDABaseModel(cfg)
                base_model = base_model.to(device)
                owned_model = base_model
            tokenizer = base_model.tokenizer
            mask_id = cfg["base_model"]["mask_token_id"]

            if not skip_baselines:
                _run_selected_baselines(
                    base_model,
                    eval_ds,
                    tokenizer,
                    cfg,
                    max_samples,
                    all_results,
                    predictions_sink=saved_predictions,
                    prediction_limit=prediction_limit,
                )

            if checkpoint_path and os.path.exists(checkpoint_path):
                if is_global_rank_zero():
                    print(f"\nLoading AOAE policy from {checkpoint_path}")
                embed_w = base_model.get_embedding_weight()
                embed_dim = embed_w.shape[1]

                soft_mask = SoftMaskedState(cfg, embed_w).to(device)
                soft_mask.set_mask_embedding(mask_id)
                policy = AOAEPolicy(cfg, input_dim=embed_dim).to(device)

                ckpt = torch.load(checkpoint_path, map_location=device)
                load_state_dict_flexible(policy, ckpt["policy"], "policy")
                if "soft_mask" in ckpt:
                    load_state_dict_flexible(soft_mask, ckpt["soft_mask"], "soft_mask")
                policy.eval()
                soft_mask.eval()

                prism_path = resolve_sidecar_artifact(
                    checkpoint_path,
                    cfg["logging"]["output_dir"],
                    "prism_adapter.pt",
                )
                prism = None
                if prism_path is not None:
                    prism = PRISMAdapter(cfg, embed_dim).to(device)
                    prism.load_state_dict(torch.load(prism_path, map_location=device))
                    prism.eval()

                if is_global_rank_zero():
                    print("\n====== AOAE Pareto Sweep ======")
                aoae_results = run_pareto_sweep(
                    base_model,
                    policy,
                    soft_mask,
                    prism,
                    eval_ds,
                    tokenizer,
                    cfg,
                    max_samples,
                    predictions_sink=saved_predictions,
                    prediction_limit=prediction_limit,
                )
                all_results.extend(aoae_results)
            else:
                if is_global_rank_zero():
                    print("\nNo AOAE checkpoint provided — skipping AOAE evaluation.")

        os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
        results_path = os.path.join(cfg["logging"]["output_dir"], "eval_results.json")
        if is_global_rank_zero():
            with open(results_path, "w") as f:
                json.dump([asdict(r) for r in all_results], f, indent=2)
            print(f"\nResults saved to {results_path}")

        predictions_path = None
        if is_global_rank_zero() and prediction_limit > 0:
            predictions_path = _save_prediction_artifact(
                saved_predictions,
                cfg["logging"]["output_dir"],
                prediction_limit,
            )
            print(f"Predictions saved to {predictions_path}")

        metadata = _build_run_metadata(
            cfg=cfg,
            mode=mode,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            max_samples=max_samples,
            results_path=results_path,
            num_results=len(all_results),
            predictions_path=predictions_path,
            saved_predictions=len(saved_predictions),
            prediction_limit=prediction_limit,
        )
        metadata_path = os.path.join(cfg["logging"]["output_dir"], "eval_metadata.json")
        if is_global_rank_zero():
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to {metadata_path}")

        if is_global_rank_zero():
            manifest_path = _append_manifest(metadata, all_results)
            print(f"Manifest updated at {manifest_path}")

            _save_kv_dynamics_artifacts(kv_dynamics_records, cfg["logging"]["output_dir"])
            _save_eval_plots(all_results, cfg["logging"]["output_dir"])

        if is_global_rank_zero():
            print("\n" + "=" * 196)
            print(
                f"{'Method':<25} {'Accuracy':>10} {'TPS':>10} {'NFE':>8} "
                f"{'CacheKeep':>10} {'Agree':>8} {'DraftAcc':>10} {'Reuse':>8} {'ReuseJS':>9} "
                f"{'AccF1':>8} {'SpecF1':>8} {'Note':<40}"
            )
            print("-" * 196)
            for r in all_results:
                print(
                    f"{r.method:<25} {r.accuracy:>10.4f} {r.avg_tokens_per_sec:>10.1f} "
                    f"{r.avg_nfe:>8.0f} {r.cache_hit_rate:>10.4f} "
                    f"{r.agreement_rate:>8.4f} {r.draft_accept_rate:>10.4f} "
                    f"{r.reuse_mean_safe:>8.4f} {r.reuse_mean_js:>9.4f} "
                    f"{r.access_next_h_f1:>8.4f} {r.access_next_h_spec_f1:>8.4f} {r.config_note:<40}"
                )
            print("=" * 196)

        return all_results
    finally:
        if owned_model is not None:
            close_fn = getattr(owned_model, "close", None)
            if callable(close_fn):
                close_fn()


if __name__ == "__main__":
    import argparse
    def _parse_float_list(raw: str) -> List[float]:
        values = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            values.append(float(chunk))
        if not values:
            raise ValueError("Expected at least one float value.")
        return values

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "speculative"],
                        help="'standard' for single-model, 'speculative' for dual-model")
    parser.add_argument("--reuse_signal_method", type=str, default=None,
                        choices=[
                            "argmax_match", "topk_overlap", "min_confidence",
                            "min_margin", "js_divergence", "temporal_confidence",
                        ],
                        help="Override inference.reuse_signal.method.")
    parser.add_argument("--reuse_signal_threshold", type=float, default=None,
                        help="Override inference.reuse_signal.threshold.")
    parser.add_argument("--track_kv_dynamics", action="store_true",
                        help="Enable analysis.track_kv_dynamics.")
    parser.add_argument("--disable_remask", action="store_true",
                        help="Set inference.disable_remask=true.")
    parser.add_argument("--enable_positional_cache", action="store_true",
                        help="Enable inference.positional_cache for next-H access experiments.")
    parser.add_argument("--positional_cache_horizon", type=int, default=None,
                        help="Override inference.positional_cache.horizon.")
    parser.add_argument("--positional_cache_refresh_budget", type=int, default=None,
                        help="Override inference.positional_cache.refresh_budget.")
    parser.add_argument("--policy_temperatures", type=str, default=None,
                        help="Comma-separated tau_pi values for speculative runs.")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip baseline decoding methods.")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Override data.eval_dataset.")
    parser.add_argument("--eval_dataset_config", type=str, default=None,
                        help="Override data.eval_dataset_config (empty string = none).")
    parser.add_argument("--eval_split", type=str, default=None,
                        help="Override data.eval_split.")
    parser.add_argument("--task_type", type=str, default=None, choices=["math", "code"],
                        help="Override evaluation.task_type.")
    parser.add_argument("--code_timeout_sec", type=float, default=None,
                        help="Override evaluation.code.timeout_sec for task_type=code.")
    parser.add_argument("--code_cpu_time_limit_sec", type=int, default=None,
                        help="Override evaluation.code.cpu_time_limit_sec.")
    parser.add_argument("--code_memory_limit_mb", type=int, default=None,
                        help="Override evaluation.code.memory_limit_mb.")
    parser.add_argument("--candidate_policy", type=str, default=None,
                        choices=["learned_topb", "sliding_window", "confidence_topb"],
                        help="Override inference.positional_cache.candidate_policy.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ic = cfg.setdefault("inference", {})
    dc = cfg.setdefault("data", {})
    if args.reuse_signal_method is not None:
        ic.setdefault("reuse_signal", {})["method"] = args.reuse_signal_method
    if args.reuse_signal_threshold is not None:
        ic.setdefault("reuse_signal", {})["threshold"] = float(args.reuse_signal_threshold)
    if args.disable_remask:
        ic["disable_remask"] = True
    if args.track_kv_dynamics:
        cfg.setdefault("analysis", {})["track_kv_dynamics"] = True
    if args.enable_positional_cache:
        ic.setdefault("positional_cache", {})["enabled"] = True
    if args.positional_cache_horizon is not None:
        ic.setdefault("positional_cache", {})["horizon"] = int(args.positional_cache_horizon)
    if args.positional_cache_refresh_budget is not None:
        ic.setdefault("positional_cache", {})["refresh_budget"] = int(args.positional_cache_refresh_budget)
    if args.candidate_policy is not None:
        ic.setdefault("positional_cache", {})["candidate_policy"] = args.candidate_policy
    if args.eval_dataset is not None:
        dc["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        dc["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        dc["eval_split"] = args.eval_split
    if args.task_type is not None:
        cfg.setdefault("evaluation", {})["task_type"] = args.task_type
    if args.code_timeout_sec is not None:
        cfg.setdefault("evaluation", {}).setdefault("code", {})["timeout_sec"] = float(args.code_timeout_sec)
    if args.code_cpu_time_limit_sec is not None:
        cfg.setdefault("evaluation", {}).setdefault("code", {})["cpu_time_limit_sec"] = int(args.code_cpu_time_limit_sec)
    if args.code_memory_limit_mb is not None:
        cfg.setdefault("evaluation", {}).setdefault("code", {})["memory_limit_mb"] = int(args.code_memory_limit_mb)

    policy_temperatures = None
    if args.policy_temperatures is not None:
        policy_temperatures = _parse_float_list(args.policy_temperatures)

    main(
        cfg,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        mode=args.mode,
        config_path=args.config,
        skip_baselines=args.skip_baselines,
        speculative_policy_temperatures=policy_temperatures,
    )
