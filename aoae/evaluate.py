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
import yaml
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datasets import load_dataset
from typing import Optional, List

from .models.base_model import LLaDABaseModel
from .models.soft_mask import SoftMaskedState
from .models.policy import AOAEPolicy, DefaultPolicy
from .models.prism import PRISMAdapter
from .inference import aoae_inference, uniform_decode, confidence_threshold_decode, block_smode_decode
from .dinfer_integration import run_policy_guided_inference, CacheStats
from .train_grpo import check_math_correctness, extract_answer


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
    cache_hit_rate: float = 0.0   # dInfer cache hit rate (0 if not tracked)
    cache_commits: int = 0        # total cache commits across all samples


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

    n_eval = min(len(dataset), max_samples) if max_samples else len(dataset)

    for i in tqdm(range(n_eval), desc=f"AOAE (tau_pi={policy_temperature})"):
        sample = dataset[i]
        question = sample.get("question", sample.get("problem", ""))
        reference = sample.get("answer", "")
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

        if check_math_correctness(gen_text, reference):
            correct += 1
        total += 1

        elapsed = t1 - t0
        total_time += elapsed
        total_nfe += T

    accuracy = correct / max(total, 1)
    avg_time = total_time / max(total, 1)
    avg_nfe = total_nfe / max(total, 1)
    avg_tps = total_gen_tokens / max(total_time, 1e-6)

    return EvalResult(
        method="AOAE",
        accuracy=accuracy,
        total_samples=total,
        correct_samples=correct,
        avg_nfe=avg_nfe,
        avg_tokens_per_sec=avg_tps,
        avg_gen_time_sec=avg_time,
        config_note=f"tau_pi={policy_temperature}",
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
) -> EvalResult:
    """Evaluate a baseline decoder on a dataset."""
    mask_id = cfg["base_model"]["mask_token_id"]
    device = base_model.device
    T = cfg["inference"]["steps"]

    correct = 0
    total = 0
    total_time = 0.0
    total_gen_tokens = 0

    n_eval = min(len(dataset), max_samples) if max_samples else len(dataset)

    # Warm-up forward pass
    warmup_ids = torch.full((1, 32), mask_id, dtype=torch.long, device=device)
    with torch.no_grad():
        base_model.forward(warmup_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Debug: show prompt format for first sample
    if n_eval > 0:
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

    for i in tqdm(range(n_eval), desc=f"Baseline ({method})"):
        sample = dataset[i]
        question = sample.get("question", sample.get("problem", ""))
        reference = sample.get("answer", "")
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

        is_correct = check_math_correctness(gen_text, reference)
        if is_correct:
            correct += 1
        total += 1
        total_time += (t1 - t0)

        # Debug output for first 3 samples
        if i < 3:
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
) -> EvalResult:
    """Evaluate speculative diffusion (dual-model) on a dataset."""
    from .speculative_inference import speculative_inference

    mask_id = cfg["base_model"]["mask_token_id"]
    device = dual_model.device
    T = cfg["inference"]["steps"]

    correct = 0
    total = 0
    total_time = 0.0
    total_gen_tokens = 0
    total_nfe = 0
    total_agreement = 0.0
    total_cache_hits = 0
    total_cache_misses = 0

    n_eval = min(len(dataset), max_samples) if max_samples else len(dataset)
    tau_r = cfg["base_model"].get("routing_temperature", 0.01)

    # Warm-up forward pass (exclude from timing)
    warmup_ids = torch.full((1, 32), mask_id, dtype=torch.long, device=device)
    with torch.no_grad():
        dual_model.auxiliary_forward(warmup_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for i in tqdm(range(n_eval), desc=f"Speculative (tau_r={tau_r})"):
        sample = dataset[i]
        question = sample.get("question", sample.get("problem", ""))
        reference = sample.get("answer", "")
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
            output_ids, trajectory = speculative_inference(
                dual_model=dual_model,
                policy=policy,
                soft_mask_module=soft_mask,
                prism_adapter=prism,
                prompt_ids=prompt_ids,
                cfg=cfg,
                record_trajectory=False,  # enables fallback, saves memory
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

        if check_math_correctness(gen_text, reference):
            correct += 1
        total += 1

        elapsed = t1 - t0
        total_time += elapsed
        total_nfe += T * 2  # 2 model forward passes per step

    accuracy = correct / max(total, 1)
    avg_time = total_time / max(total, 1)
    avg_nfe = total_nfe / max(total, 1)
    # TPS = actual generated tokens / wall time
    avg_tps = total_gen_tokens / max(total_time, 1e-6)

    return EvalResult(
        method="Speculative-AOAE",
        accuracy=accuracy,
        total_samples=total,
        correct_samples=correct,
        avg_nfe=avg_nfe,
        avg_tokens_per_sec=avg_tps,
        avg_gen_time_sec=avg_time,
        config_note=f"tau_r={tau_r},tau_pi={policy_temperature}",
        cache_hit_rate=0.0,
        cache_commits=0,
    )


def run_pareto_sweep(
    base_model, policy, soft_mask, prism, dataset, tokenizer, cfg, max_samples,
) -> List[EvalResult]:
    """Sweep policy temperature to generate Pareto curve points."""
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    results = []

    for tau_pi in temperatures:
        print(f"\n--- AOAE with tau_pi = {tau_pi} ---")
        r = evaluate_aoae(
            base_model, policy, soft_mask, prism, dataset, tokenizer, cfg,
            policy_temperature=tau_pi, max_samples=max_samples,
        )
        results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

    return results


def main(
    cfg: dict,
    checkpoint_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    mode: str = "standard",
):
    """Run full evaluation suite.

    Args:
        cfg: config dict.
        checkpoint_path: path to policy checkpoint.
        max_samples: max samples to evaluate.
        mode: 'standard' for single-model, 'speculative' for dual-model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load eval dataset ---
    dc = cfg["data"]
    print(f"Loading eval dataset: {dc['eval_dataset']}...")
    eval_ds = load_dataset(dc["eval_dataset"], "main", split=dc["eval_split"])
    if max_samples is None:
        max_samples = dc.get("eval_max_samples")

    all_results: List[EvalResult] = []

    if mode == "speculative" or cfg["base_model"].get("backend") == "dual":
        # ====== Speculative Dual-Model Evaluation ======
        from .models.dual_model import DualModelWrapper

        print("Loading dual model (hard auxiliary + soft primary)...")
        dual_model = DualModelWrapper(cfg)
        dual_model = dual_model.to(device)
        tokenizer = dual_model.tokenizer
        mask_id = cfg["base_model"]["mask_token_id"]

        embed_w = dual_model.get_embedding_weight()
        embed_dim = embed_w.shape[1]

        # --- Baselines using hard-routed model (standard LLaDA decoding) ---
        # Access the underlying model in hard-routing mode for baselines
        base_model = dual_model._model

        # Block S-Mode: paper's key TPS technique (block-wise semi-AR decoding)
        print("\n====== Baseline: Block S-Mode (tau=0.7, block=32) ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg,
                              "block_smode", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

        # Block S-Mode + MBE: with Multiple Block Editing
        print("\n====== Baseline: Block S-Mode + MBE ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg,
                              "block_smode_mbe", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

        # S-Mode: paper's speed baseline (tau_mask=0.7, aggressive)
        print("\n====== Baseline: S-Mode (tau=0.7, speed) ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg,
                              "confidence_s_mode", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

        # Q-Mode: paper's quality baseline (tau_mask=0.95, conservative)
        print("\n====== Baseline: Q-Mode (tau=0.95, quality) ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg,
                              "confidence_q_mode", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

        # --- AOAE Speculative Evaluation ---
        # Auto-detect policy checkpoint from output_dir if not explicitly provided
        if not checkpoint_path:
            for name in ("policy_best.pt", "policy_final.pt"):
                candidate = os.path.join(cfg["logging"]["output_dir"], name)
                if os.path.exists(candidate):
                    checkpoint_path = candidate
                    break
            if not checkpoint_path:
                import glob as _glob
                step_ckpts = sorted(_glob.glob(
                    os.path.join(cfg["logging"]["output_dir"], "policy_step*.pt")
                ))
                if step_ckpts:
                    checkpoint_path = step_ckpts[-1]

        has_trained_policy = checkpoint_path and os.path.exists(checkpoint_path)

        # Setup soft_mask (needed for both trained and default policy)
        soft_mask = SoftMaskedState(cfg, embed_w).to(device)
        soft_mask.set_mask_embedding(mask_id)
        soft_mask.eval()

        if has_trained_policy:
            policy = AOAEPolicy(cfg, input_dim=embed_dim).to(device)
            print(f"\nLoading trained policy from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            policy.load_state_dict(ckpt["policy"])
            if "soft_mask" in ckpt:
                soft_mask.load_state_dict(ckpt["soft_mask"])
            policy.eval()

            # PRISM adapter
            prism_path = os.path.join(cfg["logging"]["output_dir"], "prism_adapter.pt")
            prism = None
            if os.path.exists(prism_path):
                prism = PRISMAdapter(cfg, embed_dim).to(device)
                prism.load_state_dict(torch.load(prism_path, map_location=device))
                prism.eval()
                print(f"  Loaded PRISM adapter from {prism_path}")

            tau_r = cfg["base_model"].get("routing_temperature", 0.01)
            print(f"\n====== Speculative AOAE — Trained Policy (tau_r={tau_r}) ======")
            for tau_pi in [0.5, 1.0, 1.5]:
                print(f"\n--- tau_pi = {tau_pi} ---")
                r = evaluate_speculative(
                    dual_model, policy, soft_mask, prism, eval_ds, tokenizer, cfg,
                    policy_temperature=tau_pi, max_samples=max_samples,
                )
                all_results.append(r)
                print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")
        else:
            # No trained policy — use DefaultPolicy (hard auxiliary as heuristic)
            tau_r = cfg["base_model"].get("routing_temperature", 0.01)
            print(f"\n====== Speculative AOAE — Default Policy (tau_r={tau_r}) ======")
            print("  (No trained checkpoint found; using hard auxiliary as heuristic policy)")
            policy = DefaultPolicy(tau_mask=0.7).to(device)
            policy.eval()

            r = evaluate_speculative(
                dual_model, policy, soft_mask, None, eval_ds, tokenizer, cfg,
                policy_temperature=1.0, max_samples=max_samples,
            )
            all_results.append(r)
            print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

    else:
        # ====== Standard Single-Model Evaluation ======
        print("Loading base model...")
        base_model = LLaDABaseModel(cfg)
        base_model = base_model.to(device)
        tokenizer = base_model.tokenizer
        mask_id = cfg["base_model"]["mask_token_id"]

        # --- Baselines ---
        print("\n====== Baseline: Block S-Mode (tau=0.7, block=32) ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg, "block_smode", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

        print("\n====== Baseline: Block S-Mode + MBE ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg, "block_smode_mbe", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}  TPS: {r.avg_tokens_per_sec:.1f}")

        print("\n====== Baseline: S-Mode (aggressive) ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg, "confidence_s_mode", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}")

        print("\n====== Baseline: Q-Mode (conservative) ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg, "confidence_q_mode", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}")

        print("\n====== Baseline: Fast-dLLM (Wu et al. 2025) ======")
        r = evaluate_baseline(base_model, eval_ds, tokenizer, cfg, "fast_dllm", max_samples=max_samples)
        all_results.append(r)
        print(f"  Accuracy: {r.accuracy:.4f}")

        # --- AOAE ---
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nLoading AOAE policy from {checkpoint_path}")
            embed_w = base_model.get_embedding_weight()
            embed_dim = embed_w.shape[1]

            soft_mask = SoftMaskedState(cfg, embed_w).to(device)
            soft_mask.set_mask_embedding(mask_id)
            policy = AOAEPolicy(cfg, input_dim=embed_dim).to(device)

            ckpt = torch.load(checkpoint_path, map_location=device)
            policy.load_state_dict(ckpt["policy"])
            if "soft_mask" in ckpt:
                soft_mask.load_state_dict(ckpt["soft_mask"])
            policy.eval()
            soft_mask.eval()

            # PRISM adapter
            prism_path = os.path.join(cfg["logging"]["output_dir"], "prism_adapter.pt")
            prism = None
            if os.path.exists(prism_path):
                prism = PRISMAdapter(cfg, embed_dim).to(device)
                prism.load_state_dict(torch.load(prism_path, map_location=device))
                prism.eval()

            print("\n====== AOAE Pareto Sweep ======")
            aoae_results = run_pareto_sweep(
                base_model, policy, soft_mask, prism, eval_ds, tokenizer, cfg, max_samples,
            )
            all_results.extend(aoae_results)
        else:
            print("\nNo AOAE checkpoint provided — skipping AOAE evaluation.")

    # --- Save results ---
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
    results_path = os.path.join(cfg["logging"]["output_dir"], "eval_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Print summary table ---
    print("\n" + "=" * 90)
    print(f"{'Method':<25} {'Accuracy':>10} {'TPS':>10} {'NFE':>8} {'CacheHit':>10} {'Note':<20}")
    print("-" * 90)
    for r in all_results:
        print(f"{r.method:<25} {r.accuracy:>10.4f} {r.avg_tokens_per_sec:>10.1f} "
              f"{r.avg_nfe:>8.0f} {r.cache_hit_rate:>10.4f} {r.config_note:<20}")
    print("=" * 90)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "speculative"],
                        help="'standard' for single-model, 'speculative' for dual-model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.checkpoint, args.max_samples, args.mode)
