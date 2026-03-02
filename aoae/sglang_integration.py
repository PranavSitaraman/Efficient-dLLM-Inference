"""
SGLang Integration for Speculative Diffusion Inference.

Wraps the AOAE speculative diffusion loop into SGLang's serving framework
for real-world throughput benchmarking. Provides:

  1. SpeculativeDiffusionEngine: SGLang-compatible engine that runs the
     dual-model loop (hard auxiliary + soft primary) with KV cache sharing.
  2. Benchmark utilities for measuring tokens/sec and cache hit rates.

Requirements:
    pip install sglang  (or clone from https://github.com/sgl-project/sglang)
"""

import time
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


def _load_state_dict_flexible(module, state_dict: Dict[str, torch.Tensor], label: str) -> None:
    own = module.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if (k in own) and (own[k].shape == v.shape)
    }
    skipped = [
        k for k, v in state_dict.items()
        if (k not in own) or (k in own and own[k].shape != v.shape)
    ]
    module.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"  [Checkpoint] {label}: skipped {len(skipped)} incompatible keys.")


@dataclass
class BenchmarkResult:
    """Results from a throughput benchmark run."""
    total_tokens: int = 0
    wall_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    nfes: int = 0               # network function evaluations
    cache_hit_rate: float = 0.0
    agreement_rate: float = 0.0
    accuracy: float = 0.0
    tau_r: float = 0.0
    config_name: str = ""


class SpeculativeDiffusionEngine:
    """SGLang-compatible engine for speculative diffusion inference.

    This engine manages the dual-model setup:
      - Hard-routed auxiliary for fast draft predictions
      - Soft-routed primary for verification and refinement
      - KV cache sharing between auxiliary and primary

    For SGLang integration, this engine exposes:
      - generate(): single-request generation
      - batch_generate(): batched generation for throughput measurement
      - benchmark(): full throughput/accuracy benchmark
    """

    def __init__(
        self,
        dual_model,
        policy,
        soft_mask_module,
        prism_adapter,
        cfg: dict,
    ):
        self.dual_model = dual_model
        self.policy = policy
        self.soft_mask_module = soft_mask_module
        self.prism_adapter = prism_adapter
        self.cfg = cfg
        self.tokenizer = dual_model.tokenizer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        policy_temperature: float = 1.0,
    ) -> Dict[str, object]:
        """Generate a response for a single prompt.

        Args:
            prompt: input text.
            max_tokens: maximum response tokens.
            policy_temperature: tau_pi for the steering policy.

        Returns:
            dict with 'text', 'tokens', 'agreement_rate', 'cache_hit_rate', 'time_s'.
        """
        from .speculative_inference import speculative_inference

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text, return_tensors="pt", truncation=True,
            add_special_tokens=False,
            max_length=self.cfg["data"]["max_prompt_len"],
        ).to(self.dual_model.device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        cfg_gen = dict(self.cfg)
        cfg_gen["inference"] = dict(cfg_gen["inference"])
        cfg_gen["inference"]["gen_length"] = max_tokens

        start = time.perf_counter()
        output_ids, trajectory = speculative_inference(
            dual_model=self.dual_model,
            policy=self.policy,
            soft_mask_module=self.soft_mask_module,
            prism_adapter=self.prism_adapter,
            prompt_ids=prompt_ids,
            cfg=cfg_gen,
            record_trajectory=False,  # enables fallback, saves memory
            policy_temperature=policy_temperature,
        )
        elapsed = time.perf_counter() - start

        resp_ids = output_ids[0, prompt_ids.shape[1]:]
        mask_id = self.cfg["base_model"]["mask_token_id"]
        resp_ids = resp_ids[resp_ids != mask_id]
        text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)

        return {
            "text": text,
            "tokens": resp_ids.cpu().tolist(),
            "agreement_rate": trajectory.mean_agreement_rate if trajectory else 0.0,
            "cache_hits": trajectory.total_cache_hits if trajectory else 0,
            "cache_misses": trajectory.total_cache_misses if trajectory else 0,
            "time_s": elapsed,
        }

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        policy_temperature: float = 1.0,
    ) -> Tuple[List[str], BenchmarkResult]:
        """Generate responses for a batch of prompts.

        Returns:
            (texts, benchmark_result)
        """
        from .speculative_inference import speculative_inference

        results = []
        total_tokens = 0
        total_agreement = 0.0
        total_cache_hits = 0
        total_cache_misses = 0

        start = time.perf_counter()

        for prompt in prompts:
            out = self.generate(prompt, max_tokens, policy_temperature)
            results.append(out["text"])
            total_tokens += len(out["tokens"])
            total_agreement += out["agreement_rate"]
            total_cache_hits += out["cache_hits"]
            total_cache_misses += out["cache_misses"]

        elapsed = time.perf_counter() - start

        total_cache_ops = total_cache_hits + total_cache_misses
        bench = BenchmarkResult(
            total_tokens=total_tokens,
            wall_time_s=elapsed,
            tokens_per_sec=total_tokens / max(elapsed, 1e-6),
            nfes=len(prompts) * self.cfg["inference"]["steps"] * 2,  # 2 models per step
            cache_hit_rate=total_cache_hits / max(total_cache_ops, 1),
            agreement_rate=total_agreement / max(len(prompts), 1),
            tau_r=self.cfg["base_model"].get("routing_temperature", 0.01),
            config_name=self.cfg["logging"].get("run_name", "unknown"),
        )

        return results, bench

    def benchmark(
        self,
        dataset,
        extract_fn,
        max_samples: int = 100,
        max_tokens: int = 256,
        policy_temperature: float = 1.0,
    ) -> BenchmarkResult:
        """Run a full benchmark on a dataset.

        Args:
            dataset: HuggingFace dataset.
            extract_fn: function(sample) -> (prompt, reference_answer).
            max_samples: number of samples to evaluate.
            max_tokens: max generation length.
            policy_temperature: tau_pi.

        Returns:
            BenchmarkResult with accuracy and throughput metrics.
        """
        prompts = []
        references = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            prompt, ref = extract_fn(sample)
            if prompt and ref:
                prompts.append(prompt)
                references.append(ref)

        texts, bench = self.batch_generate(prompts, max_tokens, policy_temperature)

        # Compute accuracy (exact match after normalization)
        correct = 0
        for text, ref in zip(texts, references):
            if _normalize_answer(ref) in _normalize_answer(text):
                correct += 1
        bench.accuracy = correct / max(len(references), 1)

        return bench


def _normalize_answer(s: str) -> str:
    """Normalize answer string for exact match comparison."""
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def run_tau_r_sweep(
    model_name: str,
    tau_r_values: List[float],
    dataset,
    extract_fn,
    cfg_template: dict,
    max_samples: int = 100,
) -> List[BenchmarkResult]:
    """Run a complete τ_r sweep benchmark.

    Creates dual models at each τ_r value and benchmarks throughput + accuracy.

    Args:
        model_name: HuggingFace model name (e.g., 'inclusionAI/LLaDA2.1-mini').
        tau_r_values: list of routing temperatures to sweep.
        dataset: evaluation dataset.
        extract_fn: function(sample) -> (prompt, reference).
        cfg_template: base config dict (τ_r will be overridden).
        max_samples: samples per τ_r value.

    Returns:
        List of BenchmarkResult, one per τ_r value.
    """
    from .models.dual_model import DualModelWrapper
    from .models.soft_mask import SoftMaskedState
    from .models.policy import AOAEPolicy, DefaultPolicy
    from .models.prism import PRISMAdapter
    import os

    results = []

    for tau_r in tau_r_values:
        print(f"\n=== τ_r = {tau_r} ===")

        # Override τ_r in config
        cfg = _deep_copy_cfg(cfg_template)
        cfg["base_model"]["routing_temperature"] = tau_r
        cfg["logging"]["run_name"] = f"sweep_tau{tau_r}"

        # Build dual model
        dual_model = DualModelWrapper(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dual_model = dual_model.to(device)

        # Build policy and modules
        embed_w = dual_model.get_embedding_weight()
        embed_dim = embed_w.shape[1]

        soft_mask = SoftMaskedState(cfg, embed_w).to(device)
        soft_mask.set_mask_embedding(cfg["base_model"]["mask_token_id"])

        # Load trained policy if available, otherwise use DefaultPolicy
        policy_path = os.path.join(cfg["logging"]["output_dir"], "policy_best.pt")
        if os.path.exists(policy_path):
            policy = AOAEPolicy(cfg, input_dim=embed_dim).to(device)
            state = torch.load(policy_path, map_location=device)
            if isinstance(state, dict) and "policy" in state:
                _load_state_dict_flexible(policy, state["policy"], "policy")
            elif isinstance(state, dict):
                _load_state_dict_flexible(policy, state, "policy")
            else:
                raise RuntimeError(f"Unexpected checkpoint format at {policy_path}")
            policy.eval()
            print(f"  Loaded policy from {policy_path}")
        else:
            policy = DefaultPolicy(tau_mask=0.7).to(device)
            policy.eval()
            print("  No trained policy found; using DefaultPolicy (heuristic).")

        # Load PRISM adapter if available
        prism_path = os.path.join(cfg["logging"]["output_dir"], "prism_adapter.pt")
        prism = None
        if os.path.exists(prism_path):
            prism = PRISMAdapter(cfg, embed_dim).to(device)
            prism.load_state_dict(torch.load(prism_path, map_location=device))
            prism.eval()
            print(f"  Loaded PRISM from {prism_path}")

        # Build engine and benchmark
        engine = SpeculativeDiffusionEngine(
            dual_model=dual_model,
            policy=policy,
            soft_mask_module=soft_mask,
            prism_adapter=prism,
            cfg=cfg,
        )

        bench = engine.benchmark(
            dataset=dataset,
            extract_fn=extract_fn,
            max_samples=max_samples,
        )
        bench.tau_r = tau_r
        results.append(bench)

        print(f"  Accuracy:       {bench.accuracy:.4f}")
        print(f"  Tokens/sec:     {bench.tokens_per_sec:.1f}")
        print(f"  Cache hit rate: {bench.cache_hit_rate:.4f}")
        print(f"  Agreement:      {bench.agreement_rate:.4f}")

        # Free memory
        del dual_model, policy, soft_mask, prism, engine
        torch.cuda.empty_cache()

    return results


def _deep_copy_cfg(cfg: dict) -> dict:
    """Deep copy config dict."""
    import copy
    return copy.deepcopy(cfg)
