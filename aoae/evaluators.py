"""
Task evaluators for AOAE benchmarking.

This keeps benchmark correctness logic modular and allows task-specific
extensions (e.g., HumanEval pass@1) without changing inference code paths.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .code_eval import evaluate_code_sample
from .tasks import (
    check_gsm8k_correctness_official,
    check_math_correctness,
    extract_answer,
    extract_gsm8k_official_answer,
    is_gsm8k_dataset,
)


@dataclass
class EvalDecision:
    correct: bool
    detail: str = ""
    extracted_prediction: Optional[str] = None
    extracted_reference: Optional[str] = None


class BaseEvaluator:
    task_type: str = "math"
    evaluator_name: str = "unknown"

    def evaluate(self, generated: str, reference: str, sample: Optional[Dict[str, Any]] = None) -> EvalDecision:
        raise NotImplementedError


class MathEvaluator(BaseEvaluator):
    task_type = "math"

    def __init__(self, cfg: Dict[str, Any]):
        self.eval_dataset = str(cfg.get("data", {}).get("eval_dataset", "") or "")
        self.uses_gsm8k_official = is_gsm8k_dataset(cfg)
        if self.uses_gsm8k_official:
            self.evaluator_name = "gsm8k_official_openai"
        else:
            self.evaluator_name = "math_heuristic_fallback"
            warnings.warn(
                "Using the repo's ad-hoc math answer extractor as a fallback. "
                "Do not use this path for datasets that define an official evaluator.",
                RuntimeWarning,
                stacklevel=2,
            )

    def evaluate(self, generated: str, reference: str, sample: Optional[Dict[str, Any]] = None) -> EvalDecision:
        del sample
        if self.uses_gsm8k_official:
            pred = extract_gsm8k_official_answer(generated)
            gold = extract_gsm8k_official_answer(reference)
            ok = check_gsm8k_correctness_official(generated, reference)
            return EvalDecision(
                correct=bool(ok),
                detail="gsm8k_official_openai",
                extracted_prediction=pred,
                extracted_reference=gold,
            )

        pred = extract_answer(generated)
        gold = extract_answer(reference)
        ok = check_math_correctness(generated, reference)
        return EvalDecision(
            correct=bool(ok),
            detail="math_heuristic_fallback",
            extracted_prediction=pred,
            extracted_reference=gold,
        )


class CodeEvaluator(BaseEvaluator):
    """HumanEval-style evaluator with execution-based pass/fail."""

    task_type = "code"
    evaluator_name = "code_exec_or_string_match"

    def __init__(self, cfg: Dict[str, Any]):
        ec = cfg.get("evaluation", {}).get("code", {})
        self.timeout_sec = float(ec.get("timeout_sec", 3.0))
        self.cpu_time_limit_sec = int(ec.get("cpu_time_limit_sec", 2))
        self.memory_limit_mb = int(ec.get("memory_limit_mb", 1024))

    @staticmethod
    def _normalize(s: str) -> str:
        # Strip trailing whitespace differences to reduce formatting noise.
        return re.sub(r"[ \t]+$", "", s or "", flags=re.MULTILINE).strip()

    def evaluate(self, generated: str, reference: str, sample: Optional[Dict[str, Any]] = None) -> EvalDecision:
        if isinstance(sample, dict) and ("test" in sample) and ("entry_point" in sample):
            r = evaluate_code_sample(
                generated,
                sample,
                timeout_sec=self.timeout_sec,
                cpu_time_limit_sec=self.cpu_time_limit_sec,
                memory_limit_mb=self.memory_limit_mb,
            )
            return EvalDecision(correct=bool(r.passed), detail=f"code_exec:{r.status}")

        # Deterministic fallback for non-executable code datasets.
        g = self._normalize(generated)
        r = self._normalize(reference)
        return EvalDecision(correct=(g == r), detail="code_string_match")


def describe_evaluator(cfg: Dict[str, Any]) -> str:
    task_type = str(cfg.get("evaluation", {}).get("task_type", "math")).lower()
    if task_type == "math":
        return "gsm8k_official_openai" if is_gsm8k_dataset(cfg) else "math_heuristic_fallback"
    if task_type == "code":
        return "code_exec_or_string_match"
    return f"unknown:{task_type}"


def build_evaluator(cfg: Dict[str, Any]) -> BaseEvaluator:
    task_type = str(cfg.get("evaluation", {}).get("task_type", "math")).lower()
    if task_type == "math":
        return MathEvaluator(cfg)
    if task_type == "code":
        return CodeEvaluator(cfg)
    raise ValueError(f"Unknown evaluation.task_type={task_type!r}. Choose from: math, code.")
