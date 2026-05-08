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
    check_gsm8k_correctness_llada,
    check_math_correctness,
    check_math500_correctness,
    extract_answer,
    extract_gsm8k_llada_answer,
    extract_gsm8k_llada_reference,
    extract_gsm8k_official_answer,
    extract_math500_answer,
    is_gsm8k_dataset,
    is_math500_dataset,
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
    """GSM8K evaluator using flexible answer extraction for masked-diffusion models.

    LLaDA-style masked diffusion models often express answers as "The answer is X"
    or via `\\boxed{}` rather than the strict `#### X` marker required by the
    OpenAI official GSM8K grader.  The LLaDA extractor tries all common formats
    in priority order (official marker → boxed → answer lines → last number),
    recovering the ground-truth comparison without overfitting to a single format.
    The reference is still extracted using the official `#### X` pattern, which is
    always present in the GSM8K dataset's reference strings.
    """

    task_type = "math"

    def __init__(self, cfg: Dict[str, Any]):
        self.eval_dataset = str(cfg.get("data", {}).get("eval_dataset", "") or "")
        self.uses_gsm8k = is_gsm8k_dataset(cfg)
        if self.uses_gsm8k:
            self.evaluator_name = "gsm8k_llada_flexible"
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
        if self.uses_gsm8k:
            pred = extract_gsm8k_llada_answer(generated)
            gold = extract_gsm8k_llada_reference(reference)
            ok = check_gsm8k_correctness_llada(generated, reference)
            return EvalDecision(
                correct=bool(ok),
                detail="gsm8k_llada_flexible",
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


class Math500Evaluator(BaseEvaluator):
    """Evaluator for HuggingFaceH4/MATH-500.

    Uses a layered strategy: latex normalisation → numeric float comparison
    → sympy symbolic equivalence.  The reference is taken directly from the
    dataset's ``answer`` field (already stripped of \\boxed{}); the prediction
    is extracted from the last \\boxed{} in the generated text.
    """

    task_type = "math"
    evaluator_name = "math500_layered"

    def __init__(self, cfg: Dict[str, Any]):
        pass

    def evaluate(self, generated: str, reference: str, sample: Optional[Dict[str, Any]] = None) -> EvalDecision:
        pred = extract_math500_answer(generated)
        ok = check_math500_correctness(generated, reference)
        return EvalDecision(
            correct=bool(ok),
            detail="math500_layered",
            extracted_prediction=pred,
            extracted_reference=(reference or "").strip(),
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

        g = self._normalize(generated)
        r = self._normalize(reference)
        return EvalDecision(correct=(g == r), detail="code_string_match")


def describe_evaluator(cfg: Dict[str, Any]) -> str:
    task_type = str(cfg.get("evaluation", {}).get("task_type", "math")).lower()
    if task_type == "math":
        if is_math500_dataset(cfg):
            return "math500_layered"
        return "gsm8k_llada_flexible" if is_gsm8k_dataset(cfg) else "math_heuristic_fallback"
    if task_type == "code":
        return "code_exec_or_string_match"
    return f"unknown:{task_type}"


def build_evaluator(cfg: Dict[str, Any]) -> BaseEvaluator:
    task_type = str(cfg.get("evaluation", {}).get("task_type", "math")).lower()
    if task_type == "math":
        if is_math500_dataset(cfg):
            return Math500Evaluator(cfg)
        return MathEvaluator(cfg)
    if task_type == "code":
        return CodeEvaluator(cfg)
    raise ValueError(f"Unknown evaluation.task_type={task_type!r}. Choose from: math, code.")
