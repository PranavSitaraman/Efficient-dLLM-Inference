"""
Task evaluators for AOAE benchmarking.

This keeps benchmark correctness logic modular and allows task-specific
extensions (e.g., HumanEval pass@1) without changing inference code paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .code_eval import evaluate_code_sample
from .tasks import check_math_correctness


@dataclass
class EvalDecision:
    correct: bool
    detail: str = ""


class BaseEvaluator:
    task_type: str = "math"

    def evaluate(self, generated: str, reference: str, sample: Optional[Dict[str, Any]] = None) -> EvalDecision:
        raise NotImplementedError


class MathEvaluator(BaseEvaluator):
    task_type = "math"

    def evaluate(self, generated: str, reference: str, sample: Optional[Dict[str, Any]] = None) -> EvalDecision:
        del sample
        ok = check_math_correctness(generated, reference)
        return EvalDecision(correct=bool(ok), detail="math_exact")


class CodeEvaluator(BaseEvaluator):
    """HumanEval-style evaluator with execution-based pass/fail."""

    task_type = "code"

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


def build_evaluator(cfg: Dict[str, Any]) -> BaseEvaluator:
    task_type = str(cfg.get("evaluation", {}).get("task_type", "math")).lower()
    if task_type == "math":
        return MathEvaluator()
    if task_type == "code":
        return CodeEvaluator(cfg)
    raise ValueError(f"Unknown evaluation.task_type={task_type!r}. Choose from: math, code.")
