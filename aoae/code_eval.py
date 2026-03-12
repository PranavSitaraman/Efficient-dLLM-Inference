"""
Execution-based code evaluation utilities.

Provides a lightweight HumanEval-style pass@1 checker:
  - composes candidate code from model output + prompt context
  - executes dataset-provided tests in an isolated subprocess
  - applies timeout and resource limits for safety/stability
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class CodeExecutionResult:
    passed: bool
    status: str
    runtime_sec: float
    stdout: str
    stderr: str


def _extract_code_block(text: str) -> str:
    matches = _CODE_FENCE_RE.findall(text or "")
    if matches:
        return matches[0].strip()
    return (text or "").strip()


def build_candidate_code(generated: str, prompt: str, entry_point: str) -> str:
    """Normalize model output into executable candidate code."""
    candidate = _extract_code_block(generated)
    prompt = (prompt or "").rstrip()
    entry_sig = f"def {entry_point}("

    if entry_sig in candidate:
        return candidate
    if prompt and entry_sig in prompt:
        if candidate.startswith(prompt):
            return candidate
        return f"{prompt}\n{candidate}".strip()
    return candidate


def _build_exec_script(
    candidate_code: str,
    test_code: str,
    entry_point: str,
    cpu_time_limit_sec: int,
    memory_limit_mb: int,
) -> str:
    # Keep script plain and explicit; this runs in a subprocess with timeout.
    return "\n".join(
        [
            "import resource",
            f"resource.setrlimit(resource.RLIMIT_CPU, ({int(cpu_time_limit_sec)}, {int(cpu_time_limit_sec)}))",
            f"mem_bytes = {int(memory_limit_mb)} * 1024 * 1024",
            "resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))",
            "resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))",
            "",
            candidate_code.strip(),
            "",
            test_code.strip(),
            "",
            f"check(globals()[{entry_point}])",
            'print("AOAE_PASS")',
            "",
        ]
    )


def evaluate_code_sample(
    generated: str,
    sample: Dict[str, Any],
    *,
    timeout_sec: float = 3.0,
    cpu_time_limit_sec: int = 2,
    memory_limit_mb: int = 1024,
) -> CodeExecutionResult:
    """Run execution-based pass/fail for a HumanEval-style sample."""
    prompt = str(sample.get("prompt", ""))
    test_code = sample.get("test")
    entry_point = sample.get("entry_point")
    if not isinstance(test_code, str) or not isinstance(entry_point, str):
        return CodeExecutionResult(
            passed=False,
            status="unsupported_sample_schema",
            runtime_sec=0.0,
            stdout="",
            stderr="missing 'test' or 'entry_point'",
        )

    candidate_code = build_candidate_code(generated, prompt, entry_point)
    script = _build_exec_script(
        candidate_code=candidate_code,
        test_code=test_code,
        entry_point=repr(entry_point),
        cpu_time_limit_sec=cpu_time_limit_sec,
        memory_limit_mb=memory_limit_mb,
    )

    env = {
        "PYTHONHASHSEED": "0",
    }

    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="aoae_code_eval_") as tmpdir:
            path = Path(tmpdir) / "eval_candidate.py"
            path.write_text(script)
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
                cwd=tmpdir,
                env=env,
            )
    except subprocess.TimeoutExpired as exc:
        return CodeExecutionResult(
            passed=False,
            status="timeout",
            runtime_sec=time.perf_counter() - t0,
            stdout=(exc.stdout or ""),
            stderr=(exc.stderr or ""),
        )
    except Exception as exc:
        return CodeExecutionResult(
            passed=False,
            status="executor_error",
            runtime_sec=time.perf_counter() - t0,
            stdout="",
            stderr=str(exc),
        )

    elapsed = time.perf_counter() - t0
    passed = (proc.returncode == 0) and ("AOAE_PASS" in proc.stdout)
    return CodeExecutionResult(
        passed=passed,
        status="passed" if passed else "failed_tests",
        runtime_sec=elapsed,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
