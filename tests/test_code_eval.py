import pytest
from aoae.code_eval import build_candidate_code, evaluate_code_sample
from aoae.evaluators import CodeEvaluator, build_evaluator, describe_evaluator


def _sample():
    return {
        "prompt": "def add(a, b):\n    \"\"\"Return sum of a and b.\"\"\"",
        "test": "def check(fn):\n    assert fn(1, 2) == 3\n    assert fn(-1, 1) == 0",
        "entry_point": "add",
        "canonical_solution": "def add(a, b):\n    return a + b",
    }


# ---------------------------------------------------------------------------
# build_candidate_code
# ---------------------------------------------------------------------------

def test_build_candidate_full_function():
    """Model output that already contains the function signature is returned as-is."""
    generated = "def add(a, b):\n    return a + b"
    candidate = build_candidate_code(generated, _sample()["prompt"], "add")
    assert "def add(" in candidate


def test_build_candidate_body_only_prepends_prompt():
    """Body-only output (no def) gets the prompt prepended so the signature is present."""
    generated = "    return a + b"
    candidate = build_candidate_code(generated, _sample()["prompt"], "add")
    assert "def add(" in candidate


def test_build_candidate_strips_markdown_fences():
    """Code fences are stripped before the function signature check."""
    generated = "```python\ndef add(a, b):\n    return a + b\n```"
    candidate = build_candidate_code(generated, _sample()["prompt"], "add")
    assert "```" not in candidate
    assert "def add(" in candidate


# ---------------------------------------------------------------------------
# evaluate_code_sample
# ---------------------------------------------------------------------------

def test_code_execution_passes_valid_candidate():
    generated = "```python\ndef add(a, b):\n    return a + b\n```"
    r = evaluate_code_sample(generated, _sample(), timeout_sec=2.0, cpu_time_limit_sec=1, memory_limit_mb=256)
    assert r.passed
    assert r.status == "passed"


def test_code_execution_fails_invalid_candidate():
    generated = "def add(a, b):\n    return a - b"
    r = evaluate_code_sample(generated, _sample(), timeout_sec=2.0, cpu_time_limit_sec=1, memory_limit_mb=256)
    assert not r.passed
    assert r.status in {"failed_tests", "timeout", "executor_error"}


def test_code_execution_timeout():
    """Infinite-loop candidate is caught by the wall-clock timeout."""
    generated = "def add(a, b):\n    while True: pass"
    r = evaluate_code_sample(generated, _sample(), timeout_sec=1.0, cpu_time_limit_sec=1, memory_limit_mb=256)
    assert not r.passed
    assert r.status == "timeout"


def test_code_execution_missing_schema_returns_unsupported():
    """Sample without 'test'/'entry_point' fields returns unsupported_sample_schema."""
    r = evaluate_code_sample("def add(a, b): return a + b", {}, timeout_sec=2.0)
    assert not r.passed
    assert r.status == "unsupported_sample_schema"


def test_code_execution_syntax_error_fails():
    """Syntactically invalid Python does not crash the harness."""
    generated = "def add(a, b):\n    SYNTAX ERROR HERE ((("
    r = evaluate_code_sample(generated, _sample(), timeout_sec=2.0, cpu_time_limit_sec=1, memory_limit_mb=256)
    assert not r.passed
    assert r.status in {"failed_tests", "executor_error"}


# ---------------------------------------------------------------------------
# CodeEvaluator / build_evaluator dispatch
# ---------------------------------------------------------------------------

def test_code_evaluator_uses_execution_when_schema_available():
    cfg = {
        "evaluation": {
            "code": {"timeout_sec": 2.0, "cpu_time_limit_sec": 1, "memory_limit_mb": 256}
        }
    }
    ev = CodeEvaluator(cfg)
    sample = _sample()
    ok = ev.evaluate("def add(a, b):\n    return a + b", sample["canonical_solution"], sample=sample)
    bad = ev.evaluate("def add(a, b):\n    return a - b", sample["canonical_solution"], sample=sample)
    assert ok.correct
    assert not bad.correct
    assert ok.detail.startswith("code_exec:")
    assert bad.detail.startswith("code_exec:")


def test_code_evaluator_falls_back_to_string_match_without_schema():
    """Without test/entry_point in the sample, fall back to normalized string match."""
    cfg = {"evaluation": {"code": {}}}
    ev = CodeEvaluator(cfg)
    decision = ev.evaluate("hello world", "hello world", sample={})
    assert decision.correct
    assert decision.detail == "code_string_match"
    decision_bad = ev.evaluate("hello world", "goodbye", sample={})
    assert not decision_bad.correct


def test_build_evaluator_dispatches_code_task_type():
    """build_evaluator returns CodeEvaluator for task_type=code."""
    cfg = {
        "data": {"eval_dataset": "openai/openai_humaneval"},
        "evaluation": {
            "task_type": "code",
            "code": {"timeout_sec": 10.0, "cpu_time_limit_sec": 5, "memory_limit_mb": 1024},
        },
    }
    ev = build_evaluator(cfg)
    assert isinstance(ev, CodeEvaluator)
    assert ev.timeout_sec == 10.0
    assert ev.cpu_time_limit_sec == 5
    assert ev.memory_limit_mb == 1024


def test_build_evaluator_unknown_task_type_raises():
    cfg = {"data": {}, "evaluation": {"task_type": "unknown_xyz"}}
    with pytest.raises(ValueError, match="task_type"):
        build_evaluator(cfg)


def test_humaneval_config_uses_code_evaluator_and_schema_fields():
    from pathlib import Path

    from aoae.cli import _load_config
    from aoae.tasks import extract_prompt_and_reference

    root = Path(__file__).resolve().parents[1]
    cfg = _load_config(str(root / "configs" / "eval_humaneval.yaml"))
    sample = _sample()
    prompt, reference = extract_prompt_and_reference(sample)

    assert cfg["data"]["eval_dataset"] == "openai/openai_humaneval"
    assert describe_evaluator(cfg) == "code_exec_or_string_match"
    assert isinstance(build_evaluator(cfg), CodeEvaluator)
    assert prompt == sample["prompt"]
    assert reference == sample["canonical_solution"]
