from aoae.code_eval import evaluate_code_sample
from aoae.evaluators import CodeEvaluator, build_evaluator, describe_evaluator


def _sample():
    return {
        "prompt": "def add(a, b):\n    \"\"\"Return sum of a and b.\"\"\"",
        "test": "def check(fn):\n    assert fn(1, 2) == 3\n    assert fn(-1, 1) == 0",
        "entry_point": "add",
        "canonical_solution": "def add(a, b):\n    return a + b",
    }


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
