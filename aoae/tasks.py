"""Shared task parsing and correctness helpers for training and evaluation."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple


_NUMERIC_RE = re.compile(
    r"[-+]?(?:\d+\s*/\s*\d+|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def _normalize_answer_text(text: str) -> str:
    value = (text or "").strip()
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = re.sub(r"\s*/\s*", "/", value)
    value = value.strip()
    value = value.strip("[](){}")
    value = value.rstrip(".,;:!?")
    return value.strip()


def _extract_numeric_candidate(text: str) -> Optional[str]:
    matches = _NUMERIC_RE.findall(text or "")
    if not matches:
        return None
    return _normalize_answer_text(matches[-1])


def _parse_numeric_answer(value: str) -> Optional[float]:
    normalized = _normalize_answer_text(value)
    if not normalized:
        return None
    if re.fullmatch(r"[-+]?\d+/\d+", normalized):
        try:
            return float(Fraction(normalized))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return None


def extract_answer(text: str) -> Optional[str]:
    """Extract a final scalar-style answer from a model response."""
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if match:
        candidate = match[-1].strip()
        return _extract_numeric_candidate(candidate) or _normalize_answer_text(candidate)

    match = re.findall(r"####\s*(.+)", text)
    if match:
        candidate = match[-1].strip()
        return _extract_numeric_candidate(candidate) or _normalize_answer_text(candidate)

    for pattern in (
        r"(?im)(?:final answer|final|answer|result)\s*[:=]\s*([^\n]+)",
        r"(?im)(?:the answer is|therefore,? the answer is|so the answer is)\s+([^\n]+)",
    ):
        match = re.findall(pattern, text)
        if match:
            candidate = match[-1].strip()
            return _extract_numeric_candidate(candidate) or _normalize_answer_text(candidate)

    candidate = _extract_numeric_candidate(text)
    if candidate is not None:
        return candidate

    return None


def check_math_correctness(generated: str, reference: str) -> bool:
    """Compare generated and reference answers using the repo's math heuristic."""
    gen_answer = extract_answer(generated)
    ref_answer = extract_answer(reference)

    if gen_answer is None or ref_answer is None:
        return False

    gen_numeric = _parse_numeric_answer(gen_answer)
    ref_numeric = _parse_numeric_answer(ref_answer)
    if gen_numeric is not None and ref_numeric is not None:
        return abs(gen_numeric - ref_numeric) < 1e-3
    return _normalize_answer_text(gen_answer) == _normalize_answer_text(ref_answer)


def extract_prompt_and_reference(sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract a prompt/reference pair from heterogeneous benchmark schemas."""

    def _collect_text(value: Any) -> List[str]:
        out: List[str] = []
        if isinstance(value, str):
            s = value.strip()
            if s:
                out.append(s)
            return out
        if isinstance(value, dict):
            for key in ("content", "text", "value"):
                if key in value:
                    out.extend(_collect_text(value[key]))
            for key, inner in value.items():
                if key not in ("content", "text", "value"):
                    out.extend(_collect_text(inner))
            return out
        if isinstance(value, list):
            for item in value:
                out.extend(_collect_text(item))
            return out
        return out

    prompt: Optional[str] = None
    reference: Optional[str] = None

    for key in ("question", "problem", "prompt", "instruction", "input"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break

    for key in (
        "answer",
        "solution",
        "canonical_solution",
        "output",
        "response",
        "completion",
        "final_answer",
    ):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            reference = value.strip()
            break

    messages = sample.get("messages") or sample.get("conversations")
    if isinstance(messages, list):
        user_chunks: List[str] = []
        assistant_chunks: List[str] = []
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            content_chunks = _collect_text(turn.get("content"))
            if not content_chunks:
                continue
            content = "\n".join(content_chunks)
            if role == "user":
                user_chunks.append(content)
            elif role == "assistant":
                assistant_chunks.append(content)
        if prompt is None and user_chunks:
            prompt = "\n".join(user_chunks)
        if reference is None and assistant_chunks:
            reference = "\n".join(assistant_chunks)

    if prompt is None or reference is None:
        field_texts: List[Tuple[str, str]] = []
        for key, value in sample.items():
            chunks = _collect_text(value)
            if chunks:
                merged = "\n".join(chunks).strip()
                if merged:
                    field_texts.append((key, merged))

        preferred = [
            item
            for item in field_texts
            if item[0].lower() not in {"id", "source", "split", "dataset", "metadata"}
        ]
        pool = preferred if preferred else field_texts
        pool.sort(key=lambda item: len(item[1]), reverse=True)

        if pool:
            if prompt is None:
                prompt = pool[0][1]
            if reference is None:
                reference = pool[1][1] if len(pool) > 1 else pool[0][1]

    return prompt, reference
