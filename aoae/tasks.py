"""Shared task parsing and correctness helpers for training and evaluation."""

from __future__ import annotations

import re
from fractions import Fraction
from numbers import Integral
from typing import Any, Dict, Iterable, List, Optional, Tuple


_NUMERIC_RE = re.compile(
    r"[-+]?(?:\d+\s*/\s*\d+|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
_GSM8K_OFFICIAL_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_GSM8K_HASH_LINE_RE = re.compile(r"(?im)^####\s*(.+)$")
_GSM8K_ANSWER_LINE_RES = (
    re.compile(r"(?im)(?:final answer|final|answer|result)\s*[:=]\s*([^\n]+)"),
    re.compile(r"(?im)(?:the answer is|therefore,? the answer is|so the answer is)\s+([^\n]+)"),
)
_GSM8K_STANDALONE_ANSWER_HEADING_RE = re.compile(
    r"(?im)^\s*(?:[#>*-]+\s*)*(?:final answer|final|answer|result)\s*[:=]\s*$"
)
_GSM8K_INVALID_ANS = "[invalid]"


_TRUE_VALUES = {"1", "true", "yes", "on", "force"}
_FALSE_VALUES = {"0", "false", "no", "off", "none"}
_RAW_STYLE_VALUES = {"raw", "plain", "none", "off", "false", "0"}
_GSM8K_STYLE_VALUES = {"gsm8k", "cot", "math_cot", "1", "true", "on"}

_GSM8K_PROMPT_SUFFIX = (
    "Solve this carefully. Show short reasoning, then end with one final line:\n"
    "#### <answer>"
)


def _to_token_list(token_ids: Any) -> List[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, list):
        return [int(x) for x in token_ids]
    if isinstance(token_ids, tuple):
        return [int(x) for x in token_ids]
    if hasattr(token_ids, "tolist"):
        values = token_ids.tolist()
        if isinstance(values, list):
            if values and isinstance(values[0], list):
                values = values[0]
            return [int(x) for x in values]
    if isinstance(token_ids, Iterable):
        return [int(x) for x in token_ids]
    return [int(token_ids)]


def _coerce_token_id(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if hasattr(value, "item"):
        try:
            item = value.item()
        except Exception:
            return None
        if isinstance(item, bool):
            return None
        if isinstance(item, Integral):
            return int(item)
    return None


def decode_generated_tokens(
    tokenizer,
    token_ids: Any,
    *,
    mask_token_id: Optional[int] = None,
) -> str:
    """Decode generated tokens with EOS truncation and optional mask stripping."""
    summary = summarize_generated_tokens(
        tokenizer,
        token_ids,
        mask_token_id=mask_token_id,
    )
    return summary["decoded_text"]


def summarize_generated_tokens(
    tokenizer,
    token_ids: Any,
    *,
    mask_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize generated tokens after mask stripping and EOS truncation."""
    ids = _to_token_list(token_ids)
    if not ids:
        return {
            "decoded_text": "",
            "visible_token_count": 0,
            "raw_non_mask_token_count": 0,
            "mask_tokens_remaining": 0,
            "has_eos": False,
            "raw_sequence_length": 0,
        }

    eos_token_id = _coerce_token_id(getattr(tokenizer, "eos_token_id", None))
    mask_token_int = _coerce_token_id(mask_token_id)
    visible_ids: List[int] = []
    has_eos = False
    mask_tokens_remaining = 0
    raw_non_mask_token_count = 0

    for tok in ids:
        tok_int = int(tok)
        if mask_token_int is not None and tok_int == mask_token_int:
            mask_tokens_remaining += 1
            continue
        raw_non_mask_token_count += 1
        if eos_token_id is not None and tok_int == eos_token_id:
            has_eos = True
            break
        visible_ids.append(tok_int)

    decoded_text = ""
    if visible_ids:
        decoded_text = str(tokenizer.decode(visible_ids, skip_special_tokens=True)).strip()

    return {
        "decoded_text": decoded_text,
        "visible_token_count": len(visible_ids),
        "raw_non_mask_token_count": raw_non_mask_token_count,
        "mask_tokens_remaining": mask_tokens_remaining,
        "has_eos": has_eos,
        "raw_sequence_length": len(ids),
    }


def _should_apply_gsm8k_prompt(cfg: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(cfg, dict):
        return False
    data_cfg = cfg.get("data", {})
    style = str(data_cfg.get("math_prompt_style", "auto")).strip().lower()
    if style in _RAW_STYLE_VALUES:
        return False
    if style in _GSM8K_STYLE_VALUES:
        return True
    if style != "auto":
        return False

    eval_task = str(cfg.get("evaluation", {}).get("task_type", "math")).strip().lower()
    if eval_task != "math":
        return False
    dataset_hints = [
        str(data_cfg.get("eval_dataset", "")),
        str(data_cfg.get("train_dataset", "")),
    ]
    return any("gsm8k" in hint.lower() for hint in dataset_hints if hint)


def is_gsm8k_dataset(cfg: Optional[Dict[str, Any]]) -> bool:
    """Return True when the configured task is GSM8K."""
    if not isinstance(cfg, dict):
        return False
    data_cfg = cfg.get("data", {})
    dataset_hints = [
        str(data_cfg.get("eval_dataset", "")),
        str(data_cfg.get("train_dataset", "")),
    ]
    return any("gsm8k" in hint.lower() for hint in dataset_hints if hint)


def _format_task_prompt(prompt: str, cfg: Optional[Dict[str, Any]]) -> str:
    base_prompt = (prompt or "").strip()
    if not base_prompt:
        return ""
    if _should_apply_gsm8k_prompt(cfg):
        return f"{base_prompt}\n\n{_GSM8K_PROMPT_SUFFIX}"
    return base_prompt


def build_prompt(
    tokenizer,
    question: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool]:
    """Build prompt text plus the proper add_special_tokens decision."""
    prompt = _format_task_prompt(question, cfg)
    if not prompt:
        return "", True

    data_cfg = (cfg or {}).get("data", {}) if isinstance(cfg, dict) else {}
    use_chat_template = str(data_cfg.get("use_chat_template", "auto")).strip().lower()
    if use_chat_template in _FALSE_VALUES:
        return prompt, True

    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    has_apply = callable(apply_chat)

    if use_chat_template in _TRUE_VALUES:
        try_chat = has_apply
    elif use_chat_template == "auto":
        # In auto mode, attempt chat formatting whenever the tokenizer supports it.
        # Some tokenizers expose a functional default template even when the
        # `chat_template` field is unset.
        try_chat = has_apply
    else:
        # Unknown value: fail open to plain prompt for robustness.
        return prompt, True

    if try_chat:
        try:
            templated = apply_chat(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            templated_text = str(templated or "").strip()
            if templated_text:
                return templated_text, False
        except Exception:
            pass

    return prompt, True


def build_prompt_text(
    tokenizer,
    question: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Backward-compatible prompt text helper."""
    prompt_text, _ = build_prompt(tokenizer, question, cfg)
    return prompt_text


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


def _extract_first_numeric_candidate(text: str) -> Optional[str]:
    matches = _NUMERIC_RE.findall(text or "")
    if not matches:
        return None
    return _normalize_answer_text(matches[0])


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
    """
    Heuristic scalar extractor for datasets that do not ship an official grader.

    This helper is intentionally broad and can be wrong for benchmark-specific
    formats such as GSM8K. Prefer a dataset's official evaluator whenever one
    exists.
    """
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


def extract_gsm8k_official_answer(text: str) -> str:
    """
    Official OpenAI GSM8K extraction rule from grade_school_math/dataset.py.

    Source of truth:
      https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
    """
    match = _GSM8K_OFFICIAL_ANS_RE.search(text or "")
    if match:
        return match.group(1).strip().replace(",", "")
    return _GSM8K_INVALID_ANS


def check_gsm8k_correctness_official(generated: str, reference: str) -> bool:
    """Official OpenAI GSM8K exact-match correctness."""
    gold = extract_gsm8k_official_answer(reference)
    if gold == _GSM8K_INVALID_ANS:
        return False
    return extract_gsm8k_official_answer(generated) == gold


def _normalize_gsm8k_llada_answer(value: str) -> str:
    normalized = (value or "").strip()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("+", "")
    normalized = normalized.strip().strip("[](){}<>")
    normalized = normalized.rstrip(".")
    normalized = re.sub(r"(?<=\d)\.0+(?!\d)", "", normalized)
    return normalized.strip()


def _gsm8k_allows_leading_scalar_prefix(prefix: str) -> bool:
    cleaned = prefix or ""
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = cleaned.replace(r"\boxed{", "")
    cleaned = cleaned.replace(r"\(", "")
    cleaned = cleaned.replace(r"\[", "")
    cleaned = re.sub(r"[\s*_`~#$>:;,\-={}\[\]()\"']+", "", cleaned)
    return cleaned == ""


def _extract_leading_numeric_candidate(text: str) -> Optional[str]:
    match = _NUMERIC_RE.search(text or "")
    if not match:
        return None
    prefix = (text or "")[: match.start()]
    if not _gsm8k_allows_leading_scalar_prefix(prefix):
        return None
    return _normalize_answer_text(match.group(0))


def _extract_gsm8k_llada_from_line(line: str) -> str:
    candidate = _extract_leading_numeric_candidate(line)
    if candidate is None:
        return _GSM8K_INVALID_ANS
    return _normalize_gsm8k_llada_answer(candidate)


def _extract_gsm8k_llada_flexible_answer(text: str) -> str:
    candidate = _extract_numeric_candidate(text)
    if candidate is None:
        return _GSM8K_INVALID_ANS
    return _normalize_gsm8k_llada_answer(candidate)


def _extract_gsm8k_llada_multiline_answer(text: str) -> str:
    lines = (text or "").splitlines()
    for idx, line in enumerate(lines):
        if not _GSM8K_STANDALONE_ANSWER_HEADING_RE.match(line):
            continue
        for next_line in lines[idx + 1 :]:
            stripped = next_line.strip()
            if not stripped:
                continue
            return _extract_gsm8k_llada_from_line(stripped)
    return _GSM8K_INVALID_ANS


def extract_gsm8k_llada_answer(text: str) -> str:
    """
    GSM8K extractor tuned for LLaDA-style prose answers.

    It prefers explicit final-answer markers (`####`, `\\boxed{}`, answer lines)
    and only falls back to a broad last-number heuristic when no explicit cue is
    present. If an explicit answer cue exists but the associated text contains
    multiple prose numbers without a leading scalar answer, this extractor marks
    the sample invalid instead of guessing between first/last numeric tokens.
    """
    saw_explicit_answer_cue = False

    strict = _GSM8K_OFFICIAL_ANS_RE.search(text or "")
    if strict:
        return _normalize_gsm8k_llada_answer(strict.group(1))

    boxed = re.findall(r"\\boxed\{([^}]+)\}", text or "")
    if boxed:
        saw_explicit_answer_cue = True
        candidate = _extract_gsm8k_llada_from_line(boxed[-1])
        if candidate != _GSM8K_INVALID_ANS:
            return candidate

    hash_lines = _GSM8K_HASH_LINE_RE.findall(text or "")
    if hash_lines:
        saw_explicit_answer_cue = True
        candidate = _extract_gsm8k_llada_from_line(hash_lines[-1])
        if candidate != _GSM8K_INVALID_ANS:
            return candidate

    for pattern in _GSM8K_ANSWER_LINE_RES:
        matches = pattern.findall(text or "")
        if matches:
            saw_explicit_answer_cue = True
            candidate = _extract_gsm8k_llada_from_line(matches[-1])
            if candidate != _GSM8K_INVALID_ANS:
                return candidate

    multiline_candidate = _extract_gsm8k_llada_multiline_answer(text)
    if multiline_candidate != _GSM8K_INVALID_ANS:
        return multiline_candidate
    if _GSM8K_STANDALONE_ANSWER_HEADING_RE.search(text or ""):
        saw_explicit_answer_cue = True

    if saw_explicit_answer_cue:
        return _GSM8K_INVALID_ANS

    return _extract_gsm8k_llada_flexible_answer(text)


def extract_gsm8k_llada_reference(text: str) -> str:
    """Extract GSM8K references with the same normalization as predictions."""
    strict = _GSM8K_OFFICIAL_ANS_RE.search(text or "")
    if strict:
        return _normalize_gsm8k_llada_answer(strict.group(1))
    return _extract_gsm8k_llada_flexible_answer(text)


def check_gsm8k_correctness_llada(generated: str, reference: str) -> bool:
    """LLaDA-friendly GSM8K exact-match correctness with flexible extraction."""
    gold = extract_gsm8k_llada_reference(reference)
    if gold == _GSM8K_INVALID_ANS:
        return False
    return extract_gsm8k_llada_answer(generated) == gold


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
