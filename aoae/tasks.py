"""Shared task parsing and correctness helpers for training and evaluation."""

from __future__ import annotations

import re
from numbers import Integral
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Tuple


_NUMERIC_RE = re.compile(
    r"[-+]?(?:\d+\s*/\s*\d+|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


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
    ids = _to_token_list(token_ids)
    if not ids:
        return ""

    eos_token_id = _coerce_token_id(getattr(tokenizer, "eos_token_id", None))
    if eos_token_id is not None:
        try:
            eos_index = ids.index(eos_token_id)
            ids = ids[:eos_index]
        except ValueError:
            pass

    mask_token_int = _coerce_token_id(mask_token_id)
    if mask_token_int is not None:
        ids = [tok for tok in ids if int(tok) != mask_token_int]

    if not ids:
        return ""

    return str(tokenizer.decode(ids, skip_special_tokens=True)).strip()


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
