"""Shared helpers for experiment orchestration scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SUMMARY_METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    # Legacy baseline names kept for backward compatibility with older sweep scripts.
    "block_smode": ("block_smode", "llada21_speed_mode"),
    "confidence_s_mode": ("confidence_s_mode", "llada21_speed_mode"),
    "confidence_q_mode": ("confidence_q_mode", "llada21_quality_mode"),
    # Current official labels should also accept the older names so saved scripts keep working.
    "llada21_speed_mode": ("llada21_speed_mode", "block_smode", "confidence_s_mode"),
    "llada21_quality_mode": ("llada21_quality_mode", "confidence_q_mode"),
}


def parse_float_list(raw: str, *, label: str = "value") -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError(f"Expected at least one {label}.")
    return values


def tau_slug(tau_r: float) -> str:
    return f"tau_{tau_r:.4f}".replace(".", "p")


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def write_csv(rows: list[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in fields) + " |\n")


def set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def select_summary_row(
    results: Iterable[Any],
    method: str,
    note_contains: Optional[str] = None,
) -> Any:
    result_list = list(results)
    candidate_methods = SUMMARY_METHOD_ALIASES.get(method, (method,))
    rows = [r for r in result_list if getattr(r, "method", None) in candidate_methods]
    if note_contains:
        filtered = [r for r in rows if note_contains in getattr(r, "config_note", "")]
        if filtered:
            rows = filtered
    if len(rows) != 1:
        notes = [getattr(r, "config_note", "") for r in rows]
        methods = sorted({getattr(r, "method", "") for r in result_list})
        raise RuntimeError(
            f"Expected exactly one result row for method={method!r}, "
            f"note_contains={note_contains!r}; got {len(rows)} rows: {notes}. "
            f"Candidate methods tried: {list(candidate_methods)!r}. "
            f"Available methods: {methods}"
        )
    return rows[0]
