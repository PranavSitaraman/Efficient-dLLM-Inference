#!/usr/bin/env python3
"""Build a consolidated comparison table from eval artifacts.

Scans outputs/**/eval_results.json plus optional sibling eval_metadata.json,
then writes a normalized CSV and Markdown table.

Usage:
    python3 scripts/build_comparison_table.py
    python3 scripts/build_comparison_table.py --glob "outputs/**/eval_results.json"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _parse_tau_r(config_note: str) -> float | None:
    m = re.search(r"tau_r=([0-9]*\.?[0-9]+)", config_note)
    if not m:
        return None
    return _safe_float(m.group(1), default=0.0)


def _parse_remask(config_note: str, disable_remask: Any) -> str:
    if isinstance(disable_remask, bool):
        return "off" if disable_remask else "on"
    if "remask=off" in config_note:
        return "off"
    if "remask=on" in config_note:
        return "on"
    return "unknown"


def _infer_gating(backend: str, method: str) -> str:
    if method == "Speculative-AOAE" or backend == "dual":
        return "hard+soft (speculative)"
    if backend == "soft_moe":
        return "soft"
    if backend == "dinfer":
        return "hard"
    if backend == "hf":
        return "dense/hf"
    return backend or "unknown"


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _load_rows(eval_path: Path) -> List[Dict[str, Any]]:
    results = _load_json(eval_path)
    metadata_path = eval_path.with_name("eval_metadata.json")
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}

    backend = str(metadata.get("backend", ""))
    run_name = str(metadata.get("run_name", eval_path.parent.name))
    model_name = str(metadata.get("model_name_or_path", ""))
    mode = str(metadata.get("mode", "unknown"))
    config_path = str(metadata.get("config_path", ""))
    output_dir = str(metadata.get("output_dir", str(eval_path.parent)))
    routing_temperature = metadata.get("routing_temperature", None)
    reuse_signal_method = str(metadata.get("reuse_signal_method", "argmax_match"))
    positional_cache_enabled = bool(metadata.get("positional_cache_enabled", False))
    positional_cache_horizon = metadata.get("positional_cache_horizon", "")
    positional_cache_budget = metadata.get("positional_cache_refresh_budget", "")

    rows: List[Dict[str, Any]] = []
    for r in results:
        config_note = str(r.get("config_note", ""))
        tau_r = routing_temperature
        if tau_r is None:
            tau_r = _parse_tau_r(config_note)

        row = {
            "run_name": run_name,
            "config_path": config_path,
            "mode": mode,
            "output_dir": output_dir,
            "model": model_name,
            "backend": backend,
            "gating": _infer_gating(backend, str(r.get("method", ""))),
            "tau_r": "" if tau_r is None else f"{_safe_float(tau_r):.4f}",
            "reuse_signal": reuse_signal_method,
            "positional_cache": "on" if positional_cache_enabled else "off",
            "positional_horizon": positional_cache_horizon,
            "positional_budget": positional_cache_budget,
            "remask": _parse_remask(config_note, metadata.get("disable_remask")),
            "method": str(r.get("method", "")),
            "note": config_note,
            "accuracy": f"{_safe_float(r.get('accuracy')):.6f}",
            "tps": f"{_safe_float(r.get('avg_tokens_per_sec')):.3f}",
            "nfe": _safe_int(r.get("avg_nfe")),
            "cache_hit_rate": f"{_safe_float(r.get('cache_hit_rate')):.6f}",
            "agreement_rate": f"{_safe_float(r.get('agreement_rate')):.6f}",
            "draft_accept_rate": f"{_safe_float(r.get('draft_accept_rate')):.6f}",
            "reuse_mean_safe": f"{_safe_float(r.get('reuse_mean_safe')):.6f}",
            "reuse_mean_js": f"{_safe_float(r.get('reuse_mean_js')):.6f}",
            "access_rate": f"{_safe_float(r.get('access_rate')):.6f}",
            "access_mandatory_rate": f"{_safe_float(r.get('access_mandatory_rate')):.6f}",
            "access_optional_rate": f"{_safe_float(r.get('access_optional_rate')):.6f}",
            "access_budget_utilization": f"{_safe_float(r.get('access_budget_utilization')):.6f}",
            "access_next_h_precision": f"{_safe_float(r.get('access_next_h_precision')):.6f}",
            "access_next_h_recall": f"{_safe_float(r.get('access_next_h_recall')):.6f}",
            "access_next_h_f1": f"{_safe_float(r.get('access_next_h_f1')):.6f}",
            "access_next_h_spec_precision": f"{_safe_float(r.get('access_next_h_spec_precision')):.6f}",
            "access_next_h_spec_recall": f"{_safe_float(r.get('access_next_h_spec_recall')):.6f}",
            "access_next_h_spec_f1": f"{_safe_float(r.get('access_next_h_spec_f1')):.6f}",
            "total_samples": _safe_int(r.get("total_samples")),
        }
        rows.append(row)

    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in fields) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard/soft comparison table from eval artifacts.")
    parser.add_argument("--glob", default="outputs/**/eval_results.json", help="Glob for eval result files.")
    parser.add_argument("--csv", default="results/comparison_table.csv", help="Output CSV path.")
    parser.add_argument("--md", default="results/comparison_table.md", help="Output Markdown path.")
    args = parser.parse_args()

    eval_files = sorted(Path(p) for p in glob.glob(args.glob, recursive=True))
    all_rows: List[Dict[str, Any]] = []
    for eval_path in eval_files:
        all_rows.extend(_load_rows(eval_path))

    all_rows.sort(key=lambda r: (r["run_name"], r["method"], r["note"]))

    csv_path = Path(args.csv)
    md_path = Path(args.md)
    _write_csv(all_rows, csv_path)
    _write_markdown(all_rows, md_path)

    print(f"Processed eval files: {len(eval_files)}")
    print(f"Rows written: {len(all_rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
