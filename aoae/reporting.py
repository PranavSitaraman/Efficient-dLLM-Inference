"""Artifact reporting commands for AOAE experiments."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experiment_utils import load_json, write_csv, write_markdown


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


def _parse_tau_r(config_note: str) -> Optional[float]:
    match = re.search(r"tau_r=([0-9]*\.?[0-9]+)", config_note)
    if not match:
        return None
    return _safe_float(match.group(1), default=0.0)


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


def load_comparison_rows(eval_path: Path) -> List[Dict[str, Any]]:
    results = load_json(eval_path)
    metadata_path = eval_path.with_name("eval_metadata.json")
    metadata = load_json(metadata_path) if metadata_path.exists() else {}

    backend = str(metadata.get("backend", ""))
    run_name = str(metadata.get("run_name", eval_path.parent.name))
    model_name = str(metadata.get("model_name_or_path", ""))
    mode = str(metadata.get("mode", "unknown"))
    config_path = str(metadata.get("config_path", ""))
    output_dir = str(metadata.get("output_dir", str(eval_path.parent)))
    eval_dataset = str(metadata.get("eval_dataset", ""))
    eval_dataset_config = metadata.get("eval_dataset_config", None)
    routing_temperature = metadata.get("routing_temperature", None)
    reuse_signal_method = str(metadata.get("reuse_signal_method", "argmax_match"))
    positional_cache_enabled = bool(metadata.get("positional_cache_enabled", False))
    positional_cache_horizon = metadata.get("positional_cache_horizon", "")
    positional_cache_budget = metadata.get("positional_cache_refresh_budget", "")
    candidate_policy = str(metadata.get("candidate_policy", "learned_topb"))
    task_type = str(metadata.get("task_type", "math"))
    host = str(metadata.get("host", ""))
    git_commit = str(metadata.get("git_commit", ""))

    rows: List[Dict[str, Any]] = []
    for result in results:
        config_note = str(result.get("config_note", ""))
        tau_r = routing_temperature
        if tau_r is None:
            tau_r = _parse_tau_r(config_note)

        row = {
            "run_name": run_name,
            "config_path": config_path,
            "mode": mode,
            "output_dir": output_dir,
            "eval_dataset": eval_dataset,
            "eval_dataset_config": "" if eval_dataset_config in (None, "") else str(eval_dataset_config),
            "model": model_name,
            "backend": backend,
            "gating": _infer_gating(backend, str(result.get("method", ""))),
            "tau_r": "" if tau_r is None else f"{_safe_float(tau_r):.4f}",
            "reuse_signal": reuse_signal_method,
            "task_type": task_type,
            "positional_cache": "on" if positional_cache_enabled else "off",
            "positional_horizon": positional_cache_horizon,
            "positional_budget": positional_cache_budget,
            "candidate_policy": candidate_policy,
            "remask": _parse_remask(config_note, metadata.get("disable_remask")),
            "method": str(result.get("method", "")),
            "note": config_note,
            "accuracy": f"{_safe_float(result.get('accuracy')):.6f}",
            "tps": f"{_safe_float(result.get('avg_tokens_per_sec')):.3f}",
            "nfe": _safe_int(result.get("avg_nfe")),
            "cache_hit_rate": f"{_safe_float(result.get('cache_hit_rate')):.6f}",
            "agreement_rate": f"{_safe_float(result.get('agreement_rate')):.6f}",
            "draft_accept_rate": f"{_safe_float(result.get('draft_accept_rate')):.6f}",
            "reuse_mean_safe": f"{_safe_float(result.get('reuse_mean_safe')):.6f}",
            "reuse_mean_js": f"{_safe_float(result.get('reuse_mean_js')):.6f}",
            "access_rate": f"{_safe_float(result.get('access_rate')):.6f}",
            "access_mandatory_rate": f"{_safe_float(result.get('access_mandatory_rate')):.6f}",
            "access_optional_rate": f"{_safe_float(result.get('access_optional_rate')):.6f}",
            "access_budget_utilization": f"{_safe_float(result.get('access_budget_utilization')):.6f}",
            "access_effective_budget": f"{_safe_float(result.get('access_effective_budget')):.6f}",
            "access_next_h_precision": f"{_safe_float(result.get('access_next_h_precision')):.6f}",
            "access_next_h_recall": f"{_safe_float(result.get('access_next_h_recall')):.6f}",
            "access_next_h_f1": f"{_safe_float(result.get('access_next_h_f1')):.6f}",
            "access_next_h_spec_precision": f"{_safe_float(result.get('access_next_h_spec_precision')):.6f}",
            "access_next_h_spec_recall": f"{_safe_float(result.get('access_next_h_spec_recall')):.6f}",
            "access_next_h_spec_f1": f"{_safe_float(result.get('access_next_h_spec_f1')):.6f}",
            "mean_boundary_depth": f"{_safe_float(result.get('mean_boundary_depth')):.6f}",
            "boundary_distribution": str(result.get("boundary_distribution", "{}")),
            "total_samples": _safe_int(result.get("total_samples")),
            "host": host,
            "git_commit": git_commit,
        }
        commits = _safe_int(result.get("cache_commits"))
        invalidations = _safe_int(result.get("cache_invalidations"))
        row["cache_invalidation_rate"] = f"{(invalidations / max(commits + invalidations, 1)):.6f}"
        rows.append(row)

    return rows


def comparison_table_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a comparison table from eval artifacts.")
    parser.add_argument("--glob", default="outputs/**/eval_results.json", help="Glob for eval result files.")
    parser.add_argument("--csv", default="results/comparison_table.csv", help="Output CSV path.")
    parser.add_argument("--md", default="results/comparison_table.md", help="Output Markdown path.")
    args = parser.parse_args(argv)

    eval_files = sorted(Path(path) for path in glob.glob(args.glob, recursive=True))
    all_rows: List[Dict[str, Any]] = []
    for eval_path in eval_files:
        all_rows.extend(load_comparison_rows(eval_path))

    all_rows.sort(key=lambda row: (row["run_name"], row["method"], row["note"]))

    csv_path = Path(args.csv)
    md_path = Path(args.md)
    write_csv(all_rows, csv_path)
    write_markdown(all_rows, md_path)

    print(f"Processed eval files: {len(eval_files)}")
    print(f"Rows written: {len(all_rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


def kv_summary_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize kv_dynamics_summary.json files.")
    parser.add_argument("--glob", default="outputs/**/kv_dynamics_summary.json")
    parser.add_argument("--csv", default="results/kv_dynamics_table.csv")
    parser.add_argument("--md", default="results/kv_dynamics_table.md")
    args = parser.parse_args(argv)

    files = sorted(Path(path) for path in glob.glob(args.glob, recursive=True))
    rows: List[Dict[str, Any]] = []

    for path in files:
        summary = load_json(path)
        metadata_path = path.with_name("eval_metadata.json")
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        rows.append(
            {
                "run_name": metadata.get("run_name", path.parent.name),
                "output_dir": metadata.get("output_dir", str(path.parent)),
                "model": metadata.get("model_name_or_path", ""),
                "tau_r": metadata.get("routing_temperature", ""),
                "reuse_signal_method": metadata.get("reuse_signal_method", "argmax_match"),
                "disable_remask": metadata.get("disable_remask", ""),
                "num_records": summary.get("num_records", 0),
                "mean_agreement": summary.get("mean_agreement", 0.0),
                "mean_access": summary.get("mean_access", 0.0),
                "mean_layer_drift_slope": summary.get("mean_layer_drift_slope", 0.0),
                "mean_off_by_one_drift_ratio": summary.get("mean_off_by_one_drift_ratio", 0.0),
                "mean_confident_token_drift_ratio": summary.get("mean_confident_token_drift_ratio", 0.0),
                "mean_thrash_rate_given_cached": summary.get("mean_thrash_rate_given_cached", 0.0),
            }
        )

    if not rows:
        print("No kv_dynamics_summary.json files found.")
        return

    csv_path = Path(args.csv)
    md_path = Path(args.md)
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"Processed summaries: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
