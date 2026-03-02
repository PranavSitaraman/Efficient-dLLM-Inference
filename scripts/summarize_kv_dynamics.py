#!/usr/bin/env python3
"""Aggregate KV-dynamics summaries across run directories."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize kv_dynamics_summary.json files.")
    parser.add_argument("--glob", default="outputs/**/kv_dynamics_summary.json")
    parser.add_argument("--csv", default="results/kv_dynamics_table.csv")
    parser.add_argument("--md", default="results/kv_dynamics_table.md")
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.glob, recursive=True))
    rows: List[Dict] = []

    for p in files:
        summary = load_json(p)
        md_path = p.with_name("eval_metadata.json")
        md = load_json(md_path) if md_path.exists() else {}
        row = {
            "run_name": md.get("run_name", p.parent.name),
            "output_dir": md.get("output_dir", str(p.parent)),
            "model": md.get("model_name_or_path", ""),
            "tau_r": md.get("routing_temperature", ""),
            "reuse_signal_method": md.get("reuse_signal_method", "argmax_match"),
            "disable_remask": md.get("disable_remask", ""),
            "num_records": summary.get("num_records", 0),
            "mean_agreement": summary.get("mean_agreement", 0.0),
            "mean_access": summary.get("mean_access", 0.0),
            "mean_layer_drift_slope": summary.get("mean_layer_drift_slope", 0.0),
            "mean_off_by_one_drift_ratio": summary.get("mean_off_by_one_drift_ratio", 0.0),
            "mean_confident_token_drift_ratio": summary.get("mean_confident_token_drift_ratio", 0.0),
            "mean_thrash_rate_given_cached": summary.get("mean_thrash_rate_given_cached", 0.0),
        }
        rows.append(row)

    if not rows:
        print("No kv_dynamics_summary.json files found.")
        return

    fields = list(rows[0].keys())
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md_path = Path(args.md)
    with md_path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in fields) + " |\n")

    print(f"Processed summaries: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
