#!/usr/bin/env python3
"""Run a configurable AOAE ablation matrix and aggregate results."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoae.evaluate import EvalResult, main as eval_main  # noqa: E402
from aoae.runtime_checks import is_global_rank_zero  # noqa: E402


DEFAULT_ABLATIONS: List[Dict[str, Any]] = [
    {"name": "baseline_default", "overrides": {}},
    {"name": "no_composed_prediction", "overrides": {"inference.compose_gamma": 0.0}},
    {"name": "disable_remask", "overrides": {"inference.disable_remask": True}},
    {"name": "disable_positional_cache", "overrides": {"inference.positional_cache.enabled": False}},
    {"name": "candidate_sliding_window", "overrides": {"inference.positional_cache.candidate_policy": "sliding_window"}},
    {"name": "candidate_confidence_topb", "overrides": {"inference.positional_cache.candidate_policy": "confidence_topb"}},
    {"name": "no_history_features", "overrides": {"policy.use_age_feature": False, "policy.use_last_action_feature": False}},
    {"name": "boundary_head_on", "overrides": {"policy.boundary_head.enabled": True, "policy.boundary_head.num_bins": 8}},
]


def _set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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


def _select_summary_row(results: List[EvalResult], method: str) -> EvalResult:
    rows = [r for r in results if r.method == method]
    if len(rows) != 1:
        notes = [r.config_note for r in rows]
        methods = sorted({r.method for r in results})
        raise RuntimeError(
            f"Expected one row for method={method!r}; got {len(rows)} rows {notes}. "
            f"Available methods: {methods}"
        )
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AOAE ablation matrix.")
    parser.add_argument("--config", default="configs/dual_mini_tau01.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="speculative", choices=["standard", "speculative"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--summary_method", default="Speculative-AOAE")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--matrix_json", default=None, help="Optional custom matrix JSON file.")
    parser.add_argument("--keep_baselines", action="store_true",
                        help="Run baseline methods for every ablation row.")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.matrix_json:
        matrix = json.loads(Path(args.matrix_json).read_text())
        if not isinstance(matrix, list):
            raise ValueError("--matrix_json must be a list of {name, overrides}")
    else:
        matrix = copy.deepcopy(DEFAULT_ABLATIONS)

    run_name = base_cfg.get("logging", {}).get("run_name", Path(args.config).stem)
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "ablations" / f"{run_name}_matrix"
    output_root.mkdir(parents=True, exist_ok=True)

    # Load model ONCE for the entire ablation sweep
    import torch
    shared_dual_model = None
    shared_eval_ds = None
    if args.mode == "speculative" or base_cfg.get("base_model", {}).get("backend") == "dual":
        from aoae.models.dual_model import DualModelWrapper
        if is_global_rank_zero():
            print("Loading dual model ONCE for ablation sweep reuse...")
        shared_dual_model = DualModelWrapper(base_cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shared_dual_model = shared_dual_model.to(device)

    from aoae.evaluate import _load_eval_dataset
    dc = base_cfg.get("data", {})
    if is_global_rank_zero():
        print(f"Loading eval dataset: {dc.get('eval_dataset', '')}...")
    shared_eval_ds = _load_eval_dataset(dc)

    rows: List[Dict[str, Any]] = []
    for spec in matrix:
        name = str(spec["name"])
        overrides = dict(spec.get("overrides", {}))
        cfg = copy.deepcopy(base_cfg)
        for key, value in overrides.items():
            _set_nested(cfg, str(key), value)
        run_dir = output_root / name
        cfg.setdefault("logging", {})["run_name"] = f"{run_name}_{name}"
        cfg["logging"]["output_dir"] = str(run_dir)

        if is_global_rank_zero():
            print(f"\n=== ablation: {name} ===")
        results = eval_main(
            cfg,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            mode=args.mode,
            config_path=args.config,
            skip_baselines=(not args.keep_baselines),
            speculative_policy_temperatures=[1.0],
            preloaded_dual_model=shared_dual_model,
            preloaded_eval_ds=shared_eval_ds,
        )
        row = _select_summary_row(results, args.summary_method)
        rows.append(
            {
                "ablation": name,
                "method": row.method,
                "remask_enabled": int(not bool(cfg.get("inference", {}).get("disable_remask", False))),
                "accuracy": f"{row.accuracy:.6f}",
                "tps": f"{row.avg_tokens_per_sec:.3f}",
                "avg_nfe": f"{row.avg_nfe:.1f}",
                "cache_hit_rate": f"{row.cache_hit_rate:.6f}",
                "agreement_rate": f"{row.agreement_rate:.6f}",
                "draft_accept_rate": f"{row.draft_accept_rate:.6f}",
                "access_effective_budget": f"{row.access_effective_budget:.6f}",
                "access_next_h_f1": f"{row.access_next_h_f1:.6f}",
                "mean_boundary_depth": f"{row.mean_boundary_depth:.6f}",
                "config_note": row.config_note,
                "output_dir": str(run_dir),
            }
        )

    full_json = output_root / "ablation_matrix_results.json"
    full_csv = output_root / "ablation_matrix_results.csv"
    full_md = output_root / "ablation_matrix_results.md"
    if is_global_rank_zero():
        full_json.write_text(json.dumps(rows, indent=2))
        _write_csv(rows, full_csv)
        _write_markdown(rows, full_md)
        print(f"\nAblation matrix written to {full_json}")
        print(f"Ablation matrix written to {full_csv}")
        print(f"Ablation matrix written to {full_md}")


if __name__ == "__main__":
    main()
