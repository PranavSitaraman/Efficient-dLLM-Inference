#!/usr/bin/env python3
"""Run POC2 sweep over training-free reuse/agreement signals."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoae.evaluate import EvalResult, main as eval_main  # noqa: E402
from aoae.runtime_checks import is_global_rank_zero  # noqa: E402


DEFAULT_GRID: Dict[str, List[float]] = {
    "argmax_match": [0.0],
    "topk_overlap": [0.0],
    "min_confidence": [0.3, 0.5, 0.7, 0.9],
    "min_margin": [0.0, 0.1, 0.2, 0.4],
    "js_divergence": [0.01, 0.03, 0.05, 0.1],
    "temporal_confidence": [0.3, 0.5, 0.7],
}


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


def _select_summary_row(results: List[EvalResult], method: str, note_contains: Optional[str]) -> EvalResult:
    rows = [r for r in results if r.method == method]
    if note_contains:
        filtered = [r for r in rows if note_contains in r.config_note]
        if filtered:
            rows = filtered
    if len(rows) != 1:
        notes = [r.config_note for r in rows]
        methods = sorted({r.method for r in results})
        raise RuntimeError(
            f"Expected one row for method={method!r}, note_contains={note_contains!r}; "
            f"got {len(rows)} rows: {notes}. Available methods: {methods}"
        )
    return rows[0]


def _slug(method: str, threshold: Optional[float]) -> str:
    if threshold is None:
        return method
    return f"{method}_thr_{threshold:.4f}".replace(".", "p")


def _load_grid(cfg: dict) -> Dict[str, List[float]]:
    grid_cfg = cfg.get("inference", {}).get("reuse_signal", {}).get("grid")
    if not isinstance(grid_cfg, dict):
        return copy.deepcopy(DEFAULT_GRID)
    merged = copy.deepcopy(DEFAULT_GRID)
    for k, v in grid_cfg.items():
        if isinstance(v, list) and v:
            merged[str(k)] = [float(x) for x in v]
    return merged


def _read_thrash_rate(output_dir: Path) -> float:
    p = output_dir / "kv_dynamics_summary.json"
    if not p.exists():
        return 0.0
    try:
        data = json.loads(p.read_text())
        return float(data.get("mean_thrash_rate_given_cached", 0.0))
    except Exception:
        return 0.0


def _plot_reuse_pareto(rows: List[Dict[str, Any]], output_root: Path) -> None:
    """Plot Pareto front of (TPS, accuracy) colored by signal method."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Pareto plot skipped (matplotlib unavailable): {exc}")
        return

    if not rows:
        return

    methods = sorted({r["reuse_signal_method"] for r in rows})
    cmap = plt.colormaps.get_cmap("tab10")
    color_map = {m: cmap(i / max(len(methods) - 1, 1)) for i, m in enumerate(methods)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    for method in methods:
        pts = [r for r in rows if r["reuse_signal_method"] == method]
        tps = [float(p["tps"]) for p in pts]
        acc = [float(p["accuracy"]) for p in pts]
        ax.scatter(tps, acc, c=[color_map[method]], label=method, s=60, edgecolors="k", linewidths=0.5)
        for p in pts:
            thr = float(p["reuse_signal_threshold"])
            ax.annotate(
                f"{thr:.3g}",
                (float(p["tps"]), float(p["accuracy"])),
                textcoords="offset points", xytext=(4, 4), fontsize=6,
            )

    ax.set_xlabel("Tokens / sec")
    ax.set_ylabel("Accuracy")
    ax.set_title("POC 2: Reuse Signal Pareto (TPS vs Accuracy)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_root / "reuse_signal_pareto.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Pareto plot saved to {out_path}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5.5))
    for method in methods:
        pts = [r for r in rows if r["reuse_signal_method"] == method]
        hit = [float(p["cache_hit_rate"]) for p in pts]
        acc = [float(p["accuracy"]) for p in pts]
        ax2.scatter(hit, acc, c=[color_map[method]], label=method, s=60, edgecolors="k", linewidths=0.5)

    ax2.set_xlabel("Cache Hit Rate")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("POC 2: Cache Hit Rate vs Accuracy")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out_path2 = output_root / "reuse_signal_cache_vs_acc.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"Cache vs accuracy plot saved to {out_path2}")


def _decision_table(rows: List[Dict[str, Any]], argmax_acc: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    constraints = [0.01, 0.02, 0.05]
    for drop in constraints:
        feasible = [r for r in rows if float(r["accuracy"]) >= (argmax_acc - drop)]
        if not feasible:
            out.append(
                {
                    "constraint": f"acc_drop<={drop:.2f}",
                    "best_tps_method": "",
                    "best_tps": "",
                    "lowest_thrash_method": "",
                    "lowest_thrash": "",
                }
            )
            continue
        best_tps = max(feasible, key=lambda r: float(r["tps"]))
        best_thrash = min(feasible, key=lambda r: float(r["thrash_rate_given_cached"]))
        out.append(
            {
                "constraint": f"acc_drop<={drop:.2f}",
                "best_tps_method": f"{best_tps['reuse_signal_method']}@{best_tps['reuse_signal_threshold']}",
                "best_tps": best_tps["tps"],
                "lowest_thrash_method": f"{best_thrash['reuse_signal_method']}@{best_thrash['reuse_signal_threshold']}",
                "lowest_thrash": best_thrash["thrash_rate_given_cached"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POC2 reuse-signal reliability sweep.")
    parser.add_argument("--config", default="configs/dual_mini_tau01.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="speculative", choices=["standard", "speculative"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument("--summary_method", default="Speculative-AOAE")
    parser.add_argument("--summary_note_contains", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--sweep_name", default=None)
    parser.add_argument("--disable_remask", action="store_true")
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--eval_dataset_config", default=None)
    parser.add_argument("--eval_split", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.eval_dataset is not None:
        base_cfg.setdefault("data", {})["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        base_cfg.setdefault("data", {})["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        base_cfg.setdefault("data", {})["eval_split"] = args.eval_split

    grid = _load_grid(base_cfg)
    run_name = base_cfg.get("logging", {}).get("run_name", Path(args.config).stem)
    sweep_name = args.sweep_name or f"{run_name}_reuse_signal_sweep"
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "sweeps" / sweep_name
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # Load model ONCE for the entire sweep
    import torch
    shared_dual_model = None
    shared_eval_ds = None
    if args.mode == "speculative" or base_cfg.get("base_model", {}).get("backend") == "dual":
        from aoae.models.dual_model import DualModelWrapper
        if is_global_rank_zero():
            print("Loading dual model ONCE for sweep reuse...")
        shared_dual_model = DualModelWrapper(base_cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shared_dual_model = shared_dual_model.to(device)

    from aoae.evaluate import _load_eval_dataset
    dc = base_cfg.get("data", {})
    if is_global_rank_zero():
        print(f"Loading eval dataset: {dc.get('eval_dataset', '')}...")
    shared_eval_ds = _load_eval_dataset(dc)

    # no_reuse: JS threshold = -1 makes every position unsafe (never cache)
    # oracle_reuse: JS threshold = 999 makes every position safe (always cache)
    baseline_runs: List[Tuple[str, float]] = [
        ("no_reuse", -1.0),
        ("oracle_reuse", 999.0),
    ]
    baseline_runs += [("argmax_match", 0.0)]
    method_runs: List[Tuple[str, float]] = []
    for method, thresholds in grid.items():
        for thr in thresholds:
            method_runs.append((method, float(thr)))
    all_runs: List[Tuple[str, float]] = []
    seen = set()
    for method, threshold in baseline_runs + method_runs:
        key = (method, round(float(threshold), 12))
        if key in seen:
            continue
        seen.add(key)
        all_runs.append((method, float(threshold)))

    for method, threshold in all_runs:
        cfg = copy.deepcopy(base_cfg)
        ic = cfg.setdefault("inference", {})
        rs = ic.setdefault("reuse_signal", {})
        if method == "no_reuse":
            rs["method"] = "js_divergence"
            rs["threshold"] = threshold  # -1.0 → nothing passes
            label_method = "no_reuse"
        elif method == "oracle_reuse":
            rs["method"] = "js_divergence"
            rs["threshold"] = threshold  # 999.0 → everything passes
            label_method = "oracle_reuse"
        else:
            rs["method"] = method
            rs["threshold"] = float(threshold)
            label_method = method

        # POC2 favors clean reliability accounting.
        if args.disable_remask:
            ic["disable_remask"] = True

        # Force KV dynamics so thrash-rate diagnostics are available.
        cfg.setdefault("analysis", {})["track_kv_dynamics"] = True

        slug = _slug(label_method, threshold)
        run_dir = output_root / slug
        cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_{slug}"
        cfg["logging"]["output_dir"] = str(run_dir)

        if is_global_rank_zero():
            print(f"\n=== reuse={label_method} threshold={threshold} ===")
        results = eval_main(
            cfg,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            mode=args.mode,
            config_path=args.config,
            skip_baselines=True,
            speculative_policy_temperatures=[args.policy_temperature],
            preloaded_dual_model=shared_dual_model,
            preloaded_eval_ds=shared_eval_ds,
        )

        note_contains = args.summary_note_contains
        if note_contains is None and args.summary_method == "Speculative-AOAE":
            note_contains = f"tau_pi={args.policy_temperature}"
        r = _select_summary_row(results, args.summary_method, note_contains)
        thrash_rate = _read_thrash_rate(run_dir)
        cache_ops = r.cache_commits + r.cache_invalidations
        cache_invalidation_rate = r.cache_invalidations / max(cache_ops, 1)
        rows.append(
            {
                "reuse_signal_method": label_method,
                "reuse_signal_threshold": f"{threshold:.6f}",
                "remask_enabled": int(not bool(ic.get("disable_remask", False))),
                "accuracy": f"{r.accuracy:.6f}",
                "tps": f"{r.avg_tokens_per_sec:.3f}",
                "avg_nfe": f"{r.avg_nfe:.1f}",
                "agreement_rate": f"{r.agreement_rate:.6f}",
                "cache_hit_rate": f"{r.cache_hit_rate:.6f}",
                "draft_accept_rate": f"{r.draft_accept_rate:.6f}",
                "mean_safe_reuse": f"{r.reuse_mean_safe:.6f}",
                "mean_js_divergence": f"{r.reuse_mean_js:.6f}",
                "thrash_rate_given_cached": f"{thrash_rate:.6f}",
                "cache_invalidation_rate": f"{cache_invalidation_rate:.6f}",
                "access_next_h_f1": f"{r.access_next_h_f1:.6f}",
                "access_next_h_spec_f1": f"{r.access_next_h_spec_f1:.6f}",
                "access_effective_budget": f"{r.access_effective_budget:.6f}",
                "total_samples": r.total_samples,
                "output_dir": str(run_dir),
            }
        )

    # Accuracy deltas are measured against strict argmax baseline.
    argmax_rows = [r for r in rows if r["reuse_signal_method"] == "argmax_match"]
    if len(argmax_rows) != 1:
        raise RuntimeError(f"Expected exactly one argmax baseline row, found {len(argmax_rows)}")
    argmax_acc = float(argmax_rows[0]["accuracy"])

    # Pick best threshold per method by highest accuracy (tie-breaker: higher TPS).
    best_by_method: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        r["accuracy_delta_vs_argmax"] = f"{float(r['accuracy']) - argmax_acc:+.6f}"
        m = r["reuse_signal_method"]
        cur = best_by_method.get(m)
        if cur is None:
            best_by_method[m] = r
            continue
        key_new = (float(r["accuracy"]), float(r["tps"]))
        key_old = (float(cur["accuracy"]), float(cur["tps"]))
        if key_new > key_old:
            best_by_method[m] = r
    for r in rows:
        best = best_by_method[r["reuse_signal_method"]]
        r["is_best_threshold"] = int(
            r["reuse_signal_threshold"] == best["reuse_signal_threshold"]
            and r["reuse_signal_method"] == best["reuse_signal_method"]
        )

    decisions = _decision_table(rows, argmax_acc=argmax_acc)

    full_json = output_root / "reuse_signal_sweep_full.json"
    full_csv = output_root / "reuse_signal_sweep_full.csv"
    full_md = output_root / "reuse_signal_sweep_full.md"
    if is_global_rank_zero():
        full_json.write_text(json.dumps(rows, indent=2))
        _write_csv(rows, full_csv)
        _write_markdown(rows, full_md)

        decision_json = output_root / "best_method_by_constraint.json"
        decision_csv = output_root / "best_method_by_constraint.csv"
        decision_md = output_root / "best_method_by_constraint.md"
        decision_json.write_text(json.dumps(decisions, indent=2))
        _write_csv(decisions, decision_csv)
        _write_markdown(decisions, decision_md)

        print(f"\nReuse-signal full table written to {full_json}")
        print(f"Reuse-signal full table written to {full_csv}")
        print(f"Reuse-signal full table written to {full_md}")
        print(f"Decision table written to {decision_json}")
        print(f"Decision table written to {decision_csv}")
        print(f"Decision table written to {decision_md}")

        _plot_reuse_pareto(rows, output_root)


if __name__ == "__main__":
    main()
