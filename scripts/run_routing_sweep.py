#!/usr/bin/env python3
"""Run the routing-only hard-vs-soft sweep for PoC0 / PoC1A.

This keeps speculation off and compares:
  - hard routing baseline (`dinfer`) treated as tau_r = 0
  - soft routing variants (`soft_moe`) for a sweep of positive tau_r values

Each run still emits the standard eval artifacts into its own output directory.
This script additionally writes sweep-level summary/full tables and plots.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoae.evaluate import EvalResult, main as eval_main  # noqa: E402


def _parse_tau_values(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("Expected at least one tau_r value.")
    return values


def _tau_slug(tau_r: float) -> str:
    return f"tau_{tau_r:.4f}".replace(".", "p")


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


def _select_summary_row(
    results: List[EvalResult],
    method: str,
    note_contains: Optional[str],
) -> EvalResult:
    rows = [r for r in results if r.method == method]
    if note_contains:
        filtered = [r for r in rows if note_contains in r.config_note]
        if filtered:
            rows = filtered
    if len(rows) != 1:
        notes = [r.config_note for r in rows]
        raise RuntimeError(
            f"Expected exactly one result row for method={method!r}, "
            f"note_contains={note_contains!r}; got {len(rows)} rows: {notes}"
        )
    return rows[0]


def _override_eval_data(cfg: dict, args) -> None:
    if args.eval_dataset is not None:
        cfg.setdefault("data", {})["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        cfg.setdefault("data", {})["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        cfg.setdefault("data", {})["eval_split"] = args.eval_split


def _result_to_row(
    result: EvalResult,
    *,
    routing_label: str,
    routing_mode: str,
    tau_r: float,
    backend: str,
    model_name: str,
    eval_dataset: str,
    eval_dataset_config: Any,
    eval_split: str,
    output_dir: str,
    checkpoint_path: Optional[str],
) -> Dict[str, Any]:
    return {
        "routing_label": routing_label,
        "routing_mode": routing_mode,
        "tau_r": f"{tau_r:.6f}",
        "backend": backend,
        "model": model_name,
        "eval_dataset": eval_dataset,
        "eval_dataset_config": "" if eval_dataset_config in (None, "") else str(eval_dataset_config),
        "eval_split": eval_split,
        "method": result.method,
        "accuracy": f"{result.accuracy:.6f}",
        "tps": f"{result.avg_tokens_per_sec:.3f}",
        "avg_nfe": f"{result.avg_nfe:.1f}",
        "cache_hit_rate": f"{result.cache_hit_rate:.6f}",
        "agreement_rate": f"{result.agreement_rate:.6f}",
        "draft_accept_rate": f"{result.draft_accept_rate:.6f}",
        "total_samples": result.total_samples,
        "config_note": result.config_note,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path or "",
    }


def _plot_summary(rows: List[Dict[str, Any]], output_root: Path, summary_method: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Routing sweep plotting skipped (matplotlib unavailable): {exc}")
        return

    if not rows:
        return

    labels = [str(r["routing_label"]) for r in rows]
    acc_vals = [float(r["accuracy"]) for r in rows]
    tps_vals = [float(r["tps"]) for r in rows]
    is_hard = [str(r["routing_mode"]) == "hard" for r in rows]
    xs = list(range(len(rows)))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(xs, acc_vals, color="#1f77b4", linewidth=2, alpha=0.7)
    axes[1].plot(xs, tps_vals, color="#d62728", linewidth=2, alpha=0.7)
    for x, acc, tps, hard in zip(xs, acc_vals, tps_vals, is_hard):
        color = "#111111" if hard else "#1f77b4"
        axes[0].scatter(x, acc, color=color, s=80, zorder=5)
        axes[1].scatter(x, tps, color=("#111111" if hard else "#d62728"), s=80, zorder=5)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{summary_method}: Quality vs Routing Condition")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Tokens / sec")
    axes[1].set_xlabel("Routing condition")
    axes[1].set_title(f"{summary_method}: Throughput vs Routing Condition")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(xs, labels)
    fig.tight_layout()
    path = output_root / "routing_sweep_vs_condition.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Routing-condition plot saved to {path}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for row in rows:
        hard = str(row["routing_mode"]) == "hard"
        ax.scatter(
            float(row["tps"]),
            float(row["accuracy"]),
            color=("#111111" if hard else "#1f77b4"),
            s=90,
        )
        ax.annotate(
            str(row["routing_label"]),
            (float(row["tps"]), float(row["accuracy"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Tokens / sec")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{summary_method}: Hard vs Soft Routing Pareto View")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_root / "routing_sweep_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Pareto plot saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the routing-only tau_r sweep (PoC0 / PoC1A).")
    parser.add_argument("--hard_config", default="configs/llada21_mini_hard.yaml", help="Hard-routing config.")
    parser.add_argument("--soft_config", default="configs/llada21_mini_soft.yaml", help="Soft-routing config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint if you want AOAE rows in standard mode.")
    parser.add_argument(
        "--tau_r_values",
        default="0.001,0.01,0.05,0.1,0.2,0.5",
        help="Comma-separated positive tau_r values for the soft-routing sweep.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional evaluation cap.")
    parser.add_argument("--summary_method", default="block_smode", help="Eval method to summarize/plot.")
    parser.add_argument("--summary_note_contains", default=None, help="Optional config_note substring for row selection.")
    parser.add_argument("--output_root", default=None, help="Sweep output root. Defaults under outputs/sweeps/.")
    parser.add_argument("--sweep_name", default=None, help="Short name for this sweep.")
    parser.add_argument("--eval_dataset", default=None, help="Override data.eval_dataset.")
    parser.add_argument("--eval_dataset_config", default=None, help="Override data.eval_dataset_config.")
    parser.add_argument("--eval_split", default=None, help="Override data.eval_split.")
    args = parser.parse_args()

    tau_values = _parse_tau_values(args.tau_r_values)
    if any(tau <= 0.0 for tau in tau_values):
        raise ValueError("Soft-routing tau_r values must be > 0. Use the hard config for tau_r = 0.")

    with open(args.hard_config) as f:
        hard_cfg_template = yaml.safe_load(f)
    with open(args.soft_config) as f:
        soft_cfg_template = yaml.safe_load(f)

    hard_name = Path(args.hard_config).stem
    soft_name = Path(args.soft_config).stem
    sweep_name = args.sweep_name or f"{hard_name}_vs_{soft_name}_routing_sweep"
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "sweeps" / sweep_name
    output_root.mkdir(parents=True, exist_ok=True)

    full_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # Hard-routing baseline: treated as tau_r = 0.
    hard_cfg = copy.deepcopy(hard_cfg_template)
    _override_eval_data(hard_cfg, args)
    hard_dir = output_root / "hard"
    hard_cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_hard"
    hard_cfg["logging"]["output_dir"] = str(hard_dir)
    print("\n=== hard routing (tau_r = 0 / dinfer) ===")
    hard_results = eval_main(
        hard_cfg,
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        mode="standard",
        config_path=args.hard_config,
    )
    for result in hard_results:
        full_rows.append(
            _result_to_row(
                result,
                routing_label="hard",
                routing_mode="hard",
                tau_r=0.0,
                backend=hard_cfg["base_model"]["backend"],
                model_name=hard_cfg["base_model"]["name_or_path"],
                eval_dataset=hard_cfg["data"]["eval_dataset"],
                eval_dataset_config=hard_cfg["data"].get("eval_dataset_config"),
                eval_split=hard_cfg["data"]["eval_split"],
                output_dir=hard_cfg["logging"]["output_dir"],
                checkpoint_path=args.checkpoint,
            )
        )
    hard_summary = _select_summary_row(hard_results, args.summary_method, args.summary_note_contains)
    summary_rows.append(
        _result_to_row(
            hard_summary,
            routing_label="hard",
            routing_mode="hard",
            tau_r=0.0,
            backend=hard_cfg["base_model"]["backend"],
            model_name=hard_cfg["base_model"]["name_or_path"],
            eval_dataset=hard_cfg["data"]["eval_dataset"],
            eval_dataset_config=hard_cfg["data"].get("eval_dataset_config"),
            eval_split=hard_cfg["data"]["eval_split"],
            output_dir=hard_cfg["logging"]["output_dir"],
            checkpoint_path=args.checkpoint,
        )
    )

    # Soft-routing sweep.
    for tau_r in tau_values:
        soft_cfg = copy.deepcopy(soft_cfg_template)
        _override_eval_data(soft_cfg, args)
        soft_cfg.setdefault("base_model", {})["routing_temperature"] = tau_r
        tau_slug = _tau_slug(tau_r)
        soft_dir = output_root / tau_slug
        soft_cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_{tau_slug}"
        soft_cfg["logging"]["output_dir"] = str(soft_dir)

        print(f"\n=== soft routing (tau_r = {tau_r}) ===")
        soft_results = eval_main(
            soft_cfg,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            mode="standard",
            config_path=args.soft_config,
        )
        label = f"{tau_r:.4g}"
        for result in soft_results:
            full_rows.append(
                _result_to_row(
                    result,
                    routing_label=label,
                    routing_mode="soft",
                    tau_r=tau_r,
                    backend=soft_cfg["base_model"]["backend"],
                    model_name=soft_cfg["base_model"]["name_or_path"],
                    eval_dataset=soft_cfg["data"]["eval_dataset"],
                    eval_dataset_config=soft_cfg["data"].get("eval_dataset_config"),
                    eval_split=soft_cfg["data"]["eval_split"],
                    output_dir=soft_cfg["logging"]["output_dir"],
                    checkpoint_path=args.checkpoint,
                )
            )
        soft_summary = _select_summary_row(soft_results, args.summary_method, args.summary_note_contains)
        summary_rows.append(
            _result_to_row(
                soft_summary,
                routing_label=label,
                routing_mode="soft",
                tau_r=tau_r,
                backend=soft_cfg["base_model"]["backend"],
                model_name=soft_cfg["base_model"]["name_or_path"],
                eval_dataset=soft_cfg["data"]["eval_dataset"],
                eval_dataset_config=soft_cfg["data"].get("eval_dataset_config"),
                eval_split=soft_cfg["data"]["eval_split"],
                output_dir=soft_cfg["logging"]["output_dir"],
                checkpoint_path=args.checkpoint,
            )
        )

    full_json = output_root / "routing_sweep_full.json"
    full_csv = output_root / "routing_sweep_full.csv"
    full_md = output_root / "routing_sweep_full.md"
    with full_json.open("w") as f:
        json.dump(full_rows, f, indent=2)
    _write_csv(full_rows, full_csv)
    _write_markdown(full_rows, full_md)

    summary_json = output_root / "routing_sweep_summary.json"
    summary_csv = output_root / "routing_sweep_summary.csv"
    summary_md = output_root / "routing_sweep_summary.md"
    with summary_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    _write_csv(summary_rows, summary_csv)
    _write_markdown(summary_rows, summary_md)

    print(f"\nRouting sweep full table written to {full_json}")
    print(f"Routing sweep full table written to {full_csv}")
    print(f"Routing sweep full table written to {full_md}")
    print(f"Routing sweep summary written to {summary_json}")
    print(f"Routing sweep summary written to {summary_csv}")
    print(f"Routing sweep summary written to {summary_md}")

    _plot_summary(summary_rows, output_root, args.summary_method)


if __name__ == "__main__":
    main()
