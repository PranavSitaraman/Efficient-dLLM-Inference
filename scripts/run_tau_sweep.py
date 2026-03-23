#!/usr/bin/env python3
"""Run a tau_r sweep and summarize the speed/quality tradeoff for PoC1.

This script reuses the normal evaluation pipeline so each tau_r point still
produces the standard artifacts under its own output directory, then writes a
single sweep-level summary table and plots.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aoae.evaluate import EvalResult, main as eval_main, _load_eval_dataset  # noqa: E402
from aoae.runtime_checks import is_global_rank_zero, ensure_vllm_moe_runtime  # noqa: E402


def _norm_optional(value: Any) -> Any:
    if value in ("", "null"):
        return None
    return value


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


def _resolve_checkpoint(explicit: Optional[str], base_output_dir: str) -> Optional[str]:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    if not base_output_dir:
        return None

    out_dir = Path(base_output_dir)
    for name in ("policy_best.pt", "policy_final.pt"):
        candidate = out_dir / name
        if candidate.exists():
            return str(candidate)

    step_ckpts = sorted(glob.glob(str(out_dir / "policy_step*.pt")))
    return step_ckpts[-1] if step_ckpts else None


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


def _build_tau_cfg(
    base_cfg: Dict[str, Any],
    args: argparse.Namespace,
    tau_r: float,
    run_dir: Path,
    sweep_name: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("base_model", {})["routing_temperature"] = tau_r
    if args.steps is not None:
        cfg.setdefault("inference", {})["steps"] = int(args.steps)
    if args.gen_length is not None:
        cfg.setdefault("inference", {})["gen_length"] = int(args.gen_length)
    inf_cfg = cfg.setdefault("inference", {})
    schedule = str(inf_cfg.get("speculative_schedule", "aoae")).strip().lower()
    if schedule == "llada21_block":
        if args.enable_remask:
            inf_cfg["disable_remask"] = False
        else:
            inf_cfg["disable_remask"] = bool(inf_cfg.get("disable_remask", False))
    else:
        inf_cfg["disable_remask"] = not args.enable_remask
    if args.eval_dataset is not None:
        cfg.setdefault("data", {})["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        cfg.setdefault("data", {})["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        cfg.setdefault("data", {})["eval_split"] = args.eval_split

    tau_slug = _tau_slug(tau_r)
    cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_{tau_slug}"
    cfg["logging"]["output_dir"] = str(run_dir)
    return cfg


def _build_summary_row(
    cfg: Dict[str, Any],
    row: EvalResult,
    tau_r: float,
    policy_temperature: float,
    checkpoint_path: Optional[str],
) -> Dict[str, Any]:
    return {
        "tau_r": f"{tau_r:.6f}",
        "method": row.method,
        "policy_temperature": f"{policy_temperature:.4f}",
        "remask_enabled": int(not bool(cfg.get("inference", {}).get("disable_remask", False))),
        "reuse_signal_method": str(cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")),
        "reuse_signal_threshold": f"{float(cfg.get('inference', {}).get('reuse_signal', {}).get('threshold', 0.0)):.6f}",
        "eval_dataset": cfg.get("data", {}).get("eval_dataset", ""),
        "eval_dataset_config": cfg.get("data", {}).get("eval_dataset_config", ""),
        "eval_split": cfg.get("data", {}).get("eval_split", ""),
        "accuracy": f"{row.accuracy:.6f}",
        "tps": f"{row.avg_tokens_per_sec:.3f}",
        "avg_nfe": f"{row.avg_nfe:.1f}",
        "agreement_rate": f"{row.agreement_rate:.6f}",
        "cache_hit_rate": f"{row.cache_hit_rate:.6f}",
        "draft_accept_rate": f"{row.draft_accept_rate:.6f}",
        "reuse_mean_safe": f"{row.reuse_mean_safe:.6f}",
        "reuse_mean_js": f"{row.reuse_mean_js:.6f}",
        "access_effective_budget": f"{row.access_effective_budget:.6f}",
        "access_next_h_f1": f"{row.access_next_h_f1:.6f}",
        "access_next_h_spec_f1": f"{row.access_next_h_spec_f1:.6f}",
        "routing_entropy": f"{row.routing_entropy:.6f}",
        "max_routing_entropy": f"{row.max_routing_entropy:.6f}",
        "mean_boundary_depth": f"{row.mean_boundary_depth:.6f}",
        "boundary_distribution": row.boundary_distribution,
        "total_samples": row.total_samples,
        "config_note": row.config_note,
        "output_dir": cfg.get("logging", {}).get("output_dir", ""),
        "checkpoint_path": checkpoint_path or "",
    }


def _load_completed_results(
    run_dir: Path,
    cfg: Dict[str, Any],
    *,
    mode: str,
    max_samples: Optional[int],
    checkpoint_path: Optional[str],
) -> Optional[List[EvalResult]]:
    results_path = run_dir / "eval_results.json"
    metadata_path = run_dir / "eval_metadata.json"
    if not results_path.exists() or not metadata_path.exists():
        return None

    try:
        with metadata_path.open() as f:
            metadata = json.load(f)
    except Exception:
        return None

    expected_pairs = {
        "mode": mode,
        "model_name_or_path": cfg.get("base_model", {}).get("name_or_path", ""),
        "backend": cfg.get("base_model", {}).get("backend", "auto"),
        "routing_temperature": cfg.get("base_model", {}).get("routing_temperature"),
        "disable_remask": bool(cfg.get("inference", {}).get("disable_remask", False)),
        "reuse_signal_method": cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match"),
        "reuse_signal_threshold": cfg.get("inference", {}).get("reuse_signal", {}).get("threshold", 0.0),
        "compose_gamma": cfg.get("inference", {}).get("compose_gamma", 0.0),
        "eval_dataset": cfg.get("data", {}).get("eval_dataset", ""),
        "eval_dataset_config": cfg.get("data", {}).get("eval_dataset_config"),
        "eval_split": cfg.get("data", {}).get("eval_split", ""),
        "eval_max_samples": max_samples,
        "checkpoint_path": checkpoint_path,
    }
    for key, expected in expected_pairs.items():
        actual = metadata.get(key)
        if _norm_optional(actual) != _norm_optional(expected):
            return None

    try:
        with results_path.open() as f:
            rows = json.load(f)
        return [EvalResult(**row) for row in rows]
    except Exception:
        return None


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


def _plot_sweep(rows: List[Dict[str, Any]], output_root: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Sweep plotting skipped (matplotlib unavailable): {exc}")
        return

    if not rows:
        return

    tau_vals = [float(r["tau_r"]) for r in rows]
    acc_vals = [float(r["accuracy"]) for r in rows]
    tps_vals = [float(r["tps"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].semilogx(tau_vals, acc_vals, "o-", linewidth=2, color="#1f77b4")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Downstream Quality vs Routing Temperature")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(tau_vals, tps_vals, "o-", linewidth=2, color="#d62728")
    axes[1].set_xlabel("tau_r")
    axes[1].set_ylabel("Tokens / sec")
    axes[1].set_title("Throughput vs Routing Temperature")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    vs_tau_path = output_root / "tau_sweep_vs_tau.png"
    fig.savefig(vs_tau_path, dpi=150)
    plt.close(fig)
    print(f"Sweep plot saved to {vs_tau_path}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    scatter = ax.scatter(tps_vals, acc_vals, c=tau_vals, cmap="viridis", s=90)
    for row in rows:
        ax.annotate(
            f"{float(row['tau_r']):.4g}",
            (float(row["tps"]), float(row["accuracy"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel("Tokens / sec")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto View Colored by tau_r")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("tau_r")
    fig.tight_layout()
    pareto_path = output_root / "tau_sweep_pareto.png"
    fig.savefig(pareto_path, dpi=150)
    plt.close(fig)
    print(f"Pareto plot saved to {pareto_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tau_r sweep for speculative AOAE.")
    parser.add_argument("--config", default="configs/dual_mini_tau01.yaml", help="Base YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional policy checkpoint.")
    parser.add_argument(
        "--tau_r_values",
        default="0.001,0.01,0.05,0.1,0.2,0.5",
        help="Comma-separated tau_r values to sweep.",
    )
    parser.add_argument("--mode", default="speculative", choices=["standard", "speculative"])
    parser.add_argument("--max_samples", type=int, default=None, help="Optional evaluation cap.")
    parser.add_argument("--policy_temperature", type=float, default=1.0, help="tau_pi used for summary row selection.")
    parser.add_argument("--summary_method", default="Speculative-AOAE", help="Method to summarize per tau_r.")
    parser.add_argument("--summary_note_contains", default=None, help="Optional config_note substring for selecting a result row.")
    parser.add_argument("--output_root", default=None, help="Sweep output root. Defaults under outputs/sweeps/.")
    parser.add_argument("--sweep_name", default=None, help="Short name for this sweep.")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Run baseline methods during the sweep. Disabled by default for speed.")
    parser.add_argument("--keep_baselines_every_run", action="store_true",
                        help="Run baselines for every tau_r instead of only on the first point.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override inference.steps for the sweep.")
    parser.add_argument("--gen_length", type=int, default=None,
                        help="Override inference.gen_length for the sweep.")
    parser.add_argument("--enable_remask", action="store_true",
                        help="Enable remasking during the sweep (disabled by default to isolate routing effect).")
    parser.add_argument("--require_compiled_moe_ops", action="store_true",
                        help="Fail fast if compiled vLLM MoE custom ops are unavailable.")
    parser.add_argument("--eval_dataset", default=None, help="Override data.eval_dataset.")
    parser.add_argument("--eval_dataset_config", default=None, help="Override data.eval_dataset_config.")
    parser.add_argument("--eval_split", default=None, help="Override data.eval_split.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Always recompute tau points even if completed eval artifacts already exist.")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    tau_values = _parse_tau_values(args.tau_r_values)
    base_run_name = base_cfg.get("logging", {}).get("run_name", Path(args.config).stem)
    sweep_name = args.sweep_name or f"{base_run_name}_tau_sweep"
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "sweeps" / sweep_name
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(
        args.checkpoint,
        base_cfg.get("logging", {}).get("output_dir", ""),
    )
    if args.require_compiled_moe_ops:
        ensure_vllm_moe_runtime(strict=True, verbose=is_global_rank_zero(), allow_python_fallback=False)

    if is_global_rank_zero():
        if checkpoint_path:
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found; sweep will use the default heuristic policy.")

    note_contains = args.summary_note_contains
    if note_contains is None and args.summary_method == "Speculative-AOAE":
        note_contains = f"tau_pi={args.policy_temperature}"

    cached_rows: Dict[str, Dict[str, Any]] = {}
    pending_cfgs: Dict[str, Dict[str, Any]] = {}
    pending_tau_values: List[float] = []

    for tau_r in tau_values:
        tau_slug = _tau_slug(tau_r)
        run_dir = output_root / tau_slug
        cfg = _build_tau_cfg(base_cfg, args, tau_r, run_dir, sweep_name)
        if not args.no_resume:
            saved_results = _load_completed_results(
                run_dir,
                cfg,
                mode=args.mode,
                max_samples=args.max_samples,
                checkpoint_path=checkpoint_path,
            )
            if saved_results is not None:
                try:
                    row = _select_summary_row(saved_results, args.summary_method, note_contains)
                except Exception:
                    saved_results = None
                else:
                    cached_rows[tau_slug] = _build_summary_row(
                        cfg,
                        row,
                        tau_r,
                        args.policy_temperature,
                        checkpoint_path,
                    )
                    if is_global_rank_zero():
                        print(f"Reusing completed results for tau_r={tau_r} from {run_dir}")
            if saved_results is not None:
                continue
        pending_cfgs[tau_slug] = cfg
        pending_tau_values.append(tau_r)

    # Load model ONCE and reuse across all tau_r values.
    import torch
    shared_dual_model = None
    shared_eval_ds = None
    if pending_tau_values and (args.mode == "speculative" or base_cfg.get("base_model", {}).get("backend") == "dual"):
        from aoae.models.dual_model import DualModelWrapper
        if is_global_rank_zero():
            print("Loading dual model ONCE for sweep reuse...")
        first_pending_slug = _tau_slug(pending_tau_values[0])
        shared_dual_model = DualModelWrapper(copy.deepcopy(pending_cfgs[first_pending_slug]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shared_dual_model = shared_dual_model.to(device)

    dc = base_cfg.get("data", {})
    if args.eval_dataset is not None:
        dc = dict(dc)
        dc["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        dc = dict(dc) if not isinstance(dc, dict) else dc
        dc["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        dc = dict(dc) if not isinstance(dc, dict) else dc
        dc["eval_split"] = args.eval_split
    if pending_tau_values and is_global_rank_zero():
        print(f"Loading eval dataset: {dc.get('eval_dataset', '')}...")
    if pending_tau_values:
        shared_eval_ds = _load_eval_dataset(dc)

    summary_rows: List[Dict[str, Any]] = []

    for idx, tau_r in enumerate(tau_values):
        tau_slug = _tau_slug(tau_r)
        if tau_slug in cached_rows:
            summary_rows.append(cached_rows[tau_slug])
            continue

        cfg = pending_cfgs[tau_slug]

        if is_global_rank_zero():
            print(f"\n=== tau_r = {tau_r} ===")
        run_baselines = False
        if args.run_baselines:
            run_baselines = args.keep_baselines_every_run or idx == 0
        results = eval_main(
            cfg,
            checkpoint_path=checkpoint_path,
            max_samples=args.max_samples,
            mode=args.mode,
            config_path=args.config,
            skip_baselines=(not run_baselines),
            speculative_policy_temperatures=[args.policy_temperature],
            preloaded_dual_model=shared_dual_model,
            preloaded_eval_ds=shared_eval_ds,
        )

        row = _select_summary_row(results, args.summary_method, note_contains)
        summary_rows.append(
            _build_summary_row(
                cfg,
                row,
                tau_r,
                args.policy_temperature,
                checkpoint_path,
            )
        )

    json_path = output_root / "tau_sweep_summary.json"
    csv_path = output_root / "tau_sweep_summary.csv"
    md_path = output_root / "tau_sweep_summary.md"
    if is_global_rank_zero():
        with json_path.open("w") as f:
            json.dump(summary_rows, f, indent=2)
        _write_csv(summary_rows, csv_path)
        _write_markdown(summary_rows, md_path)
        print(f"\nSweep summary written to {json_path}")
        print(f"Sweep summary written to {csv_path}")
        print(f"Sweep summary written to {md_path}")

        _plot_sweep(summary_rows, output_root)


if __name__ == "__main__":
    main()
