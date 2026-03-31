"""Paper and POC orchestration commands.

This module consolidates the paper-facing experiment surface:
  - PoC 1 tau sweep
  - Routing-only hard-vs-soft sweep
  - PoC 2 reuse-signal sweep
  - Ablation matrix
  - Paper-suite orchestration
"""

from __future__ import annotations

import argparse
import copy
import glob
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .evaluate import EvalResult, _load_eval_dataset, main as eval_main
from .experiment_utils import (
    load_json,
    parse_float_list,
    select_summary_row,
    set_nested,
    tau_slug,
    write_csv,
    write_markdown,
)
from .runtime_checks import ensure_vllm_moe_runtime, is_global_rank_zero

ROOT = Path(__file__).resolve().parents[1]


def _invoke_main(main_fn, prog: str, argv: List[str]) -> None:
    prev_argv = sys.argv[:]
    try:
        sys.argv = [prog, *argv]
        if len(inspect.signature(main_fn).parameters) == 0:
            main_fn()
        else:
            main_fn(argv)
    finally:
        sys.argv = prev_argv


def _norm_optional(value: Any) -> Any:
    if value in ("", "null"):
        return None
    return value


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _close_preloaded_runtime(obj: Any) -> None:
    if obj is None:
        return
    close_fn = getattr(obj, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


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


def _override_eval_data(cfg: dict, args: argparse.Namespace) -> None:
    if args.eval_dataset is not None:
        cfg.setdefault("data", {})["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        cfg.setdefault("data", {})["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        cfg.setdefault("data", {})["eval_split"] = args.eval_split


def _override_prediction_saving(cfg: dict, args: argparse.Namespace) -> None:
    if getattr(args, "save_predictions", False):
        cfg.setdefault("evaluation", {})["save_predictions"] = True
    max_saved_predictions = getattr(args, "max_saved_predictions", None)
    if max_saved_predictions is not None:
        cfg.setdefault("evaluation", {})["save_predictions"] = True
        cfg.setdefault("evaluation", {})["max_saved_predictions"] = min(
            int(max_saved_predictions), 50
        )


def _apply_training_free_blockwise_defaults(
    cfg: Dict[str, Any],
    *,
    checkpoint_path: Optional[str],
) -> bool:
    """Default research sweeps to the training-free blockwise scheduler.

    When no learned AOAE checkpoint is provided and the config has not selected a
    speculative schedule explicitly, use the official blockwise LLaDA2.1 decode
    schedule so the sweep studies routing/reuse rather than a policy-training gap.
    """
    if checkpoint_path:
        return False
    inf_cfg = cfg.setdefault("inference", {})
    schedule = inf_cfg.get("speculative_schedule")
    if schedule not in (None, "", "null"):
        return False
    inf_cfg["speculative_schedule"] = "llada21_block"
    off_cfg = inf_cfg.setdefault("llada21_official", {})
    off_cfg.setdefault("use_block_diffusion", True)
    off_cfg.setdefault("max_post_steps", 16)
    off_cfg.setdefault("threshold", 0.7)
    off_cfg.setdefault("editing_threshold", 0.5)
    off_cfg.setdefault("enable_mbe", False)
    return True


def _slug(method: str, threshold: Optional[float]) -> str:
    if threshold is None:
        return method
    return f"{method}_thr_{threshold:.4f}".replace(".", "p")


def _load_grid(cfg: dict) -> Dict[str, List[float]]:
    default_grid: Dict[str, List[float]] = {
        "argmax_match": [0.0],
        "topk_overlap": [0.0],
        "min_confidence": [0.3, 0.5, 0.7, 0.9],
        "min_margin": [0.0, 0.1, 0.2, 0.4],
        "js_divergence": [0.01, 0.03, 0.05, 0.1],
        "temporal_confidence": [0.3, 0.5, 0.7],
    }
    grid_cfg = cfg.get("inference", {}).get("reuse_signal", {}).get("grid")
    if not isinstance(grid_cfg, dict):
        return copy.deepcopy(default_grid)
    merged = copy.deepcopy(default_grid)
    for key, value in grid_cfg.items():
        if isinstance(value, list) and value:
            merged[str(key)] = [float(x) for x in value]
    return merged


def _read_thrash_rate(output_dir: Path) -> float:
    path = output_dir / "kv_dynamics_summary.json"
    if not path.exists():
        return 0.0
    try:
        data = load_json(path)
        return float(data.get("mean_thrash_rate_given_cached", 0.0))
    except Exception:
        return 0.0


def _read_kv_dynamics_summary(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "kv_dynamics_summary.json"
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _format_layer_drift_preview(
    per_layer: List[Dict[str, Any]],
    *,
    limit: int = 6,
) -> str:
    if not per_layer:
        return "unavailable"

    chunks: List[str] = []
    for row in per_layer[:limit]:
        layer_idx = int(row.get("layer_idx", 0))
        mean_drift = _safe_float(
            row.get("mean_drift", row.get("mean_hidden_drift", 0.0))
        )
        chunks.append(f"L{layer_idx}:{mean_drift:.4f}")
    if len(per_layer) > limit:
        chunks.append("...")
    return ", ".join(chunks)


def _print_kv_dynamics_trial_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    if not is_global_rank_zero():
        return
    if not summary:
        print(f"KV dynamics summary missing for {output_dir}")
        return

    mean_age_drift = summary.get("mean_age_drift", {})
    print(
        "KV dynamics: "
        f"measure={summary.get('layer_drift_measure', 'hidden_state_proxy')} "
        f"slope={_safe_float(summary.get('mean_layer_drift_slope', 0.0)):.4f} "
        f"off_by_one={_safe_float(summary.get('mean_off_by_one_drift_ratio', 0.0)):.4f} "
        f"age1={_safe_float(mean_age_drift.get('age1', 0.0)):.4f} "
        f"age2p={_safe_float(mean_age_drift.get('age2p', 0.0)):.4f}"
    )
    print(
        "Layer-wise drift: "
        f"{_format_layer_drift_preview(summary.get('per_layer_drift', []))}"
    )


def _plot_tau_sweep(rows: List[Dict[str, Any]], output_root: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Sweep plotting skipped (matplotlib unavailable): {exc}")
        return

    if not rows:
        return

    tau_vals = [float(row["tau_r"]) for row in rows]
    acc_vals = [float(row["accuracy"]) for row in rows]
    tps_vals = [float(row["tps"]) for row in rows]

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
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("tau_r")
    fig.tight_layout()
    pareto_path = output_root / "tau_sweep_pareto.png"
    fig.savefig(pareto_path, dpi=150)
    plt.close(fig)
    print(f"Pareto plot saved to {pareto_path}")


def _plot_reuse_pareto(rows: List[Dict[str, Any]], output_root: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Pareto plot skipped (matplotlib unavailable): {exc}")
        return

    if not rows:
        return

    methods = sorted({row["reuse_signal_method"] for row in rows})
    cmap = plt.colormaps.get_cmap("tab10")
    color_map = {method: cmap(i / max(len(methods) - 1, 1)) for i, method in enumerate(methods)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    for method in methods:
        points = [row for row in rows if row["reuse_signal_method"] == method]
        tps = [float(point["tps"]) for point in points]
        acc = [float(point["accuracy"]) for point in points]
        ax.scatter(tps, acc, c=[color_map[method]], label=method, s=60, edgecolors="k", linewidths=0.5)
        for point in points:
            threshold = float(point["reuse_signal_threshold"])
            ax.annotate(
                f"{threshold:.3g}",
                (float(point["tps"]), float(point["accuracy"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
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
        points = [row for row in rows if row["reuse_signal_method"] == method]
        hit = [float(point["cache_hit_rate"]) for point in points]
        acc = [float(point["accuracy"]) for point in points]
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


def _plot_routing_summary(rows: List[Dict[str, Any]], output_root: Path, summary_method: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Routing sweep plotting skipped (matplotlib unavailable): {exc}")
        return

    if not rows:
        return

    labels = [str(row["routing_label"]) for row in rows]
    acc_vals = [float(row["accuracy"]) for row in rows]
    tps_vals = [float(row["tps"]) for row in rows]
    is_hard = [str(row["routing_mode"]) == "hard" for row in rows]
    xs = list(range(len(rows)))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(xs, acc_vals, color="#1f77b4", linewidth=2, alpha=0.7)
    axes[1].plot(xs, tps_vals, color="#d62728", linewidth=2, alpha=0.7)
    for x, acc, tps, hard in zip(xs, acc_vals, tps_vals, is_hard):
        axes[0].scatter(x, acc, color=("#111111" if hard else "#1f77b4"), s=80, zorder=5)
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
    condition_path = output_root / "routing_sweep_vs_condition.png"
    fig.savefig(condition_path, dpi=150)
    plt.close(fig)
    print(f"Routing-condition plot saved to {condition_path}")

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
    pareto_path = output_root / "routing_sweep_pareto.png"
    fig.savefig(pareto_path, dpi=150)
    plt.close(fig)
    print(f"Pareto plot saved to {pareto_path}")


def _decision_table(rows: List[Dict[str, Any]], argmax_acc: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    constraints = [0.01, 0.02, 0.05]
    for drop in constraints:
        feasible = [row for row in rows if float(row["accuracy"]) >= (argmax_acc - drop)]
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
        best_tps = max(feasible, key=lambda row: float(row["tps"]))
        best_thrash = min(feasible, key=lambda row: float(row["thrash_rate_given_cached"]))
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


def _annotate_tradeoff(summary_rows: List[Dict[str, Any]]) -> None:
    if not summary_rows:
        return
    hard_rows = [row for row in summary_rows if str(row["routing_mode"]) == "hard"]
    if len(hard_rows) != 1:
        raise RuntimeError(f"Expected exactly one hard reference row, found {len(hard_rows)}.")

    hard = hard_rows[0]
    hard_acc = float(hard["accuracy"])
    hard_tps = float(hard["tps"])
    frontier = []
    values = [(float(row["tps"]), float(row["accuracy"])) for row in summary_rows]
    for i, (t_i, a_i) in enumerate(values):
        dominated = False
        for j, (t_j, a_j) in enumerate(values):
            if i == j:
                continue
            if (t_j >= t_i and a_j >= a_i) and (t_j > t_i or a_j > a_i):
                dominated = True
                break
        frontier.append(0 if dominated else 1)

    for row, is_frontier in zip(summary_rows, frontier):
        row["delta_accuracy_vs_hard"] = f"{float(row['accuracy']) - hard_acc:+.6f}"
        row["delta_tps_vs_hard"] = f"{float(row['tps']) - hard_tps:+.3f}"
        row["frontier_index"] = int(is_frontier)


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
    _override_eval_data(cfg, args)
    cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_{tau_slug(tau_r)}"
    cfg["logging"]["output_dir"] = str(run_dir)
    return cfg


def _build_tau_summary_row(
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
        "primary_skip_ratio": f"{row.primary_skip_ratio:.6f}",
        "primary_full_steps": f"{row.primary_full_steps:.3f}",
        "primary_partial_steps": f"{row.primary_partial_steps:.3f}",
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
        metadata = load_json(metadata_path)
    except Exception:
        return None

    expected_pairs = {
        "mode": mode,
        "model_name_or_path": cfg.get("base_model", {}).get("name_or_path", ""),
        "backend": cfg.get("base_model", {}).get("backend", "auto"),
        "speculative_schedule": cfg.get("inference", {}).get("speculative_schedule", "aoae"),
        "routing_temperature": cfg.get("base_model", {}).get("routing_temperature"),
        "disable_remask": bool(cfg.get("inference", {}).get("disable_remask", False)),
        "reuse_signal_method": cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match"),
        "reuse_signal_threshold": cfg.get("inference", {}).get("reuse_signal", {}).get("threshold", 0.0),
        "compose_gamma": cfg.get("inference", {}).get("compose_gamma", 0.0),
        "llada21_use_block_diffusion": bool(cfg.get("inference", {}).get("llada21_official", {}).get("use_block_diffusion", False)),
        "llada21_threshold": cfg.get("inference", {}).get("llada21_official", {}).get("threshold"),
        "llada21_editing_threshold": cfg.get("inference", {}).get("llada21_official", {}).get("editing_threshold"),
        "llada21_max_post_steps": cfg.get("inference", {}).get("llada21_official", {}).get("max_post_steps"),
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
        rows = load_json(results_path)
        return [EvalResult(**row) for row in rows]
    except Exception:
        return None


def _result_to_routing_row(
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
    remask_enabled: bool,
    reuse_signal_method: str,
    reuse_signal_threshold: float,
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
        "remask_enabled": int(bool(remask_enabled)),
        "reuse_signal_method": reuse_signal_method,
        "reuse_signal_threshold": f"{float(reuse_signal_threshold):.6f}",
        "method": result.method,
        "accuracy": f"{result.accuracy:.6f}",
        "tps": f"{result.avg_tokens_per_sec:.3f}",
        "avg_nfe": f"{result.avg_nfe:.1f}",
        "cache_hit_rate": f"{result.cache_hit_rate:.6f}",
        "agreement_rate": f"{result.agreement_rate:.6f}",
        "draft_accept_rate": f"{result.draft_accept_rate:.6f}",
        "access_effective_budget": f"{result.access_effective_budget:.6f}",
        "mean_boundary_depth": f"{result.mean_boundary_depth:.6f}",
        "boundary_distribution": result.boundary_distribution,
        "total_samples": result.total_samples,
        "config_note": result.config_note,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path or "",
    }


def tau_sweep_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a tau_r sweep for speculative AOAE.")
    parser.add_argument("--config", default="configs/paper.yaml", help="Base YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional policy checkpoint.")
    parser.add_argument("--tau_r_values", default="0.0001,0.001,0.005,0.01,0.02,0.05", help="Comma-separated tau_r values to sweep.")
    parser.add_argument("--mode", default="speculative", choices=["standard", "speculative"])
    parser.add_argument("--max_samples", type=int, default=None, help="Optional evaluation cap.")
    parser.add_argument("--policy_temperature", type=float, default=1.0, help="tau_pi used for summary row selection.")
    parser.add_argument("--summary_method", default="Speculative-AOAE", help="Method to summarize per tau_r.")
    parser.add_argument("--summary_note_contains", default=None, help="Optional config_note substring for selecting a result row.")
    parser.add_argument("--output_root", default=None, help="Sweep output root. Defaults under outputs/sweeps/.")
    parser.add_argument("--sweep_name", default=None, help="Short name for this sweep.")
    parser.add_argument("--run_baselines", action="store_true", help="Run baseline methods during the sweep. Disabled by default for speed.")
    parser.add_argument("--keep_baselines_every_run", action="store_true", help="Run baselines for every tau_r instead of only on the first point.")
    parser.add_argument("--steps", type=int, default=None, help="Override inference.steps for the sweep.")
    parser.add_argument("--gen_length", type=int, default=None, help="Override inference.gen_length for the sweep.")
    parser.add_argument("--enable_remask", action="store_true", help="Enable remasking during the sweep (disabled by default to isolate routing effect).")
    parser.add_argument("--require_compiled_moe_ops", action="store_true", help="Fail fast if compiled vLLM MoE custom ops are unavailable.")
    parser.add_argument("--eval_dataset", default=None, help="Override data.eval_dataset.")
    parser.add_argument("--eval_dataset_config", default=None, help="Override data.eval_dataset_config.")
    parser.add_argument("--eval_split", default=None, help="Override data.eval_split.")
    parser.add_argument("--save_predictions", action="store_true", help="Save a bounded set of per-sample responses for each tau point.")
    parser.add_argument("--max_saved_predictions", type=int, default=None, help="Maximum saved predictions per run (hard-capped at 50).")
    parser.add_argument("--no_resume", action="store_true", help="Always recompute tau points even if completed eval artifacts already exist.")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    tau_values = parse_float_list(args.tau_r_values, label="tau_r")
    base_run_name = base_cfg.get("logging", {}).get("run_name", Path(args.config).stem)
    sweep_name = args.sweep_name or f"{base_run_name}_tau_sweep"
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "sweeps" / sweep_name
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(args.checkpoint, base_cfg.get("logging", {}).get("output_dir", ""))
    _apply_training_free_blockwise_defaults(
        base_cfg,
        checkpoint_path=checkpoint_path,
    )
    schedule = str(base_cfg.get("inference", {}).get("speculative_schedule", "aoae")).strip().lower()
    if args.require_compiled_moe_ops:
        ensure_vllm_moe_runtime(strict=True, verbose=is_global_rank_zero(), allow_python_fallback=False)

    if is_global_rank_zero():
        if checkpoint_path:
            print(f"Using checkpoint: {checkpoint_path}")
        elif schedule == "llada21_block":
            print("No checkpoint found; defaulting to the training-free blockwise LLaDA2.1 schedule.")
        else:
            print("No checkpoint found; sweep will use the confidence-guided default heuristic policy.")

    note_contains = args.summary_note_contains
    if note_contains is None and args.summary_method == "Speculative-AOAE":
        note_contains = f"tau_pi={args.policy_temperature}"

    cached_rows: Dict[str, Dict[str, Any]] = {}
    pending_cfgs: Dict[str, Dict[str, Any]] = {}
    pending_tau_values: List[float] = []

    for tau_r in tau_values:
        tau_dir_slug = tau_slug(tau_r)
        run_dir = output_root / tau_dir_slug
        cfg = _build_tau_cfg(base_cfg, args, tau_r, run_dir, sweep_name)
        _override_prediction_saving(cfg, args)
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
                    row = select_summary_row(saved_results, args.summary_method, note_contains)
                except Exception:
                    saved_results = None
                else:
                    cached_rows[tau_dir_slug] = _build_tau_summary_row(
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
        pending_cfgs[tau_dir_slug] = cfg
        pending_tau_values.append(tau_r)

    import torch

    shared_dual_model = None
    shared_eval_ds = None
    base_model_cfg = base_cfg.get("base_model", {}) or {}
    can_preload_model = bool(base_model_cfg.get("name_or_path"))
    if pending_tau_values and can_preload_model and (args.mode == "speculative" or base_model_cfg.get("backend") == "dual"):
        candidate_model = None
        try:
            from .models.dual_model import DualModelWrapper

            if is_global_rank_zero():
                print("Loading dual model ONCE for sweep reuse...")
            first_pending_slug = tau_slug(pending_tau_values[0])
            candidate_model = DualModelWrapper(copy.deepcopy(pending_cfgs[first_pending_slug]))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            shared_dual_model = candidate_model.to(device)
        except Exception as exc:
            _close_preloaded_runtime(candidate_model)
            shared_dual_model = None
            if is_global_rank_zero():
                print(f"Dual-model preload skipped; falling back to per-run evaluation: {exc}")

    dc = base_cfg.get("data", {})
    if args.eval_dataset is not None:
        dc = dict(dc)
        dc["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        dc = dict(dc)
        dc["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        dc = dict(dc)
        dc["eval_split"] = args.eval_split
    if pending_tau_values and dc.get("eval_dataset") and is_global_rank_zero():
        print(f"Loading eval dataset: {dc.get('eval_dataset', '')}...")
    if pending_tau_values and dc.get("eval_dataset"):
        shared_eval_ds = _load_eval_dataset(dc)

    summary_rows: List[Dict[str, Any]] = []
    for idx, tau_r in enumerate(tau_values):
        tau_dir_slug = tau_slug(tau_r)
        if tau_dir_slug in cached_rows:
            summary_rows.append(cached_rows[tau_dir_slug])
            continue

        cfg = pending_cfgs[tau_dir_slug]
        if is_global_rank_zero():
            print(f"\n=== tau_r = {tau_r} ===")
        run_baselines = args.run_baselines and (args.keep_baselines_every_run or idx == 0)
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
        row = select_summary_row(results, args.summary_method, note_contains)
        summary_rows.append(_build_tau_summary_row(cfg, row, tau_r, args.policy_temperature, checkpoint_path))

    json_path = output_root / "tau_sweep_summary.json"
    csv_path = output_root / "tau_sweep_summary.csv"
    md_path = output_root / "tau_sweep_summary.md"
    if is_global_rank_zero():
        with json_path.open("w") as f:
            json.dump(summary_rows, f, indent=2)
        write_csv(summary_rows, csv_path)
        write_markdown(summary_rows, md_path)
        print(f"\nSweep summary written to {json_path}")
        print(f"Sweep summary written to {csv_path}")
        print(f"Sweep summary written to {md_path}")
        _plot_tau_sweep(summary_rows, output_root)

    _close_preloaded_runtime(shared_dual_model)


def routing_sweep_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the routing-only tau_r sweep.")
    parser.add_argument("--hard_config", default="configs/llada21_hard.yaml", help="Hard-routing config.")
    parser.add_argument("--soft_config", default="configs/llada21_soft.yaml", help="Soft-routing config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint if you want AOAE rows in standard mode.")
    parser.add_argument("--tau_r_values", default="0.0001,0.001,0.005,0.01,0.02,0.05", help="Comma-separated positive tau_r values for the soft-routing sweep.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional evaluation cap.")
    parser.add_argument("--summary_method", default="llada21_quality_mode", help="Eval method to summarize/plot.")
    parser.add_argument("--summary_note_contains", default=None, help="Optional config_note substring for row selection.")
    parser.add_argument("--output_root", default=None, help="Sweep output root. Defaults under outputs/sweeps/.")
    parser.add_argument("--sweep_name", default=None, help="Short name for this sweep.")
    parser.add_argument("--eval_dataset", default=None, help="Override data.eval_dataset.")
    parser.add_argument("--eval_dataset_config", default=None, help="Override data.eval_dataset_config.")
    parser.add_argument("--eval_split", default=None, help="Override data.eval_split.")
    parser.add_argument("--save_predictions", action="store_true", help="Save a bounded set of per-sample responses for each routing run.")
    parser.add_argument("--max_saved_predictions", type=int, default=None, help="Maximum saved predictions per run (hard-capped at 50).")
    args = parser.parse_args(argv)

    tau_values = parse_float_list(args.tau_r_values, label="tau_r")
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

    hard_cfg = copy.deepcopy(hard_cfg_template)
    _override_eval_data(hard_cfg, args)
    _override_prediction_saving(hard_cfg, args)
    hard_cfg.setdefault("inference", {})["disable_remask"] = True
    hard_dir = output_root / "hard"
    hard_cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_hard"
    hard_cfg["logging"]["output_dir"] = str(hard_dir)
    if is_global_rank_zero():
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
            _result_to_routing_row(
                result,
                routing_label="hard",
                routing_mode="hard",
                tau_r=0.0,
                backend=hard_cfg["base_model"]["backend"],
                model_name=hard_cfg["base_model"]["name_or_path"],
                eval_dataset=hard_cfg["data"]["eval_dataset"],
                eval_dataset_config=hard_cfg["data"].get("eval_dataset_config"),
                eval_split=hard_cfg["data"]["eval_split"],
                remask_enabled=not bool(hard_cfg.get("inference", {}).get("disable_remask", False)),
                reuse_signal_method=str(hard_cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")),
                reuse_signal_threshold=float(hard_cfg.get("inference", {}).get("reuse_signal", {}).get("threshold", 0.0)),
                output_dir=hard_cfg["logging"]["output_dir"],
                checkpoint_path=args.checkpoint,
            )
        )

    hard_summary = select_summary_row(hard_results, args.summary_method, args.summary_note_contains)
    summary_rows.append(
        _result_to_routing_row(
            hard_summary,
            routing_label="hard",
            routing_mode="hard",
            tau_r=0.0,
            backend=hard_cfg["base_model"]["backend"],
            model_name=hard_cfg["base_model"]["name_or_path"],
            eval_dataset=hard_cfg["data"]["eval_dataset"],
            eval_dataset_config=hard_cfg["data"].get("eval_dataset_config"),
            eval_split=hard_cfg["data"]["eval_split"],
            remask_enabled=not bool(hard_cfg.get("inference", {}).get("disable_remask", False)),
            reuse_signal_method=str(hard_cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")),
            reuse_signal_threshold=float(hard_cfg.get("inference", {}).get("reuse_signal", {}).get("threshold", 0.0)),
            output_dir=hard_cfg["logging"]["output_dir"],
            checkpoint_path=args.checkpoint,
        )
    )

    import torch

    shared_soft_model = None
    shared_soft_base_model = None
    shared_eval_ds = None
    soft_model_cfg = soft_cfg_template.get("base_model", {}) or {}
    can_preload_soft = bool(soft_model_cfg.get("name_or_path")) and soft_model_cfg.get("backend") == "dual"
    if can_preload_soft:
        candidate_model = None
        try:
            from .models.dual_model import DualModelWrapper

            if is_global_rank_zero():
                print("Loading soft-routing model ONCE for sweep reuse...")
            init_soft = copy.deepcopy(soft_cfg_template)
            _override_eval_data(init_soft, args)
            init_soft.setdefault("base_model", {})["routing_temperature"] = tau_values[0]
            candidate_model = DualModelWrapper(init_soft)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            shared_soft_model = candidate_model.to(device)
        except Exception as exc:
            _close_preloaded_runtime(candidate_model)
            shared_soft_model = None
            if is_global_rank_zero():
                print(f"Soft-model preload skipped; falling back to per-run evaluation: {exc}")
    elif bool(soft_model_cfg.get("name_or_path")) and soft_model_cfg.get("backend") == "soft_moe":
        candidate_model = None
        try:
            import torch

            from .models.base_model import LLaDABaseModel

            if is_global_rank_zero():
                print("Loading soft-routing base model ONCE for sweep reuse...")
            init_soft = copy.deepcopy(soft_cfg_template)
            _override_eval_data(init_soft, args)
            if init_soft.get("evaluation", {}).get("baseline_methods") is None:
                hard_methods = hard_cfg_template.get("evaluation", {}).get("baseline_methods")
                if hard_methods is not None:
                    init_soft.setdefault("evaluation", {})["baseline_methods"] = list(hard_methods)
            init_soft.setdefault("base_model", {})["routing_temperature"] = tau_values[0]
            candidate_model = LLaDABaseModel(init_soft)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            shared_soft_base_model = candidate_model.to(device)
        except Exception as exc:
            _close_preloaded_runtime(candidate_model)
            shared_soft_base_model = None
            if is_global_rank_zero():
                print(f"Soft-base-model preload skipped; falling back to per-run evaluation: {exc}")

    dc = soft_cfg_template.get("data", {})
    if args.eval_dataset is not None:
        dc = dict(dc)
        dc["eval_dataset"] = args.eval_dataset
    if args.eval_dataset_config is not None:
        dc = dict(dc)
        dc["eval_dataset_config"] = args.eval_dataset_config or None
    if args.eval_split is not None:
        dc = dict(dc)
        dc["eval_split"] = args.eval_split
    if dc.get("eval_dataset") and is_global_rank_zero():
        print(f"Loading eval dataset ONCE: {dc.get('eval_dataset', '')}...")
    if dc.get("eval_dataset"):
        shared_eval_ds = _load_eval_dataset(dc)

    for tau_r in tau_values:
        soft_cfg = copy.deepcopy(soft_cfg_template)
        _override_eval_data(soft_cfg, args)
        _override_prediction_saving(soft_cfg, args)
        soft_cfg.setdefault("inference", {})["disable_remask"] = True
        if soft_cfg.get("evaluation", {}).get("baseline_methods") is None:
            hard_methods = hard_cfg_template.get("evaluation", {}).get("baseline_methods")
            if hard_methods is not None:
                soft_cfg.setdefault("evaluation", {})["baseline_methods"] = list(hard_methods)
        soft_cfg.setdefault("base_model", {})["routing_temperature"] = tau_r
        tau_dir_slug = tau_slug(tau_r)
        soft_dir = output_root / tau_dir_slug
        soft_cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_{tau_dir_slug}"
        soft_cfg["logging"]["output_dir"] = str(soft_dir)

        if is_global_rank_zero():
            print(f"\n=== soft routing (tau_r = {tau_r}) ===")
        soft_results = eval_main(
            soft_cfg,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            mode="standard",
            config_path=args.soft_config,
            preloaded_dual_model=shared_soft_model,
            preloaded_eval_ds=shared_eval_ds,
            preloaded_base_model=shared_soft_base_model,
        )
        label = f"{tau_r:.4g}"
        for result in soft_results:
            full_rows.append(
                _result_to_routing_row(
                    result,
                    routing_label=label,
                    routing_mode="soft",
                    tau_r=tau_r,
                    backend=soft_cfg["base_model"]["backend"],
                    model_name=soft_cfg["base_model"]["name_or_path"],
                    eval_dataset=soft_cfg["data"]["eval_dataset"],
                    eval_dataset_config=soft_cfg["data"].get("eval_dataset_config"),
                    eval_split=soft_cfg["data"]["eval_split"],
                    remask_enabled=not bool(soft_cfg.get("inference", {}).get("disable_remask", False)),
                    reuse_signal_method=str(soft_cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")),
                    reuse_signal_threshold=float(soft_cfg.get("inference", {}).get("reuse_signal", {}).get("threshold", 0.0)),
                    output_dir=soft_cfg["logging"]["output_dir"],
                    checkpoint_path=args.checkpoint,
                )
            )
        soft_summary = select_summary_row(soft_results, args.summary_method, args.summary_note_contains)
        summary_rows.append(
            _result_to_routing_row(
                soft_summary,
                routing_label=label,
                routing_mode="soft",
                tau_r=tau_r,
                backend=soft_cfg["base_model"]["backend"],
                model_name=soft_cfg["base_model"]["name_or_path"],
                eval_dataset=soft_cfg["data"]["eval_dataset"],
                eval_dataset_config=soft_cfg["data"].get("eval_dataset_config"),
                eval_split=soft_cfg["data"]["eval_split"],
                remask_enabled=not bool(soft_cfg.get("inference", {}).get("disable_remask", False)),
                reuse_signal_method=str(soft_cfg.get("inference", {}).get("reuse_signal", {}).get("method", "argmax_match")),
                reuse_signal_threshold=float(soft_cfg.get("inference", {}).get("reuse_signal", {}).get("threshold", 0.0)),
                output_dir=soft_cfg["logging"]["output_dir"],
                checkpoint_path=args.checkpoint,
            )
        )

    _annotate_tradeoff(summary_rows)

    full_json = output_root / "routing_sweep_full.json"
    full_csv = output_root / "routing_sweep_full.csv"
    full_md = output_root / "routing_sweep_full.md"
    if is_global_rank_zero():
        with full_json.open("w") as f:
            json.dump(full_rows, f, indent=2)
        write_csv(full_rows, full_csv)
        write_markdown(full_rows, full_md)

        summary_json = output_root / "routing_sweep_summary.json"
        summary_csv = output_root / "routing_sweep_summary.csv"
        summary_md = output_root / "routing_sweep_summary.md"
        with summary_json.open("w") as f:
            json.dump(summary_rows, f, indent=2)
        write_csv(summary_rows, summary_csv)
        write_markdown(summary_rows, summary_md)

        print(f"\nRouting sweep full table written to {full_json}")
        print(f"Routing sweep full table written to {full_csv}")
        print(f"Routing sweep full table written to {full_md}")
        print(f"Routing sweep summary written to {summary_json}")
        print(f"Routing sweep summary written to {summary_csv}")
        print(f"Routing sweep summary written to {summary_md}")
        _plot_routing_summary(summary_rows, output_root, args.summary_method)

    _close_preloaded_runtime(shared_soft_model)
    _close_preloaded_runtime(shared_soft_base_model)


def reuse_signal_sweep_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run POC2 reuse-signal reliability sweep.")
    parser.add_argument("--config", default="configs/poc2.yaml")
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
    parser.add_argument("--save_predictions", action="store_true", help="Save a bounded set of per-sample responses for each reuse run.")
    parser.add_argument("--max_saved_predictions", type=int, default=None, help="Maximum saved predictions per run (hard-capped at 50).")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    checkpoint_path = _resolve_checkpoint(args.checkpoint, base_cfg.get("logging", {}).get("output_dir", ""))
    _apply_training_free_blockwise_defaults(
        base_cfg,
        checkpoint_path=checkpoint_path,
    )
    schedule = str(base_cfg.get("inference", {}).get("speculative_schedule", "aoae")).strip().lower()
    _override_eval_data(base_cfg, args)
    grid = _load_grid(base_cfg)
    run_name = base_cfg.get("logging", {}).get("run_name", Path(args.config).stem)
    sweep_name = args.sweep_name or f"{run_name}_reuse_signal_sweep"
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "sweeps" / sweep_name
    output_root.mkdir(parents=True, exist_ok=True)

    if is_global_rank_zero():
        if checkpoint_path:
            print(f"Using checkpoint: {checkpoint_path}")
        elif schedule == "llada21_block":
            print("No checkpoint found; defaulting to the training-free blockwise LLaDA2.1 schedule.")
        else:
            print("No checkpoint found; reuse sweep will use the confidence-guided default heuristic policy.")

    rows: List[Dict[str, Any]] = []

    import torch

    shared_dual_model = None
    shared_eval_ds = None
    base_model_cfg = base_cfg.get("base_model", {}) or {}
    can_preload_model = bool(base_model_cfg.get("name_or_path"))
    if can_preload_model and (args.mode == "speculative" or base_model_cfg.get("backend") == "dual"):
        candidate_model = None
        try:
            from .models.dual_model import DualModelWrapper

            if is_global_rank_zero():
                print("Loading dual model ONCE for sweep reuse...")
            candidate_model = DualModelWrapper(base_cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            shared_dual_model = candidate_model.to(device)
        except Exception as exc:
            _close_preloaded_runtime(candidate_model)
            shared_dual_model = None
            if is_global_rank_zero():
                print(f"Dual-model preload skipped; falling back to per-run evaluation: {exc}")

    dc = base_cfg.get("data", {})
    if dc.get("eval_dataset") and is_global_rank_zero():
        print(f"Loading eval dataset: {dc.get('eval_dataset', '')}...")
    if dc.get("eval_dataset"):
        shared_eval_ds = _load_eval_dataset(dc)

    baseline_runs: List[Tuple[str, float]] = [("no_reuse", -1.0), ("oracle_reuse", 999.0), ("argmax_match", 0.0)]
    method_runs: List[Tuple[str, float]] = []
    for method, thresholds in grid.items():
        for threshold in thresholds:
            method_runs.append((method, float(threshold)))
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
        inf_cfg = cfg.setdefault("inference", {})
        reuse_signal = inf_cfg.setdefault("reuse_signal", {})
        if method == "no_reuse":
            reuse_signal["method"] = "js_divergence"
            reuse_signal["threshold"] = threshold
            label_method = "no_reuse"
        elif method == "oracle_reuse":
            reuse_signal["method"] = "js_divergence"
            reuse_signal["threshold"] = threshold
            label_method = "oracle_reuse"
        else:
            reuse_signal["method"] = method
            reuse_signal["threshold"] = float(threshold)
            label_method = method

        if args.disable_remask:
            inf_cfg["disable_remask"] = True
        cfg.setdefault("analysis", {})["track_kv_dynamics"] = True
        _override_prediction_saving(cfg, args)

        slug = _slug(label_method, threshold)
        run_dir = output_root / slug
        cfg.setdefault("logging", {})["run_name"] = f"{sweep_name}_{slug}"
        cfg["logging"]["output_dir"] = str(run_dir)

        if is_global_rank_zero():
            print(f"\n=== reuse={label_method} threshold={threshold} ===")
        results = eval_main(
            cfg,
            checkpoint_path=checkpoint_path,
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
        result = select_summary_row(results, args.summary_method, note_contains)
        thrash_rate = _read_thrash_rate(run_dir)
        kv_summary = _read_kv_dynamics_summary(run_dir)
        _print_kv_dynamics_trial_summary(run_dir, kv_summary)
        cache_ops = result.cache_commits + result.cache_invalidations
        cache_invalidation_rate = result.cache_invalidations / max(cache_ops, 1)
        rows.append(
            {
                "reuse_signal_method": label_method,
                "reuse_signal_threshold": f"{threshold:.6f}",
                "method": result.method,
                "config_note": result.config_note,
                "remask_enabled": int(not bool(inf_cfg.get("disable_remask", False))),
                "accuracy": f"{result.accuracy:.6f}",
                "tps": f"{result.avg_tokens_per_sec:.3f}",
                "avg_nfe": f"{result.avg_nfe:.1f}",
                "agreement_rate": f"{result.agreement_rate:.6f}",
                "cache_hit_rate": f"{result.cache_hit_rate:.6f}",
                "draft_accept_rate": f"{result.draft_accept_rate:.6f}",
                "mean_safe_reuse": f"{result.reuse_mean_safe:.6f}",
                "mean_js_divergence": f"{result.reuse_mean_js:.6f}",
                "thrash_rate_given_cached": f"{thrash_rate:.6f}",
                "cache_invalidation_rate": f"{cache_invalidation_rate:.6f}",
                "access_next_h_f1": f"{result.access_next_h_f1:.6f}",
                "access_next_h_spec_f1": f"{result.access_next_h_spec_f1:.6f}",
                "access_effective_budget": f"{result.access_effective_budget:.6f}",
                "total_samples": result.total_samples,
                "output_dir": str(run_dir),
                "kv_drift_measure": kv_summary.get("layer_drift_measure", "unavailable"),
                "exact_kv_drift_steps": f"{_safe_float(kv_summary.get('exact_kv_drift_steps', 0.0)):.2f}",
                "hidden_state_proxy_steps": f"{_safe_float(kv_summary.get('hidden_state_proxy_steps', 0.0)):.2f}",
                "mean_layer_drift_slope": f"{_safe_float(kv_summary.get('mean_layer_drift_slope', 0.0)):.6f}",
                "mean_off_by_one_drift_ratio": f"{_safe_float(kv_summary.get('mean_off_by_one_drift_ratio', 0.0)):.6f}",
                "mean_age_drift": kv_summary.get("mean_age_drift", {}),
                "per_layer_drift": kv_summary.get("per_layer_drift", []),
                "per_layer_drift_preview": _format_layer_drift_preview(
                    kv_summary.get("per_layer_drift", [])
                ),
            }
        )

    argmax_rows = [row for row in rows if row["reuse_signal_method"] == "argmax_match"]
    if len(argmax_rows) != 1:
        raise RuntimeError(f"Expected exactly one argmax baseline row, found {len(argmax_rows)}")
    argmax_acc = float(argmax_rows[0]["accuracy"])

    best_by_method: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row["accuracy_delta_vs_argmax"] = f"{float(row['accuracy']) - argmax_acc:+.6f}"
        method = row["reuse_signal_method"]
        current = best_by_method.get(method)
        if current is None or (float(row["accuracy"]), float(row["tps"])) > (float(current["accuracy"]), float(current["tps"])):
            best_by_method[method] = row
    for row in rows:
        best = best_by_method[row["reuse_signal_method"]]
        row["is_best_threshold"] = int(
            row["reuse_signal_threshold"] == best["reuse_signal_threshold"]
            and row["reuse_signal_method"] == best["reuse_signal_method"]
        )

    decisions = _decision_table(rows, argmax_acc=argmax_acc)

    full_json = output_root / "reuse_signal_sweep_full.json"
    full_csv = output_root / "reuse_signal_sweep_full.csv"
    full_md = output_root / "reuse_signal_sweep_full.md"
    if is_global_rank_zero():
        full_json.write_text(json.dumps(rows, indent=2))
        write_csv(rows, full_csv)
        write_markdown(rows, full_md)

        decision_json = output_root / "best_method_by_constraint.json"
        decision_csv = output_root / "best_method_by_constraint.csv"
        decision_md = output_root / "best_method_by_constraint.md"
        decision_json.write_text(json.dumps(decisions, indent=2))
        write_csv(decisions, decision_csv)
        write_markdown(decisions, decision_md)

        print(f"\nReuse-signal full table written to {full_json}")
        print(f"Reuse-signal full table written to {full_csv}")
        print(f"Reuse-signal full table written to {full_md}")
        print(f"Decision table written to {decision_json}")
        print(f"Decision table written to {decision_csv}")
        print(f"Decision table written to {decision_md}")
        _plot_reuse_pareto(rows, output_root)

    _close_preloaded_runtime(shared_dual_model)


def ablation_matrix_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run AOAE ablation matrix.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="speculative", choices=["standard", "speculative"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--summary_method", default="Speculative-AOAE")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--matrix_json", default=None, help="Optional custom matrix JSON file.")
    parser.add_argument("--keep_baselines", action="store_true", help="Run baseline methods for every ablation row.")
    parser.add_argument("--save_predictions", action="store_true", help="Save a bounded set of per-sample responses for each ablation row.")
    parser.add_argument("--max_saved_predictions", type=int, default=None, help="Maximum saved predictions per run (hard-capped at 50).")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    default_ablations: List[Dict[str, Any]] = [
        {"name": "baseline_default", "overrides": {}},
        {"name": "no_composed_prediction", "overrides": {"inference.compose_gamma": 0.0}},
        {"name": "no_agreement_feature", "overrides": {"policy.use_agreement_feature": False}},
        {"name": "disable_remask", "overrides": {"inference.disable_remask": True}},
        {"name": "disable_positional_cache", "overrides": {"inference.positional_cache.enabled": False}},
        {"name": "candidate_sliding_window", "overrides": {"inference.positional_cache.candidate_policy": "sliding_window"}},
        {"name": "candidate_confidence_topb", "overrides": {"inference.positional_cache.candidate_policy": "confidence_topb"}},
        {"name": "no_history_features", "overrides": {"policy.use_age_feature": False, "policy.use_last_action_feature": False}},
        {"name": "boundary_head_on", "overrides": {"policy.boundary_head.enabled": True, "policy.boundary_head.num_bins": 8}},
    ]

    if args.matrix_json:
        matrix = json.loads(Path(args.matrix_json).read_text())
        if not isinstance(matrix, list):
            raise ValueError("--matrix_json must be a list of {name, overrides}")
    else:
        matrix = copy.deepcopy(default_ablations)

    run_name = base_cfg.get("logging", {}).get("run_name", Path(args.config).stem)
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "ablations" / f"{run_name}_matrix"
    output_root.mkdir(parents=True, exist_ok=True)

    import torch

    shared_dual_model = None
    shared_eval_ds = None
    base_model_cfg = base_cfg.get("base_model", {}) or {}
    can_preload_model = bool(base_model_cfg.get("name_or_path"))
    if can_preload_model and (args.mode == "speculative" or base_model_cfg.get("backend") == "dual"):
        candidate_model = None
        try:
            from .models.dual_model import DualModelWrapper

            if is_global_rank_zero():
                print("Loading dual model ONCE for ablation sweep reuse...")
            candidate_model = DualModelWrapper(base_cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            shared_dual_model = candidate_model.to(device)
        except Exception as exc:
            _close_preloaded_runtime(candidate_model)
            shared_dual_model = None
            if is_global_rank_zero():
                print(f"Dual-model preload skipped; falling back to per-run evaluation: {exc}")

    dc = base_cfg.get("data", {})
    if dc.get("eval_dataset") and is_global_rank_zero():
        print(f"Loading eval dataset: {dc.get('eval_dataset', '')}...")
    if dc.get("eval_dataset"):
        shared_eval_ds = _load_eval_dataset(dc)

    rows: List[Dict[str, Any]] = []
    for spec in matrix:
        name = str(spec["name"])
        overrides = dict(spec.get("overrides", {}))
        cfg = copy.deepcopy(base_cfg)
        for key, value in overrides.items():
            set_nested(cfg, str(key), value)
        _override_prediction_saving(cfg, args)
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
        row = select_summary_row(results, args.summary_method)
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
        write_csv(rows, full_csv)
        write_markdown(rows, full_md)
        print(f"\nAblation matrix written to {full_json}")
        print(f"Ablation matrix written to {full_csv}")
        print(f"Ablation matrix written to {full_md}")

    _close_preloaded_runtime(shared_dual_model)


def paper_suite_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the paper-aligned AOAE experiment suite.")
    parser.add_argument("--config", default="configs/paper.yaml", help="Base config used for all stages unless overridden.")
    parser.add_argument("--poc1_config", default=None, help="Optional config override for the routing-temperature sweep.")
    parser.add_argument("--poc2_config", default=None, help="Optional config override for the reuse-signal sweep.")
    parser.add_argument("--ablation_config", default=None, help="Optional config override for the ablation matrix.")
    parser.add_argument("--hard_config", default="configs/llada21_hard.yaml", help="Hard-routing config used for the routing sweep.")
    parser.add_argument("--soft_config", default="configs/llada21_soft.yaml", help="Soft-routing config used for the routing sweep.")
    parser.add_argument("--checkpoint", default=None, help="Optional policy checkpoint passed to all stages.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional evaluation cap applied to all stages.")
    parser.add_argument("--output_root", default=None, help="Suite output root. Defaults under outputs/paper_suite/.")
    parser.add_argument("--tau_r_values", default="0.0001,0.001,0.005,0.01,0.02,0.05", help="Paper sweep for PoC 1.")
    parser.add_argument("--policy_temperature", type=float, default=1.0, help="Policy temperature used for sweep summaries.")
    parser.add_argument("--skip_poc1", action="store_true", help="Skip the soft-routing tradeoff sweep.")
    parser.add_argument("--skip_routing", action="store_true", help="Skip the hard-vs-soft routing sweep.")
    parser.add_argument("--skip_poc2", action="store_true", help="Skip the reuse-signal sweep.")
    parser.add_argument("--skip_ablations", action="store_true", help="Skip the ablation matrix.")
    parser.add_argument("--skip_table", action="store_true", help="Skip aggregated comparison-table generation.")
    parser.add_argument("--skip_kv_summary", action="store_true", help="Skip aggregated KV-dynamics summary generation.")
    parser.add_argument("--poc1_enable_remask", action="store_true", help="Keep remasking enabled during PoC 1. Default is disabled to isolate routing.")
    parser.add_argument("--poc2_disable_remask", action="store_true", help="Disable remasking during PoC 2 for cleaner reuse-signal accounting.")
    parser.add_argument("--save_predictions", action="store_true", help="Save bounded prediction artifacts for suite evaluation stages.")
    parser.add_argument("--max_saved_predictions", type=int, default=None, help="Maximum saved predictions per run (hard-capped at 50).")
    args = parser.parse_args(argv)

    suite_name = Path(args.config).stem
    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs" / "paper_suite" / suite_name
    output_root.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []

    def record_stage(name: str, config_path: str, stage_root: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        entry = {"stage": name, "config": config_path, "output_root": str(stage_root)}
        if extra:
            entry.update(extra)
        summary.append(entry)

    if not args.skip_poc1:
        poc1_root = output_root / "poc1"
        poc1_root.mkdir(parents=True, exist_ok=True)
        poc1_argv = [
            "--config", args.poc1_config or "configs/poc1.yaml",
            "--tau_r_values", args.tau_r_values,
            "--policy_temperature", str(args.policy_temperature),
            "--output_root", str(poc1_root),
        ]
        if args.checkpoint:
            poc1_argv.extend(["--checkpoint", args.checkpoint])
        if args.max_samples is not None:
            poc1_argv.extend(["--max_samples", str(args.max_samples)])
        if args.poc1_enable_remask:
            poc1_argv.append("--enable_remask")
        if args.save_predictions:
            poc1_argv.append("--save_predictions")
        if args.max_saved_predictions is not None:
            poc1_argv.extend(["--max_saved_predictions", str(args.max_saved_predictions)])
        _invoke_main(tau_sweep_main, "tau-sweep", poc1_argv)
        record_stage("poc1", args.poc1_config or "configs/poc1.yaml", poc1_root, {"tau_r_values": args.tau_r_values})

    if not args.skip_routing:
        routing_root = output_root / "routing"
        routing_root.mkdir(parents=True, exist_ok=True)
        routing_argv = [
            "--hard_config", args.hard_config,
            "--soft_config", args.soft_config,
            "--tau_r_values", args.tau_r_values,
            "--output_root", str(routing_root),
        ]
        if args.checkpoint:
            routing_argv.extend(["--checkpoint", args.checkpoint])
        if args.max_samples is not None:
            routing_argv.extend(["--max_samples", str(args.max_samples)])
        if args.save_predictions:
            routing_argv.append("--save_predictions")
        if args.max_saved_predictions is not None:
            routing_argv.extend(["--max_saved_predictions", str(args.max_saved_predictions)])
        _invoke_main(routing_sweep_main, "routing-sweep", routing_argv)
        record_stage(
            "routing",
            f"{args.hard_config}::{args.soft_config}",
            routing_root,
            {"tau_r_values": args.tau_r_values},
        )

    if not args.skip_poc2:
        poc2_root = output_root / "poc2"
        poc2_root.mkdir(parents=True, exist_ok=True)
        poc2_argv = [
            "--config", args.poc2_config or "configs/poc2.yaml",
            "--policy_temperature", str(args.policy_temperature),
            "--output_root", str(poc2_root),
        ]
        if args.checkpoint:
            poc2_argv.extend(["--checkpoint", args.checkpoint])
        if args.max_samples is not None:
            poc2_argv.extend(["--max_samples", str(args.max_samples)])
        if args.poc2_disable_remask:
            poc2_argv.append("--disable_remask")
        if args.save_predictions:
            poc2_argv.append("--save_predictions")
        if args.max_saved_predictions is not None:
            poc2_argv.extend(["--max_saved_predictions", str(args.max_saved_predictions)])
        _invoke_main(reuse_signal_sweep_main, "reuse-sweep", poc2_argv)
        record_stage("poc2", args.poc2_config or "configs/poc2.yaml", poc2_root, {"disable_remask": args.poc2_disable_remask})

    if not args.skip_ablations:
        ablation_root = output_root / "ablations"
        ablation_root.mkdir(parents=True, exist_ok=True)
        ablation_argv = [
            "--config", args.ablation_config or args.config,
            "--output_root", str(ablation_root),
        ]
        if args.checkpoint:
            ablation_argv.extend(["--checkpoint", args.checkpoint])
        if args.max_samples is not None:
            ablation_argv.extend(["--max_samples", str(args.max_samples)])
        if args.save_predictions:
            ablation_argv.append("--save_predictions")
        if args.max_saved_predictions is not None:
            ablation_argv.extend(["--max_saved_predictions", str(args.max_saved_predictions)])
        _invoke_main(ablation_matrix_main, "ablations", ablation_argv)
        record_stage("ablations", args.ablation_config or args.config, ablation_root)

    if not args.skip_table:
        from .reporting import comparison_table_main

        csv_path = output_root / "paper_comparison_table.csv"
        md_path = output_root / "paper_comparison_table.md"
        table_argv = [
            "--glob", str(output_root / "**" / "eval_results.json"),
            "--csv", str(csv_path),
            "--md", str(md_path),
        ]
        _invoke_main(comparison_table_main, "comparison-table", table_argv)
        record_stage("comparison_table", args.config, output_root, {"csv": str(csv_path), "md": str(md_path)})

    if not args.skip_kv_summary:
        from .reporting import kv_summary_main

        csv_path = output_root / "paper_kv_summary.csv"
        md_path = output_root / "paper_kv_summary.md"
        kv_argv = [
            "--glob", str(output_root / "**" / "kv_dynamics_summary.json"),
            "--csv", str(csv_path),
            "--md", str(md_path),
        ]
        _invoke_main(kv_summary_main, "kv-summary", kv_argv)
        record_stage("kv_summary", args.config, output_root, {"csv": str(csv_path), "md": str(md_path)})

    summary_path = output_root / "paper_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if is_global_rank_zero():
        print(f"Paper suite summary written to {summary_path}")
