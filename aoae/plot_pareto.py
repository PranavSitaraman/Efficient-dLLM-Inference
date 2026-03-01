"""
Pareto Curve Plotting Utility.

Reads eval_results.json and generates accuracy-vs-throughput Pareto curves
comparing AOAE (at various tau_pi) against baselines.

Usage:
    python3 -m aoae.plot_pareto --results outputs/eval_results.json --output outputs/pareto.png
"""

import json
import argparse
import os
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def plot_pareto(results: List[Dict], output_path: str):
    """Generate accuracy vs tokens/sec Pareto plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Separate AOAE and baseline results
    aoae_pts = [(r["avg_tokens_per_sec"], r["accuracy"], r.get("config_note", ""))
                for r in results if r["method"] == "AOAE"]
    baseline_pts = [(r["avg_tokens_per_sec"], r["accuracy"], r["method"])
                    for r in results if r["method"] != "AOAE"]

    # Plot baselines as distinct markers
    markers = {"uniform": "s", "confidence_s_mode": "^", "confidence_q_mode": "D"}
    colors = {"uniform": "#888888", "confidence_s_mode": "#E07020", "confidence_q_mode": "#2070E0"}

    for tps, acc, method in baseline_pts:
        ax.scatter(
            tps, acc,
            marker=markers.get(method, "o"),
            color=colors.get(method, "#444444"),
            s=120, zorder=5, edgecolors="black", linewidths=0.8,
            label=method.replace("_", " ").title(),
        )

    # Plot AOAE Pareto curve
    if aoae_pts:
        aoae_pts.sort(key=lambda x: x[0])
        tps_list = [p[0] for p in aoae_pts]
        acc_list = [p[1] for p in aoae_pts]
        labels = [p[2] for p in aoae_pts]

        ax.plot(tps_list, acc_list, "o-", color="#10A050", linewidth=2,
                markersize=8, zorder=4, label="AOAE (ours)")

        for tps, acc, lbl in aoae_pts:
            ax.annotate(lbl, (tps, acc), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="#10A050")

    ax.set_xlabel("Tokens / sec", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Throughput — GSM8K", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Pareto plot saved to {output_path}")


def print_table(results: List[Dict]):
    """Print a text summary table to stdout."""
    print()
    print(f"{'Method':<30} {'Accuracy':>10} {'TPS':>10} {'NFE':>8} {'Note':<25}")
    print("-" * 85)
    for r in results:
        print(f"{r['method']:<30} {r['accuracy']:>10.4f} "
              f"{r['avg_tokens_per_sec']:>10.1f} {r['avg_nfe']:>8.0f} "
              f"{r.get('config_note', ''):<25}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot AOAE Pareto curves")
    parser.add_argument("--results", type=str, default="outputs/eval_results.json")
    parser.add_argument("--output", type=str, default="outputs/pareto.png")
    args = parser.parse_args()

    results = load_results(args.results)
    print_table(results)
    plot_pareto(results, args.output)


if __name__ == "__main__":
    main()
