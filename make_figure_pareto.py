"""
make_figure_pareto.py — AOAE paper figure: Accuracy vs Throughput (GSM8K full eval)

Produces:
  figures/pareto_gsm8k.pdf
  figures/pareto_gsm8k.png

Data source: warmstart-rl-results.md (full-scale evals, 1319 GSM8K test samples)

TPS definition: total visible output tokens / wall-clock inference time (seconds),
measured on a single NVIDIA H200, tp_size=1.  This is a wall-clock throughput
metric (not theoretical FLOPs/s), so it includes all scheduling and MoE overhead.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Color / style palette
#
# Design intent:
#   • Baselines    → grey squares  (neutral, background)
#   • Post-GRPO    → filled circle, per-family color  (V4=blue, V5 kl01=green, V5 kl03=red)
#   • Warmstart    → open circle, same color as its GRPO descendant(s) — "same family, pre-RL"
#     (V5 warmstart is the ancestor of both V5 kl runs; shown in a neutral purple so it
#      reads as "shared parent" rather than belonging to one child)
# ---------------------------------------------------------------------------
v4_color     = "#1f77b4"   # blue  — V4 GRPO
v5kl01_color = "#2ca02c"   # green — V5 GRPO kl=0.01
v5kl03_color = "#d62728"   # red   — V5 GRPO kl=0.03
ws_color     = "#7b4f9e"   # purple — V5 Warmstart (shared ancestor of kl=0.01 and kl=0.03)
bl_color     = "#999999"   # grey  — LLaDA 2.1 baselines

# Marker convention:
MARKER_GRPO     = "o"   # filled circle — all post-GRPO points
MARKER_WS       = "o"   # same shape, but drawn open (facecolor=none) — warmstart
MARKER_BASELINE = "s"   # square — baselines

# ---------------------------------------------------------------------------
# Data — full-scale GSM8K eval points (1319 samples)
# Per family: keep only the best Pareto-frontier point(s).
# τ=1.0 variants are dominated by τ=0.5 in every family → omitted.
# ---------------------------------------------------------------------------

# ── Block baselines (LLaDA 2.1 official, block_length=32) ──
baselines = [
    # (label, acc, tps, label_offset)
    ("LLaDA 2.1\nSpeed mode",   74.60, 72.6, ( 4,  4)),
    ("LLaDA 2.1\nQuality mode", 77.26, 64.7, ( 4,  4)),
]

# ── V4 QBal GRPO (scalar_only, best checkpoint 625 steps) ──
# τ=0.5 lossy: 79.61% / 72.6 TPS  → Pareto dominant
# τ=0.5 lossless: 75.97% / 77.2 TPS → higher TPS at lower acc; show as frontier
v4_grpo = [
    dict(label="V4 GRPO (soft)",    acc=79.61, tps=72.6, loff=( 4,-12)),
    dict(label="V4 GRPO (hard)", acc=75.97, tps=77.2, loff=( 4,  4)),
]

# ── V5 Warmstart (hybrid features, pre-GRPO) — shared anchor for both kl runs ──
v5ws = dict(acc=78.24, tps=44.8)

# ── V5 GRPO kl=0.01 (best checkpoint) ──
# τ=0.5 lossy: 79.08% / 69.4 TPS
# τ=0.5 lossless: 79.38% / 75.2 TPS  (higher TPS, negligibly higher acc)
v5kl01 = [
    dict(label="V5 kl=0.01\n(soft)",    acc=79.08, tps=69.4, loff=(-4, -14)),
    dict(label="V5 kl=0.01\n(hard)", acc=79.38, tps=75.2, loff=( 4,   4)),
]

# ── V5 GRPO kl=0.03 (best checkpoint) ──
# τ=0.5 lossy: 79.68% / 70.0 TPS  ← overall best
# τ=0.5 lossless: 79.38% / 75.0 TPS
v5kl03 = [
    dict(label="V5 kl=0.03\n(soft)",    acc=79.68, tps=70.0, loff=( 4,   4)),
    dict(label="V5 kl=0.03\n(hard)", acc=79.38, tps=75.0, loff=( 4, -14)),
]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 5.4))

# ── Baselines — grey squares ──
for (lbl, acc, tps, loff) in baselines:
    ax.scatter(tps, acc, color=bl_color, marker=MARKER_BASELINE, s=80, zorder=4)
    ax.annotate(lbl, (tps, acc), fontsize=7.5, color=bl_color,
                ha="left", va="bottom",
                xytext=loff, textcoords="offset points")

# ── Helper: plot a GRPO family (filled circles) with a thin Pareto line ──
def plot_family(pts, color):
    spts = sorted(pts, key=lambda p: p["tps"])
    ax.plot([p["tps"] for p in spts], [p["acc"] for p in spts],
            color=color, lw=1.0, alpha=0.35, zorder=2)
    for p in spts:
        ax.scatter(p["tps"], p["acc"],
                   color=color, marker=MARKER_GRPO, s=95, zorder=5)
        dx, dy = p["loff"]
        ax.annotate(p["label"], (p["tps"], p["acc"]),
                    fontsize=7.5, color=color,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    xytext=(dx, dy), textcoords="offset points")

plot_family(v4_grpo,  v4_color)
plot_family(v5kl01,   v5kl01_color)
plot_family(v5kl03,   v5kl03_color)

# ── V5 Warmstart — open circle (same shape as GRPO, purple = shared ancestor) ──
ax.scatter(v5ws["tps"], v5ws["acc"],
           facecolors="none", edgecolors=ws_color, linewidths=2.0,
           marker=MARKER_WS, s=110, zorder=5)
ax.annotate("V5 Warmstart\n(pre-GRPO)", (v5ws["tps"], v5ws["acc"]),
            fontsize=7.5, color=ws_color, ha="right", va="top",
            xytext=(-5, -5), textcoords="offset points")

# ---------------------------------------------------------------------------
# Dotted arrows: V5 Warmstart → V5 GRPO (pre/post-RL pairs)
# One arrow per kl run, pointing to the lossy (primary) point.
# Light grey so they annotate without cluttering.
# ---------------------------------------------------------------------------
arrow_style = dict(
    arrowstyle="-|>",
    color="#cccccc",
    lw=1.1,
    mutation_scale=10,
)

# kl=0.01: arc bends upward (rad > 0)
ax.annotate("", xy=(v5kl01[0]["tps"], v5kl01[0]["acc"]),
            xytext=(v5ws["tps"], v5ws["acc"]),
            arrowprops=dict(**arrow_style,
                            connectionstyle="arc3,rad=-0.20",
                            linestyle=(0, (3, 3))),
            zorder=3)

# kl=0.03: arc bends downward (rad < 0) so the two arrows diverge visually
ax.annotate("", xy=(v5kl03[0]["tps"], v5kl03[0]["acc"]),
            xytext=(v5ws["tps"], v5ws["acc"]),
            arrowprops=dict(**arrow_style,
                            connectionstyle="arc3,rad=0.20",
                            linestyle=(0, (3, 3))),
            zorder=3)

# Single "RL-GRPO" label between the two arrow arcs
mid_x = (v5ws["tps"] + v5kl01[0]["tps"]) / 2 + 1
mid_y = (v5ws["acc"] + v5kl01[0]["acc"]) / 2 + 0.7
ax.text(mid_x, mid_y, "RL-GRPO", fontsize=7, color="#bbbbbb",
        ha="center", va="bottom", style="italic")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
L2D = matplotlib.lines.Line2D
legend_elements = [
    # Baselines — grey square
    L2D([0], [0], marker=MARKER_BASELINE, color="w",
        markerfacecolor=bl_color, markersize=8, label="LLaDA 2.1 (block baseline)"),
    # Warmstart — open circle, purple
    L2D([0], [0], marker=MARKER_WS, color="w",
        markerfacecolor="none", markeredgecolor=ws_color,
        markeredgewidth=2.0, markersize=8, label="V5 Warmstart (pre-GRPO)"),
    # Post-GRPO — filled circle, per colour
    L2D([0], [0], marker=MARKER_GRPO, color="w",
        markerfacecolor=v4_color, markersize=8, label="V4 GRPO (scalar)"),
    L2D([0], [0], marker=MARKER_GRPO, color="w",
        markerfacecolor=v5kl01_color, markersize=8, label="V5 GRPO  kl=0.01"),
    L2D([0], [0], marker=MARKER_GRPO, color="w",
        markerfacecolor=v5kl03_color, markersize=8, label="V5 GRPO  kl=0.03"),
]
ax.legend(handles=legend_elements, fontsize=7.5, loc="lower left",
          framealpha=0.88, edgecolor="#cccccc")

# ---------------------------------------------------------------------------
# Axes cosmetics
# ---------------------------------------------------------------------------
ax.set_xlabel("Throughput  (tokens / second)", fontsize=10)
ax.set_ylabel("GSM8K Accuracy", fontsize=10)
ax.set_title("AOAE — Accuracy vs. Throughput  (GSM8K, 1319 samples)", fontsize=11)
ax.set_xlim(32, 92)
ax.set_ylim(73.0, 81.5)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.grid(True, linestyle="--", alpha=0.30)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig("figures/pareto_gsm8k.pdf", bbox_inches="tight")
fig.savefig("figures/pareto_gsm8k.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved figures/pareto_gsm8k.pdf + .png")
