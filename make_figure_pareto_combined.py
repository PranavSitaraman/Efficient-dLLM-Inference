"""
make_figure_pareto_combined.py — AOAE paper figure: Accuracy vs Throughput (GSM8K full eval)
                                  Combined view: block baselines + any-order baselines + Fast-dLLM

Produces:
  figures/pareto_gsm8k_combined.pdf
  figures/pareto_gsm8k_combined.png

Data sources:
  - Block baselines: outputs/eval_full_block_gsm8k/eval_results.json (1319 samples)
  - Any-order baselines: outputs/eval_full_anyorder_gsm8k/eval_results.json (1319 samples)
  - Fast-dLLM: estimated from 50-sample pilot (paper/5_3_soft_gating_hurts_verifier-performance.md)
  - GRPO families: outputs/eval_full_*/eval_results.json (1319 samples)

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
#   • Block baselines      → grey squares  (neutral, background)
#   • Any-order baselines  → grey triangles (same neutral family, different mode)
#   • Fast-dLLM            → grey diamond  (external reference, small-sample estimate)
#   • Post-GRPO            → filled circle, per-family color  (V4=blue, V5 kl01=green, V5 kl03=red)
#   • Warmstart            → open circle, purple — "shared parent, pre-RL"
# ---------------------------------------------------------------------------
v4_color     = "#1f77b4"   # blue  — V4 GRPO
v5kl01_color = "#2ca02c"   # green — V5 GRPO kl=0.01
v5kl03_color = "#d62728"   # red   — V5 GRPO kl=0.03
ws_color     = "#7b4f9e"   # purple — V5 Warmstart (shared ancestor of kl=0.01 and kl=0.03)
bl_color     = "#999999"   # grey  — LLaDA 2.1 block baselines
ao_color     = "#aaaaaa"   # lighter grey — any-order LLaDA baselines
fd_color     = "#bbbbbb"   # faint grey — Fast-dLLM reference

# Marker convention:
MARKER_GRPO     = "o"   # filled circle — all post-GRPO points
MARKER_WS       = "o"   # same shape, but drawn open (facecolor=none) — warmstart
MARKER_BASELINE = "s"   # square — block baselines
MARKER_ANYORDER = "^"   # triangle — any-order LLaDA baselines
MARKER_FASTDLLM = "D"   # diamond — Fast-dLLM reference

# ---------------------------------------------------------------------------
# Data — full-scale GSM8K eval points (1319 samples)
# Per family: keep only the best Pareto-frontier point(s).
# τ=1.0 variants are dominated by τ=0.5 in every family → omitted.
# ---------------------------------------------------------------------------

# ── Block baselines (LLaDA 2.1 official, block_length=32) ──
baselines = [
    # (label, acc, tps, label_offset)
    ("LLaDA 2.1\nSpeed mode",   74.60, 72.6, ( 4, -14)),
    ("LLaDA 2.1\nQuality mode", 77.26, 64.7, (-4, -14)),
]

# ── Any-order baselines — excluded from figure ──
# LLaDA 2.1 any-order accuracy is too low (~15%) to be a useful reference
# on this plot and compresses the y-axis scale.  Real values (1319 samples,
# suppress_eos=False, eos_steady_passes=3, max_steps=64):
#   Speed:   acc=15.01%, tps=228.7
#   Quality: acc≈18–20%, tps≈220–240  (run in progress)
anyorder_baselines = []  # not plotted

# ── Fast-dLLM — excluded from figure ──
# (Wu et al. 2025, 50-sample est.) 62.0% acc, 46.5 TPS — Pareto-dominated by
# both LLaDA 2.1 block baselines (higher accuracy AND higher throughput).
# fast_dllm = ("Fast-dLLM*", 62.0, 46.5, (4, -14))

# ── W/S-GRPO (scalar_only features, best checkpoint 625 steps) ──
v4_grpo = [
    dict(label="W/S-GRPO (soft)",  acc=79.61, tps=72.6, loff=( 4, -12)),
    dict(label="W/S-GRPO (hard)",  acc=75.97, tps=77.2, loff=( 4,   4)),
]

# ── W/S+H_t Warmstart (hybrid features, pre-GRPO) — shared anchor for both kl runs ──
v5ws = dict(acc=78.24, tps=44.8)

# ── W/S-GRPO+H_t kl=0.01 (best checkpoint) ──
v5kl01 = [
    dict(label=r"W/S-GRPO+$H_t$ kl=0.01" + "\n(soft)",  acc=79.08, tps=69.4, loff=(-4, -14)),
    dict(label=r"W/S-GRPO+$H_t$ kl=0.01" + "\n(hard)",  acc=79.38, tps=75.2, loff=( 4,   4)),
]

# ── W/S-GRPO+H_t kl=0.03 (best checkpoint) ──
v5kl03 = [
    dict(label=r"W/S-GRPO+$H_t$ kl=0.03" + "\n(soft)",  acc=79.68, tps=70.0, loff=( 4,   4)),
    dict(label=r"W/S-GRPO+$H_t$ kl=0.03" + "\n(hard)",  acc=79.38, tps=75.0, loff=( 4, -14)),
]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.8))

# ── Block baselines — grey squares ──
for (lbl, acc, tps, loff) in baselines:
    ax.scatter(tps, acc, color=bl_color, marker=MARKER_BASELINE, s=80, zorder=4)
    dx, dy = loff
    ax.annotate(lbl, (tps, acc), fontsize=7.5, color=bl_color,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                xytext=loff, textcoords="offset points")

# ── Any-order baselines — grey triangles ──
for (lbl, acc, tps, loff) in anyorder_baselines:
    ax.scatter(tps, acc, color=ao_color, marker=MARKER_ANYORDER, s=80, zorder=4)
    dx, dy = loff
    ax.annotate(lbl, (tps, acc), fontsize=7.5, color=ao_color,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                xytext=loff, textcoords="offset points")


# ── Helper: plot a GRPO family (filled circles, no connecting line) ──
def plot_family(pts, color):
    for p in pts:
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
ax.annotate(r"W/S+$H_t$" + "\n(pre-GRPO)", (v5ws["tps"], v5ws["acc"]),
            fontsize=7.5, color=ws_color, ha="right", va="top",
            xytext=(-5, -5), textcoords="offset points")

# ---------------------------------------------------------------------------
# Dotted arrows: V5 Warmstart → V5 GRPO (pre/post-RL pairs)
# ---------------------------------------------------------------------------
arrow_style = dict(
    arrowstyle="-|>",
    color="#cccccc",
    lw=1.1,
    mutation_scale=10,
)

ax.annotate("", xy=(v5kl01[0]["tps"], v5kl01[0]["acc"]),
            xytext=(v5ws["tps"], v5ws["acc"]),
            arrowprops=dict(**arrow_style,
                            connectionstyle="arc3,rad=-0.20",
                            linestyle=(0, (3, 3))),
            zorder=3)

ax.annotate("", xy=(v5kl03[0]["tps"], v5kl03[0]["acc"]),
            xytext=(v5ws["tps"], v5ws["acc"]),
            arrowprops=dict(**arrow_style,
                            connectionstyle="arc3,rad=0.20",
                            linestyle=(0, (3, 3))),
            zorder=3)

mid_x = (v5ws["tps"] + v5kl01[0]["tps"]) / 2 + 1
mid_y = (v5ws["acc"] + v5kl01[0]["acc"]) / 2 + 0.7
ax.text(mid_x, mid_y, "RL-GRPO", fontsize=7, color="#bbbbbb",
        ha="center", va="bottom", style="italic")


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
L2D = matplotlib.lines.Line2D
legend_elements = [
    L2D([0], [0], marker=MARKER_BASELINE, color="w",
        markerfacecolor=bl_color, markersize=8, label="LLaDA 2.1 (block)"),
    L2D([0], [0], marker=MARKER_WS, color="w",
        markerfacecolor="none", markeredgecolor=ws_color,
        markeredgewidth=2.0, markersize=8, label=r"W/S+$H_t$ (pre-GRPO)"),
    L2D([0], [0], marker=MARKER_GRPO, color="w",
        markerfacecolor=v4_color, markersize=8, label="W/S-GRPO"),
    L2D([0], [0], marker=MARKER_GRPO, color="w",
        markerfacecolor=v5kl01_color, markersize=8, label=r"W/S-GRPO+$H_t$  kl=0.01"),
    L2D([0], [0], marker=MARKER_GRPO, color="w",
        markerfacecolor=v5kl03_color, markersize=8, label=r"W/S-GRPO+$H_t$  kl=0.03"),
]
ax.legend(handles=legend_elements, fontsize=7.5, loc="lower left",
          framealpha=0.88, edgecolor="#cccccc")

# ---------------------------------------------------------------------------
# Axes cosmetics
# ---------------------------------------------------------------------------
ax.set_xlabel("Throughput  (tokens / second)", fontsize=10)
ax.set_ylabel("GSM8K Accuracy", fontsize=10)
ax.set_title("Accuracy vs. Throughput  (GSM8K)", fontsize=11)
ax.set_xlim(30, 92)
ax.set_ylim(69.0, 82.0)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.grid(True, linestyle="--", alpha=0.30)
ax.set_axisbelow(True)


plt.tight_layout()
fig.savefig("figures/pareto_gsm8k_combined.pdf", bbox_inches="tight")
fig.savefig("figures/pareto_gsm8k_combined.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved figures/pareto_gsm8k_combined.pdf + .png")
