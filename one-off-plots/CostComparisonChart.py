"""
Cost comparison stacked bar chart (v3 detail) styled to match paper figures.

Run from the project root where uncertainty_sets_refactored is importable:
    python cost_chart.py

Outputs:
    Figures/cost_stacked_detail.pdf  (paper quality)
    Figures/cost_stacked_detail.png  (for Beamer)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from uncertainty_sets_refactored.plot_config import (
    setup_plotting,
    FONT_SIZES,
    IEEE_TWO_COL_WIDTH,
    FIGURE_DEFAULTS,
)

setup_plotting()

# ── Data ──────────────────────────────────────────────────────────────────────
processes = ["DAM", "DAM\nw/Res", "DAM\n+RUC", "DAM w/Res\n+RUC", "ARUC"]
energy  = [2183, 2186, 2392, 2421, 2362]
minload = [ 304,  335,  372,  374,  353]
startup = [ 103,  134,  116,  143,  151]
total   = [e + m + s for e, m, s in zip(energy, minload, startup)]

x = np.arange(len(processes))
w = 0.55

# Use paper palette where sensible; cost components get distinct but neutral tones
C_ENERGY  = "#4682B4"   # steel blue (matches "unconstrained" tone)
C_MINLOAD = "#2A9D8F"   # teal (matches "knn")
C_STARTUP = "#E63946"   # red (matches "learned")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, 3.8))

b1 = ax.bar(x, energy,  width=w, label="Energy",    color=C_ENERGY,
            edgecolor="white", linewidth=0.6)
b2 = ax.bar(x, minload, width=w, label="Min. Load", color=C_MINLOAD,
            edgecolor="white", linewidth=0.6,
            bottom=energy)
b3 = ax.bar(x, startup, width=w, label="Startup",   color=C_STARTUP,
            edgecolor="white", linewidth=0.6,
            bottom=[e + m for e, m in zip(energy, minload)])

# Total cost labels on top of each bar
for i, t in enumerate(total):
    ax.text(i, t + 12, f"${t:,}K",
            ha="center", va="bottom",
            fontsize=FONT_SIZES["small"],
            fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(processes, fontsize=FONT_SIZES["small"])
ax.set_ylabel("Cost ($K)", fontsize=FONT_SIZES["medium"], fontweight="bold")
ax.set_ylim(0, max(total) * 1.18)  # extra headroom for labels + legend
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
)

# Remove top/right spines (grid already set by setup_plotting)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Place legend above the bars, horizontally centered, outside the plot area
ax.legend(
    framealpha=0.9,
    edgecolor="gray",
    fontsize=FONT_SIZES["small"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    borderaxespad=0.3,
)

fig.tight_layout()

import pathlib
pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/cost_stacked_detail.pdf",
            dpi=FIGURE_DEFAULTS["dpi_pdf"], bbox_inches="tight")
fig.savefig("Figures/cost_stacked_detail.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/cost_stacked_detail.pdf and .png")