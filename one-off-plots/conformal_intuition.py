"""
Conformal prediction intuition figure.
Left panel: histogram of calibration scores with Q_s marked.
Right panel: forecast, base quantile, and conformalized lower bound on a timeline.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib

from uncertainty_sets_refactored.plot_config import (
    setup_plotting,
    FONT_SIZES,
    IEEE_TWO_COL_WIDTH,
    FIGURE_DEFAULTS,
)

setup_plotting()

rng = np.random.default_rng(42)

# ── Toy parameters ────────────────────────────────────────────────────────────
Y_hat      = 1420.0
q_base     = 1180.0
Q_s        = 0.04
sigma_Y    = 1180.0
correction = Q_s * sigma_Y
Y_lo       = q_base - correction

# ── Synthetic calibration scores: Q_s at true 95th percentile ────────────────
# Draw from a left-skewed distribution so the bulk is negative
# and exactly 5% of mass falls above Q_s = 0.04
scores_raw = np.random.default_rng(42).normal(-0.06, 0.07, 2000)
scores_raw = np.clip(scores_raw, -0.35, 0.25)
# Shift so the 95th percentile lands exactly at Q_s
p95 = np.percentile(scores_raw, 95)
scores = scores_raw - p95 + Q_s

# ── Figure: two panels ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(IEEE_TWO_COL_WIDTH, IEEE_TWO_COL_WIDTH * 0.38),
)

C_HIST    = "#4682B4"
C_QS      = "#E63946"
C_BASE    = "#6c757d"
C_CONF    = "#2A9D8F"
C_FORECAST = "#E63946"
C_SHADE   = "#E63946"

# ── LEFT: Score histogram ─────────────────────────────────────────────────────
counts, bins, patches = ax1.hist(
    scores, bins=35, color=C_HIST, alpha=0.75, edgecolor="white", linewidth=0.4
)

# Shade the tail beyond Q_s (the 5% that are corrected)
for patch, left in zip(patches, bins[:-1]):
    if left >= Q_s:
        patch.set_facecolor(C_QS)
        patch.set_alpha(0.85)

# Q_s vertical line
ax1.axvline(Q_s, color=C_QS, linewidth=2.0, zorder=5)
ax1.text(Q_s + 0.01, counts.max() * 0.85,
         f"$Q_s = {Q_s}$\n(95th pctile)",
         fontsize=FONT_SIZES["small"] - 1, color=C_QS, fontweight="bold", va="top")

ax1.set_xlabel("Conformal Score $s_n$",
               fontsize=FONT_SIZES["small"], fontweight="bold")
ax1.set_ylabel("Count", fontsize=FONT_SIZES["small"], fontweight="bold")
ax1.set_title("Calibration Scores",
              fontsize=FONT_SIZES["small"], fontweight="bold", pad=4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── RIGHT: Vertical level diagram ────────────────────────────────────────────
ax2.set_xlim(-0.3, 2.6)
ax2.set_ylim(1100, 1470)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.set_xticks([])
ax2.set_ylabel("Wind (MW)", fontsize=FONT_SIZES["small"],
               fontweight="bold")
ax2.set_title("Conformal Correction",
              fontsize=FONT_SIZES["small"], fontweight="bold", pad=4)

# Light background shading
ax2.axhspan(1100, Y_lo,  color=C_QS,   alpha=0.07)
ax2.axhspan(Y_lo, 1470,  color=C_CONF, alpha=0.05)

# Horizontal level lines
for y, color, lw in [
    (Y_hat,  C_FORECAST, 2.2),
    (q_base, C_BASE,     1.8),
    (Y_lo,   C_CONF,     2.2),
]:
    ax2.hlines(y, 0.05, 1.55, color=color, linewidth=lw, zorder=4)

# Right-side labels — no numbers
ax2.text(1.62, Y_hat,
         r"Forecast $\hat{Y}_h$",
         fontsize=FONT_SIZES["small"] - 1, color=C_FORECAST,
         va="center", fontweight="bold")
ax2.text(1.62, q_base,
         r"Base quantile $q_Y$",
         fontsize=FONT_SIZES["small"] - 1, color=C_BASE,
         va="center", fontweight="bold")
ax2.text(1.62, Y_lo,
         r"Lower bound $\hat{Y}_h^{\rm lo}$",
         fontsize=FONT_SIZES["small"] - 1, color=C_CONF,
         va="center", fontweight="bold")

# Correction brace: q_base → Y_lo
ax2.annotate("", xy=(0.45, Y_lo), xytext=(0.45, q_base),
             arrowprops=dict(arrowstyle="<->", color=C_QS,
                             lw=1.5, mutation_scale=9))
ax2.text(0.48, (q_base + Y_lo) / 2,
         r"$Q_s \cdot \sigma$",
         fontsize=FONT_SIZES["small"] - 1, color=C_QS,
         va="center", ha="left", fontweight="bold")

fig.tight_layout(w_pad=2.5)

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/conformal_intuition.pdf", bbox_inches="tight")
fig.savefig("Figures/conformal_intuition.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/conformal_intuition.pdf and .png")