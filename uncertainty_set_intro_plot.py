"""
2D illustration: Ellipsoidal vs. Box uncertainty set for two wind farms.
Styled to match paper plot_config.
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

# ── Uncertainty set parameters ────────────────────────────────────────────────
# Forecast center (MW)
y_hat = np.array([100.0, 100.0])

# Covariance in MW^2
rho_corr = 0.72
sigma1, sigma2 = 30.0, 42.0   # MW std devs
Sigma = np.array([
    [sigma1**2,                      rho_corr * sigma1 * sigma2],
    [rho_corr * sigma1 * sigma2,     sigma2**2],
])
radius = 2.0

# Box half-widths (same marginal coverage)
box_hw1 = radius * sigma1
box_hw2 = radius * sigma2

# ── Ellipse parametric ────────────────────────────────────────────────────────
L = np.linalg.cholesky(Sigma)
theta = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(theta), np.sin(theta)])
ellipse = y_hat[:, None] + radius * L @ circle

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH * 0.7, IEEE_TWO_COL_WIDTH * 0.5))

# Box
bx0, by0 = y_hat[0] - box_hw1, y_hat[1] - box_hw2
box = mpatches.FancyBboxPatch(
    (bx0, by0), 2 * box_hw1, 2 * box_hw2,
    boxstyle="square,pad=0",
    linewidth=1.4, edgecolor="#6c757d", facecolor="#6c757d", alpha=0.15,
    zorder=1,
)
ax.add_patch(box)
ax.plot(
    [bx0, bx0 + 2*box_hw1, bx0 + 2*box_hw1, bx0, bx0],
    [by0, by0, by0 + 2*box_hw2, by0 + 2*box_hw2, by0],
    color="#6c757d", linewidth=1.4, linestyle="--", zorder=2,
)

# Ellipsoid
ax.fill(ellipse[0], ellipse[1], color="#4682B4", alpha=0.18, zorder=3)
ax.plot(ellipse[0], ellipse[1], color="#4682B4", linewidth=2.0, zorder=4)

# Center / forecast point
ax.scatter(*y_hat, color="#E63946", s=60, zorder=6)
ax.annotate(
    r"$\hat{\mathbf{y}}_h$",
    xy=y_hat, xytext=(y_hat[0] + 4, y_hat[1] - 10),
    fontsize=FONT_SIZES["medium"],
    color="#E63946", fontweight="bold",
)

# Radius annotation along major axis
eigvals, eigvecs = np.linalg.eigh(Sigma)
major_vec = eigvecs[:, np.argmax(eigvals)]
tip = y_hat + radius * L @ (L.T @ major_vec / np.linalg.norm(L.T @ major_vec))
ax.annotate(
    "", xy=tip, xytext=y_hat,
    arrowprops=dict(arrowstyle="-|>", color="#4682B4",
                    lw=1.4, mutation_scale=12),
    zorder=5,
)
mid = (y_hat + tip) / 2
ax.text(mid[0] - 10, mid[1] + 6, r"$\rho_h$",
        fontsize=FONT_SIZES["medium"], color="#4682B4", fontweight="bold")

# Inline labels
ax.text(y_hat[0] - box_hw1 - 2, y_hat[1] + box_hw2 + 5, "Box set",
        fontsize=FONT_SIZES["small"], color="#6c757d",
        ha="right", va="bottom", style="italic")
ax.text(ellipse[0][50], ellipse[1][50] + 5, "Ellipsoidal set",
        fontsize=FONT_SIZES["small"], color="#4682B4",
        ha="center", va="bottom", fontweight="bold")

# Axes labels
ax.set_xlabel("W1 (MW)",
              fontsize=FONT_SIZES["medium"], fontweight="bold")
ax.set_ylabel("W2 (MW)",
              fontsize=FONT_SIZES["medium"], fontweight="bold")

# Equal aspect, clean spines, symmetric limits around center
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
pad = box_hw2 * 1.25
ax.set_xlim(y_hat[0] - pad, y_hat[0] + pad)
ax.set_ylim(y_hat[1] - pad, y_hat[1] + pad)
ax.axhline(y_hat[1], color="gray", linewidth=0.5, alpha=0.3)
ax.axvline(y_hat[0], color="gray", linewidth=0.5, alpha=0.3)

fig.tight_layout()

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/ellipsoid_vs_box_2d.pdf",
            dpi=FIGURE_DEFAULTS["dpi_pdf"], bbox_inches="tight")
fig.savefig("Figures/ellipsoid_vs_box_2d.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/ellipsoid_vs_box_2d.pdf and .png")