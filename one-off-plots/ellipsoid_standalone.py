"""
Standalone ellipsoid figure for the uncertainty set introduction slide.
No box — just the ellipse with rho_h and Sigma_h annotations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from uncertainty_sets_refactored.plot_config import (
    setup_plotting,
    FONT_SIZES,
    IEEE_TWO_COL_WIDTH,
    FIGURE_DEFAULTS,
)

setup_plotting()

# ── Parameters ────────────────────────────────────────────────────────────────
y_hat  = np.array([100.0, 100.0])
rho_corr = 0.72
sigma1, sigma2 = 30.0, 42.0
Sigma = np.array([
    [sigma1**2,                      rho_corr * sigma1 * sigma2],
    [rho_corr * sigma1 * sigma2,     sigma2**2],
])
radius = 2.0

L = np.linalg.cholesky(Sigma)
theta = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(theta), np.sin(theta)])
ellipse = y_hat[:, None] + radius * L @ circle

# ── Major axis tip for rho annotation ────────────────────────────────────────
eigvals, eigvecs = np.linalg.eigh(Sigma)
major_vec = eigvecs[:, np.argmax(eigvals)]
tip = y_hat + radius * L @ (L.T @ major_vec / np.linalg.norm(L.T @ major_vec))
mid = (y_hat + tip) / 2

# ── Minor axis tip for Sigma annotation ──────────────────────────────────────
minor_vec = eigvecs[:, np.argmin(eigvals)]
minor_tip = y_hat + radius * L @ (L.T @ minor_vec / np.linalg.norm(L.T @ minor_vec))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(IEEE_TWO_COL_WIDTH * 0.45, IEEE_TWO_COL_WIDTH * 0.45)
)

C_ELLIPSE = "#4682B4"
C_CENTER  = "#E63946"
C_RHO     = "#4682B4"
C_SIGMA   = "#2A9D8F"

# Ellipse fill and border
ax.fill(ellipse[0], ellipse[1], color=C_ELLIPSE, alpha=0.15, zorder=2)
ax.plot(ellipse[0], ellipse[1], color=C_ELLIPSE, linewidth=2.2, zorder=3)

# Forecast center
ax.scatter(*y_hat, color=C_CENTER, s=55, zorder=6)
ax.annotate(
    r"$\hat{\mathbf{y}}_h$",
    xy=y_hat,
    xytext=(y_hat[0] + 4, y_hat[1] - 11),
    fontsize=FONT_SIZES["medium"],
    color=C_CENTER, fontweight="bold",
)

# rho_h arrow along NEGATIVE major axis — tip on ellipse boundary
# Find the ellipse point closest to the negative major axis direction
neg_major = -major_vec
dots = np.array([neg_major @ (ellipse[:, i] - y_hat) for i in range(ellipse.shape[1])])
neg_tip = ellipse[:, np.argmax(dots)]
neg_mid = (y_hat + neg_tip) / 2
ax.annotate(
    "", xy=neg_tip, xytext=y_hat,
    arrowprops=dict(arrowstyle="-|>", color=C_RHO, lw=1.8, mutation_scale=14),
    zorder=5,
)
ax.text(neg_mid[0] + 6, neg_mid[1] - 6, r"$\rho_h$",
        fontsize=FONT_SIZES["large"], color=C_RHO, fontweight="bold")

# Sigma_h: eigenvector arrows scaled up by 1.6x for visibility
scale_factor = 1.6
for i, (lam, vec) in enumerate(zip(eigvals, eigvecs.T)):
    scale = np.sqrt(lam) * scale_factor
    end = y_hat + scale * vec
    ax.annotate(
        "", xy=end, xytext=y_hat,
        arrowprops=dict(arrowstyle="-|>", color=C_SIGMA, lw=1.5,
                        mutation_scale=12),
        zorder=5,
    )

# Sigma_h label — further right of minor eigenvector tip
minor_end = y_hat + np.sqrt(eigvals[0]) * eigvecs[:, 0] * scale_factor
ax.text(minor_end[0] + 12, minor_end[1],
        r"$\Sigma_h$",
        fontsize=FONT_SIZES["large"], color=C_SIGMA, fontweight="bold",
        ha="left", va="center")

# Axes
ax.set_xlabel("W1 (MW)", fontsize=FONT_SIZES["small"], fontweight="bold")
ax.set_ylabel("W2 (MW)", fontsize=FONT_SIZES["small"], fontweight="bold")
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
pad = sigma2 * radius * 1.35
ax.set_xlim(y_hat[0] - pad, y_hat[0] + pad)
ax.set_ylim(y_hat[1] - pad, y_hat[1] + pad)
ax.axhline(y_hat[1], color="gray", linewidth=0.4, alpha=0.3)
ax.axvline(y_hat[0], color="gray", linewidth=0.4, alpha=0.3)

fig.tight_layout()

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/ellipsoid_standalone.pdf", bbox_inches="tight")
fig.savefig("Figures/ellipsoid_standalone.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/ellipsoid_standalone.pdf and .png")