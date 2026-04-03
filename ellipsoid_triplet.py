"""
Triplet figure: ellipsoidal uncertainty sets with positive, weak, and negative
correlation, each showing the worst-case tangent hyperplane (45-degree line).

The tangent point minimizes W1 + W2 subject to lying on the ellipsoid boundary
— this is exactly the worst-case aggregate wind realization.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pathlib

from uncertainty_sets_refactored.plot_config import (
    setup_plotting,
    FONT_SIZES,
    IEEE_TWO_COL_WIDTH,
    FIGURE_DEFAULTS,
)

setup_plotting()

# ── Shared parameters ─────────────────────────────────────────────────────────
y_hat   = np.array([100.0, 100.0])   # forecast center (MW)
sigma1  = 30.0                        # W1 std dev (MW)
sigma2  = 42.0                        # W2 std dev (MW)
radius  = 2.0
e       = np.array([1.0, 1.0])       # aggregate direction

correlations = [0.75, 0.05, -0.75]
titles = [
    "Positive correlation\n($\\rho = 0.75$)",
    "Weak correlation\n($\\rho = 0.05$)",
    "Negative correlation\n($\\rho = -0.75$)",
]

C_ELLIPSE  = "#4682B4"
C_TANGENT  = "#E63946"
C_CENTER   = "#E63946"
C_POINT    = "#2A9D8F"

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3,
    figsize=(IEEE_TWO_COL_WIDTH, IEEE_TWO_COL_WIDTH * 0.42),
    sharey=True,
)

theta = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(theta), np.sin(theta)])

for ax, rho_corr, title in zip(axes, correlations, titles):

    # Covariance and Cholesky
    Sigma = np.array([
        [sigma1**2,                      rho_corr * sigma1 * sigma2],
        [rho_corr * sigma1 * sigma2,     sigma2**2],
    ])
    L = np.linalg.cholesky(Sigma)
    ellipse = y_hat[:, None] + radius * L @ circle

    # ── Worst-case tangent point ──────────────────────────────────────────────
    # Minimize e^T y s.t. (y - y_hat)^T Sigma^{-1} (y - y_hat) = rho^2
    # Solution: y* = y_hat - rho * Sigma e / sqrt(e^T Sigma e)
    Sigma_e    = Sigma @ e
    denom      = np.sqrt(e @ Sigma_e)
    y_star     = y_hat - radius * Sigma_e / denom
    worst_agg  = e @ y_star   # W1* + W2*

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Compute actual ellipse extent to set limits
    margin = 18.0
    x_min = ellipse[0].min() - margin
    x_max = ellipse[0].max() + margin
    y_min = ellipse[1].min() - margin
    y_max = ellipse[1].max() + margin
    # Make square by taking the wider extent
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        pad_y = (x_range - y_range) / 2
        y_min -= pad_y
        y_max += pad_y
    else:
        pad_x = (y_range - x_range) / 2
        x_min -= pad_x
        x_max += pad_x

    # Tangent line clipped to plot limits
    w1_tl = np.array([x_min, x_max])
    w2_tl = worst_agg - w1_tl

    ax.fill(ellipse[0], ellipse[1], color=C_ELLIPSE, alpha=0.18, zorder=2)
    ax.plot(ellipse[0], ellipse[1], color=C_ELLIPSE, linewidth=1.8, zorder=3)

    # Tangent line
    ax.plot(w1_tl, w2_tl, color=C_TANGENT, linewidth=1.6,
            linestyle="--", zorder=4,
            label=r"$W_1 + W_2 = \mathrm{const}$")

    # Forecast center
    ax.scatter(*y_hat, color=C_CENTER, s=40, zorder=6, marker="o")

    # Worst-case tangent point
    ax.scatter(*y_star, color=C_POINT, s=50, zorder=7, marker="*")

    # Leader line from center to tangent point
    ax.annotate(
        "", xy=y_star, xytext=y_hat,
        arrowprops=dict(arrowstyle="-|>", color=C_POINT,
                        lw=1.0, mutation_scale=9, linestyle="dotted"),
        zorder=5,
    )

    # Worst-case aggregate label
    ax.text(
        0.97, 0.04,
        f"$Y^{{\\rm wc}} = {worst_agg:.0f}$ MW",
        transform=ax.transAxes,
        fontsize=FONT_SIZES["small"] - 1,
        color=C_TANGENT, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=C_TANGENT, alpha=0.85),
    )

    ax.set_title(title, fontsize=FONT_SIZES["small"], pad=4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y_hat[1], color="gray", linewidth=0.4, alpha=0.3)
    ax.axvline(y_hat[0], color="gray", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("W1 (MW)", fontsize=FONT_SIZES["small"],
                  fontweight="bold")

axes[0].set_ylabel("W2 (MW)", fontsize=FONT_SIZES["small"],
                   fontweight="bold")

# Shared legend below
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=C_ELLIPSE, linewidth=1.8,
           label="Ellipsoidal set $\\mathcal{U}_h$"),
    Line2D([0], [0], color=C_TANGENT, linewidth=1.6, linestyle="--",
           label="Worst-case hyperplane"),
    plt.scatter([], [], color=C_CENTER, s=35, marker="o",
                label="Forecast $\\hat{\\mathbf{y}}_h$"),
    plt.scatter([], [], color=C_POINT, s=45, marker="*",
                label="Worst-case point $\\tilde{\\mathbf{y}}_h$"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=FONT_SIZES["small"] - 1,
    framealpha=0.9,
    edgecolor="gray",
    bbox_to_anchor=(0.5, -0.08),
)

fig.tight_layout(rect=[0, 0.08, 1, 1])

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/ellipsoid_correlation_triplet.pdf",
            dpi=FIGURE_DEFAULTS["dpi_pdf"], bbox_inches="tight")
fig.savefig("Figures/ellipsoid_correlation_triplet.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/ellipsoid_correlation_triplet.pdf and .png")