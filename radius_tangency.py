"""
Radius estimation figure: conformal lower bound hyperplane tangent to ellipsoid.
The radius rho_h is chosen so the ellipsoid just touches the hyperplane W1+W2=Y_lo.
"""
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
y_hat    = np.array([100.0, 100.0])
rho_corr = 0.72
sigma1, sigma2 = 30.0, 42.0
Sigma = np.array([
    [sigma1**2,                      rho_corr * sigma1 * sigma2],
    [rho_corr * sigma1 * sigma2,     sigma2**2],
])
e = np.array([1.0, 1.0])   # aggregate direction

# ── Compute radius and tangent point ─────────────────────────────────────────
# Conformal lower bound on aggregate: Y_lo = e^T y_hat - gap
# We set a toy gap so the hyperplane is clearly inside the ellipsoid
Y_hat = e @ y_hat          # 200 MW
gap   = 60.0               # conformal gap (MW)
Y_lo  = Y_hat - gap        # 140 MW

# Radius: rho = gap / sqrt(e^T Sigma e)
denom = np.sqrt(e @ Sigma @ e)
rho   = gap / denom

# Tangent point: y* = y_hat - rho * Sigma e / sqrt(e^T Sigma e)
Sigma_e = Sigma @ e
y_star  = y_hat - rho * Sigma_e / denom

# ── Ellipse ───────────────────────────────────────────────────────────────────
L = np.linalg.cholesky(Sigma)
theta  = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(theta), np.sin(theta)])
ellipse = y_hat[:, None] + rho * L @ circle

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(IEEE_TWO_COL_WIDTH * 0.55, IEEE_TWO_COL_WIDTH * 0.55)
)

C_ELLIPSE  = "#4682B4"
C_TANGENT  = "#E63946"
C_CENTER   = "#E63946"
C_POINT    = "#2A9D8F"
C_BOUND    = "#E63946"
C_GAP      = "#888888"

# Axis limits
pad = sigma2 * rho * 1.4
xlim = (y_hat[0] - pad, y_hat[0] + pad)
ylim = (y_hat[1] - pad, y_hat[1] + pad)

# Conformal lower bound hyperplane: W1 + W2 = Y_lo
w1_line = np.array([xlim[0], xlim[1]])
w2_line = Y_lo - w1_line
ax.plot(w1_line, w2_line, color=C_BOUND, linewidth=1.8,
        linestyle="--", zorder=3)

# Shaded region below hyperplane (infeasible / shortfall region)
ax.fill_between(
    w1_line, w2_line, ylim[0],
    color=C_BOUND, alpha=0.06, zorder=1,
)

# Ellipse
ax.fill(ellipse[0], ellipse[1], color=C_ELLIPSE, alpha=0.18, zorder=2)
ax.plot(ellipse[0], ellipse[1], color=C_ELLIPSE, linewidth=2.0, zorder=3)

# Forecast center
ax.scatter(*y_hat, color=C_CENTER, s=55, zorder=6)
ax.annotate(
    r"$\hat{\mathbf{y}}_h$",
    xy=y_hat, xytext=(y_hat[0] + 4, y_hat[1] + 6),
    fontsize=FONT_SIZES["medium"], color=C_CENTER, fontweight="bold",
)

# Tangent point
ax.scatter(*y_star, color=C_POINT, s=60, zorder=7, marker="*")
ax.annotate(
    r"$\tilde{\mathbf{y}}_h$",
    xy=y_star, xytext=(y_star[0] - 18, y_star[1] - 14),
    fontsize=FONT_SIZES["medium"], color=C_POINT, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=C_POINT, lw=1.0,
                    mutation_scale=9),
)

# rho_h arrow from center to tangent point
ax.annotate(
    "", xy=y_star, xytext=y_hat,
    arrowprops=dict(arrowstyle="-|>", color=C_ELLIPSE,
                    lw=1.5, mutation_scale=11),
    zorder=5,
)
mid = (y_hat + y_star) / 2
ax.text(mid[0] + 5, mid[1], r"$\rho_h$",
        fontsize=FONT_SIZES["medium"], color=C_ELLIPSE, fontweight="bold")

# Gap annotation: vertical arrow from Y_lo line to y_hat aggregate
# Show gap along the e direction from tangent point to center
ax.annotate(
    "", xy=y_hat, xytext=y_star,
    arrowprops=dict(arrowstyle="<->", color=C_GAP,
                    lw=1.2, mutation_scale=9),
    zorder=4,
)

# Hyperplane label — moved to upper left area
ax.text(xlim[0] + 4, ylim[1] - 4,
        r"$W_1 + W_2 = Y_h^{\rm lo}$" + "\n(conformal lower bound)",
        fontsize=FONT_SIZES["small"] - 1, color=C_BOUND,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=C_BOUND, alpha=0.85))

# Y_hat aggregate label on axis
ax.axline(y_hat, slope=-1, color="gray", linewidth=0.5,
          alpha=0.3, linestyle=":")

# Axes
ax.set_xlabel("W1 (MW)", fontsize=FONT_SIZES["small"], fontweight="bold")
ax.set_ylabel("W2 (MW)", fontsize=FONT_SIZES["small"], fontweight="bold")
ax.set_aspect("equal")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.axhline(y_hat[1], color="gray", linewidth=0.4, alpha=0.2)
ax.axvline(y_hat[0], color="gray", linewidth=0.4, alpha=0.2)

fig.tight_layout()

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/radius_tangency.pdf", bbox_inches="tight")
fig.savefig("Figures/radius_tangency.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/radius_tangency.pdf and .png")