"""
Linear Decision Rule (LDR) intuition figure.
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

# ── Parameters (MW) ───────────────────────────────────────────────────────────
rho    = 10.0
r      = np.linspace(-rho * 1.25, rho * 1.25, 300)

a1, B1 = 40.0, -1.0
a2, B2 = 70.0,  1.0

C_U1  = "#4682B4"
C_U2  = "#2A9D8F"
C_RHO = "#888888"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(IEEE_TWO_COL_WIDTH * 0.6, IEEE_TWO_COL_WIDTH * 0.45)
)

# Uncertainty set shading
ax.axvspan(-rho, rho, color=C_RHO, alpha=0.07, zorder=0)
ax.axvline(-rho, color=C_RHO, lw=1.2, ls=":", alpha=0.6)
ax.axvline( rho, color=C_RHO, lw=1.2, ls=":", alpha=0.6)
ax.text(0, 2, r"$\mathcal{U}$",
        fontsize=FONT_SIZES["small"], color=C_RHO,
        ha="center", va="bottom", style="italic")

# LDR lines
p1 = a1 + B1 * r
p2 = a2 + B2 * r
ax.plot(r, p1, color=C_U1, lw=2.2, zorder=4,
        label=r"G1: $40 - r$")
ax.plot(r, p2, color=C_U2, lw=2.2, zorder=4,
        label=r"W1: $60 + r$")

# Nominal operating point at r=0
ax.scatter([0, 0], [a1, a2], color=[C_U1, C_U2], s=60, zorder=6)
ax.axvline(0, color="gray", lw=0.6, alpha=0.4)

# Slope annotation for G1
ax.annotate("", xy=(4, a1 + B1 * 4),
            xytext=(1, a1 + B1 * 1),
            arrowprops=dict(arrowstyle="-|>", color=C_U1,
                            lw=1.0, mutation_scale=8))
ax.text(4.3, a1 + B1 * 4 + 1.5, r"slope $= B_1$",
        fontsize=FONT_SIZES["small"] - 2, color=C_U1)

ax.set_xlabel("Wind forecast error $r$ (MW)",
              fontsize=FONT_SIZES["small"], fontweight="bold")
ax.set_ylabel("Dispatch (MW)",
              fontsize=FONT_SIZES["small"], fontweight="bold")
ax.set_title("Linear Decision Rules",
             fontsize=FONT_SIZES["small"], fontweight="bold", pad=4)
ax.set_xlim(-rho * 1.25, rho * 1.25)
ax.set_ylim(0, 105)
ax.set_xticks([-rho, -rho/2, 0, rho/2, rho])
ax.set_xticklabels(["-10", "-5", "0", "5", "10"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=FONT_SIZES["small"] - 2, loc="upper right", framealpha=0.9)

fig.tight_layout()

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/ldr_intuition.pdf", bbox_inches="tight")
fig.savefig("Figures/ldr_intuition.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/ldr_intuition.pdf and .png")