"""
LDR intuition figure — shared response variant.
W1 drives uncertainty; G1 and G2 share the balancing response.
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

rho = 10.0
r   = np.linspace(-rho * 1.25, rho * 1.25, 300)

# W1 — wind (positive slope, no balancing role here)
aW, BW = 70.0,  1.0
# G1 — takes 60% of the balancing
a1, B1 = 45.0, -0.6
# G2 — takes 40% of the balancing
a2, B2 = 35.0, -0.4

C_W   = "#2A9D8F"
C_G1  = "#4682B4"
C_G2  = "#E9A227"
C_RHO = "#888888"

fig, ax = plt.subplots(
    figsize=(IEEE_TWO_COL_WIDTH * 0.6, IEEE_TWO_COL_WIDTH * 0.45)
)

# Uncertainty set shading
ax.axvspan(-rho, rho, color=C_RHO, alpha=0.07, zorder=0)
ax.axvline(-rho, color=C_RHO, lw=1.2, ls=":", alpha=0.6)
ax.axvline( rho, color=C_RHO, lw=1.2, ls=":", alpha=0.6)
# Replace the ax.text(0, 2, r"$\mathcal{U}$", ...) line with:
ax.annotate("", xy=(rho, 6), xytext=(-rho, 6),
            arrowprops=dict(arrowstyle="<->", color=C_RHO, lw=1.2))
ax.text(0, 8, r"$\mathcal{U}$",
        fontsize=FONT_SIZES["small"], color=C_RHO,
        ha="center", va="bottom", style="italic")

pW = aW + BW * r
p1 = a1 + B1 * r
p2 = a2 + B2 * r

ax.plot(r, pW, color=C_W,  lw=2.2, zorder=4, label=r"W1: $60 + r$")
ax.plot(r, p1, color=C_G1, lw=2.2, zorder=4, label=r"G1: $45 - 0.6\,r$")
ax.plot(r, p2, color=C_G2, lw=2.2, zorder=4, label=r"G2: $35 - 0.4\,r$")

# Nominal points at r=0
ax.scatter([0, 0, 0], [aW, a1, a2], color=[C_W, C_G1, C_G2], s=60, zorder=6)
ax.axvline(0, color="gray", lw=0.6, alpha=0.4)

ax.set_xlabel("Wind forecast error $r$ (MW)",
              fontsize=FONT_SIZES["small"], fontweight="bold")
ax.set_ylabel("Dispatch (MW)",
              fontsize=FONT_SIZES["small"], fontweight="bold")
# ax.set_title("Linear Decision Rules — Shared Response",
#              fontsize=FONT_SIZES["small"], fontweight="bold", pad=4)
ax.set_xlim(-rho * 1.25, rho * 1.25)
ax.set_ylim(0, 105)
ax.set_xticks([-rho, -rho/2, 0, rho/2, rho])
ax.set_xticklabels(["-10", "-5", "0", "5", "10"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=FONT_SIZES["small"] - 2, loc="upper right", framealpha=0.9)

fig.tight_layout()

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/ldr_intuition_shared.pdf", bbox_inches="tight")
fig.savefig("Figures/ldr_intuition_shared.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/ldr_intuition_shared.pdf and .png")