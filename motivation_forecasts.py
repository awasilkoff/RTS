"""
Motivation figure: point forecast (left) vs ensemble spaghetti (right).
Synthetic but realistic diurnal wind power pattern.
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

rng = np.random.default_rng(42)
hours = np.arange(0, 25)

# ── Diurnal wind pattern ──────────────────────────────────────────────────────
# Wind tends to peak overnight/early morning and dip midday
def wind_mean(h):
    base = 0.45
    overnight = 0.18 * np.cos(2 * np.pi * (h - 3) / 24)
    midday_dip = -0.10 * np.exp(-0.5 * ((h - 13) / 3) ** 2)
    ramp = 0.08 * np.sin(2 * np.pi * h / 24)
    return np.clip(base + overnight + midday_dip + ramp, 0.05, 0.95)

mean_curve = wind_mean(hours)

# ── Point forecast ────────────────────────────────────────────────────────────
# Slightly smoothed, offset from truth
point_forecast = mean_curve + 0.04 * np.sin(2 * np.pi * hours / 12) + 0.03

# ── Actual realization ────────────────────────────────────────────────────────
# Diverges significantly in the afternoon
actual = mean_curve.copy()
actual[10:18] -= 0.22 * np.exp(-0.5 * ((hours[10:18] - 14) / 3) ** 2)
actual += rng.normal(0, 0.015, len(hours))
actual = np.clip(actual, 0, 1)

# ── Ensemble members ──────────────────────────────────────────────────────────
n_members = 30
# Uncertainty grows during the day, peaks mid-afternoon
def spread(h):
    return 0.04 + 0.12 * np.exp(-0.5 * ((h - 15) / 6) ** 2)

ensemble = np.zeros((n_members, len(hours)))
for m in range(n_members):
    noise = rng.normal(0, 1, len(hours))
    # Smooth the noise
    from scipy.ndimage import gaussian_filter1d
    noise = gaussian_filter1d(noise, sigma=2.5)
    ensemble[m] = np.clip(mean_curve + spread(hours) * noise, 0, 1)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(IEEE_TWO_COL_WIDTH, IEEE_TWO_COL_WIDTH * 0.42),
    sharey=True,
)

C_FORECAST = "#4682B4"
C_ACTUAL   = "#E63946"
C_ENSEMBLE = "#4682B4"
C_MEAN     = "#2A9D8F"

# ── LEFT: Point forecast only ─────────────────────────────────────────────────
ax1.plot(hours, point_forecast, color=C_FORECAST, linewidth=2.5,
         label="Point forecast", zorder=4)
ax1.plot(hours, actual, color=C_ACTUAL, linewidth=2.0,
         linestyle="--", label="Actual realization", zorder=5)

ax1.set_title("Deterministic Forecast", fontsize=FONT_SIZES["medium"],
              fontweight="bold", pad=6)
ax1.set_xlabel("Hour of Day", fontsize=FONT_SIZES["small"], fontweight="bold")
ax1.set_ylabel("Wind Power (p.u.)", fontsize=FONT_SIZES["small"], fontweight="bold")
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 1.0)
ax1.set_xticks([0, 6, 12, 18, 24])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=FONT_SIZES["small"] - 1, framealpha=0.9,
           edgecolor="gray", loc="upper right")

# Annotation: large gap at hour 14
gap_h = 14
ax1.annotate(
    "large\nshortfall",
    xy=(gap_h, actual[gap_h]),
    xytext=(gap_h + 2, actual[gap_h] - 0.12),
    fontsize=FONT_SIZES["small"] - 1,
    color=C_ACTUAL,
    arrowprops=dict(arrowstyle="-|>", color=C_ACTUAL, lw=1.2),
    ha="left",
)

# ── RIGHT: Ensemble spaghetti ─────────────────────────────────────────────────
for m in range(n_members):
    ax2.plot(hours, ensemble[m], color=C_ENSEMBLE, linewidth=0.6,
             alpha=0.25, zorder=2)

# Ensemble mean
ens_mean = ensemble.mean(axis=0)
ax2.plot(hours, ens_mean, color=C_MEAN, linewidth=2.2,
         label="Ensemble mean", zorder=4)

# Actual
ax2.plot(hours, actual, color=C_ACTUAL, linewidth=2.0,
         linestyle="--", label="Actual realization", zorder=5)

ax2.set_title("Ensemble Forecast", fontsize=FONT_SIZES["medium"],
              fontweight="bold", pad=6)
ax2.set_xlabel("Hour of Day", fontsize=FONT_SIZES["small"], fontweight="bold")
ax2.set_xlim(0, 24)
ax2.set_ylim(0, 1.0)
ax2.set_xticks([0, 6, 12, 18, 24])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=FONT_SIZES["small"] - 1, framealpha=0.9,
           edgecolor="gray", loc="upper right")

fig.tight_layout(w_pad=2.0)

pathlib.Path("Figures").mkdir(exist_ok=True)
fig.savefig("Figures/motivation_forecast.pdf",
            bbox_inches="tight")
fig.savefig("Figures/motivation_forecast.png",
            dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
print("Saved Figures/motivation_forecast.pdf and .png")