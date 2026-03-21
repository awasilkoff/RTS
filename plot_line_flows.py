#!/usr/bin/env python3
"""
Line flow comparison visualizations: DAM w/ Res vs DARUC.

Generates four charts:
  1. Side-by-side worst-case loading heatmaps (lines x hours)
  2. Binding status diff heatmap (both / only-DAM / only-DARUC / neither)
  3. Stacked bar decomposition: nominal flow + robust margin at peak hour
  4. Network map at a specific hour showing flows on the RTS-GMLC topology

Usage:
    python plot_line_flows.py --case-dir sensitivity_suite/rho99_48h_m07d15/reserve_then_daruc
    python plot_line_flows.py --case-dir sensitivity_suite/rho99_48h_m07d15/reserve_then_daruc --map-hour 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent / "uncertainty_sets_refactored"))
from uncertainty_sets_refactored.plot_config import (
    setup_plotting,
    FONT_SIZES,
    FONT_SIZES_TWO_COL,
    IEEE_COL_WIDTH,
    IEEE_TWO_COL_WIDTH,
    FIGURE_DEFAULTS,
)

setup_plotting()

# Consistent save helper (PDF + PNG, tight bbox)
def _save_figure(fig: plt.Figure, path: Path):
    """Save figure as both PDF and PNG with paper-ready settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(
            path.with_suffix(f".{fmt}"),
            dpi=FIGURE_DEFAULTS["dpi_pdf"] if fmt == "pdf" else FIGURE_DEFAULTS["dpi"],
            bbox_inches="tight",
        )
    print(f"  Saved {path.stem}.pdf/png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_line_flow_analysis(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DAM w/ Res and DARUC line_flow_analysis CSVs."""
    dam_path = case_dir / "dam_reserve" / "line_flows" / "line_flow_analysis_dam.csv"
    daruc_path = case_dir / "daruc" / "line_flows" / "line_flow_analysis.csv"

    if not dam_path.exists():
        # fallback: some runs use line_flow_analysis.csv for DAM too
        dam_path = case_dir / "dam_reserve" / "line_flows" / "line_flow_analysis.csv"

    dam = pd.read_csv(dam_path, parse_dates=["period"])
    daruc = pd.read_csv(daruc_path, parse_dates=["period"])
    return dam, daruc


def filter_day1_binding(
    dam: pd.DataFrame, daruc: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Filter to day-1 periods and lines binding in at least one case."""
    # Day-1: first 24 hourly periods
    day1_cutoff = dam["period"].min() + pd.Timedelta(hours=24)
    dam = dam[dam["period"] < day1_cutoff].copy()
    daruc = daruc[daruc["period"] < day1_cutoff].copy()

    # Lines binding in at least one case at any hour
    dam_binding = set(dam.loc[dam["binding"], "line"].unique())
    daruc_binding = set(daruc.loc[daruc["binding"], "line"].unique())
    binding_lines = sorted(dam_binding | daruc_binding)

    dam = dam[dam["line"].isin(binding_lines)]
    daruc = daruc[daruc["line"].isin(binding_lines)]

    return dam, daruc, binding_lines


def pivot_column(df: pd.DataFrame, col: str, lines: list[str]) -> pd.DataFrame:
    """Pivot long-format to (lines x hours) matrix."""
    piv = df.pivot(index="line", columns="period", values=col)
    piv = piv.reindex(lines)
    return piv


# ---------------------------------------------------------------------------
# Chart 1: Side-by-side loading heatmaps
# ---------------------------------------------------------------------------

def plot_loading_heatmaps(
    dam: pd.DataFrame,
    daruc: pd.DataFrame,
    lines: list[str],
    out_dir: Path,
):
    """Two-panel heatmap of worst-case loading %."""
    dam_load = pivot_column(dam, "loading_worst_case_pct", lines)
    daruc_load = pivot_column(daruc, "loading_worst_case_pct", lines)
    dam_bind = pivot_column(dam, "binding", lines).fillna(False)
    daruc_bind = pivot_column(daruc, "binding", lines).fillna(False)

    hours = [t.strftime("%H:%M") for t in dam_load.columns]
    n_lines = len(lines)
    n_hours = len(hours)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(IEEE_TWO_COL_WIDTH * 1.4, max(3, n_lines * 0.6 + 1.5)),
        sharey=True,
    )

    vmin, vmax = 0, 100
    cmap = plt.cm.RdYlGn_r

    # Show every 3rd hour to avoid cramping
    tick_step = 3
    tick_positions = range(0, n_hours, tick_step)
    tick_labels = [hours[i] for i in tick_positions]

    for ax, data_load, data_bind, title in [
        (ax1, dam_load, dam_bind, "DAM + Reserve"),
        (ax2, daruc_load, daruc_bind, "DARUC"),
    ]:
        im = ax.imshow(
            data_load.values,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.grid(False)
        bind_arr = data_bind.values.astype(bool)
        ys, xs = np.where(bind_arr)
        ax.scatter(xs, ys, marker="s", s=18, c="black", alpha=0.6, label="binding")

        ax.set_xticks(list(tick_positions))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=FONT_SIZES_TWO_COL["small"])
        ax.set_yticks(range(n_lines))
        ax.set_yticklabels(lines, fontsize=FONT_SIZES_TWO_COL["small"])
        ax.set_title(title, fontsize=FONT_SIZES_TWO_COL["large"])
        ax.set_xlabel("Hour", fontsize=FONT_SIZES_TWO_COL["medium"])

    ax1.set_ylabel("Line", fontsize=FONT_SIZES_TWO_COL["medium"])

    fig.tight_layout()
    fig.colorbar(im, ax=[ax1, ax2], label="Worst-Case Loading %", shrink=0.8, pad=0.02)

    _save_figure(fig, out_dir / "fig_loading_heatmap")


# ---------------------------------------------------------------------------
# Chart 2: Binding diff heatmap
# ---------------------------------------------------------------------------

def plot_binding_diff(
    dam: pd.DataFrame,
    daruc: pd.DataFrame,
    lines: list[str],
    out_dir: Path,
):
    """Categorical heatmap: both binding / only-DAM / only-DARUC / neither."""
    dam_bind = pivot_column(dam, "binding", lines).fillna(False).values.astype(bool)
    daruc_bind = pivot_column(daruc, "binding", lines).fillna(False).values.astype(bool)

    hours = [t.strftime("%H:%M") for t in pivot_column(dam, "binding", lines).columns]
    n_lines = len(lines)
    n_hours = len(hours)

    # Encode: 0=neither, 1=DAM only, 2=DARUC only, 3=both
    cat = np.zeros_like(dam_bind, dtype=int)
    cat[dam_bind & ~daruc_bind] = 1
    cat[~dam_bind & daruc_bind] = 2
    cat[dam_bind & daruc_bind] = 3

    colors = ["#f5f5f5", "#4682B4", "#2A9D8F", "#2F4F4F"]
    labels = ["Neither", "DAM w/Res only", "DARUC only", "Both"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(
        figsize=(IEEE_TWO_COL_WIDTH, max(2.5, n_lines * 0.55 + 1)),
    )
    im = ax.imshow(cat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.grid(False)

    ax.set_xticks(range(n_hours))
    ax.set_xticklabels(hours, rotation=90, fontsize=FONT_SIZES_TWO_COL["small"])
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels(lines, fontsize=FONT_SIZES_TWO_COL["small"])
    ax.set_xlabel("Hour", fontsize=FONT_SIZES_TWO_COL["medium"])
    ax.set_ylabel("Line", fontsize=FONT_SIZES_TWO_COL["medium"])


    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="gray", label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, loc="upper center",
              fontsize=FONT_SIZES_TWO_COL["small"], ncol=4,
              bbox_to_anchor=(0.5, -0.4))

    fig.tight_layout()
    _save_figure(fig, out_dir / "fig_binding_diff")


# ---------------------------------------------------------------------------
# Chart 3: Stacked bar decomposition at peak hour
# ---------------------------------------------------------------------------

def plot_flow_decomposition(
    dam: pd.DataFrame,
    daruc: pd.DataFrame,
    lines: list[str],
    out_dir: Path,
):
    """Horizontal stacked bars: |nominal| + margin for DARUC at each line's peak hour.
    Overlays DAM w/ Res |nominal| for comparison."""

    # For each binding line, find the DARUC hour with max worst_case loading
    records = []
    for line in lines:
        d = daruc[daruc["line"] == line]
        peak_idx = d["loading_worst_case_pct"].idxmax()
        peak_row = d.loc[peak_idx]
        peak_period = peak_row["period"]

        dam_row = dam[(dam["line"] == line) & (dam["period"] == peak_period)]
        dam_nominal = abs(dam_row["flow_nominal"].values[0]) if len(dam_row) else 0

        records.append({
            "line": line,
            "hour": peak_period.strftime("%H:%M"),
            "nominal_abs": abs(peak_row["flow_nominal"]),
            "margin": peak_row["margin_rho_norm"],
            "Fmax": peak_row["Fmax"],
            "dam_nominal_abs": dam_nominal,
            "wc_loading": peak_row["loading_worst_case_pct"],
        })

    rec_df = pd.DataFrame(records).sort_values("wc_loading", ascending=True)

    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, max(2.5, len(lines) * 0.6 + 1)))

    y_pos = range(len(rec_df))
    bar_labels = [f"{r['line']} (h{r['hour']})" for _, r in rec_df.iterrows()]

    ax.barh(
        y_pos, rec_df["nominal_abs"], height=0.6,
        color="#4393c3", label="DARUC |nominal flow|", zorder=2,
    )
    ax.barh(
        y_pos, rec_df["margin"], height=0.6,
        left=rec_df["nominal_abs"].values,
        color="#f4a582", label=r"DARUC robust margin ($\rho \cdot \|Z_{\mathrm{line}}\|$)",
        zorder=2,
    )

    ax.barh(
        y_pos, rec_df["Fmax"], height=0.65,
        color="none", edgecolor="black", linewidth=1.2,
        label="Fmax", zorder=3,
    )

    ax.scatter(
        rec_df["dam_nominal_abs"], y_pos,
        marker="D", s=50, c="#2166ac", edgecolors="black", linewidths=0.5,
        zorder=4, label="DAM w/Res |nominal|",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=FONT_SIZES_TWO_COL["small"])
    ax.set_xlabel("Flow (MW)", fontsize=FONT_SIZES_TWO_COL["medium"])

    ax.legend(loc="upper right", fontsize=FONT_SIZES_TWO_COL["small"])
    ax.set_xlim(0, rec_df["Fmax"].max() * 1.08)

    fig.tight_layout()
    _save_figure(fig, out_dir / "fig_flow_decomposition")


# ---------------------------------------------------------------------------
# Chart 3b: Flow decomposition snapshot at a specific hour
# ---------------------------------------------------------------------------

def plot_flow_decomposition_hour(
    dam_raw: pd.DataFrame,
    daruc_raw: pd.DataFrame,
    hour: int,
    out_dir: Path,
    top_n: int = 20,
):
    """Horizontal stacked bars for ALL monitored lines at a specific hour.

    Shows top_n lines sorted by worst-case loading. Includes both binding
    and non-binding lines to give the full picture at that snapshot.
    """
    hour_ts = _find_hour_ts(dam_raw, hour)
    if hour_ts is None:
        print(f"  WARNING: hour {hour} not found, skipping hourly decomposition")
        return

    daruc_h = daruc_raw[daruc_raw["period"] == hour_ts].copy()
    dam_h = dam_raw[dam_raw["period"] == hour_ts].copy()

    if daruc_h.empty:
        print(f"  No DARUC data at hour {hour}")
        return

    # Sort by worst-case loading, take top N
    daruc_h = daruc_h.sort_values("loading_worst_case_pct", ascending=False).head(top_n)
    lines_sorted = daruc_h["line"].tolist()

    # Merge DAM data
    dam_lookup = dam_h.set_index("line")

    records = []
    for _, row in daruc_h.iterrows():
        line = row["line"]
        dam_nom = abs(dam_lookup.loc[line, "flow_nominal"]) if line in dam_lookup.index else 0
        records.append({
            "line": line,
            "nominal_abs": abs(row["flow_nominal"]),
            "margin": row["margin_rho_norm"],
            "Fmax": row["Fmax"],
            "dam_nominal_abs": dam_nom,
            "wc_loading": row["loading_worst_case_pct"],
            "binding": row["binding"],
        })

    rec_df = pd.DataFrame(records)
    # Reverse so highest loading is at top
    rec_df = rec_df.iloc[::-1].reset_index(drop=True)

    n = len(rec_df)
    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, max(3, n * 0.35 + 1.5)))

    y_pos = np.arange(n)

    ax.barh(y_pos, rec_df["nominal_abs"], height=0.6,
            color="#4393c3", label="DARUC |nominal|", zorder=2)
    ax.barh(y_pos, rec_df["margin"], height=0.6,
            left=rec_df["nominal_abs"].values,
            color="#e08060", label="DARUC robust margin", zorder=2)

    ax.barh(y_pos, rec_df["Fmax"], height=0.65,
            color="none", edgecolor="black", linewidth=1.0,
            label="Fmax", zorder=3)

    ax.scatter(rec_df["dam_nominal_abs"], y_pos,
               marker="D", s=40, c="#2166ac", edgecolors="black", linewidths=0.4,
               zorder=4, label="DAM w/Res |nominal|")

    labels = rec_df["line"].tolist()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES_TWO_COL["small"])
    for i, (_, row) in enumerate(rec_df.iterrows()):
        if row["binding"]:
            ax.get_yticklabels()[i].set_color("#b2182b")
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Flow (MW)", fontsize=FONT_SIZES_TWO_COL["medium"])

    ax.legend(loc="upper right", fontsize=FONT_SIZES_TWO_COL["small"])
    ax.set_xlim(0, rec_df["Fmax"].max() * 1.08)

    fig.tight_layout()
    _save_figure(fig, out_dir / f"fig_flow_decomp_h{hour:02d}")


def plot_flow_decomposition_binding(
    dam_raw: pd.DataFrame,
    daruc_raw: pd.DataFrame,
    binding_lines: list[str],
    hour: int,
    out_dir: Path,
):
    """Stacked bar decomposition at a specific hour, restricted to binding-union lines."""
    hour_ts = _find_hour_ts(dam_raw, hour)
    if hour_ts is None:
        print(f"  WARNING: hour {hour} not found, skipping binding decomposition")
        return

    daruc_h = daruc_raw[
        (daruc_raw["period"] == hour_ts) & (daruc_raw["line"].isin(binding_lines))
    ].copy()
    dam_h = dam_raw[
        (dam_raw["period"] == hour_ts) & (dam_raw["line"].isin(binding_lines))
    ].copy()

    if daruc_h.empty:
        print(f"  No DARUC data for binding lines at hour {hour}")
        return

    dam_lookup = dam_h.set_index("line")

    records = []
    for _, row in daruc_h.iterrows():
        line = row["line"]
        dam_nom = abs(dam_lookup.loc[line, "flow_nominal"]) if line in dam_lookup.index else 0
        dam_wc = dam_lookup.loc[line, "worst_case_abs_flow"] if line in dam_lookup.index else dam_nom
        records.append({
            "line": line,
            "nominal_abs": abs(row["flow_nominal"]),
            "margin": row["margin_rho_norm"],
            "Fmax": row["Fmax"],
            "dam_nominal_abs": dam_nom,
            "dam_wc_abs": dam_wc,
            "wc_loading": row["loading_worst_case_pct"],
            "binding": row["binding"],
        })

    rec_df = pd.DataFrame(records).sort_values("wc_loading", ascending=True).reset_index(drop=True)

    n = len(rec_df)
    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, max(2.5, n * 0.55 + 1.5)))
    y_pos = np.arange(n)

    ax.barh(y_pos, rec_df["nominal_abs"], height=0.6,
            color="#4393c3", label="DARUC |nominal|", zorder=2)
    ax.barh(y_pos, rec_df["margin"], height=0.6,
            left=rec_df["nominal_abs"].values,
            color="#e08060", label="DARUC robust margin", zorder=2)

    ax.barh(y_pos, rec_df["Fmax"], height=0.65,
            color="none", edgecolor="black", linewidth=1.0,
            label="Fmax", zorder=3)

    ax.scatter(rec_df["dam_wc_abs"], y_pos,
               marker="D", s=50, c="#2166ac", edgecolors="black", linewidths=0.5,
               zorder=4, label="DAM w/Res worst-case")

    labels = rec_df["line"].tolist()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES_TWO_COL["small"])
    for i, (_, row) in enumerate(rec_df.iterrows()):
        if row["binding"]:
            ax.get_yticklabels()[i].set_color("#b2182b")
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Flow (MW)", fontsize=FONT_SIZES_TWO_COL["medium"])

    ax.legend(loc="upper right", fontsize=FONT_SIZES_TWO_COL["small"])
    ax.set_xlim(0, rec_df["Fmax"].max() * 1.08)

    fig.tight_layout()
    _save_figure(fig, out_dir / f"fig_flow_decomp_binding_h{hour:02d}")


def plot_flow_decomposition_binding_only(
    dam_raw: pd.DataFrame,
    daruc_raw: pd.DataFrame,
    hour: int,
    out_dir: Path,
):
    """Stacked bar decomposition at a specific hour, restricted to lines binding in DARUC at that hour."""
    hour_ts = _find_hour_ts(dam_raw, hour)
    if hour_ts is None:
        print(f"  WARNING: hour {hour} not found, skipping binding-only decomposition")
        return

    daruc_h = daruc_raw[daruc_raw["period"] == hour_ts].copy()
    # Keep only lines that are actually binding at this hour
    daruc_h = daruc_h[daruc_h["binding"]].copy()

    if daruc_h.empty:
        print(f"  No binding lines in DARUC at hour {hour}")
        return

    binding_lines = daruc_h["line"].unique().tolist()
    dam_h = dam_raw[
        (dam_raw["period"] == hour_ts) & (dam_raw["line"].isin(binding_lines))
    ].copy()
    dam_lookup = dam_h.set_index("line")

    records = []
    for _, row in daruc_h.iterrows():
        line = row["line"]
        dam_nom = abs(dam_lookup.loc[line, "flow_nominal"]) if line in dam_lookup.index else 0
        dam_wc = dam_lookup.loc[line, "worst_case_abs_flow"] if line in dam_lookup.index else dam_nom
        records.append({
            "line": line,
            "nominal_abs": abs(row["flow_nominal"]),
            "margin": row["margin_rho_norm"],
            "Fmax": row["Fmax"],
            "dam_nominal_abs": dam_nom,
            "dam_wc_abs": dam_wc,
            "wc_loading": row["loading_worst_case_pct"],
        })

    rec_df = pd.DataFrame(records).sort_values("wc_loading", ascending=True).reset_index(drop=True)

    n = len(rec_df)
    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, max(2.5, n * 0.55 + 1.5)))
    y_pos = np.arange(n)

    ax.barh(y_pos, rec_df["nominal_abs"], height=0.6,
            color="#4393c3", label="DARUC |nominal|", zorder=2)
    ax.barh(y_pos, rec_df["margin"], height=0.6,
            left=rec_df["nominal_abs"].values,
            color="#e08060", label="DARUC robust margin", zorder=2)

    ax.barh(y_pos, rec_df["Fmax"], height=0.65,
            color="none", edgecolor="black", linewidth=1.0,
            label="Fmax", zorder=3)

    ax.scatter(rec_df["dam_wc_abs"], y_pos,
               marker="D", s=50, c="#2166ac", edgecolors="black", linewidths=0.5,
               zorder=4, label="DAM w/Res worst-case")

    labels = rec_df["line"].tolist()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES_TWO_COL["small"])
    # All lines are binding, highlight them all
    for tick in ax.get_yticklabels():
        tick.set_color("#b2182b")
        tick.set_fontweight("bold")

    ax.set_xlabel("Flow (MW)", fontsize=FONT_SIZES_TWO_COL["medium"])

    ax.set_xlim(0, rec_df["Fmax"].max() * 1.08)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=FONT_SIZES_TWO_COL["small"], frameon=False)

    fig.tight_layout()
    _save_figure(fig, out_dir / f"fig_flow_decomp_binding_only_h{hour:02d}")


def _find_hour_ts(df, hour):
    """Find the pd.Timestamp matching the given hour (0-23)."""
    for t in sorted(df["period"].unique()):
        if t.hour == hour:
            return t
    return None


# ---------------------------------------------------------------------------
# Chart 4: Network map at a specific hour
# ---------------------------------------------------------------------------

_RTS_DATA = Path(__file__).parent / "RTS_Data" / "SourceData"


def load_network_topology(
    data_dir: Path = _RTS_DATA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bus (with lat/lng) and branch (with from/to bus) data."""
    bus = pd.read_csv(data_dir / "bus.csv")
    bus = bus.set_index("Bus ID")
    branch = pd.read_csv(data_dir / "branch.csv")
    branch = branch.set_index("UID")
    return bus, branch


# Wind generator bus IDs (from gen.csv: 122_WIND_1, 303_WIND_1, 309_WIND_1, 317_WIND_1)
_WIND_BUSES = {122, 303, 309, 317}

# Shared constants
_LW_MIN, _LW_MAX = 0.8, 6.0


def _get_hour_data(flow_df, hour_ts):
    """Extract flow data for a specific hour, indexed by line UID."""
    return flow_df[flow_df["period"] == hour_ts].set_index("line")


def _draw_base_network(ax, bus, branch):
    """Draw gray background branches and bus dots (no labels)."""
    for uid, row in branch.iterrows():
        fb, tb = row["From Bus"], row["To Bus"]
        if fb not in bus.index or tb not in bus.index:
            continue
        x = [bus.loc[fb, "lng"], bus.loc[tb, "lng"]]
        y = [bus.loc[fb, "lat"], bus.loc[tb, "lat"]]
        ax.plot(x, y, color="#e0e0e0", linewidth=0.5, zorder=1)

    # Non-wind buses: small gray dots
    non_wind = bus[~bus.index.isin(_WIND_BUSES)]
    ax.scatter(non_wind["lng"], non_wind["lat"], s=8, c="#999999", zorder=5,
               edgecolors="none")

    # Wind buses: green triangles, no text labels
    wind = bus[bus.index.isin(_WIND_BUSES)]
    ax.scatter(wind["lng"], wind["lat"], s=80, c="#2ca02c", marker="^", zorder=6,
               edgecolors="black", linewidths=0.6)


def _get_branch_coords(uid, branch, bus):
    """Return (x_array, y_array) for a branch, or None if buses missing."""
    if uid not in branch.index:
        return None
    br = branch.loc[uid]
    fb, tb = br["From Bus"], br["To Bus"]
    if fb not in bus.index or tb not in bus.index:
        return None
    x = np.array([bus.loc[fb, "lng"], bus.loc[tb, "lng"]])
    y = np.array([bus.loc[fb, "lat"], bus.loc[tb, "lat"]])
    return x, y


def _perpendicular_offset(x, y, offset_pts):
    """Compute a perpendicular offset for parallel lines (in data coords)."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return np.zeros(2), np.zeros(2)
    # Unit normal (perpendicular)
    nx = -dy / length * offset_pts
    ny = dx / length * offset_pts
    return np.array([nx, nx]), np.array([ny, ny])


def _label_binding(ax, x, y, uid, extra=""):
    """Add a small label at midpoint for binding lines."""
    mx, my = 0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1])
    txt = uid + (f"\n{extra}" if extra else "")
    ax.annotate(
        txt, (mx, my), fontsize=5, fontweight="bold",
        color="#b2182b", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#b2182b",
                  alpha=0.85, lw=0.5),
        zorder=8,
    )


def _style_map_ax(ax, title):
    """Common axis styling for map panels."""
    ax.set_title(title, fontsize=FONT_SIZES_TWO_COL["large"])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FONT_SIZES_TWO_COL["small"] - 2)
    ax.set_xlabel("")
    ax.set_ylabel("")


# ---- Option A: Parallel offset lines ----

def _draw_panel_parallel(ax, bus, branch, hour_data, title):
    """Two parallel lines per branch: blue=nominal, orange=margin.
    Width proportional to MW value. Binding lines get red outline."""
    _draw_base_network(ax, bus, branch)

    offset_scale = 0.012  # data-coord offset between the two parallel lines

    for uid in hour_data.index:
        coords = _get_branch_coords(uid, branch, bus)
        if coords is None:
            continue
        x, y = coords
        row = hour_data.loc[uid]
        fmax = row["Fmax"]
        nom_abs = abs(row["flow_nominal"])
        margin = row["margin_rho_norm"]
        is_binding = bool(row["binding"])

        nom_frac = min(nom_abs / fmax, 1.0) if fmax > 0 else 0
        margin_frac = min(margin / fmax, 1.0) if fmax > 0 else 0

        lw_nom = _LW_MIN + (_LW_MAX - _LW_MIN) * nom_frac
        lw_margin = _LW_MIN + (_LW_MAX - _LW_MIN) * margin_frac

        ox, oy = _perpendicular_offset(x, y, offset_scale)

        # Nominal (blue) on one side
        ax.plot(x + ox, y + oy, color="#4393c3", linewidth=lw_nom, zorder=3,
                solid_capstyle="round")
        # Margin (orange) on the other side
        if margin > 0.1:
            ax.plot(x - ox, y - oy, color="#e08060", linewidth=lw_margin, zorder=3,
                    solid_capstyle="round")

        if is_binding:
            ax.plot(x, y, color="#b2182b", linewidth=max(lw_nom, lw_margin) + 2,
                    zorder=2, alpha=0.25)
            _label_binding(ax, x, y, uid, f"{nom_abs:.0f}+{margin:.0f}")

    _style_map_ax(ax, title)


# ---- Option B: Three-panel (DAM loading | DARUC nominal | DARUC margin) ----

def _draw_panel_single_metric(ax, bus, branch, hour_data, title,
                               metric_col, color, vmax_mw=None):
    """Single metric per line, thickness = MW value, uniform color."""
    _draw_base_network(ax, bus, branch)

    if vmax_mw is None:
        vmax_mw = hour_data[metric_col].abs().max()
    if vmax_mw < 1:
        vmax_mw = 1.0

    for uid in hour_data.index:
        coords = _get_branch_coords(uid, branch, bus)
        if coords is None:
            continue
        x, y = coords
        row = hour_data.loc[uid]
        val = abs(row[metric_col]) if metric_col == "flow_nominal" else row[metric_col]
        is_binding = bool(row["binding"])

        frac = min(val / vmax_mw, 1.0)
        lw = _LW_MIN + (_LW_MAX - _LW_MIN) * frac

        ax.plot(x, y, color=color, linewidth=lw, zorder=3, solid_capstyle="round",
                alpha=max(0.3, frac))

        if is_binding:
            ax.plot(x, y, color="#b2182b", linewidth=lw + 2, zorder=2, alpha=0.25)
            _label_binding(ax, x, y, uid)

    _style_map_ax(ax, title)


# ---- Option C: Color = nominal %, thickness = worst-case % ----

def _draw_panel_dual_encode(ax, bus, branch, hour_data, title):
    """Color encodes nominal loading %, thickness encodes worst-case loading %."""
    _draw_base_network(ax, bus, branch)

    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=0, vmax=100)

    for uid in hour_data.index:
        coords = _get_branch_coords(uid, branch, bus)
        if coords is None:
            continue
        x, y = coords
        row = hour_data.loc[uid]
        fmax = row["Fmax"]
        wc_abs = row["worst_case_abs_flow"]
        is_binding = bool(row["binding"])

        wc_frac = min(wc_abs / fmax, 1.0) if fmax > 0 else 0
        lw = _LW_MIN + (_LW_MAX - _LW_MIN) * wc_frac

        nom_loading = row["loading_nominal_pct"]
        color = cmap(norm(min(nom_loading, 100)))
        ax.plot(x, y, color=color, linewidth=lw, zorder=3, solid_capstyle="round")

        if is_binding:
            ax.plot(x, y, color="#b2182b", linewidth=lw + 2, zorder=2, alpha=0.25)
            margin = row["margin_rho_norm"]
            nom_abs = abs(row["flow_nominal"])
            _label_binding(ax, x, y, uid,
                           f"{nom_abs:.0f}+{margin:.0f}" if margin > 0.1 else "")

    _style_map_ax(ax, title)
    return cmap, norm


# ---- Master plot functions ----

def plot_network_maps(
    dam_raw: pd.DataFrame,
    daruc_raw: pd.DataFrame,
    hour: int,
    out_dir: Path,
):
    """Generate three network map options for comparison."""
    from matplotlib.lines import Line2D

    bus, branch = load_network_topology()

    hour_ts = _find_hour_ts(dam_raw, hour)
    if hour_ts is None:
        print(f"  WARNING: hour {hour} not found, skipping network maps")
        return

    dam_hour = _get_hour_data(dam_raw, hour_ts)
    daruc_hour = _get_hour_data(daruc_raw, hour_ts)

    # Shared Fmax for consistent scaling across panels
    fmax_max = max(dam_hour["Fmax"].max(), daruc_hour["Fmax"].max())

    # --- Option A: Parallel offset (2 panels) ---
    fig_a, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_TWO_COL_WIDTH, 4.0))
    _draw_panel_parallel(ax1, bus, branch, dam_hour,
                         f"DAM + Reserve (h{hour:02d})")
    _draw_panel_parallel(ax2, bus, branch, daruc_hour,
                         f"DARUC (h{hour:02d})")
    legend_a = [
        Line2D([0], [0], color="#4393c3", lw=4, label="Nominal flow"),
        Line2D([0], [0], color="#e08060", lw=4, label="Robust margin"),
        Line2D([0], [0], color="#b2182b", lw=5, alpha=0.25, label="Binding"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=8, label="Wind gen"),
    ]
    fig_a.legend(handles=legend_a, loc="lower center", ncol=4,
                 fontsize=FONT_SIZES_TWO_COL["small"],
                 bbox_to_anchor=(0.5, -0.01))
    fig_a.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save_figure(fig_a, out_dir / f"fig_map_A_parallel_h{hour:02d}")

    # --- Option B: Three panels (DAM wc | DARUC nominal | DARUC margin) ---
    fig_b, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(IEEE_TWO_COL_WIDTH * 1.5, 3.5))
    _draw_panel_single_metric(bx1, bus, branch, dam_hour,
                               f"DAM w/Res worst-case (h{hour:02d})",
                               "worst_case_abs_flow", "#4393c3", vmax_mw=fmax_max)
    _draw_panel_single_metric(bx2, bus, branch, daruc_hour,
                               f"DARUC nominal (h{hour:02d})",
                               "flow_nominal", "#4393c3", vmax_mw=fmax_max)
    _draw_panel_single_metric(bx3, bus, branch, daruc_hour,
                               f"DARUC margin only (h{hour:02d})",
                               "margin_rho_norm", "#e08060", vmax_mw=fmax_max)
    legend_b = [
        Line2D([0], [0], color="#4393c3", lw=4, label="Flow (thickness=MW)"),
        Line2D([0], [0], color="#e08060", lw=4, label="Margin (thickness=MW)"),
        Line2D([0], [0], color="#b2182b", lw=5, alpha=0.25, label="Binding"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=8, label="Wind gen"),
    ]
    fig_b.legend(handles=legend_b, loc="lower center", ncol=4,
                 fontsize=FONT_SIZES_TWO_COL["small"],
                 bbox_to_anchor=(0.5, -0.01))
    fig_b.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save_figure(fig_b, out_dir / f"fig_map_B_three_panel_h{hour:02d}")

    # --- Option C: Dual encoding (2 panels) ---
    fig_c, (cx1, cx2) = plt.subplots(1, 2, figsize=(IEEE_TWO_COL_WIDTH, 4.0))
    _draw_panel_dual_encode(cx1, bus, branch, dam_hour,
                            f"DAM + Reserve (h{hour:02d})")
    cmap_c, norm_c = _draw_panel_dual_encode(
        cx2, bus, branch, daruc_hour, f"DARUC (h{hour:02d})")
    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    fig_c.colorbar(sm, ax=[cx1, cx2], label="Nominal Loading %", shrink=0.7, pad=0.02)
    legend_c = [
        Line2D([0], [0], color="gray", lw=1, label="Thin = low worst-case"),
        Line2D([0], [0], color="gray", lw=5, label="Thick = high worst-case"),
        Line2D([0], [0], color="#b2182b", lw=5, alpha=0.25, label="Binding"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=8, label="Wind gen"),
    ]
    fig_c.legend(handles=legend_c, loc="lower center", ncol=4,
                 fontsize=FONT_SIZES_TWO_COL["small"],
                 bbox_to_anchor=(0.5, -0.01))
    fig_c.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save_figure(fig_c, out_dir / f"fig_map_C_dual_encode_h{hour:02d}")


# ---------------------------------------------------------------------------
# Chart 5: Z matrix for a single hour (DARUC only)
# ---------------------------------------------------------------------------

# Wind generator names matching the 4 uncertainty sources (k=0..3)
_WIND_GEN_NAMES = ["309_WIND_1", "317_WIND_1", "303_WIND_1", "122_WIND_1"]


def load_z_matrix(case_dir: Path) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """Load Z_coefficients.csv and return (Z_3d, gen_ids, timestamps, k_vals).

    Returns Z as shape (n_gens, n_periods, n_k).
    """
    z_path = case_dir / "daruc" / "Z_coefficients.csv"
    if not z_path.exists():
        raise FileNotFoundError(f"Z_coefficients.csv not found at {z_path}")

    # Row 0 = timestamps, Row 1 = k indices, Rows 2+ = gen data
    raw = pd.read_csv(z_path, header=None)

    timestamps_row = raw.iloc[0, 1:].values.astype(str)
    k_row = raw.iloc[1, 1:].values.astype(int)
    gen_ids = raw.iloc[2:, 0].values.astype(str).tolist()
    data = raw.iloc[2:, 1:].values.astype(float)

    # Unique timestamps and k values
    unique_times = list(dict.fromkeys(timestamps_row))  # preserves order
    unique_k = sorted(set(k_row))
    n_k = len(unique_k)
    n_t = len(unique_times)
    n_gen = len(gen_ids)

    # Reshape to (n_gen, n_t, n_k)
    Z = data.reshape(n_gen, n_t, n_k)

    return Z, gen_ids, unique_times, unique_k


def _get_hour_z(Z, gen_ids, timestamps, hour):
    """Extract Z[:, t, :] for the given hour. Returns (Z_hour, active_mask, hour_label)."""
    hour_str = f"{hour:02d}:00:00"
    t_idx = None
    for i, ts in enumerate(timestamps):
        if hour_str in ts:
            t_idx = i
            break
    if t_idx is None:
        return None, None, None

    Z_hour = Z[:, t_idx, :]  # (n_gen, n_k)
    return Z_hour, timestamps[t_idx]


def _filter_active_gens(Z_hour, gen_ids, threshold=0.001):
    """Return indices and names of generators with max|Z| > threshold."""
    max_abs = np.max(np.abs(Z_hour), axis=1)
    mask = max_abs > threshold
    active_idx = np.where(mask)[0]
    active_names = [gen_ids[i] for i in active_idx]
    return active_idx, active_names


def plot_z_matrix(case_dir: Path, hour: int, out_dir: Path, threshold: float = 0.001):
    """Generate multiple Z matrix visualizations for a single hour."""
    from matplotlib.lines import Line2D

    Z, gen_ids, timestamps, k_vals = load_z_matrix(case_dir)
    result = _get_hour_z(Z, gen_ids, timestamps, hour)
    if result[0] is None:
        print(f"  WARNING: hour {hour} not found in Z data, skipping")
        return

    Z_hour, hour_label = result
    active_idx, active_names = _filter_active_gens(Z_hour, gen_ids, threshold)

    if len(active_idx) == 0:
        print(f"  No generators with |Z| > {threshold} at hour {hour}")
        return

    Z_active = Z_hour[active_idx]  # (n_active, n_k)
    n_active = len(active_names)
    n_k = len(k_vals)

    # Wind source labels
    k_labels = [_WIND_GEN_NAMES[k] if k < len(_WIND_GEN_NAMES) else f"k={k}"
                for k in k_vals]

    # Sort: wind gens first, then by row norm descending
    is_wind = np.array([1 if "WIND" in n else 0 for n in active_names])
    row_norms = np.linalg.norm(Z_active, axis=1)
    sort_order = np.lexsort((-row_norms, -is_wind))
    Z_active = Z_active[sort_order]
    active_names = [active_names[i] for i in sort_order]
    is_wind = is_wind[sort_order]
    row_norms = row_norms[sort_order]

    # Color-code gen labels: green for wind, black for thermal
    label_colors = ["#2ca02c" if w else "black" for w in is_wind]

    print(f"  {n_active} active generators (|Z| > {threshold}) at hour {hour}")

    # --- Option A: Annotated heatmap (diverging) ---
    fig_a, ax = plt.subplots(figsize=(max(5, n_k * 1.5 + 2), max(4, n_active * 0.35 + 1.5)))
    vabs = np.max(np.abs(Z_active))
    im = ax.imshow(Z_active, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   interpolation="nearest")
    for i in range(n_active):
        for j in range(n_k):
            val = Z_active[i, j]
            if abs(val) > threshold:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=FONT_SIZES_TWO_COL["small"] - 3,
                        color="white" if abs(val) > 0.5 * vabs else "black")

    ax.set_xticks(range(n_k))
    ax.set_xticklabels(k_labels, fontsize=FONT_SIZES_TWO_COL["small"], rotation=30, ha="right")
    ax.set_yticks(range(n_active))
    ax.set_yticklabels(active_names, fontsize=FONT_SIZES_TWO_COL["small"] - 2)
    for i, c in enumerate(label_colors):
        ax.get_yticklabels()[i].set_color(c)
    ax.set_xlabel("Uncertainty Source (Wind Generator)", fontsize=FONT_SIZES_TWO_COL["medium"])
    fig_a.colorbar(im, ax=ax, label="Z coefficient", shrink=0.8)
    fig_a.tight_layout()
    _save_figure(fig_a, out_dir / f"fig_z_A_heatmap_h{hour:02d}")

    # --- Option B: Grouped horizontal bars (one color per wind source) ---
    fig_b, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, max(4, n_active * 0.4 + 1.5)))
    bar_height = 0.8 / n_k
    colors_k = ["#4393c3", "#e08060", "#66c2a5", "#984ea3"]
    for j in range(n_k):
        offsets = np.arange(n_active) + (j - n_k / 2 + 0.5) * bar_height
        ax.barh(offsets, Z_active[:, j], height=bar_height,
                color=colors_k[j % len(colors_k)], label=k_labels[j],
                edgecolor="white", linewidth=0.3)

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(range(n_active))
    ax.set_yticklabels(active_names, fontsize=FONT_SIZES_TWO_COL["small"] - 2)
    for i, c in enumerate(label_colors):
        ax.get_yticklabels()[i].set_color(c)
    ax.set_xlabel("Z coefficient value", fontsize=FONT_SIZES_TWO_COL["medium"])
    ax.legend(fontsize=FONT_SIZES_TWO_COL["small"], loc="lower right", title="Wind Source")
    ax.invert_yaxis()
    fig_b.tight_layout()
    _save_figure(fig_b, out_dir / f"fig_z_B_bars_h{hour:02d}")

    # --- Option C: Row-norm bar + stacked contribution ---
    fig_c, (cx1, cx2) = plt.subplots(1, 2, figsize=(IEEE_TWO_COL_WIDTH, max(4, n_active * 0.35 + 1.5)),
                                      gridspec_kw={"width_ratios": [1, 2]}, sharey=True)

    cx1.barh(range(n_active), np.linalg.norm(Z_active, axis=1),
             color=["#2ca02c" if w else "#4393c3" for w in is_wind],
             edgecolor="white", linewidth=0.3)
    cx1.set_xlabel(r"$\|Z_{\mathrm{row}}\|$", fontsize=FONT_SIZES_TWO_COL["medium"])
    cx1.set_yticks(range(n_active))
    cx1.set_yticklabels(active_names, fontsize=FONT_SIZES_TWO_COL["small"] - 2)
    for i, c in enumerate(label_colors):
        cx1.get_yticklabels()[i].set_color(c)
    cx1.set_title("Row Norm", fontsize=FONT_SIZES_TWO_COL["medium"])
    cx1.invert_yaxis()

    left = np.zeros(n_active)
    for j in range(n_k):
        abs_vals = np.abs(Z_active[:, j])
        cx2.barh(range(n_active), abs_vals, left=left, height=0.7,
                 color=colors_k[j % len(colors_k)], label=k_labels[j],
                 edgecolor="white", linewidth=0.3)
        left += abs_vals

    cx2.set_xlabel("|Z| contribution", fontsize=FONT_SIZES_TWO_COL["medium"])
    cx2.legend(fontsize=FONT_SIZES_TWO_COL["small"] - 2, loc="lower right",
               title="Wind Source", title_fontsize=FONT_SIZES_TWO_COL["small"] - 2)
    cx2.set_title("Stacked |Z| by Source", fontsize=FONT_SIZES_TWO_COL["medium"])

    fig_c.tight_layout()
    _save_figure(fig_c, out_dir / f"fig_z_C_norm_stack_h{hour:02d}")

    # --- Option D: Stacked |Z| by source — thermal units only ---
    thermal_mask = is_wind == 0
    if thermal_mask.any():
        Z_thermal = Z_active[thermal_mask]
        thermal_names = [n for n, w in zip(active_names, is_wind) if not w]
        n_th = len(thermal_names)

        # Sort by row norm descending
        th_norms = np.linalg.norm(Z_thermal, axis=1)
        th_order = np.argsort(-th_norms)
        Z_thermal = Z_thermal[th_order]
        thermal_names = [thermal_names[i] for i in th_order]

        fig_d, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, max(3, n_th * 0.35 + 1.5)))

        left = np.zeros(n_th)
        for j in range(n_k):
            abs_vals = np.abs(Z_thermal[:, j])
            ax.barh(range(n_th), abs_vals, left=left, height=0.7,
                    color=colors_k[j % len(colors_k)], label=k_labels[j],
                    edgecolor="white", linewidth=0.3)
            left += abs_vals

        ax.set_yticks(range(n_th))
        ax.set_yticklabels(thermal_names, fontsize=FONT_SIZES_TWO_COL["small"] - 2)
        ax.set_xlabel("|Z| contribution", fontsize=FONT_SIZES_TWO_COL["medium"])
        ax.legend(fontsize=FONT_SIZES_TWO_COL["small"], loc="lower right",
                  title="Wind Source", title_fontsize=FONT_SIZES_TWO_COL["small"])
        ax.invert_yaxis()
        fig_d.tight_layout()
        _save_figure(fig_d, out_dir / f"fig_z_D_thermal_stack_h{hour:02d}")
    else:
        print(f"  No active thermal generators at hour {hour}, skipping Option D")


# ---------------------------------------------------------------------------
# Chart 6: Network map with additionally committed units
# ---------------------------------------------------------------------------

def load_deviation_summary(case_dir: Path) -> pd.DataFrame:
    """Load deviation_summary.csv with parsed periods_added lists."""
    path = case_dir / "daruc" / "deviation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"deviation_summary.csv not found at {path}")
    df = pd.read_csv(path)
    # Parse periods_added from string repr of list
    import ast
    df["periods_added"] = df["periods_added"].apply(ast.literal_eval)
    return df


def _gen_id_to_bus(gen_id: str) -> int:
    """Extract bus ID from generator UID (e.g. '207_CT_1' -> 207)."""
    return int(gen_id.split("_")[0])


def plot_network_commitment_map(
    case_dir: Path,
    dam_raw: pd.DataFrame,
    daruc_raw: pd.DataFrame,
    hour: int,
    out_dir: Path,
):
    """Network map showing line flows + additionally committed generators at a given hour.

    Uses deviation_summary.csv to identify which generators DARUC commits
    beyond what DAM w/ Res committed, and highlights them on the map.
    """
    from matplotlib.lines import Line2D

    bus, branch = load_network_topology()
    dev = load_deviation_summary(case_dir)

    hour_ts = _find_hour_ts(dam_raw, hour)
    if hour_ts is None:
        print(f"  WARNING: hour {hour} not found, skipping commitment map")
        return

    daruc_hour = _get_hour_data(daruc_raw, hour_ts)

    # Find generators additionally committed at this hour (period index = hour for day-1)
    extra_gens = []
    for _, row in dev.iterrows():
        if hour in row["periods_added"]:
            extra_gens.append({
                "gen_id": row["gen_id"],
                "gen_type": row["gen_type"],
                "bus_id": _gen_id_to_bus(row["gen_id"]),
                "extra_hours": row["extra_committed_hours"],
            })

    extra_df = pd.DataFrame(extra_gens) if extra_gens else pd.DataFrame(
        columns=["gen_id", "gen_type", "bus_id", "extra_hours"]
    )

    # Group by bus for display (multiple units may be at same bus)
    bus_extra = {}
    if not extra_df.empty:
        for bus_id, grp in extra_df.groupby("bus_id"):
            bus_extra[bus_id] = grp["gen_id"].tolist()

    print(f"  {len(extra_gens)} extra-committed generators at hour {hour}: "
          f"{[g['gen_id'] for g in extra_gens]}")

    fig, ax = plt.subplots(figsize=(IEEE_TWO_COL_WIDTH, 5.0))

    # Draw base network
    _draw_base_network(ax, bus, branch)

    # Draw DARUC line flows (parallel style: nominal + margin)
    offset_scale = 0.012
    for uid in daruc_hour.index:
        coords = _get_branch_coords(uid, branch, bus)
        if coords is None:
            continue
        x, y = coords
        row = daruc_hour.loc[uid]
        fmax = row["Fmax"]
        nom_abs = abs(row["flow_nominal"])
        margin = row["margin_rho_norm"]
        is_binding = bool(row["binding"])

        nom_frac = min(nom_abs / fmax, 1.0) if fmax > 0 else 0
        margin_frac = min(margin / fmax, 1.0) if fmax > 0 else 0

        lw_nom = _LW_MIN + (_LW_MAX - _LW_MIN) * nom_frac
        lw_margin = _LW_MIN + (_LW_MAX - _LW_MIN) * margin_frac

        ox, oy = _perpendicular_offset(x, y, offset_scale)

        ax.plot(x + ox, y + oy, color="#4393c3", linewidth=lw_nom, zorder=3,
                solid_capstyle="round")
        if margin > 0.1:
            ax.plot(x - ox, y - oy, color="#e08060", linewidth=lw_margin, zorder=3,
                    solid_capstyle="round")

        if is_binding:
            ax.plot(x, y, color="#b2182b", linewidth=max(lw_nom, lw_margin) + 2,
                    zorder=2, alpha=0.25)
            _label_binding(ax, x, y, uid, f"{nom_abs:.0f}+{margin:.0f}")

    # Highlight additionally committed units
    for bus_id, gen_list in bus_extra.items():
        if bus_id not in bus.index:
            continue
        bx = bus.loc[bus_id, "lng"]
        by = bus.loc[bus_id, "lat"]
        n_units = len(gen_list)

        # Red square marker sized by number of units
        marker_size = 60 + 30 * n_units
        ax.scatter(bx, by, s=marker_size, c="#d62728", marker="s", zorder=7,
                   edgecolors="black", linewidths=0.8)

        # Label with short gen names
        short_names = [g.replace("_", " ") for g in gen_list]
        label_text = "\n".join(short_names)
        ax.annotate(
            label_text, (bx, by),
            fontsize=5.5, fontweight="bold", color="#d62728",
            ha="left", va="bottom",
            xytext=(5, 5), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#d62728",
                      alpha=0.9, lw=0.6),
            zorder=9,
        )

    _style_map_ax(ax, "")

    legend_elements = [
        Line2D([0], [0], color="#4393c3", lw=3, label="Nominal flow"),
        Line2D([0], [0], color="#e08060", lw=3, label="Robust margin"),
        Line2D([0], [0], color="#b2182b", lw=4, alpha=0.25, label="Binding line"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=7, label="Wind gen"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=7, label="Extra-committed unit"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=FONT_SIZES_TWO_COL["small"], framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, out_dir / f"fig_map_commitment_h{hour:02d}")


# ---------------------------------------------------------------------------
# Chart 7: Reserve distribution comparison (DAM w/Reserve vs DARUC)
# ---------------------------------------------------------------------------

def load_reserve_data(case_dir: Path):
    """Load reserve distribution (DAM) and reserve equivalent (DARUC) CSVs.

    Returns
    -------
    dam_r : DataFrame (gen_ids x day-1 times) — DAM w/Reserve r[i,t]
    daruc_eq : DataFrame (gen_ids x day-1 times) — DARUC reserve equivalent
    R_req : (T_day1,) array — system reserve requirement per period
    time_labels : list — day-1 column labels
    """
    dam_path = case_dir / "dam_reserve" / "reserve_distribution.csv"
    daruc_path = case_dir / "daruc" / "reserve_equivalent.csv"
    req_path = case_dir / "dam_reserve" / "reserve_requirement.npy"

    for p in [dam_path, daruc_path, req_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    dam_r = pd.read_csv(dam_path, index_col=0, parse_dates=True)
    daruc_eq = pd.read_csv(daruc_path, index_col=0, parse_dates=True)
    R_req = np.load(req_path)

    # Ensure RUC reserve equivalent is positive for upward response.
    # Old CSVs may have wrong sign (Z @ r_wc instead of -Z @ r_wc).
    daruc_eq = daruc_eq.abs()

    # Convert column labels to Timestamps for consistent matching
    dam_r.columns = pd.to_datetime(dam_r.columns)
    daruc_eq.columns = pd.to_datetime(daruc_eq.columns)

    # Align to common generators and time columns
    common_gens = dam_r.index.intersection(daruc_eq.index)
    common_times = dam_r.columns.intersection(daruc_eq.columns)
    dam_r = dam_r.loc[common_gens, common_times]
    daruc_eq = daruc_eq.loc[common_gens, common_times]

    # Restrict to day 1 (first 24 hourly periods)
    t0 = common_times.min()
    day1_cutoff = t0 + pd.Timedelta(hours=24)
    day1_cols = [c for c in common_times if c < day1_cutoff]
    dam_r = dam_r[day1_cols]
    daruc_eq = daruc_eq[day1_cols]
    R_req = R_req[:len(day1_cols)]

    return dam_r, daruc_eq, R_req, day1_cols


def plot_reserve_comparison(
    case_dir: Path,
    out_dir: Path,
    snapshot_hour: int = 16,
    top_n: int = 15,
):
    """Three-panel reserve distribution comparison: DAM w/Reserve vs DARUC.

    (a) System total reserve by period (day 1)
    (b) Per-unit allocation at snapshot hour (top generators)
    (c) Largest DARUC−DAM differences at snapshot hour
    """
    dam_r, daruc_eq, R_req, time_labels = load_reserve_data(case_dir)

    # Colors
    C_DAM = "#4682B4"
    C_DARUC = "#E07B39"
    C_REQ = "#808080"

    # --- Find snapshot column ---
    snap_cols = [c for c in time_labels if c.hour == snapshot_hour]
    if not snap_cols:
        # Fallback: closest hour
        hours = np.array([c.hour for c in time_labels])
        closest_idx = np.argmin(np.abs(hours - snapshot_hour))
        snap_cols = [time_labels[closest_idx]]
        print(f"  Hour {snapshot_hour}:00 not found, using {snap_cols[0]}")
    snap_col = snap_cols[0]

    # --- Hour labels for x-axis ---
    x_hours = np.array([c.hour + c.minute / 60 for c in time_labels])

    # --- Panel data ---
    dam_thermal = dam_r.loc[(dam_r != 0).any(axis=1)]  # thermal rows only
    # For DARUC, use only generators present in dam_thermal (thermals)
    thermal_gens = dam_thermal.index
    daruc_thermal = daruc_eq.loc[daruc_eq.index.isin(thermal_gens)]
    # Align indices
    common_thermal = thermal_gens.intersection(daruc_thermal.index)
    dam_thermal = dam_thermal.loc[common_thermal]
    daruc_thermal = daruc_thermal.loc[common_thermal]

    dam_total = dam_thermal.sum(axis=0).values
    daruc_total = daruc_thermal.sum(axis=0).values

    # ===================================================================
    # Build figure
    # ===================================================================
    fig, axes = plt.subplots(
        1, 3,
        figsize=(IEEE_TWO_COL_WIDTH * 1.4, 4.0),
        gridspec_kw={"width_ratios": [3, 3, 3]},
    )
    fs = FONT_SIZES_TWO_COL

    # --- (a) System total reserve by period ---
    ax = axes[0]
    ax.plot(x_hours, R_req, "--", color=C_REQ, linewidth=1.5, label="Requirement")
    ax.plot(x_hours, dam_total, "-", color=C_DAM, linewidth=1.5, label="DAM w/Res")
    ax.plot(x_hours, daruc_total, "-", color=C_DARUC, linewidth=1.5, label="DARUC")
    ax.axvline(snapshot_hour, color="k", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_xlabel("Hour", fontsize=fs["medium"])
    ax.set_ylabel("Reserve (MW)", fontsize=fs["medium"])
    ax.set_title("(a) System reserve by period", fontsize=fs["large"])
    ax.legend(fontsize=fs["small"], loc="best")
    ax.tick_params(labelsize=fs["small"])

    # --- (b) Per-unit allocation at snapshot hour ---
    ax = axes[1]
    dam_snap = dam_thermal[snap_col].sort_values(ascending=True)
    daruc_snap = daruc_thermal[snap_col].reindex(dam_snap.index)

    # Top generators by max(dam, daruc)
    combined_max = pd.concat([dam_snap.abs(), daruc_snap.abs()], axis=1).max(axis=1)
    top_gens = combined_max.nlargest(top_n).index
    dam_top = dam_snap.loc[top_gens].sort_values(ascending=True)
    daruc_top = daruc_snap.loc[dam_top.index]

    # Shorten labels
    labels = [g[:14] for g in dam_top.index]
    y_pos = np.arange(len(labels))
    bar_h = 0.35

    ax.barh(y_pos + bar_h / 2, dam_top.values, height=bar_h,
            color=C_DAM, label="DAM w/Res", edgecolor="white", linewidth=0.3)
    ax.barh(y_pos - bar_h / 2, daruc_top.values, height=bar_h,
            color=C_DARUC, label="DARUC", edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=fs["small"] - 1)
    ax.set_xlabel("Reserve (MW)", fontsize=fs["medium"])
    ax.set_title(f"(b) Top units at {snapshot_hour:02d}:00", fontsize=fs["large"])
    ax.legend(fontsize=fs["small"], loc="lower right")
    ax.tick_params(labelsize=fs["small"])

    # --- (c) Largest differences ---
    ax = axes[2]
    delta = daruc_thermal[snap_col] - dam_thermal[snap_col]
    # Filter to meaningful differences
    delta = delta[delta.abs() > 1.0].sort_values()

    if len(delta) == 0:
        ax.text(0.5, 0.5, "No significant\ndifferences",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=fs["medium"])
        ax.set_title(f"(c) DARUC\u2212DAM at {snapshot_hour:02d}:00", fontsize=fs["large"])
    else:
        # Show top differences (limit to reasonable number)
        if len(delta) > 20:
            # Keep top 10 positive and top 10 negative
            top_pos = delta.nlargest(10)
            top_neg = delta.nsmallest(10)
            delta = pd.concat([top_neg, top_pos]).sort_values()

        labels_c = [g[:14] for g in delta.index]
        y_pos_c = np.arange(len(labels_c))
        colors = [C_DARUC if v > 0 else C_DAM for v in delta.values]

        ax.barh(y_pos_c, delta.values, color=colors, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_yticks(y_pos_c)
        ax.set_yticklabels(labels_c, fontsize=fs["small"] - 1)
        ax.set_xlabel("\u0394 Reserve (MW)", fontsize=fs["medium"])
        ax.set_title(f"(c) DARUC\u2212DAM at {snapshot_hour:02d}:00", fontsize=fs["large"])
        ax.tick_params(labelsize=fs["small"])

    fig.tight_layout()
    _save_figure(fig, out_dir / "fig_reserve_comparison")


def plot_reserve_per_unit(
    case_dir: Path,
    out_dir: Path,
    snapshot_hour: int = 16,
    top_n: int = 15,
):
    """Single-column per-unit reserve allocation chart (DAM w/Reserve vs DARUC).

    Standalone version of panel (b) from plot_reserve_comparison.
    """
    dam_r, daruc_eq, _R_req, time_labels = load_reserve_data(case_dir)
    fs = FONT_SIZES  # single-column sizes

    # Colors
    C_DAM = "#4682B4"
    C_DARUC = "#E07B39"

    # Find snapshot column
    snap_cols = [c for c in time_labels if c.hour == snapshot_hour]
    if not snap_cols:
        hours = np.array([c.hour for c in time_labels])
        closest_idx = np.argmin(np.abs(hours - snapshot_hour))
        snap_cols = [time_labels[closest_idx]]
        print(f"  Hour {snapshot_hour}:00 not found, using {snap_cols[0]}")
    snap_col = snap_cols[0]

    # Thermal generators only (non-zero in DAM reserve)
    dam_thermal = dam_r.loc[(dam_r != 0).any(axis=1)]
    thermal_gens = dam_thermal.index
    daruc_thermal = daruc_eq.loc[daruc_eq.index.isin(thermal_gens)]
    common_thermal = thermal_gens.intersection(daruc_thermal.index)
    dam_thermal = dam_thermal.loc[common_thermal]
    daruc_thermal = daruc_thermal.loc[common_thermal]

    # Snapshot values
    dam_snap = dam_thermal[snap_col].sort_values(ascending=True)
    daruc_snap = daruc_thermal[snap_col].reindex(dam_snap.index)

    # Top generators by max(dam, daruc)
    combined_max = pd.concat([dam_snap.abs(), daruc_snap.abs()], axis=1).max(axis=1)
    top_gens = combined_max.nlargest(top_n).index
    dam_top = dam_snap.loc[top_gens].sort_values(ascending=True)
    daruc_top = daruc_snap.loc[dam_top.index]

    # Shorten labels
    labels = [g[:14] for g in dam_top.index]
    y_pos = np.arange(len(labels))
    bar_h = 0.35

    fig_h = max(3.0, len(labels) * 0.32 + 1.2)
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, fig_h))

    ax.barh(y_pos + bar_h / 2, dam_top.values, height=bar_h,
            color=C_DAM, label="DAM w/Res", edgecolor="white", linewidth=0.3)
    ax.barh(y_pos - bar_h / 2, daruc_top.values, height=bar_h,
            color=C_DARUC, label="DARUC", edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=fs["small"] - 1)
    ax.set_xlabel("Reserve-Equivalent (MW)", fontsize=fs["medium"])
    # ax.set_title(f"Per-unit reserve at {snapshot_hour:02d}:00", fontsize=fs["large"])
    ax.legend(fontsize=fs["small"], loc="lower right")
    ax.tick_params(labelsize=fs["small"])

    fig.tight_layout()
    _save_figure(fig, out_dir / "fig_reserve_per_unit")


def plot_reserve_per_unit_horizontal(
    case_dir: Path,
    out_dir: Path,
    snapshot_hour: int = 16,
    top_n: int = 15,
):
    """Wide-format per-unit reserve chart with generators on x-axis.

    Same data as plot_reserve_per_unit but rotated: generators along the
    horizontal axis, reserve (MW) on the vertical axis.  Better suited for
    two-column paper layouts where width is available but height is limited.
    """
    dam_r, daruc_eq, _R_req, time_labels = load_reserve_data(case_dir)
    fs = FONT_SIZES_TWO_COL

    C_DAM = "#4682B4"
    C_DARUC = "#E07B39"

    # Find snapshot column
    snap_cols = [c for c in time_labels if c.hour == snapshot_hour]
    if not snap_cols:
        hours = np.array([c.hour for c in time_labels])
        closest_idx = np.argmin(np.abs(hours - snapshot_hour))
        snap_cols = [time_labels[closest_idx]]
        print(f"  Hour {snapshot_hour}:00 not found, using {snap_cols[0]}")
    snap_col = snap_cols[0]

    # Thermal generators only (non-zero in DAM reserve)
    dam_thermal = dam_r.loc[(dam_r != 0).any(axis=1)]
    thermal_gens = dam_thermal.index
    daruc_thermal = daruc_eq.loc[daruc_eq.index.isin(thermal_gens)]
    common_thermal = thermal_gens.intersection(daruc_thermal.index)
    dam_thermal = dam_thermal.loc[common_thermal]
    daruc_thermal = daruc_thermal.loc[common_thermal]

    # Snapshot values
    dam_snap = dam_thermal[snap_col].sort_values(ascending=False)
    daruc_snap = daruc_thermal[snap_col].reindex(dam_snap.index)

    # Top generators by max(dam, daruc)
    combined_max = pd.concat([dam_snap.abs(), daruc_snap.abs()], axis=1).max(axis=1)
    top_gens = combined_max.nlargest(top_n).index
    dam_top = dam_snap.loc[top_gens].sort_values(ascending=False)
    daruc_top = daruc_snap.loc[dam_top.index]

    # Shorten labels
    labels = [g[:14] for g in dam_top.index]
    x_pos = np.arange(len(labels))
    bar_w = 0.35

    fig_w = max(IEEE_TWO_COL_WIDTH, len(labels) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 3.0))

    ax.bar(x_pos - bar_w / 2, dam_top.values, width=bar_w,
           color=C_DAM, label="DAM w/Res", edgecolor="white", linewidth=0.3)
    ax.bar(x_pos + bar_w / 2, daruc_top.values, width=bar_w,
           color=C_DARUC, label="DARUC", edgecolor="white", linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=fs["small"] - 1, rotation=45, ha="right")
    ax.set_ylabel("Reserve-Equivalent (MW)", fontsize=fs["medium"])
    # ax.set_title(f"Per-unit reserve at {snapshot_hour:02d}:00", fontsize=fs["large"])
    ax.legend(fontsize=fs["small"], loc="upper right")
    ax.tick_params(labelsize=fs["small"])

    fig.tight_layout()
    _save_figure(fig, out_dir / "fig_reserve_per_unit_h")


def plot_reserve_network_map(
    case_dir: Path,
    out_dir: Path,
    snapshot_hour: int = 16,
):
    """Two-panel network map showing per-bus reserve allocation at a snapshot hour.

    Left panel: DAM w/Reserve explicit reserves r[i,t].
    Right panel: DARUC reserve equivalent from Z coefficients.

    Circle size ∝ total reserve at each bus.  Wind buses shown as triangles
    (negative reserve = curtailment) and thermals as circles (upward reserve).
    """
    from matplotlib.lines import Line2D

    dam_r, daruc_eq, R_req, time_labels = load_reserve_data(case_dir)
    bus, branch = load_network_topology()

    # Find snapshot column
    snap_cols = [c for c in time_labels if c.hour == snapshot_hour]
    if not snap_cols:
        hours = np.array([c.hour for c in time_labels])
        closest_idx = int(np.argmin(np.abs(hours - snapshot_hour)))
        snap_cols = [time_labels[closest_idx]]
        print(f"  Hour {snapshot_hour}:00 not found, using {snap_cols[0]}")
    snap_col = snap_cols[0]

    # Aggregate per-bus reserves at snapshot
    def _aggregate_bus(reserve_series):
        """Group generator reserves by bus, return dict {bus_id: total_mw}."""
        bus_reserve = {}
        for gen_id, val in reserve_series.items():
            bid = _gen_id_to_bus(gen_id)
            bus_reserve[bid] = bus_reserve.get(bid, 0.0) + val
        return bus_reserve

    dam_bus = _aggregate_bus(dam_r[snap_col])
    daruc_bus = _aggregate_bus(daruc_eq[snap_col])

    # Shared scale: max across both panels for consistent sizing
    all_vals = list(dam_bus.values()) + list(daruc_bus.values())
    max_reserve = max(abs(v) for v in all_vals) if all_vals else 1.0

    # Circle size scaling
    S_MIN, S_MAX = 20, 600  # marker area range

    def _reserve_size(mw):
        frac = min(abs(mw) / max_reserve, 1.0) if max_reserve > 0 else 0
        return S_MIN + (S_MAX - S_MIN) * frac

    # Colors
    C_THERMAL_UP = "#4682B4"   # blue — upward thermal reserve
    C_WIND_DOWN = "#E07B39"    # orange — wind curtailment (downward)

    fs = FONT_SIZES_TWO_COL

    def _draw_reserve_panel(ax, bus_df, branch_df, bus_reserves, title):
        """Draw one reserve map panel."""
        _draw_base_network(ax, bus_df, branch_df)

        for bid, mw in bus_reserves.items():
            if bid not in bus_df.index:
                continue
            if abs(mw) < 0.5:
                continue
            bx = bus_df.loc[bid, "lng"]
            by = bus_df.loc[bid, "lat"]
            is_wind = bid in _WIND_BUSES
            color = C_WIND_DOWN if is_wind else C_THERMAL_UP
            marker = "v" if is_wind else "o"
            sz = _reserve_size(mw)
            ax.scatter(bx, by, s=sz, c=color, marker=marker, zorder=7,
                       edgecolors="black", linewidths=0.5, alpha=0.85)

            # Label buses with significant reserve (top ~5 thermal + all wind)
            if abs(mw) > max_reserve * 0.15 or is_wind:
                ax.annotate(
                    f"{mw:.0f}", (bx, by),
                    fontsize=fs["small"] - 2, fontweight="bold",
                    color=color, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec=color, alpha=0.8, lw=0.4),
                    zorder=9,
                )

        _style_map_ax(ax, title)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(IEEE_TWO_COL_WIDTH*1.5, 4.0))

    _draw_reserve_panel(ax1, bus, branch, dam_bus,
                        f"DAM w/Reserve (h{snapshot_hour:02d})")
    _draw_reserve_panel(ax2, bus, branch, daruc_bus,
                        f"DARUC equivalent (h{snapshot_hour:02d})")

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_THERMAL_UP,
               markeredgecolor="black", markersize=10,
               label="Thermal upward reserve"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=C_WIND_DOWN,
               markeredgecolor="black", markersize=10,
               label="Wind curtailment"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=8, label="Wind bus"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999",
               markeredgecolor="none", markersize=5, label="Load bus"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=fs["small"], bbox_to_anchor=(0.5, -0.02))

    # Size reference annotation
    ref_mw = round(max_reserve, -1)  # round to nearest 10
    if ref_mw > 0:
        fig.text(0.98, 0.01, f"(max circle = {ref_mw:.0f} MW)",
                 fontsize=fs["small"] - 1, ha="right", va="bottom",
                 style="italic", color="#666666")

    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    _save_figure(fig, out_dir / f"fig_reserve_map_h{snapshot_hour:02d}")


# ---------------------------------------------------------------------------
# Chart 10: Worst-case total-shortfall line flows (DAM w/Reserve vs DARUC)
# ---------------------------------------------------------------------------

def load_worst_case_flow_data(case_dir: Path, hour: int):
    """Load worst-case flow analysis CSVs for DAM w/Reserve and DARUC.

    Returns (dam_hour, daruc_hour) DataFrames indexed by line UID,
    or (None, None) if files are missing.
    """
    dam_path = case_dir / "dam_reserve" / "worst_case_flow_analysis_dam_reserve.csv"
    daruc_path = case_dir / "daruc" / "worst_case_flow_analysis_daruc.csv"

    results = []
    for path in [dam_path, daruc_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path, parse_dates=["period"])
        hour_ts = _find_hour_ts(df, hour)
        if hour_ts is None:
            raise ValueError(f"Hour {hour} not found in {path.name}")
        results.append(df[df["period"] == hour_ts].set_index("line"))

    return results[0], results[1]


def plot_worst_case_flow_map(
    case_dir: Path,
    out_dir: Path,
    hour: int = 16,
):
    """Two-panel network map: line flows under worst-case total wind shortfall.

    Left: DAM w/Reserve (likely shows violations — no network-aware LDR).
    Right: DARUC (should respect limits via adaptive Z).

    Line thickness ∝ |flow| / Fmax.  Color: blue for normal loading,
    red for violations (|flow| > Fmax).  Violated lines get labels
    showing excess MW.
    """
    from matplotlib.lines import Line2D

    dam_hour, daruc_hour = load_worst_case_flow_data(case_dir, hour)
    bus, branch = load_network_topology()

    fs = FONT_SIZES_TWO_COL

    # Shared Fmax for consistent scaling
    fmax_max = max(dam_hour["Fmax"].max(), daruc_hour["Fmax"].max())

    C_NORMAL = "#4393c3"   # blue
    C_VIOLATE = "#d62728"  # red

    def _draw_wc_panel(ax, bus_df, branch_df, hour_data, title):
        """Draw one worst-case flow panel."""
        _draw_base_network(ax, bus_df, branch_df)

        n_viols = 0
        for uid in hour_data.index:
            coords = _get_branch_coords(uid, branch_df, bus_df)
            if coords is None:
                continue
            x, y = coords
            row = hour_data.loc[uid]
            fmax = row["Fmax"]
            f_wc = row["flow_wc"]
            abs_wc = abs(f_wc)
            is_viol = bool(row["violation"])

            loading = min(abs_wc / fmax, 1.5) if fmax > 0 else 0
            lw = _LW_MIN + (_LW_MAX - _LW_MIN) * min(loading, 1.0)

            color = C_VIOLATE if is_viol else C_NORMAL
            alpha = max(0.3, min(loading, 1.0))

            ax.plot(x, y, color=color, linewidth=lw, zorder=3,
                    solid_capstyle="round", alpha=alpha)

            if is_viol:
                n_viols += 1
                # Red halo
                ax.plot(x, y, color=C_VIOLATE, linewidth=lw + 3,
                        zorder=2, alpha=0.2)
                excess = row["excess_mw"]
                _label_binding(ax, x, y, uid, f"+{excess:.0f} MW")

        _style_map_ax(ax, title)
        return n_viols

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_TWO_COL_WIDTH, 4.0))

    n_dam = _draw_wc_panel(ax1, bus, branch, dam_hour,
                           f"DAM w/Reserve (h{hour:02d})")
    n_daruc = _draw_wc_panel(ax2, bus, branch, daruc_hour,
                             f"DARUC (h{hour:02d})")

    # Subtitle with violation counts
    fig.text(0.25, 0.97, f"{n_dam} violations" if n_dam else "No violations",
             ha="center", fontsize=fs["small"],
             color=C_VIOLATE if n_dam else "#2ca02c", fontweight="bold")
    fig.text(0.75, 0.97, f"{n_daruc} violations" if n_daruc else "No violations",
             ha="center", fontsize=fs["small"],
             color=C_VIOLATE if n_daruc else "#2ca02c", fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color=C_NORMAL, lw=4, label="Normal flow"),
        Line2D([0], [0], color=C_VIOLATE, lw=4, label="Violation (|f| > Fmax)"),
        Line2D([0], [0], color=C_VIOLATE, lw=5, alpha=0.2,
               label="Violation halo"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=8, label="Wind bus"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=fs["small"], bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Line flows under worst-case total wind shortfall",
                 fontsize=fs["large"], y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    _save_figure(fig, out_dir / f"fig_worst_case_flow_map_h{hour:02d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Line flow comparison charts")
    parser.add_argument(
        "--case-dir", type=str, required=True,
        help="Path to reserve_then_daruc output directory",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for figures (default: same as case-dir)",
    )
    parser.add_argument(
        "--map-hour", type=int, default=16,
        help="Hour (0-23) for network map snapshot (default: 7)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=0.001,
        help="Min |Z| to include a generator in Z matrix plots (default: 0.001)",
    )
    parser.add_argument(
        "--reserve-hour", type=int, default=16,
        help="Hour (0-23) for reserve comparison snapshot (default: 16)",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    out_dir = Path(args.out_dir) if args.out_dir else case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading line flow data...")
    dam_raw, daruc_raw = load_line_flow_analysis(case_dir)

    print("Filtering to day-1, binding lines only...")
    dam, daruc, binding_lines = filter_day1_binding(dam_raw, daruc_raw)
    print(f"  {len(binding_lines)} binding lines: {binding_lines}")
    print(f"  {dam['period'].nunique()} periods ")

    print("\nChart 1: Loading heatmaps...")
    plot_loading_heatmaps(dam, daruc, binding_lines, out_dir)

    print("Chart 2: Binding diff...")
    plot_binding_diff(dam, daruc, binding_lines, out_dir)

    print("Chart 3: Flow decomposition...")
    plot_flow_decomposition(dam, daruc, binding_lines, out_dir)

    print(f"\nChart 3b: Flow decomposition snapshot (hour {args.map_hour})...")
    plot_flow_decomposition_hour(dam_raw, daruc_raw, args.map_hour, out_dir)

    print(f"\nChart 3c: Flow decomposition — binding lines (hour {args.map_hour})...")
    plot_flow_decomposition_binding(dam_raw, daruc_raw, binding_lines, args.map_hour, out_dir)

    print(f"\nChart 3d: Flow decomposition — binding-only lines (hour {args.map_hour})...")
    plot_flow_decomposition_binding_only(dam_raw, daruc_raw, args.map_hour, out_dir)

    print(f"\nChart 4: Network maps (hour {args.map_hour}) — 3 options...")
    plot_network_maps(dam_raw, daruc_raw, args.map_hour, out_dir)

    print(f"\nChart 6: Network map + extra commitments (hour {args.map_hour})...")
    try:
        plot_network_commitment_map(
            case_dir, dam_raw, daruc_raw, args.map_hour, out_dir,
        )
    except FileNotFoundError as e:
        print(f"  Skipping commitment map: {e}")

    print(f"\nChart 5: Z matrix (hour {args.map_hour}) — 3 options...")
    try:
        plot_z_matrix(case_dir, args.map_hour, out_dir, threshold=args.z_threshold)
    except FileNotFoundError as e:
        print(f"  Skipping Z matrix: {e}")

    print(f"\nChart 7: Reserve comparison (hour {args.reserve_hour})...")
    try:
        plot_reserve_comparison(case_dir, out_dir, snapshot_hour=args.reserve_hour)
    except FileNotFoundError as e:
        print(f"  Skipping reserve comparison: {e}")

    print(f"\nChart 8: Per-unit reserve allocation (hour {args.reserve_hour})...")
    try:
        plot_reserve_per_unit(case_dir, out_dir, snapshot_hour=args.reserve_hour)
        plot_reserve_per_unit_horizontal(case_dir, out_dir, snapshot_hour=args.reserve_hour)
    except FileNotFoundError as e:
        print(f"  Skipping per-unit reserve: {e}")

    print(f"\nChart 9: Reserve network map (hour {args.reserve_hour})...")
    try:
        plot_reserve_network_map(case_dir, out_dir, snapshot_hour=args.reserve_hour)
    except FileNotFoundError as e:
        print(f"  Skipping reserve network map: {e}")

    print(f"\nChart 10: Worst-case total-shortfall flow map (hour {args.reserve_hour})...")
    try:
        plot_worst_case_flow_map(case_dir, out_dir, hour=args.reserve_hour)
    except (FileNotFoundError, ValueError) as e:
        print(f"  Skipping worst-case flow map: {e}")

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
