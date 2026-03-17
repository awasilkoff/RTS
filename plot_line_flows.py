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
    labels = ["Neither", "DAM+Res only", "DARUC only", "Both"]
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
        zorder=4, label="DAM+Res |nominal|",
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
               zorder=4, label="DAM+Res |nominal|")

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
               zorder=4, label="DAM+Res worst-case")

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
                               f"DAM+Res worst-case (h{hour:02d})",
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
        "--map-hour", type=int, default=7,
        help="Hour (0-23) for network map snapshot (default: 7)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=0.001,
        help="Min |Z| to include a generator in Z matrix plots (default: 0.001)",
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

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
