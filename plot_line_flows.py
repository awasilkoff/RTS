#!/usr/bin/env python3
"""
Line flow comparison visualizations: DAM+Reserve vs DARUC.

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_line_flow_analysis(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DAM+Reserve and DARUC line_flow_analysis CSVs."""
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
        1, 2, figsize=(max(14, n_hours * 0.45), max(3, n_lines * 0.6 + 1.5)),
        sharey=True,
    )

    vmin, vmax = 0, 100
    cmap = plt.cm.RdYlGn_r

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
        # Mark binding cells with a black dot
        bind_arr = data_bind.values.astype(bool)
        ys, xs = np.where(bind_arr)
        ax.scatter(xs, ys, marker="s", s=18, c="black", alpha=0.6, label="binding")

        ax.set_xticks(range(n_hours))
        ax.set_xticklabels(hours, rotation=90, fontsize=7)
        ax.set_yticks(range(n_lines))
        ax.set_yticklabels(lines, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour")

    ax1.set_ylabel("Line")
    fig.suptitle("Worst-Case Line Loading: DAM+Reserve vs DARUC (Day 1)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.colorbar(im, ax=[ax1, ax2], label="Worst-Case Loading %", shrink=0.8, pad=0.02)

    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_loading_heatmap.{fmt}", dpi=200, bbox_inches="tight")
    print(f"  Saved fig_loading_heatmap.png/pdf")
    plt.close(fig)


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

    colors = ["#f0f0f0", "#4393c3", "#f4a582", "#d6604d"]
    labels = ["Neither", "DAM+Res only", "DARUC only", "Both"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(
        figsize=(max(10, n_hours * 0.4), max(2.5, n_lines * 0.55 + 1)),
    )
    im = ax.imshow(cat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n_hours))
    ax.set_xticklabels(hours, rotation=90, fontsize=7)
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels(lines, fontsize=9)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Line")
    ax.set_title("Binding Status: DAM+Reserve vs DARUC (Day 1)", fontsize=12)

    # Legend outside the plot area
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="gray", label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, loc="upper center", fontsize=8, ncol=4,
              bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_binding_diff.{fmt}", dpi=200, bbox_inches="tight")
    print(f"  Saved fig_binding_diff.png/pdf")
    plt.close(fig)


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
    Overlays DAM+Reserve |nominal| for comparison."""

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

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(lines) * 0.6 + 1)))

    y_pos = range(len(rec_df))
    bar_labels = [f"{r['line']} (h{r['hour']})" for _, r in rec_df.iterrows()]

    # Stacked: nominal + margin
    ax.barh(
        y_pos, rec_df["nominal_abs"], height=0.6,
        color="#4393c3", label="DARUC |nominal flow|", zorder=2,
    )
    ax.barh(
        y_pos, rec_df["margin"], height=0.6,
        left=rec_df["nominal_abs"].values,
        color="#f4a582", label="DARUC robust margin (ρ·‖Z_line‖)", zorder=2,
    )

    # Fmax reference
    ax.barh(
        y_pos, rec_df["Fmax"], height=0.65,
        color="none", edgecolor="black", linewidth=1.2,
        label="Fmax", zorder=3,
    )

    # DAM+Reserve nominal as diamond markers
    ax.scatter(
        rec_df["dam_nominal_abs"], y_pos,
        marker="D", s=50, c="#2166ac", edgecolors="black", linewidths=0.5,
        zorder=4, label="DAM+Res |nominal|",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=9)
    ax.set_xlabel("Flow (MW)")
    ax.set_title("Flow Decomposition at Peak Hour (DARUC, Day 1)", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, rec_df["Fmax"].max() * 1.08)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_flow_decomposition.{fmt}", dpi=200, bbox_inches="tight")
    print(f"  Saved fig_flow_decomposition.png/pdf")
    plt.close(fig)


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
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
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

def _find_hour_ts(df, hour):
    """Find the pd.Timestamp matching the given hour (0-23)."""
    for t in sorted(df["period"].unique()):
        if t.hour == hour:
            return t
    return None


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
    fig_a, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
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
    fig_a.legend(handles=legend_a, loc="lower center", ncol=4, fontsize=9,
                 bbox_to_anchor=(0.5, -0.01))
    fig_a.suptitle(f"Option A: Parallel Lines — Hour {hour:02d}:00", fontsize=13)
    fig_a.tight_layout(rect=[0, 0.04, 1, 0.96])
    for fmt in ("png", "pdf"):
        fig_a.savefig(out_dir / f"fig_map_A_parallel_h{hour:02d}.{fmt}",
                      dpi=200, bbox_inches="tight")
    print(f"  Saved fig_map_A_parallel_h{hour:02d}.png/pdf")
    plt.close(fig_a)

    # --- Option B: Three panels (DAM wc | DARUC nominal | DARUC margin) ---
    fig_b, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(24, 7))
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
    fig_b.legend(handles=legend_b, loc="lower center", ncol=4, fontsize=9,
                 bbox_to_anchor=(0.5, -0.01))
    fig_b.suptitle(f"Option B: Three Panels — Hour {hour:02d}:00", fontsize=13)
    fig_b.tight_layout(rect=[0, 0.04, 1, 0.96])
    for fmt in ("png", "pdf"):
        fig_b.savefig(out_dir / f"fig_map_B_three_panel_h{hour:02d}.{fmt}",
                      dpi=200, bbox_inches="tight")
    print(f"  Saved fig_map_B_three_panel_h{hour:02d}.png/pdf")
    plt.close(fig_b)

    # --- Option C: Dual encoding (2 panels) ---
    fig_c, (cx1, cx2) = plt.subplots(1, 2, figsize=(18, 8))
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
    fig_c.legend(handles=legend_c, loc="lower center", ncol=4, fontsize=9,
                 bbox_to_anchor=(0.5, -0.01))
    fig_c.suptitle(
        f"Option C: Color=Nominal%, Thickness=Worst-Case% — Hour {hour:02d}:00",
        fontsize=13)
    fig_c.tight_layout(rect=[0, 0.04, 1, 0.96])
    for fmt in ("png", "pdf"):
        fig_c.savefig(out_dir / f"fig_map_C_dual_encode_h{hour:02d}.{fmt}",
                      dpi=200, bbox_inches="tight")
    print(f"  Saved fig_map_C_dual_encode_h{hour:02d}.png/pdf")
    plt.close(fig_c)


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
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    out_dir = Path(args.out_dir) if args.out_dir else case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading line flow data...")
    dam_raw, daruc_raw = load_line_flow_analysis(case_dir)

    print("Filtering to day-1, binding lines only...")
    dam, daruc, binding_lines = filter_day1_binding(dam_raw, daruc_raw)
    print(f"  {len(binding_lines)} binding lines: {binding_lines}")
    print(f"  {dam['period'].nunique()} periods (day 1)")

    print("\nChart 1: Loading heatmaps...")
    plot_loading_heatmaps(dam, daruc, binding_lines, out_dir)

    print("Chart 2: Binding diff...")
    plot_binding_diff(dam, daruc, binding_lines, out_dir)

    print("Chart 3: Flow decomposition...")
    plot_flow_decomposition(dam, daruc, binding_lines, out_dir)

    print(f"\nChart 4: Network maps (hour {args.map_hour}) — 3 options...")
    plot_network_maps(dam_raw, daruc_raw, args.map_hour, out_dir)

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
