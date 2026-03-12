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


def _draw_network_panel(
    ax,
    bus: pd.DataFrame,
    branch: pd.DataFrame,
    flow_df: pd.DataFrame,
    hour_ts: pd.Timestamp,
    title: str,
    show_decomposition: bool = False,
):
    """Draw one network panel for a specific hour.

    flow_df: the full (unfiltered) line_flow_analysis for this case at this hour.
    show_decomposition: if True, use concentric halo (orange=worst-case, blue=nominal).
    """
    hour_data = flow_df[flow_df["period"] == hour_ts].set_index("line")

    # Draw all branches as light gray background
    for uid, row in branch.iterrows():
        fb, tb = row["From Bus"], row["To Bus"]
        if fb not in bus.index or tb not in bus.index:
            continue
        x = [bus.loc[fb, "lng"], bus.loc[tb, "lng"]]
        y = [bus.loc[fb, "lat"], bus.loc[tb, "lat"]]
        ax.plot(x, y, color="#d9d9d9", linewidth=0.8, zorder=1)

    # Max linewidth for scaling
    lw_max = 5.0
    lw_min = 1.0

    for uid in hour_data.index:
        if uid not in branch.index:
            continue
        row = hour_data.loc[uid]
        br = branch.loc[uid]
        fb, tb = br["From Bus"], br["To Bus"]
        if fb not in bus.index or tb not in bus.index:
            continue

        x = np.array([bus.loc[fb, "lng"], bus.loc[tb, "lng"]])
        y = np.array([bus.loc[fb, "lat"], bus.loc[tb, "lat"]])
        fmax = row["Fmax"]
        nom_abs = abs(row["flow_nominal"])
        margin = row["margin_rho_norm"]
        wc_abs = row["worst_case_abs_flow"]
        is_binding = bool(row["binding"])

        # Linewidth proportional to worst-case flow / Fmax
        wc_frac = min(wc_abs / fmax, 1.0) if fmax > 0 else 0
        nom_frac = min(nom_abs / fmax, 1.0) if fmax > 0 else 0
        lw_wc = lw_min + (lw_max - lw_min) * wc_frac
        lw_nom = lw_min + (lw_max - lw_min) * nom_frac

        if show_decomposition and margin > 0.1:
            # Concentric halo: outer orange (worst-case), inner blue (nominal)
            ax.plot(x, y, color="#f4a582", linewidth=lw_wc, zorder=3,
                    solid_capstyle="round")
            ax.plot(x, y, color="#4393c3", linewidth=lw_nom, zorder=4,
                    solid_capstyle="round")
        else:
            # Single color by loading
            cmap = plt.cm.RdYlGn_r
            norm = mcolors.Normalize(vmin=0, vmax=100)
            color = cmap(norm(min(row["loading_worst_case_pct"], 100)))
            ax.plot(x, y, color=color, linewidth=lw_wc, zorder=3)

        # Binding highlight: red outline behind everything
        if is_binding:
            ax.plot(x, y, color="red", linewidth=lw_wc + 2.5, zorder=2, alpha=0.35)

        # Label binding lines
        if is_binding:
            mx, my = 0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1])
            label_txt = uid
            if margin > 0.1:
                label_txt += f"\n{nom_abs:.0f}+{margin:.0f}"
            ax.annotate(
                label_txt, (mx, my), fontsize=5.5, fontweight="bold",
                color="red", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                zorder=6,
            )

    # Draw non-wind buses
    non_wind = bus[~bus.index.isin(_WIND_BUSES)]
    ax.scatter(
        non_wind["lng"], non_wind["lat"], s=12, c="#333333", zorder=5,
        edgecolors="white", linewidths=0.3,
    )

    # Draw wind buses with distinct marker
    wind = bus[bus.index.isin(_WIND_BUSES)]
    ax.scatter(
        wind["lng"], wind["lat"], s=100, c="#2ca02c", marker="^", zorder=6,
        edgecolors="black", linewidths=0.8, label="Wind",
    )
    for bus_id, brow in wind.iterrows():
        ax.annotate(
            f"Bus {bus_id}", (brow["lng"], brow["lat"]),
            fontsize=5.5, fontweight="bold", color="#2ca02c",
            ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#2ca02c", alpha=0.8),
            zorder=7,
        )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")


def plot_network_map(
    dam_raw: pd.DataFrame,
    daruc_raw: pd.DataFrame,
    hour: int,
    out_dir: Path,
):
    """Two-panel network map at the given hour (0-23).

    Left: DAM+Reserve (single-color edges by loading).
    Right: DARUC (two-tone edges: nominal blue + margin orange for lines with margin).
    """
    bus, branch = load_network_topology()

    # Find the timestamp for this hour
    all_periods = sorted(dam_raw["period"].unique())
    hour_ts = None
    for t in all_periods:
        if t.hour == hour:
            hour_ts = t
            break
    if hour_ts is None:
        print(f"  WARNING: hour {hour} not found in data, skipping network map")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    _draw_network_panel(
        ax1, bus, branch, dam_raw, hour_ts,
        f"DAM + Reserve (hour {hour:02d}:00)",
        show_decomposition=False,
    )
    _draw_network_panel(
        ax2, bus, branch, daruc_raw, hour_ts,
        f"DARUC (hour {hour:02d}:00)",
        show_decomposition=True,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#4393c3", lw=4, label="Nominal flow"),
        Line2D([0], [0], color="#f4a582", lw=6, label="Robust margin (halo)"),
        Line2D([0], [0], color="red", lw=5, alpha=0.35, label="Binding"),
        Line2D([0], [0], color="#d9d9d9", lw=1, label="Unmonitored"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333", markersize=5, label="Bus"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=8, label="Wind gen"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5, fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Network Line Flows: DAM+Reserve vs DARUC — Hour {hour:02d}:00",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_network_map_h{hour:02d}.{fmt}", dpi=200, bbox_inches="tight")
    print(f"  Saved fig_network_map_h{hour:02d}.png/pdf")
    plt.close(fig)


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

    print(f"\nChart 4: Network map (hour {args.map_hour})...")
    plot_network_map(dam_raw, daruc_raw, args.map_hour, out_dir)

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
