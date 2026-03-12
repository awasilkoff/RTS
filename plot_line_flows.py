#!/usr/bin/env python3
"""
Line flow comparison visualizations: DAM+Reserve vs DARUC.

Generates three charts:
  1. Side-by-side worst-case loading heatmaps (lines x hours)
  2. Binding status diff heatmap (both / only-DAM / only-DARUC / neither)
  3. Stacked bar decomposition: nominal flow + robust margin at peak hour

Usage:
    python plot_line_flows.py --case-dir sensitivity_suite/rho99_48h_m07d15/reserve_then_daruc
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
    fig.colorbar(im, ax=[ax1, ax2], label="Worst-Case Loading %", shrink=0.8)
    fig.suptitle("Worst-Case Line Loading: DAM+Reserve vs DARUC (Day 1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

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

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="gray", label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)

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

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
