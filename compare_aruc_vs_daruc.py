"""
Compare ARUC-LDR (one-shot) vs DARUC (two-step) results.

Loads pre-computed outputs from both directories and generates:
  - Figure 1: Commitment difference heatmap + dispatch by gen type + 3-way cost comparison
  - Figure 2: Z coefficient norm heatmaps (side by side)
  - Text summary to console and comparison_summary.txt

Usage:
    python compare_aruc_vs_daruc.py
    python compare_aruc_vs_daruc.py --aruc-dir aruc_outputs/quick_test_copperplate \
                                     --daruc-dir daruc_outputs/quick_test_copperplate \
                                     --out-dir comparison_outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Gen-type color palette
# ---------------------------------------------------------------------------
GENTYPE_COLORS = {
    "THERMAL": "#808080",
    "WIND": "#1f77b4",
    "SOLAR": "#f0c929",
    "HYDRO": "#17becf",
}


def _gentype_color(gt: str) -> str:
    return GENTYPE_COLORS.get(gt.upper(), "#999999")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(result_dir: Path, label: str) -> dict:
    """Load commitment, dispatch, Z analysis, and optional summary from a results dir."""
    out = {"label": label, "dir": result_dir}

    # Commitment u  (index_col=0 → gen_ids as index)
    u_path = result_dir / "commitment_u.csv"
    if not u_path.exists():
        raise FileNotFoundError(f"Missing {u_path}")
    out["u"] = pd.read_csv(u_path, index_col=0)

    # Dispatch p0
    p0_path = result_dir / "dispatch_p0.csv"
    if not p0_path.exists():
        raise FileNotFoundError(f"Missing {p0_path}")
    out["p0"] = pd.read_csv(p0_path, index_col=0)

    # Z analysis full
    za_path = result_dir / "Z_analysis_full.csv"
    if za_path.exists():
        out["z_full"] = pd.read_csv(za_path)
    else:
        out["z_full"] = None

    # Summary JSON
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            out["summary"] = json.load(f)
    else:
        out["summary"] = None

    # Deviation summary (DARUC only)
    dev_path = result_dir / "deviation_summary.csv"
    if dev_path.exists():
        out["deviation"] = pd.read_csv(dev_path)
    else:
        out["deviation"] = None

    return out


def load_dam_results(daruc_dir: Path) -> dict | None:
    """Load DAM commitment/dispatch saved alongside DARUC outputs."""
    u_path = daruc_dir / "dam_commitment_u.csv"
    p0_path = daruc_dir / "dam_dispatch_p0.csv"
    if not u_path.exists() or not p0_path.exists():
        return None
    return {
        "label": "DAM",
        "u": pd.read_csv(u_path, index_col=0),
        "p0": pd.read_csv(p0_path, index_col=0),
    }


def _round_commitment(u_df: pd.DataFrame) -> pd.DataFrame:
    """Round commitment values to clean 0/1 (Gurobi solver noise)."""
    return u_df.round().clip(0, 1).astype(int)


def _align_time(aruc: dict, daruc: dict, dam: dict | None) -> list[str]:
    """Find common time columns, warn on mismatch, return common list."""
    aruc_cols = list(aruc["u"].columns)
    daruc_cols = list(daruc["u"].columns)
    common = [c for c in aruc_cols if c in daruc_cols]
    if len(common) < len(aruc_cols) or len(common) < len(daruc_cols):
        print(f"WARNING: Time horizon mismatch — ARUC has {len(aruc_cols)} periods, "
              f"DARUC has {len(daruc_cols)} periods. Using common {len(common)} periods.")
    if dam is not None:
        dam_cols = list(dam["u"].columns)
        common = [c for c in common if c in dam_cols]
    return common


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------


def compute_cost_breakdown(u_df: pd.DataFrame, p0_df: pd.DataFrame, data) -> dict:
    """
    Compute cost breakdown matching the Gurobi objective structure.

    The model objective is:
      sum_{i,t} [ C_NL*u + C_SU*v + C_SD*w + sum_b block_cost*p_block ] + M_p*s_p

    Block structure: p[i,t] = sum_b p_block[i,t,b]  (blocks start from 0, not Pmin).

    Returns dict with keys: no_load, startup, shutdown, energy, slack, total.
    """
    u = np.round(u_df.values).astype(float)
    p0 = p0_df.values
    I, T = u.shape

    # Startup detection: v[i,t] = max(0, u[i,t] - u[i,t-1])
    v = np.zeros_like(u)
    for i in range(I):
        v[i, 0] = max(0.0, u[i, 0] - data.u_init[i])
    if T > 1:
        v[:, 1:] = np.maximum(0.0, u[:, 1:] - u[:, :-1])

    # Shutdown detection: w[i,t] = max(0, u[i,t-1] - u[i,t])
    w = np.zeros_like(u)
    for i in range(I):
        w[i, 0] = max(0.0, data.u_init[i] - u[i, 0])
    if T > 1:
        w[:, 1:] = np.maximum(0.0, u[:, :-1] - u[:, 1:])

    no_load = float((data.no_load_cost[:, None] * u).sum())
    startup = float((data.startup_cost[:, None] * v).sum())
    shutdown = float((data.shutdown_cost[:, None] * w).sum())

    # Energy cost: full dispatch through cost blocks (blocks start at 0)
    energy = 0.0
    block_cap = data.block_cap   # (I, B)
    block_cost = data.block_cost  # (I, B)
    B = block_cap.shape[1]
    for i in range(I):
        for t in range(T):
            remaining = max(0.0, p0[i, t])
            for b in range(B):
                allocated = min(remaining, block_cap[i, b])
                energy += allocated * block_cost[i, b]
                remaining -= allocated
                if remaining <= 1e-9:
                    break

    # Slack penalty: total generation shortfall × M_p
    total_gen = p0.sum(axis=0)  # (T,)
    total_load = data.d.sum(axis=0)[:T]  # (T,)
    slack = np.maximum(0.0, total_load - total_gen)
    m_penalty = 1e4  # must match M_PENALTY in runner scripts
    slack_cost = float(m_penalty * slack.sum())

    total = no_load + startup + shutdown + energy + slack_cost
    return {
        "no_load": no_load, "startup": startup, "shutdown": shutdown,
        "energy": energy, "slack": slack_cost, "total": total,
    }


# ---------------------------------------------------------------------------
# Figure 1: Commitment + Cost
# ---------------------------------------------------------------------------


def fig_commitment_and_cost(
    aruc: dict, daruc: dict, dam: dict | None, common_times: list[str],
    cost_aruc: dict | None, cost_daruc: dict | None, cost_dam: dict | None,
    out_dir: Path,
):
    """Three-panel figure: commitment diff heatmap, dispatch stacked area, cost bars."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 10), gridspec_kw={"width_ratios": [3, 4, 3]})

    u_aruc = _round_commitment(aruc["u"][common_times])
    u_daruc = _round_commitment(daruc["u"][common_times])

    # --- (a) Commitment difference heatmap ---
    ax = axes[0]
    diff = u_daruc.values - u_aruc.values  # +1 = DARUC commits more, -1 = ARUC commits more
    # Filter to generators that differ
    diff_mask = (diff != 0).any(axis=1)
    if diff_mask.sum() == 0:
        ax.text(0.5, 0.5, "Identical\ncommitments", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("(a) Commitment difference\n(DARUC − ARUC)", fontsize=10)
    else:
        diff_filtered = diff[diff_mask]
        gen_labels = u_aruc.index[diff_mask]
        vmax = max(abs(diff_filtered.min()), abs(diff_filtered.max()), 1)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(diff_filtered, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_yticks(range(len(gen_labels)))
        ax.set_yticklabels(gen_labels, fontsize=6)
        # Sparse x ticks
        n_t = diff_filtered.shape[1]
        tick_step = max(1, n_t // 6)
        xticks = list(range(0, n_t, tick_step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([common_times[i].split(" ")[1][:5] if " " in common_times[i]
                            else str(i) for i in xticks], fontsize=7, rotation=45)
        ax.set_xlabel("Hour", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.5, label="DARUC − ARUC")
        ax.set_title("(a) Commitment difference\n(blue=DARUC more, red=ARUC more)", fontsize=9)

    # --- (b) Dispatch by gen type ---
    ax = axes[1]
    # Build gen_type lookup from Z_analysis_full
    gen_type_map = {}
    for res in [aruc, daruc]:
        if res["z_full"] is not None:
            for _, row in res["z_full"][["gen_id", "gen_type"]].drop_duplicates().iterrows():
                gen_type_map[row["gen_id"]] = row["gen_type"]

    # Compute stacked dispatch by gen type
    T_common = len(common_times)
    gen_types_ordered = ["THERMAL", "WIND", "SOLAR", "HYDRO"]
    # Filter to types that actually exist
    all_types_present = sorted(set(gen_type_map.values()))
    gen_types_ordered = [gt for gt in gen_types_ordered if gt in all_types_present]
    # Add any not in the predefined list
    for gt in all_types_present:
        if gt not in gen_types_ordered:
            gen_types_ordered.append(gt)

    def _dispatch_by_type(p0_df, times):
        result = {}
        for gt in gen_types_ordered:
            gens = [g for g in p0_df.index if gen_type_map.get(g, "THERMAL") == gt]
            if gens:
                result[gt] = p0_df.loc[gens, times].sum(axis=0).values
            else:
                result[gt] = np.zeros(len(times))
        return result

    aruc_by_type = _dispatch_by_type(aruc["p0"], common_times)
    daruc_by_type = _dispatch_by_type(daruc["p0"], common_times)

    x = np.arange(T_common)
    width = 0.35

    # Stacked bars: ARUC on left, DARUC on right
    for side, by_type, offset, alpha_val in [
        ("ARUC", aruc_by_type, -width / 2, 0.85),
        ("DARUC", daruc_by_type, width / 2, 0.85),
    ]:
        bottom = np.zeros(T_common)
        for gt in gen_types_ordered:
            vals = by_type[gt]
            ax.bar(x + offset, vals, width, bottom=bottom, color=_gentype_color(gt),
                   alpha=alpha_val, label=f"{side} {gt}" if side == "ARUC" else None)
            bottom += vals

    # Load line (total demand) — use sum of dispatch as proxy if no data object
    total_aruc = sum(aruc_by_type[gt] for gt in gen_types_ordered)
    total_daruc = sum(daruc_by_type[gt] for gt in gen_types_ordered)
    ax.plot(x - width / 2, total_aruc, "k--", linewidth=0.8, alpha=0.5)
    ax.plot(x + width / 2, total_daruc, "k--", linewidth=0.8, alpha=0.5)

    # Legend for gen types only
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=_gentype_color(gt), label=gt) for gt in gen_types_ordered]
    legend_patches += [Patch(facecolor="white", edgecolor="black", label="ARUC (left)"),
                       Patch(facecolor="white", edgecolor="black", hatch="///", label="DARUC (right)")]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper left", ncol=2)

    tick_step = max(1, T_common // 8)
    ax.set_xticks(range(0, T_common, tick_step))
    ax.set_xticklabels([common_times[i].split(" ")[1][:5] if " " in common_times[i]
                        else str(i) for i in range(0, T_common, tick_step)], fontsize=7, rotation=45)
    ax.set_xlabel("Hour", fontsize=9)
    ax.set_ylabel("Dispatch (MW)", fontsize=9)
    ax.set_title("(b) Dispatch by generator type", fontsize=10)

    # --- (c) Cost comparison (3-way if DAM available) ---
    ax = axes[2]
    categories = ["No-Load", "Startup", "Shutdown", "Energy", "Slack"]
    cost_keys = ["no_load", "startup", "shutdown", "energy", "slack"]

    # Build bar data
    bar_data = {}
    bar_colors = {}
    if cost_dam is not None:
        bar_data["DAM"] = [cost_dam[k] for k in cost_keys]
        bar_colors["DAM"] = "#2ca02c"
    if cost_daruc is not None:
        bar_data["DARUC"] = [cost_daruc[k] for k in cost_keys]
        bar_colors["DARUC"] = "#ff7f0e"
    if cost_aruc is not None:
        bar_data["ARUC"] = [cost_aruc[k] for k in cost_keys]
        bar_colors["ARUC"] = "#1f77b4"

    if bar_data:
        n_groups = len(categories)
        n_bars = len(bar_data)
        bar_w = 0.8 / n_bars
        x_cat = np.arange(n_groups)
        for idx, (name, vals) in enumerate(bar_data.items()):
            offset = (idx - (n_bars - 1) / 2) * bar_w
            ax.bar(x_cat + offset, vals, bar_w, label=name, color=bar_colors[name], alpha=0.85)
        ax.set_xticks(x_cat)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Cost ($)", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title("(c) Cost breakdown", fontsize=10)
    else:
        # Fallback: just show total objectives from summary
        labels, values = [], []
        if dam is not None and daruc.get("summary") and "dam_objective" in daruc["summary"]:
            labels.append("DAM")
            values.append(daruc["summary"]["dam_objective"])
        if daruc.get("summary"):
            obj_key = "daruc_objective" if "daruc_objective" in daruc["summary"] else "objective"
            labels.append("DARUC")
            values.append(daruc["summary"][obj_key])
        if aruc.get("summary"):
            labels.append("ARUC")
            values.append(aruc["summary"]["objective"])
        if values:
            colors = ["#2ca02c", "#ff7f0e", "#1f77b4"][:len(values)]
            ax.bar(labels, values, color=colors, alpha=0.85)
            ax.set_ylabel("Objective ($)", fontsize=9)
            ax.set_title("(c) Total objective", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No cost data\navailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title("(c) Cost comparison", fontsize=10)

    fig.suptitle("ARUC vs DARUC: Commitment, Dispatch & Cost", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_commitment_cost.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_commitment_cost.pdf/.png")


# ---------------------------------------------------------------------------
# Figure 2: Z coefficient heatmaps
# ---------------------------------------------------------------------------


def fig_z_heatmaps(aruc: dict, daruc: dict, common_times: list[str], out_dir: Path):
    """Side-by-side Z-norm heatmaps for ARUC and DARUC."""
    if aruc["z_full"] is None or daruc["z_full"] is None:
        print("  Skipping Z heatmap — Z_analysis_full.csv not found in one or both dirs.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, (res, title) in enumerate([(aruc, "ARUC-LDR"), (daruc, "DARUC")]):
        ax = axes[idx]
        zf = res["z_full"].copy()

        # Filter to common times
        zf = zf[zf["time"].isin(common_times)]

        # Filter to generators with non-negligible Z
        gen_max_norm = zf.groupby("gen_id")["Z_row_norm"].max()
        active_gens = gen_max_norm[gen_max_norm > 1e-6].index.tolist()
        if not active_gens:
            ax.text(0.5, 0.5, "No active Z", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"({chr(97+idx)}) {title}", fontsize=10)
            continue

        zf_active = zf[zf["gen_id"].isin(active_gens)]

        # Pivot to (gen_id × time)
        pivot = zf_active.pivot(index="gen_id", columns="time", values="Z_row_norm")
        pivot = pivot.reindex(columns=common_times)
        pivot = pivot.fillna(0)

        # Sort: wind generators first, then by mean norm descending
        gen_type_map = dict(zip(zf_active["gen_id"], zf_active["gen_type"]))
        is_wind = [gen_type_map.get(g, "") == "WIND" for g in pivot.index]
        mean_norm = pivot.mean(axis=1)
        sort_key = pd.DataFrame({"wind": is_wind, "mean_norm": mean_norm}, index=pivot.index)
        sort_key = sort_key.sort_values(["wind", "mean_norm"], ascending=[False, False])
        pivot = pivot.loc[sort_key.index]

        # Store for shared colorbar
        if idx == 0:
            vmax_global = pivot.values.max()
        else:
            vmax_global = max(vmax_global, pivot.values.max())

    # Second pass: actually plot with shared vmax
    vmax_global = 0
    for res in [aruc, daruc]:
        zf = res["z_full"]
        if zf is not None:
            zf_t = zf[zf["time"].isin(common_times)]
            gen_max = zf_t.groupby("gen_id")["Z_row_norm"].max()
            active = gen_max[gen_max > 1e-6].index
            if len(active) > 0:
                vals = zf_t[zf_t["gen_id"].isin(active)]["Z_row_norm"].max()
                vmax_global = max(vmax_global, vals)

    if vmax_global < 1e-6:
        vmax_global = 1.0

    for idx, (res, title) in enumerate([(aruc, "ARUC-LDR"), (daruc, "DARUC")]):
        ax = axes[idx]
        ax.clear()
        zf = res["z_full"].copy()
        zf = zf[zf["time"].isin(common_times)]

        gen_max_norm = zf.groupby("gen_id")["Z_row_norm"].max()
        active_gens = gen_max_norm[gen_max_norm > 1e-6].index.tolist()
        if not active_gens:
            ax.text(0.5, 0.5, "No active Z", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"({chr(97+idx)}) {title}", fontsize=10)
            continue

        zf_active = zf[zf["gen_id"].isin(active_gens)]
        pivot = zf_active.pivot(index="gen_id", columns="time", values="Z_row_norm")
        pivot = pivot.reindex(columns=common_times).fillna(0)

        gen_type_map = dict(zip(zf_active["gen_id"], zf_active["gen_type"]))
        is_wind = [1 if gen_type_map.get(g, "") == "WIND" else 0 for g in pivot.index]
        mean_norm = pivot.mean(axis=1).values
        sort_idx = np.lexsort(([-m for m in mean_norm], [-w for w in is_wind]))
        pivot = pivot.iloc[sort_idx]

        # Annotate wind generators with color
        gen_labels = []
        for g in pivot.index:
            gt = gen_type_map.get(g, "THERMAL")
            gen_labels.append(f"{g} [{gt[0]}]" if gt == "WIND" else g)

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax_global,
                       interpolation="nearest")
        ax.set_yticks(range(len(gen_labels)))
        ax.set_yticklabels(gen_labels, fontsize=6)

        n_t = pivot.shape[1]
        tick_step = max(1, n_t // 6)
        xticks = list(range(0, n_t, tick_step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([common_times[i].split(" ")[1][:5] if " " in common_times[i]
                            else str(i) for i in xticks], fontsize=7, rotation=45)
        ax.set_xlabel("Hour", fontsize=9)
        ax.set_title(f"({chr(97+idx)}) {title}: ||Z_row||", fontsize=10)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="||Z row||₂")

    fig.suptitle("Z Coefficient Comparison (uncertainty response)", fontsize=12, y=0.98)

    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_z_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_z_comparison.pdf/.png")


# ---------------------------------------------------------------------------
# Wind curtailment helper
# ---------------------------------------------------------------------------


def compute_wind_curtailment(p0_df: pd.DataFrame, data, common_times: list[str]) -> dict:
    """
    Compute wind curtailment = available wind capacity − dispatched wind.

    Returns dict with:
      total_mwh: total curtailment in MWh over the horizon
      pct: curtailment as % of available wind
      per_farm: dict of {gen_id: avg curtailment MW}
      timeseries: (T,) array of total curtailment per period
      per_farm_timeseries: dict of {gen_id: (T,) array}
    """
    wind_mask = [gt.upper() == "WIND" for gt in data.gen_type]
    wind_idx = [i for i, m in enumerate(wind_mask) if m]
    wind_ids = [data.gen_ids[i] for i in wind_idx]

    pmax_2d = data.Pmax_2d()  # (I, T)
    T = len(common_times)

    # Map common_times to column positions in the full time axis
    # Handle string vs Timestamp mismatch between CSV columns and data.time
    time_list = [str(t) for t in data.time]
    common_str = [str(t) for t in common_times]
    time_pos = [time_list.index(t) for t in common_str]

    total_curtail_ts = np.zeros(T)
    per_farm_ts = {}
    per_farm_avg = {}

    for wi, gid in zip(wind_idx, wind_ids):
        avail = np.array([pmax_2d[wi, tp] for tp in time_pos])
        dispatched = p0_df.loc[gid, common_times].values.astype(float)
        curtail = np.maximum(0.0, avail - dispatched)
        total_curtail_ts += curtail
        per_farm_ts[gid] = curtail
        per_farm_avg[gid] = float(curtail.mean())

    total_avail = sum(
        sum(pmax_2d[wi, tp] for tp in time_pos) for wi in wind_idx
    )
    total_mwh = float(total_curtail_ts.sum())
    pct = 100.0 * total_mwh / total_avail if total_avail > 0 else 0.0

    return {
        "total_mwh": total_mwh,
        "pct": pct,
        "per_farm": per_farm_avg,
        "timeseries": total_curtail_ts,
        "per_farm_timeseries": per_farm_ts,
    }


# ---------------------------------------------------------------------------
# Figure: Wind curtailment
# ---------------------------------------------------------------------------


def fig_wind_curtailment(
    aruc: dict, daruc: dict, dam: dict | None,
    common_times: list[str], data, out_dir: Path,
):
    """Two-panel figure: (a) curtailment time series, (b) per-farm bar chart."""
    formulations = []
    colors = {}
    if dam is not None:
        formulations.append(("DAM", dam))
        colors["DAM"] = "#2ca02c"
    formulations.append(("DARUC", daruc))
    colors["DARUC"] = "#ff7f0e"
    formulations.append(("ARUC", aruc))
    colors["ARUC"] = "#1f77b4"

    curtail = {}
    for name, res in formulations:
        curtail[name] = compute_wind_curtailment(res["p0"], data, common_times)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- (a) Time series ---
    ax = axes[0]
    x = np.arange(len(common_times))
    for name, _ in formulations:
        ax.plot(x, curtail[name]["timeseries"], label=name,
                color=colors[name], linewidth=1.5)
    ax.set_xlabel("Hour", fontsize=9)
    ax.set_ylabel("Wind Curtailment (MW)", fontsize=9)
    ax.set_title("(a) Total wind curtailment over time", fontsize=10)
    ax.legend(fontsize=8)
    tick_step = max(1, len(common_times) // 8)
    ax.set_xticks(range(0, len(common_times), tick_step))
    ax.set_xticklabels(
        [common_times[i].split(" ")[1][:5] if " " in common_times[i]
         else str(i) for i in range(0, len(common_times), tick_step)],
        fontsize=7, rotation=45,
    )

    # --- (b) Per-farm bar chart ---
    ax = axes[1]
    wind_ids = list(curtail[formulations[0][0]]["per_farm"].keys())
    n_farms = len(wind_ids)
    n_forms = len(formulations)
    bar_w = 0.8 / n_forms
    x_farms = np.arange(n_farms)

    for idx, (name, _) in enumerate(formulations):
        offset = (idx - (n_forms - 1) / 2) * bar_w
        vals = [curtail[name]["per_farm"][gid] for gid in wind_ids]
        ax.bar(x_farms + offset, vals, bar_w, label=name,
               color=colors[name], alpha=0.85)

    ax.set_xticks(x_farms)
    ax.set_xticklabels([gid.replace("_WIND_1", "") for gid in wind_ids],
                       fontsize=8, rotation=30)
    ax.set_xlabel("Wind Farm", fontsize=9)
    ax.set_ylabel("Avg Curtailment (MW)", fontsize=9)
    ax.set_title("(b) Average curtailment per wind farm", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Wind Curtailment Comparison", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_wind_curtailment.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_wind_curtailment.pdf/.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def write_summary(
    aruc: dict, daruc: dict, dam: dict | None, common_times: list[str],
    cost_aruc: dict | None, cost_daruc: dict | None, cost_dam: dict | None,
    out_dir: Path,
    data=None,
):
    """Write comparison summary to console and file."""
    lines = []
    lines.append("=" * 70)
    lines.append("ARUC vs DARUC COMPARISON SUMMARY")
    lines.append("=" * 70)

    # Objective values
    aruc_obj = aruc["summary"]["objective"] if aruc.get("summary") else None
    daruc_obj = (daruc["summary"].get("daruc_objective") or daruc["summary"].get("objective")) \
        if daruc.get("summary") else None
    dam_obj = daruc["summary"].get("dam_objective") if daruc.get("summary") else None

    lines.append("\n--- Objective Values ---")
    if dam_obj is not None:
        lines.append(f"  DAM (deterministic):  {dam_obj:>14,.2f}")
    if daruc_obj is not None:
        lines.append(f"  DARUC (two-step):     {daruc_obj:>14,.2f}")
    if aruc_obj is not None:
        lines.append(f"  ARUC-LDR (one-shot):  {aruc_obj:>14,.2f}")
    if daruc_obj is not None and aruc_obj is not None:
        diff = aruc_obj - daruc_obj
        pct = 100 * diff / daruc_obj if daruc_obj else 0
        lines.append(f"  ARUC − DARUC:         {diff:>14,.2f}  ({pct:+.2f}%)")
    if dam_obj is not None and daruc_obj is not None:
        diff = daruc_obj - dam_obj
        pct = 100 * diff / dam_obj if dam_obj else 0
        lines.append(f"  DARUC − DAM:          {diff:>14,.2f}  ({pct:+.2f}%)")
    if dam_obj is not None and aruc_obj is not None:
        diff = aruc_obj - dam_obj
        pct = 100 * diff / dam_obj if dam_obj else 0
        lines.append(f"  ARUC − DAM:           {diff:>14,.2f}  ({pct:+.2f}%)")

    # Cost breakdown
    if cost_aruc or cost_daruc or cost_dam:
        lines.append("\n--- Cost Breakdown ---")
        header = (f"  {'':20s}  {'No-Load':>12s}  {'Startup':>12s}  {'Shutdown':>12s}"
                  f"  {'Energy':>12s}  {'Slack':>12s}  {'Total':>12s}")
        lines.append(header)
        for name, cost in [("DAM", cost_dam), ("DARUC", cost_daruc), ("ARUC-LDR", cost_aruc)]:
            if cost:
                lines.append(
                    f"  {name:20s}  {cost['no_load']:>12,.2f}  {cost['startup']:>12,.2f}  "
                    f"{cost['shutdown']:>12,.2f}  {cost['energy']:>12,.2f}  "
                    f"{cost['slack']:>12,.2f}  {cost['total']:>12,.2f}"
                )

    # Commitment counts
    lines.append("\n--- Commitment Summary ---")
    u_aruc = _round_commitment(aruc["u"][common_times])
    u_daruc = _round_commitment(daruc["u"][common_times])
    lines.append(f"  ARUC  total unit-hours: {u_aruc.values.sum()}")
    lines.append(f"  DARUC total unit-hours: {u_daruc.values.sum()}")
    if dam is not None:
        u_dam = _round_commitment(dam["u"][common_times])
        lines.append(f"  DAM   total unit-hours: {u_dam.values.sum()}")

    # Generators unique to each
    aruc_ever = set(u_aruc.index[u_aruc.sum(axis=1) > 0])
    daruc_ever = set(u_daruc.index[u_daruc.sum(axis=1) > 0])
    only_aruc = aruc_ever - daruc_ever
    only_daruc = daruc_ever - aruc_ever
    lines.append(f"  Generators committed only in ARUC:  {len(only_aruc)}")
    if only_aruc:
        lines.append(f"    {', '.join(sorted(only_aruc)[:15])}")
    lines.append(f"  Generators committed only in DARUC: {len(only_daruc)}")
    if only_daruc:
        lines.append(f"    {', '.join(sorted(only_daruc)[:15])}")

    # Dispatch totals by gen type
    lines.append("\n--- Dispatch Totals (MWh over common horizon) ---")
    gen_type_map = {}
    for res in [aruc, daruc]:
        if res.get("z_full") is not None:
            for _, row in res["z_full"][["gen_id", "gen_type"]].drop_duplicates().iterrows():
                gen_type_map[row["gen_id"]] = row["gen_type"]

    for name, p0_df in [("ARUC", aruc["p0"]), ("DARUC", daruc["p0"])]:
        total_by_type = {}
        for gen_id in p0_df.index:
            gt = gen_type_map.get(gen_id, "THERMAL")
            total_by_type[gt] = total_by_type.get(gt, 0) + p0_df.loc[gen_id, common_times].sum()
        parts = [f"{gt}={v:,.1f}" for gt, v in sorted(total_by_type.items())]
        lines.append(f"  {name:6s}: {', '.join(parts)}")

    # Z matrix summary
    lines.append("\n--- Z Matrix Summary ---")
    for name, res in [("ARUC", aruc), ("DARUC", daruc)]:
        if res.get("z_full") is None:
            continue
        zf = res["z_full"]
        zf_t = zf[zf["time"].isin(common_times)]
        active = zf_t[zf_t["Z_row_norm"] > 1e-6]
        active_gens = active["gen_id"].nunique()
        wind_z = active[active["gen_type"] == "WIND"]
        mean_wind_norm = wind_z["Z_row_norm"].mean() if not wind_z.empty else 0
        lines.append(f"  {name}: {active_gens} generators with active Z, "
                     f"mean wind ||Z||={mean_wind_norm:.4f}")

    # DARUC deviation summary
    if daruc.get("deviation") is not None and not daruc["deviation"].empty:
        dev = daruc["deviation"]
        lines.append("\n--- DARUC Extra Commitments (beyond DAM) ---")
        lines.append(f"  Generators with extra hours: {len(dev)}")
        lines.append(f"  Total extra unit-hours:      {dev['extra_committed_hours'].sum()}")
        for _, row in dev.head(10).iterrows():
            lines.append(f"    {row['gen_id']} ({row['gen_type']}): "
                         f"+{row['extra_committed_hours']}h (DAM={row['dam_committed_hours']}h)")

    # Wind curtailment
    if data is not None:
        lines.append("\n--- Wind Curtailment ---")
        header = f"  {'':10s}  {'Total (MWh)':>12s}  {'% Available':>12s}"
        lines.append(header)
        formulations = [("ARUC", aruc), ("DARUC", daruc)]
        if dam is not None:
            formulations.insert(0, ("DAM", dam))
        for name, res in formulations:
            c = compute_wind_curtailment(res["p0"], data, common_times)
            lines.append(f"  {name:10s}  {c['total_mwh']:>12,.1f}  {c['pct']:>11.1f}%")
            for gid, avg in c["per_farm"].items():
                lines.append(f"    {gid}: {avg:.1f} MW avg")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    print(text)

    with open(out_dir / "comparison_summary.txt", "w") as f:
        f.write(text)
    print(f"\n  Saved comparison_summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compare ARUC-LDR vs DARUC results")
    parser.add_argument("--aruc-dir", type=str, default="aruc_outputs/quick_test_copperplate",
                        help="ARUC results directory")
    parser.add_argument("--daruc-dir", type=str, default="daruc_outputs/quick_test_copperplate",
                        help="DARUC results directory")
    parser.add_argument("--out-dir", type=str, default="comparison_outputs",
                        help="Output directory for figures and summary")
    args = parser.parse_args()

    aruc_dir = Path(args.aruc_dir)
    daruc_dir = Path(args.daruc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    aruc = load_results(aruc_dir, "ARUC-LDR")
    daruc = load_results(daruc_dir, "DARUC")
    dam = load_dam_results(daruc_dir)

    common_times = _align_time(aruc, daruc, dam)
    print(f"  Common time periods: {len(common_times)}")

    # Try to build DAMData for cost breakdown + curtailment
    cost_aruc = cost_daruc = cost_dam = None
    data = None
    try:
        from io_rts import build_damdata_from_rts
        print("Building DAMData for cost breakdown...")
        data = build_damdata_from_rts(
            source_dir=Path("RTS_Data/SourceData"),
            ts_dir=Path("RTS_Data/timeseries_data_files"),
            start_time=pd.Timestamp(common_times[0]),
            horizon_hours=len(common_times),
        )
        cost_aruc = compute_cost_breakdown(aruc["u"][common_times], aruc["p0"][common_times], data)
        cost_daruc = compute_cost_breakdown(daruc["u"][common_times], daruc["p0"][common_times], data)
        if dam is not None:
            cost_dam = compute_cost_breakdown(dam["u"][common_times], dam["p0"][common_times], data)
        print("  Cost breakdown computed.")
    except Exception as e:
        print(f"  Could not compute cost breakdown: {e}")
        print("  Will show objective totals only.")

    # Generate figures
    print("\nGenerating figures...")
    fig_commitment_and_cost(aruc, daruc, dam, common_times, cost_aruc, cost_daruc, cost_dam, out_dir)
    fig_z_heatmaps(aruc, daruc, common_times, out_dir)
    if data is not None:
        fig_wind_curtailment(aruc, daruc, dam, common_times, data, out_dir)

    # Text summary
    print()
    write_summary(aruc, daruc, dam, common_times, cost_aruc, cost_daruc, cost_dam, out_dir, data=data)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
