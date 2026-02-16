"""
run_price_of_robustness.py

Sweep rho values and measure cost / wind curtailment for DAM, DARUC, and ARUC.

This is an overnight orchestrator — each rho value runs a full DARUC + ARUC solve.
Per the execution policy, the user should run this manually:

    python run_price_of_robustness.py --rhos 1.0 3.0 --hours 6 --start-month 7 --start-day 15

Full overnight run (~30-40 min for 8 rho at 12h with lines):

    python run_price_of_robustness.py --hours 12 --start-month 7 --start-day 15

Outputs (in --out-dir, default price_of_robustness/):
    sweep_results.csv           - Metrics for each rho
    fig_price_of_robustness.pdf - Cost vs rho
    fig_curtailment_vs_rho.pdf  - Wind curtailment vs rho
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_rts_daruc import run_rts_daruc
from run_rts_aruc import run_rts_aruc
from compare_aruc_vs_daruc import (
    compute_wind_curtailment,
    compute_cost_breakdown,
    _round_commitment,
    _align_time,
)


def _unit_hours(u_df: pd.DataFrame, times: list[str]) -> int:
    return int(_round_commitment(u_df[times]).values.sum())


def run_sweep(args) -> pd.DataFrame:
    """Run DARUC + ARUC for each rho and collect metrics."""
    start_time = pd.Timestamp(
        year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for rho in args.rhos:
        print("\n" + "#" * 70)
        print(f"# RHO = {rho}")
        print("#" * 70)
        t0 = time.time()

        # --- DARUC (includes DAM as step 1) ---
        try:
            daruc_out = run_rts_daruc(
                start_time=start_time,
                horizon_hours=args.hours,
                rho=rho,
                enforce_lines=args.enforce_lines,
                rho_lines_frac=args.rho_lines_frac,
                mip_gap=args.mip_gap,
                single_block=not args.three_blocks,
            )
            data = daruc_out["data"]
            daruc_res = daruc_out["daruc_results"]
            dam_res = daruc_out["dam_outputs"]["results"]

            # Build common_times from data
            common_times = list(data.time)

            dam_obj = dam_res["obj"]
            daruc_obj = daruc_res["obj"]

            # DAM dispatch uses "p" key, not "p0"
            dam_p0 = dam_res["p"]
            daruc_p0 = daruc_res["p0"]

            dam_curtail = compute_wind_curtailment(dam_p0, data, common_times)
            daruc_curtail = compute_wind_curtailment(daruc_p0, data, common_times)
            dam_uhours = _unit_hours(dam_res["u"], common_times)
            daruc_uhours = _unit_hours(daruc_res["u"], common_times)

            dam_cost = compute_cost_breakdown(dam_res["u"][common_times], dam_p0[common_times], data)
            daruc_cost = compute_cost_breakdown(daruc_res["u"][common_times], daruc_p0[common_times], data)
        except Exception as e:
            print(f"DARUC failed at rho={rho}: {e}")
            continue

        # --- ARUC ---
        try:
            aruc_out = run_rts_aruc(
                start_time=start_time,
                horizon_hours=args.hours,
                rho=rho,
                enforce_lines=args.enforce_lines,
                rho_lines_frac=args.rho_lines_frac,
                mip_gap=args.mip_gap,
                single_block=not args.three_blocks,
            )
            aruc_res = aruc_out["results"]
            aruc_obj = aruc_res["obj"]
            aruc_p0 = aruc_res["p0"]
            aruc_curtail = compute_wind_curtailment(aruc_p0, data, common_times)
            aruc_uhours = _unit_hours(aruc_res["u"], common_times)
            aruc_cost = compute_cost_breakdown(aruc_res["u"][common_times], aruc_p0[common_times], data)
        except Exception as e:
            print(f"ARUC failed at rho={rho}: {e}")
            continue

        elapsed = time.time() - t0

        row = {
            "rho": rho,
            "dam_obj": dam_obj,
            "daruc_obj": daruc_obj,
            "aruc_obj": aruc_obj,
            "dam_cost_total": dam_cost["total"],
            "daruc_cost_total": daruc_cost["total"],
            "aruc_cost_total": aruc_cost["total"],
            "dam_curtail_mwh": dam_curtail["total_mwh"],
            "daruc_curtail_mwh": daruc_curtail["total_mwh"],
            "aruc_curtail_mwh": aruc_curtail["total_mwh"],
            "dam_curtail_pct": dam_curtail["pct"],
            "daruc_curtail_pct": daruc_curtail["pct"],
            "aruc_curtail_pct": aruc_curtail["pct"],
            "dam_unit_hours": dam_uhours,
            "daruc_unit_hours": daruc_uhours,
            "aruc_unit_hours": aruc_uhours,
            "elapsed_s": elapsed,
        }
        rows.append(row)
        print(f"\n  rho={rho} done in {elapsed:.0f}s  "
              f"DAM={dam_obj:,.0f}  DARUC={daruc_obj:,.0f}  ARUC={aruc_obj:,.0f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    return df


def plot_price_of_robustness(df: pd.DataFrame, out_dir: Path):
    """Cost vs rho figure with DAM baseline, DARUC, ARUC."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axhline(df["dam_obj"].iloc[0], color="#2ca02c", linestyle="--", linewidth=1.2, label="DAM (baseline)")
    ax.plot(df["rho"], df["daruc_obj"], "o-", color="#ff7f0e", linewidth=1.5, markersize=5, label="DARUC")
    ax.plot(df["rho"], df["aruc_obj"], "s-", color="#1f77b4", linewidth=1.5, markersize=5, label="ARUC")

    # Shade region between DARUC and ARUC
    ax.fill_between(df["rho"], df["daruc_obj"], df["aruc_obj"], alpha=0.15, color="#1f77b4")

    ax.set_xlabel(r"Uncertainty radius $\rho$", fontsize=10)
    ax.set_ylabel("Objective cost ($)", fontsize=10)
    ax.set_title("Price of Robustness", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Secondary y-axis: % increase vs DAM
    dam_base = df["dam_obj"].iloc[0]
    if dam_base > 0:
        ax2 = ax.twinx()
        ax2.set_ylabel("% increase vs DAM", fontsize=9)
        y_lo, y_hi = ax.get_ylim()
        ax2.set_ylim(100 * (y_lo / dam_base - 1), 100 * (y_hi / dam_base - 1))
        ax2.tick_params(labelsize=8)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_price_of_robustness.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_price_of_robustness.pdf/.png")


def plot_curtailment_vs_rho(df: pd.DataFrame, out_dir: Path):
    """Wind curtailment vs rho figure."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axhline(df["dam_curtail_mwh"].iloc[0], color="#2ca02c", linestyle="--", linewidth=1.2, label="DAM (baseline)")
    ax.plot(df["rho"], df["daruc_curtail_mwh"], "o-", color="#ff7f0e", linewidth=1.5, markersize=5, label="DARUC")
    ax.plot(df["rho"], df["aruc_curtail_mwh"], "s-", color="#1f77b4", linewidth=1.5, markersize=5, label="ARUC")

    ax.set_xlabel(r"Uncertainty radius $\rho$", fontsize=10)
    ax.set_ylabel("Wind Curtailment (MWh)", fontsize=10)
    ax.set_title("Wind Curtailment vs Robustness", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_curtailment_vs_rho.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_curtailment_vs_rho.pdf/.png")


def main():
    parser = argparse.ArgumentParser(description="Price-of-robustness sweep over rho values")
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], help="Rho values to sweep (default: 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0)")
    parser.add_argument("--hours", type=int, default=12, help="Horizon hours (default: 12)")
    parser.add_argument("--start-month", type=int, default=7, help="Start month (default: 7)")
    parser.add_argument("--start-day", type=int, default=15, help="Start day (default: 15)")
    parser.add_argument("--start-hour", type=int, default=0, help="Start hour (default: 0)")
    parser.add_argument("--no-enforce-lines", dest="enforce_lines", action="store_false", help="Disable line flow limits (default: enforce lines)")
    parser.set_defaults(enforce_lines=True)
    parser.add_argument("--rho-lines-frac", type=float, default=None, help="Fraction of rho for line constraints")
    parser.add_argument("--mip-gap", type=float, default=0.005, help="MIP gap (default: 0.005)")
    parser.add_argument("--three-blocks", action="store_true", help="Use original 3-block piecewise cost (default: single block with weighted-average cost)")
    parser.add_argument("--out-dir", type=str, default="price_of_robustness", help="Output directory (default: price_of_robustness/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("PRICE OF ROBUSTNESS SWEEP")
    print("=" * 70)
    print(f"  Rho values: {args.rhos}")
    print(f"  Horizon:    {args.hours}h")
    print(f"  Start:      2020-{args.start_month:02d}-{args.start_day:02d} {args.start_hour:02d}:00")
    print(f"  Network:    {'with line limits' if args.enforce_lines else 'copperplate'}")
    print(f"  MIP gap:    {args.mip_gap}")
    print(f"  Output:     {out_dir}")
    print("=" * 70)

    df = run_sweep(args)

    if df.empty:
        print("No successful runs. Exiting.")
        return

    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nGenerating figures...")
    plot_price_of_robustness(df, out_dir)
    plot_curtailment_vs_rho(df, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
