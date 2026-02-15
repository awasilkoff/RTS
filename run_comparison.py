"""
Run DARUC and ARUC with identical parameters, then compare.

Single script that:
  1. Runs DARUC (two-step: DAM → robust reliability) — DAM comes for free
  2. Runs ARUC-LDR (one-shot robust) with the same horizon/rho/network settings
  3. Saves all outputs to a single comparison directory
  4. Generates comparison figures + text summary

Defaults tuned for visible differences between the two formulations:
  - July summer peak (high load → binding constraints → structural differences matter)
  - rho=3.0 (large enough uncertainty set to force meaningful hedging)
  - 12h horizon (enough structure, fast solve)
  - Copperplate (no line limits) — keeps focus on commitment/dispatch differences

Usage:
    python run_comparison.py
    python run_comparison.py --hours 24 --rho 2.0 --start-month 7 --start-day 15
    python run_comparison.py --enforce-lines   # enable line limits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_rts_daruc import run_rts_daruc
from run_rts_aruc import run_rts_aruc
from test_daruc_quick import analyze_Z
from compare_aruc_vs_daruc import (
    load_results,
    load_dam_results,
    _align_time,
    _round_commitment,
    compute_cost_breakdown,
    fig_commitment_and_cost,
    fig_z_heatmaps,
    fig_wind_curtailment,
    write_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run DARUC + ARUC with identical parameters, then compare"
    )
    parser.add_argument("--hours", type=int, default=12,
                        help="Horizon hours (default: 12)")
    parser.add_argument("--rho", type=float, default=3.0,
                        help="Ellipsoid radius (default: 3.0 for visible differences)")
    parser.add_argument("--start-month", type=int, default=7,
                        help="Start month (default: 7 = July peak load)")
    parser.add_argument("--start-day", type=int, default=15,
                        help="Start day (default: 15)")
    parser.add_argument("--start-hour", type=int, default=0,
                        help="Start hour (default: 0)")
    parser.add_argument("--enforce-lines", action="store_true",
                        help="Enforce line flow limits (default: copperplate)")
    parser.add_argument("--uncertainty-npz", type=str, default=None,
                        help="Path to time-varying uncertainty NPZ")
    parser.add_argument("--provider-start", type=int, default=0,
                        help="Start index into NPZ time series")
    parser.add_argument("--rho-lines-frac", type=float, default=None,
                        help="Fraction of rho for line flow constraints, e.g. 0.25 (default: 1.0 = same as rho)")
    parser.add_argument("--mip-gap", type=float, default=0.005,
                        help="MIP optimality gap (default: 0.005 = 0.5%%)")
    parser.add_argument("--incremental-obj", action="store_true",
                        help="DARUC: only charge commitment costs for additional units, scale dispatch by --dispatch-cost-scale")
    parser.add_argument("--dispatch-cost-scale", type=float, default=0.1,
                        help="Dispatch cost scale factor for incremental objective (default: 0.01)")
    parser.add_argument("--day2-interval", type=int, default=1,
                        help="Day-2 interval hours (default: 1 = hourly, 2 = 2-hour blocks)")
    parser.add_argument("--day1-only-robust", action="store_true",
                        help="Only enforce robust constraints for day 1 (first 24 periods)")
    parser.add_argument("--fix-wind-z", action="store_true",
                        help="Fix wind Z diagonal to 1 (wind fully tracks own realization, no curtailment)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    args = parser.parse_args()

    start_time = pd.Timestamp(year=2020, month=args.start_month,
                              day=args.start_day, hour=args.start_hour)

    # Build output directory name
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        net = "lines" if args.enforce_lines else "copperplate"
        if args.uncertainty_npz:
            rho_tag = "rho_npz"
        elif args.rho_lines_frac is not None:
            rho_tag = f"rho{args.rho}_linesfrac{args.rho_lines_frac}"
        else:
            rho_tag = f"rho{args.rho}"
        out_dir = Path(
            f"comparison_outputs/"
            f"m{args.start_month:02d}d{args.start_day:02d}_"
            f"{args.hours}h_{rho_tag}_{net}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    aruc_dir = out_dir / "aruc"
    daruc_dir = out_dir / "daruc"
    aruc_dir.mkdir(exist_ok=True)
    daruc_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"ARUC vs DARUC COMPARISON")
    print(f"  Start:    {start_time}")
    print(f"  Horizon:  {args.hours}h")
    if args.uncertainty_npz:
        print(f"  Rho:      from NPZ ({args.uncertainty_npz})")
    else:
        print(f"  Rho:      {args.rho}")
    if args.rho_lines_frac is not None:
        print(f"  Rho lines frac: {args.rho_lines_frac}")
    print(f"  MIP gap:  {args.mip_gap:.4f} ({args.mip_gap*100:.2f}%)")
    if args.incremental_obj:
        print(f"  DARUC obj: incremental (dispatch scale={args.dispatch_cost_scale})")
    if args.fix_wind_z:
        print(f"  Wind Z:   FIXED (diagonal=1, no curtailment)")
    print(f"  Network:  {'with line limits' if args.enforce_lines else 'copperplate'}")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    # ==================================================================
    # Step 1: Run DARUC (includes DAM as step 1)
    # ==================================================================
    print("\n" + "=" * 70)
    print("RUNNING DARUC (two-step: DAM → robust reliability)")
    print("=" * 70)

    daruc_outputs = run_rts_daruc(
        start_time=start_time,
        horizon_hours=args.hours,
        rho=args.rho,
        enforce_lines=args.enforce_lines,
        uncertainty_provider_path=args.uncertainty_npz,
        provider_start_idx=args.provider_start,
        rho_lines_frac=args.rho_lines_frac,
        mip_gap=args.mip_gap,
        incremental_obj=args.incremental_obj,
        dispatch_cost_scale=args.dispatch_cost_scale,
        day2_interval_hours=args.day2_interval,
        day1_only_robust=args.day1_only_robust,
        fix_wind_z=args.fix_wind_z,
    )

    daruc_results = daruc_outputs["daruc_results"]
    dam_results = daruc_outputs["dam_outputs"]["results"]
    data = daruc_outputs["data"]
    dev_df = daruc_outputs["deviation_summary"]

    # Save DARUC outputs
    daruc_results["u"].to_csv(daruc_dir / "commitment_u.csv")
    daruc_results["p0"].to_csv(daruc_dir / "dispatch_p0.csv")
    daruc_results["Z"].to_csv(daruc_dir / "Z_coefficients.csv")
    dev_df.to_csv(daruc_dir / "deviation_summary.csv", index=False)
    np.save(daruc_dir / "Sigma.npy", daruc_outputs["Sigma"])
    np.save(daruc_dir / "rho.npy", np.atleast_1d(daruc_outputs["rho"]))
    analyze_Z(daruc_results["Z"], data, daruc_dir, rho=daruc_outputs["rho"])

    # Save DAM results (DAM uses "p" not "p0")
    dam_results["u"].to_csv(daruc_dir / "dam_commitment_u.csv")
    dam_results["p"].to_csv(daruc_dir / "dam_dispatch_p0.csv")

    # Save DARUC summary
    daruc_summary = {
        "daruc_objective": daruc_results["obj"],
        "dam_objective": dam_results["obj"],
        "hours": args.hours,
        "rho_input": args.rho,
        "rho_lines_frac": args.rho_lines_frac,
        "mip_gap": args.mip_gap,
        "incremental_obj": args.incremental_obj,
        "dispatch_cost_scale": args.dispatch_cost_scale if args.incremental_obj else None,
        "time_varying": daruc_outputs["time_varying"],
        "enforce_lines": args.enforce_lines,
        "start_time": str(start_time),
    }
    with open(daruc_dir / "summary.json", "w") as f:
        json.dump(daruc_summary, f, indent=2)

    print(f"\nDARUC objective: {daruc_results['obj']:,.2f}")
    print(f"DAM objective:   {dam_results['obj']:,.2f}")

    # ==================================================================
    # Step 2: Run ARUC (one-shot) with same parameters
    # ==================================================================
    print("\n" + "=" * 70)
    print("RUNNING ARUC-LDR (one-shot robust)")
    print("=" * 70)

    aruc_outputs = run_rts_aruc(
        start_time=start_time,
        horizon_hours=args.hours,
        rho=args.rho,
        enforce_lines=args.enforce_lines,
        uncertainty_provider_path=args.uncertainty_npz,
        provider_start_idx=args.provider_start,
        rho_lines_frac=args.rho_lines_frac,
        mip_gap=args.mip_gap,
        day2_interval_hours=args.day2_interval,
        day1_only_robust=args.day1_only_robust,
        fix_wind_z=args.fix_wind_z,
    )

    aruc_results = aruc_outputs["results"]

    # Save ARUC outputs
    aruc_results["u"].to_csv(aruc_dir / "commitment_u.csv")
    aruc_results["p0"].to_csv(aruc_dir / "dispatch_p0.csv")
    aruc_results["Z"].to_csv(aruc_dir / "Z_coefficients.csv")
    np.save(aruc_dir / "Sigma.npy", aruc_outputs["Sigma"])
    np.save(aruc_dir / "rho.npy", np.atleast_1d(aruc_outputs["rho"]))
    analyze_Z(aruc_results["Z"], data, aruc_dir, rho=aruc_outputs["rho"])

    # Save ARUC summary
    aruc_summary = {
        "objective": aruc_results["obj"],
        "hours": args.hours,
        "rho_input": args.rho,
        "rho_lines_frac": args.rho_lines_frac,
        "mip_gap": args.mip_gap,
        "time_varying": aruc_outputs["time_varying"],
        "enforce_lines": args.enforce_lines,
        "start_time": str(start_time),
    }
    with open(aruc_dir / "summary.json", "w") as f:
        json.dump(aruc_summary, f, indent=2)

    print(f"\nARUC-LDR objective: {aruc_results['obj']:,.2f}")

    # ==================================================================
    # Step 3: Compare
    # ==================================================================
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON")
    print("=" * 70)

    # Load from saved files (same path the standalone compare script uses)
    aruc_loaded = load_results(aruc_dir, "ARUC-LDR")
    daruc_loaded = load_results(daruc_dir, "DARUC")
    dam_loaded = load_dam_results(daruc_dir)

    common_times = _align_time(aruc_loaded, daruc_loaded, dam_loaded)
    print(f"  Common time periods: {len(common_times)}")

    # Cost breakdown using the DAMData we already have in memory
    cost_aruc = compute_cost_breakdown(
        aruc_loaded["u"][common_times], aruc_loaded["p0"][common_times], data
    )
    cost_daruc = compute_cost_breakdown(
        daruc_loaded["u"][common_times], daruc_loaded["p0"][common_times], data
    )
    cost_dam = None
    if dam_loaded is not None:
        cost_dam = compute_cost_breakdown(
            dam_loaded["u"][common_times], dam_loaded["p0"][common_times], data
        )

    # Figures
    print("\nGenerating figures...")
    fig_commitment_and_cost(
        aruc_loaded, daruc_loaded, dam_loaded, common_times,
        cost_aruc, cost_daruc, cost_dam, out_dir,
    )
    fig_z_heatmaps(aruc_loaded, daruc_loaded, common_times, out_dir)
    fig_wind_curtailment(aruc_loaded, daruc_loaded, dam_loaded, common_times, data, out_dir)

    # Text summary
    print()
    write_summary(
        aruc_loaded, daruc_loaded, dam_loaded, common_times,
        cost_aruc, cost_daruc, cost_dam, out_dir, data=data,
    )

    # Quick delta report
    print("\n" + "=" * 70)
    print("QUICK COMPARISON")
    print("=" * 70)
    print(f"  DAM objective:   {dam_results['obj']:>14,.2f}")
    print(f"  DARUC objective: {daruc_results['obj']:>14,.2f}  "
          f"(+{daruc_results['obj'] - dam_results['obj']:,.2f} vs DAM)")
    print(f"  ARUC objective:  {aruc_results['obj']:>14,.2f}  "
          f"(+{aruc_results['obj'] - dam_results['obj']:,.2f} vs DAM)")

    u_aruc = _round_commitment(aruc_loaded["u"][common_times])
    u_daruc = _round_commitment(daruc_loaded["u"][common_times])
    diff_count = (u_aruc.values != u_daruc.values).sum()
    print(f"\n  Commitment differences: {diff_count} (gen,hour) entries differ")
    print(f"\n  All outputs: {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
