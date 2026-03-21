"""
Run DARUC and ARUC with identical parameters, then compare.

Single script that:
  1. Runs DARUC (two-step: DAM -> robust reliability) — DAM comes for free
  2. Runs ARUC-LDR (one-shot robust) with the same horizon/rho/network settings
  3. Saves all outputs to a single comparison directory
  4. Generates comparison figures + text summary

Defaults tuned for visible differences between the two formulations:
  - July summer peak (high load -> binding constraints -> structural differences matter)
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
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dam_model import build_dam_model
from run_rts_daruc import run_rts_daruc
from run_rts_aruc import run_rts_aruc, extract_line_margins
from run_rts_dam import extract_solution as extract_dam_solution
from test_daruc_quick import analyze_Z
from runner_utils import (
    compute_reserve_from_uncertainty,
    save_robust_outputs,
    save_dam_outputs,
    save_line_flows_if_enabled,
    compute_day1_metrics,
)
from compare_aruc_vs_daruc import (
    load_results,
    load_dam_results,
    load_reserve_results,
    _align_time,
    _round_commitment,
    compute_cost_breakdown,
    fig_commitment_and_cost,
    fig_z_heatmaps,
    fig_wind_curtailment,
    fig_worst_case_wind,
    fig_pmin_vs_dispatch,
    write_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run DARUC + ARUC with identical parameters, then compare"
    )
    parser.add_argument(
        "--hours", type=int, default=48, help="Horizon hours (default: 48)"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=3.0,
        help="Ellipsoid radius (default: 3.0 for visible differences)",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=7,
        help="Start month (default: 7 = July peak load)",
    )
    parser.add_argument(
        "--start-day", type=int, default=15, help="Start day (default: 15)"
    )
    parser.add_argument(
        "--start-hour", type=int, default=0, help="Start hour (default: 0)"
    )
    parser.add_argument(
        "--enforce-lines",
        action="store_true",
        help="Enforce line flow limits (default: copperplate)",
        default=True,
    )
    parser.add_argument(
        "--uncertainty-npz",
        type=str,
        default="uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz",
        help="Path to time-varying uncertainty NPZ",
    )
    parser.add_argument(
        "--provider-start",
        type=int,
        default=2448,
        help="Start index into NPZ time series",
    )
    parser.add_argument(
        "--rho-lines-frac",
        type=float,
        default=None,
        help="Fraction of rho for line flow constraints, e.g. 0.25 (default: 1.0 = same as rho)",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.005,
        help="MIP optimality gap (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--incremental-obj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DARUC: only charge commitment costs for additional units, scale dispatch by --dispatch-cost-scale (default: True)",
    )
    parser.add_argument(
        "--dispatch-cost-scale",
        type=float,
        default=0.01,
        help="Dispatch cost scale factor for incremental objective (default: 0.01)",
    )
    parser.add_argument(
        "--day2-interval",
        type=int,
        default=2,
        help="Day-2 interval hours (default: 1 = hourly, 2 = 2-hour blocks)",
    )
    parser.add_argument(
        "--day1-only-robust",
        action="store_true",
        default=True,
        help="Only enforce robust constraints for day 1 (first 24 periods)",
    )
    parser.add_argument(
        "--fix-wind-z",
        action=argparse.BooleanOptionalAction,
        help="Fix wind Z diagonal to 1 (wind fully tracks own realization, no curtailment)",
        default=True,
    )
    parser.add_argument(
        "--three-blocks",
        action="store_true",
        help="Use original 3-block piecewise cost (default: single block with weighted-average cost)",
    )
    parser.add_argument(
        "--no-worst-case-cost",
        dest="worst_case_cost",
        action="store_false",
        help="Disable worst-case dispatch cost epigraph (use nominal dispatch cost only)",
    )
    parser.set_defaults(worst_case_cost=True)
    parser.add_argument(
        "--include-renewables",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include solar (PV/RTPV) and hydro generators (default: exclude)",
    )
    parser.add_argument(
        "--include-nuclear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include nuclear generators (default: exclude)",
    )
    parser.add_argument(
        "--include-zero-marginal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override: include/exclude all zero-marginal-cost non-wind generators",
    )
    parser.add_argument(
        "--ramp-scale",
        type=float,
        default=1.0,
        help="Multiply all ramp rates (RU, RD) by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--pmin-scale",
        type=float,
        default=1.0,
        help="Multiply all Pmin by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--robust-ramp",
        action=argparse.BooleanOptionalAction,
        help="Use robust (SOC-based) ramp constraints that account for worst-case dispatch deviations",
        default=True,
    )
    parser.add_argument(
        "--with-reserve",
        action="store_true",
        help="Re-solve DAM with spinning reserve derived from uncertainty set (works with --uncertainty-npz or scalar --rho)",
        default=True,
    )
    parser.add_argument(
        "--reserve-ramp-multiplier",
        type=float,
        default=1.0,
        help="Multiplier on reserve ramp-rate cap (RU*dt*mult). 0 to disable ramp cap entirely. Default: 1.0",
    )
    parser.add_argument(
        "--line-monitor-threshold",
        type=float,
        default=0.9,
        help="DAM loading threshold for line filtering (e.g. 0.5 = keep lines loaded >=50%%)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)",
    )
    # Solver performance tuning
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Gurobi time limit in seconds (default: no limit)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Gurobi thread count (default: Gurobi auto)",
    )
    parser.add_argument(
        "--bar-qcp-conv-tol",
        type=float,
        default=None,
        help="Barrier QCP convergence tolerance (default: Gurobi default 1e-8; try 1e-4 for speed)",
    )
    parser.add_argument(
        "--mip-focus", type=int, default=None,
        help="Gurobi MIPFocus (1=feasibility, 2=optimality, 3=bound). Default: 1",
    )
    parser.add_argument(
        "--node-file-start", type=float, default=None,
        help="Gurobi NodefileStart in GB (default 2.0). Raise on machines with more RAM.",
    )
    parser.add_argument(
        "--cuts", type=int, default=None,
        help="Gurobi Cuts aggressiveness (-1=auto, 0=off, 1=moderate, 2=aggressive, 3=very aggressive)",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Performance defaults ON by default: fix-wind-z, day1-only-robust, "
             "bar-qcp-conv-tol=1e-4, time-limit=60000, line-monitor-threshold=0.5 (when --enforce-lines). "
             "Use --no-fast to disable.",
    )
    args = parser.parse_args()

    # --fast: apply performance defaults for args not explicitly set
    if args.fast:
        if not args.fix_wind_z:
            args.fix_wind_z = True
        if not args.day1_only_robust:
            args.day1_only_robust = True
        if args.bar_qcp_conv_tol is None:
            args.bar_qcp_conv_tol = 1e-4
        if args.time_limit is None:
            args.time_limit = 60000.0
        if args.line_monitor_threshold is None and args.enforce_lines:
            args.line_monitor_threshold = 0.5

    start_time = pd.Timestamp(
        year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour
    )

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
        ramp_tag = f"_ramp{args.ramp_scale}x" if args.ramp_scale != 1.0 else ""
        pmin_tag = f"_pmin{args.pmin_scale}x" if args.pmin_scale != 1.0 else ""
        out_dir = Path(
            f"comparison_outputs/"
            f"m{args.start_month:02d}d{args.start_day:02d}_"
            f"{args.hours}h_{rho_tag}_{net}{ramp_tag}{pmin_tag}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    aruc_dir = out_dir / "aruc"
    daruc_dir = out_dir / "daruc"
    aruc_dir.mkdir(exist_ok=True)
    daruc_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"ARUC vs DARUC COMPARISON")
    if not args.fast:
        print(f"  Mode:     PRECISE (--no-fast, performance defaults disabled)")
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
    if args.ramp_scale != 1.0:
        print(f"  Ramp scale: {args.ramp_scale}x")
    if args.pmin_scale != 1.0:
        print(f"  Pmin scale: {args.pmin_scale}x")
    if args.robust_ramp:
        print(f"  Robust ramp: ON (SOC-based worst-case ramp constraints)")
    if args.with_reserve:
        print(f"  DAM+Reserve: ON (spinning reserve from uncertainty set)")
    print(f"  Network:  {'with line limits' if args.enforce_lines else 'copperplate'}")
    if args.line_monitor_threshold is not None:
        print(f"  Line monitor: threshold={args.line_monitor_threshold*100:.0f}%")
    if args.time_limit is not None:
        print(f"  Time limit: {args.time_limit:.0f}s")
    if args.bar_qcp_conv_tol is not None:
        print(f"  BarQCPConvTol: {args.bar_qcp_conv_tol:.0e}")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    t_wall_start = time.time()
    timings = {}

    # ==================================================================
    # Step 1: Run DARUC (includes DAM as step 1)
    # ==================================================================
    print("\n" + "=" * 70)
    print("RUNNING DARUC (two-step: DAM -> robust reliability)")
    print("=" * 70)

    t0 = time.time()
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
        single_block=not args.three_blocks,
        worst_case_cost=args.worst_case_cost,
        include_renewables=args.include_renewables,
        include_nuclear=args.include_nuclear,
        include_zero_marginal=args.include_zero_marginal,
        ramp_scale=args.ramp_scale,
        pmin_scale=args.pmin_scale,
        robust_ramp=args.robust_ramp,
        monitored_lines_threshold=args.line_monitor_threshold,
        time_limit=args.time_limit,
        threads=args.threads,
        bar_qcp_conv_tol=args.bar_qcp_conv_tol,
        mip_focus=args.mip_focus,
        node_file_start=args.node_file_start,
        cuts=args.cuts,
    )

    daruc_results = daruc_outputs["daruc_results"]
    dam_results = daruc_outputs["dam_outputs"]["results"]
    data = daruc_outputs["data"]
    dev_df = daruc_outputs["deviation_summary"]

    # Save DARUC outputs
    daruc_margin = extract_line_margins(
        daruc_outputs["vars"], data, daruc_outputs["rho"],
        args.rho_lines_frac, daruc_outputs["time_varying"],
    )
    daruc_summary = {
        "daruc_objective": daruc_results["obj"],
        "dam_objective": dam_results["obj"],
        "hours": args.hours,
        "rho_input": args.rho,
        "rho_lines_frac": args.rho_lines_frac,
        "mip_gap": args.mip_gap,
        "incremental_obj": args.incremental_obj,
        "dispatch_cost_scale": args.dispatch_cost_scale
        if args.incremental_obj
        else None,
        "time_varying": daruc_outputs["time_varying"],
        "enforce_lines": args.enforce_lines,
        "start_time": str(start_time),
    }
    save_robust_outputs(
        daruc_results, data, daruc_dir, daruc_outputs["Sigma"], daruc_outputs["rho"],
        deviation_df=dev_df, margin_df=daruc_margin,
        summary_dict=daruc_summary, analyze_z_fn=analyze_Z,
    )

    # Save DAM results (DAM uses "p" not "p0")
    dam_results["u"].to_csv(daruc_dir / "dam_commitment_u.csv")
    dam_results["p"].to_csv(daruc_dir / "dam_dispatch_p0.csv")

    # Line flow analysis (use unfiltered data for full line coverage)
    data_full = daruc_outputs.get("data_full") or data
    if args.enforce_lines:
        print("\nLine flow analysis:")
        save_line_flows_if_enabled(True, data_full, dam_results["p"].values, None, "DAM", daruc_dir, "dam_line_flows")
        save_line_flows_if_enabled(True, data_full, daruc_results["p0"].values, daruc_margin, "DARUC", daruc_dir, "daruc_line_flows")

    timings["daruc_total"] = time.time() - t0
    # Propagate sub-timings from DARUC
    daruc_timings = daruc_outputs.get("timings", {})
    daruc_line_iters = daruc_outputs.get("line_iterations", 0)

    print(f"\nDARUC objective: {daruc_results['obj']:,.2f}")
    print(f"DAM objective:   {dam_results['obj']:,.2f}")

    # ==================================================================
    # Step 2: Run ARUC (one-shot) with same parameters
    # ==================================================================
    print("\n" + "=" * 70)
    print("RUNNING ARUC-LDR (one-shot robust)")
    print("=" * 70)

    t0 = time.time()
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
        single_block=not args.three_blocks,
        worst_case_cost=args.worst_case_cost,
        include_renewables=args.include_renewables,
        include_nuclear=args.include_nuclear,
        include_zero_marginal=args.include_zero_marginal,
        ramp_scale=args.ramp_scale,
        pmin_scale=args.pmin_scale,
        robust_ramp=args.robust_ramp,
        monitored_lines_threshold=args.line_monitor_threshold,
        dam_dispatch_for_screening=dam_results["p"].values if args.line_monitor_threshold is not None else None,
        time_limit=args.time_limit,
        threads=args.threads,
        bar_qcp_conv_tol=args.bar_qcp_conv_tol,
        mip_focus=args.mip_focus,
        node_file_start=args.node_file_start,
        cuts=args.cuts,
    )

    aruc_results = aruc_outputs["results"]

    # Save ARUC outputs
    aruc_margin = extract_line_margins(
        aruc_outputs["vars"], data, aruc_outputs["rho"],
        aruc_outputs.get("rho_lines_frac"), aruc_outputs["time_varying"],
    )
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
    save_robust_outputs(
        aruc_results, data, aruc_dir, aruc_outputs["Sigma"], aruc_outputs["rho"],
        margin_df=aruc_margin, summary_dict=aruc_summary, analyze_z_fn=analyze_Z,
    )
    save_line_flows_if_enabled(args.enforce_lines, data_full, aruc_results["p0"].values, aruc_margin, "ARUC", aruc_dir, "aruc_line_flows")

    timings["aruc_total"] = time.time() - t0
    aruc_timings = aruc_outputs.get("timings", {})
    aruc_line_iters = aruc_outputs.get("line_iterations", 0)

    print(f"\nARUC-LDR objective: {aruc_results['obj']:,.2f}")

    # ==================================================================
    # Step 2b (optional): DAM + Spinning Reserve
    # ==================================================================
    reserve_results = None
    reserve_dir = out_dir / "dam_reserve"
    if args.with_reserve:
        print("\n" + "=" * 70)
        print("RUNNING DAM + SPINNING RESERVE")
        print("=" * 70)

        Sigma = daruc_outputs["Sigma"]
        rho_val = daruc_outputs["rho"]
        R = compute_reserve_from_uncertainty(Sigma, rho_val, T=data.n_periods)
        print(
            f"  Reserve requirement R[t]: min={R.min():.1f}, max={R.max():.1f}, mean={R.mean():.1f} MW"
        )

        reserve_dir.mkdir(exist_ok=True)

        t0 = time.time()
        print("\nBuilding DAM model with spinning reserve constraint...")
        ramp_mult = args.reserve_ramp_multiplier if args.reserve_ramp_multiplier > 0 else None
        reserve_model, reserve_vars = build_dam_model(
            data,
            M_p=1e4,
            model_name="DAM_Reserve",
            enforce_lines=args.enforce_lines,
            reserve_requirement=R,
            reserve_ramp_multiplier=ramp_mult,
        )
        reserve_model.Params.MIPGap = args.mip_gap
        print("  Model built. Starting optimization...")
        reserve_model.optimize()

        from gurobipy import GRB as _GRB

        if reserve_model.Status not in [_GRB.OPTIMAL, _GRB.SUBOPTIMAL]:
            print(
                f"WARNING: DAM+Reserve did not solve optimally. Status: {reserve_model.Status}"
            )
        else:
            reserve_results = extract_dam_solution(data, reserve_model, reserve_vars)
            reserve_summary = {
                "objective": reserve_results["obj"],
                "hours": args.hours,
                "reserve_min_mw": float(R.min()),
                "reserve_max_mw": float(R.max()),
                "reserve_mean_mw": float(R.mean()),
                "reserve_ramp_multiplier": args.reserve_ramp_multiplier,
                "enforce_lines": args.enforce_lines,
                "start_time": str(start_time),
            }
            save_dam_outputs(reserve_results, reserve_dir, summary_dict=reserve_summary, reserve_R=R)
            save_line_flows_if_enabled(args.enforce_lines, data_full, reserve_results["p"].values, None, "DAM+Reserve", reserve_dir, "reserve_line_flows")

            print(f"\nDAM+Reserve objective: {reserve_results['obj']:,.2f}")
            timings["reserve_solve"] = time.time() - t0

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
    reserve_loaded = load_reserve_results(reserve_dir) if args.with_reserve else None

    common_times = _align_time(aruc_loaded, daruc_loaded, dam_loaded)
    print(f"  Common time periods: {len(common_times)}")

    # Day-1 times for metric reporting (day 2 is look-ahead only)
    d1_times = data.day1_times()
    # Intersect with common_times in case of mismatch
    d1_set = set(str(t) for t in d1_times)
    d1_common = [t for t in common_times if str(t) in d1_set]
    print(f"  Day-1 periods for metrics: {len(d1_common)}")

    # Cost breakdown using day-1 times only
    cost_aruc = compute_cost_breakdown(
        aruc_loaded["u"][d1_common], aruc_loaded["p0"][d1_common], data
    )
    cost_daruc = compute_cost_breakdown(
        daruc_loaded["u"][d1_common], daruc_loaded["p0"][d1_common], data
    )
    cost_dam = None
    if dam_loaded is not None:
        cost_dam = compute_cost_breakdown(
            dam_loaded["u"][d1_common], dam_loaded["p0"][d1_common], data
        )
    cost_reserve = None
    if reserve_loaded is not None:
        cost_reserve = compute_cost_breakdown(
            reserve_loaded["u"][d1_common], reserve_loaded["p0"][d1_common], data
        )

    # Figures
    print("\nGenerating figures...")
    fig_commitment_and_cost(
        aruc_loaded,
        daruc_loaded,
        dam_loaded,
        common_times,
        cost_aruc,
        cost_daruc,
        cost_dam,
        out_dir,
        data=data,
        reserve=reserve_loaded,
        cost_reserve=cost_reserve,
    )
    fig_z_heatmaps(aruc_loaded, daruc_loaded, common_times, out_dir)
    fig_wind_curtailment(
        aruc_loaded,
        daruc_loaded,
        dam_loaded,
        common_times,
        data,
        out_dir,
        reserve=reserve_loaded,
    )
    fig_pmin_vs_dispatch(
        aruc_loaded,
        daruc_loaded,
        dam_loaded,
        common_times,
        data,
        out_dir,
        reserve=reserve_loaded,
    )
    fig_worst_case_wind(
        aruc_loaded,
        daruc_loaded,
        dam_loaded,
        common_times,
        data,
        out_dir,
    )

    # Text summary (day-1 metrics)
    print()
    write_summary(
        aruc_loaded,
        daruc_loaded,
        dam_loaded,
        d1_common,
        cost_aruc,
        cost_daruc,
        cost_dam,
        out_dir,
        data=data,
        reserve=reserve_loaded,
        cost_reserve=cost_reserve,
    )

    # Quick delta report (day-1 costs)
    print("\n" + "=" * 70)
    print("QUICK COMPARISON (day-1 costs)")
    print("=" * 70)
    print(f"  DAM cost:        {cost_dam['total']:>14,.2f}" if cost_dam else "  DAM cost:        N/A")
    if cost_reserve is not None and cost_dam is not None:
        print(
            f"  DAM+Reserve:     {cost_reserve['total']:>14,.2f}  "
            f"(+{cost_reserve['total'] - cost_dam['total']:,.2f} vs DAM)"
        )
    print(
        f"  DARUC cost:      {cost_daruc['total']:>14,.2f}  "
        f"(+{cost_daruc['total'] - (cost_dam['total'] if cost_dam else 0):,.2f} vs DAM)"
    )
    print(
        f"  ARUC cost:       {cost_aruc['total']:>14,.2f}  "
        f"(+{cost_aruc['total'] - (cost_dam['total'] if cost_dam else 0):,.2f} vs DAM)"
    )

    u_aruc = _round_commitment(aruc_loaded["u"][d1_common])
    u_daruc = _round_commitment(daruc_loaded["u"][d1_common])
    diff_count = (u_aruc.values != u_daruc.values).sum()
    print(f"\n  Commitment differences (day 1): {diff_count} (gen,hour) entries differ")

    timings["total_wall"] = time.time() - t_wall_start

    # Print timing summary
    print("\n--- Timing ---")
    print(f"  {'DARUC (DAM + build + solve)':30s} {timings.get('daruc_total', 0):>8.1f} s")
    if daruc_timings:
        print(f"    {'DAM solve':28s} {daruc_timings.get('dam_solve', 0):>8.1f} s")
        print(f"    {'DARUC build':28s} {daruc_timings.get('daruc_build', 0):>8.1f} s")
        print(f"    {'DARUC solve':28s} {daruc_timings.get('daruc_solve', 0):>8.1f} s")
        if daruc_line_iters > 0:
            print(f"    {'Line iters (' + str(daruc_line_iters) + ' re-solves)':28s} {daruc_timings.get('line_iterations_solve', 0):>8.1f} s")
    print(f"  {'ARUC (build + solve)':30s} {timings.get('aruc_total', 0):>8.1f} s")
    if aruc_timings:
        print(f"    {'Model build':28s} {aruc_timings.get('model_build', 0):>8.1f} s")
        print(f"    {'Solve':28s} {aruc_timings.get('solve', 0):>8.1f} s")
        if aruc_line_iters > 0:
            print(f"    {'Line iters (' + str(aruc_line_iters) + ' re-solves)':28s} {aruc_timings.get('line_iterations_solve', 0):>8.1f} s")
    if "reserve_solve" in timings:
        print(f"  {'DAM+Reserve':30s} {timings['reserve_solve']:>8.1f} s")
    print(f"  {'Total wall time':30s} {timings['total_wall']:>8.1f} s")

    # Top-level summary.json (uniform interface for run_sensitivity_suite.py)
    metrics_aruc = compute_day1_metrics(data, aruc_results)
    metrics_daruc = compute_day1_metrics(data, daruc_results)
    top_summary = {
        "aruc_cost": cost_aruc,
        "daruc_cost": cost_daruc,
        "dam_cost": cost_dam,
        "reserve_cost": cost_reserve,
        "aruc_metrics": metrics_aruc,
        "daruc_metrics": metrics_daruc,
        "dam_metrics": compute_day1_metrics(data, dam_results) if dam_loaded is not None else None,
        "reserve_metrics": compute_day1_metrics(data, reserve_results) if reserve_results is not None else None,
        "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        "daruc_timings_seconds": {k: round(v, 2) for k, v in daruc_timings.items()} if daruc_timings else None,
        "aruc_timings_seconds": {k: round(v, 2) for k, v in aruc_timings.items()} if aruc_timings else None,
        "daruc_line_iterations": daruc_line_iters,
        "aruc_line_iterations": aruc_line_iters,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(top_summary, f, indent=2)

    print(f"\n  All outputs: {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
