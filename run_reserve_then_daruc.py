#!/usr/bin/env python3
"""
run_reserve_then_daruc.py

Two-step robustness gap analysis:

  Step 1: Solve DAM + Spinning Reserve (system-level hedging)
  Step 2: Feed that commitment into a DARUC with u >= u_reserve floor,
          enforce_lines=True, robust_ramp=True (full robustness features)

The DARUC uses an incremental objective (only charges for additional
commitments beyond DAM+Reserve) so the resulting cost and extra unit-hours
measure exactly what it takes to make the reserve solution fully robust.

This answers: "Given our current reserve-based commitment, what additional
commitments does a fully robust DARUC require?"

Usage:
    python run_reserve_then_daruc.py --rho 3.0 --start-month 7 --start-day 15
    python run_reserve_then_daruc.py --uncertainty-npz path/to/unc.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_rts import build_damdata_from_rts
from dam_model import build_dam_model
from aruc_model import build_aruc_ldr_model, align_uncertainty_to_aruc
from aruc_warm_start import warm_start_aruc_from_dam
from run_rts_dam import extract_solution as extract_dam_solution
from run_rts_aruc import (
    build_uncertainty_set,
    extract_solution as extract_aruc_solution,
    print_brief_summary,
    analyze_Z_patterns,
    reshape_uncertainty_for_variable_intervals,
    extract_line_margins,
)
from run_rts_daruc import extract_dam_commitment, analyze_deviations, print_deviation_summary
from runner_utils import (
    compute_reserve_from_uncertainty,
    save_robust_outputs,
    save_dam_outputs,
    save_line_flows_if_enabled,
    compute_day1_metrics,
)
from compare_aruc_vs_daruc import compute_cost_breakdown
from uncertainty_set_provider import UncertaintySetProvider


SPP_FORECASTS_PARQUET = Path(
    "uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet"
)
RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"


def run_reserve_then_daruc(
    start_time: pd.Timestamp,
    horizon_hours: int = 48,
    rho: float = 3.0,
    uncertainty_npz: str | None = None,
    provider_start_idx: int = 2448,
    rho_lines_frac: float | None = None,
    mip_gap: float = 0.005,
    dispatch_cost_scale: float = 0.01,
    day2_interval_hours: int = 2,
    day1_only_robust: bool = True,
    fix_wind_z: bool = True,
    include_renewables: bool = False,
    include_nuclear: bool = False,
    include_zero_marginal: bool | None = None,
    ramp_scale: float = 1.0,
    pmin_scale: float = 1.0,
    monitored_lines_threshold: float | None = 0.5,
    time_limit: float | None = 600.0,
    threads: int | None = None,
    bar_qcp_conv_tol: float | None = 1e-4,
    out_dir: Path | None = None,
) -> dict:
    """
    Run DAM+Reserve then DARUC with full robustness.

    Returns dict with all results and metrics.
    """

    # ==================================================================
    # Build DAMData
    # ==================================================================
    print("Building DAMData...")
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start_time,
        horizon_hours=horizon_hours,
        spp_forecasts_parquet=SPP_FORECASTS_PARQUET,
        spp_start_idx=0,
        day2_interval_hours=day2_interval_hours,
        single_block=True,
        include_renewables=include_renewables,
        include_nuclear=include_nuclear,
        include_zero_marginal=include_zero_marginal,
        ramp_scale=ramp_scale,
        pmin_scale=pmin_scale,
    )
    T = data.n_periods

    # ==================================================================
    # Build uncertainty set
    # ==================================================================
    time_varying = False
    sqrt_Sigma = None

    if uncertainty_npz is not None:
        print(f"\nLoading time-varying uncertainty from {uncertainty_npz}...")
        provider = UncertaintySetProvider.from_npz(uncertainty_npz)
        horizon = provider.get_by_indices(
            provider_start_idx, horizon_hours, compute_sqrt=True
        )
        Sigma, rho_arr, sqrt_Sigma = align_uncertainty_to_aruc(
            horizon, data, provider.get_wind_gen_ids()
        )
        time_varying = True

        if data.period_duration is not None:
            Sigma, rho_arr, sqrt_Sigma = reshape_uncertainty_for_variable_intervals(
                Sigma, rho_arr, data.period_duration, sqrt_Sigma
            )
            print(f"  Reshaped uncertainty to {T} variable-interval periods")

        print(f"  Sigma shape: {Sigma.shape}")
        print(f"  rho range: [{rho_arr.min():.3f}, {rho_arr.max():.3f}]")
        rho_val = rho_arr
    else:
        print("\nConstructing static uncertainty set...")
        Sigma, rho_val = build_uncertainty_set(data, rho=rho)

    # ==================================================================
    # Step 1: DAM + Spinning Reserve
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DAM + SPINNING RESERVE")
    print("=" * 70)

    R = compute_reserve_from_uncertainty(Sigma, rho_val, T=T)
    print(f"  Reserve R[t]: min={R.min():.1f}, max={R.max():.1f}, mean={R.mean():.1f} MW")

    # DAM+Reserve uses enforce_lines=True (matching the DARUC)
    reserve_model, reserve_vars = build_dam_model(
        data, M_p=1e4, model_name="DAM_Reserve",
        enforce_lines=True,
        reserve_requirement=R,
    )
    reserve_model.Params.MIPGap = mip_gap
    print("  Solving DAM+Reserve...")
    reserve_model.optimize()

    from gurobipy import GRB as _GRB
    if reserve_model.Status not in [_GRB.OPTIMAL, _GRB.SUBOPTIMAL]:
        raise RuntimeError(f"DAM+Reserve infeasible (status={reserve_model.Status})")

    reserve_results = extract_dam_solution(data, reserve_model, reserve_vars)
    print(f"  DAM+Reserve objective: {reserve_results['obj']:,.2f}")

    # ==================================================================
    # Extract reserve commitment as DARUC floor
    # ==================================================================
    print("\nExtracting DAM+Reserve commitment for DARUC floor...")
    reserve_commitment = extract_dam_commitment(reserve_results, data)
    rc_u_hours = reserve_commitment["u"].sum()
    rc_startups = reserve_commitment["v"].sum()
    print(f"  Reserve unit-hours: {rc_u_hours:.0f}")
    print(f"  Reserve startups:   {rc_startups:.0f}")

    # ==================================================================
    # Optional: filter to monitored lines based on reserve dispatch
    # ==================================================================
    data_full = None
    line_mask = None
    if monitored_lines_threshold is not None:
        from compute_branch_flows import filter_monitored_lines
        data_full = data
        data, line_mask = filter_monitored_lines(
            data, reserve_results["p"].values, monitored_lines_threshold
        )

    # ==================================================================
    # Step 2: DARUC with reserve commitment floor + full robustness
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: DARUC (u >= u_reserve, lines + robust ramps)")
    print("=" * 70)

    robust_mask = None
    if day1_only_robust and T > 24:
        robust_mask = np.array([True] * 24 + [False] * (T - 24))
        print(f"  day1_only_robust: {int(robust_mask.sum())} robust + "
              f"{T - int(robust_mask.sum())} nominal periods")

    print("\nBuilding DARUC model (with reserve commitment floor)...")
    model, vars_dict = build_aruc_ldr_model(
        data=data,
        Sigma=Sigma,
        rho=rho_val,
        rho_lines_frac=rho_lines_frac,
        sqrt_Sigma=sqrt_Sigma,
        M_p=1e4,
        model_name="DARUC_ReserveFloor",
        dam_commitment=reserve_commitment,
        enforce_lines=True,
        mip_gap=mip_gap,
        incremental_obj=True,
        dispatch_cost_scale=dispatch_cost_scale,
        robust_mask=robust_mask,
        fix_wind_z=fix_wind_z,
        worst_case_cost=True,
        robust_ramp=True,
        time_limit=time_limit,
        threads=threads,
        bar_qcp_conv_tol=bar_qcp_conv_tol,
        line_mask=line_mask,
    )

    # Warm start from reserve solution
    if reserve_vars is not None:
        warm_start_aruc_from_dam(model, vars_dict, reserve_vars, data)

    print("  Solving DARUC...")
    model.optimize()

    import gurobipy as gp
    if model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        if model.SolCount == 0:
            raise RuntimeError("No feasible DARUC solution found.")
        print(f"WARNING: DARUC status={model.Status}")

    # Iterative line violation resolution (if lines were filtered)
    if data_full is not None:
        from compute_branch_flows import iterative_line_resolve
        _rmask = robust_mask if robust_mask is not None else np.ones(T, dtype=bool)
        iterative_line_resolve(
            model, vars_dict, data, data_full,
            _rmask, sqrt_Sigma, rho_val,
            rho_lines_frac, time_varying,
        )

    daruc_results = extract_aruc_solution(data, model, vars_dict)
    print_brief_summary(daruc_results, data)
    analyze_Z_patterns(daruc_results["Z"], data)

    # Extract robust line margins (rho * z_line)
    daruc_margin = extract_line_margins(
        vars_dict, data, rho_val,
        rho_lines_frac, time_varying,
    )

    # ==================================================================
    # Deviation analysis
    # ==================================================================
    dev_df = analyze_deviations(data, model, vars_dict, reserve_commitment)
    print_deviation_summary(dev_df, reserve_results["obj"], daruc_results["obj"])

    # Verify u_daruc >= u_reserve
    u_daruc = daruc_results["u"].values
    u_reserve = reserve_commitment["u"]
    violations = (u_daruc < u_reserve - 0.5).sum()
    if violations > 0:
        print(f"\nWARNING: {violations} violations of u_DARUC >= u_reserve!")
    else:
        print("\nVerified: u_DARUC >= u_reserve for all (i, t).")

    # ==================================================================
    # Day-1 metrics
    # ==================================================================
    d1_mask = data.day1_period_mask()
    d1_times = [t for t, m in zip(data.time, d1_mask) if m]
    dt = data.dt[d1_mask]

    cost_reserve = compute_cost_breakdown(
        reserve_results["u"][d1_times], reserve_results["p"][d1_times], data
    )
    cost_daruc = compute_cost_breakdown(
        daruc_results["u"][d1_times], daruc_results["p0"][d1_times], data
    )

    # Extra unit-hours (day 1)
    u_res_d1 = np.round(reserve_results["u"][d1_times].values)
    u_dar_d1 = np.round(daruc_results["u"][d1_times].values)
    uh_reserve = float((u_res_d1 * dt).sum())
    uh_daruc = float((u_dar_d1 * dt).sum())
    uh_extra = uh_daruc - uh_reserve

    # Wind curtailment (day 1)
    is_wind = [i for i, gt in enumerate(data.gen_type) if gt.upper() == "WIND"]
    Pmax_2d = data.Pmax_2d()
    wind_pmax_d1 = Pmax_2d[np.ix_(is_wind, d1_mask)]
    curt_reserve = float(((wind_pmax_d1 - reserve_results["p"][d1_times].values[is_wind, :]) * dt).sum())
    curt_daruc = float(((wind_pmax_d1 - daruc_results["p0"][d1_times].values[is_wind, :]) * dt).sum())

    # ==================================================================
    # Print summary
    # ==================================================================
    lines = []
    lines.append("=" * 70)
    lines.append("RESERVE-THEN-DARUC ROBUSTNESS GAP (day-1 metrics)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'':30s} {'DAM+Reserve':>14s} {'DARUC':>14s} {'Delta':>14s}")
    lines.append("-" * 75)
    lines.append(f"{'Total cost ($)':30s} {cost_reserve['total']:>14,.2f} {cost_daruc['total']:>14,.2f} {cost_daruc['total'] - cost_reserve['total']:>+14,.2f}")
    lines.append(f"{'  Commitment (NL+SU)':30s} {cost_reserve['commitment']:>14,.2f} {cost_daruc['commitment']:>14,.2f} {cost_daruc['commitment'] - cost_reserve['commitment']:>+14,.2f}")
    lines.append(f"{'  No-load':30s} {cost_reserve['no_load']:>14,.2f} {cost_daruc['no_load']:>14,.2f} {cost_daruc['no_load'] - cost_reserve['no_load']:>+14,.2f}")
    lines.append(f"{'  Startup':30s} {cost_reserve['startup']:>14,.2f} {cost_daruc['startup']:>14,.2f} {cost_daruc['startup'] - cost_reserve['startup']:>+14,.2f}")
    lines.append(f"{'  Energy':30s} {cost_reserve['energy']:>14,.2f} {cost_daruc['energy']:>14,.2f} {cost_daruc['energy'] - cost_reserve['energy']:>+14,.2f}")
    lines.append("")
    lines.append(f"{'Unit-hours committed':30s} {uh_reserve:>14,.0f} {uh_daruc:>14,.0f} {uh_extra:>+14,.0f}")
    lines.append(f"{'Wind curtailment (MWh)':30s} {curt_reserve:>14,.1f} {curt_daruc:>14,.1f} {curt_daruc - curt_reserve:>+14,.1f}")
    if not dev_df.empty:
        lines.append("")
        lines.append(f"Generators with additional commitments: {len(dev_df)}")
        lines.append(f"Total extra unit-hours (full horizon): {dev_df['extra_committed_hours'].sum()}")
    else:
        lines.append("\nNo additional commitments needed — reserve solution is already fully robust.")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    # ==================================================================
    # Save outputs
    # ==================================================================
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        reserve_dir = out_dir / "dam_reserve"
        daruc_dir = out_dir / "daruc"
        reserve_dir.mkdir(exist_ok=True)
        daruc_dir.mkdir(exist_ok=True)

        # Reserve outputs
        reserve_summary = {
            "objective": reserve_results["obj"],
            "hours": horizon_hours,
            "reserve_min_mw": float(R.min()),
            "reserve_max_mw": float(R.max()),
            "reserve_mean_mw": float(R.mean()),
            "enforce_lines": True,
            "start_time": str(start_time),
        }
        save_dam_outputs(reserve_results, reserve_dir, summary_dict=reserve_summary, reserve_R=R)

        # DARUC outputs
        daruc_summary = {
            "daruc_objective": daruc_results["obj"],
            "reserve_objective": reserve_results["obj"],
            "hours": horizon_hours,
            "enforce_lines": True,
            "robust_ramp": True,
            "incremental_obj": True,
            "dispatch_cost_scale": dispatch_cost_scale,
            "start_time": str(start_time),
        }
        save_robust_outputs(
            daruc_results, data, daruc_dir, Sigma, rho_val,
            deviation_df=dev_df, margin_df=daruc_margin,
            summary_dict=daruc_summary,
        )

        # Line flow analysis (nominal flows + margin decomposition)
        data_out = data_full if data_full is not None else data
        print("\nLine flow analysis:")
        save_line_flows_if_enabled(True, data_out, reserve_results["p"].values, None, "DAM+Reserve", reserve_dir, "line_flows")
        save_line_flows_if_enabled(True, data_out, daruc_results["p0"].values, daruc_margin, "DARUC", daruc_dir, "line_flows")

        with open(out_dir / "summary.txt", "w") as f:
            f.write(summary_text)

        # Combined JSON for suite integration
        combined = {
            "reserve_cost": cost_reserve,
            "daruc_cost": cost_daruc,
            "unit_hours": {"reserve": uh_reserve, "daruc": uh_daruc, "extra": uh_extra},
            "wind_curtailment_mwh": {"reserve": curt_reserve, "daruc": curt_daruc},
            "extra_generators": len(dev_df),
            "extra_unit_hours_full_horizon": int(dev_df["extra_committed_hours"].sum()) if not dev_df.empty else 0,
            "reserve_requirement": {
                "min": float(R.min()), "max": float(R.max()), "mean": float(R.mean()),
            },
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(combined, f, indent=2)

        print(f"\nOutputs saved to {out_dir}/")

    return {
        "reserve_results": reserve_results,
        "daruc_results": daruc_results,
        "deviation_summary": dev_df,
        "data": data,
        "cost_reserve": cost_reserve,
        "cost_daruc": cost_daruc,
        "Sigma": Sigma,
        "rho": rho_val,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DAM+Reserve -> DARUC robustness gap analysis"
    )
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--uncertainty-npz", type=str, default="uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz")
    parser.add_argument("--provider-start", type=int, default=2448)
    parser.add_argument("--rho-lines-frac", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--dispatch-cost-scale", type=float, default=0.1)
    parser.add_argument("--day2-interval", type=int, default=2)
    parser.add_argument("--day1-only-robust", action="store_true", default=True)
    parser.add_argument("--no-day1-only-robust", dest="day1_only_robust", action="store_false")
    parser.add_argument("--fix-wind-z", action="store_true", default=True)
    parser.add_argument("--no-fix-wind-z", dest="fix_wind_z", action="store_false")
    parser.add_argument("--include-renewables", action=argparse.BooleanOptionalAction, default=False,
                        help="Include solar (PV/RTPV) and hydro generators (default: exclude)")
    parser.add_argument("--include-nuclear", action=argparse.BooleanOptionalAction, default=False,
                        help="Include nuclear generators (default: exclude)")
    parser.add_argument("--include-zero-marginal", action=argparse.BooleanOptionalAction, default=None,
                        help="Override: include/exclude all zero-marginal-cost non-wind generators")
    parser.add_argument("--ramp-scale", type=float, default=1.0)
    parser.add_argument("--pmin-scale", type=float, default=1.0)
    parser.add_argument("--line-monitor-threshold", type=float, default=0.9)
    parser.add_argument("--time-limit", type=float, default=6000)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--bar-qcp-conv-tol", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    start_time = pd.Timestamp(
        year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour
    )

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = f"rho{args.rho}_{args.hours}h_m{args.start_month:02d}d{args.start_day:02d}"
        out_dir = Path("reserve_then_daruc_outputs") / tag

    run_reserve_then_daruc(
        start_time=start_time,
        horizon_hours=args.hours,
        rho=args.rho,
        uncertainty_npz=args.uncertainty_npz,
        provider_start_idx=args.provider_start,
        rho_lines_frac=args.rho_lines_frac,
        mip_gap=args.mip_gap,
        dispatch_cost_scale=args.dispatch_cost_scale,
        day2_interval_hours=args.day2_interval,
        day1_only_robust=args.day1_only_robust,
        fix_wind_z=args.fix_wind_z,
        include_renewables=args.include_renewables,
        include_nuclear=args.include_nuclear,
        include_zero_marginal=args.include_zero_marginal,
        ramp_scale=args.ramp_scale,
        pmin_scale=args.pmin_scale,
        monitored_lines_threshold=args.line_monitor_threshold,
        time_limit=args.time_limit,
        threads=args.threads,
        bar_qcp_conv_tol=args.bar_qcp_conv_tol,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
