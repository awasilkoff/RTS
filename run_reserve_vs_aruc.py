"""
run_reserve_vs_aruc.py

Compare DAM+Reserve vs stripped-down ARUC-LDR (copperplate, no robust ramps,
no worst-case cost epigraph) to test whether they produce equivalent results.

Both hedge against wind uncertainty:
  - DAM+Reserve: system-level spinning reserve r[i,t]
  - ARUC-LDR: per-generator LDR Z coefficients

If they match, the "extra" value of full ARUC comes from robust line constraints
and/or robust ramps, not the adaptive dispatch structure itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_rts import build_damdata_from_rts
from dam_model import build_dam_model
from run_rts_dam import extract_solution as extract_dam_solution
from run_rts_aruc import run_rts_aruc, build_uncertainty_set
from runner_utils import compute_reserve_from_uncertainty
from compare_aruc_vs_daruc import compute_cost_breakdown


def main():
    parser = argparse.ArgumentParser(
        description="Compare DAM+Reserve vs stripped ARUC-LDR"
    )
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--uncertainty-npz", type=str, default=None)
    parser.add_argument("--provider-start", type=int, default=2448)
    parser.add_argument("--day2-interval", type=int, default=2)
    parser.add_argument("--day1-only-robust", action="store_true", default=True)
    parser.add_argument("--no-day1-only-robust", dest="day1_only_robust", action="store_false")
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--time-limit", type=float, default=600)
    parser.add_argument("--bar-qcp-conv-tol", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    start_time = pd.Timestamp(year=2020, month=args.start_month, day=args.start_day,
                              hour=args.start_hour)

    # Output directory
    tag = f"rho{args.rho}_{args.hours}h_m{args.start_month:02d}d{args.start_day:02d}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("reserve_vs_aruc_outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # Build DAMData
    # ======================================================================
    print("Building DAMData...")
    data = build_damdata_from_rts(
        start_time=start_time,
        horizon_hours=args.hours,
        day2_interval_hours=args.day2_interval,
    )
    d1_mask = data.day1_period_mask()
    d1_times = [t for t, m in zip(data.time, d1_mask) if m]

    # ======================================================================
    # Build uncertainty set and reserve requirement
    # ======================================================================
    if args.uncertainty_npz:
        from aruc_model import align_uncertainty_to_aruc
        from uncertainty_set_provider import UncertaintySetProvider
        provider = UncertaintySetProvider(args.uncertainty_npz)
        raw = provider.get_window(args.provider_start, args.hours)
        Sigma, rho_val = align_uncertainty_to_aruc(data, raw)
    else:
        Sigma, rho_val = build_uncertainty_set(data, rho=args.rho)

    R = compute_reserve_from_uncertainty(Sigma, rho_val, T=data.n_periods)
    print(f"Reserve requirement R[t]: min={R.min():.1f}, max={R.max():.1f}, mean={R.mean():.1f} MW")

    # ======================================================================
    # 1. Solve DAM (baseline)
    # ======================================================================
    print("\n" + "=" * 70)
    print("SOLVING DAM (baseline)")
    print("=" * 70)
    dam_model, dam_vars = build_dam_model(data, M_p=1e4, enforce_lines=False)
    dam_model.Params.MIPGap = args.mip_gap
    dam_model.optimize()
    dam_results = extract_dam_solution(data, dam_model, dam_vars)
    print(f"  DAM objective: {dam_results['obj']:,.2f}")

    # ======================================================================
    # 2. Solve DAM + Reserve
    # ======================================================================
    print("\n" + "=" * 70)
    print("SOLVING DAM + SPINNING RESERVE")
    print("=" * 70)
    reserve_model, reserve_vars = build_dam_model(
        data, M_p=1e4, model_name="DAM_Reserve", reserve_requirement=R,
        enforce_lines=False,
    )
    reserve_model.Params.MIPGap = args.mip_gap
    reserve_model.optimize()
    reserve_results = extract_dam_solution(data, reserve_model, reserve_vars)
    print(f"  DAM+Reserve objective: {reserve_results['obj']:,.2f}")

    # ======================================================================
    # 3. Solve stripped ARUC-LDR (copperplate, no robust ramps, no wc cost)
    # ======================================================================
    print("\n" + "=" * 70)
    print("SOLVING STRIPPED ARUC-LDR (copperplate, no robust ramps, no wc cost)")
    print("=" * 70)
    aruc_outputs = run_rts_aruc(
        start_time=start_time,
        horizon_hours=args.hours,
        rho=args.rho,
        enforce_lines=False,
        uncertainty_provider_path=args.uncertainty_npz,
        provider_start_idx=args.provider_start,
        mip_gap=args.mip_gap,
        day2_interval_hours=args.day2_interval,
        day1_only_robust=args.day1_only_robust,
        fix_wind_z=True,
        worst_case_cost=False,
        robust_ramp=False,
        time_limit=args.time_limit,
        bar_qcp_conv_tol=args.bar_qcp_conv_tol,
    )
    aruc_results = aruc_outputs["results"]
    print(f"  ARUC objective: {aruc_results['obj']:,.2f}")

    # ======================================================================
    # Day-1 metrics
    # ======================================================================
    cost_dam = compute_cost_breakdown(
        dam_results["u"][d1_times], dam_results["p"][d1_times], data
    )
    cost_reserve = compute_cost_breakdown(
        reserve_results["u"][d1_times], reserve_results["p"][d1_times], data
    )
    cost_aruc = compute_cost_breakdown(
        aruc_results["u"][d1_times], aruc_results["p0"][d1_times], data
    )

    # Unit-hours committed (day 1)
    dt = data.dt[d1_mask]
    uh_dam = float((np.round(dam_results["u"][d1_times].values) * dt).sum())
    uh_reserve = float((np.round(reserve_results["u"][d1_times].values) * dt).sum())
    uh_aruc = float((np.round(aruc_results["u"][d1_times].values) * dt).sum())

    # Wind curtailment (day 1)
    is_wind = [i for i, gt in enumerate(data.gen_type) if gt.upper() == "WIND"]
    Pmax_2d = data.Pmax_2d()
    wind_pmax_d1 = Pmax_2d[np.ix_(is_wind, d1_mask)]

    dam_wind_d1 = dam_results["p"][d1_times].values[is_wind, :]
    reserve_wind_d1 = reserve_results["p"][d1_times].values[is_wind, :]
    aruc_wind_d1 = aruc_results["p0"][d1_times].values[is_wind, :]

    curt_dam = float(((wind_pmax_d1 - dam_wind_d1) * dt).sum())
    curt_reserve = float(((wind_pmax_d1 - reserve_wind_d1) * dt).sum())
    curt_aruc = float(((wind_pmax_d1 - aruc_wind_d1) * dt).sum())

    # Commitment diffs (Reserve vs ARUC, day 1)
    u_reserve_d1 = np.round(reserve_results["u"][d1_times].values).astype(int)
    u_aruc_d1 = np.round(aruc_results["u"][d1_times].values).astype(int)
    diff = u_aruc_d1 - u_reserve_d1  # +1 = ARUC commits more, -1 = fewer
    gen_ids = data.gen_ids

    more_gens = sorted(set(gen_ids[i] for i in range(len(gen_ids)) if (diff[i] > 0).any()))
    fewer_gens = sorted(set(gen_ids[i] for i in range(len(gen_ids)) if (diff[i] < 0).any()))
    n_diff_pairs = int(np.abs(diff).sum())

    # ======================================================================
    # Print and save summary
    # ======================================================================
    lines = []
    lines.append("=" * 70)
    lines.append("RESERVE vs STRIPPED ARUC-LDR (day-1, copperplate, no robust ramps)")
    lines.append("=" * 70)
    lines.append(f"  Scenario: rho={args.rho}, {args.hours}h, "
                 f"start={args.start_month:02d}/{args.start_day:02d}, "
                 f"day2_interval={args.day2_interval}h")
    lines.append(f"  Reserve R[t]: min={R.min():.1f}, max={R.max():.1f}, mean={R.mean():.1f} MW")
    lines.append("")
    lines.append(f"{'':25s} {'DAM':>14s} {'DAM+Reserve':>14s} {'ARUC-LDR':>14s}")
    lines.append("-" * 70)
    lines.append(f"{'Total cost ($)':25s} {cost_dam['total']:>14,.2f} {cost_reserve['total']:>14,.2f} {cost_aruc['total']:>14,.2f}")
    lines.append(f"{'  Commitment (NL+SU)':25s} {cost_dam['commitment']:>14,.2f} {cost_reserve['commitment']:>14,.2f} {cost_aruc['commitment']:>14,.2f}")
    lines.append(f"{'  No-load':25s} {cost_dam['no_load']:>14,.2f} {cost_reserve['no_load']:>14,.2f} {cost_aruc['no_load']:>14,.2f}")
    lines.append(f"{'  Startup':25s} {cost_dam['startup']:>14,.2f} {cost_reserve['startup']:>14,.2f} {cost_aruc['startup']:>14,.2f}")
    lines.append(f"{'  Energy':25s} {cost_dam['energy']:>14,.2f} {cost_reserve['energy']:>14,.2f} {cost_aruc['energy']:>14,.2f}")
    lines.append(f"{'Cost vs DAM':25s} {'--':>14s} {cost_reserve['total'] - cost_dam['total']:>+14,.2f} {cost_aruc['total'] - cost_dam['total']:>+14,.2f}")

    dam_total = cost_dam["total"]
    if dam_total > 0:
        lines.append(f"{'Cost vs DAM (%)':25s} {'--':>14s} {(cost_reserve['total'] - dam_total) / dam_total * 100:>+14.2f}% {(cost_aruc['total'] - dam_total) / dam_total * 100:>+14.2f}%")

    lines.append("")
    lines.append(f"{'Unit-hours committed':25s} {uh_dam:>14,.0f} {uh_reserve:>14,.0f} {uh_aruc:>14,.0f}")
    lines.append(f"{'Wind curtailment (MWh)':25s} {curt_dam:>14,.1f} {curt_reserve:>14,.1f} {curt_aruc:>14,.1f}")

    # Reserve vs ARUC delta
    lines.append("")
    delta_cost = cost_aruc["total"] - cost_reserve["total"]
    lines.append(f"ARUC vs DAM+Reserve cost delta: {delta_cost:+,.2f} ({delta_cost / cost_reserve['total'] * 100:+.3f}%)" if cost_reserve["total"] > 0 else f"ARUC vs DAM+Reserve cost delta: {delta_cost:+,.2f}")

    lines.append("")
    lines.append(f"Commitment diffs (ARUC vs DAM+Reserve, day 1):")
    lines.append(f"  Differing (gen,hour) pairs: {n_diff_pairs}")
    lines.append(f"  ARUC commits MORE:  {more_gens if more_gens else '(none)'}")
    lines.append(f"  ARUC commits FEWER: {fewer_gens if fewer_gens else '(none)'}")

    lines.append("")
    if abs(delta_cost) < 0.01 * dam_total and n_diff_pairs == 0:
        lines.append(">> RESULT: Stripped ARUC matches DAM+Reserve (equivalent)")
    elif abs(delta_cost) < 0.01 * dam_total:
        lines.append(">> RESULT: Costs match but commitments differ slightly")
    else:
        lines.append(">> RESULT: Stripped ARUC differs from DAM+Reserve")
        lines.append("   Per-generator adaptive hedging (Z) provides value beyond system reserve")

    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    # Save raw data for further analysis
    summary_json = {
        "dam_cost": cost_dam,
        "reserve_cost": cost_reserve,
        "aruc_cost": cost_aruc,
        "unit_hours": {"dam": uh_dam, "reserve": uh_reserve, "aruc": uh_aruc},
        "wind_curtailment_mwh": {"dam": curt_dam, "reserve": curt_reserve, "aruc": curt_aruc},
        "commitment_diffs": {
            "n_diff_pairs": n_diff_pairs,
            "aruc_more": more_gens,
            "aruc_fewer": fewer_gens,
        },
        "reserve_requirement": {
            "min": float(R.min()), "max": float(R.max()), "mean": float(R.mean()),
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
