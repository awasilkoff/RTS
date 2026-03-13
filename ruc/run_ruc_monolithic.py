"""
run_ruc_monolithic.py — Monolithic Notification-Gated LD-RUC pipeline.

Solves the gated robust reliability UC as a single MISOCP by reusing
the existing ARUC model builder (aruc_model.py) with:
  - dam_commitment (no-decommitment floor)
  - incremental_obj=True (commitment-cost-only objective)
  - gating_mask (restrict objective to non-deferrable decisions)

Pipeline:
  1. Build DAMData from RTS-GMLC
  2. Solve deterministic DAM (market solution)
  3. Load uncertainty sets (Sigma, rho)
  4. Assign notification times from generator fuel/type
  5. Compute gating sets
  6. Solve monolithic gated robust MISOCP (single solve)
  7. Save results and artifacts

This is the monolithic alternative to run_ruc.py (which uses a
two-phase CCG decomposition). Both approaches produce the same
optimal solution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DAMData
from aruc_model import build_aruc_ldr_model, align_uncertainty_to_aruc
from aruc_warm_start import warm_start_aruc_from_dam
from run_rts_dam import run_rts_dam
from run_rts_daruc import extract_dam_commitment, analyze_deviations, print_deviation_summary
from run_rts_aruc import (
    build_uncertainty_set,
    extract_solution,
    extract_line_margins,
    print_brief_summary,
    analyze_Z_patterns,
    reshape_uncertainty_for_variable_intervals,
)
from uncertainty_set_provider import UncertaintySetProvider

# Reuse notification time and gating set logic from the CCG version
from ruc_model import compute_gating_sets
from run_ruc import assign_notification_times


# ======================================================================
# Configuration
# ======================================================================

RTS_DIR = Path(__file__).resolve().parent.parent / "RTS_Data"
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"

START_TIME = pd.Timestamp(year=2020, month=7, day=15, hour=0)
HORIZON_HOURS = 48

M_PENALTY = 1e4
UNCERTAINTY_RHO = 0.5

SPP_FORECASTS_PARQUET = (
    Path(__file__).resolve().parent.parent
    / "uncertainty_sets_refactored"
    / "data"
    / "forecasts_filtered_rts4_constellation_v2.parquet"
)
SPP_START_IDX = 0


# ======================================================================
# Main pipeline
# ======================================================================


def run_ruc_monolithic(
    source_dir: Path = SOURCE_DIR,
    ts_dir: Path = TS_DIR,
    start_time: pd.Timestamp = START_TIME,
    horizon_hours: int = HORIZON_HOURS,
    t_next: int = 25,
    notification_scale: float = 1.0,
    m_penalty: float = M_PENALTY,
    rho: float = UNCERTAINTY_RHO,
    wind_std_fraction: float = 0.15,
    uncertainty_provider_path: Optional[Union[Path, str]] = None,
    provider_start_idx: int = 0,
    spp_forecasts_parquet: Optional[Path] = SPP_FORECASTS_PARQUET,
    spp_start_idx: int = SPP_START_IDX,
    enforce_lines: bool = True,
    rho_lines_frac: Optional[float] = None,
    mip_gap: float = 0.005,
    dispatch_cost_scale: float = 0.01,
    day2_interval_hours: int = 1,
    day1_only_robust: bool = False,
    fix_wind_z: bool = False,
    worst_case_cost: bool = True,
    robust_ramp: bool = False,
    include_renewables: bool = False,
    include_nuclear: bool = False,
    include_zero_marginal: Optional[bool] = None,
    ramp_scale: float = 1.0,
    pmin_scale: float = 1.0,
    monitored_lines_threshold: Optional[float] = None,
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
    bar_qcp_conv_tol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Monolithic LD-RUC pipeline: DAM -> gating -> single MISOCP solve.

    Returns dict with all results, models, and metadata.
    """
    import gurobipy as gp

    wall_start = _time.time()

    # ==================================================================
    # STEP 1: Deterministic DAM
    # ==================================================================
    print("=" * 70)
    print("STEP 1: DETERMINISTIC DAY-AHEAD UC")
    print("=" * 70)

    dam_outputs = run_rts_dam(
        source_dir=source_dir,
        ts_dir=ts_dir,
        start_time=start_time,
        horizon_hours=horizon_hours,
        m_penalty=m_penalty,
        spp_forecasts_parquet=spp_forecasts_parquet,
        spp_start_idx=spp_start_idx,
        enforce_lines=enforce_lines,
        day2_interval_hours=day2_interval_hours,
        include_renewables=include_renewables,
        include_nuclear=include_nuclear,
        include_zero_marginal=include_zero_marginal,
        ramp_scale=ramp_scale,
        pmin_scale=pmin_scale,
    )

    dam_results = dam_outputs["results"]
    data = dam_outputs["data"]
    T = data.n_periods

    # ==================================================================
    # STEP 2: Extract DAM commitment
    # ==================================================================
    print("\nExtracting DAM commitments...")
    dam_commitment = extract_dam_commitment(dam_results, data)
    dam_u_hours = dam_commitment["u"].sum()
    dam_startups = dam_commitment["v"].sum()
    print(f"  Total DAM unit-hours: {dam_u_hours:.0f}")
    print(f"  Total DAM startups:   {dam_startups:.0f}")

    # ==================================================================
    # STEP 3: Notification times + gating sets
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: NOTIFICATION TIMES & GATING SETS")
    print("=" * 70)

    notification_times = assign_notification_times(data, scale=notification_scale)

    is_thermal = data.thermal_mask
    print(f"\nNotification times (scale={notification_scale:.2f}):")
    for i in range(data.n_gens):
        if is_thermal[i] and notification_times[i] > 0:
            print(f"  {data.gen_ids[i]:15s}  L_SU = {notification_times[i]:5.1f} h")

    gating_mask = compute_gating_sets(data, notification_times, t_next)

    n_gated = int(gating_mask.sum())
    n_gated_thermal = int(gating_mask[is_thermal].sum())
    n_total_decisions = int(is_thermal.sum()) * T
    print(f"\nGating summary (t_next={t_next}):")
    print(f"  Total gated (i,t) pairs:   {n_gated}")
    print(f"  Thermal gated (i,t) pairs: {n_gated_thermal} / {n_total_decisions} "
          f"({100*n_gated_thermal/max(n_total_decisions,1):.1f}%)")

    # ==================================================================
    # STEP 4: Optional line filtering
    # ==================================================================
    data_full = None
    line_mask = None
    flow_direction = None
    if monitored_lines_threshold is not None and enforce_lines:
        from compute_branch_flows import filter_monitored_lines
        data_full = data
        data, line_mask, flow_direction = filter_monitored_lines(
            data, dam_results["p"].values, monitored_lines_threshold
        )

    # ==================================================================
    # STEP 5: Uncertainty sets
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: UNCERTAINTY SETS")
    print("=" * 70)

    time_varying = False
    sqrt_Sigma = None

    robust_mask = None
    if day1_only_robust and T > 24:
        robust_mask = np.array([True] * 24 + [False] * (T - 24))
        print(f"  day1_only_robust: {int(robust_mask.sum())} robust + "
              f"{T - int(robust_mask.sum())} nominal periods")

    if uncertainty_provider_path is not None:
        print(f"\nLoading time-varying uncertainty from {uncertainty_provider_path}...")
        provider = UncertaintySetProvider.from_npz(uncertainty_provider_path)
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

        if rho is not None and rho != UNCERTAINTY_RHO:
            rho_arr = np.full(T, rho)
            print(f"  rho overridden to {rho}")

        print(f"  Sigma shape: {Sigma.shape}")
        print(f"  rho range: [{rho_arr.min():.3f}, {rho_arr.max():.3f}]")
        model_name = "LDRUC_TimeVarying"
        rho_val = rho_arr
    else:
        print("\nConstructing static uncertainty set...")
        Sigma, rho_val = build_uncertainty_set(
            data, rho=rho, wind_std_fraction=wind_std_fraction,
        )
        model_name = "LDRUC_RTS"

    # ==================================================================
    # STEP 6: Build & solve monolithic gated MISOCP
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4: MONOLITHIC GATED LD-RUC (MISOCP)")
    print("=" * 70)

    print("\nBuilding Gurobi model (gated incremental objective + robust dispatch)...")
    model, vars_dict = build_aruc_ldr_model(
        data=data,
        Sigma=Sigma,
        rho=rho_val,
        rho_lines_frac=rho_lines_frac,
        sqrt_Sigma=sqrt_Sigma,
        M_p=m_penalty,
        model_name=model_name,
        dam_commitment=dam_commitment,
        enforce_lines=enforce_lines,
        mip_gap=mip_gap,
        incremental_obj=True,
        dispatch_cost_scale=dispatch_cost_scale,
        robust_mask=robust_mask,
        fix_wind_z=fix_wind_z,
        worst_case_cost=worst_case_cost,
        robust_ramp=robust_ramp,
        time_limit=time_limit,
        threads=threads,
        bar_qcp_conv_tol=bar_qcp_conv_tol,
        line_mask=line_mask,
        flow_direction=flow_direction,
        gating_mask=gating_mask,
    )

    # Warm start from DAM
    dam_model = dam_outputs.get("model")
    dam_vars = dam_outputs.get("vars")
    if dam_model is not None and dam_vars is not None:
        from gurobipy import GRB as _GRB
        if dam_model.Status in [_GRB.OPTIMAL, _GRB.SUBOPTIMAL]:
            warm_start_aruc_from_dam(model, vars_dict, dam_vars, data)

    print("  Model built. Starting optimization...")
    model.optimize()

    if model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        print(f"WARNING: LD-RUC did not terminate optimally. Status: {model.Status}")
        if model.SolCount == 0:
            raise RuntimeError("No feasible LD-RUC solution found.")

    # Iterative line violation resolution (if lines were filtered)
    if data_full is not None:
        from compute_branch_flows import iterative_line_resolve
        _rmask = robust_mask if robust_mask is not None else np.ones(data.n_periods, dtype=bool)
        iterative_line_resolve(
            model, vars_dict, data, data_full,
            _rmask, sqrt_Sigma, rho_val,
            rho_lines_frac, time_varying,
        )

    ruc_results = extract_solution(data, model, vars_dict)
    print_brief_summary(ruc_results, data)
    analyze_Z_patterns(ruc_results["Z"], data)

    wall_time = _time.time() - wall_start

    # ==================================================================
    # Deviation analysis
    # ==================================================================
    dev_df = analyze_deviations(data, model, vars_dict, dam_commitment)
    print_deviation_summary(dev_df, dam_results["obj"], ruc_results["obj"])

    # Verify u_ruc >= u_dam
    u_ruc = ruc_results["u"].values
    u_dam = dam_commitment["u"]
    violations = (u_ruc < u_dam - 0.5).sum()
    if violations > 0:
        print(f"\nWARNING: {violations} violations of u_RUC >= u_DAM!")
    else:
        print("\nVerified: u_RUC >= u_DAM for all (i, t).")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("LD-RUC RESULTS SUMMARY")
    print("=" * 70)
    print(f"  DAM objective:         {dam_results['obj']:>14,.2f}")
    print(f"  LD-RUC objective:      {ruc_results['obj']:>14,.2f}")
    print(f"  DAM unit-hours:        {dam_u_hours:>10.0f}")
    print(f"  RUC unit-hours:        {u_ruc.sum():>10.0f}")
    extra_hours = int(np.maximum(u_ruc - u_dam, 0).sum())
    print(f"  Extra unit-hours:      {extra_hours:>10d}")
    print(f"  Gated (i,t) pairs:     {n_gated:>10d}")
    print(f"  Wall time:             {wall_time:>10.1f} s")

    return {
        "dam_outputs": dam_outputs,
        "dam_commitment": dam_commitment,
        "ruc_results": ruc_results,
        "deviation_summary": dev_df,
        "data": data,
        "data_full": data_full,
        "model": model,
        "vars": vars_dict,
        "notification_times": notification_times,
        "gating_mask": gating_mask,
        "Sigma": Sigma,
        "rho": rho_val,
        "rho_lines_frac": rho_lines_frac,
        "time_varying": time_varying,
        "line_mask": line_mask,
        "config": {
            "t_next": t_next,
            "notification_scale": notification_scale,
            "enforce_lines": enforce_lines,
            "rho_lines_frac": rho_lines_frac,
            "day2_interval_hours": day2_interval_hours,
            "day1_only_robust": day1_only_robust,
            "horizon_hours": horizon_hours,
            "wall_time": wall_time,
        },
    }


# ======================================================================
# Output saving
# ======================================================================


def save_outputs(outputs: Dict[str, Any], out_dir: Path) -> None:
    """Save all LD-RUC outputs to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = outputs["data"]
    ruc_results = outputs["ruc_results"]
    gen_ids = data.gen_ids
    time_labels = data.time

    # Commitment and dispatch
    ruc_results["u"].to_csv(out_dir / "commitment_u.csv")
    ruc_results["p0"].to_csv(out_dir / "dispatch_p0.csv")
    ruc_results["Z"].to_csv(out_dir / "ldr_Z.csv")

    # Gating sets
    gating_df = pd.DataFrame(
        outputs["gating_mask"].astype(int), index=gen_ids, columns=time_labels
    )
    gating_df.to_csv(out_dir / "gating_sets.csv")

    # Deviation summary
    outputs["deviation_summary"].to_csv(
        out_dir / "deviation_summary.csv", index=False
    )

    # Line margins
    margin_df = extract_line_margins(
        outputs["vars"], outputs["data"], outputs["rho"],
        outputs.get("rho_lines_frac"), outputs["time_varying"],
    )
    if margin_df is not None:
        margin_df.to_csv(out_dir / "line_margin.csv")

    # Summary JSON
    dam_obj = outputs["dam_outputs"]["results"]["obj"]
    ruc_obj = ruc_results["obj"]
    summary = {
        "dam_objective": dam_obj,
        "ruc_objective": ruc_obj,
        "dam_unit_hours": float(outputs["dam_commitment"]["u"].sum()),
        "ruc_unit_hours": float(ruc_results["u"].values.sum()),
        "extra_unit_hours": float(
            np.maximum(ruc_results["u"].values - outputs["dam_commitment"]["u"], 0).sum()
        ),
        "n_gated_decisions": int(outputs["gating_mask"].sum()),
        "config": outputs["config"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nOutputs saved to {out_dir}/")
    for fn in sorted(out_dir.iterdir()):
        print(f"  {fn.name}")


# ======================================================================
# CLI
# ======================================================================


def build_run_tag(args) -> str:
    """Build a descriptive directory name from CLI args."""
    parts = [
        f"{args.hours}h",
        f"m{args.start_month:02d}d{args.start_day:02d}",
        f"tnext{args.t_next}",
    ]
    if args.notification_scale != 1.0:
        parts.append(f"nscale{args.notification_scale}")
    if args.day2_interval > 1:
        parts.append(f"d2i{args.day2_interval}")
    if args.day1_only_robust:
        parts.append("d1rob")
    if not args.enforce_lines:
        parts.append("nolines")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Monolithic Notification-Gated LD-RUC pipeline (MISOCP)"
    )
    parser.add_argument("--hours", type=int, default=48, help="Horizon hours")
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--t-next", type=int, default=25,
                        help="First period modifiable by next process")
    parser.add_argument("--notification-scale", type=float, default=1.0,
                        help="Scale all notification times")
    parser.add_argument("--uncertainty-npz", type=str, default=None,
                        help="Path to uncertainty NPZ")
    parser.add_argument("--provider-start-idx", type=int, default=0)
    parser.add_argument("--rho", type=float, default=None, help="Override rho")
    parser.add_argument("--enforce-lines", action="store_true", default=True)
    parser.add_argument("--no-lines", dest="enforce_lines", action="store_false")
    parser.add_argument("--rho-lines-frac", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--dispatch-cost-scale", type=float, default=0.01)
    parser.add_argument("--day2-interval", type=int, default=1)
    parser.add_argument("--day1-only-robust", action="store_true", default=False)
    parser.add_argument("--fix-wind-z", action="store_true", default=False)
    parser.add_argument("--no-worst-case-cost", action="store_true", default=False)
    parser.add_argument("--robust-ramp", action="store_true", default=False)
    parser.add_argument("--include-renewables", action="store_true", default=False)
    parser.add_argument("--include-nuclear", action="store_true", default=False)
    parser.add_argument("--include-zero-marginal", action="store_true", default=None)
    parser.add_argument("--ramp-scale", type=float, default=1.0)
    parser.add_argument("--pmin-scale", type=float, default=1.0)
    parser.add_argument("--line-monitor-threshold", type=float, default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--bar-qcp-conv-tol", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory")

    args = parser.parse_args()

    start_time = pd.Timestamp(
        year=args.start_year, month=args.start_month,
        day=args.start_day, hour=0,
    )

    rho_val = args.rho if args.rho is not None else UNCERTAINTY_RHO

    outputs = run_ruc_monolithic(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start_time,
        horizon_hours=args.hours,
        t_next=args.t_next,
        notification_scale=args.notification_scale,
        m_penalty=M_PENALTY,
        rho=rho_val,
        uncertainty_provider_path=args.uncertainty_npz,
        provider_start_idx=args.provider_start_idx,
        spp_forecasts_parquet=SPP_FORECASTS_PARQUET,
        spp_start_idx=SPP_START_IDX,
        enforce_lines=args.enforce_lines,
        rho_lines_frac=args.rho_lines_frac,
        mip_gap=args.mip_gap,
        dispatch_cost_scale=args.dispatch_cost_scale,
        day2_interval_hours=args.day2_interval,
        day1_only_robust=args.day1_only_robust,
        fix_wind_z=args.fix_wind_z,
        worst_case_cost=not args.no_worst_case_cost,
        robust_ramp=args.robust_ramp,
        include_renewables=args.include_renewables,
        include_nuclear=args.include_nuclear,
        include_zero_marginal=args.include_zero_marginal,
        ramp_scale=args.ramp_scale,
        pmin_scale=args.pmin_scale,
        monitored_lines_threshold=args.line_monitor_threshold,
        time_limit=args.time_limit,
        threads=args.threads,
        bar_qcp_conv_tol=args.bar_qcp_conv_tol,
    )

    # Save outputs
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_tag = build_run_tag(args)
        out_dir = Path(__file__).resolve().parent / "ruc_outputs" / f"mono_{run_tag}"

    save_outputs(outputs, out_dir)


if __name__ == "__main__":
    main()
