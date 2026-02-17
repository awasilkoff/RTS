"""
run_ruc.py — End-to-end Notification-Gated LD-RUC pipeline.

Pipeline:
  1. Build DAMData from RTS-GMLC
  2. Solve deterministic DAM (market solution)
  3. Load uncertainty sets (Sigma, rho) from NPZ provider
  4. Assign notification times from generator fuel/type
  5. Compute gating sets
  6. Run CCG loop (Phase 1 + Phase 2)
  7. Save results and artifacts
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
from io_rts import build_damdata_from_rts
from run_rts_dam import run_rts_dam, extract_solution as extract_dam_solution
from run_rts_daruc import extract_dam_commitment
from aruc_model import align_uncertainty_to_aruc
from run_rts_aruc import (
    build_uncertainty_set,
    reshape_uncertainty_for_variable_intervals,
)
from uncertainty_set_provider import UncertaintySetProvider

from ruc_model import (
    compute_gating_sets,
    build_phase1_model,
    build_phase2_model,
    solve_ruc_ccg,
)


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
# Notification time assignment
# ======================================================================


def assign_notification_times(data: DAMData, scale: float = 1.0) -> np.ndarray:
    """
    Assign notification lead times L_i^SU for each generator.

    Uses generator metadata (Unit Type, Fuel, PMax MW) from data.gens_df
    to assign physically-motivated notification times.

    Parameters
    ----------
    data : DAMData
        UC data with gens_df available.
    scale : float
        Multiplier for all notification times (for sensitivity sweeps).

    Returns
    -------
    notification_times : np.ndarray
        (I,) array of notification lead times in hours.
    """
    I = data.n_gens
    notification_times = np.zeros(I)

    if data.gens_df is None:
        raise ValueError(
            "data.gens_df is required for notification time assignment. "
            "Ensure build_damdata_from_rts preserves gens_df."
        )

    gens_lookup = data.gens_df.set_index("GEN UID")

    for i, gen_id in enumerate(data.gen_ids):
        gen_type = data.gen_type[i]

        # Non-thermal generators: zero notification time
        if gen_type in ("WIND", "SOLAR", "HYDRO"):
            notification_times[i] = 0.0
            continue

        # For thermal, use Unit Type + Fuel + PMax to distinguish
        if gen_id not in gens_lookup.index:
            notification_times[i] = 8.0  # default for unknown thermal
            continue

        row = gens_lookup.loc[gen_id]
        unit_type = str(row.get("Unit Type", "")).upper()
        fuel = str(row.get("Fuel", "")).upper()
        pmax = float(row.get("PMax MW", 0))

        if unit_type == "NUCLEAR" or fuel == "NUCLEAR":
            notification_times[i] = 48.0
        elif unit_type == "STEAM" and fuel == "COAL":
            if pmax >= 300:
                notification_times[i] = 48.0
            elif pmax >= 100:
                notification_times[i] = 48.0
            else:
                notification_times[i] = 16.0
        elif unit_type == "STEAM" and fuel == "OIL":
            notification_times[i] = 16.0
        elif unit_type == "CC" or (unit_type == "STEAM" and "GAS" in fuel):
            notification_times[i] = 8.0
        elif unit_type == "CT":
            notification_times[i] = 2.0
        elif unit_type == "CSP":
            notification_times[i] = 0.0
        else:
            # Default for other thermal
            notification_times[i] = 8.0

    notification_times *= scale
    return notification_times


# ======================================================================
# Main pipeline
# ======================================================================


def run_ruc(
    source_dir: Path = SOURCE_DIR,
    ts_dir: Path = TS_DIR,
    start_time: pd.Timestamp = START_TIME,
    horizon_hours: int = HORIZON_HOURS,
    t_next: int = 25,
    notification_scale: float = 1.0,
    uncertainty_provider_path: Optional[Union[Path, str]] = None,
    provider_start_idx: int = 0,
    rho_override: Optional[float] = None,
    spp_forecasts_parquet: Optional[Path] = SPP_FORECASTS_PARQUET,
    spp_start_idx: int = SPP_START_IDX,
    enforce_lines: bool = True,
    rho_lines_frac: Optional[float] = None,
    max_ccg_iterations: int = 10,
    ccg_gap_tolerance: float = 1.0,
    M_p: float = M_PENALTY,
    day2_interval_hours: int = 1,
    day1_only_robust: bool = False,
    wind_std_fraction: float = 0.15,
) -> Dict[str, Any]:
    """
    Full LD-RUC pipeline: DAM -> gating -> CCG (Phase 1 + Phase 2).

    Returns dict with all results, models, and metadata.
    """
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
        m_penalty=M_p,
        spp_forecasts_parquet=spp_forecasts_parquet,
        spp_start_idx=spp_start_idx,
        enforce_lines=enforce_lines,
        day2_interval_hours=day2_interval_hours,
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
    # STEP 3: Load uncertainty sets
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: UNCERTAINTY SETS")
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

        if rho_override is not None:
            rho_arr = np.full(T, rho_override)
            print(f"  rho overridden to {rho_override}")

        print(f"  Sigma shape: {Sigma.shape}")
        print(f"  rho range: [{rho_arr.min():.3f}, {rho_arr.max():.3f}]")
    else:
        print("\nConstructing static uncertainty set...")
        rho_val = rho_override if rho_override is not None else UNCERTAINTY_RHO
        Sigma, rho_arr = build_uncertainty_set(
            data, rho=rho_val, wind_std_fraction=wind_std_fraction,
        )

    # ==================================================================
    # STEP 4: Notification times + gating sets
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: NOTIFICATION TIMES & GATING SETS")
    print("=" * 70)

    notification_times = assign_notification_times(data, scale=notification_scale)

    # Print summary
    is_thermal = data.thermal_mask
    thermal_ids = [data.gen_ids[i] for i in range(data.n_gens) if is_thermal[i]]
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
    # STEP 5: CCG loop
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4: LD-RUC CCG SOLVE")
    print("=" * 70)

    ccg_results = solve_ruc_ccg(
        data=data,
        dam_commitment=dam_commitment,
        gating_mask=gating_mask,
        Sigma=Sigma,
        rho=rho_arr,
        sqrt_Sigma=sqrt_Sigma,
        enforce_lines=enforce_lines,
        rho_lines_frac=rho_lines_frac,
        robust_mask=robust_mask,
        max_iterations=max_ccg_iterations,
        gap_tolerance=ccg_gap_tolerance,
        M_p=M_p,
    )

    wall_time = _time.time() - wall_start

    # ==================================================================
    # Summary
    # ==================================================================
    commitment = ccg_results["commitment"]
    u_ruc = commitment["u"]
    u_dam = dam_commitment["u"]
    incremental = u_ruc - u_dam

    n_extra_hours = int(np.maximum(incremental, 0).sum())
    n_extra_units = int((incremental.sum(axis=1) > 0.5).sum())

    print("\n" + "=" * 70)
    print("LD-RUC RESULTS SUMMARY")
    print("=" * 70)
    print(f"  DAM objective:         {dam_results['obj']:>14,.2f}")
    if "obj" in ccg_results["phase1_results"]:
        print(f"  Phase 1 objective:     {ccg_results['phase1_results']['obj']:>14,.2f}")
    print(f"  DAM unit-hours:        {dam_u_hours:>10.0f}")
    print(f"  RUC unit-hours:        {u_ruc.sum():>10.0f}")
    print(f"  Extra unit-hours:      {n_extra_hours:>10d}")
    print(f"  Extra units committed: {n_extra_units:>10d}")
    print(f"  CCG iterations:        {ccg_results['ccg_iterations']:>10d}")
    print(f"  Robust sufficient:     {'YES' if ccg_results['robust_sufficient'] else 'NO':>10s}")
    if ccg_results["gap_history"]:
        print(f"  Final gap (MW):        {ccg_results['gap_history'][-1]:>10.4f}")
    print(f"  Wall time:             {wall_time:>10.1f} s")

    # Deviation details
    dev_rows = []
    for i in range(data.n_gens):
        extra = int(np.maximum(incremental[i, :], 0).sum())
        if extra > 0:
            periods = [t for t in range(T) if incremental[i, t] > 0.5]
            dev_rows.append({
                "gen_id": data.gen_ids[i],
                "gen_type": data.gen_type[i],
                "L_SU": notification_times[i],
                "extra_hours": extra,
                "dam_hours": int(u_dam[i, :].sum()),
                "ruc_hours": int(u_ruc[i, :].sum()),
                "periods_added": str(periods),
            })

    dev_df = pd.DataFrame(dev_rows)
    if not dev_df.empty:
        print(f"\nIncremental commitments ({len(dev_df)} generators):")
        print(dev_df.to_string(index=False))
    else:
        print("\nNo incremental commitments (RUC = DAM).")

    return {
        "dam_outputs": dam_outputs,
        "dam_commitment": dam_commitment,
        "data": data,
        "notification_times": notification_times,
        "gating_mask": gating_mask,
        "ccg_results": ccg_results,
        "deviation_summary": dev_df,
        "Sigma": Sigma,
        "rho": rho_arr,
        "time_varying": time_varying,
        "config": {
            "t_next": t_next,
            "notification_scale": notification_scale,
            "max_ccg_iterations": max_ccg_iterations,
            "ccg_gap_tolerance": ccg_gap_tolerance,
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
    ccg = outputs["ccg_results"]
    commitment = ccg["commitment"]
    gen_ids = data.gen_ids
    time_labels = data.time

    I = data.n_gens
    T = data.n_periods

    # Commitment
    u_df = pd.DataFrame(commitment["u"], index=gen_ids, columns=time_labels)
    u_df.to_csv(out_dir / "commitment_u.csv")

    # Gating sets
    gating_df = pd.DataFrame(
        outputs["gating_mask"].astype(int), index=gen_ids, columns=time_labels
    )
    gating_df.to_csv(out_dir / "gating_sets.csv")

    # Deviation summary
    outputs["deviation_summary"].to_csv(
        out_dir / "deviation_summary.csv", index=False
    )

    # Phase 2 dispatch
    if "p0" in ccg["phase2_results"]:
        p0_df = pd.DataFrame(
            ccg["phase2_results"]["p0"], index=gen_ids, columns=time_labels
        )
        p0_df.to_csv(out_dir / "dispatch_p0.csv")

    # Phase 2 Z coefficients
    if "Z" in ccg["phase2_results"]:
        Z_arr = ccg["phase2_results"]["Z"]
        K = Z_arr.shape[2]
        Z_cols = pd.MultiIndex.from_product(
            [time_labels, range(K)], names=["time", "k"]
        )
        Z_flat = Z_arr.reshape(I, T * K)
        Z_df = pd.DataFrame(Z_flat, index=gen_ids, columns=Z_cols)
        Z_df.to_csv(out_dir / "ldr_Z.csv")

    # CCG history
    ccg_history = {
        "iterations": ccg["ccg_iterations"],
        "gap_history": ccg["gap_history"],
        "robust_sufficient": ccg["robust_sufficient"],
        "n_scenarios": len(ccg["scenarios"]),
        "timings": {
            "phase1": ccg["timings"]["phase1"],
            "phase2": ccg["timings"]["phase2"],
        },
    }
    with open(out_dir / "ccg_history.json", "w") as f:
        json.dump(ccg_history, f, indent=2, default=str)

    # Summary JSON
    dam_obj = outputs["dam_outputs"]["results"]["obj"]
    p1_obj = ccg["phase1_results"].get("obj", None)
    summary = {
        "dam_objective": dam_obj,
        "phase1_objective": p1_obj,
        "dam_unit_hours": float(outputs["dam_commitment"]["u"].sum()),
        "ruc_unit_hours": float(commitment["u"].sum()),
        "extra_unit_hours": float(
            np.maximum(commitment["u"] - outputs["dam_commitment"]["u"], 0).sum()
        ),
        "ccg_iterations": ccg["ccg_iterations"],
        "robust_sufficient": ccg["robust_sufficient"],
        "final_gap_mw": ccg["gap_history"][-1] if ccg["gap_history"] else None,
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
        description="Notification-Gated LD-RUC pipeline"
    )
    parser.add_argument("--hours", type=int, default=48, help="Horizon hours")
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--t-next", type=int, default=25, help="First period modifiable by next process")
    parser.add_argument("--notification-scale", type=float, default=1.0, help="Scale all notification times")
    parser.add_argument("--uncertainty-npz", type=str, default=None, help="Path to uncertainty NPZ")
    parser.add_argument("--provider-start-idx", type=int, default=0)
    parser.add_argument("--rho", type=float, default=None, help="Override rho")
    parser.add_argument("--enforce-lines", action="store_true", default=True)
    parser.add_argument("--no-lines", dest="enforce_lines", action="store_false")
    parser.add_argument("--rho-lines-frac", type=float, default=None)
    parser.add_argument("--max-ccg-iter", type=int, default=10)
    parser.add_argument("--ccg-tol", type=float, default=1.0, help="CCG gap tolerance (MW)")
    parser.add_argument("--day2-interval", type=int, default=1)
    parser.add_argument("--day1-only-robust", action="store_true", default=False)
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    start_time = pd.Timestamp(
        year=args.start_year, month=args.start_month,
        day=args.start_day, hour=0,
    )

    outputs = run_ruc(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start_time,
        horizon_hours=args.hours,
        t_next=args.t_next,
        notification_scale=args.notification_scale,
        uncertainty_provider_path=args.uncertainty_npz,
        provider_start_idx=args.provider_start_idx,
        rho_override=args.rho,
        spp_forecasts_parquet=SPP_FORECASTS_PARQUET,
        spp_start_idx=SPP_START_IDX,
        enforce_lines=args.enforce_lines,
        rho_lines_frac=args.rho_lines_frac,
        max_ccg_iterations=args.max_ccg_iter,
        ccg_gap_tolerance=args.ccg_tol,
        M_p=M_PENALTY,
        day2_interval_hours=args.day2_interval,
        day1_only_robust=args.day1_only_robust,
    )

    # Save outputs
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_tag = build_run_tag(args)
        out_dir = Path(__file__).resolve().parent / "ruc_outputs" / run_tag

    save_outputs(outputs, out_dir)


if __name__ == "__main__":
    main()
