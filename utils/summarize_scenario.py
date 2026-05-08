#!/usr/bin/env python3
"""
summarize_scenario.py

Print a concise scenario summary table for paper descriptions:
  - System load profile (peak, min, mean, total energy)
  - Wind forecast profile per farm and aggregate
  - Uncertainty set characteristics (rho, Sigma trace)
  - Wind penetration metrics

Usage:
    python summarize_scenario.py --start-month 7 --start-day 15
    python summarize_scenario.py --start-month 7 --start-day 15 --uncertainty-npz path/to/sigma_rho.npz
    python summarize_scenario.py --start-month 7 --start-day 15 --csv summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from io_rts import build_damdata_from_rts
from uncertainty_set_provider import UncertaintySetProvider
from aruc_model import align_uncertainty_to_aruc
from run_rts_aruc import (
    build_uncertainty_set,
    reshape_uncertainty_for_variable_intervals,
)


RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"
SPP_FORECASTS_PARQUET = Path(
    "uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet"
)


def summarize_scenario(
    start_time: pd.Timestamp,
    horizon_hours: int = 48,
    day2_interval_hours: int = 2,
    uncertainty_npz: str | None = None,
    provider_start_idx: int = 2448,
    rho: float = 3.0,
    include_renewables: bool = False,
    include_nuclear: bool = False,
    include_zero_marginal: bool | None = None,
) -> dict:
    """Build scenario summary dict."""

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
    )

    T = data.n_periods
    dt = data.dt                         # (T,)
    d1 = data.day1_period_mask()         # (T,) bool
    Pmax_2d = data.Pmax_2d()             # (I, T)

    # ---- Generator masks ----
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    is_thermal = np.array([gt.upper() == "THERMAL" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    thermal_idx = np.where(is_thermal)[0]
    wind_ids = [data.gen_ids[i] for i in wind_idx]

    # ---- Load (day-1 only) ----
    sys_load = data.d.sum(axis=0)        # (T,)
    load_d1 = sys_load[d1]
    dt_d1 = dt[d1]

    # ---- Wind forecasts (day-1 only) ----
    wind_pmax = Pmax_2d[wind_idx][:, d1]  # (K, T_d1)
    agg_wind = wind_pmax.sum(axis=0)      # (T_d1,)

    # ---- Thermal capacity ----
    thermal_pmax = Pmax_2d[thermal_idx][:, d1]  # (I_th, T_d1)
    agg_thermal = thermal_pmax.sum(axis=0)

    # ---- Wind nameplate (static) ----
    nameplate = np.array([float(np.max(Pmax_2d[i, :])) for i in wind_idx])

    # ---- Uncertainty set ----
    unc = {}
    if uncertainty_npz is not None:
        provider = UncertaintySetProvider.from_npz(uncertainty_npz)
        horizon = provider.get_by_indices(provider_start_idx, horizon_hours, compute_sqrt=True)
        Sigma, rho_arr, sqrt_Sigma = align_uncertainty_to_aruc(
            horizon, data, provider.get_wind_gen_ids()
        )
        if data.period_duration is not None:
            Sigma, rho_arr, sqrt_Sigma = reshape_uncertainty_for_variable_intervals(
                Sigma, rho_arr, data.period_duration, sqrt_Sigma
            )
        # Day-1 slices
        Sigma_d1 = Sigma[d1]
        rho_d1 = rho_arr[d1]

        # Per-hour worst-case total shortfall: rho * sqrt(1^T Sigma 1)
        ones = np.ones(Sigma.shape[1])
        wc_shortfall = np.array([
            rho_arr[t] * np.sqrt(ones @ Sigma[t] @ ones) for t in range(T)
        ])
        wc_d1 = wc_shortfall[d1]

        # Per-farm std dev (sqrt of diagonal)
        diag_std = np.array([np.sqrt(np.diag(Sigma_d1[t])) for t in range(int(d1.sum()))])  # (T_d1, K)

        unc = {
            "rho_min": float(rho_d1.min()),
            "rho_max": float(rho_d1.max()),
            "rho_mean": float(rho_d1.mean()),
            "wc_shortfall_min_mw": float(wc_d1.min()),
            "wc_shortfall_max_mw": float(wc_d1.max()),
            "wc_shortfall_mean_mw": float(wc_d1.mean()),
            "per_farm_std_mean": {
                wind_ids[k]: float(diag_std[:, k].mean())
                for k in range(len(wind_ids))
            },
            "per_farm_std_max": {
                wind_ids[k]: float(diag_std[:, k].max())
                for k in range(len(wind_ids))
            },
        }
    else:
        Sigma_static, rho_scalar = build_uncertainty_set(data, rho=rho)
        ones = np.ones(Sigma_static.shape[0])
        wc_total = rho_scalar * np.sqrt(ones @ Sigma_static @ ones)
        unc = {
            "rho": float(rho_scalar),
            "wc_shortfall_mw": float(wc_total),
        }

    # ---- Assemble ----
    summary = {
        "scenario": {
            "start": str(start_time),
            "horizon_hours": horizon_hours,
            "day2_interval_hours": day2_interval_hours,
            "n_periods": T,
            "n_day1_periods": int(d1.sum()),
        },
        "generators": {
            "total": data.n_gens,
            "thermal": int(is_thermal.sum()),
            "wind": int(is_wind.sum()),
        },
        "load_day1": {
            "peak_mw": float(load_d1.max()),
            "min_mw": float(load_d1.min()),
            "mean_mw": float((load_d1 * dt_d1).sum() / dt_d1.sum()),
            "total_mwh": float((load_d1 * dt_d1).sum()),
        },
        "thermal_capacity_day1": {
            "max_available_mw": float(agg_thermal.max()),
        },
        "wind_forecast_day1": {},
        "wind_aggregate_day1": {
            "peak_mw": float(agg_wind.max()),
            "min_mw": float(agg_wind.min()),
            "mean_mw": float((agg_wind * dt_d1).sum() / dt_d1.sum()),
            "total_mwh": float((agg_wind * dt_d1).sum()),
            "penetration_energy_pct": float((agg_wind * dt_d1).sum() / (load_d1 * dt_d1).sum() * 100),
            "penetration_peak_pct": float(agg_wind.max() / load_d1.max() * 100) if load_d1.max() > 0 else 0,
        },
        "uncertainty": unc,
    }

    # Per-farm details
    for k, gid in enumerate(wind_ids):
        farm = wind_pmax[k]  # (T_d1,)
        summary["wind_forecast_day1"][gid] = {
            "nameplate_mw": float(nameplate[k]),
            "peak_mw": float(farm.max()),
            "min_mw": float(farm.min()),
            "mean_mw": float((farm * dt_d1).sum() / dt_d1.sum()),
            "capacity_factor_pct": float((farm * dt_d1).sum() / (nameplate[k] * dt_d1.sum()) * 100),
        }

    return summary


def print_summary(s: dict) -> str:
    """Format summary dict as a readable table."""
    lines = []
    sc = s["scenario"]
    lines.append("=" * 65)
    lines.append(f"SCENARIO SUMMARY  —  {sc['start']}  ({sc['horizon_hours']}h horizon)")
    lines.append("=" * 65)

    g = s["generators"]
    lines.append(f"\nGenerators: {g['total']} total  ({g['thermal']} thermal, {g['wind']} wind)")

    # Load
    ld = s["load_day1"]
    lines.append(f"\nSystem Load (day 1):")
    lines.append(f"  Peak:   {ld['peak_mw']:>10,.1f} MW")
    lines.append(f"  Min:    {ld['min_mw']:>10,.1f} MW")
    lines.append(f"  Mean:   {ld['mean_mw']:>10,.1f} MW")
    lines.append(f"  Energy: {ld['total_mwh']:>10,.0f} MWh")

    # Thermal
    th = s["thermal_capacity_day1"]
    lines.append(f"\nThermal Capacity (day 1):")
    lines.append(f"  Max available: {th['max_available_mw']:>8,.1f} MW")

    # Aggregate wind
    wa = s["wind_aggregate_day1"]
    lines.append(f"\nWind Forecast — Aggregate (day 1):")
    lines.append(f"  Peak:   {wa['peak_mw']:>10,.1f} MW")
    lines.append(f"  Min:    {wa['min_mw']:>10,.1f} MW")
    lines.append(f"  Mean:   {wa['mean_mw']:>10,.1f} MW")
    lines.append(f"  Energy: {wa['total_mwh']:>10,.0f} MWh")
    lines.append(f"  Wind penetration (energy): {wa['penetration_energy_pct']:.1f}%")
    lines.append(f"  Wind penetration (peak):   {wa['penetration_peak_pct']:.1f}%")

    # Per-farm
    wf = s["wind_forecast_day1"]
    lines.append(f"\nWind Forecast — Per Farm (day 1):")
    lines.append(f"  {'Generator':<16s} {'Nameplate':>10s} {'Peak':>10s} {'Mean':>10s} {'CF':>8s}")
    lines.append(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for gid, v in wf.items():
        lines.append(
            f"  {gid:<16s} {v['nameplate_mw']:>9,.1f}  {v['peak_mw']:>9,.1f}  "
            f"{v['mean_mw']:>9,.1f}  {v['capacity_factor_pct']:>6.1f}%"
        )

    # Uncertainty
    unc = s["uncertainty"]
    if unc:
        lines.append(f"\nUncertainty Set (day 1):")
        if "rho_mean" in unc:
            lines.append(f"  rho:  [{unc['rho_min']:.3f}, {unc['rho_max']:.3f}]  mean={unc['rho_mean']:.3f}")
            lines.append(f"  Worst-case total shortfall:")
            lines.append(f"    Min:  {unc['wc_shortfall_min_mw']:>8,.1f} MW")
            lines.append(f"    Max:  {unc['wc_shortfall_max_mw']:>8,.1f} MW")
            lines.append(f"    Mean: {unc['wc_shortfall_mean_mw']:>8,.1f} MW")
            if "per_farm_std_mean" in unc:
                lines.append(f"  Per-farm mean std dev (sqrt diag Sigma):")
                for gid, val in unc["per_farm_std_mean"].items():
                    mx = unc["per_farm_std_max"][gid]
                    lines.append(f"    {gid:<16s}  mean={val:>7.1f} MW  max={mx:>7.1f} MW")
        else:
            lines.append(f"  rho (static): {unc.get('rho', '?')}")
            lines.append(f"  Worst-case total shortfall: {unc.get('wc_shortfall_mw', 0):,.1f} MW")

    lines.append("=" * 65)
    text = "\n".join(lines)
    return text


def main():
    parser = argparse.ArgumentParser(description="Summarize scenario load/wind/uncertainty for paper")
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--day2-interval", type=int, default=2)
    parser.add_argument("--uncertainty-npz", type=str, default=None)
    parser.add_argument("--provider-start", type=int, default=2448)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--include-renewables", action="store_true", default=False)
    parser.add_argument("--include-nuclear", action="store_true", default=False)
    parser.add_argument("--include-zero-marginal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--csv", type=str, default=None, help="Save flat summary to CSV")
    parser.add_argument("--json", type=str, default=None, help="Save full summary to JSON")
    args = parser.parse_args()

    start = pd.Timestamp(year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour)

    s = summarize_scenario(
        start_time=start,
        horizon_hours=args.hours,
        day2_interval_hours=args.day2_interval,
        uncertainty_npz=args.uncertainty_npz,
        provider_start_idx=args.provider_start,
        rho=args.rho,
        include_renewables=args.include_renewables,
        include_nuclear=args.include_nuclear,
        include_zero_marginal=args.include_zero_marginal,
    )

    text = print_summary(s)
    print(text)

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(s, f, indent=2)
        print(f"\nFull summary saved to {args.json}")

    if args.csv:
        # Flatten to single-row CSV for easy LaTeX/Excel import
        flat = {}
        flat["start"] = s["scenario"]["start"]
        flat["horizon_hours"] = s["scenario"]["horizon_hours"]
        flat["n_generators"] = s["generators"]["total"]
        flat["n_thermal"] = s["generators"]["thermal"]
        flat["n_wind"] = s["generators"]["wind"]
        for k, v in s["load_day1"].items():
            flat[f"load_{k}"] = v
        flat["thermal_max_available_mw"] = s["thermal_capacity_day1"]["max_available_mw"]
        for k, v in s["wind_aggregate_day1"].items():
            flat[f"wind_agg_{k}"] = v
        for gid, vals in s["wind_forecast_day1"].items():
            for k, v in vals.items():
                flat[f"wind_{gid}_{k}"] = v
        for k, v in s["uncertainty"].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"unc_{k}_{k2}"] = v2
            else:
                flat[f"unc_{k}"] = v

        pd.DataFrame([flat]).to_csv(args.csv, index=False)
        print(f"Flat summary saved to {args.csv}")


if __name__ == "__main__":
    main()
