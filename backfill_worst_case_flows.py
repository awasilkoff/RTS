#!/usr/bin/env python3
"""Backfill worst-case total-shortfall line flows for completed runs.

Computes PTDF-based line flows under the worst-case total wind shortfall
scenario, using saved artifacts (dispatch, Z, Sigma, rho, reserves).

Requires rebuilding DAMData from RTS source data for PTDF/load/bus mapping.

Usage:
    # Single case directory
    python backfill_worst_case_flows.py --case-dir sensitivity_suite/rho99_48h_m07d15/reserve_then_daruc

    # Recursive scan
    python backfill_worst_case_flows.py --scan-dir sensitivity_suite

    # Override run parameters if auto-detection fails
    python backfill_worst_case_flows.py --case-dir path/to/run --start-month 7 --start-day 15 --hours 48
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_rts import build_damdata_from_rts
from runner_utils import (
    compute_worst_case_total_shortfall_flows,
    save_worst_case_flow_analysis,
)
from backfill_reserves import _load_z_from_csv

RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"


def _load_run_params(case_dir: Path) -> dict:
    """Extract run parameters from summary.json files."""
    params = {"hours": 48, "start_month": 7, "start_day": 15,
              "day2_interval": 2, "enforce_lines": False}

    for subdir in ["daruc", "aruc", "dam_reserve"]:
        summary_path = case_dir / subdir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            if "hours" in s:
                params["hours"] = s["hours"]
            if "start_time" in s:
                ts = pd.Timestamp(s["start_time"])
                params["start_month"] = ts.month
                params["start_day"] = ts.day
            if "enforce_lines" in s:
                params["enforce_lines"] = s["enforce_lines"]
            break
    return params


def _load_z_array(z_path: Path, gen_ids: list[str], time_labels) -> np.ndarray:
    """Load Z_coefficients.csv into (I, T, K) array aligned to gen_ids/time."""
    Z_df = _load_z_from_csv(z_path)
    I = len(gen_ids)
    unique_times = sorted(set(t for t, _ in Z_df.columns))
    K = max(k for _, k in Z_df.columns) + 1
    T = len(time_labels)

    Z_arr = np.zeros((I, T, K))
    for i, gid in enumerate(gen_ids):
        if gid not in Z_df.index:
            continue
        for t_idx, t_label in enumerate(time_labels):
            for k in range(K):
                col = (t_label, k)
                if col in Z_df.columns:
                    Z_arr[i, t_idx, k] = Z_df.loc[gid, col]
    return Z_arr


def backfill_case(case_dir: Path, params: dict, force: bool = False) -> int:
    """Backfill worst-case flow analysis for one case directory.

    Returns number of sub-analyses written.
    """
    case_dir = Path(case_dir)
    count = 0

    # Build DAMData
    start = pd.Timestamp(year=2020, month=params["start_month"],
                         day=params["start_day"])
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR, ts_dir=TS_DIR,
        start_time=start, horizon_hours=params["hours"],
        day2_interval_hours=params.get("day2_interval", 2),
    )
    gen_ids = list(data.gen_ids)
    time_labels = list(data.time)

    # --- DARUC ---
    daruc_dir = case_dir / "daruc"
    daruc_out = daruc_dir / "worst_case_flow_analysis_daruc.csv"
    if daruc_dir.exists() and (force or not daruc_out.exists()):
        z_path = daruc_dir / "Z_coefficients.csv"
        sigma_path = daruc_dir / "Sigma.npy"
        rho_path = daruc_dir / "rho.npy"
        p0_path = daruc_dir / "dispatch_p0.csv"

        if all(p.exists() for p in [z_path, sigma_path, rho_path, p0_path]):
            print(f"\n  Computing DARUC worst-case flows: {daruc_dir}")
            Sigma = np.load(sigma_path)
            rho = np.load(rho_path)
            p0_df = pd.read_csv(p0_path, index_col=0)
            p0_arr = p0_df.reindex(gen_ids).fillna(0).values.astype(float)
            Z_arr = _load_z_array(z_path, gen_ids, time_labels)

            wc = compute_worst_case_total_shortfall_flows(
                data, p0_arr, Sigma, rho, Z_arr=Z_arr)
            save_worst_case_flow_analysis(data, wc, "daruc", daruc_dir)
            count += 1
        else:
            missing = [p.name for p in [z_path, sigma_path, rho_path, p0_path]
                       if not p.exists()]
            print(f"  SKIP DARUC — missing: {missing}")

    # --- ARUC ---
    aruc_dir = case_dir / "aruc"
    aruc_out = aruc_dir / "worst_case_flow_analysis_aruc.csv"
    if aruc_dir.exists() and (force or not aruc_out.exists()):
        z_path = aruc_dir / "Z_coefficients.csv"
        sigma_path = aruc_dir / "Sigma.npy"
        rho_path = aruc_dir / "rho.npy"
        p0_path = aruc_dir / "dispatch_p0.csv"

        if all(p.exists() for p in [z_path, sigma_path, rho_path, p0_path]):
            print(f"\n  Computing ARUC worst-case flows: {aruc_dir}")
            Sigma = np.load(sigma_path)
            rho = np.load(rho_path)
            p0_df = pd.read_csv(p0_path, index_col=0)
            p0_arr = p0_df.reindex(gen_ids).fillna(0).values.astype(float)
            Z_arr = _load_z_array(z_path, gen_ids, time_labels)

            wc = compute_worst_case_total_shortfall_flows(
                data, p0_arr, Sigma, rho, Z_arr=Z_arr)
            save_worst_case_flow_analysis(data, wc, "aruc", aruc_dir)
            count += 1

    # --- DAM+Reserve ---
    dam_dir = case_dir / "dam_reserve"
    dam_out = dam_dir / "worst_case_flow_analysis_dam_reserve.csv"
    if dam_dir.exists() and (force or not dam_out.exists()):
        p0_path = dam_dir / "dispatch_p0.csv"
        r_path = dam_dir / "reserve_distribution.csv"

        # Need Sigma/rho from a sibling robust dir
        sigma_path = rho_path = None
        for sibling in ["daruc", "aruc"]:
            sp = case_dir / sibling / "Sigma.npy"
            rp = case_dir / sibling / "rho.npy"
            if sp.exists() and rp.exists():
                sigma_path, rho_path = sp, rp
                break

        if all(p is not None and p.exists()
               for p in [p0_path, r_path, sigma_path, rho_path]):
            print(f"\n  Computing DAM+Reserve worst-case flows: {dam_dir}")
            Sigma = np.load(sigma_path)
            rho = np.load(rho_path)
            p0_df = pd.read_csv(p0_path, index_col=0)
            p0_arr = p0_df.reindex(gen_ids).fillna(0).values.astype(float)

            # Reserve array: thermal-only in CSV, expand to (I, T)
            r_df = pd.read_csv(r_path, index_col=0)
            r_arr = np.zeros_like(p0_arr)
            for gid in r_df.index:
                if gid in gen_ids:
                    idx = gen_ids.index(gid)
                    r_arr[idx, :] = r_df.loc[gid].values[:p0_arr.shape[1]]

            wc = compute_worst_case_total_shortfall_flows(
                data, p0_arr, Sigma, rho, r_arr=r_arr)
            save_worst_case_flow_analysis(data, wc, "dam_reserve", dam_dir)
            count += 1
        else:
            missing = []
            for name, p in [("dispatch", p0_path), ("reserves", r_path),
                            ("Sigma", sigma_path), ("rho", rho_path)]:
                if p is None or not p.exists():
                    missing.append(name)
            print(f"  SKIP DAM+Reserve — missing: {missing}")

    return count


def find_case_dirs(root: Path) -> list[Path]:
    """Find case directories that contain daruc/, aruc/, or dam_reserve/."""
    dirs = set()
    for subdir_name in ["daruc", "aruc", "dam_reserve"]:
        for match in root.rglob(subdir_name):
            if match.is_dir():
                dirs.add(match.parent)
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill worst-case total-shortfall line flows")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-dir", type=str)
    group.add_argument("--scan-dir", type=str)
    parser.add_argument("--start-month", type=int, default=None)
    parser.add_argument("--start-day", type=int, default=None)
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--day2-interval", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing worst-case flow CSVs")
    args = parser.parse_args()

    if args.case_dir:
        case_dirs = [Path(args.case_dir)]
    else:
        case_dirs = find_case_dirs(Path(args.scan_dir))

    print(f"Found {len(case_dirs)} case directories.\n")
    total = 0

    for case_dir in case_dirs:
        print(f"{'='*60}\n{case_dir}")
        params = _load_run_params(case_dir)
        # CLI overrides
        if args.start_month is not None:
            params["start_month"] = args.start_month
        if args.start_day is not None:
            params["start_day"] = args.start_day
        if args.hours is not None:
            params["hours"] = args.hours
        if args.day2_interval is not None:
            params["day2_interval"] = args.day2_interval

        print(f"  Params: month={params['start_month']}, day={params['start_day']}, "
              f"hours={params['hours']}")
        total += backfill_case(case_dir, params, force=args.force)

    print(f"\nBackfilled {total} worst-case flow analyses.")


if __name__ == "__main__":
    main()
