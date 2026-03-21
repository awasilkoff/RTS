"""Count unique generators committed in day 1 across commitment_u.csv files.

Usage:
    python count_committed_units.py sensitivity_suite/rho99_48h_m07d15
    python count_committed_units.py comparison_outputs/some_run --hours 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def committed_units_day1(directory, day1_hours=24):
    """Return {relative_path: n_committed} for each commitment_u.csv found."""
    directory = Path(directory)
    results = {}
    for csv_path in sorted(directory.rglob("commitment_u.csv")):
        u = pd.read_csv(csv_path, index_col=0)
        try:
            timestamps = pd.to_datetime(u.columns)
            t0 = timestamps[0]
            day1_cols = [c for c, ts in zip(u.columns, timestamps)
                         if ts < t0 + pd.Timedelta(hours=day1_hours)]
        except Exception:
            day1_cols = list(u.columns[:day1_hours])
        committed = (u[day1_cols].values >= 0.5).any(axis=1).sum()
        results[str(csv_path.relative_to(directory))] = int(committed)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count unique committed units in day 1")
    parser.add_argument("directory", help="Root directory to search for commitment_u.csv files")
    parser.add_argument("--hours", type=int, default=24, help="Day-1 duration in hours (default: 24)")
    args = parser.parse_args()

    results = committed_units_day1(args.directory, args.hours)
    if not results:
        print(f"No commitment_u.csv files found in {args.directory}")
    else:
        for path, count in results.items():
            print(f"{path}: {count} units")
