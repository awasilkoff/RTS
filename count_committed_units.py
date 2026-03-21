"""Summarize commitment and binding-line counts across cases in a directory.

Usage:
    python count_committed_units.py sensitivity_suite/rho99_48h_m07d15
    python count_committed_units.py comparison_outputs/some_run --hours 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _day1_filter(timestamps, day1_hours):
    """Return boolean mask selecting periods within the first day1_hours."""
    t0 = timestamps[0]
    return timestamps < t0 + pd.Timedelta(hours=day1_hours)


def committed_units_day1(directory, day1_hours=24):
    """Return {relative_path: n_committed} for each commitment_u.csv found."""
    directory = Path(directory)
    results = {}
    for csv_path in sorted(directory.rglob("commitment_u.csv")):
        u = pd.read_csv(csv_path, index_col=0)
        try:
            timestamps = pd.to_datetime(u.columns)
            day1_mask = _day1_filter(timestamps, day1_hours)
            day1_cols = [c for c, m in zip(u.columns, day1_mask) if m]
        except Exception:
            day1_cols = list(u.columns[:day1_hours])
        committed = (u[day1_cols].values >= 0.5).any(axis=1).sum()
        results[str(csv_path.relative_to(directory))] = int(committed)
    return results


def binding_lines_day1(directory, day1_hours=24):
    """Return {relative_path: {line_id: n_binding_hours}} for each line_flow_analysis*.csv.

    Only day-1 periods are counted.  A line-period pair counts as binding
    when the ``binding`` column is True.
    """
    directory = Path(directory)
    results = {}
    for csv_path in sorted(directory.rglob("line_flow_analysis*.csv")):
        df = pd.read_csv(csv_path)
        if "binding" not in df.columns or "period" not in df.columns:
            continue
        try:
            df["_ts"] = pd.to_datetime(df["period"])
            day1_mask = _day1_filter(df["_ts"], day1_hours)
            df = df[day1_mask]
        except Exception:
            pass
        binding = df[df["binding"] == True]  # noqa: E712
        counts = binding.groupby("line").size().sort_values(ascending=False)
        results[str(csv_path.relative_to(directory))] = counts.to_dict()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize day-1 committed units and binding lines")
    parser.add_argument("directory",
                        help="Root directory to search")
    parser.add_argument("--hours", type=int, default=24,
                        help="Day-1 duration in hours (default: 24)")
    args = parser.parse_args()

    # --- Committed units ---
    print("=" * 60)
    print("COMMITTED UNITS (day 1)")
    print("=" * 60)
    commit_results = committed_units_day1(args.directory, args.hours)
    if not commit_results:
        print(f"  No commitment_u.csv files found in {args.directory}")
    else:
        for path, count in commit_results.items():
            print(f"  {path}: {count} units")

    # --- Binding lines ---
    print()
    print("=" * 60)
    print("BINDING LINES (day 1)")
    print("=" * 60)
    line_results = binding_lines_day1(args.directory, args.hours)
    if not line_results:
        print(f"  No line_flow_analysis*.csv files found in {args.directory}")
    else:
        for path, counts in line_results.items():
            print(f"\n  {path}:")
            if not counts:
                print("    (no binding lines)")
            else:
                print(f"    {len(counts)} lines binding, "
                      f"{sum(counts.values())} total (line,hour) pairs")
                for line_id, n in counts.items():
                    print(f"      {line_id}: {n}/{args.hours}h")
