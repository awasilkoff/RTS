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


def _case_label(path):
    """Derive a short case label from a relative CSV path.

    E.g. 'lines_only/aruc/commitment_u.csv' -> 'lines_only/aruc'
         'reserve_then_daruc/daruc/line_flows/line_flow_analysis.csv' -> 'reserve_then_daruc/daruc'
    """
    parts = Path(path).parts
    # Walk up from the filename, skip known leaf dirs
    skip = {"line_flows", "aruc_line_flows", "daruc_line_flows",
            "dam_line_flows", "reserve_line_flows"}
    label_parts = [p for p in parts[:-1] if p not in skip]
    return "/".join(label_parts) if label_parts else parts[0]


def build_summary_table(commit_results, line_results, day1_hours=24):
    """Build a summary DataFrame with one row per case.

    Columns: case, committed_units, binding_lines, binding_line_hours,
             top_binding_line, top_binding_hours
    """
    # Map case label -> commitment count
    commit_by_case = {}
    for path, count in commit_results.items():
        commit_by_case[_case_label(path)] = count

    # Map case label -> line binding info
    lines_by_case = {}
    for path, counts in line_results.items():
        lines_by_case[_case_label(path)] = counts

    all_cases = sorted(set(commit_by_case) | set(lines_by_case))
    rows = []
    for case in all_cases:
        n_committed = commit_by_case.get(case, "")
        line_counts = lines_by_case.get(case, {})
        n_binding = len(line_counts)
        total_hours = sum(line_counts.values()) if line_counts else 0
        if line_counts:
            top_line = max(line_counts, key=line_counts.get)
            top_hours = line_counts[top_line]
        else:
            top_line, top_hours = "", ""
        rows.append({
            "case": case,
            "committed_units": n_committed,
            "binding_lines": n_binding,
            "binding_line_hours": total_hours,
            "top_binding_line": top_line,
            "top_binding_hours": top_hours,
        })

    # Collect all unique binding line IDs across cases
    all_lines = sorted({lid for counts in lines_by_case.values() for lid in counts})
    for row in rows:
        case = row["case"]
        line_counts = lines_by_case.get(case, {})
        for lid in all_lines:
            row[f"line_{lid}_hours"] = line_counts.get(lid, 0)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize day-1 committed units and binding lines")
    parser.add_argument("directory",
                        help="Root directory to search")
    parser.add_argument("--hours", type=int, default=24,
                        help="Day-1 duration in hours (default: 24)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: <directory>/case_summary.csv)")
    args = parser.parse_args()

    commit_results = committed_units_day1(args.directory, args.hours)
    line_results = binding_lines_day1(args.directory, args.hours)

    # --- Console output ---
    print("=" * 60)
    print("COMMITTED UNITS (day 1)")
    print("=" * 60)
    if not commit_results:
        print(f"  No commitment_u.csv files found in {args.directory}")
    else:
        for path, count in commit_results.items():
            print(f"  {path}: {count} units")

    print()
    print("=" * 60)
    print("BINDING LINES (day 1)")
    print("=" * 60)
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

    # --- Summary CSV ---
    if commit_results or line_results:
        summary = build_summary_table(commit_results, line_results, args.hours)
        out_path = Path(args.out) if args.out else Path(args.directory) / "case_summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"\nSummary table saved to {out_path}")
        print()
        print(summary.to_string(index=False))
