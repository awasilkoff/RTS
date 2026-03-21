"""Rebuild deviation_summary.csv for completed case directories.

Globs recursively for directories containing both dam_reserve/commitment_u.csv
and daruc/commitment_u.csv, then rebuilds the deviation summary with enriched
generator metadata (fuel, category, Pmax, Pmin from gen.csv).

Usage:
    # Single case directory
    python rebuild_deviation_summaries.py comparison_outputs/some_run

    # Glob over all cases under a parent directory
    python rebuild_deviation_summaries.py comparison_outputs/

    # Multiple directories
    python rebuild_deviation_summaries.py comparison_outputs/ alpha_sweep/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from runner_utils import rebuild_deviation_summary


def find_case_dirs(root):
    """Find directories that contain both dam_reserve/ and daruc/ commitment CSVs."""
    root = Path(root)
    # Find all daruc/commitment_u.csv, then check for matching dam_reserve/
    cases = []
    for daruc_csv in sorted(root.rglob("daruc/commitment_u.csv")):
        case_dir = daruc_csv.parent.parent
        dam_csv = case_dir / "dam_reserve" / "commitment_u.csv"
        if dam_csv.exists():
            cases.append(case_dir)
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild deviation summaries for completed cases")
    parser.add_argument("directories", nargs="+",
                        help="Case directories or parent directories to search")
    parser.add_argument("--dry-run", action="store_true",
                        help="List case directories without rebuilding")
    args = parser.parse_args()

    all_cases = []
    for d in args.directories:
        d = Path(d)
        # Check if this is itself a case directory
        if (d / "daruc" / "commitment_u.csv").exists() and \
           (d / "dam_reserve" / "commitment_u.csv").exists():
            all_cases.append(d)
        else:
            all_cases.extend(find_case_dirs(d))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in all_cases:
        resolved = c.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(c)
    all_cases = unique

    if not all_cases:
        print("No case directories found (need dam_reserve/ + daruc/ with commitment_u.csv)")
        return

    print(f"Found {len(all_cases)} case(s):\n")
    for c in all_cases:
        print(f"  {c}")

    if args.dry_run:
        return

    print()
    for i, case_dir in enumerate(all_cases, 1):
        print(f"[{i}/{len(all_cases)}] {case_dir}")
        try:
            rebuild_deviation_summary(case_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print(f"Done. Rebuilt {len(all_cases)} deviation summaries.")


if __name__ == "__main__":
    main()
