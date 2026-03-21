"""Rebuild deviation_summary.csv for completed case directories.

Globs recursively for directories containing daruc/commitment_u.csv plus a
DAM baseline (daruc/dam_commitment_u.csv, dam_reserve/commitment_u.csv, or
dam/commitment_u.csv), then rebuilds the deviation summary with enriched
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

# DAM baseline locations to check (relative to case_dir)
_DAM_CANDIDATES = [
    Path("daruc") / "dam_commitment_u.csv",
    Path("dam_reserve") / "commitment_u.csv",
    Path("dam") / "commitment_u.csv",
]


def _has_dam_baseline(case_dir):
    """Check if a case directory has any recognized DAM baseline commitment."""
    return any((case_dir / c).exists() for c in _DAM_CANDIDATES)


def find_case_dirs(root):
    """Find directories that contain daruc/commitment_u.csv and a DAM baseline."""
    root = Path(root)
    cases = []
    for daruc_csv in sorted(root.rglob("daruc/commitment_u.csv")):
        case_dir = daruc_csv.parent.parent
        if _has_dam_baseline(case_dir):
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
        if (d / "daruc" / "commitment_u.csv").exists() and _has_dam_baseline(d):
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
        print("No case directories found (need daruc/commitment_u.csv + DAM baseline)")
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
