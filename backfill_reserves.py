#!/usr/bin/env python3
"""Backfill reserve_equivalent.csv for completed ARUC/DARUC runs.

Reconstructs per-generator reserve distributions from saved artifacts
(Z_coefficients.csv, Sigma.npy, rho.npy) without re-running the optimizer.

Usage:
    # Single run directory (looks for aruc/ and daruc/ subdirectories)
    python backfill_reserves.py --case-dir comparison_outputs/some_run

    # Scan all runs under a parent directory
    python backfill_reserves.py --scan-dir sensitivity_suite

    # Dry run (show what would be backfilled without writing)
    python backfill_reserves.py --scan-dir sensitivity_suite --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from runner_utils import compute_reserve_equivalent, compute_reserve_from_uncertainty


def _load_z_from_csv(z_path: Path) -> pd.DataFrame:
    """Load Z_coefficients.csv into the MultiIndex DataFrame format expected
    by compute_reserve_equivalent."""
    df = pd.read_csv(z_path, header=[0, 1], index_col=0)
    # Parse the MultiIndex columns: (time_str, k_str) -> (Timestamp, int)
    new_cols = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(t), int(k)) for t, k in df.columns],
        names=["time", "k"],
    )
    df.columns = new_cols
    return df


def _infer_gen_types(gen_ids: list[str]) -> list[str]:
    """Infer generator type from generator ID string."""
    types = []
    for gid in gen_ids:
        upper = gid.upper()
        if "WIND" in upper:
            types.append("WIND")
        elif "PV" in upper or "SOLAR" in upper or "RTPV" in upper:
            types.append("SOLAR")
        elif "HYDRO" in upper or "ROR" in upper:
            types.append("HYDRO")
        elif "NUCLEAR" in upper:
            types.append("NUCLEAR")
        else:
            types.append("THERMAL")
    return types


def backfill_one(subdir: Path, dry_run: bool = False) -> bool:
    """Backfill reserve_equivalent.csv for a single aruc/ or daruc/ directory.

    Returns True if backfill was performed (or would have been in dry-run).
    """
    z_path = subdir / "Z_coefficients.csv"
    sigma_path = subdir / "Sigma.npy"
    rho_path = subdir / "rho.npy"
    reserve_path = subdir / "reserve_equivalent.csv"

    # Check required artifacts exist
    missing = [p.name for p in (z_path, sigma_path, rho_path) if not p.exists()]
    if missing:
        return False

    if reserve_path.exists():
        print(f"  SKIP {subdir} — reserve_equivalent.csv already exists")
        return False

    if dry_run:
        print(f"  WOULD backfill {subdir}")
        return True

    # Load artifacts
    Z_df = _load_z_from_csv(z_path)
    Sigma = np.load(sigma_path)
    rho = np.load(rho_path)

    gen_ids = list(Z_df.index)
    time_labels = sorted(set(t for t, _ in Z_df.columns))
    gen_types = _infer_gen_types(gen_ids)
    T = len(time_labels)

    # Build a minimal data-like namespace for compute_reserve_equivalent
    class _MinimalData:
        pass

    data = _MinimalData()
    data.gen_ids = gen_ids
    data.gen_type = gen_types
    data.time = time_labels

    results = {"Z": Z_df}

    reserve_df, stats = compute_reserve_equivalent(results, data, Sigma, rho)
    reserve_df.to_csv(reserve_path)
    print(f"  DONE {subdir} — thermal reserve: "
          f"{stats['reserve_equivalent']['thermal_total_mean_mw']:.1f} MW avg")

    # Also compute and save reserve_requirement.npy if missing
    req_path = subdir / "reserve_requirement.npy"
    if not req_path.exists():
        R = compute_reserve_from_uncertainty(Sigma, rho, T=T)
        np.save(req_path, R)
        print(f"       + saved reserve_requirement.npy (max {R.max():.1f} MW)")

    return True


def find_robust_dirs(root: Path) -> list[Path]:
    """Recursively find aruc/ and daruc/ directories with Z data."""
    dirs = []
    for z_file in root.rglob("Z_coefficients.csv"):
        subdir = z_file.parent
        if subdir.name in ("aruc", "daruc"):
            dirs.append(subdir)
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill reserve_equivalent.csv from saved Z/Sigma/rho")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-dir", type=str,
                       help="Single run directory (parent of aruc/ and daruc/)")
    group.add_argument("--scan-dir", type=str,
                       help="Recursively scan for aruc/daruc subdirectories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be backfilled without writing")
    args = parser.parse_args()

    if args.case_dir:
        case_dir = Path(args.case_dir)
        dirs = [case_dir / "aruc", case_dir / "daruc"]
        dirs = [d for d in dirs if d.exists()]
    else:
        dirs = find_robust_dirs(Path(args.scan_dir))

    if not dirs:
        print("No aruc/daruc directories found.")
        return

    print(f"Found {len(dirs)} directories to check.\n")
    count = 0
    for d in dirs:
        if backfill_one(d, dry_run=args.dry_run):
            count += 1

    action = "would backfill" if args.dry_run else "backfilled"
    print(f"\n{action} {count} directories.")


if __name__ == "__main__":
    main()
