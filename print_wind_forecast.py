"""print_wind_forecast.py

Print the mean wind forecast (mu) from an uncertainty set NPZ — this is the
center of the ellipsoidal uncertainty set and the forecast used in the DAM.

Mirrors print_worst_case_wind.py but prints mu directly instead of the
worst-case realization.

Usage:
    python print_wind_forecast.py <npz_path> [--start 2448] [--hours 48]
    python print_wind_forecast.py uncertainty_sets_refactored/data/uncertainty_sets.npz --start 2448 --hours 48
    python print_wind_forecast.py alpha_sweep/.../sigma_rho.npz --start 0 --hours 24 --out forecast.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Print mean wind forecast (mu) from uncertainty set NPZ"
    )
    parser.add_argument("npz", type=Path, help="Path to NPZ file (mu, sigma, rho, y_cols)")
    parser.add_argument("--start", type=int, default=2448,
                        help="Start index into NPZ (default: 2448, aligned to July 15)")
    parser.add_argument("--hours", type=int, default=48,
                        help="Number of hours to print (default: 48)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: print to stdout only)")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    mu = data["mu"]     # (T_total, K)
    y_cols = list(data["y_cols"]) if "y_cols" in data else [f"wind_{k}" for k in range(mu.shape[1])]

    T_total, K = mu.shape
    s, e = args.start, args.start + args.hours
    if e > T_total:
        print(f"Warning: requested [{s},{e}) but only {T_total} hours available. Truncating.")
        e = T_total

    mu_slice = mu[s:e]   # (T, K)
    T = len(mu_slice)
    hours = list(range(T))

    df = pd.DataFrame(mu_slice, index=hours, columns=y_cols)
    df.index.name = "hour"

    print(f"NPZ: {args.npz}")
    print(f"Indices [{s}, {e})  ({T} hours, {K} wind farms)")
    print(f"Farms: {', '.join(y_cols)}")
    print(f"\nMean wind forecast mu (MW)  —  {K} units  x  {T} hours\n")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.1f}".format):
        print(df.to_string())

    print(f"\nTotal wind (sum across units):")
    totals = df.sum(axis=1)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.1f}".format):
        print(totals.to_string())

    print(f"\nSummary:")
    print(f"  Mean total forecast:  {totals.mean():10.1f} MW")
    print(f"  Min  total forecast:  {totals.min():10.1f} MW  (hour {totals.idxmin()})")
    print(f"  Max  total forecast:  {totals.max():10.1f} MW  (hour {totals.idxmax()})")

    if args.out:
        df.to_csv(args.out, float_format="%.2f")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
