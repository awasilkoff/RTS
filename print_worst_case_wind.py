#!/usr/bin/env python3
"""Print worst-case wind realization from an uncertainty set NPZ.

Computes:  worst_case[t] = mu[t] - rho[t] * Sigma[t] @ e / sqrt(e^T Sigma[t] e)

This is the realization on the ellipsoid boundary that minimizes total wind.

Usage:
    python print_worst_case_wind.py <npz_path> [--start 2448] [--hours 24] [--hour 16]

Examples:
    python print_worst_case_wind.py uncertainty_sets_refactored/data/uncertainty_sets.npz --start 2448 --hours 24
    python print_worst_case_wind.py alpha_sweep/.../sigma_rho.npz --start 0 --hours 12 --hour 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Print worst-case wind from uncertainty set NPZ",
    )
    parser.add_argument("npz", type=Path, help="Path to NPZ file (mu, sigma, rho, y_cols)")
    parser.add_argument("--start", type=int, default=0, help="Start index into NPZ (default: 0)")
    parser.add_argument("--hours", type=int, default=24, help="Number of hours (default: 24)")
    parser.add_argument("--hour", type=int, default=None, help="Print detail for a specific hour offset (e.g. 16)")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    mu = data["mu"]        # (T_total, K)
    sigma = data["sigma"]  # (T_total, K, K)
    rho = data["rho"]      # (T_total,)
    y_cols = list(data["y_cols"]) if "y_cols" in data else [f"wind_{k}" for k in range(mu.shape[1])]

    T_total, K = mu.shape
    s, e = args.start, args.start + args.hours
    if e > T_total:
        print(f"Warning: requested [{s},{e}) but only {T_total} hours available. Truncating.")
        e = T_total
    T = e - s

    mu = mu[s:e]        # (T, K)
    sigma = sigma[s:e]  # (T, K, K)
    rho = rho[s:e]      # (T,)
    ones = np.ones(K)

    # Compute worst-case realization per hour
    # r* = -rho * Sigma @ e / sqrt(e^T Sigma e)   (deviation that minimizes total wind)
    # worst = mu + r* = mu - rho * Sigma @ e / sqrt(e^T Sigma e)
    wc = np.zeros_like(mu)       # (T, K)
    dev = np.zeros_like(mu)      # (T, K)
    total_dev = np.zeros(T)

    for t in range(T):
        Se = sigma[t] @ ones                    # (K,)
        denom = np.sqrt(ones @ sigma[t] @ ones) # scalar = sqrt(e^T Sigma e)
        if denom < 1e-12:
            wc[t] = mu[t]
            continue
        direction = Se / denom                   # (K,) unit direction in Sigma-norm
        dev[t] = rho[t] * direction              # per-farm deviation
        wc[t] = mu[t] - dev[t]
        total_dev[t] = rho[t] * denom            # = sum of dev[t]

    # System-level summary
    total_mu = mu.sum(axis=1)
    total_wc = wc.sum(axis=1)

    print(f"Uncertainty set: {args.npz}")
    print(f"Hours [{s}, {e})  ({T} periods, {K} wind farms)")
    print(f"Farms: {', '.join(y_cols)}")
    print()

    # Table header
    hdr = f"{'Hour':>5}  {'Forecast':>10}  {'Worst-case':>10}  {'Deviation':>10}  {'Dev%':>6}  {'rho':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t in range(T):
        pct = total_dev[t] / total_mu[t] * 100 if total_mu[t] > 1e-3 else 0.0
        print(f"{t:5d}  {total_mu[t]:10.1f}  {total_wc[t]:10.1f}  {total_dev[t]:10.1f}  {pct:5.1f}%  {rho[t]:8.3f}")

    print()
    print(f"Mean forecast:    {total_mu.mean():10.1f} MW")
    print(f"Mean worst-case:  {total_wc.mean():10.1f} MW")
    print(f"Mean deviation:   {total_dev.mean():10.1f} MW ({total_dev.mean()/total_mu.mean()*100:.1f}%)")
    print(f"Max  deviation:   {total_dev.max():10.1f} MW (hour {np.argmax(total_dev)})")

    # Per-hour detail
    if args.hour is not None:
        t = args.hour
        if t >= T:
            print(f"\nHour {t} out of range (0..{T-1})")
            return
        print(f"\n{'='*60}")
        print(f"Hour {t} detail  (rho={rho[t]:.4f})")
        print(f"{'='*60}")
        print(f"  {'Farm':>25}  {'Forecast':>10}  {'Worst':>10}  {'Dev':>10}  {'Dev%':>6}")
        for k in range(K):
            pct = dev[t, k] / mu[t, k] * 100 if mu[t, k] > 1e-3 else 0.0
            print(f"  {y_cols[k]:>25}  {mu[t,k]:10.1f}  {wc[t,k]:10.1f}  {dev[t,k]:10.1f}  {pct:5.1f}%")
        print(f"  {'TOTAL':>25}  {total_mu[t]:10.1f}  {total_wc[t]:10.1f}  {total_dev[t]:10.1f}  "
              f"{total_dev[t]/total_mu[t]*100:.1f}%")


if __name__ == "__main__":
    main()
