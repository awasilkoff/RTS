#!/usr/bin/env python3
"""Print worst-case wind deviation at a specific hour from comparison outputs.

Usage:
    python print_worst_case_wind.py <case_dir> [--hour 16]

Example:
    python print_worst_case_wind.py comparison_outputs/lines_rho2.00_48h_m07d15 --hour 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_rts import build_damdata_from_rts
from compare_aruc_vs_daruc import load_results, compute_worst_case_wind


def main():
    parser = argparse.ArgumentParser(
        description="Print worst-case wind deviation from comparison outputs",
    )
    parser.add_argument("case_dir", type=Path, help="Comparison output directory")
    parser.add_argument("--hour", type=int, default=16, help="Snapshot hour (default: 16)")
    args = parser.parse_args()

    case_dir = args.case_dir
    daruc_dir = case_dir / "daruc"
    aruc_dir = case_dir / "aruc"

    # Load summary to reconstruct data
    summary_path = daruc_dir / "summary.json"
    if not summary_path.exists():
        summary_path = aruc_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json in {daruc_dir} or {aruc_dir}")
    with open(summary_path) as f:
        summary = json.load(f)

    start_time = pd.Timestamp(summary["start_time"])
    hours = summary["hours"]

    # Rebuild DAMData
    data = build_damdata_from_rts(
        start_time=start_time,
        horizon_hours=hours,
        day2_interval_hours=2,
        single_block=True,
    )

    # Load results
    labels_results = []
    for label, rdir in [("DARUC", daruc_dir), ("ARUC", aruc_dir)]:
        if (rdir / "dispatch_p0.csv").exists():
            labels_results.append((label, load_results(rdir, label)))

    if not labels_results:
        print("No results found.")
        return

    # Common times
    all_cols = [set(r["p0"].columns) for _, r in labels_results]
    common = sorted(set.intersection(*all_cols))

    # Print per formulation
    for label, res in labels_results:
        wc = compute_worst_case_wind(res, data, common)
        if wc is None:
            print(f"\n{label}: No Z/Sigma/rho data — skipping.")
            continue

        # Find snapshot hour index
        hour_indices = []
        for i, t in enumerate(common):
            try:
                h = pd.Timestamp(t).hour
            except Exception:
                h = int(t.split(" ")[1].split(":")[0]) if " " in str(t) else i
            if h == args.hour:
                hour_indices.append(i)

        if not hour_indices:
            print(f"\n{label}: Hour {args.hour} not found in common times.")
            continue

        t_idx = hour_indices[0]

        print(f"\n{'='*60}")
        print(f"{label} — Hour {args.hour}:00  (period {t_idx}, {common[t_idx]})")
        print(f"{'='*60}")

        dev_t = wc["uncertain_ts"][t_idx]
        nom_t = wc["nominal_ts"][t_idx]
        fc_t = wc["forecast_ts"][t_idx]
        wc_t = wc["worst_case_ts"][t_idx]

        print(f"  Forecast (Pmax):     {fc_t:8.1f} MW")
        print(f"  Nominal dispatch:    {nom_t:8.1f} MW")
        print(f"  Worst-case dispatch: {wc_t:8.1f} MW")
        print(f"  Deviation:           {dev_t:8.1f} MW  "
              f"({dev_t/nom_t*100:.1f}% of nominal)" if nom_t > 1e-3 else "")
        print()

        print(f"  Per wind farm:")
        for gid, farm in wc["per_farm"].items():
            fn = farm["nominal"][t_idx]
            fw = farm["worst_case"][t_idx]
            fd = farm["deviation"][t_idx]
            ff = farm["forecast"][t_idx]
            pct = f"{fd/fn*100:.1f}%" if fn > 1e-3 else "n/a"
            print(f"    {gid:20s}  forecast={ff:7.1f}  nominal={fn:7.1f}"
                  f"  worst={fw:7.1f}  dev={fd:7.1f} MW ({pct})")

        # Also print full-horizon summary
        dev = wc["uncertain_ts"]
        print(f"\n  Full horizon summary:")
        print(f"    Total deviation:  {dev.sum():.1f} MW·h")
        print(f"    Mean:  {dev.mean():.1f} MW  |  Max: {dev.max():.1f} MW (h{np.argmax(dev)})")


if __name__ == "__main__":
    main()
