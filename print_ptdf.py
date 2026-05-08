"""print_ptdf.py

Print the RTS-GMLC PTDF matrix labeled with line and generator (or bus) IDs.

Two outputs:
  ptdf_gen.csv  — (L x I) rows=lines, columns=generators.
                  Entry [l, i] is the sensitivity of line l flow to a 1 MW
                  injection at generator i's bus (with slack absorbed at the
                  reference bus).  This is the matrix used inside ARUC line
                  constraints.

  ptdf_bus.csv  — (L x N) raw PTDF, rows=lines, columns=buses.

Usage:
    python print_ptdf.py
    python print_ptdf.py --out-dir ptdf_output/
    python print_ptdf.py --top-n 5          # also print top-5 per line
    python print_ptdf.py --no-gen --no-bus  # dry-run (just print summary)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"


def main():
    parser = argparse.ArgumentParser(description="Export labeled PTDF matrix")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="Directory for output CSVs (default: current dir)")
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--hours", type=int, default=24,
                        help="Horizon hours (only affects Pmax/load; PTDF is static)")
    parser.add_argument("--include-renewables", action="store_true", default=False)
    parser.add_argument("--include-nuclear", action="store_true", default=False)
    parser.add_argument("--top-n", type=int, default=0,
                        help="Print top-N most sensitive generators per line")
    parser.add_argument("--no-gen", action="store_true",
                        help="Skip writing ptdf_gen.csv")
    parser.add_argument("--no-bus", action="store_true",
                        help="Skip writing ptdf_bus.csv")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Zero out entries with |PTDF| < threshold before saving")
    args = parser.parse_args()

    import pandas as pd
    from io_rts import build_damdata_from_rts

    start = pd.Timestamp(year=2020, month=args.start_month, day=args.start_day)
    print(f"Loading RTS data (start={start.date()}, {args.hours}h)...")
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start,
        horizon_hours=args.hours,
        include_renewables=args.include_renewables,
        include_nuclear=args.include_nuclear,
    )

    PTDF = data.PTDF              # (L, N)
    gen_to_bus = data.gen_to_bus.astype(int)  # (I,)

    L = data.n_lines
    N = data.n_buses
    I = data.n_gens

    print(f"  {L} lines  x  {N} buses  (raw PTDF)")
    print(f"  {L} lines  x  {I} generators  (generator PTDF)")

    # Generator PTDF: column i = PTDF column of generator i's bus
    gen_ptdf = PTDF[:, gen_to_bus]  # (L, I)

    if args.threshold > 0:
        PTDF = np.where(np.abs(PTDF) >= args.threshold, PTDF, 0.0)
        gen_ptdf = np.where(np.abs(gen_ptdf) >= args.threshold, gen_ptdf, 0.0)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Generator PTDF ---
    if not args.no_gen:
        path = args.out_dir / "ptdf_gen.csv"
        df = pd.DataFrame(gen_ptdf, index=data.line_ids, columns=data.gen_ids)
        df.index.name = "line"
        df.to_csv(path, float_format="%.6f")
        print(f"\nGenerator PTDF saved to {path}  ({df.shape[0]} lines x {df.shape[1]} gens)")

    # --- Bus PTDF ---
    if not args.no_bus:
        path = args.out_dir / "ptdf_bus.csv"
        df_bus = pd.DataFrame(PTDF, index=data.line_ids, columns=data.bus_ids)
        df_bus.index.name = "line"
        df_bus.to_csv(path, float_format="%.6f")
        print(f"Bus PTDF saved      to {path}  ({df_bus.shape[0]} lines x {df_bus.shape[1]} buses)")

    # --- Summary: lines sorted by max absolute sensitivity ---
    max_sens = np.max(np.abs(gen_ptdf), axis=1)  # (L,)
    order = np.argsort(-max_sens)
    print(f"\nTop 20 most sensitive lines (max |PTDF| over all generators):")
    print(f"  {'Line':<15}  {'Fmax (MW)':>10}  {'Max |PTDF|':>12}  {'Most sensitive gen':<20}")
    print("  " + "-" * 65)
    for l in order[:20]:
        best_gen_idx = int(np.argmax(np.abs(gen_ptdf[l])))
        print(f"  {data.line_ids[l]:<15}  {data.Fmax[l]:>10.1f}  "
              f"{max_sens[l]:>12.4f}  {data.gen_ids[best_gen_idx]:<20}")

    # --- Per-line top-N detail ---
    if args.top_n > 0:
        print(f"\nTop-{args.top_n} generators by |PTDF| for each line:")
        for l in order[:20]:
            sens_l = gen_ptdf[l]
            top_idx = np.argsort(-np.abs(sens_l))[:args.top_n]
            entries = "  ".join(
                f"{data.gen_ids[i]}={sens_l[i]:+.4f}" for i in top_idx
            )
            print(f"  {data.line_ids[l]:<15}  {entries}")


if __name__ == "__main__":
    main()
