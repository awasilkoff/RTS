"""print_wind_forecast.py

Print the wind forecast (Pmax) used in the DAM for each wind generator,
for the two-day optimization horizon.

By default uses the SPP ensemble mean (same as the main scripts).
Pass --no-spp to fall back to the original DAY_AHEAD_wind.csv.

Usage:
    python print_wind_forecast.py --start-month 7 --start-day 15
    python print_wind_forecast.py --start-month 7 --start-day 15 --no-spp
    python print_wind_forecast.py --start-month 7 --start-day 15 --out forecast.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"
DEFAULT_SPP = Path(
    "uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet"
)


def main():
    parser = argparse.ArgumentParser(
        description="Print wind Pmax forecast used in the DAM"
    )
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--day2-interval", type=int, default=2,
                        help="Day-2 block size in hours (default 2, matching run_comparison.py)")
    parser.add_argument("--provider-start", type=int, default=2448,
                        help="SPP time-series start index (default 2448)")
    parser.add_argument("--no-spp", action="store_true",
                        help="Use original DAY_AHEAD_wind.csv instead of SPP ensemble mean")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: print to stdout only)")
    args = parser.parse_args()

    from io_rts import build_damdata_from_rts

    start = pd.Timestamp(year=2020, month=args.start_month, day=args.start_day,
                         hour=args.start_hour)

    spp_path = None if args.no_spp else DEFAULT_SPP
    source_label = "DAY_AHEAD_wind.csv" if args.no_spp else f"SPP ensemble mean ({DEFAULT_SPP.name})"

    print(f"Loading RTS data: start={start}  horizon={args.hours}h  source={source_label}")
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start,
        horizon_hours=args.hours,
        day2_interval_hours=args.day2_interval,
        spp_forecasts_parquet=spp_path,
        spp_start_idx=args.provider_start,
    )

    # Extract wind generators
    wind_mask = [gt.upper() == "WIND" for gt in data.gen_type]
    wind_ids = [data.gen_ids[i] for i, w in enumerate(wind_mask) if w]

    if not wind_ids:
        print("No wind generators found.")
        return

    Pmax_2d = data.Pmax_2d()                    # (I, T)
    wind_pmax = Pmax_2d[wind_mask, :]            # (n_wind, T)

    df = pd.DataFrame(wind_pmax, index=wind_ids, columns=data.time)
    df.index.name = "generator"

    # Print table
    print(f"\nWind forecast (MW)  —  {len(wind_ids)} units  x  {data.n_periods} periods\n")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.1f}".format):
        print(df.to_string())

    # Summary row
    print(f"\nTotal wind (sum across units):")
    totals = df.sum(axis=0)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.1f}".format):
        print(totals.to_string())

    print(f"\nNameplate capacity (static Pmax):")
    static_pmax = data.Pmax if data.Pmax.ndim == 1 else data.Pmax[:, 0]
    for i, gid in enumerate(data.gen_ids):
        if wind_mask[i]:
            print(f"  {gid:<20}  {static_pmax[i]:.1f} MW")

    if args.out:
        df.to_csv(args.out, float_format="%.2f")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
