"""print_wind_forecast.py

Print the actual wind Pmax (=forecast) used by the DAM/DARUC/ARUC model,
extracted directly from the built DAMData object — exactly as run_comparison.py
feeds it into the optimizer.

Optionally compare against the NPZ mu (uncertainty center) so you can verify
they are aligned.

Usage:
    python print_wind_forecast.py [--start-month 7] [--start-day 15] [--hours 48]
    python print_wind_forecast.py --start-month 7 --start-day 15 --hours 48 \
        --npz uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz \
        --provider-start 2448
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths matching run_rts_daruc.py defaults
# ---------------------------------------------------------------------------
RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"
SPP_PARQUET = Path(
    "uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet"
)


def main():
    parser = argparse.ArgumentParser(
        description="Print wind forecast (Pmax) as seen by the DAM/DARUC model"
    )
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day",   type=int, default=15)
    parser.add_argument("--start-hour",  type=int, default=0)
    parser.add_argument("--hours",       type=int, default=48,
                        help="Horizon hours (default 48)")
    parser.add_argument("--spp-start-idx", type=int, default=0,
                        help="Positional start index into SPP parquet (default 0, matches run_rts_daruc.py)")
    parser.add_argument("--npz", type=Path, default="uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz",
                        help="Optional NPZ path to compare mu vs DAM Pmax")
    parser.add_argument("--provider-start", type=int, default=2448,
                        help="NPZ start index (default 2448, matches run_comparison.py default)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save DAM wind Pmax to CSV")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Build DAMData exactly as run_rts_daruc does
    # ------------------------------------------------------------------
    from io_rts import build_damdata_from_rts

    start_time = pd.Timestamp(
        year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour
    )

    print(f"Building DAMData: start={start_time}  horizon={args.hours}h  spp_start_idx={args.spp_start_idx}")
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start_time,
        horizon_hours=args.hours,
        spp_forecasts_parquet=SPP_PARQUET,
        spp_start_idx=args.spp_start_idx,
    )

    # ------------------------------------------------------------------
    # 2. Extract wind generators
    # ------------------------------------------------------------------
    wind_mask = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_ids  = [gid for gid, m in zip(data.gen_ids, wind_mask) if m]

    pmax_2d = data.Pmax_2d()           # (I, T)
    wind_pmax = pmax_2d[wind_mask, :]  # (Kw, T)
    T = wind_pmax.shape[1]
    hours = list(range(T))

    df_dam = pd.DataFrame(wind_pmax.T, index=hours, columns=wind_ids)
    df_dam.index.name = "hour"

    # ------------------------------------------------------------------
    # 3. Print DAM Pmax
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"DAM wind Pmax (MW)  — spp_start_idx={args.spp_start_idx}  [{T} periods, {len(wind_ids)} generators]")
    print(f"  Generators: {', '.join(wind_ids)}")
    print(f"{'='*70}")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.1f}".format):
        print(df_dam.to_string())

    totals_dam = df_dam.sum(axis=1)
    print(f"\nTotal wind (sum across generators):")
    with pd.option_context("display.float_format", "{:.1f}".format):
        print(totals_dam.to_string())
    print(f"\nSummary: mean={totals_dam.mean():.1f}  min={totals_dam.min():.1f} (hr {totals_dam.idxmin()})  max={totals_dam.max():.1f} (hr {totals_dam.idxmax()})")

    # ------------------------------------------------------------------
    # 4. Optionally load NPZ mu and compare
    # ------------------------------------------------------------------
    if args.npz is not None:
        npz_data = np.load(args.npz, allow_pickle=True)
        mu_all  = npz_data["mu"]   # (N, K)
        y_cols  = list(npz_data["y_cols"]) if "y_cols" in npz_data else [f"wind_{k}" for k in range(mu_all.shape[1])]

        s, e = args.provider_start, args.provider_start + T
        if e > len(mu_all):
            print(f"\nWARNING: NPZ has {len(mu_all)} rows, requested [{s},{e}). Truncating.")
            e = len(mu_all)
        mu_slice = mu_all[s:e]  # (T, K)

        df_npz = pd.DataFrame(mu_slice, index=range(len(mu_slice)), columns=y_cols)
        df_npz.index.name = "hour"

        print(f"\n{'='*70}")
        print(f"NPZ mu (MW)  — provider_start={args.provider_start}  [{len(mu_slice)} periods, {len(y_cols)} generators]")
        print(f"  NPZ path: {args.npz}")
        print(f"  y_cols:   {', '.join(y_cols)}")
        print(f"{'='*70}")
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", "{:.1f}".format):
            print(df_npz.to_string())

        totals_npz = df_npz.sum(axis=1)
        print(f"\nNPZ total wind: mean={totals_npz.mean():.1f}  min={totals_npz.min():.1f}  max={totals_npz.max():.1f}")

        # ------------------------------------------------------------------
        # 5. Diff (DAM Pmax - NPZ mu), aligned by gen ID
        # ------------------------------------------------------------------
        common_ids = [c for c in wind_ids if c in df_npz.columns]
        if common_ids:
            T_min = min(len(df_dam), len(df_npz))
            diff = df_dam[common_ids].iloc[:T_min].values - df_npz[common_ids].iloc[:T_min].values
            df_diff = pd.DataFrame(diff, index=range(T_min), columns=common_ids)
            df_diff.index.name = "hour"

            print(f"\n{'='*70}")
            print(f"DIFF: DAM Pmax - NPZ mu  (should be ~0 if aligned)")
            print(f"{'='*70}")
            with pd.option_context("display.max_columns", None, "display.width", 200,
                                   "display.float_format", "{:+.1f}".format):
                print(df_diff.to_string())
            total_diff = np.abs(diff).sum()
            max_diff   = np.abs(diff).max()
            print(f"\n|diff| total={total_diff:.1f}  max={max_diff:.1f}")
            if max_diff < 1.0:
                print("  -> Data sources appear ALIGNED (max diff < 1 MW)")
            else:
                print(f"  -> WARNING: large discrepancy ({max_diff:.1f} MW max). "
                      f"Check spp_start_idx vs provider_start alignment.")
        else:
            print(f"\nNo common gen IDs between DAM ({wind_ids}) and NPZ ({y_cols}) — cannot diff.")

    # ------------------------------------------------------------------
    # 6. Save CSV
    # ------------------------------------------------------------------
    if args.out:
        df_dam.to_csv(args.out, float_format="%.2f")
        print(f"\nSaved DAM wind Pmax to {args.out}")


if __name__ == "__main__":
    main()
