"""print_worst_case_wind.py

Print worst-case wind realization from an uncertainty set NPZ, in the same
style as print_wind_forecast.py: full (hours × generators) table with gen IDs.

Computes:  wc[t,k] = mu[t,k] - rho[t] * (Sigma[t] @ 1)[k] / sqrt(1^T Sigma[t] 1)

This is the ellipsoid boundary point that minimises total system wind output.

Usage:
    python print_worst_case_wind.py [--start-month 7] [--start-day 15] [--hours 48]
    python print_worst_case_wind.py --npz path/to/sigma_rho.npz --provider-start 2448 --hours 48
    python print_worst_case_wind.py --hours 48 --out wc.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RTS_DIR   = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR     = RTS_DIR / "timeseries_data_files"
SPP_PARQUET = Path(
    "uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet"
)
DEFAULT_NPZ = Path(
    "uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz"
)


def compute_worst_case(mu: np.ndarray, sigma: np.ndarray, rho: np.ndarray):
    """
    Compute worst-case wind per generator per hour.

    Parameters
    ----------
    mu    : (T, K)
    sigma : (T, K, K)
    rho   : (T,)

    Returns
    -------
    wc  : (T, K)  worst-case realisation
    dev : (T, K)  per-generator deviation (mu - wc)
    """
    T, K = mu.shape
    ones = np.ones(K)
    wc  = np.zeros_like(mu)
    dev = np.zeros_like(mu)

    for t in range(T):
        Se    = sigma[t] @ ones                    # (K,)
        denom = np.sqrt(ones @ sigma[t] @ ones)   # scalar
        if denom < 1e-12:
            wc[t] = mu[t]
            continue
        dev[t] = rho[t] * Se / denom              # per-generator deviation
        wc[t]  = mu[t] - dev[t]

    return wc, dev


def main():
    parser = argparse.ArgumentParser(
        description="Print worst-case wind (per generator) from uncertainty NPZ"
    )
    parser.add_argument("--start-month",     type=int,  default=7)
    parser.add_argument("--start-day",       type=int,  default=15)
    parser.add_argument("--start-hour",      type=int,  default=0)
    parser.add_argument("--hours",           type=int,  default=48)
    parser.add_argument("--npz",             type=Path, default=DEFAULT_NPZ,
                        help="NPZ file path (default: sigma_rho_alpha99.npz)")
    parser.add_argument("--provider-start",  type=int,  default=2448,
                        help="Start index into NPZ (default 2448, matches run_comparison.py)")
    parser.add_argument("--out",             type=Path, default="99",
                        help="Save worst-case table to CSV")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load NPZ
    # ------------------------------------------------------------------
    npz_data = np.load(args.npz, allow_pickle=True)
    mu_all    = npz_data["mu"]     # (N, K)
    sigma_all = npz_data["sigma"]  # (N, K, K)
    rho_all   = npz_data["rho"]    # (N,)
    y_cols    = (list(npz_data["y_cols"]) if "y_cols" in npz_data
                 else [f"wind_{k}" for k in range(mu_all.shape[1])])

    s, e = args.provider_start, args.provider_start + args.hours
    if e > len(rho_all):
        print(f"Warning: requested [{s},{e}) but NPZ has {len(rho_all)} rows. Truncating.")
        e = len(rho_all)
    T = e - s

    mu    = mu_all[s:e]
    sigma = sigma_all[s:e]
    rho   = rho_all[s:e]

    # ------------------------------------------------------------------
    # 2. Get gen IDs from DAMData (for labeled columns)
    # ------------------------------------------------------------------
    from io_rts import build_damdata_from_rts

    start_time = pd.Timestamp(
        year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour
    )
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start_time,
        horizon_hours=args.hours,
        spp_forecasts_parquet=SPP_PARQUET,
        spp_start_idx=0,
    )
    wind_mask = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    dam_wind_ids = [gid for gid, m in zip(data.gen_ids, wind_mask) if m]

    # Match NPZ y_cols order to DAM gen IDs where possible
    # Use DAM gen IDs if they match NPZ y_cols, else fall back to y_cols
    col_ids = dam_wind_ids if set(dam_wind_ids) == set(y_cols) else y_cols

    # ------------------------------------------------------------------
    # 3. Compute worst-case
    # ------------------------------------------------------------------
    wc, dev = compute_worst_case(mu, sigma, rho)

    hours = list(range(T))
    df_mu  = pd.DataFrame(mu,  index=hours, columns=y_cols)
    df_wc  = pd.DataFrame(wc,  index=hours, columns=y_cols)
    df_dev = pd.DataFrame(dev, index=hours, columns=y_cols)
    for df in (df_mu, df_wc, df_dev):
        df.index.name = "hour"

    total_mu  = df_mu.sum(axis=1)
    total_wc  = df_wc.sum(axis=1)
    total_dev = df_dev.sum(axis=1)

    # ------------------------------------------------------------------
    # 4. Print tables
    # ------------------------------------------------------------------
    fmt = "{:.1f}".format
    ctx = dict(display__max_columns=None, display__width=200,
               display__float_format=fmt)

    print(f"NPZ: {args.npz}")
    print(f"Indices [{s}, {e})  |  {T} hours  |  {len(y_cols)} generators")
    print(f"Generators: {', '.join(y_cols)}")

    print(f"\n{'='*70}")
    print("MEAN FORECAST  mu  (MW)")
    print(f"{'='*70}")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", fmt):
        print(df_mu.to_string())

    print(f"\n{'='*70}")
    print("WORST-CASE WIND  wc = mu - deviation  (MW)")
    print(f"{'='*70}")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", fmt):
        print(df_wc.to_string())

    print(f"\n{'='*70}")
    print("DEVIATION  mu - wc  (MW, always >= 0)")
    print(f"{'='*70}")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", fmt):
        print(df_dev.to_string())

    # Summary row
    print(f"\n{'='*70}")
    print("SYSTEM TOTAL SUMMARY")
    print(f"{'='*70}")
    hdr = f"{'hour':>5}  {'forecast':>10}  {'worst-case':>10}  {'deviation':>10}  {'dev%':>6}  {'rho':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t in range(T):
        pct = total_dev.iloc[t] / total_mu.iloc[t] * 100 if total_mu.iloc[t] > 1e-3 else 0.0
        print(f"{t:5d}  {total_mu.iloc[t]:10.1f}  {total_wc.iloc[t]:10.1f}  "
              f"{total_dev.iloc[t]:10.1f}  {pct:5.1f}%  {rho[t]:8.3f}")

    print(f"\nMean forecast:    {total_mu.mean():10.1f} MW")
    print(f"Mean worst-case:  {total_wc.mean():10.1f} MW")
    print(f"Mean deviation:   {total_dev.mean():10.1f} MW  ({total_dev.mean()/total_mu.mean()*100:.1f}%)")
    print(f"Max  deviation:   {total_dev.max():10.1f} MW  (hour {total_dev.idxmax()})")

    # ------------------------------------------------------------------
    # 5. Save CSV
    # ------------------------------------------------------------------
    if args.out:
        df_wc.to_csv(str(args.out) + str("wc.csv"), float_format="%.2f")
        df_dev.to_csv(str(args.out) + str('dev.csv'), float_format="%.2f")
        print(f"\nSaved worst-case table to {args.out}")


if __name__ == "__main__":
    main()
