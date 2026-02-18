"""
Plot mean forecast vs worst-case wind dispatch from an NPZ uncertainty set file.

Shows the first 48 hours: total wind forecast (mu), worst-case dispatch under
the ellipsoid, and the shaded uncertain band between them.  When RTS load data
is available, adds system load and net-load panels for the full operational picture.

Worst-case total wind at time t:
    p_worst[t] = sum_k mu[k,t] - rho[t] * sqrt(1^T Sigma[t] 1)

Usage:
    python plot_npz_worst_case.py
    python plot_npz_worst_case.py --npz-path path/to/sigma_rho_alpha90.npz
    python plot_npz_worst_case.py --start 0 --hours 48
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_LOAD_CSV = (
    Path(__file__).parent
    / "RTS_Data/timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv"
)


def load_system_load(times, load_csv=None):
    """Load RTS system load aligned to NPZ timestamps.

    Returns (T,) numpy array of system load in MW, or None on failure.
    """
    csv_path = Path(load_csv) if load_csv else _DEFAULT_LOAD_CSV
    if not csv_path.exists():
        print(f"Warning: load CSV not found at {csv_path}, skipping load panels")
        return None

    df = pd.read_csv(csv_path)
    # Build timestamps: Period 1 = hour 0
    df["timestamp"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=df["Period"] - 1)
    )
    df["system_load"] = df["1"] + df["2"] + df["3"]
    load_series = df.set_index("timestamp")["system_load"]

    # Align to NPZ times (match month/day/hour; load CSV is year 2020,
    # NPZ may use a different year from SPP data)
    npz_idx = pd.DatetimeIndex(times)
    load_year = df["Year"].iloc[0]
    aligned_idx = npz_idx.map(
        lambda ts: pd.Timestamp(year=load_year, month=ts.month, day=ts.day, hour=ts.hour)
    )
    try:
        load_vals = load_series.loc[aligned_idx].values.astype(float)
    except KeyError:
        print("Warning: NPZ timestamps not found in load CSV, skipping load panels")
        return None

    if np.isnan(load_vals).any():
        print("Warning: NaN in aligned load data, skipping load panels")
        return None

    return load_vals


def score_days(mu, sigma, rho, times):
    """Score all 24h windows for illustration quality.

    Returns list of (start_idx, score, metrics_dict) sorted by score ascending.
    Lower score = more "moderate" day = better for illustration.
    """
    T_total, K = mu.shape
    ones = np.ones(K)

    # Pre-compute per-hour system quantities
    forecast = mu.sum(axis=1)  # (T_total,)
    envelope = np.array(
        [np.sqrt(ones @ sigma[t] @ ones) for t in range(T_total)]
    )  # (T_total,) -- sqrt(1^T Sigma 1), the MW-scale std
    shortfall = rho * envelope  # (T_total,) -- worst-case MW shortfall

    # Compute dataset-wide medians for normalization
    # Use per-day aggregates to get the median "day behavior"
    day_starts = list(range(0, T_total - 23, 24))
    day_wind_means = [forecast[i : i + 24].mean() for i in day_starts]
    day_wind_stds = [forecast[i : i + 24].std() for i in day_starts]
    day_unc_means = [shortfall[i : i + 24].mean() for i in day_starts]

    med_wind = np.median(day_wind_means)
    med_var = np.median(day_wind_stds)
    med_unc = np.median(day_unc_means)
    med_rho = np.median(rho)

    results = []
    for i in day_starts:
        w = slice(i, i + 24)
        wind_mean = forecast[w].mean()
        wind_std = forecast[w].std()
        unc_mean = shortfall[w].mean()
        rho_std = rho[w].std()
        wind_floor = forecast[w].min()

        # Normalized deviations (lower = closer to median = better)
        s_wind = abs(wind_mean - med_wind) / max(med_wind, 1)
        s_var = abs(wind_std - med_var) / max(med_var, 1)
        s_unc = abs(unc_mean - med_unc) / max(med_unc, 1)
        s_rho = rho_std / max(med_rho, 1)

        # Floor penalty: want wind > 200 MW at all hours
        s_floor = 1.0 if wind_floor < 200 else 0.0

        score = s_wind + s_var + s_unc + s_rho + s_floor
        results.append((i, score, {
            "wind_mean": wind_mean, "wind_std": wind_std,
            "unc_mean": unc_mean, "rho_std": rho_std,
            "wind_floor": wind_floor, "date": str(times[i])[:10],
            "date_end": str(times[i + 23])[:10],
        }))

    results.sort(key=lambda x: x[1])
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean vs worst-case wind from NPZ uncertainty set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path(__file__).parent
        / "uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha95.npz",
        help="Path to NPZ file",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start hour index into the NPZ time series",
    )
    parser.add_argument("--hours", type=int, default=48, help="Number of hours to plot")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output filename stem (default: npz_worst_case)",
    )
    parser.add_argument(
        "--load-csv",
        type=str,
        default=None,
        help="Path to DAY_AHEAD_regional_Load.csv (auto-detected by default)",
    )
    parser.add_argument(
        "--per-farm",
        action="store_true",
        default=False,
        help="Show per-farm wind detail panel",
    )
    parser.add_argument(
        "--find-day",
        action="store_true",
        default=False,
        help="Scan all 24h windows and rank by illustration quality",
    )
    args = parser.parse_args()

    # Load NPZ
    print(f"Loading {args.npz_path} ...")
    npz = np.load(args.npz_path, allow_pickle=True)
    mu = npz["mu"]  # (T_total, K)
    sigma = npz["sigma"]  # (T_total, K, K)
    rho = npz["rho"]  # (T_total,)
    times = npz["times"]  # (T_total,) datetime-like

    if "y_cols" in npz.files:
        wind_ids = [str(w) for w in npz["y_cols"]]
    else:
        wind_ids = [f"Wind_{k}" for k in range(mu.shape[1])]

    T_total, K = mu.shape

    # --- Find-day mode: sweep and rank, then optionally plot the winner ---
    if args.find_day:
        ranked = score_days(mu, sigma, rho, times)
        print(f"\nTop 10 illustration days (out of {len(ranked)} windows):\n")
        print(f"{'Rank':>4}  {'Start':>5}  {'Date':>12}  {'Wind Mean':>9}  "
              f"{'Wind Std':>8}  {'Unc Mean':>8}  {'Floor':>6}  {'Score':>6}")
        print("-" * 72)
        for rank, (idx, sc, m) in enumerate(ranked[:10], 1):
            print(f"{rank:4d}  {idx:5d}  {m['date']:>12}  {m['wind_mean']:9.0f}  "
                  f"{m['wind_std']:8.0f}  {m['unc_mean']:8.0f}  {m['wind_floor']:6.0f}  "
                  f"{sc:6.3f}")
        best_idx = ranked[0][0]
        print(f"\nBest day: --start {best_idx} --hours 24")

        if not args.out:
            return
        # Auto-plot the best day
        args.start = best_idx
        args.hours = 24
        print(f"\nAuto-plotting best day (start={best_idx}) ...")

    t0 = args.start
    t1 = min(t0 + args.hours, T_total)
    T = t1 - t0

    mu = mu[t0:t1]
    sigma = sigma[t0:t1]
    rho = rho[t0:t1]
    times = times[t0:t1]

    print(f"Plotting hours {t0}..{t1} ({T} periods), {K} wind farms")

    # --- System-level worst case ---
    forecast = mu.sum(axis=1)  # (T,)
    ones = np.ones(K)
    worst_case = np.array(
        [forecast[t] - rho[t] * np.sqrt(ones @ sigma[t] @ ones) for t in range(T)]
    )

    # --- Per-farm worst case (only needed with --per-farm) ---
    if args.per_farm:
        per_farm_worst = np.array(
            [
                [mu[t, k] - rho[t] * np.sqrt(sigma[t, k, k]) for k in range(K)]
                for t in range(T)
            ]
        )  # (T, K)

    # --- Load system load (graceful fallback if unavailable) ---
    load_ts = load_system_load(times, load_csv=args.load_csv)
    has_load = load_ts is not None

    x = np.arange(T)
    tick_step = max(1, T // 12)
    tick_idx = list(range(0, T, tick_step))
    tick_labels = [str(times[i])[:16] for i in tick_idx]

    farm_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    # Determine panel count
    show_per_farm = args.per_farm
    if has_load:
        n_panels = 3 if show_per_farm else 2
    else:
        n_panels = 2 if show_per_farm else 1

    # Create figure
    if n_panels == 3:
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 0.8]},
        )
    elif n_panels == 2:
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1]},
        )
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        axes = [axes]  # normalize to list

    panel_idx = 0
    panel_label = ord("a")

    # --- Panel: System Load + Wind (when load available) or Wind-only ---
    ax = axes[panel_idx]
    lbl = chr(panel_label)
    if has_load:
        ax.plot(x, load_ts, color="black", linewidth=1.4, label="System Load")
        ax.plot(x, forecast, color="gray", linestyle="--", linewidth=1.2,
                label="Wind Forecast (mean)")
        ax.plot(x, worst_case, color="#1f77b4", linestyle=":", linewidth=1.5,
                label="Wind Worst-Case")
        ax.fill_between(x, worst_case, forecast, color="#1f77b4", alpha=0.2,
                        label="Wind uncertain band")
        ax.set_ylabel("MW", fontsize=10)
        ax.set_title(f"({lbl}) System Load and Wind Generation", fontsize=11)
    else:
        ax.plot(x, forecast, color="gray", linestyle="--", linewidth=1.2,
                label="Forecast (mean)")
        ax.plot(x, worst_case, color="#1f77b4", linestyle=":", linewidth=1.5,
                label="Worst-case dispatch")
        ax.fill_between(x, worst_case, forecast, color="#1f77b4", alpha=0.2,
                        label="Uncertain band")
        ax.set_ylabel("Total Wind (MW)", fontsize=10)
        ax.set_title(f"({lbl}) System-Level: Mean Forecast vs Worst-Case", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    panel_idx += 1
    panel_label += 1

    # --- Panel: Net Load (only when load available) ---
    if has_load:
        net_load_nominal = load_ts - forecast
        net_load_worst = load_ts - worst_case
        lbl = chr(panel_label)
        ax = axes[panel_idx]
        ax.plot(x, net_load_nominal, color="#d62728", linewidth=1.3,
                label="Net Load (nominal wind)")
        ax.plot(x, net_load_worst, color="#8b0000", linestyle=":", linewidth=1.5,
                label="Net Load (worst-case wind)")
        ax.fill_between(x, net_load_nominal, net_load_worst,
                        color="#d62728", alpha=0.18, label="Additional thermal at risk")
        ax.set_ylabel("MW", fontsize=10)
        ax.set_title(f"({lbl}) Net Load (Load minus Wind)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        panel_idx += 1
        panel_label += 1

    # --- Panel: Per-farm wind (opt-in) ---
    if show_per_farm:
        lbl = chr(panel_label)
        ax = axes[panel_idx]
        for k in range(K):
            c = farm_colors[k % len(farm_colors)]
            ax.plot(x, mu[:, k], color=c, linestyle="--", linewidth=0.9, alpha=0.7)
            ax.plot(x, per_farm_worst[:, k], color=c, linestyle=":", linewidth=0.9,
                    alpha=0.7)
            ax.fill_between(x, per_farm_worst[:, k], mu[:, k], color=c, alpha=0.12,
                            label=wind_ids[k])
        ax.set_ylabel("Wind (MW)", fontsize=10)
        ax.set_title(f"({lbl}) Per-Farm: Mean Forecast vs Worst-Case", fontsize=11)
        ax.legend(fontsize=7, loc="upper right", ncol=min(K, 4))
        ax.grid(True, alpha=0.3)

    # X-axis label on the bottom panel only
    axes[-1].set_xlabel("Time", fontsize=10)

    for ax in axes:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")

    fig.suptitle(
        f"Implied Worst-Case Wind Dispatch (hours {t0}-{t1})",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    stem = args.out or "npz_worst_case"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf / {stem}.png")


if __name__ == "__main__":
    main()
