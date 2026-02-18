"""
Plot mean forecast vs worst-case wind dispatch from an NPZ uncertainty set file.

Shows the first 48 hours: total wind forecast (mu), worst-case dispatch under
the ellipsoid, and the shaded uncertain band between them.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean vs worst-case wind from NPZ uncertainty set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz-path", type=Path,
        default=Path(__file__).parent
        / "uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha90.npz",
        help="Path to NPZ file",
    )
    parser.add_argument("--start", type=int, default=0,
                        help="Start hour index into the NPZ time series")
    parser.add_argument("--hours", type=int, default=48,
                        help="Number of hours to plot")
    parser.add_argument("--out", type=str, default=None,
                        help="Output filename stem (default: npz_worst_case)")
    args = parser.parse_args()

    # Load NPZ
    print(f"Loading {args.npz_path} ...")
    npz = np.load(args.npz_path, allow_pickle=True)
    mu = npz["mu"]          # (T_total, K)
    sigma = npz["sigma"]    # (T_total, K, K)
    rho = npz["rho"]        # (T_total,)
    times = npz["times"]    # (T_total,) datetime-like

    if "y_cols" in npz.files:
        wind_ids = [str(w) for w in npz["y_cols"]]
    else:
        wind_ids = [f"Wind_{k}" for k in range(mu.shape[1])]

    T_total, K = mu.shape
    t0 = args.start
    t1 = min(t0 + args.hours, T_total)
    T = t1 - t0

    mu = mu[t0:t1]
    sigma = sigma[t0:t1]
    rho = rho[t0:t1]
    times = times[t0:t1]

    print(f"Plotting hours {t0}..{t1} ({T} periods), {K} wind farms")

    # --- System-level worst case ---
    forecast = mu.sum(axis=1)                              # (T,)
    ones = np.ones(K)
    worst_case = np.array([
        forecast[t] - rho[t] * np.sqrt(ones @ sigma[t] @ ones)
        for t in range(T)
    ])
    uncertain = forecast - worst_case                      # (T,)

    # --- Per-farm worst case ---
    per_farm_worst = np.array([
        [mu[t, k] - rho[t] * np.sqrt(sigma[t, k, k]) for k in range(K)]
        for t in range(T)
    ])  # (T, K)

    # --- Figure: 2-panel (system + per-farm) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})

    x = np.arange(T)
    tick_step = max(1, T // 12)
    tick_idx = list(range(0, T, tick_step))
    tick_labels = [str(times[i])[:16] for i in tick_idx]

    # --- (a) System total ---
    ax = axes[0]
    ax.plot(x, forecast, color="gray", linestyle="--", linewidth=1.2,
            label="Forecast (mean)")
    ax.plot(x, worst_case, color="#1f77b4", linestyle=":", linewidth=1.5,
            label="Worst-case dispatch")
    ax.fill_between(x, worst_case, forecast,
                    color="#1f77b4", alpha=0.2, label="Uncertain band")
    ax.set_ylabel("Total Wind (MW)", fontsize=10)
    ax.set_title("(a) System-Level: Mean Forecast vs Worst-Case", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- (b) Per-farm ---
    ax = axes[1]
    farm_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for k in range(K):
        c = farm_colors[k % len(farm_colors)]
        ax.plot(x, mu[:, k], color=c, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.plot(x, per_farm_worst[:, k], color=c, linestyle=":", linewidth=0.9, alpha=0.7)
        ax.fill_between(x, per_farm_worst[:, k], mu[:, k],
                        color=c, alpha=0.12, label=wind_ids[k])
    ax.set_ylabel("Wind (MW)", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_title("(b) Per-Farm: Mean Forecast vs Worst-Case", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=min(K, 4))
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")

    fig.suptitle(
        f"Implied Worst-Case Wind Dispatch (hours {t0}-{t1})",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    stem = args.out or "npz_worst_case"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf / {stem}.png")


if __name__ == "__main__":
    main()
