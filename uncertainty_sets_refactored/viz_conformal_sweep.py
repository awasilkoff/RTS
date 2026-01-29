from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df


def sweep_conformal_global_qhat(
    actuals_parquet: Path,
    forecasts_parquet: Path,
    *,
    out_png: Path,
    train_end_time: pd.Timestamp
    | None = None,  # optional: enforce same window as covariance
    alpha_targets: list[float] | None = None,
) -> pd.DataFrame:
    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)

    df_tot = build_conformal_totals_df(actuals, forecasts)

    if train_end_time is not None:
        df_train = df_tot[df_tot["TIME_HOURLY"] <= train_end_time].copy()
    else:
        df_train = df_tot.copy()

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    if alpha_targets is None:
        # coverage targets
        alpha_targets = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]

    rows = []
    for a in alpha_targets:
        bundle, metrics, _ = train_wind_lower_model_conformal_binned(
            df_train,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(a),
            binning="y_pred",
        )
        qhat = float(bundle.q_hat_global_r)

        # interpretable “typical” additive correction in MW (since margin = qhat * ens_std)
        med_scale = float(np.median(df_train["ens_std"].to_numpy()))
        med_margin = qhat * max(med_scale, bundle.min_scale)

        rows.append(
            {
                "alpha_target": float(a),
                "empirical_coverage": float(metrics["coverage"]),
                "q_hat_global_r": qhat,
                "median_scale": med_scale,
                "median_margin": med_margin,
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
            }
        )

    res = pd.DataFrame(rows).sort_values("alpha_target").reset_index(drop=True)

    # --- Plot ---
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(res["alpha_target"], res["q_hat_global_r"], marker="o")
    ax1.set_xlabel("Target coverage alpha_target")
    ax1.set_ylabel("Global conformal correction q_hat_global_r (normalized)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(res["alpha_target"], res["median_margin"], marker="s", linestyle="--")
    ax2.set_ylabel("Median additive margin (q_hat * median(scale))")

    plt.title("Global conformal correction vs target coverage")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    return res


if __name__ == "__main__":
    DATA_DIR = Path("/Users/alexwasilkoff/PycharmProjects/RTS/uncertainty_sets/data/")
    out_png = DATA_DIR / "viz_artifacts" / "conformal_global_qhat_sweep.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = sweep_conformal_global_qhat(
        actuals_parquet=DATA_DIR / "actuals_filtered_rts4_constellation_v1.parquet",
        forecasts_parquet=DATA_DIR / "forecasts_filtered_rts4_constellation_v1.parquet",
        out_png=out_png,
        train_end_time=None,  # or set to your covariance train end timestamp
    )
    print(df)
    print(f"Wrote {out_png}")
