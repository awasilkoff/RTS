from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from utils import CachedPaths, fit_standard_scaler

# Refactored modules (rename files or imports as you prefer)
from data_processing import (
    build_conformal_totals_df,
    build_XY_for_covariance_system_only,
)
from conformal_prediction import train_wind_lower_model_conformal_binned
from covariance_optimization import (
    KernelCovConfig,
    FitConfig,
    CovPredictConfig,
    fit_omega,
    predict_mu_sigma_topk_cross,
    implied_rho_from_total_lower_bound,
)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Orchestrator")

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PATHS = CachedPaths(
    actuals_parquet=DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet",
    forecasts_parquet=DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet",
    output_dir=DATA_DIR / "viz_artifacts",
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(exist_ok=True, parents=True)


def run_integration_pipeline() -> None:
    """
    1) Learn covariance geometry (omega + conditional moments) using TRAIN reference only.
    2) Train conformal lower bound model on TRAIN period.
    3) For each eval timestamp, compute implied rho from total lower bound and (mu, Sigma).
    4) Export parquet + npz.
    """
    if PATHS.output_dir is None:
        raise ValueError("PATHS.output_dir must be set")
    _ensure_dir(PATHS.output_dir)

    logger.info("Loading data...")
    actuals = pd.read_parquet(PATHS.actuals_parquet)
    forecasts = pd.read_parquet(PATHS.forecasts_parquet)


    # --------------------------------------------------------------------------
    # STEP 1: Covariance Optimization (shape)
    # --------------------------------------------------------------------------
    logger.info("--- Step 1: Learning covariance structure ---")

    X_cov, Y_cov, times_cov, x_cols, y_cols = build_XY_for_covariance_system_only(
        forecasts, actuals, drop_any_nan_rows=True
    )

    logger.info(
        "Covariance data shapes: X=%s, Y=%s, T=%s",
        X_cov.shape,
        Y_cov.shape,
        len(times_cov),
    )

    scaler_cov = fit_standard_scaler(X_cov)
    Xs_cov = scaler_cov.transform(X_cov)

    # Time-ordered split: first half train, second half eval (you can swap to a date split easily)
    n_total = Xs_cov.shape[0]
    n_train = int(n_total * 0.75)
    if n_train < 10:
        raise ValueError(
            f"Not enough samples for training split: n_total={n_total}, n_train={n_train}"
        )

    train_idx = np.arange(n_train)
    eval_idx = np.arange(n_train, n_total)

    Xs_train = Xs_cov[train_idx]
    Y_train = Y_cov[train_idx]
    times_train = times_cov[train_idx]

    Xs_eval = Xs_cov[eval_idx]
    times_eval = times_cov[eval_idx]

    logger.info(
        "Training covariance model on %d samples; eval on %d samples.",
        len(train_idx),
        len(eval_idx),
    )

    cov_cfg = KernelCovConfig(tau=20.0, ridge=1e-3)
    fit_cfg = FitConfig(
        step_size=0.03,
        grad_clip=10.0,
        max_iters=500,
        tol=0.0,
    )

    omega_hat = fit_omega(
        X=Xs_cov,  # full X is fine; train_idx controls the loss
        Y=Y_cov,
        omega0=np.ones(X_cov.shape[1], dtype=float),
        train_idx=train_idx,
        cfg=cov_cfg,
        fit_cfg=fit_cfg,
    )
    logger.info(f"Omega: {omega_hat}")
    logger.info(
        "Predicting conditional moments for eval set (NO leakage: using TRAIN reference only)..."
    )

    # IMPORTANT: use train reference set only
    mu_eval, sigma_eval = predict_mu_sigma_topk_cross(
        X_query=Xs_eval,
        X_ref=Xs_train,
        Y_ref=Y_train,
        omega=omega_hat,
        cfg=CovPredictConfig(tau=cov_cfg.tau, ridge=cov_cfg.ridge),
        k=128,
        return_type="numpy",  # easier downstream
    )
    # mu_eval: (N_eval, M), sigma_eval: (N_eval, M, M)
    logger.info("Moments predicted: mu=%s, sigma=%s", mu_eval.shape, sigma_eval.shape)

    # --------------------------------------------------------------------------
    # STEP 2: Conformal Prediction (scale)
    # --------------------------------------------------------------------------
    logger.info("--- Step 2: Calibrating conformal lower bound ---")

    df_tot = build_conformal_totals_df(actuals, forecasts)

    # Train conformal on the SAME train period (by time)
    train_end_time = times_train[-1]
    df_train_conf = df_tot[df_tot["TIME_HOURLY"] <= train_end_time].copy()

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    bundle, metrics, _df_test = train_wind_lower_model_conformal_binned(
        df_train_conf,
        feature_cols=feature_cols,
        target_col="y",
        scale_col="ens_std",
        alpha_target=0.95,
        binning="y_pred",
    )
    logger.info(
        "Conformal metrics (on its internal test split): coverage=%.2f%%, rmse=%.4f, mae=%.4f",
        100.0 * metrics["coverage"],
        metrics["rmse"],
        metrics["mae"],
    )

    # Evaluate conformal bounds on eval timestamps
    df_eval_conf = df_tot[df_tot["TIME_HOURLY"].isin(times_eval)].copy()
    df_eval_conf = df_eval_conf.sort_values("TIME_HOURLY").reset_index(drop=True)

    df_pred_eval = bundle.predict_df(df_eval_conf)
    bounds = df_pred_eval.set_index("TIME_HOURLY")["y_pred_conf"].sort_index()
    # Add actual + ensemble mean for plotting
    df_pred_eval["y"] = df_eval_conf["y"].to_numpy()
    df_pred_eval["ens_mean"] = df_eval_conf["ens_mean"].to_numpy()
    cols_keep = [
        "TIME_HOURLY",
        "y",  # actual
        "ens_mean",  # mean forecast
        "y_pred_base",  # quantile base
        "y_pred_conf",  # conformalized
    ]
    df_plot = df_pred_eval[cols_keep].copy()
    df_plot.to_parquet(
        PATHS.output_dir / "viz_eval_conformal_frame.parquet", index=False
    )

    # --------------------------------------------------------------------------
    # STEP 3: Knit (rho)
    # --------------------------------------------------------------------------
    logger.info("--- Step 3: Knitting (computing rho) ---")

    # Ensure consistent order: we’ll iterate in times_eval order and look up bounds by timestamp.
    ones = np.ones(mu_eval.shape[1], dtype=float)

    kept_i = []
    rows = []
    missing_bounds = 0

    for i, t in enumerate(times_eval):
        if t not in bounds.index:
            missing_bounds += 1
            continue

        total_lower_bound = float(bounds.loc[t])

        mu_t = mu_eval[i]
        sigma_t = sigma_eval[i]

        # Total std = sqrt(1^T Sigma 1)
        sigma_total = float(ones @ sigma_t @ ones)
        sigma_total_sqrt = float(np.sqrt(max(sigma_total, 0.0)))

        try:
            rho = float(
                implied_rho_from_total_lower_bound(
                    Sigma=sigma_t,
                    mean=mu_t,
                    total_lower_bound=total_lower_bound,
                    clip_nonneg=True,
                )
            )
        except ValueError:
            rho = 0.0

        rows.append(
            {
                "TIME_HOURLY": t,
                "rho": rho,
                "total_lower_bound": total_lower_bound,
                "mu_sum": float(mu_t.sum()),
                "sigma_total_sqrt": sigma_total_sqrt,
                "mu_vec": mu_t.tolist(),
                "sigma_diag": np.diag(sigma_t).tolist(),
            }
        )
        kept_i.append(i)  # <-- ADD THIS

    if missing_bounds:
        logger.info(
            "Skipped %d eval timestamps with missing conformal bounds.", missing_bounds
        )

    df_final = pd.DataFrame(rows).sort_values("TIME_HOURLY").reset_index(drop=True)
    kept_i = np.asarray(kept_i, dtype=int)

    times_kept = times_eval[kept_i]
    mu_kept = mu_eval[kept_i]
    sigma_kept = sigma_eval[kept_i]

    # --------------------------------------------------------------------------
    # STEP 4: Export
    # --------------------------------------------------------------------------
    out_parquet = PATHS.output_dir / "viz_integration_results.parquet"
    out_npz = PATHS.output_dir / "full_tensors.npz"

    df_final.to_parquet(out_parquet, index=False)

    times_np = (
        times_kept.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    )

    np.savez_compressed(
        out_npz,
        times=times_np,
        mu=mu_kept,
        sigma=sigma_kept,
        rho=df_final["rho"].to_numpy(dtype=float),
        omega=omega_hat,
        x_cols=np.array(x_cols, dtype=object),
        y_cols=np.array(y_cols, dtype=object),
    )

    logger.info("Success: wrote %d rows.", len(df_final))
    logger.info("Parquet: %s", out_parquet)
    logger.info("NPZ: %s", out_npz)


if __name__ == "__main__":
    run_integration_pipeline()
