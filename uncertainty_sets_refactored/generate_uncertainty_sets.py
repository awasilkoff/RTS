"""
Batch pre-computation of time-varying uncertainty sets (mu, Sigma, rho).

This script generates pre-computed uncertainty sets for all available hours,
which can then be loaded efficiently by the UncertaintySetProvider for ARUC.

Data source: Uses RTS4 data by default (rts4_constellation_v1).
Output: Saved to data/uncertainty_sets_rts4/ to differentiate from rts3 results.

Configuration defaults (optimized for day-ahead operation):
- omega_constraint: softmax (ensures sum(omega)=1, cleaner feature importance)
- n_bins: 1 (global conformal, no adaptive binning)
- safety_margin: 0.0 (no extra conservativeness buffer)
- conformal_features: baseline set (day-ahead safe, no temporal lag features)

Usage:
    python generate_uncertainty_sets.py

    # With custom parameters:
    python generate_uncertainty_sets.py \
        --tau 5.0 \
        --k 64 \
        --alpha-target 0.95 \
        --omega-path data/viz_artifacts_rts4/focused_2d/best_omega.npy
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils import CachedPaths, fit_scaler, apply_scaler
from data_processing import build_conformal_totals_df
from data_processing_extended import FEATURE_BUILDERS
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
logger = logging.getLogger("UncertaintySetGenerator")


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
@dataclass
class UncertaintySetConfig:
    """Configuration for uncertainty set generation."""

    # Covariance kernel parameters
    tau: float = 5.0
    ridge: float = 1e-3
    k: int = 64

    # Omega (learned feature weights)
    omega_path: Optional[Path] = None  # Load pre-trained omega
    retrain_omega: bool = False  # If True, retrain omega even if omega_path exists
    omega_constraint: str = "softmax"  # "none", "softmax", "simplex", or "normalize"

    # Omega training parameters (used if retraining)
    fit_max_iters: int = 500
    fit_step_size: float = 0.03
    fit_grad_clip: float = 10.0

    # Conformal prediction parameters
    alpha_target: float = 0.95
    n_bins: int = 1  # Global conformal (single bin) - always use 1
    quantile_alpha: float = 0.05  # Base quantile for lower bound
    safety_margin: float = 0.0  # No extra buffer - always 0

    # Data processing
    scaler_type: str = "minmax"  # "standard", "minmax", or "none"
    feature_set: str = "focused_2d"  # Feature set for covariance (SYS_MEAN, SYS_STD)
    train_frac: float = 0.75  # Fraction for training

    # Conformal feature columns (day-ahead safe)
    # Uses "dispersion_plus_historical" feature set - best balance of coverage & tightness
    # Coverage: 0.9481, RMSE: 117.3 (tighter bounds than forecast_dispersion+deep)
    conformal_feature_cols: list = field(
        default_factory=lambda: [
            "ens_mean",
            "ens_std",
            "ens_min",
            "ens_max",
            "n_models",
            "hour",
            "dow",
            # Forecast dispersion features (day-ahead safe)
            "forecast_range",
            "forecast_range_normalized",
            "forecast_cv",
            # Historical features (day-ahead safe: from previous days)
            "y_lag24",  # Same hour yesterday
            "forecast_error_lag24",  # Yesterday's forecast error at this hour
        ]
    )

    # Conformal model configuration - "default" config
    # Good coverage with simpler/faster model
    conformal_model_kwargs: dict = field(
        default_factory=lambda: {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "random_state": 42,
        }
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(exist_ok=True, parents=True)


def _add_conformal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for conformal prediction.

    Includes:
    - Forecast dispersion features (from ensemble)
    - Historical features (from previous days, day-ahead safe)
    """
    df = df.copy()

    # Forecast dispersion features
    df["forecast_range"] = df["ens_max"] - df["ens_min"]
    df["forecast_range_normalized"] = df["forecast_range"] / (df["ens_mean"] + 1e-6)
    df["forecast_cv"] = df["ens_std"] / (df["ens_mean"] + 1e-6)

    # Historical features (day-ahead safe: from previous days)
    df["y_lag24"] = df["y"].shift(24)  # Same hour yesterday
    df["forecast_error_lag24"] = df["y"].shift(24) - df["ens_mean"].shift(24)

    return df


# ------------------------------------------------------------------------------
# Pre-compute covariance (alpha-independent)
# ------------------------------------------------------------------------------


def pre_compute_covariance(
    config: UncertaintySetConfig,
    paths: CachedPaths,
) -> dict:
    """
    Run the alpha-independent steps: load data, build features, train omega,
    compute mu/Sigma for all timestamps, and build the conformal totals df.

    This is the expensive step (~5-10 min) that only needs to run once,
    regardless of how many alpha values will be swept.

    Returns
    -------
    dict with keys:
        mu_all : np.ndarray (N, K) — conditional means for all timestamps
        sigma_all : np.ndarray (N, K, K) — conditional covariances
        omega_hat : np.ndarray (D,) — learned feature weights
        Xs_cov : np.ndarray (N, D) — scaled covariance features
        times_cov : np.ndarray — timestamps for covariance rows
        times_train : np.ndarray — training timestamps
        df_tot : pd.DataFrame — conformal totals df (with features added)
        y_cols : list[str] — wind resource column names
        x_cols : list[str] — feature column names
    """
    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info("Loading data...")
    actuals = pd.read_parquet(paths.actuals_parquet)
    forecasts = pd.read_parquet(paths.forecasts_parquet)

    logger.info(
        "Actuals shape: %s, Forecasts shape: %s", actuals.shape, forecasts.shape
    )

    # -------------------------------------------------------------------------
    # Step 2: Build X/Y for covariance
    # -------------------------------------------------------------------------
    logger.info(
        "Building X/Y matrices for covariance (feature_set=%s)...", config.feature_set
    )

    if config.feature_set not in FEATURE_BUILDERS:
        raise ValueError(
            f"Unknown feature_set '{config.feature_set}'. "
            f"Available: {list(FEATURE_BUILDERS.keys())}"
        )
    feature_builder = FEATURE_BUILDERS[config.feature_set]

    X_cov, Y_cov, times_cov, x_cols, y_cols = feature_builder(
        forecasts, actuals, drop_any_nan_rows=True
    )

    logger.info(
        "Covariance data: X=%s, Y=%s, T=%s",
        X_cov.shape,
        Y_cov.shape,
        len(times_cov),
    )

    # Apply scaling
    scaler = fit_scaler(X_cov, config.scaler_type)
    Xs_cov = apply_scaler(X_cov, scaler)

    # Train/eval split
    n_total = Xs_cov.shape[0]
    n_train = int(n_total * config.train_frac)
    if n_train < 10:
        raise ValueError(
            f"Not enough samples for training: n_total={n_total}, n_train={n_train}"
        )

    train_idx = np.arange(n_train)

    Xs_train = Xs_cov[train_idx]
    Y_train = Y_cov[train_idx]
    times_train = times_cov[train_idx]

    logger.info(
        "Split: train=%d, eval=%d samples",
        n_train,
        n_total - n_train,
    )

    # -------------------------------------------------------------------------
    # Step 3: Load or train omega
    # -------------------------------------------------------------------------
    if config.omega_path and config.omega_path.exists() and not config.retrain_omega:
        logger.info("Loading omega from %s", config.omega_path)
        omega_hat = np.load(config.omega_path)
        if omega_hat.shape[0] != X_cov.shape[1]:
            raise ValueError(
                f"Omega shape {omega_hat.shape} doesn't match features {X_cov.shape[1]}"
            )
    else:
        logger.info("Training omega with constraint=%s...", config.omega_constraint)
        cov_cfg = KernelCovConfig(tau=config.tau, ridge=config.ridge)
        fit_cfg = FitConfig(
            step_size=config.fit_step_size,
            grad_clip=config.fit_grad_clip,
            max_iters=config.fit_max_iters,
            tol=0.0,
            omega_l2_reg=0.0,  # No L2 regularization (softmax constraint handles this)
            omega_constraint=config.omega_constraint,
        )

        omega_hat = fit_omega(
            X=Xs_cov,
            Y=Y_cov,
            omega0=np.ones(X_cov.shape[1], dtype=float),
            train_idx=train_idx,
            cfg=cov_cfg,
            fit_cfg=fit_cfg,
        )
        logger.info("Learned omega: %s (sum=%.4f)", omega_hat, omega_hat.sum())

    # -------------------------------------------------------------------------
    # Step 5: Compute mu, Sigma for ALL timestamps
    # -------------------------------------------------------------------------
    logger.info("Computing conditional moments for all timestamps...")

    mu_all, sigma_all = predict_mu_sigma_topk_cross(
        X_query=Xs_cov,
        X_ref=Xs_train,
        Y_ref=Y_train,
        omega=omega_hat,
        cfg=CovPredictConfig(tau=config.tau, ridge=config.ridge),
        k=config.k,
        return_type="numpy",
    )
    logger.info("Moments computed: mu=%s, sigma=%s", mu_all.shape, sigma_all.shape)

    # -------------------------------------------------------------------------
    # Build conformal totals df (alpha-independent preparation)
    # -------------------------------------------------------------------------
    logger.info("Building conformal totals dataframe...")
    df_tot = build_conformal_totals_df(actuals, forecasts)
    df_tot = _add_conformal_features(df_tot)

    return {
        "mu_all": mu_all,
        "sigma_all": sigma_all,
        "omega_hat": omega_hat,
        "Xs_cov": Xs_cov,
        "times_cov": times_cov,
        "times_train": times_train,
        "df_tot": df_tot,
        "y_cols": y_cols,
        "x_cols": x_cols,
    }


# ------------------------------------------------------------------------------
# Generate uncertainty sets for a single alpha (alpha-dependent)
# ------------------------------------------------------------------------------


def generate_uncertainty_sets_for_alpha(
    alpha_target: float,
    mu_all: np.ndarray,
    sigma_all: np.ndarray,
    times_cov: np.ndarray,
    times_train: np.ndarray,
    df_tot: pd.DataFrame,
    config: UncertaintySetConfig,
    omega_hat: np.ndarray,
    y_cols: list,
    x_cols: list,
    output_dir: Path,
    output_name: str = "sigma_rho",
) -> Path:
    """
    Generate uncertainty sets for a single alpha value using pre-computed
    covariance data. Only runs the alpha-dependent steps (conformal training
    + rho computation + save NPZ).

    Parameters
    ----------
    alpha_target : float
        Conformal coverage target (e.g. 0.90, 0.95, 0.99).
    mu_all : np.ndarray (N, K)
        Pre-computed conditional means.
    sigma_all : np.ndarray (N, K, K)
        Pre-computed conditional covariances.
    times_cov : np.ndarray
        Timestamps aligned to mu_all/sigma_all rows.
    times_train : np.ndarray
        Training period timestamps (for conformal split).
    df_tot : pd.DataFrame
        Conformal totals dataframe (with features already added).
    config : UncertaintySetConfig
        Used for conformal model params (n_bins, quantile_alpha, etc.).
    omega_hat : np.ndarray (D,)
        Learned feature weights.
    y_cols : list[str]
        Wind resource column names.
    x_cols : list[str]
        Feature column names.
    output_dir : Path
        Directory to save output NPZ.
    output_name : str
        Base name for the output file (without extension).

    Returns
    -------
    Path
        Path to the generated NPZ file.
    """
    _ensure_dir(output_dir)

    # -------------------------------------------------------------------------
    # Step 4: Train conformal model at this alpha
    # -------------------------------------------------------------------------
    logger.info(
        "Training conformal model for alpha=%.3f...", alpha_target
    )
    logger.info("  Features: %s", config.conformal_feature_cols)

    train_end_time = times_train[-1]
    df_train_conf = df_tot[df_tot["TIME_HOURLY"] <= train_end_time].copy()

    bundle, conf_metrics, _df_test = train_wind_lower_model_conformal_binned(
        df_train_conf,
        feature_cols=config.conformal_feature_cols,
        target_col="y",
        scale_col="ens_std",
        alpha_target=alpha_target,
        quantile_alpha=config.quantile_alpha,
        binning="y_pred",
        n_bins=config.n_bins,
        safety_margin=config.safety_margin,
        model_kwargs=config.conformal_model_kwargs,
    )
    logger.info(
        "Conformal trained (alpha=%.3f): coverage=%.2f%%, rmse=%.4f",
        alpha_target,
        100.0 * conf_metrics["coverage"],
        conf_metrics["rmse"],
    )

    # Get conformal bounds for ALL timestamps
    df_all_conf = df_tot.copy()
    df_pred_all = bundle.predict_df(df_all_conf)
    bounds_all = df_pred_all.set_index("TIME_HOURLY")["y_pred_conf"].sort_index()

    # -------------------------------------------------------------------------
    # Step 6: Compute rho for each timestamp
    # -------------------------------------------------------------------------
    logger.info("Computing rho (ellipsoid radii) for alpha=%.3f...", alpha_target)

    valid_indices = []
    mu_list = []
    sigma_list = []
    rho_list = []
    times_list = []
    missing_bounds = 0

    for i, t in enumerate(times_cov):
        if t not in bounds_all.index:
            missing_bounds += 1
            continue

        total_lower_bound = float(bounds_all.loc[t])
        mu_t = mu_all[i]
        sigma_t = sigma_all[i]

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

        valid_indices.append(i)
        mu_list.append(mu_t)
        sigma_list.append(sigma_t)
        rho_list.append(rho)
        times_list.append(t)

    if missing_bounds:
        logger.info(
            "Skipped %d timestamps with missing conformal bounds.", missing_bounds
        )

    logger.info(
        "Generated uncertainty sets for %d hours (alpha=%.3f).",
        len(valid_indices),
        alpha_target,
    )

    # Convert to arrays
    mu_arr = np.stack(mu_list, axis=0)
    sigma_arr = np.stack(sigma_list, axis=0)
    rho_arr = np.array(rho_list, dtype=float)
    times_arr = pd.DatetimeIndex(times_list)

    times_np = (
        times_arr.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    )

    # -------------------------------------------------------------------------
    # Step 7: Save outputs
    # -------------------------------------------------------------------------
    out_npz = output_dir / f"{output_name}.npz"
    out_metadata = output_dir / "metadata.json"

    logger.info("Saving to %s", out_npz)

    np.savez_compressed(
        out_npz,
        mu=mu_arr,
        sigma=sigma_arr,
        rho=rho_arr,
        omega=omega_hat,
        times=times_np,
        y_cols=np.array(y_cols, dtype=object),
        x_cols=np.array(x_cols, dtype=object),
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "rts4_constellation_v1",
        "n_hours": len(rho_arr),
        "n_wind_resources": len(y_cols),
        "wind_resource_ids": y_cols,
        "feature_cols": x_cols,
        "conformal_feature_cols": config.conformal_feature_cols,
        "config": {
            "tau": config.tau,
            "ridge": config.ridge,
            "k": config.k,
            "alpha_target": alpha_target,
            "n_bins": config.n_bins,
            "quantile_alpha": config.quantile_alpha,
            "safety_margin": config.safety_margin,
            "scaler_type": config.scaler_type,
            "feature_set": config.feature_set,
            "train_frac": config.train_frac,
            "omega_constraint": config.omega_constraint,
        },
        "conformal_metrics": {
            "coverage": conf_metrics["coverage"],
            "rmse": conf_metrics["rmse"],
        },
        "omega": omega_hat.tolist(),
        "rho_stats": {
            "min": float(rho_arr.min()),
            "max": float(rho_arr.max()),
            "mean": float(rho_arr.mean()),
            "std": float(rho_arr.std()),
        },
    }

    with open(out_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", out_metadata)
    logger.info(
        "Success: generated %d hours of uncertainty sets (alpha=%.3f).",
        len(rho_arr),
        alpha_target,
    )

    return out_npz


# ------------------------------------------------------------------------------
# Original top-level function (unchanged API, now delegates to helpers)
# ------------------------------------------------------------------------------


def generate_uncertainty_sets(
    config: UncertaintySetConfig,
    paths: CachedPaths,
    output_dir: Path,
    output_name: str = "sigma_rho",
) -> Path:
    """
    Generate pre-computed (mu, Sigma, rho) for all available hours.

    Steps:
    1. Load actuals and forecasts
    2. Build X/Y matrices for covariance
    3. Load or train omega
    4. Train conformal model for lower bounds
    5. For each hour: compute mu, Sigma, rho
    6. Save to NPZ + metadata

    Parameters
    ----------
    config : UncertaintySetConfig
        Configuration for generation
    paths : CachedPaths
        Data file paths
    output_dir : Path
        Directory to save outputs
    output_name : str
        Base name for output files (without extension)

    Returns
    -------
    Path
        Path to generated NPZ file
    """
    _ensure_dir(output_dir)

    # Phase 1: alpha-independent covariance pre-computation
    cov = pre_compute_covariance(config, paths)

    # Phase 2: alpha-dependent conformal + rho + save
    return generate_uncertainty_sets_for_alpha(
        alpha_target=config.alpha_target,
        mu_all=cov["mu_all"],
        sigma_all=cov["sigma_all"],
        times_cov=cov["times_cov"],
        times_train=cov["times_train"],
        df_tot=cov["df_tot"],
        config=config,
        omega_hat=cov["omega_hat"],
        y_cols=cov["y_cols"],
        x_cols=cov["x_cols"],
        output_dir=output_dir,
        output_name=output_name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-computed uncertainty sets for ARUC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output paths
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "uncertainty_sets_rts4",
        help="Output directory for NPZ and metadata files",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="sigma_rho",
        help="Base name for output files (without extension)",
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Data directory containing actuals/forecasts parquet files",
    )

    # Covariance parameters
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Kernel bandwidth parameter",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Ridge regularization for covariance",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=128,
        help="Number of neighbors for local covariance estimation",
    )

    # Omega
    parser.add_argument(
        "--omega-path",
        type=Path,
        default=None,
        help="Path to pre-trained omega.npy file",
    )
    parser.add_argument(
        "--retrain-omega",
        action="store_true",
        help="Retrain omega even if omega-path exists",
    )
    parser.add_argument(
        "--omega-constraint",
        type=str,
        choices=["none", "softmax", "simplex", "normalize"],
        default="softmax",
        help="Omega constraint type (softmax ensures sum(omega)=1)",
    )

    # Conformal parameters
    parser.add_argument(
        "--alpha-target",
        type=float,
        default=0.90,
        help="Target coverage level for conformal prediction",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=1,
        help="Number of bins for conformal (1=global)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.0,
        help="Safety margin for conformal prediction",
    )

    # Data processing
    parser.add_argument(
        "--scaler-type",
        type=str,
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Scaler type for features",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.75,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=list(FEATURE_BUILDERS.keys()),
        default="focused_2d",
        help="Covariance feature set to use",
    )

    args = parser.parse_args()

    # Build paths (use RTS4 data by default)
    data_dir = args.data_dir
    paths = CachedPaths(
        actuals_parquet=data_dir / "actuals_filtered_rts4_constellation_v1.parquet",
        forecasts_parquet=data_dir / "forecasts_filtered_rts4_constellation_v1.parquet",
        output_dir=args.output_dir,
    )

    # Build config
    config = UncertaintySetConfig(
        tau=args.tau,
        ridge=args.ridge,
        k=args.k,
        omega_path=args.omega_path,
        retrain_omega=args.retrain_omega,
        omega_constraint=args.omega_constraint,
        alpha_target=args.alpha_target,
        n_bins=args.n_bins,
        safety_margin=args.safety_margin,
        scaler_type=args.scaler_type,
        train_frac=args.train_frac,
        feature_set=args.feature_set,
    )

    # Generate
    generate_uncertainty_sets(
        config=config,
        paths=paths,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
