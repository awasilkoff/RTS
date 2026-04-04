"""
Per-line conformal calibration of ellipsoid radii for transmission flow constraints.

Given resource-level wind actuals/forecasts and PTDFs, this module calibrates a
separate rho_l(t) for each monitored transmission line using conformal prediction
on flow residuals.  The structure mirrors the system-level conformal pipeline
exactly:

    System level:
        1. Train quantile model on total wind, calibrate conformal correction
        2. Predict time-varying lower bound lb(t) for all hours
        3. rho(t) = (e^T mu_t - lb(t)) / sqrt(e^T Sigma_t e)

    Per line l:
        1. Train quantile model on phi_l = H_l @ actual, calibrate conformal correction
        2. Predict time-varying lower bound lb_l(t) for all hours
        3. rho_l(t) = (H_l^T mu_t - lb_l(t)) / sqrt(H_l^T Sigma_t H_l)

Each rho_l(t) carries a marginal coverage guarantee:
    P(phi_l(t) >= lb_l(t)) >= 1 - alpha   for each line l.

The per-line rho_l(t) replaces the single system-level rho(t) in the ARUC flow
constraints, giving tighter (or appropriately looser) uncertainty budgets per
constraint while preserving the existing SOC machinery.

Usage
-----
    from conformal_flow import calibrate_flow_rho_per_line

    result = calibrate_flow_rho_per_line(
        df_features=df_tot,         # DataFrame with conformal features
        Y_actual=Y_actual,          # (T, K) per-resource actuals
        H_wind=H_wind,              # (L, K) PTDF rows restricted to wind buses
        Sigma=Sigma,                # (T, K, K) conditional covariance
        mu=mu,                      # (T, K) conditional means
        config=config,              # UncertaintySetConfig
        alpha=0.95,
        times_train=times_train,
    )
    rho_lines = result.rho_lines    # (L_valid, T) time-varying per-line rho
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from conformal_prediction import train_wind_lower_model_conformal_binned
from covariance_optimization import implied_rho_from_total_lower_bound

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class FlowConformalResult:
    """Result of per-line conformal calibration."""

    rho_lines: np.ndarray
    """(L_valid, T) time-varying per-line ellipsoid radii (dimensionless)."""

    valid_mask: np.ndarray
    """(L,) bool — True for lines with non-degenerate wind flow sensitivity."""

    line_ids: list
    """Line identifiers for valid lines (length L_valid)."""

    alpha: float
    """Target coverage level used for calibration."""

    # Diagnostics
    coverage_per_line: np.ndarray
    """(L_valid,) empirical coverage on conformal test set."""

    sigma_flow_stats: dict
    """Per-line summary stats of sigma_flow (for diagnostics)."""

    rho_stats: dict
    """Summary stats of rho_lines across lines and time."""


# -----------------------------------------------------------------------------
# Core computation functions
# -----------------------------------------------------------------------------


def compute_wind_ptdf(
    PTDF: np.ndarray,
    bus_ids: list,
    wind_bus_ids: list,
) -> np.ndarray:
    """
    Extract PTDF sub-matrix for wind generator buses.

    Parameters
    ----------
    PTDF : (L, N) full PTDF matrix
    bus_ids : list of length N, canonical bus ordering matching PTDF columns
    wind_bus_ids : list of length K, bus IDs for each wind generator

    Returns
    -------
    H_wind : (L, K) PTDF columns corresponding to wind buses
    """
    bus_idx_map = {bid: i for i, bid in enumerate(bus_ids)}
    col_indices = []
    for bid in wind_bus_ids:
        if bid not in bus_idx_map:
            raise ValueError(f"Wind bus {bid} not found in bus_ids")
        col_indices.append(bus_idx_map[bid])
    return PTDF[:, col_indices]


def compute_flow_contributions(
    H_wind: np.ndarray,
    generation: np.ndarray,
) -> np.ndarray:
    """
    Compute renewable flow contributions on each line.

    Parameters
    ----------
    H_wind : (L, K) PTDF sub-matrix for wind buses
    generation : (T, K) per-resource generation (MW)

    Returns
    -------
    phi : (L, T) renewable flow contribution per line per timestep
    """
    return H_wind @ generation.T


def compute_flow_sigma(
    H_wind: np.ndarray,
    Sigma: np.ndarray,
) -> np.ndarray:
    """
    Compute per-line flow standard deviation from resource covariance.

    sigma_flow_l(t) = sqrt( H_l^T Sigma_t H_l )

    Parameters
    ----------
    H_wind : (L, K) PTDF sub-matrix for wind buses
    Sigma : (T, K, K) time-varying conditional covariance

    Returns
    -------
    sigma_flow : (L, T) per-line flow std (MW)
    """
    # var_lt = sum_k sum_j H[l,k] * Sigma[t,k,j] * H[l,j]
    variance = np.einsum("lk,tkj,lj->lt", H_wind, Sigma, H_wind)
    np.maximum(variance, 0.0, out=variance)
    return np.sqrt(variance)


def filter_degenerate_lines(
    sigma_flow: np.ndarray,
    min_sigma: float = 1e-6,
) -> np.ndarray:
    """
    Identify lines with negligible wind flow sensitivity.

    Parameters
    ----------
    sigma_flow : (L, T) per-line flow std
    min_sigma : float
        Lines where max_t sigma_flow_l(t) < min_sigma are degenerate.

    Returns
    -------
    valid_mask : (L,) bool — True for non-degenerate lines
    """
    max_sigma = sigma_flow.max(axis=1)
    return max_sigma >= min_sigma


# -----------------------------------------------------------------------------
# Per-line conformal model training
# -----------------------------------------------------------------------------


def _build_per_line_df(
    df_features: pd.DataFrame,
    phi_l: np.ndarray,
    sigma_flow_l: np.ndarray,
    time_col: str = "TIME_HOURLY",
) -> pd.DataFrame:
    """
    Build a per-line conformal DataFrame by replacing target and scale.

    Takes the system-level features DataFrame and substitutes:
      - y -> phi_l (flow contribution on this line)
      - ens_std -> sigma_flow_l (conditional flow std on this line)

    Parameters
    ----------
    df_features : DataFrame with system-level conformal features
        Must contain: TIME_HOURLY, ens_mean, ens_std, hour, dow, etc.
    phi_l : (T,) flow contribution actuals for this line
    sigma_flow_l : (T,) conditional flow std for this line

    Returns
    -------
    DataFrame with y and ens_std replaced, ready for conformal training.
    """
    df = df_features.copy()
    df["y"] = phi_l
    df["ens_std"] = sigma_flow_l
    # Drop rows where target or scale is NaN/non-finite
    valid = np.isfinite(df["y"]) & np.isfinite(df["ens_std"]) & (df["ens_std"] > 0)
    return df.loc[valid].reset_index(drop=True)


def train_per_line_conformal(
    df_line: pd.DataFrame,
    feature_cols: list[str],
    alpha_target: float,
    config_kwargs: dict,
) -> tuple:
    """
    Train a conformal model for a single line.

    Wraps train_wind_lower_model_conformal_binned with per-line data.

    Returns
    -------
    (bundle, metrics, df_test) — same as train_wind_lower_model_conformal_binned
    """
    return train_wind_lower_model_conformal_binned(
        df_line,
        feature_cols=feature_cols,
        target_col="y",
        scale_col="ens_std",
        alpha_target=alpha_target,
        **config_kwargs,
    )


# -----------------------------------------------------------------------------
# High-level pipeline
# -----------------------------------------------------------------------------


def calibrate_flow_rho_per_line(
    df_features: pd.DataFrame,
    Y_actual: np.ndarray,
    H_wind: np.ndarray,
    Sigma: np.ndarray,
    mu: np.ndarray,
    feature_cols: list[str],
    alpha: float,
    times_train: np.ndarray,
    line_ids: Optional[list] = None,
    min_sigma: float = 1e-6,
    conformal_kwargs: Optional[dict] = None,
) -> FlowConformalResult:
    """
    Full pipeline: calibrate time-varying per-line rho from data.

    For each non-degenerate line l:
      1. Train a quantile regression model on phi_l(t) = H_l @ actual(t)
      2. Calibrate conformal correction on held-out calibration set
      3. Predict lower bounds lb_l(t) for all timesteps
      4. Convert: rho_l(t) = (H_l^T mu_t - lb_l(t)) / sqrt(H_l^T Sigma_t H_l)

    Parameters
    ----------
    df_features : pd.DataFrame
        System-level conformal features DataFrame. Must contain TIME_HOURLY,
        ens_mean, ens_std, hour, dow, and all columns in feature_cols.
        Rows must align 1:1 with Y_actual, Sigma, mu.
    Y_actual : (T, K) per-resource wind actuals (MW)
    H_wind : (L, K) PTDF sub-matrix for wind generator buses
    Sigma : (T, K, K) time-varying conditional covariance
    mu : (T, K) conditional means from covariance model
    feature_cols : list[str]
        Feature columns for the conformal quantile model.
    alpha : float
        Target coverage level (e.g. 0.95).
    times_train : np.ndarray
        Training period timestamps — conformal model is trained on
        df_features[TIME_HOURLY <= times_train[-1]].
    line_ids : list, optional
        Line identifiers (length L). If None, uses 0..L-1.
    min_sigma : float
        Threshold for degenerate line detection.
    conformal_kwargs : dict, optional
        Override kwargs passed to train_wind_lower_model_conformal_binned.
        Defaults: n_bins=1, quantile_alpha=1-alpha, safety_margin=0.0.

    Returns
    -------
    FlowConformalResult with rho_lines shape (L_valid, T)
    """
    T, K = Y_actual.shape
    L = H_wind.shape[0]

    if line_ids is None:
        line_ids = list(range(L))
    assert len(line_ids) == L

    logger.info(
        "Calibrating per-line flow rho: L=%d lines, K=%d resources, T=%d timesteps, "
        "alpha=%.3f",
        L, K, T, alpha,
    )

    # 1. Flow contributions (actuals) for all lines
    phi_actual = compute_flow_contributions(H_wind, Y_actual)  # (L, T)

    # 2. Flow sigma from covariance
    sigma_flow = compute_flow_sigma(H_wind, Sigma)  # (L, T)

    # 3. Filter degenerate lines
    valid_mask = filter_degenerate_lines(sigma_flow, min_sigma=min_sigma)
    n_valid = int(valid_mask.sum())
    n_degenerate = L - n_valid
    if n_degenerate:
        logger.info(
            "Filtered %d degenerate lines (max sigma_flow < %.1e), %d lines remain",
            n_degenerate, min_sigma, n_valid,
        )

    valid_indices = np.where(valid_mask)[0]
    valid_line_ids = [line_ids[i] for i in valid_indices]

    # 4. Set up conformal kwargs
    default_kwargs = {
        "quantile_alpha": 1.0 - alpha,
        "n_bins": 1,
        "safety_margin": 0.0,
        "model_kwargs": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 32,
            "random_state": 42,
        },
    }
    if conformal_kwargs is not None:
        default_kwargs.update(conformal_kwargs)

    # Subset df_features to training window for conformal
    train_end_time = times_train[-1]
    df_train_window = df_features[df_features["TIME_HOURLY"] <= train_end_time].copy()

    # 5. Train per-line conformal models and compute rho_l(t)
    rho_lines_list = []
    coverage_list = []

    for idx_in_valid, l_idx in enumerate(valid_indices):
        lid = line_ids[l_idx]
        phi_l = phi_actual[l_idx]         # (T,)
        sf_l = sigma_flow[l_idx]          # (T,)
        H_l = H_wind[l_idx]              # (K,)

        # Build per-line training DataFrame
        df_line = _build_per_line_df(df_train_window, phi_l, sf_l)

        if len(df_line) < 50:
            logger.warning(
                "Line %s: only %d training samples, skipping", lid, len(df_line),
            )
            # Use system-level rho as fallback (zeros = will be filled by caller)
            rho_lines_list.append(np.zeros(T))
            coverage_list.append(0.0)
            continue

        # Train conformal model for this line
        try:
            bundle_l, metrics_l, _ = train_per_line_conformal(
                df_line,
                feature_cols=feature_cols,
                alpha_target=alpha,
                config_kwargs=default_kwargs,
            )
        except Exception as exc:
            logger.warning("Line %s: conformal training failed (%s), skipping", lid, exc)
            rho_lines_list.append(np.zeros(T))
            coverage_list.append(0.0)
            continue

        # Predict lower bounds for ALL timestamps
        df_all_line = _build_per_line_df(df_features, phi_l, sf_l)
        df_pred = bundle_l.predict_df(df_all_line)
        lb_all = df_pred["y_pred_conf"].to_numpy(dtype=float)  # (T,)

        # Convert lower bounds to rho_l(t) via implied_rho
        rho_l = np.zeros(T)
        for t in range(T):
            try:
                rho_l[t] = implied_rho_from_total_lower_bound(
                    Sigma=Sigma[t],
                    mean=mu[t],
                    total_lower_bound=float(lb_all[t]),
                    e=H_l,
                    clip_nonneg=True,
                )
            except (ValueError, np.linalg.LinAlgError):
                rho_l[t] = 0.0

        rho_lines_list.append(rho_l)
        coverage_list.append(float(metrics_l.get("coverage", 0.0)))

        if (idx_in_valid + 1) % 20 == 0 or idx_in_valid == 0:
            logger.info(
                "  Line %d/%d (%s): rho range [%.3f, %.3f], coverage=%.3f",
                idx_in_valid + 1, n_valid, lid,
                rho_l[rho_l > 0].min() if (rho_l > 0).any() else 0.0,
                rho_l.max(),
                metrics_l.get("coverage", 0.0),
            )

    # 6. Stack results
    rho_lines = np.stack(rho_lines_list, axis=0)  # (L_valid, T)
    coverage_per_line = np.array(coverage_list)

    # 7. Diagnostics
    sigma_flow_valid = sigma_flow[valid_mask]
    sigma_flow_stats = {
        "mean": float(sigma_flow_valid.mean()),
        "std": float(sigma_flow_valid.std()),
        "min": float(sigma_flow_valid.min()),
        "max": float(sigma_flow_valid.max()),
        "median": float(np.median(sigma_flow_valid)),
    }

    rho_positive = rho_lines[rho_lines > 0]
    rho_stats = {
        "min": float(rho_positive.min()) if rho_positive.size > 0 else 0.0,
        "max": float(rho_lines.max()),
        "mean": float(rho_lines.mean()),
        "median": float(np.median(rho_lines)),
        "std": float(rho_lines.std()),
        "frac_zero": float((rho_lines == 0).mean()),
    }

    logger.info(
        "Per-line rho: min=%.3f, median=%.3f, max=%.3f, frac_zero=%.3f (L_valid=%d)",
        rho_stats["min"], rho_stats["median"], rho_stats["max"],
        rho_stats["frac_zero"], n_valid,
    )
    logger.info(
        "Coverage: min=%.3f, mean=%.3f, max=%.3f",
        coverage_per_line.min(), coverage_per_line.mean(), coverage_per_line.max(),
    )

    return FlowConformalResult(
        rho_lines=rho_lines,
        valid_mask=valid_mask,
        line_ids=valid_line_ids,
        alpha=alpha,
        coverage_per_line=coverage_per_line,
        sigma_flow_stats=sigma_flow_stats,
        rho_stats=rho_stats,
    )


# -----------------------------------------------------------------------------
# RTS data helpers
# -----------------------------------------------------------------------------


def load_rts_wind_ptdf(
    rts_data_dir: str = "RTS_Data/SourceData",
    wind_resource_ids: Optional[list[str]] = None,
) -> tuple[np.ndarray, list, list, list]:
    """
    Build PTDF and extract wind-bus columns from raw RTS-GMLC CSV files.

    This is a convenience function for use within the uncertainty pipeline.
    It imports network_ptdf from the project root.

    Parameters
    ----------
    rts_data_dir : str
        Path to RTS_Data/SourceData directory containing bus.csv, branch.csv, gen.csv.
    wind_resource_ids : list[str], optional
        Wind generator IDs in the order matching the covariance Y columns
        (e.g. ["122_WIND_1", "303_WIND_1", "309_WIND_1", "317_WIND_1"]).
        If None, auto-detects wind generators from gen.csv.

    Returns
    -------
    H_wind : (L, K) PTDF sub-matrix for wind generator buses
    line_ids : list of line UIDs (length L)
    bus_ids : list of bus IDs (length N, canonical order)
    wind_bus_ids : list of bus IDs for each wind generator (length K, in resource order)
    """
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from network_ptdf import build_dc_ptdf

    data_dir = Path(rts_data_dir)
    buses_df = pd.read_csv(data_dir / "bus.csv")
    branches_df = pd.read_csv(data_dir / "branch.csv")
    gen_df = pd.read_csv(data_dir / "gen.csv")

    PTDF, Fmax, bus_ids, line_ids = build_dc_ptdf(buses_df, branches_df)

    if wind_resource_ids is None:
        wind_gens = gen_df[gen_df["Fuel"].str.upper() == "WIND"].sort_values("GEN UID")
        wind_resource_ids = wind_gens["GEN UID"].tolist()

    gen_bus_map = dict(zip(gen_df["GEN UID"], gen_df["Bus ID"]))
    wind_bus_ids = [gen_bus_map[gid] for gid in wind_resource_ids]

    H_wind = compute_wind_ptdf(PTDF, bus_ids, wind_bus_ids)

    return H_wind, line_ids, bus_ids, wind_bus_ids


def build_flow_forecast_matrices(
    actuals_parquet: str,
    forecasts_parquet: str,
    wind_resource_ids: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load per-resource actuals and ensemble-mean forecasts, aligned by time.

    Parameters
    ----------
    actuals_parquet : str or Path
        Path to actuals parquet file.
    forecasts_parquet : str or Path
        Path to forecasts parquet file.
    wind_resource_ids : list[str], optional
        If provided, restrict to these resource IDs (in this order).

    Returns
    -------
    Y_actual : (T, K) per-resource actuals (MW)
    Y_forecast : (T, K) per-resource ensemble mean forecasts (MW)
    times : (T,) datetime array
    resource_ids : list[str] of length K
    """
    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)

    time_col = "TIME_HOURLY"
    resource_col = "ID_RESOURCE"

    act_wide = (
        actuals.groupby([time_col, resource_col])["ACTUAL"]
        .mean()
        .reset_index()
        .pivot(index=time_col, columns=resource_col, values="ACTUAL")
        .sort_index()
    )

    fc_mean = (
        forecasts.groupby([time_col, resource_col])["FORECAST"]
        .mean()
        .reset_index()
        .pivot(index=time_col, columns=resource_col, values="FORECAST")
        .sort_index()
    )

    common_times = act_wide.index.intersection(fc_mean.index)
    act_wide = act_wide.loc[common_times]
    fc_mean = fc_mean.loc[common_times]

    if wind_resource_ids is not None:
        missing = [r for r in wind_resource_ids if r not in act_wide.columns]
        if missing:
            raise ValueError(f"Resources not found in data: {missing}")
        act_wide = act_wide[wind_resource_ids]
        fc_mean = fc_mean[wind_resource_ids]
    else:
        common_resources = sorted(set(act_wide.columns) & set(fc_mean.columns))
        act_wide = act_wide[common_resources]
        fc_mean = fc_mean[common_resources]

    valid = act_wide.notna().all(axis=1) & fc_mean.notna().all(axis=1)
    act_wide = act_wide.loc[valid]
    fc_mean = fc_mean.loc[valid]

    resource_ids = list(act_wide.columns)
    times = act_wide.index.values

    return (
        act_wide.to_numpy(dtype=np.float64),
        fc_mean.to_numpy(dtype=np.float64),
        times,
        resource_ids,
    )
