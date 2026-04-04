"""
Per-line conformal calibration of ellipsoid radii for transmission flow constraints.

Given resource-level wind actuals/forecasts and PTDFs, this module calibrates a
separate rho_l for each monitored transmission line using conformal prediction on
flow residuals. The result is a per-constraint coverage guarantee:

    P(|phi_l - phi_hat_l| <= rho_l * sigma_flow_l) >= 1 - alpha

where sigma_flow_l(t) = sqrt(H_l^T Sigma_t H_l) is the conditional flow std
derived from the resource-level covariance.

The per-line rho_l replaces the single system-level rho in the ARUC flow
constraints, giving tighter (or appropriately looser) uncertainty budgets
per constraint while preserving the existing SOC machinery.

Usage
-----
    from conformal_flow import calibrate_flow_rho_per_line

    result = calibrate_flow_rho_per_line(
        Y_actual=Y_actual,          # (T, K) per-resource actuals
        Y_forecast=Y_forecast,      # (T, K) per-resource forecasts (ensemble mean)
        H_wind=H_wind,              # (L, K) PTDF rows restricted to wind buses
        Sigma=Sigma,                # (T, K, K) conditional covariance
        cal_indices=cal_indices,    # calibration set indices
        alpha=0.95,
    )
    rho_lines = result.rho_lines    # (L_valid,)
    line_mask = result.valid_mask   # (L,) bool — which lines are non-degenerate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils import _conformal_q_level

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class FlowConformalResult:
    """Result of per-line conformal calibration."""

    rho_lines: np.ndarray
    """(L_valid,) per-line ellipsoid radii (dimensionless)."""

    valid_mask: np.ndarray
    """(L,) bool — True for lines with non-degenerate wind flow sensitivity."""

    line_ids: list
    """Line identifiers for valid lines (length L_valid)."""

    alpha: float
    """Target coverage level used for calibration."""

    # Diagnostics
    scores_cal: np.ndarray
    """(L_valid, n_cal) nonconformity scores on calibration set."""

    coverage_cal: np.ndarray
    """(L_valid,) empirical coverage at rho_lines on calibration set (should be >= alpha)."""

    sigma_flow_stats: dict
    """Per-line summary stats of sigma_flow (for diagnostics)."""


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
    # phi[l, t] = sum_k H_wind[l, k] * generation[t, k]
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
    T = Sigma.shape[0]
    L = H_wind.shape[0]

    # Vectorized: for each t, compute H @ Sigma_t @ H^T diagonal
    # H_wind: (L, K), Sigma: (T, K, K)
    # H_Sigma = H_wind @ Sigma  -> (T, L, K) via einsum
    # variance = diag(H_Sigma @ H^T) -> (T, L) via einsum
    # var_lt = sum_k sum_j H[l,k] * Sigma[t,k,j] * H[l,j]
    variance = np.einsum("lk,tkj,lj->lt", H_wind, Sigma, H_wind)  # (L, T)

    # Clamp tiny negatives from numerical noise
    np.maximum(variance, 0.0, out=variance)
    return np.sqrt(variance)


def compute_nonconformity_scores(
    residuals: np.ndarray,
    sigma_flow: np.ndarray,
    min_sigma: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized symmetric nonconformity scores.

    s_l(t) = |e_l(t)| / sigma_flow_l(t)

    Parameters
    ----------
    residuals : (L, T) flow residuals (actual - forecast)
    sigma_flow : (L, T) per-line flow std from covariance
    min_sigma : float
        Floor for sigma_flow to avoid division by near-zero.
        Lines where max_t sigma_flow_l(t) < min_sigma are flagged as degenerate.

    Returns
    -------
    scores : (L, T) nonconformity scores (inf for degenerate lines)
    valid_mask : (L,) bool — True for non-degenerate lines
    """
    L, T = residuals.shape

    # Identify degenerate lines: max sigma_flow across time is too small
    max_sigma = sigma_flow.max(axis=1)  # (L,)
    valid_mask = max_sigma >= min_sigma

    # Compute scores with safe denominator
    sigma_safe = np.maximum(sigma_flow, min_sigma)
    scores = np.abs(residuals) / sigma_safe

    return scores, valid_mask


def calibrate_rho_per_line(
    scores: np.ndarray,
    cal_indices: np.ndarray,
    alpha: float,
    safety_margin: float = 0.0,
) -> np.ndarray:
    """
    Compute per-line conformal quantile (rho_l).

    Parameters
    ----------
    scores : (L, T) nonconformity scores
    cal_indices : (n_cal,) indices into T dimension for calibration set
    alpha : float, target coverage (e.g. 0.95)
    safety_margin : float, extra buffer (default 0.0)

    Returns
    -------
    rho_lines : (L,) per-line conformal radii
    """
    L = scores.shape[0]
    n_cal = len(cal_indices)
    q_level = _conformal_q_level(n_cal, alpha, safety_margin)

    scores_cal = scores[:, cal_indices]  # (L, n_cal)
    rho_lines = np.quantile(scores_cal, q_level, axis=1)  # (L,)
    return rho_lines


def evaluate_coverage(
    scores: np.ndarray,
    rho_lines: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Compute empirical coverage on a given set of indices.

    Parameters
    ----------
    scores : (L, T) nonconformity scores
    rho_lines : (L,) per-line radii
    indices : (n,) indices into T dimension

    Returns
    -------
    coverage : (L,) fraction of indices where score <= rho_l
    """
    scores_subset = scores[:, indices]  # (L, n)
    covered = scores_subset <= rho_lines[:, np.newaxis]  # (L, n)
    return covered.mean(axis=1)


# -----------------------------------------------------------------------------
# High-level pipeline
# -----------------------------------------------------------------------------


def calibrate_flow_rho_per_line(
    Y_actual: np.ndarray,
    Y_forecast: np.ndarray,
    H_wind: np.ndarray,
    Sigma: np.ndarray,
    cal_indices: np.ndarray,
    alpha: float,
    line_ids: Optional[list] = None,
    min_sigma: float = 1e-6,
    safety_margin: float = 0.0,
    test_indices: Optional[np.ndarray] = None,
) -> FlowConformalResult:
    """
    Full pipeline: calibrate per-line rho from data.

    Parameters
    ----------
    Y_actual : (T, K) per-resource wind actuals (MW)
    Y_forecast : (T, K) per-resource wind forecasts — ensemble mean (MW)
    H_wind : (L, K) PTDF sub-matrix for wind generator buses
    Sigma : (T, K, K) time-varying conditional covariance
    cal_indices : (n_cal,) calibration set indices (into T dimension)
    alpha : float
        Target coverage level (e.g. 0.95 for 95% coverage guarantee).
    line_ids : list, optional
        Line identifiers (length L). If None, uses 0..L-1.
    min_sigma : float
        Threshold for degenerate line detection.
    safety_margin : float
        Extra conformal buffer (default 0.0).
    test_indices : (n_test,) optional
        If provided, compute test-set coverage in diagnostics.

    Returns
    -------
    FlowConformalResult
    """
    T, K = Y_actual.shape
    L = H_wind.shape[0]

    if line_ids is None:
        line_ids = list(range(L))
    assert len(line_ids) == L

    logger.info(
        "Calibrating per-line rho: L=%d lines, K=%d resources, T=%d timesteps, "
        "n_cal=%d, alpha=%.3f",
        L, K, T, len(cal_indices), alpha,
    )

    # 1. Flow contributions
    phi_actual = compute_flow_contributions(H_wind, Y_actual)  # (L, T)
    phi_forecast = compute_flow_contributions(H_wind, Y_forecast)  # (L, T)

    # 2. Flow residuals
    residuals = phi_actual - phi_forecast  # (L, T)

    # 3. Flow sigma from covariance
    sigma_flow = compute_flow_sigma(H_wind, Sigma)  # (L, T)

    # 4. Nonconformity scores
    scores, valid_mask = compute_nonconformity_scores(
        residuals, sigma_flow, min_sigma=min_sigma,
    )

    n_degenerate = int((~valid_mask).sum())
    if n_degenerate:
        logger.info(
            "Filtered %d degenerate lines (max sigma_flow < %.1e)", n_degenerate, min_sigma,
        )

    # 5. Restrict to valid lines
    scores_valid = scores[valid_mask]  # (L_valid, T)
    valid_line_ids = [lid for lid, v in zip(line_ids, valid_mask) if v]

    # 6. Per-line conformal quantile
    rho_lines = calibrate_rho_per_line(
        scores_valid, cal_indices, alpha, safety_margin,
    )

    # 7. Diagnostics
    coverage_cal = evaluate_coverage(scores_valid, rho_lines, cal_indices)
    scores_cal = scores_valid[:, cal_indices]

    sigma_flow_valid = sigma_flow[valid_mask]
    sigma_flow_stats = {
        "mean": float(sigma_flow_valid.mean()),
        "std": float(sigma_flow_valid.std()),
        "min": float(sigma_flow_valid.min()),
        "max": float(sigma_flow_valid.max()),
        "median": float(np.median(sigma_flow_valid)),
    }

    logger.info(
        "Per-line rho: min=%.3f, median=%.3f, max=%.3f (L_valid=%d)",
        rho_lines.min(), np.median(rho_lines), rho_lines.max(), len(rho_lines),
    )
    logger.info(
        "Cal coverage: min=%.3f, mean=%.3f, max=%.3f",
        coverage_cal.min(), coverage_cal.mean(), coverage_cal.max(),
    )

    if test_indices is not None and len(test_indices) > 0:
        coverage_test = evaluate_coverage(scores_valid, rho_lines, test_indices)
        logger.info(
            "Test coverage: min=%.3f, mean=%.3f, max=%.3f",
            coverage_test.min(), coverage_test.mean(), coverage_test.max(),
        )

    return FlowConformalResult(
        rho_lines=rho_lines,
        valid_mask=valid_mask,
        line_ids=valid_line_ids,
        alpha=alpha,
        scores_cal=scores_cal,
        coverage_cal=coverage_cal,
        sigma_flow_stats=sigma_flow_stats,
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

    # Add project root to path so we can import network_ptdf
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from network_ptdf import build_dc_ptdf

    data_dir = Path(rts_data_dir)
    buses_df = pd.read_csv(data_dir / "bus.csv")
    branches_df = pd.read_csv(data_dir / "branch.csv")
    gen_df = pd.read_csv(data_dir / "gen.csv")

    # Build full PTDF
    PTDF, Fmax, bus_ids, line_ids = build_dc_ptdf(buses_df, branches_df)

    # Identify wind generators and their buses
    if wind_resource_ids is None:
        wind_gens = gen_df[gen_df["Fuel"].str.upper() == "WIND"].sort_values("GEN UID")
        wind_resource_ids = wind_gens["GEN UID"].tolist()

    gen_bus_map = dict(zip(gen_df["GEN UID"], gen_df["Bus ID"]))
    wind_bus_ids = [gen_bus_map[gid] for gid in wind_resource_ids]

    # Extract wind columns from PTDF
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

    # Actuals: pivot to wide (T, K)
    act_wide = (
        actuals.groupby([time_col, resource_col])["ACTUAL"]
        .mean()
        .reset_index()
        .pivot(index=time_col, columns=resource_col, values="ACTUAL")
        .sort_index()
    )

    # Forecasts: ensemble mean per (time, resource), then pivot
    fc_mean = (
        forecasts.groupby([time_col, resource_col])["FORECAST"]
        .mean()
        .reset_index()
        .pivot(index=time_col, columns=resource_col, values="FORECAST")
        .sort_index()
    )

    # Align times
    common_times = act_wide.index.intersection(fc_mean.index)
    act_wide = act_wide.loc[common_times]
    fc_mean = fc_mean.loc[common_times]

    # Select and order resources
    if wind_resource_ids is not None:
        missing = [r for r in wind_resource_ids if r not in act_wide.columns]
        if missing:
            raise ValueError(f"Resources not found in data: {missing}")
        act_wide = act_wide[wind_resource_ids]
        fc_mean = fc_mean[wind_resource_ids]
    else:
        # Use sorted intersection of columns
        common_resources = sorted(
            set(act_wide.columns) & set(fc_mean.columns)
        )
        act_wide = act_wide[common_resources]
        fc_mean = fc_mean[common_resources]

    # Drop rows with NaN
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
