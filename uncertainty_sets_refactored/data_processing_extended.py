"""
Extended feature builders for learned omega visualization experiments.

Adds three new feature engineering approaches to explore omega learning capability:
1. Temporal nuisance (3D): [SYS_MEAN, SYS_STD, HOUR_SIN]
2. Per-resource (4D): [WIND_122_MEAN, WIND_309_MEAN, WIND_317_MEAN, HOUR_SIN]
3. Unscaled (2D): [SYS_MEAN_MW, SYS_STD_MW] in raw MW units
"""
from __future__ import annotations

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import logging

from data_processing import (
    build_features_system_total_mean_std,
    build_Y_actuals_matrix,
    build_features_per_unit_mean_std,
    build_XY_for_covariance_system_only,
)

logger = logging.getLogger(__name__)


def build_XY_temporal_nuisance_3d(
    forecasts_filtered: pd.DataFrame,
    actuals_filtered: pd.DataFrame,
    *,
    drop_any_nan_rows: bool = True,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    forecast_col: str = "FORECAST",
    actual_col: str = "ACTUAL",
    model_col: Optional[str] = "MODEL",
    return_frames: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]] | tuple:
    """
    Option 1: [SYS_MEAN, SYS_STD, HOUR_SIN] - 3D with temporal nuisance.

    Expected: Learned omega ≈ [α, β, ~0] downweights weakly-relevant temporal.

    Returns:
      X: (T, 3) float64 - [SYS_MEAN, SYS_STD, HOUR_SIN]
      Y: (T, M) float64 - per-unit actuals
      times: DatetimeIndex
      x_cols: list[str]
      y_cols: list[str]

    If return_frames=True, also returns (feat_df_aligned, Y_df_aligned) at the end.
    """
    # Get system-only features (SYS_MEAN, SYS_STD)
    feat = build_features_system_total_mean_std(
        forecasts_filtered=forecasts_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=forecast_col,
        model_col=model_col,
    )

    # Add HOUR_SIN temporal feature
    hour = feat.index.hour.to_numpy()
    feat["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24.0)

    # Get targets
    Ywide = build_Y_actuals_matrix(
        actuals_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=actual_col,
    )

    # Align
    common = feat.index.intersection(Ywide.index)
    feat = feat.loc[common].sort_index()
    Ywide = Ywide.loc[common].sort_index()

    if drop_any_nan_rows:
        mask = feat.notna().all(axis=1) & Ywide.notna().all(axis=1)
        dropped = int((~mask).sum())
        if dropped:
            logger.info(
                "build_XY_temporal_nuisance_3d: dropping %d rows due to NaNs", dropped
            )
        feat = feat.loc[mask]
        Ywide = Ywide.loc[mask]

    X = feat.to_numpy(dtype=np.float64)
    Y = Ywide.to_numpy(dtype=np.float64)
    times = feat.index
    x_cols = feat.columns.tolist()
    y_cols = [str(c) for c in Ywide.columns.tolist()]

    if return_frames:
        return X, Y, times, x_cols, y_cols, feat, Ywide

    return X, Y, times, x_cols, y_cols


def build_XY_per_resource_4d(
    forecasts_filtered: pd.DataFrame,
    actuals_filtered: pd.DataFrame,
    *,
    include_hour: bool = True,
    drop_any_nan_rows: bool = True,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    forecast_col: str = "FORECAST",
    actual_col: str = "ACTUAL",
    model_col: Optional[str] = "MODEL",
    return_frames: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]] | tuple:
    """
    Option 3: [WIND_122_MEAN, WIND_309_MEAN, WIND_317_MEAN, HOUR_SIN] - 4D per-resource.

    Natural interpretation: different farms have different predictive value.
    Expected: Learned ω discovers which farm matters most for covariance.

    Returns:
      X: (T, 3 or 4) float64 - per-resource means, optionally with HOUR_SIN
      Y: (T, M) float64 - per-unit actuals
      times: DatetimeIndex
      x_cols: list[str]
      y_cols: list[str]

    If return_frames=True, also returns (feat_df_aligned, Y_df_aligned) at the end.
    """
    # Get per-unit features (includes unit means, but NOT system-level)
    feat = build_features_per_unit_mean_std(
        forecasts_filtered=forecasts_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=forecast_col,
        model_col=model_col,
        include_system=False,  # Only per-unit, not system
    )

    # Extract only MEAN columns (drop STD)
    mean_cols = [c for c in feat.columns if c.endswith("_MEAN")]
    feat = feat[mean_cols]

    # Add HOUR_SIN temporal feature if requested
    if include_hour:
        hour = feat.index.hour.to_numpy()
        feat["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24.0)

    # Get targets
    Ywide = build_Y_actuals_matrix(
        actuals_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=actual_col,
    )

    # Align
    common = feat.index.intersection(Ywide.index)
    feat = feat.loc[common].sort_index()
    Ywide = Ywide.loc[common].sort_index()

    if drop_any_nan_rows:
        mask = feat.notna().all(axis=1) & Ywide.notna().all(axis=1)
        dropped = int((~mask).sum())
        if dropped:
            logger.info(
                "build_XY_per_resource_4d: dropping %d rows due to NaNs", dropped
            )
        feat = feat.loc[mask]
        Ywide = Ywide.loc[mask]

    X = feat.to_numpy(dtype=np.float64)
    Y = Ywide.to_numpy(dtype=np.float64)
    times = feat.index
    x_cols = feat.columns.tolist()
    y_cols = [str(c) for c in Ywide.columns.tolist()]

    if return_frames:
        return X, Y, times, x_cols, y_cols, feat, Ywide

    return X, Y, times, x_cols, y_cols


def build_XY_unscaled_2d(
    forecasts_filtered: pd.DataFrame,
    actuals_filtered: pd.DataFrame,
    *,
    drop_any_nan_rows: bool = True,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    forecast_col: str = "FORECAST",
    actual_col: str = "ACTUAL",
    model_col: Optional[str] = "MODEL",
    return_frames: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]] | tuple:
    """
    Option 4: [SYS_MEAN_MW, SYS_STD_MW] - 2D unscaled (raw MW units).

    Same as build_XY_for_covariance_system_only but NO standardization applied.
    Features remain in natural MW units with different scales.

    Expected: Equal weights dominated by large-scale feature. Learned ω discovers
    correct rescaling.

    Returns:
      X: (T, 2) float64 - system mean and std in raw MW units (unscaled)
      Y: (T, M) float64 - per-unit actuals
      times: DatetimeIndex
      x_cols: list[str]
      y_cols: list[str]

    If return_frames=True, also returns (feat_df_aligned, Y_df_aligned) at the end.
    """
    # Reuse existing function - it returns raw features before standardization
    return build_XY_for_covariance_system_only(
        forecasts_filtered=forecasts_filtered,
        actuals_filtered=actuals_filtered,
        drop_any_nan_rows=drop_any_nan_rows,
        time_col=time_col,
        resource_col=resource_col,
        forecast_col=forecast_col,
        actual_col=actual_col,
        model_col=model_col,
        return_frames=return_frames,
    )


def build_XY_focused_2d(
    forecasts_filtered: pd.DataFrame,
    actuals_filtered: pd.DataFrame,
    *,
    drop_any_nan_rows: bool = True,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    forecast_col: str = "FORECAST",
    actual_col: str = "ACTUAL",
    model_col: Optional[str] = "MODEL",
    return_frames: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]] | tuple:
    """
    Focused 2D baseline: [SYS_MEAN, SYS_STD]

    Cleanest case for omega learning. Test with different normalizations
    to see which shows clearest omega benefit.

    Returns:
      X: (T, 2) float64 - system mean and std
      Y: (T, M) float64 - per-unit actuals
      times: DatetimeIndex
      x_cols: list[str]
      y_cols: list[str]

    If return_frames=True, also returns (feat_df_aligned, Y_df_aligned) at the end.
    """
    # Reuse existing system-level builder
    return build_XY_for_covariance_system_only(
        forecasts_filtered=forecasts_filtered,
        actuals_filtered=actuals_filtered,
        drop_any_nan_rows=drop_any_nan_rows,
        time_col=time_col,
        resource_col=resource_col,
        forecast_col=forecast_col,
        actual_col=actual_col,
        model_col=model_col,
        return_frames=return_frames,
    )


def build_XY_high_dim_8d(
    forecasts_filtered: pd.DataFrame,
    actuals_filtered: pd.DataFrame,
    *,
    drop_any_nan_rows: bool = True,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    forecast_col: str = "FORECAST",
    actual_col: str = "ACTUAL",
    model_col: Optional[str] = "MODEL",
    return_frames: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]] | tuple:
    """
    High-dimensional 8D: [SYS_MEAN, SYS_STD, WIND_122_MEAN, WIND_122_STD,
                          WIND_309_MEAN, WIND_309_STD, WIND_317_MEAN, WIND_317_STD]

    Tests omega in higher dimensions. Which features does omega prioritize?

    Returns:
      X: (T, 8) float64 - system + per-unit mean/std features
      Y: (T, M) float64 - per-unit actuals
      times: DatetimeIndex
      x_cols: list[str]
      y_cols: list[str]

    If return_frames=True, also returns (feat_df_aligned, Y_df_aligned) at the end.
    """
    # Get system-level features
    feat_sys = build_features_system_total_mean_std(
        forecasts_filtered=forecasts_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=forecast_col,
        model_col=model_col,
    )

    # Get per-unit features (includes both mean and std for each unit)
    feat_per_unit = build_features_per_unit_mean_std(
        forecasts_filtered=forecasts_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=forecast_col,
        model_col=model_col,
        include_system=False,  # Only per-unit, not system
    )

    # Combine system and per-unit features
    feat = pd.concat([feat_sys, feat_per_unit], axis=1)

    # Rename per-unit columns to clean names
    feat.columns = [
        "SYS_MEAN",
        "SYS_STD",
        "WIND_122_MEAN",
        "WIND_122_STD",
        "WIND_309_MEAN",
        "WIND_309_STD",
        "WIND_317_MEAN",
        "WIND_317_STD",
    ]

    # Get targets
    Ywide = build_Y_actuals_matrix(
        actuals_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=actual_col,
    )

    # Align
    common = feat.index.intersection(Ywide.index)
    feat = feat.loc[common].sort_index()
    Ywide = Ywide.loc[common].sort_index()

    if drop_any_nan_rows:
        mask = feat.notna().all(axis=1) & Ywide.notna().all(axis=1)
        dropped = int((~mask).sum())
        if dropped:
            logger.info(
                "build_XY_high_dim_8d: dropping %d rows due to NaNs", dropped
            )
        feat = feat.loc[mask]
        Ywide = Ywide.loc[mask]

    X = feat.to_numpy(dtype=np.float64)
    Y = Ywide.to_numpy(dtype=np.float64)
    times = feat.index
    x_cols = feat.columns.tolist()
    y_cols = [str(c) for c in Ywide.columns.tolist()]

    if return_frames:
        return X, Y, times, x_cols, y_cols, feat, Ywide

    return X, Y, times, x_cols, y_cols


# Feature set dispatch
FEATURE_BUILDERS = {
    "temporal_3d": build_XY_temporal_nuisance_3d,
    "per_resource_4d": build_XY_per_resource_4d,
    "unscaled_2d": build_XY_unscaled_2d,
    "focused_2d": build_XY_focused_2d,
    "high_dim_8d": build_XY_high_dim_8d,
}

FEATURE_SET_DESCRIPTIONS = {
    "temporal_3d": "3D with temporal nuisance (SYS_MEAN, SYS_STD, HOUR_SIN)",
    "per_resource_4d": "4D per-resource (WIND_122, WIND_309, WIND_317, HOUR_SIN)",
    "unscaled_2d": "2D unscaled (SYS_MEAN_MW, SYS_STD_MW in raw units)",
    "focused_2d": "2D focused baseline (SYS_MEAN, SYS_STD)",
    "high_dim_8d": "8D high-dimensional (SYS + all units' MEAN/STD)",
}
