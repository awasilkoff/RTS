from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. Present: {list(df.columns)}"
        )


def _coerce_time_utc(series: pd.Series, *, col: str) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce")
    if out.isna().any():
        n = int(out.isna().sum())
        logger.debug("Column %s: coerced %d invalid timestamps to NaT", col, n)
    return out


def _coerce_str(series: pd.Series) -> pd.Series:
    return series.astype(str)


def _coerce_numeric(series: pd.Series, *, col: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        n = int(out.isna().sum())
        logger.debug("Column %s: coerced %d non-numeric values to NaN", col, n)
    return out


def _fill_std0(df: pd.DataFrame, std_cols: list[str]) -> None:
    if std_cols:
        df[std_cols] = df[std_cols].fillna(0.0)


# ------------------------------------------------------------------------------
# TOTAL SYSTEM HELPERS (For Conformal Prediction)
# ------------------------------------------------------------------------------


def build_total_forecast_ensemble_features(
    forecasts: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    model_col: str = "MODEL",
    resource_col: str = "ID_RESOURCE",
    value_col: str = "FORECAST",
) -> pd.DataFrame:
    """
    Aggregate forecasts to system level and calculate ensemble stats across models.

    Output columns:
      - TIME_HOURLY
      - ens_mean, ens_std, ens_min, ens_max, n_models

    Notes:
      - First sums across resources for each (time, model): FORECAST_TOTAL
      - Then aggregates FORECAST_TOTAL across models for each time.
    """
    _require_columns(
        forecasts, [time_col, model_col, resource_col, value_col], name="forecasts"
    )

    df = forecasts.copy()
    df[time_col] = _coerce_time_utc(df[time_col], col=time_col)
    df[resource_col] = _coerce_str(df[resource_col])
    df[value_col] = _coerce_numeric(df[value_col], col=value_col)
    df = df.dropna(subset=[time_col, model_col, resource_col, value_col])

    # Total per (time, model)
    tot = (
        df.groupby([time_col, model_col], observed=True)[value_col]
        .sum()
        .rename("FORECAST_TOTAL")
        .reset_index()
    )

    # Ensemble stats across models per time
    g = tot.groupby(time_col, observed=True)["FORECAST_TOTAL"]
    feat = g.agg(
        ens_mean="mean",
        ens_std="std",
        ens_min="min",
        ens_max="max",
        n_models="count",
    ).reset_index()

    feat["ens_std"] = feat["ens_std"].fillna(0.0)
    return feat


def build_total_actuals(
    actuals: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    value_col: str = "ACTUAL",
) -> pd.DataFrame:
    """
    Aggregate actuals to system level.

    Parameters:
      value_col: column name to use. Pass "RESIDUAL" to aggregate residuals instead of actuals.
                 Residuals are computed as ACTUAL - MEAN_FORECAST and must be in the dataframe.

    Output columns:
      - TIME_HOURLY
      - y
    """
    _require_columns(actuals, [time_col, resource_col, value_col], name="actuals")

    df = actuals.copy()
    df[time_col] = _coerce_time_utc(df[time_col], col=time_col)
    df[resource_col] = _coerce_str(df[resource_col])
    df[value_col] = _coerce_numeric(df[value_col], col=value_col)
    df = df.dropna(subset=[time_col, resource_col, value_col])

    y = df.groupby(time_col, observed=True)[value_col].sum().rename("y").reset_index()
    return y


def build_conformal_totals_df(
    actuals_filtered: pd.DataFrame,
    forecasts_filtered: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    value_col: str = "ACTUAL",
) -> pd.DataFrame:
    """
    Prepare the main dataframe for conformal prediction on system totals.

    Output columns include:
      - y, ens_mean, ens_std, ens_min, ens_max, n_models
      - hour, dow, month
    """
    y = build_total_actuals(actuals_filtered, time_col=time_col, value_col=value_col)
    feat = build_total_forecast_ensemble_features(forecasts_filtered, time_col=time_col)

    df = (
        y.merge(feat, on=time_col, how="inner")
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    dt = df[time_col]
    df["hour"] = dt.dt.hour.astype(int)
    df["dow"] = dt.dt.dayofweek.astype(int)
    df["month"] = dt.dt.month.astype(int)

    # Drop non-finite rows
    keep = (
        np.isfinite(df["y"]) & np.isfinite(df["ens_mean"]) & np.isfinite(df["ens_std"])
    )
    dropped = int((~keep).sum())
    if dropped:
        logger.info(
            "build_conformal_totals_df: dropping %d rows with non-finite y/ens_mean/ens_std",
            dropped,
        )

    df = df.loc[keep].reset_index(drop=True)
    return df


# ------------------------------------------------------------------------------
# MULTI-UNIT HELPERS (For Covariance Optimization)
# ------------------------------------------------------------------------------


def build_features_per_unit_mean_std(
    forecasts_filtered: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    value_col: str = "FORECAST",
    model_col: Optional[str] = "MODEL",
    include_system: bool = True,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Build a wide feature matrix with unit-specific mean/std across models and (optionally)
    system-level ensemble features.

    Unit features:
      - U_<unit>_MEAN : mean forecast across models at (time, unit)
      - U_<unit>_STD  : std  forecast across models at (time, unit), NaN->0 when only 1 sample

    System features (if include_system=True):
      - SYS_MEAN, SYS_STD, SYS_MIN, SYS_MAX, SYS_N_MODELS
        computed by summing across resources for each (time, model) to obtain system totals,
        then aggregating across models per time (same definition as build_total_forecast_ensemble_features).

    Notes:
      - If model_col is not present in the dataframe, the unit MEAN is the mean over the available rows
        for each (time, unit), and STD is std over those rows (often 0.0 if only one).
      - ddof is forwarded to pandas std. ddof=1 is default sample std.
    """
    _require_columns(
        forecasts_filtered,
        [time_col, resource_col, value_col],
        name="forecasts_filtered",
    )

    df = forecasts_filtered.copy()
    df[time_col] = _coerce_time_utc(df[time_col], col=time_col)
    df[resource_col] = _coerce_str(df[resource_col])
    df[value_col] = _coerce_numeric(df[value_col], col=value_col)

    core_cols = [time_col, resource_col, value_col]
    if model_col and model_col in df.columns:
        core_cols.append(model_col)

    df = df.dropna(subset=core_cols)

    # Unit stats across rows (typically across models)
    unit_stats = (
        df.groupby([time_col, resource_col], observed=True)[value_col]
        .agg(MEAN="mean", STD=lambda x: x.std(ddof=ddof))
        .reset_index()
    )

    mean_wide = unit_stats.pivot(index=time_col, columns=resource_col, values="MEAN")
    std_wide = unit_stats.pivot(index=time_col, columns=resource_col, values="STD")

    units = sorted(mean_wide.columns.astype(str).tolist())
    mean_wide = mean_wide[units]
    std_wide = std_wide[units]

    mean_wide.columns = [f"U_{u}_MEAN" for u in units]
    std_wide.columns = [f"U_{u}_STD" for u in units]

    feat = mean_wide.join(std_wide, how="inner").sort_index()

    # Fill std NaNs -> 0 (e.g., only one model/row in group)
    std_cols = [c for c in feat.columns if c.endswith("_STD")]
    _fill_std0(feat, std_cols)

    if include_system:
        # Prefer system ensemble definition that matches conformal totals:
        if model_col and model_col in df.columns:
            sys_feat = build_total_forecast_ensemble_features(
                df,
                time_col=time_col,
                model_col=model_col,
                resource_col=resource_col,
                value_col=value_col,
            ).set_index(time_col)
            # Back-compat names expected by earlier pipelines:
            sys_feat = sys_feat.rename(
                columns={
                    "ens_mean": "SYS_MEAN",
                    "ens_std": "SYS_STD",
                    "ens_min": "SYS_MIN",
                    "ens_max": "SYS_MAX",
                    "n_models": "SYS_N_MODELS",
                }
            )
            feat = sys_feat.join(feat, how="inner").sort_index()
        else:
            # If no model column exists, fall back to system mean/std across resources at each time.
            sys_stats = df.groupby(time_col, observed=True)[value_col].agg(
                SYS_MEAN="mean", SYS_STD=lambda x: x.std(ddof=ddof)
            )
            sys_stats["SYS_STD"] = sys_stats["SYS_STD"].fillna(0.0)
            feat = sys_stats.join(feat, how="inner").sort_index()

    out = feat.astype(np.float64)
    logger.info(
        "build_features_per_unit_mean_std: built features with shape=%s", out.shape
    )
    return out


def build_Y_actuals_matrix(
    actuals_filtered: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    value_col: str = "ACTUAL",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Build wide target matrix Y: index=time, columns=resources, values=actuals or residuals.

    Parameters:
      value_col: column name to use as target. Pass "RESIDUAL" to use residuals instead of actuals.
                 Residuals are computed as ACTUAL - MEAN_FORECAST and should be in the dataframe.
      agg: how to handle duplicates within (time, resource). Options: "mean" or "sum".
    """
    _require_columns(
        actuals_filtered, [time_col, resource_col, value_col], name="actuals_filtered"
    )
    if agg not in {"mean", "sum"}:
        raise ValueError("agg must be 'mean' or 'sum'")

    df = actuals_filtered.copy()
    df[time_col] = _coerce_time_utc(df[time_col], col=time_col)
    df[resource_col] = _coerce_str(df[resource_col])
    df[value_col] = _coerce_numeric(df[value_col], col=value_col)
    df = df.dropna(subset=[time_col, resource_col, value_col])

    # Handle duplicates
    if agg == "mean":
        df = (
            df.groupby([time_col, resource_col], observed=True)[value_col]
            .mean()
            .reset_index()
        )
    else:
        df = (
            df.groupby([time_col, resource_col], observed=True)[value_col]
            .sum()
            .reset_index()
        )

    Ywide = df.pivot(
        index=time_col, columns=resource_col, values=value_col
    ).sort_index()
    Ywide = Ywide[sorted(Ywide.columns.astype(str).tolist())]
    logger.info("build_Y_actuals_matrix: built Y with shape=%s", Ywide.shape)
    return Ywide


def build_XY_for_covariance(
    forecasts_filtered: pd.DataFrame,
    actuals_filtered: pd.DataFrame,
    *,
    include_system_features: bool = True,
    drop_any_nan_rows: bool = True,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    forecast_col: str = "FORECAST",
    actual_col: str = "ACTUAL",
    model_col: Optional[str] = "MODEL",
    return_frames: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]] | tuple:
    """
    Merge features and targets, align them by time, and return numpy arrays.

    Parameters:
      actual_col: column name to use as target. Pass "RESIDUAL" to use residuals (ACTUAL - MEAN_FORECAST)
                  instead of raw actuals. Residuals must be computed in the dataframe first.

    Returns:
      X: (T, K) float64
      Y: (T, M) float64
      times: DatetimeIndex
      x_cols: list[str]
      y_cols: list[str]

    If return_frames=True, also returns (feat_df_aligned, Y_df_aligned) at the end.
    """
    feat = build_features_per_unit_mean_std(
        forecasts_filtered=forecasts_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=forecast_col,
        model_col=model_col,
        include_system=include_system_features,
    )
    Ywide = build_Y_actuals_matrix(
        actuals_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=actual_col,
    )

    common = feat.index.intersection(Ywide.index)
    feat = feat.loc[common].sort_index()
    Ywide = Ywide.loc[common].sort_index()

    if drop_any_nan_rows:
        mask = feat.notna().all(axis=1) & Ywide.notna().all(axis=1)
        dropped = int((~mask).sum())
        if dropped:
            logger.info(
                "build_XY_for_covariance: dropping %d rows due to NaNs", dropped
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


def build_features_system_total_mean_std(
    forecasts_filtered: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    resource_col: str = "ID_RESOURCE",
    value_col: str = "FORECAST",
    model_col: Optional[str] = "MODEL",
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Build ONLY system-total ensemble mean/std across models.

    Output index: TIME_HOURLY
    Output cols:
      - SYS_MEAN: mean of system-total forecasts across models
      - SYS_STD : std  of system-total forecasts across models (NaN->0 if only 1 model)

    Notes:
      - First sums across resources for each (time, model) to get system totals
      - Then aggregates across models per time
      - If model_col missing, falls back to per-time mean/std over available rows
        (less ideal; you probably want MODEL present).
    """
    _require_columns(
        forecasts_filtered,
        [time_col, resource_col, value_col],
        name="forecasts_filtered",
    )

    df = forecasts_filtered.copy()
    df[time_col] = _coerce_time_utc(df[time_col], col=time_col)
    df[resource_col] = _coerce_str(df[resource_col])
    df[value_col] = _coerce_numeric(df[value_col], col=value_col)

    core_cols = [time_col, resource_col, value_col]
    if model_col and model_col in df.columns:
        core_cols.append(model_col)
    df = df.dropna(subset=core_cols)

    if model_col and model_col in df.columns:
        # Total per (time, model)
        tot = (
            df.groupby([time_col, model_col], observed=True)[value_col]
            .sum()
            .rename("FORECAST_TOTAL")
            .reset_index()
        )

        g = tot.groupby(time_col, observed=True)["FORECAST_TOTAL"]
        out = g.agg(
            SYS_MEAN="mean",
            SYS_STD=lambda x: x.std(ddof=ddof),
        ).reset_index()
        out["SYS_STD"] = out["SYS_STD"].fillna(0.0)
        out = out.set_index(time_col).sort_index()
    else:
        # Fallback: mean/std over whatever rows you have at each time
        out = df.groupby(time_col, observed=True)[value_col].agg(
            SYS_MEAN="mean",
            SYS_STD=lambda x: x.std(ddof=ddof),
        )
        out["SYS_STD"] = out["SYS_STD"].fillna(0.0)
        out = out.sort_index()

    out = out.astype(np.float64)
    logger.info(
        "build_features_system_total_mean_std: built features with shape=%s", out.shape
    )
    return out


def build_XY_for_covariance_system_only(
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
    Build X (system-only features) and Y (per-unit actuals) for covariance optimization.

    Uses only SYS_MEAN and SYS_STD as features (2 columns), while Y remains per-unit.

    Parameters:
      actual_col: column name to use as target. Pass "RESIDUAL" to use residuals (ACTUAL - MEAN_FORECAST)
                  instead of raw actuals. Residuals must be computed in the dataframe first.

    Returns:
      X: (T, 2) float64 - system mean and std only
      Y: (T, M) float64 - per-unit actuals or residuals
      times: DatetimeIndex
      x_cols: list[str] - ['SYS_MEAN', 'SYS_STD']
      y_cols: list[str] - resource IDs

    If return_frames=True, also returns (feat_df_aligned, Y_df_aligned) at the end.
    """
    feat = build_features_system_total_mean_std(
        forecasts_filtered=forecasts_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=forecast_col,
        model_col=model_col,
    )
    Ywide = build_Y_actuals_matrix(
        actuals_filtered,
        time_col=time_col,
        resource_col=resource_col,
        value_col=actual_col,
    )

    common = feat.index.intersection(Ywide.index)
    feat = feat.loc[common].sort_index()
    Ywide = Ywide.loc[common].sort_index()

    if drop_any_nan_rows:
        mask = feat.notna().all(axis=1) & Ywide.notna().all(axis=1)
        dropped = int((~mask).sum())
        if dropped:
            logger.info(
                "build_XY_for_covariance_system_only: dropping %d rows due to NaNs", dropped
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
