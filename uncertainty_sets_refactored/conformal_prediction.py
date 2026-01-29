from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import _conformal_q_level


BinningSpec = Literal["y_pred"] | str  # either "y_pred" or "feature:<colname>"


@dataclass(frozen=True)
class ConformalLowerBundle:
    """
    A trained quantile model + adaptive (binned) conformal correction for a *lower bound*.

    Interpretation:
      y_pred_base ~ conditional quantile at level `quantile_alpha`
      y_pred_conf = y_pred_base - q_hat(bin_feature) * scale

    Where q_hat is computed on calibration nonconformity scores:
      r = max(0, (y_pred_base - y_true) / scale)
    """

    feature_cols: list[str]
    scale_col: str
    binning: BinningSpec  # "y_pred" or "feature:<col>"
    bin_edges: list[float]

    q_hat_global_r: float
    q_hat_by_bin_r: dict  # keys are pandas Interval

    alpha_target: float
    quantile_alpha: float
    model: Any  # typically LGBMRegressor

    min_scale: float = 1e-3

    def predict_df(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the base model and conformal correction to a feature DataFrame.

        Required columns:
          - feature_cols
          - scale_col
          - if binning == "feature:<col>", that column must exist
        """
        df = df_feat.copy()

        # Base prediction
        X = df[self.feature_cols]  # keep as DataFrame with column names
        y_pred = self.model.predict(X)

        df["y_pred_base"] = np.asarray(y_pred, dtype=float)

        bin_feature = _extract_binning_feature(
            df, self.binning, y_pred_base=df["y_pred_base"].to_numpy()
        )
        df["bin_feature"] = bin_feature

        df["bin"] = pd.cut(
            df["bin_feature"], bins=self.bin_edges, include_lowest=True, right=True
        )

        # q_hat lookup (fall back to global if bin missing / unseen)
        df["q_hat_r"] = np.asarray(
            [self.q_hat_by_bin_r.get(b, self.q_hat_global_r) for b in df["bin"]],
            dtype=float,
        )

        scale = _sanitize_scale(
            df[self.scale_col].to_numpy(dtype=float), min_scale=self.min_scale
        )
        df["scale_sanitized"] = scale

        df["margin"] = df["q_hat_r"].to_numpy(dtype=float) * scale
        df["y_pred_conf"] = df["y_pred_base"].to_numpy(dtype=float) - df[
            "margin"
        ].to_numpy(dtype=float)

        return df


def _sanitize_scale(scale: np.ndarray, *, min_scale: float) -> np.ndarray:
    scale = np.asarray(scale, dtype=float)
    scale = np.where(~np.isfinite(scale) | (scale <= min_scale), min_scale, scale)
    return scale


def _extract_binning_feature(
    df: pd.DataFrame,
    binning: BinningSpec,
    *,
    y_pred_base: np.ndarray,
) -> np.ndarray:
    if binning == "y_pred":
        return np.asarray(y_pred_base, dtype=float)

    if isinstance(binning, str) and binning.startswith("feature:"):
        col = binning.split("feature:", 1)[1]
        if col not in df.columns:
            raise ValueError(f"Missing binning feature column '{col}'")
        return np.asarray(df[col].to_numpy(), dtype=float)

    raise ValueError("binning must be 'y_pred' or 'feature:<colname>'")


def compute_binned_adaptive_conformal_corrections_lower(
    *,
    bin_feature_cal: np.ndarray,
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    scale_cal: np.ndarray,
    alpha_target: float,
    bins: list[float],
    min_scale: float = 1e-3,
) -> tuple[float, dict]:
    """
    Compute global + per-bin q_hat for *lower bound* conformal correction.

    Nonconformity score (normalized miss on the lower side):
      r = max(0, (y_pred - y_true)/scale)

    q_hat is the conformal quantile of r at level computed by _conformal_q_level(n, alpha_target).

    Returns:
      (q_hat_global_r, q_hat_by_bin_r)
    """
    bin_feature_cal = np.asarray(bin_feature_cal)
    y_cal = np.asarray(y_cal)
    y_pred_cal = np.asarray(y_pred_cal)
    scale_cal = np.asarray(scale_cal, dtype=float)

    if y_cal.size == 0:
        raise ValueError("Calibration set is empty.")

    s = _sanitize_scale(scale_cal, min_scale=min_scale)
    r = np.maximum(0.0, (y_pred_cal - y_cal) / s)

    # Global
    q_level_g = _conformal_q_level(int(r.size), float(alpha_target))
    q_hat_global_r = float(np.quantile(r, q_level_g))

    # Per bin
    df = pd.DataFrame({"bin_feature": bin_feature_cal, "r": r})
    df["bin"] = pd.cut(df["bin_feature"], bins=bins, include_lowest=True, right=True)

    q_hat_by_bin_r: dict = {}
    for b in df["bin"].cat.categories:
        r_b = df.loc[df["bin"] == b, "r"].to_numpy(dtype=float)
        if r_b.size == 0:
            continue
        q_level_b = _conformal_q_level(int(r_b.size), float(alpha_target))
        q_hat_by_bin_r[b] = float(np.quantile(r_b, q_level_b))

    return q_hat_global_r, q_hat_by_bin_r


def _time_ordered_split(
    n: int,
    *,
    test_frac: float,
    cal_frac: float,
) -> tuple[slice, slice, slice]:
    if not (0.0 < test_frac < 1.0) or not (0.0 < cal_frac < 1.0):
        raise ValueError("test_frac and cal_frac must be in (0, 1).")
    if test_frac + cal_frac >= 1.0:
        raise ValueError("test_frac + cal_frac must be < 1.")

    n_test = int(test_frac * n)
    n_cal = int(cal_frac * n)
    n_train = n - n_test - n_cal
    if n_train <= 0 or n_cal <= 0 or n_test <= 0:
        raise ValueError(
            f"Split produced empty partition(s): n={n}, train={n_train}, cal={n_cal}, test={n_test}"
        )

    sl_train = slice(0, n_train)
    sl_cal = slice(n_train, n_train + n_cal)
    sl_test = slice(n_train + n_cal, n)
    return sl_train, sl_cal, sl_test


def train_wind_lower_model_conformal_binned(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "y",
    time_col: str = "TIME_HOURLY",
    test_frac: float = 0.2,
    cal_frac: float = 0.2,
    quantile_alpha: float = 0.10,
    alpha_target: float = 0.90,
    scale_col: str = "ens_std",
    binning: BinningSpec = "y_pred",
    bin_edges: list[float] | None = None,
    bin_quantiles: list[float] | None = None,
    min_scale: float = 1e-3,
    model_kwargs: dict | None = None,
) -> tuple[ConformalLowerBundle, dict[str, Any], pd.DataFrame]:
    """
    Train a quantile regression model and calibrate a conformal *lower bound* with optional binning.

    The split is time-ordered:
      train | cal | test

    Returns:
      (bundle, metrics, df_test)
    """
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")
    if scale_col not in df.columns:
        raise ValueError(f"scale_col '{scale_col}' not found.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df.sort_values(time_col).reset_index(drop=True)
    df = df[df[target_col].notna()].reset_index(drop=True)

    X = df[feature_cols]
    y = df[target_col].to_numpy(dtype=float)
    s = _sanitize_scale(df[scale_col].to_numpy(dtype=float), min_scale=min_scale)

    sl_train, sl_cal, sl_test = _time_ordered_split(
        len(df), test_frac=test_frac, cal_frac=cal_frac
    )

    X_train, y_train = X.iloc[sl_train], y[sl_train]
    X_cal, y_cal = X.iloc[sl_cal], y[sl_cal]
    X_test, y_test = X.iloc[sl_test], y[sl_test]

    s_cal = s[sl_cal]
    s_test = s[sl_test]

    # Train quantile model (LGBM)
    kw = dict(
        objective="quantile",
        alpha=quantile_alpha,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if model_kwargs:
        kw.update(model_kwargs)

    model = LGBMRegressor(**kw)
    model.fit(X_train, y_train)

    y_pred_cal = np.asarray(model.predict(X_cal), dtype=float)
    y_pred_test = np.asarray(model.predict(X_test), dtype=float)

    # Binning features for calibration and test
    if binning == "y_pred":
        bin_feature_cal = y_pred_cal
        bin_feature_test = y_pred_test
    elif isinstance(binning, str) and binning.startswith("feature:"):
        col = binning.split("feature:", 1)[1]
        if col not in df.columns:
            raise ValueError(f"Missing binning feature column '{col}'")
        arr = np.asarray(df[col].to_numpy(), dtype=float)
        bin_feature_cal = arr[sl_cal]
        bin_feature_test = arr[sl_test]
    else:
        raise ValueError("binning must be 'y_pred' or 'feature:<colname>'")

    # Choose bin edges if not provided
    if bin_edges is None:
        qs = bin_quantiles if bin_quantiles else [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        edges = np.unique(np.quantile(bin_feature_cal, qs))
        if edges.size < 2:
            raise ValueError("Not enough unique bin edges.")
        bin_edges = edges.tolist()

    (
        q_hat_global_r,
        q_hat_by_bin_r,
    ) = compute_binned_adaptive_conformal_corrections_lower(
        bin_feature_cal=bin_feature_cal,
        y_cal=y_cal,
        y_pred_cal=y_pred_cal,
        scale_cal=s_cal,
        alpha_target=alpha_target,
        bins=list(bin_edges),
        min_scale=min_scale,
    )

    bundle = ConformalLowerBundle(
        feature_cols=list(feature_cols),
        scale_col=scale_col,
        binning=binning,
        bin_edges=list(bin_edges),
        q_hat_global_r=q_hat_global_r,
        q_hat_by_bin_r=q_hat_by_bin_r,
        alpha_target=alpha_target,
        quantile_alpha=quantile_alpha,
        model=model,
        min_scale=min_scale,
    )

    # Test-set diagnostics
    df_test = pd.DataFrame(
        {
            "y": y_test,
            "y_pred_base": y_pred_test,
            "scale": s_test,
            "bin_feature": bin_feature_test,
        }
    )
    df_test["bin"] = pd.cut(
        df_test["bin_feature"], bins=bin_edges, include_lowest=True, right=True
    )

    q_hats = np.asarray(
        [q_hat_by_bin_r.get(b, q_hat_global_r) for b in df_test["bin"]], dtype=float
    )
    margin = q_hats * _sanitize_scale(
        df_test["scale"].to_numpy(dtype=float), min_scale=min_scale
    )
    df_test["y_pred_conf"] = df_test["y_pred_base"].to_numpy(dtype=float) - margin

    y_pred_conf = df_test["y_pred_conf"].to_numpy(dtype=float)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(df_test["y"], y_pred_conf))),
        "mae": float(mean_absolute_error(df_test["y"], y_pred_conf)),
        "coverage": float((df_test["y"] >= y_pred_conf).mean()),
        "pre_conformal_coverage": float((df_test["y"] >= y_pred_test).mean()),
        "q_hat_global_r": q_hat_global_r,
        "q_hat_by_bin_r": q_hat_by_bin_r,
        "n_train": int(sl_train.stop - sl_train.start),
        "n_cal": int(sl_cal.stop - sl_cal.start),
        "n_test": int(sl_test.stop - sl_test.start),
    }

    return bundle, metrics, df_test
