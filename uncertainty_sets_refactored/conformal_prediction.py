from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import _conformal_q_level


# =============================================================================
# Weighted Conformal Prediction Components
# =============================================================================


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
    *,
    include_test_weight: bool = True,
) -> float:
    """
    Compute weighted quantile using cumulative weight.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Values to compute quantile over (e.g., nonconformity scores)
    weights : np.ndarray, shape (n,)
        Weights for each value (e.g., kernel weights)
    q : float
        Quantile level (e.g., 0.95 for 95th percentile)
    include_test_weight : bool, default=True
        If True, add 1.0 to total weight (for conformal test point)

    Returns
    -------
    quantile : float
        Weighted q-th quantile

    Algorithm
    ---------
    1. Sort values and corresponding weights
    2. Compute cumulative sum of normalized weights
    3. Find first index where cumsum >= q
    4. Return value at that index
    """
    if values.size != weights.size:
        raise ValueError("values and weights must have same size")
    if values.size == 0:
        raise ValueError("Cannot compute quantile of empty array")

    # Sort by values
    idx = np.argsort(values)
    values_sorted = values[idx]
    weights_sorted = weights[idx]

    # Normalize weights
    W = weights_sorted.sum()
    if include_test_weight:
        W += 1.0  # Add test point weight (K(0) = 1.0 for Gaussian)

    if W <= 0:
        raise ValueError("Total weight must be positive")

    weights_norm = weights_sorted / W

    # Cumulative sum
    cumsum_weights = np.cumsum(weights_norm)

    # Find first index where cumsum >= q
    # Handle edge cases: if all cumsum < q, return max value
    idx_q = np.searchsorted(cumsum_weights, q, side='left')
    if idx_q >= len(values_sorted):
        idx_q = len(values_sorted) - 1

    return float(values_sorted[idx_q])


def _compute_kernel_distances(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Compute Gaussian kernel weights between query and reference points.

    Parameters
    ----------
    X_query : np.ndarray, shape (Nq, K)
        Query features
    X_ref : np.ndarray, shape (Nr, K)
        Reference features
    omega : np.ndarray, shape (K,)
        Feature weights (from covariance optimization)
    tau : float
        Kernel bandwidth parameter

    Returns
    -------
    weights : np.ndarray, shape (Nq, Nr)
        Gaussian kernel weights for each query-reference pair
        K(d/tau) = exp(-d/tau) where d = sqrt(sum_k omega[k] * (xq[k] - xr[k])^2)

    Notes
    -----
    Reuses weighted distance logic from covariance_optimization._pairwise_sqdist():
    - D[i,j] = sum_k omega[k] * (X_query[i,k] - X_ref[j,k])^2
    - K[i,j] = exp(-sqrt(D[i,j]) / tau)

    For numerical stability, compute in log-space if needed.
    """
    # Weighted squared distances
    w = np.sqrt(np.maximum(omega, 0.0))  # Apply sqrt for numerical stability
    Xq_w = X_query * w[np.newaxis, :]  # (Nq, K)
    Xr_w = X_ref * w[np.newaxis, :]     # (Nr, K)

    # ||Xq_w||^2 + ||Xr_w||^2 - 2*(Xq_w @ Xr_w.T)
    Xq_sq = np.sum(Xq_w ** 2, axis=1, keepdims=True)  # (Nq, 1)
    Xr_sq = np.sum(Xr_w ** 2, axis=1, keepdims=True)  # (Nr, 1)
    D = Xq_sq + Xr_sq.T - 2.0 * (Xq_w @ Xr_w.T)      # (Nq, Nr)
    D = np.maximum(D, 0.0)  # Numerical safety

    # Gaussian kernel: K(d/tau) = exp(-d/tau)
    d = np.sqrt(D)  # Euclidean distances
    K = np.exp(-d / tau)

    return K  # (Nq, Nr)


def compute_weighted_conformal_correction_lower(
    *,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    scale_cal: np.ndarray,
    X_query: np.ndarray,
    omega: np.ndarray,
    tau: float,
    alpha_target: float,
    min_scale: float = 1e-3,
    safety_margin: float = 0.0,
) -> np.ndarray:
    """
    Compute localized weighted conformal corrections for lower bounds.

    Parameters
    ----------
    X_cal : np.ndarray, shape (n_cal, K)
        Calibration features (for kernel weighting, matching omega dimensions)
    y_cal : np.ndarray, shape (n_cal,)
        Calibration targets (actual values)
    y_pred_cal : np.ndarray, shape (n_cal,)
        Calibration predictions from quantile model
    scale_cal : np.ndarray, shape (n_cal,)
        Calibration scale estimates (e.g., ensemble std)
    X_query : np.ndarray, shape (n_query, K)
        Query features (for kernel weighting)
    omega : np.ndarray, shape (K,)
        Learned feature weights (from covariance optimization)
    tau : float
        Kernel bandwidth parameter
    alpha_target : float
        Target coverage level (e.g., 0.95)
    min_scale : float, default=1e-3
        Minimum scale value for numerical stability
    safety_margin : float, default=0.0
        Additional conservativeness buffer (e.g., 0.02 for 2% extra coverage)

    Returns
    -------
    q_hat : np.ndarray, shape (n_query,)
        Localized conformal correction for each query point
        Apply as: y_pred_conf = y_pred_query - q_hat * scale_query

    Algorithm
    ---------
    For each query point x*:
    1. Compute nonconformity scores on calibration set:
       r_cal = max(0, (y_pred_cal - y_cal) / scale_cal)
    2. Compute kernel weights: w_cal = K(dist(X_cal, x*) / tau)
    3. Compute weighted quantile: q = weighted_quantile(r_cal, w_cal, level)
    4. Return q as localized correction factor
    """
    # Sanitize scale
    scale_cal_safe = _sanitize_scale(scale_cal, min_scale=min_scale)

    # Nonconformity scores (one-sided for lower bound)
    r_cal = np.maximum(0.0, (y_pred_cal - y_cal) / scale_cal_safe)  # (n_cal,)

    # Compute kernel weights: (n_query, n_cal)
    K = _compute_kernel_distances(X_query, X_cal, omega, tau)

    # Conformal quantile level (with safety margin)
    alpha_adjusted = min(alpha_target + safety_margin, 1.0)

    # Compute weighted quantile for each query point
    n_query = X_query.shape[0]
    q_hat = np.zeros(n_query)

    for i in range(n_query):
        w_i = K[i, :]  # Kernel weights for query point i: (n_cal,)

        # Weighted quantile with test point weight
        q_hat[i] = weighted_quantile(
            values=r_cal,
            weights=w_i,
            q=alpha_adjusted,
            include_test_weight=True,  # Add 1.0 for test point in denominator
        )

    return q_hat  # (n_query,)


# =============================================================================
# Weighted Conformal Bundle
# =============================================================================


@dataclass(frozen=True)
class WeightedConformalLowerBundle:
    """
    Trained quantile model + kernel-weighted conformal correction.

    Unlike ConformalLowerBundle (binned), this stores calibration set
    for localized prediction at each query point.

    Attributes
    ----------
    feature_cols : list[str]
        Feature columns for quantile model (e.g., ['ens_mean', 'ens_std'])
    kernel_feature_cols : list[str]
        Feature columns for kernel weighting (e.g., ['SYS_MEAN', 'SYS_STD'])
        Must match omega dimensions
    scale_col : str
        Column for nonconformity scaling (e.g., 'ens_std')

    model : LGBMRegressor
        Trained quantile regression model

    X_cal : np.ndarray, shape (n_cal, K)
        Calibration kernel features (for distance computation)
    y_cal : np.ndarray, shape (n_cal,)
        Calibration targets
    y_pred_cal : np.ndarray, shape (n_cal,)
        Calibration predictions
    scale_cal : np.ndarray, shape (n_cal,)
        Calibration scale values

    omega : np.ndarray, shape (K,)
        Learned feature weights (from covariance optimization)
    tau : float
        Kernel bandwidth parameter
    alpha_target : float
        Target coverage level

    min_scale : float
        Minimum scale value for numerical stability
    safety_margin : float
        Additional conservativeness buffer
    """

    feature_cols: list[str]
    kernel_feature_cols: list[str]
    scale_col: str

    model: Any  # LGBMRegressor

    X_cal: np.ndarray
    y_cal: np.ndarray
    y_pred_cal: np.ndarray
    scale_cal: np.ndarray

    omega: np.ndarray
    tau: float
    alpha_target: float

    min_scale: float = 1e-3
    safety_margin: float = 0.0

    def predict_df(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quantile model and localized weighted conformal correction.

        Parameters
        ----------
        df_feat : pd.DataFrame
            Features for prediction
            Must contain: feature_cols, kernel_feature_cols, scale_col

        Returns
        -------
        df_pred : pd.DataFrame
            Copy of df_feat with added columns:
            - y_pred_base: Base quantile prediction
            - q_hat_local: Localized conformal correction per query point
            - scale_sanitized: Sanitized scale values
            - margin: q_hat_local * scale_sanitized
            - y_pred_conf: Final conformal lower bound
        """
        df = df_feat.copy()

        # Base quantile prediction
        X = df[self.feature_cols]
        y_pred = self.model.predict(X)
        df['y_pred_base'] = np.asarray(y_pred, dtype=float)

        # Extract kernel features for query points
        X_query = df[self.kernel_feature_cols].to_numpy(dtype=float)

        # Compute localized q_hat for each query point
        q_hat_local = compute_weighted_conformal_correction_lower(
            X_cal=self.X_cal,
            y_cal=self.y_cal,
            y_pred_cal=self.y_pred_cal,
            scale_cal=self.scale_cal,
            X_query=X_query,
            omega=self.omega,
            tau=self.tau,
            alpha_target=self.alpha_target,
            min_scale=self.min_scale,
            safety_margin=self.safety_margin,
        )
        df['q_hat_local'] = q_hat_local

        # Sanitize scale
        scale = _sanitize_scale(
            df[self.scale_col].to_numpy(dtype=float),
            min_scale=self.min_scale
        )
        df['scale_sanitized'] = scale

        # Conformal margin and final bound
        df['margin'] = df['q_hat_local'].to_numpy(dtype=float) * scale
        df['y_pred_conf'] = df['y_pred_base'].to_numpy(dtype=float) - df['margin'].to_numpy(dtype=float)

        return df


# =============================================================================
# Training Function for Weighted Conformal
# =============================================================================


def train_wind_lower_model_weighted_conformal(
    df: pd.DataFrame,
    feature_cols: list[str],
    kernel_feature_cols: list[str],
    *,
    target_col: str = 'y',
    time_col: str = 'TIME_HOURLY',
    test_frac: float = 0.2,
    cal_frac: float = 0.2,
    quantile_alpha: float = 0.10,
    alpha_target: float = 0.90,
    scale_col: str = 'ens_std',
    omega: np.ndarray | None = None,
    omega_path: str | None = None,
    tau: float = 5.0,
    min_scale: float = 1e-3,
    model_kwargs: dict | None = None,
    safety_margin: float = 0.0,
    split_method: str = 'random',
    random_seed: int = 42,
) -> tuple[WeightedConformalLowerBundle, dict[str, Any], pd.DataFrame]:
    """
    Train quantile model + kernel-weighted conformal correction for lower bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with features, targets, and time column
    feature_cols : list[str]
        Features for quantile regression model (e.g., ['ens_mean', 'ens_std'])
    kernel_feature_cols : list[str]
        Features for kernel weighting (e.g., ['SYS_MEAN', 'SYS_STD'])
        Must match omega dimensions
    target_col : str, default='y'
        Target variable column name
    time_col : str, default='TIME_HOURLY'
        Time column for sorting (if split_method='time_ordered')
    test_frac : float, default=0.2
        Fraction of data for test set
    cal_frac : float, default=0.2
        Fraction of data for calibration set
    quantile_alpha : float, default=0.10
        Quantile level for base model (0.10 = 10th percentile)
    alpha_target : float, default=0.90
        Target conformal coverage (0.90 = 90% of actuals >= lower bound)
    scale_col : str, default='ens_std'
        Column for nonconformity scaling
    omega : np.ndarray, shape (K,), optional
        Learned feature weights (from covariance optimization)
        If None, must provide omega_path
    omega_path : str, optional
        Path to saved omega.npy file (from covariance optimization)
        Used if omega is None
    tau : float, default=5.0
        Kernel bandwidth parameter
        Smaller tau = sharper kernel, more local
        Larger tau = smoother kernel, more global
        Recommend grid search over [1.0, 2.0, 5.0, 10.0, 20.0]
    min_scale : float, default=1e-3
        Minimum scale value for numerical stability
    model_kwargs : dict, optional
        Additional kwargs for LGBMRegressor
    safety_margin : float, default=0.0
        Additional conservativeness buffer (e.g., 0.02 for 2% extra coverage)
    split_method : str, default='random'
        Data split method: 'time_ordered' or 'random'
    random_seed : int, default=42
        Random seed for reproducibility (if split_method='random')

    Returns
    -------
    bundle : WeightedConformalLowerBundle
        Trained bundle with localized conformal correction
    metrics : dict
        Evaluation metrics on test set:
        - coverage: Empirical coverage (fraction y >= y_pred_conf)
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - n_train, n_cal, n_test: Split sizes
    df_test : pd.DataFrame
        Test set with predictions and conformal bounds

    Example
    -------
    >>> # Load omega from saved covariance optimization
    >>> omega = np.load('data/best_omega.npy')
    >>>
    >>> bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
    ...     df,
    ...     feature_cols=['ens_mean', 'ens_std'],
    ...     kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
    ...     omega=omega,
    ...     tau=5.0,
    ...     alpha_target=0.95,
    ... )
    >>> print(f"Coverage: {metrics['coverage']:.3f}")
    """
    # Validate inputs
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")
    if scale_col not in df.columns:
        raise ValueError(f"scale_col '{scale_col}' not found")

    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing feature columns: {missing_feat}")

    missing_kernel = [c for c in kernel_feature_cols if c not in df.columns]
    if missing_kernel:
        raise ValueError(f"Missing kernel feature columns: {missing_kernel}")

    # Load omega if path provided
    if omega is None:
        if omega_path is None:
            raise ValueError("Must provide either omega or omega_path")
        omega = np.load(omega_path)
        print(f"Loaded omega from {omega_path}, shape: {omega.shape}")

    # Check omega dimensions match kernel features
    if omega.shape[0] != len(kernel_feature_cols):
        raise ValueError(
            f"omega shape {omega.shape} doesn't match kernel_feature_cols length {len(kernel_feature_cols)}"
        )

    # Sort and clean data
    df = df.sort_values(time_col).reset_index(drop=True)
    df = df[df[target_col].notna()].reset_index(drop=True)

    # Extract arrays
    X = df[feature_cols]
    X_kernel = df[kernel_feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    s = _sanitize_scale(df[scale_col].to_numpy(dtype=float), min_scale=min_scale)

    # Split data
    if split_method == 'time_ordered':
        sl_train, sl_cal, sl_test = _time_ordered_split(
            len(df), test_frac=test_frac, cal_frac=cal_frac
        )
        X_train, y_train = X.iloc[sl_train], y[sl_train]
        X_cal, y_cal = X.iloc[sl_cal], y[sl_cal]
        X_test, y_test = X.iloc[sl_test], y[sl_test]

        X_kernel_cal = X_kernel[sl_cal]
        X_kernel_test = X_kernel[sl_test]
        s_cal = s[sl_cal]
        s_test = s[sl_test]

    elif split_method == 'random':
        idx_train, idx_cal, idx_test = _random_split(
            len(df), test_frac=test_frac, cal_frac=cal_frac, random_seed=random_seed
        )
        X_train, y_train = X.iloc[idx_train], y[idx_train]
        X_cal, y_cal = X.iloc[idx_cal], y[idx_cal]
        X_test, y_test = X.iloc[idx_test], y[idx_test]

        X_kernel_cal = X_kernel[idx_cal]
        X_kernel_test = X_kernel[idx_test]
        s_cal = s[idx_cal]
        s_test = s[idx_test]

    else:
        raise ValueError(f"split_method must be 'time_ordered' or 'random', got '{split_method}'")

    # Train quantile model
    kw = dict(
        objective='quantile',
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

    # Predictions on calibration and test sets
    y_pred_cal = np.asarray(model.predict(X_cal), dtype=float)
    y_pred_test = np.asarray(model.predict(X_test), dtype=float)

    # Create bundle (store full calibration set)
    bundle = WeightedConformalLowerBundle(
        feature_cols=list(feature_cols),
        kernel_feature_cols=list(kernel_feature_cols),
        scale_col=scale_col,
        model=model,
        X_cal=X_kernel_cal,
        y_cal=y_cal,
        y_pred_cal=y_pred_cal,
        scale_cal=s_cal,
        omega=omega,
        tau=tau,
        alpha_target=alpha_target,
        min_scale=min_scale,
        safety_margin=safety_margin,
    )

    # Test set predictions
    df_test = pd.DataFrame({
        'y': y_test,
        'y_pred_base': y_pred_test,
        'scale': s_test,
    })

    # Add kernel features for localized q_hat computation
    for i, col in enumerate(kernel_feature_cols):
        df_test[col] = X_kernel_test[:, i]

    # Compute localized conformal corrections
    q_hat_test = compute_weighted_conformal_correction_lower(
        X_cal=X_kernel_cal,
        y_cal=y_cal,
        y_pred_cal=y_pred_cal,
        scale_cal=s_cal,
        X_query=X_kernel_test,
        omega=omega,
        tau=tau,
        alpha_target=alpha_target,
        min_scale=min_scale,
        safety_margin=safety_margin,
    )

    df_test['q_hat_local'] = q_hat_test
    df_test['margin'] = q_hat_test * s_test
    df_test['y_pred_conf'] = y_pred_test - df_test['margin'].to_numpy(dtype=float)

    # Metrics
    y_pred_conf = df_test['y_pred_conf'].to_numpy(dtype=float)

    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(df_test['y'], y_pred_conf))),
        'mae': float(mean_absolute_error(df_test['y'], y_pred_conf)),
        'coverage': float((df_test['y'] >= y_pred_conf).mean()),
        'pre_conformal_coverage': float((df_test['y'] >= y_pred_test).mean()),
        'n_train': int(len(y_train)),
        'n_cal': int(len(y_cal)),
        'n_test': int(len(y_test)),
        'split_method': split_method,
        'tau': float(tau),
        'q_hat_mean': float(q_hat_test.mean()),
        'q_hat_std': float(q_hat_test.std()),
        'q_hat_min': float(q_hat_test.min()),
        'q_hat_max': float(q_hat_test.max()),
    }

    return bundle, metrics, df_test


# =============================================================================
# Original Binned Conformal Components (unchanged below)
# =============================================================================


BinningSpec = (
    Literal["y_pred", "y_actual"] | str
)  # "y_pred", "y_actual", or "feature:<colname>"


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
    """
    Extract the binning feature for conformal prediction.

    For "y_actual" binning:
    - During calibration: bins are created from actual values (y_cal)
    - During prediction: uses y_pred_base as proxy for actual bin assignment

    The proxy mapping assumes predictions are reasonable estimates of actuals.
    Coverage guarantee is preserved if proxy quality is good.
    """
    if binning == "y_pred":
        return np.asarray(y_pred_base, dtype=float)

    if binning == "y_actual":
        # At prediction time, use y_pred as proxy for y_actual
        # (During calibration, this function is not used - y_cal is used directly)
        return np.asarray(y_pred_base, dtype=float)

    if isinstance(binning, str) and binning.startswith("feature:"):
        col = binning.split("feature:", 1)[1]
        if col not in df.columns:
            raise ValueError(f"Missing binning feature column '{col}'")
        return np.asarray(df[col].to_numpy(), dtype=float)

    raise ValueError("binning must be 'y_pred', 'y_actual', or 'feature:<colname>'")


def compute_binned_adaptive_conformal_corrections_lower(
    *,
    bin_feature_cal: np.ndarray,
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    scale_cal: np.ndarray,
    alpha_target: float,
    bins: list[float],
    min_scale: float = 1e-3,
    safety_margin: float = 0.02,
    min_q_hat_ratio: float = 0.1,
) -> tuple[float, dict]:
    """
    Compute global + per-bin q_hat for *lower bound* conformal correction.

    Nonconformity score (normalized miss on the lower side):
      r = max(0, (y_pred - y_true)/scale)

    q_hat is the conformal quantile of r at level computed by _conformal_q_level(n, alpha_target, safety_margin).

    Parameters
    ----------
    safety_margin : float, default=0.0
        Additional buffer for conservativeness (e.g., 0.02 for 2% extra coverage)
        Higher values = more conservative bounds, higher empirical coverage
        Typical values: 0.01-0.03
    min_q_hat_ratio : float, default=0.1
        Minimum ratio of bin q_hat to global q_hat (floor for bin corrections)
        If bin q_hat < global_q_hat * min_q_hat_ratio, use global q_hat instead
        Prevents bins with all under-predictions from having zero correction
        Typical values: 0.05-0.2 (0.1 = 10% of global correction minimum)

    Returns
    -------
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
    q_level_g = _conformal_q_level(
        int(r.size), float(alpha_target), safety_margin=safety_margin
    )
    q_hat_global_r = float(np.quantile(r, q_level_g))

    # Per bin
    df = pd.DataFrame({"bin_feature": bin_feature_cal, "r": r})
    df["bin"] = pd.cut(df["bin_feature"], bins=bins, include_lowest=True, right=True)

    # Minimum q_hat floor (prevents zero corrections in bins with systematic under-prediction)
    min_q_hat_floor = q_hat_global_r * min_q_hat_ratio

    q_hat_by_bin_r: dict = {}
    for b in df["bin"].cat.categories:
        r_b = df.loc[df["bin"] == b, "r"].to_numpy(dtype=float)
        if r_b.size == 0:
            continue
        q_level_b = _conformal_q_level(
            int(r_b.size), float(alpha_target), safety_margin=safety_margin
        )
        q_hat_bin = float(np.quantile(r_b, q_level_b))

        # Apply floor: use max(bin q_hat, floor) to prevent zero corrections
        # This handles bins where model systematically under-predicts (all r ≈ 0)
        q_hat_by_bin_r[b] = max(q_hat_bin, min_q_hat_floor)

    return q_hat_global_r, q_hat_by_bin_r


def _time_ordered_split(
    n: int,
    *,
    test_frac: float,
    cal_frac: float,
) -> tuple[slice, slice, slice]:
    """
    Create time-ordered train/cal/test split.

    Returns slices: train | cal | test (chronological order).
    """
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


def _random_split(
    n: int,
    *,
    test_frac: float,
    cal_frac: float,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create random train/cal/test split.

    Returns index arrays (not slices) for random assignment.
    Useful when temporal order is not important or when testing
    generalization across all time periods.
    """
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

    # Random permutation of indices
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n)

    # Split indices
    idx_train = indices[:n_train]
    idx_cal = indices[n_train : n_train + n_cal]
    idx_test = indices[n_train + n_cal :]

    return idx_train, idx_cal, idx_test


def train_wind_lower_model_conformal_binned(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "y",
    time_col: str = "TIME_HOURLY",
    test_frac: float = 0.25,
    cal_frac: float = 0.25,
    quantile_alpha: float = 0.10,
    alpha_target: float = 0.90,
    scale_col: str = "ens_std",
    binning: BinningSpec = "y_pred",
    bin_edges: list[float] | None = None,
    n_bins: int | None = None,
    bin_quantiles: list[float] | None = None,
    min_scale: float = 1e-3,
    model_kwargs: dict | None = None,
    safety_margin: float = 0.0,
    min_q_hat_ratio: float = 0.1,
    split_method: str = "random",
    random_seed: int = 42,
) -> tuple[ConformalLowerBundle, dict[str, Any], pd.DataFrame]:
    """
    Train a quantile regression model and calibrate a conformal *lower bound* with optional binning.

    The split can be time-ordered or random:
      - time_ordered: train | cal | test (chronological, default)
      - random: random assignment to train/cal/test (tests generalization across all times)

    Parameters
    ----------
    binning : BinningSpec, default="y_pred"
        Binning strategy for adaptive conformal correction:
        - "y_pred": Bin by predictions (standard approach)
        - "y_actual": Bin by actual values during calibration, use predictions as proxy during prediction
                      Useful when errors vary by actual generation level
                      Coverage guarantee depends on prediction quality as proxy
        - "feature:<colname>": Bin by a specific feature column (e.g., "feature:ens_std")
    bin_edges : list[float] | None
        Explicit bin edges (highest priority)
    n_bins : int | None
        Number of equal-width bins to create
        - n_bins=1: Single bin (no adaptive binning, uses global q_hat)
        - n_bins=2+: Adaptive binning with multiple bins
        Used if bin_edges is None
    bin_quantiles : list[float] | None
        Quantile-based bin edges (used if both bin_edges and n_bins are None)
        Default: [0.0, 0.1, 0.2, ..., 1.0] (10 quantile-based bins)
    safety_margin : float, default=0.0
        Additional buffer for conservativeness (e.g., 0.02 for 2% extra coverage)
        Higher values = wider bounds, higher empirical coverage
        Use 0.01-0.03 to reduce out-of-sample failures at lower alpha values
    min_q_hat_ratio : float, default=0.1
        Minimum ratio of bin q_hat to global q_hat (floor for bin corrections)
        Prevents bins with systematic under-prediction from having zero correction
        Especially important for y_actual binning where high actual bins may have all r=0
        Typical values: 0.05-0.2 (0.1 = 10% of global correction minimum)
        Set to 0.0 to disable floor (not recommended for y_actual binning)
    split_method : str, default="time_ordered"
        Method for splitting data into train/cal/test sets:
        - "time_ordered": Use first N points for train, next M for cal, last K for test
                          Respects temporal structure, tests future prediction
        - "random": Randomly assign points to train/cal/test sets
                    Tests generalization across all time periods, more stable estimates
    random_seed : int, default=42
        Random seed for reproducible splits when split_method="random"
        Ignored when split_method="time_ordered"

    Returns
    -------
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

    # Split data based on method
    if split_method == "time_ordered":
        sl_train, sl_cal, sl_test = _time_ordered_split(
            len(df), test_frac=test_frac, cal_frac=cal_frac
        )
        X_train, y_train = X.iloc[sl_train], y[sl_train]
        X_cal, y_cal = X.iloc[sl_cal], y[sl_cal]
        X_test, y_test = X.iloc[sl_test], y[sl_test]
        s_cal = s[sl_cal]
        s_test = s[sl_test]

    elif split_method == "random":
        idx_train, idx_cal, idx_test = _random_split(
            len(df), test_frac=test_frac, cal_frac=cal_frac, random_seed=random_seed
        )
        X_train, y_train = X.iloc[idx_train], y[idx_train]
        X_cal, y_cal = X.iloc[idx_cal], y[idx_cal]
        X_test, y_test = X.iloc[idx_test], y[idx_test]
        s_cal = s[idx_cal]
        s_test = s[idx_test]

    else:
        raise ValueError(
            f"split_method must be 'time_ordered' or 'random', got '{split_method}'"
        )

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
    elif binning == "y_actual":
        # Calibration: bin by ACTUAL values
        bin_feature_cal = y_cal
        # Test: use PREDICTIONS as proxy for actuals (for evaluation)
        # Note: At true prediction time (via bundle.predict_df), proxy mapping is automatic
        bin_feature_test = y_pred_test
    elif isinstance(binning, str) and binning.startswith("feature:"):
        col = binning.split("feature:", 1)[1]
        if col not in df.columns:
            raise ValueError(f"Missing binning feature column '{col}'")
        arr = np.asarray(df[col].to_numpy(), dtype=float)
        # Handle both slice (time_ordered) and array (random) indices
        if split_method == "time_ordered":
            bin_feature_cal = arr[sl_cal]
            bin_feature_test = arr[sl_test]
        else:  # random
            bin_feature_cal = arr[idx_cal]
            bin_feature_test = arr[idx_test]
    else:
        raise ValueError("binning must be 'y_pred', 'y_actual', or 'feature:<colname>'")

    # Choose bin edges if not provided
    if bin_edges is None:
        if n_bins is not None:
            # Equal-width bins
            # Note: n_bins=1 creates a single bin (no adaptive binning, equivalent to global q_hat)
            if n_bins < 1:
                raise ValueError("n_bins must be >= 1")
            min_val = float(np.min(bin_feature_cal))
            max_val = float(np.max(bin_feature_cal))
            # Add small margin to ensure all points are included
            margin = (max_val - min_val) * 0.001
            edges = np.linspace(min_val - margin, max_val + margin, n_bins + 1)
            bin_edges = edges.tolist()
        else:
            # Quantile-based bins
            qs = (
                bin_quantiles
                if bin_quantiles
                else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
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
        safety_margin=safety_margin,
        min_q_hat_ratio=min_q_hat_ratio,
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

    # Compute split sizes (works for both slice and array indices)
    n_train = len(y_train)
    n_cal = len(y_cal)
    n_test = len(y_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(df_test["y"], y_pred_conf))),
        "mae": float(mean_absolute_error(df_test["y"], y_pred_conf)),
        "coverage": float((df_test["y"] >= y_pred_conf).mean()),
        "pre_conformal_coverage": float((df_test["y"] >= y_pred_test).mean()),
        "q_hat_global_r": q_hat_global_r,
        "q_hat_by_bin_r": q_hat_by_bin_r,
        "n_train": int(n_train),
        "n_cal": int(n_cal),
        "n_test": int(n_test),
        "split_method": split_method,
    }

    return bundle, metrics, df_test
