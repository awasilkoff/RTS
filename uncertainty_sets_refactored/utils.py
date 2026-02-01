from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CachedPaths:
    actuals_parquet: Path
    forecasts_parquet: Path
    output_dir: Path | None = None


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8
    ddof: int = 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * self.std + self.mean


def fit_standard_scaler(
    X: np.ndarray, *, eps: float = 1e-8, ddof: int = 0, nan_policy: str = "error"
) -> StandardScaler:
    X = np.asarray(X)

    if nan_policy not in {"error", "ignore"}:
        raise ValueError("nan_policy must be 'error' or 'ignore'")

    if nan_policy == "error":
        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN/inf. Use nan_policy='ignore' or clean X first."
            )
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=ddof)
    else:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0, ddof=ddof)

    std_safe = np.where(std < eps, 1.0, std)
    return StandardScaler(mean=mean, std=std_safe, eps=eps, ddof=ddof)


@dataclass(frozen=True)
class MinMaxScaler:
    """Scale features to [0, 1] range."""
    min_val: np.ndarray
    max_val: np.ndarray
    eps: float = 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale to [0, 1]."""
        return (X - self.min_val) / (self.max_val - self.min_val + self.eps)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform from [0, 1] back to original scale."""
        return X_scaled * (self.max_val - self.min_val + self.eps) + self.min_val


def fit_minmax_scaler(X: np.ndarray, *, eps: float = 1e-8) -> MinMaxScaler:
    """Fit MinMax scaler to data."""
    X = np.asarray(X)
    return MinMaxScaler(
        min_val=X.min(axis=0),
        max_val=X.max(axis=0),
        eps=eps,
    )


def fit_scaler(X: np.ndarray, scaler_type: str) -> StandardScaler | MinMaxScaler | None:
    """Dispatch function to fit appropriate scaler."""
    if scaler_type == "standard":
        return fit_standard_scaler(X)
    elif scaler_type == "minmax":
        return fit_minmax_scaler(X)
    elif scaler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")


def apply_scaler(X: np.ndarray, scaler: StandardScaler | MinMaxScaler | None) -> np.ndarray:
    """Apply scaler to data, or return as-is if None."""
    if scaler is None:
        return X
    return scaler.transform(X)


def _conformal_q_level(n: int, alpha_target: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    q = np.ceil((n + 1) * alpha_target) / n
    return float(min(q, 1.0))
