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


def _conformal_q_level(n: int, alpha_target: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    q = np.ceil((n + 1) * alpha_target) / n
    return float(min(q, 1.0))
