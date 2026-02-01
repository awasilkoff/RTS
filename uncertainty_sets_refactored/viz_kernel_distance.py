"""
Visualize the 2D feature space with kernel distance coloring.

Shows:
- All points in (SYS_MEAN, SYS_STD) space
- A highlighted target point
- Color coding by kernel weight (distance) from the target
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils import fit_standard_scaler
from data_processing import build_XY_for_covariance_system_only


def compute_kernel_weights(
    X_target: np.ndarray,
    X_all: np.ndarray,
    omega: np.ndarray,
    tau: float,
    min_weight: float = 1e-300,
) -> np.ndarray:
    """
    Compute kernel weights from target point to all points.

    weight[i] = exp(-d^2 / tau) where d^2 = sum_k omega[k] * (X_target[k] - X_all[i,k])^2

    Parameters
    ----------
    min_weight : float
        Minimum weight value to avoid underflow to zero (important for LogNorm plotting).
        Set to 1e-300 by default to handle extreme distances while still being plottable.
    """
    diff = X_all - X_target  # (N, K)
    sq_dist = np.sum(omega * diff**2, axis=1)  # (N,)
    weights = np.exp(-sq_dist / tau)
    # Clip to minimum to avoid exact zeros (which break LogNorm)
    weights = np.maximum(weights, min_weight)
    return weights


def compute_knn_binary_weights(
    X_target: np.ndarray,
    X_all: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Compute binary k-NN weights: 1 for k nearest neighbors, 0 otherwise.
    Uses Euclidean distance.
    """
    diff = X_all - X_target  # (N, K)
    sq_dist = np.sum(diff**2, axis=1)  # (N,)

    # Find k nearest neighbors
    knn_indices = np.argpartition(sq_dist, k)[:k]

    # Binary weights: 1 for k-NN, 0 for others
    weights = np.zeros(len(X_all))
    weights[knn_indices] = 1.0
    return weights


def plot_kernel_distance(
    X: np.ndarray,
    x_cols: list[str],
    times: pd.DatetimeIndex,
    target_idx: int,
    omega: np.ndarray,
    tau: float,
    title: str | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    target_color: str = "red",
    target_size: int = 150,
    point_size: int = 20,
    show_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Axes:
    """
    Plot all points in 2D feature space, colored by kernel weight from target.

    Parameters
    ----------
    vmin, vmax : float, optional
        Explicit colorbar limits. If provided, overrides auto-scaling.
        Useful for ensuring consistent scales across multiple plots.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    X_target = X[target_idx]
    weights = compute_kernel_weights(X_target, X, omega, tau)

    # Determine color scale limits
    # Use a small floor to avoid log(0), but allow very small weights to show
    if vmin is None:
        vmin = max(weights.min(), 1e-300)
    if vmax is None:
        vmax = weights.max()

    # Plot all points colored by weight
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=weights,
        cmap=cmap,
        s=point_size,
        alpha=0.7,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    # Highlight target point
    ax.scatter(
        X_target[0],
        X_target[1],
        c=target_color,
        s=target_size,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=10,
        label=f"Target: {times[target_idx]}",
    )

    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Kernel Weight (log scale)")

    ax.set_xlabel(x_cols[0] if len(x_cols) > 0 else "Feature 0")
    ax.set_ylabel(x_cols[1] if len(x_cols) > 1 else "Feature 1")
    ax.legend(loc="upper right")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Kernel Distance from Target (tau={tau}, omega={omega})")

    return ax


def plot_global_uniform_weights(
    X: np.ndarray,
    x_cols: list[str],
    times: pd.DatetimeIndex,
    target_idx: int,
    title: str | None = None,
    ax: plt.Axes | None = None,
    uniform_color: str = "steelblue",
    target_color: str = "red",
    target_size: int = 150,
    point_size: int = 20,
) -> plt.Axes:
    """
    Plot all points in 2D feature space with UNIFORM color.
    Represents global covariance where all points get equal weight.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    X_target = X[target_idx]

    # Plot ALL points with same color (uniform weight)
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=uniform_color,
        s=point_size,
        alpha=0.6,
        label="All points (equal weight)",
    )

    # Highlight target point
    ax.scatter(
        X_target[0],
        X_target[1],
        c=target_color,
        s=target_size,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=10,
        label=f"Target: {times[target_idx]}",
    )

    ax.set_xlabel(x_cols[0] if len(x_cols) > 0 else "Feature 0")
    ax.set_ylabel(x_cols[1] if len(x_cols) > 1 else "Feature 1")
    ax.legend(loc="upper right")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Global Covariance: All Points Equal Weight")

    return ax


def plot_knn_binary_weights(
    X: np.ndarray,
    x_cols: list[str],
    times: pd.DatetimeIndex,
    target_idx: int,
    k: int,
    title: str | None = None,
    ax: plt.Axes | None = None,
    knn_color: str = "forestgreen",
    other_color: str = "lightgray",
    target_color: str = "red",
    target_size: int = 150,
    point_size: int = 20,
) -> plt.Axes:
    """
    Plot all points with binary k-NN coloring.
    k nearest neighbors get one color, all others get a different color.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    X_target = X[target_idx]
    weights = compute_knn_binary_weights(X_target, X, k)

    # Separate points into k-NN and others
    knn_mask = weights == 1.0
    other_mask = ~knn_mask

    # Plot non-neighbors first (background)
    ax.scatter(
        X[other_mask, 0],
        X[other_mask, 1],
        c=other_color,
        s=point_size,
        alpha=0.4,
        label=f"Other points (weight=0)",
    )

    # Plot k nearest neighbors
    ax.scatter(
        X[knn_mask, 0],
        X[knn_mask, 1],
        c=knn_color,
        s=point_size * 1.5,
        alpha=0.8,
        label=f"k={k} nearest neighbors (weight=1/k)",
    )

    # Highlight target point
    ax.scatter(
        X_target[0],
        X_target[1],
        c=target_color,
        s=target_size,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=10,
        label=f"Target: {times[target_idx]}",
    )

    ax.set_xlabel(x_cols[0] if len(x_cols) > 0 else "Feature 0")
    ax.set_ylabel(x_cols[1] if len(x_cols) > 1 else "Feature 1")
    ax.legend(loc="upper right")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Euclidean k-NN: k={k} Nearest Neighbors")

    return ax


def plot_kernel_distance_comparison(
    X: np.ndarray,
    x_cols: list[str],
    times: pd.DatetimeIndex,
    target_idx: int,
    omega_equal: np.ndarray,
    omega_learned: np.ndarray,
    tau: float,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (16, 7),
) -> plt.Figure:
    """
    Side-by-side comparison of kernel weights with equal vs learned omega.

    **Important:** Both plots use the same colorbar scale for direct visual comparison.

    Parameters
    ----------
    X : np.ndarray
        Standardized features, shape (N, K).
    x_cols : list[str]
        Feature column names.
    times : pd.DatetimeIndex
        Timestamps corresponding to X rows.
    target_idx : int
        Index of target point to compute distances from.
    omega_equal : np.ndarray
        Equal weights (e.g., [1, 1]).
    omega_learned : np.ndarray
        Learned omega from optimization.
    tau : float
        Temperature parameter.
    save_path : Path, optional
        If provided, save figure to this path.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Compute kernel weights for both omegas
    X_target = X[target_idx]
    weights_equal = compute_kernel_weights(X_target, X, omega_equal, tau)
    weights_learned = compute_kernel_weights(X_target, X, omega_learned, tau)

    # Find global min/max for shared colorbar scale
    vmin = max(min(weights_equal.min(), weights_learned.min()), 1e-6)
    vmax = max(weights_equal.max(), weights_learned.max())

    # Left: Equal weights
    plot_kernel_distance(
        X,
        x_cols,
        times,
        target_idx,
        omega_equal,
        tau,
        title=f"Equal Weights: ω = {omega_equal}",
        ax=axes[0],
        cmap="viridis",
        show_colorbar=True,
        vmin=vmin,
        vmax=vmax,
    )

    # Right: Learned weights
    plot_kernel_distance(
        X,
        x_cols,
        times,
        target_idx,
        omega_learned,
        tau,
        title=f"Learned Weights: ω = [{omega_learned[0]:.2f}, {omega_learned[1]:.2f}]",
        ax=axes[1],
        cmap="viridis",
        show_colorbar=True,
        vmin=vmin,
        vmax=vmax,
    )

    fig.suptitle(
        f"Kernel Weight Comparison — Target: {times[target_idx].strftime('%Y-%m-%d %H:%M')}",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_multiple_targets(
    X: np.ndarray,
    x_cols: list[str],
    times: pd.DatetimeIndex,
    target_indices: list[int],
    omega: np.ndarray,
    tau: float,
    save_path: Path | None = None,
):
    """
    Create a grid of plots for multiple target points.
    """
    n = len(target_indices)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, target_idx in enumerate(target_indices):
        plot_kernel_distance(
            X,
            x_cols,
            times,
            target_idx,
            omega,
            tau,
            title=f"Target: {times[target_idx].strftime('%Y-%m-%d %H:%M')}",
            ax=axes[i],
            show_colorbar=True,
        )

    # Hide unused axes
    n_used = len(target_indices)
    for j in range(n_used, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def main():
    DATA_DIR = Path(__file__).parent / "data"
    ART = DATA_DIR / "viz_artifacts"
    ART.mkdir(exist_ok=True, parents=True)

    # Load data
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )

    X, Y, times, x_cols, y_cols = build_XY_for_covariance_system_only(
        forecasts, actuals, drop_any_nan_rows=True
    )

    # Standardize features (same as training)
    scaler = fit_standard_scaler(X)
    Xs = scaler.transform(X)

    print(f"Data: {X.shape[0]} points, {X.shape[1]} features")
    print(f"Features: {x_cols}")
    print(f"Time range: {times[0]} to {times[-1]}")

    # Default omega (equal weights) and tau
    # TODO: Update with learned omega after sweep
    omega = np.array([1.0, 1.0])
    tau = 20.0

    # Select a few target points spread across the dataset
    n = len(times)
    target_indices = [
        n // 4,  # 25%
        n // 2,  # 50%
        3 * n // 4,  # 75%
    ]

    # Single detailed plot for middle point
    fig, ax = plt.subplots(figsize=(12, 9))
    plot_kernel_distance(
        Xs,
        x_cols,
        times,
        target_idx=n // 2,
        omega=omega,
        tau=tau,
        ax=ax,
    )
    plt.savefig(ART / "kernel_distance_single.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {ART / 'kernel_distance_single.png'}")

    # Multi-target comparison
    plot_multiple_targets(
        Xs,
        x_cols,
        times,
        target_indices=target_indices,
        omega=omega,
        tau=tau,
        save_path=ART / "kernel_distance_multi.png",
    )

    plt.show()


if __name__ == "__main__":
    main()
