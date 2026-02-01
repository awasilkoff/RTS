"""
Projection utilities for high-D → 2D visualization of learned omega.

Provides clean 2D visualizations of kernel distances and weights in high-dimensional
feature spaces by selecting the most important dimensions based on learned omega values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def project_top2_omega_weighted(
    X: np.ndarray,
    omega: np.ndarray,
    x_cols: list[str],
) -> Tuple[np.ndarray, list[str], np.ndarray]:
    """
    Select top 2 features by omega magnitude for 2D projection.

    Parameters
    ----------
    X : (N, D) array
        Feature matrix
    omega : (D,) array
        Learned feature weights
    x_cols : list[str]
        Feature names

    Returns
    -------
    X_2d : (N, 2) array
        Projected features (top 2 dimensions)
    feature_names : list[str]
        Names of selected features
    top2_idx : (2,) array
        Indices of selected features
    """
    omega = np.asarray(omega)
    top2_idx = np.argsort(np.abs(omega))[-2:][::-1]  # Descending by magnitude

    X_2d = X[:, top2_idx]
    feature_names = [x_cols[i] for i in top2_idx]

    return X_2d, feature_names, top2_idx


def plot_kernel_distance_2d_projection(
    X: np.ndarray,
    omega: np.ndarray,
    x_cols: list[str],
    times: pd.DatetimeIndex,
    target_idx: int,
    tau: float,
    projection_method: str = "top2",
    out_path: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    title_suffix: str = "",
):
    """
    Kernel distance plot for high-D features with 2D projection.

    Projects high-dimensional feature space to 2D using top-2 omega-weighted features,
    then visualizes kernel weights around a target point.

    Parameters
    ----------
    X : (N, D) array
        Full feature matrix
    omega : (D,) array
        Learned feature weights
    x_cols : list[str]
        Feature names
    times : DatetimeIndex
        Timestamps for samples
    target_idx : int
        Index of target point to visualize neighborhood
    tau : float
        Kernel bandwidth parameter
    projection_method : str
        Projection method ("top2" only for now)
    out_path : Path, optional
        Save path for figure
    figsize : tuple
        Figure size
    title_suffix : str
        Additional text for title
    """
    if projection_method != "top2":
        raise NotImplementedError("Only 'top2' projection supported")

    # Project to 2D
    X_2d, feature_names, top2_idx = project_top2_omega_weighted(X, omega, x_cols)
    omega_2d = omega[top2_idx]

    # Compute kernel weights in FULL space (not projected)
    x_target = X[target_idx]
    diff = X - x_target[None, :]  # (N, D)
    # omega[None, :] has shape (1, D) to broadcast with diff (N, D)
    dist_sq = np.sum(omega[None, :] * diff * diff, axis=1)  # (N,)
    weights = np.exp(-dist_sq / tau)
    weights /= weights.sum()

    # Visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Color by kernel weight
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=weights,
        s=50,
        alpha=0.6,
        cmap="viridis",
        edgecolors="k",
        linewidth=0.3,
    )

    # Highlight target point
    ax.scatter(
        X_2d[target_idx, 0],
        X_2d[target_idx, 1],
        c="red",
        s=300,
        marker="*",
        edgecolors="black",
        linewidth=2,
        label="Target",
        zorder=10,
    )

    # Axis labels with omega values
    ax.set_xlabel(
        f"{feature_names[0]} (ω={omega_2d[0]:.3f})",
        fontsize=12,
    )
    ax.set_ylabel(
        f"{feature_names[1]} (ω={omega_2d[1]:.3f})",
        fontsize=12,
    )

    title = f"Kernel Weights in {len(omega)}D Space (2D Projection: Top 2 Features)\nτ={tau:.1f}"
    if title_suffix:
        title += f" | {title_suffix}"
    ax.set_title(title, fontsize=13)

    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, label="Kernel Weight")
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        plt.show()


def plot_omega_bar_chart(
    omega: np.ndarray,
    x_cols: list[str],
    out_path: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Learned Feature Weights (ω)",
    highlight_top_k: int = 2,
):
    """
    Bar chart showing learned omega values for each feature.

    Parameters
    ----------
    omega : (D,) array
        Learned feature weights
    x_cols : list[str]
        Feature names
    out_path : Path, optional
        Save path for figure
    figsize : tuple
        Figure size
    title : str
        Plot title
    highlight_top_k : int
        Number of top features to highlight in different color
    """
    omega = np.asarray(omega)

    # Sort by magnitude
    sorted_idx = np.argsort(np.abs(omega))[::-1]
    omega_sorted = omega[sorted_idx]
    cols_sorted = [x_cols[i] for i in sorted_idx]

    # Colors: highlight top k
    colors = ["green" if i < highlight_top_k else "gray" for i in range(len(omega))]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        range(len(omega_sorted)),
        omega_sorted,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_xticks(range(len(cols_sorted)))
    ax.set_xticklabels(cols_sorted, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("ω (Feature Weight)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, omega_sorted)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1 if val > 0 else bar.get_height() - 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontsize=10,
            fontweight="bold" if i < highlight_top_k else "normal",
        )

    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        plt.show()


def plot_2d_projection_with_ellipses(
    X: np.ndarray,
    omega: np.ndarray,
    x_cols: list[str],
    times: pd.DatetimeIndex,
    sample_indices: list[int],
    Sigma_list: list[np.ndarray],
    rho: float = 1.0,
    out_path: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str = "Uncertainty Ellipses (2D Projection)",
):
    """
    Plot 2D projection with uncertainty ellipses at selected sample points.

    Projects to top 2 omega-weighted dimensions and overlays uncertainty ellipses
    from covariance matrices.

    Parameters
    ----------
    X : (N, D) array
        Feature matrix
    omega : (D,) array
        Learned feature weights
    x_cols : list[str]
        Feature names
    times : DatetimeIndex
        Timestamps
    sample_indices : list[int]
        Indices of samples to show ellipses for
    Sigma_list : list of (M, M) arrays
        Covariance matrices for selected samples
    rho : float
        Ellipsoid radius scaling
    out_path : Path, optional
        Save path
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    # Project to 2D
    X_2d, feature_names, top2_idx = project_top2_omega_weighted(X, omega, x_cols)
    omega_2d = omega[top2_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter all points
    ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c="lightgray",
        s=20,
        alpha=0.5,
        edgecolors="k",
        linewidth=0.2,
    )

    # Overlay ellipses at sample points
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_indices)))

    for idx, Sigma, color in zip(sample_indices, Sigma_list, colors):
        # Project covariance to 2D (diagonal of selected dimensions)
        Sigma_2d = Sigma[np.ix_(top2_idx, top2_idx)]

        # Plot ellipse (2σ contour for visualization)
        from matplotlib.patches import Ellipse

        # Eigendecomposition for ellipse parameters
        eigvals, eigvecs = np.linalg.eigh(Sigma_2d)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * rho * np.sqrt(eigvals)

        ellipse = Ellipse(
            xy=(X_2d[idx, 0], X_2d[idx, 1]),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            alpha=0.3,
            edgecolor=color,
            linewidth=2,
            label=f"Sample {idx}",
        )
        ax.add_patch(ellipse)

        # Mark center
        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            c=[color],
            s=100,
            marker="x",
            linewidth=3,
            zorder=10,
        )

    ax.set_xlabel(f"{feature_names[0]} (ω={omega_2d[0]:.3f})", fontsize=12)
    ax.set_ylabel(f"{feature_names[1]} (ω={omega_2d[1]:.3f})", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        plt.show()
