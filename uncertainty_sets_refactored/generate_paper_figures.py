"""
Paper Figure Generation Orchestrator.

Generates all IEEE paper figures using existing visualization code and saved experiment data.
Uses saved sweep results where available; generates conformal figures fresh with global (1-bin) conformal.

Usage:
    cd uncertainty_sets_refactored
    python generate_paper_figures.py

Outputs saved to: data/viz_artifacts/paper_final/
    figures/
        fig1_kernel_distance.pdf    - Kernel distance: Learned (k=64, τ=1) vs Euclidean k-NN
        fig2_ellipsoid_3d.pdf       - 3D side-by-side: Global vs Learned, k-NN vs Learned
        fig3_nll_vs_k.pdf           - NLL vs k sweep for k-NN
        fig4_nll_vs_tau.pdf         - NLL vs tau sweep for learned omega (16D)
        fig5_nll_16d.pdf            - NLL comparison bar chart (16D features)
        fig5b_nll_boxplot.pdf       - NLL box and whisker plot (16D features)
        fig6_calibration.pdf        - Global conformal calibration (with tolerance + error bars)
        fig6b_calibration_clean.pdf - Global conformal calibration (no tolerance, with error bars)
        fig6c_calibration_points.pdf - Global conformal calibration (points only, no bars)
        fig7_corrections.pdf        - Global conformal correction (q_hat) vs alpha
        fig7b_normalized_lower_bound.pdf - Std devs below mean vs alpha
        fig7c_lb_decomposition.pdf  - Stacked bar: base quantile + conformal correction
        fig8_ellipse_grid.pdf       - 2D ellipse grid with consistent axes
        fig9_omega_bar_chart.png    - Learned feature weights bar chart
        fig10_ellipse_overlay.pdf   - Ellipse overlay (k=32, 512, global)
    tables/
        tab_nll_vs_tau.tex          - NLL at different tau values
        tab_nll_vs_k.tex            - NLL at different k values
        tab_omega_weights.tex       - Learned omega per feature set
    figure_metadata.json

Required prerequisites:
    - Run sweep_and_viz_feature_set.py for focused_2d and high_dim_16d
    - Run sweep_knn_k_values.py for k-NN baseline data
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_config import (
    setup_plotting,
    IEEE_COL_WIDTH,
    IEEE_TWO_COL_WIDTH,
    COLORS,
    FIGURE_DEFAULTS,
)

# Apply global plotting config
setup_plotting()

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = Path(__file__).parent / "data"
VIZ_ARTIFACTS = DATA_DIR / "viz_artifacts"
OUTPUT_DIR = VIZ_ARTIFACTS / "paper_final"

# Experiment data paths
FOCUSED_2D_DIR = VIZ_ARTIFACTS / "focused_2d"
HIGH_DIM_16D_DIR = VIZ_ARTIFACTS / "high_dim_16d"
KNN_SWEEP_DIR = VIZ_ARTIFACTS / "knn_k_sweep"
PAPER_FIGURES_DIR = VIZ_ARTIFACTS / "paper_figures"

# Anonymized wind resource labels (Y columns are sorted: 122, 309, 317)
WIND_LABELS = {0: "Wind 1", 1: "Wind 2", 2: "Wind 3"}


def _ensure_output_dirs():
    """Create output directories."""
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)


def _load_sweep_results(feature_set_dir: Path) -> pd.DataFrame:
    """Load sweep_results.csv from a feature set directory."""
    csv_path = feature_set_dir / "sweep_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Sweep results not found: {csv_path}")
    return pd.read_csv(csv_path)


def _load_omega(feature_set_dir: Path) -> np.ndarray:
    """Load best_omega.npy from a feature set directory."""
    npy_path = feature_set_dir / "best_omega.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Omega not found: {npy_path}")
    return np.load(npy_path)


def _load_feature_config(feature_set_dir: Path) -> dict:
    """Load feature_config.json from a feature set directory."""
    json_path = feature_set_dir / "feature_config.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Feature config not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def _load_knn_sweep() -> pd.DataFrame:
    """Load k-NN sweep results."""
    csv_path = KNN_SWEEP_DIR / "knn_k_sweep_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"k-NN sweep results not found: {csv_path}")
    return pd.read_csv(csv_path)


def _load_calibration_metadata() -> dict:
    """Load conformal calibration metadata."""
    json_path = PAPER_FIGURES_DIR / "figure_metadata.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration metadata not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


# ============================================================================
# FIGURE 1: Kernel Distance Comparison (Learned vs Euclidean k-NN)
# ============================================================================
def fig1_kernel_distance_comparison(
    feature_set_dir: Path = FOCUSED_2D_DIR,
    output_path: Path = None,
    k: int = 64,
    tau: float = 1.0,
) -> plt.Figure:
    """
    Generate kernel distance comparison figure.

    Shows side-by-side: Euclidean k-NN (uniform) vs Learned omega kernel weights.
    Uses optimal hyperparameters: k=64, tau=1.0.
    """
    from viz_kernel_distance import (
        compute_kernel_weights,
        compute_knn_binary_weights,
    )
    from data_processing import build_XY_for_covariance_system_only
    from utils import fit_standard_scaler

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig1_kernel_distance"

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

    # Standardize
    scaler = fit_standard_scaler(X)
    Xs = scaler.transform(X)

    # Load learned omega
    omega_learned = _load_omega(feature_set_dir)

    # Select target point (middle of dataset)
    target_idx = len(Xs) // 2
    X_target = Xs[target_idx]

    # Compute weights
    weights_knn = compute_knn_binary_weights(X_target, Xs, k=k)
    weights_learned = compute_kernel_weights(X_target, Xs, omega_learned, tau)

    # Create figure (IEEE two-column width)
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_TWO_COL_WIDTH, 3.0))

    # Left: k-NN binary weights
    ax = axes[0]
    knn_mask = weights_knn == 1.0
    ax.scatter(Xs[~knn_mask, 0], Xs[~knn_mask, 1], c="lightgray", s=15, alpha=0.4)
    ax.scatter(
        Xs[knn_mask, 0],
        Xs[knn_mask, 1],
        c=COLORS["knn"],
        s=25,
        alpha=0.8,
        label=f"k={k} neighbors",
    )
    ax.scatter(
        X_target[0],
        X_target[1],
        c=COLORS["learned"],
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=10,
        label="Query",
    )
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    # ax.set_title(f"Euclidean k-NN (k={k})")  # caption in paper
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Right: Learned kernel weights
    ax = axes[1]
    scatter = ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=weights_learned,
        cmap="viridis",
        s=15,
        alpha=0.7,
        norm=plt.matplotlib.colors.LogNorm(
            vmin=max(weights_learned.min(), 1e-6), vmax=weights_learned.max()
        ),
    )
    ax.scatter(
        X_target[0],
        X_target[1],
        c=COLORS["learned"],
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=10,
        label="Query",
    )
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    # Title omitted — caption in paper
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Kernel Weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    # Save
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 2: 3D Ellipsoid Comparison (Learned, k-NN, Global)
# ============================================================================
def fig2_3d_ellipsoid_comparison(
    feature_set_dir: Path = HIGH_DIM_16D_DIR,
    output_path: Path = None,
    sample_idx: int = 0,
    knn_k: int = 16,
    tau: float = 1.0,
    rho: float = 1.0,
    offset: float = 20.0,
) -> plt.Figure:
    """
    Side-by-side 3D ellipsoid comparison.

    Left panel:  Global (blue) vs Learned (orange)
    Right panel: k-NN k=512 (teal) vs Learned (orange)
    Learned appears in both for direct comparison.
    """
    from sweep_knn_k_values import (
        ellipsoid_surface_3d,
        sweep_k_values,
        compute_learned_omega_baseline,
    )
    from data_processing_extended import FEATURE_BUILDERS
    from utils import fit_scaler, apply_scaler

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig2_ellipsoid_3d"

    # Load 16D features to match omega dimensionality
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )

    config = _load_feature_config(feature_set_dir)
    feature_set_name = config.get("feature_set", "high_dim_16d")
    build_fn = FEATURE_BUILDERS.get(feature_set_name)
    if build_fn is None:
        build_fn = FEATURE_BUILDERS["high_dim_16d"]

    X_raw, Y, times, x_cols, y_cols = build_fn(
        forecasts, actuals, drop_any_nan_rows=True
    )

    # Split and standardize (same seed as sweep)
    n = X_raw.shape[0]
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_train = int(0.5 * n)
    n_val = int(0.25 * n)
    train_idx = indices[:n_train]
    eval_idx = indices[n_train : n_train + n_val]

    scaler = fit_scaler(X_raw[train_idx], "standard")
    X = apply_scaler(X_raw, scaler)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_eval, Y_eval = X[eval_idx], Y[eval_idx]

    # Load learned omega
    omega = _load_omega(feature_set_dir)

    # Compute k-NN ellipsoid
    knn_results = sweep_k_values(X_train, Y_train, X_eval, Y_eval, [knn_k])

    # Compute learned omega ellipsoid
    learned_result = compute_learned_omega_baseline(
        X_train,
        Y_train,
        X_eval,
        Y_eval,
        omega=omega,
        tau=tau,
        k=128,
    )

    # Compute global covariance
    ridge = 1e-4
    Mu_global = np.mean(Y_train, axis=0)
    Sigma_global = np.cov(Y_train, rowvar=False) + ridge * np.eye(Y_train.shape[1])

    # Learned ellipsoid for this sample
    Sigma_learned = learned_result["Sigma"][sample_idx]

    # Use global mean as the common center for all ellipsoids
    Mu_common = Mu_global + offset

    # Colors: high-contrast, distinct hues
    color_global = "#4361EE"  # Blue
    color_knn = "#2A9D8F"  # Teal
    color_learned = "#F77F00"  # Orange

    # Create side-by-side 3D figure
    fig = plt.figure(figsize=(IEEE_TWO_COL_WIDTH, 3.5))

    def _labels(ax):
        ax.set_xlabel(WIND_LABELS[0], fontsize=7, labelpad=2)
        ax.set_ylabel(WIND_LABELS[1], fontsize=7, labelpad=2)
        ax.set_zlabel(WIND_LABELS[2], fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)

    n_pts = 25

    # --- Left panel: Global vs Learned ---
    ax1 = fig.add_subplot(121, projection="3d")

    X_g, Y_g, Z_g = ellipsoid_surface_3d(
        Mu_common, Sigma_global, rho=rho, n_points=n_pts
    )
    ax1.plot_surface(
        X_g,
        Y_g,
        Z_g,
        color=color_global,
        alpha=0.15,
        edgecolor=color_global,
        linewidth=0.3,
    )
    ax1.plot_wireframe(
        X_g,
        Y_g,
        Z_g,
        color=color_global,
        alpha=0.35,
        linewidth=0.5,
        rstride=5,
        cstride=5,
    )

    X_l, Y_l, Z_l = ellipsoid_surface_3d(
        Mu_common, Sigma_learned, rho=rho, n_points=n_pts
    )
    ax1.plot_surface(
        X_l,
        Y_l,
        Z_l,
        color=color_learned,
        alpha=0.25,
        edgecolor=color_learned,
        linewidth=0.3,
    )
    ax1.plot_wireframe(
        X_l,
        Y_l,
        Z_l,
        color=color_learned,
        alpha=0.5,
        linewidth=0.6,
        rstride=5,
        cstride=5,
    )

    # Common center
    ax1.scatter(
        [Mu_common[0]],
        [Mu_common[1]],
        [Mu_common[2]],
        s=50,
        c="black",
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        zorder=6,
    )

    ax1.set_title("Global vs Learned ω", fontsize=9)
    _labels(ax1)
    ax1.view_init(elev=20, azim=45)
    # Manual legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_global,
            markersize=8,
            label="Global",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=color_learned,
            markersize=8,
            label=f"Learned ω (τ={tau})",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=7)

    # --- Right panel: k-NN vs Learned ---
    ax2 = fig.add_subplot(122, projection="3d")

    Sigma_knn = knn_results[knn_k]["Sigma"][sample_idx]
    X_k, Y_k, Z_k = ellipsoid_surface_3d(Mu_common, Sigma_knn, rho=rho, n_points=n_pts)
    ax2.plot_surface(
        X_k, Y_k, Z_k, color=color_knn, alpha=0.15, edgecolor=color_knn, linewidth=0.3
    )
    ax2.plot_wireframe(
        X_k, Y_k, Z_k, color=color_knn, alpha=0.35, linewidth=0.5, rstride=5, cstride=5
    )

    ax2.plot_surface(
        X_l,
        Y_l,
        Z_l,
        color=color_learned,
        alpha=0.25,
        edgecolor=color_learned,
        linewidth=0.3,
    )
    ax2.plot_wireframe(
        X_l,
        Y_l,
        Z_l,
        color=color_learned,
        alpha=0.5,
        linewidth=0.6,
        rstride=5,
        cstride=5,
    )

    # Common center
    ax2.scatter(
        [Mu_common[0]],
        [Mu_common[1]],
        [Mu_common[2]],
        s=50,
        c="black",
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        zorder=6,
    )

    ax2.set_title(f"k-NN (k={knn_k}) vs Learned ω", fontsize=9)
    _labels(ax2)
    ax2.view_init(elev=20, azim=45)
    legend_elements_r = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_knn,
            markersize=8,
            label=f"k-NN (k={knn_k})",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=color_learned,
            markersize=8,
            label=f"Learned ω (τ={tau})",
        ),
    ]
    ax2.legend(handles=legend_elements_r, loc="upper left", fontsize=7)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 3: NLL vs k Sweep
# ============================================================================
def fig3_nll_vs_k(
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot NLL as function of k for k-NN covariance.

    Uses multi-split stats (mean +/- std, min-max shading) if available,
    falls back to single-split knn_k_sweep_summary.csv otherwise.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig3_nll_vs_k"

    # Try multi-split data first
    multi_stats_path = KNN_SWEEP_DIR / "multi_split_k_stats.csv"
    if multi_stats_path.exists():
        stats = pd.read_csv(multi_stats_path)

        k_values = stats["k"].values
        nll_mean = stats["nll_mean"].values
        nll_std = stats["nll_std"].values
        nll_min = stats["nll_min"].values
        nll_max = stats["nll_max"].values

        fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

        # Mean line with markers
        ax.plot(
            k_values,
            nll_mean,
            "o-",
            linewidth=2,
            markersize=5,
            color=COLORS["knn"],
        )

        # ±1σ shaded band (always visible even when narrow)
        ax.fill_between(
            k_values,
            nll_mean - nll_std,
            nll_mean + nll_std,
            alpha=0.25,
            color=COLORS["knn"],
            label="Mean ± 1σ",
        )

        # Mark best mean
        best_idx = np.argmin(nll_mean)
        ax.scatter(
            [k_values[best_idx]],
            [nll_mean[best_idx]],
            s=150,
            c=COLORS["knn"],
            marker="*",
            zorder=10,
            edgecolors="black",
            linewidths=1,
            label=f"Best: k={k_values[best_idx]}",
        )

        print(f"  Multi-split data: {len(k_values)} k values")
    else:
        # Fallback: single-split sweep
        print(
            f"  Multi-split data not found at {multi_stats_path}, using single-split fallback"
        )
        df = _load_knn_sweep()

        fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

        ax.plot(
            df["k"],
            df["nll"],
            "o-",
            linewidth=2,
            markersize=6,
            color=COLORS["knn"],
            label="k-NN",
        )

        # Mark minimum
        min_idx = df["nll"].idxmin()
        best_k = df.loc[min_idx, "k"]
        best_nll = df.loc[min_idx, "nll"]
        ax.scatter(
            [best_k],
            [best_nll],
            s=150,
            c=COLORS["knn"],
            marker="*",
            zorder=5,
            edgecolors="black",
            linewidths=1,
            label=f"Best: k={best_k}",
        )

    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel("Mean NLL")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 4: NLL vs tau Sweep (multi-seed)
# ============================================================================
TAU_DIAGNOSIS_DIR = VIZ_ARTIFACTS / "tau_omega_diagnosis"


def fig4_nll_vs_tau(
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot learned omega NLL as function of tau, with multi-seed error bars.

    Reads multi_seed_stats.csv from tau_omega_diagnosis directory.
    Falls back to single-seed sweep_results.csv if multi-seed data unavailable.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig4_nll_vs_tau"

    # Try multi-seed data first
    stats_path = TAU_DIAGNOSIS_DIR / "multi_seed_stats.csv"
    if stats_path.exists():
        stats = pd.read_csv(stats_path)

        tau_values = stats["tau"].values
        nll_mean = stats["val_nll_mean"].values
        nll_std = stats["val_nll_std"].values
        nll_min = stats["val_nll_min"].values
        nll_max = stats["val_nll_max"].values

        fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

        # Mean line with markers
        ax.plot(
            tau_values,
            nll_mean,
            "o-",
            linewidth=2,
            markersize=5,
            color=COLORS["learned"],
        )

        # ±1σ shaded band (always visible even when narrow)
        ax.fill_between(
            tau_values,
            nll_mean - nll_std,
            nll_mean + nll_std,
            alpha=0.25,
            color=COLORS["learned"],
            label="Mean ± 1σ",
        )

        # Mark best mean
        best_idx = np.argmin(nll_mean)
        ax.scatter(
            [tau_values[best_idx]],
            [nll_mean[best_idx]],
            s=150,
            c=COLORS["learned"],
            marker="*",
            zorder=10,
            edgecolors="black",
            linewidths=1,
            label=f"Best: τ={tau_values[best_idx]}",
        )

        print(f"  Multi-seed data: {len(tau_values)} tau values")
    else:
        # Fallback: single-seed sweep results
        print(
            f"  Multi-seed data not found at {stats_path}, using single-seed fallback"
        )
        df = _load_sweep_results(HIGH_DIM_16D_DIR)
        best_scaler = df.iloc[0]["scaler_type"]
        df_filtered = df[df["scaler_type"] == best_scaler].copy()
        tau_nll = df_filtered.groupby("tau")["val_nll_learned"].min().reset_index()
        tau_nll = tau_nll.sort_values("tau")

        fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

        ax.plot(
            tau_nll["tau"],
            tau_nll["val_nll_learned"],
            "o-",
            linewidth=2,
            markersize=6,
            color=COLORS["learned"],
            label="Learned ω",
        )

        best_idx = tau_nll["val_nll_learned"].idxmin()
        best_tau = tau_nll.loc[best_idx, "tau"]
        best_nll = tau_nll.loc[best_idx, "val_nll_learned"]
        ax.scatter(
            [best_tau],
            [best_nll],
            s=150,
            c=COLORS["learned"],
            marker="*",
            zorder=5,
            edgecolors="black",
            linewidths=1,
            label=f"Best: τ={best_tau}",
        )

    ax.set_xlabel("τ (Kernel Bandwidth)")
    ax.set_ylabel("Validation NLL")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 5: NLL Comparison Bar Chart (16D)
# ============================================================================
def fig5_nll_16d_comparison(
    feature_set_dir: Path = HIGH_DIM_16D_DIR,
    output_path: Path = None,
) -> plt.Figure:
    """
    Bar chart comparing NLL across methods for 16D feature set.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig5_nll_16d"

    # Load sweep results
    df = _load_sweep_results(feature_set_dir)

    # Get best row (sorted by improvement)
    best = df.iloc[0]

    # Extract NLL values
    nll_global = best["val_nll_global"]
    nll_knn = best["val_nll_euclidean_knn"]
    nll_learned = best["val_nll_learned"]

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.8))  # Slightly taller

    methods = ["Global", "Euclidean\nk-NN", "Learned ω"]
    nlls = [nll_global, nll_knn, nll_learned]
    colors = [COLORS["global"], COLORS["knn"], COLORS["learned"]]

    bars = ax.bar(methods, nlls, color=colors, edgecolor="black", linewidth=1)

    # Add value labels
    for bar, nll in zip(bars, nlls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{nll:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Validation NLL")
    # ax.set_title(...)  # caption in paper
    ax.grid(axis="y", alpha=0.3)

    # Add extra headroom for labels
    ymax = max(nlls) * 1.12
    ax.set_ylim(top=ymax)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 5b: NLL Box and Whisker Plot (16D)
# ============================================================================
def fig5b_nll_boxplot(
    feature_set_dir: Path = HIGH_DIM_16D_DIR,
    output_path: Path = None,
) -> plt.Figure:
    """
    Box and whisker plot comparing per-point NLL distributions across methods.

    Shows the distribution of NLL values, not just the mean.
    """
    from sweep_knn_k_values import load_data
    from covariance_optimization import (
        predict_mu_sigma_topk_cross,
        predict_mu_sigma_knn,
        CovPredictConfig,
    )

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig5b_nll_boxplot"

    # Load data and config
    config = _load_feature_config(feature_set_dir)
    omega = _load_omega(feature_set_dir)
    tau = config.get("tau", 1.0)
    k = config.get("k", 128)
    ridge = config.get("ridge", 1e-3)

    # Load raw data
    from data_processing_extended import FEATURE_BUILDERS

    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )

    build_fn = FEATURE_BUILDERS.get("high_dim_16d")
    if build_fn is None:
        print("  Warning: high_dim_16d feature builder not found")
        return None

    X_raw, Y, times, x_cols, y_cols = build_fn(
        forecasts, actuals, drop_any_nan_rows=True
    )

    # Split data (same as sweep)
    from utils import fit_scaler, apply_scaler

    n = X_raw.shape[0]
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_train = int(0.5 * n)
    n_val = int(0.25 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]

    scaler = fit_scaler(X_raw[train_idx], "standard")
    X = apply_scaler(X_raw, scaler)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    # Compute per-point NLL for each method
    def per_point_nll(Y, Mu, Sigma):
        N, M = Y.shape
        nll = np.empty(N)
        for i in range(N):
            r = (Y[i] - Mu[i]).reshape(M, 1)
            try:
                L = np.linalg.cholesky(Sigma[i])
                logdet = 2.0 * np.log(np.diag(L)).sum()
                x = np.linalg.solve(L, r)
                x = np.linalg.solve(L.T, x)
                quad = float(r.T @ x)
                nll[i] = 0.5 * (logdet + quad + M * np.log(2.0 * np.pi))
            except np.linalg.LinAlgError:
                nll[i] = 100.0
        return nll

    # Global covariance
    Mu_global = np.mean(Y_train, axis=0)
    Sigma_global = np.cov(Y_train, rowvar=False) + ridge * np.eye(Y.shape[1])
    Mu_g = np.tile(Mu_global, (len(Y_val), 1))
    Sigma_g = np.tile(Sigma_global[None, :, :], (len(Y_val), 1, 1))
    nll_global = per_point_nll(Y_val, Mu_g, Sigma_g)

    # k-NN
    Mu_knn, Sigma_knn = predict_mu_sigma_knn(X_val, X_train, Y_train, k=k, ridge=ridge)
    nll_knn = per_point_nll(Y_val, Mu_knn, Sigma_knn)

    # Learned omega
    pred_cfg = CovPredictConfig(tau=tau, ridge=ridge, enforce_nonneg_omega=True)
    Mu_learned, Sigma_learned = predict_mu_sigma_topk_cross(
        X_val, X_train, Y_train, omega=omega, cfg=pred_cfg, k=k
    )
    nll_learned = per_point_nll(Y_val, Mu_learned, Sigma_learned)

    # Create boxplot
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.8))

    data = [nll_global, nll_knn, nll_learned]
    positions = [1, 2, 3]
    labels = ["Global", "k-NN", "Learned ω"]
    colors_box = [COLORS["global"], COLORS["knn"], COLORS["learned"]]

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)

    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-Point NLL")
    # ax.set_title(...)  # caption in paper
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 6: Global Conformal Calibration Curve
# ============================================================================
def fig6_calibration_curve(
    output_path: Path = None,
    alpha_values: list[float] = None,
) -> plt.Figure:
    """
    Global conformal prediction calibration curve (1 bin).

    Shows that empirical coverage matches target coverage for global conformal.
    """
    from conformal_prediction import train_wind_lower_model_conformal_binned
    from data_processing import build_conformal_totals_df
    from viz_conformal_paper import _wilson_score_interval

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig6_calibration"

    if alpha_values is None:
        alpha_values = [0.80, 0.85, 0.90, 0.95, 0.99]

    # Load data
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_tot = build_conformal_totals_df(actuals, forecasts)

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Train global conformal (n_bins=1) for each alpha
    calibration_results = []
    for alpha in alpha_values:
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_tot,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
            n_bins=1,  # Global conformal - single bin
            cal_frac=0.35,
        )
        calibration_results.append(
            {
                "alpha_target": float(alpha),
                "coverage": float(metrics["coverage"]),
                "n_test": int(metrics["n_test"]),
            }
        )
        print(f"    α={alpha:.2f}: coverage={metrics['coverage']:.3f}")

    # Create calibration plot
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH))

    alpha_targets = np.array([r["alpha_target"] for r in calibration_results])
    coverages = np.array([r["coverage"] for r in calibration_results])
    n_tests = np.array([r["n_test"] for r in calibration_results])

    # Compute Wilson score CIs
    ci_lower, ci_upper = [], []
    for cov, n in zip(coverages, n_tests):
        lower, upper = _wilson_score_interval(cov, int(n))
        ci_lower.append(lower)
        ci_upper.append(upper)
    ci_lower, ci_upper = np.array(ci_lower), np.array(ci_upper)

    # Diagonal reference
    diag = np.linspace(0.75, 1.0, 100)
    ax.plot(diag, diag, "k--", linewidth=1.5, label="Perfect Calibration")

    # Tolerance band
    tolerance = 0.05
    ax.fill_between(
        diag,
        diag - tolerance,
        diag + tolerance,
        alpha=0.2,
        color="gray",
        label=f"±{tolerance:.0%} Tolerance",
    )

    # Points with error bars
    colors = [
        "green" if abs(cov - alpha) <= tolerance else "red"
        for cov, alpha in zip(coverages, alpha_targets)
    ]
    for alpha, cov, cl, cu, color in zip(
        alpha_targets, coverages, ci_lower, ci_upper, colors
    ):
        ax.errorbar(
            alpha,
            cov,
            yerr=[[cov - cl], [cu - cov]],
            fmt="o",
            color=color,
            markersize=8,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Target Coverage (α)")
    ax.set_ylabel("Empirical Coverage")
    # ax.set_title(...)  # caption in paper
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.75, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 6b: Global Conformal Calibration (no tolerance band)
# ============================================================================
def fig6b_calibration_no_tolerance(
    output_path: Path = None,
    alpha_values: list[float] = None,
) -> plt.Figure:
    """
    Global conformal calibration curve without the 5% tolerance band.

    Cleaner version showing just the calibration points and perfect diagonal.
    """
    from conformal_prediction import train_wind_lower_model_conformal_binned
    from data_processing import build_conformal_totals_df
    from viz_conformal_paper import _wilson_score_interval

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig6b_calibration_clean"

    if alpha_values is None:
        alpha_values = [0.80, 0.85, 0.90, 0.95, 0.99]

    # Load data
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_tot = build_conformal_totals_df(actuals, forecasts)

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Train global conformal for each alpha
    calibration_results = []
    for alpha in alpha_values:
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_tot,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
            n_bins=1,
            cal_frac=0.35,
        )
        calibration_results.append(
            {
                "alpha_target": float(alpha),
                "coverage": float(metrics["coverage"]),
                "n_test": int(metrics["n_test"]),
            }
        )

    # Create calibration plot
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH))

    alpha_targets = np.array([r["alpha_target"] for r in calibration_results])
    coverages = np.array([r["coverage"] for r in calibration_results])
    n_tests = np.array([r["n_test"] for r in calibration_results])

    # Compute Wilson score CIs
    ci_lower, ci_upper = [], []
    for cov, n in zip(coverages, n_tests):
        lower, upper = _wilson_score_interval(cov, int(n))
        ci_lower.append(lower)
        ci_upper.append(upper)
    ci_lower, ci_upper = np.array(ci_lower), np.array(ci_upper)

    # Diagonal reference
    diag = np.linspace(0.75, 1.0, 100)
    ax.plot(diag, diag, "k--", linewidth=1.5, label="Perfect Calibration")

    # Points with error bars (all same color, no tolerance coloring)
    for alpha, cov, cl, cu in zip(alpha_targets, coverages, ci_lower, ci_upper):
        ax.errorbar(
            alpha,
            cov,
            yerr=[[cov - cl], [cu - cov]],
            fmt="o",
            color=COLORS["learned"],
            markersize=8,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Target Coverage (α)")
    ax.set_ylabel("Empirical Coverage")
    # ax.set_title(...)  # caption in paper
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.75, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 6c: Global Conformal Calibration (no error bars)
# ============================================================================
def fig6c_calibration_points_only(
    output_path: Path = None,
    alpha_values: list[float] = None,
) -> plt.Figure:
    """
    Global conformal calibration curve with points only (no error bars).
    """
    from conformal_prediction import train_wind_lower_model_conformal_binned
    from data_processing import build_conformal_totals_df

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig6c_calibration_points"

    if alpha_values is None:
        alpha_values = [0.80, 0.85, 0.90, 0.95, 0.99]

    # Load data
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_tot = build_conformal_totals_df(actuals, forecasts)

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Train global conformal for each alpha
    calibration_results = []
    for alpha in alpha_values:
        bundle, metrics, _ = train_wind_lower_model_conformal_binned(
            df_tot,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
            n_bins=1,
            cal_frac=0.35,
        )
        calibration_results.append(
            {
                "alpha_target": float(alpha),
                "coverage": float(metrics["coverage"]),
            }
        )

    # Create plot
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH))

    alpha_targets = np.array([r["alpha_target"] for r in calibration_results])
    coverages = np.array([r["coverage"] for r in calibration_results])

    # Diagonal reference
    diag = np.linspace(0.75, 1.0, 100)
    ax.plot(diag, diag, "k--", linewidth=1.5, label="Perfect Calibration")

    # Points only (no error bars)
    ax.scatter(
        alpha_targets,
        coverages,
        s=60,
        c=COLORS["learned"],
        marker="o",
        edgecolors="black",
        linewidths=1,
        zorder=5,
    )

    ax.set_xlabel("Target Coverage (α)")
    ax.set_ylabel("Empirical Coverage")
    # ax.set_title(...)  # caption in paper
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.75, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 7: Global Conformal Correction vs Alpha
# ============================================================================
def fig7_conformal_corrections(
    output_path: Path = None,
    alpha_values: list[float] = None,
) -> plt.Figure:
    """
    Global conformal correction factor (q_hat) vs target alpha.

    Shows how the single correction factor varies with target coverage level.
    """
    from conformal_prediction import train_wind_lower_model_conformal_binned
    from data_processing import build_conformal_totals_df

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig7_corrections"

    if alpha_values is None:
        alpha_values = [0.80, 0.85, 0.90, 0.95, 0.99]

    # Load data
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_tot = build_conformal_totals_df(actuals, forecasts)

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Train global conformal for each alpha
    results = []
    for alpha in alpha_values:
        bundle, metrics, _ = train_wind_lower_model_conformal_binned(
            df_tot,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
            n_bins=1,  # Global conformal
            cal_frac=0.35,
        )
        results.append(
            {
                "alpha": alpha,
                "q_hat": bundle.q_hat_global_r,
                "coverage": metrics["coverage"],
            }
        )
        print(f"    α={alpha:.2f}: q_hat={bundle.q_hat_global_r:.3f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

    alphas = [r["alpha"] for r in results]
    q_hats = [r["q_hat"] for r in results]

    ax.plot(alphas, q_hats, "o-", linewidth=2, markersize=8, color=COLORS["learned"])

    # Add value labels
    for alpha, q_hat in zip(alphas, q_hats):
        ax.annotate(
            f"{q_hat:.2f}",
            (alpha, q_hat),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    ax.set_xlabel("Target Coverage (α)")
    ax.set_ylabel("Correction Factor (q̂)")
    # ax.set_title(...)  # caption in paper
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 7b: Normalized Lower Bound vs Alpha
# ============================================================================
def fig7b_normalized_lower_bound(
    output_path: Path = None,
    alpha_values: list[float] = None,
) -> plt.Figure:
    """
    Number of ensemble standard deviations below the mean forecast
    that the conformalized lower bound sits, vs target alpha.

    Y-axis: mean over test points of (ens_mean - y_pred_conf) / ens_std.
    Increases with alpha (higher coverage = lower bound further below mean).
    """
    from conformal_prediction import train_wind_lower_model_conformal_binned
    from data_processing import build_conformal_totals_df

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig7b_normalized_lower_bound"

    if alpha_values is None:
        alpha_values = [0.80, 0.85, 0.90, 0.95, 0.99]

    # Load data
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_tot = build_conformal_totals_df(actuals, forecasts)

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Replicate the same sort + split as train_wind_lower_model_conformal_binned
    # to extract raw ens_mean/ens_std for the test set
    df_sorted = df_tot.sort_values("TIME_HOURLY").reset_index(drop=True)
    df_sorted = df_sorted[df_sorted["y"].notna()].reset_index(drop=True)
    n_total = len(df_sorted)
    n_test = int(0.25 * n_total)
    test_start = n_total - n_test
    ens_mean_test = df_sorted["ens_mean"].values[test_start:]
    ens_std_test = df_sorted["ens_std"].values[test_start:]

    results = []
    for alpha in alpha_values:
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_tot,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
            n_bins=1,
            cal_frac=0.35,
        )
        # How many std devs below the mean forecast is the final lower bound?
        #   n_std_below = (ens_mean - y_pred_conf) / ens_std
        y_pred_conf = df_test["y_pred_conf"].values

        # Avoid division by zero
        valid = ens_std_test > 1e-6
        n_std_below = (ens_mean_test[valid] - y_pred_conf[valid]) / ens_std_test[valid]
        mean_n_std = float(np.mean(n_std_below))

        results.append(
            {
                "alpha": alpha,
                "mean_n_std_below": mean_n_std,
                "q_hat": bundle.q_hat_global_r,
            }
        )
        print(f"    α={alpha:.2f}: {mean_n_std:.3f} std below mean")

    # Create plot
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

    alphas = [r["alpha"] for r in results]
    n_stds = [r["mean_n_std_below"] for r in results]

    ax.plot(alphas, n_stds, "o-", linewidth=2, markersize=8, color=COLORS["learned"])

    # Add value labels
    for a, ns in zip(alphas, n_stds):
        ax.annotate(
            f"{ns:.2f}",
            (a, ns),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    ax.set_xlabel("Target Coverage (α)")
    ax.set_ylabel("Std. Deviations Below Mean")
    # ax.set_title(...)  # caption in paper
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 7c: Stacked Decomposition of Lower Bound
# ============================================================================
def fig7c_lower_bound_decomposition(
    output_path: Path = None,
    alpha_values: list[float] = None,
) -> plt.Figure:
    """
    Stacked bar decomposition of the conformalized lower bound.

    Total std devs below mean = base quantile component + conformal correction.
      base component:  mean of (ens_mean - y_pred_base) / ens_std
      conformal corr:  q_hat  (added uniformly to all points)
    """
    from conformal_prediction import train_wind_lower_model_conformal_binned
    from data_processing import build_conformal_totals_df

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig7c_lb_decomposition"

    if alpha_values is None:
        alpha_values = [0.80, 0.85, 0.90, 0.95, 0.99]

    # Load data
    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_tot = build_conformal_totals_df(actuals, forecasts)

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Extract raw test-set ens_mean / ens_std (same split as conformal training)
    df_sorted = df_tot.sort_values("TIME_HOURLY").reset_index(drop=True)
    df_sorted = df_sorted[df_sorted["y"].notna()].reset_index(drop=True)
    n_total = len(df_sorted)
    n_test = int(0.25 * n_total)
    test_start = n_total - n_test
    ens_mean_test = df_sorted["ens_mean"].values[test_start:]
    ens_std_test = df_sorted["ens_std"].values[test_start:]
    valid = ens_std_test > 1e-6

    results = []
    for alpha in alpha_values:
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_tot,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
            n_bins=1,
            cal_frac=0.35,
        )
        q_hat = bundle.q_hat_global_r
        y_pred_base = df_test["y_pred_base"].values

        # Base quantile: how far below the mean the quantile prediction already is
        base_component = float(
            np.mean((ens_mean_test[valid] - y_pred_base[valid]) / ens_std_test[valid])
        )
        # Conformal correction adds q_hat std devs on top
        results.append(
            {
                "alpha": alpha,
                "base": base_component,
                "conformal": q_hat,
                "total": base_component + q_hat,
            }
        )
        print(
            f"    α={alpha:.2f}: base={base_component:.3f}, conformal={q_hat:.3f}, total={base_component + q_hat:.3f}"
        )

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.8))

    alphas = [r["alpha"] for r in results]
    bases = [r["base"] for r in results]
    conforals = [r["conformal"] for r in results]
    totals = [r["total"] for r in results]

    x = np.arange(len(alphas))
    width = 0.5

    bars_base = ax.bar(
        x,
        bases,
        width,
        label="Base Quantile",
        color=COLORS["knn"],
        edgecolor="black",
        linewidth=0.5,
    )
    bars_conf = ax.bar(
        x,
        conforals,
        width,
        bottom=bases,
        label="Conformal Correction",
        color=COLORS["learned"],
        edgecolor="black",
        linewidth=0.5,
    )

    # Total labels on top
    for i, tot in enumerate(totals):
        ax.text(x[i], tot + 0.03, f"{tot:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("Target Coverage (α)")
    ax.set_ylabel("Std. Deviations Below Mean")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 8: 2D Ellipse Grid (different k values)
# ============================================================================
def fig8_ellipse_grid(
    output_path: Path = None,
    k_values: list[int] = None,
    sample_indices: list[int] = None,
    rho: float = 2.0,
    offset: float = 50.0,
) -> plt.Figure:
    """
    2D ellipse comparison at k=16, 512, and Global.

    Creates a grid: rows = samples, columns = k values.
    An offset is added to centers so axes stay non-negative.
    """
    from sweep_knn_k_values import ellipse_points_2d, load_data, sweep_k_values

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig8_ellipse_grid"

    # Load data
    X_train, Y_train, X_eval, Y_eval, x_cols, y_cols, _, _ = load_data(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet",
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet",
    )

    N_train = X_train.shape[0]

    if k_values is None:
        k_values = [16, 512, N_train]

    if sample_indices is None:
        n_eval = Y_eval.shape[0]
        sample_indices = [0, n_eval // 4, n_eval // 2]

    # Compute ellipsoids
    results = sweep_k_values(X_train, Y_train, X_eval, Y_eval, k_values)

    # Build display labels for k values
    k_labels = [f"Global (k={k})" if k == N_train else f"k={k}" for k in k_values]

    # Create grid figure (tight: reduced per-panel size)
    n_samples = len(sample_indices)
    n_k = len(k_values)
    fig, axes = plt.subplots(
        n_samples, n_k,
        figsize=(2.8 * n_k, 2.8 * n_samples),
    )

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    colors = ["#2A9D8F", "#E9C46A", "#E76F51"]
    dims = (1, 2)  # Wind 2 and Wind 3 (more similar scale)

    # Use global mean as common center (from largest k)
    max_k = max(k_values)
    offset_vec = np.array([offset, offset])

    # Pre-compute all ellipses to find consistent axis limits per row
    ellipses_data = {}
    for row, sample_idx in enumerate(sample_indices):
        mu_common = results[max_k]["Mu"][sample_idx][[dims[0], dims[1]]] + offset_vec

        row_xmin, row_xmax, row_ymin, row_ymax = np.inf, -np.inf, np.inf, -np.inf
        for col, k in enumerate(k_values):
            Sigma = results[k]["Sigma"][sample_idx]
            Sigma2 = Sigma[np.ix_([dims[0], dims[1]], [dims[0], dims[1]])]
            ex, ey = ellipse_points_2d(mu_common, Sigma2, rho=rho)
            ellipses_data[(row, col)] = (mu_common, ex, ey)
            row_xmin = min(row_xmin, ex.min())
            row_xmax = max(row_xmax, ex.max())
            row_ymin = min(row_ymin, ey.min())
            row_ymax = max(row_ymax, ey.max())
        pad_x = (row_xmax - row_xmin) * 0.1
        pad_y = (row_ymax - row_ymin) * 0.1
        ellipses_data[("lim", row)] = (
            row_xmin - pad_x, row_xmax + pad_x,
            row_ymin - pad_y, row_ymax + pad_y,
        )

    for row, sample_idx in enumerate(sample_indices):
        xlim_lo, xlim_hi, ylim_lo, ylim_hi = ellipses_data[("lim", row)]

        for col, k in enumerate(k_values):
            ax = axes[row, col]
            mu2, ex, ey = ellipses_data[(row, col)]

            ax.plot(ex, ey, "-", linewidth=2, color=colors[col])
            ax.fill(ex, ey, alpha=0.2, color=colors[col])
            ax.scatter([mu2[0]], [mu2[1]], s=50, c="black", marker="o", zorder=5)

            ax.set_xlabel(WIND_LABELS[dims[0]] + " (MW)", fontsize=7)
            ax.set_ylabel(WIND_LABELS[dims[1]] + " (MW)", fontsize=7)
            ax.set_title(k_labels[col], fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlim_lo, xlim_hi)
            ax.set_ylim(ylim_lo, ylim_hi)
            ax.set_aspect("equal", adjustable="box")

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# FIGURE 9: Omega Bar Chart (Feature Weights)
# ============================================================================
def fig9_omega_bar_chart(
    feature_set_dir: Path = HIGH_DIM_16D_DIR,
    output_path: Path = None,
) -> plt.Figure:
    """
    Generate omega bar chart showing learned feature weights.

    Uses saved data from feature set experiments.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig9_omega_bar_chart"

    # Check for existing figure
    existing_png = feature_set_dir / "omega_bar_chart.png"
    if existing_png.exists():
        import shutil

        shutil.copy(existing_png, output_path.parent / (output_path.stem + ".png"))
        print(f"  Copied existing omega bar chart from {existing_png}")
        return None

    # Generate from data
    omega = _load_omega(feature_set_dir)
    config = _load_feature_config(feature_set_dir)
    x_cols = config.get("x_cols", [f"Feature {i}" for i in range(len(omega))])

    from viz_projections import plot_omega_bar_chart

    plot_omega_bar_chart(
        omega,
        x_cols,
        out_path=output_path.with_suffix(".png"),
        title="Learned Feature Weights (16D)",
        highlight_top_k=3,
    )
    print(f"  Saved omega bar chart: {output_path.with_suffix('.png')}")

    return None


# ============================================================================
# FIGURE 10: 2D Ellipse Overlay with Learned Omega
# ============================================================================
def fig10_ellipse_overlay(
    output_path: Path = None,
    sample_idx: int = 0,
    rho: float = 2.0,
) -> plt.Figure:
    """
    Single sample ellipse overlay: k=32, k=512, and Global, all on one plot.
    """
    from sweep_knn_k_values import ellipse_points_2d, load_data, sweep_k_values

    if output_path is None:
        output_path = OUTPUT_DIR / "figures" / "fig10_ellipse_overlay"

    # Load data
    X_train, Y_train, X_eval, Y_eval, x_cols, y_cols, _, _ = load_data(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet",
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet",
    )

    N_train = X_train.shape[0]
    k_values = [32, 512, N_train]  # last = global
    results = sweep_k_values(X_train, Y_train, X_eval, Y_eval, k_values)

    dims = (1, 2)  # Wind 2 and Wind 3 (more similar scale)
    colors = ["#E76F51", "#2A9D8F", COLORS["global"]]  # Coral, Teal, Gray
    labels = ["k=32", "k=512", f"Global (k={N_train})"]

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH))

    # Use global mean (from k=N_train) as common center
    Mu_global = results[N_train]["Mu"][sample_idx]
    mu_common = Mu_global[[dims[0], dims[1]]]

    for i, k in enumerate(k_values):
        Sigma = results[k]["Sigma"][sample_idx]
        Sigma2 = Sigma[np.ix_([dims[0], dims[1]], [dims[0], dims[1]])]
        ex, ey = ellipse_points_2d(mu_common, Sigma2, rho=rho)
        ax.plot(ex, ey, "-", linewidth=2, color=colors[i], label=labels[i])
        ax.fill(ex, ey, alpha=0.1, color=colors[i])

    # Single common center
    ax.scatter(
        [mu_common[0]],
        [mu_common[1]],
        s=40,
        c="black",
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    ax.set_xlabel(WIND_LABELS[dims[0]])
    ax.set_ylabel(WIND_LABELS[dims[1]])
    # ax.set_title(...)  # caption in paper
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================
def table_nll_vs_tau(
    feature_set_dir: Path = HIGH_DIM_16D_DIR,
    output_path: Path = None,
) -> str:
    """
    Generate LaTeX table: NLL at different tau values.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "tables" / "tab_nll_vs_tau.tex"

    df = _load_sweep_results(feature_set_dir)

    # Get unique tau values
    tau_values = sorted(df["tau"].unique())

    # Build table rows
    rows = []
    for tau in tau_values:
        df_tau = df[df["tau"] == tau]
        best = df_tau.iloc[0]
        rows.append(
            {
                "tau": tau,
                "learned": best["val_nll_learned"],
                "kernel_equal": best["val_nll_kernel_equal"],
                "euclidean": best["val_nll_euclidean_knn"],
                "global": best["val_nll_global"],
            }
        )

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\caption{Validation NLL vs Kernel Bandwidth $\tau$}
\label{tab:nll_vs_tau}
\centering
\begin{tabular}{lcccc}
\toprule
$\tau$ & Learned $\omega$ & Kernel (ω=1) & k-NN & Global \\
\midrule
"""

    for row in rows:
        latex += f"{row['tau']:.2f} & {row['learned']:.3f} & {row['kernel_equal']:.3f} & {row['euclidean']:.3f} & {row['global']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved LaTeX table: {output_path}")

    return latex


def table_nll_vs_k(
    output_path: Path = None,
) -> str:
    """
    Generate LaTeX table: k-NN NLL at different k values.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "tables" / "tab_nll_vs_k.tex"

    df = _load_knn_sweep()

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\caption{$k$-NN Covariance: NLL vs Number of Neighbors $k$}
\label{tab:nll_vs_k}
\centering
\begin{tabular}{lc}
\toprule
$k$ & NLL \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{int(row['k'])} & {row['nll']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved LaTeX table: {output_path}")

    return latex


def table_omega_weights(
    feature_set_dirs: dict[str, Path] = None,
    output_path: Path = None,
) -> str:
    """
    Generate LaTeX table: Learned omega weights per feature set.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "tables" / "tab_omega_weights.tex"

    if feature_set_dirs is None:
        feature_set_dirs = {
            "focused_2d": FOCUSED_2D_DIR,
            "high_dim_16d": HIGH_DIM_16D_DIR,
        }

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\caption{Learned Feature Weights $\omega$}
\label{tab:omega_weights}
\centering
"""

    for name, dir_path in feature_set_dirs.items():
        if not dir_path.exists():
            continue

        omega = _load_omega(dir_path)
        config = _load_feature_config(dir_path)
        x_cols = config.get("x_cols", [f"Feature {i}" for i in range(len(omega))])

        latex += f"\\textbf{{{name.replace('_', ' ').title()}}}\n\n"
        latex += r"""\begin{tabular}{lc}
\toprule
Feature & Weight \\
\midrule
"""

        # Sort by weight (descending)
        sorted_idx = np.argsort(omega)[::-1]
        for i in sorted_idx:
            col_name = x_cols[i].replace("_", r"\_")
            latex += f"{col_name} & {omega[i]:.4f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}

"""

    latex += r"\end{table}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved LaTeX table: {output_path}")

    return latex


# ============================================================================
# UTILITIES
# ============================================================================
def _save_figure(
    fig: plt.Figure, output_path: Path, dpi: int = FIGURE_DEFAULTS["dpi_pdf"]
):
    """Save figure in both PDF and PNG formats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PDF for paper
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", format="pdf")
    print(f"  Saved PDF: {pdf_path}")

    # PNG for preview
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", format="png")
    print(f"  Saved PNG: {png_path}")

    plt.close(fig)


def save_metadata(figures_generated: list[str], tables_generated: list[str]):
    """Save generation metadata."""
    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "figures": figures_generated,
        "tables": tables_generated,
        "data_sources": {
            "focused_2d": str(FOCUSED_2D_DIR),
            "high_dim_16d": str(HIGH_DIM_16D_DIR),
            "knn_k_sweep": str(KNN_SWEEP_DIR),
            "paper_figures": str(PAPER_FIGURES_DIR),
        },
    }

    output_path = OUTPUT_DIR / "figure_metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {output_path}")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
def generate_all_figures():
    """Generate all paper figures and tables."""
    print("=" * 80)
    print("PAPER FIGURE GENERATION")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    _ensure_output_dirs()

    figures_generated = []
    tables_generated = []

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------
    print("GENERATING FIGURES")
    print("-" * 40)

    # Figure 1: Kernel distance comparison
    print("\n[1/15] Kernel distance comparison (Learned vs k-NN)...")
    try:
        fig1_kernel_distance_comparison(k=64, tau=1.0)
        figures_generated.append("fig1_kernel_distance")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 2: 3D ellipsoid (Learned, k-NN k=64/512, Global)
    print("\n[2/15] 3D ellipsoid comparison (Learned, k-NN, Global)...")
    try:
        fig2_3d_ellipsoid_comparison(knn_k=16, tau=1.0)
        figures_generated.append("fig2_ellipsoid_3d")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 3: NLL vs k
    print("\n[3/15] NLL vs k sweep...")
    try:
        fig3_nll_vs_k()
        figures_generated.append("fig3_nll_vs_k")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 4: NLL vs tau
    print("\n[4/15] NLL vs tau sweep...")
    try:
        fig4_nll_vs_tau()
        figures_generated.append("fig4_nll_vs_tau")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 5: 16D comparison bar chart
    # print("\n[5/15] 16D NLL comparison (bar chart)...")  # SKIPPED
    # fig5_nll_16d_comparison()

    # print("\n[5b/15] 16D NLL comparison (box plot)...")  # SKIPPED
    # fig5b_nll_boxplot()

    # print("\n[6/15] Global conformal calibration curve...")  # SKIPPED (use 6b or 6c)
    # fig6_calibration_curve()

    # Figure 6b: Global conformal calibration (no tolerance band)
    print("\n[6b/15] Global conformal calibration (no tolerance)...")
    try:
        fig6b_calibration_no_tolerance()
        figures_generated.append("fig6b_calibration_clean")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 6c: Global conformal calibration (points only, no error bars)
    print("\n[6c/15] Global conformal calibration (points only)...")
    try:
        fig6c_calibration_points_only()
        figures_generated.append("fig6c_calibration_points")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 7: Global conformal correction vs alpha
    print("\n[7/15] Global conformal correction vs alpha...")
    try:
        fig7_conformal_corrections()
        figures_generated.append("fig7_corrections")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 7b: Normalized lower bound vs alpha
    print("\n[7b/15] Normalized lower bound vs alpha...")
    try:
        fig7b_normalized_lower_bound()
        figures_generated.append("fig7b_normalized_lower_bound")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 7c: Lower bound decomposition (stacked bar)
    print("\n[7c/15] Lower bound decomposition (stacked bar)...")
    try:
        fig7c_lower_bound_decomposition()
        figures_generated.append("fig7c_lb_decomposition")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 8: Ellipse grid
    print("\n[8/15] 2D ellipse grid (k=16, 512, Global)...")
    try:
        fig8_ellipse_grid()
        figures_generated.append("fig8_ellipse_grid")
    except Exception as e:
        print(f"  Error: {e}")

    # Figure 9: Omega bar chart
    # print("\n[9/15] Omega bar chart (16D)...")  # SKIPPED
    # fig9_omega_bar_chart()

    # Figure 10: Ellipse overlay
    print("\n[10/15] 2D ellipse overlay...")
    try:
        fig10_ellipse_overlay()
        figures_generated.append("fig10_ellipse_overlay")
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # TABLES
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("GENERATING LATEX TABLES")
    print("-" * 40)

    # Table 1: NLL vs tau
    print("\n[1/3] NLL vs tau table...")
    try:
        table_nll_vs_tau()
        tables_generated.append("tab_nll_vs_tau")
    except Exception as e:
        print(f"  Error: {e}")

    # Table 2: NLL vs k
    print("\n[2/3] NLL vs k table...")
    try:
        table_nll_vs_k()
        tables_generated.append("tab_nll_vs_k")
    except Exception as e:
        print(f"  Error: {e}")

    # Table 3: Omega weights
    print("\n[3/3] Omega weights table...")
    try:
        table_omega_weights()
        tables_generated.append("tab_omega_weights")
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    save_metadata(figures_generated, tables_generated)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures generated: {len(figures_generated)}")
    for fig in figures_generated:
        print(f"  - {fig}")
    print(f"\nTables generated: {len(tables_generated)}")
    for tab in tables_generated:
        print(f"  - {tab}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("  figures/")
    print("  tables/")
    print("  figure_metadata.json")


if __name__ == "__main__":
    generate_all_figures()
