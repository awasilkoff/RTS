"""
Sweep over k values for k-NN covariance estimation baseline.

Visualizes how ellipsoid size/shape changes with k and computes NLL at each k.

Key insight: As k increases:
- Small k: Very local, potentially noisy (high variance)
- Large k: More global, potentially biased (high bias)
- k=N: Global covariance (no localization)

Usage:
    python sweep_knn_k_values.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from plot_config import setup_plotting
from utils import fit_standard_scaler

# Apply global plotting config
setup_plotting()
from data_processing import build_XY_for_covariance_system_only
from data_processing_extended import FEATURE_BUILDERS
from covariance_optimization import (
    predict_mu_sigma_knn,
    predict_mu_sigma_topk_cross,
    CovPredictConfig,
)


def _mean_gaussian_nll(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Compute mean NLL of Y under N(Mu, Sigma)."""
    N, M = Y.shape
    nll_total = 0.0

    for i in range(N):
        S = Sigma[i]
        r = (Y[i] - Mu[i]).reshape(M, 1)
        try:
            L = np.linalg.cholesky(S)
            logdet = 2.0 * np.log(np.diag(L)).sum()
            x = np.linalg.solve(L, r)
            x = np.linalg.solve(L.T, x)
            quad = float(r.T @ x)
            nll_total += 0.5 * (logdet + quad + M * np.log(2.0 * np.pi))
        except np.linalg.LinAlgError:
            # Non-PD matrix, use large penalty
            nll_total += 1e6

    return nll_total / N


def ellipse_points_2d(
    mu: np.ndarray, Sigma: np.ndarray, rho: float = 1.0, n_points: int = 100
):
    """
    Compute points on 2D ellipse boundary.

    Ellipse: {x : (x-mu)^T Sigma^{-1} (x-mu) <= rho^2}
    """
    # Eigendecomposition
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-10, None)

    # Parametric ellipse
    theta = np.linspace(0, 2 * np.pi, n_points)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])  # (2, n_points)

    # Scale by sqrt(eigenvalues) * rho
    scaled = np.diag(np.sqrt(w) * rho) @ unit_circle  # (2, n_points)

    # Rotate by eigenvectors and translate
    ellipse = V @ scaled + mu[:, np.newaxis]  # (2, n_points)

    return ellipse[0, :], ellipse[1, :]


def ellipsoid_surface_3d(
    mu: np.ndarray, Sigma: np.ndarray, rho: float = 1.0, n_points: int = 30
):
    """
    Compute points on 3D ellipsoid surface.

    Ellipsoid: {x : (x-mu)^T Sigma^{-1} (x-mu) <= rho^2}

    Returns:
        X, Y, Z: meshgrid arrays for plotting with plot_surface
    """
    # Eigendecomposition
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-10, None)

    # Parametric sphere
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)

    # Unit sphere
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    # Stack into (3, n_points, n_points)
    sphere = np.array([x_sphere, y_sphere, z_sphere])

    # Scale by sqrt(eigenvalues) * rho and rotate
    # Reshape for matrix multiplication
    sphere_flat = sphere.reshape(3, -1)  # (3, n_points^2)
    scaled = np.diag(np.sqrt(w) * rho) @ sphere_flat
    rotated = V @ scaled
    translated = rotated + mu[:, np.newaxis]

    # Reshape back
    X = translated[0].reshape(n_points, n_points)
    Y = translated[1].reshape(n_points, n_points)
    Z = translated[2].reshape(n_points, n_points)

    return X, Y, Z


def plot_3d_ellipsoids_overlay(
    results: dict,
    Y_eval: np.ndarray,
    sample_idx: int,
    k_values_to_plot: list[int],
    output_dir: Path,
    y_cols: list[str] = None,
    rho: float = 2.0,
    learned_omega_result: dict = None,
):
    """
    Overlay 3D ellipsoids from different k values on single plot.

    Shows how ellipsoid shape changes with k for the 3 wind farms.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    y_actual = Y_eval[sample_idx]

    # Color scale for k values
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(k_values_to_plot)))

    # Plot ellipsoids for each k (as wireframes for visibility)
    for idx, k in enumerate(k_values_to_plot):
        Mu = results[k]["Mu"][sample_idx]
        Sigma = results[k]["Sigma"][sample_idx]

        # Get ellipsoid surface
        X, Y, Z = ellipsoid_surface_3d(Mu, Sigma, rho=rho, n_points=20)

        # Plot as wireframe for better visibility
        ax.plot_wireframe(
            X,
            Y,
            Z,
            color=colors[idx],
            alpha=0.4,
            linewidth=0.8,
            label=f"k={k}",
        )

        # Plot center
        ax.scatter(
            [Mu[0]],
            [Mu[1]],
            [Mu[2]],
            s=80,
            c=[colors[idx]],
            marker="o",
            edgecolors="black",
            linewidths=1,
            zorder=5,
        )

    # Plot learned omega ellipsoid if provided
    if learned_omega_result is not None:
        Mu_omega = learned_omega_result["Mu"][sample_idx]
        Sigma_omega = learned_omega_result["Sigma"][sample_idx]

        X_omega, Y_omega, Z_omega = ellipsoid_surface_3d(
            Mu_omega, Sigma_omega, rho=rho, n_points=20
        )

        ax.plot_wireframe(
            X_omega,
            Y_omega,
            Z_omega,
            color="#E63946",
            alpha=0.6,
            linewidth=1.5,
            label=f"Learned ω (k={learned_omega_result.get('k', 64)})",
        )

        ax.scatter(
            [Mu_omega[0]],
            [Mu_omega[1]],
            [Mu_omega[2]],
            s=120,
            c="#E63946",
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=6,
        )

    # Plot actual observation
    ax.scatter(
        [y_actual[0]],
        [y_actual[1]],
        [y_actual[2]],
        s=250,
        c="lime",
        marker="*",
        zorder=10,
        edgecolors="black",
        linewidths=2,
        label="Actual",
    )

    # Labels
    if y_cols is not None and len(y_cols) >= 3:
        ax.set_xlabel(y_cols[0])
        ax.set_ylabel(y_cols[1])
        ax.set_zlabel(y_cols[2])
    else:
        ax.set_xlabel("Wind Farm 1 (MW)")
        ax.set_ylabel("Wind Farm 2 (MW)")
        ax.set_zlabel("Wind Farm 3 (MW)")

    ax.set_title(
        f"3D Uncertainty Ellipsoids: Sample {sample_idx} (ρ={rho})\n"
        + "How k affects uncertainty set shape",
        fontsize=13,
        fontweight="bold",
    )

    ax.legend(loc="upper left")

    # Improve viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    output_path = output_dir / f"knn_3d_ellipsoid_overlay_sample{sample_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved 3D ellipsoid overlay to {output_path}")

    output_path_pdf = output_dir / f"knn_3d_ellipsoid_overlay_sample{sample_idx}.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


def load_data(
    forecasts_parquet: Path,
    actuals_parquet: Path,
    train_frac: float = 0.75,
    standardize: bool = True,
    random_seed: int = 42,
    feature_set: str = "high_dim_16d",
    use_residuals: bool = False,
    actual_col: str = "ACTUAL",
):
    """Load and prepare data for k-NN sweep.

    Parameters
    ----------
    use_residuals : bool
        If True, Y becomes (actual - ensemble_mean_forecast) per resource
        instead of raw actuals. This models the forecast error distribution.
    actual_col : str
        Column name to use as Y target. Pass "RESIDUAL" to use pre-computed
        residuals from the actuals parquet.
    """
    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)

    build_fn = FEATURE_BUILDERS.get(feature_set)
    if build_fn is not None:
        X, Y, times, x_cols, y_cols = build_fn(
            forecasts, actuals, drop_any_nan_rows=True, actual_col=actual_col
        )
    else:
        # Fallback to system-only 2D
        X, Y, times, x_cols, y_cols = build_XY_for_covariance_system_only(
            forecasts, actuals, drop_any_nan_rows=True, actual_col=actual_col
        )

    if use_residuals:
        # Compute per-resource ensemble mean forecast, aligned to Y's time index
        fmean = (
            forecasts.groupby(["TIME_HOURLY", "ID_RESOURCE"])["FORECAST"]
            .mean()
            .reset_index()
            .pivot(index="TIME_HOURLY", columns="ID_RESOURCE", values="FORECAST")
            .sort_index()
        )
        # Align columns to y_cols order, reindex to times
        fmean = fmean[y_cols].reindex(times).values
        Y = Y - fmean

    if standardize:
        scaler = fit_standard_scaler(X)
        Xp = scaler.transform(X)
    else:
        Xp = X.copy()

    # Random split (consistent with conformal experiments)
    N = Xp.shape[0]
    rng = np.random.RandomState(random_seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)

    train_idx = idx[:n_train]
    eval_idx = idx[n_train:]

    X_train = Xp[train_idx]
    Y_train = Y[train_idx]
    X_eval = Xp[eval_idx]
    Y_eval = Y[eval_idx]

    return X_train, Y_train, X_eval, Y_eval, x_cols, y_cols, train_idx, eval_idx


def compute_learned_omega_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    omega: np.ndarray,
    tau: float = 5.0,
    k: int = 64,
    ridge: float = 1e-4,
    zero_mean: bool = False,
):
    """
    Compute predictions using learned omega (kernel-weighted local covariance).

    Returns dict with nll, Mu, Sigma.
    """
    cfg = CovPredictConfig(tau=tau, ridge=ridge, zero_mean=zero_mean)

    Mu_eval, Sigma_eval = predict_mu_sigma_topk_cross(
        X_eval, X_train, Y_train, omega=omega, k=k, cfg=cfg
    )

    nll = _mean_gaussian_nll(Y_eval, Mu_eval, Sigma_eval)

    return {
        "nll": nll,
        "Mu": Mu_eval,
        "Sigma": Sigma_eval,
        "omega": omega,
        "tau": tau,
        "k": k,
    }


def sweep_k_values(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    k_values: list[int],
    ridge: float = 1e-4,
):
    """
    Sweep over k values and compute NLL for each.

    Returns dict with k -> (nll, Mu_eval, Sigma_eval)
    """
    results = {}

    for k in k_values:
        print(f"  k={k}...", end=" ")

        Mu_eval, Sigma_eval = predict_mu_sigma_knn(
            X_eval, X_train, Y_train, k=k, ridge=ridge
        )

        nll = _mean_gaussian_nll(Y_eval, Mu_eval, Sigma_eval)
        print(f"NLL = {nll:.4f}")

        results[k] = {
            "nll": nll,
            "Mu": Mu_eval,
            "Sigma": Sigma_eval,
        }

    return results


def sweep_tau_values(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    omega: np.ndarray,
    tau_values: list[float],
    k: int = 64,
    ridge: float = 1e-4,
):
    """
    Sweep over tau values for learned omega and compute NLL for each.

    Returns dict with tau -> (nll, Mu_eval, Sigma_eval)
    """
    results = {}

    for tau in tau_values:
        print(f"  tau={tau}...", end=" ")

        result = compute_learned_omega_baseline(
            X_train,
            Y_train,
            X_eval,
            Y_eval,
            omega=omega,
            tau=tau,
            k=k,
            ridge=ridge,
        )

        print(f"NLL = {result['nll']:.4f}")
        results[tau] = result

    return results


def plot_nll_vs_tau(
    tau_results: dict,
    output_dir: Path,
    best_knn_nll: float = None,
):
    """Plot learned omega NLL as function of tau, with optional k-NN comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tau_values = sorted(tau_results.keys())
    nlls = [tau_results[tau]["nll"] for tau in tau_values]

    ax.plot(
        tau_values,
        nlls,
        "o-",
        linewidth=2,
        markersize=8,
        color="#E63946",
        label="Learned ω",
    )

    # Mark minimum
    min_idx = np.argmin(nlls)
    min_tau = tau_values[min_idx]
    min_nll = nlls[min_idx]
    ax.scatter(
        [min_tau],
        [min_nll],
        s=200,
        c="#E63946",
        marker="*",
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
        label=f"Best τ={min_tau}",
    )

    # Add k-NN baseline if provided
    if best_knn_nll is not None:
        ax.axhline(
            best_knn_nll,
            color="#2A9D8F",
            linestyle="--",
            linewidth=2.5,
            label=f"Best k-NN: NLL={best_knn_nll:.3f}",
        )

    ax.set_xlabel("τ (Kernel Bandwidth)")
    ax.set_ylabel("Mean Negative Log-Likelihood")
    ax.set_title("Learned Omega NLL vs Bandwidth τ")
    ax.set_xscale("log")
    ax.legend(loc="upper right")

    plt.tight_layout()

    output_path = output_dir / "learned_omega_nll_vs_tau.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved tau sweep plot to {output_path}")

    output_path_pdf = output_dir / "learned_omega_nll_vs_tau.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()

    return min_tau, min_nll


def plot_nll_vs_k(results: dict, output_dir: Path, learned_omega_result: dict = None):
    """Plot NLL as function of k, with optional learned omega comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(results.keys())
    nlls = [results[k]["nll"] for k in k_values]

    ax.plot(
        k_values,
        nlls,
        "o-",
        linewidth=2,
        markersize=8,
        color="#2A9D8F",
        label="k-NN (uniform weights)",
    )

    # Mark minimum k-NN
    min_idx = np.argmin(nlls)
    min_k = k_values[min_idx]
    min_nll = nlls[min_idx]
    ax.scatter(
        [min_k],
        [min_nll],
        s=200,
        c="#2A9D8F",
        marker="*",
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
        label=f"Best k-NN: k={min_k}",
    )

    # Add learned omega baseline if provided
    if learned_omega_result is not None:
        omega_nll = learned_omega_result["nll"]
        omega_k = learned_omega_result.get("k", 64)

        ax.axhline(
            omega_nll,
            color="#E63946",
            linestyle="--",
            linewidth=2.5,
            label=f"Learned ω (k={omega_k}): NLL={omega_nll:.3f}",
        )

        # Mark improvement
        improvement = min_nll - omega_nll
        if improvement > 0:
            ax.annotate(
                f"Learned ω is\n{improvement:.3f} better",
                xy=(k_values[len(k_values) // 2], omega_nll),
                xytext=(k_values[len(k_values) // 2], omega_nll - 0.3),
                fontsize=10,
                ha="center",
                color="#E63946",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E63946"),
            )

    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel("Mean Negative Log-Likelihood")
    ax.set_title("k-NN Covariance vs Learned Omega")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    # Add annotations for bias-variance
    # ax.annotate(
    #     f"Small k:\nHigh variance",
    #     xy=(k_values[0], nlls[0]),
    #     xytext=(k_values[0] * 2, nlls[0] + 0.3),
    #     fontsize=9,
    #     ha="left",
    #     arrowprops=dict(arrowstyle="->", color="gray"),
    # )
    #
    # if k_values[-1] > 500:
    #     ax.annotate(
    #         f"Large k:\nHigh bias",
    #         xy=(k_values[-1], nlls[-1]),
    #         xytext=(k_values[-1] / 3, nlls[-1] + 0.2),
    #         fontsize=9,
    #         ha="right",
    #         arrowprops=dict(arrowstyle="->", color="gray"),
    #     )

    plt.tight_layout()

    output_path = output_dir / "knn_nll_vs_k.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved NLL plot to {output_path}")

    output_path_pdf = output_dir / "knn_nll_vs_k.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


def plot_ellipses_at_various_k(
    results: dict,
    Y_eval: np.ndarray,
    sample_indices: list[int],
    k_values_to_plot: list[int],
    output_dir: Path,
    dims: tuple[int, int] = (0, 1),
    rho: float = 2.0,
):
    """
    Plot 2D ellipse comparison for selected samples at various k values.

    Creates a grid: rows = samples, columns = k values
    """
    n_samples = len(sample_indices)
    n_k = len(k_values_to_plot)

    fig, axes = plt.subplots(n_samples, n_k, figsize=(4 * n_k, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_k == 1:
        axes = axes.reshape(-1, 1)

    d0, d1 = dims

    # Color scale for k values
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_k))

    for row, sample_idx in enumerate(sample_indices):
        y_actual = Y_eval[sample_idx]

        for col, k in enumerate(k_values_to_plot):
            ax = axes[row, col]

            Mu = results[k]["Mu"][sample_idx]
            Sigma = results[k]["Sigma"][sample_idx]

            # Extract 2D
            mu2 = Mu[[d0, d1]]
            Sigma2 = Sigma[np.ix_([d0, d1], [d0, d1])]

            # Plot ellipse
            ex, ey = ellipse_points_2d(mu2, Sigma2, rho=rho)
            ax.plot(ex, ey, "-", linewidth=2, color=colors[col], label=f"k={k}")
            ax.fill(ex, ey, alpha=0.2, color=colors[col])

            # Plot center
            ax.scatter([mu2[0]], [mu2[1]], s=60, c=[colors[col]], marker="o", zorder=5)

            # Plot actual
            y2 = y_actual[[d0, d1]]
            ax.scatter(
                [y2[0]], [y2[1]], s=100, c="red", marker="*", zorder=6, label="Actual"
            )

            ax.set_xlabel(f"Y[{d0}]")
            ax.set_ylabel(f"Y[{d1}]")
            ax.set_title(f"Sample {sample_idx}, k={k}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="datalim")

            if row == 0 and col == 0:
                ax.legend(loc="upper right")

    fig.suptitle(
        f"Ellipsoid Comparison Across k Values (ρ={rho})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    output_path = output_dir / "knn_ellipses_by_k.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved ellipse grid to {output_path}")

    plt.close()


def plot_ellipse_size_vs_k(
    results: dict,
    output_dir: Path,
    dims: tuple[int, int] = (0, 1),
):
    """
    Plot how ellipse size (determinant, trace) changes with k.

    Shows the bias-variance tradeoff:
    - Small k: Larger variance in ellipse size
    - Large k: More consistent size (but potentially wrong shape)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = sorted(results.keys())
    d0, d1 = dims

    # Collect statistics across all eval points
    mean_det = []
    std_det = []
    mean_trace = []
    std_trace = []

    for k in k_values:
        Sigma_all = results[k]["Sigma"]  # (N_eval, M, M)
        N_eval = Sigma_all.shape[0]

        # Extract 2D and compute det/trace
        dets = []
        traces = []
        for i in range(N_eval):
            Sigma2 = Sigma_all[i][np.ix_([d0, d1], [d0, d1])]
            dets.append(np.linalg.det(Sigma2))
            traces.append(np.trace(Sigma2))

        mean_det.append(np.mean(dets))
        std_det.append(np.std(dets))
        mean_trace.append(np.mean(traces))
        std_trace.append(np.std(traces))

    mean_det = np.array(mean_det)
    std_det = np.array(std_det)
    mean_trace = np.array(mean_trace)
    std_trace = np.array(std_trace)

    # Panel 1: Determinant (ellipse area)
    ax = axes[0]
    ax.plot(k_values, mean_det, "o-", linewidth=2, markersize=8, color="#E63946")
    ax.fill_between(
        k_values, mean_det - std_det, mean_det + std_det, alpha=0.3, color="#E63946"
    )
    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel("Determinant (∝ Ellipse Area)")
    ax.set_title("Ellipse Size vs k")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 2: Trace (total variance)
    ax = axes[1]
    ax.plot(k_values, mean_trace, "s-", linewidth=2, markersize=8, color="#06AED5")
    ax.fill_between(
        k_values,
        mean_trace - std_trace,
        mean_trace + std_trace,
        alpha=0.3,
        color="#06AED5",
    )
    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel("Trace (Total Variance)")
    ax.set_title("Total Variance vs k")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Ellipsoid Statistics Across k Values (Bias-Variance Tradeoff)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    output_path = output_dir / "knn_ellipse_size_vs_k.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved ellipse size plot to {output_path}")

    output_path_pdf = output_dir / "knn_ellipse_size_vs_k.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


def plot_single_sample_overlay(
    results: dict,
    Y_eval: np.ndarray,
    sample_idx: int,
    k_values_to_plot: list[int],
    output_dir: Path,
    dims: tuple[int, int] = (0, 1),
    rho: float = 2.0,
    learned_omega_result: dict = None,
):
    """
    Overlay ellipses from different k values on single plot.

    Shows how ellipse shape changes with k, with optional learned omega comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    d0, d1 = dims
    y_actual = Y_eval[sample_idx]

    # Color scale for k values
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(k_values_to_plot)))

    for idx, k in enumerate(k_values_to_plot):
        Mu = results[k]["Mu"][sample_idx]
        Sigma = results[k]["Sigma"][sample_idx]

        # Extract 2D
        mu2 = Mu[[d0, d1]]
        Sigma2 = Sigma[np.ix_([d0, d1], [d0, d1])]

        # Plot ellipse
        ex, ey = ellipse_points_2d(mu2, Sigma2, rho=rho)
        ax.plot(ex, ey, "-", linewidth=2.5, color=colors[idx], label=f"k={k}")

        # Plot center
        ax.scatter(
            [mu2[0]],
            [mu2[1]],
            s=80,
            c=[colors[idx]],
            marker="o",
            edgecolors="black",
            linewidths=1,
            zorder=5,
        )

    # Plot learned omega ellipse if provided
    if learned_omega_result is not None:
        Mu_omega = learned_omega_result["Mu"][sample_idx]
        Sigma_omega = learned_omega_result["Sigma"][sample_idx]

        # Extract 2D
        mu2_omega = Mu_omega[[d0, d1]]
        Sigma2_omega = Sigma_omega[np.ix_([d0, d1], [d0, d1])]

        # Plot learned omega ellipse (distinct style)
        ex_omega, ey_omega = ellipse_points_2d(mu2_omega, Sigma2_omega, rho=rho)
        ax.plot(
            ex_omega,
            ey_omega,
            "-",
            linewidth=3.5,
            color="#E63946",
            label=f"Learned ω (k={learned_omega_result.get('k', 64)})",
        )

        # Plot center for learned omega
        ax.scatter(
            [mu2_omega[0]],
            [mu2_omega[1]],
            s=100,
            c="#E63946",
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=6,
        )

    # Plot actual observation
    y2 = y_actual[[d0, d1]]
    ax.scatter(
        [y2[0]],
        [y2[1]],
        s=200,
        c="lime",
        marker="*",
        zorder=10,
        edgecolors="black",
        linewidths=1.5,
        label="Actual",
    )

    ax.set_xlabel(f"Y[{d0}] (MW)")
    ax.set_ylabel(f"Y[{d1}] (MW)")
    ax.set_title(
        f"Ellipsoid Overlay: Sample {sample_idx} (ρ={rho})\n"
        + "How k affects uncertainty set shape",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()

    output_path = output_dir / f"knn_ellipse_overlay_sample{sample_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved overlay plot to {output_path}")

    output_path_pdf = output_dir / f"knn_ellipse_overlay_sample{sample_idx}.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


def save_summary(results: dict, output_dir: Path):
    """Save summary statistics to CSV."""
    rows = []
    for k in sorted(results.keys()):
        rows.append(
            {
                "k": k,
                "nll": results[k]["nll"],
            }
        )

    df = pd.DataFrame(rows)
    output_path = output_dir / "knn_k_sweep_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved summary to {output_path}")

    return df


def run_knn_k_sweep(
    k_values: list[int] = None,
    forecasts_parquet: Path = None,
    actuals_parquet: Path = None,
    output_subdir: str = "knn_k_sweep",
    sample_indices: list[int] = None,
    dims: tuple[int, int] = (0, 1),
    rho: float = 2.0,
    omega_path: str = None,
    omega_tau: float = 5.0,
    omega_k: int = 64,
    tau_values: list[float] = None,
    plot_3d: bool = False,
    feature_set: str = "high_dim_16d",
    use_residuals: bool = False,
    actual_col: str = "ACTUAL",
    zero_mean: bool = False,
):
    """
    Run complete k-NN k-value sweep and visualization.

    Parameters
    ----------
    k_values : list[int]
        List of k values to test
    forecasts_parquet : Path
        Path to forecasts parquet file
    actuals_parquet : Path
        Path to actuals parquet file
    output_subdir : str
        Subdirectory for outputs
    sample_indices : list[int]
        Indices of samples to visualize ellipses for
    dims : tuple[int, int]
        Which 2 dimensions to plot for ellipses
    rho : float
        Ellipsoid radius for visualization
    omega_path : str
        Path to saved omega.npy file for learned omega comparison
        If None, no comparison is shown
    omega_tau : float
        Bandwidth tau for learned omega kernel (default 5.0)
    omega_k : int
        Number of neighbors k for learned omega (default 64)
    tau_values : list[float]
        List of tau values to sweep for learned omega
        If None, uses [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    plot_3d : bool
        If True, generate 3D ellipsoid visualization for 3 wind farms
    actual_col : str
        Column name to use as Y target. Pass "RESIDUAL" to use pre-computed
        residuals from the actuals parquet.
    zero_mean : bool
        If True, force Mu=0 in learned omega predictions (for residuals mode).
    """
    if k_values is None:
        k_values = [8, 16, 32, 64, 128, 256, 512, 1024]

    if forecasts_parquet is None:
        forecasts_parquet = Path(
            "data/forecasts_filtered_rts3_constellation_v1.parquet"
        )
    if actuals_parquet is None:
        actuals_parquet = Path("data/actuals_filtered_rts3_constellation_v1.parquet")

    print("\n" + "=" * 80)
    print("K-NN COVARIANCE: k-VALUE SWEEP")
    print("=" * 80)
    print(f"k values: {k_values}")
    print(f"Dimensions for ellipse: {dims}")
    print(f"Ellipsoid radius (rho): {rho}")
    print("=" * 80 + "\n")

    # Load data
    print("Loading data...")
    X_train, Y_train, X_eval, Y_eval, x_cols, y_cols, _, _ = load_data(
        forecasts_parquet,
        actuals_parquet,
        feature_set=feature_set,
        use_residuals=use_residuals,
        actual_col=actual_col,
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Evaluation set: {X_eval.shape[0]} samples")
    if use_residuals or actual_col == "RESIDUAL":
        print("Target: residuals (actual - ensemble mean forecast)")
    print(f"Features: {x_cols}")
    print(f"Targets: {y_cols}\n")

    # Sweep k values
    print("Sweeping k values:")
    results = sweep_k_values(X_train, Y_train, X_eval, Y_eval, k_values)

    # Compute learned omega baseline if path provided
    learned_omega_result = None
    omega = None
    if omega_path is not None:
        omega_file = Path(omega_path)
        if omega_file.exists():
            print(f"\nLoading learned omega from {omega_path}...")
            omega = np.load(omega_file)
            print(f"  Omega shape: {omega.shape}")
            print(f"  Omega values: {omega}")
            print(
                f"  Computing learned omega baseline (tau={omega_tau}, k={omega_k})..."
            )

            learned_omega_result = compute_learned_omega_baseline(
                X_train,
                Y_train,
                X_eval,
                Y_eval,
                omega=omega,
                tau=omega_tau,
                k=omega_k,
                zero_mean=zero_mean,
            )
            print(f"  Learned omega NLL: {learned_omega_result['nll']:.4f}")
        else:
            print(
                f"\nWarning: omega_path '{omega_path}' not found, skipping comparison"
            )
            omega = None

    # Sweep tau values for learned omega
    tau_sweep_results = None
    best_tau = None
    best_tau_nll = None
    if omega_path is not None and omega is not None:
        if tau_values is None:
            tau_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        print(f"\nSweeping tau values for learned omega (k={omega_k}):")
        tau_sweep_results = sweep_tau_values(
            X_train,
            Y_train,
            X_eval,
            Y_eval,
            omega=omega,
            tau_values=tau_values,
            k=omega_k,
        )

    # Create output directory
    output_dir = Path("data/viz_artifacts") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Write config metadata
    import json
    config_meta = {
        "k_values": k_values,
        "feature_set": feature_set,
        "actual_col": actual_col,
        "use_residuals": use_residuals,
        "zero_mean": zero_mean,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (output_dir / "config.json").write_text(json.dumps(config_meta, indent=2))

    # Generate visualizations
    print("Generating visualizations...\n")

    # 1. NLL vs k plot (with learned omega comparison)
    plot_nll_vs_k(results, output_dir, learned_omega_result=learned_omega_result)

    # 1b. Tau sweep plot for learned omega
    if tau_sweep_results is not None:
        best_knn_nll = min(r["nll"] for r in results.values())
        best_tau, best_tau_nll = plot_nll_vs_tau(
            tau_sweep_results, output_dir, best_knn_nll=best_knn_nll
        )
        print(f"  Best tau: {best_tau} (NLL = {best_tau_nll:.4f})")

    # 2. Ellipse size statistics vs k
    plot_ellipse_size_vs_k(results, output_dir, dims=dims)

    # 3. Select sample indices if not provided
    if sample_indices is None:
        # Pick a few diverse samples
        n_eval = Y_eval.shape[0]
        sample_indices = [0, n_eval // 4, n_eval // 2, 3 * n_eval // 4]

    # 4. Ellipse grid (samples × k values)
    k_subset = [32, 64, 2048]
    plot_ellipses_at_various_k(
        results, Y_eval, sample_indices[:3], k_subset, output_dir, dims=dims, rho=rho
    )

    # 5. Single sample overlay (with learned omega comparison)
    plot_single_sample_overlay(
        results,
        Y_eval,
        sample_indices[0],
        k_values,
        output_dir,
        dims=dims,
        rho=rho,
        learned_omega_result=learned_omega_result,
    )

    # 5b. 3D ellipsoid overlay (if requested and Y has 3 dimensions)
    if plot_3d and Y_eval.shape[1] >= 3:
        plot_3d_ellipsoids_overlay(
            results,
            Y_eval,
            sample_indices[0],
            k_values,
            output_dir,
            y_cols=y_cols,
            rho=rho,
            learned_omega_result=learned_omega_result,
        )

    # 6. Save summary
    df_summary = save_summary(results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    best_k = df_summary.loc[df_summary["nll"].idxmin(), "k"]
    best_nll = df_summary["nll"].min()
    print(f"\nBest k-NN: k={best_k} (NLL = {best_nll:.4f})")

    if learned_omega_result is not None:
        omega_nll = learned_omega_result["nll"]
        improvement = best_nll - omega_nll
        print(f"\nLearned omega (tau={omega_tau}, k={omega_k}): NLL = {omega_nll:.4f}")
        if improvement > 0:
            print(
                f"  Improvement over best k-NN: {improvement:.4f} ({100*improvement/best_nll:.1f}%)"
            )
        else:
            print(f"  Difference from best k-NN: {improvement:.4f}")

    if best_tau is not None:
        print(f"\nTau sweep (k={omega_k}):")
        print(f"  Best tau: {best_tau}")
        print(f"  Best NLL: {best_tau_nll:.4f}")
        tau_improvement = best_nll - best_tau_nll
        if tau_improvement > 0:
            print(
                f"  Improvement over best k-NN: {tau_improvement:.4f} ({100*tau_improvement/best_nll:.1f}%)"
            )

    print("=" * 80)

    print("\n✓ k-NN k-value sweep complete!")
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - knn_nll_vs_k.png/pdf (with learned omega comparison)")
    if tau_sweep_results is not None:
        print("  - learned_omega_nll_vs_tau.png/pdf (tau sweep for learned omega)")
    print("  - knn_ellipse_size_vs_k.png/pdf")
    print("  - knn_ellipses_by_k.png")
    print("  - knn_ellipse_overlay_sample*.png/pdf (with learned omega ellipse)")
    if plot_3d and Y_eval.shape[1] >= 3:
        print("  - knn_3d_ellipsoid_overlay_sample*.png/pdf (3D wind farm ellipsoids)")
    print("  - knn_k_sweep_summary.csv")
    print()

    return results, df_summary


def run_multi_split_k_sweep(
    k_values: list[int] = None,
    forecasts_parquet: Path = None,
    actuals_parquet: Path = None,
    output_subdir: str = "knn_k_sweep",
    n_splits: int = 5,
    base_seeds: list[int] = None,
    train_frac: float = 0.75,
    ridge: float = 1e-4,
    feature_set: str = "high_dim_16d",
    use_residuals: bool = False,
    actual_col: str = "ACTUAL",
) -> pd.DataFrame:
    """
    Run k-NN k-value sweep with multiple random train/val splits.

    For each split (random seed), runs sweep_k_values and records per-k NLL.
    Saves per-k statistics (mean, std, min, max) to multi_split_k_stats.csv.

    Parameters
    ----------
    k_values : list[int]
        List of k values to test.
    n_splits : int
        Number of random train/val splits.
    base_seeds : list[int]
        Specific seeds to use. If None, uses range(n_splits).
    use_residuals : bool
        If True, model forecast error (actual - forecast mean) instead of raw actuals.
    actual_col : str
        Column name to use as Y target. Pass "RESIDUAL" to use pre-computed residuals.
    """
    if k_values is None:
        k_values = [32, 64, 128, 256, 512, 1024, 2048]

    if forecasts_parquet is None:
        forecasts_parquet = Path(
            "data/forecasts_filtered_rts3_constellation_v1.parquet"
        )
    if actuals_parquet is None:
        actuals_parquet = Path("data/actuals_filtered_rts3_constellation_v1.parquet")

    if base_seeds is None:
        base_seeds = list(range(n_splits))

    output_dir = Path("data/viz_artifacts") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("MULTI-SPLIT K-NN SWEEP")
    print("=" * 80)
    print(f"k values: {k_values}")
    print(f"Seeds: {base_seeds}")
    print("=" * 80 + "\n")

    all_rows = []

    for seed in base_seeds:
        print(f"--- Split seed={seed} ---")
        X_train, Y_train, X_eval, Y_eval, x_cols, y_cols, _, _ = load_data(
            forecasts_parquet,
            actuals_parquet,
            train_frac=train_frac,
            random_seed=seed,
            feature_set=feature_set,
            use_residuals=use_residuals,
            actual_col=actual_col,
        )
        print(f"  Train: {X_train.shape[0]}, Eval: {X_eval.shape[0]}")

        results = sweep_k_values(
            X_train, Y_train, X_eval, Y_eval, k_values, ridge=ridge
        )

        for k in k_values:
            all_rows.append({"seed": seed, "k": k, "nll": results[k]["nll"]})
        print()

    df_raw = pd.DataFrame(all_rows)

    # Compute per-k statistics
    stats = df_raw.groupby("k")["nll"].agg(["mean", "std", "min", "max"]).reset_index()
    stats.columns = ["k", "nll_mean", "nll_std", "nll_min", "nll_max"]

    # Save
    raw_path = output_dir / "multi_split_k_results.csv"
    stats_path = output_dir / "multi_split_k_stats.csv"
    df_raw.to_csv(raw_path, index=False)
    stats.to_csv(stats_path, index=False)
    print(f"Saved raw results to {raw_path}")
    print(f"Saved statistics to {stats_path}")

    # Print summary
    print("\nPer-k statistics:")
    print(stats.to_string(index=False))

    best_idx = stats["nll_mean"].idxmin()
    best_k = stats.loc[best_idx, "k"]
    best_nll = stats.loc[best_idx, "nll_mean"]
    best_std = stats.loc[best_idx, "nll_std"]
    print(f"\nBest k: {best_k} (mean NLL = {best_nll:.4f} +/- {best_std:.4f})")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="k-NN k-value sweep")
    parser.add_argument(
        "--multi-split",
        action="store_true",
        help="Run multi-split sweep (multiple random seeds for train/val split)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of random train/val splits (for --multi-split)",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="high_dim_16d",
        help="Feature set name (default: high_dim_16d)",
    )
    parser.add_argument(
        "--use-residuals",
        action="store_true",
        help="Use residuals (actual - forecast mean) instead of raw actuals",
    )

    args = parser.parse_args()

    # Derive actual_col and actuals parquet from --use-residuals
    from viz_artifacts_utils import resolve_residuals_config
    rcfg = resolve_residuals_config(args.use_residuals, Path("data"))
    actual_col = rcfg["actual_col"]
    actuals_pq = rcfg["actuals_parquet"] if args.use_residuals else None
    zero_mean = rcfg["zero_mean"]

    # Default omega path from focused_2d experiments
    default_omega_path = "data/viz_artifacts/focused_2d/best_omega.npy"

    # Use separate output directory for residuals
    output_subdir = f"knn_k_sweep{rcfg['suffix']}"

    if args.multi_split:
        run_multi_split_k_sweep(
            k_values=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            n_splits=args.n_splits,
            feature_set=args.feature_set,
            use_residuals=args.use_residuals,
            actual_col=actual_col,
            actuals_parquet=actuals_pq,
            output_subdir=output_subdir,
        )
    else:
        results, df_summary = run_knn_k_sweep(
            k_values=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            output_subdir=output_subdir,
            dims=(0, 1),
            rho=2.0,
            omega_path=None,  # default_omega_path,
            omega_tau=0.5,
            omega_k=256,
            tau_values=[
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                1.0,
                2.0,
                5.0,
                10.0,
                20.0,
            ],
            plot_3d=True,
            feature_set=args.feature_set,
            use_residuals=args.use_residuals,
            actual_col=actual_col,
            actuals_parquet=actuals_pq,
            zero_mean=zero_mean,
        )
