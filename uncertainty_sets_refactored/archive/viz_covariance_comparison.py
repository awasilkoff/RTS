"""
Covariance Method Comparison Visualization.

Compares three covariance estimation methods:
1. Learned omega (kernel-weighted with optimized feature weights)
2. Equal weights baseline (omega = [1, 1])
3. Simple KNN baseline (uniform weights over k neighbors)

Primary metric: Negative Log Likelihood (NLL) on eval set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils import fit_standard_scaler
from data_processing import build_XY_for_covariance_system_only
from covariance_optimization import (
    CovPredictConfig,
    predict_mu_sigma_topk_cross,
    predict_mu_sigma_knn,
)
from viz_ellipsoid_plane_3d import plot_multi_ellipsoid_comparison
from viz_kernel_distance import plot_kernel_distance_comparison


def _mean_gaussian_nll(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute NLL of Y under N(Mu, Sigma) row-wise.

    Returns per-sample NLL array of shape (N,).
    """
    Y = np.asarray(Y, dtype=float)
    Mu = np.asarray(Mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    N, M = Y.shape
    nll = np.empty(N, dtype=float)

    for i in range(N):
        S = Sigma[i]
        r = (Y[i] - Mu[i]).reshape(M, 1)
        L = np.linalg.cholesky(S)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        x = np.linalg.solve(L, r)
        x = np.linalg.solve(L.T, x)
        quad = float(r.T @ x)
        nll[i] = 0.5 * (logdet + quad + M * np.log(2.0 * np.pi))

    return nll


def load_and_prepare_data(
    forecasts_parquet: Path,
    actuals_parquet: Path,
    train_frac: float = 0.75,
    standardize: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """Load data and prepare train/eval splits.

    Parameters
    ----------
    standardize : bool
        If True, standardize features to zero mean and unit variance.
        If False, use raw features (omega learns the scaling).

    Returns
    -------
    X_train, Y_train, X_eval, Y_eval, Xp, Y, times, x_cols, y_cols, train_idx, eval_idx
    """
    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)

    X, Y, times, x_cols, y_cols = build_XY_for_covariance_system_only(
        forecasts, actuals, drop_any_nan_rows=True
    )

    if standardize:
        scaler = fit_standard_scaler(X)
        Xp = scaler.transform(X)
    else:
        Xp = X.copy()

    # Random train/eval split (more representative than temporal holdout)
    n = Xp.shape[0]
    n_train = int(train_frac * n)

    # Fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = np.sort(indices[:n_train])  # Sort for deterministic behavior
    eval_idx = np.sort(indices[n_train:])

    X_train, Y_train = Xp[train_idx], Y[train_idx]
    X_eval, Y_eval = Xp[eval_idx], Y[eval_idx]

    return X_train, Y_train, X_eval, Y_eval, Xp, Y, times, x_cols, y_cols, train_idx, eval_idx


def run_all_methods(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    omega_learned: np.ndarray,
    omega_equal: np.ndarray,
    tau: float,
    ridge: float,
    k_kernel: int,
    k_knn_list: list[int],
) -> dict:
    """Run all covariance estimation methods on eval set."""
    results = {}

    pred_cfg = CovPredictConfig(
        tau=tau, ridge=ridge, enforce_nonneg_omega=True, dtype="float32", device="cpu"
    )

    # Learned omega
    Mu_learned, Sigma_learned = predict_mu_sigma_topk_cross(
        X_query=X_eval,
        X_ref=X_train,
        Y_ref=Y_train,
        omega=omega_learned,
        cfg=pred_cfg,
        k=k_kernel,
        exclude_self_if_same=False,
        return_type="numpy",
    )
    results["Learned Omega"] = {"Mu": Mu_learned, "Sigma": Sigma_learned}

    # Equal weights
    Mu_equal, Sigma_equal = predict_mu_sigma_topk_cross(
        X_query=X_eval,
        X_ref=X_train,
        Y_ref=Y_train,
        omega=omega_equal,
        cfg=pred_cfg,
        k=k_kernel,
        exclude_self_if_same=False,
        return_type="numpy",
    )
    results["Equal Weights"] = {"Mu": Mu_equal, "Sigma": Sigma_equal}

    # KNN baselines
    for k in k_knn_list:
        Mu_knn, Sigma_knn = predict_mu_sigma_knn(
            X_query=X_eval,
            X_ref=X_train,
            Y_ref=Y_train,
            k=k,
            ridge=ridge,
        )
        results[f"KNN (k={k})"] = {"Mu": Mu_knn, "Sigma": Sigma_knn}

    return results


def compute_nll_metrics(
    results: dict,
    Y_eval: np.ndarray,
) -> pd.DataFrame:
    """Compute NLL metrics for all methods."""
    rows = []
    for method_name, data in results.items():
        nll = _mean_gaussian_nll(Y_eval, data["Mu"], data["Sigma"])
        rows.append(
            {
                "method": method_name,
                "mean_nll": nll.mean(),
                "std_nll": nll.std(),
                "min_nll": nll.min(),
                "max_nll": nll.max(),
                "median_nll": np.median(nll),
            }
        )
        # Store per-sample NLL for later use
        data["nll"] = nll

    return pd.DataFrame(rows).sort_values("mean_nll")


def plot_nll_bar_chart(
    nll_df: pd.DataFrame,
    out_path: Path,
    figsize: tuple[int, int] = (10, 6),
):
    """Plot NLL comparison bar chart."""
    fig, ax = plt.subplots(figsize=figsize)

    methods = nll_df["method"].tolist()
    means = nll_df["mean_nll"].tolist()
    stds = nll_df["std_nll"].tolist()

    colors = []
    for m in methods:
        if "Learned" in m:
            colors.append("green")
        elif "Equal" in m:
            colors.append("blue")
        else:
            colors.append("orange")

    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

    ax.set_ylabel("Mean NLL (lower is better)")
    ax.set_xlabel("Method")
    ax.set_title("Covariance Estimation Method Comparison\n(Eval Set NLL)")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_timestamp_nll(
    results: dict,
    times_eval: pd.DatetimeIndex,
    out_path: Path,
    figsize: tuple[int, int] = (14, 6),
):
    """Plot per-timestamp NLL comparison."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "Learned Omega": "green",
        "Equal Weights": "blue",
    }

    # Plot main methods
    for method_name in ["Learned Omega", "Equal Weights"]:
        if method_name in results:
            nll = results[method_name]["nll"]
            ax.plot(
                times_eval,
                nll,
                label=method_name,
                color=colors.get(method_name, "gray"),
                alpha=0.7,
                linewidth=1.0,
            )

    # Plot best KNN
    knn_methods = [m for m in results.keys() if m.startswith("KNN")]
    if knn_methods:
        # Find best KNN by mean NLL
        best_knn = min(knn_methods, key=lambda m: results[m]["nll"].mean())
        ax.plot(
            times_eval,
            results[best_knn]["nll"],
            label=best_knn,
            color="orange",
            alpha=0.7,
            linewidth=1.0,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("NLL")
    ax.set_title("Per-Timestamp NLL Comparison")
    ax.legend(loc="upper right")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_covariance_heatmaps(
    results: dict,
    sample_idx: int,
    y_cols: list[str],
    out_path: Path,
    figsize: tuple[int, int] = (15, 4),
):
    """Plot 3x3 covariance heatmaps side-by-side."""
    methods_to_plot = ["Equal Weights", "Learned Omega"]
    knn_methods = [m for m in results.keys() if m.startswith("KNN")]
    if knn_methods:
        best_knn = min(knn_methods, key=lambda m: results[m]["nll"].mean())
        methods_to_plot.append(best_knn)

    n_methods = len(methods_to_plot)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    # Find global vmin/vmax for consistent color scale
    all_sigmas = [results[m]["Sigma"][sample_idx] for m in methods_to_plot]
    vmin = min(s.min() for s in all_sigmas)
    vmax = max(s.max() for s in all_sigmas)

    for i, method_name in enumerate(methods_to_plot):
        ax = axes[i] if n_methods > 1 else axes
        Sigma = results[method_name]["Sigma"][sample_idx]

        im = ax.imshow(Sigma, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(method_name)
        ax.set_xticks(range(len(y_cols)))
        ax.set_yticks(range(len(y_cols)))
        ax.set_xticklabels(y_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(y_cols, fontsize=8)

        # Add value annotations
        for ii in range(Sigma.shape[0]):
            for jj in range(Sigma.shape[1]):
                ax.text(
                    jj,
                    ii,
                    f"{Sigma[ii, jj]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(Sigma[ii, jj]) > (vmax - vmin) / 2 else "black",
                )

    fig.colorbar(im, ax=axes, shrink=0.8, label="Covariance")
    fig.suptitle(f"Covariance Matrix Comparison (Sample {sample_idx})", fontsize=12)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    DATA_DIR = Path(__file__).parent / "data"
    ART = DATA_DIR / "viz_artifacts"
    ART.mkdir(exist_ok=True, parents=True)

    # Best hyperparameters from sweep
    OMEGA_LEARNED = np.array([9.758, 12.221])
    OMEGA_EQUAL = np.array([1.0, 1.0])
    TAU = 5.0
    RIDGE = 1e-3
    K_KERNEL = 128
    K_KNN_LIST = [32, 64, 128]

    print("Loading data...")
    (
        X_train,
        Y_train,
        X_eval,
        Y_eval,
        Xs,
        Y,
        times,
        x_cols,
        y_cols,
        train_idx,
        eval_idx,
    ) = load_and_prepare_data(
        forecasts_parquet=DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet",
        actuals_parquet=DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet",
        train_frac=0.75,
    )

    n_train = X_train.shape[0]
    times_eval = times[eval_idx]  # Use actual eval indices, not temporal slice

    print(f"Train: {X_train.shape[0]}, Eval: {X_eval.shape[0]}")
    print(f"Features: {x_cols}")
    print(f"Targets: {y_cols}")

    print("\nRunning all methods...")
    results = run_all_methods(
        X_train,
        Y_train,
        X_eval,
        OMEGA_LEARNED,
        OMEGA_EQUAL,
        TAU,
        RIDGE,
        K_KERNEL,
        K_KNN_LIST,
    )

    print("\nComputing NLL metrics...")
    nll_df = compute_nll_metrics(results, Y_eval)
    print("\nNLL Summary:")
    print(nll_df.to_string(index=False))

    # Save summary CSV
    nll_df.to_csv(ART / "covariance_comparison_nll.csv", index=False)
    print(f"\nSaved: {ART / 'covariance_comparison_nll.csv'}")

    # Panel 4: NLL Bar Chart
    print("\nGenerating NLL bar chart...")
    plot_nll_bar_chart(nll_df, ART / "cov_comparison_nll_bar.png")

    # Panel 5: Per-timestamp NLL
    print("Generating per-timestamp NLL plot...")
    plot_per_timestamp_nll(results, times_eval, ART / "cov_comparison_nll_timeseries.png")

    # Panel 3: Covariance heatmaps for a sample point
    print("Generating covariance heatmaps...")
    sample_idx = len(X_eval) // 2  # Middle of eval set
    plot_covariance_heatmaps(results, sample_idx, y_cols, ART / "cov_comparison_heatmaps.png")

    # Panel 1: Kernel distance comparison
    print("Generating kernel distance comparison...")
    target_idx = eval_idx[sample_idx]  # Map eval set index to full dataset index
    plot_kernel_distance_comparison(
        Xs,
        x_cols,
        times,
        target_idx,
        OMEGA_EQUAL,
        OMEGA_LEARNED,
        TAU,
        save_path=ART / "cov_comparison_kernel_weights.png",
    )

    # Panel 2: 3D Ellipsoid comparison
    print("Generating 3D ellipsoid comparison...")
    mu_dict = {
        "Equal Weights": results["Equal Weights"]["Mu"][sample_idx],
        "Learned Omega": results["Learned Omega"]["Mu"][sample_idx],
    }
    sigma_dict = {
        "Equal Weights": results["Equal Weights"]["Sigma"][sample_idx],
        "Learned Omega": results["Learned Omega"]["Sigma"][sample_idx],
    }
    # Add best KNN
    knn_methods = [m for m in results.keys() if m.startswith("KNN")]
    if knn_methods:
        best_knn = min(knn_methods, key=lambda m: results[m]["nll"].mean())
        mu_dict[best_knn] = results[best_knn]["Mu"][sample_idx]
        sigma_dict[best_knn] = results[best_knn]["Sigma"][sample_idx]

    plot_multi_ellipsoid_comparison(
        mu_dict=mu_dict,
        sigma_dict=sigma_dict,
        rho=1.0,  # Unit radius for visualization
        y_actual=Y_eval[sample_idx],
        out_png=ART / "cov_comparison_ellipsoids_3d.png",
        dims3=(0, 1, 2),
        colors={
            "Equal Weights": "blue",
            "Learned Omega": "green",
            best_knn: "orange",
        },
        title=f"Covariance Ellipsoid Comparison\n(Eval Sample {sample_idx}, rho=1.0)",
    )

    print("\nAll visualizations complete!")
    print(f"Outputs saved to: {ART}")


if __name__ == "__main__":
    main()
