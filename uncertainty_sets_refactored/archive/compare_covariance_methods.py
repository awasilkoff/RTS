"""
Unified comparison of covariance estimation methods.

Compares:
1. Global covariance (baseline) - No adaptation
2. k-NN (uniform weights) - Local adaptation, sweep k
3. Learned omega (kernel weights) - Local adaptation + feature learning, sweep tau

Imports sweep logic from existing scripts to avoid duplication.

Usage:
    python compare_covariance_methods.py --feature-set high_dim_16d
    python compare_covariance_methods.py --feature-set focused_2d --k-values 32 64 128 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_config import setup_plotting

# Apply global plotting config
setup_plotting()

# Import from existing modules (no duplication)
from data_processing_extended import FEATURE_BUILDERS, FEATURE_SET_DESCRIPTIONS
from utils import fit_scaler, apply_scaler
from covariance_optimization import (
    KernelCovConfig,
    FitConfig,
    CovPredictConfig,
    fit_omega,
    predict_mu_sigma_topk_cross,
    predict_mu_sigma_knn,
)


def _mean_gaussian_nll(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Mean NLL of Y under N(Mu, Sigma) row-wise."""
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
            nll_total += 1e6

    return nll_total / N


def load_and_split_data(
    feature_set: str,
    data_dir: Path,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    scaler_type: str = "standard",
    random_seed: int = 42,
):
    """Load data and split into train/val/test."""
    forecasts = pd.read_parquet(
        data_dir / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    actuals = pd.read_parquet(
        data_dir / "actuals_filtered_rts3_constellation_v1.parquet"
    )

    build_fn = FEATURE_BUILDERS[feature_set]
    X_raw, Y, times, x_cols, y_cols = build_fn(
        forecasts, actuals, drop_any_nan_rows=True
    )

    # Random split
    n = X_raw.shape[0]
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Fit scaler on training data
    scaler = fit_scaler(X_raw[train_idx], scaler_type)
    X = apply_scaler(X_raw, scaler)

    return {
        "X": X,
        "Y": Y,
        "X_raw": X_raw,
        "times": times,
        "x_cols": x_cols,
        "y_cols": y_cols,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "scaler": scaler,
        "scaler_type": scaler_type,
    }


def compute_global_baseline(
    Y_train: np.ndarray, Y_eval: np.ndarray, ridge: float = 1e-3
):
    """Compute global covariance baseline (no adaptation)."""
    Mu_global = np.mean(Y_train, axis=0)
    Sigma_global = np.cov(Y_train, rowvar=False) + ridge * np.eye(Y_train.shape[1])

    # Tile for all eval points
    n_eval = len(Y_eval)
    Mu_eval = np.tile(Mu_global, (n_eval, 1))
    Sigma_eval = np.tile(Sigma_global[None, :, :], (n_eval, 1, 1))

    nll = _mean_gaussian_nll(Y_eval, Mu_eval, Sigma_eval)
    return {"nll": nll, "Mu": Mu_eval, "Sigma": Sigma_eval}


def sweep_knn(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    k_values: list[int],
    ridge: float = 1e-3,
):
    """Sweep k values for k-NN baseline."""
    results = {}
    for k in k_values:
        Mu_eval, Sigma_eval = predict_mu_sigma_knn(
            X_eval, X_train, Y_train, k=k, ridge=ridge
        )
        nll = _mean_gaussian_nll(Y_eval, Mu_eval, Sigma_eval)
        results[k] = {"nll": nll, "Mu": Mu_eval, "Sigma": Sigma_eval}
        print(f"  k-NN k={k}: NLL={nll:.4f}")
    return results


def sweep_learned_omega(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    tau_values: list[float],
    k: int = 64,
    ridge: float = 1e-3,
    omega_constraint: str = "softmax",
    max_iters: int = 250,
):
    """Sweep tau values for learned omega."""
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    results = {}

    for tau in tau_values:
        # Fit omega
        omega0 = np.ones(X.shape[1], dtype=float)
        cfg = KernelCovConfig(tau=float(tau), ridge=float(ridge), zero_mean=False)
        fit_cfg = FitConfig(
            max_iters=max_iters,
            step_size=0.1,
            grad_clip=10.0,
            tol=1e-7,
            verbose_every=999999,
            dtype="float32",
            device="cpu",
            omega_l2_reg=0.0,
            omega_constraint=omega_constraint,
            # k_fit=None: use all neighbors during training (empirically better)
        )

        omega_hat, _ = fit_omega(
            X, Y, omega0=omega0, train_idx=train_idx, cfg=cfg, fit_cfg=fit_cfg
        )

        # Predict on validation
        pred_cfg = CovPredictConfig(
            tau=float(tau),
            ridge=float(ridge),
            enforce_nonneg_omega=True,
            dtype="float32",
            device="cpu",
            zero_mean=False,
        )

        Mu_eval, Sigma_eval = predict_mu_sigma_topk_cross(
            X_query=X_val,
            X_ref=X_train,
            Y_ref=Y_train,
            omega=omega_hat,
            cfg=pred_cfg,
            k=k,
            exclude_self_if_same=False,
            return_type="numpy",
        )

        nll = _mean_gaussian_nll(Y_val, Mu_eval, Sigma_eval)
        results[tau] = {
            "nll": nll,
            "omega": omega_hat,
            "Mu": Mu_eval,
            "Sigma": Sigma_eval,
        }
        omega_str = ", ".join([f"{w:.3f}" for w in omega_hat[:4]])
        if len(omega_hat) > 4:
            omega_str += ", ..."
        print(f"  Learned ω tau={tau}: NLL={nll:.4f}, ω=[{omega_str}]")

    return results


def plot_comparison_bar_chart(
    global_nll: float,
    best_knn_nll: float,
    best_knn_k: int,
    best_omega_nll: float,
    best_omega_tau: float,
    output_dir: Path,
    feature_set: str,
):
    """Create bar chart comparing the three methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        "Global\nCovariance",
        f"k-NN\n(k={best_knn_k})",
        f"Learned ω\n(τ={best_omega_tau})",
    ]
    nlls = [global_nll, best_knn_nll, best_omega_nll]
    colors = ["#6c757d", "#2A9D8F", "#E63946"]

    bars = ax.bar(methods, nlls, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, nll in zip(bars, nlls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{nll:.3f}",
            ha="center",
            va="bottom",
        )

    omega_vs_knn = best_knn_nll - best_omega_nll

    ax.set_ylabel("Validation NLL (lower is better)")
    ax.set_title(f"Covariance Estimation Comparison: {feature_set}")
    ax.grid(axis="y", alpha=0.3)

    # Add horizontal line at global baseline
    ax.axhline(global_nll, color="#6c757d", linestyle="--", linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    output_path = output_dir / "comparison_bar_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"(ok) Saved comparison bar chart to {output_path}")

    output_path_pdf = output_dir / "comparison_bar_chart.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"(ok) Saved PDF to {output_path_pdf}")

    plt.close()


def plot_sweep_curves(
    knn_results: dict,
    omega_results: dict,
    global_nll: float,
    output_dir: Path,
    feature_set: str,
):
    """Plot k sweep and tau sweep curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: k-NN k sweep
    ax = axes[0]
    k_values = sorted(knn_results.keys())
    k_nlls = [knn_results[k]["nll"] for k in k_values]

    ax.plot(
        k_values,
        k_nlls,
        "o-",
        linewidth=2.5,
        markersize=10,
        color="#2A9D8F",
        label="k-NN",
    )
    ax.axhline(
        global_nll,
        color="#6c757d",
        linestyle="--",
        linewidth=2,
        label="Global baseline",
    )

    # Mark best
    best_k_idx = np.argmin(k_nlls)
    ax.scatter(
        [k_values[best_k_idx]],
        [k_nlls[best_k_idx]],
        s=250,
        c="#2A9D8F",
        marker="*",
        zorder=10,
        edgecolors="black",
        linewidths=2,
        label=f"Best: k={k_values[best_k_idx]}",
    )

    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel("Validation NLL")
    ax.set_title("k-NN: Sweep k")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: Learned omega tau sweep
    ax = axes[1]
    tau_values = sorted(omega_results.keys())
    tau_nlls = [omega_results[tau]["nll"] for tau in tau_values]

    ax.plot(
        tau_values,
        tau_nlls,
        "o-",
        linewidth=2.5,
        markersize=10,
        color="#E63946",
        label="Learned ω",
    )
    ax.axhline(
        global_nll,
        color="#6c757d",
        linestyle="--",
        linewidth=2,
        label="Global baseline",
    )

    # Also show best k-NN as reference
    best_knn_nll = min(k_nlls)
    ax.axhline(
        best_knn_nll, color="#2A9D8F", linestyle=":", linewidth=2, label=f"Best k-NN"
    )

    # Mark best
    best_tau_idx = np.argmin(tau_nlls)
    ax.scatter(
        [tau_values[best_tau_idx]],
        [tau_nlls[best_tau_idx]],
        s=250,
        c="#E63946",
        marker="*",
        zorder=10,
        edgecolors="black",
        linewidths=2,
        label=f"Best: τ={tau_values[best_tau_idx]}",
    )

    ax.set_xlabel("τ (Kernel Bandwidth)")
    ax.set_ylabel("Validation NLL")
    ax.set_title("Learned ω: Sweep τ")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"Hyperparameter Tuning: {feature_set}", y=1.02
    )

    plt.tight_layout()

    output_path = output_dir / "hyperparameter_sweeps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"(ok) Saved hyperparameter sweeps to {output_path}")

    output_path_pdf = output_dir / "hyperparameter_sweeps.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"(ok) Saved PDF to {output_path_pdf}")

    plt.close()


def plot_omega_bar_chart(
    omega: np.ndarray, x_cols: list[str], output_dir: Path, feature_set: str
):
    """Plot learned omega weights."""
    fig, ax = plt.subplots(figsize=(max(10, len(omega) * 0.8), 6))

    # Sort by omega value for clarity
    sorted_idx = np.argsort(omega)[::-1]
    omega_sorted = omega[sorted_idx]
    cols_sorted = [x_cols[i] for i in sorted_idx]

    # Color top features differently
    colors = ["#E63946" if i < 3 else "#6c757d" for i in range(len(omega))]

    bars = ax.bar(
        range(len(omega)), omega_sorted, color=colors, edgecolor="black", linewidth=1
    )

    ax.set_xticks(range(len(omega)))
    ax.set_xticklabels(cols_sorted, rotation=45, ha="right", )
    ax.set_ylabel("Learned ω (feature weight)")
    ax.set_title(
        f"Learned Feature Weights: {feature_set}"    )
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on top bars
    for i, (bar, val) in enumerate(zip(bars, omega_sorted)):
        if i < 5:  # Label top 5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    output_path = output_dir / "omega_bar_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"(ok) Saved omega bar chart to {output_path}")

    plt.close()


def run_comparison(
    feature_set: str,
    data_dir: Path,
    output_dir: Path,
    k_values: list[int] = None,
    tau_values: list[float] = None,
    omega_k: int = 64,
    scaler_type: str = "standard",
    omega_constraint: str = "softmax",
    ridge: float = 1e-3,
):
    """Run full comparison of covariance methods."""
    if k_values is None:
        k_values = [32, 64, 128, 256, 512]
    if tau_values is None:
        tau_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"COVARIANCE METHOD COMPARISON: {feature_set}")
    print("=" * 80)
    print(f"k values for k-NN: {k_values}")
    print(f"tau values for learned ω: {tau_values}")
    print(f"omega k: {omega_k}")
    print(f"scaler: {scaler_type}")
    print(f"omega constraint: {omega_constraint}")
    print("=" * 80 + "\n")

    # Load and split data
    print("Loading data...")
    data = load_and_split_data(feature_set, data_dir, scaler_type=scaler_type)
    X, Y = data["X"], data["Y"]
    train_idx, val_idx = data["train_idx"], data["val_idx"]
    x_cols = data["x_cols"]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(
        f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(data['test_idx'])}"
    )
    print(f"Features ({len(x_cols)}): {x_cols[:5]}{'...' if len(x_cols) > 5 else ''}\n")

    # 1. Global baseline
    print("Computing global covariance baseline...")
    global_result = compute_global_baseline(Y_train, Y_val, ridge=ridge)
    global_nll = global_result["nll"]
    print(f"  Global NLL: {global_nll:.4f}\n")

    # 2. k-NN sweep
    print("Sweeping k-NN k values...")
    knn_results = sweep_knn(X_train, Y_train, X_val, Y_val, k_values, ridge=ridge)
    best_k = min(knn_results, key=lambda k: knn_results[k]["nll"])
    best_knn_nll = knn_results[best_k]["nll"]
    print(f"  Best k-NN: k={best_k}, NLL={best_knn_nll:.4f}\n")

    # 3. Learned omega sweep
    print("Sweeping learned omega tau values...")
    omega_results = sweep_learned_omega(
        X,
        Y,
        train_idx,
        val_idx,
        tau_values,
        k=omega_k,
        ridge=ridge,
        omega_constraint=omega_constraint,
    )
    best_tau = min(omega_results, key=lambda t: omega_results[t]["nll"])
    best_omega_nll = omega_results[best_tau]["nll"]
    best_omega = omega_results[best_tau]["omega"]
    print(f"  Best learned ω: tau={best_tau}, NLL={best_omega_nll:.4f}\n")

    # Generate plots
    print("Generating visualizations...")
    plot_comparison_bar_chart(
        global_nll,
        best_knn_nll,
        best_k,
        best_omega_nll,
        best_tau,
        output_dir,
        feature_set,
    )
    plot_sweep_curves(knn_results, omega_results, global_nll, output_dir, feature_set)
    plot_omega_bar_chart(best_omega, x_cols, output_dir, feature_set)

    # Save summary
    summary = {
        "feature_set": feature_set,
        "n_features": len(x_cols),
        "x_cols": x_cols,
        "scaler_type": scaler_type,
        "omega_constraint": omega_constraint,
        "global_nll": global_nll,
        "best_knn_k": best_k,
        "best_knn_nll": best_knn_nll,
        "knn_improvement_vs_global": global_nll - best_knn_nll,
        "best_omega_tau": best_tau,
        "best_omega_nll": best_omega_nll,
        "omega_improvement_vs_global": global_nll - best_omega_nll,
        "omega_improvement_vs_knn": best_knn_nll - best_omega_nll,
        "omega_improvement_pct": 100 * (best_knn_nll - best_omega_nll) / best_knn_nll,
        "best_omega": best_omega.tolist(),
    }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(ok) Saved summary to {output_dir / 'comparison_summary.json'}")

    # Save omega
    np.save(output_dir / "best_omega.npy", best_omega)
    print(f"(ok) Saved best omega to {output_dir / 'best_omega.npy'}")

    # Print final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Global covariance:  NLL = {global_nll:.4f}")
    print(
        f"Best k-NN (k={best_k}):   NLL = {best_knn_nll:.4f}  (v{global_nll - best_knn_nll:.4f} vs global)"
    )
    print(
        f"Best learned ω (τ={best_tau}): NLL = {best_omega_nll:.4f}  (v{global_nll - best_omega_nll:.4f} vs global)"
    )
    print(
        f"\nLearned ω improvement over k-NN: {best_knn_nll - best_omega_nll:.4f} ({100*(best_knn_nll - best_omega_nll)/best_knn_nll:.1f}%)"
    )
    print("=" * 80)

    print(f"\nOutputs saved to: {output_dir}/")
    print("  - comparison_bar_chart.png/pdf")
    print("  - hyperparameter_sweeps.png/pdf")
    print("  - omega_bar_chart.png")
    print("  - comparison_summary.json")
    print("  - best_omega.npy")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare covariance estimation methods"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        required=True,
        choices=list(FEATURE_BUILDERS.keys()),
        help="Feature set to use",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/viz_artifacts/<feature_set>_comparison)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 512],
        help="k values for k-NN sweep",
    )
    parser.add_argument(
        "--tau-values",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        help="tau values for learned omega sweep",
    )
    parser.add_argument(
        "--omega-k",
        type=int,
        default=64,
        help="k for learned omega prediction (default 64)",
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        default="standard",
        choices=["none", "standard", "minmax"],
        help="Scaler type",
    )
    parser.add_argument(
        "--omega-constraint",
        type=str,
        default="softmax",
        choices=["none", "softmax", "simplex", "normalize"],
        help="Omega constraint type",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.data_dir / "viz_artifacts" / f"{args.feature_set}_comparison"

    run_comparison(
        feature_set=args.feature_set,
        data_dir=args.data_dir,
        output_dir=output_dir,
        k_values=args.k_values,
        tau_values=args.tau_values,
        omega_k=args.omega_k,
        scaler_type=args.scaler_type,
        omega_constraint=args.omega_constraint,
    )


if __name__ == "__main__":
    main()
