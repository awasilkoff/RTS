"""
Diagnostic: Multi-seed analysis for tau-omega optimization.

Checks whether the learned omega is consistent across random initializations,
or if different seeds lead to different local minima.

Usage:
    python diagnose_tau_omega_seeds.py

Outputs saved to: data/viz_artifacts/tau_omega_diagnosis/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_config import setup_plotting, IEEE_COL_WIDTH, IEEE_TWO_COL_WIDTH, COLORS

setup_plotting()

from data_processing_extended import FEATURE_BUILDERS
from utils import fit_scaler, apply_scaler
from covariance_optimization import (
    KernelCovConfig,
    FitConfig,
    CovPredictConfig,
    fit_omega,
    predict_mu_sigma_topk_cross,
)


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "tau_omega_diagnosis"  # overridden by --use-residuals


def _mean_gaussian_nll(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Mean NLL of Y under N(Mu, Sigma)."""
    N, M = Y.shape
    nll_total = 0.0
    for i in range(N):
        r = (Y[i] - Mu[i]).reshape(M, 1)
        try:
            L = np.linalg.cholesky(Sigma[i])
            logdet = 2.0 * np.log(np.diag(L)).sum()
            x = np.linalg.solve(L, r)
            x = np.linalg.solve(L.T, x)
            quad = float(r.T @ x)
            nll_total += 0.5 * (logdet + quad + M * np.log(2.0 * np.pi))
        except np.linalg.LinAlgError:
            nll_total += 1e6
    return nll_total / N


def run_single_seed(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    tau: float,
    seed: int,
    k: int = 128,
    ridge: float = 1e-3,
    max_iters: int = 300,
) -> dict:
    """Run omega optimization with a specific random seed."""
    rng = np.random.RandomState(seed)

    # Random initialization for omega (positive values, roughly around 1)
    d = X.shape[1]
    omega0 = np.abs(rng.randn(d)) + 0.5  # Positive, centered around ~1

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

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
        omega_constraint="softmax",
    )

    omega_hat, hist = fit_omega(
        X,
        Y,
        omega0=omega0,
        train_idx=train_idx,
        cfg=cfg,
        fit_cfg=fit_cfg,
        return_history=True,
    )

    # Evaluate on validation
    pred_cfg = CovPredictConfig(
        tau=float(tau), ridge=float(ridge), enforce_nonneg_omega=True
    )
    Mu_val, Sigma_val = predict_mu_sigma_topk_cross(
        X_val, X_train, Y_train, omega=omega_hat, cfg=pred_cfg, k=k
    )
    val_nll = _mean_gaussian_nll(Y_val, Mu_val, Sigma_val)

    # Get final training NLL
    df_hist = pd.DataFrame(hist)
    train_nll = df_hist.iloc[-1]["loss"]
    n_iters = len(df_hist)

    return {
        "seed": seed,
        "tau": tau,
        "omega": omega_hat.copy(),
        "omega0": omega0.copy(),
        "val_nll": val_nll,
        "train_nll": train_nll,
        "n_iters": n_iters,
        "converged": n_iters < max_iters,
    }


def run_multi_seed_sweep(
    feature_set: str = "focused_2d",
    tau_values: list[float] = None,
    n_seeds: int = 10,
    k: int = 128,
    ridge: float = 1e-3,
    actual_col: str = "ACTUAL",
    actuals_parquet: Path = None,
) -> pd.DataFrame:
    """Run omega optimization with multiple seeds for each tau."""

    if tau_values is None:
        tau_values = [0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    if actuals_parquet is None:
        actuals_parquet = DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"

    # Load data
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    actuals = pd.read_parquet(actuals_parquet)

    build_fn = FEATURE_BUILDERS[feature_set]
    X_raw, Y, times, x_cols, y_cols = build_fn(
        forecasts, actuals, drop_any_nan_rows=True, actual_col=actual_col
    )

    # Split
    n = X_raw.shape[0]
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_train = int(0.5 * n)
    n_val = int(0.25 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]

    # Scale
    scaler = fit_scaler(X_raw[train_idx], "standard")
    X = apply_scaler(X_raw, scaler)

    print(f"Feature set: {feature_set}")
    print(f"Features: {x_cols}")
    print(f"Data: {n} samples, train={len(train_idx)}, val={len(val_idx)}")
    print(f"Tau values: {tau_values}")
    print(f"Seeds per tau: {n_seeds}")
    print()

    results = []

    for tau in tau_values:
        print(f"tau={tau}:")
        for seed in range(n_seeds):
            result = run_single_seed(X, Y, train_idx, val_idx, tau, seed, k, ridge)
            results.append(result)
            omega_str = ", ".join([f"{w:.3f}" for w in result["omega"]])
            print(f"  seed={seed}: val_NLL={result['val_nll']:.4f}, ω=[{omega_str}]")
        print()

    # Convert to DataFrame
    rows = []
    for r in results:
        row = {
            "tau": r["tau"],
            "seed": r["seed"],
            "val_nll": r["val_nll"],
            "train_nll": r["train_nll"],
            "n_iters": r["n_iters"],
            "converged": r["converged"],
        }
        for i, w in enumerate(r["omega"]):
            row[f"omega_{i}"] = w
        rows.append(row)

    return pd.DataFrame(rows), x_cols


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-tau statistics across seeds."""
    stats = (
        df.groupby("tau")
        .agg(
            {
                "val_nll": ["mean", "std", "min", "max"],
                "train_nll": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    stats.columns = [
        "tau",
        "val_nll_mean",
        "val_nll_std",
        "val_nll_min",
        "val_nll_max",
        "train_nll_mean",
        "train_nll_std",
    ]

    # Add omega statistics
    omega_cols = [c for c in df.columns if c.startswith("omega_")]
    for col in omega_cols:
        omega_stats = df.groupby("tau")[col].agg(["mean", "std"]).reset_index()
        stats[f"{col}_mean"] = omega_stats["mean"]
        stats[f"{col}_std"] = omega_stats["std"]

    return stats


def plot_nll_variability(df: pd.DataFrame, stats: pd.DataFrame, output_dir: Path):
    """Plot NLL with error bars showing seed variability."""
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH * 1.5, 3))

    tau_values = stats["tau"].values
    nll_mean = stats["val_nll_mean"].values
    nll_std = stats["val_nll_std"].values
    nll_min = stats["val_nll_min"].values
    nll_max = stats["val_nll_max"].values

    # Plot mean with error bars (std)
    ax.errorbar(
        tau_values,
        nll_mean,
        yerr=nll_std,
        fmt="o-",
        linewidth=2,
        markersize=8,
        color=COLORS["learned"],
        capsize=4,
        capthick=1.5,
        label="Mean ± 1 std",
    )

    # Also show min/max as shaded region
    ax.fill_between(
        tau_values,
        nll_min,
        nll_max,
        alpha=0.2,
        color=COLORS["learned"],
        label="Min-Max range",
    )

    # Mark the best mean
    best_idx = np.argmin(nll_mean)
    ax.scatter(
        [tau_values[best_idx]],
        [nll_mean[best_idx]],
        s=200,
        c=COLORS["learned"],
        marker="*",
        zorder=10,
        edgecolors="black",
        linewidths=1.5,
        label=f"Best mean: τ={tau_values[best_idx]}",
    )

    ax.set_xlabel("τ (Kernel Bandwidth)")
    ax.set_ylabel("Validation NLL")
    ax.set_title("NLL Variability Across Random Seeds")
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "nll_variability.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "nll_variability.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_omega_variability(
    df: pd.DataFrame, stats: pd.DataFrame, x_cols: list[str], output_dir: Path
):
    """Plot omega values across seeds for each tau."""
    omega_cols = [c for c in df.columns if c.startswith("omega_")]
    n_omega = len(omega_cols)

    fig, axes = plt.subplots(1, n_omega, figsize=(4 * n_omega, 3))
    if n_omega == 1:
        axes = [axes]

    tau_values = sorted(df["tau"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tau_values)))

    for ax_idx, omega_col in enumerate(omega_cols):
        ax = axes[ax_idx]

        for tau_idx, tau in enumerate(tau_values):
            df_tau = df[df["tau"] == tau]
            omega_values = df_tau[omega_col].values

            # Jitter x position
            x_pos = np.full_like(omega_values, tau_idx) + np.random.uniform(
                -0.2, 0.2, len(omega_values)
            )
            ax.scatter(x_pos, omega_values, c=[colors[tau_idx]], alpha=0.6, s=30)

        # Add mean line
        omega_means = [df[df["tau"] == tau][omega_col].mean() for tau in tau_values]
        ax.plot(
            range(len(tau_values)),
            omega_means,
            "k-",
            linewidth=2,
            marker="s",
            markersize=6,
        )

        ax.set_xticks(range(len(tau_values)))
        ax.set_xticklabels([f"{t}" for t in tau_values], rotation=45, ha="right")
        ax.set_xlabel("τ")
        feature_name = x_cols[ax_idx] if ax_idx < len(x_cols) else f"ω_{ax_idx}"
        ax.set_ylabel(f"ω ({feature_name})")
        ax.set_title(f"ω[{ax_idx}] by τ")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "omega_variability.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_seed_scatter(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: each seed as a point, colored by tau."""
    omega_cols = [c for c in df.columns if c.startswith("omega_")]

    if len(omega_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))

        tau_values = sorted(df["tau"].unique())
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tau_values)))

        for tau_idx, tau in enumerate(tau_values):
            df_tau = df[df["tau"] == tau]
            ax.scatter(
                df_tau[omega_cols[0]],
                df_tau[omega_cols[1]],
                c=[colors[tau_idx]],
                alpha=0.7,
                s=50,
                label=f"τ={tau}",
            )

        ax.set_xlabel("ω[0]")
        ax.set_ylabel("ω[1]")
        ax.set_title("Omega Solutions by Seed (colored by τ)")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / "omega_seed_scatter.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def plot_convergence_check(df: pd.DataFrame, output_dir: Path):
    """Check if all runs converged."""
    convergence_rate = df.groupby("tau")["converged"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

    ax.bar(
        range(len(convergence_rate)),
        convergence_rate["converged"] * 100,
        color=COLORS["knn"],
        edgecolor="black",
    )
    ax.set_xticks(range(len(convergence_rate)))
    ax.set_xticklabels(
        [f"{t}" for t in convergence_rate["tau"]], rotation=45, ha="right"
    )
    ax.set_xlabel("τ")
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Optimization Convergence by τ")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="green", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "convergence_rate.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def run_diagnosis(
    feature_set: str = "focused_2d",
    tau_values: list[float] = None,
    n_seeds: int = 10,
    actual_col: str = "ACTUAL",
    actuals_parquet: Path = None,
):
    """Run full multi-seed diagnosis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TAU-OMEGA MULTI-SEED DIAGNOSIS")
    print("=" * 80)

    # Run sweep
    df, x_cols = run_multi_seed_sweep(
        feature_set, tau_values, n_seeds,
        actual_col=actual_col, actuals_parquet=actuals_parquet,
    )

    # Compute statistics
    stats = compute_statistics(df)

    # Save raw results
    df.to_csv(OUTPUT_DIR / "multi_seed_results.csv", index=False)
    stats.to_csv(OUTPUT_DIR / "multi_seed_stats.csv", index=False)
    print(f"\nSaved results to {OUTPUT_DIR}/")

    # Generate plots
    print("\nGenerating plots...")
    plot_nll_variability(df, stats, OUTPUT_DIR)
    plot_omega_variability(df, stats, x_cols, OUTPUT_DIR)
    plot_seed_scatter(df, OUTPUT_DIR)
    plot_convergence_check(df, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nPer-tau statistics:")
    print(
        stats[
            ["tau", "val_nll_mean", "val_nll_std", "val_nll_min", "val_nll_max"]
        ].to_string(index=False)
    )

    # Identify problematic tau values (high std relative to improvement)
    stats["cv"] = (
        stats["val_nll_std"] / stats["val_nll_mean"]
    )  # Coefficient of variation
    high_variance_tau = stats[stats["val_nll_std"] > 0.01]["tau"].tolist()

    if high_variance_tau:
        print(f"\n⚠️  High variance tau values (std > 0.01): {high_variance_tau}")
        print("   These may be prone to local minima.")
    else:
        print("\n✓ All tau values show consistent results across seeds.")

    # Best tau
    best_idx = stats["val_nll_mean"].idxmin()
    best_tau = stats.loc[best_idx, "tau"]
    best_nll = stats.loc[best_idx, "val_nll_mean"]
    best_std = stats.loc[best_idx, "val_nll_std"]
    print(f"\nBest tau: {best_tau} (mean NLL = {best_nll:.4f} ± {best_std:.4f})")

    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  - multi_seed_results.csv (raw results)")
    print("  - multi_seed_stats.csv (per-tau statistics)")
    print("  - nll_variability.png/pdf")
    print("  - omega_variability.png")
    print("  - omega_seed_scatter.png")
    print("  - convergence_rate.png")

    return df, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-seed diagnosis for tau-omega optimization"
    )
    parser.add_argument(
        "--feature-set", type=str, default="focused_2d", help="Feature set to use"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=3, help="Number of random seeds per tau"
    )
    parser.add_argument(
        "--tau-values",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Tau values to test",
    )
    parser.add_argument(
        "--use-residuals",
        action="store_true",
        help="Use residuals (ACTUAL - MEAN_FORECAST) instead of raw actuals",
    )

    args = parser.parse_args()

    if args.use_residuals:
        actual_col = "RESIDUAL"
        actuals_pq = DATA_DIR / "residuals_filtered_rts3_constellation_v1.parquet"
        OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "tau_omega_diagnosis_residuals"
    else:
        actual_col = "ACTUAL"
        actuals_pq = None

    run_diagnosis(
        feature_set=args.feature_set,
        tau_values=args.tau_values,
        n_seeds=args.n_seeds,
        actual_col=actual_col,
        actuals_parquet=actuals_pq,
    )
