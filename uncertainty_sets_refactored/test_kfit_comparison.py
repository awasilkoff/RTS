"""
Quick comparison: k_fit=128 vs k_fit=None for focused_2d dataset.

Tests whether aligning training neighborhood with prediction k improves NLL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from covariance_optimization import (
    fit_omega,
    predict_mu_sigma_topk_cross,
    KernelCovConfig,
    FitConfig,
    CovPredictConfig,
)
from data_processing_extended import build_XY_focused_2d
from utils import fit_scaler, apply_scaler


def _mean_gaussian_nll(Y, Mu, Sigma):
    """Mean NLL of Y under N(Mu, Sigma)."""
    n = Y.shape[0]
    m = Y.shape[1]
    total = 0.0
    for i in range(n):
        diff = Y[i] - Mu[i]
        sign, logdet = np.linalg.slogdet(Sigma[i])
        if sign <= 0:
            logdet = 30.0  # Large penalty for non-PD
        quad = diff @ np.linalg.solve(Sigma[i], diff)
        total += 0.5 * (m * np.log(2 * np.pi) + logdet + quad)
    return total / n


def run_comparison():
    """Compare k_fit=128 vs k_fit=None, and also try aligning prediction to training."""
    print("=" * 70)
    print("k_fit Comparison: Training vs Prediction Alignment")
    print("=" * 70)

    # Load data
    actuals_path = Path("data/actuals_filtered_rts3_constellation_v1.parquet")
    forecasts_path = Path("data/forecasts_filtered_rts3_constellation_v1.parquet")

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)

    X_raw, Y, times, x_cols, y_cols = build_XY_focused_2d(
        forecasts, actuals, actual_col="ACTUAL", drop_any_nan_rows=True
    )

    print(f"\nDataset: focused_2d")
    print(f"  X shape: {X_raw.shape}, Y shape: {Y.shape}")
    print(f"  Features: {x_cols}")

    # Train/val split
    n = X_raw.shape[0]
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])

    print(f"  Train: {n_train}, Val: {n_val}")

    # Scale
    scaler = fit_scaler(X_raw[train_idx], "standard")
    X = apply_scaler(X_raw, scaler)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    # Common config
    tau = 2.0
    k_predict = 128
    ridge = 1e-3
    max_iters = 200

    cfg = KernelCovConfig(tau=tau, ridge=ridge, zero_mean=False)
    pred_cfg = CovPredictConfig(tau=tau, ridge=ridge, dtype="float32", device="cpu")

    print(f"\nConfig: tau={tau}, k_predict={k_predict}, max_iters={max_iters}")
    print("-" * 70)

    results = {}

    # Test 1: k_fit=None (original behavior - all neighbors during training)
    print("\n[1] Training with k_fit=None (original: softmax over all N-1 neighbors)...")
    fit_cfg_none = FitConfig(
        max_iters=max_iters,
        step_size=0.1,
        grad_clip=10.0,
        tol=1e-7,
        verbose_every=999999,
        dtype="float32",
        omega_constraint="softmax",
        k_fit=None,  # Original behavior
    )

    omega_none = fit_omega(
        X, Y, omega0=np.ones(X.shape[1]), train_idx=train_idx,
        cfg=cfg, fit_cfg=fit_cfg_none
    )

    Mu_none, Sigma_none = predict_mu_sigma_topk_cross(
        X_val, X_train, Y_train, omega_none, pred_cfg, k=k_predict
    )
    nll_none = _mean_gaussian_nll(Y_val, Mu_none, Sigma_none)

    print(f"  Omega: {omega_none}")
    print(f"  Val NLL: {nll_none:.4f}")
    results["k_fit=None"] = {"omega": omega_none, "nll": nll_none}

    # Test 2: k_fit=128 (aligned with prediction)
    print(f"\n[2] Training with k_fit={k_predict} (aligned: softmax over top-k only)...")
    fit_cfg_aligned = FitConfig(
        max_iters=max_iters,
        step_size=0.1,
        grad_clip=10.0,
        tol=1e-7,
        verbose_every=999999,
        dtype="float32",
        omega_constraint="softmax",
        k_fit=k_predict,  # Aligned with prediction
    )

    omega_aligned = fit_omega(
        X, Y, omega0=np.ones(X.shape[1]), train_idx=train_idx,
        cfg=cfg, fit_cfg=fit_cfg_aligned
    )

    Mu_aligned, Sigma_aligned = predict_mu_sigma_topk_cross(
        X_val, X_train, Y_train, omega_aligned, pred_cfg, k=k_predict
    )
    nll_aligned = _mean_gaussian_nll(Y_val, Mu_aligned, Sigma_aligned)

    print(f"  Omega: {omega_aligned}")
    print(f"  Val NLL: {nll_aligned:.4f}")
    results["k_fit=128"] = {"omega": omega_aligned, "nll": nll_aligned}

    # Test 3: k_fit=64 (smaller neighborhood)
    k_small = 64
    print(f"\n[3] Training with k_fit={k_small} (smaller neighborhood)...")
    fit_cfg_small = FitConfig(
        max_iters=max_iters,
        step_size=0.1,
        grad_clip=10.0,
        tol=1e-7,
        verbose_every=999999,
        dtype="float32",
        omega_constraint="softmax",
        k_fit=k_small,
    )

    omega_small = fit_omega(
        X, Y, omega0=np.ones(X.shape[1]), train_idx=train_idx,
        cfg=cfg, fit_cfg=fit_cfg_small
    )

    # Still predict with k=128
    Mu_small, Sigma_small = predict_mu_sigma_topk_cross(
        X_val, X_train, Y_train, omega_small, pred_cfg, k=k_predict
    )
    nll_small = _mean_gaussian_nll(Y_val, Mu_small, Sigma_small)

    print(f"  Omega: {omega_small}")
    print(f"  Val NLL: {nll_small:.4f}")
    results[f"k_fit={k_small}"] = {"omega": omega_small, "nll": nll_small}

    # Test 4: k_fit=None but predict with ALL neighbors (align prediction to training)
    k_all = len(train_idx)  # Use all training points
    print(f"\n[4] k_fit=None + predict with k={k_all} (align prediction to training)...")

    # Reuse omega_none from test 1
    Mu_all, Sigma_all = predict_mu_sigma_topk_cross(
        X_val, X_train, Y_train, omega_none, pred_cfg, k=k_all
    )
    nll_all = _mean_gaussian_nll(Y_val, Mu_all, Sigma_all)

    print(f"  Omega: {omega_none} (same as test 1)")
    print(f"  Val NLL: {nll_all:.4f}")
    results[f"k_fit=None, k_pred={k_all}"] = {"omega": omega_none, "nll": nll_all}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'NLL':>10} {'vs baseline':>12}")
    print("-" * 57)

    nll_baseline = results["k_fit=None"]["nll"]
    for name, data in sorted(results.items(), key=lambda x: x[1]["nll"]):
        diff = data["nll"] - nll_baseline
        diff_str = f"{diff:+.4f}" if name != "k_fit=None" else "baseline"
        print(f"{name:<35} {data['nll']:>10.4f} {diff_str:>12}")

    print("-" * 57)

    # Find best
    best_name = min(results, key=lambda x: results[x]["nll"])
    print(f"\nBest: {best_name}")
    print(f"  NLL: {results[best_name]['nll']:.4f}")
    print(f"  Omega: {results[best_name]['omega']}")

    return results


if __name__ == "__main__":
    results = run_comparison()
