#!/usr/bin/env python
"""
Diagnose numerical issues in kernel distance with equal omega.

Investigates why predict_mu_sigma_topk_cross() with omega=[1,1]
produces extreme NLL outliers (up to 7 million).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

from data_processing_extended import FEATURE_BUILDERS
from utils import fit_scaler, apply_scaler
from covariance_optimization import (
    CovPredictConfig,
    predict_mu_sigma_topk_cross,
    predict_mu_sigma_knn,
)


def compute_kernel_weights_manual(X_query, X_ref, omega, tau, k=128):
    """Manually compute kernel weights to understand distribution."""
    # Compute distances
    diff = X_query[:, np.newaxis, :] - X_ref[np.newaxis, :, :]
    dist_sq = np.sum(omega[np.newaxis, :] * diff * diff, axis=2)

    # Find k nearest
    idx = np.argpartition(dist_sq, k, axis=1)[:, :k]

    # Get distances to k nearest
    d_k = np.take_along_axis(dist_sq, idx, axis=1)

    # Softmax weights
    logits = -d_k / tau
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return weights, d_k, idx


def main():
    print("=" * 80)
    print("DIAGNOSING KERNEL EQUAL OMEGA OUTLIERS")
    print("=" * 80)
    print()

    # Load config
    config_path = Path("data/viz_artifacts/focused_2d/feature_config.json")
    best_config_path = Path("data/viz_artifacts/focused_2d/best_config_summary.json")

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please run experiments first to generate configuration files.")
        return

    with open(config_path) as f:
        config = json.load(f)
    with open(best_config_path) as f:
        best_config = json.load(f)

    feature_set = config["feature_set"]
    scaler_type = config["scaler_type"]
    tau = best_config["best_config"]["tau"]
    omega_learned = np.array(best_config["best_omega"])
    omega_equal = np.ones_like(omega_learned)
    k = 128
    ridge = 1e-3

    # Load data
    data_dir = Path("data")
    actuals = pd.read_parquet(data_dir / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(data_dir / "forecasts_filtered_rts3_constellation_v1.parquet")

    build_fn = FEATURE_BUILDERS[feature_set]
    X_raw, Y, times, x_cols, y_cols = build_fn(forecasts, actuals, drop_any_nan_rows=True)

    scaler = fit_scaler(X_raw, scaler_type)
    X = apply_scaler(X_raw, scaler)

    # Train/eval split
    n = X.shape[0]
    train_frac = 0.75
    n_train = int(train_frac * n)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = np.sort(indices[:n_train])
    eval_idx = np.sort(indices[n_train:])

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_eval, Y_eval = X[eval_idx], Y[eval_idx]

    print(f"Data: {X.shape[0]} points, {X.shape[1]} features")
    print(f"Train: {len(train_idx)}, Eval: {len(eval_idx)}")
    print()

    # Compute kernel weights for first few eval points
    print("Computing kernel weights for eval set...")
    weights, distances, neighbor_idx = compute_kernel_weights_manual(
        X_eval[:10], X_train, omega_equal, tau, k=k
    )

    print()
    print("=" * 80)
    print("KERNEL WEIGHT STATISTICS (First 10 eval points)")
    print("=" * 80)
    print()

    for i in range(10):
        w = weights[i]
        d = distances[i]

        print(f"Eval point {i}:")
        print(f"  Distance range: [{d.min():.3f}, {d.max():.3f}]")
        print(f"  Weight range:   [{w.min():.6f}, {w.max():.6f}]")
        print(f"  Max weight:     {w.max():.6f}")
        print(f"  Entropy:        {-np.sum(w * np.log(w + 1e-10)):.3f} (uniform = {np.log(k):.3f})")
        print(f"  Effective k:    {1.0 / (w**2).sum():.1f} / {k}")
        print()

    # Compute predictions with both methods
    print("=" * 80)
    print("COMPARING PREDICTION METHODS")
    print("=" * 80)
    print()

    # Kernel with equal omega (problematic)
    pred_cfg = CovPredictConfig(
        tau=float(tau),
        ridge=float(ridge),
        enforce_nonneg_omega=True,
        dtype="float32",
        device="cpu",
    )

    Mu_kernel, Sigma_kernel = predict_mu_sigma_topk_cross(
        X_query=X_eval[:100],
        X_ref=X_train,
        Y_ref=Y_train,
        omega=omega_equal,
        cfg=pred_cfg,
        k=k,
    )

    # True k-NN (uniform weights)
    Mu_knn, Sigma_knn = predict_mu_sigma_knn(
        X_query=X_eval[:100],
        X_ref=X_train,
        Y_ref=Y_train,
        k=k,
        ridge=ridge,
    )

    # Compare covariance condition numbers
    print("Covariance matrix condition numbers (first 10 points):")
    print()
    print("Point | Kernel(ω=1) cond | Euclidean k-NN cond | Ratio")
    print("-" * 60)
    for i in range(10):
        cond_kernel = np.linalg.cond(Sigma_kernel[i])
        cond_knn = np.linalg.cond(Sigma_knn[i])
        ratio = cond_kernel / cond_knn
        print(f"{i:4d}  | {cond_kernel:15.2e} | {cond_knn:15.2e} | {ratio:8.2f}x")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("1. Use predict_mu_sigma_knn() for true uniform k-NN baseline")
    print("2. Keep predict_mu_sigma_topk_cross(omega=[1,1]) as 'Kernel(ω=1)' baseline")
    print("3. For Kernel(ω=1), use higher ridge (1e-2) and float64 for stability")
    print()


if __name__ == "__main__":
    main()
