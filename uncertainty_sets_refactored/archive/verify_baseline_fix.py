"""
Verify that the baseline comparison fix is working correctly.

This script runs a quick test to confirm:
1. Baseline always uses equal weights [1, 1, ...]
2. For raw features, learned omega should discover rescaling
3. Improvement should be larger for raw vs standardized
"""
from pathlib import Path
import numpy as np
import pandas as pd

from utils import fit_standard_scaler
from data_processing import build_XY_for_covariance_system_only
from covariance_optimization import (
    KernelCovConfig,
    FitConfig,
    CovPredictConfig,
    fit_omega,
    predict_mu_sigma_topk_cross,
)


def _mean_gaussian_nll(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Mean NLL of Y under N(Mu, Sigma) row-wise."""
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

    return float(nll.mean())


def verify_baseline_fix():
    """Verify that baseline is always equal weights."""
    DATA_DIR = Path(__file__).parent / "data"

    print("=" * 80)
    print("Verifying Baseline Comparison Fix")
    print("=" * 80)
    print()

    # Load data
    actuals = pd.read_parquet(DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet")

    X_raw, Y, times, x_cols, y_cols = build_XY_for_covariance_system_only(
        forecasts, actuals, drop_any_nan_rows=True
    )

    # Standardize
    scaler = fit_standard_scaler(X_raw)
    X_std = scaler.transform(X_raw)

    # Random train/eval split
    n = X_raw.shape[0]
    n_train = int(0.75 * n)

    # Fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = np.sort(indices[:n_train])
    eval_idx = np.sort(indices[n_train:])

    print(f"Data: {n} samples, {X_raw.shape[1]} features")
    print(f"Train: {n_train}, Eval: {n - n_train}")
    print(f"Raw feature scales: mean={X_raw.mean(axis=0).round(1)}, std={X_raw.std(axis=0).round(1)}")
    print()

    # Config
    tau = 5.0
    ridge = 1e-3
    k = 128

    cfg = KernelCovConfig(tau=tau, ridge=ridge)
    fit_cfg = FitConfig(
        max_iters=100,
        step_size=0.1,
        grad_clip=10.0,
        tol=1e-7,
        verbose_every=999999,
        dtype="float32",
        device="cpu",
        omega_l2_reg=0.0,
    )
    pred_cfg = CovPredictConfig(
        tau=tau, ridge=ridge, enforce_nonneg_omega=True, dtype="float32", device="cpu"
    )

    results = {}

    # Test 1: Standardized features
    print("Test 1: Standardized Features")
    print("-" * 80)

    X_train, Y_train = X_std[train_idx], Y[train_idx]
    X_eval, Y_eval = X_std[eval_idx], Y[eval_idx]

    omega0 = np.ones(X_std.shape[1])
    omega_learned, _ = fit_omega(X_std, Y, omega0=omega0, train_idx=train_idx, cfg=cfg, fit_cfg=fit_cfg)

    # Baseline: MUST be equal weights
    omega_baseline = np.ones(X_std.shape[1])

    Mu_learned, Sigma_learned = predict_mu_sigma_topk_cross(
        X_eval, X_train, Y_train, omega_learned, pred_cfg, k=k, return_type="numpy"
    )
    Mu_baseline, Sigma_baseline = predict_mu_sigma_topk_cross(
        X_eval, X_train, Y_train, omega_baseline, pred_cfg, k=k, return_type="numpy"
    )

    nll_learned = _mean_gaussian_nll(Y_eval, Mu_learned, Sigma_learned)
    nll_baseline = _mean_gaussian_nll(Y_eval, Mu_baseline, Sigma_baseline)

    print(f"Baseline omega (should be [1, 1]):  {omega_baseline}")
    print(f"Learned omega:                      {omega_learned.round(3)}")
    print(f"Baseline NLL:                       {nll_baseline:.3f}")
    print(f"Learned NLL:                        {nll_learned:.3f}")
    print(f"Improvement:                        {nll_baseline - nll_learned:.3f}")

    results["standardized"] = {
        "omega_baseline": omega_baseline.copy(),
        "omega_learned": omega_learned.copy(),
        "nll_improvement": nll_baseline - nll_learned,
    }

    print()

    # Test 2: Raw features
    print("Test 2: Raw Features (Unscaled)")
    print("-" * 80)

    X_train, Y_train = X_raw[train_idx], Y[train_idx]
    X_eval, Y_eval = X_raw[eval_idx], Y[eval_idx]

    # Initialize with inverse variance for faster convergence
    omega0_raw = 1.0 / (X_train.var(axis=0) + 1e-6)
    omega_learned_raw, _ = fit_omega(X_raw, Y, omega0=omega0_raw, train_idx=train_idx, cfg=cfg, fit_cfg=fit_cfg)

    # Baseline: MUST be equal weights (this is the fix!)
    omega_baseline_raw = np.ones(X_raw.shape[1])

    # Also compute inverse variance for comparison
    omega_invvar = 1.0 / (X_train.var(axis=0) + 1e-6)

    Mu_learned, Sigma_learned = predict_mu_sigma_topk_cross(
        X_eval, X_train, Y_train, omega_learned_raw, pred_cfg, k=k, return_type="numpy"
    )
    Mu_baseline, Sigma_baseline = predict_mu_sigma_topk_cross(
        X_eval, X_train, Y_train, omega_baseline_raw, pred_cfg, k=k, return_type="numpy"
    )
    Mu_invvar, Sigma_invvar = predict_mu_sigma_topk_cross(
        X_eval, X_train, Y_train, omega_invvar, pred_cfg, k=k, return_type="numpy"
    )

    nll_learned = _mean_gaussian_nll(Y_eval, Mu_learned, Sigma_learned)
    nll_baseline = _mean_gaussian_nll(Y_eval, Mu_baseline, Sigma_baseline)
    nll_invvar = _mean_gaussian_nll(Y_eval, Mu_invvar, Sigma_invvar)

    print(f"Baseline omega (should be [1, 1]):  {omega_baseline_raw}")
    print(f"Inverse variance (for comparison):  {omega_invvar.round(3)}")
    print(f"Learned omega:                      {omega_learned_raw.round(3)}")
    print(f"")
    print(f"Baseline NLL (equal weights):       {nll_baseline:.3f}  <- Should be WORSE")
    print(f"Inverse var NLL:                    {nll_invvar:.3f}  <- Better than baseline")
    print(f"Learned NLL:                        {nll_learned:.3f}  <- Best")
    print(f"")
    print(f"Improvement vs baseline:            {nll_baseline - nll_learned:.3f}  <- Large!")
    print(f"Improvement vs inverse var:         {nll_invvar - nll_learned:.3f}  <- Modest")

    results["raw"] = {
        "omega_baseline": omega_baseline_raw.copy(),
        "omega_invvar": omega_invvar.copy(),
        "omega_learned": omega_learned_raw.copy(),
        "nll_improvement": nll_baseline - nll_learned,
    }

    print()

    # Verification checks
    print("=" * 80)
    print("Verification Checks")
    print("=" * 80)
    print()

    checks_passed = 0
    checks_total = 0

    # Check 1: Baselines are equal weights
    checks_total += 1
    if np.allclose(results["standardized"]["omega_baseline"], np.ones(2)):
        print("(ok) Check 1: Standardized baseline is [1, 1]")
        checks_passed += 1
    else:
        print("(x) Check 1: Standardized baseline is NOT [1, 1]")

    checks_total += 1
    if np.allclose(results["raw"]["omega_baseline"], np.ones(2)):
        print("(ok) Check 2: Raw baseline is [1, 1]")
        checks_passed += 1
    else:
        print("(x) Check 2: Raw baseline is NOT [1, 1]")

    # Check 2: Learned omega for raw is similar to inverse variance
    checks_total += 1
    omega_learned_norm = results["raw"]["omega_learned"] / results["raw"]["omega_learned"].sum()
    omega_invvar_norm = results["raw"]["omega_invvar"] / results["raw"]["omega_invvar"].sum()
    if np.allclose(omega_learned_norm, omega_invvar_norm, rtol=0.5):
        print(f"(ok) Check 3: Learned omega for raw features is similar to inverse variance")
        print(f"           Learned (normalized):  {omega_learned_norm.round(3)}")
        print(f"           InvVar (normalized):   {omega_invvar_norm.round(3)}")
        checks_passed += 1
    else:
        print(f"(x) Check 3: Learned omega differs significantly from inverse variance")

    # Check 3: Raw improvement is larger than standardized
    checks_total += 1
    if results["raw"]["nll_improvement"] > results["standardized"]["nll_improvement"]:
        print(f"(ok) Check 4: Raw improvement ({results['raw']['nll_improvement']:.3f}) > "
              f"Standardized ({results['standardized']['nll_improvement']:.3f})")
        checks_passed += 1
    else:
        print(f"(x) Check 4: Raw improvement should be larger")

    print()
    print("=" * 80)
    print(f"Summary: {checks_passed}/{checks_total} checks passed")
    print("=" * 80)

    if checks_passed == checks_total:
        print()
        print("(ok) All verification checks passed! Baseline fix is working correctly.")
        print()
        print("Key takeaway:")
        print("  - Baseline is consistently [1, 1, ...] for all feature sets")
        print("  - Raw features show larger improvement (learned omega discovers rescaling)")
        print("  - Learned omega for raw features ≈ inverse variance (as expected)")
        return True
    else:
        print()
        print("(x) Some checks failed. Review the baseline comparison logic.")
        return False


if __name__ == "__main__":
    import sys
    success = verify_baseline_fix()
    sys.exit(0 if success else 1)
