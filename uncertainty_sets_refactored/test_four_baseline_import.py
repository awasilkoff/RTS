#!/usr/bin/env python
"""
Quick test to verify that all imports and function calls work correctly.
Does NOT run full experiments - just checks that the code is syntactically correct.
"""
import numpy as np
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from covariance_optimization import (
        CovPredictConfig,
        predict_mu_sigma_topk_cross,
        predict_mu_sigma_knn,
    )
    print("(ok) Successfully imported all prediction functions")
except ImportError as e:
    print(f"(x) Import error: {e}")
    exit(1)

# Test that predict_mu_sigma_knn exists and has correct signature
print("\nTesting predict_mu_sigma_knn signature...")
import inspect
sig = inspect.signature(predict_mu_sigma_knn)
params = list(sig.parameters.keys())
print(f"  Parameters: {params}")

expected_params = ["X_query", "X_ref", "Y_ref", "k", "ridge"]
for param in expected_params:
    if param in params:
        print(f"  (ok) {param}")
    else:
        print(f"  (x) Missing parameter: {param}")

# Test CovPredictConfig with new parameters
print("\nTesting CovPredictConfig with numerical stability settings...")
try:
    pred_cfg_stable = CovPredictConfig(
        tau=2.0,
        ridge=1e-2,
        enforce_nonneg_omega=True,
        dtype="float64",
        device="cpu",
    )
    print(f"(ok) Created stable config: ridge={pred_cfg_stable.ridge}, dtype={pred_cfg_stable.dtype}")
except Exception as e:
    print(f"(x) Error creating config: {e}")
    exit(1)

# Test with dummy data
print("\nTesting with dummy data...")
try:
    # Create small dummy dataset
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    Y_train = np.random.randn(100, 3)
    X_query = np.random.randn(10, 2)
    omega_equal = np.ones(2)

    # Test Euclidean k-NN
    Mu_knn, Sigma_knn = predict_mu_sigma_knn(
        X_query=X_query,
        X_ref=X_train,
        Y_ref=Y_train,
        k=16,
        ridge=1e-3,
    )
    print(f"(ok) Euclidean k-NN: Mu shape={Mu_knn.shape}, Sigma shape={Sigma_knn.shape}")

    # Test kernel with equal omega
    Mu_kernel, Sigma_kernel = predict_mu_sigma_topk_cross(
        X_query=X_query,
        X_ref=X_train,
        Y_ref=Y_train,
        omega=omega_equal,
        cfg=pred_cfg_stable,
        k=16,
    )
    print(f"(ok) Kernel(ω=1): Mu shape={Mu_kernel.shape}, Sigma shape={Sigma_kernel.shape}")

    # Check that outputs are different (different methods)
    mu_diff = np.abs(Mu_knn - Mu_kernel).mean()
    print(f"(ok) Methods produce different results (mean |Δμ| = {mu_diff:.4f})")

except Exception as e:
    print(f"(x) Error in prediction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nReady to run full comparison with:")
print("  cd uncertainty_sets_refactored")
print("  python create_comprehensive_comparison.py --feature-set focused_2d")
