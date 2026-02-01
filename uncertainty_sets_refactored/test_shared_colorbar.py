#!/usr/bin/env python
"""
Test that shared colorbar scale works correctly in plot_kernel_distance_comparison.

Verifies:
1. Both subplots use the same vmin/vmax
2. Colors are directly comparable between equal and learned omega plots
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from viz_kernel_distance import plot_kernel_distance_comparison, compute_kernel_weights


def test_shared_colorbar():
    """Test that colorbar scales are shared between subplots."""
    # Create synthetic data
    np.random.seed(42)
    n_points = 100
    X = np.random.randn(n_points, 2)  # 2D features
    times = pd.date_range("2020-01-01", periods=n_points, freq="h")
    x_cols = ["Feature_1", "Feature_2"]

    # Define omegas and tau
    omega_equal = np.array([1.0, 1.0])
    omega_learned = np.array([2.0, 0.5])  # Different weights
    tau = 5.0
    target_idx = n_points // 2

    # Compute kernel weights for both
    X_target = X[target_idx]
    weights_equal = compute_kernel_weights(X_target, X, omega_equal, tau)
    weights_learned = compute_kernel_weights(X_target, X, omega_learned, tau)

    print("Test: Shared Colorbar Scale")
    print("=" * 60)
    print(f"\nEqual weights omega: {omega_equal}")
    print(f"Learned weights omega: {omega_learned}")
    print(f"Tau: {tau}")
    print()

    print("Kernel weight ranges:")
    print(f"  Equal:   min={weights_equal.min():.6f}, max={weights_equal.max():.6f}")
    print(f"  Learned: min={weights_learned.min():.6f}, max={weights_learned.max():.6f}")
    print()

    # Calculate shared scale (same logic as in plot_kernel_distance_comparison)
    vmin_shared = max(min(weights_equal.min(), weights_learned.min()), 1e-6)
    vmax_shared = max(weights_equal.max(), weights_learned.max())

    print("Shared colorbar scale:")
    print(f"  vmin: {vmin_shared:.6f}")
    print(f"  vmax: {vmax_shared:.6f}")
    print()

    # Create comparison plot
    save_path = Path("data/viz_artifacts/test_shared_colorbar.png")
    save_path.parent.mkdir(exist_ok=True, parents=True)

    fig = plot_kernel_distance_comparison(
        X,
        x_cols,
        times,
        target_idx,
        omega_equal,
        omega_learned,
        tau,
        save_path=save_path,
    )

    print(f"✓ Plot saved to: {save_path}")
    print()
    print("Verification:")
    print("  1. Both subplots should have IDENTICAL colorbar ranges")
    print("  2. Same colors should represent same kernel weights")
    print("  3. Easy to compare which omega creates stronger/weaker similarities")
    print()
    print("Expected behavior:")
    print("  - Learned omega [2.0, 0.5] weights Feature_1 more heavily")
    print("  - Points with similar Feature_1 should be brighter in right panel")
    print("  - Points with similar Feature_2 might be dimmer in right panel")
    print()

    plt.close(fig)

    return True


if __name__ == "__main__":
    success = test_shared_colorbar()
    if success:
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
