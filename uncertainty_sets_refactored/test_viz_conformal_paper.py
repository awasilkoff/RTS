#!/usr/bin/env python3
"""
Unit tests for viz_conformal_paper.py

Tests the core visualization functions with synthetic data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from viz_conformal_paper import (
    _wilson_score_interval,
    _compute_coverage_by_bin,
    plot_calibration_curve,
)


def test_wilson_score_interval():
    """Test Wilson score confidence interval computation."""
    print("Testing Wilson score interval...")

    # Perfect coverage
    lower, upper = _wilson_score_interval(0.95, 100, confidence=0.95)
    print(f"  Coverage=0.95, n=100: [{lower:.3f}, {upper:.3f}]")
    assert 0.85 < lower < 0.95 < upper < 1.0, "CI should contain true value"

    # Edge cases
    lower, upper = _wilson_score_interval(0.0, 100)
    print(f"  Coverage=0.00, n=100: [{lower:.3f}, {upper:.3f}]")
    assert lower >= 0.0, "Lower bound should be >= 0"
    assert upper < 0.1, "Upper bound should be small for 0% coverage"

    lower, upper = _wilson_score_interval(1.0, 100)
    print(f"  Coverage=1.00, n=100: [{lower:.3f}, {upper:.3f}]")
    assert upper <= 1.0, "Upper bound should be <= 1"
    assert lower > 0.9, "Lower bound should be large for 100% coverage"

    # Zero sample size
    lower, upper = _wilson_score_interval(0.5, 0)
    print(f"  Coverage=0.50, n=0: [{lower:.3f}, {upper:.3f}]")
    assert lower == 0.0 and upper == 1.0, "Zero n should return [0, 1]"

    print("  ✓ All Wilson score tests passed\n")


def test_coverage_by_bin():
    """Test coverage computation by bin."""
    print("Testing coverage by bin...")

    # Create synthetic data
    np.random.seed(42)
    n = 200

    df = pd.DataFrame(
        {
            "y": np.random.randn(n) + 100,
            "y_pred_conf": np.random.randn(n) + 98,  # Slightly lower
        }
    )

    # Create bins
    df["bin"] = pd.cut(df["y"], bins=[0, 50, 100, 150, 200], include_lowest=True)

    coverage_df = _compute_coverage_by_bin(df)
    print(f"  Computed coverage for {len(coverage_df)} bins")
    print(coverage_df)

    assert len(coverage_df) > 0, "Should compute coverage for at least one bin"
    assert all(0 <= cov <= 1 for cov in coverage_df["coverage"]), "Coverage in [0, 1]"
    assert all(n > 0 for n in coverage_df["n_samples"]), "All bins should have samples"

    print("  ✓ Coverage by bin tests passed\n")


def test_plot_calibration_curve():
    """Test calibration curve plotting with synthetic results."""
    print("Testing calibration curve plot...")

    # Synthetic calibration results
    results = [
        {"alpha_target": 0.80, "coverage": 0.795, "n_test": 100},
        {"alpha_target": 0.85, "coverage": 0.855, "n_test": 100},
        {"alpha_target": 0.90, "coverage": 0.898, "n_test": 100},
        {"alpha_target": 0.95, "coverage": 0.948, "n_test": 100},
        {"alpha_target": 0.99, "coverage": 0.992, "n_test": 100},
    ]

    # Create figure (no save)
    fig = plot_calibration_curve(results, output_path=None, figsize=(6, 5))

    assert fig is not None, "Figure should be created"
    assert len(fig.axes) == 1, "Should have one axis"

    plt.close(fig)
    print("  ✓ Calibration curve plot test passed\n")


def test_empty_results():
    """Test error handling for empty results."""
    print("Testing error handling...")

    try:
        plot_calibration_curve([], output_path=None)
        assert False, "Should raise error for empty results"
    except ValueError as e:
        print(f"  ✓ Correctly raised error: {e}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING viz_conformal_paper.py")
    print("=" * 80 + "\n")

    test_wilson_score_interval()
    test_coverage_by_bin()
    test_plot_calibration_curve()
    test_empty_results()

    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nReady to generate paper figures with:")
    print("  python run_paper_figures.py")
