#!/usr/bin/env python3
"""
Unit tests for y_actual binning implementation.

Tests:
1. Basic functionality: y_actual option works without errors
2. Calibration bins are created from actual values
3. Prediction uses y_pred as proxy
4. Coverage is maintained (gap < 5%)
5. q_hats vary across bins (evidence of adaptation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from conformal_prediction import (
    train_wind_lower_model_conformal_binned,
    _extract_binning_feature,
)


def create_synthetic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic wind data with known error patterns.

    Error pattern: errors increase with actual generation level.
    - Low generation (0-50 MW): small errors
    - High generation (200-250 MW): large errors

    This tests if y_actual binning can capture heterogeneity.
    """
    np.random.seed(seed)

    # Generate actuals with realistic distribution
    y = np.random.beta(2, 5, n) * 250  # Skewed toward lower generation

    # Generate predictions with heteroscedastic errors
    # Error variance increases with y
    error_std = 5 + 0.2 * y  # Higher errors at high generation
    error = np.random.normal(0, error_std)
    y_pred = y + error

    # Uncertainty estimate (ensemble std)
    ens_std = error_std + np.random.normal(0, 2, n)
    ens_std = np.maximum(ens_std, 1.0)  # Minimum 1 MW

    # Simple features matching conformal pipeline expectations
    ens_mean = y_pred  # Ensemble mean is the prediction
    ens_min = ens_mean - ens_std * 1.5
    ens_max = ens_mean + ens_std * 1.5

    df = pd.DataFrame({
        "TIME_HOURLY": pd.date_range("2020-01-01", periods=n, freq="h"),
        "y": y,
        "ens_mean": ens_mean,
        "ens_std": ens_std,
        "ens_min": ens_min,
        "ens_max": ens_max,
    })

    return df


def test_basic_functionality():
    """Test that y_actual binning works without errors."""
    print("Test 1: Basic functionality...")

    df = create_synthetic_data(n=1000)
    feature_cols = ["ens_mean", "ens_std"]

    try:
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df,
            feature_cols=feature_cols,
            binning="y_actual",
            n_bins=3,
            alpha_target=0.90,
        )
        print("  ✓ Training completed successfully")
        print(f"    Coverage: {metrics['coverage']:.4f}")
        assert bundle.binning == "y_actual"
        print("  ✓ Bundle has correct binning attribute")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        raise

    print("  ✓ Test passed\n")


def test_calibration_uses_actuals():
    """Test that calibration bins are created from actual values, not predictions."""
    print("Test 2: Calibration bins from actuals...")

    df = create_synthetic_data(n=1000)
    feature_cols = ["ens_mean", "ens_std"]

    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=feature_cols,
        binning="y_actual",
        n_bins=3,
        alpha_target=0.90,
    )

    # Check that bin edges span the range of ACTUALS in calibration set
    n_cal = metrics["n_cal"]
    n_train = metrics["n_train"]
    y_cal = df["y"].iloc[n_train : n_train + n_cal].values

    bin_edges = bundle.bin_edges
    y_min, y_max = y_cal.min(), y_cal.max()

    # Bin edges should roughly span calibration actual range
    assert bin_edges[0] <= y_min, f"Lower edge {bin_edges[0]} > y_min {y_min}"
    assert bin_edges[-1] >= y_max, f"Upper edge {bin_edges[-1]} < y_max {y_max}"

    print(f"  ✓ Bin edges span calibration actual range: [{bin_edges[0]:.1f}, {bin_edges[-1]:.1f}]")
    print(f"    Calibration actuals range: [{y_min:.1f}, {y_max:.1f}]")
    print("  ✓ Test passed\n")


def test_prediction_uses_proxy():
    """Test that prediction uses y_pred as proxy for bin assignment."""
    print("Test 3: Prediction uses y_pred as proxy...")

    df = create_synthetic_data(n=1000)
    feature_cols = ["ens_mean", "ens_std"]

    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=feature_cols,
        binning="y_actual",
        n_bins=3,
        alpha_target=0.90,
    )

    # Simulate prediction on new data
    df_new = df.iloc[-50:].copy()
    df_pred = bundle.predict_df(df_new)

    # Check that bin_feature equals y_pred_base (proxy behavior)
    assert "bin_feature" in df_pred.columns
    assert "y_pred_base" in df_pred.columns
    assert np.allclose(df_pred["bin_feature"], df_pred["y_pred_base"])

    print("  ✓ Bin feature equals y_pred_base at prediction time")
    print("  ✓ Test passed\n")


def test_coverage_maintained():
    """Test that coverage is maintained (gap < 5%)."""
    print("Test 4: Coverage maintained...")

    df = create_synthetic_data(n=1000)
    feature_cols = ["ens_mean", "ens_std"]

    alpha_target = 0.90
    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=feature_cols,
        binning="y_actual",
        n_bins=3,
        alpha_target=alpha_target,
    )

    coverage = metrics["coverage"]
    gap = abs(coverage - alpha_target)

    print(f"  Target coverage: {alpha_target:.4f}")
    print(f"  Actual coverage: {coverage:.4f}")
    print(f"  Gap: {gap:.4f}")

    assert gap < 0.05, f"Coverage gap {gap:.4f} >= 0.05"
    print("  ✓ Coverage gap < 5%")
    print("  ✓ Test passed\n")


def test_q_hats_vary():
    """Test that q_hats vary across bins (evidence of adaptation)."""
    print("Test 5: q_hats vary across bins...")

    df = create_synthetic_data(n=1000)
    feature_cols = ["ens_mean", "ens_std"]

    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=feature_cols,
        binning="y_actual",
        n_bins=3,
        alpha_target=0.90,
    )

    q_hats = list(bundle.q_hat_by_bin_r.values())
    q_hat_std = np.std(q_hats)

    print(f"  q_hats: {[f'{q:.3f}' for q in q_hats]}")
    print(f"  Std dev: {q_hat_std:.3f}")

    # If q_hats are all identical (or very close), adaptation is not happening
    assert q_hat_std > 0.01, f"q_hats are too similar (std={q_hat_std:.3f})"

    print("  ✓ q_hats vary across bins (adaptive correction working)")
    print("  ✓ Test passed\n")


def test_extract_binning_feature():
    """Test _extract_binning_feature for y_actual case."""
    print("Test 6: _extract_binning_feature for y_actual...")

    df = pd.DataFrame({
        "ens_std": [10, 20, 30],
        "ens_mean": [100, 150, 200],
    })
    y_pred_base = np.array([50, 100, 150])

    # y_actual should return y_pred_base (proxy behavior)
    result = _extract_binning_feature(df, "y_actual", y_pred_base=y_pred_base)
    assert np.array_equal(result, y_pred_base)
    print("  ✓ y_actual returns y_pred_base (proxy behavior)")

    # Compare with y_pred (should be identical)
    result_y_pred = _extract_binning_feature(df, "y_pred", y_pred_base=y_pred_base)
    assert np.array_equal(result, result_y_pred)
    print("  ✓ y_actual and y_pred return same value at prediction time")

    print("  ✓ Test passed\n")


def test_comparison_with_y_pred():
    """Test that y_actual performs comparably to y_pred."""
    print("Test 7: Comparison with y_pred...")

    df = create_synthetic_data(n=1000)
    feature_cols = ["ens_mean", "ens_std"]
    alpha_target = 0.95

    results = {}
    for binning in ["y_pred", "y_actual"]:
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df,
            feature_cols=feature_cols,
            binning=binning,
            n_bins=5,
            alpha_target=alpha_target,
        )
        results[binning] = {
            "coverage": metrics["coverage"],
            "gap": abs(metrics["coverage"] - alpha_target),
        }

    print(f"  y_pred:   coverage={results['y_pred']['coverage']:.4f}, gap={results['y_pred']['gap']:.4f}")
    print(f"  y_actual: coverage={results['y_actual']['coverage']:.4f}, gap={results['y_actual']['gap']:.4f}")

    # Both should achieve reasonable coverage
    assert results["y_pred"]["coverage"] > 0.90
    assert results["y_actual"]["coverage"] > 0.90

    print("  ✓ Both strategies achieve coverage > 0.90")
    print("  ✓ Test passed\n")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("Running Unit Tests for y_actual Binning")
    print("="*70)
    print()

    test_basic_functionality()
    test_calibration_uses_actuals()
    test_prediction_uses_proxy()
    test_coverage_maintained()
    test_q_hats_vary()
    test_extract_binning_feature()
    test_comparison_with_y_pred()

    print("="*70)
    print("All Tests Passed! ✓")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
