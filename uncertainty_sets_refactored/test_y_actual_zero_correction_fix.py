#!/usr/bin/env python3
"""
Test that the fix for zero correction in high bins works correctly.

The problem: When using y_actual binning, the highest actual bin often has:
- All under-predictions (y_pred < y)
- Therefore all r = max(0, (y_pred - y)/scale) = 0
- Therefore q_hat = quantile([0, 0, 0, ...], 0.95) = 0
- Therefore NO correction applied -> coverage failure

The fix: Apply a minimum q_hat floor:
- q_hat_by_bin[b] = max(q_hat_bin, global_q_hat * min_q_hat_ratio)
- Default min_q_hat_ratio = 0.1 (10% of global correction minimum)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from conformal_prediction import train_wind_lower_model_conformal_binned


def create_synthetic_data_with_underprediction(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic data where model systematically UNDER-predicts at high generation.

    This simulates the real-world pattern causing zero corrections in high bins.
    """
    np.random.seed(seed)

    # Generate actuals with realistic distribution
    y = np.random.beta(2, 5, n) * 250  # Skewed toward lower generation

    # Generate predictions with SYSTEMATIC BIAS at high generation
    # Low generation: unbiased errors
    # High generation: systematic under-prediction
    error = np.where(
        y < 150,
        np.random.normal(0, 10, n),  # Unbiased at low generation
        np.random.normal(-30, 10, n),  # Systematic under-prediction at high generation
    )
    y_pred = y + error

    # Uncertainty estimate
    ens_std = 10 + 0.1 * y + np.random.normal(0, 2, n)
    ens_std = np.maximum(ens_std, 1.0)

    # Features
    ens_mean = y_pred
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


def test_zero_correction_without_fix():
    """Test that zero correction happens without the fix (min_q_hat_ratio=0)."""
    print("=" * 70)
    print("Test 1: Zero Correction Without Fix (min_q_hat_ratio=0.0)")
    print("=" * 70)

    df = create_synthetic_data_with_underprediction(n=1000)
    feature_cols = ["ens_mean", "ens_std"]

    # Train with min_q_hat_ratio=0.0 (no floor, should show the problem)
    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=feature_cols,
        binning="y_actual",
        n_bins=5,
        alpha_target=0.95,
        min_q_hat_ratio=0.0,  # DISABLE FIX
        safety_margin=0.0,
    )

    # Check q_hats by bin
    q_hats = list(bundle.q_hat_by_bin_r.values())
    print(f"\nq_hats per bin: {[f'{q:.3f}' for q in q_hats]}")
    print(f"Global q_hat: {bundle.q_hat_global_r:.3f}")

    # Check if any bin has zero or near-zero q_hat
    zero_bins = [i for i, q in enumerate(q_hats) if q < 0.01]
    if zero_bins:
        print(f"\n⚠️  WARNING: Bins with near-zero q_hat: {zero_bins}")
        print("   This will cause coverage failures in those bins!")
    else:
        print("\n(ok) No zero q_hats (problem may not manifest with this data)")

    # Check coverage by actual bin
    df_test["y_bin"] = pd.cut(df_test["y"], bins=5)
    coverage_by_bin = df_test.groupby("y_bin").apply(
        lambda g: (g["y"] >= g["y_pred_conf"]).mean()
    )

    print(f"\nCoverage by actual bin (without fix):")
    for bin_interval, cov in coverage_by_bin.items():
        status = "(ok)" if cov >= 0.90 else "(x)"
        print(f"  {status} {bin_interval}: {cov:.3f}")

    # Overall coverage
    overall_coverage = metrics["coverage"]
    print(f"\nOverall coverage: {overall_coverage:.3f} (target: 0.95)")

    return {
        "q_hats": q_hats,
        "zero_bins": zero_bins,
        "coverage_by_bin": coverage_by_bin,
        "overall_coverage": overall_coverage,
    }


def test_zero_correction_with_fix():
    """Test that zero correction is prevented with the fix (min_q_hat_ratio=0.1)."""
    print("\n" + "=" * 70)
    print("Test 2: Zero Correction With Fix (min_q_hat_ratio=0.1)")
    print("=" * 70)

    df = create_synthetic_data_with_underprediction(n=1000)
    feature_cols = ["ens_mean", "ens_std"]

    # Train with min_q_hat_ratio=0.1 (default, fix enabled)
    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=feature_cols,
        binning="y_actual",
        n_bins=5,
        alpha_target=0.95,
        min_q_hat_ratio=0.1,  # ENABLE FIX (default)
        safety_margin=0.0,
    )

    # Check q_hats by bin
    q_hats = list(bundle.q_hat_by_bin_r.values())
    print(f"\nq_hats per bin: {[f'{q:.3f}' for q in q_hats]}")
    print(f"Global q_hat: {bundle.q_hat_global_r:.3f}")
    print(f"Minimum floor: {bundle.q_hat_global_r * 0.1:.3f} (10% of global)")

    # Check if any bin has zero or near-zero q_hat
    zero_bins = [i for i, q in enumerate(q_hats) if q < 0.01]
    if zero_bins:
        print(f"\n(x) ERROR: Bins with near-zero q_hat: {zero_bins}")
        print("   Fix did not work!")
    else:
        print(f"\n(ok) All bins have q_hat ≥ minimum floor")
        print("   Fix is working!")

    # Check that all q_hats are at least the minimum floor
    min_floor = bundle.q_hat_global_r * 0.1
    below_floor = [i for i, q in enumerate(q_hats) if q < min_floor * 0.99]  # Allow small numerical error
    if below_floor:
        print(f"\n⚠️  WARNING: Bins below floor: {below_floor}")
    else:
        print(f"(ok) All bins at or above floor")

    # Check coverage by actual bin
    df_test["y_bin"] = pd.cut(df_test["y"], bins=5)
    coverage_by_bin = df_test.groupby("y_bin").apply(
        lambda g: (g["y"] >= g["y_pred_conf"]).mean()
    )

    print(f"\nCoverage by actual bin (with fix):")
    for bin_interval, cov in coverage_by_bin.items():
        status = "(ok)" if cov >= 0.90 else "(x)"
        print(f"  {status} {bin_interval}: {cov:.3f}")

    # Overall coverage
    overall_coverage = metrics["coverage"]
    print(f"\nOverall coverage: {overall_coverage:.3f} (target: 0.95)")

    return {
        "q_hats": q_hats,
        "zero_bins": zero_bins,
        "coverage_by_bin": coverage_by_bin,
        "overall_coverage": overall_coverage,
    }


def test_comparison():
    """Compare performance with and without fix."""
    print("\n" + "=" * 70)
    print("Test 3: Comparison (With vs Without Fix)")
    print("=" * 70)

    results_without = test_zero_correction_without_fix()
    results_with = test_zero_correction_with_fix()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nZero q_hat bins:")
    print(f"  Without fix: {len(results_without['zero_bins'])} bins")
    print(f"  With fix:    {len(results_with['zero_bins'])} bins")

    print(f"\nOverall coverage:")
    print(f"  Without fix: {results_without['overall_coverage']:.3f}")
    print(f"  With fix:    {results_with['overall_coverage']:.3f}")

    improvement = results_with['overall_coverage'] - results_without['overall_coverage']
    print(f"  Improvement: {improvement:+.3f}")

    # Check if fix improves coverage
    if results_with['overall_coverage'] >= results_without['overall_coverage']:
        print("\n(ok) Fix improves or maintains coverage")
    else:
        print("\n⚠️  Fix reduces coverage (unexpected)")

    # Check if fix eliminates zero bins
    if len(results_with['zero_bins']) < len(results_without['zero_bins']):
        print("(ok) Fix eliminates or reduces zero q_hat bins")
    else:
        print("⚠️  Fix does not eliminate zero q_hat bins (may not be present in data)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    # Final verdict
    if (len(results_with['zero_bins']) == 0 and
        results_with['overall_coverage'] >= 0.90):
        print("\n✅ FIX VERIFIED: No zero bins, coverage maintained")
        return 0
    else:
        print("\n⚠️  Need to investigate further")
        return 1


if __name__ == "__main__":
    exit(test_comparison())
