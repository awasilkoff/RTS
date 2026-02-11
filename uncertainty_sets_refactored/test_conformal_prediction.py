#!/usr/bin/env python
"""
Smoke test for conformal prediction module.

Tests:
1. Import check
2. Data loading
3. Model training (small sample)
4. Prediction on new data
5. Coverage verification

Run: python test_conformal_prediction.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
from conformal_prediction import (
    train_wind_lower_model_conformal_binned,
    ConformalLowerBundle,
)
from data_processing import build_conformal_totals_df


def test_imports():
    """Test 1: Check imports."""
    print("✓ Test 1: Imports successful")


def test_data_loading():
    """Test 2: Load and validate data."""
    DATA_DIR = Path(__file__).parent / "data"

    actuals = pd.read_parquet(
        DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
    )

    df = build_conformal_totals_df(actuals, forecasts)

    assert len(df) > 0, "Empty dataframe"
    assert "y" in df.columns, "Missing target column"
    assert "ens_mean" in df.columns, "Missing feature column"

    print(f"✓ Test 2: Data loaded ({len(df)} rows, {df['TIME_HOURLY'].min()} to {df['TIME_HOURLY'].max()})")
    return df


def test_model_training(df):
    """Test 3: Train conformal model."""
    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Use small subset for speed
    df_sample = df.head(300).copy()

    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df_sample,
        feature_cols=feature_cols,
        target_col="y",
        scale_col="ens_std",
        alpha_target=0.90,
        quantile_alpha=0.10,
        binning="y_pred",
        test_frac=0.2,
        cal_frac=0.2,
        model_kwargs={"n_estimators": 50, "verbose": -1},  # Faster for testing
    )

    assert isinstance(bundle, ConformalLowerBundle), "Invalid bundle type"
    assert "coverage" in metrics, "Missing coverage metric"
    assert len(df_test) > 0, "Empty test set"

    print(f"✓ Test 3: Model trained")
    print(f"    Coverage: {metrics['coverage']:.2%} (target: 90%)")
    print(f"    Global q_hat: {metrics['q_hat_global_r']:.4f}")
    print(f"    RMSE: {metrics['rmse']:.2f} MW")
    print(f"    MAE: {metrics['mae']:.2f} MW")
    print(f"    Splits: train={metrics['n_train']}, cal={metrics['n_cal']}, test={metrics['n_test']}")

    return bundle, metrics


def test_prediction(bundle):
    """Test 4: Predict on new data."""
    # Create synthetic test data
    df_new = pd.DataFrame({
        "ens_mean": [500.0, 800.0, 1200.0],
        "ens_std": [50.0, 100.0, 150.0],
        "ens_min": [400.0, 600.0, 900.0],
        "ens_max": [600.0, 1000.0, 1500.0],
        "n_models": [10, 10, 10],
        "hour": [12, 18, 6],
        "dow": [1, 3, 5],
    })

    df_pred = bundle.predict_df(df_new)

    required_cols = ["y_pred_base", "margin", "y_pred_conf", "q_hat_r", "bin"]
    for col in required_cols:
        assert col in df_pred.columns, f"Missing column: {col}"

    assert len(df_pred) == len(df_new), "Output length mismatch"
    assert (df_pred["y_pred_conf"] <= df_pred["y_pred_base"]).all(), "Conformal bound should be <= base prediction"

    print("✓ Test 4: Prediction successful")
    print("\n    Sample predictions:")
    print(df_pred[["y_pred_base", "q_hat_r", "margin", "y_pred_conf"]].to_string(index=False))

    return df_pred


def test_coverage_property(df):
    """Test 5: Verify coverage on larger sample."""
    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Use more data for stable coverage estimate
    df_sample = df.head(1000).copy()

    alpha_targets = [0.80, 0.90, 0.95]
    results = []

    for alpha in alpha_targets:
        bundle, metrics, _ = train_wind_lower_model_conformal_binned(
            df_sample,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=alpha,
            quantile_alpha=1.0 - alpha,
            binning="y_pred",
            test_frac=0.2,
            cal_frac=0.2,
            model_kwargs={"n_estimators": 100, "verbose": -1},
        )

        coverage = metrics["coverage"]
        error = abs(coverage - alpha)

        results.append({
            "alpha_target": alpha,
            "coverage": coverage,
            "error": error,
            "status": "✓" if error < 0.10 else "⚠",
        })

    print("✓ Test 5: Coverage verification")
    print("\n    Alpha Target | Coverage | Error  | Status")
    print("    " + "-" * 45)
    for r in results:
        print(f"    {r['alpha_target']:12.2%} | {r['coverage']:8.2%} | {r['error']:6.2%} | {r['status']}")

    # Check if at least one config achieves good coverage
    good_coverage = any(r["error"] < 0.10 for r in results)
    assert good_coverage, "None of the configurations achieved target coverage"


def main():
    """Run all tests."""
    print("=" * 60)
    print("CONFORMAL PREDICTION SMOKE TEST")
    print("=" * 60)
    print()

    try:
        # Test 1: Imports
        test_imports()

        # Test 2: Data loading
        df = test_data_loading()

        # Test 3: Model training
        bundle, metrics = test_model_training(df)

        # Test 4: Prediction
        test_prediction(bundle)

        # Test 5: Coverage property
        print()
        test_coverage_property(df)

        print()
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Run full sweep: python viz_conformal_sweep.py")
        print("  2. Integrate with pipeline: python main.py")
        print("  3. Review documentation: CONFORMAL_PREDICTION_README.md")

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        raise
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
