#!/usr/bin/env python3
"""
Quick test to verify y_actual binning works with day-ahead valid script.

Tests that all three binning strategies work:
- y_pred (standard)
- ens_std (feature-based)
- y_actual (new - bin by actuals, proxy at prediction time)
"""

from pathlib import Path
import pandas as pd
from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df
from run_paper_figures_dayahead_valid import add_dayahead_valid_features


def test_binning_strategies():
    """Test that all three binning strategies work."""
    print("=" * 70)
    print("Testing Binning Strategies in Day-Ahead Valid Script")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    data_dir = Path(__file__).parent / "data"
    actuals = pd.read_parquet(data_dir / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(data_dir / "forecasts_filtered_rts3_constellation_v1.parquet")
    df_base = build_conformal_totals_df(actuals, forecasts)

    # Add day-ahead valid features
    print("[2/3] Adding day-ahead valid features...")
    df_enhanced = add_dayahead_valid_features(df_base)

    # Basic feature set
    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "hour",
        "dow",
    ]

    df_clean = df_enhanced.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    print(f"  Clean data: {len(df_clean)} time points")

    # Test three binning strategies
    print("\n[3/3] Testing binning strategies...")
    strategies = [
        ("y_pred", "Standard - bin by predictions"),
        ("feature:ens_std", "Feature-based - bin by uncertainty"),
        ("y_actual", "NEW - bin by actuals (proxy at prediction time)"),
    ]

    results = {}

    for binning, description in strategies:
        print(f"\n  Testing: {binning}")
        print(f"    ({description})")

        try:
            bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
                df_clean,
                feature_cols=feature_cols,
                target_col="y",
                scale_col="ens_std",
                alpha_target=0.95,
                binning=binning,
                n_bins=3,
                safety_margin=0.0,
            )

            coverage = metrics["coverage"]
            gap = abs(coverage - 0.95)

            results[binning] = {
                "coverage": coverage,
                "gap": gap,
                "status": "✓" if gap < 0.05 else "⚠",
            }

            print(f"    {results[binning]['status']} Coverage: {coverage:.4f} (gap: {gap:.4f})")

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            results[binning] = {"status": "✗", "error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = all(r.get("status") == "✓" for r in results.values())

    if all_passed:
        print("\n✓ All binning strategies work correctly!")
        print("\nCoverage comparison:")
        for binning, result in results.items():
            if "coverage" in result:
                print(f"  {binning:20s}: {result['coverage']:.4f} (gap: {result['gap']:.4f})")

        print("\n" + "=" * 70)
        print("SUCCESS - y_actual is ready to use!")
        print("=" * 70)
        print("\nYou can now use y_actual in run_paper_figures_dayahead_valid.py:")
        print('  binning_strategy="y_actual"')
        return 0
    else:
        print("\n⚠ Some strategies failed:")
        for binning, result in results.items():
            print(f"  {result['status']} {binning}")
            if "error" in result:
                print(f"      Error: {result['error']}")
        return 1


if __name__ == "__main__":
    exit(test_binning_strategies())
