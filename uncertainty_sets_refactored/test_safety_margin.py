#!/usr/bin/env python3
"""
Test the effect of safety_margin parameter on conformal coverage.

Demonstrates how adding a safety buffer improves coverage at lower alpha values.
"""
from pathlib import Path
import pandas as pd

from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df
from run_paper_figures_dayahead_valid import add_dayahead_valid_features


def test_safety_margins():
    """Compare coverage with different safety margins."""

    print("=" * 80)
    print("TESTING SAFETY MARGIN PARAMETER")
    print("=" * 80)

    # Load data
    DATA_DIR = Path(__file__).parent / "data"
    actuals = pd.read_parquet(DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet")
    df_base = build_conformal_totals_df(actuals, forecasts)
    df = add_dayahead_valid_features(df_base)

    # Day-ahead valid features
    feature_cols = [
        "ens_mean", "ens_std", "ens_min", "ens_max", "n_models",
        "hour", "dow", "hour_sin", "hour_cos",
        "ens_mean_lag1", "ens_std_lag1",
        "rolling_mean_3h", "rolling_std_3h",
        "forecast_range_normalized",
    ]

    df_clean = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    print(f"\nDataset: {len(df_clean)} samples")
    print(f"Features: {len(feature_cols)}")

    # Test different safety margins
    safety_margins = [0.0, 0.01, 0.02, 0.03]
    alpha_values = [0.80, 0.85, 0.90, 0.95, 0.99]

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for safety in safety_margins:
        print(f"\n\nSafety Margin = {safety:.2f}")
        print("-" * 80)

        results = []
        for alpha in alpha_values:
            _, metrics, _ = train_wind_lower_model_conformal_binned(
                df_clean,
                feature_cols=feature_cols,
                alpha_target=alpha,
                safety_margin=safety,
                bin_quantiles=[0.0, 0.33, 0.66, 1.0],  # 3 bins
            )

            coverage = metrics["coverage"]
            gap = abs(coverage - alpha)
            status = "(ok)" if gap < 0.05 else "⚠"

            results.append({
                "alpha": alpha,
                "coverage": coverage,
                "gap": gap,
                "status": status,
            })

            print(f"  {status} α={alpha:.2f}: {coverage:.3f} (gap: {gap:.3f})")

        # Summary stats
        avg_gap = sum(r["gap"] for r in results) / len(results)
        max_gap = max(r["gap"] for r in results)
        violations = sum(1 for r in results if r["status"] == "⚠")

        print(f"\n  Summary:")
        print(f"    Average gap: {avg_gap:.3f}")
        print(f"    Max gap: {max_gap:.3f}")
        print(f"    Violations (>5%): {violations}/{len(results)}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("\nBased on your results:")
    print("  • α=0.80 had 6.66% gap (73.3% vs 80%)")
    print("  • α=0.95 had 0.42% gap (95.4% vs 95%)")
    print("\nSuggested safety margins:")
    print("  • safety_margin=0.01 (1% buffer): Gentle improvement")
    print("  • safety_margin=0.02 (2% buffer): Moderate (RECOMMENDED)")
    print("  • safety_margin=0.03 (3% buffer): Conservative")
    print("\nTrade-off:")
    print("  ^ Higher safety margin -> (ok) Better coverage at low α, (x) Wider bounds")
    print("  v Lower safety margin  -> (ok) Tighter bounds, (x) More coverage failures")
    print("\nFor paper: Use safety_margin=0.02 as a reasonable middle ground")


if __name__ == "__main__":
    test_safety_margins()
