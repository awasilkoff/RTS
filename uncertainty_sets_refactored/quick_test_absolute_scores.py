"""
Quick test: Does absolute deviation work better than scaled?

Fast comparison to see if ens_std scaling is causing over-conservatism.

Usage:
    python quick_test_absolute_scores.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df


def quick_comparison(alpha_target: float = 0.95, n_bins: int = 5):
    """Quick comparison of absolute vs scaled deviation for binned conformal."""

    print("\n" + "="*70)
    print("QUICK TEST: Absolute vs Scaled Deviation")
    print("="*70 + "\n")

    # Load data
    print("Loading RTS data...")
    actuals = pd.read_parquet('data/actuals_filtered_rts3_constellation_v1.parquet')
    forecasts = pd.read_parquet('data/forecasts_filtered_rts3_constellation_v1.parquet')
    df = build_conformal_totals_df(actuals, forecasts)
    print(f"Loaded {len(df)} rows\n")

    # Test 1: ABSOLUTE deviation (no scaling)
    print("="*70)
    print("TEST 1: Binned + Absolute Deviation (r = |y_pred - y|)")
    print("="*70)
    df_abs = df.copy()
    df_abs['const_scale'] = 1.0  # Constant scale = absolute deviation

    bundle_abs, metrics_abs, _ = train_wind_lower_model_conformal_binned(
        df_abs,
        feature_cols=['ens_mean', 'ens_std'],
        scale_col='const_scale',
        binning='y_pred',
        n_bins=n_bins,
        alpha_target=alpha_target,
        split_method='random',
        random_seed=42,
    )

    print(f"\nResults:")
    print(f"  Coverage:     {metrics_abs['coverage']:.3f}")
    print(f"  Gap:          {abs(metrics_abs['coverage'] - alpha_target):.3f}")
    print(f"  RMSE:         {metrics_abs['rmse']:.2f} MW")
    print(f"  MAE:          {metrics_abs['mae']:.2f} MW")
    print(f"  q_hat global: {metrics_abs.get('q_hat_global_r', np.nan):.3f}")

    # Test 2: SCALED deviation (current approach)
    print(f"\n{'='*70}")
    print("TEST 2: Binned + Scaled Deviation (r = |y_pred - y| / ens_std)")
    print("="*70)

    bundle_scaled, metrics_scaled, _ = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=['ens_mean', 'ens_std'],
        scale_col='ens_std',
        binning='y_pred',
        n_bins=n_bins,
        alpha_target=alpha_target,
        split_method='random',
        random_seed=42,
    )

    print(f"\nResults:")
    print(f"  Coverage:     {metrics_scaled['coverage']:.3f}")
    print(f"  Gap:          {abs(metrics_scaled['coverage'] - alpha_target):.3f}")
    print(f"  RMSE:         {metrics_scaled['rmse']:.2f} MW")
    print(f"  MAE:          {metrics_scaled['mae']:.2f} MW")
    print(f"  q_hat global: {metrics_scaled.get('q_hat_global_r', np.nan):.3f}")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print("="*70)

    gap_abs = abs(metrics_abs['coverage'] - alpha_target)
    gap_scaled = abs(metrics_scaled['coverage'] - alpha_target)

    print(f"\nCoverage Gap:")
    print(f"  Absolute: {gap_abs:.3f}")
    print(f"  Scaled:   {gap_scaled:.3f}")

    if gap_abs < gap_scaled:
        improvement = gap_scaled - gap_abs
        print(f"  → Absolute is BETTER by {improvement:.3f} ✓")
        print(f"\n  INSIGHT: Scaling by ens_std is HURTING coverage!")
        print(f"           Consider using absolute deviation instead.")
    elif gap_scaled < gap_abs:
        improvement = gap_abs - gap_scaled
        print(f"  → Scaled is BETTER by {improvement:.3f}")
        print(f"\n  INSIGHT: Scaling by ens_std is helping.")
    else:
        print(f"  → No significant difference")
        print(f"\n  INSIGHT: Scaling makes minimal impact.")

    print(f"\nError Metrics:")
    print(f"  Absolute RMSE: {metrics_abs['rmse']:.2f} MW")
    print(f"  Scaled RMSE:   {metrics_scaled['rmse']:.2f} MW")

    if metrics_abs['rmse'] < metrics_scaled['rmse']:
        print(f"  → Absolute has lower error ✓")
    elif metrics_scaled['rmse'] < metrics_abs['rmse']:
        print(f"  → Scaled has lower error")

    print(f"\nConservativeness:")
    over_abs = metrics_abs['coverage'] - alpha_target
    over_scaled = metrics_scaled['coverage'] - alpha_target
    print(f"  Absolute over-coverage: {over_abs:+.3f}")
    print(f"  Scaled over-coverage:   {over_scaled:+.3f}")

    if over_abs < over_scaled:
        print(f"  → Absolute is LESS conservative ✓")
    elif over_scaled < over_abs:
        print(f"  → Scaled is LESS conservative")

    print("\n" + "="*70 + "\n")

    return metrics_abs, metrics_scaled


if __name__ == '__main__':
    metrics_abs, metrics_scaled = quick_comparison(alpha_target=0.95, n_bins=5)
