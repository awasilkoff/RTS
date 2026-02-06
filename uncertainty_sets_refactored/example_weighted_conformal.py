"""
Simple example demonstrating weighted conformal prediction usage.

This script shows:
1. How to load omega from covariance optimization
2. How to train a weighted conformal model
3. How to make predictions with the bundle
4. How to interpret the results
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from conformal_prediction import train_wind_lower_model_weighted_conformal
from data_processing import build_conformal_totals_df


def run_simple_example():
    """Run a simple example with synthetic data."""
    print("\n" + "="*70)
    print("Weighted Conformal Prediction - Simple Example")
    print("="*70 + "\n")

    # Generate synthetic data
    print("Step 1: Generating synthetic data...")
    n = 1000
    np.random.seed(42)

    # Features: two dimensions
    X = np.random.randn(n, 2)

    # Target: linear relationship + noise
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + np.random.randn(n) * 0.5

    # Create dataframe
    df = pd.DataFrame({
        'TIME_HOURLY': pd.date_range('2020-01-01', periods=n, freq='h'),
        'y': y,
        'ens_mean': X[:, 0],  # Feature for quantile model
        'ens_std': np.abs(np.random.randn(n)) + 0.5,  # Scale estimate
        'SYS_MEAN': X[:, 0],  # Kernel feature 1
        'SYS_STD': X[:, 1],   # Kernel feature 2
    })

    print(f"  Created dataframe with {len(df)} rows")
    print(f"  Columns: {list(df.columns)}\n")

    # Define feature weights (omega)
    print("Step 2: Defining feature weights (omega)...")
    # In practice, load from covariance optimization:
    # omega = np.load('data/viz_artifacts/focused_2d/best_omega.npy')
    #
    # For this example, use uniform weights:
    omega = np.array([1.0, 1.0])
    print(f"  Omega: {omega}")
    print(f"  (In practice, load from covariance optimization)\n")

    # Train weighted conformal model
    print("Step 3: Training weighted conformal model...")
    print(f"  Feature cols (quantile model): ['ens_mean', 'ens_std']")
    print(f"  Kernel feature cols: ['SYS_MEAN', 'SYS_STD']")
    print(f"  Tau (bandwidth): 5.0")
    print(f"  Alpha target (coverage): 0.95")
    print(f"  Split method: random\n")

    bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
        df,
        feature_cols=['ens_mean', 'ens_std'],
        kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
        omega=omega,
        tau=5.0,
        alpha_target=0.95,
        split_method='random',
        random_seed=42,
    )

    # Display results
    print("="*70)
    print("RESULTS")
    print("="*70 + "\n")

    print("Model Performance:")
    print(f"  RMSE:        {metrics['rmse']:.2f}")
    print(f"  MAE:         {metrics['mae']:.2f}\n")

    print("Coverage Metrics:")
    print(f"  Target:      {0.95:.3f}")
    print(f"  Achieved:    {metrics['coverage']:.3f}")
    print(f"  Gap:         {abs(metrics['coverage'] - 0.95):.3f}")
    print(f"  Pre-conf:    {metrics['pre_conformal_coverage']:.3f}\n")

    print("Conformal Correction (q_hat):")
    print(f"  Mean:        {metrics['q_hat_mean']:.3f}")
    print(f"  Std:         {metrics['q_hat_std']:.3f}  (spatial variation)")
    print(f"  Range:       [{metrics['q_hat_min']:.3f}, {metrics['q_hat_max']:.3f}]\n")

    print("Data Split:")
    print(f"  Train:       {metrics['n_train']} samples")
    print(f"  Calibration: {metrics['n_cal']} samples")
    print(f"  Test:        {metrics['n_test']} samples\n")

    # Make predictions on new data
    print("="*70)
    print("PREDICTION EXAMPLE")
    print("="*70 + "\n")

    print("Step 4: Making predictions on new data...")
    df_new = df.tail(10).copy()

    df_pred = bundle.predict_df(df_new)

    print(f"Predicted {len(df_pred)} points\n")
    print("Columns in prediction output:")
    for col in df_pred.columns:
        if col.startswith('y_') or col.startswith('q_') or col in ['margin', 'scale_sanitized']:
            print(f"  - {col}")

    print("\nSample predictions (first 3 rows):")
    cols_to_show = ['y', 'y_pred_base', 'q_hat_local', 'margin', 'y_pred_conf']
    print(df_pred[cols_to_show].head(3).to_string(index=False))

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70 + "\n")

    print("The bundle provides:")
    print("  1. y_pred_base:    Base quantile prediction (10th percentile)")
    print("  2. q_hat_local:    Localized conformal correction (query-dependent)")
    print("  3. scale_sanitized: Ensemble std (for scaling)")
    print("  4. margin:         q_hat_local * scale_sanitized")
    print("  5. y_pred_conf:    Final conformal lower bound\n")

    print("Key insight:")
    print("  - q_hat_local varies smoothly across query points (not discrete bins)")
    print("  - Uses kernel-weighted neighbors for local adaptation")
    print("  - Provides valid coverage guarantee under weighted exchangeability\n")

    print("="*70)
    print("✓ Example complete!")
    print("="*70 + "\n")

    return bundle, metrics, df_test


def run_rts_example():
    """Run example with real RTS data (if available)."""
    print("\n" + "="*70)
    print("Weighted Conformal Prediction - RTS Data Example")
    print("="*70 + "\n")

    # Check if data files exist
    actuals_path = Path('data/actuals_filtered_rts3_constellation_v1.parquet')
    forecasts_path = Path('data/forecasts_filtered_rts3_constellation_v1.parquet')
    omega_path = Path('data/viz_artifacts/focused_2d/best_omega.npy')

    if not actuals_path.exists() or not forecasts_path.exists():
        print("⚠ RTS data files not found. Run synthetic example instead.\n")
        return run_simple_example()

    print("Step 1: Loading RTS data...")
    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)
    df = build_conformal_totals_df(actuals, forecasts)
    print(f"  Loaded {len(df)} rows\n")

    # Add kernel features
    if 'SYS_MEAN' not in df.columns:
        df['SYS_MEAN'] = df['ens_mean']
    if 'SYS_STD' not in df.columns:
        df['SYS_STD'] = df['ens_std']

    # Load omega
    print("Step 2: Loading omega from covariance optimization...")
    if omega_path.exists():
        omega = np.load(omega_path)
        print(f"  Loaded omega: {omega}")
    else:
        print("  ⚠ Omega file not found, using uniform weights")
        omega = np.array([1.0, 1.0])
    print()

    # Train model
    print("Step 3: Training weighted conformal model...")
    bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
        df,
        feature_cols=['ens_mean', 'ens_std'],
        kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
        omega=omega,
        tau=5.0,
        alpha_target=0.95,
        split_method='random',
        random_seed=42,
    )

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")

    print(f"Coverage: {metrics['coverage']:.3f}  (target: 0.95, gap: {abs(metrics['coverage'] - 0.95):.3f})")
    print(f"RMSE:     {metrics['rmse']:.2f} MW")
    print(f"MAE:      {metrics['mae']:.2f} MW")
    print(f"q_hat:    mean={metrics['q_hat_mean']:.3f}, std={metrics['q_hat_std']:.3f}")

    print("\n" + "="*70)
    print("✓ RTS example complete!")
    print("="*70 + "\n")

    return bundle, metrics, df_test


if __name__ == '__main__':
    # Try RTS data first, fall back to synthetic
    try:
        bundle, metrics, df_test = run_rts_example()
    except Exception as e:
        print(f"⚠ RTS example failed: {e}")
        print("Running synthetic example instead...\n")
        bundle, metrics, df_test = run_simple_example()
