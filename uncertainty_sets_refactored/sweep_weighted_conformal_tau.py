"""
Grid search over tau values for weighted conformal prediction.

Finds optimal kernel bandwidth by testing multiple tau values and
measuring coverage gap.

Usage:
    python sweep_weighted_conformal_tau.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from conformal_prediction import train_wind_lower_model_weighted_conformal
from data_processing import build_conformal_totals_df


def load_rts_data():
    """Load RTS actuals and forecasts."""
    actuals_path = Path('data/actuals_filtered_rts3_constellation_v1.parquet')
    forecasts_path = Path('data/forecasts_filtered_rts3_constellation_v1.parquet')

    if not actuals_path.exists():
        raise FileNotFoundError(f"Actuals file not found: {actuals_path}")
    if not forecasts_path.exists():
        raise FileNotFoundError(f"Forecasts file not found: {forecasts_path}")

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)

    print(f"Loaded {len(actuals)} actual rows, {len(forecasts)} forecast rows")
    return actuals, forecasts


def sweep_tau_values(
    omega_path: str = 'data/viz_artifacts/focused_2d/best_omega.npy',
    tau_values: list[float] | None = None,
    alpha_target: float = 0.95,
):
    """
    Grid search over tau values to find optimal bandwidth.

    Parameters
    ----------
    omega_path : str
        Path to saved omega.npy from covariance optimization
    tau_values : list[float], optional
        Tau values to test. Default: [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    alpha_target : float
        Target coverage level
    """
    if tau_values is None:
        tau_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    print(f"\n{'='*70}")
    print(f"Weighted Conformal Tau Parameter Sweep")
    print(f"{'='*70}")
    print(f"Omega path: {omega_path}")
    print(f"Alpha target: {alpha_target}")
    print(f"Tau values: {tau_values}")
    print(f"{'='*70}\n")

    # Load data
    actuals, forecasts = load_rts_data()
    df = build_conformal_totals_df(actuals, forecasts)

    print(f"Built conformal dataframe: {len(df)} rows\n")

    # Load omega
    omega_file = Path(omega_path)
    if not omega_file.exists():
        print(f"⚠ Warning: Omega file not found at {omega_path}")
        print(f"Using uniform omega=[1.0, 1.0] as fallback\n")
        omega = np.array([1.0, 1.0])
    else:
        omega = np.load(omega_path)
        print(f"Loaded omega: {omega}\n")

    # Add kernel features if not present
    if 'SYS_MEAN' not in df.columns:
        df['SYS_MEAN'] = df['ens_mean']
    if 'SYS_STD' not in df.columns:
        df['SYS_STD'] = df['ens_std']

    # Sweep tau values
    results = []

    for tau in tau_values:
        print(f"{'─'*70}")
        print(f"Testing tau = {tau}")
        print(f"{'─'*70}")

        try:
            bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
                df,
                feature_cols=['ens_mean', 'ens_std'],
                kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
                omega=omega,
                tau=tau,
                alpha_target=alpha_target,
                split_method='random',
                random_seed=42,
            )

            gap = abs(metrics['coverage'] - alpha_target)

            results.append({
                'tau': tau,
                'coverage': metrics['coverage'],
                'gap': gap,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'q_hat_mean': metrics['q_hat_mean'],
                'q_hat_std': metrics['q_hat_std'],
                'q_hat_min': metrics['q_hat_min'],
                'q_hat_max': metrics['q_hat_max'],
            })

            print(f"  Coverage: {metrics['coverage']:.3f}  (gap: {gap:.3f})")
            print(f"  RMSE:     {metrics['rmse']:.2f} MW")
            print(f"  q_hat:    mean={metrics['q_hat_mean']:.3f}, std={metrics['q_hat_std']:.3f}")
            print()

        except Exception as e:
            print(f"  ✗ Failed with error: {e}\n")
            continue

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    output_csv = output_dir / 'weighted_conformal_tau_sweep.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"✓ Saved results to {output_csv}\n")

    # Find best tau
    best_idx = df_results['gap'].idxmin()
    best_tau = df_results.loc[best_idx, 'tau']
    best_coverage = df_results.loc[best_idx, 'coverage']
    best_gap = df_results.loc[best_idx, 'gap']

    print(f"{'='*70}")
    print("BEST TAU")
    print(f"{'='*70}")
    print(f"  Tau:      {best_tau}")
    print(f"  Coverage: {best_coverage:.3f}")
    print(f"  Gap:      {best_gap:.3f}")
    print(f"{'='*70}\n")

    # Plot results
    plot_tau_sweep(df_results, alpha_target, output_dir)

    return df_results


def plot_tau_sweep(df_results: pd.DataFrame, alpha_target: float, output_dir: Path):
    """Create visualization of tau sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Coverage vs Tau
    ax = axes[0, 0]
    ax.plot(df_results['tau'], df_results['coverage'], 'o-', linewidth=2, markersize=8)
    ax.axhline(alpha_target, color='red', linestyle='--', linewidth=2, label=f'Target ({alpha_target})')
    ax.set_xlabel('Tau (Bandwidth)', fontsize=12)
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Coverage vs Tau', fontsize=14, fontweight='bold')

    # 2. Coverage Gap vs Tau
    ax = axes[0, 1]
    ax.plot(df_results['tau'], df_results['gap'], 'o-', linewidth=2, markersize=8, color='orange')
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='5% threshold')
    ax.set_xlabel('Tau (Bandwidth)', fontsize=12)
    ax.set_ylabel('Coverage Gap', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Coverage Gap vs Tau', fontsize=14, fontweight='bold')

    # 3. q_hat Mean vs Tau
    ax = axes[1, 0]
    ax.plot(df_results['tau'], df_results['q_hat_mean'], 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Tau (Bandwidth)', fontsize=12)
    ax.set_ylabel('q_hat Mean', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('q_hat Mean vs Tau', fontsize=14, fontweight='bold')

    # 4. q_hat Std vs Tau (spatial variation)
    ax = axes[1, 1]
    ax.plot(df_results['tau'], df_results['q_hat_std'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Tau (Bandwidth)', fontsize=12)
    ax.set_ylabel('q_hat Std (Spatial Variation)', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('q_hat Spatial Variation vs Tau', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_png = output_dir / 'weighted_conformal_tau_sweep.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_png}")

    plt.close()


if __name__ == '__main__':
    # Default sweep
    df_results = sweep_tau_values(
        omega_path='data/viz_artifacts/focused_2d/best_omega.npy',
        tau_values=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        alpha_target=0.95,
    )

    print("\n✓ Tau sweep complete!")
