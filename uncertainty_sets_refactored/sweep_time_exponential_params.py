"""
Parameter sweep for time-exponential weighted conformal prediction.

Sweeps over:
- half_life_days: Controls decay rate (how fast old data is downweighted)
- min_lag_days: Controls minimum lag for day-ahead realism
- alpha_target: Target coverage levels

Identifies best configuration before running viz_time_exponential_conformal.py

Usage:
    python sweep_time_exponential_params.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiment_weighting_schemes import train_time_weighted_conformal
from data_processing import build_conformal_totals_df


def sweep_time_exponential_params(
    half_life_values: list[float] = None,
    min_lag_values: list[float] = None,
    alpha_values: list[float] = None,
    causal: bool = True,
    random_seed: int = 42,
    output_dir: Path = None,
):
    """
    Sweep over time-exponential parameters to find best configuration.

    Parameters
    ----------
    half_life_values : list[float]
        Half-life values to test (days)
    min_lag_values : list[float]
        Minimum lag values to test (days)
    alpha_values : list[float]
        Target coverage levels to test
    causal : bool
        Use causal constraint (only past data)
    random_seed : int
        Random seed for reproducibility
    output_dir : Path
        Output directory for results

    Returns
    -------
    df_results : pd.DataFrame
        Full results table
    """
    if half_life_values is None:
        half_life_values = [7.0, 14.0, 21.0, 30.0, 45.0, 60.0, 90.0]
    if min_lag_values is None:
        min_lag_values = [0.0, 1.0]  # 0=same day, 1=day-ahead
    if alpha_values is None:
        alpha_values = [0.90, 0.95, 0.99]
    if output_dir is None:
        output_dir = Path("data/viz_artifacts/time_exponential_sweep")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("TIME-EXPONENTIAL PARAMETER SWEEP")
    print("="*80)
    print(f"Half-life values: {half_life_values} days")
    print(f"Min lag values: {min_lag_values} days")
    print(f"Alpha values: {alpha_values}")
    print(f"Causal: {causal}")
    print("="*80 + "\n")

    # Load data
    print("Loading RTS data...")
    actuals_path = Path("data/actuals_filtered_rts3_constellation_v1.parquet")
    forecasts_path = Path("data/forecasts_filtered_rts3_constellation_v1.parquet")

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)
    df = build_conformal_totals_df(actuals, forecasts)

    print(f"Loaded {len(df)} rows")
    print(f"Time range: {df['TIME_HOURLY'].min()} to {df['TIME_HOURLY'].max()}\n")

    # Sweep
    results = []
    total_configs = len(half_life_values) * len(min_lag_values) * len(alpha_values)
    current = 0

    for alpha in alpha_values:
        print(f"\n{'='*80}")
        print(f"Alpha = {alpha}")
        print(f"{'='*80}\n")

        for min_lag in min_lag_values:
            print(f"  Min lag = {min_lag} days")

            for half_life in half_life_values:
                current += 1
                print(f"    [{current}/{total_configs}] Half-life = {half_life} days...", end=" ")

                try:
                    _, metrics, df_test = train_time_weighted_conformal(
                        df,
                        weighting='exponential',
                        half_life_days=half_life,
                        causal=causal,
                        min_lag_days=min_lag,
                        alpha_target=alpha,
                        split_method='random',
                        random_seed=random_seed,
                    )

                    gap = abs(metrics['coverage'] - alpha)

                    results.append({
                        'alpha_target': alpha,
                        'half_life_days': half_life,
                        'min_lag_days': min_lag,
                        'coverage': metrics['coverage'],
                        'gap': gap,
                        'q_hat_mean': metrics['q_hat_mean'],
                        'q_hat_std': metrics['q_hat_std'],
                        'q_hat_min': metrics['q_hat_min'],
                        'q_hat_max': metrics['q_hat_max'],
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'n_cal': metrics['n_cal'],
                        'n_test': metrics['n_test'],
                    })

                    print(f"Coverage: {metrics['coverage']:.3f} (gap: {gap:.3f})")

                except Exception as e:
                    print(f"✗ Failed: {e}")

            print()

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    results_path = output_dir / 'sweep_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to {results_path}")

    # Generate visualizations
    plot_sweep_results(df_results, output_dir)

    # Print best configs
    print_best_configs(df_results)

    return df_results


def plot_sweep_results(df_results: pd.DataFrame, output_dir: Path):
    """Create visualizations of sweep results."""
    print("\nGenerating visualizations...")

    alpha_values = sorted(df_results['alpha_target'].unique())
    min_lag_values = sorted(df_results['min_lag_days'].unique())

    # Create figure with subplots
    n_rows = len(alpha_values)
    n_cols = 2  # Coverage gap + q_hat mean

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, alpha in enumerate(alpha_values):
        df_alpha = df_results[df_results['alpha_target'] == alpha]

        # -------------------------------------------------------------------------
        # Left panel: Coverage gap vs half-life
        # -------------------------------------------------------------------------
        ax_left = axes[i, 0]

        for min_lag in min_lag_values:
            df_lag = df_alpha[df_alpha['min_lag_days'] == min_lag]

            if not df_lag.empty:
                label = f'min_lag={min_lag:.1f}d'
                if min_lag == 0.0:
                    label += ' (same-day)'
                elif min_lag == 1.0:
                    label += ' (day-ahead)'

                ax_left.plot(df_lag['half_life_days'], df_lag['gap'],
                           'o-', linewidth=2, markersize=8, label=label)

        ax_left.axhline(0.05, color='red', linestyle='--', linewidth=1.5,
                       alpha=0.5, label='5% threshold')
        ax_left.set_xlabel('Half-life (days)', fontsize=12, fontweight='bold')
        ax_left.set_ylabel('Coverage Gap', fontsize=12, fontweight='bold')
        ax_left.set_title(f'α = {alpha}: Coverage Gap vs Half-life',
                         fontsize=13, fontweight='bold')
        ax_left.set_xscale('log')
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(fontsize=10)

        # Highlight best config
        best_idx = df_alpha['gap'].idxmin()
        best = df_alpha.loc[best_idx]
        ax_left.scatter([best['half_life_days']], [best['gap']],
                       s=200, c='red', marker='*', zorder=5,
                       edgecolors='black', linewidths=1.5,
                       label=f'Best: hl={best["half_life_days"]:.0f}d')

        # -------------------------------------------------------------------------
        # Right panel: q_hat mean vs half-life
        # -------------------------------------------------------------------------
        ax_right = axes[i, 1]

        for min_lag in min_lag_values:
            df_lag = df_alpha[df_alpha['min_lag_days'] == min_lag]

            if not df_lag.empty:
                label = f'min_lag={min_lag:.1f}d'
                if min_lag == 0.0:
                    label += ' (same-day)'
                elif min_lag == 1.0:
                    label += ' (day-ahead)'

                # Plot with error bars (std)
                ax_right.errorbar(df_lag['half_life_days'], df_lag['q_hat_mean'],
                                 yerr=df_lag['q_hat_std'],
                                 fmt='o-', linewidth=2, markersize=8,
                                 capsize=5, capthick=2, label=label)

        ax_right.set_xlabel('Half-life (days)', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('Mean Conformal Correction (q̂)', fontsize=12, fontweight='bold')
        ax_right.set_title(f'α = {alpha}: Correction Magnitude vs Half-life',
                          fontsize=13, fontweight='bold')
        ax_right.set_xscale('log')
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(fontsize=10)

    fig.suptitle('Time-Exponential Parameter Sweep: Coverage and Correction Analysis',
                 fontsize=16, fontweight='bold', y=0.998)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'time_exponential_param_sweep.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved sweep visualization to {output_path}")

    output_path_pdf = output_dir / 'time_exponential_param_sweep.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()

    # -------------------------------------------------------------------------
    # Additional plot: Heatmap of coverage gap
    # -------------------------------------------------------------------------
    plot_heatmaps(df_results, output_dir)


def plot_heatmaps(df_results: pd.DataFrame, output_dir: Path):
    """Create heatmap visualization of coverage gap."""
    alpha_values = sorted(df_results['alpha_target'].unique())
    min_lag_values = sorted(df_results['min_lag_days'].unique())

    fig, axes = plt.subplots(1, len(min_lag_values), figsize=(7*len(min_lag_values), 6))

    if len(min_lag_values) == 1:
        axes = [axes]

    for idx, min_lag in enumerate(min_lag_values):
        ax = axes[idx]

        df_lag = df_results[df_results['min_lag_days'] == min_lag]

        # Pivot for heatmap
        pivot = df_lag.pivot(index='alpha_target', columns='half_life_days', values='gap')

        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=0.10, interpolation='nearest')

        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.0f}' for x in pivot.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])

        ax.set_xlabel('Half-life (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target Coverage (α)', fontsize=12, fontweight='bold')

        title = f'Coverage Gap Heatmap\nmin_lag={min_lag:.1f}d'
        if min_lag == 0.0:
            title += ' (same-day)'
        elif min_lag == 1.0:
            title += ' (day-ahead)'
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    color = 'white' if value > 0.05 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color=color, fontsize=9, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Gap', fontsize=11, fontweight='bold')

    fig.suptitle('Time-Exponential Parameter Sweep: Coverage Gap Heatmaps',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'time_exponential_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmap to {output_path}")

    output_path_pdf = output_dir / 'time_exponential_heatmap.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


def print_best_configs(df_results: pd.DataFrame):
    """Print best configurations for each alpha."""
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)

    alpha_values = sorted(df_results['alpha_target'].unique())

    for alpha in alpha_values:
        df_alpha = df_results[df_results['alpha_target'] == alpha]
        best_idx = df_alpha['gap'].idxmin()
        best = df_alpha.loc[best_idx]

        print(f"\nAlpha = {alpha}:")
        print(f"  Best half-life:  {best['half_life_days']:.1f} days")
        print(f"  Best min_lag:    {best['min_lag_days']:.1f} days")
        print(f"  Coverage:        {best['coverage']:.3f} (target: {alpha:.3f})")
        print(f"  Gap:             {best['gap']:.3f}")
        print(f"  q_hat:           mean={best['q_hat_mean']:.3f}, std={best['q_hat_std']:.3f}")
        print(f"  RMSE:            {best['rmse']:.2f} MW")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Overall best (across all alphas)
    best_overall_idx = df_results['gap'].idxmin()
    best_overall = df_results.loc[best_overall_idx]

    print(f"\nOverall best configuration:")
    print(f"  half_life_days = {best_overall['half_life_days']:.1f}")
    print(f"  min_lag_days = {best_overall['min_lag_days']:.1f}")
    print(f"  (Achieved at α={best_overall['alpha_target']}, gap={best_overall['gap']:.3f})")

    # Check if single config works for all alphas
    print(f"\nTo use with viz_time_exponential_conformal.py:")
    print(f"  run_time_exponential_visualization(")
    print(f"      half_life_days={best_overall['half_life_days']:.1f},")
    print(f"      min_lag_days={best_overall['min_lag_days']:.1f},")
    print(f"  )")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run sweep
    df_results = sweep_time_exponential_params(
        half_life_values=[7.0, 14.0, 21.0, 30.0, 45.0, 60.0, 90.0],
        min_lag_values=[0.0, 1.0],  # Compare same-day vs day-ahead
        alpha_values=[0.90, 0.95, 0.99],
        causal=True,
        random_seed=42,
    )

    print("\n✓ Time-exponential parameter sweep complete!")
