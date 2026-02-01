#!/usr/bin/env python
"""
Quick test: Visualize 2D data with Standard vs MinMax scaling.

Creates side-by-side scatter plots to show the difference.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from data_processing_extended import FEATURE_BUILDERS
from utils import fit_scaler, apply_scaler


def main():
    print("Creating side-by-side scaler comparison...")
    print()

    # Load data
    data_dir = Path("data")
    actuals = pd.read_parquet(data_dir / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(data_dir / "forecasts_filtered_rts3_constellation_v1.parquet")

    # Build features for focused_2d
    build_fn = FEATURE_BUILDERS["focused_2d"]
    X_raw, Y, times, x_cols, y_cols = build_fn(forecasts, actuals, drop_any_nan_rows=True)

    print(f"Data: {X_raw.shape[0]} points, {X_raw.shape[1]} features")
    print(f"Features: {x_cols}")
    print()

    # Apply both scalers
    scaler_std = fit_scaler(X_raw, "standard")
    X_std = apply_scaler(X_raw, scaler_std)

    scaler_mm = fit_scaler(X_raw, "minmax")
    X_mm = apply_scaler(X_raw, scaler_mm)

    # Pick a sample point to highlight
    target_idx = len(X_raw) // 2
    target_time = times[target_idx]

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: StandardScaler
    ax0 = axes[0]
    ax0.scatter(X_std[:, 0], X_std[:, 1], alpha=0.5, s=20, c='blue', label='All points')
    ax0.scatter(X_std[target_idx, 0], X_std[target_idx, 1],
                s=150, c='red', marker='*', edgecolors='black', linewidths=1,
                label=f'Target: {target_time.strftime("%Y-%m-%d %H:%M")}', zorder=10)
    ax0.set_xlabel(f'{x_cols[0]} (StandardScaler)', fontsize=12)
    ax0.set_ylabel(f'{x_cols[1]} (StandardScaler)', fontsize=12)
    ax0.set_title('StandardScaler: Data centered at 0, scaled to std=1', fontsize=14, fontweight='bold')
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)

    # Add text showing axis ranges
    xlim0 = ax0.get_xlim()
    ylim0 = ax0.get_ylim()
    ax0.text(0.02, 0.98, f'X range: [{xlim0[0]:.2f}, {xlim0[1]:.2f}]\nY range: [{ylim0[0]:.2f}, {ylim0[1]:.2f}]',
             transform=ax0.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: MinMaxScaler
    ax1 = axes[1]
    ax1.scatter(X_mm[:, 0], X_mm[:, 1], alpha=0.5, s=20, c='green', label='All points')
    ax1.scatter(X_mm[target_idx, 0], X_mm[target_idx, 1],
                s=150, c='red', marker='*', edgecolors='black', linewidths=1,
                label=f'Target: {target_time.strftime("%Y-%m-%d %H:%M")}', zorder=10)
    ax1.set_xlabel(f'{x_cols[0]} (MinMaxScaler)', fontsize=12)
    ax1.set_ylabel(f'{x_cols[1]} (MinMaxScaler)', fontsize=12)
    ax1.set_title('MinMaxScaler: Data scaled to [0, 1] range', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add text showing axis ranges
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.text(0.02, 0.98, f'X range: [{xlim1[0]:.2f}, {xlim1[1]:.2f}]\nY range: [{ylim1[0]:.2f}, {ylim1[1]:.2f}]',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Overall title
    fig.suptitle('Scaler Comparison: StandardScaler vs MinMaxScaler',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    save_path = Path("data/viz_artifacts/test_scaler_comparison.png")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    print()

    # Print statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()

    print("StandardScaler:")
    print(f"  {x_cols[0]}: range [{X_std[:, 0].min():.3f}, {X_std[:, 0].max():.3f}], "
          f"mean={X_std[:, 0].mean():.3f}, std={X_std[:, 0].std():.3f}")
    print(f"  {x_cols[1]}: range [{X_std[:, 1].min():.3f}, {X_std[:, 1].max():.3f}], "
          f"mean={X_std[:, 1].mean():.3f}, std={X_std[:, 1].std():.3f}")
    print()

    print("MinMaxScaler:")
    print(f"  {x_cols[0]}: range [{X_mm[:, 0].min():.3f}, {X_mm[:, 0].max():.3f}], "
          f"mean={X_mm[:, 0].mean():.3f}, std={X_mm[:, 0].std():.3f}")
    print(f"  {x_cols[1]}: range [{X_mm[:, 1].min():.3f}, {X_mm[:, 1].max():.3f}], "
          f"mean={X_mm[:, 1].mean():.3f}, std={X_mm[:, 1].std():.3f}")
    print()

    print("=" * 80)
    print("EXPECTED DIFFERENCES:")
    print("=" * 80)
    print("  StandardScaler plot:")
    print("    - Axes should be roughly [-3, 3]")
    print("    - Data centered at origin (0, 0)")
    print("    - Blue color")
    print()
    print("  MinMaxScaler plot:")
    print("    - Axes should be exactly [0, 1]")
    print("    - Data spread across [0, 1] box")
    print("    - Green color")
    print()
    print("If both plots look identical, something is wrong!")
    print("If they have different axis ranges, scalers are working correctly.")
    print()

    plt.show()


if __name__ == "__main__":
    main()
