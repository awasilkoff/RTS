#!/usr/bin/env python
"""
Generate kernel distance comparison plots for multiple target hours.

Uses best config from focused_2d and selects diverse target points
to show how the learned metric behaves in different regions of feature space.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

from data_processing_extended import FEATURE_BUILDERS
from utils import fit_scaler, apply_scaler
from viz_kernel_distance import plot_kernel_distance_comparison


def select_diverse_targets(X, times, n_targets=9):
    """
    Select diverse target points spanning different regions of feature space.

    Strategy:
    - Pick points from different times (morning, afternoon, evening)
    - Pick points with different feature characteristics (low/med/high)
    - Ensure good spread across the feature space
    """
    # Divide into percentiles for each feature
    percentiles = [10, 50, 90]  # Low, medium, high

    targets = []
    target_descriptions = []

    # Sample from different regions of 2D feature space
    for i, p0 in enumerate(percentiles):
        for j, p1 in enumerate(percentiles):
            # Find points near this percentile combination
            threshold0 = np.percentile(X[:, 0], p0)
            threshold1 = np.percentile(X[:, 1], p1)

            # Find closest point to this percentile combination
            distances = np.abs(X[:, 0] - threshold0) + np.abs(X[:, 1] - threshold1)
            idx = np.argmin(distances)

            targets.append(idx)

            # Create description
            feat0_label = ["Low", "Med", "High"][i]
            feat1_label = ["Low", "Med", "High"][j]
            target_descriptions.append(f"{feat0_label} SYS_MEAN, {feat1_label} SYS_STD")

    return targets[:n_targets], target_descriptions[:n_targets]


def main():
    print("=" * 80)
    print("Generating Multiple Kernel Distance Comparison Plots")
    print("=" * 80)
    print()

    # Load best config
    config_path = Path("data/viz_artifacts/focused_2d/feature_config.json")
    best_config_path = Path("data/viz_artifacts/focused_2d/best_config_summary.json")

    with open(config_path) as f:
        config = json.load(f)

    with open(best_config_path) as f:
        best_config = json.load(f)

    # Extract parameters
    scaler_type = config["scaler_type"]
    tau = best_config["best_config"]["tau"]
    omega_learned = np.array(best_config["best_omega"])
    omega_equal = np.ones_like(omega_learned)

    print(f"Configuration:")
    print(f"  Scaler: {scaler_type}")
    print(f"  Tau: {tau}")
    print(f"  Omega learned: {omega_learned}")
    print(f"  Omega equal: {omega_equal}")
    print()

    # Load data
    data_dir = Path("data")
    actuals = pd.read_parquet(data_dir / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(data_dir / "forecasts_filtered_rts3_constellation_v1.parquet")

    # Build features
    build_fn = FEATURE_BUILDERS["focused_2d"]
    X_raw, Y, times, x_cols, y_cols = build_fn(forecasts, actuals, drop_any_nan_rows=True)

    # Apply scaler
    scaler = fit_scaler(X_raw, scaler_type)
    X = apply_scaler(X_raw, scaler)

    print(f"Data: {X.shape[0]} points")
    print(f"Features: {x_cols}")
    print(f"Time range: {times[0]} to {times[-1]}")
    print()

    # Select diverse targets
    target_indices, descriptions = select_diverse_targets(X, times, n_targets=9)

    print(f"Selected {len(target_indices)} diverse target points:")
    for i, (idx, desc) in enumerate(zip(target_indices, descriptions)):
        print(f"  {i+1}. Index {idx:4d} - {times[idx].strftime('%Y-%m-%d %H:%M')} - {desc}")
        print(f"      Features: [{X[idx, 0]:.3f}, {X[idx, 1]:.3f}]")
    print()

    # Create output directory
    output_dir = Path("data/viz_artifacts/focused_2d/multiple_targets")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate plot for each target
    print("Generating plots...")
    for i, (idx, desc) in enumerate(zip(target_indices, descriptions)):
        target_time = times[idx]

        # Create filename
        filename = f"kernel_comparison_{i+1:02d}_{target_time.strftime('%Y%m%d_%H%M')}.png"
        save_path = output_dir / filename

        # Generate plot
        plot_kernel_distance_comparison(
            X=X,
            x_cols=x_cols,
            times=times,
            target_idx=idx,
            omega_equal=omega_equal,
            omega_learned=omega_learned,
            tau=tau,
            save_path=save_path,
            figsize=(16, 7),
        )

        print(f"  ✓ {i+1}/{len(target_indices)}: {filename}")

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)
    print()
    print(f"Saved {len(target_indices)} plots to:")
    print(f"  {output_dir}/")
    print()
    print("Files:")
    for i, (idx, desc) in enumerate(zip(target_indices, descriptions)):
        target_time = times[idx]
        filename = f"kernel_comparison_{i+1:02d}_{target_time.strftime('%Y%m%d_%H%M')}.png"
        print(f"  {i+1}. {filename}")
        print(f"      {desc} - {target_time.strftime('%Y-%m-%d %H:%M')}")
    print()

    # Create index file
    index_path = output_dir / "README.md"
    with open(index_path, "w") as f:
        f.write("# Multiple Target Kernel Distance Comparisons\n\n")
        f.write("Generated kernel distance comparison plots for diverse target points.\n\n")
        f.write(f"**Configuration:**\n")
        f.write(f"- Scaler: {scaler_type}\n")
        f.write(f"- Tau: {tau}\n")
        f.write(f"- Omega learned: {omega_learned}\n")
        f.write(f"- Omega equal: {omega_equal}\n\n")
        f.write("## Target Points\n\n")

        for i, (idx, desc) in enumerate(zip(target_indices, descriptions)):
            target_time = times[idx]
            filename = f"kernel_comparison_{i+1:02d}_{target_time.strftime('%Y%m%d_%H%M')}.png"
            f.write(f"### {i+1}. {desc}\n\n")
            f.write(f"- **Time:** {target_time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"- **Features:** [{X[idx, 0]:.3f}, {X[idx, 1]:.3f}]\n")
            f.write(f"- **File:** `{filename}`\n\n")
            f.write(f"![{desc}]({filename})\n\n")

    print(f"✓ Created index: {index_path}")
    print()


if __name__ == "__main__":
    main()
