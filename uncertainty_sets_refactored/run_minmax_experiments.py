#!/usr/bin/env python
"""
Quick experiments: MinMax normalization + new feature sets

Expected runtime: ~30 minutes (16 configs total)

Usage:
    python run_minmax_experiments.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 80)
    print("MinMax Normalization Experiments")
    print("=" * 80)
    print()
    print("Running 2 feature sets with 2 scalers:")
    print("  - focused_2d:  Clean 2D baseline (SYS_MEAN, SYS_STD)")
    print("  - high_dim_8d: High-D exploration (8 features)")
    print()
    print("Grid: 2 taus × 2 regs × 2 scalers = 8 configs each")
    print("Total: 16 configs")
    print()
    print("Expected runtime: ~30 minutes")
    print()

    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "run_all_feature_sets.py",
        "--feature-sets",
        "focused_2d",
        "high_dim_8d",
        "--scaler-types",
        "minmax",
        "--taus",
        "2.0",
        "5.0",
        "10.0",
        "--omega-l2-regs",
        "0.0",
        "1e-2",
    ]

    print("Command:")
    print(" ".join(cmd))
    print()
    print("=" * 80)
    print()

    try:
        # Run the experiment
        subprocess.run(cmd, check=True)

        print()
        print("=" * 80)
        print("Experiments complete!")
        print("=" * 80)
        print()
        print("Results saved to:")
        print("  data/viz_artifacts/focused_2d/")
        print("  data/viz_artifacts/high_dim_8d/")
        print("  data/viz_artifacts/feature_set_comparison.csv")
        print()
        print("Key files to check:")
        print("  - sweep_results.csv: Compare standard vs minmax")
        print("  - omega_bar_chart.png: Learned feature weights")
        print("  - kernel_distance_*.png: Visualization")
        print()

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("Experiment failed!")
        print("=" * 80)
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("Experiment interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
