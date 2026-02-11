#!/usr/bin/env python
"""
Thorough overnight grid search: Comprehensive hyperparameter sweep

Expanded grid for publication-quality results:
- 7 tau values (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0) - finer for omega constraints
- 5 omega_l2_reg values (0.0, 1e-3, 1e-2, 1e-1, 1.0) - only when omega_constraint=none
- 3 scaler types (none, standard, minmax)
- omega_constraint options (none, softmax, simplex, normalize)

Grid size per feature set:
  - With omega_constraint=none: 7 × 5 × 3 = 105 configs
  - With omega_constraint=softmax/simplex/normalize: 7 × 1 × 3 = 21 configs

Expected runtime: ~3-5 hours for default grid (2 feature sets × 105 configs)

Usage:
    # Run focused_2d and high_dim_8d (new feature sets)
    python run_thorough_overnight.py

    # Run all 5 feature sets
    python run_thorough_overnight.py --all-feature-sets

    # Run specific feature sets only
    python run_thorough_overnight.py --feature-sets focused_2d high_dim_8d temporal_3d

    # Run with both L2 reg and softmax constraint
    python run_thorough_overnight.py --omega-constraints none softmax
"""
import subprocess
import sys
import argparse
import time
from pathlib import Path


# Thorough grid for overnight run
# More tau values for finer bandwidth control (especially with omega constraints)
THOROUGH_TAUS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
THOROUGH_OMEGA_L2_REGS = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
THOROUGH_SCALER_TYPES = ["none", "standard", "minmax"]
THOROUGH_OMEGA_CONSTRAINTS = ["none"]  # Default: use L2 regularization

# Feature set options
NEW_FEATURE_SETS = ["focused_2d", "high_dim_8d"]  # New MinMax experiments
ALL_FEATURE_SETS = [
    "focused_2d",
    "high_dim_8d",
    "temporal_3d",
    "per_resource_4d",
    "unscaled_2d",
]


def estimate_runtime(n_feature_sets: int, n_configs: int) -> str:
    """Estimate total runtime based on empirical data."""
    # Empirical: ~1-2 min per config (average)
    min_minutes = n_feature_sets * n_configs * 1.0
    max_minutes = n_feature_sets * n_configs * 2.0

    min_hours = int(min_minutes // 60)
    max_hours = int(max_minutes // 60)

    if max_hours < 1:
        return f"{int(min_minutes)}-{int(max_minutes)} minutes"
    else:
        return f"{min_hours}-{max_hours} hours"


def main():
    parser = argparse.ArgumentParser(
        description="Thorough overnight grid search for feature engineering experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=ALL_FEATURE_SETS,
        default=NEW_FEATURE_SETS,
        help=f"Feature sets to run (default: {NEW_FEATURE_SETS})",
    )
    parser.add_argument(
        "--all-feature-sets",
        action="store_true",
        help="Run all 5 feature sets (overrides --feature-sets)",
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=THOROUGH_TAUS,
        help=f"Tau values (default: {THOROUGH_TAUS})",
    )
    parser.add_argument(
        "--omega-l2-regs",
        nargs="+",
        type=float,
        default=THOROUGH_OMEGA_L2_REGS,
        help=f"Omega L2 reg values (default: {THOROUGH_OMEGA_L2_REGS})",
    )
    parser.add_argument(
        "--scaler-types",
        nargs="+",
        type=str,
        choices=["none", "standard", "minmax"],
        default=THOROUGH_SCALER_TYPES,
        help=f"Scaler types (default: {THOROUGH_SCALER_TYPES})",
    )
    parser.add_argument(
        "--omega-constraints",
        nargs="+",
        type=str,
        default=THOROUGH_OMEGA_CONSTRAINTS,
        choices=["none", "softmax", "simplex", "normalize"],
        help=f"Omega constraint types to sweep (default: {THOROUGH_OMEGA_CONSTRAINTS}). "
        "When not 'none', L2 reg is ignored for that constraint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data/)",
    )
    parser.add_argument(
        "--use-residuals",
        action="store_true",
        help="Use residuals (ACTUAL - MEAN_FORECAST) instead of raw actuals as target",
    )

    args = parser.parse_args()

    # Determine feature sets to run
    if args.all_feature_sets:
        feature_sets = ALL_FEATURE_SETS
    else:
        feature_sets = args.feature_sets

    # Calculate grid size
    # For each constraint: if "none", use all L2 regs; otherwise use 1 (L2 ignored)
    n_taus = len(args.taus)
    n_scalers = len(args.scaler_types)
    n_constraint_reg_combos = sum(
        len(args.omega_l2_regs) if c == "none" else 1
        for c in args.omega_constraints
    )
    n_configs_per_fs = n_taus * n_scalers * n_constraint_reg_combos
    n_total_configs = len(feature_sets) * n_configs_per_fs

    # Estimate runtime
    estimated_time = estimate_runtime(len(feature_sets), n_configs_per_fs)

    # Print configuration summary
    print("=" * 80)
    print("THOROUGH OVERNIGHT GRID SEARCH")
    print("=" * 80)
    print()
    print(f"Feature sets ({len(feature_sets)}):")
    for fs in feature_sets:
        print(f"  - {fs}")
    print()
    print("Hyperparameter grid:")
    print(f"  Taus ({n_taus}): {args.taus}")
    print(f"  Omega constraints: {args.omega_constraints}")
    if "none" in args.omega_constraints:
        print(f"  Omega L2 regs ({len(args.omega_l2_regs)}): {args.omega_l2_regs} (for constraint='none')")
    print(f"  Scaler types ({n_scalers}): {args.scaler_types}")
    print()
    print(
        f"Grid size: {n_taus} taus × {n_scalers} scalers × {n_constraint_reg_combos} (constraint,reg) = {n_configs_per_fs} configs per feature set"
    )
    print(
        f"Total configs: {len(feature_sets)} feature sets × {n_configs_per_fs} = {n_total_configs}"
    )
    print()
    print(f"Estimated runtime: {estimated_time}")
    print()
    print("Outputs will be saved to:")
    print(f"  {args.data_dir / 'viz_artifacts'}/<feature_set>/")
    print()
    print("=" * 80)
    print()

    # Confirm before starting
    response = input("Start thorough grid search? [y/N]: ")
    if response.lower() not in ["y", "yes"]:
        print("Aborted.")
        sys.exit(0)

    print()
    print("Starting grid search...")
    print()

    # Build command
    cmd = [
        sys.executable,
        "run_all_feature_sets.py",
        "--feature-sets",
        *feature_sets,
        "--scaler-types",
        *args.scaler_types,
        "--taus",
        *[str(t) for t in args.taus],
        "--omega-l2-regs",
        *[str(r) for r in args.omega_l2_regs],
        "--omega-constraints",
        *args.omega_constraints,
        "--data-dir",
        str(args.data_dir),
    ]

    if args.use_residuals:
        cmd.append("--use-residuals")

    print("Command:")
    print(" ".join(cmd))
    print()
    print("=" * 80)
    print()

    # Record start time
    start_time = time.time()

    try:
        # Run the grid search
        subprocess.run(cmd, check=True)

        # Calculate elapsed time
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print()
        print("=" * 80)
        print("GRID SEARCH COMPLETE!")
        print("=" * 80)
        print()
        print(f"Total runtime: {hours}h {minutes}m {seconds}s")
        print()
        print("Results saved to:")
        for fs in feature_sets:
            print(f"  {args.data_dir / 'viz_artifacts' / fs}/")
        print()
        print("Comparison summary:")
        print(f"  {args.data_dir / 'viz_artifacts' / 'feature_set_comparison.csv'}")
        print()
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Check comparison summary to identify best feature set")
        print("2. Inspect best_omega.npy and visualizations for top performer")
        print("3. Use outputs for paper figures")
        print()

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("GRID SEARCH FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Check logs above for details.")
        sys.exit(1)
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print()
        print("=" * 80)
        print("Grid search interrupted by user")
        print("=" * 80)
        print(f"Partial runtime: {hours}h {minutes}m")
        print()
        print("Partial results may be available in:")
        print(f"  {args.data_dir / 'viz_artifacts'}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
