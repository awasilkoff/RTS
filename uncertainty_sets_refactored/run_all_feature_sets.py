"""
Run all three feature set experiments and generate comparison summary.

Usage:
    python run_all_feature_sets.py
    python run_all_feature_sets.py --feature-sets temporal_3d per_resource_4d
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_SETS = ["temporal_3d", "per_resource_4d", "unscaled_2d", "focused_2d", "high_dim_8d"]

FEATURE_SET_DESCRIPTIONS = {
    "temporal_3d": "3D with temporal nuisance (SYS_MEAN, SYS_STD, HOUR_SIN)",
    "per_resource_4d": "4D per-resource (WIND_122, WIND_309, WIND_317, HOUR_SIN)",
    "unscaled_2d": "2D unscaled (SYS_MEAN_MW, SYS_STD_MW in raw units)",
    "focused_2d": "2D focused baseline (SYS_MEAN, SYS_STD)",
    "high_dim_8d": "8D high-dimensional (SYS + all units' MEAN/STD)",
}

# Default hyperparameters for sweep
# These define the grid search over:
#   - tau: kernel bandwidth (controls local vs global smoothing)
#   - omega_l2_reg: L2 regularization on learned omega weights (shrink-to-baseline)
#   - standardize: whether to standardize features before learning
#
# Note: Even with shrink-to-baseline regularization, tau sweep is still necessary!
# τ controls neighborhood size (bias-variance), while ω controls feature importance.
# These are orthogonal - regularizing ω doesn't remove the need to tune τ.
DEFAULT_TAUS = [2.0, 5.0]
DEFAULT_OMEGA_L2_REGS = [0.0, 1e-2]
DEFAULT_SCALER_TYPES = ["standard", "minmax"]
DEFAULT_OMEGA_CONSTRAINTS = ["none"]


def run_feature_set(
    feature_set: str,
    taus: list[float] | None = None,
    omega_l2_regs: list[float] | None = None,
    scaler_types: list[str] | None = None,
    omega_constraints: list[str] | None = None,
    output_suffix: str = "",
    use_residuals: bool = False,
) -> bool:
    """
    Run sweep for a single feature set.

    Returns True if successful, False otherwise.
    """
    output_name = f"{feature_set}{output_suffix}" if output_suffix else feature_set

    print()
    print("=" * 80)
    print(f"Running feature set: {feature_set}")
    if output_suffix:
        print(f"Output directory: {output_name}")
    print(f"Description: {FEATURE_SET_DESCRIPTIONS[feature_set]}")
    if omega_constraints and omega_constraints != ["none"]:
        print(f"Omega constraints: {omega_constraints}")
    print("=" * 80)
    print()

    cmd = ["python", "sweep_and_viz_feature_set.py", "--feature-set", feature_set]

    if taus:
        cmd.extend(["--taus"] + [str(t) for t in taus])
    if omega_l2_regs:
        cmd.extend(["--omega-l2-regs"] + [str(r) for r in omega_l2_regs])
    if scaler_types is not None:
        cmd.extend(["--scaler-types"] + scaler_types)
    if omega_constraints is not None:
        cmd.extend(["--omega-constraints"] + omega_constraints)
    if output_suffix:
        cmd.extend(["--output-suffix", output_suffix])
    if use_residuals:
        cmd.append("--use-residuals")

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print()
        print(f"(ok) Completed: {feature_set}")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"(x) Failed: {feature_set}")
        print(f"Error: {e}")
        return False


def generate_comparison_summary(
    feature_sets: list[str],
    artifact_dir: Path,
) -> pd.DataFrame:
    """Generate comparison summary across all feature sets."""
    print()
    print("=" * 80)
    print("Generating comparison summary...")
    print("=" * 80)
    print()

    results = []
    for fs in feature_sets:
        sweep_path = artifact_dir / fs / "sweep_results.csv"
        omega_path = artifact_dir / fs / "best_omega.npy"

        if not sweep_path.exists():
            print(f"Warning: No results found for {fs}, skipping...")
            continue

        df = pd.read_csv(sweep_path)
        best = df.sort_values("nll_improvement", ascending=False).iloc[0]

        if omega_path.exists():
            omega = np.load(omega_path)
            omega_str = str(omega.round(3))
        else:
            omega_str = "N/A"

        results.append(
            {
                "feature_set": fs,
                "description": FEATURE_SET_DESCRIPTIONS.get(fs, "Unknown"),
                "nll_improvement": best["nll_improvement"],
                "nll_improvement_vs_global": best["nll_improvement_vs_global"],
                "eval_nll_learned": best["eval_nll_learned"],
                "eval_nll_baseline": best["eval_nll_baseline"],
                "eval_nll_global": best["eval_nll_global"],
                "improvement_pct": 100
                * best["nll_improvement"]
                / best["eval_nll_baseline"],
                "improvement_pct_vs_global": 100
                * best["nll_improvement_vs_global"]
                / best["eval_nll_global"],
                "tau": best["tau"],
                "omega_l2_reg": best["omega_l2_reg"],
                "scaler_type": best["scaler_type"],
                "omega": omega_str,
            }
        )

    if not results:
        print("No results found for any feature set!")
        return pd.DataFrame()

    comparison = pd.DataFrame(results).sort_values("nll_improvement", ascending=False)
    return comparison


def print_summary(comparison: pd.DataFrame, artifact_dir: Path):
    """Print formatted comparison summary."""
    print()
    print("=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print()
    print(comparison.to_string(index=False))
    print()
    print("=" * 100)

    if len(comparison) > 0:
        best = comparison.iloc[0]
        print(f"Best feature set for paper: {best['feature_set']}")
        print(f"  Description: {best['description']}")
        print(
            f"  NLL improvement vs k-NN: {best['nll_improvement']:.3f} ({best['improvement_pct']:.2f}%)"
        )
        print(
            f"  NLL improvement vs global: {best['nll_improvement_vs_global']:.3f} ({best['improvement_pct_vs_global']:.2f}%)"
        )
        print(f"  Learned omega: {best['omega']}")
        print(f"  Best tau: {best['tau']}, omega_l2_reg: {best['omega_l2_reg']}")
        print(f"  Scaler type: {best['scaler_type']}")
    print("=" * 100)
    print()
    print(f"Outputs saved to: {artifact_dir}/<feature_set>/")
    print("  - sweep_results.csv")
    print("  - best_omega.npy")
    print("  - omega_bar_chart.png")
    print("  - kernel_distance_*.png")
    print("  - feature_config.json")
    print("  - README.md")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run all feature set experiments and generate comparison"
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SETS,
        default=FEATURE_SETS,
        help="Feature sets to run (default: all)",
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=DEFAULT_TAUS,
        help=f"Tau values to sweep (default: {DEFAULT_TAUS})",
    )
    parser.add_argument(
        "--omega-l2-regs",
        nargs="+",
        type=float,
        default=DEFAULT_OMEGA_L2_REGS,
        help=f"Omega L2 regularization values (default: {DEFAULT_OMEGA_L2_REGS})",
    )
    parser.add_argument(
        "--scaler-types",
        nargs="+",
        type=str,
        choices=["none", "standard", "minmax"],
        default=DEFAULT_SCALER_TYPES,
        help=f"Scaler types to test (default: {DEFAULT_SCALER_TYPES})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data/)",
    )
    parser.add_argument(
        "--omega-constraints",
        nargs="+",
        type=str,
        default=DEFAULT_OMEGA_CONSTRAINTS,
        choices=["none", "softmax", "simplex", "normalize"],
        help=f"Omega constraint types to sweep (default: {DEFAULT_OMEGA_CONSTRAINTS}). "
             "When not 'none', L2 reg is ignored for that constraint.",
    )
    parser.add_argument(
        "--use-residuals",
        action="store_true",
        help="Use residuals (ACTUAL - MEAN_FORECAST) instead of raw actuals as target",
    )

    args = parser.parse_args()

    artifact_dir = args.data_dir / "viz_artifacts"

    print("=" * 80)
    print("Feature Engineering Experiment Suite for Learned Omega Visualization")
    print("=" * 80)
    print()
    print(
        f"Running {len(args.feature_sets)} feature set(s): {', '.join(args.feature_sets)}"
    )
    print()

    # Track timing
    start_time = time.time()
    success_count = 0
    failed = []

    # Run each feature set
    for fs in args.feature_sets:
        success = run_feature_set(
            fs,
            taus=args.taus,
            omega_l2_regs=args.omega_l2_regs,
            scaler_types=args.scaler_types,
            omega_constraints=args.omega_constraints,
            use_residuals=args.use_residuals,
        )
        if success:
            success_count += 1
        else:
            failed.append(fs)

    # Calculate elapsed time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("=" * 80)
    print(f"Completed {success_count}/{len(args.feature_sets)} feature set(s)")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Total time: {minutes}m {seconds}s")
    print("=" * 80)

    # Generate comparison summary
    if success_count > 0:
        comparison = generate_comparison_summary(args.feature_sets, artifact_dir)
        if not comparison.empty:
            print_summary(comparison, artifact_dir)

            # Save comparison to CSV
            comparison_path = artifact_dir / "feature_set_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            print(f"Comparison saved to: {comparison_path}")
        else:
            print("No valid results to compare.")
    else:
        print("No successful runs to compare.")
        sys.exit(1)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
