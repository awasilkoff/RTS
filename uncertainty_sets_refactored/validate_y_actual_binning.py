#!/usr/bin/env python3
"""
Validation script for y_actual binning strategy.

Compares three binning strategies:
1. "y_pred" (standard - bin by predictions)
2. "feature:ens_std" (bin by uncertainty estimate)
3. "y_actual" (new - bin by actuals during calibration, proxy at prediction time)

Usage:
    python validate_y_actual_binning.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df


def run_comparison(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 5,
    alpha_target: float = 0.95,
) -> dict:
    """
    Compare binning strategies and return metrics.
    """
    strategies = ["y_pred", "feature:ens_std", "y_actual"]
    results = {}

    print(f"\n{'='*70}")
    print(f"Comparing Binning Strategies (alpha={alpha_target}, n_bins={n_bins})")
    print(f"{'='*70}\n")

    for binning_strat in strategies:
        print(f"Training with binning='{binning_strat}'...")

        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_clean,
            feature_cols=feature_cols,
            binning=binning_strat,
            n_bins=n_bins,
            alpha_target=alpha_target,
            safety_margin=0.0,  # No safety margin for fair comparison
        )

        coverage = metrics["coverage"]
        gap = abs(coverage - alpha_target)
        pre_conf_coverage = metrics["pre_conformal_coverage"]

        print(f"  Coverage: {coverage:.4f} (gap: {gap:.4f})")
        print(f"  Pre-conformal: {pre_conf_coverage:.4f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  n_train={metrics['n_train']}, n_cal={metrics['n_cal']}, n_test={metrics['n_test']}")
        print()

        results[binning_strat] = {
            "bundle": bundle,
            "metrics": metrics,
            "df_test": df_test,
            "coverage": coverage,
            "gap": gap,
        }

    return results


def analyze_error_by_actual_bin(df_test: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Analyze error patterns by actual generation level.

    Returns DataFrame with mean/std/quantiles of errors per actual bin.
    """
    # Create bins from actual values
    df_test = df_test.copy()
    df_test["y_bin"] = pd.cut(df_test["y"], bins=n_bins)
    df_test["error"] = df_test["y_pred_base"] - df_test["y"]

    stats = df_test.groupby("y_bin")["error"].agg([
        "count",
        "mean",
        "std",
        ("q25", lambda x: x.quantile(0.25)),
        ("q50", lambda x: x.quantile(0.50)),
        ("q75", lambda x: x.quantile(0.75)),
    ]).round(2)

    return stats


def analyze_coverage_by_actual_bin(df_test: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Compute coverage per actual bin to check uniformity.
    """
    df_test = df_test.copy()
    df_test["y_bin"] = pd.cut(df_test["y"], bins=n_bins)
    df_test["covered"] = df_test["y"] >= df_test["y_pred_conf"]

    coverage_by_bin = df_test.groupby("y_bin")["covered"].agg([
        "count",
        "mean",
    ])
    coverage_by_bin.columns = ["n_samples", "coverage"]
    coverage_by_bin["coverage"] = coverage_by_bin["coverage"].round(4)

    return coverage_by_bin


def plot_prediction_vs_actual_scatter(
    results: dict,
    output_dir: Path,
) -> None:
    """
    Scatter plot: y_pred vs y_actual, colored by bin assignment.

    Shows how well predictions proxy for actuals (for y_actual binning).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (strat, data) in zip(axes, results.items()):
        df_test = data["df_test"]
        bundle = data["bundle"]

        # Assign bins using bundle's bin_edges
        bins = bundle.bin_edges
        bin_assignment = pd.cut(df_test["bin_feature"], bins=bins, labels=False)

        scatter = ax.scatter(
            df_test["y_pred_base"],
            df_test["y"],
            c=bin_assignment,
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        ax.plot([0, 250], [0, 250], "r--", lw=1, label="y=y_pred")
        ax.set_xlabel("Prediction (MW)")
        ax.set_ylabel("Actual (MW)")
        ax.set_title(f"Binning: {strat}")
        ax.legend()
        plt.colorbar(scatter, ax=ax, label="Bin Index")

    plt.tight_layout()
    output_path = output_dir / "prediction_vs_actual_scatter.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_distribution_by_bin(
    df_test: pd.DataFrame,
    n_bins: int,
    output_dir: Path,
) -> None:
    """
    Box plot of errors by actual bin.

    Shows if error patterns differ by actual generation level.
    """
    df_test = df_test.copy()
    df_test["y_bin"] = pd.cut(df_test["y"], bins=n_bins)
    df_test["error"] = df_test["y_pred_base"] - df_test["y"]

    fig, ax = plt.subplots(figsize=(10, 5))
    df_test.boxplot(column="error", by="y_bin", ax=ax)
    ax.set_xlabel("Actual Generation Bin (MW)")
    ax.set_ylabel("Error (y_pred - y)")
    ax.set_title("Error Distribution by Actual Generation Level")
    ax.axhline(0, color="red", linestyle="--", lw=1)
    plt.suptitle("")  # Remove automatic title

    plt.tight_layout()
    output_path = output_dir / "error_by_actual_bin.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_coverage_by_bin(
    results: dict,
    n_bins: int,
    alpha_target: float,
    output_dir: Path,
) -> None:
    """
    Bar plot of coverage by actual bin for each strategy.

    Shows if y_actual binning improves coverage uniformity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (strat, data) in zip(axes, results.items()):
        df_test = data["df_test"]
        cov_by_bin = analyze_coverage_by_actual_bin(df_test, n_bins=n_bins)

        # Extract bin midpoints for x-axis
        bin_labels = [f"{interval.left:.0f}-{interval.right:.0f}" for interval in cov_by_bin.index]

        ax.bar(range(len(cov_by_bin)), cov_by_bin["coverage"], alpha=0.7)
        ax.axhline(alpha_target, color="red", linestyle="--", lw=1, label=f"Target ({alpha_target})")
        ax.set_xticks(range(len(cov_by_bin)))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_xlabel("Actual Generation Bin (MW)")
        ax.set_ylabel("Coverage")
        ax.set_title(f"Binning: {strat}")
        ax.set_ylim([0.8, 1.0])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "coverage_by_actual_bin.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_q_hat_comparison(results: dict, output_dir: Path) -> None:
    """
    Bar plot comparing q_hat values across bins for each strategy.

    Shows how adaptive corrections differ by binning strategy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (strat, data) in zip(axes, results.items()):
        bundle = data["bundle"]
        q_hat_by_bin = bundle.q_hat_by_bin_r

        # Extract bin intervals and q_hats
        bins_sorted = sorted(q_hat_by_bin.items(), key=lambda x: x[0].left)
        bin_labels = [f"{interval.left:.0f}-{interval.right:.0f}" for interval, _ in bins_sorted]
        q_hats = [q_hat for _, q_hat in bins_sorted]

        ax.bar(range(len(q_hats)), q_hats, alpha=0.7)
        ax.axhline(bundle.q_hat_global_r, color="red", linestyle="--", lw=1, label="Global q_hat")
        ax.set_xticks(range(len(q_hats)))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_xlabel("Bin Range")
        ax.set_ylabel("q_hat (Conformal Quantile)")
        ax.set_title(f"Binning: {strat}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "q_hat_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("Loading data...")

    # Load actuals and forecasts from parquet files
    data_dir = Path(__file__).parent / "data"
    actuals_path = data_dir / "actuals_filtered_rts3_constellation_v1.parquet"
    forecasts_path = data_dir / "forecasts_filtered_rts3_constellation_v1.parquet"

    if not actuals_path.exists() or not forecasts_path.exists():
        print(f"ERROR: Data files not found!")
        print(f"  Expected actuals: {actuals_path}")
        print(f"  Expected forecasts: {forecasts_path}")
        print(f"\nPlease ensure the data files are in {data_dir}")
        return

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)
    df_clean = build_conformal_totals_df(actuals, forecasts)

    print(f"  Loaded {len(df_clean)} time points")
    print(f"  Columns: {df_clean.columns.tolist()}")

    # Feature columns for quantile regression
    feature_cols = ["ens_mean", "ens_std"]

    # Output directory
    output_dir = Path("data/viz_artifacts/y_actual_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    n_bins = 5
    alpha_target = 0.95
    results = run_comparison(df_clean, feature_cols, n_bins=n_bins, alpha_target=alpha_target)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    summary_rows = []
    for strat, data in results.items():
        summary_rows.append({
            "Binning": strat,
            "Coverage": f"{data['coverage']:.4f}",
            "Gap": f"{data['gap']:.4f}",
            "MAE": f"{data['metrics']['mae']:.2f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()

    # Save summary
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"Saved: {output_dir / 'summary.csv'}")

    # Error analysis (using y_pred results as representative)
    print(f"\n{'='*70}")
    print("Error Statistics by Actual Bin (y_pred strategy)")
    print(f"{'='*70}\n")
    error_stats = analyze_error_by_actual_bin(results["y_pred"]["df_test"], n_bins=n_bins)
    print(error_stats)
    error_stats.to_csv(output_dir / "error_stats_by_actual_bin.csv")
    print(f"\nSaved: {output_dir / 'error_stats_by_actual_bin.csv'}")

    # Coverage by bin analysis
    print(f"\n{'='*70}")
    print("Coverage by Actual Bin (all strategies)")
    print(f"{'='*70}\n")
    for strat, data in results.items():
        print(f"\n{strat}:")
        cov_by_bin = analyze_coverage_by_actual_bin(data["df_test"], n_bins=n_bins)
        print(cov_by_bin)
        cov_by_bin.to_csv(output_dir / f"coverage_by_bin_{strat}.csv")

    # Generate diagnostic plots
    print(f"\n{'='*70}")
    print("Generating Diagnostic Plots")
    print(f"{'='*70}\n")

    plot_prediction_vs_actual_scatter(results, output_dir)
    plot_error_distribution_by_bin(results["y_pred"]["df_test"], n_bins, output_dir)
    plot_coverage_by_bin(results, n_bins, alpha_target, output_dir)
    plot_q_hat_comparison(results, output_dir)

    print(f"\n{'='*70}")
    print("Validation Complete!")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review summary.csv for overall comparison")
    print("2. Check coverage_by_actual_bin.png for uniformity")
    print("3. Examine error_by_actual_bin.png for heterogeneity")
    print("4. Compare q_hat_comparison.png to see adaptive corrections")


if __name__ == "__main__":
    main()
