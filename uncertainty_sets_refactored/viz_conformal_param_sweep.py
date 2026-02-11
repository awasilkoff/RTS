#!/usr/bin/env python3
"""
Visualize conformal prediction hyperparameter sweep results.

Creates plots to understand parameter sensitivity:
- Coverage gap vs n_bins (grouped by bin_strategy)
- Coverage gap vs safety_margin (grouped by binning_strategy)
- Heatmap: n_bins × safety_margin
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_sweep_analysis(
    results_df: pd.DataFrame,
    output_dir: Path,
    alpha_target: float = 0.95,
) -> dict[str, Path]:
    """
    Create analysis plots for conformal sweep results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from sweep_conformal_config.py
    output_dir : Path
        Directory to save plots
    alpha_target : float
        Which alpha value to analyze

    Returns
    -------
    paths : dict[str, Path]
        Paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to target alpha
    df = results_df[results_df["alpha_target"] == alpha_target].copy()

    if len(df) == 0:
        raise ValueError(f"No results for alpha={alpha_target}")

    print(f"Analyzing {len(df)} configurations for α={alpha_target}")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 10

    paths = {}

    # --- PLOT 1: Coverage Gap vs n_bins ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for bin_strat in df["bin_strategy"].unique():
        for binning_strat in df["binning_strategy"].unique():
            subset = df[
                (df["bin_strategy"] == bin_strat)
                & (df["binning_strategy"] == binning_strat)
            ]
            if len(subset) == 0:
                continue

            # Average over safety margins for each n_bins
            agg = (
                subset.groupby("n_bins")["coverage_gap"]
                .agg(["mean", "std"])
                .reset_index()
            )

            label = f"{binning_strat} ({bin_strat})"
            ax.plot(
                agg["n_bins"],
                agg["mean"],
                marker="o",
                label=label,
                linewidth=2,
            )
            ax.fill_between(
                agg["n_bins"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.2,
            )

    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Number of Bins", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coverage Gap (|coverage - α|)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Coverage Gap vs Number of Bins (α={alpha_target:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    path = output_dir / f"sweep_n_bins_alpha_{alpha_target:.2f}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["n_bins_plot"] = path
    print(f"  Saved: {path.name}")

    # --- PLOT 2: Coverage Gap vs Safety Margin ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for binning_strat in df["binning_strategy"].unique():
        for bin_strat in df["bin_strategy"].unique():
            subset = df[
                (df["binning_strategy"] == binning_strat)
                & (df["bin_strategy"] == bin_strat)
            ]
            if len(subset) == 0:
                continue

            # Average over n_bins for each safety_margin
            agg = (
                subset.groupby("safety_margin")["coverage_gap"]
                .agg(["mean", "std"])
                .reset_index()
            )

            label = f"{binning_strat} ({bin_strat})"
            ax.plot(
                agg["safety_margin"],
                agg["mean"],
                marker="o",
                label=label,
                linewidth=2,
            )
            ax.fill_between(
                agg["safety_margin"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.2,
            )

    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Safety Margin", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coverage Gap (|coverage - α|)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Coverage Gap vs Safety Margin (α={alpha_target:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    path = output_dir / f"sweep_safety_margin_alpha_{alpha_target:.2f}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["safety_margin_plot"] = path
    print(f"  Saved: {path.name}")

    # --- PLOT 3: Heatmap for best binning_strategy/bin_strategy combo ---
    # Find the most common best combo
    best_combo = (
        df.nsmallest(10, "coverage_gap")[["binning_strategy", "bin_strategy"]]
        .mode()
        .iloc[0]
    )

    subset = df[
        (df["binning_strategy"] == best_combo["binning_strategy"])
        & (df["bin_strategy"] == best_combo["bin_strategy"])
    ]

    pivot = subset.pivot_table(
        index="n_bins",
        columns="safety_margin",
        values="coverage_gap",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        center=0.05,
        ax=ax,
        cbar_kws={"label": "Coverage Gap"},
    )
    ax.set_title(
        f"Coverage Gap Heatmap: {best_combo['binning_strategy']} + {best_combo['bin_strategy']} (α={alpha_target:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Safety Margin", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Bins", fontsize=11, fontweight="bold")

    path = output_dir / f"sweep_heatmap_alpha_{alpha_target:.2f}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["heatmap"] = path
    print(f"  Saved: {path.name}")

    # --- PLOT 4: Best configs comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Coverage by binning_strategy
    ax = axes[0, 0]
    df.boxplot(column="coverage_gap", by="binning_strategy", ax=ax)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Coverage Gap by Binning Strategy", fontweight="bold")
    ax.set_xlabel("Binning Strategy", fontweight="bold")
    ax.set_ylabel("Coverage Gap", fontweight="bold")
    plt.sca(ax)
    plt.xticks(rotation=0)

    # Top-right: Coverage by bin_strategy
    ax = axes[0, 1]
    df.boxplot(column="coverage_gap", by="bin_strategy", ax=ax)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Coverage Gap by Bin Strategy", fontweight="bold")
    ax.set_xlabel("Bin Strategy", fontweight="bold")
    ax.set_ylabel("Coverage Gap", fontweight="bold")
    plt.sca(ax)
    plt.xticks(rotation=0)

    # Bottom-left: Coverage by n_bins
    ax = axes[1, 0]
    df.boxplot(column="coverage_gap", by="n_bins", ax=ax)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Coverage Gap by Number of Bins", fontweight="bold")
    ax.set_xlabel("Number of Bins", fontweight="bold")
    ax.set_ylabel("Coverage Gap", fontweight="bold")

    # Bottom-right: Coverage by safety_margin
    ax = axes[1, 1]
    df.boxplot(column="coverage_gap", by="safety_margin", ax=ax)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Coverage Gap by Safety Margin", fontweight="bold")
    ax.set_xlabel("Safety Margin", fontweight="bold")
    ax.set_ylabel("Coverage Gap", fontweight="bold")
    plt.sca(ax)
    plt.xticks(rotation=45)

    plt.suptitle(
        f"Parameter Sensitivity Analysis (α={alpha_target:.2f})",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()

    path = output_dir / f"sweep_sensitivity_alpha_{alpha_target:.2f}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["sensitivity"] = path
    print(f"  Saved: {path.name}")

    return paths


if __name__ == "__main__":
    import sys

    DATA_DIR = Path(__file__).parent / "data"
    SWEEP_DIR = DATA_DIR / "viz_artifacts" / "conformal_sweep"
    VIZ_DIR = SWEEP_DIR / "visualizations"

    print("\n" + "=" * 80)
    print("CONFORMAL SWEEP VISUALIZATION")
    print("=" * 80)

    # Check if sweep results exist
    results_path = SWEEP_DIR / "conformal_sweep_results.csv"
    if not results_path.exists():
        print(f"\n⚠️  ERROR: Sweep results not found at {results_path}")
        print("\nPlease run the sweep first:")
        print("  python sweep_conformal_config.py")
        print("  OR")
        print("  python sweep_conformal_quick.py")
        sys.exit(1)

    # Load results
    print(f"\nLoading results from: {results_path}")
    results_df = pd.read_csv(results_path)
    print(f"  {len(results_df)} configurations loaded")

    # Get available alphas
    alphas = sorted(results_df["alpha_target"].unique())
    print(f"\nAvailable alpha values: {alphas}")

    # Generate visualizations for each alpha
    print("\nGenerating visualizations...")
    for alpha in alphas:
        print(f"\n📊 Alpha = {alpha:.2f}")
        paths = plot_sweep_analysis(results_df, VIZ_DIR, alpha_target=alpha)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {VIZ_DIR}")
    print("\nTo view:")
    for alpha in alphas:
        print(f"\n  Alpha = {alpha:.2f}:")
        print(f"    open {VIZ_DIR}/sweep_n_bins_alpha_{alpha:.2f}.png")
        print(f"    open {VIZ_DIR}/sweep_safety_margin_alpha_{alpha:.2f}.png")
        print(f"    open {VIZ_DIR}/sweep_heatmap_alpha_{alpha:.2f}.png")
        print(f"    open {VIZ_DIR}/sweep_sensitivity_alpha_{alpha:.2f}.png")
    print("=" * 80 + "\n")
