"""
Visualization: Time-Exponential Weighted Conformal Prediction

Produces publication-ready figures showing:
1. q_hat variation across alpha levels (coverage targets)
2. q_hat variation across hour-of-day (diurnal patterns)
3. Timeseries showing adaptive corrections
4. Coverage calibration curves

Usage:
    python viz_time_exponential_conformal.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Literal
from scipy.interpolate import UnivariateSpline

from experiment_weighting_schemes import train_time_weighted_conformal
from data_processing import build_conformal_totals_df


def train_at_multiple_alphas(
    df: pd.DataFrame,
    alpha_values: list[float],
    half_life_days: float = 30.0,
    causal: bool = True,
    min_lag_days: float = 1.0,
    safety_margin: float = 0.0,
    random_seed: int = 42,
) -> dict:
    """
    Train time-exponential conformal at multiple alpha levels.

    Returns dict mapping alpha -> (metrics, df_test)
    """
    results = {}

    for alpha in alpha_values:
        print(f"Training at alpha={alpha}...")

        _, metrics, df_test = train_time_weighted_conformal(
            df,
            weighting="exponential",
            half_life_days=half_life_days,
            causal=causal,
            min_lag_days=min_lag_days,
            safety_margin=safety_margin,
            alpha_target=alpha,
            split_method="random",
            random_seed=random_seed,
        )

        results[alpha] = (metrics, df_test)

        print(f"  Coverage: {metrics['coverage']:.3f}")
        print(
            f"  q_hat: mean={metrics['q_hat_mean']:.3f}, std={metrics['q_hat_std']:.3f}\n"
        )

    return results


def plot_time_exponential_analysis(
    results: dict,
    half_life_days: float,
    output_dir: Path,
):
    """
    Create comprehensive visualization of time-exponential conformal.

    4-panel figure:
    1. q_hat vs alpha (how correction scales with target coverage)
    2. q_hat vs hour-of-day (diurnal patterns)
    3. Timeseries of corrections for one alpha
    4. Coverage calibration curve
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    alpha_values = sorted(results.keys())

    # -------------------------------------------------------------------------
    # Panel 1: q_hat vs alpha
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    alphas = []
    q_hat_means = []
    q_hat_stds = []

    for alpha in alpha_values:
        metrics, df_test = results[alpha]
        alphas.append(alpha)
        q_hat_means.append(metrics["q_hat_mean"])
        q_hat_stds.append(metrics["q_hat_std"])

    alphas = np.array(alphas)
    q_hat_means = np.array(q_hat_means)
    q_hat_stds = np.array(q_hat_stds)

    ax1.plot(
        alphas,
        q_hat_means,
        "o-",
        linewidth=2,
        markersize=8,
        color="#2A9D8F",
        label="Mean correction",
    )
    ax1.fill_between(
        alphas,
        q_hat_means - q_hat_stds,
        q_hat_means + q_hat_stds,
        alpha=0.3,
        color="#2A9D8F",
        label="±1 std (spatial variation)",
    )

    ax1.set_xlabel("Target Coverage (α)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Conformal Correction (q̂)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Correction vs Target Coverage\n(Half-life = {half_life_days} days)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # -------------------------------------------------------------------------
    # Panel 2: q_hat vs hour-of-day (middle alpha)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    # Use middle alpha for hour analysis
    alpha_mid = alpha_values[len(alpha_values) // 2]
    metrics_mid, df_test_mid = results[alpha_mid]

    # Extract hour from TIME_HOURLY if available
    if "TIME_HOURLY" in df_test_mid.columns:
        df_test_mid["hour"] = pd.to_datetime(df_test_mid["TIME_HOURLY"]).dt.hour

        # Group by hour and compute mean q_hat
        hour_stats = df_test_mid.groupby("hour")["q_hat_local"].agg(
            ["mean", "std", "count"]
        )

        hours = hour_stats.index.values
        q_means = hour_stats["mean"].values
        q_stds = hour_stats["std"].values

        ax2.plot(
            hours,
            q_means,
            "o-",
            linewidth=2,
            markersize=6,
            color="#E63946",
            label="Mean correction",
        )
        ax2.fill_between(
            hours,
            q_means - q_stds,
            q_means + q_stds,
            alpha=0.3,
            color="#E63946",
            label="±1 std",
        )

        ax2.axhline(
            metrics_mid["q_hat_mean"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f'Overall mean ({metrics_mid["q_hat_mean"]:.3f})',
        )

        ax2.set_xlabel("Hour of Day", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Conformal Correction (q̂)", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Diurnal Pattern in Corrections\n(α = {alpha_mid})",
            fontsize=13,
            fontweight="bold",
        )
        ax2.set_xticks(range(0, 24, 3))
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
    else:
        ax2.text(
            0.5, 0.5, "TIME_HOURLY not available", ha="center", va="center", fontsize=12
        )
        ax2.set_title("Hour-of-Day Analysis", fontsize=13, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Correction vs Prediction Level
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    # Extract predictions and corrections
    y_pred = df_test_mid["y_pred_base"].values
    q_hats = df_test_mid["q_hat_local"].values

    # Scatterplot with transparency
    ax3.scatter(y_pred, q_hats, alpha=0.5, s=30, color="#06AED5", edgecolors="none")

    # Mean correction line
    ax3.axhline(
        metrics_mid["q_hat_mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean correction ({metrics_mid["q_hat_mean"]:.3f})',
    )

    # Add trend line (locally weighted smoothing)
    # Sort for spline
    sort_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sort_idx]
    q_hats_sorted = q_hats[sort_idx]

    # Only fit spline if we have enough points
    if len(y_pred) > 10:
        try:
            # Use spline with smoothing
            spline = UnivariateSpline(
                y_pred_sorted, q_hats_sorted, s=len(y_pred) * 0.5, k=3
            )
            y_pred_smooth = np.linspace(y_pred.min(), y_pred.max(), 100)
            q_hat_smooth = spline(y_pred_smooth)
            ax3.plot(
                y_pred_smooth,
                q_hat_smooth,
                color="darkgreen",
                linewidth=2.5,
                label="Adaptive trend",
                zorder=5,
            )
        except:
            # If spline fails, just skip it
            pass

    ax3.set_xlabel("Wind Prediction (MW)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Conformal Correction (q̂)", fontsize=12, fontweight="bold")
    ax3.set_title(
        f"Correction vs Prediction Level\n(α = {alpha_mid})",
        fontsize=13,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # -------------------------------------------------------------------------
    # Panel 4: Coverage calibration curve
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    target_coverages = []
    achieved_coverages = []

    for alpha in alpha_values:
        metrics, _ = results[alpha]
        target_coverages.append(alpha)
        achieved_coverages.append(metrics["coverage"])

    target_coverages = np.array(target_coverages)
    achieved_coverages = np.array(achieved_coverages)

    # Perfect calibration line
    ax4.plot(
        [0.8, 1.0],
        [0.8, 1.0],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Perfect calibration",
    )

    # Achieved calibration
    ax4.plot(
        target_coverages,
        achieved_coverages,
        "o-",
        linewidth=2.5,
        markersize=10,
        color="#2A9D8F",
        label="Time-exponential",
    )

    # Tolerance bands (±5%)
    ax4.fill_between(
        [0.8, 1.0],
        [0.75, 0.95],
        [0.85, 1.0],
        alpha=0.2,
        color="gray",
        label="±5% tolerance",
    )

    ax4.set_xlabel("Target Coverage", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Achieved Coverage", fontsize=12, fontweight="bold")
    ax4.set_title("Coverage Calibration", fontsize=13, fontweight="bold")
    ax4.set_xlim(0.8, 1.0)
    ax4.set_ylim(0.8, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10, loc="lower right")
    ax4.set_aspect("equal")

    # Overall title
    fig.suptitle(
        f"Time-Exponential Weighted Conformal Prediction Analysis\n"
        + f"(Half-life = {half_life_days} days, Causal with 1-day lag)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save
    output_path = output_dir / "time_exponential_conformal_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n(ok) Saved 4-panel analysis to {output_path}")

    # Also save PDF for paper
    output_path_pdf = output_dir / "time_exponential_conformal_analysis.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"(ok) Saved PDF to {output_path_pdf}")

    plt.close()


def plot_comparison_across_hours(
    results: dict,
    alpha_values: list[float],
    half_life_days: float,
    output_dir: Path,
):
    """
    Create detailed hour-of-day comparison across multiple alphas.

    Shows how diurnal patterns change with target coverage level.
    """
    fig, axes = plt.subplots(len(alpha_values), 1, figsize=(14, 4 * len(alpha_values)))

    if len(alpha_values) == 1:
        axes = [axes]

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]
        metrics, df_test = results[alpha]

        if "TIME_HOURLY" in df_test.columns:
            df_test["hour"] = pd.to_datetime(df_test["TIME_HOURLY"]).dt.hour

            # Group by hour
            hour_stats = df_test.groupby("hour")["q_hat_local"].agg(
                ["mean", "std", "count"]
            )

            hours = hour_stats.index.values
            q_means = hour_stats["mean"].values
            q_stds = hour_stats["std"].values
            n_counts = hour_stats["count"].values

            # Compute standard error
            q_se = q_stds / np.sqrt(n_counts)

            # Plot
            ax.plot(
                hours,
                q_means,
                "o-",
                linewidth=2.5,
                markersize=8,
                color="#E63946",
                label="Mean q̂",
            )
            ax.fill_between(
                hours,
                q_means - 2 * q_se,  # 95% CI
                q_means + 2 * q_se,
                alpha=0.3,
                color="#E63946",
                label="95% CI",
            )

            # Overall mean
            ax.axhline(
                metrics["q_hat_mean"],
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f'Overall mean ({metrics["q_hat_mean"]:.3f})',
            )

            # Formatting
            ax.set_xlabel("Hour of Day", fontsize=12, fontweight="bold")
            ax.set_ylabel("Conformal Correction (q̂)", fontsize=12, fontweight="bold")
            ax.set_title(
                f'α = {alpha} (Target {alpha:.0%} coverage, Achieved {metrics["coverage"]:.1%})',
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            # Highlight peak hours (if significant variation)
            if q_means.max() - q_means.min() > 0.1:
                peak_hour = hours[np.argmax(q_means)]
                trough_hour = hours[np.argmin(q_means)]
                ax.axvline(
                    peak_hour, color="red", linestyle=":", alpha=0.5, linewidth=1.5
                )
                ax.axvline(
                    trough_hour, color="blue", linestyle=":", alpha=0.5, linewidth=1.5
                )
                ax.text(
                    peak_hour,
                    ax.get_ylim()[1] * 0.95,
                    f"Peak\n({peak_hour}h)",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="red",
                    fontweight="bold",
                )
                ax.text(
                    trough_hour,
                    ax.get_ylim()[1] * 0.95,
                    f"Trough\n({trough_hour}h)",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="blue",
                    fontweight="bold",
                )

    fig.suptitle(
        f"Hour-of-Day Patterns Across Coverage Levels\n"
        + f"(Half-life = {half_life_days} days, Causal with 1-day lag)",
        fontsize=16,
        fontweight="bold",
        y=0.998,
    )

    plt.tight_layout()

    # Save
    output_path = output_dir / "time_exponential_hourly_patterns.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"(ok) Saved hourly patterns to {output_path}")

    output_path_pdf = output_dir / "time_exponential_hourly_patterns.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"(ok) Saved PDF to {output_path_pdf}")

    plt.close()


def save_summary_statistics(
    results: dict,
    half_life_days: float,
    output_dir: Path,
):
    """Save summary table of results."""
    rows = []

    for alpha in sorted(results.keys()):
        metrics, df_test = results[alpha]

        # Hour analysis if available
        hour_range = np.nan
        if "TIME_HOURLY" in df_test.columns:
            df_test["hour"] = pd.to_datetime(df_test["TIME_HOURLY"]).dt.hour
            hour_stats = df_test.groupby("hour")["q_hat_local"].mean()
            hour_range = hour_stats.max() - hour_stats.min()

        rows.append(
            {
                "alpha_target": alpha,
                "coverage": metrics["coverage"],
                "gap": abs(metrics["coverage"] - alpha),
                "q_hat_mean": metrics["q_hat_mean"],
                "q_hat_std": metrics["q_hat_std"],
                "q_hat_min": metrics["q_hat_min"],
                "q_hat_max": metrics["q_hat_max"],
                "hour_range": hour_range,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "half_life_days": half_life_days,
            }
        )

    df_summary = pd.DataFrame(rows)

    output_path = output_dir / "time_exponential_summary.csv"
    df_summary.to_csv(output_path, index=False)
    print(f"(ok) Saved summary statistics to {output_path}")

    return df_summary


def run_time_exponential_visualization(
    alpha_values: list[float] = None,
    half_life_days: float = 30.0,
    causal: bool = True,
    min_lag_days: float = 1.0,
    safety_margin: float = 0.0,
    output_subdir: str = "time_exponential",
):
    """
    Run complete visualization pipeline for time-exponential conformal.

    Parameters
    ----------
    alpha_values : list[float]
        Target coverage levels to test
    half_life_days : float
        Half-life for exponential decay weighting
    causal : bool
        Use causal constraint (only past data)
    min_lag_days : float
        Minimum lag (1.0 = day-ahead constraint)
    safety_margin : float
        Adjustment to target coverage (negative reduces conservatism)
        E.g., -0.02 with alpha=0.95 -> effectively targets 0.93 quantile
        Use negative values if experiencing over-coverage
    output_subdir : str
        Subdirectory name for outputs
    """
    if alpha_values is None:
        alpha_values = [0.90, 0.95, 0.99]

    print("\n" + "=" * 80)
    print("TIME-EXPONENTIAL WEIGHTED CONFORMAL PREDICTION - VISUALIZATION")
    print("=" * 80)
    print(f"Alpha values: {alpha_values}")
    print(f"Half-life: {half_life_days} days")
    print(f"Causal: {causal}")
    print(f"Min lag: {min_lag_days} days")
    print("=" * 80 + "\n")

    # Load data
    print("Loading RTS data...")
    actuals_path = Path("data/actuals_filtered_rts3_constellation_v1.parquet")
    forecasts_path = Path("data/forecasts_filtered_rts3_constellation_v1.parquet")

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)
    df = build_conformal_totals_df(actuals, forecasts)

    print(f"Loaded {len(df)} rows")
    print(f"Time range: {df['TIME_HOURLY'].min()} to {df['TIME_HOURLY'].max()}\n")

    # Train at multiple alphas
    results = train_at_multiple_alphas(
        df,
        alpha_values=alpha_values,
        half_life_days=half_life_days,
        causal=causal,
        min_lag_days=min_lag_days,
        safety_margin=safety_margin,
    )

    # Create output directory
    output_dir = Path("data/viz_artifacts") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Generate visualizations
    print("Generating visualizations...\n")

    # 4-panel analysis
    plot_time_exponential_analysis(results, half_life_days, output_dir)

    # Hour-of-day comparison
    plot_comparison_across_hours(results, alpha_values, half_life_days, output_dir)

    # Summary statistics
    df_summary = save_summary_statistics(results, half_life_days, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    print("\n(ok) Time-exponential visualization complete!")
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - time_exponential_conformal_analysis.png (4-panel)")
    print("  - time_exponential_conformal_analysis.pdf (for paper)")
    print("  - time_exponential_hourly_patterns.png (hour-of-day)")
    print("  - time_exponential_hourly_patterns.pdf (for paper)")
    print("  - time_exponential_summary.csv (statistics)")
    print()


if __name__ == "__main__":
    # Run with default settings
    # safety_margin=-0.02: Reduce conservatism for over-coverage
    run_time_exponential_visualization(
        alpha_values=[0.8, 0.85, 0.90, 0.95, 0.99],
        half_life_days=7.0,
        causal=True,
        min_lag_days=1.0,
        safety_margin=-0.01,  # Reduce over-coverage
        output_subdir="time_exponential",
    )
