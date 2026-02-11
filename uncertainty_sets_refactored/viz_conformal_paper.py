"""
IEEE paper-ready visualizations for conformal prediction.

Creates publication-quality figures for the conformal prediction section:
1. Calibration curve - validates conformal guarantee
2. Adaptive correction summary - shows why binned conformal helps

Designed for IEEE two-column format with 300 DPI output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df
from viz_timeseries_conformal import plot_conformal_timeseries


def _wilson_score_interval(
    coverage: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    Parameters
    ----------
    coverage : float
        Observed coverage (proportion, 0-1)
    n : int
        Sample size
    confidence : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    (lower, upper) : tuple[float, float]
        Confidence interval bounds
    """
    if n == 0:
        return (0.0, 1.0)

    # Z-score for confidence level
    from scipy.stats import norm

    z = norm.ppf((1 + confidence) / 2)

    p = coverage
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


def _compute_coverage_by_bin(
    df_test: pd.DataFrame,
    *,
    y_col: str = "y",
    y_pred_col: str = "y_pred_conf",
    bin_col: str = "bin",
) -> pd.DataFrame:
    """
    Compute empirical coverage for each bin.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test set with actual and predicted values
    y_col : str
        Column name for actual values
    y_pred_col : str
        Column name for conformal predictions
    bin_col : str
        Column name for bin assignments

    Returns
    -------
    coverage_df : pd.DataFrame
        Columns: bin, coverage, n_samples
    """
    df = df_test.copy()
    df["covered"] = (df[y_col] >= df[y_pred_col]).astype(int)

    coverage = (
        df.groupby(bin_col, observed=True)
        .agg(coverage=("covered", "mean"), n_samples=("covered", "size"))
        .reset_index()
    )

    return coverage


def plot_calibration_curve(
    results: list[dict[str, float]],
    *,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (8, 6),
    tolerance: float = 0.05,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot calibration validation curve for conformal prediction.

    Shows that empirical coverage matches target coverage (conformal guarantee).

    Parameters
    ----------
    results : list[dict]
        List of dicts with keys: alpha_target, coverage, n_test
        Example: [{'alpha_target': 0.95, 'coverage': 0.948, 'n_test': 100}, ...]
    output_path : Path, optional
        Save figure to this path (creates both PDF and PNG)
    figsize : tuple, default=(8, 6)
        Figure size in inches
    tolerance : float, default=0.05
        Width of acceptable band around diagonal (±5%)
    dpi : int, default=300
        Resolution for PNG output

    Returns
    -------
    fig : plt.Figure
    """
    if not results:
        raise ValueError("Results list is empty")

    # Extract data
    alpha_targets = np.array([r["alpha_target"] for r in results])
    coverages = np.array([r["coverage"] for r in results])
    n_tests = np.array([r["n_test"] for r in results])

    # Compute Wilson score CIs
    ci_lower = []
    ci_upper = []
    for cov, n in zip(coverages, n_tests):
        lower, upper = _wilson_score_interval(cov, int(n))
        ci_lower.append(lower)
        ci_upper.append(upper)

    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # Color code by tolerance
    errors = np.abs(coverages - alpha_targets)
    colors = ["green" if err <= tolerance else "red" for err in errors]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Diagonal reference line
    diag_line = np.linspace(0, 1, 100)
    ax.plot(
        diag_line, diag_line, "k--", linewidth=2, label="Perfect Calibration", zorder=1
    )

    # Tolerance band
    ax.fill_between(
        diag_line,
        diag_line - tolerance,
        diag_line + tolerance,
        alpha=0.2,
        color="gray",
        label=f"±{tolerance:.0%} Tolerance",
        zorder=0,
    )

    # Scatter with error bars
    for i, (alpha, cov, cl, cu, color) in enumerate(
        zip(alpha_targets, coverages, ci_lower, ci_upper, colors)
    ):
        # Error bar
        ax.errorbar(
            alpha,
            cov,
            yerr=[[cov - cl], [cu - cov]],
            fmt="o",
            color=color,
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )

    # Labels and styling
    ax.set_xlabel("Target Coverage (α)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Empirical Coverage", fontsize=12, fontweight="bold")
    ax.set_title(
        "Conformal Prediction Calibration Validation", fontsize=13, fontweight="bold"
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=1)

    # Legend
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # Axis limits
    ax.set_xlim(
        max(0.75, alpha_targets.min() - 0.05), min(1.0, alpha_targets.max() + 0.05)
    )
    ax.set_ylim(
        max(0.75, alpha_targets.min() - 0.05), min(1.0, alpha_targets.max() + 0.05)
    )

    # Equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # PDF for paper
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight", format="pdf")
        print(f"Saved PDF: {pdf_path}")

        # PNG for preview
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight", format="png")
        print(f"Saved PNG: {png_path}")

    return fig


def plot_adaptive_correction_summary(
    bundle: Any,  # ConformalLowerBundle
    df_test: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (12, 5),
    tolerance: float = 0.05,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot 2-panel summary of adaptive conformal corrections.

    Left panel: q_hat by bin (correction strength varies by region)
    Right panel: Coverage by bin (calibration quality per bin)

    Parameters
    ----------
    bundle : ConformalLowerBundle
        Trained model with q_hat_by_bin_r, bin_edges, alpha_target
    df_test : pd.DataFrame
        Test set with columns: y, y_pred_conf, bin
    metrics : dict
        Metrics from training (optional, can be empty)
    output_path : Path, optional
        Save figure to this path
    figsize : tuple, default=(12, 5)
        Figure size in inches (horizontal layout)
    tolerance : float, default=0.05
        Tolerance for coverage violations (±5%)
    dpi : int, default=300
        Resolution for PNG output

    Returns
    -------
    fig : plt.Figure
    """
    # Extract data from bundle (alpha_target is in bundle, not metrics)
    alpha_target = bundle.alpha_target
    q_hat_global = bundle.q_hat_global_r

    # Get bin information
    bin_edges = bundle.bin_edges
    q_hat_by_bin = bundle.q_hat_by_bin_r

    # Compute coverage by bin
    coverage_df = _compute_coverage_by_bin(df_test)

    # Sort bins
    coverage_df = coverage_df.sort_values("bin").reset_index(drop=True)

    # Extract bin intervals for x-axis labels
    bin_labels = []
    q_hat_values = []
    for interval in coverage_df["bin"]:
        bin_labels.append(f"[{interval.left:.0f}, {interval.right:.0f})")
        q_hat_values.append(q_hat_by_bin.get(interval, q_hat_global))

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- LEFT PANEL: q_hat by bin ---
    x_pos = np.arange(len(bin_labels))

    ax1.bar(
        x_pos,
        q_hat_values,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Global reference line
    ax1.axhline(
        q_hat_global,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Global q̂ = {q_hat_global:.2f}",
    )

    ax1.set_xlabel("Prediction Bin", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Correction Factor q̂", fontsize=11, fontweight="bold")
    ax1.set_title("Adaptive Correction by Bin", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)

    # Add value labels on bars
    for i, (pos, val) in enumerate(zip(x_pos, q_hat_values)):
        ax1.text(
            pos,
            val + 0.05 * max(q_hat_values),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # --- RIGHT PANEL: Coverage by bin ---
    coverages = coverage_df["coverage"].values

    # Color by tolerance
    colors = [
        "green" if abs(cov - alpha_target) <= tolerance else "red" for cov in coverages
    ]

    ax2.bar(
        x_pos,
        coverages,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Target coverage line
    ax2.axhline(
        alpha_target,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Target α = {alpha_target:.2f}",
    )

    # Tolerance band
    ax2.axhspan(
        alpha_target - tolerance,
        alpha_target + tolerance,
        alpha=0.2,
        color="gray",
        label=f"±{tolerance:.0%} Tolerance",
    )

    ax2.set_xlabel("Prediction Bin", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Empirical Coverage", fontsize=11, fontweight="bold")
    ax2.set_title("Coverage by Bin", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)
    ax2.set_ylim(max(0.7, alpha_target - 0.15), min(1.0, 1.05))

    # Add value labels on bars
    for pos, cov in zip(x_pos, coverages):
        ax2.text(
            pos,
            cov + 0.01,
            f"{cov:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # PDF for paper
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight", format="pdf")
        print(f"Saved PDF: {pdf_path}")

        # PNG for preview
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight", format="png")
        print(f"Saved PNG: {png_path}")

    return fig


def generate_paper_figures(
    data_dir: Path,
    output_dir: Path,
    *,
    alpha_values: list[float] = [0.80, 0.85, 0.90, 0.95, 0.99],
    primary_alpha: float = 0.95,
) -> dict[str, Path]:
    """
    Generate all IEEE paper figures for conformal prediction.

    Generates:
    1. Timeseries overlay (uses existing viz_timeseries_conformal.py)
    2. Calibration curve (NEW)
    3. Adaptive correction summary (NEW)

    Parameters
    ----------
    data_dir : Path
        Directory with actuals/forecasts parquet files
    output_dir : Path
        Output directory for figures (will create if needed)
    alpha_values : list[float], default=[0.80, 0.85, 0.90, 0.95, 0.99]
        Coverage targets for calibration curve
    primary_alpha : float, default=0.95
        Main alpha for correction summary figure

    Returns
    -------
    paths : dict[str, Path]
        Mapping of figure names to saved paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING IEEE PAPER FIGURES FOR CONFORMAL PREDICTION")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    actuals = pd.read_parquet(
        data_dir / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        data_dir / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_tot = build_conformal_totals_df(actuals, forecasts)
    print(f"  Loaded {len(df_tot)} time points")

    # Feature columns for model
    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # --- FIGURE 2: Calibration Curve ---
    print("\n[2/4] Generating calibration curve...")
    calibration_results = []

    for alpha in alpha_values:
        print(f"  Training model for α={alpha:.2f}...")
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_tot,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
        )

        calibration_results.append(
            {
                "alpha_target": float(alpha),
                "coverage": float(metrics["coverage"]),
                "n_test": int(metrics["n_test"]),
            }
        )
        print(
            f"    Coverage: {metrics['coverage']:.3f} (target: {alpha:.2f}, n_test: {metrics['n_test']})"
        )

    fig_calibration_path = output_dir / "fig_calibration_curve"
    fig_calibration = plot_calibration_curve(
        calibration_results, output_path=fig_calibration_path
    )
    plt.close(fig_calibration)

    # --- FIGURE 3: Adaptive Correction Summary ---
    print(f"\n[3/4] Generating adaptive correction summary (α={primary_alpha:.2f})...")
    (
        bundle_primary,
        metrics_primary,
        df_test_primary,
    ) = train_wind_lower_model_conformal_binned(
        df_tot,
        feature_cols=feature_cols,
        target_col="y",
        scale_col="ens_std",
        alpha_target=float(primary_alpha),
        binning="y_pred",
    )

    fig_correction_path = output_dir / "fig_adaptive_correction"
    fig_correction = plot_adaptive_correction_summary(
        bundle_primary,
        df_test_primary,
        metrics_primary,
        output_path=fig_correction_path,
    )
    plt.close(fig_correction)

    # --- FIGURE 1: Timeseries Overlay ---
    print("\n[4/4] Generating timeseries overlay...")
    # Use the primary alpha model predictions
    df_eval = df_test_primary.copy()
    df_eval["TIME_HOURLY"] = df_tot.loc[df_eval.index, "TIME_HOURLY"]
    df_eval["ens_mean"] = df_tot.loc[df_eval.index, "ens_mean"]

    fig_timeseries_path = output_dir / "fig_timeseries_conformal.png"
    plot_conformal_timeseries(
        df_eval,
        out_png=fig_timeseries_path,
        max_points=500,
        title=f"Wind Forecast vs Conformal Lower Bound (α={primary_alpha:.2f})",
    )

    # Save metadata
    metadata = {
        "alpha_values": alpha_values,
        "primary_alpha": primary_alpha,
        "feature_cols": feature_cols,
        "calibration_results": calibration_results,
        "primary_metrics": {
            "coverage": float(metrics_primary["coverage"]),
            "pre_conformal_coverage": float(metrics_primary["pre_conformal_coverage"]),
            "q_hat_global_r": float(metrics_primary["q_hat_global_r"]),
            "n_test": int(metrics_primary["n_test"]),
        },
    }

    metadata_path = output_dir / "figure_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")

    # Return paths
    paths = {
        "timeseries": fig_timeseries_path,
        "calibration_pdf": output_dir / "fig_calibration_curve.pdf",
        "calibration_png": output_dir / "fig_calibration_curve.png",
        "correction_pdf": output_dir / "fig_adaptive_correction.pdf",
        "correction_png": output_dir / "fig_adaptive_correction.png",
        "metadata": metadata_path,
    }

    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print("\nGenerated figures:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return paths


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "paper_figures"

    paths = generate_paper_figures(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        alpha_values=[0.80, 0.85, 0.90, 0.95, 0.99],
        primary_alpha=0.95,
    )

    print("\n" + "=" * 80)
    print("READY FOR IEEE PAPER")
    print("=" * 80)
    print("\nUse these files in your LaTeX document:")
    print(f"  Figure 1 (timeseries): {paths['timeseries']}")
    print(f"  Figure 2 (calibration): {paths['calibration_pdf']}")
    print(f"  Figure 3 (correction): {paths['correction_pdf']}")
