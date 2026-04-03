"""
Compare weighted conformal prediction to binned conformal on RTS data.

Enhanced version with parameter sweeps and visualizations:
- Sweeps over alpha_target values (coverage levels)
- Sweeps over tau values (bandwidth parameters)
- Generates comprehensive comparison plots

Usage:
    python compare_weighted_vs_binned.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List

from conformal_prediction import (
    train_wind_lower_model_weighted_conformal,
    train_wind_lower_model_conformal_binned,
)
from data_processing import build_conformal_totals_df
from plot_config import setup_plotting, IEEE_TWO_COL_WIDTH


def load_rts_data():
    """Load RTS actuals and forecasts."""
    actuals_path = Path("data/actuals_filtered_rts3_constellation_v1.parquet")
    forecasts_path = Path("data/forecasts_filtered_rts3_constellation_v1.parquet")

    if not actuals_path.exists():
        raise FileNotFoundError(f"Actuals file not found: {actuals_path}")
    if not forecasts_path.exists():
        raise FileNotFoundError(f"Forecasts file not found: {forecasts_path}")

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)

    print(f"Loaded {len(actuals)} actual rows, {len(forecasts)} forecast rows")
    return actuals, forecasts


def sweep_alpha_tau(
    df: pd.DataFrame,
    omega: np.ndarray,
    alpha_values: List[float],
    tau_values: List[float],
    n_bins: int = 5,
    random_seed: int = 42,
):
    """
    Sweep over alpha and tau values, comparing weighted vs binned conformal.

    Parameters
    ----------
    df : pd.DataFrame
        Conformal prediction dataframe
    omega : np.ndarray
        Learned feature weights
    alpha_values : list[float]
        Target coverage levels to test
    tau_values : list[float]
        Bandwidth parameters to test
    n_bins : int
        Number of bins for binned conformal
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    results : pd.DataFrame
        Combined results for all configurations
    """
    results = []

    total_configs = len(alpha_values) * len(tau_values) + len(alpha_values)
    current = 0

    print(f"\n{'='*70}")
    print(f"Parameter Sweep: {len(alpha_values)} alphas x {len(tau_values)} taus")
    print(f"Total configurations: {total_configs}")
    print(f"{'='*70}\n")

    for alpha_target in alpha_values:
        print(f"\n{'─'*70}")
        print(f"Alpha = {alpha_target}")
        print(f"{'─'*70}\n")

        # Train binned conformal (once per alpha, doesn't depend on tau)
        print(f"  Training binned conformal (n_bins={n_bins})...")
        try:
            bundle_binned, metrics_binned, _ = train_wind_lower_model_conformal_binned(
                df,
                feature_cols=["ens_mean", "ens_std"],
                binning="y_pred",
                n_bins=n_bins,
                alpha_target=alpha_target,
                split_method="random",
                random_seed=random_seed,
            )

            results.append(
                {
                    "method": "binned",
                    "alpha_target": alpha_target,
                    "tau": np.nan,  # Binned doesn't use tau
                    "coverage": metrics_binned["coverage"],
                    "gap": abs(metrics_binned["coverage"] - alpha_target),
                    "rmse": metrics_binned["rmse"],
                    "mae": metrics_binned["mae"],
                    "q_hat_mean": np.nan,
                    "q_hat_std": np.nan,
                    "q_hat_global": metrics_binned.get("q_hat_global_r", np.nan),  # Global q_hat for binned
                    "n_bins": n_bins,
                }
            )

            current += 1
            print(
                f"    Coverage: {metrics_binned['coverage']:.3f} (gap: {abs(metrics_binned['coverage'] - alpha_target):.3f})"
            )
            print(f"    Progress: {current}/{total_configs}\n")

        except Exception as e:
            print(f"    (x) Failed: {e}\n")

        # Train weighted conformal for each tau
        for tau in tau_values:
            print(f"  Training weighted conformal (tau={tau})...")
            try:
                (
                    bundle_weighted,
                    metrics_weighted,
                    _,
                ) = train_wind_lower_model_weighted_conformal(
                    df,
                    feature_cols=["ens_mean", "ens_std"],
                    kernel_feature_cols=["SYS_MEAN", "SYS_STD"],
                    omega=omega,
                    tau=tau,
                    alpha_target=alpha_target,
                    split_method="random",
                    random_seed=random_seed,
                )

                results.append(
                    {
                        "method": "weighted",
                        "alpha_target": alpha_target,
                        "tau": tau,
                        "coverage": metrics_weighted["coverage"],
                        "gap": abs(metrics_weighted["coverage"] - alpha_target),
                        "rmse": metrics_weighted["rmse"],
                        "mae": metrics_weighted["mae"],
                        "q_hat_mean": metrics_weighted["q_hat_mean"],
                        "q_hat_std": metrics_weighted["q_hat_std"],
                        "q_hat_global": np.nan,  # Weighted uses localized q_hat
                        "n_bins": np.nan,
                    }
                )

                current += 1
                print(
                    f"    Coverage: {metrics_weighted['coverage']:.3f} (gap: {abs(metrics_weighted['coverage'] - alpha_target):.3f})"
                )
                print(
                    f"    q_hat: mean={metrics_weighted['q_hat_mean']:.3f}, std={metrics_weighted['q_hat_std']:.3f}"
                )
                print(f"    Progress: {current}/{total_configs}\n")

            except Exception as e:
                print(f"    (x) Failed: {e}\n")

    df_results = pd.DataFrame(results)
    return df_results


def plot_comparison_results(df_results: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive visualization of comparison results.

    Generates:
    1. Coverage vs Tau for different alphas
    2. Coverage Gap comparison (weighted vs binned)
    3. q_hat spatial variation vs Tau
    4. Error metrics comparison
    5. Heatmap of coverage gap across (alpha, tau) space
    """
    print("\nGenerating visualizations...")

    # Set up paper-ready fonts
    setup_plotting()
    colors = {"weighted": "#2E86AB", "binned": "#A23B72"}

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Get unique alpha values
    alpha_values = sorted(df_results["alpha_target"].unique())

    # -------------------------------------------------------------------------
    # 1. Coverage vs Tau for different alphas
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])

    for alpha in alpha_values:
        # Weighted results for this alpha
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["alpha_target"] == alpha)
        ]
        if not df_w.empty:
            ax1.plot(
                df_w["tau"],
                df_w["coverage"],
                "o-",
                label=f"Weighted (α={alpha})",
                linewidth=2,
                markersize=8,
            )

        # Binned result for this alpha (horizontal line)
        df_b = df_results[
            (df_results["method"] == "binned") & (df_results["alpha_target"] == alpha)
        ]
        if not df_b.empty:
            coverage_binned = df_b["coverage"].iloc[0]
            ax1.axhline(
                coverage_binned, linestyle="--", alpha=0.7, label=f"Binned (α={alpha})"
            )

    # Add target lines
    for alpha in alpha_values:
        ax1.axhline(alpha, color="red", linestyle=":", alpha=0.3, linewidth=1)

    ax1.set_xlabel("Tau (Bandwidth)", fontweight="bold")
    ax1.set_ylabel("Coverage", fontweight="bold")
    ax1.set_xscale("log")
    ax1.legend(ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        "Coverage vs Tau (Different Target Levels)", fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # 2. Coverage Gap comparison
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])

    gaps_weighted = []
    gaps_binned = []
    labels = []

    for alpha in alpha_values:
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["alpha_target"] == alpha)
        ]
        df_b = df_results[
            (df_results["method"] == "binned") & (df_results["alpha_target"] == alpha)
        ]

        if not df_w.empty and not df_b.empty:
            # Use best tau for weighted (min gap)
            best_w = df_w.loc[df_w["gap"].idxmin()]
            gaps_weighted.append(best_w["gap"])
            gaps_binned.append(df_b["gap"].iloc[0])
            labels.append(f"α={alpha}")

    x = np.arange(len(labels))
    width = 0.35

    ax2.bar(
        x - width / 2,
        gaps_weighted,
        width,
        label="Weighted (best τ)",
        color=colors["weighted"],
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2,
        gaps_binned,
        width,
        label="Binned",
        color=colors["binned"],
        alpha=0.8,
    )

    ax2.set_xlabel("Target Coverage", fontweight="bold")
    ax2.set_ylabel("Coverage Gap", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title(
        "Coverage Gap\n(Weighted Best vs Binned)", fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # 3. q_hat Spatial Variation vs Tau (with binned global reference)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    for alpha in alpha_values:
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["alpha_target"] == alpha)
        ]
        if not df_w.empty:
            ax3.plot(
                df_w["tau"],
                df_w["q_hat_std"],
                "o-",
                label=f"Weighted α={alpha}",
                linewidth=2,
                markersize=8,
            )

        # Add binned global q_hat as horizontal reference
        df_b = df_results[
            (df_results["method"] == "binned") & (df_results["alpha_target"] == alpha)
        ]
        if not df_b.empty and pd.notna(df_b["q_hat_global"].iloc[0]):
            q_global = df_b["q_hat_global"].iloc[0]
            ax3.axhline(
                q_global,
                linestyle=":",
                alpha=0.5,
                linewidth=2,
                label=f"Binned global α={alpha} ({q_global:.2f})",
            )

    ax3.set_xlabel("Tau (Bandwidth)", fontweight="bold")
    ax3.set_ylabel("q_hat (Std for Weighted, Global for Binned)", fontweight="bold")
    ax3.set_xscale("log")
    ax3.legend(ncol=1)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(
        "q_hat: Spatial Variation (Weighted) vs Global (Binned)", fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # 4. RMSE comparison
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    rmse_weighted = []
    rmse_binned = []
    labels = []

    for alpha in alpha_values:
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["alpha_target"] == alpha)
        ]
        df_b = df_results[
            (df_results["method"] == "binned") & (df_results["alpha_target"] == alpha)
        ]

        if not df_w.empty and not df_b.empty:
            # Use best tau for weighted (min gap)
            best_w = df_w.loc[df_w["gap"].idxmin()]
            rmse_weighted.append(best_w["rmse"])
            rmse_binned.append(df_b["rmse"].iloc[0])
            labels.append(f"α={alpha}")

    x = np.arange(len(labels))
    width = 0.35

    ax4.bar(
        x - width / 2,
        rmse_weighted,
        width,
        label="Weighted (best τ)",
        color=colors["weighted"],
        alpha=0.8,
    )
    ax4.bar(
        x + width / 2,
        rmse_binned,
        width,
        label="Binned",
        color=colors["binned"],
        alpha=0.8,
    )

    ax4.set_xlabel("Target Coverage", fontweight="bold")
    ax4.set_ylabel("RMSE (MW)", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_title("Error Comparison\n(Lower is Better)", fontweight="bold")

    # -------------------------------------------------------------------------
    # 5. Heatmap: Coverage Gap vs (Alpha, Tau) for Weighted
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])

    df_w_all = df_results[df_results["method"] == "weighted"].copy()
    if not df_w_all.empty:
        pivot = df_w_all.pivot(index="tau", columns="alpha_target", values="gap")

        im = ax5.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn_r",
            interpolation="nearest",
            vmin=0,
            vmax=0.1,
        )

        # Set ticks
        ax5.set_xticks(np.arange(len(pivot.columns)))
        ax5.set_yticks(np.arange(len(pivot.index)))
        ax5.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax5.set_yticklabels([f"{y:.1f}" for y in pivot.index])

        ax5.set_xlabel("Target Coverage (α)", fontweight="bold")
        ax5.set_ylabel("Tau (Bandwidth)", fontweight="bold")
        ax5.set_title(
            "Weighted Conformal:\nCoverage Gap Heatmap", fontweight="bold"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label("Coverage Gap")

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax5.text(
                    j,
                    i,
                    f"{pivot.values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                )

    # -------------------------------------------------------------------------
    # 6. Coverage vs Alpha (for different taus)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 0])

    tau_values = sorted(
        df_results[df_results["method"] == "weighted"]["tau"].dropna().unique()
    )

    for tau in tau_values[:5]:  # Plot up to 5 tau values
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["tau"] == tau)
        ]
        if not df_w.empty:
            ax6.plot(
                df_w["alpha_target"],
                df_w["coverage"],
                "o-",
                label=f"τ={tau}",
                linewidth=2,
                markersize=6,
            )

    # Binned
    df_b_all = df_results[df_results["method"] == "binned"]
    if not df_b_all.empty:
        ax6.plot(
            df_b_all["alpha_target"],
            df_b_all["coverage"],
            "s--",
            label="Binned",
            linewidth=2,
            markersize=8,
            color=colors["binned"],
        )

    # Perfect coverage line
    alpha_range = np.linspace(
        df_results["alpha_target"].min(), df_results["alpha_target"].max(), 100
    )
    ax6.plot(alpha_range, alpha_range, "r:", linewidth=2, label="Perfect", alpha=0.5)

    ax6.set_xlabel("Target Coverage (α)", fontweight="bold")
    ax6.set_ylabel("Achieved Coverage", fontweight="bold")
    ax6.legend(ncol=2)
    ax6.grid(True, alpha=0.3)
    ax6.set_title("Achieved vs Target Coverage", fontweight="bold")

    # -------------------------------------------------------------------------
    # 7. q_hat Value Comparison (mean for weighted vs global for binned)
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 1])

    qhat_weighted = []
    qhat_binned = []
    labels = []

    for alpha in alpha_values:
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["alpha_target"] == alpha)
        ]
        df_b = df_results[
            (df_results["method"] == "binned") & (df_results["alpha_target"] == alpha)
        ]

        if not df_w.empty and not df_b.empty:
            best_w = df_w.loc[df_w["gap"].idxmin()]
            qhat_weighted.append(best_w["q_hat_mean"])
            qhat_binned.append(df_b["q_hat_global"].iloc[0])
            labels.append(f"α={alpha}")

    x = np.arange(len(labels))
    width = 0.35

    bars_w = ax7.bar(
        x - width / 2,
        qhat_weighted,
        width,
        label="Weighted (mean, best τ)",
        color=colors["weighted"],
        alpha=0.8,
    )
    bars_b = ax7.bar(
        x + width / 2,
        qhat_binned,
        width,
        label="Binned (global)",
        color=colors["binned"],
        alpha=0.8,
    )

    # Add value labels on bars
    for bar in bars_w:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    for bar in bars_b:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    ax7.set_xlabel("Target Coverage", fontweight="bold")
    ax7.set_ylabel("q_hat (Correction Factor)", fontweight="bold")
    ax7.set_xticks(x)
    ax7.set_xticklabels(labels)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis="y")
    ax7.set_title("q_hat Comparison\n(Localized Mean vs Global)", fontweight="bold")

    # -------------------------------------------------------------------------
    # 8. Best tau vs alpha
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 2])

    best_taus = []
    best_gaps = []
    alpha_labels = []

    for alpha in alpha_values:
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["alpha_target"] == alpha)
        ]
        if not df_w.empty:
            best_row = df_w.loc[df_w["gap"].idxmin()]
            best_taus.append(best_row["tau"])
            best_gaps.append(best_row["gap"])
            alpha_labels.append(alpha)

    ax8_color = "tab:blue"
    ax8.set_xlabel("Target Coverage (α)", fontweight="bold")
    ax8.set_ylabel("Best Tau", fontweight="bold", color=ax8_color)
    ax8.plot(alpha_labels, best_taus, "o-", color=ax8_color, linewidth=2, markersize=8)
    ax8.tick_params(axis="y", labelcolor=ax8_color)
    ax8.grid(True, alpha=0.3)

    # Add coverage gap on secondary y-axis
    ax8_twin = ax8.twinx()
    ax8_twin_color = "tab:red"
    ax8_twin.set_ylabel(
        "Coverage Gap at Best Tau", fontweight="bold", color=ax8_twin_color
    )
    ax8_twin.plot(
        alpha_labels, best_gaps, "s--", color=ax8_twin_color, linewidth=2, markersize=6
    )
    ax8_twin.tick_params(axis="y", labelcolor=ax8_twin_color)

    ax8.set_title("Optimal Tau vs Target Coverage", fontweight="bold")

    # Overall title
    fig.suptitle(
        "Weighted vs Binned Conformal Prediction: Comprehensive Comparison",
                fontweight="bold",
        y=0.995,
    )

    # Save figure
    output_path = output_dir / "weighted_vs_binned_comprehensive.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"(ok) Saved comprehensive visualization to {output_path}")

    plt.close()


def print_summary_table(df_results: pd.DataFrame):
    """Print summary table of best results."""
    print("\n" + "=" * 70)
    print("SUMMARY: Best Configuration for Each Alpha")
    print("=" * 70 + "\n")

    alpha_values = sorted(df_results["alpha_target"].unique())

    for alpha in alpha_values:
        print(f"Alpha = {alpha}:")
        print("-" * 70)

        # Best weighted
        df_w = df_results[
            (df_results["method"] == "weighted") & (df_results["alpha_target"] == alpha)
        ]
        if not df_w.empty:
            best_w = df_w.loc[df_w["gap"].idxmin()]
            print(f"  Weighted (best τ={best_w['tau']:.1f}):")
            print(f"    Coverage: {best_w['coverage']:.3f}  (gap: {best_w['gap']:.3f})")
            print(f"    RMSE:     {best_w['rmse']:.2f} MW")
            print(f"    MAE:      {best_w['mae']:.2f} MW")
            print(
                f"    q_hat:    mean={best_w['q_hat_mean']:.3f}, std={best_w['q_hat_std']:.3f}"
            )

        # Binned
        df_b = df_results[
            (df_results["method"] == "binned") & (df_results["alpha_target"] == alpha)
        ]
        if not df_b.empty:
            binned = df_b.iloc[0]
            print(f"  Binned (n_bins={int(binned['n_bins'])}):")
            print(f"    Coverage: {binned['coverage']:.3f}  (gap: {binned['gap']:.3f})")
            print(f"    RMSE:     {binned['rmse']:.2f} MW")
            print(f"    MAE:      {binned['mae']:.2f} MW")
            if pd.notna(binned['q_hat_global']):
                print(f"    q_hat_global: {binned['q_hat_global']:.3f}  (single value for all points)")

        # Winner
        if not df_w.empty and not df_b.empty:
            if best_w["gap"] < binned["gap"]:
                improvement = binned["gap"] - best_w["gap"]
                print(f"  -> Weighted wins by {improvement:.3f} coverage gap (ok)")
            elif binned["gap"] < best_w["gap"]:
                improvement = best_w["gap"] - binned["gap"]
                print(f"  -> Binned wins by {improvement:.3f} coverage gap")
            else:
                print(f"  -> Tie")

        print()


def run_comprehensive_comparison(
    alpha_values: List[float] = None,
    tau_values: List[float] = None,
    n_bins: int = 5,
    omega_path: str = "data/viz_artifacts/focused_2d/best_omega.npy",
):
    """
    Run comprehensive comparison with parameter sweeps and visualizations.

    Parameters
    ----------
    alpha_values : list[float], optional
        Target coverage levels. Default: [0.90, 0.95, 0.99]
    tau_values : list[float], optional
        Bandwidth parameters. Default: [0.5, 1.0, 2.0, 5.0, 10.0]
    n_bins : int, default=5
        Number of bins for binned conformal
    omega_path : str
        Path to learned omega from covariance optimization
    """
    if alpha_values is None:
        alpha_values = [0.90, 0.95, 0.99]
    if tau_values is None:
        tau_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    print("\n" + "=" * 70)
    print("Weighted vs Binned Conformal: Comprehensive Comparison")
    print("=" * 70)
    print(f"Alpha values: {alpha_values}")
    print(f"Tau values:   {tau_values}")
    print(f"Binned bins:  {n_bins}")
    print("=" * 70 + "\n")

    # Load data
    print("Loading RTS data...")
    actuals, forecasts = load_rts_data()
    df = build_conformal_totals_df(actuals, forecasts)
    print(f"Built conformal dataframe: {len(df)} rows\n")

    # Load omega
    omega_file = Path(omega_path)
    if omega_file.exists():
        omega = np.load(omega_path)
        print(f"Loaded omega from {omega_path}")
        print(f"Omega: {omega}\n")
    else:
        print(f"⚠ Omega file not found, using uniform weights")
        omega = np.array([1.0, 1.0])

    # Add kernel features
    if "SYS_MEAN" not in df.columns:
        df["SYS_MEAN"] = df["ens_mean"]
    if "SYS_STD" not in df.columns:
        df["SYS_STD"] = df["ens_std"]

    # Run parameter sweep
    df_results = sweep_alpha_tau(
        df=df,
        omega=omega,
        alpha_values=alpha_values,
        tau_values=tau_values,
        n_bins=n_bins,
    )

    # Save results
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "weighted_vs_binned_sweep.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n(ok) Saved sweep results to {results_path}")

    # Generate visualizations
    plot_comparison_results(df_results, output_dir)

    # Print summary
    print_summary_table(df_results)

    print("\n" + "=" * 70)
    print("(ok) Comprehensive comparison complete!")
    print("=" * 70 + "\n")

    return df_results


if __name__ == "__main__":
    # Run comprehensive comparison with parameter sweeps
    df_results = run_comprehensive_comparison(
        alpha_values=[0.90, 0.95, 0.99],
        tau_values=[0.5, 1.0, 2.0, 5.0, 10.0],
        n_bins=1,
        omega_path="data/viz_artifacts/focused_2d/best_omega.npy",
    )
