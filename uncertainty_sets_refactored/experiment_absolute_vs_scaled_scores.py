"""
Experiment: Compare absolute deviation vs scaled (studentized) conformity scores.

Tests whether ensemble std scaling provides most of the adaptation benefit,
or if kernel-weighted localization adds meaningful value.

Configurations tested:
1. Binned + Absolute deviation
2. Binned + Scaled deviation (current)
3. Weighted + Absolute deviation
4. Weighted + Scaled deviation (current)

This helps identify:
- Does scaling help? (absolute vs scaled)
- Does localization help? (binned vs weighted)
- What's the interaction?

Usage:
    python experiment_absolute_vs_scaled_scores.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from conformal_prediction import (
    train_wind_lower_model_weighted_conformal,
    train_wind_lower_model_conformal_binned,
)
from data_processing import build_conformal_totals_df


def compute_normalized_margin_stats(
    df_test: pd.DataFrame, df_full: pd.DataFrame
) -> dict:
    """
    Compute margin statistics normalized by ens_std for direct comparison.

    For absolute deviation: margin_normalized = margin_mw / ens_std
    For scaled deviation: margin_normalized = q_hat (already dimensionless)

    This allows apples-to-apples comparison of correction magnitudes.
    """
    if "y_pred_conf" not in df_test or "y_pred_base" not in df_test:
        return {
            "margin_mw_mean": np.nan,
            "margin_mw_std": np.nan,
            "margin_normalized_mean": np.nan,
            "margin_normalized_std": np.nan,
        }

    # Margin in MW
    margin_mw = df_test["y_pred_base"] - df_test["y_pred_conf"]

    # Get ens_std for test points
    # Match test points back to full dataframe to get ens_std
    # Use TIME_HOURLY as key
    if "TIME_HOURLY" in df_test.columns and "TIME_HOURLY" in df_full.columns:
        df_test_with_std = df_test.merge(
            df_full[["TIME_HOURLY", "ens_std"]],
            on="TIME_HOURLY",
            how="left",
            suffixes=("", "_orig"),
        )
        ens_std_test = df_test_with_std["ens_std_orig"].fillna(
            df_test_with_std.get("ens_std", 1.0)
        )
    elif "ens_std" in df_test.columns:
        ens_std_test = df_test["ens_std"]
    else:
        # Fallback: use mean ens_std from full data
        ens_std_test = df_full["ens_std"].mean()

    # Normalized margin
    margin_normalized = margin_mw / ens_std_test

    return {
        "margin_mw_mean": float(margin_mw.mean()),
        "margin_mw_std": float(margin_mw.std()),
        "margin_normalized_mean": float(margin_normalized.mean()),
        "margin_normalized_std": float(margin_normalized.std()),
        "ens_std_test_mean": float(
            ens_std_test.mean() if hasattr(ens_std_test, "mean") else ens_std_test
        ),
    }


def train_binned_absolute(
    df: pd.DataFrame,
    alpha_target: float = 0.95,
    n_bins: int = 5,
    random_seed: int = 42,
):
    """
    Train binned conformal with ABSOLUTE deviation scores (no scaling).

    Instead of: r = |y_pred - y| / ens_std
    Uses:       r = |y_pred - y|

    This is implemented by setting scale_col to a constant (all 1s).
    """
    # Add constant scale column (all ones)
    df_mod = df.copy()
    df_mod["const_scale"] = 1.0

    bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
        df_mod,
        feature_cols=["ens_mean", "ens_std"],
        scale_col="const_scale",  # Use constant scale = absolute deviation
        binning="y_pred",
        n_bins=n_bins,
        alpha_target=alpha_target,
        split_method="random",
        random_seed=random_seed,
    )

    # Compute normalized metrics for comparison
    # margin = q_hat * scale, but scale=1.0, so margin = q_hat (in MW)
    # To normalize: divide by ens_std from original data
    df_test_full = df_test.copy()

    # Get ens_std for test indices (need to match back to original df)
    # For now, compute average margin normalized by ens_std
    if "y_pred_conf" in df_test and "y_pred_base" in df_test:
        margin_mw = (
            df_test["y_pred_base"] - df_test["y_pred_conf"]
        )  # Actual margin in MW

        # Get ens_std for these test points
        # Since we're using random split, we need to extract ens_std carefully
        # Simpler approach: just report the margin in MW and let comparison script normalize
        metrics["margin_mw_mean"] = float(margin_mw.mean())
        metrics["margin_mw_std"] = float(margin_mw.std())

    return bundle, metrics, df_test


def train_weighted_absolute(
    df: pd.DataFrame,
    omega: np.ndarray,
    tau: float,
    alpha_target: float = 0.95,
    random_seed: int = 42,
):
    """
    Train weighted conformal with ABSOLUTE deviation scores (no scaling).

    Instead of: r = |y_pred - y| / ens_std
    Uses:       r = |y_pred - y|

    This is implemented by setting scale_col to a constant (all 1s).
    """
    # Add constant scale column (all ones)
    df_mod = df.copy()
    df_mod["const_scale"] = 1.0

    bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
        df_mod,
        feature_cols=["ens_mean", "ens_std"],
        kernel_feature_cols=["SYS_MEAN", "SYS_STD"],
        scale_col="const_scale",  # Use constant scale = absolute deviation
        omega=omega,
        tau=tau,
        alpha_target=alpha_target,
        split_method="random",
        random_seed=random_seed,
    )

    # Compute normalized metrics
    if "y_pred_conf" in df_test and "y_pred_base" in df_test:
        margin_mw = df_test["y_pred_base"] - df_test["y_pred_conf"]
        metrics["margin_mw_mean"] = float(margin_mw.mean())
        metrics["margin_mw_std"] = float(margin_mw.std())

    return bundle, metrics, df_test


def run_absolute_vs_scaled_experiments(
    alpha_values: list[float] = None,
    tau_values: list[float] = None,
    n_bins: int = 5,
    omega_path: str = "data/viz_artifacts/focused_2d/best_omega.npy",
):
    """
    Run comprehensive experiments comparing absolute vs scaled conformity scores.

    Tests all 4 combinations:
    - Binned + Absolute
    - Binned + Scaled (current)
    - Weighted + Absolute
    - Weighted + Scaled (current)
    """
    if alpha_values is None:
        alpha_values = [0.90, 0.95, 0.99]
    if tau_values is None:
        tau_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    print("\n" + "=" * 80)
    print("EXPERIMENT: Absolute Deviation vs Scaled Deviation Conformity Scores")
    print("=" * 80)
    print(f"Alpha values: {alpha_values}")
    print(f"Tau values (weighted): {tau_values}")
    print(f"Bins (binned): {n_bins}")
    print("=" * 80 + "\n")

    # Load data
    print("Loading RTS data...")
    actuals_path = Path("data/actuals_filtered_rts3_constellation_v1.parquet")
    forecasts_path = Path("data/forecasts_filtered_rts3_constellation_v1.parquet")

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)
    df = build_conformal_totals_df(actuals, forecasts)

    print(f"Loaded {len(df)} rows\n")

    # Load omega
    omega_file = Path(omega_path)
    if omega_file.exists():
        omega = np.load(omega_path)
        print(f"Loaded omega: {omega}")
    else:
        print("⚠ Omega not found, using uniform weights")
        omega = np.array([1.0, 1.0])

    # Add kernel features
    if "SYS_MEAN" not in df.columns:
        df["SYS_MEAN"] = df["ens_mean"]
    if "SYS_STD" not in df.columns:
        df["SYS_STD"] = df["ens_std"]

    print()

    results = []

    # Progress tracking
    total_configs = (
        len(alpha_values) * 2
        + len(alpha_values)  # Binned (absolute + scaled)
        * len(tau_values)
        * 2  # Weighted (absolute + scaled)
    )
    current = 0

    for alpha in alpha_values:
        print(f"\n{'='*80}")
        print(f"Alpha = {alpha}")
        print(f"{'='*80}\n")

        # =====================================================================
        # 1. BINNED + ABSOLUTE
        # =====================================================================
        print(f"  [1/4] Binned + Absolute deviation...")
        try:
            bundle, metrics, df_test = train_binned_absolute(
                df, alpha_target=alpha, n_bins=n_bins, random_seed=42
            )

            # Compute normalized margin stats
            margin_stats = compute_normalized_margin_stats(df_test, df)

            results.append(
                {
                    "method": "binned",
                    "score_type": "absolute",
                    "alpha_target": alpha,
                    "tau": np.nan,
                    "coverage": metrics["coverage"],
                    "gap": abs(metrics["coverage"] - alpha),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "q_hat_global": metrics.get("q_hat_global_r", np.nan),
                    "margin_mw_mean": margin_stats["margin_mw_mean"],
                    "margin_normalized_mean": margin_stats[
                        "margin_normalized_mean"
                    ],  # ← KEY METRIC
                    "margin_normalized_std": margin_stats["margin_normalized_std"],
                }
            )

            current += 1
            print(
                f"    Coverage: {metrics['coverage']:.3f} (gap: {abs(metrics['coverage'] - alpha):.3f})"
            )
            print(
                f"    Margin: {margin_stats['margin_mw_mean']:.2f} MW = {margin_stats['margin_normalized_mean']:.2f}σ"
            )
            print(f"    Progress: {current}/{total_configs}\n")
        except Exception as e:
            print(f"    ✗ Failed: {e}\n")

        # =====================================================================
        # 2. BINNED + SCALED
        # =====================================================================
        print(f"  [2/4] Binned + Scaled deviation (current)...")
        try:
            bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
                df,
                feature_cols=["ens_mean", "ens_std"],
                scale_col="ens_std",
                binning="y_pred",
                n_bins=n_bins,
                alpha_target=alpha,
                split_method="random",
                random_seed=42,
            )

            # Compute normalized margin stats
            margin_stats = compute_normalized_margin_stats(df_test, df)

            results.append(
                {
                    "method": "binned",
                    "score_type": "scaled",
                    "alpha_target": alpha,
                    "tau": np.nan,
                    "coverage": metrics["coverage"],
                    "gap": abs(metrics["coverage"] - alpha),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "q_hat_global": metrics.get("q_hat_global_r", np.nan),
                    "margin_mw_mean": margin_stats["margin_mw_mean"],
                    "margin_normalized_mean": margin_stats[
                        "margin_normalized_mean"
                    ],  # ← KEY METRIC
                    "margin_normalized_std": margin_stats["margin_normalized_std"],
                }
            )

            current += 1
            print(
                f"    Coverage: {metrics['coverage']:.3f} (gap: {abs(metrics['coverage'] - alpha):.3f})"
            )
            print(
                f"    Margin: {margin_stats['margin_mw_mean']:.2f} MW = {margin_stats['margin_normalized_mean']:.2f}σ"
            )
            print(f"    Progress: {current}/{total_configs}\n")
        except Exception as e:
            print(f"    ✗ Failed: {e}\n")

        # =====================================================================
        # 3. WEIGHTED + ABSOLUTE (for each tau)
        # =====================================================================
        print(f"  [3/4] Weighted + Absolute deviation...")
        for tau in tau_values:
            print(f"    tau={tau}...")
            try:
                bundle, metrics, df_test = train_weighted_absolute(
                    df, omega=omega, tau=tau, alpha_target=alpha, random_seed=42
                )

                # Compute normalized margin stats
                margin_stats = compute_normalized_margin_stats(df_test, df)

                results.append(
                    {
                        "method": "weighted",
                        "score_type": "absolute",
                        "alpha_target": alpha,
                        "tau": tau,
                        "coverage": metrics["coverage"],
                        "gap": abs(metrics["coverage"] - alpha),
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "q_hat_global": np.nan,
                        "margin_mw_mean": margin_stats["margin_mw_mean"],
                        "margin_normalized_mean": margin_stats[
                            "margin_normalized_mean"
                        ],
                        "margin_normalized_std": margin_stats["margin_normalized_std"],
                    }
                )

                current += 1
                print(
                    f"      Coverage: {metrics['coverage']:.3f} (gap: {abs(metrics['coverage'] - alpha):.3f})"
                )
                print(
                    f"      Margin: {margin_stats['margin_mw_mean']:.2f} MW = {margin_stats['margin_normalized_mean']:.2f}σ"
                )
                print(f"      Progress: {current}/{total_configs}")
            except Exception as e:
                print(f"      ✗ Failed: {e}")
        print()

        # =====================================================================
        # 4. WEIGHTED + SCALED (for each tau)
        # =====================================================================
        print(f"  [4/4] Weighted + Scaled deviation (current)...")
        for tau in tau_values:
            print(f"    tau={tau}...")
            try:
                bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
                    df,
                    feature_cols=["ens_mean", "ens_std"],
                    kernel_feature_cols=["SYS_MEAN", "SYS_STD"],
                    scale_col="ens_std",
                    omega=omega,
                    tau=tau,
                    alpha_target=alpha,
                    split_method="random",
                    random_seed=42,
                )

                # Compute normalized margin stats
                margin_stats = compute_normalized_margin_stats(df_test, df)

                results.append(
                    {
                        "method": "weighted",
                        "score_type": "scaled",
                        "alpha_target": alpha,
                        "tau": tau,
                        "coverage": metrics["coverage"],
                        "gap": abs(metrics["coverage"] - alpha),
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "q_hat_global": np.nan,
                        "margin_mw_mean": margin_stats["margin_mw_mean"],
                        "margin_normalized_mean": margin_stats[
                            "margin_normalized_mean"
                        ],
                        "margin_normalized_std": margin_stats["margin_normalized_std"],
                    }
                )

                current += 1
                print(
                    f"      Coverage: {metrics['coverage']:.3f} (gap: {abs(metrics['coverage'] - alpha):.3f})"
                )
                print(
                    f"      Margin: {margin_stats['margin_mw_mean']:.2f} MW = {margin_stats['margin_normalized_mean']:.2f}σ"
                )
                print(f"      Progress: {current}/{total_configs}")
            except Exception as e:
                print(f"      ✗ Failed: {e}")
        print()

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "absolute_vs_scaled_experiments.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to {results_path}")

    # Generate visualizations
    plot_absolute_vs_scaled(df_results, output_dir)

    # Print summary
    print_experiment_summary(df_results)

    return df_results


def plot_absolute_vs_scaled(df_results: pd.DataFrame, output_dir: Path):
    """Create visualization comparing absolute vs scaled scores."""
    print("\nGenerating visualizations...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    alpha_values = sorted(df_results["alpha_target"].unique())
    colors = {
        "binned_absolute": "#E63946",
        "binned_scaled": "#F77F00",
        "weighted_absolute": "#06AED5",
        "weighted_scaled": "#2A9D8F",
    }

    # -------------------------------------------------------------------------
    # 1. Coverage Gap: Binned (Absolute vs Scaled)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    gaps_abs = []
    gaps_scaled = []
    labels = []

    for alpha in alpha_values:
        df_abs = df_results[
            (df_results["method"] == "binned")
            & (df_results["score_type"] == "absolute")
            & (df_results["alpha_target"] == alpha)
        ]
        df_scaled = df_results[
            (df_results["method"] == "binned")
            & (df_results["score_type"] == "scaled")
            & (df_results["alpha_target"] == alpha)
        ]

        if not df_abs.empty and not df_scaled.empty:
            gaps_abs.append(df_abs["gap"].iloc[0])
            gaps_scaled.append(df_scaled["gap"].iloc[0])
            labels.append(f"α={alpha}")

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(
        x - width / 2,
        gaps_abs,
        width,
        label="Absolute",
        color=colors["binned_absolute"],
        alpha=0.8,
    )
    ax1.bar(
        x + width / 2,
        gaps_scaled,
        width,
        label="Scaled",
        color=colors["binned_scaled"],
        alpha=0.8,
    )

    ax1.set_xlabel("Target Coverage", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Coverage Gap", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_title(
        "Binned Conformal:\nAbsolute vs Scaled Scores", fontsize=12, fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # 2. Coverage Gap: Weighted (Absolute vs Scaled) - Best Tau
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    gaps_abs_w = []
    gaps_scaled_w = []
    labels_w = []

    for alpha in alpha_values:
        df_abs = df_results[
            (df_results["method"] == "weighted")
            & (df_results["score_type"] == "absolute")
            & (df_results["alpha_target"] == alpha)
        ]
        df_scaled = df_results[
            (df_results["method"] == "weighted")
            & (df_results["score_type"] == "scaled")
            & (df_results["alpha_target"] == alpha)
        ]

        if not df_abs.empty and not df_scaled.empty:
            # Use best tau (min gap)
            best_abs = df_abs.loc[df_abs["gap"].idxmin()]
            best_scaled = df_scaled.loc[df_scaled["gap"].idxmin()]
            gaps_abs_w.append(best_abs["gap"])
            gaps_scaled_w.append(best_scaled["gap"])
            labels_w.append(f"α={alpha}")

    x = np.arange(len(labels_w))

    ax2.bar(
        x - width / 2,
        gaps_abs_w,
        width,
        label="Absolute (best τ)",
        color=colors["weighted_absolute"],
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2,
        gaps_scaled_w,
        width,
        label="Scaled (best τ)",
        color=colors["weighted_scaled"],
        alpha=0.8,
    )

    ax2.set_xlabel("Target Coverage", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Coverage Gap", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_w)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title(
        "Weighted Conformal:\nAbsolute vs Scaled Scores", fontsize=12, fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # 3. Coverage vs Tau (Weighted only, for one alpha)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])

    alpha_mid = alpha_values[len(alpha_values) // 2]  # Middle alpha

    df_abs = df_results[
        (df_results["method"] == "weighted")
        & (df_results["score_type"] == "absolute")
        & (df_results["alpha_target"] == alpha_mid)
    ]
    df_scaled = df_results[
        (df_results["method"] == "weighted")
        & (df_results["score_type"] == "scaled")
        & (df_results["alpha_target"] == alpha_mid)
    ]

    if not df_abs.empty:
        ax3.plot(
            df_abs["tau"],
            df_abs["coverage"],
            "o-",
            label="Absolute",
            linewidth=2,
            markersize=8,
            color=colors["weighted_absolute"],
        )
    if not df_scaled.empty:
        ax3.plot(
            df_scaled["tau"],
            df_scaled["coverage"],
            "s-",
            label="Scaled",
            linewidth=2,
            markersize=8,
            color=colors["weighted_scaled"],
        )

    ax3.axhline(
        alpha_mid,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Target ({alpha_mid})",
    )

    ax3.set_xlabel("Tau (Bandwidth)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Coverage", fontsize=11, fontweight="bold")
    ax3.set_xscale("log")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(
        f"Weighted Coverage vs Tau\n(α={alpha_mid})", fontsize=12, fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # 4. Overall Comparison: All Methods
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, :])

    # Prepare data for grouped bar chart
    methods_labels = []
    gaps_all = []

    for alpha in alpha_values:
        # Binned absolute
        df_ba = df_results[
            (df_results["method"] == "binned")
            & (df_results["score_type"] == "absolute")
            & (df_results["alpha_target"] == alpha)
        ]
        # Binned scaled
        df_bs = df_results[
            (df_results["method"] == "binned")
            & (df_results["score_type"] == "scaled")
            & (df_results["alpha_target"] == alpha)
        ]
        # Weighted absolute (best)
        df_wa = df_results[
            (df_results["method"] == "weighted")
            & (df_results["score_type"] == "absolute")
            & (df_results["alpha_target"] == alpha)
        ]
        # Weighted scaled (best)
        df_ws = df_results[
            (df_results["method"] == "weighted")
            & (df_results["score_type"] == "scaled")
            & (df_results["alpha_target"] == alpha)
        ]

        methods_labels.append(f"α={alpha}")

        row_gaps = []
        if not df_ba.empty:
            row_gaps.append(df_ba["gap"].iloc[0])
        else:
            row_gaps.append(np.nan)

        if not df_bs.empty:
            row_gaps.append(df_bs["gap"].iloc[0])
        else:
            row_gaps.append(np.nan)

        if not df_wa.empty:
            row_gaps.append(df_wa["gap"].min())
        else:
            row_gaps.append(np.nan)

        if not df_ws.empty:
            row_gaps.append(df_ws["gap"].min())
        else:
            row_gaps.append(np.nan)

        gaps_all.append(row_gaps)

    gaps_all = np.array(gaps_all).T

    x = np.arange(len(methods_labels))
    width = 0.2

    ax4.bar(
        x - 1.5 * width,
        gaps_all[0],
        width,
        label="Binned + Absolute",
        color=colors["binned_absolute"],
        alpha=0.8,
    )
    ax4.bar(
        x - 0.5 * width,
        gaps_all[1],
        width,
        label="Binned + Scaled",
        color=colors["binned_scaled"],
        alpha=0.8,
    )
    ax4.bar(
        x + 0.5 * width,
        gaps_all[2],
        width,
        label="Weighted + Absolute (best τ)",
        color=colors["weighted_absolute"],
        alpha=0.8,
    )
    ax4.bar(
        x + 1.5 * width,
        gaps_all[3],
        width,
        label="Weighted + Scaled (best τ)",
        color=colors["weighted_scaled"],
        alpha=0.8,
    )

    ax4.set_xlabel("Target Coverage", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Coverage Gap", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods_labels)
    ax4.legend(fontsize=10, ncol=4, loc="upper right")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_title(
        "Overall Comparison: All Configurations", fontsize=14, fontweight="bold"
    )

    # Overall title
    fig.suptitle(
        "Absolute vs Scaled Deviation Conformity Scores: Comprehensive Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save
    output_path = output_dir / "absolute_vs_scaled_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to {output_path}")

    plt.close()


def print_experiment_summary(df_results: pd.DataFrame):
    """Print summary of experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80 + "\n")

    alpha_values = sorted(df_results["alpha_target"].unique())

    for alpha in alpha_values:
        print(f"Alpha = {alpha}:")
        print("-" * 80)

        # Binned absolute
        df_ba = df_results[
            (df_results["method"] == "binned")
            & (df_results["score_type"] == "absolute")
            & (df_results["alpha_target"] == alpha)
        ]
        if not df_ba.empty:
            row = df_ba.iloc[0]
            print(f"  Binned + Absolute:")
            print(
                f"    Coverage:        {row['coverage']:.3f}  (gap: {row['gap']:.3f})"
            )
            print(f"    RMSE:            {row['rmse']:.2f} MW")
            print(f"    Margin:          {row['margin_mw_mean']:.2f} MW")
            print(f"    Margin (norm):   {row['margin_normalized_mean']:.2f}σ  ← KEY")

        # Binned scaled
        df_bs = df_results[
            (df_results["method"] == "binned")
            & (df_results["score_type"] == "scaled")
            & (df_results["alpha_target"] == alpha)
        ]
        if not df_bs.empty:
            row = df_bs.iloc[0]
            print(f"  Binned + Scaled:")
            print(
                f"    Coverage:        {row['coverage']:.3f}  (gap: {row['gap']:.3f})"
            )
            print(f"    RMSE:            {row['rmse']:.2f} MW")
            print(f"    Margin:          {row['margin_mw_mean']:.2f} MW")
            print(f"    Margin (norm):   {row['margin_normalized_mean']:.2f}σ  ← KEY")

        # Weighted absolute (best)
        df_wa = df_results[
            (df_results["method"] == "weighted")
            & (df_results["score_type"] == "absolute")
            & (df_results["alpha_target"] == alpha)
        ]
        if not df_wa.empty:
            best = df_wa.loc[df_wa["gap"].idxmin()]
            print(f"  Weighted + Absolute (best τ={best['tau']:.1f}):")
            print(
                f"    Coverage:        {best['coverage']:.3f}  (gap: {best['gap']:.3f})"
            )
            print(f"    RMSE:            {best['rmse']:.2f} MW")
            print(f"    Margin:          {best['margin_mw_mean']:.2f} MW")
            print(f"    Margin (norm):   {best['margin_normalized_mean']:.2f}σ  ← KEY")

        # Weighted scaled (best)
        df_ws = df_results[
            (df_results["method"] == "weighted")
            & (df_results["score_type"] == "scaled")
            & (df_results["alpha_target"] == alpha)
        ]
        if not df_ws.empty:
            best = df_ws.loc[df_ws["gap"].idxmin()]
            print(f"  Weighted + Scaled (best τ={best['tau']:.1f}):")
            print(
                f"    Coverage:        {best['coverage']:.3f}  (gap: {best['gap']:.3f})"
            )
            print(f"    RMSE:            {best['rmse']:.2f} MW")
            print(f"    Margin:          {best['margin_mw_mean']:.2f} MW")
            print(f"    Margin (norm):   {best['margin_normalized_mean']:.2f}σ  ← KEY")

        # Analysis
        print(f"\n  Analysis:")

        # Compare binned: absolute vs scaled
        if not df_ba.empty and not df_bs.empty:
            gap_diff = df_ba["gap"].iloc[0] - df_bs["gap"].iloc[0]
            margin_diff = (
                df_ba["margin_normalized_mean"].iloc[0]
                - df_bs["margin_normalized_mean"].iloc[0]
            )

            print(f"    Binned comparison:")
            if abs(gap_diff) < 0.01:
                print(f"      Coverage: Similar ({gap_diff:+.3f} gap difference)")
            elif gap_diff > 0:
                print(
                    f"      Coverage: Scaling helps (gap reduced by {abs(gap_diff):.3f})"
                )
            else:
                print(
                    f"      Coverage: Absolute better (gap reduced by {abs(gap_diff):.3f}) ✓"
                )

            print(
                f"      Normalized margin: {df_ba['margin_normalized_mean'].iloc[0]:.2f}σ (abs) vs "
                + f"{df_bs['margin_normalized_mean'].iloc[0]:.2f}σ (scaled)"
            )
            if margin_diff > 0:
                print(
                    f"      → Scaled uses {abs(margin_diff):.2f}σ LESS margin (less conservative) ✓"
                )
            else:
                print(
                    f"      → Absolute uses {abs(margin_diff):.2f}σ LESS margin (less conservative) ✓"
                )

        # Compare weighted: absolute vs scaled
        if not df_wa.empty and not df_ws.empty:
            best_abs = df_wa.loc[df_wa["gap"].idxmin()]
            best_scaled = df_ws.loc[df_ws["gap"].idxmin()]
            gap_diff = best_abs["gap"] - best_scaled["gap"]
            margin_diff = (
                best_abs["margin_normalized_mean"]
                - best_scaled["margin_normalized_mean"]
            )

            print(f"    Weighted comparison:")
            if abs(gap_diff) < 0.01:
                print(f"      Coverage: Similar ({gap_diff:+.3f} gap difference)")
            elif gap_diff > 0:
                print(
                    f"      Coverage: Scaling helps (gap reduced by {abs(gap_diff):.3f})"
                )
            else:
                print(
                    f"      Coverage: Absolute better (gap reduced by {abs(gap_diff):.3f}) ✓"
                )

            print(
                f"      Normalized margin: {best_abs['margin_normalized_mean']:.2f}σ (abs) vs "
                + f"{best_scaled['margin_normalized_mean']:.2f}σ (scaled)"
            )
            if margin_diff > 0:
                print(
                    f"      → Scaled uses {abs(margin_diff):.2f}σ LESS margin (less conservative) ✓"
                )
            else:
                print(
                    f"      → Absolute uses {abs(margin_diff):.2f}σ LESS margin (less conservative) ✓"
                )

        print()

    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("1. If absolute ≈ scaled: ens_std scaling provides little benefit")
    print("2. If absolute < scaled: ens_std scaling is hurting coverage")
    print("3. If weighted ≈ binned: localization provides little benefit")
    print("4. If weighted is too conservative: consider absolute scores only")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run experiments
    df_results = run_absolute_vs_scaled_experiments(
        alpha_values=[0.90, 0.95, 0.99],
        tau_values=[0.5, 1.0, 2.0, 5.0, 10.0],
        n_bins=5,
        omega_path="data/viz_artifacts/focused_2d/best_omega.npy",
    )

    print("\n✓ Experiments complete!")
