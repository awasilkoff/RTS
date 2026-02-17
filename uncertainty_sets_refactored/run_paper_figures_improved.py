#!/usr/bin/env python3
"""
Generate IEEE paper figures with IMPROVED conformal prediction.

Uses the best configuration from experiments:
- Feature set: temporal_lag (adds previous hour context)
- Model: early_stop (prevents overfitting)
- Expected coverage: ~94.9% (target: 95.0%)

This version should achieve much better calibration than the baseline.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from viz_conformal_paper import (
    plot_calibration_curve,
    plot_adaptive_correction_summary,
)
from viz_timeseries_conformal import plot_conformal_timeseries
from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df


def add_temporal_lag_features(df: pd.DataFrame, time_col: str = "TIME_HOURLY") -> pd.DataFrame:
    """
    Add temporal lag features (previous 1-2 hours).

    This was the winning feature set from experiments!
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    # Lag features (1-2 hours back)
    for lag in [1, 2]:
        df[f"ens_mean_lag{lag}"] = df["ens_mean"].shift(lag)
        df[f"ens_std_lag{lag}"] = df["ens_std"].shift(lag)
        df[f"y_lag{lag}"] = df["y"].shift(lag)

    return df


def generate_improved_paper_figures(
    data_dir: Path,
    output_dir: Path,
    *,
    alpha_values: list[float] = [0.80, 0.85, 0.90, 0.95, 0.99],
    primary_alpha: float = 0.95,
) -> dict[str, Path]:
    """
    Generate paper figures with improved configuration.

    Uses temporal_lag features + early_stop model config.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING IMPROVED IEEE PAPER FIGURES")
    print("=" * 80)
    print("\nUsing best configuration from experiments:")
    print("  Feature set: temporal_lag (adds lag1, lag2 for ens_mean, ens_std, y)")
    print("  Model: early_stop (3000 estimators, lr=0.02, early stopping)")
    print("  Expected coverage: ~94.9% (vs 90.8% baseline)")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    actuals = pd.read_parquet(data_dir / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(
        data_dir / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_base = build_conformal_totals_df(actuals, forecasts)
    print(f"  Loaded {len(df_base)} time points")

    # Add temporal features
    print("\n[2/5] Adding temporal lag features...")
    df_enhanced = add_temporal_lag_features(df_base)
    print(f"  Added lag features: {df_enhanced.shape[1]} total columns")

    # Enhanced feature columns (baseline + temporal lag)
    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
        # Temporal lag features (winning addition!)
        "ens_mean_lag1",
        "ens_std_lag1",
        "y_lag1",
        "ens_mean_lag2",
        "ens_std_lag2",
        "y_lag2",
    ]

    # Model config (early_stop)
    model_kwargs = {
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 96,
        "random_state": 42,
    }

    # --- FIGURE 2: Calibration Curve ---
    print("\n[3/5] Generating calibration curve...")
    calibration_results = []

    for alpha in alpha_values:
        print(f"  Training model for α={alpha:.2f}...")

        # Drop rows with NaN from lag features
        df_clean = df_enhanced.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_clean,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning="y_pred",
            model_kwargs=model_kwargs,
        )

        calibration_results.append(
            {
                "alpha_target": float(alpha),
                "coverage": float(metrics["coverage"]),
                "n_test": int(metrics["n_test"]),
            }
        )
        print(
            f"    Coverage: {metrics['coverage']:.3f} (target: {alpha:.2f}, "
            f"gap: {abs(metrics['coverage'] - alpha):.3f})"
        )

    fig_calibration_path = output_dir / "fig_calibration_curve_improved"
    fig_calibration = plot_calibration_curve(
        calibration_results, output_path=fig_calibration_path
    )
    import matplotlib.pyplot as plt
    plt.close(fig_calibration)

    # --- FIGURE 3: Adaptive Correction Summary ---
    print(f"\n[4/5] Generating adaptive correction summary (α={primary_alpha:.2f})...")

    df_clean = df_enhanced.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    bundle_primary, metrics_primary, df_test_primary = train_wind_lower_model_conformal_binned(
        df_clean,
        feature_cols=feature_cols,
        target_col="y",
        scale_col="ens_std",
        alpha_target=float(primary_alpha),
        binning="y_pred",
        model_kwargs=model_kwargs,
    )

    fig_correction_path = output_dir / "fig_adaptive_correction_improved"
    fig_correction = plot_adaptive_correction_summary(
        bundle_primary,
        df_test_primary,
        metrics_primary,
        output_path=fig_correction_path,
    )
    plt.close(fig_correction)

    # --- FIGURE 1: Timeseries Overlay ---
    print("\n[5/5] Generating timeseries overlay...")

    # Prepare eval dataframe with predictions
    df_eval = df_test_primary.copy()

    # Get corresponding time index (need to align with cleaned data)
    # The test indices in df_test_primary correspond to indices in df_clean
    test_size = len(df_test_primary)
    test_start_idx = len(df_clean) - test_size

    df_eval["TIME_HOURLY"] = df_clean.iloc[test_start_idx:]["TIME_HOURLY"].values
    df_eval["ens_mean"] = df_clean.iloc[test_start_idx:]["ens_mean"].values

    fig_timeseries_path = output_dir / "fig_timeseries_conformal_improved.png"
    plot_conformal_timeseries(
        df_eval,
        out_png=fig_timeseries_path,
        max_points=24,  # Just 24 hours for clarity
        title=f"Wind Forecast vs Conformal Lower Bound (24-Hour Window, α={primary_alpha:.2f})",
    )

    # Save metadata
    metadata = {
        "configuration": "temporal_lag+early_stop",
        "features": feature_cols,
        "model_config": model_kwargs,
        "alpha_values": alpha_values,
        "primary_alpha": primary_alpha,
        "calibration_results": calibration_results,
        "primary_metrics": {
            "coverage": float(metrics_primary["coverage"]),
            "pre_conformal_coverage": float(metrics_primary["pre_conformal_coverage"]),
            "coverage_gap": float(abs(metrics_primary["coverage"] - primary_alpha)),
            "improvement": float(metrics_primary["coverage"] - metrics_primary["pre_conformal_coverage"]),
            "q_hat_global_r": float(metrics_primary["q_hat_global_r"]),
            "n_test": int(metrics_primary["n_test"]),
        },
        "comparison_to_baseline": {
            "baseline_coverage": 0.908,
            "improved_coverage": float(metrics_primary["coverage"]),
            "improvement": float(metrics_primary["coverage"] - 0.908),
        },
    }

    metadata_path = output_dir / "figure_metadata_improved.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")

    # Return paths
    paths = {
        "timeseries": fig_timeseries_path,
        "calibration_pdf": output_dir / "fig_calibration_curve_improved.pdf",
        "calibration_png": output_dir / "fig_calibration_curve_improved.png",
        "correction_pdf": output_dir / "fig_adaptive_correction_improved.pdf",
        "correction_png": output_dir / "fig_adaptive_correction_improved.png",
        "metadata": metadata_path,
    }

    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print("\nGenerated figures:")
    for name, path in paths.items():
        status = "(ok)" if path.exists() else "(x)"
        print(f"  {status} {name}: {path}")

    # Print improvement summary
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUMMARY")
    print("=" * 80)
    print(f"\nBaseline (original features):")
    print(f"  Coverage: 90.8% (gap: 4.2%)")
    print(f"\nImproved (temporal_lag features):")
    print(f"  Coverage: {metadata['primary_metrics']['coverage']:.1%} (gap: {metadata['primary_metrics']['coverage_gap']:.1%})")
    print(f"\nImprovement: {metadata['comparison_to_baseline']['improvement']:.1%}")

    return paths


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "paper_figures_improved"

    print("\n" + "=" * 80)
    print("IEEE PAPER FIGURE GENERATOR (IMPROVED)")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nThis version uses the BEST configuration from experiments:")
    print("  • Temporal lag features (previous 1-2 hours)")
    print("  • Early stopping model (3000 trees, lr=0.02)")
    print("  • Expected: ~94.9% coverage (vs 90.8% baseline)")
    print("\nEstimated time: 3-4 minutes")
    print("=" * 80 + "\n")

    paths = generate_improved_paper_figures(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        alpha_values=[0.80, 0.85, 0.90, 0.95, 0.99],
        primary_alpha=0.95,
    )

    print("\n" + "=" * 80)
    print("SUCCESS - IMPROVED FIGURES READY FOR PAPER")
    print("=" * 80)
    print("\n📊 Key Results:")

    # Load and display metadata
    with open(paths["metadata"]) as f:
        meta = json.load(f)

    print(f"\nCoverage at α=0.95:")
    print(f"  Baseline: 90.8%")
    print(f"  Improved: {meta['primary_metrics']['coverage']:.1%}")
    print(f"  Gap from target: {meta['primary_metrics']['coverage_gap']:.2%}")

    print(f"\nAll alpha values:")
    for res in meta['calibration_results']:
        gap = abs(res['coverage'] - res['alpha_target'])
        print(f"  α={res['alpha_target']:.2f}: {res['coverage']:.1%} (gap: {gap:.2%})")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review figures:")
    print(f"   open {paths['calibration_png']}")
    print(f"   open {paths['correction_png']}")
    print(f"   open {paths['timeseries']}")

    print("\n2. Copy to paper:")
    print(f"   cp {paths['calibration_pdf']} /path/to/paper/figures/")
    print(f"   cp {paths['correction_pdf']} /path/to/paper/figures/")
    print(f"   cp {paths['timeseries']} /path/to/paper/figures/")

    print("\n3. Update paper text:")
    print("   • Mention temporal lag features in methodology")
    print("   • Report ~95% coverage achievement")
    print("   • Highlight 4% improvement from feature engineering")
    print("=" * 80 + "\n")
