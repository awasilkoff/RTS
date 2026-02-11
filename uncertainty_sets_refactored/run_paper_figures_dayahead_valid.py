#!/usr/bin/env python3
"""
Generate IEEE paper figures with DAY-AHEAD VALID conformal prediction.

CRITICAL FIX: Only uses features available at day-ahead forecast time.
- NO actual generation features (y_lag1, y_lag2) - those are in the future!
- YES ensemble forecast features for current and adjacent hours
- YES time-of-day features

This version respects the day-ahead operational constraint.
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


def add_dayahead_valid_features(
    df: pd.DataFrame, time_col: str = "TIME_HOURLY"
) -> pd.DataFrame:
    """
    Add ONLY features available at day-ahead forecast time.

    Valid features:
    - Lagged ENSEMBLE forecasts (ens_mean_lag1, etc.) - these are forecasts for adjacent hours
    - Cyclical time encoding (hour_sin/cos, dow_sin/cos)
    - Rolling ensemble statistics (computed from forecast, not actuals)

    INVALID features (NOT included):
    - y_lag1, y_lag2 - actual generation is in the future!
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    # Lagged ENSEMBLE forecasts (valid if forecasts produced for all hours at once)
    for lag in [1, 2]:
        df[f"ens_mean_lag{lag}"] = df["ens_mean"].shift(lag)
        df[f"ens_std_lag{lag}"] = df["ens_std"].shift(lag)

    # Lead ENSEMBLE forecasts (next hour forecast, if available)
    df["ens_mean_lead1"] = df["ens_mean"].shift(-1)
    df["ens_std_lead1"] = df["ens_std"].shift(-1)

    # Cyclical hour encoding (handles midnight wraparound)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclical day of week encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Rolling statistics (computed from ENSEMBLE forecasts, not actuals)
    df["rolling_mean_3h"] = df["ens_mean"].rolling(window=3, min_periods=1).mean()
    df["rolling_std_3h"] = (
        df["ens_mean"].rolling(window=3, min_periods=1).std().fillna(0)
    )

    # Forecast range (dispersion measure)
    df["forecast_range"] = df["ens_max"] - df["ens_min"]
    df["forecast_range_normalized"] = df["forecast_range"] / (df["ens_mean"] + 1e-6)

    # Day part indicators
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)

    return df


def generate_dayahead_valid_figures(
    data_dir: Path,
    output_dir: Path,
    *,
    alpha_values: list[float] = [0.85, 0.90, 0.95, 0.99],
    viz_alphas: list[float] | None = None,
    n_bins: int = 3,
    bin_strategy: str = "equal_width",
    safety_margin: float = 0.005,
    max_value_filter: float | None = None,
    binning_strategy: str = "y_pred",
) -> dict[str, Path]:
    """
    Generate paper figures with DAY-AHEAD VALID features only.

    Parameters
    ----------
    alpha_values : list[float]
        Coverage targets for calibration curve
    viz_alphas : list[float] | None
        Alpha values to generate detailed visualizations for (timeseries + correction summary)
        If None, defaults to [0.95]
        Example: [0.90, 0.95, 0.99] will create 3 sets of detailed figures
    n_bins : int, default=3
        Number of bins for adaptive conformal correction
        Typical values: 3-10 (fewer bins = more stable, more bins = more adaptive)
    bin_strategy : str, default="equal_width"
        Strategy for creating bin boundaries
        - "equal_width": Bins span equal ranges of the binning feature
        - "quantile": Bins contain equal numbers of calibration datapoints
    safety_margin : float, default=0.005
        Safety margin for conformal quantile (0.005 = 0.5% extra conservativeness)
    max_value_filter : float | None, default=None
        If set, filter out any data points where y (actual generation) exceeds this value
        Use to remove suspicious/erroneous high values (e.g., max_value_filter=290)
    binning_strategy : str, default="y_pred"
        Strategy for creating adaptive bins
        - "y_pred": Bin by predicted value (default)
        - "y_actual": Bin by actual values during calibration, use predictions as proxy at prediction time
        - "ens_std": Bin by ensemble standard deviation
        - Can also specify any feature column name (e.g., "ens_mean", "hour")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert binning_strategy to BinningSpec format for conformal_prediction.py
    if binning_strategy == "y_pred":
        binning = "y_pred"
    elif binning_strategy == "y_actual":
        binning = "y_actual"
    else:
        binning = f"feature:{binning_strategy}"

    # Convert bin_strategy and n_bins to conformal_prediction parameters
    if bin_strategy == "equal_width":
        # Use n_bins parameter for equal-width bins
        bin_params = {"n_bins": n_bins}
    elif bin_strategy == "quantile":
        # Use bin_quantiles for equal-count bins
        # Generate n_bins+1 quantile points from 0 to 1
        bin_quantiles = [i / n_bins for i in range(n_bins + 1)]
        bin_params = {"bin_quantiles": bin_quantiles}
    else:
        raise ValueError(
            f"bin_strategy must be 'equal_width' or 'quantile', got '{bin_strategy}'"
        )

    # Default viz_alphas if not specified
    if viz_alphas is None:
        viz_alphas = [0.95]

    print("=" * 80)
    print("GENERATING DAY-AHEAD VALID IEEE PAPER FIGURES")
    print("=" * 80)
    print(
        "\n⚠️  CRITICAL FIX: Only using features available at day-ahead forecast time"
    )
    print("   ❌ NO y_lag features (actual generation in the future)")
    print("   ✅ YES ensemble forecast features (available at forecast time)")
    print("\n📊 Configuration:")
    print(f"   • Bins: {n_bins} {bin_strategy} bins")
    print(f"   • Binning strategy: {binning_strategy}")
    print(
        f"   • Safety margin: {safety_margin:.3f} ({safety_margin*100:.1f}% extra coverage)"
    )
    print(
        f"   • Max value filter: {max_value_filter if max_value_filter else 'None (no filtering)'}"
    )
    print(f"   • Viz alphas: {viz_alphas}")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    actuals = pd.read_parquet(
        data_dir / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        data_dir / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_base = build_conformal_totals_df(actuals, forecasts)
    print(f"  Loaded {len(df_base)} time points")

    # Filter out suspicious high values if requested
    if max_value_filter is not None:
        n_before = len(df_base)
        df_base = df_base[df_base["y"] <= max_value_filter].reset_index(drop=True)
        n_after = len(df_base)
        n_filtered = n_before - n_after
        print(f"  ⚠️  Filtered {n_filtered} suspicious points (y > {max_value_filter})")
        print(f"  Remaining: {n_after} time points")

    # Add day-ahead valid features
    print("\n[2/5] Adding day-ahead valid features...")
    df_enhanced = add_dayahead_valid_features(df_base)
    print(f"  Added valid features: {df_enhanced.shape[1]} total columns")

    # Feature sets to test
    print("\n[3/5] Testing feature configurations...")

    # Baseline: current features only
    baseline_features = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Enhanced: add cyclical encoding
    cyclical_features = baseline_features + [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    # Advanced: add ensemble lag/lead and rolling stats
    advanced_features = cyclical_features + [
        "ens_mean_lag1",
        "ens_std_lag1",
        "ens_mean_lag2",
        "ens_std_lag2",
        "ens_mean_lead1",
        "ens_std_lead1",
        "rolling_mean_3h",
        "rolling_std_3h",
        "forecast_range_normalized",
    ]

    # Use advanced features (most comprehensive valid set)
    feature_cols = advanced_features

    print(f"  Using {len(feature_cols)} features (all day-ahead valid):")
    for feat in feature_cols:
        print(f"    • {feat}")

    # Model config (tuned)
    model_kwargs = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 96,
        "random_state": 42,
    }

    # --- FIGURE 2: Calibration Curve ---
    print("\n[4/5] Generating calibration curve...")
    calibration_results = []

    for alpha in alpha_values:
        print(f"  Training model for α={alpha:.2f}...")

        df_clean = df_enhanced.dropna(subset=feature_cols + ["y"]).reset_index(
            drop=True
        )

        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_clean,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha),
            binning=binning,
            model_kwargs=model_kwargs,
            **bin_params,
            safety_margin=safety_margin,
        )

        calibration_results.append(
            {
                "alpha_target": float(alpha),
                "coverage": float(metrics["coverage"]),
                "n_test": int(metrics["n_test"]),
            }
        )
        gap = abs(metrics["coverage"] - alpha)
        print(
            f"    Coverage: {metrics['coverage']:.3f} (target: {alpha:.2f}, gap: {gap:.3f})"
        )

    fig_calibration_path = output_dir / "fig_calibration_curve_dayahead"
    fig_calibration = plot_calibration_curve(
        calibration_results, output_path=fig_calibration_path
    )
    import matplotlib.pyplot as plt

    plt.close(fig_calibration)

    # --- FIGURES 3 & 1: Generate for each viz_alpha ---
    print(
        f"\n[5/6] Generating detailed visualizations for {len(viz_alphas)} alpha values..."
    )

    viz_results = {}  # Store results for each alpha

    for i, alpha_viz in enumerate(viz_alphas, 1):
        print(f"\n  [{i}/{len(viz_alphas)}] Alpha = {alpha_viz:.2f}")
        print(f"  " + "-" * 60)

        df_clean = df_enhanced.dropna(subset=feature_cols + ["y"]).reset_index(
            drop=True
        )

        # Train model for this alpha
        bundle_viz, metrics_viz, df_test_viz = train_wind_lower_model_conformal_binned(
            df_clean,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=float(alpha_viz),
            binning=binning,
            model_kwargs=model_kwargs,
            **bin_params,
            safety_margin=safety_margin,
        )

        print(f"    Coverage: {metrics_viz['coverage']:.3f} (target: {alpha_viz:.2f})")

        # Format alpha for filename (replace dot with underscore to avoid Path issues)
        alpha_str = f"{alpha_viz:.2f}".replace(".", "_")

        # Adaptive correction summary
        fig_correction_path = output_dir / f"fig_adaptive_correction_alpha_{alpha_str}"
        fig_correction = plot_adaptive_correction_summary(
            bundle_viz,
            df_test_viz,
            metrics_viz,
            output_path=fig_correction_path,
        )
        plt.close(fig_correction)
        print(f"    ✓ Saved correction summary")

        # Timeseries overlay
        df_eval = df_test_viz.copy()
        test_size = len(df_test_viz)
        test_start_idx = len(df_clean) - test_size

        df_eval["TIME_HOURLY"] = df_clean.iloc[test_start_idx:]["TIME_HOURLY"].values
        df_eval["ens_mean"] = df_clean.iloc[test_start_idx:]["ens_mean"].values

        fig_timeseries_path = (
            output_dir / f"fig_timeseries_conformal_alpha_{alpha_str}.png"
        )
        plot_conformal_timeseries(
            df_eval,
            out_png=fig_timeseries_path,
            max_points=24,
            title=f"Day-Ahead Wind Forecast vs Conformal Bound (α={alpha_viz:.2f})",
        )
        print(f"    ✓ Saved timeseries (24h window)")

        # Store results
        viz_results[alpha_viz] = {
            "correction_pdf": output_dir
            / f"fig_adaptive_correction_alpha_{alpha_str}.pdf",
            "correction_png": output_dir
            / f"fig_adaptive_correction_alpha_{alpha_str}.png",
            "timeseries": fig_timeseries_path,
            "metrics": {
                "coverage": float(metrics_viz["coverage"]),
                "pre_conformal_coverage": float(metrics_viz["pre_conformal_coverage"]),
                "coverage_gap": float(abs(metrics_viz["coverage"] - alpha_viz)),
                "improvement": float(
                    metrics_viz["coverage"] - metrics_viz["pre_conformal_coverage"]
                ),
                "q_hat_global_r": float(metrics_viz["q_hat_global_r"]),
                "n_test": int(metrics_viz["n_test"]),
            },
        }

    # Save metadata
    metadata = {
        "configuration": "day_ahead_valid_advanced",
        "features": feature_cols,
        "model_config": model_kwargs,
        "validation_note": "All features available at day-ahead forecast time",
        "invalid_features_removed": ["y_lag1", "y_lag2"],
        "alpha_values": alpha_values,
        "viz_alphas": viz_alphas,
        "calibration_results": calibration_results,
        "visualization_results": {
            str(alpha): viz_results[alpha]["metrics"] for alpha in viz_alphas
        },
    }

    metadata_path = output_dir / "figure_metadata_dayahead.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")

    # Build paths dictionary with all generated files
    paths = {
        "calibration_pdf": output_dir / "fig_calibration_curve_dayahead.pdf",
        "calibration_png": output_dir / "fig_calibration_curve_dayahead.png",
        "metadata": metadata_path,
    }

    # Add visualization files for each alpha
    for alpha_viz in viz_alphas:
        alpha_str = f"{alpha_viz:.2f}".replace(".", "_")
        paths[f"timeseries_alpha_{alpha_str}"] = viz_results[alpha_viz]["timeseries"]
        paths[f"correction_pdf_alpha_{alpha_str}"] = viz_results[alpha_viz][
            "correction_pdf"
        ]
        paths[f"correction_png_alpha_{alpha_str}"] = viz_results[alpha_viz][
            "correction_png"
        ]

    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print("\nGenerated figures (DAY-AHEAD VALID):")
    for name, path in paths.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nGenerated visualizations for {len(viz_alphas)} alpha values:")
    for alpha_viz in viz_alphas:
        metrics_viz = viz_results[alpha_viz]["metrics"]
        print(f"\n  α={alpha_viz:.2f}:")
        print(
            f"    Coverage: {metrics_viz['coverage']:.1%} (gap: {metrics_viz['coverage_gap']:.2%})"
        )
        print(f"    Improvement: {metrics_viz['improvement']:.1%}")

    return paths


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "paper_figures_dayahead"

    print("\n" + "=" * 80)
    print("IEEE PAPER FIGURE GENERATOR (DAY-AHEAD VALID)")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\n⚠️  CRITICAL FIX:")
    print("   This version ONLY uses features available at day-ahead forecast time")
    print("   Removed: y_lag1, y_lag2 (actual generation not available)")
    print("   Added: Cyclical encoding, ensemble lag/lead, rolling stats")
    print("\nEstimated time: 3-4 minutes")
    print("=" * 80 + "\n")

    # Generate figures for multiple alpha values
    # alpha_values: for calibration curve
    # viz_alphas: detailed visualizations (timeseries + correction summary)
    # n_bins: number of bins for adaptive conformal
    # bin_strategy: "equal_width" (equal spacing) or "quantile" (equal counts)
    # binning_strategy: bin by "y_pred" (default), "y_actual", "ens_std" (or any feature)
    # safety_margin: conservativeness buffer (0.005 = 0.5% extra coverage)
    # max_value_filter: remove suspicious high values (e.g., 290)
    paths = generate_dayahead_valid_figures(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        alpha_values=[0.8, 0.85, 0.90, 0.95, 0.99],  # Calibration curve
        viz_alphas=[0.90, 0.95, 0.99],  # Detailed visualizations
        n_bins=1,  # ← ADJUST THIS to test different bin counts (e.g., 3, 5, 10, 30)
        bin_strategy="quantile",  # ← ADJUST THIS: "equal_width" or "quantile"
        binning_strategy="y_actual",  # ← ADJUST THIS: "y_pred", "y_actual", "ens_std", or any feature
        safety_margin=0.000,  # ← ADJUST THIS for conservativeness (e.g., 0.0, 0.005, 0.01, 0.02)
        max_value_filter=None,  # ← ADJUST THIS to filter suspicious high values (or None to disable)
    )

    print("\n" + "=" * 80)
    print("SUCCESS - DAY-AHEAD VALID FIGURES READY")
    print("=" * 80)

    with open(paths["metadata"]) as f:
        meta = json.load(f)

    print(f"\n📊 Results with day-ahead valid features:")

    print(f"\nCalibration curve (all alphas):")
    for res in meta["calibration_results"]:
        gap = abs(res["coverage"] - res["alpha_target"])
        status = "✓" if gap < 0.05 else "⚠"
        print(
            f"  {status} α={res['alpha_target']:.2f}: {res['coverage']:.1%} (gap: {gap:.2%})"
        )

    print(f"\nDetailed visualizations generated for:")
    for alpha_str, metrics_viz in meta["visualization_results"].items():
        alpha = float(alpha_str)
        gap = metrics_viz["coverage_gap"]
        status = "✓" if gap < 0.05 else "⚠"
        print(
            f"  {status} α={alpha:.2f}: {metrics_viz['coverage']:.1%} "
            f"(gap: {gap:.2%}, improvement: {metrics_viz['improvement']:.1%})"
        )

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review results above to see if performance is acceptable")
    print("2. If coverage gap is <5%, use these figures for paper")
    print("3. Update paper methodology to mention day-ahead valid features")
    print("\n4. View figures:")
    print(f"   Calibration curve:")
    print(f"     open {paths['calibration_png']}")
    print(f"\n   Detailed visualizations by alpha:")
    for alpha in meta["viz_alphas"]:
        alpha_str = f"{alpha:.2f}".replace(".", "_")
        print(f"\n   α={alpha:.2f}:")
        print(f"     open {paths[f'timeseries_alpha_{alpha_str}']}")
        print(f"     open {paths[f'correction_png_alpha_{alpha_str}']}")
    print("=" * 80 + "\n")
