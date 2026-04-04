#!/usr/bin/env python3
"""
Test day-ahead compatible conformal prediction feature sets.

This script tests ONLY feature sets that are available at day-ahead time,
meaning no features that require intra-day actual values or rolling computations.

Day-Ahead Safe Feature Sets:
1. baseline: Basic ensemble features (ens_mean, ens_std, etc.)
2. cyclical_time: + sin/cos encoding for hour/dow
3. day_parts: + time-of-day indicators
4. forecast_dispersion: + forecast range features

NOT Day-Ahead Safe (excluded):
- temporal_lag: Uses y_lag (actual values from recent hours)
- rolling_stats: Uses rolling computations that could leak intra-day info
- combined: Includes lag features

Configuration:
- n_bins=1: Global conformal (single bin, no adaptive binning)
- safety_margin=0.0: No extra conservativeness buffer
- Data source: RTS4 (rts4_constellation_v1)
"""
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df


def add_dayahead_features(
    df: pd.DataFrame, time_col: str = "TIME_HOURLY"
) -> pd.DataFrame:
    """
    Add day-ahead compatible features to improve conformal prediction.

    Only adds features that are available at day-ahead time:
    - Cyclical time encodings (from timestamp, known in advance)
    - Day part indicators (from timestamp, known in advance)
    - Forecast dispersion features (from ensemble forecasts, available day-ahead)
    - Same-hour historical actuals (y_lag24, y_lag48, y_lag168)
    - Adjacent hour forecasts (intra-day forecast relationships)
    - Daily aggregate features (computed from day D forecasts)

    Does NOT add:
    - Short-term lag features (y_lag1, y_lag2 - not available for intra-day hours)
    - Rolling statistics on actuals (could leak intra-day information)
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Cyclical time encodings
    # -------------------------------------------------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # -------------------------------------------------------------------------
    # Day part indicators
    # -------------------------------------------------------------------------
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 24)).astype(int)

    # -------------------------------------------------------------------------
    # Forecast dispersion features (from ensemble, available day-ahead)
    # -------------------------------------------------------------------------
    df["forecast_range"] = df["ens_max"] - df["ens_min"]
    df["forecast_range_normalized"] = df["forecast_range"] / (df["ens_mean"] + 1e-6)
    df["forecast_cv"] = df["ens_std"] / (df["ens_mean"] + 1e-6)

    # -------------------------------------------------------------------------
    # Same-hour historical actuals (SAFE: from previous days)
    # -------------------------------------------------------------------------
    df["y_lag24"] = df["y"].shift(24)    # Same hour yesterday
    df["y_lag48"] = df["y"].shift(48)    # Same hour 2 days ago
    df["y_lag168"] = df["y"].shift(168)  # Same hour last week

    # Yesterday's forecast error at this hour
    df["forecast_error_lag24"] = df["y"].shift(24) - df["ens_mean"].shift(24)

    # -------------------------------------------------------------------------
    # Adjacent hour forecasts (SAFE: all from day D forecasts)
    # These are FORECAST values for adjacent hours, not actuals
    # -------------------------------------------------------------------------
    df["ens_mean_prev_hour"] = df["ens_mean"].shift(1)   # Forecast for hour t-1
    df["ens_mean_next_hour"] = df["ens_mean"].shift(-1)  # Forecast for hour t+1
    df["ens_std_prev_hour"] = df["ens_std"].shift(1)
    df["ens_std_next_hour"] = df["ens_std"].shift(-1)

    # Forecast gradient/trend (how forecast changes across hours)
    df["forecast_slope"] = df["ens_mean_next_hour"] - df["ens_mean_prev_hour"]
    df["forecast_slope_normalized"] = df["forecast_slope"] / (df["ens_mean"] + 1e-6)

    # -------------------------------------------------------------------------
    # Daily aggregate features (SAFE: computed from day D forecasts)
    # -------------------------------------------------------------------------
    df["_date"] = df[time_col].dt.date
    df["daily_mean_forecast"] = df.groupby("_date")["ens_mean"].transform("mean")
    df["daily_max_forecast"] = df.groupby("_date")["ens_mean"].transform("max")
    df["daily_min_forecast"] = df.groupby("_date")["ens_mean"].transform("min")
    df["daily_std_forecast"] = df.groupby("_date")["ens_mean"].transform("std")

    # How this hour compares to daily average
    df["hour_vs_daily_mean"] = df["ens_mean"] - df["daily_mean_forecast"]
    df["hour_vs_daily_mean_normalized"] = df["hour_vs_daily_mean"] / (df["daily_mean_forecast"] + 1e-6)

    # Clean up
    df = df.drop(columns=["_date"])

    return df


def get_dayahead_feature_sets() -> dict[str, list[str]]:
    """
    Define day-ahead compatible feature set configurations.

    All feature sets here use only information available at day-ahead time.

    Returns
    -------
    feature_sets : dict[str, list[str]]
        Mapping of experiment name to feature column list
    """
    # Baseline: Basic ensemble features only
    baseline = ["ens_mean", "ens_std", "ens_min", "ens_max", "n_models", "hour", "dow"]

    # Add cyclical time encoding (periodic, handles wraparound)
    cyclical_time = baseline + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    # Add day part indicators (categorical time-of-day)
    day_parts = baseline + ["is_night", "is_morning", "is_afternoon", "is_evening"]

    # Add forecast dispersion features (BEST from previous run)
    forecast_dispersion = baseline + [
        "forecast_range",
        "forecast_range_normalized",
        "forecast_cv",
    ]

    # Same-hour historical actuals (y from previous days)
    historical_actuals = baseline + [
        "y_lag24",           # Same hour yesterday
        "y_lag48",           # Same hour 2 days ago
        "forecast_error_lag24",  # Yesterday's forecast error
    ]

    # Adjacent hour forecasts (intra-day forecast relationships)
    forecast_gradient = baseline + [
        "ens_mean_prev_hour",
        "ens_mean_next_hour",
        "forecast_slope",
        "forecast_slope_normalized",
    ]

    # Daily context (how this hour fits in the day)
    daily_context = baseline + [
        "daily_mean_forecast",
        "daily_max_forecast",
        "hour_vs_daily_mean",
        "hour_vs_daily_mean_normalized",
    ]

    # Forecast dispersion + historical (combine two promising sets)
    dispersion_plus_historical = baseline + [
        "forecast_range",
        "forecast_range_normalized",
        "forecast_cv",
        "y_lag24",
        "forecast_error_lag24",
    ]

    # Forecast dispersion + gradient (combine dispersion with trend)
    dispersion_plus_gradient = baseline + [
        "forecast_range",
        "forecast_range_normalized",
        "forecast_cv",
        "forecast_slope",
        "forecast_slope_normalized",
    ]

    # Kitchen sink: all day-ahead safe features
    all_features = list(
        set(
            baseline
            + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
            + ["is_night", "is_morning", "is_afternoon", "is_evening"]
            + ["forecast_range", "forecast_range_normalized", "forecast_cv"]
            + ["y_lag24", "y_lag48", "y_lag168", "forecast_error_lag24"]
            + ["ens_mean_prev_hour", "ens_mean_next_hour", "forecast_slope"]
            + ["daily_mean_forecast", "daily_max_forecast", "hour_vs_daily_mean"]
        )
    )

    return {
        "baseline": baseline,
        "cyclical_time": cyclical_time,
        "day_parts": day_parts,
        "forecast_dispersion": forecast_dispersion,
        "historical_actuals": historical_actuals,
        "forecast_gradient": forecast_gradient,
        "daily_context": daily_context,
        "dispersion_plus_historical": dispersion_plus_historical,
        "dispersion_plus_gradient": dispersion_plus_gradient,
        "all_features": all_features,
    }


def get_model_configs() -> dict[str, dict]:
    """
    Define LightGBM model configurations to test.

    Returns
    -------
    model_configs : dict[str, dict]
        Mapping of config name to LightGBM kwargs
    """
    # Default configuration
    default = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "random_state": 42,
    }

    # Regularized (L1 + L2)
    regularized = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
    }

    # Deeper trees, slower learning
    deep = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 64,
        "random_state": 42,
    }

    return {
        "default": default,
        "regularized": regularized,
        "deep": deep,
    }


def run_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_kwargs: dict,
    alpha_target: float = 0.95,
    quantile_alpha: float = 0.05,
    experiment_name: str = "experiment",
) -> dict[str, Any] | None:
    """
    Run single conformal prediction experiment and return metrics.

    Uses global conformal (n_bins=1) with no safety margin.
    """
    print(f"\n  Running: {experiment_name}")
    print(f"    Features: {len(feature_cols)} ({', '.join(feature_cols[:5])}...)")
    print(
        f"    Model: {model_kwargs.get('n_estimators')} trees, lr={model_kwargs.get('learning_rate')}"
    )

    # Drop rows with NaN in feature columns
    df_clean = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    print(f"    Samples: {len(df_clean)} (dropped {len(df) - len(df_clean)} with NaN)")

    if len(df_clean) < 100:
        print("    Too few samples, skipping")
        return None

    try:
        bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
            df_clean,
            feature_cols=feature_cols,
            target_col="y",
            scale_col="ens_std",
            alpha_target=alpha_target,
            quantile_alpha=quantile_alpha,
            binning="y_pred",
            n_bins=1,  # Global conformal (single bin)
            test_frac=0.25,
            cal_frac=0.25,
            model_kwargs=model_kwargs,
            safety_margin=0.0,  # No extra buffer
            split_method="random",
            random_seed=42,
        )

        coverage = metrics["coverage"]
        pre_coverage = metrics["pre_conformal_coverage"]
        coverage_gap = abs(coverage - alpha_target)

        print(
            f"    Coverage: {coverage:.3f} (target: {alpha_target}, gap: {coverage_gap:.3f})"
        )
        print(
            f"      Pre-conformal: {pre_coverage:.3f}, Improvement: {coverage - pre_coverage:.3f}"
        )

        return {
            "experiment_name": experiment_name,
            "n_features": len(feature_cols),
            "n_samples": len(df_clean),
            "coverage": coverage,
            "pre_conformal_coverage": pre_coverage,
            "coverage_gap": coverage_gap,
            "improvement": coverage - pre_coverage,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "q_hat_global_r": metrics["q_hat_global_r"],
            "n_test": metrics["n_test"],
        }

    except Exception as e:
        print(f"    Error: {e}")
        return None


def run_all_experiments(
    data_dir: Path,
    output_dir: Path,
    alpha_target: float = 0.95,
    use_rts4: bool = True,
) -> pd.DataFrame:
    """
    Run all day-ahead compatible feature set and model configuration experiments.

    Parameters
    ----------
    data_dir : Path
        Directory containing parquet data files
    output_dir : Path
        Directory for output files
    alpha_target : float
        Target coverage level (e.g., 0.95)
    use_rts4 : bool
        If True, use RTS4 data; otherwise use RTS3

    Returns
    -------
    results_df : pd.DataFrame
        Comparison of all experiments
    """
    print("=" * 80)
    print("DAY-AHEAD COMPATIBLE CONFORMAL PREDICTION EXPERIMENTS")
    print("=" * 80)
    print(f"Configuration: n_bins=1 (global), safety_margin=0.0, alpha={alpha_target}")

    # Load data
    print("\nLoading data...")
    data_suffix = "rts4" if use_rts4 else "rts3"
    actuals = pd.read_parquet(
        data_dir / f"actuals_filtered_{data_suffix}_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        data_dir / f"forecasts_filtered_{data_suffix}_constellation_v1.parquet"
    )
    df_base = build_conformal_totals_df(actuals, forecasts)
    print(f"  Data source: {data_suffix}")
    print(f"  Base data: {len(df_base)} time points")

    # Add day-ahead compatible features
    print("\nAdding day-ahead compatible features...")
    df_enhanced = add_dayahead_features(df_base)
    print(f"  Enhanced data: {df_enhanced.shape[1]} columns")

    # Get configurations
    feature_sets = get_dayahead_feature_sets()
    model_configs = get_model_configs()

    print(
        f"\nRunning {len(feature_sets)} x {len(model_configs)} = "
        f"{len(feature_sets) * len(model_configs)} experiments..."
    )

    results = []

    # Test each combination
    for feat_name, feat_cols in feature_sets.items():
        for model_name, model_kwargs in model_configs.items():
            exp_name = f"{feat_name}+{model_name}"

            result = run_experiment(
                df_enhanced,
                feature_cols=feat_cols,
                model_kwargs=model_kwargs,
                alpha_target=alpha_target,
                experiment_name=exp_name,
            )

            if result:
                result["feature_set"] = feat_name
                result["model_config"] = model_name
                result["data_source"] = data_suffix
                results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by coverage gap (smaller is better)
    results_df = results_df.sort_values("coverage_gap").reset_index(drop=True)

    return results_df


def plot_experiment_comparison(
    results_df: pd.DataFrame,
    output_path: Path,
    alpha_target: float = 0.95,
):
    """
    Plot comparison of experiment results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Coverage vs Target (top-left)
    ax = axes[0, 0]
    x_pos = np.arange(len(results_df))

    colors = [
        "green" if abs(row["coverage"] - alpha_target) <= 0.02 else "orange"
        for _, row in results_df.iterrows()
    ]

    ax.barh(x_pos, results_df["coverage"], color=colors, alpha=0.7, edgecolor="black")
    ax.axvline(
        alpha_target,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target ({alpha_target})",
    )
    ax.axvline(alpha_target - 0.02, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(alpha_target + 0.02, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_yticks(x_pos)
    ax.set_yticklabels(results_df["experiment_name"], fontsize=8)
    ax.set_xlabel("Empirical Coverage", fontweight="bold")
    ax.set_title("Coverage vs Target (Day-Ahead Features)", fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # 2. Coverage Gap (top-right)
    ax = axes[0, 1]
    ax.barh(
        x_pos,
        results_df["coverage_gap"],
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_yticks(x_pos)
    ax.set_yticklabels(results_df["experiment_name"], fontsize=8)
    ax.set_xlabel("Coverage Gap (|empirical - target|)", fontweight="bold")
    ax.set_title("Coverage Gap (Lower is Better)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # 3. Improvement over pre-conformal (bottom-left)
    ax = axes[1, 0]
    ax.barh(
        x_pos, results_df["improvement"], color="purple", alpha=0.7, edgecolor="black"
    )
    ax.set_yticks(x_pos)
    ax.set_yticklabels(results_df["experiment_name"], fontsize=8)
    ax.set_xlabel("Improvement over Pre-Conformal", fontweight="bold")
    ax.set_title("Coverage Improvement", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # 4. Feature count vs Coverage Gap (bottom-right)
    ax = axes[1, 1]
    scatter = ax.scatter(
        results_df["n_features"],
        results_df["coverage_gap"],
        c=results_df["coverage"],
        s=100,
        alpha=0.7,
        cmap="RdYlGn",
        edgecolors="black",
        vmin=alpha_target - 0.1,
        vmax=alpha_target + 0.05,
    )
    ax.set_xlabel("Number of Features", fontweight="bold")
    ax.set_ylabel("Coverage Gap", fontweight="bold")
    ax.set_title("Feature Count vs Performance", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Coverage")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved comparison plot: {output_path}")


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "conformal_dayahead_rts4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments with RTS4 data
    results_df = run_all_experiments(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        alpha_target=0.95,
        use_rts4=True,
    )

    # Save results
    results_csv = OUTPUT_DIR / "experiment_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results: {results_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print("TOP 5 BEST EXPERIMENTS (by coverage gap)")
    print("=" * 80)
    print(
        results_df.head(5)[
            ["experiment_name", "coverage", "coverage_gap", "improvement", "n_features"]
        ]
    )

    # Plot comparison
    plot_path = OUTPUT_DIR / "experiment_comparison.png"
    plot_experiment_comparison(results_df, plot_path, alpha_target=0.95)

    # Save best configuration
    best = results_df.iloc[0]
    best_config = {
        "experiment_name": best["experiment_name"],
        "feature_set": best["feature_set"],
        "model_config": best["model_config"],
        "coverage": float(best["coverage"]),
        "coverage_gap": float(best["coverage_gap"]),
        "improvement": float(best["improvement"]),
        "n_features": int(best["n_features"]),
        "data_source": best["data_source"],
        "config": {
            "n_bins": 1,
            "safety_margin": 0.0,
            "alpha_target": 0.95,
        },
    }

    best_config_path = OUTPUT_DIR / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved best config: {best_config_path}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nBest day-ahead configuration: {best['experiment_name']}")
    print(
        f"  Coverage: {best['coverage']:.3f} (target: 0.95, gap: {best['coverage_gap']:.3f})"
    )
    print(f"  Improvement: {best['improvement']:.3f}")
    print(f"  Features: {best['n_features']}")
    print(f"  Data source: {best['data_source']}")
    print("\nThis configuration is safe for day-ahead operation.")
