#!/usr/bin/env python3
"""
Sweep conformal prediction hyperparameters to find optimal configuration.

Tests combinations of:
- binning_strategy: What feature to bin by (y_pred, ens_std, etc.)
- n_bins: Number of bins (1, 3, 5, 10, 30)
- bin_strategy: How to create bins (equal_width, quantile)
- safety_margin: Conservativeness buffer (0.0, 0.005, 0.01, 0.02, 0.03)

Outputs:
- CSV with all results ranked by coverage gap
- JSON with best configuration for each alpha value
"""
from pathlib import Path
import json
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm

from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df
from run_paper_figures_dayahead_valid import add_dayahead_valid_features


def run_conformal_sweep(
    data_dir: Path,
    output_dir: Path,
    *,
    alpha_values: list[float] = [0.85, 0.90, 0.95, 0.99],
    binning_strategies: list[str] | None = None,
    n_bins_values: list[int] | None = None,
    bin_strategies: list[str] | None = None,
    safety_margins: list[float] | None = None,
    max_value_filter: float | None = None,
) -> pd.DataFrame:
    """
    Run hyperparameter sweep for conformal prediction.

    Parameters
    ----------
    alpha_values : list[float]
        Coverage targets to test (e.g., [0.90, 0.95])
    binning_strategies : list[str]
        Features to bin by (e.g., ["y_pred", "ens_std"])
    n_bins_values : list[int]
        Number of bins to test (e.g., [1, 3, 5, 10])
    bin_strategies : list[str]
        Binning methods (e.g., ["equal_width", "quantile"])
    safety_margins : list[float]
        Safety margin values (e.g., [0.0, 0.005, 0.01, 0.02])
    max_value_filter : float | None
        Filter out data points above this value

    Returns
    -------
    results_df : pd.DataFrame
        All sweep results sorted by coverage gap
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default parameter grids
    if binning_strategies is None:
        binning_strategies = ["y_pred", "ens_std"]
    if n_bins_values is None:
        n_bins_values = [1, 3, 5, 10, 30]
    if bin_strategies is None:
        bin_strategies = ["equal_width", "quantile"]
    if safety_margins is None:
        safety_margins = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]

    print("=" * 80)
    print("CONFORMAL PREDICTION HYPERPARAMETER SWEEP")
    print("=" * 80)
    print(f"\nParameter grid:")
    print(f"  • alpha_values: {alpha_values}")
    print(f"  • binning_strategies: {binning_strategies}")
    print(f"  • n_bins_values: {n_bins_values}")
    print(f"  • bin_strategies: {bin_strategies}")
    print(f"  • safety_margins: {safety_margins}")

    # Calculate total combinations
    n_combos = (
        len(alpha_values)
        * len(binning_strategies)
        * len(n_bins_values)
        * len(bin_strategies)
        * len(safety_margins)
    )
    print(f"\nTotal configurations: {n_combos}")
    print(f"Estimated time: ~{n_combos * 2 / 60:.1f} minutes (2 sec per config)")
    print("=" * 80)

    # Load data once
    print("\n[1/3] Loading data...")
    actuals = pd.read_parquet(
        data_dir / "actuals_filtered_rts3_constellation_v1.parquet"
    )
    forecasts = pd.read_parquet(
        data_dir / "forecasts_filtered_rts3_constellation_v1.parquet"
    )
    df_base = build_conformal_totals_df(actuals, forecasts)

    if max_value_filter is not None:
        df_base = df_base[df_base["y"] <= max_value_filter].reset_index(drop=True)
        print(f"  Filtered to y <= {max_value_filter}: {len(df_base)} points")

    df_enhanced = add_dayahead_valid_features(df_base)
    print(f"  Loaded {len(df_enhanced)} time points")

    # Feature columns (day-ahead valid)
    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
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

    df_clean = df_enhanced.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    print(f"  After dropna: {len(df_clean)} points")

    # Model config
    model_kwargs = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 96,
        "random_state": 42,
    }

    # Run sweep
    print("\n[2/3] Running hyperparameter sweep...")
    results = []

    param_combos = list(
        itertools.product(
            alpha_values,
            binning_strategies,
            n_bins_values,
            bin_strategies,
            safety_margins,
        )
    )

    for alpha, binning_strat, n_bins, bin_strat, safety in tqdm(
        param_combos, desc="Sweep progress"
    ):
        # Convert parameters to conformal_prediction format
        if binning_strat == "y_pred":
            binning = "y_pred"
        else:
            binning = f"feature:{binning_strat}"

        if bin_strat == "equal_width":
            bin_params = {"n_bins": n_bins}
        else:  # quantile
            bin_quantiles = [i / n_bins for i in range(n_bins + 1)]
            bin_params = {"bin_quantiles": bin_quantiles}

        try:
            _, metrics, _ = train_wind_lower_model_conformal_binned(
                df_clean,
                feature_cols=feature_cols,
                target_col="y",
                scale_col="ens_std",
                alpha_target=float(alpha),
                binning=binning,
                model_kwargs=model_kwargs,
                **bin_params,
                safety_margin=safety,
            )

            coverage = metrics["coverage"]
            gap = abs(coverage - alpha)

            results.append(
                {
                    "alpha_target": alpha,
                    "binning_strategy": binning_strat,
                    "n_bins": n_bins,
                    "bin_strategy": bin_strat,
                    "safety_margin": safety,
                    "coverage": coverage,
                    "coverage_gap": gap,
                    "pre_conformal_coverage": metrics["pre_conformal_coverage"],
                    "improvement": coverage - metrics["pre_conformal_coverage"],
                    "q_hat_global": metrics["q_hat_global_r"],
                    "n_test": metrics["n_test"],
                    "within_5pct": gap < 0.05,
                }
            )

        except Exception as e:
            print(
                f"\n  ⚠️  Failed: {alpha=}, {binning_strat=}, {n_bins=}, {bin_strat=}, {safety=}"
            )
            print(f"      Error: {e}")
            continue

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["alpha_target", "coverage_gap"]).reset_index(
        drop=True
    )

    # Save results
    print("\n[3/3] Saving results...")
    results_path = output_dir / "conformal_sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # Find best config for each alpha
    best_configs = {}
    for alpha in alpha_values:
        alpha_results = results_df[results_df["alpha_target"] == alpha]
        if len(alpha_results) == 0:
            continue

        best = alpha_results.iloc[0]  # Already sorted by coverage_gap
        best_configs[f"alpha_{alpha:.2f}"] = {
            "alpha_target": float(alpha),
            "binning_strategy": str(best["binning_strategy"]),
            "n_bins": int(best["n_bins"]),
            "bin_strategy": str(best["bin_strategy"]),
            "safety_margin": float(best["safety_margin"]),
            "coverage": float(best["coverage"]),
            "coverage_gap": float(best["coverage_gap"]),
            "improvement": float(best["improvement"]),
            "within_5pct": bool(best["within_5pct"]),
        }

    best_configs_path = output_dir / "conformal_best_configs.json"
    with open(best_configs_path, "w") as f:
        json.dump(best_configs, f, indent=2)
    print(f"  Saved: {best_configs_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)

    for alpha in alpha_values:
        print(f"\n🎯 Alpha = {alpha:.2f}")
        print("-" * 80)

        alpha_results = results_df[results_df["alpha_target"] == alpha]
        if len(alpha_results) == 0:
            print("  No results")
            continue

        # Best config
        best = alpha_results.iloc[0]
        print(f"\n  ✅ BEST CONFIG:")
        print(f"     Binning strategy: {best['binning_strategy']}")
        print(f"     Bins: {best['n_bins']} ({best['bin_strategy']})")
        print(f"     Safety margin: {best['safety_margin']:.3f}")
        print(
            f"     Coverage: {best['coverage']:.3f} (gap: {best['coverage_gap']:.3f})"
        )
        print(f"     Within 5%: {'✓' if best['within_5pct'] else '✗'}")

        # Top 5 configs
        print(f"\n  📊 Top 5 configurations:")
        for i, row in alpha_results.head(5).iterrows():
            status = "✓" if row["within_5pct"] else "✗"
            print(
                f"     {status} {row['binning_strategy']:8s} | "
                f"{row['n_bins']:2d} {row['bin_strategy']:11s} bins | "
                f"safety={row['safety_margin']:.3f} | "
                f"gap={row['coverage_gap']:.3f}"
            )

        # Statistics
        n_within_5pct = alpha_results["within_5pct"].sum()
        print(f"\n  📈 {n_within_5pct}/{len(alpha_results)} configs within 5% of target")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n1. Review results:")
    print(f"   cat {results_path}")
    print(f"\n2. Use best configs in run_paper_figures_dayahead_valid.py:")
    print(f"   cat {best_configs_path}")
    print("\n3. Generate figures with optimal settings")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "conformal_sweep"

    print("\n" + "=" * 80)
    print("CONFORMAL PREDICTION HYPERPARAMETER SWEEP")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nThis will test all combinations of:")
    print("  • Binning strategies: y_pred, ens_std")
    print("  • Number of bins: 1, 3, 5, 10, 30")
    print("  • Bin strategies: equal_width, quantile")
    print("  • Safety margins: 0.0, 0.005, 0.01, 0.015, 0.02, 0.03")
    print("\nEstimated time: ~10-15 minutes")
    print("=" * 80 + "\n")

    # Run sweep with default parameters
    results_df = run_conformal_sweep(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        alpha_values=[0.85, 0.90, 0.95, 0.99],
        binning_strategies=["ens_std"],
        n_bins_values=[
            1,
        ],
        bin_strategies=[
            "equal_width",
        ],
        safety_margins=[
            0.0,
        ],
        max_value_filter=None,
    )

    print("\n✅ Sweep complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
