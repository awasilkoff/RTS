#!/usr/bin/env python3
"""
Quick conformal prediction sweep for rapid testing (~3-5 minutes).

Smaller parameter grid than sweep_conformal_config.py for fast iteration.
"""
from pathlib import Path
from sweep_conformal_config import run_conformal_sweep


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "conformal_sweep_quick"

    print("\n" + "=" * 80)
    print("QUICK CONFORMAL PREDICTION SWEEP")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nReduced parameter grid for fast testing:")
    print("  • Binning strategies: y_pred, ens_std")
    print("  • Number of bins: 1, 3, 10")
    print("  • Bin strategies: equal_width, quantile")
    print("  • Safety margins: 0.0, 0.01, 0.02")
    print("\nTotal: 4 alphas x 2 binning x 3 n_bins x 2 bin_strat x 3 safety = 144 configs")
    print("Estimated time: ~5 minutes")
    print("=" * 80 + "\n")

    # Quick sweep with reduced grid
    results_df = run_conformal_sweep(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        alpha_values=[0.85, 0.90, 0.95, 0.99],
        binning_strategies=["y_pred", "ens_std"],
        n_bins_values=[1, 3, 10],  # Reduced: skip 5, 30
        bin_strategies=["equal_width", "quantile"],
        safety_margins=[0.0, 0.01, 0.02],  # Reduced: skip 0.005, 0.015, 0.03
        max_value_filter=200,
    )

    print("\n✅ Quick sweep complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nTo run full sweep: python sweep_conformal_config.py")
