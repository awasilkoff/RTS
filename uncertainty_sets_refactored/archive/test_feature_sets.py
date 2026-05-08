"""
Quick smoke test for new feature builders.

Verifies that all three feature sets load correctly and produce expected dimensions.
"""
from pathlib import Path
import pandas as pd

from data_processing_extended import (
    build_XY_temporal_nuisance_3d,
    build_XY_per_resource_4d,
    build_XY_unscaled_2d,
)


def test_feature_sets():
    """Test all three feature sets."""
    DATA_DIR = Path(__file__).parent / "data"

    forecasts = pd.read_parquet(DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet")
    actuals = pd.read_parquet(DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet")

    print("Testing feature set builders...\n")

    # Test 1: Temporal nuisance 3D
    print("1. Temporal nuisance 3D (SYS_MEAN, SYS_STD, HOUR_SIN)")
    X, Y, times, x_cols, y_cols = build_XY_temporal_nuisance_3d(forecasts, actuals)
    print(f"   X shape: {X.shape} (expected: (N, 3))")
    print(f"   Y shape: {Y.shape}")
    print(f"   Features: {x_cols}")
    print(f"   Expected: ['SYS_MEAN', 'SYS_STD', 'HOUR_SIN']")
    assert X.shape[1] == 3, "Expected 3 features"
    assert x_cols == ["SYS_MEAN", "SYS_STD", "HOUR_SIN"], "Unexpected feature names"
    print("   (ok) PASS\n")

    # Test 2: Per-resource 4D
    print("2. Per-resource 4D (3 wind farms + HOUR_SIN)")
    X, Y, times, x_cols, y_cols = build_XY_per_resource_4d(forecasts, actuals, include_hour=True)
    print(f"   X shape: {X.shape} (expected: (N, 4))")
    print(f"   Y shape: {Y.shape}")
    print(f"   Features: {x_cols}")
    print(f"   Expected: 3 wind farm means + HOUR_SIN")
    assert X.shape[1] == 4, "Expected 4 features"
    assert all("_MEAN" in c or c == "HOUR_SIN" for c in x_cols), "Unexpected feature pattern"
    print("   (ok) PASS\n")

    # Test 3: Per-resource 3D (without hour)
    print("3. Per-resource 3D (3 wind farms, no hour)")
    X, Y, times, x_cols, y_cols = build_XY_per_resource_4d(forecasts, actuals, include_hour=False)
    print(f"   X shape: {X.shape} (expected: (N, 3))")
    print(f"   Features: {x_cols}")
    assert X.shape[1] == 3, "Expected 3 features"
    assert "HOUR_SIN" not in x_cols, "HOUR_SIN should not be present"
    print("   (ok) PASS\n")

    # Test 4: Unscaled 2D
    print("4. Unscaled 2D (SYS_MEAN_MW, SYS_STD_MW)")
    X, Y, times, x_cols, y_cols = build_XY_unscaled_2d(forecasts, actuals)
    print(f"   X shape: {X.shape} (expected: (N, 2))")
    print(f"   Y shape: {Y.shape}")
    print(f"   Features: {x_cols}")
    print(f"   Feature ranges (raw MW units):")
    print(f"     {x_cols[0]}: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}]")
    print(f"     {x_cols[1]}: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}]")
    assert X.shape[1] == 2, "Expected 2 features"
    assert X[:, 0].mean() > 100, "SYS_MEAN should be in MW scale (not standardized)"
    print("   (ok) PASS\n")

    print("=" * 60)
    print("All feature set tests passed! (ok)")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_sets()
