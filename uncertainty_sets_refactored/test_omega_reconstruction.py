"""
Test that omega reconstruction from DataFrame works correctly.

Verifies the regex fix that prevents omega_l2_reg from being included
in the reconstructed omega vector.
"""
import re
import numpy as np
import pandas as pd


def test_omega_reconstruction():
    """Test that omega columns are correctly extracted."""

    # Simulate a sweep results row with omega values
    test_row = {
        "standardize": True,
        "tau": 5.0,
        "omega_l2_reg": 0.01,  # ← This should NOT be included in omega
        "omega_0_SYS_MEAN": 9.5,
        "omega_1_SYS_STD": 12.3,
        "eval_nll_learned": 5.2,
        "eval_nll_baseline": 5.3,
        "nll_improvement": 0.1,
    }

    x_cols = ["SYS_MEAN", "SYS_STD"]  # 2 features
    expected_omega = np.array([9.5, 12.3])

    # Create DataFrame
    df = pd.DataFrame([test_row])

    # OLD METHOD (BUGGY) - would include omega_l2_reg
    omega_cols_buggy = [c for c in df.columns if c.startswith("omega_")]
    print(f"Buggy method found columns: {omega_cols_buggy}")
    assert "omega_l2_reg" in omega_cols_buggy, "Bug test: should include omega_l2_reg"

    # NEW METHOD (FIXED) - strict regex
    omega_cols_fixed = [c for c in df.columns if re.match(r"^omega_\d+_", c)]
    omega_cols_fixed = sorted(omega_cols_fixed, key=lambda c: int(c.split("_")[1]))
    print(f"Fixed method found columns: {omega_cols_fixed}")

    # Reconstruct omega
    omega_reconstructed = np.array([test_row[c] for c in omega_cols_fixed], dtype=float)

    # Verify
    print(f"Expected omega: {expected_omega}")
    print(f"Reconstructed omega: {omega_reconstructed}")

    assert len(omega_cols_fixed) == 2, f"Should find 2 omega columns, got {len(omega_cols_fixed)}"
    assert "omega_l2_reg" not in omega_cols_fixed, "Should NOT include omega_l2_reg"
    assert omega_cols_fixed == ["omega_0_SYS_MEAN", "omega_1_SYS_STD"], f"Wrong columns: {omega_cols_fixed}"
    assert np.allclose(omega_reconstructed, expected_omega), "Omega values don't match"
    assert omega_reconstructed.shape[0] == len(x_cols), "Omega dimension should match x_cols"

    print("✓ All checks passed!")
    print()

    # Test with 4D features
    print("Testing 4D features...")
    test_row_4d = {
        "omega_l2_reg": 0.01,
        "omega_0_WIND_122": 1.2,
        "omega_1_WIND_309": 5.3,
        "omega_2_WIND_317": 2.1,
        "omega_3_HOUR_SIN": 0.5,
    }
    df_4d = pd.DataFrame([test_row_4d])

    omega_cols_4d = [c for c in df_4d.columns if re.match(r"^omega_\d+_", c)]
    omega_cols_4d = sorted(omega_cols_4d, key=lambda c: int(c.split("_")[1]))
    omega_4d = np.array([test_row_4d[c] for c in omega_cols_4d], dtype=float)

    expected_4d = np.array([1.2, 5.3, 2.1, 0.5])

    print(f"4D omega columns: {omega_cols_4d}")
    print(f"4D omega values: {omega_4d}")

    assert len(omega_cols_4d) == 4, "Should find 4 omega columns"
    assert "omega_l2_reg" not in omega_cols_4d, "Should NOT include omega_l2_reg"
    assert np.allclose(omega_4d, expected_4d), "4D omega values don't match"

    print("✓ 4D test passed!")
    print()

    # Test sorting (make sure omega_10 comes after omega_9)
    print("Testing index sorting...")
    test_row_sorting = {
        "omega_2_feat": 2.0,
        "omega_10_feat": 10.0,
        "omega_1_feat": 1.0,
    }
    df_sort = pd.DataFrame([test_row_sorting])

    omega_cols_sort = [c for c in df_sort.columns if re.match(r"^omega_\d+_", c)]
    omega_cols_sort = sorted(omega_cols_sort, key=lambda c: int(c.split("_")[1]))

    print(f"Sorted columns: {omega_cols_sort}")
    expected_order = ["omega_1_feat", "omega_2_feat", "omega_10_feat"]
    assert omega_cols_sort == expected_order, f"Wrong sort order: {omega_cols_sort}"

    print("✓ Sorting test passed!")
    print()

    print("=" * 60)
    print("All omega reconstruction tests passed! ✓")
    print("=" * 60)
    print()
    print("The fix prevents omega_l2_reg from being included in omega.")
    print("The regex r'^omega_\\d+_' only matches omega_<digit>_<name>.")


if __name__ == "__main__":
    test_omega_reconstruction()
