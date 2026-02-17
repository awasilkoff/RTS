"""
Test that random split indexing works correctly.

Verifies that sample indices are valid after switching from temporal to random split.
"""
import numpy as np


def test_temporal_split_bug():
    """Demonstrate the bug with temporal split logic."""
    n = 2952
    n_train = int(0.75 * n)  # 2214

    # OLD LOGIC (temporal split)
    sample_idx_eval = n_train // 4  # 553 (middle of eval set)
    target_idx_old = n_train + sample_idx_eval  # 2214 + 553 = 2767 (ok) Valid

    # But with n=2952:
    sample_idx_eval_end = (n - n_train) // 2  # 369 (middle of 738 eval samples)
    target_idx_end = n_train + sample_idx_eval_end  # 2214 + 369 = 2583 (ok) Valid

    # Edge case that caused the bug:
    sample_idx_edge = (n - n_train)  # 738 (beyond eval set!)
    target_idx_edge = n_train + sample_idx_edge  # 2214 + 738 = 2952 (x) OUT OF BOUNDS!

    print("Temporal split (old logic):")
    print(f"  n={n}, n_train={n_train}, n_eval={n - n_train}")
    print(f"  Valid indices: 0-{n-1}")
    print(f"  Middle of eval: sample_idx={sample_idx_eval_end}, target_idx={target_idx_end} (ok)")
    print(f"  Edge case: sample_idx={sample_idx_edge}, target_idx={target_idx_edge} (x) OUT OF BOUNDS")
    print()


def test_random_split_fix():
    """Show the fix with random split."""
    n = 2952
    n_train = int(0.75 * n)

    # NEW LOGIC (random split)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = np.sort(indices[:n_train])
    eval_idx = np.sort(indices[n_train:])

    # Pick middle of eval set
    sample_idx_in_eval = len(eval_idx) // 2  # 369 (index within eval set)

    # OLD (BUGGY): Assumes eval starts at n_train
    target_idx_buggy = n_train + sample_idx_in_eval  # 2214 + 369 = 2583
    # This is wrong! eval_idx[369] might be 1523, not 2583

    # NEW (FIXED): Use actual eval indices
    target_idx_fixed = eval_idx[sample_idx_in_eval]  # Correct index in full dataset

    print("Random split (new logic):")
    print(f"  n={n}, n_train={n_train}, n_eval={len(eval_idx)}")
    print(f"  Eval indices are scattered: {eval_idx[:10]}...{eval_idx[-10:]}")
    print(f"  Middle of eval set: index {sample_idx_in_eval} within eval")
    print(f"  Buggy calculation: target_idx = n_train + {sample_idx_in_eval} = {target_idx_buggy}")
    print(f"  Fixed calculation: target_idx = eval_idx[{sample_idx_in_eval}] = {target_idx_fixed}")
    print()

    # Verify all eval indices are valid
    assert eval_idx.max() < n, f"Max eval index {eval_idx.max()} >= {n}"
    assert eval_idx.min() >= 0, f"Min eval index {eval_idx.min()} < 0"
    print(f"  (ok) All {len(eval_idx)} eval indices are valid (0-{n-1})")
    print()


def test_simple_fix():
    """Show the simplest fix for visualization."""
    n = 2952

    # SIMPLEST FIX: Just pick middle of full dataset
    sample_idx_simple = n // 2  # 1476 (always valid!)

    print("Simplest fix (for visualization):")
    print(f"  n={n}")
    print(f"  sample_idx = n // 2 = {sample_idx_simple} (ok) Always valid")
    print(f"  No need to track eval_idx if just picking a sample for viz")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Random Split Indexing Test")
    print("=" * 70)
    print()

    test_temporal_split_bug()
    test_random_split_fix()
    test_simple_fix()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("With temporal split:")
    print("  target_idx = n_train + sample_idx_in_eval  (ok) Works")
    print()
    print("With random split:")
    print("  target_idx = n_train + sample_idx_in_eval  (x) WRONG")
    print("  target_idx = eval_idx[sample_idx_in_eval]  (ok) Correct")
    print()
    print("For visualization only:")
    print("  sample_idx = len(X) // 2  (ok) Simplest (no eval_idx needed)")
    print()
