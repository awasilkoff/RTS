# Bug Fix: Omega Reconstruction from DataFrame

## The Bug

**File:** `sweep_and_viz_feature_set.py` (line 245-246)

**Problem:**
```python
# BUGGY CODE
omega_cols = [c for c in sweep_df.columns if c.startswith("omega_")]
omega_best = np.array([best_row[c] for c in omega_cols], dtype=float)
```

This incorrectly matches **both**:
- `omega_0_SYS_MEAN`, `omega_1_SYS_STD` (✓ correct - per-feature omega values)
- `omega_l2_reg` (✗ wrong - regularization hyperparameter!)

**Result:**
- `omega_best` had extra dimension: `[0.01, 9.5, 12.3]` instead of `[9.5, 12.3]`
- Plotting failed: tried to access `x_cols[2]` when only 2 features exist
- Omega bar chart showed wrong values

## The Fix

**Option A (Implemented): Strict Regex**
```python
# FIXED CODE
import re

omega_cols = [c for c in sweep_df.columns if re.match(r"^omega_\d+_", c)]
omega_cols = sorted(omega_cols, key=lambda c: int(c.split("_")[1]))  # Sort by index
omega_best = np.array([best_row[c] for c in omega_cols], dtype=float)

# Sanity check
assert omega_best.shape[0] == len(x_cols), (
    f"Omega dimension mismatch: got {omega_best.shape[0]} "
    f"but expected {len(x_cols)} features"
)
```

**Why this works:**
- Pattern `r"^omega_\d+_"` matches `omega_<digit>_` only
- `omega_l2_reg` doesn't match (no digit after omega)
- Sorting ensures correct order even with 10+ features

## Testing

Run the verification test:
```bash
python test_omega_reconstruction.py
```

Expected output:
```
✓ All checks passed!
✓ 4D test passed!
✓ Sorting test passed!
All omega reconstruction tests passed! ✓
```

## Alternative Fixes (Not Implemented)

**Option B: Explicit Exclusion**
```python
omega_cols = [
    c for c in sweep_df.columns
    if c.startswith("omega_") and c != "omega_l2_reg"
]
```
**Downside:** Fragile - breaks if we add other `omega_*` metadata columns

**Option C: Store omega_vec Directly**
```python
# In loop:
row["omega_vec"] = omega_hat.copy()

# After sweep:
omega_best = best_row["omega_vec"]
```
**Downside:** DataFrame stores as object, CSV serialization doesn't preserve numpy arrays cleanly

## Other Fixes in This Commit

**Fixed `best_row_idx`:**
```python
# Before (wrong - gives original index, not rank)
best_row_idx = sweep_df.index[0]

# After (correct - gives rank in sorted df)
best_row_idx = 0
```

## Files Modified

1. ✓ `sweep_and_viz_feature_set.py` - Fixed omega reconstruction + added sanity check
2. ✓ `test_omega_reconstruction.py` - Created comprehensive test
3. ✓ `BUGFIX_OMEGA_RECONSTRUCTION.md` - This documentation

## Root Cause

**Naming collision:** Both hyperparameters and learned values use `omega_*` prefix:
- Hyperparameter: `omega_l2_reg`
- Learned values: `omega_0_<feature>`, `omega_1_<feature>`, ...

**Better naming convention (future):**
- Use `omega_reg` instead of `omega_l2_reg` for hyperparameter
- Or use `learned_omega_*` for per-feature values

But the regex fix is robust and doesn't require renaming.

## Verification Checklist

After running sweeps, verify:

1. ✓ Check omega shape matches number of features:
   ```python
   omega = np.load("best_omega.npy")
   print(f"Omega shape: {omega.shape}")  # Should be (n_features,)
   ```

2. ✓ Check first value is NOT omega_l2_reg:
   ```python
   # Should be feature weight, not 0.01 or similar reg value
   print(f"First omega value: {omega[0]}")  # Should be ~1-10, not 0.01
   ```

3. ✓ Plotting should work without index errors:
   ```python
   # This should not crash:
   plot_omega_bar_chart(omega, x_cols, ...)
   ```

## Impact

**Before fix:**
- Sweeps would complete but omega_best had wrong dimension
- Plotting would crash with index errors
- Results were invalid

**After fix:**
- Omega reconstruction is correct
- Plotting works
- Sanity check catches future issues

This was a critical bug that would have invalidated all sweep results!
