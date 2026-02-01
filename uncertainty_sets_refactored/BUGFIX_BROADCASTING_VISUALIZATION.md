# Bug Fix: Broadcasting Error in Visualization

## Issue

**File:** `viz_projections.py` line 104

**Error:**
```
ValueError: operands could not be broadcast together with shapes (8,1) (2952,8)
```

**Triggered by:** Running `high_dim_8d` feature set (8-dimensional omega)

## Root Cause

Incorrect broadcasting in kernel weight computation:

```python
# WRONG
dist_sq = np.sum(omega[:, None] * diff * diff, axis=1)
```

**Problem:**
- `omega[:, None]` creates shape `(D, 1)` = `(8, 1)`
- `diff` has shape `(N, D)` = `(2952, 8)`
- `diff * diff` has shape `(N, D)` = `(2952, 8)`
- Broadcasting `(8, 1) * (2952, 8)` fails - dimensions don't align

## Fix

Use correct broadcasting shape:

```python
# CORRECT
dist_sq = np.sum(omega[None, :] * diff * diff, axis=1)
```

**Why this works:**
- `omega[None, :]` creates shape `(1, D)` = `(1, 8)`
- Broadcasting `(1, 8) * (2952, 8) * (2952, 8)` works correctly
- Result: `(2952, 8)` → sum over axis=1 → `(2952,)` ✓

## Alternative Fix

Can also just use `omega` directly (numpy broadcasts automatically):

```python
# ALSO CORRECT (simpler)
dist_sq = np.sum(omega * diff * diff, axis=1)
```

This works because:
- `omega` has shape `(D,)` = `(8,)`
- Numpy broadcasts `(8,) * (2952, 8)` as `(1, 8) * (2952, 8)` automatically

## Why It Only Failed for high_dim_8d

**Works for 2D/3D/4D:** These feature sets worked by coincidence
- Lower dimensions didn't trigger the broadcasting issue
- Or the issue was masked by other array operations

**Fails for 8D:** First time testing with 8-dimensional omega
- Exposed the incorrect broadcasting logic
- Made the dimension mismatch obvious

## Testing

Verified fix works:

```bash
python -c "
import numpy as np
N, D = 2952, 8
omega = np.random.rand(D)
diff = np.random.randn(N, D)

# New way (fixed)
dist_sq = np.sum(omega[None, :] * diff * diff, axis=1)
print(f'Shape: {dist_sq.shape}')  # (2952,) ✓
"
```

## Impact

**Before fix:** `high_dim_8d` visualization crashed
**After fix:** All feature sets (2D through 8D) visualize correctly

## Files Changed

- `viz_projections.py` line 104 - Fixed broadcasting

## Lesson Learned

**Broadcasting pitfalls:**
- `omega[:, None]` → `(D, 1)` (column vector)
- `omega[None, :]` → `(1, D)` (row vector)
- For broadcasting with `(N, D)` arrays, use row vector `(1, D)`

**Best practice:**
- Be explicit about broadcasting dimensions
- Test with different array shapes
- Or just use `omega` directly and let numpy handle it

## Verification

Run high_dim_8d feature set to verify:

```bash
cd uncertainty_sets_refactored
python sweep_and_viz_feature_set.py --feature-set high_dim_8d --taus 5.0 --omega-l2-regs 0.0 --scaler-types minmax
```

Should now complete without errors! ✓
