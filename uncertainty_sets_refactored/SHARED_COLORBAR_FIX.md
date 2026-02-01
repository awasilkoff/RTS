# Shared Colorbar Scale for Direct Comparison

## Change Summary

Updated `viz_kernel_distance.py` to enforce the same colorbar scale across both subplots in `plot_kernel_distance_comparison()`. This allows direct visual comparison of kernel weights between equal and learned omega.

## Problem Before

```python
# Each subplot used its own independent scale
# Left plot:  vmin=0.098, vmax=1.000 (range of equal weights)
# Right plot: vmin=0.012, vmax=1.000 (range of learned weights)
```

**Issue:** Same color (e.g., yellow) could represent different kernel weights in each plot, making visual comparison misleading.

## Solution After

```python
# Both subplots share the same global scale
# Both plots: vmin=0.012, vmax=1.000 (global min/max across both omegas)
```

**Benefit:** Same color represents same kernel weight across both plots. Directly comparable.

## Changes Made

### 1. Added `vmin`/`vmax` parameters to `plot_kernel_distance()`

**Before:**
```python
def plot_kernel_distance(...):
    weights = compute_kernel_weights(X_target, X, omega, tau)

    scatter = ax.scatter(
        ...,
        norm=LogNorm(vmin=max(weights.min(), 1e-6), vmax=weights.max()),
    )
```

**After:**
```python
def plot_kernel_distance(..., vmin=None, vmax=None):
    """
    Parameters
    ----------
    vmin, vmax : float, optional
        Explicit colorbar limits. If provided, overrides auto-scaling.
        Useful for ensuring consistent scales across multiple plots.
    """
    weights = compute_kernel_weights(X_target, X, omega, tau)

    # Determine color scale limits
    if vmin is None:
        vmin = max(weights.min(), 1e-6)
    if vmax is None:
        vmax = weights.max()

    scatter = ax.scatter(
        ...,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
```

### 2. Compute global scale in `plot_kernel_distance_comparison()`

**Added:**
```python
# Compute kernel weights for both omegas
X_target = X[target_idx]
weights_equal = compute_kernel_weights(X_target, X, omega_equal, tau)
weights_learned = compute_kernel_weights(X_target, X, omega_learned, tau)

# Find global min/max for shared colorbar scale
vmin = max(min(weights_equal.min(), weights_learned.min()), 1e-6)
vmax = max(weights_equal.max(), weights_learned.max())

# Pass to both subplots
plot_kernel_distance(..., vmin=vmin, vmax=vmax)  # Left
plot_kernel_distance(..., vmin=vmin, vmax=vmax)  # Right
```

### 3. Fixed pre-existing bug in `plot_multiple_targets()`

**Before:**
```python
for i, target_idx in enumerate(target_indices):
    ...

# Bug: i undefined if target_indices is empty
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
```

**After:**
```python
for i, target_idx in enumerate(target_indices):
    ...

# Fixed: use len(target_indices) instead
n_used = len(target_indices)
for j in range(n_used, len(axes)):
    axes[j].set_visible(False)
```

## Visual Impact

### Example: focused_2d with equal vs learned omega

**Before fix:**
- Left plot (equal ω=[1,1]): Yellow = weight 0.5 (relative to equal range)
- Right plot (learned ω=[9.5, 12.3]): Yellow = weight 0.3 (relative to learned range)
- ❌ **Misleading:** Yellow means different things in each plot!

**After fix:**
- Left plot: Yellow = weight 0.5 (global scale)
- Right plot: Yellow = weight 0.5 (same global scale)
- ✅ **Clear:** Yellow means the same thing in both plots!

## How to Interpret Comparison Plots Now

With shared colorbar scale, you can directly compare:

1. **Brightness:** Brighter = higher kernel weight (more similar to target)
2. **Pattern differences:**
   - If learned omega shows brighter regions → learned metric found stronger similarities
   - If learned omega shows dimmer regions → learned metric found weaker similarities
3. **Spatial patterns:**
   - Different bright regions → learned omega prioritizes different features
   - Same bright regions but different intensity → learned omega rescales similarities

### Example Interpretation

```
Equal weights:     Learned weights:
[1.0, 1.0]        [2.0, 0.5]

Left plot:         Right plot:
Points spread      Points clustered
along diagonal     along Feature_1 axis

→ Learned omega prioritizes Feature_1 (weight=2.0)
→ Similarities are stronger along Feature_1 dimension
```

## Backward Compatibility

**100% compatible:** The change adds optional parameters with sensible defaults.

- `plot_kernel_distance()` still works without `vmin`/`vmax` (auto-scales)
- `plot_kernel_distance_comparison()` automatically computes shared scale
- Existing code continues to work unchanged

## Testing

Created `test_shared_colorbar.py` to verify:
- ✅ Both subplots use identical `vmin`/`vmax`
- ✅ Colors are directly comparable
- ✅ Visual comparison is clear and unambiguous

**Run test:**
```bash
cd uncertainty_sets_refactored
python test_shared_colorbar.py
```

**Expected output:**
```
✓ Test passed!
Saved: data/viz_artifacts/test_shared_colorbar.png
```

## Files Modified

1. ✅ `viz_kernel_distance.py`
   - Added `vmin`/`vmax` parameters to `plot_kernel_distance()`
   - Compute global scale in `plot_kernel_distance_comparison()`
   - Fixed `plot_multiple_targets()` bug
2. ✅ `test_shared_colorbar.py` (new test file)
3. ✅ `SHARED_COLORBAR_FIX.md` (this documentation)

## Impact on Existing Outputs

**Recommendation:** Re-run visualizations to get updated plots with shared scales.

```bash
# Re-run feature sets to get updated visualizations
python sweep_and_viz_feature_set.py --feature-set focused_2d
python sweep_and_viz_feature_set.py --feature-set high_dim_8d
```

New visualizations will have:
- ✅ Shared colorbar scale
- ✅ Directly comparable colors
- ✅ Clearer visual comparison

## Summary

**What changed:** Colorbar scales are now shared across comparison plots

**Why:** Enables direct visual comparison of kernel weights

**Impact:** Visualizations are now more interpretable and scientifically rigorous

**Backward compatibility:** 100% - existing code works unchanged

**Testing:** ✅ Verified with `test_shared_colorbar.py`

**Next steps:** Re-run experiments to get updated visualizations with shared scales
