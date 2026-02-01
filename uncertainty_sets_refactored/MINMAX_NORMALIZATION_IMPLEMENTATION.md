# MinMax Normalization & Extended Feature Sets - Implementation Summary

## Overview

Successfully implemented MinMax [0,1] normalization and two new feature sets (focused_2d, high_dim_8d) to explore learned omega benefits across different scaling regimes and dimensionalities.

**Total changes:** 4 files modified
**Status:** ✅ All implementations complete and tested

---

## 1. MinMaxScaler Infrastructure (utils.py)

### Added Components

**MinMaxScaler class:**
- Scales features to [0, 1] range
- Supports inverse transform
- Handles edge cases with eps parameter

**Dispatcher functions:**
- `fit_scaler(X, scaler_type)` - unified scaler fitting
- `apply_scaler(X, scaler)` - unified scaler application
- Supports: "none", "standard", "minmax"

### Verification

```bash
✓ MinMax scales to [0, 1] range
✓ Inverse transform reconstructs original
✓ Standard scaler still works
✓ None scaler returns unchanged data
```

---

## 2. New Feature Builders (data_processing_extended.py)

### focused_2d

**Features:** [SYS_MEAN, SYS_STD]
**Purpose:** Clean 2D baseline to test normalization impact on omega learning
**Expected:** MinMax should show clearest omega benefit (no scale dominance)

### high_dim_8d

**Features:** [SYS_MEAN, SYS_STD, WIND_122_MEAN, WIND_122_STD, WIND_309_MEAN, WIND_309_STD, WIND_317_MEAN, WIND_317_STD]
**Purpose:** Explore which features omega prioritizes in higher dimensions
**Expected:** Omega will reveal feature importance patterns (no prior hypothesis)

### Registry Updates

Both feature sets added to:
- `FEATURE_BUILDERS` dispatch dict
- `FEATURE_SET_DESCRIPTIONS` for documentation

---

## 3. Sweep Script Updates (sweep_and_viz_feature_set.py)

### Parameter Changes

**Old:** `standardize_options: tuple[bool, ...]`
**New:** `scaler_types: tuple[str, ...]`

Supports: "none", "standard", "minmax"

### Key Implementation Details

1. **Scaler fitting on train data only:**
   ```python
   scaler = fit_scaler(X_raw[train_idx], scaler_type)
   X = apply_scaler(X_raw, scaler)
   ```

2. **Initial omega based on scaler:**
   - `scaler_type == "none"`: Start with inverse variance (1/σ²)
   - Otherwise: Start with equal weights [1, 1, ...]

3. **Results CSV updated:**
   - Column `standardize` → `scaler_type`
   - Values: "none", "standard", "minmax"

### Visualization Updates

- `generate_visualizations()` now accepts `scaler_type_best` parameter
- Applies best scaler before generating plots
- No changes needed to omega bar chart or projection logic (already handles all dimensionalities)

---

## 4. Runner Script Updates (run_all_feature_sets.py)

### Default Configuration (Quick Runtime)

```python
DEFAULT_TAUS = [2.0, 5.0]              # 2 taus
DEFAULT_OMEGA_L2_REGS = [0.0, 1e-2]    # 2 regs
DEFAULT_SCALER_TYPES = ["standard", "minmax"]  # 2 scalers
```

**Grid size per feature set:** 2 × 2 × 2 = 8 configs
**For focused_2d + high_dim_8d:** 16 total configs
**Expected runtime:** ~30 minutes

### Feature Set Registry

Added to `FEATURE_SETS` and `FEATURE_SET_DESCRIPTIONS`:
- `focused_2d`
- `high_dim_8d`

### Summary Reports

Updated to show `scaler_type` instead of `standardize`:
- `generate_comparison_summary()`
- `print_summary()`

---

## Usage Examples

### Quick Run: focused_2d with MinMax

```bash
cd uncertainty_sets_refactored

python sweep_and_viz_feature_set.py \
    --feature-set focused_2d \
    --scaler-types standard minmax \
    --taus 2.0 5.0 \
    --omega-l2-regs 0.0 1e-2
```

**Output:** `data/viz_artifacts/focused_2d/`
- `sweep_results.csv` - Compare standard vs minmax
- `omega_bar_chart.png` - Learned omega weights
- `kernel_distance_comparison.png` - 2D scatter with contours
- `feature_config.json` - Best config metadata

### Quick Run: high_dim_8d Exploration

```bash
python sweep_and_viz_feature_set.py \
    --feature-set high_dim_8d \
    --scaler-types standard minmax \
    --taus 2.0 5.0 \
    --omega-l2-regs 0.0 1e-2
```

**Output:** `data/viz_artifacts/high_dim_8d/`
- `sweep_results.csv` - Which scaler works best for 8D?
- `omega_bar_chart.png` - **Which features does omega prioritize?**
- `kernel_distance_projection_2d.png` - 2D projection (top 2 features)
- `best_omega.npy` - Full learned omega vector

### Run Both Together (Parallel)

```bash
python run_all_feature_sets.py \
    --feature-sets focused_2d high_dim_8d \
    --scaler-types standard minmax \
    --taus 2.0 5.0 \
    --omega-l2-regs 0.0 1e-2
```

**Total configs:** 16 (8 per feature set)
**Expected runtime:** ~30 min
**Outputs:** Comparison table + individual artifact directories

---

## Thorough Run (Overnight)

For comprehensive exploration:

```bash
python sweep_and_viz_feature_set.py \
    --feature-set high_dim_8d \
    --scaler-types none standard minmax \
    --taus 1.0 5.0 20.0 \
    --omega-l2-regs 0.0 0.01 0.1 1.0
```

**Grid size:** 3 × 3 × 4 = 36 configs
**Expected runtime:** ~2-3 hours

---

## Analysis Questions

### For focused_2d:

1. **Does MinMax show clearer omega benefit than Standard?**
   - Compare `nll_improvement` for minmax vs standard in `sweep_results.csv`
   - Hypothesis: MinMax eliminates scale dominance → omega learns feature importance more clearly

2. **How do learned weights differ between scalers?**
   - Check `omega_0_SYS_MEAN` and `omega_1_SYS_STD` columns
   - MinMax should show clearer differentiation

### For high_dim_8d:

3. **Which features does omega prioritize?**
   - View `omega_bar_chart.png` - top weighted features?
   - Possible patterns:
     - System-level (SYS_MEAN, SYS_STD) dominate (aggregate info)
     - Specific farm (e.g., WIND_309) most predictive
     - Mean features downweight Std features (redundancy)

4. **Does learned omega improve over baseline in 8D?**
   - Compare `eval_nll_learned` vs `eval_nll_baseline`
   - Higher-D should show larger benefit (more room for optimization)

5. **Which scaler works best for 8D?**
   - Sort by `nll_improvement` in `sweep_results.csv`
   - Does MinMax benefit extend to higher dimensions?

---

## Verification Checklist

### Code Verification
- [x] MinMaxScaler transforms to [0, 1] range
- [x] fit_scaler() dispatch works for all types
- [x] focused_2d returns shape (N, 2)
- [x] high_dim_8d returns shape (N, 8) with correct column names
- [x] Sweep script accepts --scaler-types argument
- [x] Runner script defaults to ["standard", "minmax"]

### Output Verification (After Run)
- [ ] `scaler_type` column appears in sweep_results.csv
- [ ] Quick grid completes in ~30 min
- [ ] Best config identified correctly
- [ ] omega_bar_chart.png shows all 8 features for high_dim_8d
- [ ] 2D projection shows top 2 features for high_dim_8d
- [ ] focused_2d uses standard 2D scatter

### Results Verification (After Analysis)
- [ ] MinMax improves NLL vs Standard for focused_2d
- [ ] Learned omega shows clear prioritization in high_dim_8d
- [ ] NLL improvement larger for high_dim_8d vs focused_2d
- [ ] Feature importance pattern makes physical sense

---

## File Structure

```
uncertainty_sets_refactored/
├── utils.py                          # ✅ MinMaxScaler + dispatch
├── data_processing_extended.py       # ✅ focused_2d + high_dim_8d
├── sweep_and_viz_feature_set.py      # ✅ scaler_type parameter
├── run_all_feature_sets.py           # ✅ updated defaults
│
└── data/viz_artifacts/
    ├── focused_2d/                   # 🔄 To be generated
    │   ├── sweep_results.csv
    │   ├── omega_bar_chart.png
    │   ├── kernel_distance_comparison.png
    │   └── feature_config.json
    │
    └── high_dim_8d/                  # 🔄 To be generated
        ├── sweep_results.csv
        ├── omega_bar_chart.png
        ├── kernel_distance_projection_2d.png
        └── feature_config.json
```

---

## Next Steps

1. **Run quick experiments** (~30 min):
   ```bash
   cd uncertainty_sets_refactored
   python run_all_feature_sets.py \
       --feature-sets focused_2d high_dim_8d \
       --scaler-types standard minmax \
       --taus 2.0 5.0 \
       --omega-l2-regs 0.0 1e-2
   ```

2. **Analyze results:**
   - Does MinMax show clearer omega benefit for focused_2d?
   - Which features does omega prioritize in high_dim_8d?
   - Does learned omega improve over baseline in 8D?

3. **If promising, expand:**
   - Add "none" scaler to compare unscaled baseline
   - Expand tau/reg grid for thorough search
   - Test on existing feature sets (temporal_3d, per_resource_4d)

4. **Optional: Add more scalers:**
   - RobustScaler (median/IQR)
   - MaxAbsScaler (max absolute value)

---

## Implementation Notes

### Why fit scaler on train data only?

```python
# CORRECT: Fit on train, apply to all
scaler = fit_scaler(X_raw[train_idx], scaler_type)
X = apply_scaler(X_raw, scaler)

# WRONG: Fit on all data (data leakage)
scaler = fit_scaler(X_raw, scaler_type)
```

**Rationale:** MinMax parameters (min/max) should not see eval data to avoid information leakage. This ensures fair evaluation.

### Why omega initialization differs by scaler?

- **none:** Features have different scales → start with inverse variance to rescale
- **standard/minmax:** Features already balanced → start with equal weights [1, 1, ...]

This provides better initialization and faster convergence.

---

## Testing Summary

All tests passed:
```
✓ utils.py imports OK
✓ Feature builders: ['temporal_3d', 'per_resource_4d', 'unscaled_2d', 'focused_2d', 'high_dim_8d']
✓ sweep_and_viz_feature_set.py --help shows --scaler-types
✓ run_all_feature_sets.py --help shows --scaler-types
✓ MinMaxScaler [0,1] normalization verified
✓ Inverse transform reconstruction verified
✓ Standard scaler still works
✓ None scaler returns unchanged data
```

**Status:** Ready to run experiments! 🚀
