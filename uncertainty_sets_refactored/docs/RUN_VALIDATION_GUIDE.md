# Y_ACTUAL Binning Validation Guide

## Quick Start

The implementation is complete and tested. Follow these steps to validate on real RTS wind data.

## Step 1: Verify Data is Available

Run this quick test (~5 seconds):

```bash
cd /Users/alexwasilkoff/PycharmProjects/RTS/uncertainty_sets_refactored
python test_data_loading.py
```

**Expected output:**
```
✓ Actuals loaded: 11013 rows
✓ Forecasts loaded: 487080 rows
✓ Conformal totals built: 2952 time points
SUCCESS - Data is ready!
```

If you see errors, the data files need to be generated first using the covariance pipeline.

## Step 2: Run Full Validation (~2-3 minutes)

```bash
python validate_y_actual_binning.py 2>&1 | tee validation_output.txt
```

This will:
1. Compare three binning strategies: "y_pred", "feature:ens_std", "y_actual"
2. Train conformal models at α=0.95 with 5 bins
3. Compute coverage metrics
4. Generate diagnostic plots
5. Save results to `data/viz_artifacts/y_actual_validation/`

**Expected runtime:** 2-3 minutes

## What to Look For

### 1. Coverage Comparison

Check the summary output:

```
Binning              Coverage    Gap       MAE
y_pred               0.9500      0.0000    XX.XX
feature:ens_std      0.9520      0.0020    XX.XX
y_actual             0.9480      0.0020    XX.XX
```

**Success criteria:**
- All strategies have gap < 0.05 (5%)
- y_actual is comparable to other methods

### 2. Error Heterogeneity

Check `error_by_actual_bin.png`:
- Do errors vary systematically by actual generation level?
- If yes → y_actual binning is justified
- If no → errors don't depend on actuals, use standard binning

### 3. Coverage Uniformity

Check `coverage_by_actual_bin.png`:
- Are coverage bars more uniform for y_actual?
- If yes → y_actual improves coverage consistency
- Ideal: all bars close to target line (0.95)

### 4. Proxy Quality

Check `prediction_vs_actual_scatter.png`:
- Do predictions cluster near the diagonal (y=y_pred)?
- Are colors consistent within regions (correct bin assignment)?
- If scattered → proxy mapping is poor, coverage may degrade

### 5. Adaptive Corrections

Check `q_hat_comparison.png`:
- Do q_hat values vary across bins?
- Are patterns different for y_actual vs. other strategies?
- Variation indicates adaptive correction is working

## Outputs Generated

All files saved to `data/viz_artifacts/y_actual_validation/`:

**Summary files:**
- `summary.csv` - Coverage comparison table
- `error_stats_by_actual_bin.csv` - Error statistics per bin
- `coverage_by_bin_*.csv` - Coverage per bin for each strategy

**Diagnostic plots:**
- `coverage_by_actual_bin.png` - Coverage uniformity comparison
- `error_by_actual_bin.png` - Error distribution by actual generation level
- `q_hat_comparison.png` - Adaptive correction magnitudes
- `prediction_vs_actual_scatter.png` - Proxy mapping quality

## Interpreting Results

### Scenario A: y_actual Outperforms

**Indicators:**
- Coverage gap ≤ other methods
- More uniform coverage across actual bins
- Clear error heterogeneity in `error_by_actual_bin.png`
- Good proxy quality in scatter plot

**Action:** Use `binning="y_actual"` for production

**Example use case:**
```python
bundle, metrics, _ = train_wind_lower_model_conformal_binned(
    df,
    feature_cols=["ens_mean", "ens_std"],
    binning="y_actual",  # ← USE THIS
    n_bins=5,
    alpha_target=0.95,
)
```

### Scenario B: y_actual Comparable

**Indicators:**
- Coverage gap similar to other methods (within 1%)
- Coverage uniformity similar
- Some error heterogeneity but not dramatic

**Action:** Either approach is fine, use `"y_pred"` (simpler, stronger theory)

### Scenario C: y_actual Underperforms

**Indicators:**
- Coverage gap > 5% (worse than other methods)
- Poor proxy quality (scatter plot shows misalignment)
- High variance in bin-wise coverage

**Action:** Use `"y_pred"` or `"feature:ens_std"` instead

**Why it might fail:**
- Predictions are biased (systematic over/under-forecasting)
- Error patterns don't actually depend on generation level
- Insufficient calibration data (bins too sparse)

## Expected Results for RTS Data

Based on the loaded data:
- **2952 time points** total
- **~1772 training points** (60%)
- **~590 calibration points** (20%)
- **~590 test points** (20%)

With 5 bins, each calibration bin has ~118 samples (sufficient for stable q_hat estimation).

## Troubleshooting

### Issue: "Data files not found"

**Solution:** Run the covariance pipeline first to generate:
- `actuals_filtered_rts3_constellation_v1.parquet`
- `forecasts_filtered_rts3_constellation_v1.parquet`

### Issue: "Coverage gap too large"

**Potential causes:**
1. Bins too sparse (try `n_bins=3`)
2. Poor proxy quality (check scatter plot)
3. Need safety margin (try `safety_margin=0.02`)

**Fix:**
```python
bundle, metrics, _ = train_wind_lower_model_conformal_binned(
    df,
    binning="y_actual",
    n_bins=3,              # ← Fewer bins
    safety_margin=0.02,    # ← More conservative
    alpha_target=0.95,
)
```

### Issue: "q_hats are all similar"

**This is OK!** If error patterns don't vary by actual level, adaptive binning won't help much. The global q_hat is already optimal.

**Recommendation:** Use `binning="y_pred"` (simpler, equivalent performance)

## Next Steps After Validation

### If y_actual is beneficial:

1. **Document findings:**
   - Save plots to `figures/` directory
   - Note coverage improvement in comments
   - Reference in paper/reports

2. **Use in production:**
   ```python
   bundle = train_wind_lower_model_conformal_binned(
       df, binning="y_actual", n_bins=5, alpha_target=0.95
   )
   bundle.predict_df(new_data)
   ```

3. **Monitor performance:**
   - Track coverage on new data
   - Verify proxy quality remains good
   - Retrain periodically with fresh calibration data

### If y_actual is not beneficial:

1. **Use standard binning:**
   ```python
   bundle = train_wind_lower_model_conformal_binned(
       df, binning="y_pred", n_bins=5, alpha_target=0.95
   )
   ```

2. **Document why:**
   - Error patterns don't depend on actual level
   - Predictions are biased (poor proxies)
   - Standard binning achieves same coverage

## Alternative Experiments

### Try different bin counts:

```bash
# Edit validate_y_actual_binning.py, line 246:
# n_bins = 3  # or 7, 10
python validate_y_actual_binning.py
```

### Try with safety margin:

```python
# Edit validate_y_actual_binning.py, line 59:
safety_margin=0.02,  # Add this parameter
```

### Try different alpha targets:

```python
# Edit validate_y_actual_binning.py, line 247:
alpha_target = 0.90  # or 0.99
```

## Summary of Implementation

**What was implemented:**
- Proxy-based binning by actual values in `conformal_prediction.py`
- Comprehensive validation script with diagnostics
- Unit test suite (7 tests, all passing)
- Detailed documentation (3 guides, ~1500 lines)

**Key files:**
- `conformal_prediction.py` - Core implementation
- `validate_y_actual_binning.py` - Validation with plots
- `test_y_actual_binning.py` - Unit tests
- `CONFORMAL_Y_ACTUAL_README.md` - Complete usage guide
- `Y_ACTUAL_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `RUN_VALIDATION_GUIDE.md` - This guide

**Total code + docs:** ~1800 lines

## Questions?

**Theory:** See `CONFORMAL_Y_ACTUAL_README.md` sections:
- "How It Works"
- "Theoretical Considerations"
- "When to Use"

**Usage:** See `CONFORMAL_Y_ACTUAL_README.md` section "Usage"

**Troubleshooting:** See `CONFORMAL_Y_ACTUAL_README.md` section "Troubleshooting"

**Implementation:** See `Y_ACTUAL_IMPLEMENTATION_SUMMARY.md`

---

**Ready to run?**

```bash
# Quick data check (5 seconds)
python test_data_loading.py

# Full validation (2-3 minutes)
python validate_y_actual_binning.py

# View results
ls -lh data/viz_artifacts/y_actual_validation/
cat data/viz_artifacts/y_actual_validation/summary.csv
```
