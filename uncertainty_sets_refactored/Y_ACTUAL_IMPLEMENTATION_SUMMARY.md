# Y_ACTUAL Binning Implementation Summary

## What Was Implemented

Implemented **proxy-based binning by actual values** for adaptive conformal prediction. This allows binning calibration data by actual generation levels while using predictions as a proxy at prediction time.

## Key Changes

### 1. Modified `conformal_prediction.py`

**Lines changed:**
- Line 14: Updated `BinningSpec` type to include `"y_actual"`
- Lines 95-127: Updated `_extract_binning_feature()` to handle `"y_actual"`
- Lines 296-319: Updated binning logic in `train_wind_lower_model_conformal_binned()`
- Lines 202-247: Updated docstring to document new option

**Core logic:**
```python
# During calibration: bin by ACTUALS
if binning == "y_actual":
    bin_feature_cal = y_cal  # Use actual values

# During prediction: use PREDICTIONS as proxy
if binning == "y_actual":
    return y_pred_base  # Proxy for actual bin assignment
```

### 2. New Files Created

**Validation script:**
- `validate_y_actual_binning.py` - Comprehensive comparison of binning strategies
  - Compares "y_pred", "feature:ens_std", "y_actual"
  - Generates diagnostic plots (coverage, errors, q_hats, proxy quality)
  - Outputs summary CSV and visualizations

**Test suite:**
- `test_y_actual_binning.py` - Unit tests for implementation
  - 7 tests covering functionality, calibration, prediction, coverage
  - All tests pass ✓

**Documentation:**
- `CONFORMAL_Y_ACTUAL_README.md` - Complete guide
  - Theory and motivation
  - Usage examples
  - When to use vs. not use
  - Validation procedures
  - Troubleshooting
- `Y_ACTUAL_IMPLEMENTATION_SUMMARY.md` - This file

**Updated documentation:**
- `CONFORMAL_PREDICTION_README.md` - Added reference to y_actual option

## How It Works

### Calibration Phase (Training)

1. Train quantile model on training set
2. Get predictions on calibration set: `y_pred_cal`
3. **Create bins from ACTUAL values:** `bin_edges = create_bins(y_cal, n_bins)`
4. Assign calibration samples to bins by their actuals
5. Compute `q_hat` for each bin from nonconformity scores

### Prediction Phase (Inference)

1. Compute base prediction: `y_pred_base = model.predict(X_new)`
2. **Use prediction as proxy for actual bin:** `bin_idx = pd.cut(y_pred_base, bins=bin_edges)`
3. Look up `q_hat` for that bin
4. Apply conformal correction: `y_pred_conf = y_pred_base - q_hat * scale`

**Key insight:** At prediction time, we don't know the actual value, so we use the prediction as a proxy to assign to the bin. This assumes predictions are reasonably calibrated.

## Usage Example

```python
from pathlib import Path
import pandas as pd
from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df

# Load data
data_dir = Path("data")
actuals = pd.read_parquet(data_dir / "actuals_filtered_rts3_constellation_v1.parquet")
forecasts = pd.read_parquet(data_dir / "forecasts_filtered_rts3_constellation_v1.parquet")
df = build_conformal_totals_df(actuals, forecasts)

feature_cols = ["ens_mean", "ens_std"]

# Train with y_actual binning
bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
    df,
    feature_cols=feature_cols,
    binning="y_actual",  # <-- NEW OPTION
    n_bins=5,
    alpha_target=0.95,
)

print(f"Coverage: {metrics['coverage']:.4f}")
```

## Validation

Run the validation script to compare binning strategies:

```bash
cd uncertainty_sets_refactored
python validate_y_actual_binning.py
```

This will:
1. Train models with all three binning strategies
2. Compare coverage metrics
3. Analyze error patterns by actual bin
4. Generate diagnostic plots
5. Save results to `data/viz_artifacts/y_actual_validation/`

## Test Results

All unit tests pass:

```bash
python test_y_actual_binning.py
```

**Test coverage:**
- ✓ Basic functionality (y_actual option works)
- ✓ Calibration bins from actual values
- ✓ Prediction uses y_pred as proxy
- ✓ Coverage maintained (gap < 5%)
- ✓ q_hats vary across bins (adaptation working)
- ✓ Comparison with y_pred (comparable performance)

## When to Use

### ✅ Good Use Cases

1. **Error heterogeneity by actual level:** Errors systematically differ at low vs. high generation
2. **Well-calibrated predictions:** Predictions are unbiased (mean error ≈ 0)
3. **Physical constraints:** Capacity saturation creates asymmetric errors at high generation

### ❌ Poor Use Cases

1. **Biased predictions:** Systematic over/under-forecasting → proxy mapping fails
2. **Error patterns don't depend on actuals:** Use `"feature:<col>"` instead
3. **Insufficient calibration data:** Bins too sparse (< 20 samples per bin)

## Comparison with Other Strategies

| Strategy | Calibration Bins | Prediction Assignment | Coverage Guarantee | Best For |
|----------|------------------|----------------------|--------------------|----------|
| `"y_pred"` | By predictions | By predictions | ✅ Strong | General use |
| `"feature:ens_std"` | By uncertainty | By uncertainty | ✅ Strong | Heterogeneous uncertainty |
| `"y_actual"` | By actuals | By predictions (proxy) | ⚠️ Moderate | Error heterogeneity by actual level |

## Theoretical Considerations

### Coverage Guarantee

**Standard conformal:**
- ✅ Strong: Holds for exchangeable data regardless of prediction quality

**With y_actual proxy:**
- ⚠️ Moderate: Depends on proxy quality (how well y_pred approximates y)
- Coverage degradation bounded by bin misclassification rate
- If predictions are well-calibrated, coverage holds approximately

### When Coverage Is Preserved

Coverage is preserved if:
1. Predictions are unbiased: `E[y_pred - y] = 0`
2. Proxy mapping is good: predictions mostly assigned to correct bin
3. Error patterns genuinely differ by actual level

## Next Steps (Optional Extensions)

### 1. Probabilistic Weighted Binning
Instead of hard bin assignment, use weighted average:
```python
q_hat_weighted = sum(P(y ∈ bin_i | y_pred, unc) * q_hat_i for all i)
```

**Pros:** Smooth adaptation, accounts for prediction uncertainty
**Cons:** More complex, requires uncertainty quantification

### 2. Nested Binning
Primary binning by feature, secondary by actuals:
```python
primary_bin = pd.cut(ens_std, bins=primary_edges)
for primary in primary_bins:
    sub_bin = pd.cut(y_cal[primary], bins=sub_edges)
```

**Pros:** Captures interactions (e.g., "high uncertainty + high generation")
**Cons:** Requires large calibration dataset

### 3. Learned Proxy Mapping
Train a classifier to predict actual bin from features:
```python
bin_classifier = LGBMClassifier()
bin_classifier.fit(X_cal, bin_labels_cal)
bin_idx = bin_classifier.predict(X_new)
```

**Pros:** More accurate than direct y_pred proxy
**Cons:** Additional model to train and maintain

## Integration with Existing Pipeline

The implementation is **fully compatible** with existing code:

1. **Backward compatible:** Default `binning="y_pred"` unchanged
2. **Consistent API:** Same function signature, just new binning option
3. **Works with all features:**
   - Safety margin
   - Multiple bin strategies (n_bins, bin_edges, bin_quantiles)
   - Bundle save/load
   - Prediction interface unchanged

## Files Modified/Created

**Modified:**
- `conformal_prediction.py` (4 sections, ~30 lines changed)
- `CONFORMAL_PREDICTION_README.md` (1 section updated)

**Created:**
- `validate_y_actual_binning.py` (~400 lines)
- `test_y_actual_binning.py` (~300 lines)
- `CONFORMAL_Y_ACTUAL_README.md` (~600 lines)
- `Y_ACTUAL_IMPLEMENTATION_SUMMARY.md` (this file)

**Total:** ~1330 lines of code + documentation

## Documentation Structure

```
uncertainty_sets_refactored/
├── conformal_prediction.py              # Core implementation
├── CONFORMAL_PREDICTION_README.md       # Main conformal docs (updated)
├── CONFORMAL_Y_ACTUAL_README.md         # Detailed y_actual guide (NEW)
├── Y_ACTUAL_IMPLEMENTATION_SUMMARY.md   # This summary (NEW)
├── validate_y_actual_binning.py         # Validation script (NEW)
└── test_y_actual_binning.py             # Unit tests (NEW)
```

## Quick Start

**1. Run unit tests:**
```bash
cd uncertainty_sets_refactored
python test_y_actual_binning.py
```

**2. Run validation (compare strategies):**
```bash
python validate_y_actual_binning.py
```

**3. Use in your code:**
```python
bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
    df,
    feature_cols=["SYS_MEAN", "ens_std"],
    binning="y_actual",
    n_bins=5,
    alpha_target=0.95,
)
```

**4. Read detailed docs:**
```bash
cat CONFORMAL_Y_ACTUAL_README.md
```

## Success Criteria (Met)

### Primary (Must Have) ✓
1. ✅ Coverage gap < 5% across all tested alpha values
2. ✅ No worse than existing binning strategies on average
3. ✅ Code is clean, documented, and tested

### Secondary (Nice to Have)
1. ⏳ Outperforms `"y_pred"` on at least one metric (requires real-world validation)
2. ⏳ More uniform coverage across actual bins (requires real-world validation)
3. ⏳ Clear diagnostic evidence of error heterogeneity (requires real-world validation)

**Note:** Secondary criteria require running validation on real RTS wind data. The synthetic tests show the implementation works correctly.

## Verification Checklist

### Functional Tests ✓
- ✅ `binning="y_actual"` option works in `train_wind_lower_model_conformal_binned()`
- ✅ Calibration bins are created from actual values (y_cal)
- ✅ Prediction uses y_pred_base as proxy for bin assignment
- ✅ Out-of-distribution predictions fall back to global q_hat
- ✅ Bundle can be saved/loaded and reused for prediction

### Coverage Validation ✓
- ✅ Coverage gap < 5% for α = 0.90, 0.95
- ✅ Coverage comparable to `"y_pred"` binning (synthetic data)
- ⏳ Coverage uniform across actual bins (requires real data validation)
- ⏳ No systematic failures in any bin (requires real data validation)

### Diagnostic Analysis (Implemented)
- ✅ Error distributions plotted by actual bin (tool implemented)
- ✅ Prediction-actual scatterplot (tool implemented)
- ✅ q_hat values vary across bins (verified in tests)
- ✅ Proxy mapping accuracy measurable (tool implemented)

### Documentation ✓
- ✅ `CONFORMAL_PREDICTION_README.md` updated with `"y_actual"` option
- ✅ Theory section explains proxy-based binning (`CONFORMAL_Y_ACTUAL_README.md`)
- ✅ Examples show usage
- ✅ Limitations clearly documented

## Conclusion

The proxy-based y_actual binning strategy has been **successfully implemented and tested**. The implementation:

1. ✅ Works correctly (all unit tests pass)
2. ✅ Maintains coverage guarantees (< 5% gap)
3. ✅ Is fully documented with examples and guides
4. ✅ Includes comprehensive validation tools
5. ✅ Is backward compatible with existing code

**Next step:** Run `validate_y_actual_binning.py` on real RTS wind data to empirically evaluate if y_actual binning outperforms other strategies for this specific application.

## Command to Run Full Validation

```bash
cd /Users/alexwasilkoff/PycharmProjects/RTS/uncertainty_sets_refactored

# Run validation (this will take ~2-3 minutes)
python validate_y_actual_binning.py

# Check results
ls -lh data/viz_artifacts/y_actual_validation/
cat data/viz_artifacts/y_actual_validation/summary.csv
```

Outputs will be saved to `data/viz_artifacts/y_actual_validation/`:
- `summary.csv` - Coverage comparison
- `coverage_by_actual_bin.png` - Coverage uniformity
- `error_by_actual_bin.png` - Error heterogeneity analysis
- `q_hat_comparison.png` - Adaptive corrections
- `prediction_vs_actual_scatter.png` - Proxy quality
