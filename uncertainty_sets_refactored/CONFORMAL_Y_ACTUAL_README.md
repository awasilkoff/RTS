# Binning Conformal Prediction by Actual Values

## Overview

This document describes the **"y_actual" binning strategy** for adaptive conformal prediction, which bins calibration data by actual generation values rather than predictions or features.

## Motivation

Standard conformal prediction uses binning strategies that are available at prediction time:
- `"y_pred"`: Bin by model predictions
- `"feature:<col>"`: Bin by a specific feature (e.g., uncertainty estimate)

However, error patterns may genuinely depend on the **actual generation level**, not the predicted level. For example:
- **Low generation (0-50 MW):** Forecasts tend to over-predict (positive bias)
- **High generation (200-250 MW):** Forecasts tend to under-predict due to capacity constraints
- **Operational regimes:** Different physical processes dominate at different generation levels

The `"y_actual"` strategy addresses this by binning calibration samples by their actual values, then using predictions as a proxy for bin assignment at prediction time.

## How It Works

### Calibration Phase

1. **Create bins from actual values (y_cal):**
   ```python
   bin_edges = create_bins(y_cal, n_bins=5)
   # Example: [0, 50, 100, 150, 200, 250] MW
   ```

2. **Assign calibration samples to bins by their actuals:**
   ```python
   bin_assignment_cal = pd.cut(y_cal, bins=bin_edges)
   ```

3. **Compute q_hat for each bin:**
   ```python
   for bin_b in bins:
       samples_in_b = (bin_assignment_cal == bin_b)
       r_b = nonconformity_scores[samples_in_b]
       q_hat_by_bin[bin_b] = quantile(r_b, conformal_level)
   ```

### Prediction Phase (Proxy Mapping)

1. **Compute base prediction:**
   ```python
   y_pred_base = model.predict(X_new)
   ```

2. **Use prediction as proxy for actual bin:**
   ```python
   bin_idx = pd.cut(y_pred_base, bins=bin_edges)  # Same edges as calibration
   q_hat = q_hat_by_bin[bin_idx]
   ```

3. **Apply conformal correction:**
   ```python
   y_pred_conf = y_pred_base - q_hat * scale
   ```

**Key Insight:** At prediction time, `"y_actual"` behaves identically to `"y_pred"` (both use predictions for bin assignment), but during calibration, `"y_actual"` bins by actuals instead of predictions.

## Usage

### Basic Example

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
    binning="y_actual",  # <-- Use actual values for binning
    n_bins=5,
    alpha_target=0.95,
)

print(f"Coverage: {metrics['coverage']:.4f}")
print(f"Gap: {abs(metrics['coverage'] - 0.95):.4f}")
```

### Comparison with Other Strategies

```python
strategies = ["y_pred", "feature:ens_std", "y_actual"]
results = {}

for binning in strategies:
    bundle, metrics, _ = train_wind_lower_model_conformal_binned(
        df,
        feature_cols=feature_cols,
        binning=binning,
        n_bins=5,
        alpha_target=0.95,
    )
    results[binning] = metrics["coverage"]

# Compare coverage
for binning, coverage in results.items():
    print(f"{binning:20s}: coverage={coverage:.4f}")
```

### Prediction on New Data

```python
# Load bundle (trained with y_actual binning)
bundle = ...  # From training above

# Predict on new data
df_new = pd.DataFrame({
    "ens_mean": [100, 150, 200],
    "ens_std": [10, 15, 20],
})

df_pred = bundle.predict_df(df_new)
print(df_pred[["y_pred_base", "y_pred_conf", "margin"]])
```

## When to Use "y_actual" Binning

### ✅ Good Use Cases

1. **Error heterogeneity by actual level:**
   - Errors systematically differ at low vs. high generation
   - Physical constraints create asymmetric errors (e.g., capacity saturation)

2. **Well-calibrated predictions:**
   - Predictions are unbiased (mean error ≈ 0)
   - Prediction quality is reasonable (proxy mapping works)

3. **Operational regime effects:**
   - Different physical processes dominate at different generation levels
   - Want to ensure coverage across full actual range

### ❌ Poor Use Cases

1. **Biased predictions:**
   - Systematic over/under-forecasting
   - Proxy mapping will be poor

2. **Error patterns don't depend on actual level:**
   - Errors are driven by features available at prediction time (uncertainty, weather, etc.)
   - Use `"feature:<col>"` instead

3. **Insufficient calibration data:**
   - Bins become sparse (< 20 samples per bin)
   - Global q_hat is more reliable

## Theoretical Considerations

### Coverage Guarantee

**Standard conformal guarantee:**
> For exchangeable data, with probability ≥ α, the conformal prediction interval contains the true value.

**With y_actual binning:**
- ✅ **During calibration:** Guarantee holds (we observe actuals)
- ⚠️ **During prediction:** Guarantee depends on **proxy quality**

**Coverage Degradation Bound:**

Coverage is preserved if predictions are good proxies for actuals:

```
|coverage_actual - α| ≤ f(proxy_error)
```

Where `f(proxy_error)` depends on the bin misclassification rate.

**When Proxy Is Perfect:**
- If `y_pred = y` exactly, coverage guarantee holds exactly
- In practice, if `y_pred ≈ y`, coverage degradation is small

**When Proxy Is Poor:**
- Predictions frequently assigned to wrong bin
- Coverage may degrade beyond target α
- Fallback: Use `"y_pred"` or `"feature:<col>"` instead

### Adaptive Correction Benefits

If error patterns genuinely differ by actual level, `"y_actual"` can **improve coverage uniformity**:

| Actual Bin | y_pred Binning | y_actual Binning |
|------------|----------------|------------------|
| 0-50 MW    | 0.92           | 0.95             |
| 50-100 MW  | 0.94           | 0.95             |
| 100-150 MW | 0.97           | 0.95             |
| 150-200 MW | 0.98           | 0.95             |
| 200-250 MW | 0.99           | 0.95             |

**Interpretation:** `"y_actual"` produces more uniform coverage across actual bins, avoiding over-conservative bounds at high generation.

## Validation

Run the validation script to compare all binning strategies:

```bash
cd uncertainty_sets_refactored
python validate_y_actual_binning.py
```

**Outputs** (saved to `data/viz_artifacts/y_actual_validation/`):

1. **summary.csv** - Coverage comparison table
2. **coverage_by_actual_bin.png** - Coverage uniformity across actual bins
3. **error_by_actual_bin.png** - Error distribution by actual generation level
4. **q_hat_comparison.png** - Adaptive corrections per bin
5. **prediction_vs_actual_scatter.png** - Proxy mapping quality visualization

### Diagnostic Questions

**Q1: Do error patterns vary by actual generation level?**
- Check `error_by_actual_bin.png`
- Look for systematic differences in mean/variance across bins

**Q2: How well do predictions proxy for actuals?**
- Check `prediction_vs_actual_scatter.png`
- Points should cluster near the diagonal (y = y_pred)
- Colors should be consistent within regions (correct bin assignment)

**Q3: Does y_actual improve coverage uniformity?**
- Check `coverage_by_actual_bin.png`
- Compare bars across strategies
- `"y_actual"` should have more uniform bar heights

## Expected Outcomes

### If It Works Well

**Indicators:**
- Coverage gap < 5% (comparable to other methods)
- More uniform coverage across actual bins
- q_hats capture true error heterogeneity
- Proxy mapping accuracy > 70%

**When to use:**
- Errors genuinely vary by actual generation level
- Predictions are reasonably accurate
- Want to ensure coverage across full actual range

### If It Doesn't Work Well

**Indicators:**
- Coverage gap > 5% (worse than `"y_pred"`)
- High variance in bin-wise coverage
- Proxy mapping frequently wrong

**Why it might fail:**
- Predictions are biased (poor proxies)
- Error patterns don't depend on actual level
- Insufficient calibration data (bins too sparse)

**Fallback:**
- Use `"y_pred"` or `"feature:ens_std"` instead
- Consider probabilistic weighted binning (future work)

## Comparison Table

| Strategy | Calibration Bins | Prediction Assignment | Coverage Guarantee | Best For |
|----------|------------------|----------------------|--------------------|----------|
| `"y_pred"` | By predictions | By predictions | ✅ Strong | General use |
| `"feature:ens_std"` | By uncertainty | By uncertainty | ✅ Strong | Heterogeneous uncertainty |
| `"y_actual"` | By actuals | By predictions (proxy) | ⚠️ Moderate | Error heterogeneity by actual level |

## Unit Tests

Run the test suite to verify implementation:

```bash
cd uncertainty_sets_refactored
python test_y_actual_binning.py
```

**Tests:**
1. ✓ Basic functionality (y_actual option works)
2. ✓ Calibration bins from actual values
3. ✓ Prediction uses y_pred as proxy
4. ✓ Coverage maintained (gap < 5%)
5. ✓ q_hats vary across bins (adaptation working)
6. ✓ Comparison with y_pred (comparable performance)

## Advanced Topics

### Probabilistic Weighted Binning (Future Work)

Instead of hard bin assignment, use weighted average of q_hats:

```python
# Compute probability distribution over bins
for bin_i in bins:
    p_i = P(y ∈ bin_i | y_pred, uncertainty)

# Weighted q_hat
q_hat_weighted = sum(p_i * q_hat_by_bin[i] for all i)
```

**Advantage:** Smooth adaptation, accounts for prediction uncertainty

**Disadvantage:** Requires uncertainty quantification, more complex

### Nested Binning (Future Work)

Primary binning by feature, secondary binning by actuals:

```python
# Primary: bin by uncertainty
primary_bin = pd.cut(ens_std, bins=primary_edges)

# Secondary: within each primary bin, sub-bin by actuals
for primary in primary_bins:
    sub_bin = pd.cut(y_cal[primary], bins=sub_edges)
    q_hat_nested[primary][sub_bin] = ...
```

**Advantage:** Captures interactions (e.g., "high uncertainty + high generation")

**Disadvantage:** Requires large calibration dataset, risk of overfitting

## References

- Standard conformal prediction: Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Adaptive conformal prediction: Romano et al. (2019) "Conformalized Quantile Regression"
- Binning strategies: Xu and Xie (2021) "Conformal prediction interval for dynamic time-series"

## Implementation Details

### Code Structure

**Modified Files:**
1. `conformal_prediction.py:14` - Updated `BinningSpec` type
2. `conformal_prediction.py:95-127` - Updated `_extract_binning_feature()`
3. `conformal_prediction.py:296-319` - Updated binning logic in training function
4. `conformal_prediction.py:202-247` - Updated docstring

**New Files:**
1. `validate_y_actual_binning.py` - Validation script with diagnostics
2. `test_y_actual_binning.py` - Unit test suite
3. `CONFORMAL_Y_ACTUAL_README.md` - This documentation

### Key Implementation Decision

**At prediction time, `"y_actual"` maps to `y_pred_base`:**

```python
def _extract_binning_feature(df, binning, *, y_pred_base):
    if binning == "y_actual":
        return y_pred_base  # Proxy for actual values
```

This ensures:
- ✅ Calibration bins based on actuals (captures true heterogeneity)
- ✅ Prediction works without knowing actuals (uses proxy)
- ✅ Code is simple (no separate prediction logic)

## Troubleshooting

**Q: Coverage is worse with y_actual than y_pred. Why?**

A: Likely causes:
1. Predictions are biased → poor proxy mapping
2. Error patterns don't depend on actual level → no benefit from y_actual
3. Bins are too sparse → use fewer bins or more calibration data

**Fix:** Use `"y_pred"` or `"feature:ens_std"` instead, or increase calibration data size.

---

**Q: q_hats are all very similar. Is adaptation working?**

A: Check if error patterns genuinely vary by actual level:
```bash
python validate_y_actual_binning.py
# Examine error_by_actual_bin.png
```

If error distributions look similar across bins, `"y_actual"` won't improve over global q_hat.

---

**Q: Can I use y_actual with safety_margin?**

A: Yes! Safety margin works with all binning strategies:
```python
bundle, metrics, _ = train_wind_lower_model_conformal_binned(
    df,
    binning="y_actual",
    n_bins=5,
    alpha_target=0.95,
    safety_margin=0.02,  # Extra 2% conservativeness
)
```

This makes bounds more conservative (higher empirical coverage).

## Summary

**When to use `"y_actual"`:**
- ✅ Error patterns vary by actual generation level
- ✅ Predictions are well-calibrated (good proxies)
- ✅ Want uniform coverage across actual range

**When NOT to use `"y_actual"`:**
- ❌ Predictions are biased or inaccurate
- ❌ Errors driven by features available at prediction time
- ❌ Insufficient calibration data (< 100 samples)

**Validation checklist:**
1. Run `validate_y_actual_binning.py`
2. Check coverage gap < 5%
3. Verify proxy mapping quality (scatterplot)
4. Confirm error heterogeneity by actual bin
5. Compare with `"y_pred"` baseline

**Expected result:**
- Comparable or better overall coverage
- More uniform coverage across actual bins
- Adaptive q_hats reflect true error patterns
