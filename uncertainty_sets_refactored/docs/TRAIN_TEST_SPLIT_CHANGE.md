# Train/Test Split Change: Temporal → Random

## Change Summary

**Before:** Temporal holdout (last 25% of time series)
```python
train_idx = np.arange(n_train)  # First 75%
eval_idx = np.arange(n_train, n)  # Last 25%
```

**After:** Random holdout (random 25% of data)
```python
rng = np.random.RandomState(42)  # Fixed seed for reproducibility
indices = rng.permutation(n)
train_idx = np.sort(indices[:n_train])  # Random 75%
eval_idx = np.sort(indices[n_train:])   # Random 25%
```

## Motivation

### Problem with Temporal Holdout

The last 25% of the time series might not be representative if:
- **Seasonality:** Last quarter might be winter/summer only
- **Non-stationarity:** Data characteristics change over time
- **Limited coverage:** Doesn't test performance across all conditions

**Example:**
```
Data: Aug-Dec 2023 (5 months)
Temporal split:
  Train: Aug-Oct (summer/fall)
  Eval:  Nov-Dec (winter only)  ← Not representative!
```

### Benefits of Random Holdout

1. **Representative evaluation:** Test set samples all time periods
2. **Reduce variance:** More stable NLL estimates
3. **Better generalization test:** Not biased by end-of-series conditions

**Example:**
```
Random split:
  Train: 75% of samples from all months
  Eval:  25% of samples from all months  ← Representative!
```

## Tradeoffs

### Random Split
**Pros:**
- ✓ More representative of overall distribution
- ✓ Lower variance in evaluation metrics
- ✓ Tests generalization across all conditions

**Cons:**
- ✗ Not testing on truly "future" data
- ✗ Slight information leakage if strong temporal correlation

### Temporal Split
**Pros:**
- ✓ Tests on future data (realistic forecasting scenario)
- ✓ No information leakage from future

**Cons:**
- ✗ May not be representative (seasonality, drift)
- ✗ Higher variance if end period is atypical
- ✗ Can't use if dataset is too short

## When to Use Each

### Use Random Split (Our Choice)
- **Covariance estimation** (not time series forecasting)
- **Distribution is stationary** (no major drift)
- **Dataset is short** (5 months - temporal split loses too much coverage)
- **Goal:** Test generalization to new samples from same distribution

### Use Temporal Split
- **Time series forecasting** (predicting future values)
- **Strong temporal dependence** (AR, ARIMA models)
- **Goal:** Test performance on truly future data

## Implementation Details

### Fixed Random Seed
```python
rng = np.random.RandomState(42)  # Reproducible splits
```
**Why:** Ensures same train/test split across runs for fair comparison

### Sorted Indices
```python
train_idx = np.sort(indices[:n_train])
```
**Why:** Maintains deterministic ordering within train/test sets (helpful for debugging)

### Train Fraction
```python
train_frac = 0.75  # 75% train, 25% eval
```
**Standard:** Common split ratio in ML

## Files Modified

1. ✓ `sweep_and_viz_feature_set.py`
2. ✓ `sweep_regularization.py`
3. ✓ `viz_covariance_comparison.py`
4. ✓ `verify_baseline_fix.py`

All now use random split with seed=42.

## Impact on Results

### Expected Changes

**Baseline NLL:** May change (different test samples)
**Learned NLL:** May change (different test samples)
**NLL Improvement:** Should be more stable (less variance)

**Results are NOT directly comparable to old runs** because test set changed.

### Verification

After running with random split:

1. **Check NLL is stable across runs:**
   ```bash
   # Run twice with same seed
   python sweep_and_viz_feature_set.py --feature-set temporal_3d
   python sweep_and_viz_feature_set.py --feature-set temporal_3d
   # → Should get identical results
   ```

2. **Check test set representativeness:**
   ```python
   # Test set should span all months
   print(times[eval_idx].month.value_counts())
   # Should see Aug-Dec, not just Nov-Dec
   ```

## Alternative: Stratified Split

For even better representativeness, could use stratified split:

```python
from sklearn.model_selection import train_test_split

# Stratify by month to ensure balanced representation
months = times.month.values
train_idx, eval_idx = train_test_split(
    np.arange(n),
    test_size=0.25,
    stratify=months,
    random_state=42
)
train_idx = np.sort(train_idx)
eval_idx = np.sort(eval_idx)
```

**Benefit:** Guarantees proportional month representation in train/test

**Not implemented:** Simple random split is sufficient for our purposes

## Cross-Validation (Future Enhancement)

For even more robust evaluation:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, eval_idx) in enumerate(kf.split(X)):
    # Run sweep for each fold
    # Average NLL across folds
```

**Benefit:** More robust estimates, uses all data for both train and eval

**Downside:** 5x slower (5 folds)

## Bottom Line

**Random split is the right choice for covariance estimation:**
- Not doing time series forecasting (don't need temporal holdout)
- Dataset is short (5 months - need representative test set)
- Reduces evaluation variance
- Standard practice in ML for non-temporal problems

**All results now use seed=42 for reproducibility.**
