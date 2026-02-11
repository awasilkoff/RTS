# Absolute vs Scaled Deviation Experiments

## Motivation

Weighted conformal prediction is showing **over-conservative** behavior (98% coverage vs 95% target). This suggests that either:

1. **Scaling by `ens_std` is already capturing adaptation**, making kernel weighting redundant
2. **The combination of scaling + localization is over-adapting**, causing excessive conservatism
3. **Absolute deviation might work better** than studentized (scaled) scores

## Key Question

**Is the ensemble std scaling already providing most of the adaptive benefit?**

If yes, we should:
- Use absolute deviation: `r = |y_pred - y|`
- Skip the scaling: don't divide by `ens_std`
- This would make the method simpler and potentially less conservative

## Conformity Score Types

### Scaled (Studentized) - Current Approach
```python
r = |y_pred - y| / ens_std
```

**Rationale:** Normalize by prediction uncertainty. Errors are "more acceptable" when uncertainty is high.

**Potential issue:** If `ens_std` already captures local variation, this might double-count adaptation when combined with kernel weighting.

### Absolute Deviation - Alternative
```python
r = |y_pred - y|
```

**Rationale:** Raw prediction error without normalization.

**Benefit:** Simpler, might avoid over-conservatism if `ens_std` correlation is already strong.

## Experiments

### Quick Test (`quick_test_absolute_scores.py`)

**Runtime:** ~1-2 minutes

**What it tests:**
- Binned + Absolute deviation
- Binned + Scaled deviation

**Use case:** Fast diagnosis of whether scaling helps or hurts

```bash
cd uncertainty_sets_refactored
python quick_test_absolute_scores.py
```

**Expected output:**
```
COMPARISON
======================================================================

Coverage Gap:
  Absolute: 0.015
  Scaled:   0.030
  → Absolute is BETTER by 0.015 ✓

  INSIGHT: Scaling by ens_std is HURTING coverage!
           Consider using absolute deviation instead.
```

### Comprehensive Experiments (`experiment_absolute_vs_scaled_scores.py`)

**Runtime:** ~15-20 minutes (multiple alphas and taus)

**What it tests:**
1. Binned + Absolute
2. Binned + Scaled
3. Weighted + Absolute (all taus)
4. Weighted + Scaled (all taus)

**Use case:** Full analysis across configurations

```bash
cd uncertainty_sets_refactored
python experiment_absolute_vs_scaled_scores.py
```

**Outputs:**
- `data/absolute_vs_scaled_experiments.csv` - Full results
- `data/absolute_vs_scaled_comparison.png` - 4-panel visualization

## Interpreting Results

### Scenario 1: Absolute >> Scaled (Better coverage)
**Conclusion:** Scaling by `ens_std` is hurting. The ensemble std is redundant or causing over-normalization.

**Recommendation:**
- Use absolute deviation for all methods
- Skip the `ens_std` scaling
- Simpler and better performing

### Scenario 2: Scaled >> Absolute (Better coverage)
**Conclusion:** Scaling is helpful. Studentized scores are working as intended.

**Recommendation:**
- Keep scaled approach
- Issue is likely with tau selection or kernel weighting
- Try smaller tau values to reduce conservatism

### Scenario 3: Absolute ≈ Scaled (Similar coverage)
**Conclusion:** Scaling makes minimal difference.

**Recommendation:**
- Use absolute for simplicity
- Or investigate other factors (tau, bins, etc.)

### Scenario 4: Binned ≈ Weighted (Localization doesn't help)
**Conclusion:** Kernel weighting adds complexity without benefit.

**Recommendation:**
- Stick with simpler binned conformal
- Focus on optimizing bins instead of tau

## Example Analysis

After running experiments, check:

1. **Is absolute better than scaled for binned?**
   - If yes → scaling is the problem

2. **Is weighted better than binned for absolute?**
   - If no → localization doesn't help

3. **What's the best combination?**
   - Could be: Binned + Absolute (simplest, if it works best)
   - Or: Weighted + Absolute (if localization helps without scaling)

## Implementation Details

### How We Test Absolute Deviation

The trick is to set `scale_col` to a constant column of 1s:

```python
# Add constant scale column
df['const_scale'] = 1.0

# Train with constant scale = absolute deviation
bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
    df,
    feature_cols=['ens_mean', 'ens_std'],
    scale_col='const_scale',  # ← Use constant instead of ens_std
    binning='y_pred',
    n_bins=5,
    alpha_target=0.95,
)
```

This works because:
- Conformity score: `r = |y_pred - y| / scale`
- With `scale = 1.0`: `r = |y_pred - y| / 1.0 = |y_pred - y|`
- Exactly what we want!

## Next Steps Based on Results

### If Absolute Wins:
1. Update default in `train_wind_lower_model_conformal_binned()` to use `scale_col='const_scale'`
2. Simplify documentation (no need to explain studentization)
3. Re-run weighted conformal experiments with absolute scores
4. Compare binned vs weighted with absolute scores only

### If Scaled Wins:
1. Keep current approach
2. Focus on tau optimization to reduce conservatism
3. Consider adaptive tau selection
4. Investigate safety_margin parameter

### If No Clear Winner:
1. Provide both options to users
2. Make `score_type` a parameter
3. Document trade-offs
4. Let users choose based on their data

## Files Created

```
uncertainty_sets_refactored/
├── quick_test_absolute_scores.py              # Fast test (2 configs)
├── experiment_absolute_vs_scaled_scores.py    # Full test (4 configs × alphas × taus)
└── ABSOLUTE_VS_SCALED_EXPERIMENTS.md          # This file
```

## Running the Experiments

### Quick Test (Recommended First)
```bash
cd uncertainty_sets_refactored
python quick_test_absolute_scores.py
```

Wait ~2 minutes, check if absolute is better.

### Full Experiments (If Quick Test Shows Promise)
```bash
python experiment_absolute_vs_scaled_scores.py
```

Wait ~15-20 minutes, get comprehensive analysis.

## Understanding Your Current Issue

**Current behavior:** Weighted conformal gives 98% coverage (target 95%)

**Possible explanations:**

1. **Double adaptation:** `ens_std` already varies with conditions, kernel weights also vary with conditions → double counting
2. **Over-smoothing:** Large tau → kernel averages too much → conservative
3. **Scaling amplifies conservatism:** Small `ens_std` → large `r` → large `q_hat` → very conservative bounds

**These experiments will isolate which factor is dominant.**

## References

- **Studentized scores:** Romano et al. (2019) "Conformalized Quantile Regression"
- **Absolute scores:** Lei et al. (2018) "Distribution-Free Predictive Inference"
- **Adaptive conformal:** Gibbs & Candès (2021) "Adaptive Conformal Inference"

The key insight is that studentization (scaling) is meant to account for heteroskedasticity, but if your ensemble std already captures this, scaling becomes redundant or harmful.
