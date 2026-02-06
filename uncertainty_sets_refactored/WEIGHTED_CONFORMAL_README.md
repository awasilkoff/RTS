# Weighted Conformal Prediction

## Overview

This implementation provides **localized weighted conformal prediction** using kernel distances and learned feature weights (omega) from covariance optimization. Unlike binned conformal which assumes local exchangeability via fixed bins, this approach uses kernel-weighted calibration points for smooth, adaptive conformal corrections.

## Key Differences from Binned Conformal

| Aspect | Binned Conformal | Weighted Conformal |
|--------|------------------|-------------------|
| **Exchangeability** | Fixed bins | Kernel-weighted, continuous |
| **Calibration points** | All points in same bin | Distance-weighted neighbors |
| **Quantile** | Uniform weights per bin | Kernel-weighted quantile |
| **Adaptation** | Discrete (per bin) | Continuous (per query point) |
| **Coverage** | Global bin-based | Localized query-dependent |
| **Bundle storage** | Bin edges + q_hat per bin | Full calibration set |

## Algorithm

### Input
- `x_cal, y_cal`: Calibration features and targets
- `x_star`: Query point
- `model fhat(x)`: Trained quantile regressor
- `scale shat(x)`: Scale estimator (e.g., ensemble std)
- `kernel K(dist/h)`: Gaussian kernel with bandwidth h (tau)
- `alpha`: Target coverage (e.g., 0.95)
- `omega`: Learned feature weights (from covariance optimization)

### Steps

**1) Compute calibration (studentized) scores:**
```python
yhat_cal = fhat(x_cal)         # Quantile predictions on calibration set
sigma_cal = shat(x_cal)        # Scale estimates
r_cal = max(0, (yhat_cal - y_cal) / sigma_cal)  # One-sided for lower bound
```

**2) Compute local weights for query point:**
```python
d = dist(x_cal, x_star)         # Weighted distances using omega
w_cal = K(d / h)                # Gaussian kernel weights
w_star = K(0.0) = 1.0           # Test point weight
```

**3) Compute weighted quantile:**
```python
W = w_cal.sum() + w_star        # Total weight including test point
idx = argsort(r_cal)            # Sort scores
r_sorted = r_cal[idx]
w_sorted = w_cal[idx] / W       # Normalized weights

# Find first k where cumsum(w_sorted) >= 1 - alpha
q = r_sorted[first_k_where(cumsum(w_sorted) >= 1 - alpha)]
```

**4) Form prediction interval (invert score):**
```python
yhat_star = fhat(x_star)
sigma_star = shat(x_star)
y_pred_conf = yhat_star - q * sigma_star  # Lower bound only
```

## Usage

### Quick Start

```python
import numpy as np
import pandas as pd
from conformal_prediction import train_wind_lower_model_weighted_conformal
from data_processing import build_conformal_totals_df

# Load data
actuals = pd.read_parquet('data/actuals_filtered_rts3_constellation_v1.parquet')
forecasts = pd.read_parquet('data/forecasts_filtered_rts3_constellation_v1.parquet')
df = build_conformal_totals_df(actuals, forecasts)

# Add kernel features (matching omega dimensions)
df['SYS_MEAN'] = df['ens_mean']
df['SYS_STD'] = df['ens_std']

# Load learned omega from covariance optimization
omega = np.load('data/viz_artifacts/focused_2d/best_omega.npy')

# Train weighted conformal model
bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
    df,
    feature_cols=['ens_mean', 'ens_std'],        # For quantile model
    kernel_feature_cols=['SYS_MEAN', 'SYS_STD'], # For kernel weighting
    omega=omega,
    tau=5.0,
    alpha_target=0.95,
    split_method='random',
    random_seed=42,
)

print(f"Coverage: {metrics['coverage']:.3f}")
print(f"Gap from target: {abs(metrics['coverage'] - 0.95):.3f}")
print(f"q_hat mean: {metrics['q_hat_mean']:.3f}")
print(f"q_hat std: {metrics['q_hat_std']:.3f}")

# Predict on new data
df_new = ...  # New forecast data with required columns
df_pred = bundle.predict_df(df_new)
lower_bound = df_pred['y_pred_conf']
```

### Parameter Selection

**Tau (Kernel Bandwidth):**
- Smaller tau → sharper kernel, more local adaptation
- Larger tau → smoother kernel, more global behavior
- Recommend grid search: `[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]`
- Use `sweep_weighted_conformal_tau.py` to find optimal value

**Omega (Feature Weights):**
- Load from previous covariance optimization: `np.load('data/viz_artifacts/focused_2d/best_omega.npy')`
- Or use uniform weights: `omega = np.array([1.0, 1.0])`
- Must match dimensions of `kernel_feature_cols`

**Feature Sets:**
- **Quantile model features**: `['ens_mean', 'ens_std']` (can include any predictive features)
- **Kernel features**: `['SYS_MEAN', 'SYS_STD']` (must match omega dimensions from covariance)

## Scripts

### 1. Unit Tests
```bash
cd uncertainty_sets_refactored
python test_weighted_conformal.py
```

Tests:
- `weighted_quantile()` correctness
- Kernel distance computation
- Coverage validation
- Edge cases and error handling

### 2. Comparison with Binned Conformal
```bash
python compare_weighted_vs_binned.py
```

Outputs:
- Side-by-side coverage metrics
- Error comparison (RMSE, MAE)
- `data/weighted_vs_binned_comparison.csv`

Expected results:
- Similar overall coverage
- Smoother q_hat variation (no step function)
- Better local coverage uniformity

### 3. Tau Parameter Sweep
```bash
python sweep_weighted_conformal_tau.py
```

Outputs:
- `data/weighted_conformal_tau_sweep.csv`
- `data/weighted_conformal_tau_sweep.png` (4-panel visualization)

Plots:
1. Coverage vs Tau
2. Coverage Gap vs Tau
3. q_hat Mean vs Tau
4. q_hat Spatial Variation vs Tau

Expected pattern:
- Low tau → under-coverage (too local, noisy)
- High tau → over-coverage (too global, conservative)
- Optimal tau: minimal coverage gap

## Implementation Details

### Core Functions

**`weighted_quantile(values, weights, q, include_test_weight=True)`**
- Computes weighted quantile with test point weight included
- Handles edge cases (empty arrays, all zeros)
- Numerical stability for small weights

**`_compute_kernel_distances(X_query, X_ref, omega, tau)`**
- Gaussian kernel: `K(d/tau) = exp(-d/tau)`
- Weighted distance: `d = sqrt(sum_k omega[k] * (x_q[k] - x_r[k])^2)`
- Efficient pairwise distance computation

**`compute_weighted_conformal_correction_lower(...)`**
- Computes localized q_hat for each query point
- One-sided nonconformity: `r = max(0, (y_pred - y) / scale)`
- Kernel-weighted quantile with safety margin

### Bundle Class

**`WeightedConformalLowerBundle`**

Stores:
- Quantile model (LGBMRegressor)
- Calibration set: `X_cal, y_cal, y_pred_cal, scale_cal`
- Learned omega and tau
- Configuration: `alpha_target, min_scale, safety_margin`

Methods:
- `predict_df(df_feat)`: Apply model + localized conformal correction

Storage size (typical):
- n_cal ~ 500-1000 points
- K ~ 2 features (focused_2d)
- X_cal: 8 KB, y_cal/pred/scale: 12 KB
- Total: ~20 KB (negligible)

### Training Function

**`train_wind_lower_model_weighted_conformal(...)`**

Steps:
1. Validate inputs (columns, omega dimensions)
2. Split data (time_ordered or random)
3. Train quantile model (LGBM)
4. Compute calibration scores
5. Create bundle with full calibration set
6. Evaluate on test set
7. Return bundle, metrics, test predictions

Metrics returned:
- `coverage`: Empirical coverage (fraction y >= y_pred_conf)
- `pre_conformal_coverage`: Before conformal correction
- `rmse, mae`: Prediction errors
- `q_hat_mean, q_hat_std, q_hat_min, q_hat_max`: q_hat statistics
- `n_train, n_cal, n_test`: Split sizes

## Expected Outcomes

### Success Criteria

**Primary (Must Have):**
1. ✅ Coverage gap < 5% for alpha=0.95
2. ✅ Smooth variation in q_hat across query points (not step function)
3. ✅ Bundle predict_df() works correctly

**Secondary (Nice to Have):**
1. ✅ Better coverage than binned conformal (especially in smooth regions)
2. ✅ Lower coverage variance across query points
3. ✅ Tau sweep shows clear optimum

### Diagnostic Questions

**How does coverage vary with tau?**
- Plot coverage vs. tau (log scale)
- Expect: low tau → under-coverage, high tau → over-coverage

**How does q_hat vary spatially?**
- Scatterplot: query features (SYS_MEAN, SYS_STD) colored by q_hat_local
- Expect: smooth gradients, no sharp jumps

**Does weighted outperform binned?**
- Compare coverage, gap, MAE, RMSE
- Expect: similar overall coverage, smoother local coverage

## When to Use Weighted vs Binned

### Use Weighted Conformal When:
- ✅ Have learned omega from covariance optimization
- ✅ Want smooth, continuous adaptation
- ✅ Sufficient calibration data (n_cal > 100)
- ✅ Willing to store calibration set in bundle
- ✅ Want localized, query-dependent corrections

### Use Binned Conformal When:
- ✅ No omega available (use uniform features)
- ✅ Want simpler, more interpretable bins
- ✅ Limited calibration data (binning more stable)
- ✅ Smaller bundle size (no calibration storage)
- ✅ Prefer discrete adaptation levels

## Integration with Existing Pipeline

### Workflow

```python
# Step 1: Learn omega (covariance optimization)
from covariance_optimization import fit_omega, KernelCovConfig, FitConfig

omega = fit_omega(X_scaled, Y, omega0=np.ones(K), train_idx=train_idx,
                  cfg=KernelCovConfig(tau=5.0, ridge=1e-4),
                  fit_cfg=FitConfig(max_iters=200))
np.save('data/best_omega.npy', omega)

# Step 2: Weighted conformal (new!)
bundle, metrics, df_test = train_wind_lower_model_weighted_conformal(
    df,
    feature_cols=['ens_mean', 'ens_std'],
    kernel_feature_cols=['SYS_MEAN', 'SYS_STD'],
    omega_path='data/best_omega.npy',
    tau=5.0,
    alpha_target=0.95,
)

# Step 3: Predict on new data
df_new = ...  # New forecast data
df_pred = bundle.predict_df(df_new)
lower_bound = df_pred['y_pred_conf']

# Step 4: Use in unit commitment
# (Pass lower_bound to ARUC model as constraint)
```

## Limitations and Caveats

1. **Computational Cost**:
   - Weighted conformal computes distances for all query-reference pairs
   - For large test sets, this is O(n_query * n_cal * K)
   - Consider k-NN truncation if too slow

2. **Storage**:
   - Bundle stores full calibration set (vs. bin edges + q_hat dict)
   - Typically ~20 KB (negligible for most applications)

3. **Coverage Guarantee**:
   - Theoretical guarantee requires exchangeability assumption
   - Weighted conformal relaxes this to "kernel-weighted exchangeability"
   - Empirical coverage may vary if distribution shifts significantly

4. **Tau Selection**:
   - No universal "best" tau (data-dependent)
   - Use cross-validation or hold-out validation to select
   - Trade-off: local vs. global behavior

## References

**Weighted Conformal Prediction:**
- Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
- Barber et al. (2021) "Conformal Prediction Beyond Exchangeability"
- Lei et al. (2018) "Distribution-Free Predictive Inference For Regression"

**Kernel Methods:**
- Gaussian kernel: `K(d/h) = exp(-d/h)`
- Bandwidth selection: median heuristic, cross-validation
- Feature weighting: learned from covariance optimization

**Implementation:**
- Current binned conformal: `conformal_prediction.py:train_wind_lower_model_conformal_binned()`
- Covariance kernel weighting: `covariance_optimization.py:fit_omega()`
- Feature engineering: `data_processing.py:build_conformal_totals_df()`

## Troubleshooting

### Common Issues

**1. Coverage gap too large (>10%)**
- Try different tau values (sweep recommended)
- Check omega dimensions match kernel features
- Increase calibration set size
- Add safety_margin (e.g., 0.02)

**2. q_hat all zeros or very small**
- Check that y_pred is not systematically over-predicting (all r_cal ≈ 0)
- Verify scale_col is not all zeros or very large
- Try different quantile_alpha (e.g., 0.05 instead of 0.10)

**3. Runtime too slow**
- Reduce calibration set size (e.g., cal_frac=0.15)
- Consider k-NN truncation (use only top-k neighbors)
- Use time_ordered split instead of random (faster indexing)

**4. Omega file not found**
- Run covariance optimization first: `python sweep_and_viz_feature_set.py --feature-set focused_2d`
- Or use uniform omega: `omega = np.array([1.0, 1.0])`

**5. Dimension mismatch**
- Ensure `kernel_feature_cols` length matches `omega.shape[0]`
- If omega is from focused_2d, use `kernel_feature_cols=['SYS_MEAN', 'SYS_STD']`

## Next Steps

After implementing and testing weighted conformal:

1. **Validate on RTS data** with learned omega from focused_2d
2. **Tau grid search** to find optimal bandwidth
3. **Compare with binned** conformal (y_pred, y_actual, feature:ens_std)
4. **Visualize q_hat spatial variation** (scatterplot)
5. **Document best practices** for tau selection
6. **Optional: Implement k-NN truncation** if full kernel is too slow
7. **Integrate with ARUC model** (use conformal bounds as constraints)

## File Structure

```
uncertainty_sets_refactored/
├── conformal_prediction.py              # Core implementation
│   ├── weighted_quantile()
│   ├── _compute_kernel_distances()
│   ├── compute_weighted_conformal_correction_lower()
│   ├── WeightedConformalLowerBundle
│   └── train_wind_lower_model_weighted_conformal()
├── test_weighted_conformal.py           # Unit tests
├── compare_weighted_vs_binned.py        # Comparison script
├── sweep_weighted_conformal_tau.py      # Tau parameter sweep
└── WEIGHTED_CONFORMAL_README.md         # This file
```

## Support

For questions or issues:
1. Check this README for common troubleshooting
2. Run unit tests: `python test_weighted_conformal.py`
3. Verify omega file exists and matches dimensions
4. Check RTS data files are present in `data/` directory
