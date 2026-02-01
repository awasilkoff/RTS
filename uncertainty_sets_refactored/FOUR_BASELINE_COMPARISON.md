# Four-Baseline Comparison Implementation

## Overview

This document describes the implementation of a comprehensive 4-baseline comparison to distinguish between different k-NN approaches and properly evaluate the benefit of learned omega feature weights.

## Problem Statement

The previous 3-baseline comparison conflated two different approaches:
1. **Kernel distance with equal weights (ω=[1,1])** - uses softmax-weighted k-NN based on Euclidean distances
2. **Standard Euclidean k-NN** - uses uniform 1/k weights over k nearest Euclidean neighbors

This led to:
- Confusion about what "k-NN Equal" meant (kernel with equal omega vs. true uniform k-NN)
- Numerical issues: Kernel(ω=1) had extreme NLL outliers (up to 7 million) due to softmax concentration
- Unclear baseline comparisons: Hard to isolate the benefit of learned omega vs. kernel method itself

## Solution: Four Distinct Baselines

### 1. Learned Omega (Current Best)
- **Method:** `predict_mu_sigma_topk_cross()` with learned omega
- **Weighting:** Softmax on learned metric distances
- **Purpose:** Demonstrates full benefit of feature learning + kernel adaptation
- **Label:** "Learned Omega"

### 2. Kernel with Equal Feature Weights
- **Method:** `predict_mu_sigma_topk_cross()` with omega=[1,1,...]
- **Weighting:** Softmax on Euclidean distances
- **Purpose:** Shows kernel adaptation without feature learning
- **Label:** "Kernel (ω=1)"
- **Numerical stability:** Uses higher ridge (1e-2) and float64 precision

### 3. Standard Euclidean k-NN (NEW!)
- **Method:** `predict_mu_sigma_knn()`
- **Weighting:** Uniform 1/k weights
- **Purpose:** True uniform k-NN baseline without kernel
- **Label:** "Euclidean k-NN"

### 4. Global Covariance
- **Method:** Empirical mean/cov from all training data
- **Weighting:** N/A (no adaptation)
- **Purpose:** Shows benefit of any local adaptation
- **Label:** "Global Cov"

## Key Comparisons and Insights

| Comparison | What It Shows |
|------------|---------------|
| Learned vs Kernel(ω=1) | Value of learning feature weights (omega) |
| Learned vs Euclidean k-NN | Value of kernel method + learning combined |
| Kernel(ω=1) vs Euclidean k-NN | Impact of softmax vs uniform weighting |
| Euclidean k-NN vs Global | Value of local adaptation |

## Implementation Details

### File: `create_comprehensive_comparison.py`

**Changes:**

1. **Import added:**
   ```python
   from covariance_optimization import (
       CovPredictConfig,
       predict_mu_sigma_topk_cross,
       predict_mu_sigma_knn  # NEW
   )
   ```

2. **Kernel(ω=1) numerical stability:**
   ```python
   pred_cfg_stable = CovPredictConfig(
       tau=float(tau),
       ridge=1e-2,  # 10x higher ridge
       dtype="float64",  # Better precision
       ...
   )
   ```

3. **Euclidean k-NN baseline:**
   ```python
   Mu_euclidean_knn, Sigma_euclidean_knn = predict_mu_sigma_knn(
       X_query=X_eval,
       X_ref=X_train,
       Y_ref=Y_train,
       k=k,
       ridge=ridge,
   )
   ```

4. **Updated methods dictionary:**
   ```python
   methods = {
       "Learned Omega": nll_learned,
       "Kernel (ω=1)": nll_kernel_equal,
       "Euclidean k-NN": nll_euclidean_knn,
       "Global Cov": nll_global,
   }
   ```

5. **Pairwise comparisons (all show Learned Omega on left):**
   - Learned vs Kernel(ω=1) - both show kernel weight visualization
   - Learned vs Euclidean k-NN - binary coloring (k neighbors vs others)
   - Learned vs Global Cov - uniform color (all points equal weight)

6. **Visualization handling:**
   - **Learned Omega / Kernel(ω=1):** Kernel weight coloring (LogNorm scale)
   - **Euclidean k-NN:** Binary coloring - k neighbors in green, others in gray
   - **Global Cov:** Uniform color - all points same color (steelblue)

### File: `diagnose_kernel_equal_outliers.py` (NEW)

**Purpose:** Diagnostic tool to understand why Kernel(ω=1) produces extreme outliers

**Features:**
- Manually computes kernel weights to show softmax concentration
- Compares condition numbers between Kernel(ω=1) and Euclidean k-NN
- Shows effective k (how many neighbors actually contribute)
- Demonstrates weight entropy vs. uniform

**Usage:**
```bash
cd uncertainty_sets_refactored
python diagnose_kernel_equal_outliers.py
```

## Expected Results

### NLL Ordering (Best to Worst)

1. **Learned Omega** - Lowest NLL (learns optimal feature weights)
2. **Kernel(ω=1)** or **Euclidean k-NN** - Moderate NLL (depends on data)
3. **Global Cov** - Highest NLL (no adaptation)

### Key Insights

**If Learned >> Kernel(ω=1) ≈ Euclidean k-NN:**
- Feature learning is highly beneficial
- Kernel weighting scheme doesn't matter much
- Omega discovers important feature scaling

**If Learned >> Kernel(ω=1) >> Euclidean k-NN:**
- Both feature learning AND kernel weighting help
- Softmax weighting provides additional benefit beyond omega
- Adaptive weighting improves covariance estimation

**If Learned ≈ Kernel(ω=1) >> Euclidean k-NN:**
- Kernel method is key (softmax weighting)
- Feature learning provides marginal benefit
- May indicate features are already well-scaled

## Running the Comparison

```bash
cd uncertainty_sets_refactored

# Run comprehensive comparison
python create_comprehensive_comparison.py \
    --feature-set focused_2d \
    --k 128 \
    --ridge 1e-3

# Run diagnostics (optional)
python diagnose_kernel_equal_outliers.py
```

**Outputs:**
- `nll_boxplot.png` - Box-and-whisker plot of all 4 baselines
- `comparison_learned_vs_kernel_equal.png` - Learned (kernel) vs Kernel(ω=1) (kernel)
- `comparison_learned_vs_euclidean_knn.png` - Learned (kernel) vs Euclidean k-NN (binary)
- `comparison_learned_vs_global.png` - Learned (kernel) vs Global Cov (uniform)

## Why This Matters

### Research Contribution

The 4-baseline comparison allows us to:

1. **Isolate the benefit of learned omega** (Learned vs Kernel(ω=1))
2. **Show the benefit of kernel method** (Kernel(ω=1) vs Euclidean k-NN)
3. **Demonstrate local adaptation value** (Euclidean k-NN vs Global)
4. **Understand feature scaling** (unscaled features: omega discovers rescaling)

### Paper Claims

With this comparison, we can make precise claims:

✅ "Learning omega improves NLL by X over equal-weight kernel"
✅ "Kernel weighting improves NLL by Y over uniform k-NN"
✅ "Local adaptation improves NLL by Z over global covariance"
✅ "Combined benefit (learned omega + kernel) is X+Y"

### Methodology Clarity

The distinction between Kernel(ω=1) and Euclidean k-NN:

- **Kernel(ω=1):** Softmax weighting can concentrate on few neighbors
  - Pro: Adapts to local density
  - Con: Can be numerically unstable (needs higher ridge)

- **Euclidean k-NN:** Uniform weighting over k neighbors
  - Pro: Numerically stable, interpretable
  - Con: Doesn't adapt to varying data density

## Technical Notes

### Why Softmax Weighting Can Cause Outliers

```python
# Distance-based logits
logits = -distances / tau

# Softmax weights
weights = exp(logits) / sum(exp(logits))
```

When Euclidean distances vary widely:
- Nearest neighbor gets weight ≈ 0.9
- Other neighbors get weight ≈ 0.01 each
- Covariance becomes nearly singular (rank 1)
- Ridge = 1e-3 insufficient for float32 precision
- NLL explodes

**Solution:** Higher ridge (1e-2) + float64 precision

### Why Euclidean k-NN is Stable

```python
# Uniform weights (always 1/k)
mu = Y_neighbors.mean(axis=0)
Sigma = (centered_Y.T @ centered_Y) / k
```

- Always uses exactly k neighbors equally
- Covariance is proper sample covariance
- No numerical concentration issues
- Standard ridge (1e-3) sufficient

## Future Work

1. **Adaptive tau:** Learn temperature parameter per-point
2. **Hybrid weighting:** Combine softmax and uniform (e.g., top-k softmax)
3. **Outlier detection:** Flag points with extreme kernel concentration
4. **Feature importance:** Analyze omega patterns across feature sets

## References

- `BASELINE_COMPARISON_FIX.md` - Why baseline always uses ω=[1,1,...]
- `GLOBAL_COVARIANCE_BASELINE.md` - Three-level baseline comparison
- `TRAIN_TEST_SPLIT_CHANGE.md` - Random vs temporal holdout
- `FEATURE_ENGINEERING_README.md` - Feature set descriptions
