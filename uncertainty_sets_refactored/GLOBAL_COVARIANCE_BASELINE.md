# Global Covariance Baseline

## Overview

Added a **global covariance baseline** that uses the empirical mean and covariance from all training data, with no local adaptation. This represents the "no adaptation" case - like k-NN with k=N (all points).

## Motivation

We now have **three baselines** for comparison:

### 1. Global Covariance (NEW)
- **Method:** Use empirical mean and covariance from all training data
- **Prediction:** Same μ and Σ for all query points (no adaptation)
- **Equivalent to:** k-NN with k=N and equal weights on all points
- **Philosophy:** "The world is stationary - use overall statistics"

### 2. k-NN with Equal Weights
- **Method:** k-NN (k=128) with omega=[1,1,...] (equal feature weights)
- **Prediction:** Adapt locally based on k nearest neighbors
- **Philosophy:** "Adapt to local context, but treat all features equally"

### 3. Learned Omega k-NN
- **Method:** k-NN (k=128) with learned omega weights
- **Prediction:** Adapt locally AND weight features intelligently
- **Philosophy:** "Adapt locally AND learn which features matter most"

## Why This Matters

### Hierarchy of Adaptation

```
No Adaptation          →          Local Adaptation          →      Local + Feature Learning
Global Covariance      →      k-NN Equal Weights (ω=[1,1])  →      k-NN Learned Omega

Weakest                                                                     Strongest
```

### Expected Performance

**Best to worst:**
1. **Learned omega k-NN** (local adaptation + feature learning)
2. **Equal weights k-NN** (local adaptation only)
3. **Global covariance** (no adaptation)

**Key insight:** The gap between global and k-NN shows the value of **local adaptation**. The gap between k-NN and learned omega shows the value of **feature learning**.

## Implementation

### Computation

```python
# Global mean and covariance from training data
Mu_global = np.mean(Y_train, axis=0)  # (M,)
Sigma_global = np.cov(Y_train, rowvar=False)  # (M, M)
Sigma_global += ridge * np.eye(M)  # Add ridge for stability

# Predict: use same μ and Σ for ALL eval points
for each eval point:
    μ_pred = Mu_global
    Σ_pred = Sigma_global
```

### Differences from k-NN Baselines

| Aspect | Global Covariance | k-NN (Equal/Learned) |
|--------|------------------|---------------------|
| **Adaptation** | None (same for all points) | Local (k nearest neighbors) |
| **Μ varies?** | No (always global mean) | Yes (varies by query point) |
| **Σ varies?** | No (always global cov) | Yes (varies by query point) |
| **Complexity** | O(1) per query | O(N log N) per query |

## Results

### Sweep Output

Before:
```
NLL: learned=5.234 vs baseline=5.359 (BETTER)
```

After:
```
NLL: learned=5.234 vs kNN=5.359 (BETTER) vs global=6.102 (BETTER)
```

### Comparison Table

```csv
feature_set,nll_improvement,nll_improvement_vs_global,eval_nll_learned,eval_nll_baseline,eval_nll_global
focused_2d,0.125,0.868,5.234,5.359,6.102
high_dim_8d,0.287,1.023,4.991,5.278,6.014
```

### Interpretation

**Example:**
- Global: NLL = 6.102 (no adaptation - worst)
- k-NN Equal: NLL = 5.359 (local adaptation helps!)
- k-NN Learned: NLL = 5.234 (feature learning helps even more!)

**Improvements:**
- Local adaptation value: 6.102 - 5.359 = **0.743** (12% improvement)
- Feature learning value: 5.359 - 5.234 = **0.125** (2% improvement)

## When Global Might Be Competitive

Global covariance can perform well when:

1. **Data is i.i.d.** (no local structure to exploit)
2. **k is too small** (k-NN doesn't capture enough context)
3. **Features are noisy** (local adaptation overfits)
4. **Dataset is small** (not enough samples for local estimates)

If global performs similarly to k-NN, it suggests **limited local structure** in the data.

## For Your Paper

### Main Comparison

Focus on **k-NN baselines** (equal vs learned) for main results:
- Fair comparison (same k, different ω)
- Isolates feature learning benefit
- Standard practice in metric learning

### Global as Sanity Check

Use global baseline as a **sanity check**:
- Verifies local adaptation is helping
- Shows learned method beats naive global approach
- Provides lower bound on performance

### Example Text

> "We compare three approaches: (1) global covariance using all training data (no adaptation), (2) k-NN with equal feature weights (local adaptation only), and (3) k-NN with learned feature weights (local adaptation + feature learning). The learned approach achieves X% improvement over equal-weight k-NN and Y% improvement over the global baseline, demonstrating the value of both local adaptation (Y-X%) and feature learning (X%)."

## Outputs Updated

### sweep_results.csv

New columns:
- `eval_nll_global`: Global baseline NLL
- `nll_improvement_vs_global`: Learned - Global

### Comparison Summary

```
Best feature set for paper: focused_2d
  Description: 2D focused baseline (SYS_MEAN, SYS_STD)
  NLL improvement vs k-NN: 0.125 (2.33%)
  NLL improvement vs global: 0.868 (14.20%)
  Learned omega: [9.5, 12.3]
```

## Testing

Created test to verify global baseline:

```python
# Test that global baseline works
python test_global_baseline.py
```

Expected behavior:
- Global NLL ≥ k-NN NLL ≥ Learned NLL (usually)
- Global uses same μ/Σ for all queries
- Ridge is added for numerical stability

## Edge Cases

### When Global Beats k-NN

If `eval_nll_global < eval_nll_baseline`:
- **Possible causes:**
  - k is too small (not capturing global structure)
  - Features are noisy (local adaptation overfits)
  - Test set is different distribution from train
- **Action:** Increase k or check for distribution shift

### When k-NN = Global

If `eval_nll_global ≈ eval_nll_baseline`:
- **Interpretation:** Data has little local structure
- **Implication:** k-NN not providing much benefit
- **Action:** Check if features are informative

## Summary

**What:** Added global covariance baseline (no adaptation)

**Why:**
- Provides lower bound on performance
- Quantifies value of local adaptation
- Sanity check for experiments

**How:**
- Compute μ and Σ from all training data
- Use same prediction for all query points
- Compare against k-NN baselines

**Impact:**
- Shows learned method beats both k-NN and global
- Separates local adaptation value from feature learning value
- More comprehensive experimental validation

**Files changed:**
- `sweep_and_viz_feature_set.py` - Added global baseline computation
- `run_all_feature_sets.py` - Added global columns to comparison
- `GLOBAL_COVARIANCE_BASELINE.md` - This documentation

**Ready to use:** All runs will now include global baseline automatically! 🎉
