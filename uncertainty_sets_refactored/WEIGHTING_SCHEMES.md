# Weighting Schemes for Conformal Prediction

## Overview

Different ways to weight calibration points when computing conformal corrections. The key question: **which calibration points are most relevant for predicting at a given query point?**

## Available Schemes

### 1. Uniform (Binned Conformal)

**Weight formula:** All points in same bin get weight 1, others get weight 0

**When similar to query:** Points fall in same discrete bin (e.g., same y_pred range)

**Pros:**
- Simple, interpretable
- No hyperparameters (just bin edges)
- Fast computation

**Cons:**
- Discrete jumps at bin boundaries
- No smooth adaptation
- Fixed bin structure may not capture true similarity

**Use when:**
- Want simplicity and interpretability
- Have enough data per bin
- Discrete adaptation is acceptable

---

### 2. Feature-based Kernel (Current Weighted Conformal)

**Weight formula:** `K(x_query, x_cal) = exp(-d(x_query, x_cal) / τ)`

where `d` = weighted Euclidean distance using learned omega

**When similar to query:** Feature vectors are close in omega-weighted space

**Pros:**
- Smooth, continuous adaptation
- Learns feature importance (omega)
- Theory-backed (covariate shift literature)

**Cons:**
- Requires learned omega
- Sensitive to tau selection
- Stores full calibration set

**Use when:**
- Have learned omega from covariance optimization
- Want smooth query-dependent corrections
- Feature space captures relevant similarity

---

### 3. Time-based Exponential Decay

**Weight formula:** `w(t_query, t_cal) = exp(-λ * |t_query - t_cal|)`

where `λ = ln(2) / half_life`

**When similar to query:** Calibration point is temporally recent

**Concept:** Recent calibration points are more relevant than old ones due to:
- Concept drift (wind patterns change over time)
- Forecast model updates
- Seasonal/climatic shifts

**Half-life interpretation:**
- half_life = 14 days → weights decay 50% after 2 weeks
- half_life = 30 days → weights decay 50% after 1 month
- half_life = 60 days → weights decay 50% after 2 months

**Pros:**
- Handles temporal drift naturally
- Simple interpretable parameter (half-life)
- Smooth decay (no hard cutoffs)

**Cons:**
- Assumes stationarity within half-life
- May discard useful older data
- Requires sufficient recent calibration data

**Use when:**
- Suspect temporal drift in wind patterns or forecast quality
- Recent observations are more predictive
- Have continuous time-series data

---

### 4. Time-based Sliding Window

**Weight formula:** `w = 1 if |t_query - t_cal| <= window, else 0`

**When similar to query:** Calibration point is within time window

**Concept:** Hard cutoff - only use recent calibration points

**Window interpretation:**
- window = 14 days → use only last 2 weeks
- window = 30 days → use only last month
- window = 60 days → use only last 2 months

**Pros:**
- Very simple (binary weights)
- Guarantees minimum calibration set size if window is large enough
- Clear temporal locality

**Cons:**
- Hard cutoff (discontinuous)
- May have too few calibration points if window is small
- Discards potentially useful older data

**Use when:**
- Want simple recency bias
- Know there's a clear temporal horizon (e.g., "only last month matters")
- Have dense time-series data

---

### 5. Combined (Feature + Time)

**Weight formula:** `w = K_feature(x_query, x_cal) × w_time(t_query, t_cal)`

**When similar to query:** Both feature-wise close AND temporally recent

**Concept:** Multiplicative combination - calibration point must be similar in BOTH feature space and time

**Pros:**
- Captures both spatial and temporal similarity
- Most flexible approach
- Can identify if one dominates

**Cons:**
- More hyperparameters (tau, half_life, omega)
- Risk of over-fitting if not enough data
- Computationally more expensive

**Use when:**
- Both feature similarity and recency likely matter
- Have enough calibration data for joint weighting
- Want to test if temporal drift is significant

---

## Data Size Considerations

With **~3000 hourly timepoints (~125 days)**:

**Sufficient for:**
- ✅ Time exponential decay (14-60 day half-life)
- ✅ Time sliding window (14-60 day window)
- ✅ Combined feature+time (with reasonable parameters)

**Might need more for:**
- ❓ Very long time horizons (>90 days)
- ❓ Seasonal patterns (need full year ideally)

**Why 3k is enough:**
- 125 days covers ~4 months of patterns
- Sufficient to test if recent data matters more
- Enough calibration points for exponential weighting
- Can test multiple half-lives/windows

---

## Experimental Questions

### Q1: Does temporal drift matter?

**Test:** Compare feature kernel vs time exponential

**If time exponential has better coverage:**
→ Yes, recent calibration points are more relevant

**If similar coverage:**
→ No, temporal drift is not significant in this dataset

---

### Q2: What's the right time horizon?

**Test:** Sweep over half_life or window values

**Look for:**
- Optimal half-life (best coverage gap)
- Sensitivity to time horizon

**Interpretation:**
- Short optimal half-life (14 days) → fast concept drift
- Long optimal half-life (60+ days) → slow drift, most data useful

---

### Q3: Does feature + time help?

**Test:** Compare combined vs feature-only vs time-only

**If combined is best:**
→ Both dimensions matter, use hybrid approach

**If one dominates:**
→ Use simpler single-dimension weighting

---

## Usage Example

```python
from experiment_weighting_schemes import run_weighting_schemes_experiment

# Run comprehensive comparison
df_results = run_weighting_schemes_experiment(
    alpha_values=[0.95],
    tau_values=[2.0, 5.0],                 # Feature kernel bandwidth
    half_life_values=[14.0, 30.0, 60.0],   # Time decay (days)
    window_values=[14.0, 30.0, 60.0],      # Time window (days)
)

# Results saved to: data/weighting_schemes_comparison.csv
# Visualization: data/weighting_schemes_comparison.png
```

**Runtime:** ~10-15 minutes (tests 5 methods × multiple configs)

---

## Interpretation Guide

After running experiments, check:

### 1. Coverage Gap Table

```
Method              | Coverage | Gap
--------------------|----------|-------
Binned              | 0.950    | 0.000  ← Simple baseline
Feature kernel      | 0.980    | 0.030  ← Over-conservative
Time exponential    | 0.955    | 0.005  ← If best: time matters!
Time window         | 0.960    | 0.010
Combined            | 0.958    | 0.008
```

**Best method = smallest gap**

### 2. Key Insights

**If time-based ≈ feature-based:**
→ Temporal drift not significant, stick with feature kernel

**If time-based < feature-based:**
→ Recency matters! Use time-based or combined

**If binned ≈ weighted:**
→ Complexity not justified, use simpler binned method

### 3. Spatial Variation (q_hat std)

**Higher std = more adaptive** (corrections vary across query points)

**Expected pattern:**
- Binned: ~0 std (global per bin)
- Feature/Time: ~0.05-0.15 std (smooth variation)
- Combined: similar or higher (if both dimensions matter)

---

## Recommendations

### If you have ~3k timepoints (125 days):

**Try first:**
1. Feature kernel (tau=2.0, 5.0)
2. Time exponential (half_life=30 days)

**If time exponential is better:**
→ Use time-based or combined going forward

**If feature kernel is better:**
→ Temporal drift not significant, stick with current approach

### If over-conservative (98% coverage):

**Priority fixes:**
1. Test absolute deviation (no ens_std scaling) ← Current focus
2. Try time-based weighting (may reduce over-conservatism)
3. Reduce tau (sharper kernel, more local)

### If under-conservative (<90% coverage):

**Priority fixes:**
1. Increase tau (smoother kernel, more conservative)
2. Add safety margin (e.g., 0.02)
3. Check if temporal drift causing miscalibration

---

## Advanced: Custom Weighting Functions

You can define custom weighting schemes by modifying `train_time_weighted_conformal()`:

```python
# Example: Periodic (hour-of-day) weighting
def compute_periodic_weights(hours_cal, hours_query):
    # Circular distance on 24-hour clock
    dt = np.abs(hours_query[:, None] - hours_cal[None, :])
    dt = np.minimum(dt, 24 - dt)  # Wrap around
    return np.exp(-dt / 6.0)  # 6-hour bandwidth

# Example: Seasonal (day-of-year) weighting
def compute_seasonal_weights(doy_cal, doy_query):
    # Circular distance on 365-day year
    dt = np.abs(doy_query[:, None] - doy_cal[None, :])
    dt = np.minimum(dt, 365 - dt)  # Wrap around
    return np.exp(-dt / 60.0)  # 60-day bandwidth
```

---

## Files

```
uncertainty_sets_refactored/
├── experiment_weighting_schemes.py     # Main experiment script
├── WEIGHTING_SCHEMES.md                # This file (documentation)
└── data/
    ├── weighting_schemes_comparison.csv       # Results table
    └── weighting_schemes_comparison.png       # Visualization
```

---

## References

**Feature-based weighting:**
- Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
- Barber et al. (2021) "Conformal Prediction Beyond Exchangeability"

**Time-based weighting:**
- Gibbs & Candès (2021) "Adaptive Conformal Inference"
- Angelopoulos et al. (2023) "Conformal Prediction with Temporal Dependence"

**Exponential weighting for non-stationarity:**
- Vapnik (1998) "Statistical Learning Theory" (importance weighting)
- Sugiyama et al. (2012) "Density Ratio Estimation in Machine Learning"
