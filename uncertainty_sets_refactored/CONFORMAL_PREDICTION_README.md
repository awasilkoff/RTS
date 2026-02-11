# Conformal Prediction for Wind Power Lower Bounds

This document explains the conformal prediction approach used to calibrate probabilistic lower bounds for total wind power generation.

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Architecture](#architecture)
4. [Usage Guide](#usage-guide)
5. [Integration with Covariance Pipeline](#integration-with-covariance-pipeline)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Technical Details](#technical-details)

---

## Overview

**Goal**: Provide a calibrated lower bound for total wind power that achieves a target coverage probability (e.g., 95% of actual values exceed the bound).

**Why Conformal Prediction?**
- **Distribution-free**: No parametric assumptions about the data
- **Finite-sample guarantees**: Coverage holds for any sample size (not just asymptotically)
- **Adaptive**: Can adjust corrections based on prediction context (via binning)

**Pipeline**:
```
Historical Data → Train Quantile Model → Calibrate Conformal Correction → Deploy Lower Bound
```

---

## Key Concepts

### 1. Quantile Regression (Base Model)

We use LightGBM to train a quantile regressor at level `quantile_alpha` (e.g., 0.10 for 10th percentile):

```python
y_pred_base ≈ Q_α(y | features)
```

**Features**:
- `ens_mean`: Ensemble mean forecast
- `ens_std`: Ensemble standard deviation (uncertainty estimate)
- `ens_min`, `ens_max`: Ensemble range
- `n_models`: Number of contributing models
- `hour`, `dow`: Time features

**Problem**: Quantile models are often miscalibrated. The nominal 10th percentile might actually be the 15th percentile empirically.

### 2. Conformal Correction (Calibration Layer)

Conformal prediction adds a **data-driven margin** to ensure coverage:

```python
y_pred_conf = y_pred_base - margin
```

where `margin = q_hat * scale`.

**Nonconformity Score** (measures how wrong the base prediction is):
```python
r = max(0, (y_pred_base - y_true) / scale)
```

- `r > 0`: Prediction was too high (underestimated wind)
- Normalized by `scale` (e.g., ensemble std) for heteroscedasticity

**Conformal Quantile** `q_hat`:
- Computed on a **calibration set** (separate from training)
- `q_hat = quantile(r_cal, level)` where `level = ceil((n+1)*(1-α))/n`
- For `α=0.95` (95% coverage target), we use ~95th percentile of calibration errors

### 3. Binned Adaptive Conformal Prediction

**Problem**: A single global `q_hat` may be too conservative in some contexts and insufficient in others.

**Solution**: Compute **bin-specific** corrections based on a binning feature.

**Binning Options**:
1. **`binning="y_pred"`**: Bin by predicted value
   - Intuition: Correction varies by forecast magnitude (e.g., low wind vs high wind)
2. **`binning="feature:ens_std"`**: Bin by ensemble spread
   - Intuition: High uncertainty → larger correction needed
3. **`binning="y_actual"`**: Bin by actual values (calibration), use predictions as proxy (prediction)
   - Intuition: Error patterns vary by actual generation level (e.g., capacity constraints)
   - See `CONFORMAL_Y_ACTUAL_README.md` for detailed documentation

**Example bins** (quantile-based):
```python
[0%, 10%, 25%, 50%, 75%, 90%, 100%]
```

Each bin gets its own `q_hat_bin`, computed from calibration samples falling in that bin.

**Fallback**: If a bin is empty (rare in test data), use global `q_hat` as fallback.

---

## Architecture

### Core Components

```
conformal_prediction.py
├── ConformalLowerBundle       # Trained model + conformal corrections
│   ├── model                  # LGBMRegressor (quantile)
│   ├── q_hat_global_r         # Global conformal correction
│   ├── q_hat_by_bin_r         # Dict[bin → q_hat]
│   ├── bin_edges              # Bin boundaries
│   └── predict_df()           # Apply to new data
│
├── train_wind_lower_model_conformal_binned()
│   # End-to-end training function
│   # Returns: (bundle, metrics, df_test)
│
└── compute_binned_adaptive_conformal_corrections_lower()
    # Core conformal calibration logic
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Time-Ordered Split: TRAIN | CAL | TEST                      │
└─────────────────────────────────────────────────────────────┘
                │
                ├─ TRAIN: Fit quantile model (LightGBM)
                │         Input: (X_train, y_train)
                │         Output: y_pred_base = model(X)
                │
                ├─ CAL:   Compute conformal corrections
                │         1. Predict: y_pred_cal = model(X_cal)
                │         2. Compute errors: r = max(0, (y_pred - y_cal)/scale)
                │         3. Bin by feature: assign each sample to bin
                │         4. Per-bin q_hat: quantile(r_bin, conformal_level)
                │
                └─ TEST:  Evaluate coverage
                          1. Predict: y_pred_test = model(X_test)
                          2. Lookup: q_hat = q_hat_by_bin[bin(x_test)]
                          3. Correct: y_pred_conf = y_pred_test - q_hat * scale
                          4. Measure: coverage = mean(y_test >= y_pred_conf)
```

**Key Property**: The model never sees TEST data. CAL data is used only for correction (not model fitting).

---

## Usage Guide

### Basic Example

```python
from pathlib import Path
import pandas as pd
from conformal_prediction import train_wind_lower_model_conformal_binned
from data_processing import build_conformal_totals_df

# Load data
DATA_DIR = Path("data")
actuals = pd.read_parquet(DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet")
forecasts = pd.read_parquet(DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet")

# Build features
df = build_conformal_totals_df(actuals, forecasts)

# Train model
feature_cols = ["ens_mean", "ens_std", "ens_min", "ens_max", "n_models", "hour", "dow"]

bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
    df,
    feature_cols=feature_cols,
    target_col="y",
    scale_col="ens_std",
    alpha_target=0.95,  # 95% coverage target
    quantile_alpha=0.10,  # Start with 10th percentile
    binning="y_pred",
    test_frac=0.2,
    cal_frac=0.2,
)

print(f"Coverage: {metrics['coverage']:.2%}")  # Should be close to 95%
print(f"RMSE: {metrics['rmse']:.2f} MW")
print(f"MAE: {metrics['mae']:.2f} MW")
```

### Prediction on New Data

```python
# Apply trained model to new data
df_new = pd.DataFrame({
    "ens_mean": [500.0, 800.0],
    "ens_std": [50.0, 100.0],
    "ens_min": [400.0, 600.0],
    "ens_max": [600.0, 1000.0],
    "n_models": [10, 10],
    "hour": [12, 18],
    "dow": [1, 3],
})

df_pred = bundle.predict_df(df_new)

print(df_pred[["y_pred_base", "margin", "y_pred_conf"]])
#   y_pred_base  margin  y_pred_conf
# 0        450.0   120.0        330.0  (base - margin)
# 1        750.0   240.0        510.0
```

**Output Columns**:
- `y_pred_base`: Quantile regression prediction
- `q_hat_r`: Conformal correction factor (unitless)
- `scale_sanitized`: Clipped scale (min 1e-3 MW)
- `margin`: q_hat * scale (MW)
- `y_pred_conf`: Final conformal lower bound (MW)
- `bin`: Which bin this sample belongs to
- `bin_feature`: The feature used for binning

---

## Integration with Covariance Pipeline

The conformal prediction module provides the **scale** (ρ) for the ellipsoidal uncertainty set, while covariance optimization provides the **shape** (Σ).

### Workflow in `main.py`

```python
# Step 1: Learn covariance structure (Ω, τ, Σ)
omega_hat = fit_omega(X_cov, Y_cov, ...)
mu_eval, sigma_eval = predict_mu_sigma_topk_cross(X_query, X_ref, Y_ref, omega_hat, ...)

# Step 2: Calibrate conformal lower bound
bundle = train_wind_lower_model_conformal_binned(df, ...)

# Step 3: Compute ρ from total lower bound
for each timestamp t:
    total_lower_bound = bundle.predict_df(features[t])["y_pred_conf"]
    mu_t = mu_eval[t]  # Per-resource mean
    Sigma_t = sigma_eval[t]  # Per-resource covariance

    # Solve for ρ such that: sum(mu) - sqrt(1ᵀ Σ 1) * ρ = total_lower_bound
    rho_t = implied_rho_from_total_lower_bound(Sigma_t, mu_t, total_lower_bound)
```

**Key Insight**: Conformal prediction gives us a calibrated total wind bound. We then **back out** the ellipsoid radius ρ that makes our uncertainty set consistent with this bound.

**Why This Works**:
1. Conformal prediction is **marginal**: calibrates the total (sum of all resources)
2. Covariance optimization is **structural**: learns correlations between resources
3. Together: calibrated bounds + realistic per-resource correlations

---

## Hyperparameter Tuning

### Coverage Target (`alpha_target`)

**Typical Values**: 0.90, 0.95, 0.99

**Trade-off**:
- Higher α → More conservative bounds → Higher ρ → More expensive unit commitment
- Lower α → Tighter bounds → Risk of constraint violations

**Recommendation**: Start with 0.95 (95% coverage), then adjust based on operational risk tolerance.

### Quantile Level (`quantile_alpha`)

**Role**: Starting point for base quantile model (before conformal correction).

**Typical Values**: 0.05, 0.10, 0.15

**Important**: Final coverage is determined by `alpha_target` (via conformal correction), NOT `quantile_alpha`.

**Recommendation**: Use `quantile_alpha = 1 - alpha_target` as a starting point (e.g., 0.05 for 95% coverage), but conformal correction will adjust.

### Binning Strategy

**Options**:
1. **`binning="y_pred"`** (default): Adaptive correction by forecast level
   - Best when model errors vary by prediction magnitude
2. **`binning="feature:ens_std"`**: Adaptive correction by uncertainty
   - Best when ensemble spread is a good indicator of miscalibration
3. **No binning** (single global `q_hat`): Use fixed `bin_edges=[min, max]`

**Bin Edges**:
- Default: Quantile-based `[0%, 10%, 25%, 50%, 75%, 90%, 100%]`
- Custom: Pass `bin_edges=[0, 200, 400, 800, 1200]` (MW thresholds)

**Visualization**: Use `viz_conformal_sweep.py` to compare α targets.

### Split Fractions

**Default**: `train_frac=0.6`, `cal_frac=0.2`, `test_frac=0.2`

**Time-Ordered Split**: Data is sorted by time before splitting (no shuffling).

**Rationale**:
- **60% train**: Enough data for quantile model
- **20% cal**: Compute conformal corrections
- **20% test**: Held-out evaluation (unbiased coverage estimate)

**Important**: Test set simulates future deployment. Coverage on test set is the expected operational coverage.

---

## Technical Details

### Conformal Quantile Level

The exact quantile level for `q_hat` is computed via:

```python
from utils import _conformal_q_level

n = len(calibration_set)
alpha_target = 0.95

q_level = _conformal_q_level(n, alpha_target)
# Returns: ceil((n+1) * (1 - alpha_target)) / n

q_hat = np.quantile(r_cal, q_level)
```

**Why `(n+1)` and `ceil`?**
- Ensures **marginal coverage** holds in finite samples
- See Vovk et al. (2005), "Algorithmic Learning in a Random World"

### Scale Sanitization

To avoid division by zero and numerical instability:

```python
scale = np.where(scale <= min_scale, min_scale, scale)
```

Default `min_scale=1e-3` (1 kW). Adjust if needed.

### Coverage Guarantee

**Theorem** (Conformal Prediction): If calibration and test data are **exchangeable**, then:

```
P(y_test >= y_pred_conf) >= 1 - α_target
```

**In Practice**:
- Exchangeability ≈ i.i.d. or stationary time series
- If distribution shifts over time, coverage may degrade → retrain periodically
- Test coverage is an empirical estimate (may vary due to finite sample size)

### Model Diagnostics

**Check These**:
1. **Pre-conformal coverage**: `mean(y_test >= y_pred_base)`
   - If already >> α_target, quantile model is too conservative → try higher `quantile_alpha`
   - If << α_target, quantile model is too aggressive → conformal correction will be large

2. **Per-bin coverage**: Check if coverage is uniform across bins
   - If some bins have low coverage, increase `cal_frac` or use coarser bins

3. **RMSE/MAE**: Not the primary goal (coverage is), but track to avoid overly conservative bounds

---

## File Reference

**Core Module**:
- `conformal_prediction.py`: Training and inference

**Data Processing**:
- `data_processing.py`: `build_conformal_totals_df()` aggregates actuals/forecasts

**Visualization**:
- `viz_conformal_sweep.py`: Sweep over α_target, plot q_hat vs coverage
- `viz_timeseries_conformal.py`: Timeseries overlay plot
- `viz_conformal_paper.py`: **IEEE paper-ready figures** (NEW)

**Integration**:
- `main.py`: Full pipeline (covariance + conformal + rho calibration)

**Utilities**:
- `utils.py`: `_conformal_q_level()` for exact quantile computation

---

## Quick Start Checklist

- [ ] Verify data files exist: `actuals_filtered_rts3_*.parquet`, `forecasts_filtered_rts3_*.parquet`
- [ ] Run smoke test: `python -c "from conformal_prediction import train_wind_lower_model_conformal_binned; print('OK')"`
- [ ] Train baseline model: `python viz_conformal_sweep.py`
- [ ] Check coverage: Should be within ±5% of `alpha_target` on test set
- [ ] Integrate with main pipeline: `python main.py`
- [ ] Monitor operational coverage: Retrain if coverage degrades

---

## Troubleshooting

**Problem**: Coverage much lower than target (e.g., 80% when targeting 95%)

**Solutions**:
1. Increase `cal_frac` (more calibration data → better q_hat estimate)
2. Use coarser bins (fewer, wider bins → more stable per-bin estimates)
3. Check for distribution shift (calibration period vs test period)

**Problem**: Bounds are too wide (high RMSE/MAE)

**Solutions**:
1. Decrease `alpha_target` (accept more risk for tighter bounds)
2. Improve base quantile model (better features, hyperparameters)
3. Use adaptive binning to make corrections context-dependent

**Problem**: Some bins have 0% or 100% coverage

**Solutions**:
1. Increase calibration sample size
2. Merge adjacent bins (use quantile-based bins with fewer splits)
3. Rely on global `q_hat` fallback

**Problem**: Negative predictions

**Solutions**:
- Post-process: `y_pred_conf = max(0, y_pred_conf)` (physical constraint)
- Adjust `quantile_alpha` to avoid very low base predictions

---

## References

- **Conformal Prediction**: Shafer & Vovk (2008), "A Tutorial on Conformal Prediction"
- **Adaptive Conformal**: Romano et al. (2019), "Conformalized Quantile Regression"
- **Binned Conformal**: Lei & Wasserman (2014), "Distribution-Free Prediction Bands"

---

## IEEE Paper Figures

### Quick Generation

Generate all publication-ready figures for the conformal prediction section:

```bash
cd uncertainty_sets_refactored
python run_paper_figures.py
```

**Output**: `data/viz_artifacts/paper_figures/`
- `fig_timeseries_conformal.png` - Method demonstration
- `fig_calibration_curve.pdf/.png` - Validation of conformal guarantee
- `fig_adaptive_correction.pdf/.png` - Binned correction summary
- `figure_metadata.json` - Metadata for reproducibility

**Estimated time**: 2-3 minutes

### Figure Descriptions

**Figure 1: Timeseries Overlay**
- Shows actual wind generation vs ensemble mean forecast vs conformal lower bound
- Demonstrates method visually over time
- 4 lines: actual, mean forecast, base quantile, conformal bound
- Publication-ready: 14×6 inches, 200 DPI

**Figure 2: Calibration Curve** (NEW)
- Validates conformal guarantee: empirical coverage ≈ target coverage
- Scatter plot with diagonal reference line (y=x)
- Wilson score 95% confidence intervals
- Tolerance band (±5% shaded region)
- Color-coded: green (within tolerance), red (violation)
- Multiple alpha values: 0.80, 0.85, 0.90, 0.95, 0.99

**Figure 3: Adaptive Correction Summary** (NEW)
- 2-panel horizontal layout showing why binned conformal helps
- **Left panel**: q_hat by bin (correction strength varies)
  - Bar chart: correction factor per prediction bin
  - Reference line: global q_hat baseline
- **Right panel**: Coverage by bin (calibration quality)
  - Bar chart: empirical coverage per bin
  - Reference line: target alpha (e.g., 0.95)
  - Color-coded: green (within ±5%), red (violation)

### LaTeX Integration

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_timeseries_conformal.png}
\caption{Conformal prediction lower bound compared to actual wind generation,
ensemble mean forecast, and base quantile prediction (10th percentile).
The conformalized bound achieves 95\% empirical coverage.}
\label{fig:conformal_timeseries}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_calibration_curve.pdf}
\caption{Calibration validation: empirical coverage vs target coverage for
conformal prediction. Points near the diagonal indicate well-calibrated
predictions. Error bars show 95\% Wilson score confidence intervals.}
\label{fig:conformal_calibration}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_adaptive_correction.pdf}
\caption{Adaptive conformal correction summary. (Left) Correction factors
$\hat{q}$ vary by prediction bin, enabling context-specific adjustments.
(Right) Per-bin coverage shows calibration quality across prediction regions,
all within 5\% of the 95\% target.}
\label{fig:conformal_adaptive}
\end{figure*}
```

### Customization

Generate figures with custom parameters:

```python
from pathlib import Path
from viz_conformal_paper import generate_paper_figures

paths = generate_paper_figures(
    data_dir=Path("data"),
    output_dir=Path("custom_output"),
    alpha_values=[0.85, 0.90, 0.95, 0.99],  # Custom alpha sweep
    primary_alpha=0.90,  # Use 90% for correction summary
)
```

Individual figure generation:

```python
from viz_conformal_paper import (
    plot_calibration_curve,
    plot_adaptive_correction_summary,
)

# Calibration curve only
results = [
    {"alpha_target": 0.90, "coverage": 0.895, "n_test": 100},
    {"alpha_target": 0.95, "coverage": 0.948, "n_test": 100},
]
fig = plot_calibration_curve(results, output_path=Path("calibration.pdf"))

# Correction summary only (requires trained bundle)
bundle, metrics, df_test = train_wind_lower_model_conformal_binned(...)
fig = plot_adaptive_correction_summary(
    bundle, df_test, metrics, output_path=Path("correction.pdf")
)
```

### Figure Interpretation

**Calibration Curve**:
- Points on diagonal → method is well-calibrated
- Points above diagonal → conservative (over-coverage)
- Points below diagonal → aggressive (under-coverage)
- Error bars crossing diagonal → statistically consistent

**Adaptive Correction**:
- **Left panel**: Varying bar heights show adaptive behavior
  - Higher bars = larger corrections needed in that region
  - Bars near global line = region-specific correction unnecessary
- **Right panel**: All green bars = good calibration across all bins
  - Red bars = need more calibration data or coarser bins

---

## Contact / Support

For questions or issues related to this module:
1. Check the smoke test: `python conformal_prediction.py` (if `__main__` block exists)
2. Run visualization: `python viz_conformal_sweep.py`
3. Generate paper figures: `python run_paper_figures.py`
4. Review integration: `python main.py`

**Note**: This module is part of the Adaptive Robust Unit Commitment (ARUC) research project for RTS-GMLC.
