# Time-Exponential Weighted Conformal Visualization

## Overview

Produces publication-ready figures analyzing **time-exponential weighted conformal prediction** - a method that gives more weight to recent calibration data, suitable for day-ahead operations.

## Key Features

### 4-Panel Analysis Figure

**Panel 1: Correction vs Target Coverage**
- Shows how `q_hat` (conformal correction) scales with target coverage level (α)
- Includes spatial variation (±1 std bands)
- Demonstrates relationship between desired coverage and correction magnitude

**Panel 2: Diurnal Pattern**
- Hour-of-day analysis of conformal corrections
- Identifies if certain hours need more/less correction
- Useful for understanding when wind is harder to predict

**Panel 3: Temporal Variation**
- Timeseries showing how corrections evolve over time
- First 200 test points for clarity
- Shows smooth adaptation vs constant correction

**Panel 4: Coverage Calibration**
- Perfect calibration: achieved coverage = target coverage
- Includes ±5% tolerance bands
- Validates that method achieves target coverage

### Hourly Patterns Figure

- Separate panel for each alpha level
- Detailed hour-of-day analysis with 95% confidence intervals
- Highlights peak/trough hours if significant variation exists
- Shows how diurnal patterns change with coverage level

### Summary Statistics Table

CSV with:
- Coverage metrics (target, achieved, gap)
- q_hat statistics (mean, std, min, max, hour_range)
- Error metrics (RMSE, MAE)
- Configuration (half_life_days)

## Usage

### Basic Run

```bash
cd uncertainty_sets_refactored
python viz_time_exponential_conformal.py
```

**Runtime:** ~3-5 minutes (trains at 3 alpha levels)

### Custom Settings

```python
from viz_time_exponential_conformal import run_time_exponential_visualization

# Custom alpha levels
run_time_exponential_visualization(
    alpha_values=[0.90, 0.95, 0.97, 0.99],
    half_life_days=30.0,
    causal=True,
    min_lag_days=1.0,
    output_subdir='time_exponential',
)

# Test different half-lives
for half_life in [14.0, 30.0, 60.0]:
    run_time_exponential_visualization(
        alpha_values=[0.95],
        half_life_days=half_life,
        output_subdir=f'time_exp_hl{int(half_life)}',
    )
```

## Outputs

All saved to `data/viz_artifacts/time_exponential/`:

```
time_exponential_conformal_analysis.png       # 4-panel figure (PNG, 150 DPI)
time_exponential_conformal_analysis.pdf       # 4-panel figure (PDF, 300 DPI, for paper)
time_exponential_hourly_patterns.png          # Hour-of-day detailed (PNG)
time_exponential_hourly_patterns.pdf          # Hour-of-day detailed (PDF, for paper)
time_exponential_summary.csv                  # Summary statistics
```

## Key Insights from Visualizations

### 1. Coverage Scaling (Panel 1)

**If q_hat increases linearly with α:**
→ Coverage is well-calibrated across all levels

**If q_hat plateaus:**
→ Method may be conservative at high α

**If spatial variation (std) is high:**
→ Corrections are query-dependent (adaptive)

### 2. Diurnal Patterns (Panel 2)

**If peak hours visible:**
→ Certain times of day are harder to predict (need more correction)

**Expected pattern for wind:**
- Higher corrections during ramp hours (morning/evening)
- Lower corrections during stable hours (night/midday)

**If hour_range > 0.2:**
→ Significant diurnal variation, worth modeling explicitly

### 3. Temporal Variation (Panel 3)

**If timeseries is smooth:**
→ Corrections adapt gradually (good for stability)

**If timeseries is noisy:**
→ May need larger half_life or more calibration data

**If trending upward/downward:**
→ Possible concept drift (forecast quality changing)

### 4. Coverage Calibration (Panel 4)

**If points lie on diagonal:**
→ Perfect calibration ✓

**If points above diagonal:**
→ Over-conservative (achieving > target coverage)

**If points below diagonal:**
→ Under-conservative (achieving < target coverage) ⚠

## Comparison with Other Methods

After running this, compare with:

**Binned conformal:**
- Discrete bins vs smooth time decay
- q_hat std = 0 (per bin) vs continuous variation

**Feature-kernel weighted:**
- Feature similarity vs temporal proximity
- Which matters more for your application?

**Combined (feature + time):**
- Does adding time improve feature-only?
- Or is time sufficient?

## Parameter Tuning

### Half-life Selection

**14 days (fast decay):**
- For rapidly changing wind patterns
- More reactive to recent data
- Risk: insufficient calibration data

**30 days (moderate decay):**
- Good default for monthly-scale patterns
- Balances recency vs data sufficiency

**60 days (slow decay):**
- For stable patterns
- More calibration data used
- Less reactive to recent changes

**Rule of thumb:** Choose half_life ≈ time horizon over which wind patterns change

### Causal Constraint

**Always use `causal=True` for operational deployment**
- Only uses past data (no cheating)
- Realistic for day-ahead

**Can test `causal=False` for comparison:**
- Symmetric window (experimental)
- Shows upper bound on performance

### Minimum Lag

**`min_lag_days=1.0` for day-ahead:**
- Can't use same-day data
- Predicting tomorrow using yesterday or earlier

**`min_lag_days=0.0` for real-time:**
- Can use all past data up to current time
- Not realistic for batch day-ahead process

## Example Interpretation

```
Alpha = 0.95
  Coverage: 0.948 (gap: 0.002)  ← Excellent calibration!
  q_hat: mean=1.652, std=0.089  ← Moderate spatial variation

Hour-of-day range: 0.243  ← Significant diurnal pattern
  Peak hour: 18 (6 PM)
  Trough hour: 4 (4 AM)

Interpretation:
- Method is well-calibrated (0.2% gap)
- Evening hours need ~12% more correction than early morning
- Corrections vary smoothly in time (std=0.089)
- Consider modeling hour-of-day explicitly if targeting <1% gap
```

## LaTeX Integration

For paper figures:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{time_exponential_conformal_analysis.pdf}
\caption{Time-exponential weighted conformal prediction analysis.
  (a) Correction magnitude scales with target coverage.
  (b) Diurnal pattern shows higher corrections during evening hours.
  (c) Corrections adapt smoothly over time.
  (d) Coverage calibration validates method accuracy.}
\label{fig:time_exp_analysis}
\end{figure}
```

## Related Scripts

- `experiment_weighting_schemes.py` - Compare all weighting methods
- `run_paper_figures_dayahead_valid.py` - Original paper figures
- `viz_conformal_paper.py` - General conformal visualization

## References

**Exponential time weighting:**
- Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
- Angelopoulos et al. (2023) "Conformal Prediction with Temporal Dependence"

**Importance weighting:**
- Sugiyama et al. (2012) "Density Ratio Estimation in Machine Learning"
- Vapnik (1998) "Statistical Learning Theory"
