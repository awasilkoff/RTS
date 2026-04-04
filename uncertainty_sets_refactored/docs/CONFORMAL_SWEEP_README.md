# Conformal Prediction Hyperparameter Sweep

This directory contains tools for optimizing conformal prediction configuration through systematic hyperparameter sweeps.

## Overview

The sweep tests all combinations of:
- **binning_strategy**: What feature to bin by (`y_pred`, `ens_std`)
- **n_bins**: Number of bins (1, 3, 5, 10, 30)
- **bin_strategy**: How to create bins (`equal_width`, `quantile`)
- **safety_margin**: Conservativeness buffer (0.0, 0.005, 0.01, 0.015, 0.02, 0.03)

## Quick Start

### Option 1: Quick Sweep (~5 minutes)

Reduced parameter grid for rapid iteration:

```bash
cd uncertainty_sets_refactored
python sweep_conformal_quick.py
```

**Parameters tested:**
- Binning strategies: `y_pred`, `ens_std`
- Number of bins: 1, 3, 10
- Bin strategies: `equal_width`, `quantile`
- Safety margins: 0.0, 0.01, 0.02

**Total:** 144 configurations (4 alphas × 2 binning × 3 n_bins × 2 bin_strat × 3 safety)

### Option 2: Full Sweep (~15 minutes)

Complete parameter grid for thorough optimization:

```bash
cd uncertainty_sets_refactored
python sweep_conformal_config.py
```

**Parameters tested:**
- Binning strategies: `y_pred`, `ens_std`
- Number of bins: 1, 3, 5, 10, 30
- Bin strategies: `equal_width`, `quantile`
- Safety margins: 0.0, 0.005, 0.01, 0.015, 0.02, 0.03

**Total:** 480 configurations (4 alphas × 2 binning × 5 n_bins × 2 bin_strat × 6 safety)

## Workflow

### 1. Run Sweep

```bash
# Quick version (~5 min)
python sweep_conformal_quick.py

# OR full version (~15 min)
python sweep_conformal_config.py
```

**Outputs:**
- `data/viz_artifacts/conformal_sweep/conformal_sweep_results.csv` - All results
- `data/viz_artifacts/conformal_sweep/conformal_best_configs.json` - Best config per alpha

### 2. Visualize Results

```bash
python viz_conformal_param_sweep.py
```

**Generates 4 plots per alpha value:**
1. **Coverage gap vs n_bins** - How many bins is optimal?
2. **Coverage gap vs safety_margin** - How conservative to be?
3. **Heatmap (n_bins × safety_margin)** - Joint optimization
4. **Sensitivity analysis** - Which parameters matter most?

**Outputs:**
- `data/viz_artifacts/conformal_sweep/visualizations/sweep_n_bins_alpha_0.95.png`
- `data/viz_artifacts/conformal_sweep/visualizations/sweep_safety_margin_alpha_0.95.png`
- `data/viz_artifacts/conformal_sweep/visualizations/sweep_heatmap_alpha_0.95.png`
- `data/viz_artifacts/conformal_sweep/visualizations/sweep_sensitivity_alpha_0.95.png`

### 3. Review Results

```bash
# View best configurations
cat data/viz_artifacts/conformal_sweep/conformal_best_configs.json

# View all results (sorted by coverage gap)
cat data/viz_artifacts/conformal_sweep/conformal_sweep_results.csv | head -20
```

### 4. Apply Best Configuration

Use the best config in `run_paper_figures_dayahead_valid.py`:

```python
# Example from conformal_best_configs.json
paths = generate_dayahead_valid_figures(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    alpha_values=[0.85, 0.90, 0.95, 0.99],
    viz_alphas=[0.90, 0.95, 0.99],
    n_bins=5,                        # ← From sweep results
    bin_strategy="quantile",         # ← From sweep results
    binning_strategy="ens_std",      # ← From sweep results
    safety_margin=0.01,              # ← From sweep results
    max_value_filter=200,
)
```

## Interpreting Results

### Success Criterion

**Target:** Coverage gap < 5% (i.e., `|coverage - α| < 0.05`)

Example:
- α = 0.95: Acceptable coverage range is 90%-100%
- Coverage = 94.2% → Gap = 0.8% → ✅ Success
- Coverage = 88.0% → Gap = 7.0% → ⚠️ Failure

### Key Metrics

From `conformal_sweep_results.csv`:

- `coverage_gap`: Distance from target coverage (minimize this)
- `within_5pct`: True if gap < 0.05 (success indicator)
- `improvement`: Gain over pre-conformal coverage
- `q_hat_global`: Global correction factor

### Parameter Interpretation

**binning_strategy:**
- `y_pred`: Bin by predicted value
  - Pro: Natural interpretation (low/medium/high predictions)
  - Con: May have unbalanced uncertainty across bins
- `ens_std`: Bin by ensemble standard deviation
  - Pro: Directly segments by forecast uncertainty
  - Con: Requires ensemble forecasts

**bin_strategy:**
- `equal_width`: Equal spacing across feature range
  - Pro: Interpretable bin boundaries
  - Con: May have very few samples in some bins
- `quantile`: Equal number of samples per bin
  - Pro: Statistical power balanced across bins
  - Con: Bin boundaries adapt to data distribution

**n_bins:**
- 1: No adaptive binning (global correction only)
- 3-5: Coarse adaptation (stable, interpretable)
- 10-30: Fine-grained adaptation (may overfit)

**safety_margin:**
- 0.0: Standard conformal guarantee
- 0.01-0.02: Recommended for conservative bounds
- 0.03+: Very conservative (wider bounds, higher coverage)

## Expected Patterns

### Typical Best Configurations

**For α = 0.95:**
- `binning_strategy`: `ens_std` (binning by uncertainty works well)
- `n_bins`: 3-10 (moderate adaptation)
- `bin_strategy`: `quantile` (balanced sample sizes)
- `safety_margin`: 0.01-0.02 (slight conservativeness helps)

**For α = 0.80-0.85:**
- May need higher `safety_margin` (lower α is harder to achieve)
- Fewer bins often better (less data in each bin at low α)

**For α = 0.99:**
- Lower `safety_margin` sufficient (high α easier to achieve)
- More bins acceptable (plenty of coverage data)

### Trade-offs

**More bins:**
- ✅ More adaptive corrections
- ✅ Better handles heterogeneous data
- ⚠️ Requires more calibration data
- ⚠️ May overfit

**Higher safety margin:**
- ✅ Better coverage at low α
- ✅ Fewer out-of-sample failures
- ⚠️ Wider prediction intervals
- ⚠️ More conservative (may over-cover)

**Quantile vs equal-width:**
- Quantile typically better for skewed features (like `ens_std`)
- Equal-width better for uniform features (like `hour`)

## Output Files

```
data/viz_artifacts/conformal_sweep/
├── conformal_sweep_results.csv          # All configurations tested
├── conformal_best_configs.json          # Best config per alpha
└── visualizations/
    ├── sweep_n_bins_alpha_0.85.png     # n_bins sensitivity (α=0.85)
    ├── sweep_n_bins_alpha_0.90.png     # n_bins sensitivity (α=0.90)
    ├── sweep_n_bins_alpha_0.95.png     # n_bins sensitivity (α=0.95)
    ├── sweep_n_bins_alpha_0.99.png     # n_bins sensitivity (α=0.99)
    ├── sweep_safety_margin_alpha_0.85.png
    ├── sweep_safety_margin_alpha_0.90.png
    ├── sweep_safety_margin_alpha_0.95.png
    ├── sweep_safety_margin_alpha_0.99.png
    ├── sweep_heatmap_alpha_0.85.png
    ├── sweep_heatmap_alpha_0.90.png
    ├── sweep_heatmap_alpha_0.95.png
    ├── sweep_heatmap_alpha_0.99.png
    ├── sweep_sensitivity_alpha_0.85.png
    ├── sweep_sensitivity_alpha_0.90.png
    ├── sweep_sensitivity_alpha_0.95.png
    └── sweep_sensitivity_alpha_0.99.png
```

## Customization

### Test Different Parameters

Edit `sweep_conformal_config.py` main block:

```python
results_df = run_conformal_sweep(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    alpha_values=[0.85, 0.90, 0.95, 0.99],           # ← Coverage targets
    binning_strategies=["y_pred", "ens_std"],        # ← Add more features
    n_bins_values=[1, 3, 5, 10, 30],                 # ← Test different bin counts
    bin_strategies=["equal_width", "quantile"],       # ← Both strategies
    safety_margins=[0.0, 0.005, 0.01, 0.02, 0.03],   # ← Conservativeness range
    max_value_filter=200,                             # ← Data filtering
)
```

### Add Custom Binning Features

To test binning by other features (e.g., `hour`, `ens_mean`):

```python
binning_strategies=["y_pred", "ens_std", "ens_mean", "hour"]
```

Note: Feature must be available in the DataFrame (see `add_dayahead_valid_features()`).

## FAQ

**Q: Which sweep should I run first?**
A: Start with `sweep_conformal_quick.py` (~5 min) to get a sense of the parameter landscape. If results look good, run the full sweep for final optimization.

**Q: How do I know if my results are good?**
A: Check `within_5pct` column in results CSV. If most/all configs achieve this, you're in good shape. If none do, you may need to expand the parameter grid or add more features.

**Q: Can I test more alpha values?**
A: Yes, edit `alpha_values=[...]` in the sweep script. But be aware this multiplies the number of configurations linearly.

**Q: Why does quantile binning usually win?**
A: Ensemble standard deviation (`ens_std`) is right-skewed (many small values, few large). Quantile binning ensures each bin has enough data for reliable calibration, while equal-width bins may have sparse high-uncertainty bins.

**Q: What if all configs fail (gap > 5%)?**
A: Try:
1. Increase `safety_margin` range (test 0.03-0.05)
2. Add more features (see `CONFORMAL_PREDICTION_README.md`)
3. Check data quality (outliers, missing values)
4. Increase model complexity (`n_estimators`, `num_leaves`)

## References

- Main conformal prediction module: `conformal_prediction.py`
- Feature engineering: `run_paper_figures_dayahead_valid.py`
- Conformal prediction theory: `CONFORMAL_PREDICTION_README.md`
