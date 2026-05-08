# Thorough Overnight Grid Search Configuration

## Overview

For publication-quality results, we provide an expanded hyperparameter grid that comprehensively explores:
- **Tau (kernel bandwidth):** 5 values from local to global smoothing
- **Omega L2 regularization:** 5 values from no shrinkage to strong shrinkage
- **Scaler types:** All 3 normalization approaches

## Grid Configuration

### Hyperparameters

**Tau (Kernel Bandwidth):** `[1.0, 2.0, 5.0, 10.0, 20.0]`
- `1.0`: Very local (small neighborhoods)
- `2.0`: Local (default for quick runs)
- `5.0`: Moderate (default for quick runs)
- `10.0`: Semi-global
- `20.0`: Global (large neighborhoods)

**Omega L2 Regularization:** `[0.0, 1e-3, 1e-2, 1e-1, 1.0]`
- `0.0`: No shrinkage (learn freely)
- `1e-3`: Very weak shrinkage
- `1e-2`: Weak shrinkage (default for quick runs)
- `1e-1`: Moderate shrinkage
- `1.0`: Strong shrinkage (close to baseline)

**Scaler Types:** `["none", "standard", "minmax"]`
- `none`: No scaling (raw features)
- `standard`: StandardScaler (mean=0, std=1)
- `minmax`: MinMaxScaler ([0,1] range)

### Grid Size

**Per feature set:** 5 taus × 5 regs × 3 scalers = **75 configs**

**For 2 feature sets** (focused_2d, high_dim_8d): **150 total configs**

**For all 5 feature sets:** **375 total configs**

## Runtime Estimates

Based on empirical data (~1-2 min per config):

| Feature Sets | Configs | Estimated Runtime |
|--------------|---------|-------------------|
| 2 (new only) | 150     | 3-5 hours        |
| 5 (all)      | 375     | 6-12 hours       |

**Recommendation:** Run 2 new feature sets first (focused_2d, high_dim_8d) to complete overnight. If results are promising, run all 5 on a subsequent night.

## Usage

### Default: New Feature Sets (Recommended)

```bash
cd uncertainty_sets_refactored
python run_thorough_overnight.py
```

Runs: `focused_2d`, `high_dim_8d`
Expected: 3-5 hours

### All Feature Sets

```bash
python run_thorough_overnight.py --all-feature-sets
```

Runs: All 5 feature sets
Expected: 6-12 hours

### Custom Feature Sets

```bash
python run_thorough_overnight.py --feature-sets focused_2d temporal_3d
```

### Custom Hyperparameters

```bash
python run_thorough_overnight.py \
    --taus 1.0 5.0 10.0 \
    --omega-l2-regs 0.0 1e-2 1e-1 \
    --scaler-types standard minmax
```

This reduces the grid to: 3 × 3 × 2 = 18 configs per feature set

## What to Expect

### During Execution

The script will:
1. Print configuration summary and runtime estimate
2. Ask for confirmation before starting
3. Show progress for each feature set
4. Print timing and best configs as it runs

### Outputs

All results saved to: `data/viz_artifacts/<feature_set>/`

**Per feature set:**
- `sweep_results.csv` - All 75 configs with NLL scores
- `best_omega.npy` - Best learned weights
- `omega_bar_chart.png` - Feature importance visualization
- `kernel_distance_*.png` - Learned metric visualization
- `feature_config.json` - Best configuration metadata

**Global comparison:**
- `feature_set_comparison.csv` - Best config from each feature set

## Analysis After Completion

### 1. Check Overall Best

```bash
cat data/viz_artifacts/feature_set_comparison.csv
```

Look for:
- Largest `nll_improvement`
- Clearest omega pattern
- Best scaler type for each feature set

### 2. Inspect Hyperparameter Sensitivity

```python
import pandas as pd

# Load full sweep for focused_2d
df = pd.read_csv("data/viz_artifacts/focused_2d/sweep_results.csv")

# Best scaler type?
print(df.groupby("scaler_type")["nll_improvement"].max())

# Best tau?
print(df.groupby("tau")["nll_improvement"].max())

# Best regularization?
print(df.groupby("omega_l2_reg")["nll_improvement"].max())

# Top 10 configs
print(df.sort_values("nll_improvement", ascending=False).head(10))
```

### 3. Analyze Omega Patterns

```python
import numpy as np
import pandas as pd

df = pd.read_csv("data/viz_artifacts/high_dim_8d/sweep_results.csv")

# Get omega columns
omega_cols = [c for c in df.columns if c.startswith("omega_") and "_" in c.split("omega_")[1]]

# Top 5 configs
top5 = df.sort_values("nll_improvement", ascending=False).head(5)

# Print learned omega for each
for idx, row in top5.iterrows():
    omega = row[omega_cols].values
    print(f"\nConfig: tau={row['tau']}, reg={row['omega_l2_reg']}, scaler={row['scaler_type']}")
    print(f"Omega: {omega}")
    print(f"NLL improvement: {row['nll_improvement']:.3f}")
```

### 4. Compare Scalers Head-to-Head

```python
import pandas as pd

df = pd.read_csv("data/viz_artifacts/focused_2d/sweep_results.csv")

# For each (tau, omega_l2_reg) pair, compare scaler types
comparison = df.pivot_table(
    index=["tau", "omega_l2_reg"],
    columns="scaler_type",
    values="nll_improvement"
)

print(comparison.sort_values("minmax", ascending=False))

# Which scaler wins most often?
print("\nWins by scaler:")
print((comparison.idxmax(axis=1).value_counts()))
```

## Expected Findings

### For focused_2d

**Hypothesis:** MinMax should show clearest omega benefit
- MinMax eliminates scale dominance
- Omega learns feature importance more clearly
- Larger NLL improvement than Standard

**Expected best config:**
- Scaler: `minmax`
- Tau: 2.0-5.0 (moderate)
- Reg: 0.0-1e-2 (low regularization)

### For high_dim_8d

**Hypothesis:** Omega prioritizes most predictive features
- System-level (SYS_MEAN, SYS_STD) might dominate
- Or specific farm (e.g., WIND_309) might be most predictive
- Std features might be downweighted if mean is sufficient

**Expected best config:**
- Scaler: `minmax` or `standard` (unclear which)
- Tau: 5.0-10.0 (need larger neighborhoods for 8D)
- Reg: 0.0-1e-2 (low regularization to learn 8 weights)

**Key question:** Which features does omega prioritize?
- Check `omega_bar_chart.png` for visual hierarchy
- Check `best_omega.npy` for exact values

### For temporal_3d (if running all)

**Expected:** ω ≈ [α, β, ~0] with HOUR_SIN downweighted
- Demonstrates omega suppresses irrelevant features

### For per_resource_4d (if running all)

**Expected:** 1-2 farms dominate
- Shows omega discovers differential farm importance

## Interrupting Long Runs

If you need to stop early:

1. **Ctrl+C**: Gracefully stops and shows partial results
2. **Partial results**: Available in `data/viz_artifacts/` for completed feature sets
3. **Resume**: Simply re-run - completed feature sets will show "already exists" warning

**To force re-run a specific feature set:**
```bash
rm -rf data/viz_artifacts/focused_2d
python run_thorough_overnight.py --feature-sets focused_2d
```

## Reducing Grid Size (If Needed)

If 3-5 hours is too long, reduce the grid:

**Moderate grid (45 configs/fs, ~90-150 min):**
```bash
python run_thorough_overnight.py \
    --taus 1.0 5.0 20.0 \
    --omega-l2-regs 0.0 1e-2 1e-1 \
    --scaler-types standard minmax
```
3 taus × 3 regs × 2 scalers = 18 configs/fs

**Minimal grid (12 configs/fs, ~30-60 min):**
```bash
python run_thorough_overnight.py \
    --taus 2.0 10.0 \
    --omega-l2-regs 0.0 1e-2 \
    --scaler-types standard minmax
```
2 taus × 2 regs × 2 scalers = 8 configs/fs

## Paper-Ready Outputs

After thorough grid search, you'll have:

1. **Best feature set identified** (largest NLL improvement)
2. **Best hyperparameters** (tau, omega_l2_reg, scaler_type)
3. **Learned omega** showing feature importance
4. **Visualizations** (omega bar chart, kernel distance)
5. **Quantitative results** (NLL improvement, statistical significance)

Use these for:
- **Figure 1:** Omega bar chart from best feature set
- **Figure 2:** Kernel distance visualization
- **Table 1:** NLL comparison across feature sets
- **Methodology:** Explain hyperparameter selection (cross-validation)

## Troubleshooting

### Out of Memory

If you run out of memory:
- Reduce number of feature sets (run sequentially)
- Reduce grid size (fewer taus/regs)
- Close other applications

### Very Slow

If slower than expected:
- Check CPU usage (should be ~100% during optimization)
- Check if disk I/O is bottleneck (SSD vs HDD)
- Consider reducing `k` (number of neighbors) in sweep script

### Inconsistent Results

If results vary between runs:
- Check random seed is fixed (seed=42 in train/test split)
- Verify no stochastic components in optimization
- May indicate optimization didn't converge (increase max_iters)

## Summary

**Default command:**
```bash
python run_thorough_overnight.py
```

**What it does:**
- Runs 75 configs for focused_2d (2D baseline)
- Runs 75 configs for high_dim_8d (8D exploration)
- Total: 150 configs in 3-5 hours

**What you get:**
- Comprehensive hyperparameter exploration
- Best config for each feature set
- Clear answer: Which scaler works best?
- Clear answer: Which features does omega prioritize?
- Publication-ready figures and results

**Next morning:**
```bash
# Check results
cat data/viz_artifacts/feature_set_comparison.csv

# Inspect best feature set
ls data/viz_artifacts/focused_2d/
ls data/viz_artifacts/high_dim_8d/
```

Ready for your paper! 📊✨
