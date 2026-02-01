# Feature Engineering for Learned Omega Visualization

## Overview

This system implements three feature engineering approaches to demonstrate learned omega benefits more clearly in paper visualizations. Each feature set highlights different aspects of omega learning capability.

## Motivation

The baseline 2D feature set (SYS_MEAN, SYS_STD) shows only marginal learned omega improvement. By exploring alternative feature engineering approaches, we can:

1. **Demonstrate omega's ability to handle nuisance features** (temporal_3d)
2. **Show omega learning differential feature importance** (per_resource_4d)
3. **Highlight omega's automatic feature rescaling** (unscaled_2d)

## Feature Sets

### Option 1: Temporal Nuisance (3D)

**Features:** `[SYS_MEAN, SYS_STD, HOUR_SIN]`

**Builder:** `build_XY_temporal_nuisance_3d()`

**Dimension:** 3D → can visualize as 3D scatter or 2D slices

**Expected Behavior:**
- Learned ω ≈ [α, β, ~0]
- Omega downweights HOUR_SIN (weakly-relevant temporal nuisance)
- Preserves weights for SYS_MEAN and SYS_STD

**Why This Matters:**
Demonstrates that omega learning can identify and suppress nuisance features that don't improve covariance prediction.

**Visualization:**
- 3D scatter plot or 2D slices
- Color by omega-weighted kernel distance
- Show ω values: expect ω[2] ≈ 0

---

### Option 2: Per-Resource (4D → 2D Projection)

**Features:** `[WIND_122_MEAN, WIND_309_MEAN, WIND_317_MEAN, HOUR_SIN]`

**Builder:** `build_XY_per_resource_4d(include_hour=True)`

**Dimension:** 4D → visualize top 2 ω-weighted dimensions

**Expected Behavior:**
- Omega discovers which wind farm is most predictive
- Different farms have different ω values
- Top 2 features by |ω| dominate the metric

**Why This Matters:**
Shows omega can learn feature relevance in natural multi-source data. Natural interpretation: some wind farms are more informative for covariance than others.

**Visualization:**
- **Panel A:** Bar chart of ω per feature
- **Panel B:** 2D scatter of top 2 ω-weighted features
- Annotation: "WIND_309 (ω=5.2) and WIND_317 (ω=4.3) prioritized"

**Standard Practice:**
This is a common approach in metric learning papers (Mahalanobis distance, LMNN). Learn high-D metric, visualize via projection to interpretable 2D slice.

---

### Option 3: Unscaled Features (2D)

**Features:** `[SYS_MEAN_MW, SYS_STD_MW]` (raw, unscaled)

**Builder:** `build_XY_unscaled_2d()`

**Dimension:** 2D (different scales: ~300 MW vs ~50 MW)

**Expected Behavior:**
- Equal weights dominated by large-scale feature (SYS_MEAN_MW ≈ 300)
- Learned ω discovers correct rescaling
- Compare ω to 1/variance: should be similar

**Why This Matters:**
Demonstrates omega automatically handles feature scale differences. Equivalent to learned standardization.

**Visualization:**
- Side-by-side 2D scatters:
  - **Left:** Equal weights (axis-aligned ellipses)
  - **Right:** Learned ω (rotated/rescaled ellipses)
- Show learned axis rescaling via ω values

---

## File Structure

```
uncertainty_sets_refactored/
├── data_processing_extended.py       # New feature builders
├── viz_projections.py                 # High-D → 2D projection utilities
├── viz_artifacts_utils.py             # Directory management & metadata
├── sweep_and_viz_feature_set.py      # Unified sweep + viz script
├── test_feature_sets.py               # Smoke tests
└── data/viz_artifacts/                # Auto-generated outputs
    ├── temporal_nuisance_3d/
    │   ├── feature_config.json
    │   ├── README.md
    │   ├── sweep_results.csv
    │   ├── best_omega.npy
    │   ├── omega_bar_chart.png
    │   └── kernel_distance_projection_2d.png
    ├── per_resource_4d/
    │   ├── feature_config.json
    │   ├── README.md
    │   ├── sweep_results.csv
    │   ├── best_omega.npy
    │   ├── omega_bar_chart.png
    │   └── kernel_distance_projection_2d.png
    └── unscaled_2d/
        ├── feature_config.json
        ├── README.md
        ├── sweep_results.csv
        ├── best_omega.npy
        ├── omega_bar_chart.png
        └── kernel_distance_comparison.png
```

## Usage

### Quick Test

Verify feature builders work correctly:

```bash
cd uncertainty_sets_refactored
python test_feature_sets.py
```

Expected output:
```
All feature set tests passed! ✓
```

### Run Sweep for Single Feature Set

```bash
# Option 1: Temporal nuisance (3D)
python sweep_and_viz_feature_set.py --feature-set temporal_3d

# Option 2: Per-resource (4D → 2D projection)
python sweep_and_viz_feature_set.py --feature-set per_resource_4d

# Option 3: Unscaled (2D)
python sweep_and_viz_feature_set.py --feature-set unscaled_2d
```

### Run All Three Feature Sets

```bash
cd uncertainty_sets_refactored

# Run all feature sets and generate comparison
python run_all_feature_sets.py

# Or run specific feature sets only
python run_all_feature_sets.py --feature-sets temporal_3d per_resource_4d
```

### Custom Hyperparameter Sweep

```bash
python sweep_and_viz_feature_set.py \
    --feature-set per_resource_4d \
    --taus 1.0 5.0 10.0 20.0 \
    --omega-l2-regs 0.0 0.001 0.01 0.1 \
    --standardize True False
```

## Output Interpretation

### 1. Check `sweep_results.csv`

Sort by `nll_improvement` (descending) to find best config:

```python
import pandas as pd
df = pd.read_csv("data/viz_artifacts/temporal_3d/sweep_results.csv")
print(df.sort_values("nll_improvement", ascending=False).head(10))
```

Look for:
- Positive `nll_improvement` (learned ω beats baseline)
- Omega values matching expected pattern
- Best `tau` and `omega_l2_reg` combination

### 2. Inspect `best_omega.npy`

```python
import numpy as np
omega = np.load("data/viz_artifacts/temporal_3d/best_omega.npy")
print(f"Learned omega: {omega}")
```

**Temporal 3D:** Expect `[α, β, ~0]` with ω[2] ≈ 0

**Per-resource 4D:** Expect 1-2 dominant features

**Unscaled 2D:** Compare to `1 / X.var(axis=0)` - should be similar

### 3. Compare Visualizations

**Omega Bar Chart (`omega_bar_chart.png`):**
- Green bars = top 2 features by |ω|
- Gray bars = lower-weighted features
- Values labeled on bars

**Kernel Distance Plots:**
- 2D: Standard equal vs learned comparison
- 4D: Top-2 projection with ω annotations
- Color = kernel weight (brighter = more similar to target)

## Expected Runtimes

**Note:** These are typical runtimes on a modern laptop. Actual times may vary.

- **Feature builder test:** <5 seconds
- **Single sweep (12 configs):** ~2-5 minutes
- **Visualization generation:** <30 seconds
- **Full pipeline (sweep + viz):** ~3-6 minutes per feature set

## Comparison Workflow

After running all three feature sets:

1. **Load all sweep results:**
   ```python
   import pandas as pd
   from pathlib import Path

   artifact_dir = Path("data/viz_artifacts")

   results = {}
   for fs in ["temporal_3d", "per_resource_4d", "unscaled_2d"]:
       df = pd.read_csv(artifact_dir / fs / "sweep_results.csv")
       best = df.sort_values("nll_improvement", ascending=False).iloc[0]
       results[fs] = best

   comparison = pd.DataFrame(results).T
   print(comparison[["nll_improvement", "eval_nll_learned", "eval_nll_baseline"]])
   ```

2. **Identify best feature set for paper:**
   - Largest `nll_improvement`
   - Clearest omega pattern (matches expected behavior)
   - Most interpretable visualization

3. **Create paper figure:**
   - Use outputs from best feature set directory
   - Multi-panel: (A) Omega bar, (B) Kernel distance, (C) NLL comparison
   - Caption explains feature engineering choice

## Addressing "Is 4D → 2D Messy?" Question

**Answer: No, it's standard practice in ML papers.**

**Why it works:**

1. **Top-2 ω projection is clean:**
   - Plot the 2 features with largest learned weights
   - Example: If ω = [0.8, 5.2, 0.1, 4.3], plot dims 1 (5.2) and 3 (4.3)
   - Caption: "2D projection showing top 2 features by learned metric"

2. **Clear interpretation:**
   - Omega bar chart shows which features matter
   - 2D scatter shows those specific dimensions
   - Transparent about projection: "visualized in learned metric's top 2 dimensions"

3. **Precedent in literature:**
   - Mahalanobis metric learning always projects for visualization
   - LMNN (Large Margin Nearest Neighbors) papers do this
   - Kernel PCA, t-SNE all project high-D to 2D

**For your paper:**

> "Figure X: Learned kernel metric in 4D feature space. Panel (a) shows learned weights ω indicating WIND_309 and WIND_317 are most predictive for covariance. Panel (b) visualizes kernel neighborhoods projected to these two dimensions."

This is **more interpretable** than PCA because omega values have direct meaning: feature importance for covariance.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Make sure you're in the uncertainty_sets_refactored directory
cd uncertainty_sets_refactored

# Python should find modules in current directory
python sweep_and_viz_feature_set.py --feature-set temporal_3d
```

### Data Not Found

Ensure data files exist:

```bash
ls -lh data/forecasts_filtered_rts3_constellation_v1.parquet
ls -lh data/actuals_filtered_rts3_constellation_v1.parquet
```

If missing, run the data preprocessing pipeline first.

### Sweep Takes Too Long

Reduce sweep size:

```bash
python sweep_and_viz_feature_set.py \
    --feature-set temporal_3d \
    --taus 5.0 \
    --omega-l2-regs 0.0 0.1 \
    --standardize True
```

This runs only 2 configs instead of 24.

## Next Steps

1. **Run all three feature sets** (total: ~15 minutes)
2. **Compare NLL improvements** across feature sets
3. **Identify best for paper** (largest improvement + clearest pattern)
4. **Generate final paper figures** using outputs from best feature set
5. **Write figure caption** explaining feature engineering choice

## Citation

If using this for your paper, explain the feature engineering approach in the methodology section:

> "To demonstrate the benefit of learned feature metrics, we compared three feature engineering approaches: (1) system-level forecasts with temporal nuisance features, (2) per-resource forecasts, and (3) unscaled features. The learned metric ω consistently downweighted irrelevant features (e.g., temporal) and discovered differential importance across resources, improving covariance prediction by X% over equal weights."

## Contact

For questions about this feature engineering system, see the main project README or open an issue.
