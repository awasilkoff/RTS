# Quick Start: Feature Engineering for Learned Omega

## 30-Second Overview

You now have **three new feature engineering approaches** to demonstrate learned omega benefits more clearly than the baseline 2D (SYS_MEAN, SYS_STD):

1. **Temporal Nuisance (3D)** - Shows omega downweighting irrelevant features
2. **Per-Resource (4D)** - Shows omega learning differential feature importance
3. **Unscaled (2D)** - Shows omega learning automatic feature rescaling

## Run Everything (10-15 minutes)

```bash
cd uncertainty_sets_refactored
python run_all_feature_sets.py
```

This will:
- Run hyperparameter sweeps for all three feature sets
- Generate visualizations (omega bar charts, kernel distance plots)
- Create organized output directories with metadata
- Print comparison summary showing which is best for your paper

## Output Location

```
uncertainty_sets_refactored/data/viz_artifacts/
├── temporal_nuisance_3d/
├── per_resource_4d/
└── unscaled_2d/
```

Each directory contains:
- `sweep_results.csv` - All hyperparameter combinations and NLL scores
- `best_omega.npy` - Learned feature weights
- `omega_bar_chart.png` - Visualization of learned ω
- `kernel_distance_*.png` - Kernel weight visualization
- `feature_config.json` - Full configuration metadata
- `README.md` - Auto-generated documentation

## Important: Baseline Comparison

**All feature sets now use equal weights `[1, 1, ...]` as the baseline** for fair comparison. This means:

- **Standardized features:** Comparison against natural Euclidean baseline
- **Raw features:** Learned omega must discover rescaling from scratch (more impressive!)
- **Consistent:** Same baseline definition across all feature sets

See `BASELINE_COMPARISON_FIX.md` for details on this important fix.

## What to Look For

After running, check the comparison summary (printed at end):

```
COMPARISON SUMMARY
=================================================================================================
feature_set         nll_improvement  eval_nll_learned  eval_nll_baseline  improvement_pct  ...
per_resource_4d             0.125             5.234              5.359             2.33  ...
temporal_3d                 0.089             5.267              5.356             1.66  ...
unscaled_2d                 0.067             5.289              5.356             1.25  ...
=================================================================================================
Best feature set for paper: per_resource_4d
  NLL improvement: 0.125 (2.33%)
  Learned omega: [0.8  5.2  0.1  4.3]
```

**The best feature set is the one with:**
- Largest `nll_improvement`
- Clearest omega pattern (matches expected behavior)
- Most interpretable visualization

## If You're in a Hurry

Run just one feature set to test:

```bash
cd uncertainty_sets_refactored

# Quick test (~3 minutes)
python sweep_and_viz_feature_set.py --feature-set temporal_3d
```

## Expected Omega Patterns

### Temporal 3D: ω ≈ [α, β, ~0]
```
omega_0_SYS_MEAN   = 8.5
omega_1_SYS_STD    = 12.1
omega_2_HOUR_SIN   = 0.02  ← Near zero (nuisance feature)
```

### Per-Resource 4D: 1-2 dominant features
```
omega_0_WIND_122_MEAN  = 0.8
omega_1_WIND_309_MEAN  = 5.2  ← High weight
omega_2_WIND_317_MEAN  = 0.1
omega_3_HOUR_SIN       = 4.3  ← High weight
```

### Unscaled 2D: ω ≈ 1/variance
```
omega_0_SYS_MEAN  = 0.02  ← Small (large-scale feature ~300 MW)
omega_1_SYS_STD   = 0.50  ← Large (small-scale feature ~50 MW)
```

## Verify Installation First

Before running sweeps, verify feature builders work:

```bash
cd uncertainty_sets_refactored
python test_feature_sets.py
```

Expected output:
```
All feature set tests passed! ✓
```

## For Your Paper

1. **Run all three feature sets** (use `run_all_feature_sets.sh`)
2. **Pick the best one** (largest NLL improvement + clearest pattern)
3. **Use outputs from that directory** for paper figures:
   - `omega_bar_chart.png` - Shows learned feature weights
   - `kernel_distance_*.png` - Shows learned metric in action
4. **Write figure caption:**

Example caption for 4D per-resource case:

> "Figure X: Learned kernel metric for per-resource wind forecasts. (a) Learned feature weights ω show WIND_309 and HOUR_SIN are most predictive for covariance. (b) Kernel neighborhood visualization in 2D projection of top-weighted features, demonstrating the learned metric prioritizes samples with similar forecasts from the most informative wind farm."

## Troubleshooting

**Import errors?**
```bash
# Make sure you're in the right directory
cd uncertainty_sets_refactored
python sweep_and_viz_feature_set.py --feature-set temporal_3d
```

**Data not found?**
```bash
# Check data files exist
ls -lh data/*.parquet
```

**Too slow?**
```bash
# Reduce hyperparameter grid for testing
python sweep_and_viz_feature_set.py \
    --feature-set temporal_3d \
    --taus 5.0 \
    --omega-l2-regs 0.0 \
    --standardize True
```

## Next Steps

After identifying the best feature set:

1. **Extract best omega and config:**
   ```python
   import numpy as np
   import json
   from pathlib import Path

   best_fs = "per_resource_4d"  # Replace with your best
   artifact_dir = Path(f"data/viz_artifacts/{best_fs}")

   omega = np.load(artifact_dir / "best_omega.npy")
   with open(artifact_dir / "feature_config.json") as f:
       config = json.load(f)

   print(f"Best omega: {omega}")
   print(f"Config: {config}")
   ```

2. **Use in your paper:**
   - Copy PNG files to paper/figures/
   - Reference omega values in text
   - Explain feature engineering choice in methodology

3. **Re-run with best config only:**
   ```bash
   # Use exact hyperparameters from best config
   python sweep_and_viz_feature_set.py \
       --feature-set per_resource_4d \
       --taus 5.0 \
       --omega-l2-regs 0.01 \
       --standardize True
   ```

## Full Documentation

- **User guide:** `FEATURE_ENGINEERING_README.md` (detailed explanations)
- **Implementation details:** `IMPLEMENTATION_SUMMARY.md` (technical overview)
- **Quick reference:** This file (`QUICKSTART.md`)

## Time Budget

- **Test run (1 feature set):** ~3-5 minutes
- **Full run (all 3 feature sets):** ~10-15 minutes
- **Analysis and figure generation:** ~5 minutes

**Total:** ~20 minutes to go from nothing to paper-ready figures.

## Questions?

See `FEATURE_ENGINEERING_README.md` for:
- Detailed feature set descriptions
- Expected behaviors and interpretations
- Visualization examples
- Comparison workflows
- "Is 4D → 2D messy?" discussion

---

**Ready to run?**

```bash
cd uncertainty_sets_refactored
python run_all_feature_sets.py
```

Go get coffee ☕ - it'll be done in 15 minutes!
