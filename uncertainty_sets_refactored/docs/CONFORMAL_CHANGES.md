# Conformal Prediction: Fixes and Documentation

## Date
2026-02-05

## Summary
Fixed path inconsistencies in conformal prediction scripts and created comprehensive documentation.

---

## Changes Made

### 1. Fixed `viz_conformal_sweep.py`

**Issues Fixed:**
- ✅ Hardcoded absolute path → Changed to relative path using `Path(__file__).parent`
- ✅ Dataset version mismatch → Changed from `rts4_constellation` to `rts3_constellation` for consistency

**Before:**
```python
DATA_DIR = Path("/Users/alexwasilkoff/PycharmProjects/RTS/uncertainty_sets/data/")
actuals_parquet=DATA_DIR / "actuals_filtered_rts4_constellation_v1.parquet"
forecasts_parquet=DATA_DIR / "forecasts_filtered_rts4_constellation_v1.parquet"
```

**After:**
```python
DATA_DIR = Path(__file__).parent / "data"
actuals_parquet=DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet"
forecasts_parquet=DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet"
```

**Impact:** Script now works consistently across different environments and matches the data used in `main.py`.

---

### 2. Created Documentation: `CONFORMAL_PREDICTION_README.md`

**Contents:**
- **Overview**: What conformal prediction is and why we use it
- **Key Concepts**: Quantile regression, conformal correction, binned adaptation
- **Architecture**: Component breakdown and data flow diagrams
- **Usage Guide**: Complete code examples for training and prediction
- **Integration**: How conformal prediction fits into the covariance pipeline
- **Hyperparameter Tuning**: Guide for `alpha_target`, `quantile_alpha`, binning strategies
- **Technical Details**: Mathematical foundations, coverage guarantees
- **Troubleshooting**: Common issues and solutions

**Location:** `uncertainty_sets_refactored/CONFORMAL_PREDICTION_README.md`

---

### 3. Created Test Suite: `test_conformal_prediction.py`

**Test Coverage:**
1. ✓ Import checks
2. ✓ Data loading and validation
3. ✓ Model training (small sample)
4. ✓ Prediction on new data
5. ✓ Coverage property verification (80%, 90%, 95% targets)

**Usage:**
```bash
python test_conformal_prediction.py
```

**Results:** All tests passing ✓
- Coverage achieved within 5% of target for all test cases
- Predictions generate valid bounds (conf ≤ base)
- Model handles edge cases (small samples, various alpha targets)

**Location:** `uncertainty_sets_refactored/test_conformal_prediction.py`

---

## Verification Status

| Component | Status | Notes |
|-----------|--------|-------|
| `conformal_prediction.py` | ✅ Ready | Core module working correctly |
| `viz_conformal_sweep.py` | ✅ Fixed | Path issues resolved |
| `main.py` integration | ✅ Verified | Full pipeline tested |
| `data_processing.py` | ✅ Working | `build_conformal_totals_df()` validated |
| Documentation | ✅ Complete | Comprehensive README created |
| Test suite | ✅ Passing | All 5 tests pass |

---

## Quick Start

### Run Tests
```bash
cd uncertainty_sets_refactored
python test_conformal_prediction.py
```

### Run Conformal Sweep
```bash
python viz_conformal_sweep.py
```

### Run Full Integration Pipeline
```bash
python main.py
```

### Read Documentation
```bash
cat CONFORMAL_PREDICTION_README.md
# Or open in your favorite markdown viewer
```

---

## File Locations

```
uncertainty_sets_refactored/
├── conformal_prediction.py              # Core module (unchanged)
├── viz_conformal_sweep.py               # Fixed: paths updated
├── test_conformal_prediction.py         # NEW: test suite
├── CONFORMAL_PREDICTION_README.md       # NEW: documentation
└── CONFORMAL_CHANGES.md                 # NEW: this file
```

---

## Data Consistency

All scripts now use consistent dataset names:
- ✅ `actuals_filtered_rts3_constellation_v1.parquet`
- ✅ `forecasts_filtered_rts3_constellation_v1.parquet`
- ✅ `residuals_filtered_rts3_constellation_v1.parquet`

**Note:** `rts4_constellation` files also exist if needed for different experiments.

---

## Next Steps

1. **For Development:**
   - Run `python test_conformal_prediction.py` before making changes
   - Update documentation if adding new features
   - Maintain consistency with `main.py` integration

2. **For Research:**
   - Review `CONFORMAL_PREDICTION_README.md` for theory and intuition
   - Experiment with different `alpha_target` values (trade-off: coverage vs tightness)
   - Try different binning strategies (`y_pred` vs `feature:ens_std`)

3. **For Production:**
   - Monitor operational coverage (should match `alpha_target` ± 5%)
   - Retrain periodically if distribution shifts
   - Consider ensemble of conformal models for robustness

---

## References

**Documentation:**
- `CONFORMAL_PREDICTION_README.md` - Complete technical guide
- `CLAUDE.md` - Project overview (see "Uncertainty Set Analysis" section)

**Code:**
- `conformal_prediction.py` - Core implementation
- `main.py` - Integration example (Steps 2-3)
- `test_conformal_prediction.py` - Usage examples

**Related Papers:**
- Romano et al. (2019): Conformalized Quantile Regression
- Lei & Wasserman (2014): Distribution-Free Prediction Bands
- Vovk et al. (2005): Algorithmic Learning in a Random World
