# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **Adaptive Robust Unit Commitment (ARUC)** with Linear Decision Rules for the RTS-GMLC power system test case. It uses Gurobi for optimization and supports both deterministic day-ahead market (DAM) and robust unit commitment under wind uncertainty.

## Key Commands

```bash
# Run deterministic day-ahead unit commitment
python run_rts_dam.py

# Run adaptive robust unit commitment with LDR
python run_rts_aruc.py

# Run uncertainty set calibration pipeline
python uncertainty_sets_refactored/main.py

# Run individual module smoke tests (most modules have __main__ blocks)
python models.py
python dam_model.py
python network_ptdf.py
```

**Requirements:** Python 3.7+, Gurobi (with valid license), pandas, numpy, pydantic

## Architecture

### Data Flow Pipeline

```
RTS_Data/SourceData/*.csv  →  io_rts.py  →  DAMData (Pydantic model)
                                              ↓
                              dam_model.py (deterministic) or
                              aruc_model.py (robust with Sigma, rho)
                                              ↓
                              Gurobi optimization  →  results CSVs
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `models.py` | `DAMData` Pydantic class - canonical data container for UC models |
| `io_rts.py` | ETL: RTS-GMLC CSV files → `DAMData` object |
| `network_ptdf.py` | DC power flow PTDF matrix computation |
| `dam_model.py` | Deterministic DAM UC model builder (Gurobi) |
| `aruc_model.py` | Adaptive robust UC with linear decision rules |
| `run_rts_dam.py` | End-to-end deterministic DAM pipeline |
| `run_rts_aruc.py` | End-to-end robust ARUC pipeline |
| `run_rts_daruc.py` | Two-step DARUC: deterministic DAM → robust reliability commitments |

### Uncertainty Set Analysis (`uncertainty_sets_refactored/`)

| Module | Purpose |
|--------|---------|
| `main.py` | Integration pipeline: covariance fitting → conformal prediction → rho calibration |
| `covariance_optimization.py` | Kernel-based covariance fitting (omega, tau, ridge) |
| `conformal_prediction.py` | Binned and weighted conformal prediction for wind lower bounds |
| `data_processing.py` | Build X/Y matrices for covariance and conformal models |
| `data_processing_extended.py` | Extended feature builders for omega learning experiments |
| `sweep_and_viz_feature_set.py` | Hyperparameter sweep + visualization for feature sets |
| `run_all_feature_sets.py` | Run all feature set experiments and generate comparison |

#### Conformal Prediction Methods

The system supports **two conformal prediction approaches**:

**1. Binned Conformal (Original):**
- Fixed discrete bins for local adaptation
- Bin by predictions, actuals, or features
- Step-function q_hat variation
- Smaller bundle size (~1 KB)
- File: `conformal_prediction.py:train_wind_lower_model_conformal_binned()`

**2. Weighted Conformal (New):**
- Kernel-weighted continuous adaptation
- Uses learned omega from covariance optimization
- Smooth q_hat variation across query points
- Localized, query-dependent corrections
- File: `conformal_prediction.py:train_wind_lower_model_weighted_conformal()`

**Quick Start:**
```bash
cd uncertainty_sets_refactored

# Run unit tests
python test_weighted_conformal.py

# Simple example (synthetic data)
python example_weighted_conformal.py

# Compare weighted vs binned (RTS data)
python compare_weighted_vs_binned.py

# Find optimal tau (bandwidth parameter)
python sweep_weighted_conformal_tau.py
```

**Documentation:**
- `WEIGHTED_CONFORMAL_README.md` - Comprehensive guide
- `WEIGHTED_CONFORMAL_IMPLEMENTATION_SUMMARY.md` - Implementation details

**When to use weighted conformal:**
- Have learned omega from covariance optimization
- Want smooth, continuous adaptation (vs discrete bins)
- Sufficient calibration data (n_cal > 100)
- Prefer query-dependent localized corrections

#### Feature Engineering for Learned Omega

The system supports **5 feature sets** to demonstrate learned omega benefits:

**2D Feature Sets:**
- `focused_2d`: [SYS_MEAN, SYS_STD] - Clean baseline for testing normalization impact
- `unscaled_2d`: [SYS_MEAN_MW, SYS_STD_MW] - Raw units, omega learns automatic rescaling

**Higher-Dimensional Feature Sets:**
- `temporal_3d`: [SYS_MEAN, SYS_STD, HOUR_SIN] - Omega downweights nuisance features
- `per_resource_4d`: [WIND_122, WIND_309, WIND_317, HOUR_SIN] - Omega learns per-farm importance
- `high_dim_8d`: [SYS_MEAN, SYS_STD, + 3 farms × (MEAN, STD)] - Explore feature prioritization

**Normalization Options:**
- `none`: No scaling (raw features)
- `standard`: StandardScaler (mean=0, std=1)
- `minmax`: MinMaxScaler ([0,1] range) - Shows clearest omega benefit

**Omega Constraint Options:**
- `none`: No constraint on omega scale. L2 regularization pulls toward 1.0.
- `softmax`: Learn unconstrained α, ω = softmax(α). Ensures sum(ω)=1, ω≥0. L2 reg ignored.
- `simplex`: Project ω onto probability simplex at each step. Ensures sum(ω)=1, ω≥0. L2 reg ignored.
- `normalize`: Learn ω directly, divide by sum(ω) at end. Post-hoc normalization. L2 reg ignored.

When using constraints (softmax/simplex/normalize), omega represents **relative feature importance** (sums to 1), cleanly separating feature weighting from kernel bandwidth (tau).

**Quick Start:**
```bash
cd uncertainty_sets_refactored

# Quick experiments (~30 min, 16 configs)
python run_minmax_experiments.py

# Thorough overnight grid search (~3-5 hours, 105 configs per feature set)
python run_thorough_overnight.py

# Run with softmax constraint (decouples scale from tau)
python run_thorough_overnight.py --omega-constraint softmax

# Or run specific feature set
python sweep_and_viz_feature_set.py \
    --feature-set focused_2d \
    --scaler-types standard minmax \
    --taus 2.0 5.0 \
    --omega-l2-regs 0.0 1e-2

# With omega constraint (L2 reg ignored when constraint is not 'none')
python sweep_and_viz_feature_set.py \
    --feature-set focused_2d \
    --omega-constraint softmax \
    --scaler-types standard minmax \
    --taus 0.5 1.0 2.0 5.0 10.0
```

**Outputs:** `data/viz_artifacts/<feature_set>/`
- `sweep_results.csv` - Hyperparameter sweep results (compare scalers, tau, regularization)
- `best_omega.npy` - Learned feature weights
- `omega_bar_chart.png` - Visualization of learned ω per feature
- `kernel_distance_*.png` - Learned metric visualization
- `feature_config.json` - Best configuration metadata

**Important Methodology Notes:**
- **Baseline comparisons:** Three levels of sophistication (see `GLOBAL_COVARIANCE_BASELINE.md`)
  1. **Global covariance:** No adaptation (uses all training data, k=N)
  2. **k-NN equal weights:** Local adaptation with ω=[1,1,...]
  3. **k-NN learned omega:** Local adaptation + feature learning
- **Baseline comparison philosophy:** Always uses equal weights `[1, 1, ...]` for k-NN baseline (see `BASELINE_COMPARISON_FIX.md`)
  - Standardized features: Natural Euclidean baseline
  - Raw features: Learned omega discovers rescaling from scratch (more impressive!)
- **Train/test split:** Random holdout (75/25, seed=42) for representative evaluation (see `TRAIN_TEST_SPLIT_CHANGE.md`)
  - More stable than temporal holdout for covariance estimation
  - Tests generalization across all time periods, not just end-of-series

**Documentation:**
- `QUICKSTART.md` - Fast start guide (~20 min to paper figures)
- `THOROUGH_GRID_SEARCH.md` - Overnight comprehensive sweep (3-5 hours)
- `FEATURE_ENGINEERING_README.md` - Detailed feature set descriptions
- `MINMAX_NORMALIZATION_IMPLEMENTATION.md` - MinMax scaler implementation details
- `GLOBAL_COVARIANCE_BASELINE.md` - Three-level baseline comparison (global/k-NN/learned)
- `FOUR_BASELINE_COMPARISON.md` - Four-baseline comparison (Learned/Kernel(ω=1)/Euclidean k-NN/Global)
- `BASELINE_COMPARISON_FIX.md` - Why baseline is always `[1,1,...]`
- `TRAIN_TEST_SPLIT_CHANGE.md` - Random vs temporal holdout rationale

### SPP Forecast Integration

The DAM/ARUC/DARUC models can replace the default `DAY_AHEAD_wind.csv` forecasts with scaled SPP ensemble mean forecasts. This is controlled by two parameters on `build_damdata_from_rts()` and each runner script:

- `spp_forecasts_parquet`: Path to v2 forecasts parquet (default: `uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet`). Set to `None` to use original `DAY_AHEAD_wind.csv`.
- `spp_start_idx`: Positional index into the SPP time series for time alignment (default: 0).

**V2 parquet pipeline** (`mapping.py`):
1. **SUBMODEL filter** — keeps only the 51 SUBMODEL#00–#50 ensemble members, drops 4 MODEL# variants
2. **Capacity scaling** — multiplies ACTUAL and FORECAST by `RTS_Pmax / max(SPP_actuals)` per resource

Scale factors (v2 mapping CSV):

| RTS Generator | SPP Site | Scale Factor | RTS Pmax |
|---|---|---|---|
| 309_WIND_1 | OKGE.GRANTPLNS.WIND.1 | 1.42 | 148.3 MW |
| 317_WIND_1 | OKGE.SDWE.SDWE | 4.51 | 799.1 MW |
| 303_WIND_1 | OKGE.SWCV.CV | 4.19 | 847.0 MW |
| 122_WIND_1 | MIDW.KACY.ALEX | 29.14 | 713.5 MW |

**Regenerating v2 parquets:**
```bash
cd uncertainty_sets_refactored && python mapping.py
```
This writes v2 files alongside existing v1 files — no existing scripts are affected.

**Wind block capacity fix:** Wind generators in gen.csv have `Output_pct_1/2/3 = 0` (no heat rate curve), which previously caused `block_cap = [0,0,0]` and forced zero dispatch. `io_rts.py` now sets `block_cap[i,0] = Pmax[i]` for wind generators with zero marginal cost, allowing dispatch up to the time-varying `Pmax_2d` forecast.

### Data Location

- **Static data:** `RTS_Data/SourceData/` (bus.csv, gen.csv, branch.csv)
- **Time series:** `RTS_Data/timeseries_data_files/` (Load/, WIND/, PV/, HYDRO/)
- **SPP forecasts (v2):** `uncertainty_sets_refactored/data/*_v2.parquet` (scaled, SUBMODEL-filtered)
- **Outputs:** `dam_outputs/`, `aruc_outputs/`, `daruc_outputs/` (generated at runtime)

## Key Concepts

### DAMData Structure

The `DAMData` Pydantic model (in `models.py`) is the canonical interface between ETL and optimization:

- **Indices:** `gen_ids`, `bus_ids`, `line_ids`, `time`
- **Generator params:** `Pmin`, `Pmax` (can be I×T for wind), `RU`, `RD`, `MUT`, `MDT`, costs
- **Network:** `PTDF` (L×N matrix), `Fmax` (line limits)
- **Load:** `d` (N×T nodal demand array)
- **Initial conditions:** `u_init`, `init_up_time`, `init_down_time`

Generator types: `"THERMAL"`, `"WIND"`, `"SOLAR"`, `"HYDRO"` (stored in `gen_type` list)

### ARUC Model Variables

- `u[i,t]`: Binary commitment status
- `v[i,t]`, `w[i,t]`: Startup/shutdown indicators
- `p0[i,t]`: Nominal dispatch (first stage)
- `Z[i,t,k]`: LDR coefficients - dispatch adjusts as `p(r) = p0 + Z @ r`
- Uncertainty: ellipsoidal set `{r : r^T Sigma r <= rho^2}`

### Configuration

Edit constants at the top of runner scripts:
- `START_TIME`: Simulation start (e.g., `pd.Timestamp(year=2020, month=1, day=1)`)
- `HORIZON_HOURS`: Optimization horizon (default: 48)
- `UNCERTAINTY_RHO`: Ellipsoid radius for robust model
- `M_PENALTY`: Power balance slack penalty (big-M)

### Expected Omega Patterns (Feature Engineering)

When running feature set experiments, expect these patterns:

**focused_2d** (MinMax vs Standard comparison):
- MinMax should show clearer NLL improvement than Standard
- Omega values differentiate between SYS_MEAN and SYS_STD
- Eliminates scale dominance → cleaner feature importance learning

**high_dim_8d** (Feature prioritization):
- Omega reveals which features are most predictive for covariance
- Possible patterns: system-level dominance, specific farm importance, or mean vs std tradeoffs
- Bar chart shows learned feature importance hierarchy

**temporal_3d** (Nuisance downweighting):
- Expect ω ≈ [α, β, ~0] with HOUR_SIN downweighted
- Demonstrates omega suppresses irrelevant temporal features

**per_resource_4d** (Differential importance):
- 1-2 farms dominate the metric
- Shows omega discovers which wind farms are most informative

**unscaled_2d** (Automatic rescaling):
- Learned ω ≈ 1/variance (similar to standardization)
- Demonstrates omega handles scale differences automatically

## Debugging

Use `debugging.py` for systematic infeasibility diagnosis:
1. Check deterministic feasibility first
2. Apply progressive constraint relaxation
3. Extract IIS (Irreducible Inconsistent Subsystem) from Gurobi

Models can be saved/loaded via `utils.save_full_state()` and `utils.load_full_state()`.

## IEEE Paper Figures

Generate publication-ready conformal prediction figures:

```bash
cd uncertainty_sets_refactored
python run_paper_figures.py
```

**Outputs** (saved to `data/viz_artifacts/paper_figures/`):
1. `fig_timeseries_conformal.png` - Timeseries overlay (method demo)
2. `fig_calibration_curve.pdf` - Calibration validation curve
3. `fig_adaptive_correction.pdf` - 2-panel adaptive correction summary

**Key Features**:
- IEEE two-column format optimized
- 300 DPI for final submission
- Both PDF (for paper) and PNG (for preview)
- Wilson score confidence intervals on calibration curve
- Color-coded tolerance bands

**Runtime**: ~2-3 minutes (trains models at 5 alpha values)

See `CONFORMAL_PREDICTION_README.md` section "IEEE Paper Figures" for LaTeX integration and customization.

## Commit Policy

After completing a major change (new feature, bug fix, multi-file refactor), **automatically commit and push** with a descriptive message. Do not wait for the user to ask — commit and push proactively at natural completion points. This keeps the history clean, prevents losing work across sessions, and maintains remote backups.

## Execution Policy
For any script that will take >1 minute to run:
1. Create the script
2. Tell me to run it manually
3. Suggest the command as a one line terminal command and where to save output
4. Wait for me to paste results

Do NOT execute long-running tasks directly.

**Note:** Feature engineering experiments (`run_minmax_experiments.py`, `run_all_feature_sets.py`) take ~30 minutes with default parameters. Paper figure generation (`run_paper_figures.py`) takes ~2-3 minutes. These should be run manually by the user.

## Formatting Preferences

- **Command lines:** Always provide shell commands as single-line one-liners (no backslash continuations or multi-line formatting).

## Analysis Workflow

After running feature set experiments:

1. **Check comparison summary:**
   ```bash
   cat uncertainty_sets_refactored/data/viz_artifacts/feature_set_comparison.csv
   ```

2. **Identify best feature set:**
   - Largest `nll_improvement`
   - Clearest omega pattern (matches expected behavior)
   - Most interpretable visualization

3. **Extract best config:**
   ```python
   import numpy as np
   import json
   omega = np.load("uncertainty_sets_refactored/data/viz_artifacts/<best_feature_set>/best_omega.npy")
   config = json.load(open("uncertainty_sets_refactored/data/viz_artifacts/<best_feature_set>/feature_config.json"))
   ```

4. **Use for paper:**
   - Copy omega_bar_chart.png and kernel_distance_*.png to figures/
   - Reference omega values and NLL improvement in text
   - Explain feature engineering choice in methodology
```