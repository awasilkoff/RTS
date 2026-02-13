# CLAUDE.md — uncertainty_sets_refactored

This subdirectory implements **kernel-weighted covariance estimation** and **conformal prediction** for wind forecast uncertainty sets, used by the parent ARUC optimization models.

## Policies (inherited from root CLAUDE.md)

**Commit Policy:** After completing a major change, automatically commit and push with a descriptive message. Do not wait for the user to ask.

**Execution Policy:** For any script that will take >1 minute to run: create the script, tell the user to run it manually, suggest the command as a single-line one-liner, and wait for pasted results. Do NOT execute long-running tasks directly.

**Formatting:** Always provide shell commands as single-line one-liners (no backslash continuations).

## Core Modules (do not rewrite — extend only)

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `covariance_optimization.py` | Kernel-weighted local covariance estimation with learned omega | `fit_omega`, `predict_mu_sigma_topk_cross`, `predict_mu_sigma_knn`, `KernelCovConfig`, `FitConfig`, `CovPredictConfig` |
| `conformal_prediction.py` | Binned and weighted conformal prediction for wind lower bounds | `train_wind_lower_model_conformal_binned`, `train_wind_lower_model_weighted_conformal`, `ConformalLowerBundle`, `WeightedConformalLowerBundle` |
| `generate_paper_figures.py` | All IEEE paper figures and LaTeX tables | `generate_all_figures`, `fig1_*` through `fig11_*`, `fig_nll_heatmap`, `fig_nll_delta_surface`, `table_*` |
| `data_processing.py` | ETL: parquet files to X/Y matrices (system-level features) | `build_XY_for_covariance_system_only`, `build_XY_for_covariance` |
| `data_processing_extended.py` | Extended feature builders (2D through 16D) | `FEATURE_BUILDERS`, `FEATURE_SET_DESCRIPTIONS`, `build_XY_*` |
| `utils.py` | Scalers, conformal helpers | `fit_scaler`, `apply_scaler`, `StandardScaler`, `MinMaxScaler` |
| `plot_config.py` | IEEE plot styling, colors, constants | `setup_plotting`, `IEEE_COL_WIDTH`, `IEEE_TWO_COL_WIDTH`, `COLORS` |
| `generate_uncertainty_sets.py` | Batch pre-computation of time-varying (mu, Sigma, rho) for ARUC | `UncertaintySetConfig`, `pre_compute_covariance`, `generate_uncertainty_sets` |
| `mapping.py` | SPP-to-RTS resource mapping and scaling | `cache_matched_rts_named_data`, `compute_scale_factors` |

## Key Function Locations

### covariance_optimization.py
```
KernelCovConfig                    :23   — tau, ridge, enforce_nonneg_omega
FitConfig                          :35   — max_iters, step_size, omega_l2_reg, omega_constraint, k_fit
CovPredictConfig                   :65   — tau, ridge, dtype, device
fit_omega                          :179  — Train omega via gradient descent (leave-one-out on training set)
predict_mu_sigma_topk_cross        :515  — Predict (mu, Sigma) using learned omega + top-k neighbors
predict_mu_sigma_knn               :649  — Euclidean k-NN baseline (no learned weights)
implied_rho_from_total_lower_bound :726  — Convert conformal lower bound to ellipsoid rho
```

### conformal_prediction.py
```
train_wind_lower_model_conformal_binned     :883  — Train binned conformal model
train_wind_lower_model_weighted_conformal   :351  — Train weighted conformal model (uses learned omega)
compute_binned_adaptive_conformal_corrections_lower :737
compute_weighted_conformal_correction_lower :136
ConformalLowerBundle                        :627  — Serializable binned conformal model
WeightedConformalLowerBundle                :226  — Serializable weighted conformal model
```

### generate_paper_figures.py
```
_load_omega             :97   — Load best_omega.npy
_load_feature_config    :105  — Load feature_config.json
_per_point_gaussian_nll :2313 — Per-point Gaussian NLL (copied from sweep code)
_save_figure            :2646 — Save PDF + PNG
fig_nll_heatmap         :2334 — Two-panel NLL scatter (k-NN vs learned)
fig_nll_delta_surface   :2460 — Smoothed ΔNLL surface with zero-contour
generate_all_figures    :2688 — Main orchestrator
```

### data_processing_extended.py
```
FEATURE_BUILDERS          :511  — Dict mapping feature set name → builder function
build_XY_focused_2d       :227  — [SYS_MEAN, SYS_STD]
build_XY_high_dim_16d     :364  — [8D base + temporal + ramp + spread]
build_XY_high_dim_8d      :268  — [SYS + per-farm MEAN/STD]
build_XY_temporal_nuisance_3d :27 — [SYS_MEAN, SYS_STD, HOUR_SIN]
build_XY_per_resource_4d  :101  — [WIND_122, WIND_309, WIND_317, HOUR_SIN]
build_XY_unscaled_2d      :183  — [SYS_MEAN_MW, SYS_STD_MW] raw units
```

### generate_uncertainty_sets.py
```
UncertaintySetConfig              :66   — Dataclass: all config (tau, k, alpha, omega_path, etc.)
pre_compute_covariance            :160  — Fit omega + predict (mu, Sigma) for all hours
generate_uncertainty_sets_for_alpha :317 — Generate sets at a single alpha (conformal + rho)
generate_uncertainty_sets         :536  — Full pipeline: covariance + conformal at multiple alphas
main                              :591  — CLI entry point
```

### sweep_and_viz_feature_set.py
```
run_sweep                    :83   — Main hyperparameter sweep (tau, l2_reg, constraint)
run_multi_seed_validation    :533  — Multi-seed omega re-run at each tau (--n-seeds)
_per_point_gaussian_nll      :55   — Per-point NLL (canonical copy)
_mean_gaussian_nll           :78   — Mean NLL wrapper
```

### sweep_knn_k_values.py
```
sweep_k_values              :363  — Sweep k for Euclidean k-NN
compute_learned_omega_baseline :330 — Compute learned omega predictions
ellipsoid_surface_3d        :84   — 3D ellipsoid mesh for visualization
run_knn_k_sweep             :913  — Full k-NN sweep pipeline
run_multi_split_k_sweep     :1159 — Multi-seed k sweep for error bars
```

## Data Layout

```
data/
├── *_rts3_constellation_v1.parquet   — RTS-3 wind data (3 farms: 122, 309, 317)
├── *_rts4_constellation_v2.parquet   — RTS-4 wind data (scaled, SUBMODEL-filtered)
├── rts_to_spp_mapping_*.csv          — SPP-to-RTS resource mapping
└── viz_artifacts/
    ├── focused_2d/                   — 2D sweep results (softmax constraint)
    │   ├── best_omega.npy
    │   ├── feature_config.json
    │   ├── sweep_results.csv
    │   ├── multi_seed_results.csv    — Per (tau, seed) results (from --n-seeds)
    │   └── multi_seed_stats.csv      — Per-tau aggregated stats (mean/std/min/max)
    ├── high_dim_16d/                 — 16D sweep results (unconstrained)
    │   ├── best_omega.npy            — Trained at tau=0.1, constraint=none, l2_reg=0.0
    │   ├── feature_config.json
    │   ├── sweep_results.csv
    │   ├── multi_seed_results.csv    — Per (tau, seed) results (from --n-seeds)
    │   └── multi_seed_stats.csv      — Per-tau aggregated stats (mean/std/min/max)
    ├── knn_k_sweep/                  — k-NN baseline sweep
    ├── paper_final/                  — Generated paper figures and tables
    │   ├── figures/
    │   └── tables/
    └── paper_figures/                — Conformal calibration metadata
```

## Current Best Omega (high_dim_16d)

Trained with: `tau=0.1, omega_constraint=none, l2_reg=0.0, k_fit=None, step_size=0.1, max_iters=250`

Top-weighted features (unconstrained, sum=1.48):
| Feature | ω |
|---------|------|
| WIND_309_MEAN | 0.362 |
| WIND_122_STD | 0.313 |
| DOW_COS | 0.222 |
| WIND_122_MEAN | 0.183 |
| WIND_309_STD | 0.130 |
| SYS_MEAN, SYS_STD, etc. | ~0.01 (floor) |

Test NLL: 10.94 (learned) vs 11.88 (k-NN k=64) vs 13.25 (global)

## Standard Data Split

All experiments use the same random 50/25/25 split:
```python
rng = np.random.RandomState(42)
indices = rng.permutation(n)
n_train = int(0.5 * n)
n_val = int(0.25 * n)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]
```

## Sweep Training Parameters

The sweep in `sweep_and_viz_feature_set.py:run_sweep` uses:
```python
FitConfig(max_iters=250, step_size=0.1, grad_clip=10.0, tol=1e-7,
          omega_l2_reg=..., omega_constraint=..., k_fit=None)
```
**Important:** `k_fit=None` during training (softmax over all N-1 neighbors). Prediction uses `k=128` top-k. This train/predict mismatch is intentional and beneficial.

## One-Off / Legacy Scripts (do not modify unless asked)

These scripts were created for specific experiments and are NOT part of the active pipeline. Do not modify, refactor, or extend them:

| Script | Purpose |
|--------|---------|
| `compare_covariance_methods.py` | Early comparison (global/k-NN/learned) |
| `compare_nn_vs_gbm.py` | NN vs GBM experiment |
| `compare_weighted_vs_binned.py` | Weighted vs binned conformal comparison |
| `create_comprehensive_comparison.py` | Four-baseline comprehensive comparison |
| `diagnose_kernel_equal_outliers.py` | One-off kernel diagnosis |
| `diagnose_tau_omega_seeds.py` | Tau/omega seed sensitivity (superseded by `sweep_and_viz_feature_set.py --n-seeds`) |
| `example_weighted_conformal.py` | Tutorial example |
| `experiment_absolute_vs_scaled_scores.py` | Absolute vs scaled score experiment |
| `experiment_weighting_schemes.py` | Weighting scheme comparison |
| `generate_multiple_kernel_comparisons.py` | Multi-kernel comparison figures |
| `improve_conformal_coverage.py` | Coverage improvement experiment |
| `quick_test_absolute_scores.py` | Quick diagnostic |
| `run_minmax_experiments.py` | MinMax normalization experiments |
| `run_paper_figures.py` | Old paper figure generation (superseded by `generate_paper_figures.py`) |
| `run_paper_figures_dayahead_valid.py` | Day-ahead validation figures |
| `run_paper_figures_improved.py` | Improved paper figures (superseded) |
| `run_thorough_overnight.py` | Overnight grid search launcher |
| `run_all_feature_sets.py` | All feature set experiments |
| `sweep_conformal_config.py` | Conformal config sweep |
| `sweep_conformal_quick.py` | Quick conformal sweep |
| `sweep_cov_fit.py` | Covariance fitting sweep |
| `sweep_regularization.py` | Regularization sweep |
| `sweep_time_exponential_params.py` | Time-exponential parameter sweep |
| `sweep_weighted_conformal_tau.py` | Weighted conformal tau sweep |
| `test_*.py` | Various one-off test scripts |
| `validate_*.py`, `verify_*.py` | Validation scripts |
| `viz_*.py` (except `viz_kernel_distance.py`) | Legacy visualization scripts |

## Active Scripts (the ones that matter)

1. **`generate_paper_figures.py`** — All IEEE paper figures. Run with: `cd uncertainty_sets_refactored && python generate_paper_figures.py`
2. **`sweep_and_viz_feature_set.py`** — Hyperparameter sweep for a feature set. Use `--n-seeds N` for multi-seed validation at best config. Long-running (~30 min, longer with `--n-seeds`).
3. **`sweep_knn_k_values.py`** — k-NN baseline k sweep. Long-running.
4. **`conformal_prediction.py`** — Conformal prediction library (imported, not run directly).
5. **`covariance_optimization.py`** — Covariance estimation library (imported, not run directly).
6. **`mapping.py`** — SPP data mapping. Run with: `cd uncertainty_sets_refactored && python mapping.py`
7. **`generate_uncertainty_sets.py`** — Batch pre-computation of (mu, Sigma, rho) for ARUC. Run with: `cd uncertainty_sets_refactored && python generate_uncertainty_sets.py`
8. **`main.py`** — Original integration pipeline.

## Generating Paper Figures

```bash
cd uncertainty_sets_refactored && python generate_paper_figures.py
```

Output: `data/viz_artifacts/paper_final/figures/` and `tables/`

Runtime: ~3-5 minutes (loads saved sweep data, generates conformal figures fresh).

Prerequisites: must have run `sweep_and_viz_feature_set.py --n-seeds 5` for focused_2d and high_dim_16d (multi-seed data enables error bars in fig4), and `sweep_knn_k_values.py --multi-split`.
