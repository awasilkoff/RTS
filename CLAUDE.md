# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **Adaptive Robust Unit Commitment (ARUC)** with Linear Decision Rules for the RTS-GMLC power system test case. It uses Gurobi for optimization and supports both deterministic day-ahead market (DAM) and robust unit commitment under wind uncertainty.

The project has two main parts:
1. **Unit Commitment models** (root directory) -- DAM, ARUC, DARUC formulations solved via Gurobi
2. **Uncertainty Set Calibration** (`uncertainty_sets_refactored/`) -- Learned covariance estimation, conformal prediction, paper figures. See `uncertainty_sets_refactored/CLAUDE.md` for detailed documentation.

> **`ruc/` is NOT part of this project.** It contains early-stage work for a future Paper 3 formulation (Notification-Gated RUC). It shares data and infrastructure with this repo for convenience but will not be shipped with the Paper 2 release. See `ruc/CLAUDE.md` for details.

## Directory Layout

```
RTS/
├── CLAUDE.md
├── environment.yml / requirements.txt
│
├── [Core library — always in root, imported by runners]
│   ├── models.py                  DAMData Pydantic model
│   ├── io_rts.py                  ETL: RTS-GMLC CSVs -> DAMData
│   ├── dam_model.py               Deterministic DAM Gurobi model
│   ├── aruc_model.py              ARUC with LDR (Gurobi MISOCP)
│   ├── aruc_warm_start.py         Warm-start initialization for ARUC
│   ├── network_ptdf.py            DC power flow PTDF computation
│   ├── compute_branch_flows.py    Branch flow helpers (imported by runners)
│   ├── uncertainty_set_provider.py  NPZ uncertainty set loader
│   ├── runner_utils.py            Shared functions across runner scripts
│   ├── utils.py                   Save/load model state helpers
│   ├── debugging.py               IIS extraction and infeasibility tools
│   └── plot_line_flows.py         Line flow visualization
│
├── [Runner scripts — entry points, run directly]
│   ├── run_comparison.py          PRIMARY: DARUC + ARUC comparison
│   ├── run_alpha_sweep.py         Conformal alpha -> cost/curtailment sweep
│   ├── run_price_of_robustness.py Rho sweep
│   ├── run_sensitivity_suite.py   Multi-scenario sensitivity runner
│   ├── run_reserve_then_daruc.py  DAM+Reserve -> DARUC pipeline
│   ├── run_reserve_vs_aruc.py     Reserve vs ARUC comparison
│   ├── run_rts_dam.py             Standalone DAM
│   ├── run_rts_aruc.py            Standalone ARUC
│   ├── run_rts_daruc.py           Standalone DARUC
│   └── compare_aruc_vs_daruc.py   Comparison figures and summary (used by run_comparison.py)
│
├── uncertainty_sets_refactored/   Uncertainty pipeline (own CLAUDE.md)
│   ├── [core libs, runners, viz support — see CLAUDE.md]
│   ├── docs/                      Markdown documentation
│   └── archive/                   One-off experiments and legacy scripts
│
├── utils/                         Infrequently-used utility scripts
│   │                              (add root to sys.path when run directly)
│   ├── backfill_reserves.py       Backfill reserve_equivalent.csv for past runs
│   ├── backfill_worst_case_flows.py  Backfill worst-case line flows for past runs
│   ├── count_committed_units.py   Count committed units across output dirs
│   ├── rebuild_deviation_summaries.py  Rebuild deviation_summary.csv for past runs
│   ├── summarize_scenario.py      Print scenario summary table (load, wind, uncertainty)
│   └── verify_dam_wind_forecasts.py   Visualize wind Pmax to confirm SPP forecast loading
│
├── one-off-plots/                 Standalone plot scripts (not part of pipeline)
│   │                              (add root to sys.path when run directly)
│   └── [CostComparisonChart, ellipsoid_*, ldr_*, motivation_forecasts, etc.]
│
├── docs/                          Project documentation
│   ├── formulations.tex/.pdf      Standalone ARUC/DARUC formulations writeup
│   └── [archived markdown notes]
│
├── archive/                       Superseded scripts and one-off artifacts
│   └── [old test scripts, debug logs, one-off output files]
│
├── ruc/                           Paper 3 experimental (NOT shipping with Paper 2)
│   └── [see ruc/CLAUDE.md]
│
└── RTS_Data/                      Static RTS-GMLC test case data (read-only)
```

**Running utils/ and one-off-plots/ scripts:** These insert the project root into `sys.path` at startup, so they resolve imports correctly when run directly from anywhere: `python utils/summarize_scenario.py --start-month 7 --start-day 15`.

## Key Commands

**Primary script — `run_comparison.py`** is the main entry point for all ARUC/DARUC analysis:

```bash
# Basic comparison (uses --fast defaults: 48h, day2-interval=2, day1-only-robust, etc.)
python run_comparison.py --rho 2.0 --start-month 7 --start-day 15

# With time-varying uncertainty from NPZ
python run_comparison.py --uncertainty-npz path/to/uncertainty.npz --start-month 7 --start-day 15

# With line limits and DAM+Reserve baseline
python run_comparison.py --enforce-lines --with-reserve --uncertainty-npz path/to/uncertainty.npz

# Copperplate, custom horizon
python run_comparison.py --hours 24 --rho 3.0 --start-month 1 --start-day 5
```

Other scripts (standalone or sweep):

```bash
# Standalone DAM / ARUC / DARUC (less commonly used)
python run_rts_dam.py
python run_rts_aruc.py
python run_rts_daruc.py

# Alpha sweep: conformal alpha -> uncertainty sets -> DARUC/ARUC (long-running)
python run_alpha_sweep.py --hours 12 --start-month 7 --start-day 15

# Price of robustness sweep (rho values, long-running)
python run_price_of_robustness.py --hours 12 --start-month 7 --start-day 15

# Generate all IEEE paper figures (~3-5 min)
cd uncertainty_sets_refactored && python generate_paper_figures.py

# Pre-compute uncertainty sets for ARUC
cd uncertainty_sets_refactored && python generate_uncertainty_sets.py
```

**Requirements:** Python 3.7+, Gurobi (with valid license), pandas, numpy, pydantic, torch (for omega learning)

## Architecture

### Data Flow Pipeline

```
RTS_Data/SourceData/*.csv  ->  io_rts.py  ->  DAMData (Pydantic model)
                                              v
                              dam_model.py (deterministic) or
                              aruc_model.py (robust with Sigma, rho)
                                              v
                              Gurobi optimization  ->  results
```

For robust models, time-varying (Sigma, rho) come from the uncertainty set pipeline:
```
SPP wind data (parquet)  ->  covariance_optimization.py (learn omega, predict Sigma)
                          ->  conformal_prediction.py (calibrate rho via alpha)
                          ->  NPZ file (mu, Sigma, rho per hour)
                          ->  aruc_model.py / run_rts_aruc.py
```

### Root-Level Modules

**Core library (imported by runners):**

| Module | Purpose |
|--------|---------|
| `models.py` | `DAMData` Pydantic class — canonical data container for UC models |
| `io_rts.py` | ETL: RTS-GMLC CSV files -> `DAMData` object |
| `network_ptdf.py` | DC power flow PTDF matrix computation |
| `dam_model.py` | Deterministic DAM UC model builder (Gurobi) |
| `aruc_model.py` | Adaptive robust UC with linear decision rules (MISOCP) |
| `aruc_warm_start.py` | Warm-start: initialize ARUC binaries/dispatch/Z from DAM solution |
| `compute_branch_flows.py` | Branch flow helpers: `filter_monitored_lines`, `iterative_line_resolve` |
| `uncertainty_set_provider.py` | Load time-varying (mu, Sigma, rho) from NPZ files |
| `runner_utils.py` | Shared functions across runner scripts (reserve, deviation summary, etc.) |
| `utils.py` | Save/load full model state helpers |
| `debugging.py` | Infeasibility diagnosis (IIS extraction, progressive relaxation) |
| `plot_line_flows.py` | Line flow visualization against thermal limits |

**Runner scripts (run directly):**

| Script | Purpose |
|--------|---------|
| `run_comparison.py` | **Primary**: DARUC + ARUC, generates figures + summary |
| `compare_aruc_vs_daruc.py` | Comparison figures and `write_summary` (used by `run_comparison.py`) |
| `run_rts_dam.py` | Standalone deterministic DAM pipeline |
| `run_rts_aruc.py` | Standalone robust ARUC pipeline |
| `run_rts_daruc.py` | Standalone two-step DARUC pipeline |
| `run_reserve_then_daruc.py` | DAM+Reserve -> DARUC pipeline |
| `run_reserve_vs_aruc.py` | DAM+Reserve vs ARUC comparison |
| `run_sensitivity_suite.py` | Multi-scenario sensitivity sweep across configurations |
| `run_price_of_robustness.py` | Rho sweep: cost/curtailment vs uncertainty budget |
| `run_alpha_sweep.py` | Conformal alpha -> NPZ -> DARUC/ARUC cost/curtailment sweep |

### Uncertainty Set Pipeline (`uncertainty_sets_refactored/`)

See `uncertainty_sets_refactored/CLAUDE.md` for full documentation including function locations, data layout, and sweep parameters.

Key modules: `covariance_optimization.py`, `conformal_prediction.py`, `generate_paper_figures.py`, `generate_uncertainty_sets.py`, `data_processing.py`, `data_processing_extended.py`.

### SPP Forecast Integration

The DAM/ARUC/DARUC models can replace the default `DAY_AHEAD_wind.csv` forecasts with scaled SPP ensemble mean forecasts. Controlled by two parameters on `build_damdata_from_rts()` and each runner script:

- `spp_forecasts_parquet`: Path to v2 forecasts parquet (default: `uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet`). Set to `None` to use original `DAY_AHEAD_wind.csv`.
- `spp_start_idx`: Positional index into the SPP time series for time alignment (default: 0).

Scale factors (v2 mapping):

| RTS Generator | SPP Site | Scale Factor | RTS Pmax |
|---|---|---|---|
| 309_WIND_1 | OKGE.GRANTPLNS.WIND.1 | 1.42 | 148.3 MW |
| 317_WIND_1 | OKGE.SDWE.SDWE | 4.51 | 799.1 MW |
| 303_WIND_1 | OKGE.SWCV.CV | 4.19 | 847.0 MW |
| 122_WIND_1 | MIDW.KACY.ALEX | 29.14 | 713.5 MW |

Regenerate v2 parquets: `cd uncertainty_sets_refactored && python mapping.py`

**Wind block capacity fix:** Wind generators in gen.csv have `Output_pct_1/2/3 = 0` (no heat rate curve), which previously caused `block_cap = [0,0,0]` and forced zero dispatch. `io_rts.py` now sets `block_cap[i,0] = Pmax[i]` for wind generators with zero marginal cost.

**Ramp rate units fix:** gen.csv stores `Ramp Rate MW/Min` but the ramp constraints in `dam_model.py` and `aruc_model.py` apply `RU * dt_ramp` where `dt_ramp` is in hours. `io_rts.py` now multiplies by 60 at read time (MW/min -> MW/h). Without this, ramp rates were 60x too tight, forcing thermals to stay elevated overnight and causing ~54% wind curtailment in DAM.

## Runner Scripts

### `run_comparison.py` — Primary Script

The main entry point for all single-scenario analysis. Runs DARUC (two-step: DAM -> robust reliability) and ARUC-LDR (one-shot robust) with identical parameters, then generates comparison figures and a text summary.

```bash
python run_comparison.py --rho 2.0 --start-month 7 --start-day 15
python run_comparison.py --uncertainty-npz path/to/uncertainty.npz --enforce-lines --with-reserve
```

**Output directory:** `comparison_outputs/<auto-tag>/` containing:
- `aruc/` and `daruc/` subdirs with commitment CSVs, dispatch CSVs, Z coefficients, Sigma/rho arrays, summary JSON
- `dam_reserve/` (when `--with-reserve`) with DAM+Reserve results
- Figures: commitment heatmaps, dispatch bars, Z coefficient heatmaps, wind curtailment, Pmin vs dispatch, worst-case wind
- `comparison_summary.txt` — full text report with day-1 cost breakdown, curtailment, commitment diffs

**CLI reference:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--hours` | 48 | Horizon hours |
| `--rho` | 3.0 | Ellipsoid radius (ignored if `--uncertainty-npz`) |
| `--start-month` / `--start-day` / `--start-hour` | 7 / 15 / 0 | Simulation start |
| `--enforce-lines` | **on** | Line flow limits (`--no-enforce-lines` for copperplate) |
| `--uncertainty-npz` | **`.../uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz`** | Time-varying uncertainty NPZ; overrides `--rho`, so `--rho` is ignored unless this is cleared |
| `--provider-start` | 2448 | Start index into NPZ time series |
| `--rho-lines-frac` | None | Fraction of rho for line constraints (e.g. 0.25) |
| `--mip-gap` | 0.005 | MIP optimality gap (0.5%) |
| `--day2-interval` | 2 | Day-2 period duration in hours (2 = 2h blocks) |
| `--day1-only-robust` | True | Only day 1 gets SOC/Z constraints |
| `--fix-wind-z` | off (True with `--fast`) | Fix wind Z diagonal to 1 |
| `--three-blocks` | off | Use 3-block piecewise cost (default: single block) |
| `--no-worst-case-cost` | off | Disable worst-case cost epigraph |
| `--incremental-obj` / `--no-incremental-obj` | True | DARUC: charge only incremental commitment costs |
| `--dispatch-cost-scale` | 0.01 | DARUC dispatch cost scale (with incremental obj) |
| `--robust-ramp` | **on** | SOC-based robust ramp constraints |
| `--with-reserve` | **on** | DAM+Reserve baseline. Declared `store_true` with `default=True`, so there is no way to turn it off |
| `--ramp-scale` | 1.0 | Multiply all ramp rates by this factor |
| `--pmin-scale` | 1.0 | Multiply all Pmin by this factor |
| `--line-monitor-threshold` | **0.95** | Per-hour line filtering threshold (the paper suite uses 0.9) |
| `--include-renewables` / `--include-nuclear` / `--include-zero-marginal` | off | Generator filtering (see section) |
| `--fast` / `--no-fast` | **False** | Performance defaults bundle (see Solver Tuning) |
| `--time-limit` | None (60000 with `--fast`) | Gurobi time limit (seconds) |
| `--threads` | None | Gurobi thread count |
| `--bar-qcp-conv-tol` | None (1e-4 with `--fast`) | Barrier QCP convergence tolerance |
| `--out-dir` | auto | Override output directory |

### Price of Robustness Sweep (rho)

```bash
python run_price_of_robustness.py --hours 12 --start-month 7 --start-day 15
```

Outputs (in `price_of_robustness/`): `sweep_results.csv`, cost vs rho figure, curtailment vs rho figure.

### Alpha Sweep (Conformal Alpha -> Cost)

`run_alpha_sweep.py` sweeps conformal alpha values end-to-end: pre-computes covariance (once), generates per-alpha NPZ uncertainty sets, then runs DARUC + ARUC for each.

```bash
python run_alpha_sweep.py --hours 12 --start-month 7 --start-day 15
python run_alpha_sweep.py --hours 48 --day2-interval 2 --day1-only-robust --start-month 7 --start-day 15
```

Output dir is auto-named from params (e.g. `alpha_sweep/lines_rlf0.25_12h_m07d15_a0.80_0.90_0.95_0.99/`), overridable with `--out-dir`.

CLI args for variable intervals: `--day2-interval 2`, `--day1-only-robust`.

**Resume support:** If interrupted, re-run with the same args -- completed alphas are read from `sweep_results.csv` and skipped. Results are flushed to CSV after every alpha point.

**Per-alpha artifacts:** After each solve, full outputs are saved to `alpha_X.XXXX/{dam,daruc,aruc}/` subdirectories (commitment, dispatch, Z coefficients, Sigma, rho, deviation summary, Z analysis, summary JSON).

Outputs: `sweep_results.csv`, per-alpha artifact dirs, cost vs alpha figure, curtailment vs alpha figure.

### IEEE Paper Figures

```bash
cd uncertainty_sets_refactored && python generate_paper_figures.py
```

Outputs in `data/viz_artifacts/paper_final/figures/` and `tables/`. Runtime: ~3-5 minutes. Requires pre-run sweep data for focused_2d, high_dim_16d, and knn_k_sweep.

## Key Concepts

### Day-1 Metric Reporting

All metric reporting (costs, curtailment, unit-hours) uses **day-1 periods only**. Day 2 is treated as a look-ahead horizon that prevents end-of-horizon effects but is not reported.

- **`data.day1_times()`**: Returns time labels for the first 24 hours (uses cumulative period durations for variable-interval support)
- **`data.day1_period_mask()`**: Boolean mask `(T,)` selecting day-1 periods
- **Affected scripts**: `run_alpha_sweep.py`, `run_price_of_robustness.py`, `run_comparison.py`, `run_rts_aruc.py`, `run_rts_daruc.py`
- **CSV columns**: `dam_cost_total`, `daruc_cost_total`, `aruc_cost_total` are day-1 cost breakdowns (no separate `*_obj` columns for full-horizon Gurobi objectives)
- **Figures**: Visualizations (commitment heatmaps, dispatch bars, Z heatmaps) still show the full horizon for context

### DAMData Structure

The `DAMData` Pydantic model (in `models.py`) is the canonical interface between ETL and optimization:

- **Indices:** `gen_ids`, `bus_ids`, `line_ids`, `time`
- **Generator params:** `Pmin`, `Pmax` (can be IxT for wind), `RU`, `RD`, `MUT`, `MDT`, costs
- **Network:** `PTDF` (LxN matrix), `Fmax` (line limits)
- **Load:** `d` (NxT nodal demand array)
- **Initial conditions:** `u_init`, `init_up_time`, `init_down_time`

Generator types: `"THERMAL"`, `"WIND"`, `"SOLAR"`, `"HYDRO"` (stored in `gen_type` list)

### Generator Filtering (Solar/Hydro/Nuclear Exclusion)

By default, solar (PV, RTPV), hydro (HYDRO, ROR), and nuclear generators are **excluded** from the model so that wind is the only low-marginal-cost resource. This makes wind uncertainty the dominant driver of commitment decisions.

Controlled by three parameters on `build_damdata_from_rts()` and all runner scripts:

| Parameter | Default | Controls |
|-----------|---------|----------|
| `include_renewables` | `False` | PV, RTPV, HYDRO, ROR generators |
| `include_nuclear` | `False` | NUCLEAR generators |
| `include_zero_marginal` | `None` | Override: when explicitly True/False, overrides both above |

CLI flags (available on `run_comparison.py`, `run_alpha_sweep.py`, `run_price_of_robustness.py`):
- `--include-renewables` / `--no-include-renewables`
- `--include-nuclear` / `--no-include-nuclear`
- `--include-zero-marginal` / `--no-include-zero-marginal` (overrides both)

With defaults (~77 generators): 39 CT + 23 STEAM + 10 CC + 1 CSP + 4 WIND. With `--include-zero-marginal` (~154 generators): adds 56 solar + 20 hydro + 1 nuclear.

Filtering happens at the earliest point in `io_rts.py` (extending the existing `keep_mask` that excludes STORAGE and SYNC_COND), so all downstream code works unchanged.

### Variable-Duration Periods

When `day2_interval_hours > 1`, the 48-hour horizon uses hourly periods for day 1 (24 periods) and multi-hour blocks for day 2. For example, `day2_interval_hours=2` gives T=36 periods: 24x1h + 12x2h.

- **`period_duration`** (on `DAMData`): Optional `(T,)` array of period durations in hours. `None` = all 1.0 (backward compatible).
- **`data.dt`** property: Returns period durations (always available, defaults to ones).
- **`data.total_hours`** property: Sum of all period durations.
- **Load/Pmax**: Averaged across hours in each block.
- **Objective**: No-load and energy costs scale by `dt[t]`; startup/shutdown are one-time events.
- **Ramp**: Scales by transition time `(dt[t-1] + dt[t]) / 2`.
- **MUT/MDT**: Look-forward, counting hours not periods (a 2-hour block = 2 hours toward MUT).

### Day-1-Only Robustness

When `day1_only_robust=True`, the ARUC/DARUC model only creates Z variables and SOC constraints for the first 24 periods. Day 2 uses nominal (DAM-like) constraints. This significantly reduces solve time by eliminating all SOC constraints for day 2.

- **`robust_mask`** (on `build_aruc_ldr_model`): Optional `(T,)` bool array. `True` = robust period. `None` = all robust (backward compatible).

### Robust Ramp Constraints

When `robust_ramp=True` (`--robust-ramp` CLI flag), ramp constraints account for worst-case dispatch deviations under uncertainty:

- **Ramp up:** `p0[i,t] + rho_t * ||Z_{i,t} L_t|| - p0[i,t-1] + rho_{t-1} * ||Z_{i,t-1} L_{t-1}|| <= RU_i * dt_ramp * (u[i,t-1] + v[i,t])`
- **Ramp down:** `p0[i,t-1] + rho_{t-1} * ||Z_{i,t-1} L_{t-1}|| - p0[i,t] + rho_t * ||Z_{i,t} L_t|| <= RD_i * dt_ramp * (u[i,t] + w[i,t])`

The norm terms reuse the existing `z_gen[i,t]` SOC variables (no new SOC constraints needed). Only applied when both periods `t` and `t-1` are robust (per `robust_mask`); otherwise falls back to nominal ramp constraints. Default is `False` (nominal ramps only) for backward compatibility.

### DAM + Spinning Reserve Baseline

The `--with-reserve` flag on `run_comparison.py` re-solves the DAM with a system-level spinning reserve constraint derived from the same uncertainty set used by ARUC/DARUC. This provides a "what if we just added reserves?" baseline to quantify the value of adaptive robust optimization (LDR) vs a naive reserve policy.

**Reserve requirement:** `R[t] = rho[t] * sqrt(1^T Sigma[t] 1)` -- worst-case total wind shortfall over the ellipsoid.

**Spinning reserve constraints** (thermal generators only):
```
r[i,t] <= Pmax[i,t] * u[i,t] - p[i,t]    (capacity headroom)
r[i,t] <= RU[i] * dt[t]                    (ramp capability)
sum_{i in THERMAL} r[i,t] >= R[t]          (system requirement)
```

Works with both `--uncertainty-npz` (time-varying) and scalar `--rho` (static covariance). Results saved to `dam_reserve/` subdirectory and included as a fourth series in all comparison figures and summary.

**Reserve ramp cap vs inter-period ramp:** The reserve constraint uses `r[i,t] <= RU[i] * dt[t]` (full period duration), not `dt_ramp = (dt[t-1]+dt[t])/2` (transition time). This is intentional: spinning reserve measures how much a unit can ramp **within** the current period, so the full `dt[t]` is the relevant time window. Inter-period ramp constraints use transition time `dt_ramp` because they span the boundary between two periods.

Expected cost ordering: DAM < DAM+Reserve < DARUC/ARUC.

### ARUC Model Variables

- `u[i,t]`: Binary commitment status
- `v[i,t]`, `w[i,t]`: Startup/shutdown indicators
- `p0[i,t]`: Nominal dispatch (first stage)
- `Z[i,t,k]`: LDR coefficients -- dispatch adjusts as `p(r) = p0 + Z @ r`
- Uncertainty: ellipsoidal set `{r : r^T Sigma^{-1} r <= rho^2}`

### Solver Performance Tuning

The `--fast` flag is **OFF by default** on `run_comparison.py` (`default=False`); pass `--fast` to enable it. Note that several of the defaults it would set are already the standalone defaults (`day1_only_robust=True`), while others are not (`fix_wind_z`, `bar_qcp_conv_tol`, `time_limit`). It applies these performance defaults:

| Default | Value | Effect |
|---------|-------|--------|
| `fix_wind_z` | `True` | Eliminates wind availability SOC constraints |
| `day1_only_robust` | `True` | Only day 1 gets SOC/Z constraints (48h horizon) |
| `bar_qcp_conv_tol` | `1e-4` | Relaxed barrier convergence (default 1e-8) |
| `time_limit` | `600s` | Safety backstop to prevent runaway solves |
| `line_monitor_threshold` | `0.5` | Per-hour line filtering at >=50% DAM loading (when `--enforce-lines`) |

Individual overrides still work (e.g. `--fast --time-limit 300`).

**Per-hour line filtering:** When `--line-monitor-threshold` is set, lines are filtered per `(line, period)` pair based on DAM dispatch loading (`|flow_l(t)| / Fmax_l >= threshold`). Lines never above threshold at any hour are dropped entirely (reducing PTDF row count). For remaining lines, only `(l, t)` pairs where loading exceeds threshold get flow constraints and SOC cones in ARUC. This can reduce line SOC constraints by 40-60% compared to a static (horizon-wide) filter. DAM always enforces all lines; the per-hour mask applies only to ARUC/DARUC.

Additional Gurobi params always applied in `aruc_model.py`:
- `Presolve=2`, `PreSparsify=1` — aggressive presolve helps SOC-heavy models
- `NodefileStart=4.0` — GB of B&B tree held in RAM before spilling to disk. Was 2.0 (documented here as 0.5, which was wrong). Kept modest on purpose: **no runner in this repo ever passes `node_file_start`**, so the value applies unchanged on the 32 GB laptop, where a large in-memory tree risks an OOM kill. Raise via `--node-file-start` on large-memory hosts.
- `Heuristics=0.05` — was `0.2` (chasing incumbents). On the 48 h reserve_then_daruc case the incumbent is found early and never improves while the bound crawls, so the gap is entirely bound-side.
- `MIPFocus` fallback `1` → `3`. **Narrower than it looks:** `run_comparison.py` already defaults `--mip-focus` to 3 and passes it explicitly, so this fallback is unreachable from it *and from the whole sensitivity suite*, which shells out to it. It governs only callers passing `mip_focus=None` — `run_reserve_then_daruc.py` and `ruc/run_ruc_monolithic.py`.
- The measured 8 h A/B (−14% solve time, −29% nodes, identical solution) ran through `run_comparison.py`, where MIPFocus was **already** 3 — so that gain is attributable to `Heuristics` and `NodefileStart` only. The MIPFocus change itself is unvalidated on the path it actually affects.

Warm start (`aruc_warm_start.py`) initializes binary u/v/w from DAM, p0 from DAM dispatch, and Z with constraint-consistent values. Z satisfies `sum_i Z[i,t,k] = 0` (power balance response): wind diagonal = identity, thermals distribute `-1` proportional to committed Pmax. SOC auxiliaries (z_gen) are also hinted. This avoids the previous infeasibility where Z[thermal]=0 + Z[wind]=identity violated power balance.

### Configuration

Edit constants at the top of runner scripts:
- `START_TIME`: Simulation start (e.g., `pd.Timestamp(year=2020, month=1, day=1)`)
- `HORIZON_HOURS`: Optimization horizon (default: 48)
- `UNCERTAINTY_RHO`: Ellipsoid radius for robust model
- `M_PENALTY`: Power balance slack penalty (big-M)

### Data Location

- **Static data:** `RTS_Data/SourceData/` (bus.csv, gen.csv, branch.csv)
- **Time series:** `RTS_Data/timeseries_data_files/` (Load/, WIND/, PV/, HYDRO/)
- **SPP forecasts (v2):** `uncertainty_sets_refactored/data/*_v2.parquet` (scaled, SUBMODEL-filtered)
- **Uncertainty set data:** `uncertainty_sets_refactored/data/viz_artifacts/` (sweep results, learned omega, paper figures)
- **Outputs:** `dam_outputs/`, `aruc_outputs/`, `daruc_outputs/`, `comparison_outputs/`, `price_of_robustness/`, `alpha_sweep/` (generated at runtime)

## Debugging

Use `debugging.py` for systematic infeasibility diagnosis:
1. Check deterministic feasibility first
2. Apply progressive constraint relaxation
3. Extract IIS (Irreducible Inconsistent Subsystem) from Gurobi

Models can be saved/loaded via `utils.save_full_state()` and `utils.load_full_state()`.

## Commit Policy

After completing a major change (new feature, bug fix, multi-file refactor), **automatically commit and push** with a descriptive message. Do not wait for the user to ask -- commit and push proactively at natural completion points. This keeps the history clean, prevents losing work across sessions, and maintains remote backups.

## Execution Policy

For any script that will take >1 minute to run:
1. Create the script
2. Tell me to run it manually
3. Suggest the command as a one line terminal command and where to save output
4. Wait for me to paste results

Do NOT execute long-running tasks directly.

**Long-running scripts:** `run_alpha_sweep.py` (~40-60 min), `run_price_of_robustness.py` (~30-40 min), `sweep_and_viz_feature_set.py` (~30 min), `generate_paper_figures.py` (~3-5 min). These should be run manually by the user.

## Formatting Preferences

- **Command lines:** Always provide shell commands as single-line one-liners (no backslash continuations or multi-line formatting).