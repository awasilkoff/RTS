# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **Adaptive Robust Unit Commitment (ARUC)** with Linear Decision Rules for the RTS-GMLC power system test case. It uses Gurobi for optimization and supports both deterministic day-ahead market (DAM) and robust unit commitment under wind uncertainty.

The project has two main parts:
1. **Unit Commitment models** (root directory) — DAM, ARUC, DARUC formulations solved via Gurobi
2. **Uncertainty Set Calibration** (`uncertainty_sets_refactored/`) — Learned covariance estimation, conformal prediction, paper figures. See `uncertainty_sets_refactored/CLAUDE.md` for detailed documentation.

## Key Commands

```bash
# Deterministic day-ahead unit commitment
python run_rts_dam.py

# Adaptive robust unit commitment with LDR
python run_rts_aruc.py

# Two-step DARUC: DAM → robust reliability commitments
python run_rts_daruc.py

# ARUC vs DARUC comparison (single scenario)
python run_comparison.py --hours 6 --start-month 7 --start-day 15 --rho 2.0

# Price of robustness sweep (rho values, long-running)
python run_price_of_robustness.py --hours 12 --start-month 7 --start-day 15

# Alpha sweep: conformal alpha → uncertainty sets → DARUC/ARUC (long-running)
python run_alpha_sweep.py --hours 12 --start-month 7 --start-day 15

# Generate all IEEE paper figures (~3-5 min)
cd uncertainty_sets_refactored && python generate_paper_figures.py

# Pre-compute uncertainty sets for ARUC
cd uncertainty_sets_refactored && python generate_uncertainty_sets.py
```

**Requirements:** Python 3.7+, Gurobi (with valid license), pandas, numpy, pydantic, torch (for omega learning)

## Architecture

### Data Flow Pipeline

```
RTS_Data/SourceData/*.csv  →  io_rts.py  →  DAMData (Pydantic model)
                                              ↓
                              dam_model.py (deterministic) or
                              aruc_model.py (robust with Sigma, rho)
                                              ↓
                              Gurobi optimization  →  results
```

For robust models, time-varying (Sigma, rho) come from the uncertainty set pipeline:
```
SPP wind data (parquet)  →  covariance_optimization.py (learn omega, predict Sigma)
                          →  conformal_prediction.py (calibrate rho via alpha)
                          →  NPZ file (mu, Sigma, rho per hour)
                          →  aruc_model.py / run_rts_aruc.py
```

### Root-Level Modules

| Module | Purpose |
|--------|---------|
| `models.py` | `DAMData` Pydantic class — canonical data container for UC models |
| `io_rts.py` | ETL: RTS-GMLC CSV files → `DAMData` object |
| `network_ptdf.py` | DC power flow PTDF matrix computation |
| `dam_model.py` | Deterministic DAM UC model builder (Gurobi) |
| `aruc_model.py` | Adaptive robust UC with linear decision rules |
| `run_rts_dam.py` | End-to-end deterministic DAM pipeline |
| `run_rts_aruc.py` | End-to-end robust ARUC pipeline |
| `run_rts_daruc.py` | Two-step DARUC: deterministic DAM → robust reliability commitments |
| `run_comparison.py` | DARUC vs ARUC comparison orchestrator |
| `compare_aruc_vs_daruc.py` | Comparison figures: commitment heatmaps, dispatch bars, curtailment |
| `run_price_of_robustness.py` | Rho sweep: cost/curtailment vs uncertainty budget |
| `run_alpha_sweep.py` | Alpha sweep: conformal alpha → NPZ → DARUC/ARUC cost/curtailment |
| `debugging.py` | Infeasibility diagnosis (IIS extraction, progressive relaxation) |

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

## Runner Scripts

### ARUC vs DARUC Comparison

```bash
python run_comparison.py --hours 6 --start-month 7 --start-day 15 --rho 2.0
```

Outputs (in `comparison_outputs/<run_tag>/`): commitment heatmaps, dispatch bars, Z coefficient heatmaps, wind curtailment figures, and text summary.

### Price of Robustness Sweep (rho)

```bash
python run_price_of_robustness.py --hours 12 --start-month 7 --start-day 15
```

Outputs (in `price_of_robustness/`): `sweep_results.csv`, cost vs rho figure, curtailment vs rho figure.

### Alpha Sweep (Conformal Alpha → Cost)

`run_alpha_sweep.py` sweeps conformal alpha values end-to-end: pre-computes covariance (once), generates per-alpha NPZ uncertainty sets, then runs DARUC + ARUC for each.

```bash
python run_alpha_sweep.py --hours 12 --start-month 7 --start-day 15
```

Output dir is auto-named from params (e.g. `alpha_sweep/lines_rlf0.25_12h_m07d15_a0.80_0.90_0.95_0.99/`), overridable with `--out-dir`.

Outputs: `sweep_results.csv`, cost vs alpha figure, curtailment vs alpha figure.

### IEEE Paper Figures

```bash
cd uncertainty_sets_refactored && python generate_paper_figures.py
```

Outputs in `data/viz_artifacts/paper_final/figures/` and `tables/`. Runtime: ~3-5 minutes. Requires pre-run sweep data for focused_2d, high_dim_16d, and knn_k_sweep.

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
- `Z[i,t,k]`: LDR coefficients — dispatch adjusts as `p(r) = p0 + Z @ r`
- Uncertainty: ellipsoidal set `{r : r^T Sigma^{-1} r <= rho^2}`

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

After completing a major change (new feature, bug fix, multi-file refactor), **automatically commit and push** with a descriptive message. Do not wait for the user to ask — commit and push proactively at natural completion points. This keeps the history clean, prevents losing work across sessions, and maintains remote backups.

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