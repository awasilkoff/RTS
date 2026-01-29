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

### Uncertainty Set Analysis (`uncertainty_sets_refactored/`)

| Module | Purpose |
|--------|---------|
| `main.py` | Integration pipeline: covariance fitting → conformal prediction → rho calibration |
| `covariance_optimization.py` | Kernel-based covariance fitting (omega, tau, ridge) |
| `conformal_prediction.py` | Binned conformal prediction for wind lower bounds |
| `data_processing.py` | Build X/Y matrices for covariance and conformal models |

### Data Location

- **Static data:** `RTS_Data/SourceData/` (bus.csv, gen.csv, branch.csv)
- **Time series:** `RTS_Data/timeseries_data_files/` (Load/, WIND/, PV/, HYDRO/)
- **Outputs:** `dam_outputs/`, `aruc_outputs/` (generated at runtime)

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

## Debugging

Use `debugging.py` for systematic infeasibility diagnosis:
1. Check deterministic feasibility first
2. Apply progressive constraint relaxation
3. Extract IIS (Irreducible Inconsistent Subsystem) from Gurobi

Models can be saved/loaded via `utils.save_full_state()` and `utils.load_full_state()`.
