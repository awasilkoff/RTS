# RTS — Adaptive Robust Unit Commitment with Linear Decision Rules

Research code for the NewRAMP paper: adaptive robust unit commitment (ARUC) and
day-ahead reliability unit commitment (DARUC) on the RTS-GMLC test system, with
ellipsoidal wind-uncertainty sets calibrated by conformal prediction.

Two halves:

1. **Unit commitment models** (repo root) — DAM, ARUC, DARUC solved with Gurobi.
2. **Uncertainty set calibration** (`uncertainty_sets_refactored/`) — learned
   covariance, conformal radius, paper figures.

`ruc/` is **not** part of this project. It is early work for a future
Notification-Gated RUC paper and shares infrastructure only for convenience.

Detailed documentation lives in `CLAUDE.md` (root) and
`uncertainty_sets_refactored/CLAUDE.md`. This file covers how to run things.

---

## Setup

Requires Python 3.11+, **Gurobi with a full licence** (developed against
13.0.2), pandas, numpy, pydantic, matplotlib, and torch for the covariance
learning step.

```
conda env create -f environment.yml
conda activate RTS
```

The environment is named `RTS` (see `environment.yml`); conda environment names
are case-sensitive on Linux.

**Gurobi licence:** the size-limited licence bundled with `pip install gurobipy`
is **not** sufficient. A 48 h robust solve on RTS-GMLC has tens of thousands of
variables and roughly 1,850 second-order cones, well past the restricted limits.
An academic named-user or WLS licence is needed.

**Non-interactive contexts** (scheduled tasks, SSH one-liners, detached
processes) do not get the conda hook, so bare `python` will not resolve. Use the
absolute interpreter path, e.g.

```
C:\Users\alexw\miniforge3\envs\RTS\python.exe run_comparison.py ...
```

---

## Primary entry point — `run_comparison.py`

Runs DARUC (two-step: DAM → robust reliability) and optionally ARUC (one-shot)
with identical parameters, then writes comparison figures and a text summary.

```
python run_comparison.py --rho 2.0 --start-month 7 --start-day 15
```

With time-varying uncertainty from a calibrated NPZ, network limits, and the
DAM+Reserve baseline:

```
python run_comparison.py --uncertainty-npz uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz --provider-start 2448 --enforce-lines --with-reserve
```

Most-used flags (full list via `--help`):

| Flag | Default | Effect |
|---|---|---|
| `--hours` | 48 | Horizon length |
| `--start-month` / `--start-day` | 7 / 15 | Simulation start |
| `--uncertainty-npz` | None | Time-varying (Sigma, rho); overrides `--rho` |
| `--enforce-lines` | off | Network limits (otherwise copperplate) |
| `--with-reserve` | on | Adds the DAM+spinning-reserve baseline |
| `--no-fix-wind-z` | off | Free Z — let the LDR choose the wind response |
| `--robust-ramp` | off | SOC-based robust ramp constraints |
| `--skip-aruc` | off | Skip the one-shot ARUC solve (usually the bulk of runtime) |
| `--line-monitor-threshold` | 0.95 | Per-hour line filtering by DAM loading |
| `--mip-gap` | 0.005 | MIP optimality gap |
| `--time-limit` | None | Gurobi time limit, seconds |
| `--out-dir` | auto | Override output directory |

`--skip-aruc` matters for runtime. In one measured 48 h free-Z run the ARUC
solve took 8,235 s of an 8,788 s total — 94% — while DARUC took 539 s. Skip it
unless you specifically need the one-shot benchmark.

---

## Other runners

| Script | Purpose |
|---|---|
| `run_sensitivity_suite.py` | Multi-scenario sweep; the paper's case matrix |
| `run_reserve_then_daruc.py` | DAM+Reserve → DARUC (reserve-based commitment) |
| `run_alpha_sweep.py` | Conformal alpha → cost/curtailment (~40–60 min) |
| `run_price_of_robustness.py` | Rho sweep (~30–40 min) |
| `run_rts_dam.py` / `run_rts_aruc.py` / `run_rts_daruc.py` | Standalone single models |
| `compare_aruc_vs_daruc.py` | Figures and summary; used by `run_comparison.py` |

Utilities in `utils/` and standalone plots in `one-off-plots/` insert the repo
root into `sys.path`, so they can be run directly from anywhere.

---

## The paper's four-case comparison

All four reported cases come from **two** suite scenarios:

| Case | Reported as | Source |
|---|---|---|
| 1 | DAM | `full_free_z/` |
| 2 | DAM w/Res | `reserve_then_daruc/` (first stage) |
| 3 | DAM + RUC | `full_free_z/` (DARUC) |
| 4 | DAM w/Res + RUC | `reserve_then_daruc/` (DARUC) |

Note that **DAM w/Res is taken from `reserve_then_daruc/`**, not from
`full_free_z/dam_reserve/`. Both solve the same problem, but Case 4 must be
compared against the first stage it actually amends, and the two solves differ
slightly within the MIP gap.

**The suite defaults are the canonical paper configuration**, so running both
scenarios needs no flags:

```
python run_sensitivity_suite.py
```

That is equivalent to `--start-month 7 --start-day 15 --hours 48
--day2-interval 2 --provider-start 2448 --line-monitor-threshold 0.9 --mip-gap
0.005` with the α=0.99 uncertainty NPZ, writing to `sensitivity_suite/freeZ/`.
The resolved configuration is printed at startup and recorded in `config.json`.

Then backfill worst-case line flows (reserve deployment vs LDR re-dispatch at
worst-case wind), which produces `worst_case_flow_analysis_*.csv` per case:

```
python utils/backfill_worst_case_flows.py --scan-dir sensitivity_suite/freeZ
```

A single scenario can be run alone with `--scenarios reserve_then_daruc`, and an
interrupted suite resumed with `--resume`.

Because the default output directory is a fixed path, a run with non-default
parameters would otherwise overwrite the canonical results silently. The suite
compares against the previous run's `config.json` and prints a loud warning if
they disagree. Use `--out-dir <path>` for variants, or `--out-dir auto` for a
name generated from rho, horizon and start date.

### Checking a run is trustworthy

Every run writes solve provenance into `summary.json`. Check it before using any
number:

```
python -c "import json;d=json.load(open('sensitivity_suite/<tag>/reserve_then_daruc/summary.json'));print(d['solve_diagnostics']);print(d['line_resolve_max_iter_hit'])"
```

- `converged` must be `true` for every solve. `status_name: TIME_LIMIT` means the
  solve was truncated and `mip_gap_achieved` is the gap you actually got, not the
  0.5% you asked for.
- `line_resolve_max_iter_hit: true` means the worst-case line-violation loop
  exhausted its iteration cap, so the solution may still violate worst-case line
  limits. Raise `max_iter` and re-run before reporting that case.

---

## Outputs

Generated at runtime, not all tracked: `comparison_outputs/`, `dam_outputs/`,
`aruc_outputs/`, `daruc_outputs/`, `alpha_sweep/`, `price_of_robustness/`,
`sensitivity_suite/`.

`alpha_sweep/` and `sensitivity_suite/` **are** deliberately tracked in git —
they hold results of long sweeps so collaborators need not re-run them. Do not
add them to `.gitignore`.

Each robust case directory contains commitment and dispatch CSVs, Z
coefficients, Sigma/rho arrays, line-flow analysis, and `summary.json`.
Reported metrics are **day-1 only**; day 2 is a look-ahead horizon that prevents
end-of-horizon effects and is not reported.

---

## Notes

- **Long runs.** The robust solves are MISOCPs and can take hours. Prefer
  launching them yourself rather than from an automated context, and capture the
  log.
- **Logging on Windows.** PowerShell's `Tee-Object` defaults to UTF-16LE, which
  makes logs awkward to grep. Use `-Encoding utf8`.
- **Data.** `RTS_Data/` is RTS-GMLC source data (see its own READMEs). The
  `uncertainty_sets_refactored/data/*_constellation_*.parquet` files are derived
  from SPP ensemble forecasts; confirm redistribution terms before sharing this
  repo outside the team.
