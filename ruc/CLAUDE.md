# CLAUDE.md — ruc (Paper 3)

This subdirectory implements a **new Robust Unit Commitment (RUC) formulation** for Paper 3, building on the ARUC/DARUC infrastructure from Paper 2.

## Policies (inherited from root CLAUDE.md)

**Commit Policy:** After completing a major change, automatically commit and push with a descriptive message. Do not wait for the user to ask.

**Execution Policy:** For any script that will take >1 minute to run: create the script, tell the user to run it manually, suggest the command as a single-line one-liner, and wait for pasted results. Do NOT execute long-running tasks directly.

**Formatting:** Always provide shell commands as single-line one-liners (no backslash continuations).

## Shared Infrastructure (from parent repo)

This formulation reuses:
- `models.py` — `DAMData` Pydantic model (canonical data container)
- `io_rts.py` — RTS-GMLC ETL pipeline
- `dam_model.py` — Deterministic DAM model (baseline)
- `network_ptdf.py` — DC power flow PTDF computation
- `RTS_Data/` — RTS-GMLC test case data
- `uncertainty_sets_refactored/` — Learned covariance, conformal prediction, uncertainty set generation

## Directory Structure

```
ruc/
├── CLAUDE.md          # This file
├── ruc_model.py       # New RUC formulation (Gurobi model builder)
├── run_ruc.py         # End-to-end RUC pipeline
└── ...                # Additional modules as needed
```

## Status

Project setup complete. Formulation development in progress.
