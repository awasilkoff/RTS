# NewRAMP — freeZ Rerun Plan

Plan for the clean **freeZ** sensitivity rerun and the case comparison behind the NewRAMP
paper. Fixed-Z is dropped (established as the wrong approach); **all cases use free Z**
(`--no-fix-wind-z`).

## Conceptual comparison — three cases

All three share one theme: **how much extra commitment does full robustness require, and from
what starting point?**

| # | Case | First-stage commitment | Robustness added by | Increment reported |
|---|------|------------------------|---------------------|--------------------|
| 1 | **DARUC from DAM** | Deterministic DAM (nominal, no reserve) | LDR + worst-case lines (2nd step) | DARUC − DAM |
| 2 | **DARUC from Reserve** | DAM + spinning reserve (capacity-robust) | LDR + worst-case lines (2nd step) | DARUC − Reserve |
| 3 | **ARUC free-Z** | *co-optimized with robustness* | LDR + worst-case lines (one-shot) | benchmark (optimum) |

**Story:**
- **Case 1 → 3 gap** = value of co-optimizing commitment with robustness vs robustifying a naive
  deterministic plan after the fact.
- **Case 2 → 3 gap** = how much of that value reserves capture cheaply; the residual increment is
  the true cost of adaptivity.
- **Case 2 transmission problems** = reserve provisions aggregate capacity only
  (`R[t] = ρ·√(1ᵀΣ1)`, a scalar that ignores the network), so feeding it through DARUC's
  worst-case line constraints forces extra commitments purely for deliverability.

## Scenario wiring (`run_sensitivity_suite.py`)

| Scenario | Script | Extra flags | Produces |
|----------|--------|-------------|----------|
| `full_free_z` | `run_comparison.py` | `--enforce-lines --robust-ramp --with-reserve --no-fix-wind-z` | DAM, DAM+Reserve, **DARUC-from-DAM** (Case 1), **ARUC free-Z** (Case 3) |
| `reserve_then_daruc` | `run_reserve_then_daruc.py` | `--no-fix-wind-z` | **DARUC-from-Reserve** (Case 2) |

Notes:
- `run_comparison.py` `--incremental-obj` defaults **True**, so DARUC-from-DAM reports the
  *incremental* commitment over the deterministic DAM automatically.
- `run_reserve_then_daruc.py` **hardcodes** `enforce_lines=True`, `robust_ramp=True`,
  `incremental_obj=True`; only `--no-fix-wind-z` needs passing for Case 2.

## Shared run parameters (match prior freeB run)

- Uncertainty: **α=0.99 NPZ** — `uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz`, `--provider-start 2448`
- Horizon: **48 h**, `--day2-interval 2`, `--day1-only-robust`; metrics are **day-1 only**
- Start: **m07 d15 h0**
- Network: `--enforce-lines`, `--line-monitor-threshold 0.9`
- `--mip-gap 0.005`, `--bar-qcp-conv-tol 1e-4`
- **Free Z** (`--no-fix-wind-z`), `--robust-ramp`
- **Reserve ramp multiplier = 1.0** (single; `--reserve-ramp-multiplier` default — no sweep)
- No α sweep
- Output: `sensitivity_suite/rho99_48h_m07d15_freeZmatrix/`

## Line-flow handling — verified correct, no fix needed

- **ARUC / DARUC (robust):** `run_comparison.py` delegates to `run_rts_aruc` / `run_rts_daruc`,
  which call `iterative_line_resolve` with the **worst-case** checker
  (`find_worst_case_line_violations`, nominal + ρ·‖Lᵀa‖). Correct.
- **DAM+Reserve (capacity-only):** `build_dam_model` has no Σ/ρ/Z — it enforces **nominal** line
  limits + a scalar capacity reserve only. **Correct by design**; robust line checking here would
  defeat the capacity-only comparison.

## Commands (long-running — run manually)

Optional cleanup of the stale empty skeleton from the aborted run (not required — the new
scenarios use different dir names and overwrite their own outputs):
```
Remove-Item -Recurse -Force sensitivity_suite/rho99_48h_m07d15_freeZmatrix/full_robust_fixed
```
Run the suite (output → `sensitivity_suite/rho99_48h_m07d15_freeZmatrix/`):
```
python run_sensitivity_suite.py --start-month 7 --start-day 15 --hours 48 --day2-interval 2 --uncertainty-npz uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz --provider-start 2448 --line-monitor-threshold 0.9 --mip-gap 0.005
```
Backfill worst-case line flows (reserve & LDR dispatch at worst-case wind):
```
python utils/backfill_worst_case_flows.py --scan-dir sensitivity_suite/rho99_48h_m07d15_freeZmatrix
```
Writes `worst_case_flow_analysis_{daruc,aruc,dam_reserve}.csv` in each case dir.

## Deliverables — tables & charts

### Case comparison
- Table of totals — per case: total cost, energy/commitment split, unit-hours, wind
  curtailment, **incremental commitment** (Δ vs first stage). Day-1 only.
- Drill into commitments — which units each path adds (commitment heatmap diff).
- Z matrix — LDR structure for the free-Z full robust solve (Case 3).
- **Line loading: Reserve vs DARUC** — **dispatch at worst-case wind**: reserve plan
  (proportional reserve deployment) vs robust LDR (`p0 − Z·r_wc`), and resulting line loading.
  From `compute_worst_case_total_shortfall_flows` (`r_arr` for reserve, `Z_arr` for LDR) via the
  backfill step. Shows reserve overloads lines under worst-case while the LDR re-dispatches to
  respect them. *Reporting only — reserve solve stays nominal/capacity-only.*

### Charts
- Selected interesting hour — difference of line loading at one binding hour (Reserve vs DARUC).
- Finalize the ellipsoid charts.
- Consistent scaling across all figures.
- Heatmap figure → stacked version.
- Line-decomp figure: fix squish, larger fonts, consistent naming.
- Add load-to-wind capacity ratio annotation.
- Reserve vs robust capacity — are they different? Compare specific hours.
- Show the Z (B) matrix relative to reserve capacity, across scenarios.

### Paper structure
- Remove robust transmission case (superseded by the reserve-transmission-problem story).

## Decisions log

- **Fixed-Z dropped** — established as the wrong approach; all cases use free Z.
- **Reserve ramp multiplier = 1.0** (single; not sweeping).
- **Line-flow handling verified** — ARUC/DARUC do worst-case resolve; DAM+Reserve correctly
  nominal/capacity-only. No solve-side change.
- **DARUC objective = incremental** for Cases 1 & 2 (isolates added commitment).
- **"One-sided line filtering" already exists** — `filter_monitored_lines` returns
  `flow_direction`; the model adds only the binding-direction constraint. "Alternating" resolves
  to the existing iterative worst-case re-solve.
- **Worst-case line loading comparison (Option B)** — reserve dispatch at worst-case wind vs LDR
  dispatch at worst-case, via the existing backfill utility (no new compute code).
