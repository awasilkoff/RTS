#!/usr/bin/env python
"""
evaluate_commitment_recourse.py

Given a completed comparison run directory, evaluates each model's fixed commitment
against the worst-case wind realization by:
  1. Nominal deficit  — how much load goes unserved if the nominal dispatch is held
                        fixed while wind drops to its worst-case level
  2. LP recourse      — minimum-cost re-dispatch (fixed commitment, worst-case wind)
  3. LDR dispatch     — ARUC/DARUC only: cost via built-in Z response at worst-case wind

This analysis is independent of MIP gap uncertainty in the commitment solve.
Commit costs (no-load, startup) are fixed by the commitment; only energy re-dispatches.

Usage:
    python evaluate_commitment_recourse.py <run_dir> --start-month 7 --start-day 15 --hours 48 --day2-interval 2 --provider-start 2448
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from io_rts import build_damdata_from_rts

M_SHED = 5_000.0    # $/MWh Value of Lost Load (VoLL) penalty for unserved load in recourse LP

SOURCE_DIR = ROOT / "RTS_Data" / "SourceData"
TS_DIR     = ROOT / "RTS_Data" / "timeseries_data_files"
DEFAULT_SPP = ROOT / "uncertainty_sets_refactored" / "data" / "forecasts_filtered_rts4_constellation_v2.parquet"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", help="Comparison run output directory (contains aruc/, daruc/ subdirs)")
    p.add_argument("--start-month",   type=int, default=7)
    p.add_argument("--start-day",     type=int, default=15)
    p.add_argument("--start-hour",    type=int, default=0)
    p.add_argument("--hours",         type=int, default=48)
    p.add_argument("--day2-interval", type=int, default=2,    help="Day-2 period duration (hours)")
    p.add_argument("--provider-start",type=int, default=2448, help="SPP time-series start index")
    p.add_argument("--spp-parquet",   type=Path, default=DEFAULT_SPP)
    p.add_argument("--no-spp",        action="store_true",    help="Use DAY_AHEAD_wind.csv instead of SPP")
    p.add_argument("--three-blocks",  action="store_true", default=False, help="Use 3-block piecewise cost (must match original run setting; default: single block)")
    p.add_argument("--day1-hours",    type=int, default=24)
    p.add_argument("--out-csv",       type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _align_csv(df, gen_ids, time_labels):
    """Reindex DataFrame rows → gen_ids, columns → time_labels (all as str)."""
    df.index   = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.reindex(
        index=[str(g) for g in gen_ids],
        columns=[str(t) for t in time_labels],
        fill_value=0.0,
    ).values.astype(float)


def load_commitment_csv(path, gen_ids, time_labels):
    return _align_csv(pd.read_csv(path, index_col=0), gen_ids, time_labels)


def load_dispatch_csv(path, gen_ids, time_labels):
    return _align_csv(pd.read_csv(path, index_col=0), gen_ids, time_labels)


def load_z_array(z_path, gen_ids, time_labels, K):
    """Load Z_coefficients.csv (MultiIndex columns). Returns (I, T, K) float array."""
    I = len(gen_ids)
    T = len(time_labels)
    Z_df = pd.read_csv(z_path, index_col=0, header=[0, 1])
    Z_df.index = Z_df.index.astype(str)

    gen_strs  = [str(g) for g in gen_ids]
    time_strs = [str(t) for t in time_labels]

    # Build lookup: (t_str, k_int) -> column key in Z_df
    col_lookup = {}
    for col in Z_df.columns:
        t_str = str(col[0])
        try:
            k_int = int(col[1])
        except (ValueError, TypeError):
            continue
        col_lookup[(t_str, k_int)] = col

    Z_arr = np.zeros((I, T, K))
    for t_idx, t_str in enumerate(time_strs):
        for k in range(K):
            col = col_lookup.get((t_str, k))
            if col is None:
                continue
            series = Z_df[col]
            for i_idx, g_str in enumerate(gen_strs):
                if g_str in series.index:
                    Z_arr[i_idx, t_idx, k] = series[g_str]

    return Z_arr


# ---------------------------------------------------------------------------
# Worst-case wind
# ---------------------------------------------------------------------------

def compute_worst_case_pmax(data, Sigma, rho):
    """
    Worst-case total-wind-shortfall direction:
        r_wc[t] = rho[t] * Sigma[t] @ 1 / sqrt(1^T Sigma[t] 1)

    Returns
    -------
    Pmax_wc : (I, T)  — worst-case Pmax for each generator (nominal for non-wind)
    r_wc    : (T, K)  — worst-case deviation vector per wind generator
    R_total : (T,)    — total MW shortfall magnitude
    """
    is_wind  = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    K        = int(is_wind.sum())
    T        = len(data.time)

    Sigma_3d = np.broadcast_to(
        Sigma[None] if Sigma.ndim == 2 else Sigma, (T, K, K)
    ).copy()
    rho_arr  = np.broadcast_to(np.atleast_1d(rho).astype(float), (T,)).copy()

    ones    = np.ones(K)
    r_wc    = np.zeros((T, K))
    R_total = np.zeros(T)
    for t in range(T):
        Se    = Sigma_3d[t] @ ones
        denom = float(np.sqrt(ones @ Se))
        if denom > 1e-12:
            r_wc[t]    = rho_arr[t] * Se / denom
            R_total[t] = rho_arr[t] * denom

    Pmax_2d = data.Pmax_2d()  # (I, T)
    Pmax_wc = Pmax_2d.copy()
    wind_idx = np.where(is_wind)[0]
    for k, i in enumerate(wind_idx):
        Pmax_wc[i, :] = np.maximum(0.0, Pmax_2d[i, :] - r_wc[:, k])

    return Pmax_wc, r_wc, R_total


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def nominal_deficit(data, p0, Pmax_wc, dt, day1_mask):
    """
    Hold nominal dispatch fixed; cap wind to Pmax_wc.
    Returns (deficit_mwh_day1, deficit_per_period, surplus_per_period).
    """
    is_wind  = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    p_actual = p0.copy()
    p_actual[is_wind] = np.minimum(p0[is_wind], Pmax_wc[is_wind])

    gen_total  = p_actual.sum(axis=0)
    load_total = data.d.sum(axis=0)

    deficit  = np.maximum(0.0, load_total - gen_total)
    surplus  = np.maximum(0.0, gen_total  - load_total)
    def_d1   = float((deficit[day1_mask] * dt[day1_mask]).sum())
    return def_d1, deficit, surplus


def compute_ldr_dispatch(p0, Z_arr, r_wc):
    """LDR worst-case: p_wc[i,t] = p0[i,t] - Z[i,t,:] @ r_wc[t,:]."""
    p_wc = p0.copy()
    for t in range(p0.shape[1]):
        p_wc[:, t] = p0[:, t] - Z_arr[:, t, :] @ r_wc[t, :]
    return p_wc


def solve_recourse_lp(data, u_arr, p0_arr, Pmax_wc, name="model"):
    """
    Fixed-commitment re-dispatch LP at worst-case wind.

    Commitment u_arr is fixed; dispatch p[i,t] is free within bounds.
    Wind Pmax is Pmax_wc (worst-case). Load shedding allowed at M_SHED $/MWh.
    Ramp constraints enforced for thermal generators; t=0 ramps from Pmin*u_init.

    Returns dict with dispatch array and cost components, or None if solver failed.
    """
    import gurobipy as gp
    from gurobipy import GRB

    I        = len(data.gen_ids)
    T        = len(data.time)
    dt       = data.dt
    is_therm = np.array([gt == "THERMAL" for gt in data.gen_type])

    # Startup / shutdown indicators from fixed commitment
    u_init   = data.u_init.astype(float)
    u_full   = np.hstack([u_init[:, None], u_arr])           # (I, T+1)
    v_arr    = np.maximum(u_full[:, 1:] - u_full[:, :-1], 0) # (I, T) startup
    w_arr    = np.maximum(u_full[:, :-1] - u_full[:, 1:], 0) # (I, T) shutdown

    mc       = data.block_cost[:, 0]   # $/MWh marginal cost (first block)
    D_total  = data.d.sum(axis=0)      # (T,)

    # Initial dispatch for t=0 ramp: units were at Pmin when on, 0 when off
    p_init   = data.Pmin * u_init      # (I,)

    m = gp.Model(f"recourse_{name}")
    m.setParam("OutputFlag", 0)
    m.setParam("Method", 1)  # dual simplex — fast for LPs

    p    = m.addVars(I, T, lb=0.0, name="p")
    shed = m.addVars(T, lb=0.0, name="shed")

    # Generator dispatch bounds (worst-case Pmax for wind, nominal for thermal)
    for i in range(I):
        for t in range(T):
            p[i, t].lb = float(data.Pmin[i] * u_arr[i, t])
            p[i, t].ub = float(Pmax_wc[i, t] * u_arr[i, t])

    # Power balance (with load shedding)
    for t in range(T):
        m.addConstr(
            gp.quicksum(p[i, t] for i in range(I)) + shed[t] == float(D_total[t]),
            name=f"bal_{t}",
        )

    # Ramp constraints — thermal only
    therm_idx = np.where(is_therm)[0]
    for i in therm_idx:
        # t = 0: ramp from p_init (units at Pmin before horizon)
        dt_ramp0 = float(dt[0])
        m.addConstr(
            p[i, 0] - float(p_init[i]) <= float(data.RU[i]) * dt_ramp0 * float(u_init[i] + v_arr[i, 0]),
            name=f"ru_{i}_0",
        )
        m.addConstr(
            float(p_init[i]) - p[i, 0] <= float(data.RD[i]) * dt_ramp0 * float(u_arr[i, 0] + w_arr[i, 0]),
            name=f"rd_{i}_0",
        )
        # t >= 1
        for t in range(1, T):
            dt_ramp = float((dt[t - 1] + dt[t]) / 2.0)
            m.addConstr(
                p[i, t] - p[i, t - 1] <= float(data.RU[i]) * dt_ramp * float(u_arr[i, t - 1] + v_arr[i, t]),
                name=f"ru_{i}_{t}",
            )
            m.addConstr(
                p[i, t - 1] - p[i, t] <= float(data.RD[i]) * dt_ramp * float(u_arr[i, t] + w_arr[i, t]),
                name=f"rd_{i}_{t}",
            )

    # Objective: energy (single-block approx for LP; post-processed with piecewise)
    # + load shed at VoLL. Using block_cost[:,0] for optimization is fine — the
    # ranking of generators by marginal cost is approximately correct for dispatch.
    m.setObjective(
        gp.quicksum(float(mc[i] * dt[t]) * p[i, t] for i in range(I) for t in range(T))
        + M_SHED * gp.quicksum(float(dt[t]) * shed[t] for t in range(T)),
        GRB.MINIMIZE,
    )

    m.optimize()

    if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None

    p_val    = np.array([[p[i, t].X for t in range(T)] for i in range(I)])
    shed_val = np.array([shed[t].X for t in range(T)])

    return {"p": p_val, "shed": shed_val}


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def energy_cost_d1(data, p_arr, day1_mask):
    """Day-1 energy cost using piecewise-linear block allocation (matches comparison summary)."""
    dt         = data.dt
    block_cap  = data.block_cap   # (I, B)
    block_cost = data.block_cost  # (I, B)
    B          = block_cap.shape[1]
    cost       = 0.0
    for i in range(len(data.gen_ids)):
        for t in range(len(data.time)):
            if not day1_mask[t]:
                continue
            remaining = float(max(0.0, p_arr[i, t]))
            for b in range(B):
                allocated  = min(remaining, float(block_cap[i, b]))
                cost      += allocated * float(block_cost[i, b]) * float(dt[t])
                remaining -= allocated
                if remaining <= 1e-9:
                    break
    return cost


def commitment_cost_d1(data, u_arr, day1_mask):
    """Day-1 no-load + startup cost from commitment array."""
    dt     = data.dt
    u_prev = data.u_init.astype(float)
    v_arr  = np.maximum(
        np.hstack([u_prev[:, None], u_arr])[:, 1:] - np.hstack([u_prev[:, None], u_arr])[:, :-1],
        0,
    )
    no_load = float(np.sum(data.no_load_cost[:, None] * u_arr[:, day1_mask] * dt[None, day1_mask]))
    startup = float(np.sum(data.startup_cost[:, None] * v_arr[:, day1_mask]))
    return no_load + startup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}")
        sys.exit(1)

    spp = None if args.no_spp else args.spp_parquet
    if spp and not Path(spp).exists():
        print(f"Warning: SPP parquet not found at {spp} — using DAY_AHEAD_wind.csv")
        spp = None

    import pandas as _pd
    start_time = _pd.Timestamp(year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour)

    print(f"Building DAMData: {start_time.date()}, {args.hours}h, day2_interval={args.day2_interval}h")
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start_time,
        horizon_hours=args.hours,
        day2_interval_hours=args.day2_interval,
        spp_forecasts_parquet=spp,
        spp_start_idx=args.provider_start,
        single_block=not args.three_blocks,
    )

    I, T      = len(data.gen_ids), len(data.time)
    dt        = data.dt
    day1_mask = data.day1_period_mask(args.day1_hours)
    is_wind   = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    K         = int(is_wind.sum())
    gen_ids   = data.gen_ids
    t_labels  = data.time
    D1        = int(day1_mask.sum())

    print(f"  {I} generators, {T} periods ({D1} day-1), {K} wind generators")

    # Load Sigma / rho — prefer aruc/, fall back to daruc/
    Sigma, rho = None, None
    for sub in ["aruc", "daruc"]:
        sp = run_dir / sub / "Sigma.npy"
        rp = run_dir / sub / "rho.npy"
        if sp.exists() and rp.exists():
            Sigma = np.load(sp)
            rho   = np.load(rp)
            rho   = float(rho.flat[0]) if rho.size == 1 else rho
            print(f"  Sigma {Sigma.shape}, rho {np.atleast_1d(rho).shape} loaded from {sub}/")
            break

    if Sigma is None:
        print("ERROR: Sigma.npy / rho.npy not found in aruc/ or daruc/")
        sys.exit(1)

    # Worst-case wind
    Pmax_wc, r_wc, R_total = compute_worst_case_pmax(data, Sigma, rho)
    Pmax_nom = data.Pmax_2d()
    wc_wind  = float((Pmax_wc[is_wind]  * dt).sum())
    nom_wind = float((Pmax_nom[is_wind] * dt).sum())
    D1_load  = float((data.d.sum(axis=0)[day1_mask] * dt[day1_mask]).sum())

    print(f"\n  WC wind (full horizon): {wc_wind:,.0f} MWh  (nominal: {nom_wind:,.0f} MWh)")
    print(f"  Wind reduction at WC  : {nom_wind - wc_wind:,.0f} MWh  ({100*(nom_wind-wc_wind)/nom_wind:.1f}%)")
    print(f"  Day-1 total load      : {D1_load:,.0f} MWh")

    # Cases: (label, commitment_csv, dispatch_csv, z_csv_or_None)
    cases = [
        ("DAM",
         run_dir / "daruc" / "dam_commitment_u.csv",
         run_dir / "daruc" / "dam_dispatch_p0.csv",
         None),
        ("DAM+Reserve",
         run_dir / "dam_reserve" / "commitment_u.csv",
         run_dir / "dam_reserve" / "dispatch_p0.csv",
         None),
        ("DARUC",
         run_dir / "daruc" / "commitment_u.csv",
         run_dir / "daruc" / "dispatch_p0.csv",
         run_dir / "daruc" / "Z_coefficients.csv"),
        ("ARUC",
         run_dir / "aruc" / "commitment_u.csv",
         run_dir / "aruc" / "dispatch_p0.csv",
         run_dir / "aruc" / "Z_coefficients.csv"),
    ]

    rows = []

    for label, u_path, p0_path, z_path in cases:
        if not u_path.exists():
            print(f"\n  {label}: skipping (missing {u_path.name})")
            continue

        print(f"\n{'='*60}\n{label}\n{'='*60}")

        u_arr  = load_commitment_csv(u_path, gen_ids, t_labels)
        p0_arr = load_dispatch_csv(p0_path, gen_ids, t_labels) if p0_path.exists() else None

        comm_cost_d1 = commitment_cost_d1(data, u_arr, day1_mask)
        nom_energy_d1 = energy_cost_d1(data, p0_arr, day1_mask) if p0_arr is not None else None
        nom_total_d1  = (comm_cost_d1 + nom_energy_d1) if nom_energy_d1 is not None else None

        print(f"  Nominal day-1 commit cost  : ${comm_cost_d1:>12,.0f}")
        if nom_energy_d1 is not None:
            print(f"  Nominal day-1 energy cost  : ${nom_energy_d1:>12,.0f}")
            print(f"  Nominal day-1 total        : ${nom_total_d1:>12,.0f}")

        # 1. Nominal deficit at worst-case wind
        def_d1 = None
        if p0_arr is not None:
            def_d1, deficit_t, _ = nominal_deficit(data, p0_arr, Pmax_wc, dt, day1_mask)
            def_pct = 100 * def_d1 / D1_load if D1_load > 0 else 0.0
            print(f"\n  [Nominal dispatch, WC wind — no re-dispatch]")
            print(f"  Load deficit (day-1)       : {def_d1:>10,.1f} MWh  ({def_pct:.2f}% of load)")
            if def_d1 > 0:
                worst_t = int(np.argmax(deficit_t))
                print(f"  Worst deficit period       : period {worst_t}  ({deficit_t[worst_t]:,.1f} MW)")

        # 2. LP recourse
        lp_energy_d1 = lp_shed_d1 = lp_total_d1 = None
        print(f"\n  [LP recourse — optimal re-dispatch at WC wind]")
        lp = solve_recourse_lp(data, u_arr, p0_arr, Pmax_wc, name=label)
        if lp is not None:
            lp_energy_d1 = energy_cost_d1(data, lp["p"], day1_mask)
            lp_shed_d1   = float((lp["shed"][day1_mask] * dt[day1_mask]).sum())
            lp_total_d1  = comm_cost_d1 + lp_energy_d1
            print(f"  Energy cost (day-1)        : ${lp_energy_d1:>12,.0f}")
            print(f"  Total cost  (day-1)        : ${lp_total_d1:>12,.0f}")
            print(f"  Load shed   (day-1)        : {lp_shed_d1:>10,.1f} MWh")
            if nom_energy_d1 is not None:
                delta = lp_energy_d1 - nom_energy_d1
                print(f"  Energy premium vs nominal  : ${delta:>+12,.0f}  ({100*delta/nom_energy_d1:+.2f}%)")
        else:
            print(f"  LP recourse: solver failed")

        # 3. LDR dispatch (ARUC/DARUC)
        ldr_energy_d1 = ldr_total_d1 = None
        if z_path and z_path.exists() and p0_arr is not None:
            print(f"\n  [LDR response — Z @ (-r_wc) applied to nominal dispatch]")
            Z_arr        = load_z_array(z_path, gen_ids, t_labels, K)
            p_ldr        = compute_ldr_dispatch(p0_arr, Z_arr, r_wc)
            ldr_energy_d1 = energy_cost_d1(data, p_ldr, day1_mask)
            ldr_total_d1  = comm_cost_d1 + ldr_energy_d1
            # Check power balance (should be ~0 by Z construction)
            imbalance    = p_ldr.sum(axis=0) - data.d.sum(axis=0)
            max_imbal    = float(np.abs(imbalance[day1_mask]).max())
            print(f"  Energy cost (day-1)        : ${ldr_energy_d1:>12,.0f}")
            print(f"  Total cost  (day-1)        : ${ldr_total_d1:>12,.0f}")
            print(f"  Max power imbalance        : {max_imbal:>10.2f} MW  (should be ~0)")
            if nom_energy_d1 is not None:
                delta = ldr_energy_d1 - nom_energy_d1
                print(f"  Energy premium vs nominal  : ${delta:>+12,.0f}  ({100*delta/nom_energy_d1:+.2f}%)")

        rows.append({
            "model":             label,
            "commit_cost_d1":    comm_cost_d1,
            "nom_energy_d1":     nom_energy_d1,
            "nom_total_d1":      nom_total_d1,
            "nom_deficit_mwh":   def_d1,
            "lp_energy_d1":      lp_energy_d1,
            "lp_shed_mwh_d1":    lp_shed_d1,
            "lp_total_d1":       lp_total_d1,
            "ldr_energy_d1":     ldr_energy_d1,
            "ldr_total_d1":      ldr_total_d1,
        })

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RECOURSE EVALUATION SUMMARY  (day-1 metrics, worst-case wind)")
    print("=" * 90)

    hdr = f"  {'Model':<13}  {'Commit':>10}  {'Nom Energy':>11}  {'Nom Total':>11}  {'WC Deficit':>11}  {'LP Total':>11}  {'LP Shed':>9}  {'LDR Total':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def _f(v, fmt=">11,.0f", prefix="$"):
        return f"{prefix}{v:{fmt}}" if v is not None else f"{'N/A':>11}"

    def _mwh(v):
        return f"{v:>9,.1f}" if v is not None else f"{'N/A':>9}"

    for r in rows:
        print(
            f"  {r['model']:<13}"
            f"  {_f(r['commit_cost_d1'])}"
            f"  {_f(r['nom_energy_d1'])}"
            f"  {_f(r['nom_total_d1'])}"
            f"  {_mwh(r['nom_deficit_mwh'])} MWh"
            f"  {_f(r['lp_total_d1'])}"
            f"  {_mwh(r['lp_shed_mwh_d1'])} MWh"
            f"  {_f(r['ldr_total_d1'])}"
        )

    # Cost premium vs DAM LP recourse
    dam_lp = next((r["lp_total_d1"] for r in rows if r["model"] == "DAM"), None)
    if dam_lp is not None:
        print(f"\n  LP recourse cost premium vs DAM (day-1):")
        for r in rows:
            if r["lp_total_d1"] is not None and r["model"] != "DAM":
                delta = r["lp_total_d1"] - dam_lp
                pct   = 100 * delta / dam_lp
                print(f"    {r['model']:<13}  ${delta:>+12,.0f}  ({pct:+.2f}%)")

    print()

    # Save CSV
    out_csv = args.out_csv or (run_dir / "recourse_evaluation.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    main()
