"""
validate_dam.py

Post-solve validation of the DAM UC formulation.

Runs a DAM solve, extracts all Gurobi variables, checks every constraint
post-hoc with numerical tolerance, and prints a comprehensive diagnostic
report including dispatch sanity checks and cost breakdown.

Usage:
    python validate_dam.py
    python validate_dam.py --start-month 1 --start-day 1 --hours 24
    python validate_dam.py --start-month 7 --start-day 15 --hours 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_rts_dam import run_rts_dam

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M_PENALTY = 1e4
TOL = 1e-4


# ---------------------------------------------------------------------------
# Section 1: Extract full solution arrays
# ---------------------------------------------------------------------------

def extract_all_vars(vars_dict, I, T, B):
    """Extract all 6 variable groups from Gurobi vars_dict into numpy arrays."""
    u = np.zeros((I, T))
    v = np.zeros((I, T))
    w = np.zeros((I, T))
    p = np.zeros((I, T))
    p_block = np.zeros((I, T, B))
    s_p = np.zeros(T)

    for i in range(I):
        for t in range(T):
            u[i, t] = vars_dict["u"][i, t].X
            v[i, t] = vars_dict["v"][i, t].X
            w[i, t] = vars_dict["w"][i, t].X
            p[i, t] = vars_dict["p"][i, t].X
            for b in range(B):
                p_block[i, t, b] = vars_dict["p_block"][i, t, b].X

    for t in range(T):
        s_p[t] = vars_dict["s_p"][t].X

    return u, v, w, p, p_block, s_p


# ---------------------------------------------------------------------------
# Section 2: Constraint checks
# ---------------------------------------------------------------------------

def check_constraint(name, violations, max_viol):
    """Print PASS/FAIL for a constraint check."""
    n = int(np.sum(violations))
    if n == 0:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: {n} violations, max = {max_viol:.6e}")
    return n == 0


def run_constraint_checks(data, u, v, w, p, p_block, s_p):
    """Run all post-hoc constraint checks. Returns True if all pass."""
    I = data.n_gens
    T = data.n_periods
    B = data.n_blocks
    Pmin = data.Pmin
    Pmax_2d = data.Pmax_2d()
    RU = data.RU
    RD = data.RD
    MUT = data.MUT.astype(int)
    MDT = data.MDT.astype(int)
    block_cap = data.block_cap
    d = data.d
    dt = data.dt
    u_init = data.u_init

    all_pass = True

    print("\n" + "=" * 60)
    print("CONSTRAINT CHECKS (tol = {:.0e})".format(TOL))
    print("=" * 60)

    # 1. Power balance: sum_i p[i,t] + s_p[t] = sum_n d[n,t]
    viol = np.zeros(T, dtype=bool)
    max_v = 0.0
    for t in range(T):
        lhs = p[:, t].sum() + s_p[t]
        rhs = d[:, t].sum()
        err = abs(lhs - rhs)
        max_v = max(max_v, err)
        if err > TOL:
            viol[t] = True
    all_pass &= check_constraint("Power balance", viol, max_v)

    # 2. Pmin bound: p[i,t] >= Pmin[i] * u[i,t]
    viol = np.zeros((I, T), dtype=bool)
    max_v = 0.0
    for i in range(I):
        for t in range(T):
            deficit = Pmin[i] * u[i, t] - p[i, t]
            if deficit > TOL:
                viol[i, t] = True
                max_v = max(max_v, deficit)
    all_pass &= check_constraint("Pmin bound", viol, max_v)

    # 3. Pmax bound: p[i,t] <= Pmax_2d[i,t] * u[i,t]
    viol = np.zeros((I, T), dtype=bool)
    max_v = 0.0
    for i in range(I):
        for t in range(T):
            excess = p[i, t] - Pmax_2d[i, t] * u[i, t]
            if excess > TOL:
                viol[i, t] = True
                max_v = max(max_v, excess)
    all_pass &= check_constraint("Pmax bound", viol, max_v)

    # 4. Block cap: p_block[i,t,b] <= block_cap[i,b] * u[i,t]
    viol = np.zeros((I, T, B), dtype=bool)
    max_v = 0.0
    for i in range(I):
        for t in range(T):
            for b in range(B):
                excess = p_block[i, t, b] - block_cap[i, b] * u[i, t]
                if excess > TOL:
                    viol[i, t, b] = True
                    max_v = max(max_v, excess)
    all_pass &= check_constraint("Block cap", viol, max_v)

    # 5. Block aggregation: p[i,t] = sum_b p_block[i,t,b]
    viol = np.zeros((I, T), dtype=bool)
    max_v = 0.0
    for i in range(I):
        for t in range(T):
            err = abs(p[i, t] - p_block[i, t, :].sum())
            if err > TOL:
                viol[i, t] = True
                max_v = max(max_v, err)
    all_pass &= check_constraint("Block aggregation", viol, max_v)

    # 6. Commitment logic (t > 0): u[i,t] - u[i,t-1] = v[i,t] - w[i,t]
    viol = np.zeros((I, T - 1), dtype=bool)
    max_v = 0.0
    for i in range(I):
        for t in range(1, T):
            err = abs((u[i, t] - u[i, t - 1]) - (v[i, t] - w[i, t]))
            if err > TOL:
                viol[i, t - 1] = True
                max_v = max(max_v, err)
    all_pass &= check_constraint("Commitment logic (t>0)", viol, max_v)

    # 7. Initial logic: u[i,0] - u_init[i] = v[i,0] - w[i,0]
    viol = np.zeros(I, dtype=bool)
    max_v = 0.0
    for i in range(I):
        err = abs((u[i, 0] - u_init[i]) - (v[i, 0] - w[i, 0]))
        if err > TOL:
            viol[i] = True
            max_v = max(max_v, err)
    all_pass &= check_constraint("Initial logic (t=0)", viol, max_v)

    # 8. Ramp up (thermal only): p[i,t] - p[i,t-1] <= RU[i]*dt_ramp*(u[i,t-1]+v[i,t])
    viol_list = []
    max_v = 0.0
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            lhs = p[i, t] - p[i, t - 1]
            rhs = RU[i] * dt_ramp * (u[i, t - 1] + v[i, t])
            excess = lhs - rhs
            if excess > TOL:
                viol_list.append((i, t, excess))
                max_v = max(max_v, excess)
    n_ramp_up_viol = len(viol_list)
    if n_ramp_up_viol == 0:
        print(f"  PASS  Ramp up (thermal)")
    else:
        print(f"  FAIL  Ramp up (thermal): {n_ramp_up_viol} violations, max = {max_v:.6e}")
        for i, t, exc in viol_list[:5]:
            print(f"         gen {data.gen_ids[i]} t={t}: excess={exc:.4f}")
        all_pass = False

    # 9. Ramp down (thermal only): p[i,t-1] - p[i,t] <= RD[i]*dt_ramp*(u[i,t]+w[i,t])
    viol_list = []
    max_v = 0.0
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            lhs = p[i, t - 1] - p[i, t]
            rhs = RD[i] * dt_ramp * (u[i, t] + w[i, t])
            excess = lhs - rhs
            if excess > TOL:
                viol_list.append((i, t, excess))
                max_v = max(max_v, excess)
    n_ramp_dn_viol = len(viol_list)
    if n_ramp_dn_viol == 0:
        print(f"  PASS  Ramp down (thermal)")
    else:
        print(f"  FAIL  Ramp down (thermal): {n_ramp_dn_viol} violations, max = {max_v:.6e}")
        for i, t, exc in viol_list[:5]:
            print(f"         gen {data.gen_ids[i]} t={t}: excess={exc:.4f}")
        all_pass = False

    # 10. MUT: if v[i,t]=1, u must stay on for enough periods to cover MUT hours
    viol_list = []
    for i in range(I):
        mut_hrs = int(MUT[i])
        if mut_hrs <= 0:
            continue
        for t in range(T):
            if v[i, t] < 0.5:
                continue
            # Count forward: accumulate hours until >= MUT
            cum = 0.0
            must_on = []
            for s in range(t, T):
                must_on.append(s)
                cum += dt[s]
                if cum >= mut_hrs:
                    break
            # Check all periods in must_on have u=1
            for s in must_on:
                if u[i, s] < 0.5:
                    viol_list.append((i, t, s))
    if len(viol_list) == 0:
        print(f"  PASS  Minimum up time (MUT)")
    else:
        print(f"  FAIL  Minimum up time (MUT): {len(viol_list)} violations")
        for i, t, s in viol_list[:5]:
            print(f"         gen {data.gen_ids[i]}: startup at t={t}, off at t={s} (MUT={MUT[i]}h)")
        all_pass = False

    # 11. MDT: if w[i,t]=1, u must stay off for enough periods to cover MDT hours
    viol_list = []
    for i in range(I):
        mdt_hrs = int(MDT[i])
        if mdt_hrs <= 0:
            continue
        for t in range(T):
            if w[i, t] < 0.5:
                continue
            cum = 0.0
            must_off = []
            for s in range(t, T):
                must_off.append(s)
                cum += dt[s]
                if cum >= mdt_hrs:
                    break
            for s in must_off:
                if u[i, s] > 0.5:
                    viol_list.append((i, t, s))
    if len(viol_list) == 0:
        print(f"  PASS  Minimum down time (MDT)")
    else:
        print(f"  FAIL  Minimum down time (MDT): {len(viol_list)} violations")
        for i, t, s in viol_list[:5]:
            print(f"         gen {data.gen_ids[i]}: shutdown at t={t}, on at t={s} (MDT={MDT[i]}h)")
        all_pass = False

    # 12. Zero dispatch when off: u[i,t]=0 => p[i,t]=0
    viol = np.zeros((I, T), dtype=bool)
    max_v = 0.0
    for i in range(I):
        for t in range(T):
            if u[i, t] < 0.5 and p[i, t] > TOL:
                viol[i, t] = True
                max_v = max(max_v, p[i, t])
    all_pass &= check_constraint("Zero dispatch when off", viol, max_v)

    # 13. Wind/solar/hydro u=1
    viol = np.zeros((I, T), dtype=bool)
    max_v = 0.0
    for i in range(I):
        if data.gen_type[i] in ("WIND", "SOLAR", "HYDRO"):
            for t in range(T):
                if u[i, t] < 0.5:
                    viol[i, t] = True
                    max_v = 1.0
    all_pass &= check_constraint("Non-thermal u=1", viol, max_v)

    # 14. Slack usage
    total_slack = s_p.sum()
    max_slack = s_p.max()
    if total_slack < TOL:
        print(f"  PASS  Slack usage: all zero")
    else:
        print(f"  INFO  Slack usage: total = {total_slack:.4f} MW, max = {max_slack:.4f} MW")
        nonzero_t = np.where(s_p > TOL)[0]
        for t in nonzero_t[:5]:
            print(f"         t={t}: s_p = {s_p[t]:.4f} MW")

    return all_pass


# ---------------------------------------------------------------------------
# Section 3: Dispatch sanity report
# ---------------------------------------------------------------------------

def dispatch_sanity_report(data, u, v, w, p, p_block, s_p, obj_val):
    """Print comprehensive dispatch diagnostics."""
    I = data.n_gens
    T = data.n_periods
    B = data.n_blocks
    dt = data.dt
    Pmax_2d = data.Pmax_2d()
    d = data.d

    print("\n" + "=" * 60)
    print("DISPATCH SANITY REPORT")
    print("=" * 60)

    # --- 3.1 Energy balance table ---
    print("\n--- Energy Balance (per period) ---")
    print(f"{'Period':>6}  {'Load MW':>10}  {'Gen MW':>10}  {'Slack MW':>10}  {'Imbal MW':>10}")
    for t in range(T):
        load_t = d[:, t].sum()
        gen_t = p[:, t].sum()
        slack_t = s_p[t]
        imbal = gen_t + slack_t - load_t
        if t < 10 or t >= T - 3 or abs(imbal) > TOL:
            print(f"{t:>6}  {load_t:>10.2f}  {gen_t:>10.2f}  {slack_t:>10.2f}  {imbal:>10.4f}")
    if T > 13:
        print(f"  ... ({T - 13} periods omitted, showing first 10 and last 3)")

    # --- 3.2 Generation by type ---
    print("\n--- Generation by Type (MWh over horizon) ---")
    type_mwh = {}
    type_avail = {}
    for gtype in sorted(set(data.gen_type)):
        mwh = 0.0
        avail = 0.0
        for i in range(I):
            if data.gen_type[i] == gtype:
                for t in range(T):
                    mwh += p[i, t] * dt[t]
                    avail += Pmax_2d[i, t] * dt[t]
        type_mwh[gtype] = mwh
        type_avail[gtype] = avail

    total_mwh = sum(type_mwh.values())
    for gtype in sorted(type_mwh.keys()):
        mwh = type_mwh[gtype]
        avail = type_avail[gtype]
        util = (mwh / avail * 100) if avail > 0 else 0.0
        pct = (mwh / total_mwh * 100) if total_mwh > 0 else 0.0
        print(f"  {gtype:<10} {mwh:>12,.1f} MWh  ({pct:>5.1f}% of total)  utilization: {util:>5.1f}%")
    print(f"  {'TOTAL':<10} {total_mwh:>12,.1f} MWh")

    total_load_mwh = sum(d[:, t].sum() * dt[t] for t in range(T))
    print(f"  Total load: {total_load_mwh:>12,.1f} MWh")

    # --- 3.3 Cost breakdown ---
    print("\n--- Cost Breakdown ---")
    C_NL = data.no_load_cost
    C_SU = data.startup_cost
    C_SD = data.shutdown_cost
    block_cost = data.block_cost

    noload_cost = sum(C_NL[i] * u[i, t] * dt[t] for i in range(I) for t in range(T))
    startup_cost = sum(C_SU[i] * v[i, t] for i in range(I) for t in range(T))
    shutdown_cost = sum(C_SD[i] * w[i, t] for i in range(I) for t in range(T))
    energy_cost = sum(
        block_cost[i, b] * p_block[i, t, b] * dt[t]
        for i in range(I) for t in range(T) for b in range(B)
    )
    slack_cost = sum(M_PENALTY * s_p[t] * dt[t] for t in range(T))

    computed_total = noload_cost + startup_cost + shutdown_cost + energy_cost + slack_cost

    print(f"  No-load:   ${noload_cost:>14,.2f}")
    print(f"  Startup:   ${startup_cost:>14,.2f}")
    print(f"  Shutdown:  ${shutdown_cost:>14,.2f}")
    print(f"  Energy:    ${energy_cost:>14,.2f}")
    print(f"  Slack:     ${slack_cost:>14,.2f}")
    print(f"  --------------------")
    print(f"  Computed:  ${computed_total:>14,.2f}")
    print(f"  Gurobi:    ${obj_val:>14,.2f}")
    obj_err = abs(computed_total - obj_val)
    if obj_err < 1.0:
        print(f"  Match: PASS (diff = ${obj_err:.4f})")
    else:
        print(f"  Match: FAIL (diff = ${obj_err:.2f})")

    # --- 3.4 Commitment summary ---
    print("\n--- Commitment Summary ---")
    thermal_mask = np.array([gt == "THERMAL" for gt in data.gen_type])
    n_thermal = thermal_mask.sum()
    committed_per_t = np.array([u[thermal_mask, t].sum() for t in range(T)])
    startups_per_t = np.array([v[thermal_mask, t].sum() for t in range(T)])
    shutdowns_per_t = np.array([w[thermal_mask, t].sum() for t in range(T)])

    print(f"  Thermal units: {n_thermal}")
    print(f"  Committed (thermal): min={committed_per_t.min():.0f}, "
          f"max={committed_per_t.max():.0f}, mean={committed_per_t.mean():.1f}")
    print(f"  Total startups:  {startups_per_t.sum():.0f}")
    print(f"  Total shutdowns: {shutdowns_per_t.sum():.0f}")

    # --- 3.5 Top 5 most expensive generators ---
    print("\n--- Top 5 Most Expensive Generators (energy cost) ---")
    gen_energy_cost = np.zeros(I)
    for i in range(I):
        for t in range(T):
            for b in range(B):
                gen_energy_cost[i] += block_cost[i, b] * p_block[i, t, b] * dt[t]

    top5 = np.argsort(gen_energy_cost)[::-1][:5]
    for rank, i in enumerate(top5):
        total_mwh_i = sum(p[i, t] * dt[t] for t in range(T))
        print(f"  {rank+1}. {data.gen_ids[i]:<20} ${gen_energy_cost[i]:>12,.2f}  "
              f"({total_mwh_i:>8,.1f} MWh, type={data.gen_type[i]})")

    # --- 3.6 Renewable dispatch profile ---
    print("\n--- Renewable Dispatch vs Available ---")
    for i in range(I):
        if data.gen_type[i] not in ("WIND", "SOLAR"):
            continue
        total_disp = sum(p[i, t] * dt[t] for t in range(T))
        total_avail = sum(Pmax_2d[i, t] * dt[t] for t in range(T))
        util = (total_disp / total_avail * 100) if total_avail > 0 else 0.0
        curtailed_periods = sum(1 for t in range(T) if Pmax_2d[i, t] > TOL and p[i, t] < Pmax_2d[i, t] - TOL)
        print(f"  {data.gen_ids[i]:<20} {total_disp:>8,.1f}/{total_avail:>8,.1f} MWh "
              f"({util:>5.1f}% util)  curtailed in {curtailed_periods}/{T} periods")


# ---------------------------------------------------------------------------
# Section 4: Line flow check
# ---------------------------------------------------------------------------

def check_line_flows(data, p):
    """Check PTDF-based line flow limits."""
    I = data.n_gens
    N = data.n_buses
    L = data.n_lines
    T = data.n_periods
    PTDF = data.PTDF
    Fmax = data.Fmax
    d = data.d
    gen_to_bus = data.gen_to_bus.astype(int)

    print("\n" + "=" * 60)
    print("LINE FLOW CHECK")
    print("=" * 60)

    # Build net injection per bus per period
    inj = np.zeros((N, T))
    for i in range(I):
        n = gen_to_bus[i]
        inj[n, :] += p[i, :]
    inj -= d  # inj[n,t] = gen_at_bus_n - load_at_bus_n

    # Compute flows: flow[l,t] = PTDF[l,:] @ inj[:,t]
    flow = PTDF @ inj  # (L, T)

    violations = 0
    max_viol = 0.0
    most_congested = []

    for l in range(L):
        for t in range(T):
            excess = abs(flow[l, t]) - Fmax[l]
            if excess > TOL:
                violations += 1
                max_viol = max(max_viol, excess)

        # Track max loading ratio per line
        max_loading = np.max(np.abs(flow[l, :])) / Fmax[l] if Fmax[l] > 0 else 0.0
        most_congested.append((l, max_loading))

    if violations == 0:
        print(f"  PASS  Line flow limits: no violations")
    else:
        print(f"  FAIL  Line flow limits: {violations} violations, max excess = {max_viol:.4f} MW")

    # Top 5 most congested lines
    most_congested.sort(key=lambda x: -x[1])
    print(f"\n  Top 5 most congested lines:")
    for l, ratio in most_congested[:5]:
        print(f"    Line {data.line_ids[l]:<15} max loading: {ratio*100:.1f}%  "
              f"(Fmax = {Fmax[l]:.1f} MW)")

    return violations == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate DAM UC formulation")
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--no-lines", action="store_true", help="Disable line flow constraints")
    args = parser.parse_args()

    start_time = pd.Timestamp(year=2020, month=args.start_month, day=args.start_day, hour=0)
    enforce_lines = not args.no_lines

    print("=" * 60)
    print("DAM FORMULATION VALIDATION")
    print(f"  Period: {start_time.strftime('%Y-%m-%d')} for {args.hours}h")
    print(f"  Lines: {'enforced' if enforce_lines else 'disabled'}")
    print("=" * 60)

    # Step 1: Run DAM
    outputs = run_rts_dam(
        start_time=start_time,
        horizon_hours=args.hours,
        m_penalty=M_PENALTY,
        enforce_lines=enforce_lines,
    )

    data = outputs["data"]
    model = outputs["model"]
    vars_dict = outputs["vars"]
    obj_val = outputs["results"]["obj"]

    I = data.n_gens
    T = data.n_periods
    B = data.n_blocks

    # Step 2: Extract all variables
    print("\nExtracting solution variables...")
    u, v, w, p, p_block, s_p = extract_all_vars(vars_dict, I, T, B)
    print(f"  u: {u.shape}, v: {v.shape}, w: {w.shape}")
    print(f"  p: {p.shape}, p_block: {p_block.shape}, s_p: {s_p.shape}")

    # Step 3: Constraint checks
    all_pass = run_constraint_checks(data, u, v, w, p, p_block, s_p)

    # Step 4: Dispatch sanity report
    dispatch_sanity_report(data, u, v, w, p, p_block, s_p, obj_val)

    # Step 5: Line flow check (if lines enforced)
    if enforce_lines:
        lines_pass = check_line_flows(data, p)
        all_pass &= lines_pass

    # Final verdict
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see details above")
    print("=" * 60)


if __name__ == "__main__":
    main()
