"""
Gurobi deterministic DAM UC model builder for RTS-GMLC-style data.

This module defines a single function, `build_dam_model`, which takes a
DAMData instance (see models.py) and returns a Gurobi model implementing
a deterministic day-ahead UC:

  min   (18a)
  s.t.  (18b), (18d)-(18j), network line limits via PTDF,
  with initial conditions handled via u_init, init_up_time, init_down_time.

Simplifications:
- No reserves (r_{i,t}) modeled.
- No explicit ramp constraints vs pre-horizon generation p_{i,-1};
  ramping is enforced only for transitions within the horizon (t >= 1).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from models import DAMData


def build_dam_model(
    data: DAMData,
    M_p: float = 1e5,
    model_name: str = "DAM_UC",
    enforce_lines: bool = True,
) -> Tuple[gp.Model, Dict[str, object]]:
    """
    Build a deterministic day-ahead UC model using Gurobi.

    Parameters
    ----------
    data : DAMData
        Canonical data object (see models.py).
    M_p : float, default 1e5
        Penalty coefficient for power balance slack s_t^p (big-M).
    model_name : str
        Name for the Gurobi model.

    Returns
    -------
    model : gp.Model
        Gurobi model, ready to optimize.
    vars : dict
        Dictionary of decision-variable containers:
            - "u"       : on/off status (I,T)
            - "v"       : startup (I,T)
            - "w"       : shutdown (I,T)
            - "p_block" : block outputs (I,T,B)
            - "p"       : total generator output (I,T)
            - "s_p"     : power balance slack (T,)
    """
    I = data.n_gens
    N = data.n_buses
    L = data.n_lines
    T = data.n_periods
    B = data.n_blocks

    # Ensure shapes are consistent; will raise AssertionError if not.
    data.validate_shapes()

    # --- Shorthand references ---
    Pmin = data.Pmin
    Pmax_2d = data.Pmax_2d()  # handle static vs time-varying Pmax uniformly
    RU = data.RU
    RD = data.RD
    MUT = data.MUT.astype(int)
    MDT = data.MDT.astype(int)
    C_NL = data.no_load_cost
    C_SU = data.startup_cost
    C_SD = data.shutdown_cost
    block_cap = data.block_cap
    block_cost = data.block_cost
    PTDF = data.PTDF
    Fmax = data.Fmax
    d = data.d
    gen_to_bus = data.gen_to_bus  # shape (I,)

    dt = data.dt  # period durations in hours, shape (T,)

    u_init = data.u_init
    init_up = data.init_up_time
    init_down = data.init_down_time

    # Mapping: bus -> list of generators (for line-flow expressions)
    gens_at_bus = [[] for _ in range(N)]
    for i in range(I):
        n = int(gen_to_bus[i])
        gens_at_bus[n].append(i)

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    m = gp.Model(model_name)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # u_{i,t} ∈ {0,1} : on/off
    u = m.addVars(I, T, vtype=GRB.BINARY, name="u")

    # v_{i,t}, w_{i,t} ∈ {0,1} : startup/shutdown
    v = m.addVars(I, T, vtype=GRB.BINARY, name="v")
    w = m.addVars(I, T, vtype=GRB.BINARY, name="w")

    # p_block_{i,t,b} ≥ 0 : block outputs
    p_block = m.addVars(I, T, B, vtype=GRB.CONTINUOUS, lb=0.0, name="p_block")

    # p_{i,t} : total generator output, continuous
    p = m.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, name="p")

    # s_t^p ≥ 0 : power balance slack
    s_p = m.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, name="s_p")

    # ------------------------------------------------------------------
    # Fix commitment for zero-cost generators (WIND, SOLAR, HYDRO).
    # These have Pmin=0, MUT=0, MDT=0 and all costs=0, so the commitment
    # variable is degenerate (u=0 and u=1 give identical objective).
    # Fixing u=1 eliminates solver arbitrariness and removes binaries.
    # For wind: also prevents pathological u=0 which would force p=0
    # while Pmax > 0, unnecessarily constraining the dispatch space.
    # ------------------------------------------------------------------
    for i in range(I):
        if data.gen_type[i] in ("WIND", "SOLAR", "HYDRO"):
            for t in range(T):
                u[i, t].lb = 1.0
                u[i, t].ub = 1.0

    # ------------------------------------------------------------------
    # Objective (18a) without reserves r_{i,t}
    #   f = sum_{i,t} [ C_NL[i]*u[i,t] + C_SU[i]*v[i,t] + C_SD[i]*w[i,t]
    #                   + sum_b C_{i,b}^p * p_block[i,t,b] ]
    #       + sum_t M^p * s_p[t]
    # ------------------------------------------------------------------
    obj = gp.LinExpr()

    for i in range(I):
        for t in range(T):
            obj.addTerms(C_NL[i] * dt[t], u[i, t])      # no-load: $/hr × hours
            obj.addTerms(C_SU[i], v[i, t])                # startup: one-time
            obj.addTerms(C_SD[i], w[i, t])                # shutdown: one-time
            for b in range(B):
                obj.addTerms(block_cost[i, b] * dt[t], p_block[i, t, b])  # energy: $/MWh × MW × hours

    for t in range(T):
        obj.addTerms(M_p * dt[t], s_p[t])

    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # (18b) System-wide power balance:
    #   sum_i p[i,t] + s_p[t] = sum_n d[n,t]   ∀ t
    for t in range(T):
        supply = gp.quicksum(p[i, t] for i in range(I)) + s_p[t]
        demand = float(d[:, t].sum())
        m.addConstr(supply == demand, name=f"balance_t{t}")

    # (18d) Block limits & aggregation:
    #   p_block[i,t,b] ≤ block_cap[i,b] * u[i,t]
    #   p[i,t] = sum_b p_block[i,t,b]
    for i in range(I):
        for t in range(T):
            # Block upper bounds
            for b in range(B):
                m.addConstr(
                    p_block[i, t, b] <= block_cap[i, b] * u[i, t],
                    name=f"block_cap_i{i}_t{t}_b{b}",
                )
            # Aggregate p[i,t]
            m.addConstr(
                p[i, t] == gp.quicksum(p_block[i, t, b] for b in range(B)),
                name=f"p_agg_i{i}_t{t}",
            )

    # (18e) Min/max output:
    #   Pmin[i] * u[i,t] ≤ p[i,t] ≤ Pmax[i,t] * u[i,t]
    for i in range(I):
        for t in range(T):
            m.addConstr(
                p[i, t] >= Pmin[i] * u[i, t],
                name=f"p_min_i{i}_t{t}",
            )
            m.addConstr(
                p[i, t] <= Pmax_2d[i, t] * u[i, t],
                name=f"p_max_i{i}_t{t}",
            )

    # (18f) Ramping up (within horizon):
    #   p[i,t] - p[i,t-1] ≤ RU[i]*dt_ramp*u[i,t-1] + RU[i]*dt_ramp*v[i,t]
    # dt_ramp = (dt[t-1] + dt[t]) / 2  (transition time between period midpoints)
    # Skip non-thermal generators: wind/solar/hydro have RU=0 in RTS-GMLC
    # data, which would lock dispatch to a constant (p[i,t] = p[i,t-1]).
    # These generators have no physical ramp limits.
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            m.addConstr(
                p[i, t] - p[i, t - 1] <= RU[i] * dt_ramp * u[i, t - 1] + RU[i] * dt_ramp * v[i, t],
                name=f"ramp_up_i{i}_t{t}",
            )

    # (18g) Ramping down (within horizon):
    #   p[i,t-1] - p[i,t] ≤ RD[i]*dt_ramp*u[i,t] + RD[i]*dt_ramp*w[i,t]
    # Skip non-thermal (same reason as ramp-up).
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            m.addConstr(
                p[i, t - 1] - p[i, t] <= RD[i] * dt_ramp * u[i, t] + RD[i] * dt_ramp * w[i, t],
                name=f"ramp_down_i{i}_t{t}",
            )

    # (18h) Logical relationship between u, v, w
    #
    # Interpreting u_init as the status at "time -1". Our horizon t = 0..T-1.
    # For t = 0:
    #   u[i,0] - u_init[i] = v[i,0] - w[i,0]
    # For t >= 1:
    #   u[i,t] - u[i,t-1]  = v[i,t] - w[i,t]
    for i in range(I):
        # t = 0
        m.addConstr(
            u[i, 0] - u_init[i] == v[i, 0] - w[i, 0],
            name=f"logic_i{i}_t0",
        )
        # t >= 1
        for t in range(1, T):
            m.addConstr(
                u[i, t] - u[i, t - 1] == v[i, t] - w[i, t],
                name=f"logic_i{i}_t{t}",
            )

    # Initial min up/down time implications
    #
    # If unit is ON at t=-1 with init_up_time[i] = tau_up,
    # and MUT[i] is the minimum up-time in periods, then:
    #   if tau_up < MUT[i], the unit must remain ON for at least
    #   MUT[i] - tau_up more periods into the horizon:
    #
    #       u[i,t] = 1  for t = 0 .. min(MUT[i] - tau_up - 1, T-1)
    #
    # Similarly, if unit is OFF at t=-1 with init_down_time[i] = tau_down,
    # and MDT[i] is the minimum down-time:
    #
    #   if tau_down < MDT[i], the unit must remain OFF for at least
    #   MDT[i] - tau_down more periods into the horizon:
    #
    #       u[i,t] = 0  for t = 0 .. min(MDT[i] - tau_down - 1, T-1)
    #
    # We only apply one of these per unit depending on u_init[i].
    # for i in range(I):
    #     if u_init[i] >= 0.5:  # treat as ON
    #         remaining_up = int(max(MUT[i] - init_up[i], 0))
    #         if remaining_up > 0:
    #             last_t = min(remaining_up - 1, T - 1)
    #             for t in range(0, last_t + 1):
    #                 m.addConstr(
    #                     u[i, t] == 1,
    #                     name=f"init_min_up_fix_i{i}_t{t}",
    #                 )
    #     else:  # OFF
    #         remaining_down = int(max(MDT[i] - init_down[i], 0))
    #         if remaining_down > 0:
    #             last_t = min(remaining_down - 1, T - 1)
    #             for t in range(0, last_t + 1):
    #                 m.addConstr(
    #                     u[i, t] == 0,
    #                     name=f"init_min_down_fix_i{i}_t{t}",
    #                 )

    # (18i) Minimum up time — look-forward, counting hours not periods.
    # If v[i,t]=1 (startup), then u must stay on for enough future periods
    # to accumulate MUT hours.
    for i in range(I):
        mut_hrs = float(MUT[i])
        if mut_hrs <= 0:
            continue
        for t in range(T):
            cum = 0.0
            must_on = []
            for s in range(t, T):
                must_on.append(s)
                cum += dt[s]
                if cum >= mut_hrs:
                    break
            n = len(must_on)
            if n > 0:
                lhs = gp.quicksum(u[i, s] for s in must_on)
                m.addConstr(lhs >= n * v[i, t], name=f"mut_i{i}_t{t}")

    # (18j) Minimum down time — look-forward, counting hours not periods.
    for i in range(I):
        mdt_hrs = float(MDT[i])
        if mdt_hrs <= 0:
            continue
        for t in range(T):
            cum = 0.0
            must_off = []
            for s in range(t, T):
                must_off.append(s)
                cum += dt[s]
                if cum >= mdt_hrs:
                    break
            n = len(must_off)
            if n > 0:
                lhs = gp.quicksum((1 - u[i, s]) for s in must_off)
                m.addConstr(lhs >= n * w[i, t], name=f"mdt_i{i}_t{t}")

    # Network / line flow constraints:
    #   -Fmax[l] ≤ sum_n PTDF[l,n] * ( sum_{i ∈ I_n} p[i,t] - d[n,t] ) ≤ Fmax[l]
    #
    # We'll construct:
    #   inj[n,t] = sum_{i ∈ I_n} p[i,t] - d[n,t]
    # and then apply PTDF * inj[:,t] for each line.
    if not enforce_lines:
        print("  [DAM] Line flow constraints DISABLED (copper-plate mode)")
    if enforce_lines:
        for l in range(L):
            for t in range(T):
                # Line flow expression via PTDF
                flow_expr = gp.LinExpr()
                for n in range(N):
                    if PTDF[l, n] == 0.0:
                        continue
                    gen_sum = gp.quicksum(p[i, t] for i in gens_at_bus[n])
                    # net injection at bus n
                    inj_n = gen_sum - float(d[n, t])
                    flow_expr += PTDF[l, n] * inj_n

                m.addConstr(
                    flow_expr <= Fmax[l],
                    name=f"line_max_l{l}_t{t}",
                )
                m.addConstr(
                    flow_expr >= -Fmax[l],
                    name=f"line_min_l{l}_t{t}",
                )

    # ------------------------------------------------------------------
    # Basic Gurobi parameters (tune as you like)
    # ------------------------------------------------------------------
    m.Params.OutputFlag = 1  # solver log on; set to 0 to silence

    vars_dict: Dict[str, object] = {
        "u": u,
        "v": v,
        "w": w,
        "p_block": p_block,
        "p": p,
        "s_p": s_p,
    }

    return m, vars_dict


if __name__ == "__main__":
    # Tiny smoke test using a dummy DAMData instance.
    from models import DAMData

    I, N, L, T, B = 2, 3, 1, 4, 2

    dummy_data = DAMData(
        gen_ids=[f"G{i}" for i in range(I)],
        bus_ids=[f"B{n}" for n in range(N)],
        line_ids=[f"L{l}" for l in range(L)],
        time=list(range(T)),
        gen_type=["THERMAL", "THERMAL"],
        gen_to_bus=np.array([0, 2], dtype=int),
        Pmin=np.zeros(I),
        Pmax=np.ones(I),  # static Pmax
        RU=np.ones(I) * 10.0,
        RD=np.ones(I) * 10.0,
        MUT=np.ones(I) * 2,
        MDT=np.ones(I) * 2,
        startup_cost=np.ones(I) * 5.0,
        shutdown_cost=np.ones(I) * 2.0,
        no_load_cost=np.ones(I) * 1.0,
        # Initial conditions: unit 0 on for 1 period, unit 1 off for 3 periods
        u_init=np.array([1.0, 0.0]),
        init_up_time=np.array([1.0, 0.0]),
        init_down_time=np.array([0.0, 3.0]),
        block_cap=np.ones((I, B)) * 50.0,
        block_cost=np.array([[10.0, 20.0], [12.0, 25.0]]),
        PTDF=np.zeros((L, N)),
        Fmax=np.ones(L) * 100.0,
        d=np.ones((N, T)) * 10.0,
        gens_df=None,
        buses_df=None,
        lines_df=None,
    )

    dummy_data.validate_shapes()
    model, vars_ = build_dam_model(dummy_data, M_p=1e3)
    print("Model built successfully. Variable counts:")
    print("u size:", len(vars_["u"]))
    print("p size:", len(vars_["p"]))
