"""
Notification-Gated Reliability Unit Commitment (LD-RUC) model builder.

Two-phase decomposition:
  Phase 1: Deterministic Gated MIP — commitment with gated objective
  Phase 2: Robust Feasibility SOCP — dispatch verification with fixed binaries
  CCG:     Column-and-constraint generation loop when Phase 2 fails

Reuses UC constraint patterns from dam_model.py and SOC patterns from
aruc_model.py.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DAMData


# ======================================================================
# 1. Gating set computation
# ======================================================================


def compute_gating_sets(
    data: DAMData,
    notification_times: np.ndarray,
    t_next: int,
) -> np.ndarray:
    """
    Compute the notification-gated set for each generator and period.

    A period t is gated for generator i if the startup decision at t
    cannot be deferred to the next decision process:

        gated[i, t]  iff  t_hours - L_i^SU < t_next_hours

    where t_hours is the cumulative hour at the *start* of period t.

    Parameters
    ----------
    data : DAMData
        UC problem data (provides period_duration via data.dt).
    notification_times : np.ndarray
        (I,) notification lead times in hours per generator.
    t_next : int
        First period index modifiable by the next process (e.g. 25 for
        DA-RUC where the next process starts at hour 25).

    Returns
    -------
    gated : np.ndarray
        (I, T) boolean mask. True = gated (non-deferrable).
    """
    I = data.n_gens
    T = data.n_periods
    dt = data.dt

    # Compute cumulative hours at the start of each period
    cum_hours = np.zeros(T)
    for t in range(1, T):
        cum_hours[t] = cum_hours[t - 1] + dt[t - 1]

    # t_next in hours
    t_next_hours = float(cum_hours[min(t_next, T - 1)] if t_next < T else cum_hours[-1] + dt[-1])

    gated = np.zeros((I, T), dtype=bool)
    for i in range(I):
        L_i = float(notification_times[i])
        if L_i <= 0:
            continue
        for t in range(T):
            # A startup at period t requires notification by (t_hours - L_i).
            # If that notification deadline is before t_next, it's non-deferrable.
            if cum_hours[t] - L_i < t_next_hours:
                gated[i, t] = True

    return gated


# ======================================================================
# 2. Phase 1: Deterministic Gated MIP
# ======================================================================


def build_phase1_model(
    data: DAMData,
    dam_commitment: Dict[str, np.ndarray],
    gating_mask: np.ndarray,
    scenarios: Optional[List[np.ndarray]] = None,
    M_p: float = 1e5,
    enforce_lines: bool = True,
) -> Tuple[gp.Model, Dict[str, object]]:
    """
    Build the Phase 1 deterministic gated MIP.

    Objective penalises only *incremental* startups and no-load costs
    inside the gating window, plus power-balance slack.

    Parameters
    ----------
    data : DAMData
        UC problem data.
    dam_commitment : dict
        {"u": (I,T), "v": (I,T), "w": (I,T)} from the market solution.
    gating_mask : np.ndarray
        (I, T) boolean mask from compute_gating_sets.
    scenarios : list of np.ndarray, optional
        CCG scenario deviations [(K,) arrays]. Each is a wind deviation
        vector r_m; dispatch must also be feasible at forecast + r_m.
    M_p : float
        Big-M penalty for power-balance slack.
    enforce_lines : bool
        Whether to enforce transmission line limits.

    Returns
    -------
    model : gp.Model
    vars_dict : dict
    """
    I = data.n_gens
    N = data.n_buses
    L = data.n_lines
    T = data.n_periods
    B = data.n_blocks

    data.validate_shapes()

    Pmin = data.Pmin
    Pmax_2d = data.Pmax_2d()
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
    gen_to_bus = data.gen_to_bus
    dt = data.dt

    u_init = data.u_init
    u_dam = dam_commitment["u"]

    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    K = len(wind_idx)

    gens_at_bus = [[] for _ in range(N)]
    for i in range(I):
        gens_at_bus[int(gen_to_bus[i])].append(i)

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    m = gp.Model("Phase1_Gated_MIP")

    # ------------------------------------------------------------------
    # Decision variables (total commitment, not incremental)
    # ------------------------------------------------------------------
    u = m.addVars(I, T, vtype=GRB.BINARY, name="u")
    v = m.addVars(I, T, vtype=GRB.BINARY, name="v")
    w = m.addVars(I, T, vtype=GRB.BINARY, name="w")
    p_block = m.addVars(I, T, B, vtype=GRB.CONTINUOUS, lb=0.0, name="p_block")
    p = m.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, name="p")
    s_p = m.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, name="s_p")

    # Incremental commitment tracking
    u_prime = m.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, name="u_prime")

    # Fix renewable generators committed
    for i in range(I):
        if data.gen_type[i] in ("WIND", "SOLAR", "HYDRO"):
            for t in range(T):
                u[i, t].lb = 1.0
                u[i, t].ub = 1.0

    # ------------------------------------------------------------------
    # Objective: gated incremental costs only
    # ------------------------------------------------------------------
    obj = gp.LinExpr()
    for i in range(I):
        for t in range(T):
            if gating_mask[i, t]:
                # Incremental startup: v[i,t] includes DAM startups,
                # but we only want to penalise NEW startups.
                # v'[i,t] = v[i,t] - v_dam[i,t] captures this, but since
                # v_dam is a constant, we can write it as:
                #   C_SU * v[i,t] - C_SU * v_dam[i,t]   (constant term drops)
                # Actually we penalise v[i,t] and subtract the constant later,
                # or equivalently just penalise (v[i,t] - v_dam[i,t]).
                # Since v_dam is constant, we just penalise v[i,t]:
                obj.addTerms(C_SU[i], v[i, t])
                obj.addTerms(C_NL[i] * dt[t], u_prime[i, t])

    for t in range(T):
        obj.addTerms(M_p * dt[t], s_p[t])

    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # No-decommitment: u >= u_dam
    for i in range(I):
        for t in range(T):
            m.addConstr(u[i, t] >= float(u_dam[i, t]), name=f"no_decommit_i{i}_t{t}")

    # Incremental definition: u' = u - u_dam
    for i in range(I):
        for t in range(T):
            m.addConstr(
                u_prime[i, t] == u[i, t] - float(u_dam[i, t]),
                name=f"u_prime_def_i{i}_t{t}",
            )

    # Power balance (nominal forecast)
    for t in range(T):
        supply = gp.quicksum(p[i, t] for i in range(I)) + s_p[t]
        demand = float(d[:, t].sum())
        m.addConstr(supply == demand, name=f"balance_t{t}")

    # Block limits and aggregation
    for i in range(I):
        for t in range(T):
            for b in range(B):
                m.addConstr(
                    p_block[i, t, b] <= block_cap[i, b] * u[i, t],
                    name=f"block_cap_i{i}_t{t}_b{b}",
                )
            m.addConstr(
                p[i, t] == gp.quicksum(p_block[i, t, b] for b in range(B)),
                name=f"p_agg_i{i}_t{t}",
            )

    # Min/max output
    for i in range(I):
        for t in range(T):
            m.addConstr(p[i, t] >= Pmin[i] * u[i, t], name=f"p_min_i{i}_t{t}")
            m.addConstr(p[i, t] <= Pmax_2d[i, t] * u[i, t], name=f"p_max_i{i}_t{t}")

    # Ramp constraints (thermal only)
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            m.addConstr(
                p[i, t] - p[i, t - 1]
                <= RU[i] * dt_ramp * u[i, t - 1] + RU[i] * dt_ramp * v[i, t],
                name=f"ramp_up_i{i}_t{t}",
            )
            m.addConstr(
                p[i, t - 1] - p[i, t]
                <= RD[i] * dt_ramp * u[i, t] + RD[i] * dt_ramp * w[i, t],
                name=f"ramp_down_i{i}_t{t}",
            )

    # Logic constraints
    for i in range(I):
        m.addConstr(
            u[i, 0] - u_init[i] == v[i, 0] - w[i, 0],
            name=f"logic_i{i}_t0",
        )
        for t in range(1, T):
            m.addConstr(
                u[i, t] - u[i, t - 1] == v[i, t] - w[i, t],
                name=f"logic_i{i}_t{t}",
            )

    # MUT
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
            n_must = len(must_on)
            if n_must > 0:
                lhs = gp.quicksum(u[i, s] for s in must_on)
                m.addConstr(lhs >= n_must * v[i, t], name=f"mut_i{i}_t{t}")

    # MDT
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
            n_must = len(must_off)
            if n_must > 0:
                lhs = gp.quicksum((1 - u[i, s]) for s in must_off)
                m.addConstr(lhs >= n_must * w[i, t], name=f"mdt_i{i}_t{t}")

    # Line flow constraints
    if not enforce_lines:
        print("  [Phase1] Line flow constraints DISABLED (copper-plate mode)")
    if enforce_lines:
        for l_idx in range(L):
            for t in range(T):
                flow_expr = gp.LinExpr()
                for n in range(N):
                    if PTDF[l_idx, n] == 0.0:
                        continue
                    gen_sum = gp.quicksum(p[i, t] for i in gens_at_bus[n])
                    flow_expr += PTDF[l_idx, n] * (gen_sum - float(d[n, t]))
                m.addConstr(flow_expr <= Fmax[l_idx], name=f"line_max_l{l_idx}_t{t}")
                m.addConstr(flow_expr >= -Fmax[l_idx], name=f"line_min_l{l_idx}_t{t}")

    # ------------------------------------------------------------------
    # CCG scenario constraints
    # ------------------------------------------------------------------
    scen_vars = {}
    if scenarios:
        for m_idx, r_m in enumerate(scenarios):
            sv = add_scenario_constraints(
                m, {"u": u, "v": v, "w": w},
                data, r_m, m_idx, enforce_lines,
            )
            scen_vars[m_idx] = sv
            # Penalise scenario slack — without this, Phase 1 can satisfy
            # any scenario trivially via free slack and never adds commitments
            for t in range(T):
                obj.addTerms(M_p * dt[t], sv["s_p"][t])
        m.setObjective(obj, GRB.MINIMIZE)

    m.Params.OutputFlag = 1
    m.Params.MIPGap = 0.005

    vars_dict = {
        "u": u,
        "v": v,
        "w": w,
        "p_block": p_block,
        "p": p,
        "s_p": s_p,
        "u_prime": u_prime,
        "scen_vars": scen_vars,
    }

    return m, vars_dict


# ======================================================================
# 3. Scenario augmentation for CCG
# ======================================================================


def add_scenario_constraints(
    model: gp.Model,
    commitment_vars: Dict[str, object],
    data: DAMData,
    scenario_r: np.ndarray,
    scenario_idx: int,
    enforce_lines: bool = True,
) -> Dict[str, object]:
    """
    Add dispatch feasibility constraints for a specific wind scenario.

    Creates a copy of dispatch variables that must be feasible under
    wind forecast + scenario_r, sharing the commitment variables (u, v, w).

    Parameters
    ----------
    model : gp.Model
        The Phase 1 model to augment.
    commitment_vars : dict
        {"u", "v", "w"} — shared commitment variables from Phase 1.
    data : DAMData
        UC problem data.
    scenario_r : np.ndarray
        (K,) wind deviation vector.
    scenario_idx : int
        Index for naming variables/constraints.
    enforce_lines : bool
        Whether to enforce line limits for this scenario.

    Returns
    -------
    scen_vars : dict
        {"p": scenario dispatch, "p_block": scenario blocks, "s_p": scenario slack}
    """
    I = data.n_gens
    N = data.n_buses
    L = data.n_lines
    T = data.n_periods
    B = data.n_blocks

    Pmin = data.Pmin
    Pmax_2d = data.Pmax_2d()
    RU = data.RU
    RD = data.RD
    block_cap = data.block_cap
    PTDF = data.PTDF
    Fmax = data.Fmax
    d = data.d
    gen_to_bus = data.gen_to_bus
    dt = data.dt

    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]

    u = commitment_vars["u"]
    v = commitment_vars["v"]
    w = commitment_vars["w"]

    tag = f"s{scenario_idx}"

    gens_at_bus = [[] for _ in range(N)]
    for i in range(I):
        gens_at_bus[int(gen_to_bus[i])].append(i)

    # Scenario dispatch variables
    ps_block = model.addVars(I, T, B, lb=0.0, name=f"p_block_{tag}")
    ps = model.addVars(I, T, lb=0.0, name=f"p_{tag}")
    ss_p = model.addVars(T, lb=0.0, name=f"s_p_{tag}")

    # Adjusted Pmax for wind under scenario
    Pmax_scen = Pmax_2d.copy()
    for k_w, i in enumerate(wind_idx):
        for t in range(T):
            Pmax_scen[i, t] = max(0.0, Pmax_2d[i, t] + scenario_r[k_w])

    # Power balance under scenario
    for t in range(T):
        supply = gp.quicksum(ps[i, t] for i in range(I)) + ss_p[t]
        # Load doesn't change; wind Pmax changes
        demand = float(d[:, t].sum())
        model.addConstr(supply == demand, name=f"bal_{tag}_t{t}")

    # Block limits and aggregation
    for i in range(I):
        for t in range(T):
            cap_it = Pmax_scen[i, t] if is_wind[i] else None
            for b in range(B):
                bc = block_cap[i, b]
                if is_wind[i]:
                    bc = min(bc, Pmax_scen[i, t])
                model.addConstr(
                    ps_block[i, t, b] <= bc * u[i, t],
                    name=f"block_{tag}_i{i}_t{t}_b{b}",
                )
            model.addConstr(
                ps[i, t] == gp.quicksum(ps_block[i, t, b] for b in range(B)),
                name=f"agg_{tag}_i{i}_t{t}",
            )

    # Pmin/Pmax
    for i in range(I):
        for t in range(T):
            model.addConstr(
                ps[i, t] >= Pmin[i] * u[i, t],
                name=f"pmin_{tag}_i{i}_t{t}",
            )
            model.addConstr(
                ps[i, t] <= Pmax_scen[i, t] * u[i, t],
                name=f"pmax_{tag}_i{i}_t{t}",
            )

    # Ramp constraints
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            model.addConstr(
                ps[i, t] - ps[i, t - 1]
                <= RU[i] * dt_ramp * u[i, t - 1] + RU[i] * dt_ramp * v[i, t],
                name=f"rup_{tag}_i{i}_t{t}",
            )
            model.addConstr(
                ps[i, t - 1] - ps[i, t]
                <= RD[i] * dt_ramp * u[i, t] + RD[i] * dt_ramp * w[i, t],
                name=f"rdn_{tag}_i{i}_t{t}",
            )

    # Line flow
    if enforce_lines:
        for l_idx in range(L):
            for t in range(T):
                flow = gp.LinExpr()
                for n in range(N):
                    if PTDF[l_idx, n] == 0.0:
                        continue
                    gen_sum = gp.quicksum(ps[i, t] for i in gens_at_bus[n])
                    flow += PTDF[l_idx, n] * (gen_sum - float(d[n, t]))
                model.addConstr(flow <= Fmax[l_idx], name=f"lmax_{tag}_l{l_idx}_t{t}")
                model.addConstr(flow >= -Fmax[l_idx], name=f"lmin_{tag}_l{l_idx}_t{t}")

    return {"p": ps, "p_block": ps_block, "s_p": ss_p}


# ======================================================================
# 4. Phase 2: Robust Feasibility SOCP
# ======================================================================


def build_phase2_model(
    data: DAMData,
    commitment: Dict[str, np.ndarray],
    Sigma: np.ndarray,
    rho: Union[float, np.ndarray],
    sqrt_Sigma: Optional[np.ndarray] = None,
    enforce_lines: bool = True,
    rho_lines_frac: Optional[float] = None,
    robust_mask: Optional[np.ndarray] = None,
) -> Tuple[gp.Model, Dict[str, object]]:
    """
    Build the Phase 2 robust feasibility verification SOCP.

    All commitment variables are fixed from Phase 1. Only continuous
    dispatch variables (p0, Z) and slack remain.

    Parameters
    ----------
    data : DAMData
        UC problem data.
    commitment : dict
        {"u": (I,T), "v": (I,T), "w": (I,T)} — fixed binary values.
    Sigma : np.ndarray
        Covariance matrix. (K,K) or (T,K,K).
    rho : float or np.ndarray
        Ellipsoid radius. Scalar or (T,).
    sqrt_Sigma : np.ndarray or None
        Pre-computed Cholesky factor. (K,K) or (T,K,K).
    enforce_lines : bool
        Whether to enforce line flow limits.
    rho_lines_frac : float or None
        Scale rho for line constraints.
    robust_mask : np.ndarray or None
        (T,) bool mask. True = robust period. None = all robust.

    Returns
    -------
    model : gp.Model
    vars_dict : dict
    """
    I = data.n_gens
    N = data.n_buses
    L = data.n_lines
    T = data.n_periods
    B = data.n_blocks

    data.validate_shapes()

    Pmin = data.Pmin
    Pmax_2d = data.Pmax_2d()
    RU = data.RU
    RD = data.RD
    block_cap = data.block_cap
    PTDF = data.PTDF
    Fmax = data.Fmax
    d = data.d
    gen_to_bus = data.gen_to_bus
    dt = data.dt

    u_fix = commitment["u"]
    v_fix = commitment["v"]
    w_fix = commitment["w"]

    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    n_wind = len(wind_idx)
    K = n_wind

    is_z_eligible = np.array(
        [gt.upper() in ("THERMAL", "WIND") for gt in data.gen_type]
    )
    z_elig = [i for i in range(I) if is_z_eligible[i]]
    z_elig_set = set(z_elig)
    thermal_idx = [i for i in range(I) if is_z_eligible[i] and not is_wind[i]]

    if robust_mask is None:
        robust_mask = np.ones(T, dtype=bool)
    else:
        robust_mask = np.asarray(robust_mask, dtype=bool)

    # Handle time-varying vs static uncertainty
    time_varying = (Sigma.ndim == 3) or (isinstance(rho, np.ndarray) and rho.ndim == 1)

    if time_varying:
        if Sigma.ndim == 2:
            Sigma = np.tile(Sigma[np.newaxis, :, :], (T, 1, 1))
        if isinstance(rho, (int, float)):
            rho = np.full(T, rho)
        if rho_lines_frac is not None:
            rho_lines = rho_lines_frac * rho
        else:
            rho_lines = rho
        if sqrt_Sigma is None:
            sqrt_Sigma = np.zeros_like(Sigma)
            for t in range(T):
                sqrt_Sigma[t] = np.linalg.cholesky(Sigma[t])
    else:
        if rho_lines_frac is not None:
            rho_lines = rho_lines_frac * rho
        else:
            rho_lines = rho
        if sqrt_Sigma is None:
            sqrt_Sigma = np.linalg.cholesky(Sigma)

    gens_at_bus = [[] for _ in range(N)]
    for i in range(I):
        gens_at_bus[int(gen_to_bus[i])].append(i)

    # ------------------------------------------------------------------
    m = gp.Model("Phase2_Robust_SOCP")

    # Nominal dispatch
    p0 = m.addVars(I, T, lb=0.0, name="p0")
    p0_block = m.addVars(I, T, B, lb=0.0, name="p0_block")

    # Power balance slack (objective)
    s = m.addVars(T, lb=0.0, name="s")

    # LDR coefficients (only for z-eligible generators in robust periods)
    Z = m.addVars(
        [(i, t, k) for i in z_elig for t in range(T) if robust_mask[t] for k in range(K)],
        lb=-GRB.INFINITY, name="Z",
    )

    # SOC auxiliaries for thermal generators
    z_gen = m.addVars(
        [(i, t) for i in thermal_idx for t in range(T) if robust_mask[t]],
        lb=0.0, name="z_gen",
    )
    y_gen = m.addVars(
        [(i, t, k) for i in thermal_idx for t in range(T) if robust_mask[t] for k in range(K)],
        lb=-GRB.INFINITY, name="y_gen",
    )

    # Objective: minimise total slack
    obj = gp.LinExpr()
    for t in range(T):
        obj.addTerms(1.0, s[t])
    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Capacity constraints with SOC (same pattern as aruc_model.py)
    # ------------------------------------------------------------------
    for i in range(I):
        for t in range(T):
            u_it = float(u_fix[i, t])

            if not robust_mask[t] or u_it < 0.5:
                # Nominal constraints only
                m.addConstr(p0[i, t] <= Pmax_2d[i, t] * u_it, name=f"p0max_i{i}_t{t}")
                m.addConstr(p0[i, t] >= Pmin[i] * u_it, name=f"p0min_i{i}_t{t}")
                for b in range(B):
                    m.addConstr(
                        p0_block[i, t, b] <= block_cap[i, b] * u_it,
                        name=f"blk_i{i}_t{t}_b{b}",
                    )
                m.addConstr(
                    p0[i, t] <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                    name=f"agg_i{i}_t{t}",
                )
                continue

            # --- Robust period ---
            sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
            rho_t = float(rho[t]) if time_varying else float(rho)

            if is_wind[i]:
                # Wind: nominal block constraints; SOC via wind availability
                for b in range(B):
                    m.addConstr(
                        p0_block[i, t, b] <= block_cap[i, b] * u_it,
                        name=f"blk_i{i}_t{t}_b{b}",
                    )
                m.addConstr(
                    p0[i, t] <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                    name=f"agg_i{i}_t{t}",
                )
                continue

            if not is_z_eligible[i]:
                # Solar/hydro: nominal only
                m.addConstr(p0[i, t] <= Pmax_2d[i, t] * u_it, name=f"p0max_i{i}_t{t}")
                m.addConstr(p0[i, t] >= Pmin[i] * u_it, name=f"p0min_i{i}_t{t}")
                for b in range(B):
                    m.addConstr(
                        p0_block[i, t, b] <= block_cap[i, b] * u_it,
                        name=f"blk_i{i}_t{t}_b{b}",
                    )
                m.addConstr(
                    p0[i, t] <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                    name=f"agg_i{i}_t{t}",
                )
                continue

            # --- Thermal, robust, committed ---
            # y_gen definition
            for k_idx in range(K):
                expr = gp.LinExpr()
                for j_idx in range(K):
                    coef = float(sqrt_Sigma_t[k_idx, j_idx])
                    if abs(coef) < 1e-12:
                        continue
                    expr += coef * Z[i, t, j_idx]
                m.addConstr(y_gen[i, t, k_idx] == expr, name=f"yg_i{i}_t{t}_k{k_idx}")

            # SOC: z_gen >= ||y_gen||
            m.addConstr(
                z_gen[i, t] * z_gen[i, t]
                >= gp.quicksum(y_gen[i, t, k] * y_gen[i, t, k] for k in range(K)),
                name=f"soc_gen_i{i}_t{t}",
            )

            # Block caps
            for b in range(B):
                m.addConstr(
                    p0_block[i, t, b] <= block_cap[i, b] * u_it,
                    name=f"blk_i{i}_t{t}_b{b}",
                )

            # Robust aggregation
            m.addConstr(
                p0[i, t] + rho_t * z_gen[i, t]
                <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                name=f"agg_rob_i{i}_t{t}",
            )

            # Robust Pmax
            m.addConstr(
                p0[i, t] + rho_t * z_gen[i, t] <= Pmax_2d[i, t] * u_it,
                name=f"pmax_rob_i{i}_t{t}",
            )

            # Robust Pmin (Gurobi expr must be on LHS to avoid numpy __le__)
            m.addConstr(
                p0[i, t] - rho_t * z_gen[i, t] >= float(Pmin[i]) * u_it,
                name=f"pmin_rob_i{i}_t{t}",
            )

    # ------------------------------------------------------------------
    # Ramp constraints (nominal, thermal only)
    # ------------------------------------------------------------------
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            u_it = float(u_fix[i, t])
            u_itm1 = float(u_fix[i, t - 1])
            v_it = float(v_fix[i, t])
            w_it = float(w_fix[i, t])
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            m.addConstr(
                p0[i, t] - p0[i, t - 1]
                <= RU[i] * dt_ramp * u_itm1 + RU[i] * dt_ramp * v_it,
                name=f"rup_i{i}_t{t}",
            )
            m.addConstr(
                p0[i, t - 1] - p0[i, t]
                <= RD[i] * dt_ramp * u_it + RD[i] * dt_ramp * w_it,
                name=f"rdn_i{i}_t{t}",
            )

    # ------------------------------------------------------------------
    # Power balance (nominal + recourse balance)
    # ------------------------------------------------------------------
    for t in range(T):
        total_load = float(d[:, t].sum())
        m.addConstr(
            gp.quicksum(p0[i, t] for i in range(I)) + s[t] == total_load,
            name=f"bal_t{t}",
        )
        if robust_mask[t]:
            for k in range(K):
                m.addConstr(
                    gp.quicksum(Z[i, t, k] for i in z_elig) == 0.0,
                    name=f"bal_Z_t{t}_k{k}",
                )

    # ------------------------------------------------------------------
    # Line flow SOC (robust periods)
    # ------------------------------------------------------------------
    z_line = m.addVars(
        [(l, t) for l in range(L) for t in range(T) if robust_mask[t]],
        lb=0.0, name="z_line",
    )
    y_line = m.addVars(
        [(l, t, k) for l in range(L) for t in range(T) if robust_mask[t] for k in range(K)],
        lb=-GRB.INFINITY, name="y_line",
    )

    if not enforce_lines:
        print("  [Phase2] Line flow constraints DISABLED")
    if enforce_lines:
        for l_idx in range(L):
            for t in range(T):
                flow_nom = gp.LinExpr()
                for n in range(N):
                    if abs(PTDF[l_idx, n]) < 1e-10:
                        continue
                    gen_sum = gp.quicksum(p0[i, t] for i in gens_at_bus[n])
                    flow_nom += PTDF[l_idx, n] * (gen_sum - float(d[n, t]))

                if not robust_mask[t]:
                    m.addConstr(flow_nom <= Fmax[l_idx], name=f"lmax_i{l_idx}_t{t}")
                    m.addConstr(flow_nom >= -Fmax[l_idx], name=f"lmin_i{l_idx}_t{t}")
                    continue

                sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
                rho_lines_t = float(rho_lines[t]) if time_varying else float(rho_lines)

                a_expr = [gp.LinExpr() for _ in range(K)]
                for i in range(I):
                    if i not in z_elig_set:
                        continue
                    n = int(gen_to_bus[i])
                    if abs(PTDF[l_idx, n]) < 1e-10:
                        continue
                    for k in range(K):
                        a_expr[k] += PTDF[l_idx, n] * Z[i, t, k]

                for i_k in range(K):
                    expr = gp.LinExpr()
                    for j_k in range(K):
                        coef = sqrt_Sigma_t[i_k, j_k]
                        if abs(coef) < 1e-10:
                            continue
                        expr += coef * a_expr[j_k]
                    m.addConstr(y_line[l_idx, t, i_k] == expr, name=f"yl_l{l_idx}_t{t}_k{i_k}")

                m.addConstr(
                    z_line[l_idx, t] * z_line[l_idx, t]
                    >= gp.quicksum(y_line[l_idx, t, k] * y_line[l_idx, t, k] for k in range(K)),
                    name=f"soc_line_l{l_idx}_t{t}",
                )

                m.addConstr(
                    flow_nom + rho_lines_t * z_line[l_idx, t] <= Fmax[l_idx],
                    name=f"lmax_rob_l{l_idx}_t{t}",
                )
                m.addConstr(
                    -flow_nom + rho_lines_t * z_line[l_idx, t] <= Fmax[l_idx],
                    name=f"lmin_rob_l{l_idx}_t{t}",
                )

    # ------------------------------------------------------------------
    # Wind availability SOC (robust periods)
    # ------------------------------------------------------------------
    z_wind = m.addVars(
        [(kw, t) for kw in range(n_wind) for t in range(T) if robust_mask[t]],
        lb=0.0, name="z_wind",
    )
    y_wind = m.addVars(
        [(kw, t, k) for kw in range(n_wind) for t in range(T) if robust_mask[t] for k in range(K)],
        lb=-GRB.INFINITY, name="y_wind",
    )

    for k_wind, i in enumerate(wind_idx):
        for t in range(T):
            if not robust_mask[t]:
                continue
            u_it = float(u_fix[i, t])
            if u_it < 0.5:
                continue

            sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
            rho_t = float(rho[t]) if time_varying else float(rho)

            Pbar_it = Pmax_2d[i, t]
            alpha_expr = p0[i, t] - Pbar_it

            a_expr = []
            for k in range(K):
                expr = gp.LinExpr()
                expr += Z[i, t, k]
                if k == k_wind:
                    expr += -1.0
                a_expr.append(expr)

            for q in range(K):
                expr = gp.LinExpr()
                for j in range(K):
                    coef = float(sqrt_Sigma_t[q, j])
                    if abs(coef) < 1e-12:
                        continue
                    expr += coef * a_expr[j]
                m.addConstr(y_wind[k_wind, t, q] == expr, name=f"yw_w{k_wind}_t{t}_k{q}")

            m.addConstr(
                z_wind[k_wind, t] * z_wind[k_wind, t]
                >= gp.quicksum(y_wind[k_wind, t, k] * y_wind[k_wind, t, k] for k in range(K)),
                name=f"soc_wind_w{k_wind}_t{t}",
            )

            m.addConstr(
                alpha_expr + rho_t * z_wind[k_wind, t] <= 0.0,
                name=f"wind_rob_w{k_wind}_t{t}",
            )

    # Gurobi params — use homogeneous barrier for numerical stability
    m.Params.OutputFlag = 1
    m.Params.NumericFocus = 2
    m.Params.BarHomogeneous = 1
    m.Params.ScaleFlag = 2

    vars_dict = {
        "p0": p0,
        "p0_block": p0_block,
        "Z": Z,
        "s": s,
        "z_gen": z_gen,
        "y_gen": y_gen,
        "z_line": z_line,
        "y_line": y_line,
        "z_wind": z_wind,
        "y_wind": y_wind,
    }

    return m, vars_dict


# ======================================================================
# 5. Worst-case scenario extraction
# ======================================================================


def extract_worst_case_scenario(
    phase2_model: gp.Model,
    phase2_vars: Dict[str, object],
    data: DAMData,
    Sigma: np.ndarray,
    rho: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Extract the worst-case wind deviation from Phase 2 solution.

    Uses the simple fallback: the direction that minimises total
    renewable output (worst case for system balance).

    For each robust period t with slack s_t > 0, compute the worst-case
    direction from the aggregate Z balance constraint structure.

    Parameters
    ----------
    phase2_model : gp.Model
        Solved Phase 2 model.
    phase2_vars : dict
        Phase 2 variables.
    data : DAMData
        UC problem data.
    Sigma : np.ndarray
        Covariance matrix. (K,K) or (T,K,K).
    rho : float or np.ndarray
        Ellipsoid radius.

    Returns
    -------
    r_star : np.ndarray
        (K,) worst-case deviation vector.
    """
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    K = int(is_wind.sum())
    T = data.n_periods

    time_varying = Sigma.ndim == 3

    # Find the period with maximum slack (if solution exists)
    t_worst = 0
    if phase2_model.SolCount > 0:
        s_vals = np.array([phase2_vars["s"][t].X for t in range(T)])
        t_worst = int(np.argmax(s_vals))

    if time_varying:
        Sigma_t = Sigma[t_worst]
        rho_t = float(rho[t_worst]) if isinstance(rho, np.ndarray) else float(rho)
    else:
        Sigma_t = Sigma
        rho_t = float(rho)

    # Simple fallback: worst case = direction that maximally reduces
    # total wind output. Direction: -Sigma^{1/2} @ ones / ||...||
    try:
        L_chol = np.linalg.cholesky(Sigma_t)
    except np.linalg.LinAlgError:
        L_chol = np.linalg.cholesky(Sigma_t + 1e-6 * np.eye(K))

    direction = -L_chol @ np.ones(K)
    norm = np.linalg.norm(L_chol.T @ direction)
    if norm < 1e-12:
        # Degenerate: use first principal component
        direction = -L_chol[:, 0]
        norm = np.linalg.norm(L_chol.T @ direction)

    if norm > 1e-12:
        r_star = rho_t * Sigma_t @ direction / norm
    else:
        r_star = np.zeros(K)

    return r_star


# ======================================================================
# 6. CCG orchestration
# ======================================================================


def solve_ruc_ccg(
    data: DAMData,
    dam_commitment: Dict[str, np.ndarray],
    gating_mask: np.ndarray,
    Sigma: np.ndarray,
    rho: Union[float, np.ndarray],
    sqrt_Sigma: Optional[np.ndarray] = None,
    enforce_lines: bool = True,
    rho_lines_frac: Optional[float] = None,
    robust_mask: Optional[np.ndarray] = None,
    max_iterations: int = 10,
    gap_tolerance: float = 1.0,
    M_p: float = 1e5,
) -> Dict[str, object]:
    """
    Solve the LD-RUC via column-and-constraint generation.

    Phase 1 (deterministic gated MIP) → Phase 2 (robust SOCP) → iterate.

    Parameters
    ----------
    data : DAMData
        UC problem data.
    dam_commitment : dict
        {"u": (I,T), "v": (I,T), "w": (I,T)} market commitment.
    gating_mask : np.ndarray
        (I, T) boolean gating mask.
    Sigma, rho, sqrt_Sigma :
        Uncertainty set parameters.
    enforce_lines : bool
        Enforce line limits.
    rho_lines_frac : float or None
        Scale rho for lines.
    robust_mask : np.ndarray or None
        Which periods are robust.
    max_iterations : int
        Maximum CCG iterations.
    gap_tolerance : float
        Stop when total slack <= this (MW).
    M_p : float
        Big-M penalty.

    Returns
    -------
    results : dict
        Full solution including commitment, dispatch, gap history, etc.
    """
    I = data.n_gens
    T = data.n_periods

    scenarios: List[np.ndarray] = []
    gap_history: List[float] = []
    timings: Dict[str, List[float]] = {"phase1": [], "phase2": []}

    print("\n" + "=" * 70)
    print("LD-RUC CCG LOOP")
    print("=" * 70)

    for iteration in range(max_iterations):
        print(f"\n--- CCG Iteration {iteration + 1}/{max_iterations} ---")

        # --- Phase 1 ---
        t0 = _time.time()
        print(f"  Phase 1: Building model ({len(scenarios)} scenarios)...")
        p1_model, p1_vars = build_phase1_model(
            data, dam_commitment, gating_mask,
            scenarios=scenarios if scenarios else None,
            M_p=M_p,
            enforce_lines=enforce_lines,
        )
        print("  Phase 1: Solving...")
        p1_model.optimize()
        t_p1 = _time.time() - t0
        timings["phase1"].append(t_p1)

        if p1_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            print(f"  WARNING: Phase 1 status {p1_model.Status}")
            if p1_model.SolCount == 0:
                raise RuntimeError("Phase 1 infeasible — no solution found.")

        p1_obj = p1_model.ObjVal
        print(f"  Phase 1 objective: {p1_obj:,.2f} (solve time: {t_p1:.1f}s)")

        # Extract commitment
        u_hat = np.zeros((I, T))
        v_hat = np.zeros((I, T))
        w_hat = np.zeros((I, T))
        for i in range(I):
            for t in range(T):
                u_hat[i, t] = round(p1_vars["u"][i, t].X)
                v_hat[i, t] = round(p1_vars["v"][i, t].X)
                w_hat[i, t] = round(p1_vars["w"][i, t].X)

        commitment_hat = {"u": u_hat, "v": v_hat, "w": w_hat}

        # Count incremental commitments
        u_prime_sum = 0
        for i in range(I):
            for t in range(T):
                u_prime_sum += max(0, p1_vars["u_prime"][i, t].X)
        print(f"  Incremental unit-hours: {u_prime_sum:.0f}")

        # --- Phase 2 ---
        t0 = _time.time()
        print("  Phase 2: Building SOCP...")
        p2_model, p2_vars = build_phase2_model(
            data, commitment_hat, Sigma, rho,
            sqrt_Sigma=sqrt_Sigma,
            enforce_lines=enforce_lines,
            rho_lines_frac=rho_lines_frac,
            robust_mask=robust_mask,
        )
        print("  Phase 2: Solving...")
        p2_model.optimize()
        t_p2 = _time.time() - t0
        timings["phase2"].append(t_p2)

        if p2_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            print(f"  WARNING: Phase 2 status {p2_model.Status}")
            if p2_model.SolCount == 0:
                print("  Phase 2 infeasible — commitment may be insufficient.")
                gap_history.append(float("inf"))
                # Try to add worst-case anyway
                r_star = extract_worst_case_scenario(
                    p2_model, p2_vars, data, Sigma, rho,
                )
                scenarios.append(r_star)
                continue

        total_slack = sum(p2_vars["s"][t].X for t in range(T))
        gap_history.append(total_slack)
        print(f"  Phase 2 total slack: {total_slack:.4f} MW (solve time: {t_p2:.1f}s)")

        if total_slack <= gap_tolerance:
            print(f"\n  ROBUST SUFFICIENT (slack {total_slack:.4f} <= tol {gap_tolerance})")
            break

        # Extract worst-case scenario and add to Phase 1
        r_star = extract_worst_case_scenario(
            p2_model, p2_vars, data, Sigma, rho,
        )
        scenarios.append(r_star)
        print(f"  Added scenario {len(scenarios)}: ||r*|| = {np.linalg.norm(r_star):.2f}")

    else:
        print(f"\n  CCG reached max iterations ({max_iterations})")

    robust_sufficient = len(gap_history) > 0 and gap_history[-1] <= gap_tolerance

    # Extract Phase 2 solution if available
    p2_solution = {}
    if p2_model.SolCount > 0:
        p0_arr = np.zeros((I, T))
        for i in range(I):
            for t in range(T):
                p0_arr[i, t] = p2_vars["p0"][i, t].X
        p2_solution["p0"] = p0_arr
        p2_solution["obj"] = p2_model.ObjVal

        # Extract Z
        is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
        K = int(is_wind.sum())
        is_z_elig = np.array(
            [gt.upper() in ("THERMAL", "WIND") for gt in data.gen_type]
        )
        z_elig = [i for i in range(I) if is_z_elig[i]]
        rm = robust_mask if robust_mask is not None else np.ones(T, dtype=bool)

        Z_arr = np.zeros((I, T, K))
        for i in z_elig:
            for t in range(T):
                if rm[t]:
                    for k in range(K):
                        if (i, t, k) in p2_vars["Z"]:
                            Z_arr[i, t, k] = p2_vars["Z"][i, t, k].X
        p2_solution["Z"] = Z_arr

    # Phase 1 solution
    p1_solution = {}
    if p1_model.SolCount > 0:
        p_arr = np.zeros((I, T))
        for i in range(I):
            for t in range(T):
                p_arr[i, t] = p1_vars["p"][i, t].X
        p1_solution["p"] = p_arr
        p1_solution["obj"] = p1_model.ObjVal

    print(f"\n  CCG Summary: {len(gap_history)} iterations, "
          f"final gap={gap_history[-1]:.4f} MW, "
          f"robust={'YES' if robust_sufficient else 'NO'}")

    return {
        "commitment": commitment_hat,
        "phase1_results": p1_solution,
        "phase2_results": p2_solution,
        "ccg_iterations": len(gap_history),
        "gap_history": gap_history,
        "scenarios": scenarios,
        "robust_sufficient": robust_sufficient,
        "timings": timings,
        "phase1_model": p1_model,
        "phase2_model": p2_model,
        "phase1_vars": p1_vars,
        "phase2_vars": p2_vars,
    }
