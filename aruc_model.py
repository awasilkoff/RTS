from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from models import DAMData


def build_aruc_ldr_model(
    data: DAMData,
    Sigma: np.ndarray,
    rho: float,
    M_p: float = 1e5,
    model_name: str = "ARUC_LDR",
) -> Tuple[gp.Model, Dict[str, object]]:
    """
    Adaptive robust UC with linear decision rules:

        p(i,t)(r) = p0(i,t) + Z(i,t)^T r

    with r in ellipsoidal uncertainty set:
        { r : r^T Sigma r <= rho^2 }

    This shares the same DAMData as the deterministic DAM model, but:
    - p is represented as an affine function of r.
    - Constraints are enforced for all r in the ellipsoid (robust counterpart).
    """

    I = data.n_gens
    N = data.n_buses
    L = data.n_lines
    T = data.n_periods
    B = data.n_blocks

    data.validate_shapes()

    # --- data shortcuts ---
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

    u_init = data.u_init
    init_up = data.init_up_time
    init_down = data.init_down_time

    # Identify wind units if you've tagged them in data.gen_type
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    n_wind = len(wind_idx)

    # Dimension of uncertainty vector r:
    # simplest: one r per wind generator (aggregate over time),
    # or one per wind gen and time. For now, assume K = n_wind.
    K = n_wind
    assert Sigma.shape == (K, K), f"Expected Sigma shape ({K}, {K}), got {Sigma.shape}"
    sqrt_Sigma = np.linalg.cholesky(Sigma)  # K x K

    # Map bus -> list of generators
    gens_at_bus = [[] for _ in range(N)]
    for i in range(I):
        n = int(gen_to_bus[i])
        gens_at_bus[n].append(i)

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    m = gp.Model(model_name)

    # ------------------------------------------------------------------
    # First-stage decision variables
    # ------------------------------------------------------------------
    u = m.addVars(I, T, vtype=GRB.BINARY, name="u")
    v = m.addVars(I, T, vtype=GRB.BINARY, name="v")
    w = m.addVars(I, T, vtype=GRB.BINARY, name="w")

    # Nominal dispatch part p0[i,t]
    p0 = m.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, name="p0")

    # Optional: keep block structure only for p0 (not the recourse part)
    p0_block = m.addVars(I, T, B, vtype=GRB.CONTINUOUS, lb=0.0, name="p0_block")

    # Slack on power balance (still first-stage)
    s_p = m.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, name="s_p")

    # ------------------------------------------------------------------
    # Second-stage LDR variables: Z[i,t,k]
    # ------------------------------------------------------------------
    # Z encodes sensitivity of p(i,t) to each component of r:
    #   p(i,t)(r) = p0[i,t] + sum_k Z[i,t,k] * r_k
    Z = m.addVars(I, T, K, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Z")

    # ------------------------------------------------------------------
    # Objective: expected or worst-case cost?
    # ------------------------------------------------------------------
    # For a first version, you can just use the nominal cost at p0 and u,v,w,
    # ignoring Z in the objective (i.e. be conservative only in constraints).
    obj = gp.LinExpr()

    for i in range(I):
        for t in range(T):
            obj.addTerms(C_NL[i], u[i, t])
            obj.addTerms(C_SU[i], v[i, t])
            obj.addTerms(C_SD[i], w[i, t])
            for b in range(B):
                obj.addTerms(block_cost[i, b], p0_block[i, t, b])

    for t in range(T):
        obj.addTerms(M_p, s_p[t])

    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Constraints – UC logic, min/max, ramps stay on p0
    # ------------------------------------------------------------------

    # Block limits on p0
    for i in range(I):
        for t in range(T):
            for b in range(B):
                m.addConstr(
                    p0_block[i, t, b] <= block_cap[i, b] * u[i, t],
                    name=f"p0_block_cap_i{i}_t{t}_b{b}",
                )
            m.addConstr(
                p0[i, t] == gp.quicksum(p0_block[i, t, b] for b in range(B)),
                name=f"p0_agg_i{i}_t{t}",
            )

    # Min/max on nominal part (you'll later add robust margins using Z)
    for i in range(I):
        for t in range(T):
            m.addConstr(
                p0[i, t] >= Pmin[i] * u[i, t],
                name=f"p0_min_i{i}_t{t}",
            )
            m.addConstr(
                p0[i, t] <= Pmax_2d[i, t] * u[i, t],
                name=f"p0_max_i{i}_t{t}",
            )

    # Ramps on p0 (again, later robustify using Z if needed)
    for i in range(I):
        for t in range(1, T):
            m.addConstr(
                p0[i, t] - p0[i, t - 1] <= RU[i] * u[i, t - 1] + RU[i] * v[i, t],
                name=f"ramp_up_i{i}_t{t}",
            )
            m.addConstr(
                p0[i, t - 1] - p0[i, t] <= RD[i] * u[i, t] + RD[i] * w[i, t],
                name=f"ramp_down_i{i}_t{t}",
            )

    # Commitment logic u,v,w exactly as in DAM (you can keep or relax init)
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

    # ------------------------------------------------------------------
    # MUT / MDT Constraints
    # ------------------------------------------------------------------
    for i in range(I):
        mut_val = int(MUT[i])
        mdt_val = int(MDT[i])

        # Minimum up time constraints
        for t in range(T):
            end_period = min(T, t + mut_val)
            lhs = gp.quicksum(u[i, s] for s in range(t, end_period))
            m.addConstr(lhs >= mut_val * v[i, t], name=f"mut_i{i}_t{t}")

        # Minimum down time constraints
        for t in range(T):
            end_period = min(T, t + mdt_val)
            lhs = gp.quicksum((1 - u[i, s]) for s in range(t, end_period))
            m.addConstr(lhs >= mdt_val * w[i, t], name=f"mdt_i{i}_t{t}")

        # # Initial conditions for MUT
        # if u_init[i] > 0.5 and init_up[i] < mut_val:
        #     remaining = int(mut_val - init_up[i])
        #     for t in range(min(remaining, T)):
        #         m.addConstr(u[i, t] == 1, name=f"init_mut_i{i}_t{t}")
        #
        # # Initial conditions for MDT
        # if u_init[i] < 0.5 and init_down[i] < mdt_val:
        #     remaining = int(mdt_val - init_down[i])
        #     for t in range(min(remaining, T)):
        #         m.addConstr(u[i, t] == 0, name=f"init_mdt_i{i}_t{t}")

    # ------------------------------------------------------------------
    # Robust constraints
    # ------------------------------------------------------------------

    # 1) Power balance must hold for all r in ellipsoid
    for t in range(T):
        total_load_t = float(d[:, t].sum())
        # nominal power balance:
        m.addConstr(
            gp.quicksum(p0[i, t] for i in range(I)) + s_p[t] == total_load_t,
            name=f"bal_nom_t{t}",
        )
        # zero-net-response condition on Z (keeps balance for all r)
        for k in range(K):
            m.addConstr(
                gp.quicksum(Z[i, t, k] for i in range(I)) == 0.0,
                name=f"bal_Z_t{t}_k{k}",
            )

    # 2) Line flows robust
    z_line = m.addVars(L, T, lb=0.0, name="z_line")
    # Auxiliary variables for SOC constraints
    y_line = m.addVars(L, T, K, lb=-GRB.INFINITY, name="y_line")

    for l in range(L):
        for t in range(T):
            # 1) flow_nom_expr
            flow_nom = gp.LinExpr()
            for n in range(N):
                if abs(PTDF[l, n]) < 1e-10:
                    continue
                gen_sum = gp.quicksum(p0[i, t] for i in gens_at_bus[n])
                flow_nom += PTDF[l, n] * (gen_sum - float(d[n, t]))

            # 2) a_expr[k] = d(flow) / d r_k
            a_expr = [gp.LinExpr() for _ in range(K)]
            for i in range(I):
                n = int(gen_to_bus[i])
                if abs(PTDF[l, n]) < 1e-10:
                    continue
                for k in range(K):
                    a_expr[k] += PTDF[l, n] * Z[i, t, k]

            # 3) Define auxiliary variables: y_line[l,t,k] = (sqrt_Sigma @ a_expr)[k]
            for i_k in range(K):
                expr = gp.LinExpr()
                for j_k in range(K):
                    coef = sqrt_Sigma[i_k, j_k]
                    if abs(coef) < 1e-10:
                        continue
                    expr += coef * a_expr[j_k]
                m.addConstr(
                    y_line[l, t, i_k] == expr, name=f"y_line_def_l{l}_t{t}_k{i_k}"
                )

            # 4) SOC: z_line[l,t] >= ||y_line[l,t,:]||
            # Use addConstr with cone constraint: (z, y) in quadratic cone
            m.addConstr(
                z_line[l, t] * z_line[l, t]
                >= gp.quicksum(y_line[l, t, k] * y_line[l, t, k] for k in range(K)),
                name=f"soc_line_l{l}_t{t}",
            )

            # 5) Robust line limits
            m.addConstr(
                flow_nom + rho * z_line[l, t] <= Fmax[l],
                name=f"line_max_rob_l{l}_t{t}",
            )
            m.addConstr(
                -flow_nom + rho * z_line[l, t] <= Fmax[l],
                name=f"line_min_rob_l{l}_t{t}",
            )

    # 3) Wind availability
    z_wind = m.addVars(n_wind, T, lb=0.0, name="z_wind")
    # Auxiliary variables for SOC constraints
    y_wind = m.addVars(n_wind, T, K, lb=-GRB.INFINITY, name="y_wind")

    for k_wind, i in enumerate(wind_idx):
        for t in range(T):
            # Deterministic forecast for this wind unit (Pbar)
            Pbar_it = Pmax_2d[i, t]  # interpret Pmax_2d for wind as forecast

            # alpha = p0(i,t) - Pbar(i,t)
            alpha_expr = p0[i, t] - Pbar_it

            # a_expr[j] = Z[i,t,j] - 1{j == k_wind}
            a_expr = []
            for k in range(K):
                expr = gp.LinExpr()
                expr += Z[i, t, k]
                if k == k_wind:
                    expr += -1.0
                a_expr.append(expr)

            # Define auxiliary variables: y_wind[k_wind,t,q] = (sqrt_Sigma @ a_expr)[q]
            for q in range(K):
                expr = gp.LinExpr()
                for j in range(K):
                    coef = float(sqrt_Sigma[q, j])
                    if abs(coef) < 1e-12:
                        continue
                    expr += coef * a_expr[j]
                m.addConstr(
                    y_wind[k_wind, t, q] == expr, name=f"y_wind_def_w{k_wind}_t{t}_k{q}"
                )

            # SOC: z_wind[k_wind, t] >= || y_wind[k_wind,t,:] ||_2
            # Use addConstr with cone constraint: (z, y) in quadratic cone
            m.addConstr(
                z_wind[k_wind, t] * z_wind[k_wind, t]
                >= gp.quicksum(
                    y_wind[k_wind, t, k] * y_wind[k_wind, t, k] for k in range(K)
                ),
                name=f"soc_wind_i{i}_t{t}",
            )

            # Robust inequality: alpha + rho * z_wind <= 0
            m.addConstr(
                alpha_expr + rho * z_wind[k_wind, t] <= 0.0,
                name=f"wind_max_rob_i{i}_t{t}",
            )

    # Basic Gurobi params
    m.Params.OutputFlag = 1

    vars_dict: Dict[str, object] = {
        "u": u,
        "v": v,
        "w": w,
        "p0": p0,
        "p0_block": p0_block,
        "Z": Z,
        "s_p": s_p,
        "z_line": z_line,
        "y_line": y_line,
        "z_wind": z_wind,
        "y_wind": y_wind,
    }

    return m, vars_dict


if __name__ == "__main__":
    from models import DAMData

    I, N, L, T, B = 2, 3, 1, 4, 2

    dummy_data = DAMData(
        gen_ids=[f"G{i}" for i in range(I)],
        bus_ids=[f"B{n}" for n in range(N)],
        line_ids=[f"L{l}" for l in range(L)],
        time=list(range(T)),
        gen_type=["THERMAL", "WIND"],
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
    model, vars_ = build_aruc_ldr_model(dummy_data, np.array([[1.0]]), 2.0, M_p=1e3)
    print("Model built successfully. Variable counts:")
    print("u size:", len(vars_["u"]))
    print("p0 size:", len(vars_["p0"]))  # Fixed: was vars_["p"]
    print("Z size:", len(vars_["Z"]))
    print("z_line size:", len(vars_["z_line"]))
    print("z_wind size:", len(vars_["z_wind"]))
    model.optimize()
