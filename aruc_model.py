from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from models import DAMData


def align_uncertainty_to_aruc(
    horizon: "HorizonUncertaintySet",
    data: DAMData,
    provider_wind_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """
    Align provider uncertainty sets to ARUC wind generator ordering.

    The provider may have wind generators in a different order than the ARUC
    model. This function permutes the covariance matrices to match ARUC's
    wind generator ordering.

    Parameters
    ----------
    horizon : HorizonUncertaintySet
        Horizon uncertainty set from the provider
    data : DAMData
        ARUC data object containing generator information
    provider_wind_ids : List[str]
        Wind generator IDs from the provider (in provider's order)

    Returns
    -------
    Sigma_aligned : np.ndarray
        Covariance matrices aligned to ARUC ordering, shape (T, K, K)
    rho : np.ndarray
        Ellipsoid radii, shape (T,)
    sqrt_Sigma_aligned : np.ndarray or None
        Cholesky factors aligned to ARUC ordering, shape (T, K, K),
        or None if horizon.sqrt_sigma is None
    """
    # Get ARUC wind ordering
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    aruc_wind_ids = [data.gen_ids[i] for i in range(len(data.gen_ids)) if is_wind[i]]

    # Validate all ARUC wind IDs are in provider
    provider_set = set(provider_wind_ids)
    for wid in aruc_wind_ids:
        if wid not in provider_set:
            raise ValueError(
                f"ARUC wind generator '{wid}' not in provider: {provider_wind_ids}"
            )

    # Build permutation: provider index -> ARUC index
    provider_idx_map = {wid: i for i, wid in enumerate(provider_wind_ids)}
    perm = [provider_idx_map[wid] for wid in aruc_wind_ids]

    # Apply permutation to covariance matrices
    # For Sigma[t], we need to permute both rows and columns
    Sigma_aligned = horizon.sigma[:, perm, :][:, :, perm]
    sqrt_Sigma_aligned = None
    if horizon.sqrt_sigma is not None:
        sqrt_Sigma_aligned = horizon.sqrt_sigma[:, perm, :][:, :, perm]

    return Sigma_aligned, horizon.rho.copy(), sqrt_Sigma_aligned


def build_aruc_ldr_model(
    data: DAMData,
    Sigma: np.ndarray,
    rho: Union[float, np.ndarray],
    rho_lines_frac: Optional[float] = None,
    sqrt_Sigma: Union[np.ndarray, None] = None,
    M_p: float = 1e5,
    model_name: str = "ARUC_LDR",
    dam_commitment: Optional[Dict[str, np.ndarray]] = None,
    enforce_lines: bool = True,
    mip_gap: float = 0.005,
    incremental_obj: bool = False,
    dispatch_cost_scale: float = 0.1,
    gurobi_numeric_mode: str = "balanced",
    robust_mask: Optional[np.ndarray] = None,
    fix_wind_z: bool = False,
    worst_case_cost: bool = True,
    robust_ramp: bool = False,
) -> Tuple[gp.Model, Dict[str, object]]:
    """
    Adaptive robust UC with linear decision rules:

        p(i,t)(r) = p0(i,t) + Z(i,t)^T r

    with r in ellipsoidal uncertainty set:
        { r : r^T Sigma r <= rho^2 }

    Supports both static and time-varying uncertainty:
    - Static: Sigma (K, K), rho scalar - same for all time periods
    - Time-varying: Sigma (T, K, K), rho (T,) - different for each period

    Parameters
    ----------
    data : DAMData
        UC problem data
    Sigma : np.ndarray
        Covariance matrix. Shape (K, K) for static or (T, K, K) for time-varying.
    rho : float or np.ndarray
        Ellipsoid radius. Scalar for static or shape (T,) for time-varying.
    sqrt_Sigma : np.ndarray or None
        Pre-computed Cholesky factor(s). Shape (K, K) or (T, K, K).
        If None, computed from Sigma.
    M_p : float
        Big-M penalty for power balance slack
    model_name : str
        Gurobi model name
    dam_commitment : dict, optional
        DAM commitment solution for DARUC mode. Keys:
        - "u": (I, T) array of binary commitment status
        - "v": (I, T) array of startup indicators
        - "w": (I, T) array of shutdown indicators
        When provided, adds constraints u >= u_DAM (can only add commitments,
        never decommit) and tracks deviations u' = u - u_DAM.

    Returns
    -------
    model : gp.Model
        Gurobi model
    vars_dict : dict
        Dictionary of variable containers

    Notes
    -----
    This shares the same DAMData as the deterministic DAM model, but:
    - p is represented as an affine function of r.
    - Constraints are enforced for all r in the ellipsoid (robust counterpart).
    """

    I = data.n_gens
    N = data.n_buses
    L = data.n_lines
    T = data.n_periods
    B = data.n_blocks

    if worst_case_cost:
        assert B == 1, (
            f"worst_case_cost requires single_block (B=1), got B={B}"
        )

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

    dt = data.dt  # period durations in hours, shape (T,)

    u_init = data.u_init
    init_up = data.init_up_time
    init_down = data.init_down_time

    # Robust mask: which periods have full robust (SOC/Z) constraints
    if robust_mask is None:
        robust_mask = np.ones(T, dtype=bool)
    else:
        robust_mask = np.asarray(robust_mask, dtype=bool)
        assert robust_mask.shape == (T,), (
            f"robust_mask shape {robust_mask.shape} != ({T},)"
        )
    n_robust = int(robust_mask.sum())
    n_nominal = T - n_robust
    if n_nominal > 0:
        print(f"  [ARUC] robust_mask: {n_robust} robust + {n_nominal} nominal periods")

    # Identify wind units if you've tagged them in data.gen_type
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    n_wind = len(wind_idx)

    # Only thermal and wind generators participate in uncertainty response (Z matrix).
    # Solar/hydro have zero cost, so allowing them to absorb uncertainty via Z
    # makes robust hedging free and eliminates cost differentiation between
    # ARUC and DARUC.
    is_z_eligible = np.array(
        [gt.upper() in ("THERMAL", "WIND") for gt in data.gen_type]
    )
    z_elig = [i for i in range(I) if is_z_eligible[i]]
    z_elig_set = set(z_elig)
    thermal_idx = [i for i in range(I) if is_z_eligible[i] and not is_wind[i]]

    # Dimension of uncertainty vector r:
    # simplest: one r per wind generator (aggregate over time),
    # or one per wind gen and time. For now, assume K = n_wind.
    K = n_wind

    # Detect time-varying mode
    time_varying = (Sigma.ndim == 3) or (
        isinstance(rho, np.ndarray) and rho.ndim == 1
    )

    if time_varying:
        # Normalize to (T, K, K) and (T,)
        if Sigma.ndim == 2:
            Sigma = np.tile(Sigma[np.newaxis, :, :], (T, 1, 1))
        if isinstance(rho, (int, float)):
            rho = np.full(T, rho)

        assert Sigma.shape == (T, K, K), (
            f"Expected Sigma shape ({T}, {K}, {K}), got {Sigma.shape}"
        )
        assert rho.shape == (T,), f"Expected rho shape ({T},), got {rho.shape}"

        # Resolve rho_lines from fraction: scales with time-varying rho
        if rho_lines_frac is not None:
            rho_lines = rho_lines_frac * rho  # (T,) array
        else:
            rho_lines = rho  # Same object — full backward compat

        # Use provided sqrt_Sigma or compute
        if sqrt_Sigma is None:
            sqrt_Sigma = np.zeros_like(Sigma)
            for t_idx in range(T):
                sqrt_Sigma[t_idx] = np.linalg.cholesky(Sigma[t_idx])
    else:
        # Static mode (original)
        assert Sigma.shape == (K, K), (
            f"Expected Sigma shape ({K}, {K}), got {Sigma.shape}"
        )
        # Resolve rho_lines from fraction
        if rho_lines_frac is not None:
            rho_lines = rho_lines_frac * rho  # scalar
        else:
            rho_lines = rho  # Same object — full backward compat
        if sqrt_Sigma is None:
            sqrt_Sigma = np.linalg.cholesky(Sigma)

    if rho_lines is not rho:
        if time_varying:
            print(f"  [ARUC] rho_lines_frac={rho_lines_frac} -> rho_lines range "
                  f"[{rho_lines.min():.3f}, {rho_lines.max():.3f}]")
        else:
            print(f"  [ARUC] rho_lines_frac={rho_lines_frac} -> rho_lines={rho_lines:.3f}")

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
    # Worst-case dispatch cost epigraph variables
    # ------------------------------------------------------------------
    if worst_case_cost:
        gamma_cost = m.addVar(lb=0.0, name="gamma_cost")
        z_cost = m.addVars(
            [t for t in range(T) if robust_mask[t]],
            lb=0.0, name="z_cost",
        )
        y_cost = m.addVars(
            [(t, k) for t in range(T) if robust_mask[t] for k in range(K)],
            lb=-GRB.INFINITY, name="y_cost",
        )

    # ------------------------------------------------------------------
    # Fix commitment for zero-cost generators (WIND, SOLAR, HYDRO).
    # These have Pmin=0, MUT=0, MDT=0 and all costs=0, so the commitment
    # variable is degenerate (u=0 and u=1 give identical objective).
    # Fixing u=1 eliminates solver arbitrariness and removes binaries.
    # For wind in ARUC: also ensures Z variables for wind interact
    # correctly with dispatch (u=0 would force p0=0 while Z could
    # remain nonzero, creating inconsistent uncertainty response).
    # ------------------------------------------------------------------
    for i in range(I):
        if data.gen_type[i] in ("WIND", "SOLAR", "HYDRO"):
            for t in range(T):
                u[i, t].lb = 1.0
                u[i, t].ub = 1.0

    # ------------------------------------------------------------------
    # Second-stage LDR variables: Z[i,t,k]
    # ------------------------------------------------------------------
    # Z encodes sensitivity of p(i,t) to each component of r:
    #   p(i,t)(r) = p0[i,t] + sum_k Z[i,t,k] * r_k
    # Only z-eligible generators (thermal + wind) participate; solar/hydro
    # have zero uncertainty response, so we skip them entirely.
    # Only created for robust periods (robust_mask[t] == True).
    Z = m.addVars(
        [(i, t, k) for i in z_elig for t in range(T) if robust_mask[t] for k in range(K)],
        vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Z",
    )

    # ------------------------------------------------------------------
    # Fix wind Z diagonal: Z[wind_k, t, k_wind] = 1, off-diagonal = 0
    # Forces wind to fully track its own realization, eliminating
    # curtailment and forcing thermal units to provide all hedging.
    # ------------------------------------------------------------------
    if fix_wind_z:
        print(f"  [ARUC] fix_wind_z=True: fixing Z[wind_k,t,k]=1 (diagonal), "
              f"Z[wind_k,t,j!=k]=0 (off-diagonal)")
        for k_wind, i in enumerate(wind_idx):
            for t in range(T):
                if not robust_mask[t]:
                    continue
                for k in range(K):
                    if k == k_wind:
                        Z[i, t, k].lb = 1.0
                        Z[i, t, k].ub = 1.0
                    else:
                        Z[i, t, k].lb = 0.0
                        Z[i, t, k].ub = 0.0

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    obj = gp.LinExpr()

    if incremental_obj and dam_commitment is not None:
        # Incremental objective: only pay commitment costs for additional
        # units (u_dam=0), scale dispatch costs down to break ties.
        u_dam_arr = dam_commitment["u"]
        if worst_case_cost:
            print(f"  [ARUC] Incremental objective: commitment costs for "
                  f"additional units only, worst-case dispatch scaled by {dispatch_cost_scale}")
        else:
            print(f"  [ARUC] Incremental objective: commitment costs for "
                  f"additional units only, dispatch scaled by {dispatch_cost_scale}")
        for i in range(I):
            for t in range(T):
                if u_dam_arr[i, t] < 0.5:  # not committed by DAM
                    obj.addTerms(C_NL[i] * dt[t], u[i, t])
                    obj.addTerms(C_SU[i], v[i, t])
                    obj.addTerms(C_SD[i], w[i, t])
        if worst_case_cost:
            obj.addTerms(dispatch_cost_scale, gamma_cost)
        else:
            for i in range(I):
                for t in range(T):
                    for b in range(B):
                        obj.addTerms(
                            dispatch_cost_scale * block_cost[i, b] * dt[t],
                            p0_block[i, t, b],
                        )
    else:
        # Full cost objective (standard)
        for i in range(I):
            for t in range(T):
                obj.addTerms(C_NL[i] * dt[t], u[i, t])     # no-load: $/hr × hours
                obj.addTerms(C_SU[i], v[i, t])               # startup: one-time
                obj.addTerms(C_SD[i], w[i, t])               # shutdown: one-time
        if worst_case_cost:
            obj.addTerms(1.0, gamma_cost)
        else:
            for i in range(I):
                for t in range(T):
                    for b in range(B):
                        obj.addTerms(block_cost[i, b] * dt[t], p0_block[i, t, b])

    for t in range(T):
        obj.addTerms(M_p * dt[t], s_p[t])

    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Robust Pmin/Pmax/Block constraints
    # ------------------------------------------------------------------
    # These require ||Z_{i,t} L_t|| for each generator i at time t.
    # We introduce auxiliary variables z_gen[i,t] >= ||y_gen[i,t,:]||
    # where y_gen[i,t,:] = L_t @ Z_{i,t}

    z_gen = m.addVars(
        [(i, t) for i in thermal_idx for t in range(T) if robust_mask[t]],
        lb=0.0, name="z_gen",
    )
    y_gen = m.addVars(
        [(i, t, k) for i in thermal_idx for t in range(T) if robust_mask[t] for k in range(K)],
        lb=-GRB.INFINITY, name="y_gen",
    )

    for i in range(I):
        for t in range(T):
            # --- Non-robust (nominal) period: DAM-like constraints ---
            if not robust_mask[t]:
                m.addConstr(
                    p0[i, t] <= Pmax_2d[i, t] * u[i, t],
                    name=f"p0_max_nom_i{i}_t{t}",
                )
                m.addConstr(
                    Pmin[i] * u[i, t] <= p0[i, t],
                    name=f"p0_min_nom_i{i}_t{t}",
                )
                for b in range(B):
                    m.addConstr(
                        p0_block[i, t, b] <= block_cap[i, b] * u[i, t],
                        name=f"p0_block_cap_i{i}_t{t}_b{b}",
                    )
                m.addConstr(
                    p0[i, t]
                    <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                    name=f"p0_agg_nom_i{i}_t{t}",
                )
                continue

            # --- Robust period: full uncertainty constraints ---

            # Get time-specific Cholesky factor and rho
            sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
            rho_t = rho[t] if time_varying else rho

            # ----------------------------------------------------------
            # Wind generators: skip generic Pmax/Pmin/block robust
            # constraints.  Their physical upper bound p(r) <= Pbar + r_k
            # moves with the realization and is handled by the dedicated
            # wind availability SOC (section 3 below).  The generic
            # robust Pmax enforces the FIXED ceiling p(r) <= Pbar, which
            # combined with wind availability yields Z diagonal = 0.5
            # instead of the correct ~1.0.
            # ----------------------------------------------------------
            if is_wind[i]:
                # Block capacity (nominal, not robust)
                for b in range(B):
                    m.addConstr(
                        p0_block[i, t, b] <= block_cap[i, b] * u[i, t],
                        name=f"p0_block_cap_i{i}_t{t}_b{b}",
                    )
                # Nominal block aggregation: p0 = sum_b p0_block
                m.addConstr(
                    p0[i, t]
                    <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                    name=f"p0_agg_nom_i{i}_t{t}",
                )
                continue

            # --- Solar/Hydro: no uncertainty response, nominal constraints only ---
            if not is_z_eligible[i]:
                m.addConstr(
                    p0[i, t] <= Pmax_2d[i, t] * u[i, t],
                    name=f"p0_max_nom_i{i}_t{t}",
                )
                m.addConstr(
                    Pmin[i] * u[i, t] <= p0[i, t],
                    name=f"p0_min_nom_i{i}_t{t}",
                )
                for b in range(B):
                    m.addConstr(
                        p0_block[i, t, b] <= block_cap[i, b] * u[i, t],
                        name=f"p0_block_cap_i{i}_t{t}_b{b}",
                    )
                m.addConstr(
                    p0[i, t]
                    <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                    name=f"p0_agg_nom_i{i}_t{t}",
                )
                continue

            # --- Thermal generators: full robust constraints ---

            # Define y_gen[i,t,k] = (L^T @ Z_{i,t})_k = sum_j L[j,k] * Z[i,t,j]
            # where L = chol(Sigma) (lower triangular, Sigma = LL^T).
            # We need L^T (not L) so that ||y|| = ||L^T z|| = sqrt(z^T Sigma z).
            for k_idx in range(K):
                expr = gp.LinExpr()
                for j_idx in range(K):
                    coef = float(sqrt_Sigma_t[j_idx, k_idx])  # L^T[k,j] = L[j,k]
                    if abs(coef) < 1e-12:
                        continue
                    expr += coef * Z[i, t, j_idx]
                m.addConstr(
                    y_gen[i, t, k_idx] == expr, name=f"y_gen_def_i{i}_t{t}_k{k_idx}"
                )

            # SOC: z_gen[i,t] >= ||y_gen[i,t,:]||
            m.addConstr(
                z_gen[i, t] * z_gen[i, t]
                >= gp.quicksum(y_gen[i, t, k] * y_gen[i, t, k] for k in range(K)),
                name=f"soc_gen_i{i}_t{t}",
            )

            # Block capacity limits
            for b in range(B):
                m.addConstr(
                    p0_block[i, t, b] <= block_cap[i, b] * u[i, t],
                    name=f"p0_block_cap_i{i}_t{t}_b{b}",
                )

            # Robust block aggregation: a + rho*||Z L|| <= sum_b p_{i,t,b}
            m.addConstr(
                p0[i, t] + rho_t * z_gen[i, t]
                <= gp.quicksum(p0_block[i, t, b] for b in range(B)),
                name=f"p0_agg_rob_i{i}_t{t}",
            )

            # Robust Pmax: a + rho*||Z L|| <= Pmax * u
            m.addConstr(
                p0[i, t] + rho_t * z_gen[i, t] <= Pmax_2d[i, t] * u[i, t],
                name=f"p0_max_rob_i{i}_t{t}",
            )

            # Robust Pmin: Pmin * u <= a - rho*||Z L||
            m.addConstr(
                Pmin[i] * u[i, t] <= p0[i, t] - rho_t * z_gen[i, t],
                name=f"p0_min_rob_i{i}_t{t}",
            )

    # Ramps — nominal or robust depending on robust_ramp flag.
    # dt_ramp = (dt[t-1] + dt[t]) / 2  (transition time between period midpoints)
    # Skip non-thermal generators: wind/solar/hydro have RU=PMax in RTS-GMLC
    # data (effectively unlimited ramp), so these constraints are non-binding.
    #
    # Robust ramp (when robust_ramp=True and both periods are robust):
    #   p0[i,t] + rho_t * z_gen[i,t] - p0[i,t-1] + rho_{t-1} * z_gen[i,t-1]
    #       <= RU_i * dt_ramp * (u[i,t-1] + v[i,t])
    # This ensures worst-case dispatch stays within ramp limits.
    if robust_ramp:
        print(f"  [ARUC] robust_ramp=True: using SOC-based ramp constraints for robust periods")
    for i in range(I):
        if data.gen_type[i] != "THERMAL":
            continue
        for t in range(1, T):
            dt_ramp = (dt[t - 1] + dt[t]) / 2.0
            if robust_ramp and robust_mask[t] and robust_mask[t - 1] and i in z_elig_set:
                rho_t = rho[t] if time_varying else rho
                rho_tm1 = rho[t - 1] if time_varying else rho
                m.addConstr(
                    p0[i, t] + rho_t * z_gen[i, t]
                    - p0[i, t - 1] + rho_tm1 * z_gen[i, t - 1]
                    <= RU[i] * dt_ramp * u[i, t - 1] + RU[i] * dt_ramp * v[i, t],
                    name=f"ramp_up_rob_i{i}_t{t}",
                )
                m.addConstr(
                    p0[i, t - 1] + rho_tm1 * z_gen[i, t - 1]
                    - p0[i, t] + rho_t * z_gen[i, t]
                    <= RD[i] * dt_ramp * u[i, t] + RD[i] * dt_ramp * w[i, t],
                    name=f"ramp_down_rob_i{i}_t{t}",
                )
            else:
                m.addConstr(
                    p0[i, t] - p0[i, t - 1] <= RU[i] * dt_ramp * u[i, t - 1] + RU[i] * dt_ramp * v[i, t],
                    name=f"ramp_up_i{i}_t{t}",
                )
                m.addConstr(
                    p0[i, t - 1] - p0[i, t] <= RD[i] * dt_ramp * u[i, t] + RD[i] * dt_ramp * w[i, t],
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
    # MUT / MDT Constraints — look-forward, counting hours not periods
    # ------------------------------------------------------------------
    for i in range(I):
        mut_hrs = float(MUT[i])
        mdt_hrs = float(MDT[i])

        # Minimum up time constraints
        if mut_hrs > 0:
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

        # Minimum down time constraints
        if mdt_hrs > 0:
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

    # ------------------------------------------------------------------
    # DARUC constraints (if DAM commitment is provided)
    # ------------------------------------------------------------------
    if dam_commitment is not None:
        u_dam = dam_commitment["u"]  # (I, T) array
        v_dam = dam_commitment["v"]  # (I, T)
        w_dam = dam_commitment["w"]  # (I, T)

        # Deviation tracking variables
        u_prime = m.addVars(I, T, lb=0.0, name="u_prime")  # u' >= 0
        v_prime = m.addVars(I, T, lb=-GRB.INFINITY, name="v_prime")
        w_prime = m.addVars(I, T, lb=-GRB.INFINITY, name="w_prime")

        for i in range(I):
            for t in range(T):
                # u' = u - u_DAM  (>= 0 enforced by lb)
                m.addConstr(
                    u_prime[i, t] == u[i, t] - float(u_dam[i, t]),
                    name=f"u_prime_i{i}_t{t}",
                )
                # v' = v - v_DAM
                m.addConstr(
                    v_prime[i, t] == v[i, t] - float(v_dam[i, t]),
                    name=f"v_prime_i{i}_t{t}",
                )
                # w' = w - w_DAM
                m.addConstr(
                    w_prime[i, t] == w[i, t] - float(w_dam[i, t]),
                    name=f"w_prime_i{i}_t{t}",
                )

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
        # Only for robust periods (Z doesn't exist for nominal periods)
        if robust_mask[t]:
            for k in range(K):
                m.addConstr(
                    gp.quicksum(Z[i, t, k] for i in z_elig) == 0.0,
                    name=f"bal_Z_t{t}_k{k}",
                )

    # 2) Line flows robust (skip entirely in copper-plate mode)
    # Only create SOC auxiliaries for robust periods
    z_line = m.addVars(
        [(l, t) for l in range(L) for t in range(T) if robust_mask[t]],
        lb=0.0, name="z_line",
    )
    y_line = m.addVars(
        [(l, t, k) for l in range(L) for t in range(T) if robust_mask[t] for k in range(K)],
        lb=-GRB.INFINITY, name="y_line",
    )

    if not enforce_lines:
        print("  [ARUC] Line flow constraints DISABLED (copper-plate mode)")
    if enforce_lines:
        for l in range(L):
            for t in range(T):
                # 1) flow_nom_expr (needed for both robust and nominal)
                flow_nom = gp.LinExpr()
                for n in range(N):
                    if abs(PTDF[l, n]) < 1e-10:
                        continue
                    gen_sum = gp.quicksum(p0[i, t] for i in gens_at_bus[n])
                    flow_nom += PTDF[l, n] * (gen_sum - float(d[n, t]))

                if not robust_mask[t]:
                    # Nominal line flow constraints (like DAM)
                    m.addConstr(
                        flow_nom <= Fmax[l],
                        name=f"line_max_nom_l{l}_t{t}",
                    )
                    m.addConstr(
                        flow_nom >= -Fmax[l],
                        name=f"line_min_nom_l{l}_t{t}",
                    )
                    continue

                # --- Robust period ---
                # Get time-specific values
                sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
                rho_lines_t = rho_lines[t] if time_varying else rho_lines

                # 2) a_expr[k] = d(flow) / d r_k
                a_expr = [gp.LinExpr() for _ in range(K)]
                for i in range(I):
                    if i not in z_elig_set:
                        continue
                    n = int(gen_to_bus[i])
                    if abs(PTDF[l, n]) < 1e-10:
                        continue
                    for k in range(K):
                        a_expr[k] += PTDF[l, n] * Z[i, t, k]

                # 3) Define auxiliary variables: y_line[l,t,k] = (L^T @ a_expr)[k]
                #    L^T[k,j] = L[j,k] where L = chol(Sigma).
                for i_k in range(K):
                    expr = gp.LinExpr()
                    for j_k in range(K):
                        coef = sqrt_Sigma_t[j_k, i_k]  # L^T[i_k, j_k] = L[j_k, i_k]
                        if abs(coef) < 1e-10:
                            continue
                        expr += coef * a_expr[j_k]
                    m.addConstr(
                        y_line[l, t, i_k] == expr, name=f"y_line_def_l{l}_t{t}_k{i_k}"
                    )

                # 4) SOC: z_line[l,t] >= ||y_line[l,t,:]||
                m.addConstr(
                    z_line[l, t] * z_line[l, t]
                    >= gp.quicksum(y_line[l, t, k] * y_line[l, t, k] for k in range(K)),
                    name=f"soc_line_l{l}_t{t}",
                )

                # 5) Robust line limits
                m.addConstr(
                    flow_nom + rho_lines_t * z_line[l, t] <= Fmax[l],
                    name=f"line_max_rob_l{l}_t{t}",
                )
                m.addConstr(
                    -flow_nom + rho_lines_t * z_line[l, t] <= Fmax[l],
                    name=f"line_min_rob_l{l}_t{t}",
                )

    # 3) Wind availability — only for robust periods
    #    When fix_wind_z=True, Z[wind_k,t,k]=1 and Z[wind_k,t,j!=k]=0 are
    #    already fixed, so (Z - e_k) = 0 and the SOC is trivially satisfied.
    #    We add a simple p0 <= Pbar constraint instead and skip the SOC.
    if fix_wind_z:
        z_wind = {}
        y_wind = {}
        for k_wind, i in enumerate(wind_idx):
            for t in range(T):
                if not robust_mask[t]:
                    continue
                # With Z fixed: p0 + 1*r_k <= Pbar + r_k  =>  p0 <= Pbar
                m.addConstr(
                    p0[i, t] <= Pmax_2d[i, t],
                    name=f"wind_max_fixed_i{i}_t{t}",
                )
    else:
        z_wind = m.addVars(
            [(k_wind, t) for k_wind in range(n_wind) for t in range(T) if robust_mask[t]],
            lb=0.0, name="z_wind",
        )
        y_wind = m.addVars(
            [(k_wind, t, k) for k_wind in range(n_wind) for t in range(T) if robust_mask[t] for k in range(K)],
            lb=-GRB.INFINITY, name="y_wind",
        )

        for k_wind, i in enumerate(wind_idx):
            for t in range(T):
                if not robust_mask[t]:
                    # Nominal wind: p0 <= Pmax*u (already handled in dispatch section above)
                    continue

                # Get time-specific values
                sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
                rho_t = rho[t] if time_varying else rho

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

                # Define auxiliary variables: y_wind[k_wind,t,q] = (L^T @ a_expr)[q]
                #    L^T[q,j] = L[j,q] where L = chol(Sigma).
                for q in range(K):
                    expr = gp.LinExpr()
                    for j in range(K):
                        coef = float(sqrt_Sigma_t[j, q])  # L^T[q,j] = L[j,q]
                        if abs(coef) < 1e-12:
                            continue
                        expr += coef * a_expr[j]
                    m.addConstr(
                        y_wind[k_wind, t, q] == expr, name=f"y_wind_def_w{k_wind}_t{t}_k{q}"
                    )

                # SOC: z_wind[k_wind, t] >= || y_wind[k_wind,t,:] ||_2
                m.addConstr(
                    z_wind[k_wind, t] * z_wind[k_wind, t]
                    >= gp.quicksum(
                        y_wind[k_wind, t, k] * y_wind[k_wind, t, k] for k in range(K)
                    ),
                    name=f"soc_wind_i{i}_t{t}",
                )

                # Robust inequality: alpha + rho * z_wind <= 0
                m.addConstr(
                    alpha_expr + rho_t * z_wind[k_wind, t] <= 0.0,
                    name=f"wind_max_rob_i{i}_t{t}",
                )

    # ------------------------------------------------------------------
    # 4) Worst-case dispatch cost epigraph
    # ------------------------------------------------------------------
    if worst_case_cost:
        C_dispatch = block_cost[:, 0]  # (I,) — one marginal cost per generator

        epigraph_rhs = gp.LinExpr()
        # Nominal dispatch cost for ALL periods
        for i in range(I):
            for t in range(T):
                epigraph_rhs += C_dispatch[i] * dt[t] * p0[i, t]

        # Worst-case redispatch cost for ROBUST periods only
        for t in range(T):
            if not robust_mask[t]:
                continue
            sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
            rho_t = rho[t] if time_varying else rho

            # y_cost[t,k] = (L_t^T @ c_t)[k] = sum_j L[j,k] * sum_i C_i * Z[i,t,j]
            for k_idx in range(K):
                expr = gp.LinExpr()
                for j_idx in range(K):
                    coef_L = float(sqrt_Sigma_t[j_idx, k_idx])
                    if abs(coef_L) < 1e-12:
                        continue
                    for i in z_elig:
                        if abs(C_dispatch[i]) < 1e-12:
                            continue
                        expr += coef_L * C_dispatch[i] * Z[i, t, j_idx]
                m.addConstr(y_cost[t, k_idx] == expr, name=f"y_cost_def_t{t}_k{k_idx}")

            # SOC: z_cost[t] >= ||y_cost[t,:]||
            m.addConstr(
                z_cost[t] * z_cost[t]
                >= gp.quicksum(y_cost[t, k] * y_cost[t, k] for k in range(K)),
                name=f"soc_cost_t{t}",
            )

            epigraph_rhs += rho_t * dt[t] * z_cost[t]

        m.addConstr(gamma_cost >= epigraph_rhs, name="epigraph_cost")

    # Gurobi params — numeric tuning for MISOCP with wide coefficient range
    _NUMERIC_MODES = {
        "fast":     {"NumericFocus": 0, "BarHomogeneous": -1, "ScaleFlag": 1},
        "balanced": {"NumericFocus": 1, "BarHomogeneous": -1, "ScaleFlag": 1},
        "robust":   {"NumericFocus": 2, "BarHomogeneous": 1,  "ScaleFlag": 2},
    }
    if gurobi_numeric_mode not in _NUMERIC_MODES:
        raise ValueError(
            f"Unknown gurobi_numeric_mode={gurobi_numeric_mode!r}; "
            f"choose from {list(_NUMERIC_MODES)}"
        )
    _nmode = _NUMERIC_MODES[gurobi_numeric_mode]
    m.Params.OutputFlag = 1
    m.Params.NumericFocus = _nmode["NumericFocus"]
    m.Params.BarHomogeneous = _nmode["BarHomogeneous"]
    m.Params.ScaleFlag = _nmode["ScaleFlag"]
    m.Params.MIPGap = mip_gap       # Default 0.5% — UC doesn't need 0.01% precision

    # Heuristic tuning for MISOCP — default Gurobi heuristics produce
    # terrible incumbents for this problem class (SOC + integer).
    # Spend more effort finding good feasible solutions early.
    m.Params.Heuristics = 0.2       # 20% of node time on heuristics (default 5%)
    m.Params.MIPFocus = 1           # Focus on finding feasible solutions quickly

    vars_dict: Dict[str, object] = {
        "u": u,
        "v": v,
        "w": w,
        "p0": p0,
        "p0_block": p0_block,
        "Z": Z,
        "s_p": s_p,
        "z_gen": z_gen,
        "y_gen": y_gen,
        "z_line": z_line,
        "y_line": y_line,
        "z_wind": z_wind,
        "y_wind": y_wind,
    }

    if worst_case_cost:
        vars_dict["gamma_cost"] = gamma_cost
        vars_dict["z_cost"] = z_cost
        vars_dict["y_cost"] = y_cost

    if dam_commitment is not None:
        vars_dict["u_prime"] = u_prime
        vars_dict["v_prime"] = v_prime
        vars_dict["w_prime"] = w_prime

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
