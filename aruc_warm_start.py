"""
Warm-starting ARUC-LDR model from deterministic DAM solution.

The key insight: The deterministic DAM solution provides good initial values for:
- u[i,t], v[i,t], w[i,t]: commitment decisions (binary)
- p0[i,t]: nominal dispatch (corresponds to DAM's p[i,t])
- Z[i,t,k]: can be initialized to 0 (no response to uncertainty)

This can significantly reduce solve time, especially for large instances.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from models import DAMData


def warm_start_aruc_from_dam(
    aruc_model: gp.Model,
    aruc_vars: Dict[str, Any],
    dam_vars: Dict[str, Any],
    data: DAMData,
) -> None:
    """
    Warm start ARUC model using deterministic DAM solution.

    Parameters
    ----------
    aruc_model : gp.Model
        The ARUC-LDR model to warm start
    aruc_vars : dict
        Variable dictionary from build_aruc_ldr_model
    dam_vars : dict
        Variable dictionary from build_dam_model (with solution values)
    data : DAMData
        The problem data
    """

    I = data.n_gens
    T = data.n_periods

    # Get DAM solution values
    dam_u = dam_vars["u"]
    dam_v = dam_vars["v"]
    dam_w = dam_vars["w"]
    dam_p = dam_vars["p"]

    # Get ARUC variables
    aruc_u = aruc_vars["u"]
    aruc_v = aruc_vars["v"]
    aruc_w = aruc_vars["w"]
    aruc_p0 = aruc_vars["p0"]
    aruc_Z = aruc_vars["Z"]

    # Count how many start values we set
    n_vars_set = 0

    # Set commitment variables from DAM solution
    for i in range(I):
        for t in range(T):
            # Binary commitment decisions
            aruc_u[i, t].Start = dam_u[i, t].X
            aruc_v[i, t].Start = dam_v[i, t].X
            aruc_w[i, t].Start = dam_w[i, t].X
            n_vars_set += 3

            # Nominal dispatch
            aruc_p0[i, t].Start = dam_p[i, t].X
            n_vars_set += 1

    # Initialize Z to zero (no response to uncertainty in starting point)
    for key in aruc_Z.keys():
        aruc_Z[key].Start = 0.0
        n_vars_set += 1

    # Optionally set slack variables if they exist
    if "s_p" in aruc_vars and "s_p" in dam_vars:
        dam_s_p = dam_vars["s_p"]
        aruc_s_p = aruc_vars["s_p"]
        for t in range(T):
            aruc_s_p[t].Start = dam_s_p[t].X
            n_vars_set += 1

    # Also set p0_block if it exists
    if "p0_block" in aruc_vars and "p_block" in dam_vars:
        dam_p_block = dam_vars["p_block"]
        aruc_p0_block = aruc_vars["p0_block"]
        B = data.n_blocks
        for i in range(I):
            for t in range(T):
                for b in range(B):
                    aruc_p0_block[i, t, b].Start = dam_p_block[i, t, b].X
                    n_vars_set += 1

    print(f"Warm start: set {n_vars_set} variable start values from DAM solution")
    aruc_model.update()


def build_aruc_with_warm_start(
    data: DAMData,
    Sigma: np.ndarray,
    rho: float,
    dam_model: Optional[gp.Model] = None,
    dam_vars: Optional[Dict[str, Any]] = None,
    M_p: float = 1e5,
    model_name: str = "ARUC_LDR_WarmStart",
) -> tuple[gp.Model, Dict[str, Any]]:
    """
    Build ARUC model and optionally warm start from DAM solution.

    Parameters
    ----------
    data : DAMData
        Problem data
    Sigma : np.ndarray
        Covariance matrix for uncertainty
    rho : float
        Ellipsoid radius
    dam_model : gp.Model, optional
        Solved deterministic DAM model
    dam_vars : dict, optional
        Variable dictionary from DAM model
    M_p : float
        Big-M penalty for power balance slack
    model_name : str
        Name for the ARUC model

    Returns
    -------
    model : gp.Model
        The ARUC model (possibly warm started)
    vars_dict : dict
        Dictionary of ARUC variables
    """

    from aruc_model import build_aruc_ldr_model

    # Build the ARUC model
    print("\nBuilding ARUC-LDR model...")
    model, vars_dict = build_aruc_ldr_model(
        data=data,
        Sigma=Sigma,
        rho=rho,
        M_p=M_p,
        model_name=model_name,
    )

    # Apply warm start if DAM solution is provided
    if dam_model is not None and dam_vars is not None:
        if dam_model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            print("\nApplying warm start from DAM solution...")
            warm_start_aruc_from_dam(model, vars_dict, dam_vars, data)
        else:
            print(
                f"\nWARNING: DAM model status is {dam_model.Status}, skipping warm start"
            )

    return model, vars_dict


# ---------------------------------------------------------------------
# Enhanced run script with warm start option
# ---------------------------------------------------------------------


def run_aruc_with_warm_start(
    data: DAMData,
    Sigma: np.ndarray,
    rho: float,
    solve_dam_first: bool = True,
    M_p: float = 1e4,
) -> Dict[str, Any]:
    """
    Run ARUC with optional warm start from DAM.

    Parameters
    ----------
    data : DAMData
        Problem data
    Sigma : np.ndarray
        Covariance matrix
    rho : float
        Ellipsoid radius
    solve_dam_first : bool
        If True, solve DAM first and use for warm start
    M_p : float
        Big-M penalty

    Returns
    -------
    results : dict
        {
            "dam_model": DAM model (if solved),
            "dam_vars": DAM variables,
            "dam_time": DAM solve time,
            "aruc_model": ARUC model,
            "aruc_vars": ARUC variables,
            "aruc_time": ARUC solve time,
            "speedup": speedup factor (if warm start used),
        }
    """

    from dam_model import build_dam_model

    results = {}

    # Step 1: Optionally solve DAM first
    if solve_dam_first:
        print("=" * 70)
        print("STEP 1: Solving Deterministic DAM")
        print("=" * 70)

        dam_model, dam_vars = build_dam_model(data, M_p=M_p)
        dam_model.Params.OutputFlag = 1

        import time

        start = time.time()
        dam_model.optimize()
        dam_time = time.time() - start

        if dam_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            raise RuntimeError(f"DAM model failed with status {dam_model.Status}")

        print(f"\n✓ DAM solved in {dam_time:.2f}s")
        print(f"  Objective: {dam_model.ObjVal:,.2f}")

        results["dam_model"] = dam_model
        results["dam_vars"] = dam_vars
        results["dam_time"] = dam_time
    else:
        dam_model = None
        dam_vars = None

    # Step 2: Build and solve ARUC (with warm start if DAM solved)
    print("\n" + "=" * 70)
    print("STEP 2: Solving ARUC-LDR")
    if solve_dam_first:
        print("         (with warm start from DAM)")
    print("=" * 70)

    aruc_model, aruc_vars = build_aruc_with_warm_start(
        data=data,
        Sigma=Sigma,
        rho=rho,
        dam_model=dam_model,
        dam_vars=dam_vars,
        M_p=M_p,
    )

    aruc_model.Params.OutputFlag = 1

    import time

    start = time.time()
    aruc_model.optimize()
    aruc_time = time.time() - start

    if aruc_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print(f"WARNING: ARUC model status {aruc_model.Status}")
    else:
        print(f"\n✓ ARUC solved in {aruc_time:.2f}s")
        print(f"  Objective: {aruc_model.ObjVal:,.2f}")

    results["aruc_model"] = aruc_model
    results["aruc_vars"] = aruc_vars
    results["aruc_time"] = aruc_time

    # Compute speedup if we have both solve times
    if solve_dam_first:
        # Note: This is just the ARUC time, not total time
        # A fairer comparison might be to solve ARUC without warm start
        print(f"\nTiming Summary:")
        print(f"  DAM solve time:  {dam_time:.2f}s")
        print(f"  ARUC solve time: {aruc_time:.2f}s")
        print(f"  Total time:      {dam_time + aruc_time:.2f}s")

    return results


# ---------------------------------------------------------------------
# Comparison: with vs without warm start
# ---------------------------------------------------------------------


def compare_warm_start_benefit(
    data: DAMData,
    Sigma: np.ndarray,
    rho: float,
    M_p: float = 1e4,
) -> None:
    """
    Compare solve times with and without warm start.
    """

    print("\n" + "=" * 80)
    print(" WARM START COMPARISON")
    print("=" * 80)

    from dam_model import build_dam_model
    import time

    # Solve DAM once
    print("\nSolving DAM for warm start...")
    dam_model, dam_vars = build_dam_model(data, M_p=M_p)
    dam_model.Params.OutputFlag = 0
    start = time.time()
    dam_model.optimize()
    dam_time = time.time() - start
    print(f"  DAM solved in {dam_time:.2f}s")

    # Test 1: ARUC without warm start
    print("\n--- Test 1: ARUC WITHOUT warm start ---")
    aruc_cold, aruc_vars_cold = build_aruc_with_warm_start(
        data=data,
        Sigma=Sigma,
        rho=rho,
        dam_model=None,
        dam_vars=None,
        M_p=M_p,
        model_name="ARUC_Cold",
    )
    aruc_cold.Params.OutputFlag = 0
    start = time.time()
    aruc_cold.optimize()
    time_cold = time.time() - start
    print(f"  Solved in {time_cold:.2f}s")
    print(f"  Iterations: {aruc_cold.BarIterCount}")

    # Test 2: ARUC with warm start
    print("\n--- Test 2: ARUC WITH warm start ---")
    aruc_warm, aruc_vars_warm = build_aruc_with_warm_start(
        data=data,
        Sigma=Sigma,
        rho=rho,
        dam_model=dam_model,
        dam_vars=dam_vars,
        M_p=M_p,
        model_name="ARUC_Warm",
    )
    aruc_warm.Params.OutputFlag = 0
    start = time.time()
    aruc_warm.optimize()
    time_warm = time.time() - start
    print(f"  Solved in {time_warm:.2f}s")
    print(f"  Iterations: {aruc_warm.BarIterCount}")

    # Summary
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f"Without warm start: {time_cold:.2f}s")
    print(f"With warm start:    {time_warm:.2f}s (+ {dam_time:.2f}s for DAM)")
    print(f"Total with WS:      {time_warm + dam_time:.2f}s")

    if time_cold > time_warm + dam_time:
        speedup = time_cold / (time_warm + dam_time)
        print(f"\n✓ Warm start is {speedup:.2f}x faster overall!")
    else:
        slowdown = (time_warm + dam_time) / time_cold
        print(f"\n✗ Warm start is {slowdown:.2f}x slower overall")
        print("  (This can happen for small problems or when DAM overhead dominates)")

    # Check both found same objective
    if aruc_cold.Status == GRB.OPTIMAL and aruc_warm.Status == GRB.OPTIMAL:
        obj_diff = abs(aruc_cold.ObjVal - aruc_warm.ObjVal)
        print(f"\nObjective difference: {obj_diff:.2e}")
        if obj_diff < 1e-3:
            print("✓ Both methods found the same solution")


if __name__ == "__main__":
    # Example usage
    from io_rts import build_damdata_from_rts
    from pathlib import Path
    import pandas as pd

    # Load data
    source_dir = Path("RTS_Data/SourceData")
    ts_dir = Path("RTS_Data/timeseries_data_files")
    start_time = pd.Timestamp("2020-01-01 00:00:00")

    data = build_damdata_from_rts(source_dir, ts_dir, start_time, horizon_hours=24)

    # Build uncertainty set
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    n_wind = len(wind_idx)

    Pmax_2d = data.Pmax_2d()
    variances = [(0.15 * Pmax_2d[i, :].mean()) ** 2 for i in wind_idx]
    Sigma = np.diag(variances)
    rho = 2.0

    # Run comparison
    compare_warm_start_benefit(data, Sigma, rho)
