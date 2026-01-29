"""
debug_aruc_infeasibility.py

Systematic debugging script for ARUC-LDR infeasibility issues.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from models import DAMData
from aruc_model import build_aruc_ldr_model
from dam_model import build_dam_model
from io_rts import build_damdata_from_rts


def debug_step_1_check_deterministic(data: DAMData) -> bool:
    """
    Step 1: Verify the deterministic DAM model is feasible.
    If this fails, the data itself has issues.
    """
    print("=" * 70)
    print("STEP 1: Check if deterministic DAM is feasible")
    print("=" * 70)

    model, vars_dict = build_dam_model(data, M_p=1e4)
    model.Params.OutputFlag = 1
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print(f"✓ Deterministic DAM is FEASIBLE (obj = {model.ObjVal:,.2f})")
        return True
    else:
        print(f"✗ Deterministic DAM is INFEASIBLE (status = {model.Status})")
        print("   Problem is in the base data, not the robust formulation!")
        return False


def debug_step_2_relax_robust_constraints(
    data: DAMData, Sigma: np.ndarray, rho: float
) -> None:
    """
    Step 2: Progressively relax robust constraints to find the culprit.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Test with progressively smaller uncertainty sets")
    print("=" * 70)

    # Test with very small rho values
    test_rhos = [0.0, 0.1, 0.5, 1.0, rho]

    for test_rho in test_rhos:
        print(f"\nTesting with rho = {test_rho:.2f}...")
        model, vars_dict = build_aruc_ldr_model(
            data=data,
            Sigma=Sigma,
            rho=test_rho,
            M_p=1e4,
            model_name=f"ARUC_rho_{test_rho}",
        )
        model.Params.OutputFlag = 0
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print(f"  ✓ FEASIBLE with rho={test_rho:.2f} (obj = {model.ObjVal:,.2f})")
        else:
            print(f"  ✗ INFEASIBLE with rho={test_rho:.2f}")
            if test_rho == 0.0:
                print(
                    "    → Even rho=0 is infeasible! Issue is in nominal constraints."
                )
            break


def debug_step_3_compute_iis(data: DAMData, Sigma: np.ndarray, rho: float) -> None:
    """
    Step 3: Compute Irreducible Inconsistent Subsystem (IIS) to identify
    which constraints are causing infeasibility.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Compute IIS (Irreducible Inconsistent Subsystem)")
    print("=" * 70)

    model, vars_dict = build_aruc_ldr_model(
        data=data, Sigma=Sigma, rho=rho, M_p=1e4, model_name="ARUC_IIS"
    )
    model.Params.OutputFlag = 1
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print("\nModel is infeasible. Computing IIS...")
        model.computeIIS()

        # Write IIS to file
        iis_file = Path("aruc_debug") / "model.ilp"
        iis_file.parent.mkdir(exist_ok=True)
        model.write(str(iis_file))
        print(f"\nIIS written to: {iis_file}")

        # Analyze IIS constraints
        print("\nConstraints in IIS:")
        iis_constrs = [c for c in model.getConstrs() if c.IISConstr]

        # Group by constraint name prefix
        constraint_types = {}
        for c in iis_constrs:
            prefix = c.ConstrName.split("_")[0] if "_" in c.ConstrName else c.ConstrName
            constraint_types[prefix] = constraint_types.get(prefix, 0) + 1

        print("\nIIS Constraint Summary:")
        for ctype, count in sorted(constraint_types.items(), key=lambda x: -x[1]):
            print(f"  {ctype}: {count} constraints")

        # Show first few IIS constraints
        print("\nFirst 10 IIS constraints:")
        for c in iis_constrs[:10]:
            print(f"  {c.ConstrName}")


def debug_step_4_check_wind_availability(data: DAMData) -> None:
    """
    Step 4: Check if wind availability constraints might be too restrictive.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Analyze wind unit constraints")
    print("=" * 70)

    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]

    if len(wind_idx) == 0:
        print("No wind units found.")
        return

    Pmax_2d = data.Pmax_2d()

    print(f"\nFound {len(wind_idx)} wind units:")
    for i in wind_idx:
        gen_id = data.gen_ids[i]
        avg_cap = Pmax_2d[i, :].mean()
        min_cap = Pmax_2d[i, :].min()
        max_cap = Pmax_2d[i, :].max()
        print(f"  {gen_id}: avg={avg_cap:.2f}, min={min_cap:.2f}, max={max_cap:.2f} MW")

    # Check if any wind units have zero or very low capacity
    for i in wind_idx:
        if Pmax_2d[i, :].max() < 1e-6:
            print(f"  WARNING: {data.gen_ids[i]} has near-zero capacity!")


def debug_step_5_disable_constraint_groups(
    data: DAMData, Sigma: np.ndarray, rho: float
) -> None:
    """
    Step 5: Build model and selectively disable constraint groups to isolate issue.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Test with constraint groups disabled")
    print("=" * 70)

    # Build full model
    model, vars_dict = build_aruc_ldr_model(
        data=data, Sigma=Sigma, rho=rho, M_p=1e4, model_name="ARUC_Debug"
    )

    # Test configurations: disable different robust constraint groups
    test_configs = [
        ("All constraints enabled", []),
        ("Wind constraints disabled", ["wind_max_rob", "soc_wind"]),
        ("Line constraints disabled", ["line_max_rob", "line_min_rob", "soc_line"]),
        (
            "All robust constraints disabled",
            ["wind_max_rob", "soc_wind", "line_max_rob", "line_min_rob", "soc_line"],
        ),
    ]

    for config_name, disabled_prefixes in test_configs:
        print(f"\n{config_name}...")

        # Reset model
        model, vars_dict = build_aruc_ldr_model(
            data=data, Sigma=Sigma, rho=rho, M_p=1e4, model_name="ARUC_Test"
        )

        # Disable specified constraints
        disabled_count = 0
        for constr in model.getConstrs():
            for prefix in disabled_prefixes:
                if constr.ConstrName.startswith(prefix):
                    model.remove(constr)
                    disabled_count += 1
                    break

        model.update()
        if disabled_count > 0:
            print(f"  Disabled {disabled_count} constraints")

        model.Params.OutputFlag = 0
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print(f"  ✓ FEASIBLE (obj = {model.ObjVal:,.2f})")
        else:
            print(f"  ✗ INFEASIBLE")


def debug_step_6_check_z_bounds(data: DAMData, Sigma: np.ndarray, rho: float) -> None:
    """
    Step 6: Check if Z variables need better bounds or if they're causing issues.
    """
    print("\n" + "=" * 70)
    print("STEP 6: Analyze Z variable bounds and constraints")
    print("=" * 70)

    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    n_wind = len(wind_idx)

    print(
        f"\nZ variables dimensions: {data.n_gens} gens × {data.n_periods} periods × {n_wind} uncertainty dims"
    )
    print(f"Total Z variables: {data.n_gens * data.n_periods * n_wind}")

    # Check if non-wind generators should have Z = 0
    non_wind_idx = np.where(~is_wind)[0]
    print(f"\nNon-wind generators: {len(non_wind_idx)}")
    print("  Suggestion: Consider fixing Z[i,t,k] = 0 for non-wind generators")
    print("  This reduces problem size and ensures only wind responds to uncertainty")


def run_full_debug_suite(data: DAMData, Sigma: np.ndarray, rho: float) -> None:
    """
    Run all debugging steps in sequence.
    """
    print("\n" + "=" * 80)
    print(" ARUC-LDR INFEASIBILITY DEBUGGING SUITE")
    print("=" * 80)

    # Step 1: Check base feasibility
    if not debug_step_1_check_deterministic(data):
        print("\n" + "=" * 80)
        print("DIAGNOSIS: Base data is infeasible. Fix deterministic model first!")
        print("=" * 80)
        return

    # Step 2: Test with smaller uncertainty sets
    debug_step_2_relax_robust_constraints(data, Sigma, rho)

    # Step 3: Compute IIS
    debug_step_3_compute_iis(data, Sigma, rho)

    # Step 4: Check wind data
    debug_step_4_check_wind_availability(data)

    # Step 5: Test with constraint groups disabled
    debug_step_5_disable_constraint_groups(data, Sigma, rho)

    # Step 6: Analyze Z variables
    debug_step_6_check_z_bounds(data, Sigma, rho)

    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE - Check output above for issues")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configuration
    RTS_DIR = Path("RTS_Data")
    SOURCE_DIR = RTS_DIR / "SourceData"
    TS_DIR = RTS_DIR / "timeseries_data_files"

    START_TIME = pd.Timestamp(year=2020, month=1, day=1, hour=0)
    HORIZON_HOURS = 48

    # Build data
    print("Loading RTS-GMLC data...")
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=START_TIME,
        horizon_hours=HORIZON_HOURS,
    )

    # Build uncertainty set
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    n_wind = len(wind_idx)

    if n_wind == 0:
        print("ERROR: No wind units found!")
        exit(1)

    # Simple diagonal covariance
    Pmax_2d = data.Pmax_2d()
    wind_std_fraction = 0.15
    variances = [(wind_std_fraction * Pmax_2d[i, :].mean()) ** 2 for i in wind_idx]
    Sigma = np.diag(variances)
    rho = 2.0

    # Run debugging suite
    run_full_debug_suite(data, Sigma, rho)
