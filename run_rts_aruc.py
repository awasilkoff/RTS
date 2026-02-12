"""
run_rts_aruc.py

End-to-end script for Adaptive Robust UC with Linear Decision Rules:
RTS-GMLC -> DAMData -> Gurobi ARUC-LDR -> results as DataFrames.

Assumes the following modules/files already exist in your project:

- models.py            (defines DAMData and BaseModel)
- aruc_ldr_model.py    (defines build_aruc_ldr_model(data, Sigma, rho, ...))
- io_rts.py            (defines build_damdata_from_rts(...))

You will likely need to adjust:
- RTS_DIR / SOURCE_DIR / TS_DIR
- start_time / horizon_hours
- Uncertainty set parameters (Sigma, rho)
to match your local setup and chosen study period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import gurobipy as gp

from models import DAMData
from aruc_model import build_aruc_ldr_model, align_uncertainty_to_aruc
from io_rts import build_damdata_from_rts
from uncertainty_set_provider import UncertaintySetProvider


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root RTS directory (adjust as needed)
RTS_DIR = Path("RTS_Data")


# Where you put the RTS-GMLC source tables and time-series
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"

# Example horizon: 48-hour day-ahead starting from a given timestamp
START_TIME = pd.Timestamp(year=2020, month=1, day=1, hour=0)
HORIZON_HOURS = 48

# Big-M penalty for power-balance slack
M_PENALTY = 1e4

# Uncertainty set parameters
# rho: radius of uncertainty ellipsoid (e.g., 2.0 for ~95% confidence)
# Sigma: covariance matrix for wind forecast errors
UNCERTAINTY_RHO = 2.0

# SPP forecast override for wind Pmax (set to None to use DAY_AHEAD_wind.csv)
SPP_FORECASTS_PARQUET = Path("uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet")
SPP_START_IDX = 0


# ---------------------------------------------------------------------------
# Uncertainty set construction
# ---------------------------------------------------------------------------


def build_uncertainty_set(
    data: DAMData,
    rho: float = UNCERTAINTY_RHO,
    wind_std_fraction: float = 0.15,
) -> tuple[np.ndarray, float]:
    """
    Construct uncertainty set parameters (Sigma, rho) for wind generators.

    Parameters
    ----------
    data : DAMData
        The DAM data object containing generator information
    rho : float
        Radius of uncertainty ellipsoid (e.g., 2.0 for ~95% confidence)
    wind_std_fraction : float
        Standard deviation as fraction of wind capacity (e.g., 0.15 = 15%)

    Returns
    -------
    Sigma : np.ndarray
        Covariance matrix (K x K) where K = number of wind units
    rho : float
        Ellipsoid radius
    """
    # Identify wind units
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    n_wind = len(wind_idx)

    if n_wind == 0:
        print("WARNING: No wind units found. Using identity covariance.")
        return np.eye(1), rho

    print(f"Found {n_wind} wind units for uncertainty modeling:")
    for idx in wind_idx:
        print(f"  - {data.gen_ids[idx]}")

    # Simple diagonal covariance: each wind unit has independent uncertainty
    # Variance proportional to squared capacity
    variances = []
    for i in wind_idx:
        # Use average Pmax over time horizon as nominal capacity
        avg_capacity = data.Pmax_2d()[i, :].mean()
        variance = (wind_std_fraction * avg_capacity) ** 2
        variances.append(variance)

    Sigma = np.diag(variances)

    print(f"\nUncertainty set configuration:")
    print(f"  Ellipsoid radius (rho): {rho:.2f}")
    print(f"  Wind std dev fraction: {wind_std_fraction:.1%}")
    print(f"  Covariance matrix shape: {Sigma.shape}")

    return Sigma, rho


# ---------------------------------------------------------------------------
# Helpers for extracting results
# ---------------------------------------------------------------------------


def extract_solution(
    data: DAMData,
    model: gp.Model,
    vars_dict: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Extract u[i,t], p0[i,t], and Z[i,t,k] from the ARUC-LDR model.

    Returns
    -------
    results : dict
        {
          "u": on/off status (index=gen_ids, columns=time),
          "p0": nominal dispatch (MW) (index=gen_ids, columns=time),
          "Z": LDR coefficients (index=gen_ids, columns=multiindex(time, k)),
          "obj": objective value (float),
        }
    """
    gen_ids = data.gen_ids
    time_labels = data.time

    I = len(gen_ids)
    T = len(time_labels)

    u = vars_dict["u"]
    p0 = vars_dict["p0"]
    Z = vars_dict["Z"]

    u_array = np.zeros((I, T), dtype=float)
    p0_array = np.zeros((I, T), dtype=float)

    for i in range(I):
        for t in range(T):
            u_array[i, t] = u[i, t].X
            p0_array[i, t] = p0[i, t].X

    u_df = pd.DataFrame(u_array, index=gen_ids, columns=time_labels)
    p0_df = pd.DataFrame(p0_array, index=gen_ids, columns=time_labels)

    # Extract Z coefficients as 2D matrix: I x (T*K)
    # Determine K from Z keys
    K = 0
    for key in Z.keys():
        K = max(K, key[2] + 1)

    # Create column names: (time, k) pairs
    Z_columns = pd.MultiIndex.from_product([time_labels, range(K)], names=["time", "k"])

    Z_array = np.zeros((I, T * K), dtype=float)
    for i in range(I):
        for t in range(T):
            for k in range(K):
                if (i, t, k) in Z:
                    Z_array[i, t * K + k] = Z[i, t, k].X

    Z_df = pd.DataFrame(Z_array, index=gen_ids, columns=Z_columns)

    results = {
        "u": u_df,
        "p0": p0_df,
        "Z": Z_df,
        "obj": float(model.ObjVal),
    }
    return results


def reshape_Z_for_generator(Z_df: pd.DataFrame, gen_id: str) -> pd.DataFrame:
    """
    Reshape Z coefficients for a single generator from (T*K,) to (T, K).

    Parameters
    ----------
    Z_df : pd.DataFrame
        Full Z dataframe with multiindex columns (time, k)
    gen_id : str
        Generator ID to extract

    Returns
    -------
    Z_gen : pd.DataFrame
        Z coefficients for this generator, shape (T, K)
        Index = time, Columns = k values
    """
    # Get row for this generator
    Z_row = Z_df.loc[gen_id]

    # Reshape from flat multiindex to 2D
    Z_reshaped = Z_row.unstack(level="k")

    return Z_reshaped


def analyze_Z_patterns(Z_df: pd.DataFrame, data: DAMData) -> None:
    """
    Print analysis of Z coefficient patterns.
    """
    print("\n" + "=" * 70)
    print("Z COEFFICIENT ANALYSIS")
    print("=" * 70)

    # Identify which generators have non-zero Z
    Z_array = Z_df.values
    nonzero_by_gen = (np.abs(Z_array) > 1e-6).sum(axis=1)

    active_gens = Z_df.index[nonzero_by_gen > 0]
    print(f"\nGenerators with non-zero Z coefficients: {len(active_gens)}")

    # Check against wind generators
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_gen_ids = [data.gen_ids[i] for i in range(len(data.gen_ids)) if is_wind[i]]

    print(f"Wind generators in problem: {len(wind_gen_ids)}")

    # Show top 5 generators by total |Z|
    total_abs_Z = np.abs(Z_array).sum(axis=1)
    top_idx = np.argsort(total_abs_Z)[-5:][::-1]

    print("\nTop 5 generators by total |Z|:")
    for idx in top_idx:
        gen_id = Z_df.index[idx]
        total = total_abs_Z[idx]
        gen_type = data.gen_type[data.gen_ids.index(gen_id)]
        print(f"  {gen_id} ({gen_type}): {total:.2f}")

    # Show example Z pattern for one wind unit
    if len(active_gens) > 0:
        example_gen = active_gens[0]
        Z_gen = reshape_Z_for_generator(Z_df, example_gen)

        print(f"\nExample Z pattern for {example_gen}:")
        print(f"  Shape: {Z_gen.shape} (time x K)")
        print(f"  First few rows:")
        print(Z_gen.head())


def print_brief_summary(results: Dict[str, Any], data: DAMData) -> None:
    """
    Print a small human-readable summary to the console.
    """
    u_df = results["u"]
    p0_df = results["p0"]
    obj = results["obj"]

    print("\n=== ARUC-LDR UC Summary ===")
    print(f"Objective value: {obj:,.2f}")
    print(f"Number of generators: {len(data.gen_ids)}")
    print(f"Number of buses:      {len(data.bus_ids)}")
    print(f"Number of lines:      {len(data.line_ids)}")
    print(f"Number of periods:    {len(data.time)}")

    # Total nominal generation vs total load
    total_gen = p0_df.to_numpy().sum()
    total_load = data.d.sum()

    print(f"Total nominal generation (MWh): {total_gen:,.2f}")
    print(f"Total load (MWh):               {total_load:,.2f}")

    # Show which units are ever committed
    committed_any = u_df.sum(axis=1) > 0.5
    committed_units = committed_any[committed_any].index.tolist()
    print(f"Units committed at least once: {len(committed_units)}")
    if len(committed_units) <= 20:
        print("  " + ", ".join(committed_units))


def compare_with_deterministic(
    aruc_results: Dict[str, Any],
    dam_results: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print comparison between ARUC and deterministic DAM results.
    """
    if dam_results is None:
        return

    print("\n=== Comparison: ARUC vs Deterministic DAM ===")

    aruc_obj = aruc_results["obj"]
    dam_obj = dam_results["obj"]

    print(f"Deterministic DAM objective: {dam_obj:,.2f}")
    print(f"ARUC-LDR objective:          {aruc_obj:,.2f}")
    print(
        f"Difference (ARUC - DAM):     {aruc_obj - dam_obj:,.2f} ({100*(aruc_obj/dam_obj - 1):.2f}%)"
    )

    # Compare commitment decisions
    aruc_u = aruc_results["u"]
    dam_u = dam_results["u"]

    # Units committed in ARUC but not DAM
    aruc_committed = aruc_u.sum(axis=1) > 0.5
    dam_committed = dam_u.sum(axis=1) > 0.5

    extra_committed = aruc_committed & ~dam_committed
    fewer_committed = ~aruc_committed & dam_committed

    if extra_committed.any():
        print(f"\nUnits committed in ARUC but not DAM ({extra_committed.sum()}):")
        print("  " + ", ".join(extra_committed[extra_committed].index.tolist()[:10]))

    if fewer_committed.any():
        print(f"\nUnits committed in DAM but not ARUC ({fewer_committed.sum()}):")
        print("  " + ", ".join(fewer_committed[fewer_committed].index.tolist()[:10]))


# ---------------------------------------------------------------------------
# Main end-to-end driver
# ---------------------------------------------------------------------------


def run_rts_aruc(
    source_dir: Path = SOURCE_DIR,
    ts_dir: Path = TS_DIR,
    start_time: pd.Timestamp = START_TIME,
    horizon_hours: int = HORIZON_HOURS,
    m_penalty: float = M_PENALTY,
    rho: float = UNCERTAINTY_RHO,
    wind_std_fraction: float = 0.15,
    slack_bus_id: Optional[int | str] = None,
    dam_results: Optional[Dict[str, Any]] = None,
    uncertainty_provider_path: Optional[Union[Path, str]] = None,
    provider_start_idx: int = 0,
    spp_forecasts_parquet: Optional[Path] = SPP_FORECASTS_PARQUET,
    spp_start_idx: int = SPP_START_IDX,
    enforce_lines: bool = True,
    rho_lines_frac: Optional[float] = None,
    mip_gap: float = 0.005,
    gurobi_numeric_mode: str = "balanced",
) -> Dict[str, Any]:
    """
    Full pipeline for ARUC-LDR:
      RTS input -> DAMData -> Uncertainty set -> Gurobi ARUC model -> solve -> outputs.

    Parameters
    ----------
    source_dir : Path
        Directory containing RTS source data
    ts_dir : Path
        Directory containing RTS time series data
    start_time : pd.Timestamp
        Start time for the horizon
    horizon_hours : int
        Number of hours in the horizon
    m_penalty : float
        Big-M penalty for power balance slack
    rho : float
        Ellipsoid radius for static uncertainty (ignored if provider used)
    wind_std_fraction : float
        Standard deviation fraction for static uncertainty (ignored if provider used)
    slack_bus_id : int or str, optional
        Slack bus ID for PTDF computation
    dam_results : dict, optional
        Results from deterministic DAM run for comparison
    uncertainty_provider_path : Path or str, optional
        Path to pre-computed uncertainty set NPZ file. If provided, uses
        time-varying uncertainty instead of static.
    provider_start_idx : int
        Starting index for provider query (default 0)

    Returns
    -------
    outputs : dict
        {
          "data": DAMData,
          "model": gp.Model,
          "vars": dict of Gurobi var containers,
          "results": dict of DataFrames (u, p0, Z, obj),
          "Sigma": covariance matrix (K,K) or (T,K,K),
          "rho": ellipsoid radius (scalar or (T,) array),
          "time_varying": bool indicating if time-varying uncertainty was used,
        }
    """
    print("Building DAMData from RTS-GMLC...")
    data: DAMData = build_damdata_from_rts(
        source_dir=source_dir,
        ts_dir=ts_dir,
        start_time=start_time,
        horizon_hours=horizon_hours,
        spp_forecasts_parquet=spp_forecasts_parquet,
        spp_start_idx=spp_start_idx,
    )
    print("  Done. Data shapes:")
    print(f"    n_gens   = {data.n_gens}")
    print(f"    n_buses  = {data.n_buses}")
    print(f"    n_lines  = {data.n_lines}")
    print(f"    n_periods= {data.n_periods}")

    # Determine whether to use time-varying or static uncertainty
    time_varying = False
    sqrt_Sigma = None

    if uncertainty_provider_path is not None:
        # Time-varying uncertainty from pre-computed provider
        print(f"\nLoading time-varying uncertainty from {uncertainty_provider_path}...")
        provider = UncertaintySetProvider.from_npz(uncertainty_provider_path)
        horizon = provider.get_by_indices(
            provider_start_idx, horizon_hours, compute_sqrt=True
        )

        Sigma, rho, sqrt_Sigma = align_uncertainty_to_aruc(
            horizon, data, provider.get_wind_gen_ids()
        )
        time_varying = True

        print(f"  Wind IDs from provider: {provider.get_wind_gen_ids()}")
        print(f"  Sigma shape: {Sigma.shape}")
        print(f"  rho range: [{rho.min():.3f}, {rho.max():.3f}]")
        model_name = "ARUC_LDR_TimeVarying"
    else:
        # Static uncertainty (original behavior)
        print("\nConstructing static uncertainty set...")
        Sigma, rho = build_uncertainty_set(
            data, rho=rho, wind_std_fraction=wind_std_fraction
        )
        model_name = "ARUC_LDR_RTS"

    print("\nBuilding Gurobi ARUC-LDR model...")
    model, vars_dict = build_aruc_ldr_model(
        data=data,
        Sigma=Sigma,
        rho=rho,
        rho_lines_frac=rho_lines_frac,
        sqrt_Sigma=sqrt_Sigma,
        M_p=m_penalty,
        model_name=model_name,
        enforce_lines=enforce_lines,
        mip_gap=mip_gap,
        gurobi_numeric_mode=gurobi_numeric_mode,
    )
    print("  Model built. Starting optimization...")

    model.optimize()

    if model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        print(f"WARNING: Model did not terminate optimally. Status: {model.Status}")
        if model.SolCount == 0:
            raise RuntimeError("No feasible solution found by Gurobi.")

    results = extract_solution(data, model, vars_dict)
    print_brief_summary(results, data)

    # Analyze Z patterns
    analyze_Z_patterns(results["Z"], data)

    # Compare with deterministic if provided
    if dam_results is not None:
        compare_with_deterministic(results, dam_results)

    return {
        "data": data,
        "model": model,
        "vars": vars_dict,
        "results": results,
        "Sigma": Sigma,
        "rho": rho,
        "rho_lines_frac": rho_lines_frac,
        "time_varying": time_varying,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Option 1: Run ARUC standalone
    outputs_aruc = run_rts_aruc()

    # Option 2: Run both DAM and ARUC for comparison
    # Uncomment below to compare:
    """
    from run_rts_dam import run_rts_dam
    
    print("=" * 70)
    print("RUNNING DETERMINISTIC DAM")
    print("=" * 70)
    outputs_dam = run_rts_dam()
    
    print("\n" + "=" * 70)
    print("RUNNING ADAPTIVE ROBUST UC (ARUC-LDR)")
    print("=" * 70)
    outputs_aruc = run_rts_aruc(dam_results=outputs_dam["results"])
    """

    # Write results to CSV
    results = outputs_aruc["results"]
    u_df = results["u"]
    p0_df = results["p0"]
    Z_df = results["Z"]

    out_dir = Path("aruc_outputs")
    out_dir.mkdir(exist_ok=True)

    u_df.to_csv(out_dir / "commitment_u_aruc.csv")
    p0_df.to_csv(out_dir / "dispatch_p0_aruc.csv")
    Z_df.to_csv(out_dir / "ldr_coefficients_Z_aruc.csv")

    print("\nWrote:")
    print(f"  {out_dir / 'commitment_u_aruc.csv'}")
    print(f"  {out_dir / 'dispatch_p0_aruc.csv'}")
    print(f"  {out_dir / 'ldr_coefficients_Z_aruc.csv'}")

    # Print Z statistics
    Z_array = Z_df.values
    print(f"\nZ coefficient statistics:")
    print(f"  Shape: {Z_array.shape} (generators x time*K)")
    print(f"  Non-zero: {(np.abs(Z_array) > 1e-6).sum()}")
    print(f"  Mean abs value: {np.abs(Z_array).mean():.4f}")
    print(f"  Max abs value: {np.abs(Z_array).max():.4f}")
