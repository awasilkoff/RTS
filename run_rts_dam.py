"""
run_rts_dam.py

End-to-end script:
RTS-GMLC -> DAMData -> Gurobi DAM UC -> results as DataFrames.

Assumes the following modules/files already exist in your project:

- models.py            (defines DAMData and BaseModel)
- dam_model.py         (defines build_dam_model(data: DAMData) -> (model, vars_dict))
- io_rts.py            (defines build_damdata_from_rts(...))

You will likely need to adjust:
- RTS_DIR / SOURCE_DIR / TS_DIR
- start_time / horizon_hours
to match your local setup and chosen study period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import gurobipy as gp

from models import DAMData
from dam_model import build_dam_model
from io_rts import build_damdata_from_rts  # <- use your existing function


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root RTS directory (adjust as needed)
RTS_DIR = Path("RTS_Data")

# Where you put the RTS-GMLC source tables and time-series
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"  # adjust to your actual path

# Example horizon: 48-hour day-ahead starting from a given timestamp
# Adjust this to match your RTS year / dates
START_TIME = pd.Timestamp(year=2020, month=1, day=1, hour=0)
HORIZON_HOURS = 48

# Big-M penalty for power-balance slack
M_PENALTY = 1e4

# SPP forecast override for wind Pmax (set to None to use DAY_AHEAD_wind.csv)
SPP_FORECASTS_PARQUET = Path("uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet")
SPP_START_IDX = 0


# ---------------------------------------------------------------------------
# Helpers for extracting results
# ---------------------------------------------------------------------------


def extract_solution(
    data: DAMData,
    model: gp.Model,
    vars_dict: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Extract u[i,t] and p[i,t] from the Gurobi model into pandas DataFrames.

    Returns
    -------
    results : dict
        {
          "u": on/off status (index=gen_ids, columns=time),
          "p": dispatch (MW) (index=gen_ids, columns=time),
          "obj": objective value (float),
        }
    """
    gen_ids = data.gen_ids
    time_labels = data.time

    I = len(gen_ids)
    T = len(time_labels)

    u = vars_dict["u"]
    p = vars_dict["p"]

    u_array = np.zeros((I, T), dtype=float)
    p_array = np.zeros((I, T), dtype=float)

    for i in range(I):
        for t in range(T):
            u_array[i, t] = u[i, t].X
            p_array[i, t] = p[i, t].X

    u_df = pd.DataFrame(u_array, index=gen_ids, columns=time_labels)
    p_df = pd.DataFrame(p_array, index=gen_ids, columns=time_labels)

    results = {
        "u": u_df,
        "p": p_df,
        "obj": float(model.ObjVal),
    }
    return results


def print_brief_summary(results: Dict[str, Any], data: DAMData) -> None:
    """
    Print a small human-readable summary to the console.
    """
    u_df = results["u"]
    p_df = results["p"]
    obj = results["obj"]

    print("\n=== DAM UC Summary ===")
    print(f"Objective value: {obj:,.2f}")
    print(f"Number of generators: {len(data.gen_ids)}")
    print(f"Number of buses:      {len(data.bus_ids)}")
    print(f"Number of lines:      {len(data.line_ids)}")
    print(f"Number of periods:    {len(data.time)}")

    # Example: total generation vs total load over the horizon
    total_gen = p_df.to_numpy().sum()
    total_load = data.d.sum()

    print(f"Total generation (MWh across horizon): {total_gen:,.2f}")
    print(f"Total load (MWh across horizon):       {total_load:,.2f}")

    # Show which units are ever committed
    committed_any = u_df.sum(axis=1) > 0.5
    committed_units = committed_any[committed_any].index.tolist()
    print(f"Units committed at least once: {len(committed_units)}")
    if len(committed_units) <= 20:
        print("  " + ", ".join(committed_units))


# ---------------------------------------------------------------------------
# Main end-to-end driver
# ---------------------------------------------------------------------------


def run_rts_dam(
    source_dir: Path = SOURCE_DIR,
    ts_dir: Path = TS_DIR,
    start_time: pd.Timestamp = START_TIME,
    horizon_hours: int = HORIZON_HOURS,
    m_penalty: float = M_PENALTY,
    slack_bus_id: Optional[int | str] = None,  # if your build_damdata uses it
    spp_forecasts_parquet: Optional[Path] = SPP_FORECASTS_PARQUET,
    spp_start_idx: int = SPP_START_IDX,
    enforce_lines: bool = True,
    day2_interval_hours: int = 1,
    single_block: bool = True,
    include_renewables: bool = False,
    include_nuclear: bool = False,
    include_zero_marginal: Optional[bool] = None,
    ramp_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Full pipeline:
      RTS input -> DAMData -> Gurobi model -> solve -> DataFrame outputs.

    Returns
    -------
    outputs : dict
        {
          "data": DAMData,
          "model": gp.Model,
          "vars": dict of Gurobi var containers,
          "results": dict of DataFrames (u, p, obj),
        }
    """
    print("Building DAMData from RTS-GMLC...")
    # NOTE: adjust args to match your actual build_damdata_from_rts signature.
    # I’m assuming: (source_dir, ts_dir, start_time, horizon_hours)
    data: DAMData = build_damdata_from_rts(
        source_dir=source_dir,
        ts_dir=ts_dir,
        start_time=start_time,
        horizon_hours=horizon_hours,
        spp_forecasts_parquet=spp_forecasts_parquet,
        spp_start_idx=spp_start_idx,
        day2_interval_hours=day2_interval_hours,
        single_block=single_block,
        include_renewables=include_renewables,
        include_nuclear=include_nuclear,
        include_zero_marginal=include_zero_marginal,
        ramp_scale=ramp_scale,
    )
    print("  Done. Data shapes:")
    print(f"    n_gens   = {data.n_gens}")
    print(f"    n_buses  = {data.n_buses}")
    print(f"    n_lines  = {data.n_lines}")
    print(f"    n_periods= {data.n_periods}")

    print("\nBuilding Gurobi DAM UC model...")
    model, vars_dict = build_dam_model(data, M_p=m_penalty, enforce_lines=enforce_lines)
    print("  Model built. Starting optimization...")

    model.optimize()

    if model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        print(f"WARNING: Model did not terminate optimally. Status: {model.Status}")
        # You can decide here whether to raise or still try to extract something.
        # For now, we'll still attempt extraction if there is any solution.
        if model.SolCount == 0:
            raise RuntimeError("No feasible solution found by Gurobi.")

    results = extract_solution(data, model, vars_dict)
    print_brief_summary(results, data)

    return {
        "data": data,
        "model": model,
        "vars": vars_dict,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    outputs = run_rts_dam()

    # Example: write dispatch and commitment to CSV for inspection
    results = outputs["results"]
    u_df = results["u"]
    p_df = results["p"]

    out_dir = Path("dam_outputs")
    out_dir.mkdir(exist_ok=True)

    u_df.to_csv(out_dir / "commitment_u.csv")
    p_df.to_csv(out_dir / "dispatch_p.csv")

    print("\nWrote:")
    print(f"  {out_dir / 'commitment_u.csv'}")
    print(f"  {out_dir / 'dispatch_p.csv'}")
