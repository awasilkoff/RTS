"""
run_rts_daruc.py

Two-step Day-Ahead Robust Unit Commitment (Setup 1 from the paper):

  Step 1 (Noon):  Deterministic DA-UC  ->  u_DAM commitments
  Step 2 (5 PM):  DARUC adds reliability commitments under uncertainty
                  with constraint u >= u_DAM (can only add, never decommit)

Uses:
  - run_rts_dam.py   for deterministic DAM (Step 1)
  - aruc_model.py    for robust DARUC (Step 2) with dam_commitment parameter
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import gurobipy as gp

from models import DAMData
from aruc_model import build_aruc_ldr_model, align_uncertainty_to_aruc
from run_rts_dam import run_rts_dam
from run_rts_aruc import (
    build_uncertainty_set,
    extract_solution,
    print_brief_summary,
    analyze_Z_patterns,
)
from uncertainty_set_provider import UncertaintySetProvider


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"

START_TIME = pd.Timestamp(year=2020, month=1, day=1, hour=0)
HORIZON_HOURS = 48

M_PENALTY = 1e4
UNCERTAINTY_RHO = 0.5

# SPP forecast override for wind Pmax (set to None to use DAY_AHEAD_wind.csv)
SPP_FORECASTS_PARQUET = Path(
    "uncertainty_sets_refactored/data/forecasts_filtered_rts4_constellation_v2.parquet"
)
SPP_START_IDX = 0


# ---------------------------------------------------------------------------
# DAM commitment extraction
# ---------------------------------------------------------------------------


def extract_dam_commitment(
    dam_results: Dict[str, Any],
    data: DAMData,
) -> Dict[str, np.ndarray]:
    """
    Extract DAM commitment arrays (u, v, w) for use as DARUC input.

    The DAM model exposes u and p; we derive v and w from u transitions:
        v[i,t] = max(0, u[i,t] - u[i,t-1])   (startup)
        w[i,t] = max(0, u[i,t-1] - u[i,t])   (shutdown)

    Parameters
    ----------
    dam_results : dict
        Results from run_rts_dam(), must contain "u" DataFrame.
    data : DAMData
        DAM data object (for initial conditions).

    Returns
    -------
    dam_commitment : dict
        {"u": (I, T) array, "v": (I, T) array, "w": (I, T) array}
    """
    u_df = dam_results["u"]
    u_dam = np.round(u_df.values).astype(float)  # (I, T), round to clean 0/1

    I, T = u_dam.shape
    v_dam = np.zeros_like(u_dam)
    w_dam = np.zeros_like(u_dam)

    # t = 0: use initial conditions
    for i in range(I):
        v_dam[i, 0] = max(0.0, u_dam[i, 0] - data.u_init[i])
        w_dam[i, 0] = max(0.0, data.u_init[i] - u_dam[i, 0])

    # t >= 1: transitions
    for t in range(1, T):
        v_dam[:, t] = np.maximum(0.0, u_dam[:, t] - u_dam[:, t - 1])
        w_dam[:, t] = np.maximum(0.0, u_dam[:, t - 1] - u_dam[:, t])

    return {"u": u_dam, "v": v_dam, "w": w_dam}


# ---------------------------------------------------------------------------
# Deviation analysis
# ---------------------------------------------------------------------------


def analyze_deviations(
    data: DAMData,
    model: gp.Model,
    vars_dict: Dict[str, Any],
    dam_commitment: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Analyze DARUC deviations from DAM commitments.

    Returns a DataFrame summarizing which generators were additionally
    committed by the DARUC step.
    """
    gen_ids = data.gen_ids
    I = data.n_gens
    T = data.n_periods

    u_prime = vars_dict["u_prime"]

    rows = []
    for i in range(I):
        extra_hours = 0
        periods_added = []
        for t in range(T):
            val = u_prime[i, t].X
            if val > 0.5:
                extra_hours += 1
                periods_added.append(t)

        if extra_hours > 0:
            rows.append(
                {
                    "gen_id": gen_ids[i],
                    "gen_type": data.gen_type[i],
                    "extra_committed_hours": extra_hours,
                    "dam_committed_hours": int(dam_commitment["u"][i, :].sum()),
                    "daruc_committed_hours": int(
                        dam_commitment["u"][i, :].sum() + extra_hours
                    ),
                    "periods_added": str(periods_added),
                }
            )

    if rows:
        dev_df = pd.DataFrame(rows)
    else:
        dev_df = pd.DataFrame(
            columns=[
                "gen_id",
                "gen_type",
                "extra_committed_hours",
                "dam_committed_hours",
                "daruc_committed_hours",
                "periods_added",
            ]
        )

    return dev_df


def print_deviation_summary(
    dev_df: pd.DataFrame,
    dam_obj: float,
    daruc_obj: float,
) -> None:
    """Print a human-readable deviation summary."""
    print("\n" + "=" * 70)
    print("DARUC DEVIATION ANALYSIS (DARUC vs DAM)")
    print("=" * 70)

    print(f"\nDAM objective:   {dam_obj:>14,.2f}")
    print(f"DARUC objective: {daruc_obj:>14,.2f}")
    diff = daruc_obj - dam_obj
    pct = 100 * diff / dam_obj if dam_obj != 0 else float("inf")
    print(f"Cost increase:   {diff:>14,.2f}  ({pct:+.2f}%)")

    if dev_df.empty:
        print("\nNo additional commitments made by DARUC.")
    else:
        total_extra = dev_df["extra_committed_hours"].sum()
        print(f"\nGenerators with additional commitments: {len(dev_df)}")
        print(f"Total extra unit-hours committed: {total_extra}")
        print(f"\nDetails:")
        print(dev_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main two-step pipeline
# ---------------------------------------------------------------------------


def run_rts_daruc(
    source_dir: Path = SOURCE_DIR,
    ts_dir: Path = TS_DIR,
    start_time: pd.Timestamp = START_TIME,
    horizon_hours: int = HORIZON_HOURS,
    m_penalty: float = M_PENALTY,
    rho: float = UNCERTAINTY_RHO,
    wind_std_fraction: float = 0.15,
    uncertainty_provider_path: Optional[Union[Path, str]] = None,
    provider_start_idx: int = 0,
    spp_forecasts_parquet: Optional[Path] = SPP_FORECASTS_PARQUET,
    spp_start_idx: int = SPP_START_IDX,
    enforce_lines: bool = True,
    rho_lines_frac: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Two-step DARUC pipeline (Setup 1):

      Step 1: Deterministic DA-UC  ->  u_DAM
      Step 2: DARUC with u >= u_DAM  ->  additional reliability commitments

    Parameters
    ----------
    source_dir, ts_dir, start_time, horizon_hours, m_penalty :
        Same as run_rts_dam / run_rts_aruc.
    rho : float
        Ellipsoid radius (used for static uncertainty).
    wind_std_fraction : float
        Wind std dev fraction (used for static uncertainty).
    uncertainty_provider_path : Path or str, optional
        Path to pre-computed uncertainty NPZ for time-varying mode.
    provider_start_idx : int
        Starting index for provider query.

    Returns
    -------
    outputs : dict
        {
          "dam_outputs": dict from run_rts_dam(),
          "daruc_results": dict (u, p0, Z, obj),
          "deviation_summary": pd.DataFrame,
          "data": DAMData,
        }
    """

    # ==================================================================
    # STEP 1: Deterministic DA-UC
    # ==================================================================
    print("=" * 70)
    print("STEP 1: DETERMINISTIC DAY-AHEAD UC")
    print("=" * 70)

    dam_outputs = run_rts_dam(
        source_dir=source_dir,
        ts_dir=ts_dir,
        start_time=start_time,
        horizon_hours=horizon_hours,
        m_penalty=m_penalty,
        spp_forecasts_parquet=spp_forecasts_parquet,
        spp_start_idx=spp_start_idx,
        enforce_lines=enforce_lines,
    )

    dam_results = dam_outputs["results"]
    data = dam_outputs["data"]

    # ==================================================================
    # Extract DAM commitments
    # ==================================================================
    print("\nExtracting DAM commitments for DARUC input...")
    dam_commitment = extract_dam_commitment(dam_results, data)

    dam_u_hours = dam_commitment["u"].sum()
    dam_startups = dam_commitment["v"].sum()
    print(f"  Total DAM unit-hours: {dam_u_hours:.0f}")
    print(f"  Total DAM startups:   {dam_startups:.0f}")

    # ==================================================================
    # STEP 2: DARUC — Robust reliability commitments
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: DARUC (ROBUST RELIABILITY COMMITMENT)")
    print("=" * 70)

    # Build uncertainty set
    time_varying = False
    sqrt_Sigma = None

    if uncertainty_provider_path is not None:
        print(f"\nLoading time-varying uncertainty from {uncertainty_provider_path}...")
        provider = UncertaintySetProvider.from_npz(uncertainty_provider_path)
        horizon = provider.get_by_indices(
            provider_start_idx, horizon_hours, compute_sqrt=True
        )
        Sigma, rho_arr, sqrt_Sigma = align_uncertainty_to_aruc(
            horizon, data, provider.get_wind_gen_ids()
        )
        time_varying = True
        print(f"  Sigma shape: {Sigma.shape}")
        print(f"  rho range: [{rho_arr.min():.3f}, {rho_arr.max():.3f}]")
        model_name = "DARUC_TimeVarying"
        rho_val = rho_arr
    else:
        print("\nConstructing static uncertainty set...")
        Sigma, rho_val = build_uncertainty_set(
            data, rho=rho, wind_std_fraction=wind_std_fraction
        )
        model_name = "DARUC_RTS"

    print("\nBuilding Gurobi DARUC model (with DAM commitment floor)...")
    model, vars_dict = build_aruc_ldr_model(
        data=data,
        Sigma=Sigma,
        rho=rho_val,
        rho_lines_frac=rho_lines_frac,
        sqrt_Sigma=sqrt_Sigma,
        M_p=m_penalty,
        model_name=model_name,
        dam_commitment=dam_commitment,
        enforce_lines=enforce_lines,
    )
    print("  Model built. Starting optimization...")

    model.optimize()

    if model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        print(f"WARNING: DARUC did not terminate optimally. Status: {model.Status}")
        if model.SolCount == 0:
            raise RuntimeError("No feasible DARUC solution found.")

    daruc_results = extract_solution(data, model, vars_dict)
    print_brief_summary(daruc_results, data)
    analyze_Z_patterns(daruc_results["Z"], data)

    # ==================================================================
    # Deviation analysis
    # ==================================================================
    dev_df = analyze_deviations(data, model, vars_dict, dam_commitment)
    print_deviation_summary(dev_df, dam_results["obj"], daruc_results["obj"])

    # Verify u_daruc >= u_dam
    u_daruc = daruc_results["u"].values
    u_dam = dam_commitment["u"]
    violations = (u_daruc < u_dam - 0.5).sum()
    if violations > 0:
        print(f"\nWARNING: {violations} violations of u_DARUC >= u_DAM!")
    else:
        print("\nVerified: u_DARUC >= u_DAM for all (i, t).")

    return {
        "dam_outputs": dam_outputs,
        "daruc_results": daruc_results,
        "deviation_summary": dev_df,
        "data": data,
        "model": model,
        "vars": vars_dict,
        "Sigma": Sigma,
        "rho": rho_val,
        "rho_lines_frac": rho_lines_frac,
        "time_varying": time_varying,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    outputs = run_rts_daruc()

    # Write results to CSV
    daruc_results = outputs["daruc_results"]
    dev_df = outputs["deviation_summary"]

    out_dir = Path("daruc_outputs")
    out_dir.mkdir(exist_ok=True)

    daruc_results["u"].to_csv(out_dir / "commitment_u_daruc.csv")
    daruc_results["p0"].to_csv(out_dir / "dispatch_p0_daruc.csv")
    daruc_results["Z"].to_csv(out_dir / "ldr_coefficients_Z_daruc.csv")
    dev_df.to_csv(out_dir / "deviation_summary.csv", index=False)

    print("\nWrote:")
    print(f"  {out_dir / 'commitment_u_daruc.csv'}")
    print(f"  {out_dir / 'dispatch_p0_daruc.csv'}")
    print(f"  {out_dir / 'ldr_coefficients_Z_daruc.csv'}")
    print(f"  {out_dir / 'deviation_summary.csv'}")
