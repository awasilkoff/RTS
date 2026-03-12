"""Shared utilities for runner scripts.

Functions extracted from run_comparison.py and run_reserve_then_daruc.py to
avoid cross-runner imports and reduce duplication.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_branch_flows import compute_branch_flows


# ---------------------------------------------------------------------------
# Reserve requirement from uncertainty set
# ---------------------------------------------------------------------------

def compute_reserve_from_uncertainty(Sigma, rho, T=None):
    """Compute spinning reserve requirement from ellipsoidal uncertainty set.

    R[t] = rho[t] * sqrt(1^T Sigma[t] 1) -- worst-case total wind shortfall.

    Parameters
    ----------
    Sigma : (T, K, K) or (K, K) array -- covariance matrices per period or static
    rho : scalar or (T,) array -- ellipsoid radii
    T : int, optional -- number of periods (required when Sigma is 2D static)

    Returns
    -------
    R : (T,) array -- reserve requirement per period (MW)
    """
    if Sigma.ndim == 2:
        if T is None:
            raise ValueError("T must be provided when Sigma is 2D (static)")
        Sigma = np.broadcast_to(Sigma[None, :, :], (T, Sigma.shape[0], Sigma.shape[1]))

    rho_arr = np.atleast_1d(rho)
    n_periods = Sigma.shape[0]
    if rho_arr.shape[0] == 1:
        rho_arr = np.full(n_periods, rho_arr[0])
    ones = np.ones(Sigma.shape[1])
    R = np.array([rho_arr[t] * np.sqrt(ones @ Sigma[t] @ ones) for t in range(n_periods)])
    return R


# ---------------------------------------------------------------------------
# Line flow analysis
# ---------------------------------------------------------------------------

def save_line_flow_analysis(data, dispatch_arr, margin_df, label, out_dir):
    """Save line flow decomposition CSVs for a single model run.

    Outputs (all L x T DataFrames):
      - line_flow_nominal.csv   : PTDF-based nominal flows (MW)
      - line_flow_analysis.csv  : long-format table with columns:
            line, period, flow_nominal, margin, worst_case_abs,
            Fmax, loading_nominal, loading_worst_case, binding

    Parameters
    ----------
    data : DAMData (must have PTDF, Fmax, line_ids, etc.)
    dispatch_arr : (I, T) array — p0 for ARUC/DARUC, p for DAM
    margin_df : DataFrame (L x T) of rho*z_line margins, or None (DAM)
    label : str — model label for printing
    out_dir : Path — directory to write CSVs
    """
    flow_df = compute_branch_flows(data, dispatch_arr)
    flow_df.to_csv(out_dir / "line_flow_nominal.csv")

    L = len(data.line_ids)
    T = len(data.time)
    Fmax = data.Fmax

    flow_vals = flow_df.values  # (L, T)
    margin_vals = margin_df.values if margin_df is not None else np.zeros((L, T))

    rows = []
    for l_idx in range(L):
        for t_idx in range(T):
            f_nom = flow_vals[l_idx, t_idx]
            m = margin_vals[l_idx, t_idx]
            fmax = Fmax[l_idx]
            wc_abs = abs(f_nom) + m
            rows.append({
                "line": data.line_ids[l_idx],
                "period": data.time[t_idx],
                "flow_nominal": round(f_nom, 2),
                "margin_rho_norm": round(m, 2),
                "worst_case_abs_flow": round(wc_abs, 2),
                "Fmax": round(fmax, 2),
                "loading_nominal_pct": round(abs(f_nom) / fmax * 100, 1) if fmax > 0 else 0.0,
                "loading_worst_case_pct": round(wc_abs / fmax * 100, 1) if fmax > 0 else 0.0,
                "binding": wc_abs >= fmax - 1.0,
            })

    analysis_df = pd.DataFrame(rows)
    analysis_df.to_csv(out_dir / "line_flow_analysis.csv", index=False)

    # Print summary of binding lines
    binding = analysis_df[analysis_df["binding"]]
    n_binding = binding.groupby("line").size()
    if len(n_binding) > 0:
        print(f"  [{label}] {len(n_binding)} lines binding at least one period, "
              f"{len(binding)} total (line,period) pairs")
        top = n_binding.nlargest(5)
        for lid, cnt in top.items():
            print(f"    {lid}: binding {cnt}/{T} periods")
    else:
        print(f"  [{label}] No binding lines")

    # Summary of margin reservation (robust models only)
    if margin_df is not None and margin_vals.max() > 0:
        active = margin_vals > 0.01
        if active.any():
            fmax_2d = Fmax[:, None] * np.ones((1, T))
            pct_reserved = margin_vals[active] / fmax_2d[active] * 100
            print(f"  [{label}] Margin reservation (where active): "
                  f"mean={pct_reserved.mean():.1f}%, max={pct_reserved.max():.1f}%")

    return flow_df, analysis_df


# ---------------------------------------------------------------------------
# Output saving helpers
# ---------------------------------------------------------------------------

def save_robust_outputs(results, data, out_dir, Sigma, rho,
                        deviation_df=None, margin_df=None,
                        summary_dict=None, analyze_z_fn=None):
    """Save standard robust model (ARUC/DARUC) outputs to a directory.

    Saves: commitment_u.csv, dispatch_p0.csv, Z_coefficients.csv,
           Sigma.npy, rho.npy, summary.json, and optionally
           deviation_summary.csv, line_margin.csv, Z_analysis.

    Parameters
    ----------
    results : dict with keys "u", "p0", "Z", "obj"
    data : DAMData
    out_dir : Path — directory to save to (must exist)
    Sigma : array — covariance matrix(es)
    rho : scalar or array — ellipsoid radius
    deviation_df : DataFrame, optional — DARUC deviation from DAM
    margin_df : DataFrame, optional — line margins (rho * z_line)
    summary_dict : dict, optional — metadata for summary.json
    analyze_z_fn : callable, optional — function(Z, data, out_dir, rho=rho)
    """
    results["u"].to_csv(out_dir / "commitment_u.csv")
    results["p0"].to_csv(out_dir / "dispatch_p0.csv")
    results["Z"].to_csv(out_dir / "Z_coefficients.csv")
    if deviation_df is not None:
        deviation_df.to_csv(out_dir / "deviation_summary.csv", index=False)
    np.save(out_dir / "Sigma.npy", Sigma)
    np.save(out_dir / "rho.npy", np.atleast_1d(rho))
    if analyze_z_fn is not None:
        analyze_z_fn(results["Z"], data, out_dir, rho=rho)
    if margin_df is not None:
        margin_df.to_csv(out_dir / "line_margin.csv")
    if summary_dict is not None:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2)


def save_dam_outputs(results, out_dir, summary_dict=None,
                     reserve_R=None, dispatch_key="p"):
    """Save DAM or DAM+Reserve outputs to a directory.

    Saves: commitment_u.csv, dispatch_p0.csv, summary.json,
           and optionally reserve_requirement.npy.

    Parameters
    ----------
    results : dict with keys "u", dispatch_key, "obj"
    out_dir : Path — directory to save to (must exist)
    summary_dict : dict, optional — metadata for summary.json
    reserve_R : array, optional — reserve requirement R[t]
    dispatch_key : str — key for dispatch in results ("p" for DAM, "p0" for ARUC)
    """
    results["u"].to_csv(out_dir / "commitment_u.csv")
    results[dispatch_key].to_csv(out_dir / "dispatch_p0.csv")
    if reserve_R is not None:
        np.save(out_dir / "reserve_requirement.npy", reserve_R)
    if summary_dict is not None:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2)


def save_line_flows_if_enabled(enforce_lines, data_full, dispatch_arr,
                               margin_df, label, parent_dir, subdir_name):
    """Conditionally save line flow analysis to a subdirectory.

    Handles the common pattern: check enforce_lines, mkdir, reindex margin
    to full line set, call save_line_flow_analysis.

    Parameters
    ----------
    enforce_lines : bool — skip if False
    data_full : DAMData — unfiltered data with full line set
    dispatch_arr : (I, T) array
    margin_df : DataFrame or None — line margins on filtered lines
    label : str — model label for printing
    parent_dir : Path — parent directory (e.g. daruc_dir)
    subdir_name : str — subdirectory name (e.g. "daruc_line_flows")
    """
    if not enforce_lines:
        return
    margin_full = None
    if margin_df is not None:
        margin_full = margin_df.reindex(data_full.line_ids, fill_value=0.0)
    flow_dir = parent_dir / subdir_name
    flow_dir.mkdir(exist_ok=True)
    save_line_flow_analysis(data_full, dispatch_arr, margin_full, label, flow_dir)


def compute_day1_metrics(data, results_dict):
    """Compute day-1 unit-hours and wind curtailment for a model's results.

    Parameters
    ----------
    data : DAMData
    results_dict : dict — must have "u" DataFrame and dispatch DataFrame
                   (keyed as "p0" or "p")

    Returns
    -------
    dict with keys: unit_hours, wind_curtailment_mwh
    """
    d1_mask = data.day1_period_mask()
    d1_times = [t for t, m in zip(data.time, d1_mask) if m]
    dt = data.dt[d1_mask]

    u = np.round(results_dict["u"][d1_times].values)
    uh = float((u * dt).sum())

    dispatch_key = "p0" if "p0" in results_dict else "p"
    dispatch = results_dict[dispatch_key][d1_times].values

    is_wind = [i for i, gt in enumerate(data.gen_type) if gt.upper() == "WIND"]
    Pmax_2d = data.Pmax_2d()
    wind_pmax_d1 = Pmax_2d[np.ix_(is_wind, d1_mask)]
    curt = float(((wind_pmax_d1 - dispatch[is_wind, :]) * dt).sum())

    return {"unit_hours": uh, "wind_curtailment_mwh": curt}
