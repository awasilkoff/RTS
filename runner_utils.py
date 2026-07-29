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
# Generator metadata from gen.csv
# ---------------------------------------------------------------------------

_GEN_CSV_PATH = Path(__file__).parent / "RTS_Data" / "SourceData" / "gen.csv"

_GEN_META_COLS = ["Fuel", "Category", "PMax MW", "PMin MW"]


def load_gen_metadata(gen_csv=None):
    """Load generator metadata from gen.csv, indexed by GEN UID.

    Returns a DataFrame with columns: Fuel, Category, PMax MW, PMin MW.
    Returns None if the file cannot be read.
    """
    path = Path(gen_csv) if gen_csv else _GEN_CSV_PATH
    try:
        df = pd.read_csv(path)
        df = df.set_index("GEN UID")[_GEN_META_COLS]
        return df
    except Exception:
        return None


def _enrich_deviation_df(dev_df, gen_meta):
    """Add fuel, category, pmax_mw, pmin_mw columns to a deviation DataFrame.

    Merges on gen_id.  If gen_meta is None, adds empty columns.
    """
    if gen_meta is not None and not dev_df.empty:
        merged = dev_df.merge(
            gen_meta.rename(columns={
                "Fuel": "fuel",
                "Category": "category",
                "PMax MW": "pmax_mw",
                "PMin MW": "pmin_mw",
            }),
            left_on="gen_id",
            right_index=True,
            how="left",
        )
        return merged
    else:
        for col in ("fuel", "category", "pmax_mw", "pmin_mw"):
            if col not in dev_df.columns:
                dev_df[col] = ""
        return dev_df


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
# Reserve equivalent from LDR under worst-case total wind shortfall
# ---------------------------------------------------------------------------

def compute_reserve_equivalent(results, data, Sigma, rho):
    """Compute per-generator reserve equivalent under worst-case total wind shortfall.

    The worst-case direction that maximizes total wind shortfall over the
    ellipsoid {r : r^T Sigma^{-1} r <= rho^2} is:

        r_wc[t] = rho[t] * Sigma[t] @ e / sqrt(e^T Sigma[t] e)

    The reserve equivalent for generator i at period t is then:

        reserve_eq[i,t] = Z[i,t,:] @ r_wc[t,:]

    Parameters
    ----------
    results : dict — must contain "Z" (DataFrame with MultiIndex columns (time, k))
    data : DAMData
    Sigma : (K, K) or (T, K, K) array — covariance matrices
    rho : scalar or (T,) array — ellipsoid radii

    Returns
    -------
    reserve_df : DataFrame (gen_ids x time) — per-generator reserve equivalent (MW)
    stats : dict — summary statistics for inclusion in summary.json
    """
    Z_df = results["Z"]
    gen_ids = data.gen_ids
    time_labels = data.time
    I = len(gen_ids)
    T = len(time_labels)

    # Normalize Sigma to (T, K, K)
    if Sigma.ndim == 2:
        Sigma_3d = np.broadcast_to(Sigma[None, :, :], (T, Sigma.shape[0], Sigma.shape[1]))
    else:
        Sigma_3d = Sigma

    K = Sigma_3d.shape[1]

    # Normalize rho to (T,)
    rho_arr = np.atleast_1d(rho).astype(float)
    if rho_arr.shape[0] == 1:
        rho_arr = np.full(T, rho_arr[0])

    # Compute worst-case deviation vector per period
    ones = np.ones(K)
    r_wc = np.zeros((T, K))
    for t in range(T):
        Se = Sigma_3d[t] @ ones
        denom = np.sqrt(ones @ Se)
        if denom > 1e-12:
            r_wc[t] = rho_arr[t] * Se / denom

    # Extract Z as (I, T, K) array from MultiIndex DataFrame
    Z_arr = np.zeros((I, T, K))
    for i in range(I):
        for t in range(T):
            t_label = time_labels[t]
            for k in range(K):
                col = (t_label, k)
                if col in Z_df.columns:
                    Z_arr[i, t, k] = Z_df.iloc[i][col]

    # Reserve equivalent: -Z[i,t,:] @ r_wc[t,:]
    # Negated because the LDR is p(r) = p0 + Z @ r where r is wind shortfall
    # (positive).  Thermal Z values are negative (they ramp UP to compensate),
    # so -Z @ r_wc gives the positive upward reserve contribution for thermals
    # and negative for wind (which loses output under shortfall).
    reserve_eq = -np.einsum("itk,tk->it", Z_arr, r_wc)

    reserve_df = pd.DataFrame(reserve_eq, index=gen_ids, columns=time_labels)

    # Summary statistics
    system_total = reserve_eq.sum(axis=0)  # (T,) — should sum to ~0 by power balance
    thermal_mask = [gt == "THERMAL" for gt in data.gen_type]
    thermal_total = reserve_eq[thermal_mask].sum(axis=0)  # (T,)
    wind_mask = [gt.upper() == "WIND" for gt in data.gen_type]
    wind_total = reserve_eq[wind_mask].sum(axis=0)

    stats = {
        "reserve_equivalent": {
            "thermal_total_min_mw": round(float(thermal_total.min()), 2),
            "thermal_total_max_mw": round(float(thermal_total.max()), 2),
            "thermal_total_mean_mw": round(float(thermal_total.mean()), 2),
            "wind_total_min_mw": round(float(wind_total.min()), 2),
            "wind_total_max_mw": round(float(wind_total.max()), 2),
            "wind_total_mean_mw": round(float(wind_total.mean()), 2),
            "system_balance_max_abs_mw": round(float(np.abs(system_total).max()), 4),
        },
    }

    return reserve_df, stats


# ---------------------------------------------------------------------------
# Worst-case total-shortfall line flows
# ---------------------------------------------------------------------------

def compute_worst_case_total_shortfall_flows(
    data, p0_arr, Sigma, rho, Z_arr=None, r_arr=None,
):
    """Compute line flows under the worst-case total wind shortfall scenario.

    Given the worst-case deviation vector:
        r_wc[t] = rho[t] * Sigma[t] @ e / sqrt(e^T Sigma[t] e)

    For DARUC/ARUC (Z_arr provided):
        p_wc[i,t] = p0[i,t] + Z[i,t,:] @ r_wc[t,:]
        (LDR adjusts all generators — thermals ramp up, wind curtails)

    For DAM+Reserve (r_arr provided, no Z):
        Wind loses: p_wc[wind_k,t] = p0[wind_k,t] - r_wc[t,k]
        Thermals deploy: p_wc[i,t] = p0[i,t] + r[i,t] * R[t] / sum_j(r[j,t])
        (Reserves deployed proportionally; may exceed sum -> capped to r[i,t])

    Parameters
    ----------
    data : DAMData — must have PTDF, gen_to_bus, d, gen_type, Fmax
    p0_arr : (I, T) array — nominal dispatch
    Sigma : (K, K) or (T, K, K) — covariance
    rho : scalar or (T,) — ellipsoid radii
    Z_arr : (I, T, K) array or None — LDR coefficients (DARUC/ARUC)
    r_arr : (I, T) array or None — explicit reserves (DAM+Reserve, thermal-only)

    Returns
    -------
    dict with keys:
        flow_nominal : (L, T) array — PTDF @ (p0 - load)
        flow_wc : (L, T) array — PTDF @ (p_wc - load)
        p_wc : (I, T) array — worst-case dispatch per generator
        r_wc : (T, K) array — worst-case deviation vector
        violations : list of (line_idx, period, excess_mw, flow_mw)
    """
    I, T = p0_arr.shape
    is_wind = np.array([gt.upper() == "WIND" for gt in data.gen_type])
    wind_idx = np.where(is_wind)[0]
    K = len(wind_idx)

    # Normalize Sigma/rho
    if Sigma.ndim == 2:
        Sigma_3d = np.broadcast_to(Sigma[None, :, :], (T, K, K))
    else:
        Sigma_3d = Sigma
    rho_arr = np.atleast_1d(rho).astype(float)
    if rho_arr.shape[0] == 1:
        rho_arr = np.full(T, rho_arr[0])

    # Compute worst-case deviation: r_wc[t] = rho[t] * Sigma[t] @ e / sqrt(e^T Sigma[t] e)
    ones = np.ones(K)
    r_wc = np.zeros((T, K))
    R_total = np.zeros(T)
    for t in range(T):
        Se = Sigma_3d[t] @ ones
        denom = np.sqrt(ones @ Se)
        if denom > 1e-12:
            r_wc[t] = rho_arr[t] * Se / denom
            R_total[t] = rho_arr[t] * denom

    # Compute worst-case dispatch
    p_wc = p0_arr.copy()

    # Pmax for clamping (handle both (I,) and (I,T) shapes)
    Pmax = np.array(data.Pmax, dtype=float)
    if Pmax.ndim == 1:
        Pmax_2d = np.broadcast_to(Pmax[:, None], (I, T))
    else:
        Pmax_2d = Pmax

    if Z_arr is not None:
        # DARUC/ARUC: p_wc = p0 - Z @ r_wc
        # The LDR is p(r) = p0 + Z @ r.  The worst-case total shortfall
        # direction is r = -r_wc (wind produces less), so:
        #   p_adjusted = p0 + Z @ (-r_wc) = p0 - Z @ r_wc
        # Thermals go UP (Z negative -> -Z positive), wind goes DOWN
        # (Z ~identity -> -Z ~negative).
        #
        # The ARUC model's SOC constraints ensure p_wc ∈ [Pmin, Pmax]
        # for the solved Z.  Do NOT clip here -- even tiny adjustments
        # break sum_i Z[i,t,k] = 0 (power balance response) and create
        # spurious line-flow violations.
        for t in range(T):
            p_wc[:, t] = p0_arr[:, t] - Z_arr[:, t, :] @ r_wc[t, :]

    elif r_arr is not None:
        # DAM+Reserve: no adaptive policy.
        #
        # Wind curtailment matters: if wind is already dispatched below
        # forecast (p0 < Pmax), the effective shortfall from dispatched
        # level is less than r_wc.  Actual wind under worst case:
        #   p_wc[wind_k] = max(0, Pmax[wind_k] - r_wc[k])
        # but can't exceed what was dispatched (curtailed):
        #   p_wc[wind_k] = min(p0[wind_k], max(0, Pmax[wind_k] - r_wc[k]))
        # Effective reduction from dispatch:
        #   delta_wind[k] = p0[wind_k] - p_wc[wind_k]
        actual_net_shortfall = np.zeros(T)
        for k, i in enumerate(wind_idx):
            for t in range(T):
                available_wc = max(0.0, Pmax_2d[i, t] - r_wc[t, k])
                p_wc[i, t] = min(p0_arr[i, t], available_wc)
                actual_net_shortfall[t] += p0_arr[i, t] - p_wc[i, t]

        # Thermals deploy reserves proportionally to cover the actual
        # net shortfall (which may be < R_total due to wind curtailment)
        is_thermal = np.array([gt == "THERMAL" for gt in data.gen_type])
        for t in range(T):
            if actual_net_shortfall[t] < 1e-6:
                continue
            thermal_reserve_total = r_arr[is_thermal, t].sum()
            if thermal_reserve_total < 1e-6:
                continue
            # Deploy just enough to cover actual shortfall
            scale = min(actual_net_shortfall[t] / thermal_reserve_total, 1.0)
            for i in np.where(is_thermal)[0]:
                deployed = r_arr[i, t] * scale
                p_wc[i, t] = min(p0_arr[i, t] + deployed, Pmax_2d[i, t])
    else:
        raise ValueError("Must provide either Z_arr (DARUC) or r_arr (DAM+Reserve)")

    # Compute flows via PTDF
    from compute_branch_flows import compute_branch_flows
    flow_nom_df = compute_branch_flows(data, p0_arr)
    flow_wc_df = compute_branch_flows(data, p_wc)

    flow_nom = flow_nom_df.values
    flow_wc = flow_wc_df.values
    Fmax = data.Fmax

    # Detect violations
    violations = []
    for l in range(len(Fmax)):
        for t in range(T):
            excess = abs(flow_wc[l, t]) - Fmax[l]
            if excess > 1.0:
                violations.append((l, t, excess, flow_wc[l, t]))

    return {
        "flow_nominal": flow_nom,
        "flow_wc": flow_wc,
        "p_wc": p_wc,
        "r_wc": r_wc,
        "R_total": R_total,
        "violations": violations,
    }


def save_worst_case_flow_analysis(data, wc_result, label, out_dir):
    """Save worst-case total-shortfall flow analysis CSV.

    Columns: line, period, flow_nominal, flow_wc, Fmax,
             loading_nominal_pct, loading_wc_pct, violation, excess_mw
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flow_nom = wc_result["flow_nominal"]
    flow_wc = wc_result["flow_wc"]
    Fmax = data.Fmax
    L, T = flow_nom.shape

    rows = []
    for l in range(L):
        for t in range(T):
            f_nom = flow_nom[l, t]
            f_wc = flow_wc[l, t]
            fmax = Fmax[l]
            excess = max(abs(f_wc) - fmax, 0.0)
            rows.append({
                "line": data.line_ids[l],
                "period": data.time[t],
                "flow_nominal": round(f_nom, 2),
                "flow_wc": round(f_wc, 2),
                "Fmax": round(fmax, 2),
                "loading_nominal_pct": round(abs(f_nom) / fmax * 100, 1) if fmax > 0 else 0.0,
                "loading_wc_pct": round(abs(f_wc) / fmax * 100, 1) if fmax > 0 else 0.0,
                "violation": abs(f_wc) > fmax + 1.0,
                "excess_mw": round(excess, 2),
            })

    df = pd.DataFrame(rows)
    fname = out_dir / f"worst_case_flow_analysis_{label}.csv"
    df.to_csv(fname, index=False)

    # Summary
    viols = df[df["violation"]]
    n_lines = viols["line"].nunique()
    n_pairs = len(viols)
    if n_pairs > 0:
        print(f"  [{label}] {n_lines} lines violated ({n_pairs} line-period pairs), "
              f"max excess {viols['excess_mw'].max():.1f} MW")
    else:
        print(f"  [{label}] No line violations under worst-case total shortfall")

    return df


# ---------------------------------------------------------------------------
# Output saving helpers
# ---------------------------------------------------------------------------

def rebuild_deviation_summary(case_dir):
    """Rebuild deviation_summary.csv from DAM and DARUC commitment CSVs.

    This avoids re-running the full pipeline when only the summary is needed.
    Looks for the DAM baseline commitment in this order:
      1. daruc/dam_commitment_u.csv  (saved alongside DARUC output)
      2. dam_reserve/commitment_u.csv (reserve baseline layout)
      3. dam/commitment_u.csv (plain DAM layout)

    Parameters
    ----------
    case_dir : Path — root directory containing daruc/ subdir (and DAM baseline)

    Returns
    -------
    pd.DataFrame — the deviation summary (also saved to daruc/deviation_summary.csv)
    """
    case_dir = Path(case_dir)
    daruc_u = pd.read_csv(case_dir / "daruc" / "commitment_u.csv", index_col=0)

    # Find DAM baseline commitment
    dam_candidates = [
        case_dir / "daruc" / "dam_commitment_u.csv",
        case_dir / "dam_reserve" / "commitment_u.csv",
        case_dir / "dam" / "commitment_u.csv",
    ]
    dam_path = None
    for candidate in dam_candidates:
        if candidate.exists():
            dam_path = candidate
            break
    if dam_path is None:
        raise FileNotFoundError(
            f"No DAM baseline commitment found in {case_dir}. "
            f"Searched: {[str(c) for c in dam_candidates]}"
        )
    dam_u = pd.read_csv(dam_path, index_col=0)
    print(f"  DAM baseline: {dam_path.relative_to(case_dir)}")

    # Binarize (handle -0.0 and float noise)
    dam_arr = (dam_u.values > 0.5).astype(int)
    daruc_arr = (daruc_u.values > 0.5).astype(int)
    gen_ids = dam_u.index.tolist()

    # Extra commitment = DARUC on but DAM off
    extra = daruc_arr - dam_arr
    extra = np.clip(extra, 0, 1)  # only additions

    rows = []
    for i, gen_id in enumerate(gen_ids):
        periods_added = list(np.where(extra[i] > 0)[0])
        if not periods_added:
            continue

        extra_hours = len(periods_added)

        # Count extra startups: transitions from off->on in combined schedule
        # that weren't already on in the DAM schedule
        extra_startups = 0
        for t in periods_added:
            if t == 0:
                prev_on = False  # conservative: no initial state info available
            else:
                prev_on = daruc_arr[i, t - 1] > 0
            if not prev_on:
                extra_startups += 1

        rows.append({
            "gen_id": gen_id,
            "gen_type": "THERMAL",
            "extra_committed_hours": extra_hours,
            "extra_startups": extra_startups,
            "dam_committed_hours": int(dam_arr[i].sum()),
            "daruc_committed_hours": int(daruc_arr[i].sum()),
            "periods_added": str(periods_added),
        })

    if rows:
        dev_df = pd.DataFrame(rows)
    else:
        dev_df = pd.DataFrame(columns=[
            "gen_id", "gen_type", "extra_committed_hours", "extra_startups",
            "dam_committed_hours", "daruc_committed_hours", "periods_added",
        ])

    # Enrich with fuel type and capacity from gen.csv
    gen_meta = load_gen_metadata()
    dev_df = _enrich_deviation_df(dev_df, gen_meta)

    dev_df.to_csv(case_dir / "daruc" / "deviation_summary.csv", index=False)
    print(f"Rebuilt deviation_summary.csv: {len(dev_df)} generators with extra commitments")
    return dev_df


def save_robust_outputs(results, data, out_dir, Sigma, rho,
                        deviation_df=None, margin_df=None,
                        summary_dict=None, analyze_z_fn=None, mu=None):
    """Save standard robust model (ARUC/DARUC) outputs to a directory.

    Saves: commitment_u.csv, dispatch_p0.csv, Z_coefficients.csv,
           reserve_equivalent.csv, Sigma.npy, rho.npy, summary.json,
           and optionally deviation_summary.csv, line_margin.csv, Z_analysis.

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
    if mu is not None:
        np.save(out_dir / "mu.npy", mu)
    if analyze_z_fn is not None:
        analyze_z_fn(results["Z"], data, out_dir, rho=rho)
    if margin_df is not None:
        margin_df.to_csv(out_dir / "line_margin.csv")

    # Reserve equivalent under worst-case total wind shortfall
    reserve_df, reserve_stats = compute_reserve_equivalent(results, data, Sigma, rho)
    reserve_df.to_csv(out_dir / "reserve_equivalent.csv")
    if summary_dict is not None:
        summary_dict.update(reserve_stats)

    if summary_dict is not None:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2)


def save_dam_outputs(results, out_dir, summary_dict=None,
                     reserve_R=None, dispatch_key="p"):
    """Save DAM or DAM+Reserve outputs to a directory.

    Saves: commitment_u.csv, dispatch_p0.csv, summary.json,
           and optionally reserve_requirement.npy and reserve_distribution.csv.

    Parameters
    ----------
    results : dict with keys "u", dispatch_key, "obj", and optionally "r"
    out_dir : Path — directory to save to (must exist)
    summary_dict : dict, optional — metadata for summary.json
    reserve_R : array, optional — reserve requirement R[t]
    dispatch_key : str — key for dispatch in results ("p" for DAM, "p0" for ARUC)
    """
    results["u"].to_csv(out_dir / "commitment_u.csv")
    results[dispatch_key].to_csv(out_dir / "dispatch_p0.csv")
    if reserve_R is not None:
        np.save(out_dir / "reserve_requirement.npy", reserve_R)
    if "r" in results:
        results["r"].to_csv(out_dir / "reserve_distribution.csv")
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


def committed_units_day1(directory, day1_hours=24):
    """Count unique generators committed during day 1 across commitment_u.csv files.

    Searches *directory* recursively for files named ``commitment_u.csv``,
    reads each one, and counts the number of generators (rows) with u >= 0.5
    in at least one day-1 period.

    Day-1 periods are identified by taking the first *day1_hours* worth of
    columns.  For hourly schedules this is simply the first 24 columns; for
    variable-interval horizons the column timestamps are parsed and all
    columns within 24 hours of the first timestamp are included.

    Parameters
    ----------
    directory : str or Path
        Root directory to search (recursively) for commitment_u.csv files.
    day1_hours : int, optional
        Number of hours in day 1 (default 24).

    Returns
    -------
    dict[str, int]
        Mapping from the relative path of each commitment_u.csv (relative to
        *directory*) to the number of unique committed units in day 1.
    """
    directory = Path(directory)
    results = {}
    for csv_path in sorted(directory.rglob("commitment_u.csv")):
        u = pd.read_csv(csv_path, index_col=0)
        # Determine day-1 columns
        try:
            timestamps = pd.to_datetime(u.columns)
            t0 = timestamps[0]
            day1_cols = [c for c, ts in zip(u.columns, timestamps)
                         if ts < t0 + pd.Timedelta(hours=day1_hours)]
        except Exception:
            day1_cols = list(u.columns[:day1_hours])
        committed = (u[day1_cols].values >= 0.5).any(axis=1).sum()
        results[str(csv_path.relative_to(directory))] = int(committed)
    return results


def analyze_Z(Z_df: "pd.DataFrame", data, out_dir: Path, rho=None):
    """Analyze Z matrix structure: row sums, wind vs non-wind, per-period.

    Saves Z_analysis_full.csv and Z_analysis_per_gen.csv to out_dir.
    """
    gen_ids = list(Z_df.index)
    gen_type = data.gen_type
    time_labels = Z_df.columns.get_level_values("time").unique()
    K = Z_df.columns.get_level_values("k").unique().size

    rho_arr = np.atleast_1d(rho) if rho is not None else None
    time_varying_rho = rho_arr is not None and rho_arr.shape[0] > 1

    rows = []
    for t in time_labels:
        Z_t = Z_df[t].values
        for i, gid in enumerate(gen_ids):
            z_row = Z_t[i, :]
            rows.append({
                "gen_id": gid,
                "gen_type": gen_type[i],
                "time": t,
                "Z_row_sum": z_row.sum(),
                "Z_row_abs_sum": np.abs(z_row).sum(),
                "Z_row_norm": np.linalg.norm(z_row),
                **{f"Z_k{k}": z_row[k] for k in range(K)},
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "Z_analysis_full.csv", index=False)

    agg = (
        df.groupby(["gen_id", "gen_type"])
        .agg({
            "Z_row_sum": ["mean", "std", "min", "max"],
            "Z_row_abs_sum": ["mean", "max"],
            "Z_row_norm": ["mean", "max"],
        })
        .reset_index()
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg.to_csv(out_dir / "Z_analysis_per_gen.csv", index=False)

    print("\n" + "=" * 70)
    print("Z MATRIX ANALYSIS")
    print("=" * 70)

    for t_idx, t in enumerate(time_labels):
        Z_t = Z_df[t].values
        col_sums = Z_t.sum(axis=0)
        wind_mask = np.array([gt.upper() == "WIND" for gt in gen_type])
        wind_col_sums = Z_t[wind_mask].sum(axis=0)
        nonwind_col_sums = Z_t[~wind_mask].sum(axis=0)

        if time_varying_rho:
            rho_t = rho_arr[t_idx]
        elif rho_arr is not None:
            rho_t = float(rho_arr[0])
        else:
            rho_t = None
        rho_str = f"  rho={rho_t:.4f}" if rho_t is not None else ""

        print(f"\n  Period {t}:{rho_str}")
        print(f"    Column sums (all gens):  {col_sums}")
        print(f"    Column sums (wind only): {wind_col_sums}")
        print(f"    Column sums (non-wind):  {nonwind_col_sums}")

        wind_idx = np.where(wind_mask)[0]
        for wi in wind_idx:
            z_row = Z_t[wi, :]
            print(f"    {gen_ids[wi]:20s}  Z=[{', '.join(f'{v:+.4f}' for v in z_row)}]  sum={z_row.sum():+.4f}")

    nonwind_active = df[(df["gen_type"] != "WIND") & (df["Z_row_abs_sum"] > 1e-6)]
    if not nonwind_active.empty:
        print(f"\n  Non-wind generators with non-zero Z ({len(nonwind_active)} entries):")
        summary = (
            nonwind_active.groupby("gen_id")
            .agg({"Z_row_sum": "mean", "Z_row_abs_sum": "mean"})
            .sort_values("Z_row_abs_sum", ascending=False)
            .head(20)
        )
        print(summary.to_string())
    else:
        print("\n  No non-wind generators have non-zero Z.")

    print("=" * 70)
    return df


# ---------------------------------------------------------------------------
# Solve provenance
# ---------------------------------------------------------------------------

#: Gurobi status codes worth distinguishing in the saved artifacts.
_GRB_STATUS_NAMES = {
    2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
    6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
    14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


def solve_diagnostics(model, label: str = ""):
    """Capture solve status and the *achieved* MIP gap from a Gurobi model.

    Without this, a solve truncated at ``time_limit`` is indistinguishable in
    the saved artifacts from one that converged to ``mip_gap`` -- both leave
    behind nothing but a cost number.

    Returns None when the model is unavailable.  Never raises: attribute access
    on a model that never solved would otherwise abort an already-complete run
    at reporting time.
    """
    if model is None:
        return None
    try:
        status = int(model.Status)
    except Exception:
        return None
    diag = {
        "status": status,
        "status_name": _GRB_STATUS_NAMES.get(status, f"UNKNOWN_{status}"),
        "converged": status == 2,
    }
    for attr, key in (("MIPGap", "mip_gap_achieved"), ("Runtime", "runtime_seconds"),
                      ("ObjVal", "obj_val"), ("ObjBound", "obj_bound"),
                      ("NodeCount", "node_count")):
        try:
            diag[key] = float(getattr(model, attr))
        except Exception:
            diag[key] = None
    if label and not diag["converged"]:
        print(f"  NOTE: {label} finished with status={diag['status_name']}, "
              f"achieved MIP gap={diag.get('mip_gap_achieved')}")
    return diag
