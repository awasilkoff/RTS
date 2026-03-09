"""Compute DC branch flows from a saved dispatch CSV using the PTDF matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from io_rts import build_damdata_from_rts
from models import DAMData

RTS_DIR = Path("RTS_Data")
SOURCE_DIR = RTS_DIR / "SourceData"
TS_DIR = RTS_DIR / "timeseries_data_files"


def compute_branch_flows(data: DAMData, p: np.ndarray) -> pd.DataFrame:
    """Compute DC branch flows from generator dispatch via PTDF.

    Parameters
    ----------
    data : DAMData
        Must contain PTDF (L,N), gen_to_bus (I,), d (N,T), line_ids, time.
    p : np.ndarray
        Dispatch array of shape (I, T) in MW.

    Returns
    -------
    pd.DataFrame
        Branch flows (L, T) with index=line_ids, columns=time.
    """
    N = data.n_buses
    T = data.n_periods
    gen_to_bus = data.gen_to_bus.astype(int)

    # Net injection per bus: generation minus load
    inj = np.zeros((N, T))
    for i in range(p.shape[0]):
        inj[gen_to_bus[i], :] += p[i, :]
    inj -= data.d

    flow = data.PTDF @ inj  # (L, T)
    return pd.DataFrame(flow, index=data.line_ids, columns=data.time)


def report_congestion(flow_df: pd.DataFrame, data: DAMData, top_n: int = 10) -> int:
    """Print congestion summary and return number of violations."""
    flow = flow_df.values  # (L, T)
    Fmax = data.Fmax

    # Per-line max absolute flow and loading ratio
    max_abs = np.max(np.abs(flow), axis=1)  # (L,)
    loading = np.where(Fmax > 0, max_abs / Fmax, 0.0)

    # Violations
    tol = 1e-4
    viol_mask = max_abs > Fmax + tol
    n_violations = int(viol_mask.sum())

    print("=" * 60)
    print("BRANCH FLOW SUMMARY")
    print("=" * 60)

    if n_violations == 0:
        print(f"  No line flow violations (tol={tol} MW)")
    else:
        print(f"  {n_violations} lines with violations:")
        for l in np.where(viol_mask)[0]:
            excess = max_abs[l] - Fmax[l]
            print(f"    {data.line_ids[l]:<15}  excess {excess:.2f} MW  "
                  f"(Fmax={Fmax[l]:.1f} MW)")

    # Top-N most congested
    order = np.argsort(-loading)[:top_n]
    print(f"\n  Top {top_n} most congested lines:")
    for l in order:
        print(f"    {data.line_ids[l]:<15}  loading {loading[l]*100:.1f}%  "
              f"(max {max_abs[l]:.1f} / {Fmax[l]:.1f} MW)")

    return n_violations


def compute_line_loading_mask(
    data: DAMData, p: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-hour line monitoring mask based on dispatch loading.

    Parameters
    ----------
    data : DAMData
        System data with PTDF, Fmax, etc.
    p : np.ndarray
        Dispatch array (I, T) used for screening (e.g. DAM solution).
    threshold : float
        Loading fraction cutoff. A (line, period) pair is monitored
        when ``|flow_l(t)| / Fmax_l >= threshold``.

    Returns
    -------
    mask : np.ndarray of bool, shape (L, T)
        True where line *l* should be enforced at period *t*.
    loading : np.ndarray of float, shape (L, T)
        Per-hour loading fractions for diagnostics.
    """
    flow = compute_branch_flows(data, p).values  # (L, T)
    Fmax = data.Fmax[:, None]  # (L, 1) for broadcasting
    loading = np.where(Fmax > 0, np.abs(flow) / Fmax, 0.0)  # (L, T)
    mask = loading >= threshold  # (L, T) bool
    return mask, loading


def filter_monitored_lines(
    data: DAMData, p: np.ndarray, threshold: float = 0.8
) -> tuple[DAMData, np.ndarray | None]:
    """Return filtered data and per-hour monitoring mask.

    Two-stage filter:
    1. **Static pre-filter** — drop lines never above *threshold* at any
       hour (reduces PTDF row count).
    2. **Per-hour mask** — for the remaining lines, mark which ``(l, t)``
       pairs exceed *threshold*.  Returned as ``(L_filtered, T)`` bool
       array indexed by the filtered line indices.

    Parameters
    ----------
    data : DAMData
        Full system data with all lines.
    p : np.ndarray
        Dispatch array (I, T) used for screening (e.g. DAM solution).
    threshold : float
        Loading fraction cutoff (0.5 = 50 %).

    Returns
    -------
    filtered_data : DAMData
        Copy with rows dropped for lines never above *threshold*.
    line_mask : np.ndarray of bool, shape (L_filtered, T)
        Per-hour mask over the *filtered* line indices.
    """
    mask_full, loading_full = compute_line_loading_mask(data, p, threshold)

    # Stage 1: static — keep lines that are above threshold at ANY hour
    ever_active = mask_full.any(axis=1)  # (L,) bool

    n_kept = int(ever_active.sum())
    n_total = len(ever_active)

    # Per-hour stats for kept lines
    line_mask = mask_full[ever_active, :]  # (n_kept, T)
    n_lt_pairs = int(line_mask.sum())
    n_lt_total = n_kept * mask_full.shape[1]

    print(f"\n  Monitored line filtering (threshold={threshold*100:.0f}%):")
    print(f"    {n_kept}/{n_total} lines kept (static pre-filter)")
    print(f"    {n_lt_pairs}/{n_lt_total} (line,period) pairs monitored "
          f"({n_lt_pairs / max(n_lt_total, 1) * 100:.0f}%)")

    # Show top monitored lines by max loading
    kept_idx = np.where(ever_active)[0]
    max_loading = np.max(loading_full, axis=1)
    top_order = np.argsort(-max_loading[kept_idx])[:5]
    for rank, idx in enumerate(top_order):
        l = kept_idx[idx]
        n_hrs = int(mask_full[l].sum())
        print(f"    [{rank+1}] {data.line_ids[l]:<15} peak {max_loading[l]*100:.1f}%  "
              f"({n_hrs}/{mask_full.shape[1]} periods)")

    filtered_line_ids = [data.line_ids[l] for l in range(n_total) if ever_active[l]]
    filtered_data = data.copy(update={
        "PTDF": data.PTDF[ever_active],
        "Fmax": data.Fmax[ever_active],
        "line_ids": filtered_line_ids,
    })
    return filtered_data, line_mask


def find_line_violations(
    data_full: DAMData, p0_arr: np.ndarray, viol_tol: float = 1.0
) -> list[tuple[int, int, float]]:
    """Check all lines for flow violations.

    Parameters
    ----------
    data_full : DAMData
        Full system data with all lines (unfiltered PTDF/Fmax).
    p0_arr : np.ndarray
        Dispatch array (I, T) in MW.
    viol_tol : float
        Excess MW above Fmax to count as a violation.

    Returns
    -------
    list of (line_idx, period, excess_MW) tuples.
    """
    flow = compute_branch_flows(data_full, p0_arr).values  # (L_full, T)
    Fmax = data_full.Fmax
    violations = []
    for l in range(flow.shape[0]):
        for t in range(flow.shape[1]):
            excess = abs(flow[l, t]) - Fmax[l]
            if excess > viol_tol:
                violations.append((l, t, excess))
    return violations


def iterative_line_resolve(
    model,
    vars_dict: dict,
    data: DAMData,
    data_full: DAMData,
    robust_mask: np.ndarray,
    sqrt_Sigma,
    rho,
    rho_lines_frac: float | None,
    time_varying: bool,
    max_iter: int = 5,
    viol_tol: float = 1.0,
) -> int:
    """Iteratively add violated line constraints and re-solve.

    After the initial ARUC/DARUC solve, checks ALL lines for flow
    violations. Any violated (line, period) pairs are added to the
    existing Gurobi model as new constraints, the previous solution is
    used as a warm start, and the model is re-solved.  Repeats until
    no violations remain or *max_iter* iterations are exhausted.

    Parameters
    ----------
    model : gurobipy.Model
        Solved ARUC/DARUC model.
    vars_dict : dict
        Variable dict from ``build_aruc_ldr_model``.
    data : DAMData
        Filtered data used to build the model.
    data_full : DAMData
        Original (unfiltered) data with all lines.
    robust_mask : np.ndarray of bool, shape (T,)
    sqrt_Sigma : np.ndarray
        Cholesky factor(s). Shape (T,K,K) if time-varying, else (K,K).
    rho : float or np.ndarray
        Uncertainty radius. Scalar or (T,) if time-varying.
    rho_lines_frac : float or None
        If set, ``rho_lines = rho_lines_frac * rho``.
    time_varying : bool
    max_iter : int
    viol_tol : float
        MW threshold for violations.

    Returns
    -------
    int
        Number of re-solve iterations performed (0 = no violations).
    """
    import gurobipy as gp
    from gurobipy import GRB

    p0 = vars_dict["p0"]
    Z = vars_dict["Z"]
    gens_at_bus = vars_dict["_gens_at_bus"]
    z_elig_set = vars_dict["_z_elig_set"]
    K = vars_dict["_K"]

    I = data.n_gens
    T = data.n_periods
    N = data.n_buses
    gen_to_bus = data.gen_to_bus.astype(int)
    d = data.d  # (N, T) — same as data_full.d

    # Resolve rho_lines (mirrors aruc_model.py logic)
    if rho_lines_frac is not None:
        rho_lines = rho_lines_frac * rho
    else:
        rho_lines = rho

    added_pairs: set[tuple[int, int]] = set()

    for iteration in range(1, max_iter + 1):
        # Extract p0 dispatch
        p0_arr = np.zeros((I, T))
        for (i, t), var in p0.items():
            p0_arr[i, t] = var.X

        violations = find_line_violations(data_full, p0_arr, viol_tol)
        if not violations:
            print(f"  [Line iter] No violations (tol={viol_tol} MW) — converged.")
            return iteration - 1

        # Filter already-added pairs
        new_viols = [(l, t, ex) for l, t, ex in violations if (l, t) not in added_pairs]
        if not new_viols:
            print(f"  [Line iter {iteration}] {len(violations)} violations but all "
                  f"already constrained — cannot resolve further.")
            return iteration

        print(f"  [Line iter {iteration}] {len(new_viols)} new violated (line,period) pairs:")
        for l, t, ex in sorted(new_viols, key=lambda x: -x[2])[:10]:
            print(f"    {data_full.line_ids[l]:<15} t={t:2d}  excess={ex:.1f} MW")
        if len(new_viols) > 10:
            print(f"    ... and {len(new_viols) - 10} more")

        PTDF_full = data_full.PTDF
        Fmax_full = data_full.Fmax

        n_added = 0
        for l_full, t, _ex in new_viols:
            added_pairs.add((l_full, t))
            tag = f"l{l_full}_t{t}"

            # Build flow_nom expression for this (l_full, t) using full PTDF
            flow_nom = gp.LinExpr()
            for n in range(N):
                ptdf_val = PTDF_full[l_full, n]
                if abs(ptdf_val) < 1e-10:
                    continue
                gen_sum = gp.quicksum(p0[i, t] for i in gens_at_bus[n])
                flow_nom += ptdf_val * (gen_sum - float(d[n, t]))

            fmax_l = float(Fmax_full[l_full])

            if robust_mask[t]:
                # Robust constraints with SOC
                sqrt_Sigma_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma
                rho_lines_t = float(rho_lines[t]) if time_varying else float(rho_lines)

                z_var = model.addVar(lb=0.0, name=f"z_line_add_{tag}")
                y_vars = {}
                for k in range(K):
                    y_vars[k] = model.addVar(lb=-GRB.INFINITY, name=f"y_line_add_{tag}_k{k}")

                # sensitivity: a[k] = sum_n PTDF[l,n] * sum_{i in z_elig at n} Z[i,t,k]
                a_expr = [gp.LinExpr() for _ in range(K)]
                for i in range(I):
                    if i not in z_elig_set:
                        continue
                    n = int(gen_to_bus[i])
                    ptdf_val = PTDF_full[l_full, n]
                    if abs(ptdf_val) < 1e-10:
                        continue
                    for k in range(K):
                        if (i, t, k) in Z:
                            a_expr[k] += ptdf_val * Z[i, t, k]

                # y = L^T @ a  (L = chol(Sigma))
                for i_k in range(K):
                    expr = gp.LinExpr()
                    for j_k in range(K):
                        coef = sqrt_Sigma_t[j_k, i_k]  # L^T[i_k, j_k]
                        if abs(coef) < 1e-10:
                            continue
                        expr += coef * a_expr[j_k]
                    model.addConstr(y_vars[i_k] == expr, name=f"y_line_add_def_{tag}_k{i_k}")

                # SOC: z >= ||y||
                model.addConstr(
                    z_var * z_var >= gp.quicksum(y_vars[k] * y_vars[k] for k in range(K)),
                    name=f"soc_line_add_{tag}",
                )

                # Robust flow limits
                model.addConstr(flow_nom + rho_lines_t * z_var <= fmax_l,
                                name=f"line_max_add_{tag}")
                model.addConstr(-flow_nom + rho_lines_t * z_var <= fmax_l,
                                name=f"line_min_add_{tag}")
            else:
                # Nominal flow limits
                model.addConstr(flow_nom <= fmax_l, name=f"line_max_add_{tag}")
                model.addConstr(flow_nom >= -fmax_l, name=f"line_min_add_{tag}")

            n_added += 1

        print(f"  [Line iter {iteration}] Added {n_added} constraint sets. Warm-starting re-solve...")

        # Warm start from previous solution
        for var in model.getVars():
            try:
                var.Start = var.X
            except AttributeError:
                pass  # new variables without .X yet

        model.optimize()

        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            print(f"  [Line iter {iteration}] WARNING: re-solve status={model.Status}")
            if model.SolCount == 0:
                print("  [Line iter] No feasible solution after adding line constraints!")
                return iteration

    # Exhausted max_iter — final check
    p0_arr = np.zeros((I, T))
    for (i, t), var in p0.items():
        p0_arr[i, t] = var.X
    remaining = find_line_violations(data_full, p0_arr, viol_tol)
    if remaining:
        print(f"  [Line iter] WARNING: {len(remaining)} violations remain after {max_iter} iterations")
    else:
        print(f"  [Line iter] Converged after {max_iter} iterations.")
    return max_iter


def main():
    parser = argparse.ArgumentParser(
        description="Compute DC branch flows from a dispatch CSV.")
    parser.add_argument("dispatch_csv", type=Path,
                        help="Path to dispatch CSV (gen_ids x time)")
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument(
        "--day2-interval",
        type=int,
        default=2,
        help="Day-2 interval hours (default: 1 = hourly, 2 = 2-hour blocks)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("branch_flows.csv"),
                        help="Output CSV path (default: branch_flows.csv)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of congested lines to report")
    parser.add_argument("--include-renewables", action="store_true", default=False)
    parser.add_argument("--include-nuclear", action="store_true", default=False)
    args = parser.parse_args()

    # Load dispatch CSV
    p_df = pd.read_csv(args.dispatch_csv, index_col=0)
    print(f"Loaded dispatch: {p_df.shape[0]} generators x {p_df.shape[1]} periods")

    # Rebuild DAMData with matching parameters
    start = pd.Timestamp(year=2020, month=args.start_month, day=args.start_day)
    data = build_damdata_from_rts(
        source_dir=SOURCE_DIR,
        ts_dir=TS_DIR,
        start_time=start,
        horizon_hours=args.hours,
        day2_interval_hours=args.day2_interval,
        include_renewables=args.include_renewables,
        include_nuclear=args.include_nuclear,
    )

    # Validate generator alignment
    data_ids = list(data.gen_ids)
    csv_ids = list(p_df.index.astype(str))
    if data_ids != csv_ids:
        missing = set(csv_ids) - set(data_ids)
        extra = set(data_ids) - set(csv_ids)
        if missing:
            print(f"WARNING: {len(missing)} generators in CSV not in DAMData: {sorted(missing)[:5]}...")
        if extra:
            print(f"WARNING: {len(extra)} generators in DAMData not in CSV: {sorted(extra)[:5]}...")
        # Use intersection in DAMData order
        common = [g for g in data_ids if g in set(csv_ids)]
        p_df = p_df.loc[common]
        print(f"Using {len(common)} common generators")

    p_array = p_df.values.astype(float)

    # Compute flows
    flow_df = compute_branch_flows(data, p_array)

    # Save
    flow_df.to_csv(args.output)
    print(f"\nBranch flows saved to {args.output}")

    # Report
    report_congestion(flow_df, data, top_n=args.top_n)


if __name__ == "__main__":
    main()
