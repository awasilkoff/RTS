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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    flow : np.ndarray of float, shape (L, T)
        Signed flow values (MW) for determining binding direction.
    """
    flow = compute_branch_flows(data, p).values  # (L, T)
    Fmax = data.Fmax[:, None]  # (L, 1) for broadcasting
    loading = np.where(Fmax > 0, np.abs(flow) / Fmax, 0.0)  # (L, T)
    mask = loading >= threshold  # (L, T) bool
    return mask, loading, flow


def filter_monitored_lines(
    data: DAMData, p: np.ndarray, threshold: float = 0.8
) -> tuple[DAMData, np.ndarray | None, np.ndarray | None]:
    """Return filtered data, per-hour monitoring mask, and flow direction.

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
    flow_direction : np.ndarray of float, shape (L_filtered, T)
        Signed screening flow (MW) for the kept lines.  Used by
        ``build_aruc_ldr_model`` to add only the binding-direction
        line constraint.
    """
    mask_full, loading_full, flow_full = compute_line_loading_mask(data, p, threshold)

    # Stage 1: static — keep lines that are above threshold at ANY hour
    ever_active = mask_full.any(axis=1)  # (L,) bool

    n_kept = int(ever_active.sum())
    n_total = len(ever_active)

    # Per-hour stats for kept lines
    line_mask = mask_full[ever_active, :]  # (n_kept, T)
    flow_direction = flow_full[ever_active, :]  # (n_kept, T)
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
    return filtered_data, line_mask, flow_direction


def find_line_violations(
    data_full: DAMData, p0_arr: np.ndarray, viol_tol: float = 1.0
) -> list[tuple[int, int, float, float]]:
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
    list of (line_idx, period, excess_MW, signed_flow_MW) tuples.
        signed_flow_MW indicates the flow direction (positive or negative)
        so callers can add only the binding-direction constraint.
    """
    flow = compute_branch_flows(data_full, p0_arr).values  # (L_full, T)
    Fmax = data_full.Fmax
    violations = []
    for l in range(flow.shape[0]):
        for t in range(flow.shape[1]):
            excess = abs(flow[l, t]) - Fmax[l]
            if excess > viol_tol:
                violations.append((l, t, excess, flow[l, t]))
    return violations


def find_worst_case_line_violations(
    data_full: DAMData,
    p0_arr: np.ndarray,
    Z_arr: np.ndarray,
    robust_mask: np.ndarray,
    sqrt_Sigma,
    rho,
    rho_lines_frac: float | None,
    time_varying: bool,
    viol_tol: float = 1.0,
) -> list[tuple[int, int, float, float]]:
    """Check all lines for worst-case flow violations under uncertainty.

    For robust periods, the worst-case flow on line l at time t is:
        |f_nom[l,t]| + rho_t * ||L_t^T @ a_t||
    where a_t[k] = sum_n PTDF[l,n] * sum_{i at bus n} Z[i,t,k].

    For non-robust periods, falls back to nominal flow check.

    Parameters
    ----------
    data_full : DAMData
        Full system data with all lines (unfiltered PTDF/Fmax).
    p0_arr : np.ndarray
        Nominal dispatch array (I, T) in MW.
    Z_arr : np.ndarray
        LDR coefficient array (I, T, K) in MW.
    robust_mask : np.ndarray of bool, shape (T,)
        Which periods have robust constraints.
    sqrt_Sigma : np.ndarray
        Cholesky factor(s). Shape (T,K,K) if time-varying, else (K,K).
    rho : float or np.ndarray
        Uncertainty radius. Scalar or (T,) if time-varying.
    rho_lines_frac : float or None
        If set, rho_lines = rho_lines_frac * rho.
    time_varying : bool
    viol_tol : float
        Excess MW above Fmax to count as a violation.

    Returns
    -------
    list of (line_idx, period, excess_MW, signed_flow_MW) tuples.
        signed_flow_MW is the nominal flow direction (for binding-direction
        constraint addition). excess_MW is the worst-case excess.
    """
    PTDF = data_full.PTDF  # (L, N)
    Fmax = data_full.Fmax  # (L,)
    gen_to_bus = data_full.gen_to_bus.astype(int)
    N = data_full.n_buses
    I, T = p0_arr.shape
    L = PTDF.shape[0]

    if rho_lines_frac is not None:
        rho_lines = rho_lines_frac * rho
    else:
        rho_lines = rho

    # Nominal flows: PTDF @ (gen_injection - load)
    inj = np.zeros((N, T))
    for i in range(I):
        inj[gen_to_bus[i], :] += p0_arr[i, :]
    inj -= data_full.d
    flow_nom = PTDF @ inj  # (L, T)

    # Z aggregated by bus: Z_bus[n, t, k] = sum_{i at bus n} Z[i, t, k]
    K = Z_arr.shape[2]
    Z_bus = np.zeros((N, T, K))
    for i in range(I):
        Z_bus[gen_to_bus[i], :, :] += Z_arr[i, :, :]

    # Line sensitivity: a[l, t, k] = sum_n PTDF[l,n] * Z_bus[n,t,k]
    # = PTDF @ Z_bus  reshaped appropriately
    # PTDF is (L, N), Z_bus is (N, T, K) -> a is (L, T, K)
    a = np.einsum("ln,ntk->ltk", PTDF, Z_bus)

    violations = []
    for l in range(L):
        fmax_l = Fmax[l]
        for t in range(T):
            f_nom = flow_nom[l, t]
            if robust_mask[t]:
                # Worst-case margin: rho_t * ||L_t^T @ a[l,t,:]||
                L_t = sqrt_Sigma[t] if time_varying else sqrt_Sigma  # (K, K)
                rho_t = float(rho_lines[t]) if time_varying else float(rho_lines)
                # y = L^T @ a  -> ||y||
                y = L_t.T @ a[l, t, :]  # (K,)
                norm_y = np.linalg.norm(y)
                # Worst case in positive direction: f_nom + rho_t * norm_y
                # Worst case in negative direction: -f_nom + rho_t * norm_y
                wc_pos = f_nom + rho_t * norm_y
                wc_neg = -f_nom + rho_t * norm_y
                excess = max(wc_pos - fmax_l, wc_neg - fmax_l)
            else:
                excess = abs(f_nom) - fmax_l
            if excess > viol_tol:
                violations.append((l, t, excess, f_nom))
    return violations


#: Default cap on worst-case line-violation re-solves.  Hitting this cap means
#: the returned solution may still violate worst-case line limits, so callers
#: should compare their reported iteration count against it.
LINE_RESOLVE_MAX_ITER = 5


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
    max_iter: int = LINE_RESOLVE_MAX_ITER,
    viol_tol: float = 1.0,
    time_limit: float | None = None,
    solve_start: float | None = None,
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
    time_limit : float or None
        Total wall-clock budget in seconds shared between the initial solve
        and all re-solve iterations.  Each re-solve gets only the remaining
        time (``time_limit - elapsed``).  None = no limit.
    solve_start : float or None
        ``time.time()`` timestamp taken just before the initial
        ``model.optimize()`` call.  Required when *time_limit* is set.

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

    def _extract_solution():
        """Extract p0 and Z arrays from solved Gurobi variables."""
        p0_arr = np.zeros((I, T))
        for (i, t), var in p0.items():
            p0_arr[i, t] = var.X
        Z_arr = np.zeros((I, T, K))
        for (i, t, k), var in Z.items():
            Z_arr[i, t, k] = var.X
        return p0_arr, Z_arr

    for iteration in range(1, max_iter + 1):
        p0_arr, Z_arr = _extract_solution()

        # Check worst-case flows (nominal + uncertainty margin)
        violations = find_worst_case_line_violations(
            data_full, p0_arr, Z_arr, robust_mask,
            sqrt_Sigma, rho, rho_lines_frac, time_varying, viol_tol,
        )
        if not violations:
            print(f"  [Line iter] No worst-case violations (tol={viol_tol} MW) — converged.")
            return iteration - 1

        # Filter already-added pairs
        new_viols = [(l, t, ex, fv) for l, t, ex, fv in violations if (l, t) not in added_pairs]
        if not new_viols:
            print(f"  [Line iter {iteration}] {len(violations)} violations but all "
                  f"already constrained — cannot resolve further.")
            return iteration

        print(f"  [Line iter {iteration}] {len(new_viols)} new worst-case violated (line,period) pairs:")
        for l, t, ex, fv in sorted(new_viols, key=lambda x: -x[2])[:10]:
            print(f"    {data_full.line_ids[l]:<15} t={t:2d}  wc_excess={ex:.1f} MW  nom_flow={fv:+.1f}")
        if len(new_viols) > 10:
            print(f"    ... and {len(new_viols) - 10} more")

        PTDF_full = data_full.PTDF
        Fmax_full = data_full.Fmax

        n_added = 0
        for l_full, t, _ex, flow_val in new_viols:
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

                # Always add both directions: once (l,t) is in added_pairs it
                # can never be revisited, so a single-direction add here would
                # permanently miss the other side if the flow reverses after
                # re-solve.  The two extra linear constraints are cheap vs a
                # full re-solve.
                model.addConstr(flow_nom + rho_lines_t * z_var <= fmax_l,
                                name=f"line_max_add_{tag}")
                model.addConstr(-flow_nom + rho_lines_t * z_var <= fmax_l,
                                name=f"line_min_add_{tag}")
            else:
                # Nominal flow limit — add both directions for the same reason.
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

        if time_limit is not None and solve_start is not None:
            import time as _time
            elapsed = _time.time() - solve_start
            remaining = time_limit - elapsed
            if remaining <= 0:
                print(f"  [Line iter {iteration}] Global time budget exhausted ({elapsed:.0f}s >= {time_limit:.0f}s) — stopping.")
                return iteration
            model.Params.TimeLimit = remaining
            print(f"  [Line iter {iteration}] Remaining budget: {remaining:.0f}s of {time_limit:.0f}s")

        model.optimize()

        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            print(f"  [Line iter {iteration}] WARNING: re-solve status={model.Status}")
            if model.SolCount == 0:
                print("  [Line iter] No feasible solution after adding line constraints!")
                return iteration

    # Exhausted max_iter — final check
    p0_arr, Z_arr = _extract_solution()
    remaining = find_worst_case_line_violations(
        data_full, p0_arr, Z_arr, robust_mask,
        sqrt_Sigma, rho, rho_lines_frac, time_varying, viol_tol,
    )
    if remaining:
        print(f"  [Line iter] WARNING: {len(remaining)} worst-case violations remain after {max_iter} iterations")
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
    parser.add_argument("--top-n", type=int, default=30,
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
