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
