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


def main():
    parser = argparse.ArgumentParser(
        description="Compute DC branch flows from a dispatch CSV.")
    parser.add_argument("dispatch_csv", type=Path,
                        help="Path to dispatch CSV (gen_ids x time)")
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--hours", type=int, default=48)
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
