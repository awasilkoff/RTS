#!/usr/bin/env python3
"""Verify the DC PTDF against an independent angle-based power flow.

The PTDF is the one piece of network data every line constraint depends on, in
DAM, ARUC and DARUC alike, and an error in it is invisible in the results -- it
shows up only as congestion on the wrong lines.  This script checks it against a
completely separate computation path.

Method
------
For a power-balanced injection vector ``p`` (sum zero):

  1. PTDF path      : ``flow = PTDF @ p``
  2. Independent path: solve ``B_red theta = p_red`` with the slack angle fixed
     at zero, then ``flow_l = b_l (theta_from - theta_to)``

These must agree to solver precision.  The second path shares no code with
``build_dc_ptdf`` beyond reading the same CSVs, so it catches errors in the
reduction, assembly and slack handling.

Also asserts the structural property that caught a real bug: under the
single-slack convention, the slack column must be identically zero.

Usage
-----
    python utils/verify_ptdf.py
    python utils/verify_ptdf.py --trials 500 --slack-bus 113
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network_ptdf import build_dc_ptdf  # noqa: E402

_DATA = Path(__file__).resolve().parent.parent / "RTS_Data" / "SourceData"

# Flow agreement tolerance (MW).  Both paths solve the same linear system by
# different routes, so disagreement should be at round-off scale.
_TOL_MW = 1e-6


def angle_based_flows(buses_df, branches_df, inj, slack_idx):
    """Reference DC flows via nodal angles -- independent of build_dc_ptdf."""
    bus_ids = sorted(buses_df["Bus ID"].unique())
    idx = {b: i for i, b in enumerate(bus_ids)}
    n = len(bus_ids)

    branches_df = branches_df.reset_index(drop=True)
    f = branches_df["From Bus"].map(idx).to_numpy(int)
    t = branches_df["To Bus"].map(idx).to_numpy(int)
    b = 1.0 / branches_df["X"].to_numpy(float)

    Bbus = np.zeros((n, n))
    for e in range(len(b)):
        i, j = f[e], t[e]
        Bbus[i, i] += b[e]
        Bbus[j, j] += b[e]
        Bbus[i, j] -= b[e]
        Bbus[j, i] -= b[e]

    keep = [k for k in range(n) if k != slack_idx]
    theta = np.zeros(n)
    # Sparse LU (SuperLU), not np.linalg.solve: network_ptdf.py avoids BLAS
    # here deliberately ("to avoid MKL crashes on some Windows envs"), and a
    # verifier that cannot run on those environments is useless.
    B_red = _sp.csc_matrix(Bbus[np.ix_(keep, keep)])
    theta[keep] = _spla.splu(B_red).solve(inj[keep])
    return b * (theta[f] - theta[t])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=200,
                    help="Random balanced injection vectors to test (default: 200)")
    ap.add_argument("--slack-bus", type=int, default=None,
                    help="Slack bus ID (default: smallest, matching build_dc_ptdf)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    buses_df = pd.read_csv(_DATA / "bus.csv")
    branches_df = pd.read_csv(_DATA / "branch.csv")

    PTDF, Fmax, bus_ids, line_ids = build_dc_ptdf(
        buses_df, branches_df, slack_bus_id=args.slack_bus
    )
    slack_id = args.slack_bus if args.slack_bus is not None else bus_ids[0]
    slack_idx = bus_ids.index(slack_id)

    print(f"PTDF {PTDF.shape[0]} lines x {PTDF.shape[1]} buses, slack bus {slack_id}")

    failures = []

    # --- Structural check: slack column must be identically zero -------------
    slack_col_max = float(np.abs(PTDF[:, slack_idx]).max())
    ok = slack_col_max <= _TOL_MW
    print(f"\n[{'PASS' if ok else 'FAIL'}] slack column is zero "
          f"(max|PTDF[:, slack]| = {slack_col_max:.3e})")
    if not ok:
        failures.append(
            "slack column is non-zero: injecting and withdrawing at the slack "
            "cannot move power, so any net imbalance at the slack bus will "
            "corrupt every line flow"
        )

    # --- Magnitude sanity ----------------------------------------------------
    max_abs = float(np.abs(PTDF).max())
    worst_line = line_ids[int(np.unravel_index(np.abs(PTDF).argmax(), PTDF.shape)[0])]
    print(f"[INFO] max|PTDF| = {max_abs:.3f} (line {worst_line})")

    # --- Numerical check against the independent path ------------------------
    rng = np.random.default_rng(args.seed)
    n = len(bus_ids)
    worst_err = 0.0
    for _ in range(args.trials):
        inj = rng.normal(0.0, 100.0, n)
        inj -= inj.mean()                       # enforce power balance
        ref = angle_based_flows(buses_df, branches_df, inj, slack_idx)
        worst_err = max(worst_err, float(np.abs(PTDF @ inj - ref).max()))

    ok = worst_err <= _TOL_MW
    print(f"[{'PASS' if ok else 'FAIL'}] PTDF flows match angle-based flows over "
          f"{args.trials} balanced injections (max error {worst_err:.3e} MW)")
    if not ok:
        failures.append(
            f"PTDF disagrees with the angle-based solve by up to {worst_err:,.1f} MW"
        )

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All PTDF checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
