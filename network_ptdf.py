# network_ptdf.py

from __future__ import annotations

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


def build_dc_ptdf(
    buses_df: pd.DataFrame,
    branches_df: pd.DataFrame,
    slack_bus_id: Optional[int | str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Build DC PTDF matrix and line limits from RTS-GMLC bus/branch tables.

    Parameters
    ----------
    buses_df : pd.DataFrame
        Must contain at least a 'Bus ID' column.
    branches_df : pd.DataFrame
        Must contain at least:
            - 'UID'        : line identifier
            - 'From Bus'   : from-bus ID
            - 'To Bus'     : to-bus ID
            - 'X'          : series reactance (p.u. on system base)
        For limits (Fmax), will use the first available of:
            - 'Cont Rating'
            - 'LTE Rating'
            - 'STE Rating'
        (interpreted as MW; for RTS this is standard.)

    slack_bus_id : int or str, optional
        ID of the slack bus. If None, the smallest Bus ID in buses_df is used.

    Returns
    -------
    PTDF : np.ndarray, shape (L, N)
        PTDF[l, n] = flow on line l (MW) for a 1 MW injection at bus n
        and a -1 MW injection at the slack bus (single-slack PTDF).
    Fmax : np.ndarray, shape (L,)
        Line flow limits (MW) from branch ratings.
    bus_ids : list
        Ordered list of bus IDs corresponding to PTDF columns.
    line_ids : list
        Ordered list of line UIDs corresponding to PTDF rows.

    Notes
    -----
    - Assumes R is small and ignores it (DC power flow approximation).
    - Ignores shunts (B) for the PTDF; uses only series X.
    - Ignores transformer tap ratios for now (treats Tr Ratio == 0 as 1.0).
      You can refine this if you care about off-nominal taps.
    """
    # ---------- 1. Indexing ----------
    # Canonical bus ordering
    bus_ids = sorted(buses_df["Bus ID"].unique())
    bus_idx = {bus_id: i for i, bus_id in enumerate(bus_ids)}
    N = len(bus_ids)

    # Canonical line ordering
    # Ensure branches_df is in a stable order
    branches_df = branches_df.copy().reset_index(drop=True)
    line_ids = branches_df["UID"].tolist()
    L = len(line_ids)

    # Slack bus
    if slack_bus_id is None:
        slack_bus_id = bus_ids[0]
    if slack_bus_id not in bus_idx:
        raise ValueError(f"Slack bus {slack_bus_id!r} not found in buses_df['Bus ID'].")

    slack = bus_idx[slack_bus_id]

    # ---------- 2. Extract branch data ----------
    # Map bus IDs to indices
    from_idx = branches_df["From Bus"].map(bus_idx).to_numpy(dtype=int)
    to_idx = branches_df["To Bus"].map(bus_idx).to_numpy(dtype=int)

    # Reactances
    x = branches_df["X"].to_numpy(dtype=float)

    # Transformer ratio: if 0, treat as 1 (no tap); for now, we ignore tap in DC PTDF.
    if "Tr Ratio" in branches_df.columns:
        tap = branches_df["Tr Ratio"].to_numpy(dtype=float)
        tap = np.where(tap == 0.0, 1.0, tap)
    else:
        tap = np.ones(L, dtype=float)

    # Series susceptance magnitude; simple DC approximation ignoring off-nominal taps
    # If you want to include taps later, you'll adjust this section.
    b = 1.0 / x  # p.u. susceptance on system base

    # ---------- 3. Build incidence matrix A (L x N) ----------
    A = np.zeros((L, N), dtype=float)
    for ell in range(L):
        i = from_idx[ell]
        j = to_idx[ell]
        A[ell, i] = 1.0
        A[ell, j] = -1.0

    # ---------- 4. Build Bbus = A^T * diag(b) * A ----------
    Bbus = np.zeros((N, N), dtype=float)
    for ell in range(L):
        i = from_idx[ell]
        j = to_idx[ell]
        bij = b[ell]
        # Standard Bbus stamping for line between i and j with susceptance bij
        Bbus[i, i] += bij
        Bbus[j, j] += bij
        Bbus[i, j] -= bij
        Bbus[j, i] -= bij

    # ---------- 5. Reduce Bbus by removing slack bus ----------
    keep = [n for n in range(N) if n != slack]
    B_red = Bbus[np.ix_(keep, keep)]

    # Invert reduced B
    B_red_inv = np.linalg.inv(B_red)

    # ---------- 6. Compute PTDF ----------
    # We want PTDF for injections at each non-slack bus with withdrawal at slack.
    # A_red is A with the slack column removed
    A_red = A[:, keep]  # shape (L, N-1)

    # diag(b) * A_red
    BA = b[:, None] * A_red  # shape (L, N-1)

    # PTDF on non-slack buses: (L x (N-1))
    PTDF_red = BA @ B_red_inv

    # Assemble full PTDF (L x N)
    PTDF = np.zeros((L, N), dtype=float)
    # Fill non-slack columns
    for col_pos, bus in enumerate(keep):
        PTDF[:, bus] = PTDF_red[:, col_pos]
    # Slack column = -sum of others (1 MW injection at slack with equal withdrawals)
    PTDF[:, slack] = -PTDF_red.sum(axis=1)

    # ---------- 7. Line limits Fmax ----------
    if "Cont Rating" in branches_df.columns:
        Fmax = branches_df["Cont Rating"].to_numpy(dtype=float)
    elif "LTE Rating" in branches_df.columns:
        Fmax = branches_df["LTE Rating"].to_numpy(dtype=float)
    elif "STE Rating" in branches_df.columns:
        Fmax = branches_df["STE Rating"].to_numpy(dtype=float)
    else:
        # Fallback: large number
        Fmax = np.ones(L, dtype=float) * 1e4

    return PTDF, Fmax, bus_ids, line_ids


if __name__ == "__main__":
    # Tiny smoke test with your sample branch row and fake buses
    buses = pd.DataFrame({"Bus ID": [101, 102, 103]})
    branches = pd.DataFrame(
        {
            "UID": ["A1", "B1", "C1"],
            "From Bus": [101, 102, 103],
            "To Bus": [102, 103, 101],
            "R": [0.003, 0, 0],
            "X": [0.014, 0.0461, 0.03],
            "B": [0.461, 0.678, 0.234],
            "Cont Rating": [175.0, 100, 100],
            "LTE Rating": [193.0, 100, 100],
            "STE Rating": [200.0, 100, 100],
            "Perm OutRate": [0.24, 0, 0],
            "Duration": [16, 0, 0],
            "Tr Ratio": [0.0, 100, 100],
            "Tran OutRate": [0.0, 100, 100],
            "Length": [3.0, 100, 100],
        }
    )

    PTDF, Fmax, bus_ids, line_ids = build_dc_ptdf(buses, branches, slack_bus_id=101)
    print("Bus IDs:", bus_ids)
    print("Line IDs:", line_ids)
    print("PTDF shape:", PTDF.shape)
    print("PTDF:", PTDF)
    print("Fmax:", Fmax)
