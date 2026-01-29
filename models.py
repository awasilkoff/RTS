"""
Models for deterministic day-ahead unit commitment on RTS-GMLC.

Defines the canonical Pydantic BaseModel `DAMData`, which bundles all
arrays and index lists you need to build a deterministic DAM UC model.

Design choices:
- Use pandas for ETL / raw RTS-GMLC ingestion (outside this file).
- Convert to numpy arrays before constructing DAMData.
- No explicit reserves modeled (energy + commitment + network only).
- Supports time-varying Pmax, which you can use for dispatchable wind/solar.
- Encodes generator type (e.g. THERMAL vs WIND) for future RUC / renewable logic.
- Includes generator initial conditions (commitment + up/down times).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class DAMData(BaseModel):
    """
    Canonical deterministic DAM data object.

    This is the "boundary" between:
      - your RTS-GMLC ETL / preprocessing layer (pandas-heavy),
      - and your Gurobi model builder (array-based).

    Conventions
    -----------
    - Indices:
        i ∈ {0, ..., I-1}  generators
        n ∈ {0, ..., N-1}  buses
        l ∈ {0, ..., L-1}  lines
        t ∈ {0, ..., T-1}  time periods
        b ∈ {0, ..., B-1}  cost blocks
    - All numpy arrays use these integer indices.
    - `gen_ids[k]` is the external ID for generator i = k, etc.
    - `gen_to_bus[i]` is an integer bus index n for generator i.
    """

    # ---------- Identity / index lists ----------
    gen_ids: List[str | int] = Field(
        ..., description="External IDs for generators (length I)."
    )
    bus_ids: List[str | int] = Field(
        ..., description="External IDs for buses (length N)."
    )
    line_ids: List[str | int] = Field(
        ..., description="External IDs for lines (length L)."
    )
    time: List[pd.Timestamp] | List[int] = Field(
        ...,
        description="Time index for periods t (length T). Can be ints or timestamps.",
    )

    # Generator types, one per generator.
    # Recommended conventions (but not enforced here):
    #   - 'THERMAL'  for conventional dispatchable units
    #   - 'WIND', 'SOLAR' for variable/forecast-driven renewables
    #   - 'HYDRO', 'NUCLEAR', etc. as needed
    gen_type: List[str] = Field(
        ...,
        description=(
            "Generator type labels (length I). "
            "Use e.g. 'THERMAL', 'WIND', 'SOLAR', 'HYDRO'. "
            "Future RUC logic can distinguish thermal vs variable units via these labels."
        ),
    )

    # ---------- Mapping arrays ----------
    # gen_to_bus[i] ∈ {0, ..., N-1}
    gen_to_bus: np.ndarray = Field(
        ..., description="Integer bus index for each generator (shape (I,))."
    )

    # ---------- Generator-level scalars ----------
    # All shape (I,)
    Pmin: np.ndarray = Field(
        ..., description="Minimum output for each generator (MW). shape (I,)."
    )

    # Pmax can be either (I,) or (I, T); see Pmax_2d() helper below.
    Pmax: np.ndarray = Field(
        ...,
        description=(
            "Maximum output for each generator. "
            "Either shape (I,) if static, or (I, T) if time-varying (e.g. dispatchable wind)."
        ),
    )

    RU: np.ndarray = Field(
        ..., description="Ramp-up limit (MW per period). shape (I,)."
    )
    RD: np.ndarray = Field(
        ..., description="Ramp-down limit (MW per period). shape (I,)."
    )
    MUT: np.ndarray = Field(
        ..., description="Minimum up time (in periods). shape (I,)."
    )
    MDT: np.ndarray = Field(
        ..., description="Minimum down time (in periods). shape (I,)."
    )
    startup_cost: np.ndarray = Field(
        ..., description="Startup cost for each generator. shape (I,)."
    )
    shutdown_cost: np.ndarray = Field(
        ..., description="Shutdown cost for each generator. shape (I,)."
    )
    no_load_cost: np.ndarray = Field(
        ..., description="No-load cost for each generator per period. shape (I,)."
    )

    # ---------- Initial conditions ----------
    # Initial commitment and time-in-state before the scheduling horizon.
    #
    # u_init[i]         : 1 if generator i is ON at "time 0" (before t=0 of the model),
    #                     0 if OFF. This is the status that should connect
    #                     to u[i,0] via logic constraints.
    #
    # init_up_time[i]   : number of consecutive periods generator i has been ON
    #                     immediately before time 0 (history). 0 means it just
    #                     turned on now; a large value means it's been on for a long time.
    #
    # init_down_time[i] : number of consecutive periods generator i has been OFF
    #                     immediately before time 0. 0 means it just turned off.
    #
    # Typically, for each i, exactly one of init_up_time[i], init_down_time[i]
    # is positive, and u_init[i] is consistent with that (1 if up_time>0, 0 otherwise).
    u_init: np.ndarray = Field(
        ...,
        description="Initial on/off status at time 0; 1=on, 0=off. shape (I,).",
    )
    init_up_time: np.ndarray = Field(
        ...,
        description="Historical consecutive on-time before period 0 (in periods). shape (I,).",
    )
    init_down_time: np.ndarray = Field(
        ...,
        description="Historical consecutive off-time before period 0 (in periods). shape (I,).",
    )

    # ---------- Piecewise linear energy cost ----------
    # block_cap[i, b] is the capacity (MW) of block b for generator i.
    # block_cost[i, b] is the marginal cost of that block (e.g. $/MWh).
    # You can set B = 1 for simple linear cost.
    block_cap: np.ndarray = Field(
        ..., description="Cost block capacities (MW). shape (I, B)."
    )
    block_cost: np.ndarray = Field(
        ...,
        description="Cost block marginal prices (same units as energy price). shape (I, B).",
    )

    # ---------- Network data ----------
    # PTDF[l, n]: power transfer distribution factor for line l at bus n.
    PTDF: np.ndarray = Field(..., description="Line-bus PTDF matrix. shape (L, N).")
    Fmax: np.ndarray = Field(
        ..., description="Line flow limits (MW) for each line. shape (L,)."
    )

    # ---------- Load / net injection ----------
    # d[n, t]: nominal net load at bus n and time t.
    # Convention: positive d is demand:
    #   sum_i p[i,t] + s_p[t] = sum_n d[n,t]
    # For future RUC, treat this as the *forecast* (nominal) net load.
    d: np.ndarray = Field(
        ...,
        description="Nominal net load at each bus and time (after any fixed injections). shape (N, T).",
    )

    # ---------- Optional debug / provenance ----------
    # These are not required for the optimization model; they are for debugging / introspection.
    gens_df: Optional[pd.DataFrame] = Field(
        default=None,
        description="Optional: original generator table used to create arrays.",
    )
    buses_df: Optional[pd.DataFrame] = Field(
        default=None,
        description="Optional: original bus table used to create arrays.",
    )
    lines_df: Optional[pd.DataFrame] = Field(
        default=None,
        description="Optional: original line table used to create arrays.",
    )

    class Config:
        # Allow numpy arrays and pandas objects as fields.
        arbitrary_types_allowed = True

    # ---------- Convenience properties ----------
    @property
    def n_gens(self) -> int:
        return len(self.gen_ids)

    @property
    def n_buses(self) -> int:
        return len(self.bus_ids)

    @property
    def n_lines(self) -> int:
        return len(self.line_ids)

    @property
    def n_periods(self) -> int:
        return len(self.time)

    @property
    def n_blocks(self) -> int:
        # Handle possible 1D edge cases gracefully
        return int(self.block_cap.shape[1]) if self.block_cap.ndim == 2 else 0

    # ---------- Convenience masks for generator types ----------
    @property
    def thermal_mask(self) -> np.ndarray:
        """
        Boolean mask of generators considered 'thermal/static'.

        By default, anything with gen_type == 'THERMAL' is treated as thermal.
        You can refine this logic later (e.g. include 'NUCLEAR', 'HYDRO' if you like).
        """
        types = np.array(self.gen_type, dtype=object)
        return types == "THERMAL"

    @property
    def variable_mask(self) -> np.ndarray:
        """
        Boolean mask of generators considered 'variable/forecast-driven'.

        By default, anything with gen_type != 'THERMAL' is treated as variable
        (e.g. 'WIND', 'SOLAR'). You can refine this logic later.
        """
        return ~self.thermal_mask

    # ---------- Helper for treating Pmax as 2D ----------
    def Pmax_2d(self) -> np.ndarray:
        """
        Return Pmax as a 2D array of shape (I, T), broadcasting if necessary.

        - If Pmax is (I,), broadcast along time: Pmax_2d[i,t] = Pmax[i].
        - If Pmax is (I,T), return as-is.
        """
        I = self.n_gens
        T = self.n_periods

        if self.Pmax.ndim == 1:
            assert self.Pmax.shape == (
                I,
            ), f"Pmax shape {self.Pmax.shape} incompatible with I={I}"
            return np.tile(self.Pmax[:, None], (1, T))
        elif self.Pmax.ndim == 2:
            assert self.Pmax.shape == (
                I,
                T,
            ), f"Pmax shape {self.Pmax.shape} incompatible with (I,T)=({I},{T})"
            return self.Pmax
        else:
            raise ValueError(f"Pmax must be 1D or 2D, got ndim={self.Pmax.ndim}")

    # ---------- Optional: simple shape + length checks ----------
    def validate_shapes(self) -> None:
        """
        Run sanity checks on array shapes vs index lengths.

        Call this once after constructing DAMData if you want a quick
        assertion-based sanity check.
        """
        I = self.n_gens
        N = self.n_buses
        L = self.n_lines
        T = self.n_periods

        # gen_type length
        assert (
            len(self.gen_type) == I
        ), f"gen_type must have length I={I}, but has length {len(self.gen_type)}"

        # gen_to_bus
        assert self.gen_to_bus.shape == (
            I,
        ), f"gen_to_bus must have shape ({I},) but has {self.gen_to_bus.shape}"

        # Scalar generator arrays
        for name, arr in [
            ("Pmin", self.Pmin),
            ("RU", self.RU),
            ("RD", self.RD),
            ("MUT", self.MUT),
            ("MDT", self.MDT),
            ("startup_cost", self.startup_cost),
            ("shutdown_cost", self.shutdown_cost),
            ("no_load_cost", self.no_load_cost),
            ("u_init", self.u_init),
            ("init_up_time", self.init_up_time),
            ("init_down_time", self.init_down_time),
        ]:
            assert arr.shape == (
                I,
            ), f"{name} must have shape ({I},) but has {arr.shape}"

        # Pmax: allow (I,) or (I, T)
        if self.Pmax.ndim == 1:
            assert self.Pmax.shape == (
                I,
            ), f"Pmax must have shape ({I},) or ({I}, {T}), got {self.Pmax.shape}"
        elif self.Pmax.ndim == 2:
            assert self.Pmax.shape == (
                I,
                T,
            ), f"Pmax must have shape ({I}, {T}) if 2D, got {self.Pmax.shape}"
        else:
            raise AssertionError(f"Pmax must be 1D or 2D, got ndim={self.Pmax.ndim}")

        # Block arrays: (I, B)
        assert (
            self.block_cap.shape[0] == I
        ), f"block_cap first dimension must be I={I}, got {self.block_cap.shape}"
        assert (
            self.block_cost.shape == self.block_cap.shape
        ), f"block_cost shape {self.block_cost.shape} must match block_cap shape {self.block_cap.shape}"

        # Network
        assert self.Fmax.shape == (
            L,
        ), f"Fmax must have shape ({L},) but has {self.Fmax.shape}"
        assert self.PTDF.shape == (
            L,
            N,
        ), f"PTDF must have shape ({L}, {N}) but has {self.PTDF.shape}"

        # Load
        assert self.d.shape == (
            N,
            T,
        ), f"d must have shape ({N}, {T}) but has {self.d.shape}"


if __name__ == "__main__":
    # Minimal smoke test with tiny dummy data.
    I, N, L, T, B = 3, 3, 1, 4, 2

    dummy = DAMData(
        gen_ids=[f"G{i}" for i in range(I)],
        bus_ids=[f"B{n}" for n in range(N)],
        line_ids=[f"L{l}" for l in range(L)],
        time=list(range(T)),
        gen_type=["THERMAL", "WIND", "THERMAL"],
        gen_to_bus=np.array([0, 2, 1], dtype=int),
        Pmin=np.zeros(I),
        Pmax=np.ones(I),  # static Pmax
        RU=np.ones(I) * 10.0,
        RD=np.ones(I) * 10.0,
        MUT=np.ones(I) * 1,
        MDT=np.ones(I) * 1,
        startup_cost=np.ones(I) * 5.0,
        shutdown_cost=np.ones(I) * 2.0,
        no_load_cost=np.ones(I) * 1.0,
        # Initial conditions (dummy)
        u_init=np.array([1, 0, 1], dtype=float),
        init_up_time=np.array([3, 0, 5], dtype=float),
        init_down_time=np.array([0, 4, 0], dtype=float),
        block_cap=np.ones((I, B)) * 50.0,
        block_cost=np.array([[10.0, 20.0], [0.0, 0.0], [12.0, 25.0]]),
        PTDF=np.zeros((L, N)),
        Fmax=np.ones(L) * 100.0,
        d=np.ones((N, T)) * 10.0,
        gens_df=None,
        buses_df=None,
        lines_df=None,
    )

    dummy.validate_shapes()
    print("DAMData smoke test passed.")
    print("Thermal mask:", dummy.thermal_mask)
    print("Variable mask:", dummy.variable_mask)
    print("Pmax_2d shape:", dummy.Pmax_2d().shape)
