# io_rts.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from network_ptdf import build_dc_ptdf

from models import DAMData

import pandas as pd


def add_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DF with Year, Month, Day, Period columns,
    add a 'time' column as a pandas.Timestamp.
    """
    # Period is 1..24 → hour = Period-1
    hour = df["Period"] - 1

    df = df.copy()
    df["time"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=df["Day"],
            hour=hour,
        )
    )
    return df


def build_wind_pmax_array(
    wind_df: pd.DataFrame,
    gens_df: pd.DataFrame,
    gen_ids: list,
    time_index: list,
    gen_type: np.ndarray,
) -> np.ndarray:
    """
    Build time-varying Pmax array for wind generators.

    Parameters
    ----------
    wind_df : DataFrame
        index = time, columns = wind generator names
        values = availability factor (0-1) or absolute MW
    gens_df : DataFrame
        Generator metadata with 'GEN UID' and 'PMax MW'
    gen_ids : list
        Ordered list of generator IDs
    time_index : list
        Ordered list of times
    gen_type : np.ndarray
        Array of generator types (length = len(gen_ids))

    Returns
    -------
    pmax_wind : np.ndarray
        I x T array where pmax_wind[i,t] is the wind forecast for generator i at time t
        (or 0 for non-wind generators)
    """
    I = len(gen_ids)
    T = len(time_index)

    pmax_wind = np.zeros((I, T), dtype=float)

    # Create mapping from gen_id to index
    gen_idx = {gen_id: i for i, gen_id in enumerate(gen_ids)}

    # Get static Pmax for each generator
    gens_lookup = gens_df.set_index("GEN UID")

    # Identify wind generators
    is_wind = np.array([gt.upper() == "WIND" for gt in gen_type])

    # For each wind generator, look up its time series
    for i, gen_id in enumerate(gen_ids):
        if not is_wind[i]:
            continue

        # Try to find this generator in wind_df columns
        # RTS wind column names might need mapping (e.g., "301_WIND_1" in wind_df)
        matching_cols = [col for col in wind_df.columns if gen_id in str(col)]

        if not matching_cols:
            print(f"WARNING: No wind time series found for {gen_id}, using 0")
            continue

        # Use first matching column
        wind_col = matching_cols[0]

        # Get static capacity
        static_pmax = gens_lookup.loc[gen_id, "PMax MW"]

        # Get wind time series for our time window
        wind_series = wind_df.loc[time_index, wind_col].to_numpy(dtype=float)

        # Wind_df values are typically capacity factors (0-1), multiply by Pmax
        # If they're already in MW, this might need adjustment
        if wind_series.max() <= 1.0:
            # Capacity factors
            pmax_wind[i, :] = wind_series * static_pmax
        else:
            # Already in MW
            pmax_wind[i, :] = wind_series

    return pmax_wind


def build_solar_pmax_array(
    pv_df: pd.DataFrame,
    rtpv_df: pd.DataFrame,
    gens_df: pd.DataFrame,
    gen_ids: list,
    time_index: list,
    gen_type: np.ndarray,
) -> np.ndarray:
    """
    Build time-varying Pmax for PV and RTPV generators.

    Solar CSVs are already in absolute MW, so no capacity-factor conversion
    is needed. Values are clipped to nameplate Pmax.

    Parameters
    ----------
    pv_df : DataFrame
        index = time, columns = PV generator names, values = MW availability.
    rtpv_df : DataFrame
        index = time, columns = RTPV generator names, values = MW availability.
    gens_df : DataFrame
        Generator metadata with 'GEN UID' and 'PMax MW'.
    gen_ids : list
        Ordered list of generator IDs.
    time_index : list
        Ordered list of times.
    gen_type : np.ndarray
        Array of generator types (length = len(gen_ids)).

    Returns
    -------
    pmax_solar : np.ndarray
        I x T array where pmax_solar[i,t] is the solar availability for
        generator i at time t (or 0 for non-solar generators).
    """
    I = len(gen_ids)
    T = len(time_index)
    pmax_solar = np.zeros((I, T), dtype=float)

    gens_lookup = gens_df.set_index("GEN UID")
    is_solar = np.array([gt.upper() == "SOLAR" for gt in gen_type])

    # Combine PV + RTPV into one lookup
    solar_df = pd.concat([pv_df, rtpv_df], axis=1)

    for i, gen_id in enumerate(gen_ids):
        if not is_solar[i]:
            continue

        matching_cols = [c for c in solar_df.columns if gen_id in str(c)]
        if not matching_cols:
            print(f"WARNING: No solar time series for {gen_id}, using static Pmax")
            pmax_solar[i, :] = gens_lookup.loc[gen_id, "PMax MW"]
            continue

        series = solar_df.loc[time_index, matching_cols[0]].to_numpy(dtype=float)
        static_pmax = gens_lookup.loc[gen_id, "PMax MW"]
        pmax_solar[i, :] = np.clip(series, 0, static_pmax)

    return pmax_solar


def build_wind_pmax_from_spp_ensemble(
    forecasts_parquet: str | Path,
    gen_ids: list[str],
    gen_type: list[str],
    static_pmax: np.ndarray,
    n_hours: int,
    start_idx: int = 0,
) -> np.ndarray:
    """
    Build time-varying wind Pmax from SPP ensemble mean forecast.

    Returns (I, T) array: ensemble mean per wind gen, 0 for non-wind.
    Clips to static_pmax to prevent exceeding nameplate.
    Uses positional indexing (start_idx) for time alignment.

    Parameters
    ----------
    forecasts_parquet : Path
        Path to scaled/filtered forecasts parquet (v2).
    gen_ids : list[str]
        Ordered generator IDs.
    gen_type : list[str]
        Generator types (length I).
    static_pmax : np.ndarray
        Static Pmax from gen.csv (length I or (I,) or (I,T) — uses [i] or [i,0]).
    n_hours : int
        Number of hours (T) for the horizon.
    start_idx : int
        Starting row index in the sorted time series.

    Returns
    -------
    pmax_wind : np.ndarray
        (I, T) array.
    """
    I = len(gen_ids)
    T = n_hours
    pmax_wind = np.zeros((I, T), dtype=float)

    is_wind = [gt.upper() == "WIND" for gt in gen_type]
    wind_gen_ids = {gen_ids[i] for i in range(I) if is_wind[i]}

    if not wind_gen_ids:
        return pmax_wind

    # Load forecasts
    df = pd.read_parquet(forecasts_parquet)

    # Keep only wind resources that match our gen_ids
    df = df[df["ID_RESOURCE"].isin(wind_gen_ids)].copy()

    if df.empty:
        print("WARNING: No matching wind resources in SPP forecasts parquet")
        return pmax_wind

    # Compute ensemble mean per (time, resource)
    df["TIME_HOURLY"] = pd.to_datetime(df["TIME_HOURLY"])
    mean_fc = (
        df.groupby(["TIME_HOURLY", "ID_RESOURCE"], as_index=False)["FORECAST"]
        .mean()
    )

    # Pivot to wide: rows=time, cols=resource
    pivot = mean_fc.pivot(index="TIME_HOURLY", columns="ID_RESOURCE", values="FORECAST")
    pivot = pivot.sort_index()

    # Slice by positional index
    if start_idx + T > len(pivot):
        raise ValueError(
            f"SPP forecasts have {len(pivot)} timesteps but need "
            f"start_idx={start_idx} + n_hours={T} = {start_idx + T}"
        )
    pivot_slice = pivot.iloc[start_idx : start_idx + T]

    # Fill (I, T) array
    # Get static pmax per generator (handle 1D or 2D)
    static_pmax_arr = np.asarray(static_pmax)
    for i in range(I):
        if not is_wind[i]:
            continue
        gen_id = gen_ids[i]
        if gen_id not in pivot_slice.columns:
            print(f"WARNING: {gen_id} not in SPP forecasts, using 0")
            continue

        vals = pivot_slice[gen_id].to_numpy(dtype=float)

        # Get nameplate for clipping
        if static_pmax_arr.ndim == 1:
            cap = static_pmax_arr[i]
        else:
            cap = static_pmax_arr[i, 0]

        pmax_wind[i, :] = np.clip(vals, 0, cap)

    return pmax_wind


def load_rts_source_tables(source_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load RTS-GMLC SourceData tables into pandas DataFrames.

    Adjust the filenames/column names to the actual RTS-GMLC repo.
    """
    src = Path(source_dir)

    buses = pd.read_csv(src / "bus.csv")  # TODO: adjust filenames
    lines = pd.read_csv(src / "branch.csv")
    gens = pd.read_csv(src / "gen.csv")
    return {
        "buses": buses,
        "lines": lines,
        "gens": gens,
    }


def load_rts_timeseries(
    ts_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load RTS-GMLC time series: load, wind, PV, and RTPV.

    Returns
    -------
    load_df :
        DataFrame indexed by time, columns = regions, values = regional load MW.
    wind_df :
        DataFrame indexed by time, columns = wind generators.
    pv_df :
        DataFrame indexed by time, columns = PV generators (MW).
    rtpv_df :
        DataFrame indexed by time, columns = RTPV generators (MW).
    """
    ts_path = Path(ts_dir)

    load_raw = pd.read_csv(ts_path / "Load" / "DAY_AHEAD_regional_Load.csv")
    load_raw = add_timestamp_column(load_raw)
    load_df = load_raw.set_index(["time"]).sort_index()
    load_df = load_df.drop(columns={"Year", "Month", "Day", "Period"})

    # Wind:
    wind_raw = pd.read_csv(ts_path / "WIND" / "DAY_AHEAD_wind.csv")
    wind_raw = add_timestamp_column(wind_raw)
    wind_df = wind_raw.set_index(["time"]).sort_index()
    wind_df = wind_df.drop(columns={"Year", "Month", "Day", "Period"})

    # Solar PV:
    pv_raw = pd.read_csv(ts_path / "PV" / "DAY_AHEAD_pv.csv")
    pv_raw = add_timestamp_column(pv_raw)
    pv_df = pv_raw.set_index(["time"]).sort_index()
    pv_df = pv_df.drop(columns={"Year", "Month", "Day", "Period"})

    # Rooftop PV:
    rtpv_raw = pd.read_csv(ts_path / "RTPV" / "DAY_AHEAD_rtpv.csv")
    rtpv_raw = add_timestamp_column(rtpv_raw)
    rtpv_df = rtpv_raw.set_index(["time"]).sort_index()
    rtpv_df = rtpv_df.drop(columns={"Year", "Month", "Day", "Period"})

    return load_df, wind_df, pv_df, rtpv_df


import numpy as np
import pandas as pd


def compute_bus_load_factors(
    buses_df: pd.DataFrame,
    region_col: str = "Area",
    load_col: str = "MW Load",
) -> pd.DataFrame:
    """
    Given bus metadata with a Region and MW Load column, compute
    per-bus load share (alpha) within each region.

    Returns a DataFrame with columns:
        ['Bus ID', region_col, load_col, 'alpha']
    """
    df = buses_df.copy()

    # Total MW load by region
    region_totals = df.groupby(region_col)[load_col].sum()

    # Avoid divide-by-zero; if a region total is 0, just set alpha=0 for those buses
    df["region_total"] = df[region_col].map(region_totals)
    df["alpha"] = np.where(
        df["region_total"] > 0,
        df[load_col] / df["region_total"],
        0.0,
    )

    return df[["Bus ID", region_col, load_col, "alpha"]]


def build_nodal_load_array(
    load_df: pd.DataFrame,
    bus_factors: pd.DataFrame,
    bus_ids: list,
    time_index: list,
    region_col: str = "Area",
) -> np.ndarray:
    """
    Distribute regional load to buses using static alpha factors.

    Parameters
    ----------
    load_df : DataFrame
        index = time, columns = regions (e.g. 1,2,3),
        values = regional load MW.
    bus_factors : DataFrame
        Output of compute_bus_load_factors, with columns:
        ['Bus ID', region_col, 'alpha'].
    bus_ids : list
        Ordered list of bus IDs (the order you'll use in DAMData.bus_ids).
    time_index : list
        Ordered list of times for the horizon (subset of load_df.index).

    Returns
    -------
    d : np.ndarray
        N x T nodal load array, where N=len(bus_ids), T=len(time_index).
    """
    N = len(bus_ids)
    T = len(time_index)

    # Map bus_id -> row index
    bus_idx = {bus_id: n for n, bus_id in enumerate(bus_ids)}

    # Ensure we only consider buses in bus_ids
    bf = bus_factors[bus_factors["Bus ID"].isin(bus_ids)].copy()

    # Initialize d[n,t]
    d = np.zeros((N, T), dtype=float)

    # We’ll map times to positions once
    time_pos = {t: j for j, t in enumerate(time_index)}

    # For each region, get its buses + alphas
    for region in load_df.columns:
        region_buses = bf[bf[region_col] == int(region)]
        if region_buses.empty:
            continue

        # For each bus in this region, apply alpha * regional load at each time
        for _, row in region_buses.iterrows():
            bus_id = row["Bus ID"]
            alpha = row["alpha"]
            n = bus_idx[bus_id]

            # vectorize across time
            # load_df.loc[time_index, region] is a Series of length T
            d[n, :] += alpha * load_df.loc[time_index, region].to_numpy(dtype=float)

    return d


def _aggregate_to_blocks(hourly: np.ndarray, block_size: int) -> np.ndarray:
    """Average groups of consecutive columns (time axis) into blocks.

    Parameters
    ----------
    hourly : np.ndarray
        Array with time as the last axis. Shape (..., T_hourly).
    block_size : int
        Number of consecutive hours per block.

    Returns
    -------
    np.ndarray
        Averaged array with shape (..., T_hourly // block_size).
    """
    *leading, T = hourly.shape
    n_blocks = T // block_size
    # Reshape last axis into (n_blocks, block_size) then average
    reshaped = hourly[..., : n_blocks * block_size].reshape(*leading, n_blocks, block_size)
    return reshaped.mean(axis=-1)


def build_damdata_from_rts(
    source_dir: str | Path,
    ts_dir: str | Path,
    start_time: pd.Timestamp,
    horizon_hours: int = 48,
    spp_forecasts_parquet: str | Path | None = None,
    spp_start_idx: int = 0,
    day2_interval_hours: int = 1,
    single_block: bool = False,
    include_renewables: bool = False,
    include_nuclear: bool = False,
    include_zero_marginal: bool | None = None,
    ramp_scale: float = 1.0,
    pmin_scale: float = 1.0,
) -> DAMData:
    """
    High-level function:
    RTS-GMLC SourceData + time series -> DAMData for the Gurobi builder.

    Parameters
    ----------
    source_dir : path to RTS_Data/SourceData
    ts_dir     : path to RTS_Data/timeseries_data_files (or similar)
    time_selector : indices of hours to include (e.g. range(24))

    Returns
    -------
    DAMData
    """

    tables = load_rts_source_tables(source_dir)
    buses_df = tables["buses"]
    lines_df = tables["lines"]
    gens_df = tables["gens"]

    # 1) Simplify/filter generators
    # Resolve combined flag
    if include_zero_marginal is not None:
        include_renewables = include_zero_marginal
        include_nuclear = include_zero_marginal

    # Base filter: always exclude STORAGE and SYNC_COND
    exclude_types = {"STORAGE", "SYNC_COND"}
    if not include_renewables:
        exclude_types |= {"PV", "RTPV", "HYDRO", "ROR"}
    if not include_nuclear:
        exclude_types |= {"NUCLEAR"}

    keep_mask = ~gens_df["Unit Type"].isin(exclude_types)
    n_excluded = (~keep_mask).sum()
    if n_excluded > 0:
        excluded_types = gens_df.loc[~keep_mask, "Unit Type"].value_counts()
        print(f"  Excluded {n_excluded} generators: {dict(excluded_types)}")
    gens_df = gens_df.loc[keep_mask].copy()

    # Example: add type labels compatible with DAMData.gen_type
    # (map RTS technology codes to 'THERMAL', 'WIND', etc.)
    tech = gens_df["Unit Type"].astype(str)

    # 2) Build indices

    bus_ids = sorted(buses_df["Bus ID"].unique())
    gen_ids = sorted(gens_df["GEN UID"].unique())
    line_ids = sorted(lines_df["UID"].unique())

    bus_idx = {bus_id: n for n, bus_id in enumerate(bus_ids)}
    gen_idx = {gen_id: i for i, gen_id in enumerate(gen_ids)}

    I = len(gen_ids)
    N = len(bus_ids)
    L = len(line_ids)
    T = horizon_hours
    B = 3

    # 3) Core generator arrays
    gen_to_bus = np.zeros(I, dtype=int)
    Pmin = np.zeros(I)
    Pmax = np.zeros(I)
    RU = np.zeros(I)
    RD = np.zeros(I)
    MUT = np.zeros(I)
    MDT = np.zeros(I)
    startup_cost = np.zeros(I)
    shutdown_cost = np.zeros(I)
    no_load_cost = np.zeros(I)
    u_init = np.zeros(I)
    init_up_time = np.zeros(I)
    init_down_time = np.zeros(I)
    block_cap = np.zeros((I, B))
    block_cost = np.zeros((I, B))
    gen_type = [""] * I

    # Populate in sorted order inside the loop
    for gen_id, row in gens_df.set_index("GEN UID").iterrows():
        i = gen_idx[gen_id]  # Sorted position
        unit_type_upper = row["Unit Type"].upper()
        if "WIND" in unit_type_upper:
            gen_type[i] = "WIND"
        elif unit_type_upper in ("PV", "RTPV"):
            gen_type[i] = "SOLAR"
        elif unit_type_upper in ("HYDRO", "ROR"):
            gen_type[i] = "HYDRO"
        else:
            gen_type[i] = "THERMAL"  # CT, CC, STEAM, NUCLEAR, CSP

        gen_to_bus[i] = bus_idx[row["Bus ID"]]
        Pmin[i] = row["PMin MW"]
        Pmax[i] = row["PMax MW"]
        RU[i] = row["Ramp Rate MW/Min"]
        RD[i] = row["Ramp Rate MW/Min"]
        MUT[i] = row["Min Up Time Hr"]
        MDT[i] = row["Min Down Time Hr"]
        startup_cost[i] = (
            row["Start Heat Warm MBTU"] * row["Fuel Price $/MMBTU"]
            + row["Non Fuel Start Cost $"]
        )
        shutdown_cost[i] = row.get(
            "ShutdownCost", 0.0
        )  # RTS doesn't have shutdown costs
        no_load_cost[i] = (
            row["Fuel Price $/MMBTU"]
            * (row["HR_avg_0"] - row["HR_incr_1"])
            * Pmin[i]
            / 1000
        )  # RTS doesn't have no-load costs

        # Initial conditions: adjust based on actual RTS columns.
        # e.g., 'InitialStatus', 'InitialUpTime', 'InitialDownTime'
        u_init[i] = min(int(row.get("MW Inj", 0)), 1)
        init_up_time[i] = row.get("InitialUpTime", 0.0)
        init_down_time[i] = row.get("InitialDownTime", 0.0)

        # 4) Cost blocks
        if gen_type[i] in ("WIND", "SOLAR", "HYDRO"):
            # Renewables/hydro: single block covering full capacity, zero marginal cost.
            # Time-varying availability enforced by Pmax_2d constraint.
            block_cap[i, 0] = Pmax[i]
            block_cap[i, 1] = 0.0
            block_cap[i, 2] = 0.0
            # Tiny negative cost for wind to break LP degeneracy:
            # ensures optimizer maximizes wind dispatch before other zero-cost gen.
            block_cost[i, 0] = -0.001 if gen_type[i] == "WIND" else 0.0
            block_cost[i, 1] = 0.0
            block_cost[i, 2] = 0.0
        else:
            block_cap[i, 0] = np.clip(row.get("Output_pct_1", 0), 0, 1) * Pmax[i]
            block_cap[i, 1] = (
                np.clip(row.get("Output_pct_2", 0) - row.get("Output_pct_1", 0), 0, 1)
                * Pmax[i]
            )
            block_cap[i, 2] = (
                np.clip(row.get("Output_pct_3", 0) - row.get("Output_pct_2", 0), 0, 1)
                * Pmax[i]
            )

            block_cost[i, 0] = row.get("HR_incr_1") * row["Fuel Price $/MMBTU"] / 1000
            block_cost[i, 1] = row.get("HR_incr_2") * row["Fuel Price $/MMBTU"] / 1000
            block_cost[i, 2] = row.get("HR_incr_3") * row["Fuel Price $/MMBTU"] / 1000

    # Scale Pmin if requested (sensitivity test for headroom constraints)
    if pmin_scale != 1.0:
        Pmin *= pmin_scale
        # Clamp: Pmin must stay >= 0 and <= Pmax
        Pmin = np.clip(Pmin, 0.0, Pmax)
        print(f"  Pmin scaled by {pmin_scale}x")

    # Scale ramp rates if requested
    if ramp_scale != 1.0:
        RU *= ramp_scale
        RD *= ramp_scale
        print(f"  Ramp rates scaled by {ramp_scale}x")

    # Collapse 3 blocks → 1 block with weighted-average marginal cost
    if single_block:
        avg_cost = np.zeros(I)
        for i in range(I):
            total_cap = block_cap[i, :].sum()
            if total_cap > 0:
                avg_cost[i] = (block_cost[i, :] * block_cap[i, :]).sum() / total_cap
        block_cap_1 = np.zeros((I, 1))
        block_cap_1[:, 0] = Pmax
        block_cost_1 = np.zeros((I, 1))
        block_cost_1[:, 0] = avg_cost
        block_cap = block_cap_1
        block_cost = block_cost_1

    # 5) Load / net injection array d[n,t]
    # 0) Build full time-series once
    load_df, wind_df, pv_df, rtpv_df = load_rts_timeseries(ts_dir)

    end_time = start_time + pd.Timedelta(hours=horizon_hours)

    # Slice to [start_time, end_time)
    load_window = load_df.loc[
        (load_df.index.get_level_values("time") >= start_time)
        & (load_df.index.get_level_values("time") < end_time)
    ].copy()

    # Build local time index (sorted unique timestamps)
    time_index = sorted(load_window.index.get_level_values("time").unique())

    # Example: assume load_df has index (bus_id, time), col 'load_mw'
    bus_factors = compute_bus_load_factors(
        buses_df,
        region_col="Area",  # or "Area"/whatever matches your RTS
        load_col="MW Load",
    )
    d = build_nodal_load_array(
        load_df=load_window,
        bus_factors=bus_factors,
        bus_ids=bus_ids,
        time_index=time_index,
        region_col="Area",
    )
    # For now, ignore wind_df in d; later you can subtract fixed injections
    # 6) Build time-varying Pmax for wind generators
    # This is CRITICAL for the ARUC model!
    if spp_forecasts_parquet is not None:
        # Use SPP ensemble mean forecast (scaled to RTS capacity)
        print(f"  Using SPP ensemble mean for wind Pmax: {spp_forecasts_parquet}")
        pmax_wind = build_wind_pmax_from_spp_ensemble(
            forecasts_parquet=spp_forecasts_parquet,
            gen_ids=gen_ids,
            gen_type=gen_type,
            static_pmax=Pmax,
            n_hours=T,
            start_idx=spp_start_idx,
        )
    else:
        # Original: read DAY_AHEAD_wind.csv
        wind_window = wind_df.loc[
            (wind_df.index >= start_time) & (wind_df.index < end_time)
        ].copy()

        pmax_wind = build_wind_pmax_array(
            wind_df=wind_window,
            gens_df=gens_df,
            gen_ids=gen_ids,
            time_index=time_index,
            gen_type=gen_type,
        )

    # 6b) Build time-varying Pmax for solar generators (PV + RTPV)
    has_solar = any(gt == "SOLAR" for gt in gen_type)
    if has_solar:
        pv_window = pv_df.loc[
            (pv_df.index >= start_time) & (pv_df.index < end_time)
        ].copy()
        rtpv_window = rtpv_df.loc[
            (rtpv_df.index >= start_time) & (rtpv_df.index < end_time)
        ].copy()

        pmax_solar = build_solar_pmax_array(
            pv_df=pv_window,
            rtpv_df=rtpv_window,
            gens_df=gens_df,
            gen_ids=gen_ids,
            time_index=time_index,
            gen_type=gen_type,
        )
    else:
        pmax_solar = np.zeros((I, T))

    # Create Pmax_t array: I x T
    # Wind and solar use time-varying forecasts; thermal/hydro use static Pmax
    is_wind = np.array([gt.upper() == "WIND" for gt in gen_type])
    is_solar = np.array([gt.upper() == "SOLAR" for gt in gen_type])

    Pmax_t = np.zeros((I, T), dtype=float)
    for i in range(I):
        if is_wind[i]:
            Pmax_t[i, :] = pmax_wind[i, :]
        elif is_solar[i]:
            Pmax_t[i, :] = pmax_solar[i, :]
        else:
            Pmax_t[i, :] = Pmax[i]  # Static capacity for thermal/hydro
    # 6) Network: PTDF and Fmax
    PTDF, Fmax, bus_ids, line_ids = build_dc_ptdf(buses_df, lines_df)

    # ---- Variable-duration period aggregation ----
    period_duration = None
    if day2_interval_hours > 1 and horizon_hours > 24:
        day1_hours = 24
        day2_hourly = horizon_hours - day1_hours
        if day2_hourly % day2_interval_hours != 0:
            raise ValueError(
                f"Day-2 hours ({day2_hourly}) must be divisible by "
                f"day2_interval_hours ({day2_interval_hours})"
            )
        day2_periods = day2_hourly // day2_interval_hours
        T_new = day1_hours + day2_periods

        # Aggregate load: average over block
        d_day1 = d[:, :day1_hours]
        d_day2 = _aggregate_to_blocks(d[:, day1_hours:], day2_interval_hours)
        d = np.concatenate([d_day1, d_day2], axis=1)

        # Aggregate Pmax_t: average over block
        pmax_day1 = Pmax_t[:, :day1_hours]
        pmax_day2 = _aggregate_to_blocks(Pmax_t[:, day1_hours:], day2_interval_hours)
        Pmax_t = np.concatenate([pmax_day1, pmax_day2], axis=1)

        # Aggregate time_index: use start timestamp of each block
        time_day1 = time_index[:day1_hours]
        time_day2 = [time_index[day1_hours + j * day2_interval_hours]
                     for j in range(day2_periods)]
        time_index = time_day1 + time_day2

        # Build period_duration vector
        period_duration = np.array(
            [1.0] * day1_hours + [float(day2_interval_hours)] * day2_periods
        )

        print(f"  Variable intervals: {day1_hours} × 1h + {day2_periods} × {day2_interval_hours}h = {T_new} periods")

    # 7) Build DAMData
    dam = DAMData(
        gen_ids=gen_ids,
        bus_ids=bus_ids,
        line_ids=line_ids,
        time=time_index,
        gen_type=list(gen_type),
        gen_to_bus=gen_to_bus,
        Pmin=Pmin,
        Pmax=Pmax_t,
        RU=RU,
        RD=RD,
        MUT=MUT,
        MDT=MDT,
        startup_cost=startup_cost,
        shutdown_cost=shutdown_cost,
        no_load_cost=no_load_cost,
        u_init=u_init,
        init_up_time=init_up_time,
        init_down_time=init_down_time,
        block_cap=block_cap,
        block_cost=block_cost,
        PTDF=PTDF,
        Fmax=Fmax,
        d=d,
        period_duration=period_duration,
        gens_df=gens_df,
        buses_df=buses_df,
        lines_df=lines_df,
    )

    dam.validate_shapes()
    return dam


if __name__ == "__main__":
    # Example usage
    source_dir = "RTS_Data/SourceData"
    ts_dir = "RTS_Data/timeseries_data_files"
    start_time = pd.Timestamp("2020-01-01 00:00:00")
    damdata = build_damdata_from_rts(source_dir, ts_dir, start_time, horizon_hours=48)
    print(damdata)
