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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RTS-GMLC time series: load and (optionally) wind/solar.

    Returns
    -------
    load_df :
        DataFrame indexed by ['bus_id', 't'], with column 'load_mw'.
    wind_df :
        DataFrame indexed by ['gen_id', 't'], with column 'pmax_mw'
        for dispatchable wind (can be empty if you want at first).
    """
    ts_path = Path(ts_dir)

    load_raw = pd.read_csv(ts_path / "Load" / "DAY_AHEAD_regional_Load.csv")
    load_raw = add_timestamp_column(load_raw)
    # Example normalization:
    # - assume columns: ["Year","Month","Day","Period","1","2","3"]
    load_df = load_raw.set_index(["time"]).sort_index()
    load_df = load_df.drop(columns={"Year", "Month", "Day", "Period"})
    # Wind:
    wind_raw = pd.read_csv(ts_path / "WIND" / "DAY_AHEAD_wind.csv")
    wind_raw = add_timestamp_column(wind_raw)
    wind_df = wind_raw.set_index(["time"]).sort_index()
    wind_df = wind_df.drop(columns={"Year", "Month", "Day", "Period"})
    return load_df, wind_df


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


def build_damdata_from_rts(
    source_dir: str | Path,
    ts_dir: str | Path,
    start_time: pd.Timestamp,
    horizon_hours: int = 48,
    spp_forecasts_parquet: str | Path | None = None,
    spp_start_idx: int = 0,
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
    # e.g., drop storage, sync cond, maybe hydro, keep thermal + wind.
    keep_mask = ~gens_df["Unit Type"].isin(["STORAGE", "SYNC_COND"])
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
        gen_type[i] = "WIND" if "WIND" in row["Unit Type"].upper() else "THERMAL"

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
        if gen_type[i] == "WIND":
            # Wind: single block covering full capacity, zero marginal cost.
            # Time-varying availability enforced by Pmax_2d constraint.
            block_cap[i, 0] = Pmax[i]
            block_cap[i, 1] = 0.0
            block_cap[i, 2] = 0.0
            block_cost[i, :] = 0.0
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

    # 5) Load / net injection array d[n,t]
    # 0) Build full time-series once
    load_df, wind_df = load_rts_timeseries(ts_dir)

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

    # For thermal units, use static Pmax
    # For wind units, use time-varying forecast
    is_wind = np.array([gt.upper() == "WIND" for gt in gen_type])

    # Store as Pmax_t attribute (will be used by Pmax_2d() method)
    # This requires modifying DAMData to accept time-varying Pmax
    # For now, we'll store it and let DAMData handle it

    # Create Pmax_t array: I x T
    Pmax_t = np.zeros((I, T), dtype=float)
    for i in range(I):
        if is_wind[i]:
            Pmax_t[i, :] = pmax_wind[i, :]
        else:
            Pmax_t[i, :] = Pmax[i]  # Static capacity for thermal
    # 6) Network: PTDF and Fmax
    PTDF, Fmax, bus_ids, line_ids = build_dc_ptdf(buses_df, lines_df)
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
