from __future__ import annotations
from sklearn.neighbors import BallTree
import plotly.graph_objects as go
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


EARTH_R = 6_371_000.0  # meters


# ----------------------------
# Geometry helpers
# ----------------------------
def latlon_to_xy_m(lat, lon, lat0, lon0):
    """Equirectangular projection around (lat0, lon0). Good for regional footprints."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    x = (lon - lon0) * np.cos(lat0) * EARTH_R
    y = (lat - lat0) * EARTH_R
    return np.column_stack([x, y])


def xy_to_latlon(x, y, lat0, lon0):
    """Inverse of latlon_to_xy_m for the same (lat0, lon0) origin."""
    lat0r = np.radians(lat0)
    lon0r = np.radians(lon0)

    lat = (y / EARTH_R) + lat0r
    lon = (x / (EARTH_R * np.cos(lat0r))) + lon0r
    return np.column_stack([np.degrees(lat), np.degrees(lon)])


# ----------------------------
# Constellation matcher
# ----------------------------
def best_constellation_match(A_latlon, B_latlon, allow_scale=True):
    """
    Find the best 4-point constellation match of A inside B, up to translation + rotation
    (+ optional uniform scale), using an anchor-edge transform search.

    Inputs:
      A_latlon: (4,2) array [[lat, lon], ...]
      B_latlon: (M,2) array [[lat, lon], ...]
      allow_scale: whether to allow uniform scaling

    Returns dict:
      score: float (sum of NN distances in meters)
      s: float scale
      R: (2,2) rotation matrix
      t: (2,) translation vector in projected meters
      snapped_B_idx: (4,) indices into B of the 4 matched points (unique enforced)
      anchor_A: (i0, i1) anchor edge indices in A
      anchor_B: (j0, j1) anchor edge indices in B
    """
    A_latlon = np.asarray(A_latlon, float)
    B_latlon = np.asarray(B_latlon, float)

    if A_latlon.shape != (4, 2):
        raise ValueError(f"Expected A_latlon shape (4,2), got {A_latlon.shape}")
    if B_latlon.ndim != 2 or B_latlon.shape[1] != 2:
        raise ValueError(f"Expected B_latlon shape (M,2), got {B_latlon.shape}")

    # Project both to local XY around A centroid
    lat0, lon0 = A_latlon[:, 0].mean(), A_latlon[:, 1].mean()
    A = latlon_to_xy_m(A_latlon[:, 0], A_latlon[:, 1], lat0, lon0)
    B = latlon_to_xy_m(B_latlon[:, 0], B_latlon[:, 1], lat0, lon0)

    # NN structure on B
    tree = BallTree(B, leaf_size=40, metric="euclidean")

    # Anchor edge in A: farthest pair (stable)
    dA = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    i0, i1 = np.unravel_index(np.argmax(dA), dA.shape)
    if i0 == i1:
        i0, i1 = 0, 1

    a0, a1 = A[i0], A[i1]
    va = a1 - a0
    la = np.linalg.norm(va)
    if la == 0:
        raise ValueError("Template has duplicate points (zero anchor edge length).")

    best = {
        "score": np.inf,
        "s": None,
        "R": None,
        "t": None,
        "snapped_B_idx": None,
        "anchor_A": (int(i0), int(i1)),
        "anchor_B": None,
        "lat0": float(lat0),
        "lon0": float(lon0),
    }

    M = B.shape[0]

    # Candidate anchor edges in B
    for j0 in range(M):
        for j1 in range(j0 + 1, M):
            b0, b1 = B[j0], B[j1]
            vb = b1 - b0
            lb = np.linalg.norm(vb)
            if lb == 0:
                continue

            s = (lb / la) if allow_scale else 1.0

            # rotation mapping va -> vb in 2D
            u = va / la
            v = vb / lb
            c = float(u @ v)
            cross = float(u[0] * v[1] - u[1] * v[0])  # 2D cross scalar

            R = np.array([[c, -cross], [cross, c]], dtype=float)

            # translation so a0 -> b0
            t = b0 - (s * (R @ a0))

            # transform all A points
            A_hat = (s * (A @ R.T)) + t

            # snap each transformed point to nearest B point
            dist, ind = tree.query(A_hat, k=1)
            dist = dist.flatten()
            ind = ind.flatten().astype(int)

            # enforce uniqueness (no collisions)
            if len(set(ind.tolist())) < 4:
                continue

            score = float(dist.sum())

            if score < best["score"]:
                best.update(
                    {
                        "score": score,
                        "s": float(s),
                        "R": R,
                        "t": t.astype(float),
                        "snapped_B_idx": ind,
                        "anchor_B": (int(j0), int(j1)),
                    }
                )

    return best


# ----------------------------
# Your readers (unchanged)
# ----------------------------
def read_rts_wind(gen_path, bus_path):
    df_rts_wind = pd.read_csv(gen_path)
    df_buses = pd.read_csv(bus_path)

    df_rts_wind = df_rts_wind[df_rts_wind["Fuel"] == "Wind"][["GEN UID", "Bus ID"]]
    df_buses = df_buses[["Bus ID", "lat", "lng"]]

    df = df_rts_wind.merge(df_buses, how="left", on="Bus ID")
    return df


def read_spp_latlong(spp_path):
    df = pd.read_csv(spp_path)
    df = df.rename(
        columns={"RESOURCEASSETNAME": "SPP_name", "LATITUDE": "lat", "LONGITUDE": "lng"}
    )
    return df


# ----------------------------
# Main: run + plot
# ----------------------------
def find_and_plot_rts_to_spp_constellation(
    gen_path: str,
    bus_path: str,
    spp_path: str,
    *,
    allow_scale: bool = True,
    show: bool = True,
    label_col: str | None = None,
    zoom: float = 5.0,
):
    """
    Read RTS wind points (expects 4) and SPP resource points, find best 4-point constellation
    match (translation + rotation + optional scale), and produce a Plotly map overlay.

    Returns:
        result: dict from best_constellation_match
        matched_spp: df subset of spp with the matched 4 points
        fig: plotly.graph_objects.Figure
    """
    rts = (
        read_rts_wind(gen_path, bus_path)
        .dropna(subset=["lat", "lng"])
        .reset_index(drop=True)
    )
    spp = (
        read_spp_latlong(spp_path).dropna(subset=["lat", "lng"]).reset_index(drop=True)
    )

    if len(rts) != 4:
        raise ValueError(
            f"Expected 4 RTS points, got {len(rts)}. Check filters / missing lat/lng."
        )

    # Match using raw arrays (lat, lng)
    A_latlon = rts[["lat", "lng"]].to_numpy()
    B_latlon = spp[["lat", "lng"]].to_numpy()

    result = best_constellation_match(A_latlon, B_latlon, allow_scale=allow_scale)

    if result["snapped_B_idx"] is None:
        raise RuntimeError(
            "No valid match found (likely due to degeneracy or uniqueness constraint)."
        )

    matched_spp = spp.iloc[result["snapped_B_idx"]].copy()
    # --- Build RTS -> SPP name mapping ---
    rts_name_col = "GEN UID" if "GEN UID" in rts.columns else "Bus ID"

    mapping_df = pd.DataFrame(
        {
            "RTS_name": rts[rts_name_col].astype(str).values,
            "SPP_name": matched_spp["SPP_name"].astype(str).values,
            "RTS_lat": rts["lat"].values,
            "RTS_lng": rts["lng"].values,
            "SPP_lat": matched_spp["lat"].values,
            "SPP_lng": matched_spp["lng"].values,
        }
    )

    print("\n=== RTS -> SPP Mapping ===")
    print(mapping_df.to_string(index=False))

    # Print summary
    print(f"RTS points: {len(rts)} | SPP points: {len(spp)}")
    print("\n=== Best match ===")
    print(f"Total geometric error: {result['score']:.2f} meters")
    print(f"Matched SPP row indices: {result['snapped_B_idx']}")
    print(f"Scale s: {result['s']:.6f}")
    print(f"Rotation R:\n{result['R']}")
    print(f"Translation t: {result['t']}")
    print("\nMatched SPP resources:")
    print(matched_spp[["SPP_name", "lat", "lng"]].to_string(index=True))

    # Build transformed RTS overlay (convert RTS -> xy, apply transform, back to lat/lon)
    lat0, lon0 = result["lat0"], result["lon0"]
    A_xy = latlon_to_xy_m(A_latlon[:, 0], A_latlon[:, 1], lat0, lon0)
    A_hat_xy = (result["s"] * (A_xy @ result["R"].T)) + result["t"]
    A_hat_ll = xy_to_latlon(A_hat_xy[:, 0], A_hat_xy[:, 1], lat0, lon0)

    rts_overlay = rts.copy()
    rts_overlay["lat_overlay"] = A_hat_ll[:, 0]
    rts_overlay["lng_overlay"] = A_hat_ll[:, 1]

    if label_col is None:
        label_col = (
            "GEN UID"
            if "GEN UID" in rts_overlay.columns
            else ("Bus ID" if "Bus ID" in rts_overlay.columns else None)
        )
    rts_labels = (
        rts_overlay[label_col].astype(str)
        if label_col
        else [f"RTS_{i}" for i in range(len(rts_overlay))]
    )

    # Plotly map overlay
    fig = go.Figure()

    fig.add_trace(
        go.Scattermap(
            lat=spp["lat"],
            lon=spp["lng"],
            mode="markers",
            marker=dict(size=6, opacity=0.25),
            text=spp["SPP_name"] if "SPP_name" in spp.columns else None,
            name="All SPP resources",
            hovertemplate=(
                "SPP: %{text}<br>(%{lat:.4f}, %{lon:.4f})<extra></extra>"
                if "SPP_name" in spp.columns
                else "(%{lat:.4f}, %{lon:.4f})<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scattermap(
            lat=matched_spp["lat"],
            lon=matched_spp["lng"],
            mode="markers+text",
            marker=dict(size=12),
            text=matched_spp["SPP_name"] if "SPP_name" in matched_spp.columns else None,
            textposition="top center",
            name="Matched SPP (best 4)",
            hovertemplate=(
                "MATCHED SPP: %{text}<br>(%{lat:.4f}, %{lon:.4f})<extra></extra>"
                if "SPP_name" in matched_spp.columns
                else "(%{lat:.4f}, %{lon:.4f})<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scattermap(
            lat=rts_overlay["lat_overlay"],
            lon=rts_overlay["lng_overlay"],
            mode="markers+text",
            marker=dict(size=14, symbol="star"),
            text=rts_labels,
            textposition="bottom center",
            name="RTS constellation (transformed)",
            hovertemplate="RTS: %{text}<br>(%{lat:.4f}, %{lon:.4f})<extra></extra>",
        )
    )

    center_lat = float(matched_spp["lat"].mean())
    center_lon = float(matched_spp["lng"].mean())

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        title="RTS 4-point constellation overlaid onto best-matching 4 SPP resources",
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )

    if show:
        fig.show()

    return result, matched_spp, mapping_df, fig


# -----------------------------
# Basic filter + prep
# -----------------------------
def filter_actuals_forecasts_to_resources(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    keep_resources: Iterable[str],
    *,
    time_col: str = "TIME_HOURLY",
    id_col: str = "ID_RESOURCE",
    actual_col: str = "ACTUAL",
    model_col: str = "MODEL",
    forecast_col: str = "FORECAST",
    utc: bool = True,
    dedupe: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_set = {str(x) for x in keep_resources}

    req_a = {time_col, id_col, actual_col}
    req_f = {time_col, model_col, id_col, forecast_col}
    if miss := (req_a - set(actuals.columns)):
        raise ValueError(f"actuals missing columns: {sorted(miss)}")
    if miss := (req_f - set(forecasts.columns)):
        raise ValueError(f"forecasts missing columns: {sorted(miss)}")

    a = actuals[[time_col, id_col, actual_col]].copy()
    f = forecasts[[time_col, model_col, id_col, forecast_col]].copy()

    a[time_col] = pd.to_datetime(a[time_col], utc=utc)
    f[time_col] = pd.to_datetime(f[time_col], utc=utc)

    a[id_col] = a[id_col].astype(str)
    f[id_col] = f[id_col].astype(str)
    f[model_col] = f[model_col].astype(str)

    a = a[a[id_col].isin(keep_set)]
    f = f[f[id_col].isin(keep_set)]

    a[actual_col] = pd.to_numeric(a[actual_col], errors="coerce")
    f[forecast_col] = pd.to_numeric(f[forecast_col], errors="coerce")

    a = a.dropna(subset=[time_col, id_col, actual_col])
    f = f.dropna(subset=[time_col, model_col, id_col, forecast_col])

    if dedupe:
        a = a.groupby([time_col, id_col], as_index=False, observed=True)[
            actual_col
        ].mean()
        f = f.groupby([time_col, model_col, id_col], as_index=False, observed=True)[
            forecast_col
        ].mean()

    a = a.sort_values([time_col, id_col]).reset_index(drop=True)
    f = f.sort_values([time_col, model_col, id_col]).reset_index(drop=True)
    return a, f


# -----------------------------
# Rename SPP resource IDs -> RTS names using mapping_df
# -----------------------------
def rename_resources_to_rts_names(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    mapping_df: pd.DataFrame,
    *,
    id_col: str = "ID_RESOURCE",
    spp_col: str = "SPP_name",
    rts_col: str = "RTS_name",
    keep_original_id: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    mapping_df must contain columns: [RTS_name, SPP_name]
    Replaces ID_RESOURCE values (currently SPP_name) with RTS_name.
    """
    needed = {spp_col, rts_col}
    if missing := (needed - set(mapping_df.columns)):
        raise ValueError(f"mapping_df missing columns: {sorted(missing)}")

    # SPP_name -> RTS_name map
    spp_to_rts = dict(
        zip(
            mapping_df[spp_col].astype(str).values,
            mapping_df[rts_col].astype(str).values,
        )
    )

    a = actuals.copy()
    f = forecasts.copy()

    a[id_col] = a[id_col].astype(str)
    f[id_col] = f[id_col].astype(str)

    if keep_original_id:
        a["ID_RESOURCE_ORIG"] = a[id_col]
        f["ID_RESOURCE_ORIG"] = f[id_col]

    a[id_col] = a[id_col].map(spp_to_rts)
    f[id_col] = f[id_col].map(spp_to_rts)

    # sanity: everything should have mapped (since we filtered first)
    if a[id_col].isna().any() or f[id_col].isna().any():
        bad_a = a[a[id_col].isna()].head(5)
        bad_f = f[f[id_col].isna()].head(5)
        raise RuntimeError(
            "Some IDs did not map to RTS names. "
            "Did you filter before renaming? "
            f"Example unmapped actuals rows:\n{bad_a}\n"
            f"Example unmapped forecasts rows:\n{bad_f}\n"
        )

    return a, f


# -----------------------------
# Resource subsetting (e.g., 4-resource -> 3-resource)
# -----------------------------
def filter_rts_data_to_subset(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    mapping_df: pd.DataFrame,
    spp_names_to_keep: list[str],
    out_dir: Path,
    *,
    tag: str = "rts3",
    id_col: str = "ID_RESOURCE",
    spp_col: str = "SPP_name",
    rts_col: str = "RTS_name",
    compute_residuals: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter RTS data (already matched to SPP resources) to a subset of resources.

    Parameters:
      actuals: actuals DataFrame with ID_RESOURCE column (RTS names)
      forecasts: forecasts DataFrame with ID_RESOURCE column (RTS names)
      mapping_df: mapping from find_and_plot_rts_to_spp_constellation with SPP_name and RTS_name
      spp_names_to_keep: list of SPP_name values to keep
      out_dir: output directory for parquet files
      tag: tag for output filenames (e.g., "rts3")
      compute_residuals: if True, also compute and write residuals parquet

    Returns: (actuals_subset, forecasts_subset, mapping_subset)
    Writes:
      - actuals_filtered_{tag}.parquet
      - forecasts_filtered_{tag}.parquet
      - residuals_filtered_{tag}.parquet (if compute_residuals=True)
      - rts_to_spp_mapping_{tag}.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter mapping to keep only selected SPP resources
    keep_set = {str(x) for x in spp_names_to_keep}
    mapping_subset = mapping_df[mapping_df[spp_col].astype(str).isin(keep_set)].copy()

    if len(mapping_subset) == 0:
        raise ValueError(f"No resources matched from {spp_names_to_keep}")

    # Get corresponding RTS names
    rts_names_to_keep = set(mapping_subset[rts_col].astype(str).unique())

    # Filter actuals and forecasts to these RTS names
    a_subset = actuals[actuals[id_col].astype(str).isin(rts_names_to_keep)].copy()
    f_subset = forecasts[forecasts[id_col].astype(str).isin(rts_names_to_keep)].copy()

    # Write parquets
    actuals_path = out_dir / f"actuals_filtered_{tag}.parquet"
    forecasts_path = out_dir / f"forecasts_filtered_{tag}.parquet"
    mapping_path = out_dir / f"rts_to_spp_mapping_{tag}.csv"

    a_subset.to_parquet(actuals_path, index=False)
    f_subset.to_parquet(forecasts_path, index=False)
    mapping_subset.to_csv(mapping_path, index=False)

    print(f"\nWrote {tag} subset files:")
    print(f" - {actuals_path} ({len(a_subset)} rows)")
    print(f" - {forecasts_path} ({len(f_subset)} rows)")

    # Compute and write residuals if requested
    if compute_residuals:
        residuals_subset = compute_residuals_from_ensemble_mean(a_subset, f_subset)
        residuals_path = out_dir / f"residuals_filtered_{tag}.parquet"
        residuals_subset.to_parquet(residuals_path, index=False)
        print(f" - {residuals_path} ({len(residuals_subset)} rows)")

    print(f" - {mapping_path}")
    print(f"Resources: {sorted(rts_names_to_keep)}")

    return a_subset, f_subset, mapping_subset


# -----------------------------
# Residual computation
# -----------------------------
def compute_residuals_from_ensemble_mean(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    id_col: str = "ID_RESOURCE",
    actual_col: str = "ACTUAL",
    forecast_col: str = "FORECAST",
) -> pd.DataFrame:
    """
    Compute residuals = actuals - mean(forecast ensemble).

    Returns DataFrame with same structure as actuals, but with:
    - ACTUAL: original actual value (preserved)
    - MEAN_FORECAST: mean across ensemble members
    - RESIDUAL: ACTUAL - MEAN_FORECAST
    """
    # Compute mean forecast per (time, resource)
    forecast_mean = (
        forecasts.groupby([time_col, id_col], as_index=False)[forecast_col]
        .mean()
        .rename(columns={forecast_col: "MEAN_FORECAST"})
    )

    # Merge with actuals
    result = actuals.merge(forecast_mean, on=[time_col, id_col], how="left")

    # Compute residual
    result["RESIDUAL"] = result[actual_col] - result["MEAN_FORECAST"]

    return result


# -----------------------------
# Capacity scaling helpers
# -----------------------------
def compute_scale_factors(
    actuals: pd.DataFrame,
    mapping_df: pd.DataFrame,
    gen_path: str | Path,
    *,
    id_col: str = "ID_RESOURCE",
    actual_col: str = "ACTUAL",
    rts_col: str = "RTS_name",
) -> dict[str, float]:
    """
    Compute per-resource scale factors: RTS_Pmax / max(SPP_actuals).

    Parameters
    ----------
    actuals : DataFrame
        Actuals with ID_RESOURCE already renamed to RTS names.
    mapping_df : DataFrame
        Mapping with RTS_name column.
    gen_path : Path
        Path to gen.csv for RTS Pmax lookup.

    Returns
    -------
    scale_factors : dict[rts_name -> float]
    """
    gens_df = pd.read_csv(gen_path)
    rts_pmax = dict(zip(gens_df["GEN UID"].astype(str), gens_df["PMax MW"]))

    scale_factors = {}
    for rts_name in mapping_df[rts_col].astype(str).unique():
        # RTS nameplate capacity
        if rts_name not in rts_pmax:
            print(f"WARNING: {rts_name} not found in gen.csv, skipping scale factor")
            continue
        nameplate = rts_pmax[rts_name]

        # Max observed SPP actual for this resource
        mask = actuals[id_col].astype(str) == rts_name
        spp_max = actuals.loc[mask, actual_col].max()

        if spp_max <= 0:
            print(f"WARNING: max(ACTUAL) for {rts_name} is {spp_max}, using scale=1.0")
            scale_factors[rts_name] = 1.0
        else:
            scale_factors[rts_name] = nameplate / spp_max

    return scale_factors


def apply_scale_factors(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    scale_factors: dict[str, float],
    *,
    id_col: str = "ID_RESOURCE",
    actual_col: str = "ACTUAL",
    forecast_col: str = "FORECAST",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multiply ACTUAL and FORECAST columns by per-resource scale factor.

    Returns copies of actuals and forecasts with scaled values.
    """
    a = actuals.copy()
    f = forecasts.copy()

    for rts_name, sf in scale_factors.items():
        a_mask = a[id_col].astype(str) == rts_name
        f_mask = f[id_col].astype(str) == rts_name
        a.loc[a_mask, actual_col] = a.loc[a_mask, actual_col] * sf
        f.loc[f_mask, forecast_col] = f.loc[f_mask, forecast_col] * sf

    return a, f


# -----------------------------
# One-shot: match -> filter -> rename -> write cache
# -----------------------------
@dataclass(frozen=True)
class ConstellationPaths:
    gen_path: Path
    bus_path: Path
    spp_latlong_path: Path  # resource_latlong.csv


def cache_matched_rts_named_data(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    paths: ConstellationPaths,
    out_dir: Path,
    *,
    allow_scale: bool = True,
    show_map: bool = False,
    zoom: float = 6.0,
    tag: str = "rts4",
    model_prefix_filter: str | None = "SUBMODEL",
    capacity_scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Uses your existing find_and_plot_rts_to_spp_constellation(...) to get a 4-point match,
    filters actuals/forecasts to those 4 resources, renames to RTS names, and writes:

      - actuals_filtered_<tag>.parquet
      - forecasts_filtered_<tag>.parquet
      - rts_to_spp_mapping_<tag>.csv

    Returns: (actuals_rts, forecasts_rts, mapping_df)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- run constellation match (your function must be in scope) ----
    result, matched_spp, mapping_df, fig = find_and_plot_rts_to_spp_constellation(
        gen_path=str(paths.gen_path),
        bus_path=str(paths.bus_path),
        spp_path=str(paths.spp_latlong_path),
        allow_scale=allow_scale,
        show=show_map,
        label_col="GEN UID",
        zoom=zoom,
    )

    matched_spp_names = matched_spp["SPP_name"].astype(str).tolist()

    # ---- filter to just those 4 resources ----
    a4, f4 = filter_actuals_forecasts_to_resources(
        actuals=actuals,
        forecasts=forecasts,
        keep_resources=matched_spp_names,
    )

    # ---- rename ID_RESOURCE to RTS names ----
    a4_rts, f4_rts = rename_resources_to_rts_names(
        a4, f4, mapping_df, keep_original_id=True
    )

    # ---- MODEL filter (e.g., keep only SUBMODEL* rows) ----
    if model_prefix_filter is not None:
        n_before = len(f4_rts)
        f4_rts = f4_rts[f4_rts["MODEL"].str.startswith(model_prefix_filter)].copy()
        print(f"Filtered MODEL to '{model_prefix_filter}*': {n_before} -> {len(f4_rts)} rows")

    # ---- capacity scaling (RTS_Pmax / max(SPP_actuals)) ----
    if capacity_scale:
        scale_factors = compute_scale_factors(
            a4_rts, mapping_df, paths.gen_path
        )
        a4_rts, f4_rts = apply_scale_factors(a4_rts, f4_rts, scale_factors)
        mapping_df = mapping_df.copy()
        mapping_df["scale_factor"] = mapping_df["RTS_name"].map(scale_factors)
        print(f"Applied capacity scale factors: {scale_factors}")

    # ---- write cache ----
    actuals_path = out_dir / f"actuals_filtered_{tag}.parquet"
    forecasts_path = out_dir / f"forecasts_filtered_{tag}.parquet"
    mapping_path = out_dir / f"rts_to_spp_mapping_{tag}.csv"

    a4_rts.to_parquet(actuals_path, index=False)
    f4_rts.to_parquet(forecasts_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)

    # ---- compute and write residuals ----
    residuals = compute_residuals_from_ensemble_mean(a4_rts, f4_rts)
    residuals_path = out_dir / f"residuals_filtered_{tag}.parquet"
    residuals.to_parquet(residuals_path, index=False)

    print("\nWrote cached files:")
    print(" -", actuals_path)
    print(" -", forecasts_path)
    print(" -", residuals_path)
    print(" -", mapping_path)
    print(f"Geometric error (meters): {result['score']:.2f}")

    return a4_rts, f4_rts, mapping_df


if __name__ == "__main__":
    # Use paths relative to this module's location
    _MODULE_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _MODULE_DIR.parent
    _DATA_DIR = _MODULE_DIR / "data"

    df_actuals = pd.read_parquet(_DATA_DIR / "actuals_filtered.parquet")
    df_forecasts = pd.read_parquet(_DATA_DIR / "forecasts_filtered.parquet")

    paths = ConstellationPaths(
        gen_path=_PROJECT_ROOT / "RTS_Data" / "SourceData" / "gen.csv",
        bus_path=_PROJECT_ROOT / "RTS_Data" / "SourceData" / "bus.csv",
        spp_latlong_path=_DATA_DIR / "resource_latlong.csv",
    )

    out_dir = _DATA_DIR

    # --- v1: original (no MODEL filter, no scaling) ---
    a4_rts, f4_rts, mapping_df = cache_matched_rts_named_data(
        actuals=df_actuals,
        forecasts=df_forecasts,
        paths=paths,
        out_dir=out_dir,
        tag="rts4_constellation_v1",
        show_map=False,
        model_prefix_filter=None,
        capacity_scale=False,
    )

    # Create rts3 subset (exclude 303_WIND_1 / OKGE.SWCV.CV)
    rts3_spp_names = [
        "OKGE.GRANTPLNS.WIND.1",
        "OKGE.SDWE.SDWE",
        "MIDW.KACY.ALEX",
    ]
    a3_rts, f3_rts, mapping_df_3 = filter_rts_data_to_subset(
        actuals=a4_rts,
        forecasts=f4_rts,
        mapping_df=mapping_df,
        spp_names_to_keep=rts3_spp_names,
        out_dir=out_dir,
        tag="rts3_constellation_v1",
    )

    # --- v2: SUBMODEL filter + capacity scaling ---
    a4_rts_v2, f4_rts_v2, mapping_df_v2 = cache_matched_rts_named_data(
        actuals=df_actuals,
        forecasts=df_forecasts,
        paths=paths,
        out_dir=out_dir,
        tag="rts4_constellation_v2",
        show_map=False,
        model_prefix_filter="SUBMODEL",
        capacity_scale=True,
    )

    a3_rts_v2, f3_rts_v2, mapping_df_3_v2 = filter_rts_data_to_subset(
        actuals=a4_rts_v2,
        forecasts=f4_rts_v2,
        mapping_df=mapping_df_v2,
        spp_names_to_keep=rts3_spp_names,
        out_dir=out_dir,
        tag="rts3_constellation_v2",
    )
