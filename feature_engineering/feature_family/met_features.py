#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree  
from feature_registry import register_stage_features


DATE_COL = "date_utc"
LAT_COL = "latitude"  
LON_COL = "longitude"  
PM_COL = "pm25_daily_mean"
CITY_COL = "City"
NAME_COL = "Name"
DENY_NAMES = {"region", "abbrevs", "names", "__inpoly"}

# -----------------------------
# Daily aggregate vars
# -----------------------------
DEFAULT_DAILY_AGG_VARS = [
    "WS10", "blh", "RH", "t2m", "VPD", "VC",
    # common precip candidates (use whatever exists)
    "tp", "precip", "precipitation", "pr",
]

# -----------------------------
# Snapping helpers (to avoid NaNs near coasts/water)
# -----------------------------
SNAP_REF_VARS = ["sp", "t2m", "u10", "v10", "d2m"]  # core vars used to define "valid" land-like cells


# -----------------------------
# Generic helpers
# -----------------------------
def _ensure_utc_naive(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    return t.dt.tz_convert("UTC").dt.tz_localize(None)


def _slugify(*parts) -> str:
    s = "_".join(str(p) for p in parts if pd.notna(p) and str(p).strip())
    s = s.lower()
    s = re.sub(r"[^\w]+", "_", s)
    return re.sub(r"__+", "_", s).strip("_") or "unknown"


def _create_deterministic_sensor_id(lat: float, lon: float) -> int:
    """
    Create stable, deterministic sensor_id that never changes across pipeline runs.
    
    Uses SHA1 hash of ONLY rounded coordinates for maximum stability.
    Returns consistent integer ID regardless of data subset, processing order, 
    or presence/absence of city/name metadata.
    
    Args:
        lat, lon: Sensor coordinates (rounded to 5 decimal places for stability)
        
    Returns:
        Deterministic integer sensor_id (never changes for same location)
    """
    lat_rounded = round(float(lat), 5)
    lon_rounded = round(float(lon), 5)
    id_string = f"{lat_rounded:.5f},{lon_rounded:.5f}"
    h = hashlib.sha1(id_string.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _numeric_feature_vars(ds: xr.Dataset) -> list[str]:
    keep: list[str] = []
    for v in ds.data_vars:
        if v in DENY_NAMES:
            continue
        if np.issubdtype(ds[v].dtype, np.number):
            keep.append(v)
    return keep


def _strip_coords(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.drop_vars(list(DENY_NAMES), errors="ignore")
    return ds.reset_coords(drop=True)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized haversine distance (km).
    """
    R = 6371.0
    l1, L1, l2, L2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((l2 - l1) / 2) ** 2 + np.cos(l1) * np.cos(l2) * np.sin((L2 - L1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _normalize_lon_to_ds(ds_lon_vals: np.ndarray, sensor_lons: np.ndarray) -> np.ndarray:
    """
    Make sensor longitudes compatible with dataset longitude convention.
    If ds lon looks like 0..360 and sensors are -180..180, convert sensors.
    """
    lon_max = float(np.nanmax(ds_lon_vals))
    smin = float(np.nanmin(sensor_lons))
    if lon_max > 180 and smin < 0:
        return (sensor_lons + 360.0) % 360.0
    return sensor_lons


def _build_valid_mask_2d(ds: xr.Dataset, ref_vars: list[str] | None = None) -> xr.DataArray:
    """
    Build boolean (latitude, longitude) mask for grid-cells that look valid ("land-like").
    Rule: all available ref_vars must be finite at a reference time (time index 0 if present).
    """
    if ref_vars is None:
        ref_vars = SNAP_REF_VARS

    present = [v for v in ref_vars if v in ds.data_vars]
    if not present:
        nums = _numeric_feature_vars(ds)
        if not nums:
            raise ValueError("No numeric variables found to build valid mask.")
        present = [nums[0]]

    base = ds[present]
    if "time" in base.dims and base.sizes.get("time", 0) > 0:
        base = base.isel(time=0)

    mask = None
    for v in present:
        m = np.isfinite(base[v])
        mask = m if mask is None else (mask & m)

    if "latitude" not in mask.dims or "longitude" not in mask.dims:
        raise ValueError("Expected (latitude, longitude) dims for valid mask.")
    return mask.astype(bool)


def _build_kdtree_for_valid_cells(ds: xr.Dataset, valid_mask_2d: xr.DataArray):
    """
    Build KDTree over valid grid-cell centers in (lat, lon) space.
    Returns (tree, valid_lat, valid_lon, lon_scale).
    ASSUMES scipy.spatial.cKDTree is available.
    """
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    valid = valid_mask_2d.values.astype(bool)
    vlat = lat2d[valid].astype("float64")
    vlon = lon2d[valid].astype("float64")

    if vlat.size == 0:
        raise ValueError("Valid-mask has zero valid cells; cannot build snap tree.")

    lat0 = float(np.nanmedian(vlat))
    lon_scale = float(np.cos(np.deg2rad(lat0)))
    lon_scale = max(lon_scale, 1e-3)

    pts = np.column_stack([vlat, vlon * lon_scale])
    tree = cKDTree(pts)
    return tree, vlat, vlon, lon_scale


def _snap_points_to_valid(
    tree,
    vlat: np.ndarray,
    vlon: np.ndarray,
    lon_scale: float,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    q = np.column_stack([lat.astype("float64"), lon.astype("float64") * lon_scale])
    _, idx = tree.query(q, k=1)
    return vlat[idx], vlon[idx]


def _nearest_grid_latlon(ds: xr.Dataset, lat_arr: np.ndarray, lon_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat_nn = ds["latitude"].sel(latitude=xr.DataArray(lat_arr, dims="points"), method="nearest").values
    lon_nn = ds["longitude"].sel(longitude=xr.DataArray(lon_arr, dims="points"), method="nearest").values
    return lat_nn, lon_nn


def load_daily_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in [DATE_COL, LAT_COL, LON_COL, PM_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Daily CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # time: daily UTC-naive
    df["time"] = _ensure_utc_naive(df[DATE_COL]).dt.floor("D")

    # coords
    df["latitude"] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df["longitude"] = pd.to_numeric(df[LON_COL], errors="coerce")

    # target
    df["pm25"] = pd.to_numeric(df[PM_COL], errors="coerce")

    # FIXED: Always enforce deterministic sensor_id that never changes across runs
    # Extract optional city/name fields for ID generation
    city_vals = df[CITY_COL].fillna("") if CITY_COL in df.columns else [""] * len(df)
    name_vals = df[NAME_COL].fillna("") if NAME_COL in df.columns else [""] * len(df)
    
    # Generate deterministic hash-based sensor IDs based ONLY on coordinates (always stable)
    df["sensor_id"] = [
        _create_deterministic_sensor_id(lat, lon)
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    
    # Create readable sensor names for human reference (not used as IDs)
    if CITY_COL in df.columns or NAME_COL in df.columns:
        df["sensor_name"] = [_slugify(c, n) for c, n in zip(city_vals, name_vals)]
    else:
        df["sensor_name"] = [
            f"pt_{lat:.4f}_{lon:.4f}" for lat, lon in zip(df["latitude"], df["longitude"])
        ]

    # keep only valid rows
    df = df.dropna(subset=["time", "latitude", "longitude"]).copy()
    df = df.drop_duplicates(subset=["sensor_id", "time"]).copy()

    # stable column order: core cols first, then everything else in original order
    base_cols = ["sensor_id", "sensor_name", "time", "latitude", "longitude", "pm25"]
    rest_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + rest_cols].copy()

    df["ym"] = df["time"].dt.to_period("M")
    return df


# -----------------------------
# 2) Find monthly feature files features_YYYY_MM.nc
# -----------------------------
def find_feature_files(features_dir: Path) -> dict[pd.Period, Path]:
    rx = re.compile(r"features_(\d{4})_(\d{2})\.nc$")
    idx: dict[pd.Period, Path] = {}
    for p in features_dir.rglob("*.nc"):
        m = rx.search(p.name)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            idx[pd.Period(year=y, month=mo, freq="M")] = p
    return idx


def _present_daily_agg_vars(ds: xr.Dataset, requested: list[str]) -> list[str]:
    return [v for v in requested if v in ds.data_vars]


def sample_month_daily_snapshot_and_agg(
    df_month: pd.DataFrame,
    nc_path: Path,
    chunk: int = 50000,
    interp_method: str = "linear",
    daily_agg_vars: list[str] | None = None,
) -> pd.DataFrame:
    """
    For one month:
      A) Snapshot sample at (time, lat, lon) for each row.
         If NaNs appear at a point (coast/water), snap to nearest valid cell and resample.
      B) Optional true daily aggregates for selected vars (mean/max/min across the day)
         sampled at each sensor location.
    """
    if df_month.empty:
        return df_month.copy()

    ds = xr.open_dataset(nc_path, drop_variables=["step"])
    ds = _strip_coords(ds)

    # time slice (just the days we need)
    try:
        tmin = df_month["time"].min()
        tmax = df_month["time"].max() + pd.Timedelta(days=1)
        ds = ds.sel(time=slice(tmin, tmax))
    except Exception:
        pass

    # build snapping tree for this month
    valid_mask = _build_valid_mask_2d(ds, ref_vars=SNAP_REF_VARS)
    tree, vlat, vlon, lon_scale = _build_kdtree_for_valid_cells(ds, valid_mask)

    data_vars = _numeric_feature_vars(ds)
    if not data_vars:
        ds.close()
        raise ValueError(f"No numeric feature variables found in {nc_path.name}")

    ds_lon_vals = ds["longitude"].values
    n = len(df_month)
    out_parts: list[pd.DataFrame] = []

    # -------------------------
    # A) SNAPSHOT sampling
    # -------------------------
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = df_month.iloc[i0:i1].copy()

        obs_lat = block["latitude"].values.astype("float64")
        obs_lon = block["longitude"].values.astype("float64")
        obs_lon_n = _normalize_lon_to_ds(ds_lon_vals, obs_lon)

        times = xr.DataArray(block["time"].values, dims=("points",))
        lats = xr.DataArray(obs_lat, dims=("points",))
        lons = xr.DataArray(obs_lon_n, dims=("points",))

        # nearest gridpoint diagnostics from original coords
        grid_lat0, grid_lon0 = _nearest_grid_latlon(ds, obs_lat, obs_lon_n)
        grid_dist0 = haversine_km(obs_lat, obs_lon_n, grid_lat0, grid_lon0)

        sampled = ds[data_vars].interp(time=times, latitude=lats, longitude=lons, method=interp_method)
        sampled = sampled.reset_coords(drop=True)
        feat_df = sampled.to_dataframe().reset_index(drop=True)

        # "bad" rows: NaNs in core vars (or any vars if core not present)
        core_present = [v for v in SNAP_REF_VARS if v in feat_df.columns]
        if core_present:
            bad = feat_df[core_present].isna().any(axis=1).to_numpy()
        else:
            bad = feat_df.isna().any(axis=1).to_numpy()

        # Default: sample coords = obs coords
        sample_lat = obs_lat.copy()
        sample_lon = obs_lon_n.copy()
        snapped_flag = np.zeros_like(sample_lat, dtype=bool)
        snap_dist_km = np.zeros_like(sample_lat, dtype="float64")

        if bad.any():
            idx_bad = np.where(bad)[0]
            lat_s, lon_s = _snap_points_to_valid(
                tree, vlat, vlon, lon_scale,
                lat=obs_lat[idx_bad], lon=obs_lon_n[idx_bad]
            )

            sample_lat[idx_bad] = lat_s
            sample_lon[idx_bad] = lon_s
            snapped_flag[idx_bad] = True
            snap_dist_km[idx_bad] = haversine_km(obs_lat[idx_bad], obs_lon_n[idx_bad], lat_s, lon_s)

            # re-sample for only bad points
            times_bad = xr.DataArray(block["time"].values[idx_bad], dims=("points",))
            lats_bad = xr.DataArray(sample_lat[idx_bad], dims=("points",))
            lons_bad = xr.DataArray(sample_lon[idx_bad], dims=("points",))

            sampled2 = ds[data_vars].interp(time=times_bad, latitude=lats_bad, longitude=lons_bad, method=interp_method)
            sampled2 = sampled2.reset_coords(drop=True)
            feat_df2 = sampled2.to_dataframe().reset_index(drop=True)

            # overwrite bad rows
            feat_df.iloc[idx_bad, :] = feat_df2.values

        # Diagnostics for "used" sample coords
        grid_lat1, grid_lon1 = _nearest_grid_latlon(ds, sample_lat, sample_lon)
        grid_dist1 = haversine_km(sample_lat, sample_lon, grid_lat1, grid_lon1)

        # merge back
        base_block_cols = [c for c in block.columns if c not in {"ym"}]
        block_out = (
            block[base_block_cols]
            .reset_index(drop=True)
            .rename(columns={"latitude": "obs_lat", "longitude": "obs_lon"})
        )

        feat_df["grid_lat"] = grid_lat0
        feat_df["grid_lon"] = grid_lon0
        feat_df["grid_dist_km"] = grid_dist0.astype("float32")

        feat_df["sample_lat"] = sample_lat
        feat_df["sample_lon"] = sample_lon
        feat_df["snapped_flag"] = snapped_flag.astype("int8")
        feat_df["snap_dist_km"] = snap_dist_km.astype("float32")

        feat_df["sample_grid_lat"] = grid_lat1
        feat_df["sample_grid_lon"] = grid_lon1
        feat_df["sample_grid_dist_km"] = grid_dist1.astype("float32")

        merged = pd.concat([block_out, feat_df], axis=1)
        out_parts.append(merged)

    snap = pd.concat(out_parts, ignore_index=True)

    # -------------------------
    # B) TRUE DAILY aggregates (optional)
    # -------------------------
    if daily_agg_vars is None:
        daily_agg_vars = DEFAULT_DAILY_AGG_VARS

    present_agg_vars = _present_daily_agg_vars(ds, daily_agg_vars)
    if not present_agg_vars:
        ds.close()
        return snap

    sensors = (
        df_month[["sensor_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("sensor_id")
        .rename(columns={"latitude": "lat", "longitude": "lon"})
        .reset_index(drop=True)
    )
    if sensors.empty:
        ds.close()
        return snap

    sensors["lon_n"] = _normalize_lon_to_ds(ds_lon_vals, sensors["lon"].to_numpy())

    # Use same snapping logic as snapshot:
    # if a sensor is "bad" at time0, snap it for daily-aggregate sampling.
    base = ds[present_agg_vars]
    if "time" in base.dims and base.sizes.get("time", 0) > 0:
        base0 = base.isel(time=0)
    else:
        base0 = base

    lats0 = xr.DataArray(sensors["lat"].to_numpy(), dims="points")
    lons0 = xr.DataArray(sensors["lon_n"].to_numpy(), dims="points")
    sampled0 = base0.interp(latitude=lats0, longitude=lons0, method=interp_method).to_dataframe().reset_index(drop=True)
    bad0 = sampled0.isna().any(axis=1).to_numpy()

    lat_use = sensors["lat"].to_numpy().astype("float64")
    lon_use = sensors["lon_n"].to_numpy().astype("float64")

    if bad0.any():
        lat_s, lon_s = _snap_points_to_valid(
            tree, vlat, vlon, lon_scale,
            lat=lat_use[bad0], lon=lon_use[bad0]
        )
        lat_use[bad0] = lat_s
        lon_use[bad0] = lon_s

    sensors["agg_lat"] = lat_use
    sensors["agg_lon"] = lon_use
    sensors["agg_snapped_flag"] = bad0.astype("int8")
    sensors["agg_snap_dist_km"] = haversine_km(
        sensors["lat"].to_numpy(), sensors["lon_n"].to_numpy(), lat_use, lon_use
    ).astype("float32")

    # sample full hourly (or sub-daily) time series at each sensor point, then aggregate by day
    lats_p = xr.DataArray(sensors["agg_lat"].to_numpy(), dims=("points",))
    lons_p = xr.DataArray(sensors["agg_lon"].to_numpy(), dims=("points",))

    ts_sampled = ds[present_agg_vars].interp(latitude=lats_p, longitude=lons_p, method=interp_method)
    ts_df = ts_sampled.to_dataframe().reset_index()  # time, points, vars...
    ts_df["time"] = pd.to_datetime(ts_df["time"])
    ts_df["date"] = ts_df["time"].dt.floor("D")

    sensors_points = sensors.reset_index().rename(columns={"index": "points"})
    ts_df = ts_df.merge(sensors_points[["points", "sensor_id"]], on="points", how="left")

    # aggregate per sensor-day: mean/max/min
    agg = ts_df.groupby(["sensor_id", "date"], as_index=False)[present_agg_vars].agg(["mean", "max", "min"])

    # flatten columns
    new_cols = []
    for var, stat in agg.columns:
        if var in ("sensor_id", "date"):
            new_cols.append(var)
        else:
            new_cols.append(f"{var}_daily_{stat}")
    agg.columns = new_cols
    agg = agg.reset_index(drop=True).rename(columns={"date": "time"})

    out = snap.merge(agg, on=["sensor_id", "time"], how="left")
    out = out.merge(sensors[["sensor_id", "agg_snapped_flag", "agg_snap_dist_km"]], on="sensor_id", how="left")

    ds.close()
    return out


def compute_met_for_observations(
    df_obs: pd.DataFrame,
    met_dir: Path,
    chunk: int = 50000,
    interp_method: str = "linear",
    daily_agg_vars: list[str] | None = None,
    add_level2: bool = True,
) -> pd.DataFrame:
    """
    ETL wrapper: Compute MET features for a DataFrame of observations.

    This is the feature_store-compatible interface that takes an already-loaded
    DataFrame and returns features without writing to disk.

    Args:
        df_obs: DataFrame with columns: date_utc/date, latitude, longitude, pm25_daily_mean
                (same format as paqi_network_daily.csv)
        met_dir: Path to directory containing features_YYYY_MM.nc files
        chunk: Processing chunk size
        interp_method: Interpolation method ('linear' or 'nearest')
        daily_agg_vars: Variables for daily aggregation (uses defaults if None)
        add_level2: If True, add level 2 engineered features (rolling, lags, etc.)

    Returns:
        DataFrame with MET features for each observation row

    Example:
        >>> df_obs = pd.read_csv("data/paqi_network_daily.csv")
        >>> df_met = compute_met_for_observations(df_obs, Path("datasets/features_met"))
    """
    # Prepare the observations DataFrame (same logic as load_daily_sensor_csv)
    df = df_obs.copy()

    # Ensure required columns exist
    required_cols = [DATE_COL, LAT_COL, LON_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # Try alternate column names
        if "date" in df.columns and DATE_COL not in df.columns:
            df[DATE_COL] = df["date"]
        if "lat" in df.columns and LAT_COL not in df.columns:
            df[LAT_COL] = df["lat"]
        if "lon" in df.columns and LON_COL not in df.columns:
            df[LON_COL] = df["lon"]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Parse time
    df["time"] = _ensure_utc_naive(df[DATE_COL]).dt.floor("D")

    # Coordinates
    df["latitude"] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df["longitude"] = pd.to_numeric(df[LON_COL], errors="coerce")

    # Target (optional for feature computation)
    if PM_COL in df.columns:
        df["pm25"] = pd.to_numeric(df[PM_COL], errors="coerce")

    # Sensor ID (deterministic hash-based)
    df["sensor_id"] = [
        _create_deterministic_sensor_id(lat, lon)
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]

    # Sensor name for readability
    city_vals = df[CITY_COL].fillna("") if CITY_COL in df.columns else [""] * len(df)
    name_vals = df[NAME_COL].fillna("") if NAME_COL in df.columns else [""] * len(df)
    if CITY_COL in df.columns or NAME_COL in df.columns:
        df["sensor_name"] = [_slugify(c, n) for c, n in zip(city_vals, name_vals)]
    else:
        df["sensor_name"] = [f"pt_{lat:.4f}_{lon:.4f}" for lat, lon in zip(df["latitude"], df["longitude"])]

    # Clean up
    df = df.dropna(subset=["time", "latitude", "longitude"]).copy()
    df = df.drop_duplicates(subset=["sensor_id", "time"]).copy()
    df["ym"] = df["time"].dt.to_period("M")

    # Find feature files
    feat_map = find_feature_files(Path(met_dir))

    if df.empty:
        raise ValueError("No valid rows after parsing observations DataFrame")

    months = sorted(df["ym"].unique())
    print(f"[met-etl] Processing {len(df):,} observations across {len(months)} month(s)")

    all_parts: list[pd.DataFrame] = []
    ok = missing_count = 0

    for ym in months:
        p = feat_map.get(ym)
        label = f"{ym.year}-{ym.month:02d}"
        if not p or not p.exists():
            print(f"[met-etl] MISSING {label}: features_{ym.year}_{ym.month:02d}.nc")
            missing_count += 1
            continue

        sub = df.loc[df["ym"] == ym].copy()
        print(f"[met-etl] SAMPLING {label}: {len(sub):,} rows from {p.name}")

        sampled = sample_month_daily_snapshot_and_agg(
            sub, p,
            chunk=chunk,
            interp_method=interp_method,
            daily_agg_vars=daily_agg_vars,
        )
        all_parts.append(sampled)
        ok += 1

    if not all_parts:
        raise ValueError("No monthly met feature files found for the observation dates")

    out = pd.concat(all_parts, ignore_index=True)
    out = out.drop(columns=["ym"], errors="ignore")

    print(f"[met-etl] Sampling done. Months OK: {ok} | Months missing: {missing_count}")

    # Add level 2 features if requested
    if add_level2:
        print(f"[met-etl] Adding level 2 features...")
        out = add_met_level2_daily(out)
        print(f"[met-etl] Final: {len(out):,} rows, {len(out.columns)} columns")

    return out


def sample_met_features_daily(
    daily_csv: Path,
    features_dir: Path,
    chunk: int = 50000,
    interp_method: str = "linear",
    daily_agg_vars: list[str] | None = None,
) -> pd.DataFrame:
    df = load_daily_sensor_csv(daily_csv)
    feat_map = find_feature_files(features_dir)

    if df.empty:
        raise SystemExit("Daily CSV has no valid rows after parsing.")

    months = sorted(df["ym"].unique())
    print(f"[daily-met] Found {len(df):,} sensor-day rows across {len(months)} month(s)")
    print(f"[daily-met] Snapshot interpolation method: {interp_method}")
    print(f"[daily-met] Daily aggregates vars: {daily_agg_vars or DEFAULT_DAILY_AGG_VARS}")

    all_parts: list[pd.DataFrame] = []
    ok = missing = 0

    for ym in months:
        p = feat_map.get(ym)
        label = f"{ym.year}-{ym.month:02d}"
        if not p or not p.exists():
            print(f"[daily-met] MISSING {label}: features_{ym.year}_{ym.month:02d}.nc")
            missing += 1
            continue

        sub = df.loc[df["ym"] == ym].copy()
        print(f"[daily-met] SAMPLING {label}: {len(sub):,} rows from {p.name}")

        sampled = sample_month_daily_snapshot_and_agg(
            sub,
            p,
            chunk=chunk,
            interp_method=interp_method,
            daily_agg_vars=daily_agg_vars,
        )
        all_parts.append(sampled)
        ok += 1

    if not all_parts:
        raise SystemExit("No monthly met feature files were found for the daily CSV's months.")

    out = pd.concat(all_parts, ignore_index=True)
    out = out.drop(columns=["ym"], errors="ignore")

    print(f"[daily-met] Done. Months OK: {ok} | Months missing: {missing}")
    return out


# -----------------------------
# 3) DAILY Met Level-2 features
# -----------------------------
def _rolling_feat_daily(g: pd.core.groupby.DataFrameGroupBy, col: str, window: int, how: str) -> pd.Series:
    roll = g[col].rolling(window, min_periods=1)
    if how == "mean":
        out = roll.mean()
    elif how == "std":
        out = roll.std()
    elif how == "min":
        out = roll.min()
    elif how == "max":
        out = roll.max()
    elif how == "sum":
        out = roll.sum()
    else:
        raise ValueError(f"Unknown rolling op {how}")
    return out.reset_index(level=0, drop=True)


def _create_time_segments(df: pd.DataFrame, gap_threshold_days: float = 2.0) -> pd.Series:
    """
    FIXED: Create segment IDs that increment whenever there's a time gap > gap_threshold_days.
    This allows time-aware rolling operations within continuous segments only.
    
    Args:
        df: DataFrame with 'sensor_id' and 'time' columns, sorted by (sensor_id, time)
        gap_threshold_days: Time gap threshold in days to trigger new segment
        
    Returns:
        Series of segment_id values (sensor_id specific, increments at gaps)
    """
    # Compute time differences within each sensor group
    g_time = df.groupby("sensor_id", sort=False)["time"]
    dt_days = g_time.diff().dt.total_seconds() / (3600.0 * 24.0)
    
    # Mark gaps (including first record of each sensor as gap)
    gap_mask = (dt_days > gap_threshold_days) | dt_days.isna()
    
    # Create segment increments (cumulative gap count per sensor)
    g_gaps = gap_mask.groupby(df["sensor_id"], sort=False)
    segment_increments = g_gaps.cumsum()
    
    # Create global segment IDs by combining sensor_id and segment increment
    # This ensures segments are unique across sensors
    sensor_codes = pd.factorize(df["sensor_id"], sort=False)[0]
    max_segments_per_sensor = segment_increments.max() + 1
    global_segment_id = sensor_codes * max_segments_per_sensor + segment_increments
    
    return global_segment_id


def _time_aware_rolling(
    df: pd.DataFrame, 
    segment_col: str, 
    target_col: str, 
    window: int, 
    agg: str, 
    min_pct: float = 0.5
) -> pd.Series:
    """
    FIXED: Perform rolling operation within time segments only (never across gaps).
    
    Args:
        df: DataFrame containing data
        segment_col: Column name containing segment IDs
        target_col: Column to apply rolling operation to
        window: Rolling window size
        agg: Aggregation method ('mean', 'std', 'min', 'max', 'sum')
        min_pct: Minimum fraction of window required for non-null result
        
    Returns:
        Series with rolling aggregation results
    """
    g = df.groupby(segment_col, sort=False)[target_col]
    min_periods = max(1, int(window * min_pct))
    return getattr(g.rolling(window, min_periods=min_periods), agg)().reset_index(level=0, drop=True)


def _time_aware_shift(df: pd.DataFrame, segment_col: str, target_col: str, periods: int) -> pd.Series:
    """
    FIXED: Perform lag shift within time segments only (never across gaps).
    
    Args:
        df: DataFrame containing data  
        segment_col: Column name containing segment IDs
        target_col: Column to shift
        periods: Number of periods to shift (positive for lag)
        
    Returns:
        Series with shifted values (NaN where shift crosses segment boundary)
    """
    g = df.groupby(segment_col, sort=False)[target_col]
    return g.shift(periods).reset_index(level=0, drop=True)


def add_met_level2_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily analogue of the hourly met L2 script.
    Uses day-based lags/rollings. Exogenous only (no PM history).
    """
    df = df.copy()

    if "time" not in df.columns or "sensor_id" not in df.columns:
        raise ValueError("Expected 'time' and 'sensor_id' columns.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["sensor_id", "time"])
    g = df.groupby("sensor_id", sort=False)
    
    # FIXED: Create time segments early to prevent rolling across gaps
    df["time_segment_id"] = _create_time_segments(df, gap_threshold_days=2.0)

    df["day_of_year"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month

    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    df["doy_sin_2"] = np.sin(4 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos_2"] = np.cos(4 * np.pi * df["day_of_year"] / 366.0)
    df["doy_sin_3"] = np.sin(6 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos_3"] = np.cos(6 * np.pi * df["day_of_year"] / 366.0)

    df["heating_season_flag"] = df["month"].isin([11, 12, 1, 2]).astype("int8")
    df["burning_season_flag"] = df["month"].isin([10, 11]).astype("int8")

    # WS10 - FIXED: Time-aware operations (never cross gaps)
    if "WS10" in df.columns:
        df["WS10_lag1d"] = _time_aware_shift(df, "time_segment_id", "WS10", 1)
        df["WS10_lag3d"] = _time_aware_shift(df, "time_segment_id", "WS10", 3)

        df["WS10_rollmean_3d"] = _time_aware_rolling(df, "time_segment_id", "WS10", 3, "mean")
        df["WS10_rollstd_3d"] = _time_aware_rolling(df, "time_segment_id", "WS10", 3, "std")
        df["WS10_rollmin_7d"] = _time_aware_rolling(df, "time_segment_id", "WS10", 7, "min")
        df["WS10_rollmax_7d"] = _time_aware_rolling(df, "time_segment_id", "WS10", 7, "max")

        inst_calm = (df["WS10"] < 2.0).astype("int8")
        df["WS10_calm_inst"] = inst_calm  # temporary column for rolling
        df["calm3_count"] = _time_aware_rolling(df, "time_segment_id", "WS10_calm_inst", 3, "sum")
        df["calm3_flag"] = (df["calm3_count"] >= 2).astype("int8")

        df["calm7_count"] = _time_aware_rolling(df, "time_segment_id", "WS10_calm_inst", 7, "sum")
        df["calm7_flag"] = (df["calm7_count"] >= 5).astype("int8")
        
        # Clean up temporary column
        df = df.drop(columns=["WS10_calm_inst"], errors="ignore")

    # BLH - FIXED: Time-aware operations (never cross gaps)
    if "blh" in df.columns:
        df["blh_lag1d"] = _time_aware_shift(df, "time_segment_id", "blh", 1)
        df["blh_lag3d"] = _time_aware_shift(df, "time_segment_id", "blh", 3)

        df["blh_rollmean_3d"] = _time_aware_rolling(df, "time_segment_id", "blh", 3, "mean")
        df["blh_rollmin_7d"] = _time_aware_rolling(df, "time_segment_id", "blh", 7, "min")

        df["blh_rollmean_7d"] = _time_aware_rolling(df, "time_segment_id", "blh", 7, "mean")
        df["blh_anom_7d"] = df["blh"] - df["blh_rollmean_7d"]

        df["blh_rollmean_14d"] = _time_aware_rolling(df, "time_segment_id", "blh", 14, "mean")
        df["blh_anom_14d"] = df["blh"] - df["blh_rollmean_14d"]

    # RH - FIXED: Time-aware operations (never cross gaps)
    if "RH" in df.columns:
        df["RH_rollmean_3d"] = _time_aware_rolling(df, "time_segment_id", "RH", 3, "mean")
        df["RH_rollmax_7d"] = _time_aware_rolling(df, "time_segment_id", "RH", 7, "max")
        df["RH_rollmean_7d"] = _time_aware_rolling(df, "time_segment_id", "RH", 7, "mean")
        df["RH_anom_7d"] = df["RH"] - df["RH_rollmean_7d"]
        df["RH_rollstd_7d"] = _time_aware_rolling(df, "time_segment_id", "RH", 7, "std")

    # VPD - FIXED: Time-aware operations (never cross gaps)
    if "VPD" in df.columns:
        df["VPD_rollmean_3d"] = _time_aware_rolling(df, "time_segment_id", "VPD", 3, "mean")
        df["VPD_rollmean_7d"] = _time_aware_rolling(df, "time_segment_id", "VPD", 7, "mean")

    # VC - FIXED: Time-aware operations (never cross gaps)
    if "VC" in df.columns:
        df["VC_rollmean_3d"] = _time_aware_rolling(df, "time_segment_id", "VC", 3, "mean")
        df["VC_rollmin_7d"] = _time_aware_rolling(df, "time_segment_id", "VC", 7, "min")

        df["VC_rollmean_7d"] = _time_aware_rolling(df, "time_segment_id", "VC", 7, "mean")
        df["VC_anom_7d"] = df["VC"] - df["VC_rollmean_7d"]

        df["VC_rollmean_14d"] = _time_aware_rolling(df, "time_segment_id", "VC", 14, "mean")
        df["VC_anom_14d"] = df["VC"] - df["VC_rollmean_14d"]

    # Stagnation proxy - FIXED: Time-aware operations (never cross gaps)
    if ("WS10" in df.columns) and ("blh" in df.columns):
        inst_stag = ((df["WS10"] < 2.0) & (df["blh"] < 300.0)).astype("int8")
        df["stagnant_inst"] = inst_stag  # temporary column for rolling

        df["stagnant3_count"] = _time_aware_rolling(df, "time_segment_id", "stagnant_inst", 3, "sum")
        df["stagnant3_flag"] = (df["stagnant3_count"] >= 2).astype("int8")

        df["stagnant7_count"] = _time_aware_rolling(df, "time_segment_id", "stagnant_inst", 7, "sum")
        df["stagnant7_flag"] = (df["stagnant7_count"] >= 5).astype("int8")
        
        # Clean up temporary column
        df = df.drop(columns=["stagnant_inst"], errors="ignore")

    # Wind direction stability - FIXED: Time-aware operations (never cross gaps)
    if "WD10" in df.columns:
        df["WD10_sin"] = np.sin(np.deg2rad(df["WD10"]))
        df["WD10_cos"] = np.cos(np.deg2rad(df["WD10"]))

        df["WD10_sin_rm_7d"] = _time_aware_rolling(df, "time_segment_id", "WD10_sin", 7, "mean")
        df["WD10_cos_rm_7d"] = _time_aware_rolling(df, "time_segment_id", "WD10_cos", 7, "mean")
        R7 = np.sqrt(df["WD10_sin_rm_7d"] ** 2 + df["WD10_cos_rm_7d"] **  2)
        df["WD10_var_7d"] = (1.0 - R7).clip(0, 1)

        df["WD10_sin_rm_14d"] = _time_aware_rolling(df, "time_segment_id", "WD10_sin", 14, "mean")
        df["WD10_cos_rm_14d"] = _time_aware_rolling(df, "time_segment_id", "WD10_cos", 14, "mean")
        R14 = np.sqrt(df["WD10_sin_rm_14d"] ** 2 + df["WD10_cos_rm_14d"] ** 2)
        df["WD10_var_14d"] = (1.0 - R14).clip(0, 1)

    # Shear / tendencies - FIXED: Time-aware operations (never cross gaps)
    if "dWS" in df.columns:
        df["dWS_abs"] = df["dWS"].abs()
        df["dWS_rollmean_3d"] = _time_aware_rolling(df, "time_segment_id", "dWS", 3, "mean")
        df["dWS_rollstd_3d"] = _time_aware_rolling(df, "time_segment_id", "dWS", 3, "std")

    if "dWD" in df.columns:
        df["dWD_abs"] = df["dWD"].abs()
        df["dWD_rollstd_3d"] = _time_aware_rolling(df, "time_segment_id", "dWD", 3, "std")

    for col in ["BLH_tend", "MSLP_tend", "SP_tend", "dT"]:
        if col in df.columns:
            df[f"{col}_rollmean_3d"] = _time_aware_rolling(df, "time_segment_id", col, 3, "mean")
            df[f"{col}_rollstd_3d"] = _time_aware_rolling(df, "time_segment_id", col, 3, "std")
            df[f"{col}_rollmean_7d"] = _time_aware_rolling(df, "time_segment_id", col, 7, "mean")
            df[f"{col}_rollstd_7d"] = _time_aware_rolling(df, "time_segment_id", col, 7, "std")

    # FIXED: No longer need gap reset logic - time-aware operations handle gaps correctly
    # Add dt_days for potential diagnostic use (but gap handling now built into features)
    df["dt_days"] = g["time"].diff().dt.total_seconds() / (3600.0 * 24.0)
    
    # Clean up time_segment_id (internal helper column)
    df = df.drop(columns=["time_segment_id"], errors="ignore")

    return df


# -----------------------------
# Orchestrator (NO satellite PM2.5 step)
# -----------------------------
def run_pipeline(
    daily_csv: Path,
    features_dir: Path,
    out_csv: Path,
    chunk: int = 50000,
    interp_method: str = "linear",
    daily_agg_vars: list[str] | None = None,
):
    print(f"[pipeline] Loading DAILY targets: {daily_csv}")
    print(f"[pipeline] Using HARD-CODED columns: {DATE_COL}, {LAT_COL}, {LON_COL}, {PM_COL}")

    # Check if output file already exists with features
    skip_processing = False
    if out_csv.exists():
        try:
            existing_df = pd.read_csv(out_csv)
            # Check for key features that indicate processing is complete
            key_features = ["WS10_lag1d", "blh_rollmean_7d", "doy_sin", "heating_season_flag"]
            if all(col in existing_df.columns for col in key_features):
                print(f"[pipeline] Output file {out_csv} already exists with features - skipping processing")
                print(f"[pipeline] Existing file has {len(existing_df):,} rows, {len(existing_df.columns)} cols")
                skip_processing = True
        except Exception as e:
            print(f"[pipeline] Warning: Could not read existing file {out_csv}: {e}")

    if not skip_processing:
        # 1) met snapshot + optional true daily aggregates
        df_met = sample_met_features_daily(
            daily_csv,
            features_dir,
            chunk=chunk,
            interp_method=interp_method,
            daily_agg_vars=daily_agg_vars,
        )
        print(f"[pipeline] After met sampling+agg: {len(df_met):,} rows, {len(df_met.columns)} cols")

        # 2) daily met L2
        df_final = add_met_level2_daily(df_met)
        print(f"[pipeline] After daily metL2: {len(df_final):,} rows, {len(df_final.columns)} cols")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # DON'T write df_final to out_csv; let register_stage_features() update the master
        
    # Always register features with the feature registry (even if skipped processing)
    if not skip_processing:
        # Register from the processed data
        register_met_features(df_final)
        print(f"[met_pipeline] Registry updated for {len(df_final):,} rows, {len(df_final.columns)} cols")
        return df_final
    else:
        # If skipped, load existing and register
        if out_csv.exists():
            df_final = pd.read_csv(out_csv)
            register_met_features(df_final)
            print(f"[met_pipeline] Registry updated for {len(df_final):,} rows, {len(df_final.columns)} cols")
            return df_final
    
    return None


def parse_args():
    ap = argparse.ArgumentParser(
        description="DAILY PAQI feature engineering pipeline (MET sampling + daily metL2)."
    )
    ap.add_argument("--daily_csv", default="data/paqi_network_daily.csv", type=Path)
    ap.add_argument("--features_dir", default="datasets/features_met", type=Path,
                    help="Folder containing features_YYYY_MM.nc files")
    ap.add_argument("--csv_file", default="output/paqi_with_all_features.csv", type=Path,
                    help="Output CSV path")
    ap.add_argument("--chunk", type=int, default=50000)

    ap.add_argument("--interp_method", type=str, default="linear",
                    choices=["nearest", "linear"],
                    help="Interpolation method for snapshot + spatial sampling.")
    ap.add_argument("--daily_agg_vars", default="",
                    help="Comma-separated list of vars to compute daily mean/max/min for. "
                         "Default uses a sensible list (WS10, blh, RH, t2m, VPD, VC, precip candidates).")

    return ap.parse_args()


def main():
    args = parse_args()

    daily_agg_vars = None
    if args.daily_agg_vars.strip():
        daily_agg_vars = [v.strip() for v in args.daily_agg_vars.split(",") if v.strip()]

    run_pipeline(
        daily_csv=args.daily_csv,
        features_dir=args.features_dir,
        out_csv=args.csv_file,
        chunk=args.chunk,
        interp_method=args.interp_method,
        daily_agg_vars=daily_agg_vars,
    )


def register_met_features(data: pd.DataFrame) -> None:
    """
    Register meteorological features with the feature registry system.
    """
    # Base meteorological features (from file reading)
    base_met_features = []
    
    # Level 2 engineered features (time-based)
    engineered_features = [
        # Temporal encodings
        "doy_sin", "doy_cos", "doy_sin_2", "doy_cos_2", "doy_sin_3", "doy_cos_3",
        "heating_season_flag", "burning_season_flag",
        
        # Wind speed features
        "WS10_lag1d", "WS10_lag3d", "WS10_rollmean_3d", "WS10_rollstd_3d",
        "WS10_rollmin_7d", "WS10_rollmax_7d",
        "calm3_count", "calm3_flag", "calm7_count", "calm7_flag",
        
        # Boundary layer height features
        "blh_lag1d", "blh_lag3d", "blh_rollmean_3d", "blh_rollmin_7d",
        "blh_rollmean_7d", "blh_anom_7d", "blh_rollmean_14d", "blh_anom_14d",
        
        # Relative humidity features
        "RH_rollmean_3d", "RH_rollmax_7d", "RH_rollmean_7d", "RH_anom_7d", "RH_rollstd_7d",
        
        # Vapor pressure deficit features
        "VPD_rollmean_3d", "VPD_rollmean_7d",
        
        # Vegetation condition features
        "VC_rollmean_3d", "VC_rollmin_7d", "VC_rollmean_7d", "VC_anom_7d",
    ]
    
    # Find base meteorological variables in the dataset
    possible_met_vars = ["WS10", "blh", "RH", "t2m", "VPD", "VC", "tp", "precip", "precipitation", "pr",
                        "sp", "u10", "v10", "d2m", "WD10", "snowd", "sf"]  # Add more as needed
    
    for var in possible_met_vars:
        if var in data.columns:
            base_met_features.append(var)
            # Also check for daily aggregated versions (the correct naming)
            for stat in ["mean", "max", "min", "std"]:
                agg_var = f"{var}_daily_{stat}"
                if agg_var in data.columns:
                    base_met_features.append(agg_var)
    
    # Sampling diagnostics (metadata)
    sampling_metadata = [
        "grid_lat", "grid_lon", "grid_dist_km", "sample_lat", "sample_lon", 
        "snapped_flag", "snap_dist_km", "agg_snapped_flag", "agg_snap_dist_km", 
        "sample_grid_lat", "sample_grid_lon", "sample_grid_dist_km"
    ]
    
    # Met metadata patterns: snapping + target-quality aggregates + sampling diagnostics
    meta_cols = [c for c in data.columns if (
        c.startswith(("grid_", "sample_", "snap_", "snapped_", "agg_"))
        or c in {
            "snapped_flag", "agg_snapped_flag", "agg_snap_dist_km",
            "pm25_daily_mean","pm25_daily_median","pm25_daily_p95","n_hours_valid","daily_complete_flag", 
            "dt_days", "day_of_year", "month"
        }
    )]
    
    # Only include columns that actually exist in the dataset
    existing_base_features = [col for col in base_met_features if col in data.columns]
    existing_engineered_features = [col for col in engineered_features if col in data.columns]
    # Feature columns: everything else that's numeric and not in meta_cols or join_keys/core_meta/target
    join_keys = {"sensor_id","time"}
    core_meta = {"obs_lat","obs_lon","City","Name","date_utc","latitude","longitude"}
    target = {"pm25"}

    feat_cols = []
    for c in data.columns:
        if c in join_keys or c in core_meta or c in target or c in meta_cols:
            continue
        # only keep numeric as features
        if pd.api.types.is_numeric_dtype(data[c]):
            feat_cols.append(c)
    
    # Add descriptions
    descriptions = {}
    
    # Base met variable descriptions
    met_descriptions = {
        "WS10": "Wind speed at 10m height",
        "WD10": "Wind direction at 10m height", 
        "blh": "Boundary layer height",
        "RH": "Relative humidity",
        "t2m": "Temperature at 2m height",
        "VPD": "Vapor pressure deficit",
        "VC": "Vegetation condition index",
        "sp": "Surface pressure",
        "u10": "10m u-component of wind",
        "v10": "10m v-component of wind",
        "d2m": "2m dewpoint temperature",
        "tp": "Total precipitation",
        "precip": "Precipitation",
        "precipitation": "Precipitation",
        "pr": "Precipitation rate",
        "snowd": "Snow depth",
        "sf": "Snowfall",
    }
    
    for col in feat_cols:
        base_var = col.split("_")[0]  # Remove suffixes like _mean, _max
        desc = met_descriptions.get(base_var, f"Meteorological variable: {base_var}")
        if "doy_sin" in col or "doy_cos" in col:
            descriptions[col] = f"Seasonal encoding: {col}"
        elif "season_flag" in col:
            descriptions[col] = f"Seasonal indicator: {col}"
        elif "lag" in col:
            descriptions[col] = f"Lagged feature: {col}"
        elif "roll" in col:
            descriptions[col] = f"Rolling window feature: {col}"
        elif "anom" in col:
            descriptions[col] = f"Anomaly feature: {col}"
        elif "calm" in col:
            descriptions[col] = f"Low wind condition feature: {col}"
        elif "_" in col:
            suffix = col.split("_", 1)[1]
            descriptions[col] = f"{desc} ({suffix} aggregation)"
        else:
            descriptions[col] = desc
    
    # Metadata descriptions
    for col in meta_cols:
        descriptions[col] = f"Meteorological metadata: {col}"
    
    # Register with the feature registry
    register_stage_features(
        stage="met",
        data=data,
        feature_columns=feat_cols,
        metadata_columns=meta_cols,
        descriptions=descriptions
    )


if __name__ == "__main__":
    main()
