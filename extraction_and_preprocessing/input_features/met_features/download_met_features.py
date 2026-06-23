"""
Feature Engineering Batching Pipeline (FULL BBOX / NO MASKING) ============================================================= - Loads ERA5-Land + ERA5 for your requested rectangular AREA (the GRIB already contains only that bbox) - Does NOT mask / drop anything spatially (no ds.where(inside) anywhere) - Adds a Pakistan boundary indicator: __inpoly (0/1) as a data_var (NOT used to create NaNs) - Computes all derived features on the full bbox grid - Computes regime flags using thresholds computed ONLY over Pakistan (__inpoly==1), but outputs flags for the whole bbox (outside Pakistan => -1) This is designed as a drop-in replacement for your current feature-engineering batching script.
"""

import warnings
warnings.filterwarnings("ignore")

import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
from datetime import datetime
import geopandas as gpd
import regionmask
import re

# ------------------------------ User paths -------------------------------------
pakistan_geojson = "pakistan.geojson"

# ------------------------------ Helpers: I/O -----------------------------------
def resolve_path(p: Path):
    """Return a working path. If 'p' missing, try swapping .zip <-> .grib."""
    if p.exists():
        return p

    alt = None
    if p.suffix == ".zip":
        alt = p.with_suffix("")  # drop .zip -> .grib expected
    else:
        alt = p.with_suffix(p.suffix + ".zip")  # add .zip

    if alt and alt.exists():
        print(f"[resolver] Provided path missing, using: {alt}")
        return alt

    if p.suffix == ".grib":
        z = p.with_suffix(".grib.zip")
        if z.exists():
            print(f"[resolver] Provided path missing, using: {z}")
            return z

    if p.suffixes[-2:] == [".grib", ".zip"] and p.with_suffix("").exists():
        alt2 = p.with_suffix("")
        print(f"[resolver] Provided path missing, using: {alt2}")
        return alt2

    raise FileNotFoundError(str(p))


def _open_cfgrib_merge(path: Path):
    """Open a GRIB (possibly multi-group) and merge groups."""
    ds_list = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    ds = xr.merge(ds_list, compat="override")
    return ds


def load_era5land_data(file_path, cache_dir="_cache"):
    p = resolve_path(Path(file_path))
    if zipfile.is_zipfile(p):
        print("ERA5-Land: ZIP detected → extracting to cache...")
        cache_root = Path(cache_dir) / p.stem
        cache_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as zf:
            zf.extractall(cache_root)
        gribs = sorted(cache_root.glob("*.grib"))
        if not gribs:
            raise FileNotFoundError("No .grib found inside the zip.")
        print(f" Extracted: {gribs[0].name}")
        ds = _open_cfgrib_merge(gribs[0])
        print(" ERA5-Land loaded from extracted GRIB")
    else:
        print("ERA5-Land: loading GRIB directly")
        ds = _open_cfgrib_merge(p)
    return ds.load()


def load_era5_data(file_path, cache_dir="_cache"):
    p = resolve_path(Path(file_path))
    if zipfile.is_zipfile(p):
        print("ERA5: ZIP detected → extracting to cache...")
        cache_root = Path(cache_dir) / p.stem
        cache_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as zf:
            zf.extractall(cache_root)
        gribs = sorted(cache_root.glob("*.grib"))
        if not gribs:
            raise FileNotFoundError("No .grib found inside the zip.")
        print(f" Extracted: {gribs[0].name}")
        ds = _open_cfgrib_merge(gribs[0])
        print(" ERA5 loaded from extracted GRIB")
    else:
        print("ERA5: loading GRIB directly")
        ds = _open_cfgrib_merge(p)
    return ds.load()


# -------------------- Helpers: Unfold (time, step) -> 1D time ------------------
def drop_or_fix_allnan_first_hour(ds: xr.Dataset, vars_: list, fix="copy_next") -> xr.Dataset:
    """
    Hardening: detect and fix all-NaN first hour that survived unfold.

    Args:
        ds: Dataset to check
        vars_: List of critical variables to check for NaNs
        fix: 'copy_next' (copy from next hour) or 'drop' (remove first hour)
    """
    if 'time' not in ds.dims or ds.time.size < 2:
        return ds

    t0 = ds.time.values[0]
    bad = False

    # Check if first hour has all-NaN data for critical variables
    for v in vars_:
        if v not in ds:
            continue
        frac = float(ds[v].sel(time=t0).isnull().mean().values)
        if frac > 0.999:  # Essentially all NaN
            bad = True
            print(f"[SANITY] {v} is all-NaN at first hour {t0}")

    if not bad:
        return ds

    if fix == "copy_next" and ds.time.size >= 2:
        t1 = ds.time.values[1]
        print(f"[SANITY] Patching first hour by copying from {t1}")
        for v in vars_:
            if v in ds:
                ds[v].loc[dict(time=t0)] = ds[v].sel(time=t1)
        return ds
    elif fix == "drop":
        print(f"[SANITY] Dropping corrupted first hour {t0}")
        return ds.isel(time=slice(1, None))

    print(f"[SANITY] WARNING: Could not fix corrupted first hour {t0}")
    return ds


def unfold_time_to_valid_1d(ds: xr.Dataset, prefer="first_valid") -> xr.Dataset:
    """
    Convert (time, step) into a single 'time' = valid_time (= time + step).

    If duplicates at same valid_time exist, reduce using:
      - 'first_valid' (fill NaNs from later steps, then take earliest) [RECOMMENDED]
      - 'first' (smallest lead when sorted, can return NaN if step=0 is missing)
      - 'mean'/'median' for averaging approaches
    """
    if "step" not in ds.dims:
        if "time" in ds.dims:
            return ds.sortby("time")
        return ds

    st = ds.stack(ts=("time", "step"))
    vt = (st["time"] + st["step"])
    st = st.assign_coords(valid_time=("ts", vt.data))

    if prefer == "first_valid":
        # Fill earlier-step NaNs from later-step values (same valid_time),
        # then take the earliest step.
        def _pick_first_valid(g: xr.Dataset) -> xr.Dataset:
            g = g.sortby("step")  # Sort by step (0h, 1h, 3h, etc.)
            g = g.bfill("ts")     # Backfill NaNs from later steps
            return g.isel(ts=0)   # Take the earliest step (now filled)

        sort_keys = ["valid_time", "step"]
        st = st.sortby(sort_keys)
        out = st.groupby("valid_time").map(_pick_first_valid)

    elif prefer == "first":
        sort_keys = ["valid_time"]
        if "step" in st.coords:
            sort_keys.append("step")
        st = st.sortby(sort_keys)
        out = st.groupby("valid_time").first()

    else:
        gb = st.groupby("valid_time")
        out = gb.mean() if prefer == "mean" else gb.median()

    # Handle naming conflict - drop existing 'time' if it conflicts
    if "time" in out.coords and "time" != out.dims.get("valid_time"):
        out = out.drop_vars("time", errors="ignore")

    out = out.rename({"valid_time": "time"}).sortby("time")

    if "ts" in out.dims:
        out = out.drop_dims("ts")

    core = {"time", "latitude", "longitude"}
    extras = [c for c in out.coords if c not in core]
    out = out.reset_coords(extras, drop=True)
    return out


def intersect_hours(ds_a, ds_b):
    """Find exact hourly intersection of two datasets"""
    ta = pd.to_datetime(ds_a.time.values).round("1h")
    tb = pd.to_datetime(ds_b.time.values).round("1h")
    common = np.intersect1d(ta.values, tb.values)
    common_da = xr.DataArray(common, dims="time")
    return ds_a.sel(time=common), ds_b.sel(time=common), common_da


# --------- Helpers: clamp to filename month --------
def clamp_to_modal_month(ds: xr.Dataset):
    if "time" not in ds.coords or ds["time"].size == 0:
        return ds
    t = pd.to_datetime(ds["time"].values)
    ym = pd.Series([(d.year, d.month) for d in t])
    year, month = ym.mode().iat[0]
    start = pd.Timestamp(year=year, month=month, day=1, hour=0)
    end = (
        pd.Timestamp(
            year=year + (month == 12),
            month=(1 if month == 12 else month + 1),
            day=1,
            hour=0
        ) - pd.Timedelta(hours=1)
    )
    return ds.sel(time=slice(np.datetime64(start), np.datetime64(end)))


def clamp_to_filename_month(ds, fname):
    """Use filename year-month to clamp time (avoid modal ambiguity)"""
    m = re.search(r"(\d{4})[_-](\d{2})", Path(fname).stem)
    if not m or "time" not in ds:
        return clamp_to_modal_month(ds)
    y, mo = map(int, m.groups())
    start = np.datetime64(pd.Timestamp(year=y, month=mo, day=1, hour=0))
    end = np.datetime64(
        (pd.Timestamp(
            year=y + (mo == 12),
            month=(1 if mo == 12 else mo + 1),
            day=1,
            hour=0
        ) - pd.Timedelta(hours=1))
    )
    return ds.sel(time=slice(start, end))


# ------------------------ Pakistan geometry + inpoly mask -----------------------
def load_pakistan_polygon(geojson_path):
    """Load Pakistan boundary (single dissolved polygon) in EPSG:4326."""
    gdf = gpd.read_file(geojson_path)
    pak = gdf.dissolve().to_crs("EPSG:4326")
    return pak.geometry.iloc[0]


def build_inpoly_mask(ds: xr.Dataset, poly):
    """
    Return a boolean (lat, lon) mask for whether grid cell center lies inside polygon.
    Does NOT apply the mask to data (no NaNs introduced).
    """
    lon_arr = ds["longitude"].values
    lat_arr = ds["latitude"].values
    reg = regionmask.Regions([poly])

    try:
        try:
            m3 = reg.mask_3D(lon_arr, lat_arr)  # old sig
        except TypeError:
            m3 = reg.mask_3D(lon=lon_arr, lat=lat_arr)  # new sig

        if "lat" in m3.dims or "lon" in m3.dims:
            m3 = m3.rename({"lat": "latitude", "lon": "longitude"})

        inside = m3.isel(region=0).fillna(False).astype(bool)

    except Exception:
        # fallback
        try:
            m2 = reg.mask(lon_arr, lat_arr)
        except TypeError:
            m2 = reg.mask(lon=lon_arr, lat=lat_arr)

        if "lat" in m2.dims or "lon" in m2.dims:
            m2 = m2.rename({"lat": "latitude", "lon": "longitude"})

        if np.issubdtype(m2.dtype, np.floating):
            inside = m2.notnull()
        else:
            inside = (m2 >= 0)

        inside = inside.transpose("latitude", "longitude")

    return inside


# ------------------------------- Feature fns -----------------------------------
def wind_dir_from_uv(u, v):
    # meteorological direction (deg FROM which wind blows)
    return (np.degrees(np.arctan2(-u, -v)) % 360.0)


def tendency_causal(da: xr.DataArray):
    """Causal tendency calculation without future information leakage"""
    s = da.rolling(time=3, center=False, min_periods=1).mean()
    d = s.diff("time")
    dt = (da["time"].diff("time") / np.timedelta64(1, "h"))
    out = d / dt
    out = out.reindex(time=da.time)  # first time becomes NaN (expected)
    return out


def compute_thermo(ds: xr.Dataset):
    T, Td, sp = ds["t2m"], ds["d2m"], ds["sp"]  # K, K, Pa
    T_C, Td_C = T - 273.15, Td - 273.15
    e_s = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))  # hPa
    e = 6.112 * np.exp((17.67 * Td_C) / (Td_C + 243.5))  # hPa
    RH = (100.0 * (e / e_s)).clip(0, 100)
    q = (0.622 * e) / ((sp / 100.0) - 0.378 * e)  # kg/kg
    VPD = e_s - e  # hPa
    dT = T_C - Td_C  # °C
    R_d, c_p = 287.0, 1004.0
    theta = T * (1e5 / sp) ** (R_d / c_p)  # K

    return xr.Dataset(
        {
            "RH": RH.assign_attrs(long_name="Relative humidity", units="%"),
            "q": q.assign_attrs(long_name="Specific humidity", units="kg kg-1"),
            "VPD": VPD.assign_attrs(long_name="Vapor pressure deficit", units="hPa"),
            "dT": dT.assign_attrs(long_name="Dew-point depression", units="degC"),
            "theta": theta.assign_attrs(long_name="Potential temperature", units="K"),
        }
    )


def compute_wind(ds: xr.Dataset):
    u10, v10 = ds["u10"], ds["v10"]
    u100 = ds["u100"] if "u100" in ds else u10
    v100 = ds["v100"] if "v100" in ds else v10

    WS10, WD10 = np.hypot(u10, v10), wind_dir_from_uv(u10, v10)
    WS100, WD100 = np.hypot(u100, v100), wind_dir_from_uv(u100, v100)
    dWS = WS100 - WS10
    dWD = ((WD100 - WD10 + 180.0) % 360.0) - 180.0

    return xr.Dataset(
        {
            "WS10": WS10.assign_attrs(long_name="10m wind speed", units="m s-1"),
            "WD10": WD10.assign_attrs(long_name="10m wind direction (from north)", units="degrees"),
            "WS100": WS100.assign_attrs(long_name="100m wind speed", units="m s-1"),
            "WD100": WD100.assign_attrs(long_name="100m wind direction (from north)", units="degrees"),
            "dWS": dWS.assign_attrs(long_name="Vertical wind speed shear", units="m s-1"),
            "dWD": dWD.assign_attrs(long_name="Vertical wind dir shear", units="degrees"),
        }
    )


def compute_mixing(ds: xr.Dataset):
    if "blh" not in ds:
        raise ValueError("BLH is required for mixing features")
    WS10 = np.hypot(ds["u10"], ds["v10"])
    BLH = ds["blh"]  # m
    VC = WS10 * BLH
    med = VC.median("time")
    iqr = (VC.quantile(0.75, "time") - VC.quantile(0.25, "time"))
    iqr = xr.where(iqr > 0, iqr, np.nan)
    VCi = ((VC - med) / iqr).clip(-3, 3)
    BLH_tend = tendency_causal(BLH)

    return xr.Dataset(
        {
            "VC": VC.assign_attrs(long_name="Ventilation coefficient", units="m2 s-1"),
            "VCi": VCi.assign_attrs(long_name="Inverse ventilation index (robust z-like)", units="standardized"),
            "BLH_tend": BLH_tend.assign_attrs(long_name="BLH tendency (causal)", units="m h-1"),
        }
    )


def compute_clouds(ds: xr.Dataset):
    if "tcc" not in ds:
        raise ValueError("tcc is required to compute CLR")
    tcc = ds["tcc"]
    try:
        if float(tcc.max()) > 1.5:  # ERA5 tcc can be 0-100
            tcc = tcc / 100.0
    except Exception:
        pass
    CLR = 1.0 - tcc
    return xr.Dataset({"CLR": CLR.assign_attrs(long_name="Clear-sky fraction", units="0-1")})


def compute_synoptic(ds: xr.Dataset):
    out = xr.Dataset()
    if "msl" in ds:
        out["MSLP_tend"] = tendency_causal(ds["msl"]).assign_attrs(long_name="MSLP tendency (causal)", units="Pa h-1")
    if "sp" in ds:
        out["SP_tend"] = tendency_causal(ds["sp"]).assign_attrs(long_name="Surface pressure tendency (causal)", units="Pa h-1")
    return out


def compute_flags_from_thermo_mixing(thermo_ds, mixing_ds, inpoly_bool, rh_thresh=85.0, vc_thresh_percentile=40.0):
    """
    Compute flags for the FULL bbox, but compute VC threshold using ONLY in-Pakistan cells.
    Outside Pakistan => flags set to -1 (int8), so you can ignore easily later.
    """
    RH, VC = thermo_ds["RH"], mixing_ds["VC"]
    vc_in = VC.where(inpoly_bool)  # broadcast over time
    vc_thresh = float(vc_in.quantile(vc_thresh_percentile / 100.0))
    stagnant = (VC < vc_thresh)
    high_rh = (RH > rh_thresh)

    # output int8 with -1 outside Pakistan
    stagnant_i8 = xr.where(inpoly_bool, stagnant, False).astype("int8")
    highrh_i8 = xr.where(inpoly_bool, high_rh, False).astype("int8")
    stagnant_i8 = xr.where(inpoly_bool, stagnant_i8, np.int8(-1))
    highrh_i8 = xr.where(inpoly_bool, highrh_i8, np.int8(-1))

    return xr.Dataset(
        {
            "Stagnant": stagnant_i8.assign_attrs(
                long_name=f"Stagnation flag (VC<P{vc_thresh_percentile:.0f}={vc_thresh:.0f} m²/s), outside=-1",
                units="0/1/-1",
            ),
            "HighRH": highrh_i8.assign_attrs(long_name="High-RH flag (RH>85%), outside=-1", units="0/1/-1"),
        }
    )


# ================================ MAIN PIPELINE =====================================
def process_month(era5land_file, era5_file, output_dir="features_nc_bbox"):
    """Process a single month and save features as NetCDF over FULL bbox (no masking)."""
    print(f"\n🚀 PROCESSING MONTH (BBOX): {Path(era5land_file).stem}")
    print("=" * 90)

    # 0) Load Pakistan polygon (only for __inpoly; does NOT mask data)
    pak_poly = load_pakistan_polygon(pakistan_geojson)

    # 1) Load FULL bbox datasets (no masking)
    print("Loading ERA5-Land (full bbox from GRIB)...")
    era5land_raw = load_era5land_data(era5land_file)

    print("Loading ERA5 (full bbox from GRIB)...")
    era5_raw = load_era5_data(era5_file)

    # 2) Unfold time and align by exact hourly intersection
    print("Unfolding (time, step) → valid time (ROBUST) ...")
    era5land_1d = unfold_time_to_valid_1d(era5land_raw, prefer="first_valid")
    era5_1d = unfold_time_to_valid_1d(era5_raw, prefer="first_valid")

    print("Aligning times by exact hourly intersection...")
    era5land_1d, era5_1d, common_times = intersect_hours(era5land_1d, era5_1d)
    print(f" Common times: {len(common_times)} hours")

    # 3) Collect raw + regrid ERA5 to ERA5-Land grid (spatial only)
    era5land_vars = ["t2m", "d2m", "u10", "v10", "sp"]
    era5_vars = ["u100", "v100", "blh", "msl", "tcc"]

    avail_land = [v for v in era5land_vars if v in era5land_1d]
    avail_e5 = [v for v in era5_vars if v in era5_1d]

    if not avail_land:
        raise ValueError("No ERA5-Land core vars found (expected: t2m,d2m,u10,v10,sp).")

    raw_feats = {v: era5land_1d[v] for v in avail_land}

    print("Regridding ERA5 vars to ERA5-Land grid (linear, spatial only)...")
    for v in avail_e5:
        print(f" {v} ...")
        raw_feats[v] = era5_1d[v].interp(
            latitude=era5land_1d.latitude,
            longitude=era5land_1d.longitude,
            method="linear"
        )

    raw_ds = xr.Dataset(raw_feats)
    raw_ds = clamp_to_filename_month(raw_ds, era5land_file)

    # HARDENING: Fix any ghost timestamps that survived unfold AND month clamping
    raw_ds = drop_or_fix_allnan_first_hour(raw_ds, ['blh', 'msl', 'tcc', 'u100', 'v100'])

    print(f"✓ Combined raw dataset: {len(raw_ds.data_vars)} vars | Dims: {dict(raw_ds.dims)}")
    print(
        f" lat: {float(raw_ds.latitude.min()):.3f} … {float(raw_ds.latitude.max()):.3f} "
        f"lon: {float(raw_ds.longitude.min()):.3f} … {float(raw_ds.longitude.max()):.3f}"
    )

    # 4) Build __inpoly (NO masking)
    inpoly = build_inpoly_mask(raw_ds, pak_poly)  # (lat, lon) bool
    inpoly_ds = xr.Dataset(
        {
            "__inpoly": inpoly.astype("int8").assign_attrs(
                long_name="Inside Pakistan boundary (grid-cell center)",
                units="0/1"
            )
        }
    )

    # 5) Derived features
    print("Computing derived features...")
    thermo = compute_thermo(raw_ds)
    if "VPD" in thermo:
        thermo["VPD"] = thermo["VPD"].clip(min=0)

    wind = compute_wind(raw_ds)
    mixing = compute_mixing(raw_ds)
    clouds = compute_clouds(raw_ds)
    syn = compute_synoptic(raw_ds)

    # 6) Regime flags: thresholds computed over Pakistan only, outputs full bbox
    print("Computing regime flags (thresholds from Pakistan only)...")
    flags = compute_flags_from_thermo_mixing(
        thermo,
        mixing,
        inpoly_bool=inpoly,
        rh_thresh=85.0,
        vc_thresh_percentile=40.0
    )

    # 7) Merge all (everything already on same grid/time)
    final_ds = xr.merge([raw_ds, thermo, wind, mixing, clouds, syn, inpoly_ds, flags], compat="override", join="inner")

    if "CLR" in final_ds:
        final_ds["CLR"] = final_ds["CLR"].clip(0, 1)

    print(f"Final dataset: {len(final_ds.data_vars)} features | Dims: {dict(final_ds.dims)}")

    # 8) Save NetCDF in INFERENCE format
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract year/month from filename/path for inference naming convention
    year_month_match = re.search(r"(\d{4})_(\d{2})", str(era5land_file))
    if year_month_match:
        year, month = year_month_match.groups()
        # Save in inference format: features_YYYY_MM.nc
        output_file = output_dir / f"features_{year}_{month}.nc"
    else:
        stem = Path(era5land_file).stem
        output_file = output_dir / f"features_{stem}.nc"

    final_ds.attrs.update(
        {
            "title": "PM2.5 Prediction Features (FULL BBOX)",
            "description": "Feature-engineered ERA5-Land + ERA5 drivers over the full requested bbox; includes __inpoly mask as a variable (no NaN masking).",
            "source_era5land": str(era5land_file),
            "source_era5": str(era5_file),
            "processing_date": datetime.now().isoformat(),
            "temporal_resolution": "1 hour",
            "spatial_resolution": "ERA5-Land native (~0.1 degree)",
            "feature_count": int(len(final_ds.data_vars)),
            "grid_cells": int(final_ds.sizes["latitude"] * final_ds.sizes["longitude"]),
            "time_steps": int(final_ds.sizes["time"]),
        }
    )

    # Optional: tag feature types
    for v in raw_ds.data_vars:
        if v in final_ds:
            final_ds[v].attrs["feature_type"] = "raw"

    for tag, grp in [
        ("thermodynamic", thermo),
        ("wind", wind),
        ("mixing", mixing),
        ("cloud", clouds),
        ("synoptic", syn),
        ("regime", flags),
    ]:
        for v in grp.data_vars:
            if v in final_ds:
                final_ds[v].attrs["feature_type"] = tag

    if "__inpoly" in final_ds:
        final_ds["__inpoly"].attrs["feature_type"] = "mask"

    # Compression / dtype encoding
    encoding = {}
    for var in final_ds.data_vars:
        if var == "__inpoly":
            encoding[var] = {"dtype": "int8", "zlib": True, "complevel": 4}
            continue
        if final_ds[var].dtype == np.float64:
            encoding[var] = {"dtype": "float32", "zlib": True, "complevel": 4}
        elif final_ds[var].dtype in [np.int64, np.int32]:
            encoding[var] = {"dtype": "int16", "zlib": True, "complevel": 4}
        else:
            encoding[var] = {"zlib": True, "complevel": 4}

    print(f"💾 Saving to: {output_file}")
    final_ds.to_netcdf(output_file, encoding=encoding)
    print(f"✅ Saved NetCDF: {output_file}")

    return final_ds, output_file


# ------------------------- Batch driver -------------------------
def month_range(start="2020-01", end="2025-10"):
    ts = pd.period_range(start=start, end=end, freq="M")
    for p in ts:
        yield p.year, p.month


def build_paths(y, m):
    # Use the copied data directories
    land_file = f"Pak_Era5_land_18-25/era5land_singlelevels_18_25/era5land_pk_{y}_{m:02d}.grib"
    era5_file = f"Pak_Era5_18-25/era5_singlelevels_pk_2018_2025_grib/era5_pk_{y}_{m:02d}.grib"
    return land_file, era5_file


def run_batch(start="2020-01", end="2025-10", output_dir="../../inference/datasets/features_met", overwrite=False):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    successes, failures, skipped = [], [], []
    for y, m in month_range(start, end):
        out_nc = out_dir / f"features_{y}_{m:02d}.nc"
        if out_nc.exists() and not overwrite:
            print(f"⏭️ {y}-{m:02d}: exists → {out_nc.name} (skip)")
            skipped.append((y, m, str(out_nc)))
            continue

        era5land_file, era5_file = build_paths(y, m)

        try:
            ds, out_path = process_month(era5land_file, era5_file, output_dir=output_dir)
            print(f"✅ {y}-{m:02d}: saved → {out_path}")
            successes.append((y, m, str(out_path)))

        except FileNotFoundError as e:
            print(f"🚫 {y}-{m:02d}: missing input → {e}")
            failures.append((y, m, "missing input"))

        except Exception as e:
            print(f"❌ {y}-{m:02d}: error → {e}")
            failures.append((y, m, repr(e)))

    print("\n=== BATCH SUMMARY ===")
    print(f" OK: {len(successes)} | Skipped(existing): {len(skipped)} | Failed: {len(failures)}")
    return successes, skipped, failures


def main():
    # Generate data from 2020-2025 in inference format
    successes, skipped, failures = run_batch(
        "2020-01",
        "2025-09",
        output_dir="../../inference/datasets/features_met",
        overwrite=True
    )

    print("\n=== DONE ===")
    print(f"OK={len(successes)} Skipped(existing)={len(skipped)} Failed={len(failures)}")

    if successes:
        print(f"\n✅ Generated {len(successes)} meteorological files in inference format:")
        for y, m, path in successes[:5]:  # Show first 5
            print(f" {y}-{m:02d}: {path}")
        if len(successes) > 5:
            print(f" ... and {len(successes) - 5} more")


if __name__ == "__main__":
    main()
