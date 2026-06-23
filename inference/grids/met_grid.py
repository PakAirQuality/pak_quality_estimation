# met_grid.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

DENY_NAMES = {"region", "abbrevs", "names", "__inpoly"}
SNAP_REF_VARS = ["sp", "t2m", "u10", "v10", "d2m"]

DEFAULT_DAILY_AGG_VARS = [
    "WS10", "blh", "RH", "t2m", "VPD", "VC",
    "tp", "precip", "precipitation", "pr",
]


def _numeric_feature_vars(ds: xr.Dataset) -> List[str]:
    keep: List[str] = []
    for v in ds.data_vars:
        if v in DENY_NAMES:
            continue
        if np.issubdtype(ds[v].dtype, np.number):
            keep.append(v)
    return keep


def _strip_coords(ds: xr.Dataset) -> xr.Dataset:
    keep = set(ds.dims)
    drop = [c for c in ds.coords if c not in keep]
    return ds.drop_vars(drop, errors="ignore")


def _infer_is_hourly_time(ds: xr.Dataset) -> bool:
    if "time" not in ds.dims or ds.sizes.get("time", 0) < 2:
        return False
    t = pd.to_datetime(ds["time"].values)
    dt = np.median(np.diff(t.values).astype("timedelta64[h]").astype(int))
    return dt <= 6


def _to_daily_midnight_robust(ds: xr.Dataset) -> xr.Dataset:
    """
    Interpolate to daily midnight snapshots, but avoid NaNs at the edges
    by filling times outside the available range with nearest values.
    """
    if "time" not in ds.dims:
        return ds
        
    time_vals = pd.to_datetime(ds["time"].values)
    tmin = pd.Timestamp(time_vals.min())
    tmax = pd.Timestamp(time_vals.max())

    start = tmin.floor("D")
    end = tmax.floor("D")
    daily_times = pd.date_range(start=start, end=end, freq="D")

    # Linear interpolation (what you want for training alignment)
    ds_lin = ds.interp(time=daily_times)

    # Nearest interpolation (safe fallback for out-of-range)
    ds_near = ds.interp(time=daily_times, method="nearest")

    outside = (daily_times < tmin) | (daily_times > tmax)
    outside_da = xr.DataArray(outside, coords={"time": daily_times}, dims=("time",))

    # Replace only the out-of-range days (typically the very first / last)
    ds_fixed = ds_lin.where(~outside_da, ds_near)
    
    # Diagnostic logging
    if len(daily_times) > 0:
        sample_time = daily_times[0]
        n_outside = outside.sum()
        print(f"[met] TRAINING ALIGNMENT: Using daily snapshots at midnight (e.g., {sample_time})")
        print(f"[met] This matches training snapshot interpolation (not daily means)")
        if n_outside > 0:
            print(f"[met] EDGE FIX: Filled {n_outside} out-of-range timestamps with nearest values")
            print(f"[met] Time coverage: {tmin} to {tmax}")

    return ds_fixed


def _to_daily(ds: xr.Dataset) -> xr.Dataset:
    """
    CRITICAL FIX: Convert to daily snapshots (not daily means) to match training.
    Training uses snapshot interpolation at midnight, not daily aggregation.
    """
    if "time" not in ds.dims:
        return ds
    
    if _infer_is_hourly_time(ds):
        return _to_daily_midnight_robust(ds)
    else:
        # Already daily or irregular - use existing logic
        t = pd.to_datetime(ds["time"].values).floor("D")
        ds2 = ds.assign_coords(time=("time", t))
        return ds2.groupby("time").mean(skipna=True)


def _build_valid_mask_2d(ds: xr.Dataset, ref_vars: List[str]) -> xr.DataArray:
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


def _snap_points_to_valid(tree, vlat, vlon, lon_scale, lat, lon):
    q = np.column_stack([lat.astype("float64"), lon.astype("float64") * lon_scale])
    _, idx = tree.query(q, k=1)
    return vlat[idx], vlon[idx]


@dataclass
class MetGridConfig:
    features_met_dir: Path
    grid_lats_1d: np.ndarray
    grid_lons_1d: np.ndarray


class MetGrid:
    def __init__(self, cfg: MetGridConfig):
        self.cfg = cfg

    def _met_file_for_year_month(self, ym: pd.Period) -> Path:
        return self.cfg.features_met_dir / f"features_{ym.year}_{ym.month:02d}.nc"

    def _open_met_range(self, t0: pd.Timestamp, t1: pd.Timestamp) -> xr.Dataset:
        # Add buffer to avoid edge NaNs in interpolation
        t0_buffered = pd.Timestamp(t0) - pd.Timedelta(hours=12)
        t1_buffered = pd.Timestamp(t1) + pd.Timedelta(hours=12)
        
        months = pd.period_range(t0_buffered.to_period("M"), t1_buffered.to_period("M"), freq="M")
        parts = []
        missing = []
        for ym in months:
            p = self._met_file_for_year_month(ym)
            if not p.exists():
                missing.append(p.name)
                continue
            ds = xr.open_dataset(p, drop_variables=["step"])
            ds = _strip_coords(ds)
            if "time" in ds.dims:
                ds = ds.sel(time=slice(t0_buffered, t1_buffered))
            parts.append(ds)

        if not parts:
            raise FileNotFoundError(
                f"No met feature files found for range {t0.date()}..{t1.date()}.\n"
                f"Missing examples: {missing[:5]}"
            )

        if len(parts) == 1:
            combined = parts[0]
        else:
            combined = xr.concat(parts, dim="time", coords="minimal", data_vars="minimal", compat="override")
            
        # Diagnostic: check time coverage vs requested range
        if "time" in combined.dims and combined.sizes.get("time", 0) > 0:
            time_vals = pd.to_datetime(combined["time"].values)
            actual_min = time_vals.min()
            actual_max = time_vals.max()
            print(f"[met] Time coverage: {actual_min} to {actual_max}")
            print(f"[met] Requested range: {t0} to {t1}")
            
            # Check if prediction day midnight is covered
            pred_midnight_covered = (t0 >= actual_min) and (t0 <= actual_max)
            if not pred_midnight_covered:
                print(f"[met] WARNING: Prediction day {t0} not fully covered by available data")
            
        return combined

    def _ensure_core_met_vars(self, ds_daily: xr.Dataset) -> xr.Dataset:
        ds = ds_daily

        if "WS10" not in ds.data_vars and ("u10" in ds.data_vars and "v10" in ds.data_vars):
            ds["WS10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

        if "WD10" not in ds.data_vars and ("u10" in ds.data_vars and "v10" in ds.data_vars):
            ds["WD10"] = (np.degrees(np.arctan2(-ds["u10"], -ds["v10"])) % 360)

        if "WS100" not in ds.data_vars and ("u100" in ds.data_vars and "v100" in ds.data_vars):
            ds["WS100"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)

        if "WD100" not in ds.data_vars and ("u100" in ds.data_vars and "v100" in ds.data_vars):
            ds["WD100"] = (np.degrees(np.arctan2(-ds["u100"], -ds["v100"])) % 360)

        if "CLR" not in ds.data_vars and "tcc" in ds.data_vars:
            ds["CLR"] = 1.0 - ds["tcc"]

        def _add_tend(name, base):
            if name not in ds.data_vars and base in ds.data_vars:
                ds[name] = ds[base].diff("time", label="upper")
                ds[name] = ds[name].reindex(time=ds["time"])

        _add_tend("BLH_tend", "blh")
        _add_tend("MSLP_tend", "msl")
        _add_tend("SP_tend", "sp")
        _add_tend("dT", "t2m")

        if "dWS" not in ds.data_vars and ("WS10" in ds.data_vars and "WS100" in ds.data_vars):
            ds["dWS"] = ds["WS100"] - ds["WS10"]
        if "dWD" not in ds.data_vars and ("WD10" in ds.data_vars and "WD100" in ds.data_vars):
            ds["dWD"] = ds["WD100"] - ds["WD10"]

        if "Stagnant" not in ds.data_vars and ("WS10" in ds.data_vars and "blh" in ds.data_vars):
            ds["Stagnant"] = ((ds["WS10"] < 2.0) & (ds["blh"] < 300.0)).astype("int8")
        if "HighRH" not in ds.data_vars and ("RH" in ds.data_vars):
            ds["HighRH"] = (ds["RH"] > 80.0).astype("int8")

        return ds

    def _snap_daily_to_valid_land(self, ds_daily: xr.Dataset) -> xr.Dataset:
        ds = ds_daily
        present = [v for v in SNAP_REF_VARS if v in ds.data_vars]
        if not present:
            return ds

        valid_mask_2d = _build_valid_mask_2d(ds, ref_vars=SNAP_REF_VARS)
        tree, vlat, vlon, lon_scale = _build_kdtree_for_valid_cells(ds, valid_mask_2d)

        lats = ds["latitude"].values
        lons = ds["longitude"].values
        lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")

        base0 = ds[present].isel(time=0) if ("time" in ds.dims and ds.sizes.get("time", 0) > 0) else ds[present]

        bad = None
        for v in present:
            m = ~np.isfinite(base0[v].values)
            bad = m if bad is None else (bad | m)
        bad = bad.astype(bool)

        snapped_flag2d = bad.astype("float32")
        if not bad.any():
            ds["snapped_flag"] = xr.DataArray(snapped_flag2d, dims=("latitude", "longitude"))
            return ds

        bad_lat = lat2d[bad].astype("float64")
        bad_lon = lon2d[bad].astype("float64")
        s_lat, s_lon = _snap_points_to_valid(tree, vlat, vlon, lon_scale, bad_lat, bad_lon)

        snapped_lat2d = lat2d.copy()
        snapped_lon2d = lon2d.copy()
        snapped_lat2d[bad] = s_lat
        snapped_lon2d[bad] = s_lon

        lat_da = xr.DataArray(snapped_lat2d, dims=("latitude", "longitude"))
        lon_da = xr.DataArray(snapped_lon2d, dims=("latitude", "longitude"))

        ds2 = ds.interp(latitude=lat_da, longitude=lon_da, method="nearest")

        # restore 1D coordinates
        lats_1d = ds["latitude"].values
        lons_1d = ds["longitude"].values
        if "latitude" in ds2.coords and ds2["latitude"].ndim != 1:
            ds2 = ds2.drop_vars("latitude", errors="ignore")
        if "longitude" in ds2.coords and ds2["longitude"].ndim != 1:
            ds2 = ds2.drop_vars("longitude", errors="ignore")

        ds2 = ds2.assign_coords(latitude=("latitude", lats_1d), longitude=("longitude", lons_1d))
        ds2["snapped_flag"] = xr.DataArray(snapped_flag2d, dims=("latitude", "longitude"))
        return ds2

    def load_met_daily_history(self, pred_day: pd.Timestamp, history_days: int) -> xr.Dataset:
        pred_day = pd.Timestamp(pred_day).floor("D")
        t0 = pred_day - pd.Timedelta(days=history_days)
        t1 = pred_day

        ds_raw = self._open_met_range(t0, t1 + pd.Timedelta(days=1))
        is_hourly = _infer_is_hourly_time(ds_raw)
        ds_daily = _to_daily(ds_raw)

        agg_vars = [v for v in DEFAULT_DAILY_AGG_VARS if v in ds_raw.data_vars or v in ds_daily.data_vars]
        if agg_vars:
            if is_hourly and "time" in ds_raw.dims:
                base = ds_raw[agg_vars]
                ds_mean = base.resample(time="1D").mean(skipna=True)
                ds_max = base.resample(time="1D").max(skipna=True)
                ds_min = base.resample(time="1D").min(skipna=True)
            else:
                base = ds_daily[agg_vars]
                ds_mean, ds_max, ds_min = base, base, base

            ds_mean = ds_mean.rename({v: f"{v}_daily_mean" for v in agg_vars})
            ds_max = ds_max.rename({v: f"{v}_daily_max" for v in agg_vars})
            ds_min = ds_min.rename({v: f"{v}_daily_min" for v in agg_vars})
            ds_daily = xr.merge([ds_daily, ds_mean, ds_max, ds_min], compat="override")

        ds_daily = ds_daily.sel(time=slice(t0, t1))
        ds_daily = self._ensure_core_met_vars(ds_daily)
        ds_daily = self._snap_daily_to_valid_land(ds_daily)

        ds_daily = ds_daily.sel(
            latitude=xr.DataArray(self.cfg.grid_lats_1d, dims="latitude"),
            longitude=xr.DataArray(self.cfg.grid_lons_1d, dims="longitude"),
            method="nearest",
        )
        return ds_daily

    def met_level2_for_day(self, ds_hist: xr.Dataset, pred_day: pd.Timestamp) -> Dict[str, np.ndarray]:
        pred_day = pd.Timestamp(pred_day).floor("D")
        ds_hist = ds_hist.sortby("time")

        doy = int(pred_day.dayofyear)
        month = int(pred_day.month)

        nlat = len(self.cfg.grid_lats_1d)
        nlon = len(self.cfg.grid_lons_1d)
        n = nlat * nlon

        out: Dict[str, np.ndarray] = {}
        day = ds_hist.sel(time=pred_day, method="nearest")

        if "__inpoly" in ds_hist.data_vars:
            inpoly = ds_hist["__inpoly"]
            inpoly_day = inpoly.isel(time=0) if "time" in inpoly.dims else inpoly
            out["__inpoly"] = inpoly_day.values.astype("float32").ravel()

        # Handle tendency variables specially - they need last valid value, not exact prediction day
        tendency_vars = ["SP_tend", "MSLP_tend", "BLH_tend", "dT"]
        
        for v in _numeric_feature_vars(day):
            if v in tendency_vars and v in ds_hist.data_vars:
                # For tendency vars: get last valid (non-NaN) value from history
                da = ds_hist[v]
                # Use forward fill to get last valid value at prediction day
                da_filled = da.ffill("time")
                out[v] = da_filled.sel(time=pred_day, method="nearest").values.astype("float32").ravel()
            else:
                # Standard extraction for non-tendency variables
                out[v] = day[v].values.astype("float32").ravel()

        out["doy_sin"] = np.full(n, np.sin(2 * np.pi * doy / 366.0), dtype="float32")
        out["doy_cos"] = np.full(n, np.cos(2 * np.pi * doy / 366.0), dtype="float32")
        out["doy_sin_2"] = np.full(n, np.sin(4 * np.pi * doy / 366.0), dtype="float32")
        out["doy_cos_2"] = np.full(n, np.cos(4 * np.pi * doy / 366.0), dtype="float32")
        out["doy_sin_3"] = np.full(n, np.sin(6 * np.pi * doy / 366.0), dtype="float32")
        out["doy_cos_3"] = np.full(n, np.cos(6 * np.pi * doy / 366.0), dtype="float32")

        out["heating_season_flag"] = np.full(n, int(month in [11, 12, 1, 2]), dtype="float32")
        out["burning_season_flag"] = np.full(n, int(month in [10, 11]), dtype="float32")

        # Ensure these exist (your training expects them)
        out.setdefault("snapped_flag", np.zeros(n, dtype="float32"))
        out.setdefault("agg_snapped_flag", np.zeros(n, dtype="float32"))
        out.setdefault("agg_snap_dist_km", np.zeros(n, dtype="float32"))
        out.setdefault("dt_days", np.zeros(n, dtype="float32"))

        def _last(da: xr.DataArray) -> np.ndarray:
            # For tendency variables, use forward fill to get last valid value
            if da.name in tendency_vars:
                da_filled = da.ffill("time")
                return da_filled.sel(time=pred_day, method="nearest").values.astype("float32").ravel()
            else:
                return da.sel(time=pred_day, method="nearest").values.astype("float32").ravel()

        if "WS10" in ds_hist.data_vars:
            WS10 = ds_hist["WS10"]
            out["WS10_lag1d"] = _last(WS10.shift(time=1))
            out["WS10_lag3d"] = _last(WS10.shift(time=3))
            # TRAINING ALIGNMENT: Use ceil(0.5*window) for min_periods like training
            out["WS10_rollmean_3d"] = _last(WS10.rolling(time=3, min_periods=2).mean())
            out["WS10_rollstd_3d"] = _last(WS10.rolling(time=3, min_periods=2).std(ddof=1))
            out["WS10_rollmin_7d"] = _last(WS10.rolling(time=7, min_periods=4).min())
            out["WS10_rollmax_7d"] = _last(WS10.rolling(time=7, min_periods=4).max())

            inst_calm = (WS10 < 2.0).astype("int8")
            calm3 = inst_calm.rolling(time=3, min_periods=2).sum()
            calm7 = inst_calm.rolling(time=7, min_periods=4).sum()
            out["calm3_count"] = _last(calm3)
            out["calm3_flag"] = (_last(calm3) >= 2).astype("float32")
            out["calm7_count"] = _last(calm7)
            out["calm7_flag"] = (_last(calm7) >= 5).astype("float32")

        if "blh" in ds_hist.data_vars:
            blh = ds_hist["blh"]
            out["blh_lag1d"] = _last(blh.shift(time=1))
            out["blh_lag3d"] = _last(blh.shift(time=3))
            out["blh_rollmean_3d"] = _last(blh.rolling(time=3, min_periods=2).mean())
            out["blh_rollmin_7d"] = _last(blh.rolling(time=7, min_periods=4).min())

            rm7 = blh.rolling(time=7, min_periods=4).mean()
            rm14 = blh.rolling(time=14, min_periods=7).mean()
            out["blh_rollmean_7d"] = _last(rm7)
            out["blh_anom_7d"] = (_last(blh) - _last(rm7)).astype("float32")
            out["blh_rollmean_14d"] = _last(rm14)
            out["blh_anom_14d"] = (_last(blh) - _last(rm14)).astype("float32")

        if "RH" in ds_hist.data_vars:
            RH = ds_hist["RH"]
            out["RH_rollmean_3d"] = _last(RH.rolling(time=3, min_periods=2).mean())
            out["RH_rollmax_7d"] = _last(RH.rolling(time=7, min_periods=4).max())
            rm7 = RH.rolling(time=7, min_periods=4).mean()
            out["RH_rollmean_7d"] = _last(rm7)
            out["RH_anom_7d"] = (_last(RH) - _last(rm7)).astype("float32")
            out["RH_rollstd_7d"] = _last(RH.rolling(time=7, min_periods=4).std(ddof=1))

        if "VPD" in ds_hist.data_vars:
            VPD = ds_hist["VPD"]
            out["VPD_rollmean_3d"] = _last(VPD.rolling(time=3, min_periods=2).mean())
            out["VPD_rollmean_7d"] = _last(VPD.rolling(time=7, min_periods=4).mean())

        if "VC" in ds_hist.data_vars:
            VC = ds_hist["VC"]
            out["VC_rollmean_3d"] = _last(VC.rolling(time=3, min_periods=2).mean())
            out["VC_rollmin_7d"] = _last(VC.rolling(time=7, min_periods=4).min())
            rm7 = VC.rolling(time=7, min_periods=4).mean()
            rm14 = VC.rolling(time=14, min_periods=7).mean()
            out["VC_rollmean_7d"] = _last(rm7)
            out["VC_anom_7d"] = (_last(VC) - _last(rm7)).astype("float32")
            out["VC_rollmean_14d"] = _last(rm14)
            out["VC_anom_14d"] = (_last(VC) - _last(rm14)).astype("float32")

        if ("WS10" in ds_hist.data_vars) and ("blh" in ds_hist.data_vars):
            inst_stag = ((ds_hist["WS10"] < 2.0) & (ds_hist["blh"] < 300.0)).astype("int8")
            s3 = inst_stag.rolling(time=3, min_periods=2).sum()
            s7 = inst_stag.rolling(time=7, min_periods=4).sum()
            out["stagnant3_count"] = _last(s3)
            out["stagnant3_flag"] = (_last(s3) >= 2).astype("float32")
            out["stagnant7_count"] = _last(s7)
            out["stagnant7_flag"] = (_last(s7) >= 5).astype("float32")

        if "WD10" in ds_hist.data_vars:
            WD10 = ds_hist["WD10"]
            WD10_sin = np.sin(np.deg2rad(WD10))
            WD10_cos = np.cos(np.deg2rad(WD10))
            out["WD10_sin"] = _last(WD10_sin)
            out["WD10_cos"] = _last(WD10_cos)

            sin7 = WD10_sin.rolling(time=7, min_periods=4).mean()
            cos7 = WD10_cos.rolling(time=7, min_periods=4).mean()
            R7 = np.sqrt(sin7**2 + cos7**2)
            var7 = (1.0 - R7).clip(0, 1)

            sin14 = WD10_sin.rolling(time=14, min_periods=7).mean()
            cos14 = WD10_cos.rolling(time=14, min_periods=7).mean()
            R14 = np.sqrt(sin14**2 + cos14**2)
            var14 = (1.0 - R14).clip(0, 1)

            out["WD10_sin_rm_7d"] = _last(sin7)
            out["WD10_cos_rm_7d"] = _last(cos7)
            out["WD10_var_7d"] = _last(var7)

            out["WD10_sin_rm_14d"] = _last(sin14)
            out["WD10_cos_rm_14d"] = _last(cos14)
            out["WD10_var_14d"] = _last(var14)

        if "dWS" in ds_hist.data_vars:
            dWS = ds_hist["dWS"]
            out["dWS_abs"] = _last(np.abs(dWS))
            out["dWS_rollmean_3d"] = _last(dWS.rolling(time=3, min_periods=2).mean())
            out["dWS_rollstd_3d"] = _last(dWS.rolling(time=3, min_periods=2).std(ddof=1))

        if "dWD" in ds_hist.data_vars:
            dWD = ds_hist["dWD"]
            out["dWD_abs"] = _last(np.abs(dWD))
            out["dWD_rollstd_3d"] = _last(dWD.rolling(time=3, min_periods=2).std(ddof=1))

        for col in ["BLH_tend", "MSLP_tend", "SP_tend", "dT"]:
            if col in ds_hist.data_vars:
                da = ds_hist[col]
                out[f"{col}_rollmean_3d"] = _last(da.rolling(time=3, min_periods=2).mean())
                out[f"{col}_rollstd_3d"] = _last(da.rolling(time=3, min_periods=2).std(ddof=1))
                out[f"{col}_rollmean_7d"] = _last(da.rolling(time=7, min_periods=4).mean())
                out[f"{col}_rollstd_7d"] = _last(da.rolling(time=7, min_periods=4).std(ddof=1))

        return {k: np.asarray(v, dtype="float32") for k, v in out.items()}

    def compute(self, pred_date: datetime, history_days: int) -> Dict[str, np.ndarray]:
        pred_day = pd.Timestamp(pred_date).floor("D")
        ds_hist = self.load_met_daily_history(pred_day, history_days=history_days)
        return self.met_level2_for_day(ds_hist, pred_day)