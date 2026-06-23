
from __future__ import annotations
import argparse
import hashlib
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from feature_registry import register_stage_features

warnings.filterwarnings("ignore")


def _create_deterministic_sensor_id(lat: float, lon: float) -> int:
    """
    Create stable, deterministic sensor_id using same logic as met_features.py.

    Uses SHA1 hash of ONLY rounded coordinates for maximum stability.
    This MUST match the implementation in met_features.py exactly.
    """
    lat_rounded = round(float(lat), 5)
    lon_rounded = round(float(lon), 5)
    id_string = f"{lat_rounded:.5f},{lon_rounded:.5f}"
    h = hashlib.sha1(id_string.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


# -----------------------------
# Column resolver helpers
# -----------------------------
def _resolve_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}. Available: {list(df.columns)}")
    return None


# -----------------------------
# Metadata helpers
# -----------------------------
def _get_tag_float(tags: Dict[str, str], keys, default=None) -> Optional[float]:
    for k in keys:
        if k in tags:
            try:
                return float(tags[k])
            except Exception:
                pass
    return default


def _get_fill_value(dataset: rasterio.DatasetReader, band: int) -> Optional[float]:
    if dataset.nodata is not None:
        return float(dataset.nodata)
    tags = dataset.tags(band)
    fv = _get_tag_float(tags, keys=["_FillValue", "FILL_VALUE", "fill_value", "nodata", "NODATA"], default=None)
    return fv


def _apply_scale_offset(arr: np.ndarray, tags: Dict[str, str]) -> Tuple[np.ndarray, float, float]:
    scale = _get_tag_float(tags, keys=["scale_factor", "SCALE_FACTOR", "ScaleFactor"], default=1.0)
    offset = _get_tag_float(tags, keys=["add_offset", "ADD_OFFSET", "AddOffset"], default=0.0)
    out = arr.astype(np.float64) * scale + offset
    return out, float(scale), float(offset)


# -----------------------------
# CRS helpers
# -----------------------------
def _lonlat_to_dataset_xy(dataset: rasterio.DatasetReader, lon: float, lat: float) -> Tuple[float, float]:
    if dataset.crs is None:
        return lon, lat

    epsg = dataset.crs.to_epsg() if dataset.crs else None
    if epsg == 4326:
        return lon, lat

    if Transformer is None:
        raise ValueError(f"Dataset CRS is {dataset.crs} but pyproj not installed; cannot transform from EPSG:4326.")

    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def _estimate_pixels_per_km(dataset: rasterio.DatasetReader, lat: float) -> float:
    epsg = dataset.crs.to_epsg() if dataset.crs else 4326
    res_x, res_y = abs(dataset.res[0]), abs(dataset.res[1])

    if epsg == 4326:
        lat_factor = np.cos(np.radians(lat))
        km_per_deg_x = 111.32 * max(lat_factor, 1e-6)
        km_per_deg_y = 110.54
        pix_per_km_x = 1.0 / (res_x * km_per_deg_x) if res_x > 0 else 1.0
        pix_per_km_y = 1.0 / (res_y * km_per_deg_y) if res_y > 0 else 1.0
        return float((pix_per_km_x + pix_per_km_y) / 2.0)

    # projected meters
    meters_per_pixel = (res_x + res_y) / 2.0
    return float(1000.0 / meters_per_pixel) if meters_per_pixel > 0 else 1.0


# -----------------------------
# QA-band auto-detection
# -----------------------------
def _detect_qa_band(ds: rasterio.DatasetReader, requested_qa_band: int) -> int:
    """
    Returns 0 if QA should be disabled, else a 1-indexed band.
    Preference:
      1) requested_qa_band if exists
      2) band whose tags/description suggest QA
      3) band 2 if exists
    """
    if requested_qa_band and requested_qa_band > 0 and ds.count >= requested_qa_band:
        return int(requested_qa_band)

    # tags / descriptions heuristic
    descs = list(ds.descriptions) if ds.descriptions else []
    for b in range(1, ds.count + 1):
        tags = ds.tags(b)
        keyblob = " ".join([str(k).lower() for k in tags.keys()])
        valblob = " ".join([str(v).lower() for v in tags.values()])
        d = (descs[b - 1].lower() if (b - 1) < len(descs) and descs[b - 1] else "")
        blob = " ".join([keyblob, valblob, d])
        if "qa" in blob or "quality" in blob or "qa_value" in blob:
            return int(b)

    if ds.count >= 2:
        return 2
    return 0


# -----------------------------
# File discovery
# -----------------------------
def find_product_files(product_dir: Path) -> Dict[datetime.date, Dict[str, Any]]:
    """
    Find *.tif and map by date token in filename (YYYY-MM-DD anywhere).
    """
    file_map: Dict[datetime.date, Dict[str, Any]] = {}
    if not product_dir.exists():
        return file_map

    n = 0
    for tif in sorted(product_dir.glob("*.tif")):
        mo = DATE_RE.search(tif.name)
        if not mo:
            continue
        y, m, d = map(int, mo.groups())
        dt = datetime(y, m, d).date()
        if dt not in file_map:
            file_map[dt] = {"path": tif, "date": dt, "year": y, "month": m, "day": d}
            n += 1

    print(f"  Found {n:,} TIFFs across {len(file_map)} days in {product_dir}")
    return file_map


# -----------------------------
# Stats helpers (feature engineering)
# -----------------------------
def _nan_skew(x: np.ndarray) -> float:
    if x.size < 3:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x))
    if s <= 1e-12:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z ** 3))


def _nan_kurtosis_excess(x: np.ndarray) -> float:
    if x.size < 4:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x))
    if s <= 1e-12:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z ** 4) - 3.0)


def _summarize_valid(valid: np.ndarray) -> Dict[str, float]:
    """
    Assumes valid is 1D and has no NaNs.
    """
    if valid.size == 0:
        return {
            "median": np.nan, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "p10": np.nan, "p25": np.nan, "p75": np.nan, "p90": np.nan,
            "iqr": np.nan, "mad": np.nan, "range": np.nan, "cv": np.nan,
            "skew": np.nan, "kurt": np.nan,
        }

    q10, q25, q75, q90 = np.percentile(valid, [10, 25, 75, 90])
    med = float(np.median(valid))
    mean = float(np.mean(valid))
    std = float(np.std(valid)) if valid.size > 1 else 0.0
    mad = float(np.median(np.abs(valid - med))) if valid.size > 0 else np.nan
    rng = float(np.max(valid) - np.min(valid)) if valid.size > 0 else np.nan
    cv = float(std / (abs(mean) + 1e-12)) if np.isfinite(mean) else np.nan
    skew = _nan_skew(valid)
    kurt = _nan_kurtosis_excess(valid)
    return {
        "median": med,
        "mean": mean,
        "std": std,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p10": float(q10),
        "p25": float(q25),
        "p75": float(q75),
        "p90": float(q90),
        "iqr": float(q75 - q25),
        "mad": mad,
        "range": rng,
        "cv": cv,
        "skew": float(skew) if np.isfinite(skew) else np.nan,
        "kurt": float(kurt) if np.isfinite(kurt) else np.nan,
    }


# -----------------------------
# Extraction core (multi-scale from one read)
# -----------------------------
def extract_product_multiscale(
    ds: rasterio.DatasetReader,
    lat: float,
    lon: float,
    product: str,
    radius_km: float,
    data_band: int,
    requested_qa_band: int,
    qa_threshold: float,
    min_valid_pixels: int,
) -> Dict[str, Any]:
    """
    Computes features at radii [0.5R, R, 2R] using ONE max-window read (2R),
    then picks a "best" radius:
      - first radius with n_valid >= min_valid_pixels, else radius with max n_valid, else none.
    Returns:
      - best features (no suffix)
      - multiscale summaries with suffixes _r05, _r1, _r2
      - delta features between scales
    """
    p = product.lower()

    # base output scaffold
    out: Dict[str, Any] = {}

    # default/robust guards
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        out[f"{p}_invalid_reason"] = "latlon_out_of_range"
        out[f"{p}_n_pixels"] = 0
        return out

    try:
        x, y = _lonlat_to_dataset_xy(ds, lon, lat)
        row, col = ds.index(x, y)
        row, col = int(row), int(col)
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            out[f"{p}_invalid_reason"] = "point_outside_raster"
            out[f"{p}_n_pixels"] = 0
            return out

        pix_per_km = max(_estimate_pixels_per_km(ds, lat), 1e-6)

        # radii and pixel radii
        radii_km = [0.5 * radius_km, radius_km, 2.0 * radius_km]
        rpxs = [max(1, int(round(r * pix_per_km))) for r in radii_km]
        rpx_max = rpxs[-1]

        # read max window once
        r0 = max(0, row - rpx_max)
        r1 = min(ds.height, row + rpx_max + 1)
        c0 = max(0, col - rpx_max)
        c1 = min(ds.width, col + rpx_max + 1)
        if r0 >= r1 or c0 >= c1:
            out[f"{p}_invalid_reason"] = "empty_window"
            out[f"{p}_n_pixels"] = 0
            return out

        wmax = Window(c0, r0, c1 - c0, r1 - r0)

        # value scaling/masking
        tags_val = ds.tags(data_band)
        fill_val = _get_fill_value(ds, data_band)

        raw_val = ds.read(data_band, window=wmax).astype(np.float64)
        val, scale_used, offset_used = _apply_scale_offset(raw_val, tags_val)
        if fill_val is not None:
            val[raw_val == fill_val] = np.nan
        val[(val < -1e12) | (val > 1e12)] = np.nan  # ultra-sanity only

        # QA
        qa_band = _detect_qa_band(ds, requested_qa_band) if requested_qa_band else 0
        qa_available = int(qa_band > 0 and qa_band != data_band and ds.count >= qa_band)
        qa = None
        if qa_available:
            tags_qa = ds.tags(qa_band)
            fill_qa = _get_fill_value(ds, qa_band)
            raw_qa = ds.read(qa_band, window=wmax).astype(np.float64)
            qa, _, _ = _apply_scale_offset(raw_qa, tags_qa)
            if fill_qa is not None:
                qa[raw_qa == fill_qa] = np.nan

        # center within max window
        rr = row - r0
        cc = col - c0

        # compute per-scale stats
        scale_suffixes = ["r05", "r1", "r2"]
        scale_rows: List[Dict[str, Any]] = []

        for rpx, rkm, suf in zip(rpxs, radii_km, scale_suffixes):
            rs0 = max(0, rr - rpx)
            rs1 = min(val.shape[0], rr + rpx + 1)
            cs0 = max(0, cc - rpx)
            cs1 = min(val.shape[1], cc + rpx + 1)

            sub_val = val[rs0:rs1, cs0:cs1]
            total_pix = int(sub_val.size)

            qa_pass_fraction = 1.0
            if qa_available and qa is not None:
                sub_qa = qa[rs0:rs1, cs0:cs1]
                qa_mask = (sub_qa >= qa_threshold)
                qa_mask = np.where(np.isnan(sub_qa), False, qa_mask)
                passed = int(np.sum(qa_mask))
                qa_pass_fraction = float(passed / total_pix) if total_pix > 0 else 0.0
                sub_val = np.where(qa_mask, sub_val, np.nan)

            valid = sub_val[~np.isnan(sub_val)]
            stats = _summarize_valid(valid)

            row_out: Dict[str, Any] = {
                "suffix": suf,
                "radius_km": float(rkm),
                "window_size_used": int(2 * rpx + 1),
                "n_pixels": int(valid.size),
                "window_coverage": float(valid.size / total_pix) if total_pix > 0 else 0.0,
                "qa_pass_fraction": float(qa_pass_fraction),
                "has_valid": int(valid.size > 0),
                "invalid_reason": "" if valid.size > 0 else "no_valid_pixels",
            }
            row_out.update(stats)
            scale_rows.append(row_out)

        # choose best scale
        best_idx = None
        for i, r in enumerate(scale_rows):
            if r["n_pixels"] >= int(min_valid_pixels):
                best_idx = i
                break
        if best_idx is None:
            # pick the scale with the most pixels, if any
            nps = [r["n_pixels"] for r in scale_rows]
            mx = max(nps) if nps else 0
            if mx > 0:
                best_idx = int(np.argmax(nps))

        # export multiscale columns (lightweight but useful)
        for r in scale_rows:
            suf = r["suffix"]
            out[f"{p}_median_{suf}"] = r["median"]
            out[f"{p}_std_{suf}"] = r["std"]
            out[f"{p}_n_pixels_{suf}"] = r["n_pixels"]
            out[f"{p}_window_coverage_{suf}"] = r["window_coverage"]
            out[f"{p}_qa_pass_fraction_{suf}"] = r["qa_pass_fraction"]
            out[f"{p}_invalid_reason_{suf}"] = r["invalid_reason"]

        # delta features (gradients / scale sensitivity)
        out[f"{p}_delta_median_r1_r05"] = out.get(f"{p}_median_r1", np.nan) - out.get(f"{p}_median_r05", np.nan)
        out[f"{p}_delta_median_r2_r1"] = out.get(f"{p}_median_r2", np.nan) - out.get(f"{p}_median_r1", np.nan)
        out[f"{p}_delta_std_r2_r05"] = out.get(f"{p}_std_r2", np.nan) - out.get(f"{p}_std_r05", np.nan)

        # best-scale features (the "main" ones you train on)
        out[f"{p}_scale_factor"] = float(scale_used)
        out[f"{p}_add_offset"] = float(offset_used)
        out[f"{p}_qa_available"] = int(qa_available)
        out[f"{p}_qa_threshold"] = float(qa_threshold)

        if best_idx is None:
            out[f"{p}_invalid_reason"] = "no_valid_pixels_after_masking"
            out[f"{p}_n_pixels"] = 0
            out[f"{p}_radius_km_used"] = np.nan
            out[f"{p}_window_size_used"] = 0
            out[f"{p}_window_coverage"] = 0.0
            out[f"{p}_qa_pass_fraction"] = 0.0

            # main stats
            for k in ["median", "mean", "std", "min", "max", "p10", "p25", "p75", "p90", "iqr", "mad", "range", "cv", "skew", "kurt"]:
                out[f"{p}_{k}"] = np.nan
            return out

        b = scale_rows[best_idx]
        out[f"{p}_invalid_reason"] = "" if b["n_pixels"] > 0 else b["invalid_reason"]
        out[f"{p}_n_pixels"] = int(b["n_pixels"])
        out[f"{p}_radius_km_used"] = float(b["radius_km"])
        out[f"{p}_window_size_used"] = int(b["window_size_used"])
        out[f"{p}_window_coverage"] = float(b["window_coverage"])
        out[f"{p}_qa_pass_fraction"] = float(b["qa_pass_fraction"])

        for k in ["median", "mean", "std", "min", "max", "p10", "p25", "p75", "p90", "iqr", "mad", "range", "cv", "skew", "kurt"]:
            out[f"{p}_{k}"] = b[k]

        return out

    except Exception as e:
        out[f"{p}_invalid_reason"] = f"exception:{type(e).__name__}"
        out[f"{p}_n_pixels"] = 0
        return out


# -----------------------------
# Per-day processing
# -----------------------------
_PRINTED_PRODUCT_INFO: set[str] = set()


def _product_feature_cols(product: str) -> List[str]:
    p = product.lower()
    # best-scale engineered features
    cols = [
        f"{p}_median", f"{p}_mean", f"{p}_std", f"{p}_min", f"{p}_max",
        f"{p}_p10", f"{p}_p25", f"{p}_p75", f"{p}_p90",
        f"{p}_iqr", f"{p}_mad", f"{p}_range", f"{p}_cv", f"{p}_skew", f"{p}_kurt",
        f"{p}_n_pixels", f"{p}_radius_km_used", f"{p}_window_size_used",
        f"{p}_window_coverage", f"{p}_scale_factor", f"{p}_add_offset",
        f"{p}_qa_available", f"{p}_qa_threshold", f"{p}_qa_pass_fraction",
        f"{p}_invalid_reason",
        # multiscale
        f"{p}_median_r05", f"{p}_std_r05", f"{p}_n_pixels_r05", f"{p}_window_coverage_r05", f"{p}_qa_pass_fraction_r05", f"{p}_invalid_reason_r05",
        f"{p}_median_r1",  f"{p}_std_r1",  f"{p}_n_pixels_r1",  f"{p}_window_coverage_r1",  f"{p}_qa_pass_fraction_r1",  f"{p}_invalid_reason_r1",
        f"{p}_median_r2",  f"{p}_std_r2",  f"{p}_n_pixels_r2",  f"{p}_window_coverage_r2",  f"{p}_qa_pass_fraction_r2",  f"{p}_invalid_reason_r2",
        # deltas
        f"{p}_delta_median_r1_r05", f"{p}_delta_median_r2_r1", f"{p}_delta_std_r2_r05",
        # file audit
        f"{p}_file_available", f"{p}_file_used",
    ]
    return cols


def process_date_group(
    df_date: pd.DataFrame,
    file_info_by_product: Dict[str, Optional[dict]],
    products: List[str],
    chunk: int,
    radius_km: float,
    data_band: int,
    qa_band: int,
    qa_threshold: float,
    min_valid_pixels: int,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    if df_date.empty:
        return df_date.copy()

    out = df_date.copy()

    # initialize columns
    for prod in products:
        p = prod.lower()
        cols = _product_feature_cols(prod)
        for c in cols:
            if c.endswith("_n_pixels") or c.endswith("_window_size_used") or c.endswith("_qa_available") or c.endswith("_file_available"):
                out[c] = 0
            elif c.endswith("_window_coverage") or c.endswith("_qa_pass_fraction") or c.endswith("_qa_threshold"):
                out[c] = 0.0
            elif c.endswith("_invalid_reason") or c.endswith("_file_used"):
                out[c] = ""
            else:
                out[c] = np.nan

        fi = file_info_by_product.get(prod)
        out[f"{p}_file_available"] = 1 if fi else 0
        out[f"{p}_file_used"] = fi["path"].name if fi else ""

    n = len(out)
    out_parts: List[pd.DataFrame] = []

    # open datasets once per date (huge speedup)
    ds_by_prod: Dict[str, rasterio.DatasetReader] = {}
    qa_band_by_prod: Dict[str, int] = {}
    try:
        for prod in products:
            fi = file_info_by_product.get(prod)
            if not fi:
                continue
            ds = rasterio.open(fi["path"])
            ds_by_prod[prod] = ds
            qa_band_detected = _detect_qa_band(ds, qa_band) if qa_band > 0 else 0
            qa_ok = int(qa_band_detected > 0 and qa_band_detected != data_band and ds.count >= qa_band_detected)
            qa_band_by_prod[prod] = qa_band_detected if qa_ok else 0

            if prod not in _PRINTED_PRODUCT_INFO:
                print(
                    f"  INFO {prod}: {fi['path'].name} | CRS={ds.crs} | size={ds.width}x{ds.height} "
                    f"| bands={ds.count} | data_band={data_band} | QA={'yes' if qa_ok else 'no'}(band={qa_band_by_prod[prod]})"
                )
                _PRINTED_PRODUCT_INFO.add(prod)

        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            block = out.iloc[i0:i1].copy()

            lats = block[lat_col].to_numpy(dtype=float)
            lons = block[lon_col].to_numpy(dtype=float)

            for prod in products:
                fi = file_info_by_product.get(prod)
                if not fi:
                    continue
                ds = ds_by_prod.get(prod)
                if ds is None:
                    continue

                p = prod.lower()

                # prealloc arrays for speed
                cols = _product_feature_cols(prod)
                tmp: Dict[str, Any] = {}
                for c in cols:
                    if c.endswith("_n_pixels") or c.endswith("_window_size_used") or c.endswith("_qa_available") or c.endswith("_file_available"):
                        tmp[c] = np.zeros(len(block), dtype=np.int32)
                    elif c.endswith("_window_coverage") or c.endswith("_qa_pass_fraction") or c.endswith("_qa_threshold"):
                        tmp[c] = np.zeros(len(block), dtype=np.float64)
                    elif c.endswith("_invalid_reason") or c.endswith("_file_used") or "invalid_reason" in c:
                        tmp[c] = np.array([""] * len(block), dtype=object)
                    else:
                        tmp[c] = np.full(len(block), np.nan, dtype=np.float64)

                # keep file audit constant
                tmp[f"{p}_file_available"][:] = 1
                tmp[f"{p}_file_used"][:] = fi["path"].name

                qa_use = qa_band_by_prod.get(prod, 0)

                for j in range(len(block)):
                    lat, lon = float(lats[j]), float(lons[j])

                    res = extract_product_multiscale(
                        ds,
                        lat=lat,
                        lon=lon,
                        product=prod,
                        radius_km=radius_km,
                        data_band=data_band,
                        requested_qa_band=qa_use,
                        qa_threshold=qa_threshold,
                        min_valid_pixels=min_valid_pixels,
                    )

                    # write results (only keys we created)
                    for k, v in res.items():
                        if k in tmp:
                            tmp[k][j] = v

                # commit to block
                for c in cols:
                    block[c] = tmp[c]

            out_parts.append(block)

        return pd.concat(out_parts, axis=0)

    finally:
        for ds in ds_by_prod.values():
            try:
                ds.close()
            except Exception:
                pass


# -----------------------------
# Product dir override parsing
# -----------------------------
def _parse_dir_overrides(s: str) -> Dict[str, Path]:
    """
    "NH3=/path/to/NH3,CO=/path/to/CO"
    """
    out: Dict[str, Path] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad override '{p}'. Expected like 'NH3=/path/to/dir'.")
        k, v = p.split("=", 1)
        out[k.strip().upper()] = Path(v.strip())
    return out


# -----------------------------
# Driver
# -----------------------------
def compute_tropomi_for_observations(
    df_obs: pd.DataFrame,
    tropomi_base_dir: Path,
    geos_cf_base_dir: Optional[Path] = None,
    products: List[str] = None,
    chunk: int = 1000,
    radius_km: float = 10.0,
    data_band: int = 1,
    qa_band: int = 2,
    qa_threshold: float = 0.75,
    min_valid_pixels: int = 5,
    product_dir_overrides: Dict[str, Path] = None,
) -> pd.DataFrame:
    """
    ETL wrapper: Compute TROPOMI features for a DataFrame of observations.

    This is the feature_store-compatible interface that takes an already-loaded
    DataFrame and returns features without writing to disk.

    Args:
        df_obs: DataFrame with columns: time (or date_utc), latitude/obs_lat, longitude/obs_lon
        tropomi_base_dir: Path to base directory containing TROPOMI product subdirectories
        geos_cf_base_dir: Optional path for GEOS-CF products (e.g., NH3)
        products: List of products to process (default: NO2, SO2, CO, etc.)
        chunk: Processing chunk size
        radius_km: Search radius in km
        data_band: 1-indexed band for product value
        qa_band: QA band (1-indexed, 0 to disable)
        qa_threshold: Minimum QA value for valid pixels
        min_valid_pixels: Minimum valid pixels required

    Returns:
        DataFrame with TROPOMI features added to each observation row

    Example:
        >>> df_aod = pd.read_csv("output/paqi_with_aod_features.csv")
        >>> df_tropomi = compute_tropomi_for_observations(
        ...     df_aod,
        ...     Path("datasets/tropomi_pakistan_2020_2025"),
        ...     geos_cf_base_dir=Path("datasets/geos_cf_pakistan_2020_2025"),
        ... )
    """
    if products is None:
        products = ["NO2", "SO2", "CO", "HCHO", "AAI", "ALH", "CH4", "CLOUD", "O3"]
    if product_dir_overrides is None:
        product_dir_overrides = {}

    df = df_obs.copy()

    # Resolve columns robustly
    time_col = _resolve_col(df, ["time", "date_utc", "datetime", "date"], required=True)
    lat_col = _resolve_col(df, ["obs_lat", "latitude", "lat"], required=True)
    lon_col = _resolve_col(df, ["obs_lon", "longitude", "lon", "lng"], required=True)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()
    if df.empty:
        raise ValueError(f"No valid rows after parsing '{time_col}'")

    df["date"] = df[time_col].dt.date

    # CRITICAL: Generate deterministic sensor_id if missing (must match MET exactly)
    if "sensor_id" not in df.columns:
        print("[tropomi-etl] Generating sensor_id from coordinates...")
        df["sensor_id"] = [
            _create_deterministic_sensor_id(lat, lon)
            for lat, lon in zip(df[lat_col], df[lon_col])
        ]

    # Scan files for each product
    file_maps: Dict[str, Dict[datetime.date, Dict[str, Any]]] = {}
    print("[tropomi-etl] Scanning product directories...")

    for prod in products:
        if prod in product_dir_overrides:
            pdir = product_dir_overrides[prod]
        elif prod == "NH3" and geos_cf_base_dir is not None:
            pdir = geos_cf_base_dir / "NH3"
        else:
            pdir = Path(tropomi_base_dir) / prod

        print(f"  - {prod}: {pdir}")
        file_maps[prod] = find_product_files(pdir)

    dates = sorted(df["date"].unique())
    print(f"[tropomi-etl] Processing {len(df):,} observations across {len(dates)} days ({dates[0]} to {dates[-1]})")

    all_parts = []

    for i, d in enumerate(dates):
        sub = df[df["date"] == d].copy()
        date_str = d.strftime("%Y-%m-%d")

        file_info_by_product = {prod: file_maps[prod].get(d) for prod in products}

        if i < 5 or i % 30 == 0:
            avail = ", ".join([f"{p}:{'Y' if file_info_by_product[p] else 'N'}" for p in products])
            print(f"[tropomi-etl] {date_str}: {len(sub):,} obs | {avail}")

        processed = process_date_group(
            sub,
            file_info_by_product=file_info_by_product,
            products=products,
            chunk=chunk,
            radius_km=radius_km,
            data_band=data_band,
            qa_band=qa_band,
            qa_threshold=qa_threshold,
            min_valid_pixels=min_valid_pixels,
            lat_col=lat_col,
            lon_col=lon_col,
        )

        processed = processed.drop(columns=["date"], errors="ignore")
        all_parts.append(processed)

    result = pd.concat(all_parts, ignore_index=True) if all_parts else df.copy()

    print(f"[tropomi-etl] Done. {len(result):,} rows, {len(result.columns)} columns")
    return result


def run_multi_product(
    input_csv: Path,
    tropomi_base_dir: Path,
    output_csv: Path,
    products: List[str],
    chunk: int,
    radius_km: float,
    data_band: int,
    qa_band: int,
    qa_threshold: float,
    min_valid_pixels: int,
    overwrite: bool,
    geos_cf_base_dir: Optional[Path],
    product_dir_overrides: Dict[str, Path],
):
    df = pd.read_csv(input_csv)

    # resolve cols robustly
    time_col = _resolve_col(df, ["time", "date_utc", "datetime", "date"], required=True)
    lat_col = _resolve_col(df, ["obs_lat", "latitude", "lat"], required=True)
    lon_col = _resolve_col(df, ["obs_lon", "longitude", "lon", "lng"], required=True)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()
    if df.empty:
        raise SystemExit(f"No valid rows after parsing '{time_col}'.")

    df["date"] = df[time_col].dt.date

    # scan files for each product
    file_maps: Dict[str, Dict[datetime.date, Dict[str, Any]]] = {}
    print("Scanning product directories...")

    for prod in products:
        # directory selection priority:
        # 1) explicit override
        # 2) NH3 -> geos_cf_base_dir/NH3 (if provided)
        # 3) default -> tropomi_base_dir/PROD
        if prod in product_dir_overrides:
            pdir = product_dir_overrides[prod]
        elif prod == "NH3" and geos_cf_base_dir is not None:
            pdir = geos_cf_base_dir / "NH3"
        else:
            pdir = tropomi_base_dir / prod

        print(f"- {prod}: {pdir}")
        file_maps[prod] = find_product_files(pdir)

    dates = sorted(df["date"].unique())
    print(f"\nRows: {len(df):,} | Unique days: {len(dates):,} | Range: {dates[0]} .. {dates[-1]}")

    output_csv = Path(output_csv)
    if output_csv.exists():
        if overwrite:
            output_csv.unlink()
        else:
            raise SystemExit(f"Output exists: {output_csv} (use --overwrite)")

    wrote_header = False
    total_written = 0

    for i, d in enumerate(dates):
        sub = df[df["date"] == d].copy()
        date_str = d.strftime("%Y-%m-%d")

        file_info_by_product = {prod: file_maps[prod].get(d) for prod in products}

        if i < 10 or i % 10 == 0:
            avail = ", ".join([f"{p}:{'Y' if file_info_by_product[p] else 'N'}" for p in products])
            print(f"\nPROCESS {date_str}: {len(sub):,} rows | {avail}")

        processed = process_date_group(
            sub,
            file_info_by_product=file_info_by_product,
            products=products,
            chunk=chunk,
            radius_km=radius_km,
            data_band=data_band,
            qa_band=qa_band,
            qa_threshold=qa_threshold,
            min_valid_pixels=min_valid_pixels,
            lat_col=lat_col,
            lon_col=lon_col,
        )

        processed = processed.drop(columns=["date"])
        processed.to_csv(
            output_csv,
            index=False,
            mode=("w" if not wrote_header else "a"),
            header=(not wrote_header),
        )
        wrote_header = True
        total_written += len(processed)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(dates)} days | {total_written:,} rows written")

    print(f"\n✅ Done → {output_csv}")
    print(f"Total rows written: {total_written:,}")
    
    # Return the final dataset for feature registration
    return pd.read_csv(output_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_file", default="output/paqi_with_all_features.csv", type=Path)
    ap.add_argument("--tropomi_base_dir", default="datasets/tropomi_pakistan_2020_2025", type=Path)
    ap.add_argument("--geos_cf_base_dir", default="datasets/geos_cf_pakistan_2020_2025", type=Path, help="Optional base dir for GEOS-CF products (e.g., NH3)")

    ap.add_argument(
        "--products",
        default="NO2,SO2,CO,HCHO,AAI,ALH,CH4,CLOUD,O3,NH3",
        help="Comma list of subfolder names. Add NH3 if you have GEOS-CF NH3.",
    )

    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--radius_km", type=float, default=10.0)
    ap.add_argument("--min_valid_pixels", type=int, default=5)

    ap.add_argument("--data_band", type=int, default=1, help="1-indexed band for product value")
    ap.add_argument("--qa_band", type=int, default=2, help="Preferred QA band (1-indexed). Set 0 to disable QA globally.")
    ap.add_argument("--qa_threshold", type=float, default=0.75)

    ap.add_argument(
        "--product_dir_overrides",
        default="",
        help="Optional overrides like 'NH3=/abs/path/to/NH3,CO=/abs/path/to/CO'",
    )

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    # Check if CSV already has TROPOMI features
    skip_processing = False
    if args.csv_file.exists():
        try:
            df_check = pd.read_csv(args.csv_file, nrows=1)
            # Check for key TROPOMI features that indicate processing is complete
            key_tropomi_features = ["no2_median", "so2_median", "co_median", "hcho_median", "aai_median"]
            existing_tropomi_features = [col for col in key_tropomi_features if col in df_check.columns]
            
            if len(existing_tropomi_features) >= 1:  # At least 1 key feature present
                print(f"[tropomi_pipeline] Output file {args.csv_file} already exists with TROPOMI features - skipping processing")
                print(f"[tropomi_pipeline] Found TROPOMI features: {existing_tropomi_features}")
                skip_processing = True
        except Exception as e:
            print(f"[tropomi_pipeline] Warning: Could not read existing file {args.csv_file}: {e}")
    
    # Create output directory if it doesn't exist
    args.csv_file.parent.mkdir(parents=True, exist_ok=True)

    products = [p.strip().upper() for p in args.products.split(",") if p.strip()]
    allowed = {"NO2", "SO2", "CO", "HCHO", "AAI", "ALH", "CH4", "CLOUD", "O3", "NH3"}
    bad = [p for p in products if p not in allowed]
    if bad:
        raise SystemExit(f"Unknown products: {bad}. Allowed: {sorted(allowed)}")

    overrides = _parse_dir_overrides(args.product_dir_overrides)

    if not skip_processing:
        # Use temp file to avoid polluting master CSV
        tmp_full = Path(args.csv_file).with_suffix(".tropomi_full.csv")
        
        output_data = run_multi_product(
            input_csv=args.csv_file,
            tropomi_base_dir=args.tropomi_base_dir,
            geos_cf_base_dir=args.geos_cf_base_dir,
            output_csv=str(tmp_full),
            products=products,
            chunk=args.chunk,
            radius_km=args.radius_km,
            data_band=args.data_band,
            qa_band=args.qa_band,
            qa_threshold=args.qa_threshold,
            min_valid_pixels=args.min_valid_pixels,
            overwrite=True,
            product_dir_overrides=overrides,
        )
        
        # Register from temp file
        feat_cols, meta_cols = tropomi_feature_lists(products)
        df_full = pd.read_csv(tmp_full)
        
        register_stage_features(
            stage="tropomi",
            data=df_full,
            feature_columns=[c for c in feat_cols if c in df_full.columns],
            metadata_columns=[c for c in meta_cols if c in df_full.columns],
        )
        
        # Clean up temp file
        tmp_full.unlink()
        print(f"[tropomi_pipeline] Registry updated for {len(df_full):,} rows, {len(df_full.columns)} cols")
        
    else:
        # If skipped, register from existing master (limited columns)
        if args.csv_file.exists():
            feat_cols, meta_cols = tropomi_feature_lists(products)
            usecols = [c for c in (["sensor_id", "time"] + feat_cols + meta_cols) if c in pd.read_csv(args.csv_file, nrows=1).columns]
            df_reg = pd.read_csv(args.csv_file, usecols=usecols)
            
            register_stage_features(
                stage="tropomi",
                data=df_reg,
                feature_columns=[c for c in feat_cols if c in df_reg.columns],
                metadata_columns=[c for c in meta_cols if c in df_reg.columns],
            )
            print(f"[tropomi_pipeline] Registry updated for {len(df_reg):,} rows, {len(df_reg.columns)} cols")


def tropomi_feature_lists(products):
    """Generate lists of feature and metadata columns for TROPOMI products."""
    # keep "physical/stat" columns as features
    stats = ["median","mean","std","min","max","p10","p25","p75","p90","iqr","mad","range","cv","skew","kurt"]
    feature_cols = []
    meta_cols = []
    
    for p in [x.lower() for x in products]:
        feature_cols += [f"{p}_{s}" for s in stats]
        feature_cols += [f"{p}_delta_median_r1_r05", f"{p}_delta_median_r2_r1", f"{p}_delta_std_r2_r05"]

        # treat quality/file/coverage/invalid_reason as metadata by default
        meta_cols += [
            f"{p}_n_pixels", f"{p}_radius_km_used", f"{p}_window_size_used",
            f"{p}_window_coverage", f"{p}_scale_factor", f"{p}_add_offset",
            f"{p}_qa_available", f"{p}_qa_threshold", f"{p}_qa_pass_fraction",
            f"{p}_invalid_reason", f"{p}_file_available", f"{p}_file_used",
        ]

        # multiscale outputs: recommend metadata unless you *explicitly* want them as training features
        for suf in ["r05","r1","r2"]:
            meta_cols += [
                f"{p}_median_{suf}", f"{p}_std_{suf}", f"{p}_n_pixels_{suf}",
                f"{p}_window_coverage_{suf}", f"{p}_qa_pass_fraction_{suf}",
                f"{p}_invalid_reason_{suf}",
            ]

    return feature_cols, meta_cols




if __name__ == "__main__":
    main()