from __future__ import annotations
import argparse
import hashlib
import re
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

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

AOD_FEATURES = [
    "optical_depth_047", "optical_depth_055", "aod_uncertainty",
    "qa_cloudmask", "qa_adjacency", "qa_aod",
]
AOD_METADATA = [
    "qa_n_pixels", "aod_total_valid_pixels", "aod_files_used",
    "aod_window_size_used", "aod_n_hdf_files", "aod_hdf_tiles_used",
]

# MODIS MAIAC AOD Science Data Sets (SDS) to extract (numeric science layers)
AOD_SDS_NAMES = [
    "Optical_Depth_047",
    "Optical_Depth_055", 
    "AOD_Uncertainty",
]

# QA-derived categorical features decoded from AOD_QA bitfield
QA_FEATURE_NAMES = [
    "qa_cloudmask",   # bits 0-2
    "qa_adjacency",   # bits 5-7
    "qa_aod",         # bits 8-11
    "qa_n_pixels",    # how many QA pixels contributed (window-valid)
]

TARGET_DATASETS = ["Optical_Depth_047", "Optical_Depth_055", "AOD_Uncertainty", "AOD_QA"]

@dataclass
class OpenHDF:
    hdf: Any
    sds: Dict[str, Any]                 # dataset name -> SDS handle
    meta: Dict[str, Tuple[float, float, float]]  # name -> (scale_factor, add_offset, fill_value)
    shape: Dict[str, Tuple[int, int, int]]       # name -> (rank, height, width)

def _open_hdf_cached(path: Path) -> OpenHDF:
    hdf = SD(str(path), SDC.READ)
    available = set(hdf.datasets().keys())

    sds = {}
    meta = {}
    shape = {}

    for name in TARGET_DATASETS:
        if name not in available:
            continue
        ds = hdf.select(name)
        sds[name] = ds

        attrs = ds.attributes()
        scale_factor = float(attrs.get("scale_factor", 1.0))
        add_offset  = float(attrs.get("add_offset", 0.0))
        fill_value  = float(attrs.get("_FillValue", -32768))

        name0, rank, dims, _dtype, _nattrs = ds.info()
        # rank is int, dims is tuple
        if rank == 2:
            height, width = dims
        elif rank == 3:
            _, height, width = dims
        else:
            # unexpected rank; skip this dataset
            continue

        meta[name] = (scale_factor, add_offset, fill_value)
        shape[name] = (rank, int(height), int(width))

    return OpenHDF(hdf=hdf, sds=sds, meta=meta, shape=shape)

def _close_hdf_cache(cache: Dict[Path, OpenHDF]) -> None:
    for ohdf in cache.values():
        try:
            for ds in ohdf.sds.values():
                try:
                    ds.endaccess()
                except Exception:
                    pass
            ohdf.hdf.end()
        except Exception:
            pass

def _read_window(ds: Any, rank: int, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
    # IMPORTANT: slicing reads only the requested hyperslab (fast) rather than full tile
    if rank == 2:
        return ds[row0:row1, col0:col1]
    if rank == 3:
        return ds[:, row0:row1, col0:col1]
    return np.array([], dtype=np.float64)


def _mode_int(x: np.ndarray, max_value: int) -> float:
    """
    Return the mode (most frequent) integer in x.
    Returns np.nan if x is empty.
    """
    if x.size == 0:
        return np.nan
    x = x.astype(np.int64)
    x = x[(x >= 0) & (x <= max_value)]
    if x.size == 0:
        return np.nan
    counts = np.bincount(x, minlength=max_value + 1)
    return float(np.argmax(counts))


def decode_aod_qa_bitfield(valid_qa_values_1d: np.ndarray) -> dict:
    """
    Decode MAIAC AOD_QA bitfield for a window of pixels.

    Bits (per MAIAC QA table):
      - 0-2  : Cloud Mask (3 bits)
      - 5-7  : Adjacency Mask (3 bits)
      - 8-11 : QA for AOD over land/water (4 bits)

    We summarize each decoded field by MODE across the window.
    """
    v = valid_qa_values_1d.astype(np.uint16)

    cloudmask = (v >> 0) & 0b111       # 0..7
    adjacency = (v >> 5) & 0b111       # 0..7
    qa_aod    = (v >> 8) & 0b1111      # 0..15

    return {
        "qa_cloudmask": _mode_int(cloudmask, max_value=7),
        "qa_adjacency": _mode_int(adjacency, max_value=7),
        "qa_aod": _mode_int(qa_aod, max_value=15),
        "qa_n_pixels": float(v.size),
    }


def latlon_to_modis_tile_pixel(lat, lon):
    """Convert lat/lon to MODIS tile (h,v) and pixel (row,col) coordinates"""
    R = 6371007.181
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = R * lon_rad * np.cos(lat_rad)
    y = R * lat_rad

    xmin = -20015109.353729
    ymax =  10007554.676865

    tile_size  = 1111950.5197665554
    pixel_size = tile_size / 1200.0

    h = int((x - xmin) / tile_size)
    v = int((ymax - y) / tile_size)

    x_in = (x - xmin) - h * tile_size
    y_in = (ymax - y) - v * tile_size

    col = int(x_in / pixel_size)
    row = int(y_in / pixel_size)

    if not (0 <= h <= 35 and 0 <= v <= 17 and 0 <= row < 1200 and 0 <= col < 1200):
        return None  # mapping failed

    return h, v, row, col


def find_hdf_files(aod_base_dir: Path):
    """Find all MCD19A2 HDF files across all years and organize by date"""
    # Pattern: MCD19A2.AYYYYDDD.hXXvYY.061.YYMMDDHHMMSS.hdf
    pattern = re.compile(r"MCD19A2\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.061\..*\.hdf$")
    
    file_map = {}
    
    # Look for year directories under the base AOD directory
    for year_dir in aod_base_dir.glob("*/"):
        if year_dir.is_dir() and year_dir.name.isdigit():
            print(f"Scanning {year_dir} for HDF files...")
            year_files = 0
            
            for hdf_file in year_dir.glob("*.hdf"):
                match = pattern.match(hdf_file.name)
                if match:
                    year = int(match.group(1))
                    doy = int(match.group(2))  # Day of year
                    h_tile = int(match.group(3))
                    v_tile = int(match.group(4))
                    
                    # Convert day of year to date
                    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                    date_key = date.date()
                    
                    if date_key not in file_map:
                        file_map[date_key] = []
                    file_map[date_key].append({
                        'path': hdf_file,
                        'h_tile': h_tile,
                        'v_tile': v_tile,
                        'date': date_key,
                        'year': year
                    })
                    year_files += 1
            
            print(f"  Found {year_files:,} HDF files in {year_dir.name}")
    
    print(f"Total: {sum(len(files) for files in file_map.values()):,} HDF files across {len(file_map)} days")
    return file_map




def extract_aod_from_open_hdf_progressive_window(
    ohdf: OpenHDF,
    row: int,
    col: int,
    debug: bool = False,
) -> dict:
    """
    Same logic as before, but:
      - HDF is already open (cached)
      - Reads ONLY the local window (3/5/7) via hyperslab slice
    """
    try:
        if not ohdf.sds:
            return {}

        # pick a reference dataset to get bounds (prefer OD_055)
        ref_name = "Optical_Depth_055" if "Optical_Depth_055" in ohdf.shape else next(iter(ohdf.shape))
        ref_rank, height, width = ohdf.shape[ref_name]

        aod_data = {}

        for window_size in (3, 5, 7):
            r = window_size // 2
            row0 = max(0, row - r)
            row1 = min(height, row + r + 1)
            col0 = max(0, col - r)
            col1 = min(width, col + r + 1)

            if row0 >= row1 or col0 >= col1:
                continue

            if debug:
                print(f"  DEBUG: Trying {window_size}x{window_size} window at ({row},{col})")

            window_success = False
            temp = {}

            for name in TARGET_DATASETS:
                if name not in ohdf.sds or name not in ohdf.shape:
                    continue

                ds = ohdf.sds[name]
                rank, _, _ = ohdf.shape[name]
                scale_factor, add_offset, fill_value = ohdf.meta.get(name, (1.0, 0.0, -32768.0))

                win = _read_window(ds, rank, row0, row1, col0, col1).astype(np.float64).ravel()
                if win.size == 0:
                    continue

                win[win == fill_value] = np.nan

                if name == "AOD_QA":
                    valid = win[~np.isnan(win)]
                    if valid.size > 0:
                        temp.update(decode_aod_qa_bitfield(valid.astype(np.uint16)))
                    else:
                        temp.update({
                            "qa_cloudmask": np.nan,
                            "qa_adjacency": np.nan,
                            "qa_aod": np.nan,
                            "qa_n_pixels": 0.0,
                        })
                    continue

                # numeric layers
                scaled = win * scale_factor + add_offset
                scaled[(scaled < -0.05) | (scaled > 5.0)] = np.nan
                vv = scaled[~np.isnan(scaled)]

                key = name.lower()  # optical_depth_055, optical_depth_047, aod_uncertainty
                if vv.size > 0:
                    temp[key] = float(np.median(vv))
                    temp[f"{key}_n_pixels"] = int(vv.size)
                    window_success = True
                    if debug and name == "Optical_Depth_055":
                        print(f"  DEBUG: {name} success @ {window_size}x{window_size}: {temp[key]:.3f} ({vv.size} px)")
                else:
                    temp[key] = np.nan
                    temp[f"{key}_n_pixels"] = 0

            if window_success:
                aod_data = temp
                aod_data["window_size_used"] = window_size
                if debug:
                    print(f"  DEBUG: Using {window_size}x{window_size} window results")
                break
            elif debug:
                print(f"  DEBUG: {window_size}x{window_size} window failed, trying larger")

        # fallback: fill NaNs if nothing worked
        if not aod_data:
            for sds_name in AOD_SDS_NAMES:
                aod_data[sds_name.lower()] = np.nan
                aod_data[f"{sds_name.lower()}_n_pixels"] = 0
            for qn in QA_FEATURE_NAMES:
                aod_data[qn] = np.nan if qn != "qa_n_pixels" else 0.0
            aod_data["window_size_used"] = np.nan

        return aod_data

    except Exception as e:
        if debug:
            print(f"  DEBUG: cached extractor error: {e}")
        return {}


def process_date_group_enhanced(df_date: pd.DataFrame, hdf_files: list, chunk=1000) -> pd.DataFrame:
    """Process sensors for a single date (FAST): tile-index + cached HDF + window reads + itertuples"""
    if df_date.empty or not hdf_files:
        result = df_date.copy()
        for sds_name in AOD_SDS_NAMES:
            result[sds_name.lower()] = np.nan
        for qn in QA_FEATURE_NAMES:
            result[qn] = np.nan if qn != "qa_n_pixels" else 0.0
        result["aod_total_valid_pixels"] = 0
        result["aod_files_used"] = 0
        result["aod_window_size_used"] = np.nan
        result["aod_n_hdf_files"] = 0
        result["aod_hdf_tiles_used"] = ""
        return result

    # 1) Pre-index files by tile (massive speedup vs scanning hdf_files inside sensor loop)
    tile_to_files = defaultdict(list)
    for info in hdf_files:
        tile_to_files[(info["h_tile"], info["v_tile"])].append(info["path"])

    out_parts = []
    n = len(df_date)

    # 2) Cache open HDF per day (reused across all chunks)
    hdf_cache: Dict[Path, OpenHDF] = {}

    try:
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            block = df_date.iloc[i0:i1].copy()

            # init columns
            for sds_name in AOD_SDS_NAMES:
                block[sds_name.lower()] = np.nan
            for qn in QA_FEATURE_NAMES:
                block[qn] = np.nan if qn != "qa_n_pixels" else 0.0

            block["aod_total_valid_pixels"] = 0
            block["aod_files_used"] = 0
            block["aod_window_size_used"] = np.nan
            block["aod_n_hdf_files"] = len(hdf_files)
            block["aod_hdf_tiles_used"] = ""

            # 3) Iterate fast
            first_index = block.index[0] if len(block.index) else None

            for row in block.itertuples(index=True):
                idx = row.Index
                lat = getattr(row, "obs_lat")
                lon = getattr(row, "obs_lon")

                debug_mode = (first_index is not None) and (idx == first_index) and (i0 == 0)

                mapping = latlon_to_modis_tile_pixel(lat, lon)
                if mapping is None:
                    if debug_mode:
                        print(f"  DEBUG: Sensor at ({lat:.4f},{lon:.4f}) mapping failed (out of bounds)")
                    continue

                h_tile, v_tile, pixel_row, pixel_col = mapping
                if debug_mode:
                    print(f"  DEBUG: Sensor maps to h{h_tile:02d}v{v_tile:02d} pixel ({pixel_row},{pixel_col})")

                matching_paths = tile_to_files.get((h_tile, v_tile), [])
                if not matching_paths:
                    if debug_mode:
                        print(f"  DEBUG: No HDF files for tile h{h_tile:02d}v{v_tile:02d}")
                    continue

                # accumulate across files for this tile/day
                all_vals = {sds.lower(): [] for sds in AOD_SDS_NAMES}
                for qn in QA_FEATURE_NAMES:
                    all_vals[qn] = []
                n_pixels = []
                window_sizes = []
                n_files_used = 0

                for p in matching_paths:
                    ohdf = hdf_cache.get(p)
                    if ohdf is None:
                        ohdf = _open_hdf_cached(p)
                        hdf_cache[p] = ohdf

                    aod = extract_aod_from_open_hdf_progressive_window(
                        ohdf,
                        pixel_row,
                        pixel_col,
                        debug=(debug_mode and p == matching_paths[0]),
                    )

                    file_contributed = False
                    for sds_name in AOD_SDS_NAMES:
                        key = sds_name.lower()
                        if key in aod and not np.isnan(aod[key]):
                            all_vals[key].append(aod[key])
                            file_contributed = True

                    for qn in QA_FEATURE_NAMES:
                        if qn in aod and not (isinstance(aod[qn], float) and np.isnan(aod[qn])):
                            all_vals[qn].append(aod[qn])

                    if file_contributed:
                        n_files_used += 1
                        if "window_size_used" in aod and not np.isnan(aod["window_size_used"]):
                            window_sizes.append(aod["window_size_used"])

                        for band in ("optical_depth_055", "optical_depth_047"):
                            pk = f"{band}_n_pixels"
                            if pk in aod:
                                n_pixels.append(aod[pk])

                # aggregate
                aggregated = {}
                for key, values in all_vals.items():
                    if not values:
                        aggregated[key] = np.nan
                        continue
                    arr = np.array(values, dtype=float)
                    if key in ("qa_cloudmask", "qa_adjacency", "qa_aod"):
                        vv = arr[~np.isnan(arr)].astype(int)
                        aggregated[key] = np.nan if vv.size == 0 else float(np.argmax(np.bincount(vv)))
                    elif key == "qa_n_pixels":
                        aggregated[key] = float(np.nansum(arr))
                    else:
                        aggregated[key] = float(np.nanmedian(arr))

                aggregated["aod_total_valid_pixels"] = int(np.nansum(n_pixels)) if n_pixels else 0
                aggregated["aod_files_used"] = int(n_files_used)
                aggregated["aod_window_size_used"] = float(np.nanmedian(window_sizes)) if window_sizes else np.nan

                # write back
                for sds_name in AOD_SDS_NAMES:
                    key = sds_name.lower()
                    if not np.isnan(aggregated.get(key, np.nan)):
                        block.at[idx, key] = aggregated[key]

                for qn in QA_FEATURE_NAMES:
                    if qn in aggregated:
                        block.at[idx, qn] = aggregated[qn]

                block.at[idx, "aod_total_valid_pixels"] = aggregated["aod_total_valid_pixels"]
                block.at[idx, "aod_files_used"] = aggregated["aod_files_used"]
                block.at[idx, "aod_window_size_used"] = aggregated["aod_window_size_used"]
                block.at[idx, "aod_hdf_tiles_used"] = f"h{h_tile:02d}v{v_tile:02d}"
                block.at[idx, "aod_n_hdf_files"] = len(matching_paths)

            out_parts.append(block)

        result = pd.concat(out_parts, ignore_index=True) if out_parts else df_date.copy()

        mask_no_aod = result[["optical_depth_055", "optical_depth_047"]].isna().all(axis=1)
        valid_aod_sensors = (~mask_no_aod).sum()
        if valid_aod_sensors > 0:
            print(f"  SUCCESS: {valid_aod_sensors}/{len(result)} sensors have AOD (047/055)")
        else:
            print(f"  Warning: {len(result)} sensors have no AOD data")

        return result

    finally:
        # 4) Close per-day cache (important)
        _close_hdf_cache(hdf_cache)


def compute_aod_for_observations(
    df_obs: pd.DataFrame,
    aod_dir: Path,
    chunk: int = 1000,
) -> pd.DataFrame:
    """
    ETL wrapper: Compute AOD features for a DataFrame of observations.

    This is the feature_store-compatible interface that takes an already-loaded
    DataFrame and returns features without writing to disk.

    Args:
        df_obs: DataFrame with columns: time (or date_utc), latitude (or obs_lat), longitude (or obs_lon)
        aod_dir: Path to base directory containing year subdirectories with MCD19A2 HDF files
        chunk: Processing chunk size

    Returns:
        DataFrame with AOD features added to each observation row

    Example:
        >>> df_met = pd.read_csv("output/paqi_with_met_features.csv")
        >>> df_aod = compute_aod_for_observations(df_met, Path("datasets/MCD19A2.061"))
    """
    df = df_obs.copy()

    # Ensure required columns exist
    if "time" not in df.columns and "date_utc" in df.columns:
        df["time"] = pd.to_datetime(df["date_utc"])
    elif "time" not in df.columns:
        raise ValueError("Missing required 'time' or 'date_utc' column")

    # Ensure lat/lon columns exist (may be obs_lat/obs_lon from MET output)
    if "obs_lat" not in df.columns:
        if "latitude" in df.columns:
            df["obs_lat"] = df["latitude"]
        else:
            raise ValueError("Missing required latitude column (obs_lat or latitude)")
    if "obs_lon" not in df.columns:
        if "longitude" in df.columns:
            df["obs_lon"] = df["longitude"]
        else:
            raise ValueError("Missing required longitude column (obs_lon or longitude)")

    # Convert time and extract date
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    # CRITICAL: Generate deterministic sensor_id if missing (must match MET exactly)
    if "sensor_id" not in df.columns:
        print("[aod-etl] Generating sensor_id from coordinates...")
        df["sensor_id"] = [
            _create_deterministic_sensor_id(lat, lon)
            for lat, lon in zip(df["obs_lat"], df["obs_lon"])
        ]

    print(f"[aod-etl] Scanning for HDF files in {aod_dir}...")
    hdf_map = find_hdf_files(Path(aod_dir))

    if df.empty:
        raise ValueError("Input DataFrame has no valid rows")

    dates = sorted(df["date"].unique())
    date_range = f"{dates[0]} to {dates[-1]}"
    print(f"[aod-etl] Processing {len(df):,} observations across {len(dates)} days ({date_range})")
    print(f"[aod-etl] Available HDF files for {len(hdf_map)} days")

    overlap_days = len([d for d in dates if d in hdf_map])
    print(f"[aod-etl] Days with both observation and HDF data: {overlap_days}/{len(dates)} ({overlap_days/len(dates)*100:.1f}%)")

    all_parts = []
    processed_days = 0
    missing_days = 0

    for i, date_key in enumerate(dates):
        hdf_files = hdf_map.get(date_key, [])
        date_str = date_key.strftime('%Y-%m-%d')

        if not hdf_files:
            missing_days += 1

        sub = df[df["date"] == date_key].copy()

        # Show progress every 30 days
        if i < 5 or i % 30 == 0:
            n_files = len(hdf_files) if hdf_files else 0
            print(f"[aod-etl] {date_str}: {len(sub):,} obs, {n_files} HDF files")

        processed = process_date_group_enhanced(sub, hdf_files, chunk=chunk)

        # Remove temporary date column
        processed = processed.drop(columns=["date"], errors="ignore")
        all_parts.append(processed)
        processed_days += 1

    result = pd.concat(all_parts, ignore_index=True) if all_parts else df.copy()

    print(f"[aod-etl] Done. Days processed: {processed_days} | Days missing HDF: {missing_days}")
    print(f"[aod-etl] Final: {len(result):,} rows, {len(result.columns)} columns")

    return result


def run_enhanced(input_csv: Path, aod_base_dir: Path, output_csv: Path, chunk=1000, overwrite=False) -> Path:
    """Main processing function for enhanced features"""

    print(f"Loading enhanced feature data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Convert time column to datetime and extract date
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    
    print(f"Scanning for HDF files in {aod_base_dir}...")
    hdf_map = find_hdf_files(aod_base_dir)
    
    if df.empty:
        raise SystemExit("Input CSV has no valid rows after parsing.")
    
    dates = sorted(df["date"].unique())
    date_range = f"{dates[0]} to {dates[-1]}"
    print(f"\\nFound {len(df):,} feature rows covering {len(dates)} days ({date_range})")
    print(f"Available HDF files for {len(hdf_map)} days")
    
    # Check data coverage
    sensor_years = set(d.year for d in dates)
    hdf_years = set(info['year'] for files in hdf_map.values() for info in files)
    print(f"Feature data years: {sorted(sensor_years)}")
    print(f"HDF data years: {sorted(hdf_years)}")
    
    overlap_days = len([d for d in dates if d in hdf_map])
    print(f"Days with both feature and HDF data: {overlap_days}/{len(dates)} ({overlap_days/len(dates)*100:.1f}%)")
    
    output_csv = Path(output_csv)
    if output_csv.exists() and overwrite:
        output_csv.unlink()
    
    wrote_header = False
    processed_days = 0
    missing_days = 0
    total_sensors_processed = 0
    
    print(f"\\nProcessing {len(dates)} days...")
    
    for i, date_key in enumerate(dates):
        hdf_files = hdf_map.get(date_key, [])
        date_str = date_key.strftime('%Y-%m-%d')
        
        if not hdf_files:
            if missing_days < 5:  # Only show first few missing days to avoid spam
                print(f"MISSING {date_str}: No HDF files found")
            missing_days += 1
            # Still process but with NaN AOD values
        
        sub = df[df["date"] == date_key].copy()
        
        # Show progress every 10 days or for first few days
        if i < 10 or i % 10 == 0:
            if hdf_files:
                print(f"PROCESSING {date_str}: {len(sub):,} sensors, {len(hdf_files)} HDF files")
            else:
                print(f"PROCESSING {date_str}: {len(sub):,} sensors, NO HDF files (AOD=NaN)")
        
        processed = process_date_group_enhanced(sub, hdf_files, chunk=chunk)
        
        # Remove temporary date column
        processed = processed.drop(columns=["date"])
        
        # Save to CSV
        processed.to_csv(output_csv, index=False, 
                        mode=("w" if not wrote_header else "a"),
                        header=not wrote_header)
        
        wrote_header = True
        processed_days += 1
        total_sensors_processed += len(processed)
        
        # Show progress update every 50 days
        if processed_days % 50 == 0:
            print(f"  Progress: {processed_days}/{len(dates)} days processed ({total_sensors_processed:,} feature records)")
    
    if missing_days > 5:
        print(f"... and {missing_days-5} more missing days")
    
    print(f"\\nDone → {output_csv}")
    print(f"Days processed: {processed_days} | Days with missing AOD: {missing_days}")
    print(f"Total feature records with AOD enhancement: {total_sensors_processed:,}")
    
    return output_csv


def parse_args():
    ap = argparse.ArgumentParser(
        description="Add AOD features to existing PAQI feature dataset"
    )
    ap.add_argument("--csv_file", default="output/paqi_with_all_features.csv", type=Path,
                    help="CSV file to add AOD features to (input and output)")
    ap.add_argument("--aod_dir", default="datasets/MCD19A2.061", type=Path,
                    help="Base directory containing year subdirectories with MCD19A2 HDF files")
    ap.add_argument("--chunk", type=int, default=1000,
                    help="Processing chunk size")
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Check if CSV already has AOD features
    skip_processing = False
    if args.csv_file.exists():
        try:
            df_check = pd.read_csv(args.csv_file, nrows=1)
            # Check for key AOD features that indicate processing is complete
            key_aod_features = ["optical_depth_055", "optical_depth_047", "aod_uncertainty", "qa_cloudmask"]
            existing_aod_features = [col for col in key_aod_features if col in df_check.columns]
            
            if len(existing_aod_features) >= 2:  # At least 2 key features present
                print(f"[aod_pipeline] Output file {args.csv_file} already exists with AOD features - skipping processing")
                print(f"[aod_pipeline] Found AOD features: {existing_aod_features}")
                skip_processing = True
        except Exception as e:
            print(f"[aod_pipeline] Warning: Could not read existing file {args.csv_file}: {e}")
    
    args.csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not skip_processing:
        tmp_full = Path(args.csv_file).with_suffix(".aod_full.csv")
        
        output_data = run_enhanced(args.csv_file, args.aod_dir, str(tmp_full), 
                                  args.chunk, overwrite=True)
        
        df_full = pd.read_csv(tmp_full)
        register_stage_features("aod", df_full,
                               feature_columns=[c for c in AOD_FEATURES if c in df_full.columns],
                               metadata_columns=[c for c in AOD_METADATA if c in df_full.columns])
        
        tmp_full.unlink()
        print(f"[aod_pipeline] Registry updated for {len(df_full):,} rows, {len(df_full.columns)} cols")
        
    else:
        if args.csv_file.exists():
            usecols = ["sensor_id","time"] + AOD_FEATURES + AOD_METADATA
            df_reg = pd.read_csv(args.csv_file, usecols=[c for c in usecols if c in pd.read_csv(args.csv_file, nrows=1).columns])
            register_stage_features("aod", df_reg,
                                   feature_columns=[c for c in AOD_FEATURES if c in df_reg.columns],
                                   metadata_columns=[c for c in AOD_METADATA if c in df_reg.columns])
            print(f"[aod_pipeline] Registry updated for {len(df_reg):,} rows, {len(df_reg.columns)} cols")




if __name__ == "__main__":
    main()