# aod_grid.py
"""
Optimized AOD grid processing with major performance improvements:

1. Only compute AOD for Pakistan mask pixels (huge speedup)
2. Dtype optimization: float32 for numeric, uint16 for QA
3. Cached coordinate mapping (no recomputation of lat/lon -> (h,v,row,col))
4. Reduced window extraction work with early stopping
5. Vectorized operations where possible

"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date as date_cls
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC


AOD_SDS_NUM = ["Optical_Depth_047", "Optical_Depth_055", "AOD_Uncertainty"]
AOD_SDS_QA = "AOD_QA"

AOD_OUT_COLS = [
    "optical_depth_047",
    "optical_depth_055", 
    "aod_uncertainty",
    "qa_cloudmask",
    "qa_adjacency",
    "qa_aod",
    "qa_n_pixels",
    "aod_total_valid_pixels",
    "aod_files_used",
    "aod_window_size_used",
]

AOD_FILE_RE = re.compile(r"MCD19A2\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.061\..*\.hdf$")

# QA invalid sentinel value (constant across all files)
_QA_INVALID_SENTINEL = np.uint16(65535)


def _stable_digest(obj: dict, digest_size: int = 8) -> str:
    """Create deterministic hash from dict (stable across Python runs)"""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def _stable_array_digest(a: np.ndarray, digest_size: int = 8) -> str:
    """Create deterministic hash from array (stable across Python runs)"""
    # no-copy view into bytes
    mv = memoryview(np.ascontiguousarray(a)).cast("B")
    return hashlib.blake2b(mv, digest_size=digest_size).hexdigest()


def _mode_int(x: np.ndarray, max_value: int) -> float:
    if x.size == 0:
        return np.nan
    x = x.astype(np.int64)
    x = x[(x >= 0) & (x <= max_value)]
    if x.size == 0:
        return np.nan
    counts = np.bincount(x, minlength=max_value + 1)
    return float(np.argmax(counts))


def decode_aod_qa_bitfield(valid_qa_values_1d: np.ndarray) -> dict:
    v = valid_qa_values_1d.astype(np.uint16)
    cloudmask = (v >> 0) & 0b111
    adjacency = (v >> 5) & 0b111
    qa_aod = (v >> 8) & 0b1111
    return {
        "qa_cloudmask": _mode_int(cloudmask, max_value=7),
        "qa_adjacency": _mode_int(adjacency, max_value=7), 
        "qa_aod": _mode_int(qa_aod, max_value=15),
        "qa_n_pixels": float(v.size),
    }


def latlon_to_modis_tile_pixel(lat, lon):
    """Convert lat/lon to MODIS tile coordinates. Same as original."""
    R = 6371007.181
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = R * lon_rad * np.cos(lat_rad)
    y = R * lat_rad

    xmin = -20015109.353729
    ymax = 10007554.676865

    tile_size = 1111950.5197665554
    pixel_size = tile_size / 1200.0

    h = int((x - xmin) / tile_size)
    v = int((ymax - y) / tile_size)

    x_in = (x - xmin) - h * tile_size
    y_in = (ymax - y) - v * tile_size

    col = int(x_in / pixel_size)
    row = int(y_in / pixel_size)

    if not (0 <= h <= 35 and 0 <= v <= 17 and 0 <= row < 1200 and 0 <= col < 1200):
        return None

    return h, v, row, col


@dataclass
class AODGridConfig:
    datasets_dir: Path
    aod_dir: Path
    grid_lats: np.ndarray
    grid_lons: np.ndarray


@dataclass
class CachedCoordMapping:
    """Cached coordinate mapping to avoid recomputation"""
    mask_indices: np.ndarray  # Indices where inpoly==1
    tile_to_points: Dict[Tuple[int, int], List[Tuple[int, int, int]]]  # (h,v) -> [(mask_idx, r, c)]
    # Cache validation metadata
    n_total: int
    grid_shape: Tuple[int, ...]
    lat_bounds: Tuple[float, float]  # (min, max)
    lon_bounds: Tuple[float, float]  # (min, max)
    mask_hash: str  # Stable hash of mask for validation


class AODGridOptimized:
    """
    High-performance AOD grid processor with major optimizations:
    1. Pakistan mask filtering (only compute where needed)
    2. Cached coordinate mapping
    3. Dtype optimization (float32 for data, uint16 for QA)
    4. Early window size stopping
    """
    
    def __init__(self, cfg: AODGridConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self._aod_index = None
        self._aod_index_loaded = False
        self._coord_cache = None
        
    def _load_or_build_aod_index(self):
        """Load or build AOD file index. Same as original."""
        if self._aod_index_loaded:
            return

        cache_path = self.cfg.datasets_dir / "aod_file_index.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self._aod_index = pickle.load(f)
                self._aod_index_loaded = True
                return
            except Exception:
                pass

        file_map: Dict[date_cls, List[dict]] = {}
        if not self.cfg.aod_dir.exists():
            self._aod_index = {}
            self._aod_index_loaded = True
            return

        for year_dir in self.cfg.aod_dir.glob("*/"):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for hdf_file in year_dir.glob("*.hdf"):
                m = AOD_FILE_RE.match(hdf_file.name)
                if not m:
                    continue
                y = int(m.group(1))
                doy = int(m.group(2))
                h_tile = int(m.group(3))
                v_tile = int(m.group(4))
                dt = (datetime(y, 1, 1) + timedelta(days=doy - 1)).date()
                file_map.setdefault(dt, []).append({
                    "path": hdf_file,
                    "h_tile": h_tile,
                    "v_tile": v_tile,
                    "date": dt,
                    "year": y,
                })

        self._aod_index = file_map
        self._aod_index_loaded = True

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(file_map, f)
        except Exception:
            pass

    def _build_coord_cache(self, pakistan_mask: Optional[np.ndarray] = None) -> CachedCoordMapping:
        """
        Build cached coordinate mapping for Pakistan mask pixels only.
        This is the KEY optimization - only process pixels where we actually predict.
        """
        n_total = self.cfg.grid_lats.size
        
        # Ensure grid coordinates are 1D (fix issue #3)
        grid_lats_1d = self.cfg.grid_lats.ravel()
        grid_lons_1d = self.cfg.grid_lons.ravel()
        
        # Calculate cache validation metadata
        lat_bounds = (float(grid_lats_1d.min()), float(grid_lats_1d.max()))
        lon_bounds = (float(grid_lons_1d.min()), float(grid_lons_1d.max()))
        
        # If no mask provided, use all points (fallback to original behavior)
        if pakistan_mask is None:
            mask_indices = np.arange(n_total)
            mask_hash = "nomask"  # Special value for no mask
        else:
            # Only process points where mask == 1 (inside Pakistan)
            pakistan_mask_1d = pakistan_mask.ravel()
            mask_indices = np.where(pakistan_mask_1d == 1)[0]
            # Stable hash of mask for validation
            mask_hash = _stable_array_digest(pakistan_mask_1d.astype(np.uint8))
        
        # Create stable cache key based on grid configuration
        cache_id = _stable_digest({
            "n_total": n_total,
            "lat_bounds": lat_bounds,
            "lon_bounds": lon_bounds,
            "mask_hash": mask_hash,
        })
        cache_path = self.cfg.datasets_dir / f"aod_coord_cache_{cache_id}.pkl"
        
        # Try to load from cache with validation
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                    if (isinstance(cached, CachedCoordMapping) and
                        cached.n_total == n_total and
                        cached.lat_bounds == lat_bounds and
                        cached.lon_bounds == lon_bounds and
                        cached.mask_hash == mask_hash):
                        if self.verbose:
                            print(f"[AOD] Using cached coordinate mapping: {len(cached.mask_indices)}/{n_total} grid points")
                        return cached
                    else:
                        if self.verbose:
                            print(f"[AOD] Cache validation failed, rebuilding...")
            except Exception:
                if self.verbose:
                    print(f"[AOD] Cache loading failed, rebuilding...")
            
        if self.verbose:
            print(f"[AOD] Building coordinate cache for {len(mask_indices)}/{n_total} grid points")
        
        tile_to_points = {}
        
        for mask_idx in mask_indices:
            i = int(mask_idx)  # Original grid index
            lat = float(grid_lats_1d[i])  # Use 1D arrays
            lon = float(grid_lons_1d[i])
            m = latlon_to_modis_tile_pixel(lat, lon)
            if m is None:
                continue
                
            h, v, r, c = m
            tile_to_points.setdefault((h, v), []).append((mask_idx, r, c))
        
        coord_cache = CachedCoordMapping(
            mask_indices=mask_indices,
            tile_to_points=tile_to_points,
            n_total=n_total,
            grid_shape=tuple(self.cfg.grid_lats.shape),  # Store original shape
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            mask_hash=mask_hash
        )
        
        # Cache for future use
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(coord_cache, f)
            if self.verbose:
                print(f"[AOD] Cached coordinate mapping to {cache_path}")
        except Exception as e:
            if self.verbose:
                print(f"[AOD] Failed to cache coordinate mapping: {e}")
            
        return coord_cache

    def _read_hdf_sds_arrays_optimized(self, hdf_path: Path):
        """
        Read HDF arrays with dtype optimization:
        - Numeric data as float32 (not float64)
        - QA data as uint16 (not float64)
        """
        hdf = SD(str(hdf_path), SDC.READ)
        out = {}

        for sds_name in AOD_SDS_NUM:
            sds = hdf.select(sds_name)
            # Read as original dtype, then convert to float32 after scaling
            arr = sds.get()
            attrs = sds.attributes()
            scale = float(attrs.get("scale_factor", 1.0))
            offset = float(attrs.get("add_offset", 0.0))
            fill = attrs.get("_FillValue", -32768)

            # Convert to float32 for memory efficiency
            arr = arr.astype(np.float32)
            arr[arr == fill] = np.nan
            arr = arr * scale + offset
            arr[(arr < -0.05) | (arr > 5.0)] = np.nan
            out[sds_name] = arr
            sds.endaccess()

        # Keep QA as integers (uint16) for efficient bitfield operations
        if AOD_SDS_QA in hdf.datasets():
            sds = hdf.select(AOD_SDS_QA)
            qa_raw = sds.get()  # likely int16
            attrs = sds.attributes()
            fill = int(attrs.get("_FillValue", -32768))
            
            # Safe fill-value handling: work in int32 space, then convert to uint16
            qa_i32 = qa_raw.astype(np.int32)
            qa_i32[qa_i32 == fill] = -1  # mark invalid in signed space
            qa_u16 = qa_i32.astype(np.uint16)  # -1 becomes 65535 in uint16
            out[AOD_SDS_QA] = qa_u16
            sds.endaccess()
        else:
            out[AOD_SDS_QA] = None

        hdf.end()
        return out

    def _extract_window_vals_optimized(self, arr: np.ndarray, row: int, col: int, w: int) -> np.ndarray:
        """Optimized window extraction with early bounds checking."""
        r = w // 2
        
        if arr.ndim == 3:
            bands, height, width = arr.shape
        else:
            height, width = arr.shape
            
        # Early bounds check - if center pixel is outside, skip
        if not (0 <= row < height and 0 <= col < width):
            return np.array([], dtype=arr.dtype)
            
        r0 = max(0, row - r)
        r1 = min(height, row + r + 1)
        c0 = max(0, col - r)
        c1 = min(width, col + r + 1)
        
        if r0 >= r1 or c0 >= c1:
            return np.array([], dtype=arr.dtype)
            
        if arr.ndim == 3:
            return arr[:, r0:r1, c0:c1].reshape(-1)
        else:
            return arr[r0:r1, c0:c1].reshape(-1)

    def compute(self, pred_date: datetime, pakistan_mask: Optional[np.ndarray] = None, verbose: bool = None) -> Dict[str, np.ndarray]:
        """
        Compute AOD features with major optimizations.
        
        Args:
            pred_date: Date to process
            pakistan_mask: Optional mask array (1=inside Pakistan, 0=outside)
                          If provided, only processes pixels where mask==1
            verbose: Optional verbose flag, overrides instance setting if provided
        """
        if verbose is None:
            verbose = self.verbose
            
        self._load_or_build_aod_index()
        
        n_total = self.cfg.grid_lats.size
        
        # Initialize outputs for full grid (filled with NaN)
        out = {c: np.full(n_total, np.nan, dtype="float32") for c in AOD_OUT_COLS}
        out["aod_total_valid_pixels"] = np.zeros(n_total, dtype="float32")
        out["aod_files_used"] = np.zeros(n_total, dtype="float32")
        out["qa_n_pixels"] = np.zeros(n_total, dtype="float32")

        dt_key = pd.Timestamp(pred_date).date()
        infos = (self._aod_index or {}).get(dt_key, [])
        if not infos:
            return out

        # Build coordinate cache for masked pixels only
        coord_cache = self._build_coord_cache(pakistan_mask)
        
        # Diagnostic: Check tile coverage vs availability
        if verbose and coord_cache.tile_to_points:
            needed = set(coord_cache.tile_to_points.keys())
            available = set((i["h_tile"], i["v_tile"]) for i in infos)
            missing = needed - available
            print(f"[AOD] needed tiles: {sorted(needed)}")
            print(f"[AOD] available tiles: {sorted(available)}")
            if missing:
                print(f"[AOD] missing tiles: {sorted(missing)}")
            print(f"[AOD] n_files_today: {len(infos)}")
        
        # If no valid pixels in mask, return empty results
        if len(coord_cache.mask_indices) == 0:
            if verbose:
                print(f"[AOD] No valid pixels in Pakistan mask, skipping")
            return out
            
        if verbose:
            print(f"[AOD] Processing {len(coord_cache.mask_indices)} Pakistan pixels for {pd.Timestamp(pred_date).date()}")

        # Group files by tile  
        by_tile_files: Dict[Tuple[int, int], List[Path]] = {}
        for info in infos:
            hv = (info["h_tile"], info["v_tile"])
            by_tile_files.setdefault(hv, []).append(Path(info["path"]))

        # Process each tile that has both files and pixels
        tiles_processed = 0
        for hv, pts in coord_cache.tile_to_points.items():
            files = by_tile_files.get(hv, [])
            if not files:
                continue

            tiles_processed += 1
            if verbose:
                print(f"[AOD] Processing tile h{hv[0]:02d}v{hv[1]:02d}: {len(files)} files, {len(pts)} pixels")

            # Load all files for this tile
            file_arrays = []
            for fp in files:
                try:
                    file_arrays.append(self._read_hdf_sds_arrays_optimized(fp))
                except Exception as e:
                    print(f"[AOD] Failed to read {fp}: {e}")
                    continue

            if not file_arrays:
                continue

            # Process each pixel in this tile
            for (mask_idx, row, col) in pts:
                vals_047, vals_055, vals_unc = [], [], []
                qa_cloud, qa_adj, qa_aod = [], [], []
                qa_npix_total = 0.0
                valid_pix_total = 0.0
                window_sizes_used = []
                files_used = 0

                # Process each file for this pixel
                for fa in file_arrays:
                    # CRITICAL FIX: Match training "window success" criterion
                    # Training considers window successful if ANY numeric layer has valid pixels
                    # (including AOD_Uncertainty alone)
                    best = None
                    for w in (3, 5, 7):
                        v055 = self._extract_window_vals_optimized(fa["Optical_Depth_055"], row, col, w)
                        v047 = self._extract_window_vals_optimized(fa["Optical_Depth_047"], row, col, w)
                        vunc = self._extract_window_vals_optimized(fa["AOD_Uncertainty"], row, col, w)

                        v055_valid = v055[np.isfinite(v055)]
                        v047_valid = v047[np.isfinite(v047)]
                        vunc_valid = vunc[np.isfinite(vunc)]

                        # TRAINING-ALIGNED: success if ANY numeric layer has valid pixels
                        if (v055_valid.size > 0) or (v047_valid.size > 0) or (vunc_valid.size > 0):
                            best = (v047_valid, v055_valid, vunc_valid, w)
                            break
                    
                    if best is None:
                        continue

                    v047, v055, vunc, w = best
                    window_sizes_used.append(float(w))

                    # Aggregate values
                    if v047.size > 0:
                        vals_047.append(float(np.median(v047)))  # Use median, not nanmedian (already filtered)
                        valid_pix_total += float(v047.size)
                    if v055.size > 0:
                        vals_055.append(float(np.median(v055)))
                        valid_pix_total += float(v055.size)
                    if vunc.size > 0:
                        vals_unc.append(float(np.median(vunc)))

                    # Process QA (keep as uint16 for efficiency)
                    qa_arr = fa.get("AOD_QA", None)
                    if qa_arr is not None:
                        q = self._extract_window_vals_optimized(qa_arr, row, col, w)
                        # Filter out only the invalid sentinel (keep 0 as valid QA)
                        qv = q[q != _QA_INVALID_SENTINEL]
                        if qv.size > 0:
                            feats = decode_aod_qa_bitfield(qv)
                            qa_cloud.append(feats["qa_cloudmask"])
                            qa_adj.append(feats["qa_adjacency"])
                            qa_aod.append(feats["qa_aod"])
                            qa_npix_total += feats["qa_n_pixels"]

                    files_used += 1

                if files_used == 0:
                    continue

                # Convert mask_idx back to original grid index for output
                idx = int(mask_idx)

                # Aggregate final values
                if vals_047:
                    out["optical_depth_047"][idx] = float(np.median(vals_047))
                if vals_055:
                    out["optical_depth_055"][idx] = float(np.median(vals_055))
                if vals_unc:
                    out["aod_uncertainty"][idx] = float(np.median(vals_unc))

                def _mode_from_list(lst):
                    if not lst:
                        return np.nan
                    vv = np.array(lst, dtype=float)
                    vv = vv[np.isfinite(vv)].astype(int)
                    if vv.size == 0:
                        return np.nan
                    return float(np.argmax(np.bincount(vv)))

                out["qa_cloudmask"][idx] = _mode_from_list(qa_cloud)
                out["qa_adjacency"][idx] = _mode_from_list(qa_adj)
                out["qa_aod"][idx] = _mode_from_list(qa_aod)
                out["qa_n_pixels"][idx] = float(qa_npix_total)

                out["aod_total_valid_pixels"][idx] = float(valid_pix_total)
                out["aod_files_used"][idx] = float(files_used)
                out["aod_window_size_used"][idx] = float(np.median(window_sizes_used)) if window_sizes_used else np.nan

        if verbose:
            print(f"[AOD] Processed {tiles_processed} tiles for {pd.Timestamp(pred_date).date()}")
            
            # Diagnostic: Check file usage vs actual retrieval success
            if pakistan_mask is not None:
                mask = pakistan_mask.ravel() == 1
                fu = out["aod_files_used"][mask]
                od = out["optical_depth_055"][mask]
                print(f"[AOD] frac files_used>0: {np.mean(fu > 0):.3f}")
                print(f"[AOD] frac OD055 finite: {np.mean(np.isfinite(od)):.3f}")
                if np.mean(fu > 0) > 0.5 and np.mean(np.isfinite(od)) < 0.3:
                    print("[AOD] ^ Files available but low AOD retrieval (clouds/snow/bright surfaces)")
                elif np.mean(fu > 0) < 0.3:
                    print("[AOD] ^ Low file usage indicates missing tile coverage")
        
        return {k: np.asarray(v, dtype="float32") for k, v in out.items()}


# Backwards compatibility alias
AODGrid = AODGridOptimized