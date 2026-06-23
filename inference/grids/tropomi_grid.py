# tropomi_grid.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import rasterio
import time
import traceback
from rasterio.warp import reproject, Resampling
from scipy.ndimage import convolve
from numpy.lib.stride_tricks import sliding_window_view


TROPOMI_PRODUCTS = ["no2", "so2", "co", "hcho", "aai", "alh", "ch4", "cloud", "o3", "nh3"]
TROPOMI_MS_FIELDS = ["median", "std", "n_pixels", "window_coverage", "qa_pass_fraction"]
TROPOMI_DELTA_FIELDS = ["delta_median_r1_r05", "delta_median_r2_r1", "delta_std_r2_r05"]


@dataclass
class TropomiGridConfig:
    tropomi_dir: Path
    geos_cf_dir: Path
    grid_shape: tuple
    grid_lats_1d: np.ndarray
    grid_lons_1d: np.ndarray
    grid_resolution_deg: float
    tropomi_window: int


class TropomiGrid:
    """
    Fast + training-faithful grid implementation.

    Key alignments with training tropomi_features.py:
      - Best-scale selection uses ONLY n_pixels (valid count): first >= min_valid_pixels else max count.
      - When QA band is absent (our preprocessed single-band TIFs), qa_pass_fraction is 1.0 per-scale.
        (Best-scale qa_pass_fraction is 1.0 if any data exists, else 0.0 when best_idx is None.)
      - window_coverage denominator matches training edge behavior via truncated-window total_pix
        (computed by convolving an in-bounds mask with the kernel).
      - Expensive stats computed only for the chosen scale per cell (two-pass).
    """

    def __init__(self, cfg: TropomiGridConfig):
        self.cfg = cfg

    # -----------------------------
    # Feature selection helpers
    # -----------------------------
    def _needed_base_stats(self, p: str, wanted: set[str]) -> set[str]:
        BASE_STATS = {
            "median", "mean", "std", "min", "max", "p10", "p25", "p75", "p90",
            "iqr", "mad", "range", "cv", "skew", "kurt"
        }
        return {s for s in BASE_STATS if f"{p}_{s}" in wanted}

    def _needs_delta(self, p: str, wanted: set[str]) -> bool:
        return any(f"{p}_{d}" in wanted for d in TROPOMI_DELTA_FIELDS)

    def _needed_ms_fields(self, p: str, suffix: str, wanted: set[str]) -> set[str]:
        MS_STATS = {"median", "std", "n_pixels", "window_coverage", "qa_pass_fraction"}
        return {m for m in MS_STATS if f"{p}_{m}_{suffix}" in wanted}

    # -----------------------------
    # File discovery
    # -----------------------------
    def _find_file(self, product: str, pred_date: datetime) -> Optional[Path]:
        ds = pd.Timestamp(pred_date).strftime("%Y-%m-%d")
        product = product.lower()

        if product == "nh3":
            candidates = [
                self.cfg.geos_cf_dir / "NH3" / f"geos_cf_nh3_pakistan_{ds}.tif",
                self.cfg.geos_cf_dir / "NH3" / f"nh3_{ds}.tif",
                self.cfg.geos_cf_dir / "NH3" / f"{ds}.tif",
            ]
            for p in candidates:
                if p.exists():
                    return p
            if self.cfg.geos_cf_dir.exists():
                hits = [
                    p for p in self.cfg.geos_cf_dir.rglob("*.tif")
                    if ("nh3" in p.name.lower() and ds in p.name)
                ]
                if hits:
                    return sorted(hits)[0]
            return None

        candidates = [
            self.cfg.tropomi_dir / product / f"{ds}.tif",
            self.cfg.tropomi_dir / product / f"{product}_{ds}.tif",
            self.cfg.tropomi_dir / f"{product}_{ds}.tif",
            self.cfg.tropomi_dir / f"{ds}_{product}.tif",
        ]
        for p in candidates:
            if p.exists():
                return p

        if self.cfg.tropomi_dir.exists():
            hits = [
                p for p in self.cfg.tropomi_dir.rglob("*.tif")
                if (product in p.name.lower() and ds in p.name)
            ]
            if hits:
                return sorted(hits)[0]
        return None

    # -----------------------------
    # Reprojection to inference grid
    # -----------------------------
    def _grid_transform(self) -> rasterio.Affine:
        lats = self.cfg.grid_lats_1d
        lons = self.cfg.grid_lons_1d
        res = float(self.cfg.grid_resolution_deg)
        west = float(lons.min() - res / 2)
        north = float(lats.max() + res / 2)
        return rasterio.transform.from_origin(west, north, res, res)

    def _reproject_to_grid(self, src_path: Path, *, verbose: bool = False, product: str = "") -> np.ndarray:
        dst = np.full(self.cfg.grid_shape, np.nan, dtype="float32")
        dst_transform = self._grid_transform()
        dst_crs = "EPSG:4326"

        with rasterio.open(src_path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band = np.where(band == nodata, np.nan, band)

            tags = src.tags(1)

            def _get_tag_float(keys, default=None):
                for k in keys:
                    if k in tags:
                        try:
                            return float(tags[k])
                        except Exception:
                            pass
                return default

            scale = _get_tag_float(["scale_factor", "SCALE_FACTOR", "ScaleFactor"], 1.0)
            offset = _get_tag_float(["add_offset", "ADD_OFFSET", "AddOffset"], 0.0)

            # Apply scaling if present (usually 1.0, 0.0 for preprocessed TIFs)
            if scale != 1.0 or offset != 0.0:
                if verbose:
                    print(f"[tropomi] {product} applying scale={scale} offset={offset}")
                band = band * np.float32(scale) + np.float32(offset)

            reproject(
                source=band,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs if src.crs is not None else dst_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
        return dst

    # -----------------------------
    # Window utilities (training-faithful borders)
    # -----------------------------
    @staticmethod
    def _flat_windows_nanpad(a2d: np.ndarray, w: int) -> np.ndarray:
        """
        Returns shape (H*W, w*w) with NaN padding beyond edges.
        This matches training's truncated reads (outside raster == missing).
        """
        r = w // 2
        ap = np.pad(a2d, ((r, r), (r, r)), mode="constant", constant_values=np.nan)
        win = sliding_window_view(ap, (w, w))  # (H, W, w, w) view
        flat = win.reshape(a2d.shape[0] * a2d.shape[1], -1)
        # keep float32 throughout
        return flat.astype(np.float32, copy=False)

    @staticmethod
    def _count_and_coverage(finite_mask: np.ndarray, kernel: np.ndarray, total_pix_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        finite_mask: (H,W) float32 (1.0 for finite, 0.0 for non-finite)
        kernel: (w,w) float32 ones
        total_pix_2d: (H,W) float32 = convolve(ones(H,W), kernel) with constant padding
        """
        cnt2d = convolve(finite_mask, kernel, mode="constant", cval=0.0).astype(np.float32)
        cov2d = np.where(total_pix_2d > 0, cnt2d / total_pix_2d, 0.0).astype(np.float32)
        return cnt2d, cov2d

    @staticmethod
    def _summarize_rows(vals: np.ndarray, need: set[str]) -> Dict[str, np.ndarray]:
        """
        vals: (m, k) float32 with NaNs
        returns dict stat -> (m,) float32
        Implements training semantics:
          - std is population (ddof=0)
          - cv = std/(abs(mean)+1e-12)
          - skew nan if <3 valid; kurt nan if <4 valid; if std<=1e-12 => 0.0
          - percentiles computed on valid values (nan ignored)
          - mad uses median(|x - median|)
        """
        need = set(need)
        out: Dict[str, np.ndarray] = {}

        with np.errstate(all="ignore"):
            mask = np.isfinite(vals)
            cnt = mask.sum(axis=1).astype(np.int32)

            # Mean/std
            mean = std = None
            if any(k in need for k in {"mean", "std", "cv", "skew", "kurt"}):
                mean = np.nanmean(vals, axis=1).astype(np.float32)
                std = np.nanstd(vals, axis=1, ddof=0).astype(np.float32)
                if "mean" in need:
                    out["mean"] = mean
                if "std" in need:
                    out["std"] = std

            # Median
            median = None
            if "median" in need or "mad" in need:
                median = np.nanmedian(vals, axis=1).astype(np.float32)
                if "median" in need:
                    out["median"] = median

            # Min/max
            if "min" in need:
                out["min"] = np.nanmin(vals, axis=1).astype(np.float32)
            if "max" in need:
                out["max"] = np.nanmax(vals, axis=1).astype(np.float32)

            # Percentiles
            # Compute only those requested
            def _pct(pct: int) -> np.ndarray:
                return np.nanpercentile(vals, pct, axis=1).astype(np.float32)

            p10 = p25 = p75 = p90 = None
            if "p10" in need:
                p10 = _pct(10)
                out["p10"] = p10
            if "p25" in need or "iqr" in need:
                p25 = _pct(25)
                if "p25" in need:
                    out["p25"] = p25
            if "p75" in need or "iqr" in need:
                p75 = _pct(75)
                if "p75" in need:
                    out["p75"] = p75
            if "p90" in need:
                p90 = _pct(90)
                out["p90"] = p90

            if "iqr" in need:
                if p25 is None:
                    p25 = _pct(25)
                if p75 is None:
                    p75 = _pct(75)
                out["iqr"] = (p75 - p25).astype(np.float32)

            if "mad" in need:
                if median is None:
                    median = np.nanmedian(vals, axis=1).astype(np.float32)
                out["mad"] = np.nanmedian(np.abs(vals - median[:, None]), axis=1).astype(np.float32)

            if "range" in need:
                vmin = out.get("min")
                vmax = out.get("max")
                if vmin is None:
                    vmin = np.nanmin(vals, axis=1).astype(np.float32)
                if vmax is None:
                    vmax = np.nanmax(vals, axis=1).astype(np.float32)
                out["range"] = (vmax - vmin).astype(np.float32)

            if "cv" in need:
                if mean is None:
                    mean = np.nanmean(vals, axis=1).astype(np.float32)
                if std is None:
                    std = np.nanstd(vals, axis=1, ddof=0).astype(np.float32)
                out["cv"] = (std / (np.abs(mean) + 1e-12)).astype(np.float32)

            if "skew" in need or "kurt" in need:
                if mean is None:
                    mean = np.nanmean(vals, axis=1).astype(np.float32)
                if std is None:
                    std = np.nanstd(vals, axis=1, ddof=0).astype(np.float32)
                denom = std + 1e-12
                z = (vals - mean[:, None]) / denom[:, None]

                if "skew" in need:
                    skew = np.nanmean(z ** 3, axis=1).astype(np.float32)
                    # training: if <3 valid => nan; if std<=1e-12 => 0.0
                    skew = np.where(cnt >= 3, skew, np.nan).astype(np.float32)
                    skew = np.where((cnt >= 3) & (std <= 1e-12), 0.0, skew).astype(np.float32)
                    out["skew"] = skew

                if "kurt" in need:
                    kurt = (np.nanmean(z ** 4, axis=1) - 3.0).astype(np.float32)
                    kurt = np.where(cnt >= 4, kurt, np.nan).astype(np.float32)
                    kurt = np.where((cnt >= 4) & (std <= 1e-12), 0.0, kurt).astype(np.float32)
                    out["kurt"] = kurt

        return out

    # -----------------------------
    # Main compute
    # -----------------------------
    def compute(self, pred_date: datetime, verbose: bool = False, feature_cols: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        H, W = self.cfg.grid_shape
        n = H * W
        out: Dict[str, np.ndarray] = {}

        # 1) Match your existing window->km mapping
        base_window = int(self.cfg.tropomi_window)
        if base_window < 1 or base_window % 2 == 0:
            raise ValueError("tropomi_window must be odd and >=1 (e.g., 3,5).")

        res_km = float(self.cfg.grid_resolution_deg) * 111.0
        pix_per_km = 1.0 / res_km

        base_radius_km = (base_window - 1) / 2.0 / pix_per_km
        radii_km = [0.5 * base_radius_km, base_radius_km, 2.0 * base_radius_km]
        scale_suffixes = ["r05", "r1", "r2"]
        rpxs = [max(1, int(round(r * pix_per_km))) for r in radii_km]
        ws = [2 * rpx + 1 for rpx in rpxs]

        if verbose:
            print(f"[tropomi] TROPOMI multiscale setup:")
            print(f"  Base window: {base_window} pixels")
            for r_km, rpx, suffix, w in zip(radii_km, rpxs, scale_suffixes, ws):
                print(f"  {suffix}: r={r_km:.2f}km → {rpx}px → w={w}")
            if len(set(ws)) < 3:
                print(f"  WARNING: Some window sizes are identical. Consider tropomi_window 5 or 7 for distinct multiscale.")

        # 2) Selective feature optimization
        wanted = set(feature_cols) if feature_cols else set()
        use_selective = bool(feature_cols)

        # 3) Precompute total_pix per unique window size (border-faithful)
        ones_hw = np.ones((H, W), dtype=np.float32)
        unique_ws = sorted(set(ws))
        kernels: Dict[int, np.ndarray] = {w: np.ones((w, w), dtype=np.float32) for w in unique_ws}
        total_pix_by_w: Dict[int, np.ndarray] = {
            w: convolve(ones_hw, kernels[w], mode="constant", cval=0.0).astype(np.float32)
            for w in unique_ws
        }

        # Training default
        min_valid_pixels = 5

        for prod in TROPOMI_PRODUCTS:
            p = prod.lower()
            t0p = time.perf_counter()

            # init "main" columns
            for stat in ["median", "mean", "std", "min", "max", "p10", "p25", "p75", "p90", "iqr", "mad", "range", "cv", "skew", "kurt"]:
                out[f"{p}_{stat}"] = np.full(n, np.nan, dtype="float32")

            for qfeat in ["n_pixels", "window_size_used", "window_coverage", "radius_km_used",
                          "file_available", "qa_pass_fraction", "qa_available", "qa_threshold",
                          "scale_factor", "add_offset"]:
                if qfeat in ["n_pixels", "window_size_used", "qa_available", "file_available"]:
                    out[f"{p}_{qfeat}"] = np.zeros(n, dtype="float32")
                else:
                    out[f"{p}_{qfeat}"] = np.full(n, np.nan, dtype="float32")

            for suffix in ["r05", "r1", "r2"]:
                for field in TROPOMI_MS_FIELDS:
                    out[f"{p}_{field}_{suffix}"] = np.full(n, np.nan, dtype="float32")

            for field in TROPOMI_DELTA_FIELDS:
                out[f"{p}_{field}"] = np.full(n, np.nan, dtype="float32")

            # determine needed features for this product (if selective)
            if use_selective:
                need_best = self._needed_base_stats(p, wanted)
                need_delta = self._needs_delta(p, wanted)
                need_ms = {
                    "r05": self._needed_ms_fields(p, "r05", wanted),
                    "r1":  self._needed_ms_fields(p, "r1",  wanted),
                    "r2":  self._needed_ms_fields(p, "r2",  wanted),
                }

                # If model uses nothing from this product, skip heavy work
                if not (need_best or need_delta or need_ms["r05"] or need_ms["r1"] or need_ms["r2"]):
                    if verbose:
                        print(f"[tropomi] Skipping {p} (not needed by model)")
                    continue
            else:
                need_best = {"median", "mean", "std", "min", "max", "p10", "p25", "p75", "p90", "iqr", "mad", "range", "cv", "skew", "kurt"}
                need_delta = True
                need_ms = {"r05": set(TROPOMI_MS_FIELDS), "r1": set(TROPOMI_MS_FIELDS), "r2": set(TROPOMI_MS_FIELDS)}

            path = self._find_file(p, pred_date)
            t_find = time.perf_counter()

            if path is None:
                out[f"{p}_file_available"] = np.zeros(n, dtype="float32")
                continue

            try:
                grid_val = self._reproject_to_grid(path, verbose=verbose, product=p).astype(np.float32)
                t_rep = time.perf_counter()

                if verbose:
                    fin = float(np.isfinite(grid_val).mean())
                    mn = float(np.nanmin(grid_val)) if np.isfinite(grid_val).any() else np.nan
                    mx = float(np.nanmax(grid_val)) if np.isfinite(grid_val).any() else np.nan
                    print(f"[tropomi] {p} finite_frac={fin:.3f} min={mn:.3g} max={mx:.3g}")

                out[f"{p}_file_available"] = np.ones(n, dtype="float32")
                out[f"{p}_scale_factor"] = np.ones(n, dtype="float32")
                out[f"{p}_add_offset"] = np.zeros(n, dtype="float32")

                # Training-faithful: preprocessed single-band => QA band not available
                out[f"{p}_qa_available"] = np.zeros(n, dtype="float32")
                out[f"{p}_qa_threshold"] = np.full(n, 0.75, dtype="float32")

                # Pass A: counts/coverage for ALL scales (cheap), plus optional per-scale median/std (only if needed)
                finite_mask = np.isfinite(grid_val).astype(np.float32)

                # Cache per window size (w) to reuse if duplicate ws exist
                w_cache: Dict[int, Dict[str, np.ndarray]] = {}

                # Decide which per-scale med/std are needed
                # - deltas require median at r05/r1/r2 and std at r05/r2
                # - multiscale outputs may request median/std explicitly
                need_scale_median = {"r05": False, "r1": False, "r2": False}
                need_scale_std = {"r05": False, "r1": False, "r2": False}

                if need_delta:
                    need_scale_median["r05"] = True
                    need_scale_median["r1"] = True
                    need_scale_median["r2"] = True
                    # delta_std_r2_r05 needs std at r05 and r2
                    need_scale_std["r05"] = True
                    need_scale_std["r2"] = True

                for suf in ["r05", "r1", "r2"]:
                    if "median" in need_ms[suf]:
                        need_scale_median[suf] = True
                    if "std" in need_ms[suf]:
                        need_scale_std[suf] = True

                # Compute per-scale arrays
                scale_counts_flat: Dict[str, np.ndarray] = {}
                scale_cov_flat: Dict[str, np.ndarray] = {}
                scale_med_flat: Dict[str, np.ndarray] = {}
                scale_std_flat: Dict[str, np.ndarray] = {}

                for suf, w, rkm in zip(scale_suffixes, ws, radii_km):
                    if w not in w_cache:
                        kernel = kernels[w]
                        total2d = total_pix_by_w[w]
                        cnt2d, cov2d = self._count_and_coverage(finite_mask, kernel, total2d)

                        w_cache[w] = {
                            "cnt2d": cnt2d,
                            "cov2d": cov2d,
                            # lazily filled:
                            "flat": None,
                            "median": None,
                            "std": None,
                        }

                    cnt2d = w_cache[w]["cnt2d"]
                    cov2d = w_cache[w]["cov2d"]

                    cnt_flat = cnt2d.reshape(-1).astype(np.float32, copy=False)
                    cov_flat = cov2d.reshape(-1).astype(np.float32, copy=False)

                    scale_counts_flat[suf] = cnt_flat
                    scale_cov_flat[suf] = cov_flat

                    # Write MS "cheap" fields always (they’re used for selection + often meta)
                    out[f"{p}_n_pixels_{suf}"] = cnt_flat
                    out[f"{p}_window_coverage_{suf}"] = cov_flat

                    # Training semantics: QA absent => qa_pass_fraction = 1.0 per-scale (even if no valid pixels)
                    out[f"{p}_qa_pass_fraction_{suf}"] = np.ones(n, dtype="float32")

                    # Optional per-scale median/std
                    if need_scale_median[suf] or need_scale_std[suf]:
                        if w_cache[w]["flat"] is None:
                            w_cache[w]["flat"] = self._flat_windows_nanpad(grid_val, w)

                        flat = w_cache[w]["flat"]

                        if need_scale_median[suf] and w_cache[w]["median"] is None:
                            with np.errstate(all="ignore"):
                                w_cache[w]["median"] = np.nanmedian(flat, axis=1).astype(np.float32)

                        if need_scale_std[suf] and w_cache[w]["std"] is None:
                            with np.errstate(all="ignore"):
                                w_cache[w]["std"] = np.nanstd(flat, axis=1, ddof=0).astype(np.float32)

                        if need_scale_median[suf]:
                            med = w_cache[w]["median"]
                            scale_med_flat[suf] = med
                            out[f"{p}_median_{suf}"] = med

                        if need_scale_std[suf]:
                            std = w_cache[w]["std"]
                            scale_std_flat[suf] = std
                            out[f"{p}_std_{suf}"] = std

                # Pass A done; choose best scale per cell (training rule)
                n_pix_stack = np.vstack([scale_counts_flat["r05"], scale_counts_flat["r1"], scale_counts_flat["r2"]]).astype(np.float32)
                meets_min = n_pix_stack >= float(min_valid_pixels)
                has_enough = np.any(meets_min, axis=0)
                best_idx_enough = np.argmax(meets_min, axis=0)  # first True (r05->r1->r2)
                best_idx_fallback = np.argmax(n_pix_stack, axis=0)
                best_scale_idx = np.where(has_enough, best_idx_enough, best_idx_fallback).astype(np.int32)

                max_pix = np.max(n_pix_stack, axis=0)
                has_any_data = max_pix > 0

                # Best n_pixels
                n_best = n_pix_stack[best_scale_idx, np.arange(n)]
                out[f"{p}_n_pixels"] = np.where(has_any_data, n_best, 0.0).astype(np.float32)

                # Best window_coverage from chosen scale
                cov_stack = np.vstack([scale_cov_flat["r05"], scale_cov_flat["r1"], scale_cov_flat["r2"]]).astype(np.float32)
                cov_best = cov_stack[best_scale_idx, np.arange(n)]
                out[f"{p}_window_coverage"] = np.where(has_any_data, cov_best, 0.0).astype(np.float32)

                # Best QA pass fraction:
                # training: per-scale is 1.0 when QA absent; best is 1.0 if best_idx exists else 0.0
                out[f"{p}_qa_pass_fraction"] = np.where(has_any_data, 1.0, 0.0).astype(np.float32)

                # Best window_size_used + radius_km_used
                window_sizes = np.array(ws, dtype=np.float32)  # [w_r05, w_r1, w_r2]
                radii_arr = np.array(radii_km, dtype=np.float32)

                w_best = window_sizes[best_scale_idx]
                r_best = radii_arr[best_scale_idx]
                out[f"{p}_window_size_used"] = np.where(has_any_data, w_best, 0.0).astype(np.float32)
                out[f"{p}_radius_km_used"] = np.where(has_any_data, r_best, np.nan).astype(np.float32)

                # Pass B: compute expensive "best-scale" stats only for the chosen scale per cell
                # (This is the big speed win vs computing all stats for all scales.)
                if need_best:
                    # allocate outputs (already allocated in out dict)
                    # compute per-scale subsets
                    for idx_scale, (suf, w) in enumerate(zip(scale_suffixes, ws)):
                        sel = has_any_data & (best_scale_idx == idx_scale)
                        if not np.any(sel):
                            continue
                        sel_idx = np.where(sel)[0]

                        if w not in w_cache or w_cache[w].get("flat") is None:
                            # should exist, but keep safe
                            if w not in w_cache:
                                w_cache[w] = {"flat": None}
                            w_cache[w]["flat"] = self._flat_windows_nanpad(grid_val, w)

                        flat = w_cache[w]["flat"]
                        vals = flat[sel_idx, :]  # (m, w*w)

                        stats = self._summarize_rows(vals, need_best)

                        # Write stats into the correct global positions
                        for k, arr in stats.items():
                            out[f"{p}_{k}"][sel_idx] = arr.astype(np.float32, copy=False)

                # Optional debug timing
                if verbose:
                    print(
                        f"[tropomi] {p} find={t_find - t0p:.3f}s "
                        f"reproject={t_rep - t_find:.3f}s total={time.perf_counter() - t0p:.3f}s"
                    )

            except Exception as e:
                print(f"[tropomi] FAILED product={p} date={pd.Timestamp(pred_date).date()} path={path} err={repr(e)}")
                traceback.print_exc(limit=2)
                out[f"{p}_file_available"] = np.zeros(n, dtype="float32")
                continue

        # Delta features AFTER all processing (safe; only meaningful if med/std scales exist)
        for prod in TROPOMI_PRODUCTS:
            p = prod.lower()
            r05_m = out.get(f"{p}_median_r05", np.full(n, np.nan, dtype="float32"))
            r1_m = out.get(f"{p}_median_r1", np.full(n, np.nan, dtype="float32"))
            r2_m = out.get(f"{p}_median_r2", np.full(n, np.nan, dtype="float32"))
            r05_s = out.get(f"{p}_std_r05", np.full(n, np.nan, dtype="float32"))
            r2_s = out.get(f"{p}_std_r2", np.full(n, np.nan, dtype="float32"))

            out[f"{p}_delta_median_r1_r05"] = (r1_m - r05_m).astype("float32")
            out[f"{p}_delta_median_r2_r1"] = (r2_m - r1_m).astype("float32")
            out[f"{p}_delta_std_r2_r05"] = (r2_s - r05_s).astype("float32")

        return out
