"""
GEOS-CF CTM Baseline
====================

Independent chemical transport model baseline for PM2.5 evaluation.
"""

import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from training.utils.metrics import evaluate_predictions


class GEOSCFBaseline:
    """
    GEOS-CF Chemical Transport Model baseline for PM2.5 evaluation.

    GEOS-CF provides near-real-time analysis + 5-day forecasts at 0.25 deg.
    This is an independent CTM comparator (not tuned to station network).

    Data source: NASA GEOS-CF Air Quality Concentrations Diagnostics
    - Variable: PM25_RH35_GCC (PM2.5 at 35% RH)
    - Resolution: 0.25 deg (~25 km)

    Usage
    -----
    >>> geos = GEOSCFBaseline(cache_dir="data/geos_cf")
    >>> baseline_pred = geos.get_station_predictions(station_df)
    >>> metrics = geos.evaluate(y_obs, baseline_pred)
    """

    RESOLUTION = 0.25  # degrees
    GCS_PATH_PATTERN = "gs://paqi-raw-hawanama-data/raw/geos_cf_pakistan_2020_2025/PM25/geos_cf_pm25_pakistan_{date}.tif"

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/geos_cf",
        use_gcs: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_gcs = use_gcs
        self._raster_cache: Dict[str, Tuple[np.ndarray, dict]] = {}

    def _get_gcs_path(self, date: dt.date) -> str:
        return self.GCS_PATH_PATTERN.format(date=date.strftime("%Y-%m-%d"))

    def _get_local_path(self, date: dt.date) -> Path:
        return self.cache_dir / f"geos_cf_pm25_{date.strftime('%Y%m%d')}.tif"

    def _download_from_gcs(self, date: dt.date) -> Optional[Path]:
        import subprocess

        gcs_path = self._get_gcs_path(date)
        local_path = self._get_local_path(date)

        if local_path.exists():
            return local_path

        try:
            result = subprocess.run(
                ["gsutil", "-q", "cp", gcs_path, str(local_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return local_path
            else:
                return None
        except Exception:
            return None

    def _load_raster(self, date: dt.date) -> Optional[Tuple[np.ndarray, dict]]:
        import rasterio

        date_str = date.strftime("%Y%m%d")

        if date_str in self._raster_cache:
            return self._raster_cache[date_str]

        local_path = self._get_local_path(date)

        if not local_path.exists() and self.use_gcs:
            local_path = self._download_from_gcs(date)

        if local_path is None or not local_path.exists():
            return None

        try:
            with rasterio.open(local_path) as src:
                data = src.read(1)
                meta = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "bounds": src.bounds,
                    "width": src.width,
                    "height": src.height,
                }

            self._raster_cache[date_str] = (data, meta)
            return (data, meta)

        except Exception:
            return None

    def get_station_value(
        self,
        lat: float,
        lon: float,
        date: Union[str, dt.date],
        method: str = "nearest",
    ) -> float:
        """Get GEOS-CF PM2.5 value at a station location for a specific date."""
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()

        raster_data = self._load_raster(date)
        if raster_data is None:
            return np.nan

        data, meta = raster_data
        transform = meta["transform"]
        bounds = meta["bounds"]

        if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
            return np.nan

        if method == "nearest":
            col = int((lon - transform.c) / transform.a)
            row = int((lat - transform.f) / transform.e)

            if 0 <= row < meta["height"] and 0 <= col < meta["width"]:
                return float(data[row, col])
            return np.nan

        elif method == "bilinear":
            col_f = (lon - transform.c) / transform.a
            row_f = (lat - transform.f) / transform.e

            col0, row0 = int(col_f), int(row_f)
            col1, row1 = col0 + 1, row0 + 1

            if not (0 <= row0 < meta["height"] - 1 and 0 <= col0 < meta["width"] - 1):
                return self.get_station_value(lat, lon, date, method="nearest")

            t = row_f - row0
            u = col_f - col0

            v00 = data[row0, col0]
            v01 = data[row0, col1]
            v10 = data[row1, col0]
            v11 = data[row1, col1]

            return float((1-t)*(1-u)*v00 + (1-t)*u*v01 + t*(1-u)*v10 + t*u*v11)

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_station_predictions(
        self,
        station_df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        date_col: str = "date",
        method: str = "nearest",
    ) -> np.ndarray:
        """Get GEOS-CF predictions for all station-days in a DataFrame."""
        predictions = np.full(len(station_df), np.nan)

        for i, (_, row) in enumerate(station_df.iterrows()):
            predictions[i] = self.get_station_value(
                row[lat_col], row[lon_col], row[date_col], method=method
            )

        return predictions

    def evaluate(
        self,
        y_obs: np.ndarray,
        y_geoscf: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """Evaluate GEOS-CF baseline against observations."""
        return evaluate_predictions(y_obs, y_geoscf, verbose=verbose)

    def sync_from_gcs(
        self,
        start_date: str,
        end_date: str,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """Download GEOS-CF data from GCS to local cache for a date range."""
        dates = pd.date_range(start_date, end_date, freq="D")
        counts = {"downloaded": 0, "already_cached": 0, "not_found": 0}

        for date in dates:
            local_path = self._get_local_path(date.date())

            if local_path.exists():
                counts["already_cached"] += 1
                continue

            result = self._download_from_gcs(date.date())
            if result is not None:
                counts["downloaded"] += 1
                if verbose:
                    print(f"  Downloaded: {date.strftime('%Y-%m-%d')}")
            else:
                counts["not_found"] += 1
                if verbose:
                    print(f"  Not found:  {date.strftime('%Y-%m-%d')}")

        if verbose:
            print(f"\nGEOS-CF sync complete: {counts['downloaded']} downloaded, "
                  f"{counts['already_cached']} cached, {counts['not_found']} not found")

        return counts
