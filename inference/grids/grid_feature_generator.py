# grid_feature_generator.py
from __future__ import annotations

import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import xarray as xr

# Ensure repo root is importable when running from inference/
_REPO_ROOT = Path(__file__).resolve().parents[2]  # inference/grids/../.. = repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

@contextmanager
def timed(name: str, timings: dict | None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + dt

from .met_grid import MetGrid, MetGridConfig
from .aod_grid import AODGrid, AODGridConfig, AOD_OUT_COLS
from .tropomi_grid import TropomiGrid, TropomiGridConfig, TROPOMI_PRODUCTS

warnings.filterwarnings("ignore")

# Optional FeatureRegistry integration
_THIS = Path(__file__).resolve()
_MFE = _THIS.parent.parent.parent / "feature_engineering"
if _MFE.exists():
    sys.path.append(str(_MFE))

try:
    from feature_registry import FeatureRegistry
    REGISTRY_AVAILABLE = True
except Exception:
    REGISTRY_AVAILABLE = False

# Import shared stage mapping
from shared_features.specs.stage_map import split_by_stage


@dataclass
class GridSpec:
    lats_1d: np.ndarray
    lons_1d: np.ndarray
    lat2d: np.ndarray
    lon2d: np.ndarray
    shape: tuple
    resolution_deg: float


class PakistanGridFeatureGenerator:
    def __init__(
        self,
        master_feature_dir: str = "inference",
        grid_resolution: float = 0.1,
        pakistan_bounds: Optional[Dict[str, float]] = None,
        aod_dir: Optional[str] = None,
        tropomi_dir: Optional[str] = None,
        geos_cf_dir: Optional[str] = None,
        tropomi_window: int = 3,
        use_feature_registry: bool = True,
    ):
        self.master_dir = Path(master_feature_dir)
        self.datasets_dir = self.master_dir / "datasets"
        self.grid_resolution = float(grid_resolution)

        self.pakistan_bounds = pakistan_bounds or {
            "lat_min": 23.3, "lat_max": 37.3,
            "lon_min": 60.5, "lon_max": 77.9,
        }

        self.features_met_dir = self.datasets_dir / "features_met"

        self.aod_dir = Path(aod_dir) if aod_dir else (self.datasets_dir / "MCD19A2.061")
        self.tropomi_dir = Path(tropomi_dir) if tropomi_dir else (self.datasets_dir / "tropomi")
        self.geos_cf_dir = Path(geos_cf_dir) if geos_cf_dir else (self.datasets_dir / "geos_cf_pakistan_2020_2025")
        self.tropomi_window = int(tropomi_window)

        self.grid = self._init_grid()

        self.grid_lats = self.grid.lat2d.ravel().astype("float32")
        self.grid_lons = self.grid.lon2d.ravel().astype("float32")

        # Expose for compatibility
        self.lat_grid = self.grid.lat2d
        self.lon_grid = self.grid.lon2d

        # modules
        self.met = MetGrid(MetGridConfig(
            features_met_dir=self.features_met_dir,
            grid_lats_1d=self.grid.lats_1d,
            grid_lons_1d=self.grid.lons_1d,
        ))
        self.aod = AODGrid(AODGridConfig(
            datasets_dir=self.datasets_dir,
            aod_dir=self.aod_dir,
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
        ), verbose=False)  # Will be overridden in compute call
        self.trop = TropomiGrid(TropomiGridConfig(
            tropomi_dir=self.tropomi_dir,
            geos_cf_dir=self.geos_cf_dir,
            grid_shape=self.grid.shape,
            grid_lats_1d=self.grid.lats_1d,
            grid_lons_1d=self.grid.lons_1d,
            grid_resolution_deg=self.grid.resolution_deg,
            tropomi_window=self.tropomi_window,
        ))

        # registry
        self.use_feature_registry = bool(use_feature_registry and REGISTRY_AVAILABLE)
        self._training_feature_list: Optional[List[str]] = None
        self.registry_path = None
        if self.use_feature_registry:
            self._load_feature_registry()

        print(f"[grid] master_dir   : {self.master_dir}")
        print(f"[grid] datasets_dir : {self.datasets_dir}")
        print(f"[grid] met_dir      : {self.features_met_dir}")
        print(f"[grid] aod_dir      : {self.aod_dir}")
        print(f"[grid] geos_cf_dir  : {self.geos_cf_dir}")
        print(f"[grid] tropomi_dir  : {self.tropomi_dir}")
        print(f"[grid] Using met-file grid: {self.grid.shape[0]}x{self.grid.shape[1]} (res≈{self.grid.resolution_deg:.4f}°)")

    def _init_grid(self) -> GridSpec:
        met_files = sorted(self.features_met_dir.glob("features_*.nc"))
        if met_files:
            try:
                with xr.open_dataset(met_files[0]) as ds0:
                    lats = ds0["latitude"].values
                    lons = ds0["longitude"].values
                lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
                res_lat = float(np.nanmedian(np.abs(np.diff(lats)))) if len(lats) > 1 else self.grid_resolution
                res_lon = float(np.nanmedian(np.abs(np.diff(lons)))) if len(lons) > 1 else self.grid_resolution
                res = float(np.nanmedian([res_lat, res_lon]))
                return GridSpec(
                    lats_1d=lats, lons_1d=lons,
                    lat2d=lat2d, lon2d=lon2d,
                    shape=lat2d.shape, resolution_deg=res,
                )
            except Exception:
                pass

        b = self.pakistan_bounds
        # Use linspace for more precise boundary alignment
        # Calculate number of grid points to ensure exact boundary coverage
        lat_range = b["lat_max"] - b["lat_min"]
        lon_range = b["lon_max"] - b["lon_min"]
        
        n_lat = int(np.round(lat_range / self.grid_resolution)) + 1
        n_lon = int(np.round(lon_range / self.grid_resolution)) + 1
        
        # Use linspace for exact boundary alignment
        lats = np.linspace(b["lat_max"], b["lat_min"], n_lat, dtype=np.float64)
        lons = np.linspace(b["lon_min"], b["lon_max"], n_lon, dtype=np.float64)
        
        # Ensure consistent ordering (north-to-south for lats, west-to-east for lons)
        if lats[0] < lats[-1]:
            lats = lats[::-1]
        if lons[0] > lons[-1]:
            lons = lons[::-1]
            
        lon2d, lat2d = np.meshgrid(lons, lats, indexing='xy')
        
        # Calculate actual resolution for verification
        actual_lat_res = np.abs(lats[1] - lats[0]) if len(lats) > 1 else self.grid_resolution
        actual_lon_res = np.abs(lons[1] - lons[0]) if len(lons) > 1 else self.grid_resolution
        actual_res = np.mean([actual_lat_res, actual_lon_res])
        
        return GridSpec(
            lats_1d=lats, lons_1d=lons,
            lat2d=lat2d, lon2d=lon2d,
            shape=lat2d.shape, resolution_deg=actual_res,
        )

    def _load_feature_registry(self):
        try:
            registry_paths = [
                self.master_dir / "data" / "features_registry.csv",
                self.master_dir.parent / "training" / "data" / "features_registry.csv",
                self.master_dir.parent / "feature_engineering" / "output" / "features_registry.csv",
            ]
            self.registry_path = next((p for p in registry_paths if p.exists()), None)
            if self.registry_path is None:
                self.use_feature_registry = False
                return

            print(f"[grid] Loading feature registry from: {self.registry_path}")
            reg = FeatureRegistry()
            reg.registry_path = self.registry_path
            reg.registry = pd.read_csv(self.registry_path)
            self._training_feature_list = list(reg.get_feature_columns())
            print(f"[grid] Found {len(self._training_feature_list)} training features in registry")
        except Exception:
            self.use_feature_registry = False

    def generate_daily_features(
        self,
        pred_date: datetime,
        history_days: int = 21,
        required_features: Optional[List[str]] = None,
        timings: dict | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        pred_day = pd.Timestamp(pred_date).floor("D")
        n = self.grid_lats.size

        print(f"[features] Generating TRUE features for {pred_day.strftime('%Y-%m-%d')} (history_days={history_days})")

        # Replace stage routing with shared split_by_stage
        if required_features:
            stage_feats = split_by_stage(required_features, registry_csv=self.registry_path)
            met_feats = stage_feats.get("met", [])
            aod_feats = stage_feats.get("aod", [])
            trop_feats = stage_feats.get("tropomi", [])
            static_feats = stage_feats.get("static", [])
            
            if verbose:
                print(f"[grid] Stage breakdown: met={len(met_feats)}, aod={len(aod_feats)}, tropomi={len(trop_feats)}, static={len(static_feats)}")
        else:
            # No filtering - generate all
            met_feats = aod_feats = trop_feats = static_feats = None

        # MET always needed (also gives __inpoly)
        with timed("met.compute", timings):
            met = self.met.compute(pred_date, history_days=history_days)

        # Extract Pakistan mask for AOD optimization
        pakistan_mask = met.get("__inpoly", None)

        # No static features (elevation and coast removed)
        elev = {}
        coast = {}

        # AOD features
        if aod_feats is None or aod_feats:
            with timed("aod.compute", timings):
                aod = self.aod.compute(pred_date, pakistan_mask=pakistan_mask, verbose=verbose)
        else:
            aod = {}

        # TROPOMI features
        if trop_feats is None or trop_feats:
            with timed("tropomi.compute", timings):
                trop = self.trop.compute(pred_date, verbose=verbose, feature_cols=trop_feats)
        else:
            trop = {}

        with timed("df.assemble", timings):
            all_cols: Dict[str, np.ndarray] = {}
            all_cols.update(met)
            all_cols.update(elev)
            all_cols.update(coast)
            all_cols.update(aod)
            all_cols.update(trop)

            all_cols["grid_lat"] = self.grid_lats
            all_cols["grid_lon"] = self.grid_lons
            all_cols["date"] = np.full(n, pred_day.to_datetime64())

            df = pd.DataFrame(all_cols)

        with timed("df.reindex", timings):
            # If registry is available but required_features is NOT provided, filter to registry features.
            if required_features is None and self.use_feature_registry and self._training_feature_list:
                coord_cols = ["grid_lat", "grid_lon", "date"]
                available = [c for c in self._training_feature_list if c in df.columns]
                df = df[coord_cols + available].copy()

        return df