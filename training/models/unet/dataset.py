"""
Grid PM2.5 Dataset (v3)
=======================

Fixes from v2:
- Differentiable bilinear sampling: stations stored as continuous normalized
  grid coords in [-1,1]×[-1,1] (PyTorch grid_sample convention) instead of
  integer pixel indices. Eliminates discretization error.

Carried from v2:
- Station-list supervision (no pixel collisions)
- Proper normalization (binary flags untouched, std_floor, 50-date stats)
- Availability mask channels for NaN-aware learning
- Fixed latitude-axis direction
- 0.05° output grid

Each sample is one day:
  X:            [C_total, 141, 175]  — 307 feature channels + 298 availability masks = 605
  border:       [1, 141, 175]        — Pakistan boundary mask at 0.1°
  station_grid: [N_stations, 2]      — (x, y) coords in [-1,1] for grid_sample
  station_pm25: [N_stations]         — observed PM2.5 values
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from training.models.unet.prior import build_pm25_lag1_prior_0p1

# ---------------------------------------------------------------------------
# Feature channel definitions (matches features_registry.csv used_in_training)
# ---------------------------------------------------------------------------

MET_FEATURES = [
    # Base meteorological variables
    "t2m", "d2m", "blh", "sp", "tcc", "msl", "q", "theta",
    "RH", "VPD", "VC", "VCi", "CLR", "Stagnant", "HighRH",
    # Wind (10m and 100m)
    "u10", "v10", "WS10", "WD10", "WD10_sin", "WD10_cos",
    "u100", "v100", "WS100", "WD100",
    # Wind shear
    "dWS", "dWS_abs", "dWS_rollmean_3d", "dWS_rollstd_3d",
    "dWD", "dWD_abs", "dWD_rollstd_3d",
    # Temperature change
    "dT", "dT_rollmean_3d", "dT_rollmean_7d", "dT_rollstd_3d", "dT_rollstd_7d",
    # Tendency features
    "BLH_tend", "BLH_tend_rollmean_3d", "BLH_tend_rollmean_7d",
    "BLH_tend_rollstd_3d", "BLH_tend_rollstd_7d",
    "MSLP_tend", "MSLP_tend_rollmean_3d", "MSLP_tend_rollmean_7d",
    "MSLP_tend_rollstd_3d", "MSLP_tend_rollstd_7d",
    "SP_tend", "SP_tend_rollmean_3d", "SP_tend_rollmean_7d",
    "SP_tend_rollstd_3d", "SP_tend_rollstd_7d",
    # Daily aggregates
    "t2m_daily_mean", "t2m_daily_max", "t2m_daily_min",
    "WS10_daily_mean", "WS10_daily_max", "WS10_daily_min",
    "blh_daily_mean", "blh_daily_max", "blh_daily_min",
    "RH_daily_mean", "RH_daily_max", "RH_daily_min",
    "VPD_daily_mean", "VPD_daily_max", "VPD_daily_min",
    "VC_daily_mean", "VC_daily_max", "VC_daily_min",
    # Lag features
    "WS10_lag1d", "WS10_lag3d", "blh_lag1d", "blh_lag3d",
    # Rolling statistics
    "WS10_rollmean_3d", "WS10_rollstd_3d", "WS10_rollmin_7d", "WS10_rollmax_7d",
    "blh_rollmean_3d", "blh_rollmean_7d", "blh_rollmean_14d",
    "blh_rollmin_7d", "blh_anom_7d", "blh_anom_14d",
    "RH_rollmean_3d", "RH_rollmean_7d", "RH_rollmax_7d",
    "RH_rollstd_7d", "RH_anom_7d",
    "VPD_rollmean_3d", "VPD_rollmean_7d",
    "VC_rollmean_3d", "VC_rollmean_7d", "VC_rollmean_14d",
    "VC_rollmin_7d", "VC_anom_7d", "VC_anom_14d",
    # Wind direction rolling
    "WD10_sin_rm_7d", "WD10_cos_rm_7d", "WD10_var_7d",
    "WD10_sin_rm_14d", "WD10_cos_rm_14d", "WD10_var_14d",
    # Calm / stagnation indices
    "calm3_count", "calm3_flag", "calm7_count", "calm7_flag",
    "stagnant3_count", "stagnant3_flag", "stagnant7_count", "stagnant7_flag",
]

AOD_FEATURES = [
    "optical_depth_047", "optical_depth_055", "aod_uncertainty",
    "qa_cloudmask", "qa_adjacency", "qa_aod",
]

# TROPOMI: 10 products × 18 statistics = 180 features
_TROPOMI_PRODUCTS = [
    "no2", "so2", "co", "hcho", "aai", "alh", "ch4", "cloud", "o3", "nh3",
]
_TROPOMI_STATS = [
    "median", "mean", "std", "min", "max",
    "p10", "p25", "p75", "p90",
    "iqr", "mad", "range", "cv", "skew", "kurt",
    "delta_median_r1_r05", "delta_median_r2_r1", "delta_std_r2_r05",
]
TROPOMI_FEATURES = [f"{p}_{s}" for p in _TROPOMI_PRODUCTS for s in _TROPOMI_STATS]

TIME_FEATURES = [
    "doy_sin", "doy_cos", "doy_sin_2", "doy_cos_2", "doy_sin_3", "doy_cos_3",
    "heating_season_flag", "burning_season_flag",
]

PM25_PRIOR_FEATURES = ["pm25_lag1_prior", "pm25_lag1_cov"]

# Binary flags: do NOT normalize — keep as 0/1
BINARY_FEATURES = {"burning_season_flag", "heating_season_flag"}

ALL_FEATURES = MET_FEATURES + AOD_FEATURES + TROPOMI_FEATURES + PM25_PRIOR_FEATURES + TIME_FEATURES

# Channels that get an availability mask (can have NaN)
# pm25_lag1_prior is maskable (NaN where no coverage); pm25_lag1_cov is always ≥0
MASKED_FEATURES = MET_FEATURES + AOD_FEATURES + TROPOMI_FEATURES + ["pm25_lag1_prior"]

# Grid dimensions (input at 0.1°)
N_ROWS = 141
N_COLS = 175

# Output grid at 0.05° (2x resolution) for reduced station collisions
N_OUT_ROWS = 281
N_OUT_COLS = 349
OUT_RESOLUTION = 0.05

# Normalization
STD_FLOOR = 1e-6
N_NORM_SAMPLE_DATES = 50

# Total input channels: 307 features + 298 availability masks = 605
N_INPUT_CHANNELS = len(ALL_FEATURES) + len(MASKED_FEATURES)


class GridPM25Dataset(Dataset):
    """
    PyTorch Dataset that yields daily gridded feature tensors and station
    PM2.5 targets as coordinate lists (not rasterized).

    Parameters
    ----------
    dates : list of str
        Dates in 'YYYY-MM-DD' format.
    grid_store : str or Path
        Root of grid feature store.
    master_lake : str or Path
        Root of master_lake hive-partitioned directory.
    norm_stats : dict, optional
        Pre-computed {"mean": [...], "std": [...]} per channel.
    """

    def __init__(
        self,
        dates: List[str],
        grid_store: str | Path,
        master_lake: str | Path,
        norm_stats: Optional[Dict] = None,
    ):
        self.dates = sorted(dates)
        self.grid_store = Path(grid_store)
        self.master_lake = Path(master_lake)

        # Load static grid
        static_path = self.grid_store / "static" / "pakistan_grid_0p1.parquet"
        self.static_grid = pd.read_parquet(static_path)
        self.lat_max = float(self.static_grid["lat"].max())  # 37.3
        self.lon_min = float(self.static_grid["lon"].min())  # 60.5
        self._grid_merge = self.static_grid[["cell_id", "row", "col"]].copy()

        # Border mask at 0.1°
        self._border_tensor = torch.from_numpy(self._build_border_mask())

        # Lazy station cache: date → (grid_coords [N,2], pm25 [N])
        self._station_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

        # Normalization
        if norm_stats is not None:
            self.norm_mean = np.array(norm_stats["mean"], dtype=np.float32)
            self.norm_std = np.array(norm_stats["std"], dtype=np.float32)
        else:
            self.norm_mean, self.norm_std = self._compute_norm_stats()

    def _build_border_mask(self) -> np.ndarray:
        """Build [1, 141, 175] border mask from __inpoly in met data."""
        sample_date = self.dates[0]
        met_path = self.grid_store / "met" / f"date={sample_date}"
        met_df = pd.read_parquet(met_path, columns=["cell_id", "__inpoly"])
        merged = self._grid_merge.merge(met_df, on="cell_id", how="left")
        mask = np.zeros((1, N_ROWS, N_COLS), dtype=np.float32)
        inpoly = merged["__inpoly"].fillna(0).values.astype(np.float32)
        mask[0, merged["row"].values, merged["col"].values] = inpoly
        return mask

    def _load_station_obs(
        self, date_str: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load station obs as continuous normalized coords.

        Returns (grid_coords, pm25) where grid_coords is [N, 2] float32
        with (x, y) in [-1, 1] for PyTorch grid_sample (align_corners=True).
        """
        if date_str in self._station_cache:
            return self._station_cache[date_str]

        ml_path = self.master_lake / f"date={date_str}"
        if not ml_path.exists():
            self._station_cache[date_str] = None
            return None

        try:
            obs = pd.read_parquet(ml_path, columns=["obs_lat", "obs_lon", "pm25"])
        except Exception:
            self._station_cache[date_str] = None
            return None

        obs = obs.dropna(subset=["pm25"])
        if obs.empty:
            self._station_cache[date_str] = None
            return None

        # Continuous position in the 0.05° output grid (no rounding)
        # lat axis: row 0 = lat_max (north), row increases southward
        r_cont = (self.lat_max - obs["obs_lat"].values) / OUT_RESOLUTION
        c_cont = (obs["obs_lon"].values - self.lon_min) / OUT_RESOLUTION

        # Normalize to [-1, 1] for grid_sample with align_corners=True
        # -1 maps to pixel 0, +1 maps to pixel (N-1)
        y_norm = 2.0 * r_cont / (N_OUT_ROWS - 1) - 1.0
        x_norm = 2.0 * c_cont / (N_OUT_COLS - 1) - 1.0

        # Clip to valid range (stations slightly outside grid edge)
        x_norm = np.clip(x_norm, -1.0, 1.0)
        y_norm = np.clip(y_norm, -1.0, 1.0)

        # grid_sample convention: last dim is (x, y) where x=W, y=H
        grid_coords = np.stack([x_norm, y_norm], axis=1).astype(np.float32)

        result = (grid_coords, obs["pm25"].values.astype(np.float32))
        self._station_cache[date_str] = result
        return result

    def _load_day_features(self, date_str: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load features for one day.

        Returns:
            features:   [307, 141, 175] raw feature grid (may contain NaN)
            valid_mask: [298, 141, 175] binary mask (1=valid, 0=NaN) for MASKED_FEATURES
        """
        n_feat = len(ALL_FEATURES)
        grid = np.full((n_feat, N_ROWS, N_COLS), np.nan, dtype=np.float32)

        def _read_stage(stage: str, columns: List[str]) -> Optional[pd.DataFrame]:
            path = self.grid_store / stage / f"date={date_str}"
            if not path.exists():
                return None
            available_cols = ["cell_id"] + columns
            try:
                df = pd.read_parquet(path, columns=available_cols)
            except Exception:
                df = pd.read_parquet(path)
                available = [c for c in available_cols if c in df.columns]
                df = df[available]
            return df

        met_df = _read_stage("met", MET_FEATURES + TIME_FEATURES)
        aod_df = _read_stage("aod", AOD_FEATURES)
        trop_df = _read_stage("tropomi", TROPOMI_FEATURES)

        merged = self._grid_merge.copy()
        for df in [met_df, aod_df, trop_df]:
            if df is not None:
                df = df.drop_duplicates(subset=["cell_id"])
                merged = merged.merge(df, on="cell_id", how="left")

        rows = merged["row"].values
        cols = merged["col"].values

        for i, feat in enumerate(ALL_FEATURES):
            if feat in merged.columns:
                grid[i, rows, cols] = merged[feat].values.astype(np.float32)

        # Inject lag-1 PM2.5 prior channels
        border_np = self._border_tensor.numpy()
        prior, cov = build_pm25_lag1_prior_0p1(
            date_str, self.master_lake,
            self.lat_max, self.lon_min, border_np,
            N_ROWS, N_COLS,
        )
        idx_prior = ALL_FEATURES.index("pm25_lag1_prior")
        idx_cov = ALL_FEATURES.index("pm25_lag1_cov")
        grid[idx_prior] = prior
        grid[idx_cov] = cov

        # Build availability mask for MASKED_FEATURES
        n_masked = len(MASKED_FEATURES)
        valid_mask = np.zeros((n_masked, N_ROWS, N_COLS), dtype=np.float32)
        for j, feat in enumerate(MASKED_FEATURES):
            i = ALL_FEATURES.index(feat)
            valid_mask[j] = (~np.isnan(grid[i])).astype(np.float32)

        return grid, valid_mask

    def _compute_norm_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-channel mean/std from training dates."""
        n_sample = min(N_NORM_SAMPLE_DATES, len(self.dates))
        indices = np.linspace(0, len(self.dates) - 1, n_sample, dtype=int)
        sample_dates = [self.dates[i] for i in indices]

        print(f"  Computing normalization stats from {len(sample_dates)} sampled dates...")

        n_feat = len(ALL_FEATURES)
        sums = np.zeros(n_feat, dtype=np.float64)
        sq_sums = np.zeros(n_feat, dtype=np.float64)
        counts = np.zeros(n_feat, dtype=np.float64)

        for d in sample_dates:
            grid, _ = self._load_day_features(d)
            for i in range(n_feat):
                valid = ~np.isnan(grid[i])
                sums[i] += np.nansum(grid[i])
                sq_sums[i] += np.nansum(grid[i] ** 2)
                counts[i] += valid.sum()
            del grid
            gc.collect()

        mean = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)
        var = np.where(counts > 0, sq_sums / counts - mean**2, 1.0)
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        # Apply std floor
        std = np.maximum(std, STD_FLOOR)

        # Binary features: skip normalization (identity transform)
        for feat in BINARY_FEATURES:
            if feat in ALL_FEATURES:
                idx = ALL_FEATURES.index(feat)
                mean[idx] = 0.0
                std[idx] = 1.0

        return mean, std

    def get_norm_stats_dict(self) -> Dict:
        """Return normalization stats as JSON-serializable dict."""
        return {
            "mean": self.norm_mean.tolist(),
            "std": self.norm_std.tolist(),
            "features": ALL_FEATURES,
            "masked_features": MASKED_FEATURES,
            "binary_features": sorted(BINARY_FEATURES),
            "n_input_channels": N_INPUT_CHANNELS,
            "std_floor": STD_FLOOR,
            "n_norm_sample_dates": N_NORM_SAMPLE_DATES,
        }

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> Dict:
        date_str = self.dates[idx]

        # Load features and validity masks
        grid, valid_mask = self._load_day_features(date_str)

        # Normalize features: (x - mean) / std, NaN → 0 (neutral)
        mean = self.norm_mean[:, None, None]
        std = self.norm_std[:, None, None]
        grid = (grid - mean) / std
        grid = np.nan_to_num(grid, nan=0.0)

        # Concatenate: [307 features, 298 availability masks] = 605 channels
        combined = np.concatenate([grid, valid_mask], axis=0)
        X = torch.from_numpy(combined.copy())

        # Border mask at 0.1°
        border = self._border_tensor.clone()

        # Station coords as continuous normalized grid for grid_sample
        station_data = self._load_station_obs(date_str)
        if station_data is not None:
            grid_coords, pm25 = station_data
            station_grid = torch.from_numpy(grid_coords)   # [N, 2]
            station_pm25 = torch.from_numpy(pm25)           # [N]
        else:
            station_grid = torch.zeros(0, 2, dtype=torch.float32)
            station_pm25 = torch.zeros(0, dtype=torch.float32)

        return {
            "X": X,
            "border": border,
            "station_grid": station_grid,
            "station_pm25": station_pm25,
        }


class MemmapGridDataset(Dataset):
    """
    Fast dataset backed by a single memory-mapped float16 .npy file.

    The memmap is pre-normalized (from preprocess.py), so __getitem__ is just:
      1. Read one (602, 141, 175) float16 slice from memmap (~28 MB page-in)
      2. Cast to float32 tensor
      3. Look up station data from in-memory dict

    Expects a preprocessed directory containing:
      X.npy                    — (N_all, 602, 141, 175) float16, already normalized
      normalization_stats.json — mean/std used during preprocessing
      stations.pkl             — dict: date_str -> (grid_coords [N,2], pm25 [N])
      border.npy               — (1, 141, 175) float32
      dates.json               — ordered list of date strings

    Parameters
    ----------
    dates : list of str
        Dates to include (subset of all preprocessed dates).
    preprocessed_dir : str or Path
        Directory containing preprocessed files.
    """

    def __init__(
        self,
        dates: List[str],
        preprocessed_dir: str | Path,
    ):
        preprocessed_dir = Path(preprocessed_dir)

        # Load date index → memmap row mapping
        with open(preprocessed_dir / "dates.json") as f:
            all_dates = json.load(f)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # Filter to requested dates that exist in the preprocessed data
        self.dates = sorted([d for d in dates if d in date_to_idx])
        self.indices = [date_to_idx[d] for d in self.dates]

        # Memory-mapped feature tensor (read-only, float16)
        self._X = np.load(str(preprocessed_dir / "X.npy"), mmap_mode="r")

        # Border mask (small, in RAM)
        self._border_tensor = torch.from_numpy(
            np.load(str(preprocessed_dir / "border.npy"))
        )

        # Station data (small, in RAM)
        with open(preprocessed_dir / "stations.pkl", "rb") as f:
            self._stations = pickle.load(f)

        # Load normalization stats (baked into memmap, needed for results config)
        with open(preprocessed_dir / "normalization_stats.json") as f:
            self._norm_stats = json.load(f)

    def get_norm_stats_dict(self) -> Dict:
        """Return normalization stats used during preprocessing."""
        return self._norm_stats

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> Dict:
        date_str = self.dates[idx]
        memmap_idx = self.indices[idx]

        # Read float16 from memmap, cast to float32 for model input.
        # Clip to ±20 to suppress extreme outliers (e.g. cv features) that
        # cause NaN loss during training.
        X = torch.from_numpy(self._X[memmap_idx].astype(np.float32)).clamp(-20, 20)

        # Station data
        station_data = self._stations.get(date_str)
        if station_data is not None:
            grid_coords, pm25 = station_data
            station_grid = torch.from_numpy(grid_coords.copy())
            station_pm25 = torch.from_numpy(pm25.copy())
        else:
            station_grid = torch.zeros(0, 2, dtype=torch.float32)
            station_pm25 = torch.zeros(0, dtype=torch.float32)

        return {
            "X": X,
            "border": self._border_tensor.clone(),
            "station_grid": station_grid,
            "station_pm25": station_pm25,
        }
