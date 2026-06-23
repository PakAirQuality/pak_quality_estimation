"""
Preprocess grid parquet files into a single memory-mapped float16 .npy file.

Converts thousands of per-date parquet reads + pandas merges into a single
contiguous array for fast training. One-time cost (~15-30 min) that drops
per-epoch data loading from ~14 min to ~1 min.

Pipeline:
  1. Compute normalization stats from training dates (mean/std per feature)
  2. For each date: load parquet → normalize → NaN→0 → concat masks → float16
  3. Write to memmap incrementally (one sample at a time, no RAM pressure)

Output files:
    X.npy                  — (N_dates, 605, 141, 175) float16 memmap, normalized
    normalization_stats.json — per-channel mean/std used for normalization
    stations.pkl           — dict: date_str -> (grid_coords [N,2], pm25 [N])
    border.npy             — (1, 141, 175) float32, Pakistan boundary mask
    dates.json             — ordered list of date strings (index i -> dates[i])

Usage:
    python training/models/unet/preprocess.py \
        --grid_store ~/data/grid \
        --master_lake ~/data/master_lake/master \
        --output_dir ~/data/preprocessed \
        --train_start 2020-01-01 --train_end 2023-12-31
"""

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from training.models.unet.dataset import (
    ALL_FEATURES,
    AOD_FEATURES,
    BINARY_FEATURES,
    MASKED_FEATURES,
    MET_FEATURES,
    N_COLS,
    N_INPUT_CHANNELS,
    N_NORM_SAMPLE_DATES,
    N_OUT_COLS,
    N_OUT_ROWS,
    N_ROWS,
    OUT_RESOLUTION,
    STD_FLOOR,
    TIME_FEATURES,
    TROPOMI_FEATURES,
)
from training.models.unet.prior import build_pm25_lag1_prior_0p1


def discover_dates(grid_store: Path) -> list:
    """Find all dates with met data in the grid store."""
    met_dir = grid_store / "met"
    dates = []
    for d in sorted(met_dir.iterdir()):
        if d.name.startswith("date="):
            dates.append(d.name.split("=")[1])
    return dates


def load_static_grid(grid_store: Path):
    """Load static grid and extract merge keys + coordinate bounds."""
    static_path = grid_store / "static" / "pakistan_grid_0p1.parquet"
    static_grid = pd.read_parquet(static_path)
    grid_merge = static_grid[["cell_id", "row", "col"]].copy()
    lat_max = float(static_grid["lat"].max())
    lon_min = float(static_grid["lon"].min())
    return grid_merge, lat_max, lon_min


def load_day_features(
    date_str: str,
    grid_store: Path,
    grid_merge: pd.DataFrame,
    master_lake: Path = None,
    lat_max: float = None,
    lon_min: float = None,
    border: np.ndarray = None,
):
    """Load raw features + availability masks for one day.

    Returns:
        features:   (307, 141, 175) float32 with NaN where data is missing
        valid_mask: (298, 141, 175) float32, 1=valid 0=NaN
    """
    n_feat = len(ALL_FEATURES)
    grid = np.full((n_feat, N_ROWS, N_COLS), np.nan, dtype=np.float32)

    def _read_stage(stage, columns):
        path = grid_store / stage / f"date={date_str}"
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

    merged = grid_merge.copy()
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
    if master_lake is not None:
        prior, cov = build_pm25_lag1_prior_0p1(
            date_str, master_lake, lat_max, lon_min, border,
            N_ROWS, N_COLS,
        )
        idx_prior = ALL_FEATURES.index("pm25_lag1_prior")
        idx_cov = ALL_FEATURES.index("pm25_lag1_cov")
        grid[idx_prior] = prior
        grid[idx_cov] = cov

    # Build availability masks for MASKED_FEATURES
    n_masked = len(MASKED_FEATURES)
    valid_mask = np.zeros((n_masked, N_ROWS, N_COLS), dtype=np.float32)
    for j, feat in enumerate(MASKED_FEATURES):
        i = ALL_FEATURES.index(feat)
        valid_mask[j] = (~np.isnan(grid[i])).astype(np.float32)

    return grid, valid_mask


def load_station_obs(date_str, master_lake, lat_max, lon_min):
    """Load station observations as normalized grid coords.

    Returns (grid_coords [N,2], pm25 [N]) or None.
    """
    ml_path = master_lake / f"date={date_str}"
    if not ml_path.exists():
        return None

    try:
        obs = pd.read_parquet(ml_path, columns=["obs_lat", "obs_lon", "pm25"])
    except Exception:
        return None

    obs = obs.dropna(subset=["pm25"])
    if obs.empty:
        return None

    r_cont = (lat_max - obs["obs_lat"].values) / OUT_RESOLUTION
    c_cont = (obs["obs_lon"].values - lon_min) / OUT_RESOLUTION

    y_norm = 2.0 * r_cont / (N_OUT_ROWS - 1) - 1.0
    x_norm = 2.0 * c_cont / (N_OUT_COLS - 1) - 1.0
    x_norm = np.clip(x_norm, -1.0, 1.0)
    y_norm = np.clip(y_norm, -1.0, 1.0)

    grid_coords = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    pm25 = obs["pm25"].values.astype(np.float32)
    return (grid_coords, pm25)


def build_border_mask(grid_store, grid_merge, sample_date):
    """Build (1, 141, 175) Pakistan boundary mask."""
    met_path = grid_store / "met" / f"date={sample_date}"
    met_df = pd.read_parquet(met_path, columns=["cell_id", "__inpoly"])
    merged = grid_merge.merge(met_df, on="cell_id", how="left")
    mask = np.zeros((1, N_ROWS, N_COLS), dtype=np.float32)
    inpoly = merged["__inpoly"].fillna(0).values.astype(np.float32)
    mask[0, merged["row"].values, merged["col"].values] = inpoly
    return mask


def compute_norm_stats(dates, grid_store, grid_merge, master_lake, lat_max, lon_min, border):
    """Compute per-feature mean/std from a sample of dates."""
    n_feat = len(ALL_FEATURES)
    n_sample = min(N_NORM_SAMPLE_DATES, len(dates))
    indices = np.linspace(0, len(dates) - 1, n_sample, dtype=int)
    sample_dates = [dates[i] for i in indices]

    print(f"  Computing normalization stats from {len(sample_dates)} sampled dates...")

    sums = np.zeros(n_feat, dtype=np.float64)
    sq_sums = np.zeros(n_feat, dtype=np.float64)
    counts = np.zeros(n_feat, dtype=np.float64)

    for d in sample_dates:
        grid, _ = load_day_features(
            d, grid_store, grid_merge, master_lake, lat_max, lon_min, border,
        )
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
    std = np.maximum(std, STD_FLOOR)

    # Binary features: identity transform (mean=0, std=1)
    for feat in BINARY_FEATURES:
        if feat in ALL_FEATURES:
            idx = ALL_FEATURES.index(feat)
            mean[idx] = 0.0
            std[idx] = 1.0

    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess grid parquet files to float16 memmap"
    )
    parser.add_argument("--grid_store", required=True)
    parser.add_argument("--master_lake", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_start", default="2020-01-01",
                        help="Training start date (for normalization stats)")
    parser.add_argument("--train_end", default="2023-12-31",
                        help="Training end date (for normalization stats)")
    args = parser.parse_args()

    grid_store = Path(args.grid_store)
    master_lake = Path(args.master_lake)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout.reconfigure(line_buffering=True)

    # --- Discover dates ---
    print("Discovering available dates...")
    dates = discover_dates(grid_store)
    N = len(dates)
    print(f"Found {N} dates: {dates[0]} to {dates[-1]}")

    # --- Static grid ---
    grid_merge, lat_max, lon_min = load_static_grid(grid_store)

    # --- Border mask ---
    print("Building border mask...")
    border = build_border_mask(grid_store, grid_merge, dates[0])
    np.save(output_dir / "border.npy", border)

    # --- Normalization stats (computed from training dates only) ---
    train_dates = [d for d in dates if args.train_start <= d <= args.train_end]
    print(f"Training dates for normalization: {len(train_dates)}")
    mean, std = compute_norm_stats(
        train_dates, grid_store, grid_merge, master_lake, lat_max, lon_min, border,
    )

    norm_stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "features": ALL_FEATURES,
        "masked_features": MASKED_FEATURES,
        "binary_features": sorted(BINARY_FEATURES),
        "n_input_channels": N_INPUT_CHANNELS,
        "std_floor": STD_FLOOR,
        "n_norm_sample_dates": N_NORM_SAMPLE_DATES,
        "train_start": args.train_start,
        "train_end": args.train_end,
    }
    with open(output_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("  Normalization stats saved.")

    # --- Create float16 memmap ---
    X_path = output_dir / "X.npy"
    shape = (N, N_INPUT_CHANNELS, N_ROWS, N_COLS)
    size_gb = N * N_INPUT_CHANNELS * N_ROWS * N_COLS * 2 / 1e9  # float16 = 2 bytes
    print(f"Creating float16 memmap: {X_path}")
    print(f"  Shape: {shape}  Size: {size_gb:.1f} GB")
    X = np.lib.format.open_memmap(
        str(X_path), mode="w+", dtype=np.float16, shape=shape,
    )

    # --- Process each date: normalize → NaN→0 → concat masks → float16 ---
    n_feat = len(ALL_FEATURES)
    mean_3d = mean[:, None, None]
    std_3d = std[:, None, None]
    stations = {}
    t_start = time.time()

    for i, date_str in enumerate(dates):
        features, valid_mask = load_day_features(
            date_str, grid_store, grid_merge, master_lake, lat_max, lon_min, border,
        )

        # Normalize features (float32), then NaN → 0
        features = (features - mean_3d) / std_3d
        np.nan_to_num(features, copy=False, nan=0.0)

        # Concatenate [normalized features, availability masks] → float16
        combined = np.concatenate([features, valid_mask], axis=0)
        X[i] = combined.astype(np.float16)

        # Station observations
        station_data = load_station_obs(date_str, master_lake, lat_max, lon_min)
        if station_data is not None:
            stations[date_str] = station_data

        if (i + 1) % 100 == 0 or i == 0 or (i + 1) == N:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(
                f"  [{i+1:>{len(str(N))}}/{N}] {date_str}"
                f" | {rate:.1f} dates/s | ETA {eta/60:.1f} min"
            )

    X.flush()
    del X

    elapsed = time.time() - t_start
    print(f"\nFeature processing complete in {elapsed/60:.1f} min")

    # --- Save station data ---
    print(f"Saving station data ({len(stations)} dates with observations)...")
    with open(output_dir / "stations.pkl", "wb") as f:
        pickle.dump(stations, f)

    # --- Save date index ---
    print("Saving date index...")
    with open(output_dir / "dates.json", "w") as f:
        json.dump(dates, f)

    # --- Summary ---
    print(f"\nPreprocessing complete!")
    print(f"  Dates:    {N}")
    print(f"  X.npy:    {size_gb:.1f} GB (float16)")
    print(f"  Stations: {len(stations)} dates with observations")
    print(f"  Output:   {output_dir}/")


if __name__ == "__main__":
    main()
