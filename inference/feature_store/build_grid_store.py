#!/usr/bin/env python3
"""
Build Grid Feature Store
========================

Build grid-level Parquet partitions from raw data (NetCDF, HDF, GeoTIFF).

This script processes raw satellite and meteorological data to create
date-partitioned Parquet files for fast inference.

Usage:
    # Build all stages for a date range
    python build_grid_store.py --stages met,aod,tropomi --start 2020-01-01 --end 2025-12-31

    # Build only MET stage
    python build_grid_store.py --stages met --start 2024-01-01 --end 2024-12-31

    # Overwrite existing partitions
    python build_grid_store.py --stages met --start 2024-03-01 --end 2024-03-31 --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup path for imports
_THIS_DIR = Path(__file__).resolve().parent
_INFERENCE_DIR = _THIS_DIR.parent
_REPO_ROOT = _INFERENCE_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

from .paths import GridStoreConfig, VALID_STAGES
from .writer import GridPartitionWriter

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Pakistan grid bounds (matches inference/grids/)
PAKISTAN_BOUNDS = {
    "lat_min": 23.3,
    "lat_max": 37.3,
    "lon_min": 60.5,
    "lon_max": 77.9,
}


def build_static_grid(
    config: GridStoreConfig,
    resolution: float = 0.1,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Build and write the static grid definition file.

    Creates a parquet file with grid cell definitions:
    - cell_id: Unique identifier (e.g., "28.50_67.30")
    - lat: Latitude of cell center
    - lon: Longitude of cell center
    - row: Row index (0 to n_lat-1)
    - col: Column index (0 to n_lon-1)

    Args:
        config: Store configuration
        resolution: Grid resolution in degrees
        overwrite: If True, overwrite existing file

    Returns:
        DataFrame with grid definition
    """
    static_path = config.static_path()

    # Check if already exists
    if config.static_exists() and not overwrite:
        logger.info(f"Static grid already exists: {static_path}")
        # Read and return existing
        if config.is_gcs:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(static_path.replace("gs://", ""), "rb") as f:
                return pd.read_parquet(f)
        else:
            return pd.read_parquet(static_path)

    # Build grid
    lat_range = PAKISTAN_BOUNDS["lat_max"] - PAKISTAN_BOUNDS["lat_min"]
    lon_range = PAKISTAN_BOUNDS["lon_max"] - PAKISTAN_BOUNDS["lon_min"]

    n_lat = int(np.round(lat_range / resolution)) + 1
    n_lon = int(np.round(lon_range / resolution)) + 1

    # Create coordinate arrays (north to south, west to east)
    lats_1d = np.linspace(PAKISTAN_BOUNDS["lat_max"], PAKISTAN_BOUNDS["lat_min"], n_lat)
    lons_1d = np.linspace(PAKISTAN_BOUNDS["lon_min"], PAKISTAN_BOUNDS["lon_max"], n_lon)

    # Create meshgrid
    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)

    # Flatten to create DataFrame
    n_cells = n_lat * n_lon
    rows = []

    for i in range(n_lat):
        for j in range(n_lon):
            lat = lats_1d[i]
            lon = lons_1d[j]
            cell_id = f"{lat:.2f}_{lon:.2f}"
            rows.append({
                "cell_id": cell_id,
                "lat": np.float32(lat),
                "lon": np.float32(lon),
                "row": np.int16(i),
                "col": np.int16(j),
            })

    df = pd.DataFrame(rows)

    logger.info(f"Built static grid: {n_lat} x {n_lon} = {n_cells} cells")

    # Write to parquet
    if config.is_gcs:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(static_path.replace("gs://", ""), "wb") as f:
            df.to_parquet(f, index=False, engine="pyarrow")
    else:
        static_file = Path(static_path)
        static_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(static_file, index=False, engine="pyarrow")

    logger.info(f"Wrote static grid: {static_path}")
    return df


def build_met_store(
    start: date,
    end: date,
    config: GridStoreConfig,
    met_dir: Path,
    history_days: int = 21,
    overwrite: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Build MET stage from raw NetCDF files.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
        config: Store configuration
        met_dir: Path to features_met directory
        history_days: Days of history for rolling features
        overwrite: If True, overwrite existing partitions
        verbose: Enable verbose logging

    Returns:
        Statistics dict with partitions_written, partitions_skipped
    """
    from grids.met_grid import MetGrid, MetGridConfig

    logger.info(f"Building MET store: {start} to {end}")

    # Load static grid
    static_df = build_static_grid(config, overwrite=False)
    lats_1d = static_df.groupby("row")["lat"].first().sort_index().values
    lons_1d = static_df.groupby("col")["lon"].first().sort_index().values

    # Initialize MET generator
    met_config = MetGridConfig(
        features_met_dir=met_dir,
        grid_lats_1d=lats_1d,
        grid_lons_1d=lons_1d,
    )
    met_grid = MetGrid(met_config)

    # Initialize writer
    writer = GridPartitionWriter(config, stage="met")

    stats = {
        "partitions_written": 0,
        "partitions_skipped": 0,
        "partitions_failed": 0,
    }

    current = start
    total_days = (end - start).days + 1

    try:
        from tqdm import tqdm
        date_iter = tqdm(range(total_days), desc="Building MET", unit="day")
    except ImportError:
        date_iter = range(total_days)

    for i in date_iter:
        current = start + timedelta(days=i)

        # Check if already exists
        if config.partition_exists("met", current) and not overwrite:
            stats["partitions_skipped"] += 1
            continue

        try:
            # Generate features
            features = met_grid.compute(current, history_days=history_days)

            # Convert to DataFrame
            n_cells = len(static_df)
            df = pd.DataFrame({
                "date": [current] * n_cells,
                "cell_id": static_df["cell_id"].values,
                "lat": static_df["lat"].values,
                "lon": static_df["lon"].values,
            })

            # Add feature columns
            for name, arr in features.items():
                if arr is not None:
                    flat = np.asarray(arr, dtype=np.float32).ravel()
                    if len(flat) == n_cells:
                        df[name] = flat

            # Write partition
            written = writer.write_single_date(df, current, overwrite=overwrite)
            if written:
                stats["partitions_written"] += 1
            else:
                stats["partitions_skipped"] += 1

            if verbose:
                logger.info(f"MET {current}: {len(df.columns)} columns")

        except Exception as e:
            logger.error(f"MET {current} failed: {e}")
            stats["partitions_failed"] += 1

    logger.info(
        f"MET complete: {stats['partitions_written']} written, "
        f"{stats['partitions_skipped']} skipped, {stats['partitions_failed']} failed"
    )
    return stats


def build_aod_store(
    start: date,
    end: date,
    config: GridStoreConfig,
    aod_dir: Path,
    overwrite: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Build AOD stage from raw HDF files.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
        config: Store configuration
        aod_dir: Path to MCD19A2.061 directory
        overwrite: If True, overwrite existing partitions
        verbose: Enable verbose logging

    Returns:
        Statistics dict
    """
    from grids.aod_grid import AODGrid, AODGridConfig

    logger.info(f"Building AOD store: {start} to {end}")

    # Load static grid
    static_df = build_static_grid(config, overwrite=False)
    lats = static_df["lat"].values
    lons = static_df["lon"].values

    # Create pakistan mask (all True for now, can be refined)
    n_lat = static_df["row"].max() + 1
    n_lon = static_df["col"].max() + 1
    pakistan_mask = np.ones((n_lat, n_lon), dtype=bool)

    # Initialize AOD generator
    aod_config = AODGridConfig(
        datasets_dir=aod_dir.parent,
        aod_dir=aod_dir,
        grid_lats=lats.reshape(n_lat, n_lon),
        grid_lons=lons.reshape(n_lat, n_lon),
    )
    aod_grid = AODGrid(aod_config)

    # Initialize writer
    writer = GridPartitionWriter(config, stage="aod")

    stats = {
        "partitions_written": 0,
        "partitions_skipped": 0,
        "partitions_failed": 0,
    }

    current = start
    total_days = (end - start).days + 1

    try:
        from tqdm import tqdm
        date_iter = tqdm(range(total_days), desc="Building AOD", unit="day")
    except ImportError:
        date_iter = range(total_days)

    for i in date_iter:
        current = start + timedelta(days=i)

        if config.partition_exists("aod", current) and not overwrite:
            stats["partitions_skipped"] += 1
            continue

        try:
            # Generate features
            features = aod_grid.compute(current, pakistan_mask=pakistan_mask, verbose=verbose)

            # Convert to DataFrame
            n_cells = len(static_df)
            df = pd.DataFrame({
                "date": [current] * n_cells,
                "cell_id": static_df["cell_id"].values,
                "lat": static_df["lat"].values,
                "lon": static_df["lon"].values,
            })

            # Add feature columns
            for name, arr in features.items():
                if arr is not None:
                    flat = np.asarray(arr, dtype=np.float32).ravel()
                    if len(flat) == n_cells:
                        df[name] = flat

            written = writer.write_single_date(df, current, overwrite=overwrite)
            if written:
                stats["partitions_written"] += 1
            else:
                stats["partitions_skipped"] += 1

            if verbose:
                logger.info(f"AOD {current}: {len(df.columns)} columns")

        except Exception as e:
            logger.error(f"AOD {current} failed: {e}")
            stats["partitions_failed"] += 1

    logger.info(
        f"AOD complete: {stats['partitions_written']} written, "
        f"{stats['partitions_skipped']} skipped, {stats['partitions_failed']} failed"
    )
    return stats


def build_tropomi_store(
    start: date,
    end: date,
    config: GridStoreConfig,
    tropomi_dir: Path,
    geos_cf_dir: Path,
    tropomi_window: int = 5,
    overwrite: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Build TROPOMI stage from raw GeoTIFF files.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
        config: Store configuration
        tropomi_dir: Path to TROPOMI directory
        geos_cf_dir: Path to GEOS-CF directory
        tropomi_window: Window size for multiscale features
        overwrite: If True, overwrite existing partitions
        verbose: Enable verbose logging

    Returns:
        Statistics dict
    """
    from grids.tropomi_grid import TropomiGrid, TropomiGridConfig

    logger.info(f"Building TROPOMI store: {start} to {end}")

    # Load static grid
    static_df = build_static_grid(config, overwrite=False)
    n_lat = static_df["row"].max() + 1
    n_lon = static_df["col"].max() + 1

    lats_1d = static_df.groupby("row")["lat"].first().sort_index().values
    lons_1d = static_df.groupby("col")["lon"].first().sort_index().values

    # Initialize TROPOMI generator
    trop_config = TropomiGridConfig(
        tropomi_dir=tropomi_dir,
        geos_cf_dir=geos_cf_dir,
        grid_shape=(n_lat, n_lon),
        grid_lats_1d=lats_1d,
        grid_lons_1d=lons_1d,
        grid_resolution_deg=config.grid_resolution,
        tropomi_window=tropomi_window,
    )
    trop_grid = TropomiGrid(trop_config)

    # Initialize writer
    writer = GridPartitionWriter(config, stage="tropomi")

    stats = {
        "partitions_written": 0,
        "partitions_skipped": 0,
        "partitions_failed": 0,
    }

    total_days = (end - start).days + 1

    try:
        from tqdm import tqdm
        date_iter = tqdm(range(total_days), desc="Building TROPOMI", unit="day")
    except ImportError:
        date_iter = range(total_days)

    for i in date_iter:
        current = start + timedelta(days=i)

        if config.partition_exists("tropomi", current) and not overwrite:
            stats["partitions_skipped"] += 1
            continue

        try:
            # Generate features
            features = trop_grid.compute(current, verbose=verbose)

            # Convert to DataFrame
            n_cells = len(static_df)
            df = pd.DataFrame({
                "date": [current] * n_cells,
                "cell_id": static_df["cell_id"].values,
                "lat": static_df["lat"].values,
                "lon": static_df["lon"].values,
            })

            # Add feature columns
            for name, arr in features.items():
                if arr is not None:
                    flat = np.asarray(arr, dtype=np.float32).ravel()
                    if len(flat) == n_cells:
                        df[name] = flat

            written = writer.write_single_date(df, current, overwrite=overwrite)
            if written:
                stats["partitions_written"] += 1
            else:
                stats["partitions_skipped"] += 1

            if verbose:
                logger.info(f"TROPOMI {current}: {len(df.columns)} columns")

        except Exception as e:
            logger.error(f"TROPOMI {current} failed: {e}")
            stats["partitions_failed"] += 1

    logger.info(
        f"TROPOMI complete: {stats['partitions_written']} written, "
        f"{stats['partitions_skipped']} skipped, {stats['partitions_failed']} failed"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build grid feature store from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Date range
    parser.add_argument(
        "--start",
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date YYYY-MM-DD",
    )

    # Stages
    parser.add_argument(
        "--stages",
        default="met,aod,tropomi",
        help="Comma-separated stages to build (default: met,aod,tropomi)",
    )

    # Paths
    parser.add_argument(
        "--store_path",
        default="derived/feature_store/grid",
        help="Base path for feature store",
    )
    parser.add_argument(
        "--datasets_dir",
        default="datasets",
        help="Path to datasets directory",
    )
    parser.add_argument(
        "--met_dir",
        help="Path to MET NetCDF files (default: {datasets_dir}/features_met)",
    )
    parser.add_argument(
        "--aod_dir",
        help="Path to AOD HDF files (default: {datasets_dir}/MCD19A2.061)",
    )
    parser.add_argument(
        "--tropomi_dir",
        help="Path to TROPOMI GeoTIFFs (default: {datasets_dir}/tropomi)",
    )
    parser.add_argument(
        "--geos_cf_dir",
        help="Path to GEOS-CF data (default: {datasets_dir}/geos_cf_pakistan_2020_2025)",
    )

    # Options
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="Grid resolution in degrees (default: 0.1)",
    )
    parser.add_argument(
        "--history_days",
        type=int,
        default=21,
        help="MET history window in days (default: 21)",
    )
    parser.add_argument(
        "--tropomi_window",
        type=int,
        default=5,
        help="TROPOMI window size (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing partitions",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Parse dates
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if start > end:
        parser.error("start date must be <= end date")

    # Parse stages
    stages = [s.strip() for s in args.stages.split(",")]
    for stage in stages:
        if stage not in VALID_STAGES:
            parser.error(f"Invalid stage: {stage}. Must be one of: {VALID_STAGES}")

    # Setup paths
    datasets_dir = Path(args.datasets_dir)
    met_dir = Path(args.met_dir) if args.met_dir else datasets_dir / "features_met"
    aod_dir = Path(args.aod_dir) if args.aod_dir else datasets_dir / "MCD19A2.061"
    tropomi_dir = Path(args.tropomi_dir) if args.tropomi_dir else datasets_dir / "tropomi_pakistan_2020_2025"
    geos_cf_dir = Path(args.geos_cf_dir) if args.geos_cf_dir else datasets_dir / "geos_cf_pakistan_2020_2025"

    # Setup config
    config = GridStoreConfig(
        base_path=args.store_path,
        grid_resolution=args.resolution,
    )

    logger.info(f"Building grid store: {start} to {end}")
    logger.info(f"Stages: {stages}")
    logger.info(f"Store path: {config.base_path}")

    # Build static grid first
    build_static_grid(config, resolution=args.resolution, overwrite=args.overwrite)

    # Build each stage
    all_stats = {}

    if "met" in stages:
        all_stats["met"] = build_met_store(
            start, end, config, met_dir,
            history_days=args.history_days,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

    if "aod" in stages:
        all_stats["aod"] = build_aod_store(
            start, end, config, aod_dir,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

    if "tropomi" in stages:
        all_stats["tropomi"] = build_tropomi_store(
            start, end, config, tropomi_dir, geos_cf_dir,
            tropomi_window=args.tropomi_window,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    for stage, stats in all_stats.items():
        print(f"{stage:10s}: {stats['partitions_written']:5d} written, "
              f"{stats['partitions_skipped']:5d} skipped, "
              f"{stats['partitions_failed']:5d} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
