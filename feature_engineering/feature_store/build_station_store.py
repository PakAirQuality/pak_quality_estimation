#!/usr/bin/env python3
"""
Build Station Feature Store
============================

ETL entrypoint that builds station-level feature stores from raw data.

This replaces the "build one huge CSV from raw" habit. After running once,
you have partitioned Parquet stores that don't need resampling for every experiment.

Output structure:
    {store_path}/met/date=YYYY-MM-DD/part-000.parquet
    {store_path}/aod/date=YYYY-MM-DD/part-000.parquet
    {store_path}/tropomi/date=YYYY-MM-DD/part-000.parquet

Usage:
    python -m feature_store.build_station_store \\
        --stages met,aod,tropomi \\
        --daily_csv data/paqi_network_daily.csv \\
        --met_dir datasets/features_met \\
        --aod_dir datasets/MCD19A2.061 \\
        --tropomi_dir datasets/tropomi_pakistan_2020_2025 \\
        --geos_cf_dir datasets/geos_cf_pakistan_2020_2025 \\
        --store_path feature_store \\
        --start 2020-01-01 \\
        --end 2025-07-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_store.paths import StoreConfig, VALID_STAGES
from feature_store.writer import PartitionWriter

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_observations(
    daily_csv: Path,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load observation data (stations with dates and target values).

    Args:
        daily_csv: Path to daily observations CSV
        start: Optional start date filter
        end: Optional end date filter

    Returns:
        DataFrame with columns: date_utc, latitude, longitude, pm25_daily_mean, etc.
    """
    logger.info(f"Loading observations from {daily_csv}")
    df = pd.read_csv(daily_csv)

    # Parse date column
    if "date_utc" in df.columns:
        df["date"] = pd.to_datetime(df["date_utc"]).dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        raise ValueError("Cannot find date column in observations CSV")

    # Filter by date range
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]

    logger.info(f"Loaded {len(df)} observations from {df['date'].min()} to {df['date'].max()}")
    return df


def build_met_store(
    df_obs: pd.DataFrame,
    met_dir: Path,
    config: StoreConfig,
    overwrite: bool = False,
) -> dict:
    """
    Build MET feature store from raw NetCDF files.

    Args:
        df_obs: Observations DataFrame with dates and locations
        met_dir: Path to raw MET features directory
        config: Store configuration
        overwrite: Overwrite existing partitions

    Returns:
        Statistics dict
    """
    from feature_family.met_features import compute_met_for_observations

    logger.info("Building MET feature store...")

    # Sample MET features using the ETL wrapper
    df_met = compute_met_for_observations(
        df_obs=df_obs,
        met_dir=met_dir,
        interp_method="linear",
        add_level2=True,
    )

    # Write to store
    writer = PartitionWriter(config=config, stage="met", date_column="date", time_column="time")
    return writer.write(df_met, overwrite=overwrite)


def build_aod_store(
    df_obs: pd.DataFrame,
    aod_dir: Path,
    config: StoreConfig,
    overwrite: bool = False,
) -> dict:
    """
    Build AOD feature store from raw HDF files.

    Args:
        df_obs: Observations DataFrame with dates and locations
        aod_dir: Path to raw AOD data directory
        config: Store configuration
        overwrite: Overwrite existing partitions

    Returns:
        Statistics dict
    """
    from feature_family.aod_features import compute_aod_for_observations

    logger.info("Building AOD feature store...")

    # Process AOD features using the ETL wrapper
    df_aod = compute_aod_for_observations(
        df_obs=df_obs,
        aod_dir=aod_dir,
    )

    # Write to store
    writer = PartitionWriter(config=config, stage="aod", date_column="date", time_column="time")
    return writer.write(df_aod, overwrite=overwrite)


def build_tropomi_store(
    df_obs: pd.DataFrame,
    tropomi_dir: Path,
    geos_cf_dir: Path,
    config: StoreConfig,
    products: List[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Build TROPOMI feature store from raw GeoTIFF files.

    Args:
        df_obs: Observations DataFrame with dates and locations
        tropomi_dir: Path to raw TROPOMI data directory
        geos_cf_dir: Path to GEOS-CF data directory
        config: Store configuration
        products: List of products to process
        overwrite: Overwrite existing partitions

    Returns:
        Statistics dict
    """
    from feature_family.tropomi_features import compute_tropomi_for_observations

    logger.info("Building TROPOMI feature store...")

    if products is None:
        products = ["NO2", "SO2", "CO", "HCHO", "AAI", "ALH", "CH4", "CLOUD", "O3"]

    # Process TROPOMI features using the ETL wrapper
    df_tropomi = compute_tropomi_for_observations(
        df_obs=df_obs,
        tropomi_base_dir=tropomi_dir,
        geos_cf_base_dir=geos_cf_dir,
        products=products,
    )

    # Write to store
    writer = PartitionWriter(config=config, stage="tropomi", date_column="date", time_column="time")
    return writer.write(df_tropomi, overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(
        description="Build station feature stores from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Stage selection
    parser.add_argument(
        "--stages",
        type=str,
        default="met,aod,tropomi",
        help="Comma-separated stages to build (met,aod,tropomi)",
    )

    # Input paths
    parser.add_argument(
        "--daily_csv",
        type=Path,
        default=Path("data/paqi_network_daily.csv"),
        help="Path to daily observations CSV",
    )
    parser.add_argument(
        "--met_dir",
        type=Path,
        default=Path("datasets/features_met"),
        help="Path to raw MET features directory",
    )
    parser.add_argument(
        "--aod_dir",
        type=Path,
        default=Path("datasets/MCD19A2.061"),
        help="Path to raw AOD data directory",
    )
    parser.add_argument(
        "--tropomi_dir",
        type=Path,
        default=Path("datasets/tropomi_pakistan_2020_2025"),
        help="Path to raw TROPOMI data directory",
    )
    parser.add_argument(
        "--geos_cf_dir",
        type=Path,
        default=Path("datasets/geos_cf_pakistan_2020_2025"),
        help="Path to GEOS-CF data directory",
    )

    # Output configuration
    parser.add_argument(
        "--store_path",
        type=str,
        default="feature_store",
        help="Base path for feature store (local path or gs:// URI)",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )

    # TROPOMI options
    parser.add_argument(
        "--products",
        type=str,
        default="NO2,SO2,CO,HCHO,AAI,ALH,CH4,CLOUD,O3",
        help="Comma-separated TROPOMI products to process",
    )

    # Processing options
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
    setup_logging(args.verbose)

    # Parse arguments
    stages = [s.strip() for s in args.stages.split(",")]
    for stage in stages:
        if stage not in VALID_STAGES:
            logger.error(f"Invalid stage: {stage}. Must be one of: {VALID_STAGES}")
            sys.exit(1)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    products = [p.strip() for p in args.products.split(",")]

    # Create store config
    if args.store_path.startswith("gs://"):
        config = StoreConfig(base_path=args.store_path)
    else:
        config = StoreConfig.local(args.store_path)

    logger.info(f"Feature store path: {config.base_path}")
    logger.info(f"Stages to build: {stages}")

    # Load observations
    df_obs = load_observations(args.daily_csv, start=start_date, end=end_date)

    # Build each stage
    results = {}

    if "met" in stages:
        try:
            results["met"] = build_met_store(
                df_obs=df_obs,
                met_dir=args.met_dir,
                config=config,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.error(f"Failed to build MET store: {e}")
            if args.verbose:
                raise

    if "aod" in stages:
        try:
            results["aod"] = build_aod_store(
                df_obs=df_obs,
                aod_dir=args.aod_dir,
                config=config,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.error(f"Failed to build AOD store: {e}")
            if args.verbose:
                raise

    if "tropomi" in stages:
        try:
            results["tropomi"] = build_tropomi_store(
                df_obs=df_obs,
                tropomi_dir=args.tropomi_dir,
                geos_cf_dir=args.geos_cf_dir,
                config=config,
                products=products,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.error(f"Failed to build TROPOMI store: {e}")
            if args.verbose:
                raise

    # Summary
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    for stage, stats in results.items():
        logger.info(
            f"  {stage}: {stats.get('partitions_written', 0)} partitions written, "
            f"{stats.get('partitions_skipped', 0)} skipped, "
            f"{stats.get('total_rows', 0)} total rows"
        )


if __name__ == "__main__":
    main()
