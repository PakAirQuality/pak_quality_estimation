#!/usr/bin/env python3
"""
Main Feature Engineering Orchestrator - Lake Data Source
=========================================================

Builds partitioned Parquet feature stores for PM2.5 prediction using
the station_labels/lake data source.

Output is saved to feature_engineering/output/master_lake/ by default
in partitioned parquet (lake) format.

Usage:
    # Default: outputs to feature_engineering/output/master_lake/ (lake format)
    python main_feature_pipeline.py --verbose

    # With custom lake output path
    python main_feature_pipeline.py \\
        --output_lake /custom/path/master_lake \\
        --start 2020-01-01 \\
        --end 2025-07-13

    # Output as single files instead of lake format
    python main_feature_pipeline.py \\
        --output_parquet output/master.parquet \\
        --output_csv output/master.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Default lake path (relative to feature_engineering directory)
DEFAULT_LAKE_PATH = Path(__file__).parent.parent / "extraction_and_preprocessing" / "station_labels" / "lake"

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def load_observations_from_lake(
    lake_path: Path,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load daily observations from the lake partitioned parquet files.

    Args:
        lake_path: Path to the lake directory containing station_daily/
        start: Start date filter
        end: End date filter

    Returns:
        DataFrame with columns matching paqi_network_daily.csv format:
        - date_utc, City, Name, latitude, longitude, pm25_daily_mean
    """
    station_daily_path = lake_path / "station_daily"

    if not station_daily_path.exists():
        raise FileNotFoundError(f"Lake station_daily directory not found: {station_daily_path}")

    # List all date partitions
    date_dirs = sorted([d for d in station_daily_path.iterdir() if d.is_dir() and d.name.startswith("date=")])

    logger.info(f"Found {len(date_dirs)} date partitions in lake")

    # Filter partitions by date range
    filtered_dirs = []
    for d in date_dirs:
        date_str = d.name.replace("date=", "")
        try:
            partition_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start and partition_date < start:
                continue
            if end and partition_date > end:
                continue
            filtered_dirs.append(d)
        except ValueError:
            logger.warning(f"Skipping invalid date partition: {d.name}")
            continue

    logger.info(f"Loading {len(filtered_dirs)} partitions after date filtering")

    if not filtered_dirs:
        return pd.DataFrame()

    # Load all parquet files
    dfs = []
    for partition_dir in filtered_dirs:
        parquet_files = list(partition_dir.glob("*.parquet"))
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {pf}: {e}")

    if not dfs:
        return pd.DataFrame()

    df_combined = pd.concat(dfs, ignore_index=True)

    # Rename columns to match paqi_network_daily.csv format
    df_obs = df_combined.rename(columns={
        "date": "date_utc",
        "city_name": "City",
        "station_name": "Name",
        "pm25_ugm3_mean": "pm25_daily_mean",
    })

    # Select only required columns
    required_cols = ["date_utc", "City", "Name", "latitude", "longitude", "pm25_daily_mean"]
    df_obs = df_obs[required_cols].copy()

    # Drop rows with missing pm25 values
    df_obs = df_obs.dropna(subset=["pm25_daily_mean"])

    # Convert date to datetime
    df_obs["date_utc"] = pd.to_datetime(df_obs["date_utc"])

    logger.info(f"Loaded {len(df_obs)} observations from lake")
    logger.info(f"Date range: {df_obs['date_utc'].min()} to {df_obs['date_utc'].max()}")
    logger.info(f"Unique stations: {df_obs['Name'].nunique()}")

    return df_obs


def filter_to_missing_dates(
    df_obs: pd.DataFrame,
    config,  # StoreConfig
    stage: str,
    date_column: str = "date",
) -> Tuple[pd.DataFrame, List, List]:
    """
    Filter observations to only dates that don't have partitions yet.

    This is the key optimization for incremental processing:
    - First run: all dates are missing → heavy compute
    - Later runs: only new dates → fast (Cloud Run friendly)

    Args:
        df_obs: Observations DataFrame with date column
        config: StoreConfig for checking partition existence
        stage: Stage name (met, aod, tropomi)
        date_column: Name of date column

    Returns:
        Tuple of (filtered_df, missing_dates, existing_dates)
    """
    all_dates = sorted(df_obs[date_column].unique())

    existing_dates = []
    missing_dates = []

    for dt in all_dates:
        if config.partition_exists(stage, dt):
            existing_dates.append(dt)
        else:
            missing_dates.append(dt)

    logger.info(f"[{stage}] Date check: {len(existing_dates)} existing, {len(missing_dates)} missing")

    if not missing_dates:
        logger.info(f"[{stage}] All partitions exist, nothing to compute")
        return pd.DataFrame(), missing_dates, existing_dates

    # Filter to only missing dates
    df_filtered = df_obs[df_obs[date_column].isin(missing_dates)].copy()
    logger.info(f"[{stage}] Will compute {len(df_filtered)} rows for {len(missing_dates)} missing dates")

    return df_filtered, missing_dates, existing_dates


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure logging with optional file output."""
    level = logging.INFO  # Always use INFO for root to suppress library debug spam

    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w', encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Suppress noisy third-party library loggers
    noisy_libs = [
        "findlibs", "gribapi", "gribapi.bindings",
        "rasterio", "rasterio.env", "rasterio.session",
        "h5py", "h5py._conv",
        "urllib3", "botocore", "boto3",
        "pyproj", "fiona", "shapely",
    ]
    for lib in noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)


# =============================================================================
# STORE MODE (NEW FLOW)
# =============================================================================

def run_store_pipeline(
    stages: List[str],
    lake_path: Path,
    met_dir: Path,
    aod_dir: Path,
    tropomi_dir: Path,
    geos_cf_dir: Path,
    store_path: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    products: List[str] = None,
    output_parquet: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    output_lake: Optional[Path] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run the feature store pipeline using lake data source.

    This is a 2-step process:
    1. Build partitioned stores from raw data (can be skipped if stores exist)
    2. Join stores into master dataset (fast)

    Args:
        stages: List of stages to process (met, aod, tropomi)
        lake_path: Path to lake directory containing station_daily/
        met_dir: Path to MET features directory
        aod_dir: Path to AOD data directory
        tropomi_dir: Path to TROPOMI data directory
        geos_cf_dir: Path to GEOS-CF data directory
        store_path: Base path for feature store
        start: Start date filter
        end: End date filter
        products: TROPOMI products to process
        output_parquet: Output single parquet path
        output_csv: Output CSV path
        output_lake: Output path for partitioned parquet (lake format)
        overwrite: Overwrite existing partitions

    Returns:
        Master DataFrame
    """
    # Add directories to path for imports
    base_dir = Path(__file__).parent
    sys.path.insert(0, str(base_dir))
    sys.path.insert(0, str(base_dir / "feature_family"))  # For feature_registry imports

    from feature_store.paths import StoreConfig
    from feature_store.writer import PartitionWriter
    from feature_store.reader import PartitionReader

    logger.info("=" * 60)
    logger.info("FEATURE STORE PIPELINE (Lake Data Source)")
    logger.info("=" * 60)

    # Create store config
    if store_path.startswith("gs://"):
        config = StoreConfig(base_path=store_path)
    else:
        config = StoreConfig.local(store_path)

    logger.info(f"Store path: {config.base_path}")
    logger.info(f"Lake path: {lake_path}")
    logger.info(f"Stages: {stages}")
    logger.info(f"Date range: {start} to {end}")

    # Default products
    if products is None:
        products = ["NO2", "SO2", "CO", "HCHO", "AAI", "ALH", "CH4", "CLOUD", "O3"]

    # -------------------------------------------------------------------------
    # STEP 1: Load observations from lake
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("STEP 1: Loading observations from lake")
    logger.info("-" * 60)

    df_obs = load_observations_from_lake(lake_path, start=start, end=end)

    if df_obs.empty:
        logger.warning("No observations loaded from lake!")
        return pd.DataFrame()

    # Parse date column
    df_obs["date"] = pd.to_datetime(df_obs["date_utc"]).dt.date

    logger.info(f"Loaded {len(df_obs)} observations from {df_obs['date'].min()} to {df_obs['date'].max()}")

    # -------------------------------------------------------------------------
    # STEP 2: Build partitioned stores
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("STEP 2: Building partitioned feature stores")
    logger.info("-" * 60)

    build_results = {}

    if "met" in stages:
        logger.info("[MET] Building MET feature store...")
        try:
            # Filter to only missing dates (incremental processing)
            df_obs_met, missing_dates, existing_dates = filter_to_missing_dates(
                df_obs, config, "met", date_column="date"
            )

            if df_obs_met.empty:
                build_results["met"] = {
                    "partitions_written": 0,
                    "partitions_skipped": len(existing_dates),
                    "total_rows": 0,
                }
            else:
                from feature_family.met_features import compute_met_for_observations

                df_met = compute_met_for_observations(
                    df_obs=df_obs_met,
                    met_dir=met_dir,
                    interp_method="linear",
                    add_level2=True,
                )

                writer = PartitionWriter(config=config, stage="met", date_column="date", time_column="time")
                build_results["met"] = writer.write(df_met, overwrite=overwrite)
                build_results["met"]["partitions_skipped"] = len(existing_dates)

            logger.info(f"[MET] Done: {build_results['met'].get('partitions_written', 0)} written, {build_results['met'].get('partitions_skipped', 0)} skipped")
        except Exception as e:
            logger.error(f"[MET] Failed: {e}")
            raise

    if "aod" in stages:
        logger.info("[AOD] Building AOD feature store...")
        try:
            # Filter to only missing dates (incremental processing)
            df_obs_aod, missing_dates, existing_dates = filter_to_missing_dates(
                df_obs, config, "aod", date_column="date"
            )

            if df_obs_aod.empty:
                build_results["aod"] = {
                    "partitions_written": 0,
                    "partitions_skipped": len(existing_dates),
                    "total_rows": 0,
                }
            else:
                from feature_family.aod_features import compute_aod_for_observations

                df_aod = compute_aod_for_observations(
                    df_obs=df_obs_aod,
                    aod_dir=aod_dir,
                )

                writer = PartitionWriter(config=config, stage="aod", date_column="date", time_column="time")
                build_results["aod"] = writer.write(df_aod, overwrite=overwrite)
                build_results["aod"]["partitions_skipped"] = len(existing_dates)

            logger.info(f"[AOD] Done: {build_results['aod'].get('partitions_written', 0)} written, {build_results['aod'].get('partitions_skipped', 0)} skipped")
        except Exception as e:
            logger.error(f"[AOD] Failed: {e}")
            raise

    if "tropomi" in stages:
        logger.info("[TROPOMI] Building TROPOMI feature store...")
        try:
            # Filter to only missing dates (incremental processing)
            df_obs_tropomi, missing_dates, existing_dates = filter_to_missing_dates(
                df_obs, config, "tropomi", date_column="date"
            )

            if df_obs_tropomi.empty:
                build_results["tropomi"] = {
                    "partitions_written": 0,
                    "partitions_skipped": len(existing_dates),
                    "total_rows": 0,
                }
            else:
                from feature_family.tropomi_features import compute_tropomi_for_observations

                df_tropomi = compute_tropomi_for_observations(
                    df_obs=df_obs_tropomi,
                    tropomi_base_dir=tropomi_dir,
                    geos_cf_base_dir=geos_cf_dir,
                    products=products,
                )

                writer = PartitionWriter(config=config, stage="tropomi", date_column="date", time_column="time")
                build_results["tropomi"] = writer.write(df_tropomi, overwrite=overwrite)
                build_results["tropomi"]["partitions_skipped"] = len(existing_dates)

            logger.info(f"[TROPOMI] Done: {build_results['tropomi'].get('partitions_written', 0)} written, {build_results['tropomi'].get('partitions_skipped', 0)} skipped")
        except Exception as e:
            logger.error(f"[TROPOMI] Failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # STEP 3: Join stores into master dataset
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("STEP 3: Joining stores into master dataset")
    logger.info("-" * 60)

    from feature_store.build_master_from_store import build_master

    master = build_master(
        config=config,
        start=start,
        end=end,
        stages=stages,
        output_parquet=output_parquet,
        output_csv=output_csv,
        output_lake=output_lake,
        overwrite=overwrite,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Master dataset: {len(master)} rows, {len(master.columns)} columns")

    for stage, stats in build_results.items():
        logger.info(
            f"  {stage}: {stats.get('partitions_written', 0)} partitions written, "
            f"{stats.get('partitions_skipped', 0)} skipped"
        )

    if output_parquet:
        logger.info(f"Output parquet: {output_parquet}")
    if output_csv:
        logger.info(f"Output CSV: {output_csv}")
    if output_lake:
        logger.info(f"Output lake: {output_lake}")

    return master


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run feature engineering pipeline (lake data source)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Lake path (replaces daily_csv)
    parser.add_argument("--lake_path", default=None, type=Path,
                        help="Path to lake directory (default: ../extraction_and_preprocessing/station_labels/lake)")
    parser.add_argument("--chunk", type=int, default=50000,
                        help="Processing chunk size")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--log_file", type=Path,
                        help="Log file path (auto-generated if not specified)")

    # Stage control
    parser.add_argument("--stages", type=str, default="met,aod,tropomi",
                        help="Comma-separated stages to process")
    parser.add_argument("--skip_met", action="store_true",
                        help="Skip MET processing")
    parser.add_argument("--skip_aod", action="store_true",
                        help="Skip AOD processing")
    parser.add_argument("--skip_tropomi", action="store_true",
                        help="Skip TROPOMI processing")

    # Data directories
    parser.add_argument("--features_dir", default="datasets/features_met", type=Path,
                        help="MET features directory")
    parser.add_argument("--aod_dir", default="datasets/MCD19A2.061", type=Path,
                        help="AOD data directory")
    parser.add_argument("--tropomi_base_dir", default="datasets/tropomi_pakistan_2020_2025", type=Path,
                        help="TROPOMI data directory")
    parser.add_argument("--geos_cf_base_dir", default="datasets/geos_cf_pakistan_2020_2025", type=Path,
                        help="GEOS-CF data directory")

    # Store arguments
    parser.add_argument("--store_path", type=str, default="feature_store",
                        help="Feature store base path (local or gs://)")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-07-13",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_parquet", type=Path, default=None,
                        help="Output single parquet path (disabled by default, use --output_lake)")
    parser.add_argument("--output_csv", type=Path, default=None,
                        help="Output CSV path (disabled by default, use --output_lake)")
    parser.add_argument("--output_lake", type=Path, default=None,
                        help="Output path for partitioned parquet lake format (default: feature_engineering/output/master_lake)")

    parser.add_argument("--products", default="NO2,SO2,CO,HCHO,AAI,ALH,CH4,CLOUD,O3,NH3",
                        help="TROPOMI products")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing partitions")

    args = parser.parse_args()

    # Default lake path
    if args.lake_path is None:
        args.lake_path = DEFAULT_LAKE_PATH

    # Default output paths (ensure they're in feature_engineering/output/)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Default to lake format output
    if args.output_lake is None and args.output_parquet is None and args.output_csv is None:
        args.output_lake = DEFAULT_OUTPUT_DIR / "master_lake"

    # Set up log file
    if args.log_file:
        log_file = args.log_file
    else:
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f"pipeline_{timestamp}.log"

    # Determine stages from skip flags
    stages = [s.strip() for s in args.stages.split(",")]
    if args.skip_met and "met" in stages:
        stages.remove("met")
    if args.skip_aod and "aod" in stages:
        stages.remove("aod")
    if args.skip_tropomi and "tropomi" in stages:
        stages.remove("tropomi")

    print(f"[orchestrator] Logging to: {log_file}", flush=True)

    setup_logging(args.verbose, log_file)

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    # Parse products
    products = [p.strip() for p in args.products.split(",")]

    pipeline_start = time.time()

    try:
        master = run_store_pipeline(
            stages=stages,
            lake_path=args.lake_path,
            met_dir=args.features_dir,
            aod_dir=args.aod_dir,
            tropomi_dir=args.tropomi_base_dir,
            geos_cf_dir=args.geos_cf_base_dir,
            store_path=args.store_path,
            start=start_date,
            end=end_date,
            products=products,
            output_parquet=args.output_parquet,
            output_csv=args.output_csv,
            output_lake=args.output_lake,
            overwrite=args.overwrite,
        )

        elapsed = time.time() - pipeline_start
        logger.info(f"Pipeline completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
