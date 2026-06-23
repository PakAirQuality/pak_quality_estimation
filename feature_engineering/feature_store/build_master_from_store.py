#!/usr/bin/env python3
"""
Build Master Dataset from Feature Store
========================================

Join step that builds the master dataset cheaply from partitioned stores.

This is now a fast table join, not a raw-data sampling job.

Process:
1. Read met partitions (date range)
2. Read aod partitions (date range)
3. Read tropomi partitions (date range)
4. Join them on (sensor_id, date) or key columns
5. Write: master parquet (recommended) + optional master CSV

Usage:
    python -m feature_store.build_master_from_store \\
        --store_path feature_store \\
        --start 2020-01-01 \\
        --end 2025-07-01 \\
        --output_parquet output/master.parquet \\
        --output_csv output/master.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_store.paths import StoreConfig
from feature_store.reader import PartitionReader
from feature_store.writer import PartitionWriter

logger = logging.getLogger(__name__)

# Canonical join key options - in order of preference
# For daily data: sensor_id + date_utc or sensor_id + date
# For hourly data: sensor_id + time
CANONICAL_JOIN_KEY_OPTIONS = [
    ["sensor_id", "date_utc"],   # Daily data (preferred)
    ["sensor_id", "date"],       # Daily data (alternative)
    ["sensor_id", "time"],       # Hourly data
]


def detect_join_keys(df: pd.DataFrame, stage: str) -> List[str]:
    """
    Detect which join keys are available in the DataFrame.

    Tries each option in order of preference.

    Returns:
        List of join key column names

    Raises:
        ValueError if no valid join key combination is found
    """
    for keys in CANONICAL_JOIN_KEY_OPTIONS:
        if all(k in df.columns for k in keys):
            return keys

    raise ValueError(
        f"[{stage}] CRITICAL: No valid join keys found. "
        f"Expected one of {CANONICAL_JOIN_KEY_OPTIONS}. "
        f"Available columns: {list(df.columns)[:20]}..."
    )


def require_canonical_keys(df: pd.DataFrame, stage: str) -> List[str]:
    """
    Enforce canonical join keys are present. Fail fast if missing.

    This prevents silent data corruption from partial joins.

    Returns:
        The detected join keys for this DataFrame
    """
    return detect_join_keys(df, stage)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def identify_join_keys(df: pd.DataFrame) -> List[str]:
    """
    Identify the best join keys from a DataFrame.

    Looks for common key columns in order of preference.

    Returns:
        List of column names to use as join keys
    """
    # Priority order of key columns
    key_candidates = [
        # Location + time keys
        ["latitude", "longitude", "date"],
        ["lat", "lon", "date"],
        ["latitude", "longitude", "time"],
        # Sensor ID + time keys
        ["sensor_id", "date"],
        ["station_id", "date"],
        ["sensor_id", "time"],
        # Fallback to just location
        ["latitude", "longitude"],
    ]

    for keys in key_candidates:
        if all(k in df.columns for k in keys):
            return keys

    raise ValueError(
        f"Cannot identify join keys. Available columns: {list(df.columns)}"
    )


def read_stage_data(
    config: StoreConfig,
    stage: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Read data for a stage from the store.

    Args:
        config: Store configuration
        stage: Stage name (met, aod, tropomi)
        start: Start date
        end: End date

    Returns:
        DataFrame with stage data
    """
    reader = PartitionReader(config=config, stage=stage)
    df = reader.read(start=start, end=end)

    if df.empty:
        logger.warning(f"No data found for stage '{stage}'")
        return df

    logger.info(f"Read {len(df)} rows for stage '{stage}'")
    return df


# Debug/metadata column patterns to exclude from final output
# These are diagnostic columns that inflate file size without adding predictive value
DEBUG_COLUMN_PATTERNS = [
    # TROPOMI debug columns
    "_file_available",
    "_file_used",
    "_invalid_reason",
    "_qa_available",
    "_qa_threshold",
    "_add_offset",
    "_scale_factor",
    "_window_coverage",
    "_window_size_used",
    "_radius_km_used",
    "_n_pixels",           # pixel counts are diagnostic
    "_qa_pass_fraction",   # QA fractions are diagnostic
    "agg_snap_dist_km",
    "agg_snapped_flag",
    # Multi-radius variants (not in original output)
    "_r05",                # 0.5km radius variant
    "_r1",                 # 1km radius variant
    "_r2",                 # 2km radius variant
    # AOD metadata columns
    "aod_files_used",
    "aod_hdf_tiles_used",
    "aod_n_hdf_files",
    "aod_total_valid_pixels",
]

# Columns that should not be duplicated from AOD/TROPOMI during join
# These are identifier columns already present in MET base
DUPLICATE_ID_COLUMNS = [
    "City", "Name", "latitude", "longitude", "date", "date_utc",
    "pm25_daily_mean", "sensor_id", "sensor_name",
    # Coordinate variants
    "obs_lat", "obs_lon", "obs_lat_aod", "obs_lon_aod",
    "sample_lat", "sample_lon",
    # Time variants
    "time", "time_aod",
]

# Additional metadata columns to exclude (grid/sampling diagnostics)
GRID_METADATA_COLUMNS = [
    "grid_dist_km", "grid_lat", "grid_lon",
    "sample_grid_dist_km", "sample_grid_lat", "sample_grid_lon",
    "sample_lat", "sample_lon",  # sampling coordinates
    "snap_dist_km", "snapped_flag",
    "agg_snap_dist_km", "agg_snapped_flag",
    # Time-derived columns (can be recreated from date_utc)
    "day_of_year", "dt_days", "month",
    "date",  # duplicate of date_utc
    # Other metadata
    "sensor_name",  # duplicate info (sensor_id is sufficient)
    "pm25_daily_mean",  # duplicate of pm25
]


def _filter_debug_columns(columns: List[str], exclude_ids: bool = True) -> List[str]:
    """
    Filter out debug/metadata columns from column list.

    These columns are useful for debugging but inflate file size
    without adding predictive value.

    Args:
        columns: List of column names
        exclude_ids: If True, also exclude duplicate ID columns
    """
    def should_exclude(col: str) -> bool:
        # Keep delta columns (spatial gradients) - they're valuable features
        if "_delta_" in col:
            return False
        # Check against debug patterns
        if any(pattern in col for pattern in DEBUG_COLUMN_PATTERNS):
            return True
        # Check against grid metadata
        if col in GRID_METADATA_COLUMNS:
            return True
        return False

    filtered = [c for c in columns if not should_exclude(c)]
    if exclude_ids:
        filtered = [c for c in filtered if c not in DUPLICATE_ID_COLUMNS]
    return filtered


def _normalize_join_key_dtypes(df: pd.DataFrame, join_keys: List[str]) -> pd.DataFrame:
    """
    Normalize join key dtypes to ensure compatible merges.

    Converts datetime columns to string for consistent joining.
    """
    df = df.copy()
    for key in join_keys:
        if key not in df.columns:
            continue
        # Convert datetime to string for consistent joining
        if pd.api.types.is_datetime64_any_dtype(df[key]):
            df[key] = df[key].astype(str)
        # Ensure string type for date-like columns
        elif key in ["date_utc", "date", "time"] and df[key].dtype == "object":
            # Already string, ensure consistent format
            df[key] = df[key].astype(str)
    return df


def join_stages(
    df_met: pd.DataFrame,
    df_aod: pd.DataFrame,
    df_tropomi: pd.DataFrame,
    join_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Join stage DataFrames on canonical keys.

    CRITICAL: All non-empty stages MUST have compatible join keys.
    This function will FAIL FAST if keys are missing to prevent silent data corruption.

    Args:
        df_met: MET features DataFrame
        df_aod: AOD features DataFrame
        df_tropomi: TROPOMI features DataFrame
        join_keys: Keys to join on (auto-detected if None)

    Returns:
        Joined master DataFrame

    Raises:
        ValueError: If any non-empty stage is missing required join keys
    """
    # Start with MET as base (usually has the most rows/dates)
    if df_met.empty:
        logger.warning("MET data is empty, cannot build master")
        return pd.DataFrame()

    # Detect join keys from MET (base) if not specified
    if join_keys is None:
        join_keys = require_canonical_keys(df_met, "met")
    logger.info(f"Using join keys: {join_keys}")

    # Verify other stages have the same keys
    if not df_aod.empty:
        aod_keys = require_canonical_keys(df_aod, "aod")
        if aod_keys != join_keys:
            logger.warning(f"AOD has different keys {aod_keys}, using {join_keys}")
    if not df_tropomi.empty:
        tropomi_keys = require_canonical_keys(df_tropomi, "tropomi")
        if tropomi_keys != join_keys:
            logger.warning(f"TROPOMI has different keys {tropomi_keys}, using {join_keys}")

    # Normalize join key dtypes to prevent merge errors
    df_met = _normalize_join_key_dtypes(df_met, join_keys)
    if not df_aod.empty:
        df_aod = _normalize_join_key_dtypes(df_aod, join_keys)
    if not df_tropomi.empty:
        df_tropomi = _normalize_join_key_dtypes(df_tropomi, join_keys)

    master = df_met.copy()
    initial_rows = len(master)

    # Join AOD if available
    if not df_aod.empty:
        aod_feature_cols = [c for c in df_aod.columns if c not in join_keys]
        aod_feature_cols = _filter_debug_columns(aod_feature_cols)

        if aod_feature_cols:
            # Deduplicate AOD on join keys to prevent row explosion
            df_aod_dedup = df_aod[join_keys + aod_feature_cols].drop_duplicates(subset=join_keys, keep="first")
            if len(df_aod_dedup) < len(df_aod):
                logger.warning(f"AOD had {len(df_aod) - len(df_aod_dedup)} duplicate rows on join keys, keeping first")

            logger.info(f"Joining AOD features ({len(aod_feature_cols)} columns) on {join_keys}")
            master = master.merge(
                df_aod_dedup,
                on=join_keys,
                how="left",
                suffixes=("", "_aod"),
            )
            logger.info(f"After AOD join: {len(master)} rows ({len(master) - initial_rows:+d})")

    # Join TROPOMI if available
    if not df_tropomi.empty:
        tropomi_feature_cols = [c for c in df_tropomi.columns if c not in join_keys]
        tropomi_feature_cols_raw = len(tropomi_feature_cols)
        tropomi_feature_cols = _filter_debug_columns(tropomi_feature_cols)
        logger.info(f"Filtered TROPOMI: {tropomi_feature_cols_raw} -> {len(tropomi_feature_cols)} columns (removed {tropomi_feature_cols_raw - len(tropomi_feature_cols)} debug cols)")

        if tropomi_feature_cols:
            # Deduplicate TROPOMI on join keys to prevent row explosion
            df_tropomi_dedup = df_tropomi[join_keys + tropomi_feature_cols].drop_duplicates(subset=join_keys, keep="first")
            if len(df_tropomi_dedup) < len(df_tropomi):
                logger.warning(f"TROPOMI had {len(df_tropomi) - len(df_tropomi_dedup)} duplicate rows on join keys, keeping first")

            logger.info(f"Joining TROPOMI features ({len(tropomi_feature_cols)} columns) on {join_keys}")
            master = master.merge(
                df_tropomi_dedup,
                on=join_keys,
                how="left",
                suffixes=("", "_tropomi"),
            )
            logger.info(f"After TROPOMI join: {len(master)} rows")

    # Final cleanup: remove any remaining debug/metadata columns from master
    cols_before = len(master.columns)
    cols_to_drop = [c for c in master.columns if c in GRID_METADATA_COLUMNS]
    # Also drop any columns with _aod or _tropomi suffix that are duplicates
    cols_to_drop += [c for c in master.columns if c.endswith(("_aod", "_tropomi")) and
                     c.replace("_aod", "").replace("_tropomi", "") in DUPLICATE_ID_COLUMNS]
    if cols_to_drop:
        master = master.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"Final cleanup: dropped {len(cols_to_drop)} metadata columns ({cols_before} -> {len(master.columns)})")

    return master


def write_master(
    df: pd.DataFrame,
    output_parquet: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    output_lake: Optional[Path] = None,
    config: Optional[StoreConfig] = None,
    overwrite: bool = False,
) -> None:
    """
    Write the master dataset to output files.

    Args:
        df: Master DataFrame
        output_parquet: Path for single parquet output
        output_csv: Path for CSV output (optional, for paper export)
        output_lake: Path for partitioned parquet output (lake format)
        config: StoreConfig (used for lake output if output_lake is not specified)
        overwrite: Overwrite existing partitions (for lake output)
    """
    if df.empty:
        logger.warning("Master DataFrame is empty, nothing to write")
        return

    if output_parquet:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_parquet, index=False)
        logger.info(f"Wrote master parquet: {output_parquet} ({len(df)} rows, {len(df.columns)} cols)")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        # Use %.16g format to match original pipeline precision (avoids extra decimal digits)
        df.to_csv(output_csv, index=False, float_format='%.16g')
        logger.info(f"Wrote master CSV: {output_csv} ({len(df)} rows, {len(df.columns)} cols)")

    if output_lake:
        write_master_lake(df, output_lake, overwrite=overwrite)


def write_master_lake(
    df: pd.DataFrame,
    output_path: Path,
    date_column: str = "date_utc",
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Write the master dataset to partitioned parquet (lake format).

    Args:
        df: Master DataFrame
        output_path: Base path for the lake output
        date_column: Column to use for partitioning (default: date_utc)
        overwrite: Overwrite existing partitions

    Returns:
        Statistics dict with partition counts
    """
    if df.empty:
        logger.warning("Master DataFrame is empty, nothing to write")
        return {"partitions_written": 0, "partitions_skipped": 0, "total_rows": 0}

    # Create a custom config for the master output
    lake_config = StoreConfig.local(str(output_path.parent))

    # We need to write to output_path/master/date=YYYY-MM-DD/
    # Override the base_path to point to output_path directly
    lake_config.base_path = str(output_path)

    writer = PartitionWriter(
        config=lake_config,
        stage="master",
        date_column="date",  # The writer expects 'date'
        time_column=date_column,
    )

    # Ensure we have a date column for partitioning
    df_copy = df.copy()
    if "date" not in df_copy.columns and date_column in df_copy.columns:
        df_copy["date"] = pd.to_datetime(df_copy[date_column]).dt.date

    stats = writer.write(df_copy, overwrite=overwrite)

    logger.info(
        f"Wrote master lake: {output_path} "
        f"({stats['partitions_written']} partitions, {stats['total_rows']} rows)"
    )

    return stats


def build_master(
    config: StoreConfig,
    start: Optional[date] = None,
    end: Optional[date] = None,
    stages: List[str] = ["met", "aod", "tropomi"],
    join_keys: Optional[List[str]] = None,
    output_parquet: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    output_lake: Optional[Path] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Build master dataset from feature stores.

    Args:
        config: Store configuration
        start: Start date
        end: End date
        stages: Stages to include
        join_keys: Keys to join on
        output_parquet: Output single parquet path
        output_csv: Output CSV path
        output_lake: Output path for partitioned parquet (lake format)
        overwrite: Overwrite existing partitions (for lake output)

    Returns:
        Master DataFrame
    """
    logger.info("=" * 60)
    logger.info("BUILDING MASTER DATASET FROM STORE")
    logger.info("=" * 60)
    logger.info(f"Store path: {config.base_path}")
    logger.info(f"Date range: {start} to {end}")
    logger.info(f"Stages: {stages}")

    # Read each stage
    df_met = pd.DataFrame()
    df_aod = pd.DataFrame()
    df_tropomi = pd.DataFrame()

    if "met" in stages:
        df_met = read_stage_data(config, "met", start, end)

    if "aod" in stages:
        df_aod = read_stage_data(config, "aod", start, end)

    if "tropomi" in stages:
        df_tropomi = read_stage_data(config, "tropomi", start, end)

    # Join stages
    logger.info("Joining stages...")
    master = join_stages(df_met, df_aod, df_tropomi, join_keys=join_keys)

    # Write outputs
    if output_parquet or output_csv or output_lake:
        write_master(
            master,
            output_parquet=output_parquet,
            output_csv=output_csv,
            output_lake=output_lake,
            config=config,
            overwrite=overwrite,
        )

    # Summary
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Master dataset: {len(master)} rows, {len(master.columns)} columns")

    if not master.empty:
        date_col = "date_utc" if "date_utc" in master.columns else "date"
        logger.info(f"Date range: {master[date_col].min()} to {master[date_col].max()}")

        # Feature counts by prefix
        met_cols = [c for c in master.columns if any(
            c.startswith(p) for p in ["u10", "v10", "t2m", "sp", "blh", "tp", "RH", "WS", "VPD", "VC"]
        )]
        aod_cols = [c for c in master.columns if "aod" in c.lower() or "AOD" in c]
        tropomi_cols = [c for c in master.columns if any(
            c.startswith(p) for p in ["NO2", "SO2", "CO", "HCHO", "AAI", "ALH", "CH4", "CLOUD", "O3"]
        )]

        logger.info(f"  MET features: ~{len(met_cols)}")
        logger.info(f"  AOD features: ~{len(aod_cols)}")
        logger.info(f"  TROPOMI features: ~{len(tropomi_cols)}")

    return master


def main():
    parser = argparse.ArgumentParser(
        description="Build master dataset from feature stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Store configuration
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

    # Stage selection
    parser.add_argument(
        "--stages",
        type=str,
        default="met,aod,tropomi",
        help="Comma-separated stages to include",
    )

    # Join configuration
    parser.add_argument(
        "--join_keys",
        type=str,
        default=None,
        help="Comma-separated join keys (auto-detected if not provided)",
    )

    # Output paths
    parser.add_argument(
        "--output_parquet",
        type=Path,
        default=Path("output/master.parquet"),
        help="Output parquet path (recommended)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path (optional, for paper export)",
    )
    parser.add_argument(
        "--no_parquet",
        action="store_true",
        help="Skip parquet output",
    )

    # Processing options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Parse arguments
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    stages = [s.strip() for s in args.stages.split(",")]
    join_keys = [k.strip() for k in args.join_keys.split(",")] if args.join_keys else None

    # Create store config
    if args.store_path.startswith("gs://"):
        config = StoreConfig(base_path=args.store_path)
    else:
        config = StoreConfig.local(args.store_path)

    # Determine output paths
    output_parquet = None if args.no_parquet else args.output_parquet
    output_csv = args.output_csv

    # Build master
    master = build_master(
        config=config,
        start=start_date,
        end=end_date,
        stages=stages,
        join_keys=join_keys,
        output_parquet=output_parquet,
        output_csv=output_csv,
    )

    return master


if __name__ == "__main__":
    main()
