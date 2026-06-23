#!/usr/bin/env python3
"""
Sync Station Data from GCS to Local Lake
=========================================

Downloads station hourly and daily readings from GCS bucket to local
Hive-style partitioned lake structure. Supports incremental sync.

GCS Source:
    gs://station-hourly-readings/exports/
        hourly/date=YYYY-MM-DD/part-*.parquet
        daily/date=YYYY-MM-DD/station_daily_*.parquet

Local Lake (mirrors GCS):
    lake/
        station_hourly/date=YYYY-MM-DD/part-*.parquet
        station_daily/date=YYYY-MM-DD/station_daily_*.parquet

Usage:
    # Sync all data
    python sync_station_lake.py

    # Sync specific date range
    python sync_station_lake.py --start 2024-01-01 --end 2024-12-31

    # Sync only daily (faster for feature pipeline)
    python sync_station_lake.py --daily-only

    # Dry run (show what would be synced)
    python sync_station_lake.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# GCS bucket configuration
GCS_BUCKET = "gs://paqi-raw-hawanama-data"
GCS_RAW = f"{GCS_BUCKET}/raw"
GCS_HOURLY = f"{GCS_RAW}/station_hourly"
GCS_DAILY = f"{GCS_RAW}/station_daily"

# Default local lake path (relative to repo root)
DEFAULT_LAKE_PATH = "lake"


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def list_gcs_partitions(gcs_path: str) -> List[str]:
    """List all date partitions in a GCS path."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", f"{gcs_path}/"],
            capture_output=True,
            text=True,
            check=True,
        )
        partitions = []
        for line in result.stdout.strip().split("\n"):
            if line and "date=" in line:
                # Extract date from path like gs://.../date=2024-01-01/
                date_part = line.rstrip("/").split("/")[-1]
                if date_part.startswith("date="):
                    partitions.append(date_part.replace("date=", ""))
        return sorted(partitions)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list GCS partitions: {e.stderr}")
        return []


def list_local_partitions(local_path: Path) -> Set[str]:
    """List all date partitions in local lake."""
    partitions = set()
    if local_path.exists():
        for partition_dir in local_path.iterdir():
            if partition_dir.is_dir() and partition_dir.name.startswith("date="):
                # Check if partition has _SUCCESS marker or parquet files
                has_success = (partition_dir / "_SUCCESS.json").exists()
                has_parquet = any(partition_dir.glob("*.parquet"))
                if has_success or has_parquet:
                    partitions.add(partition_dir.name.replace("date=", ""))
    return partitions


def sync_partition(
    gcs_source: str,
    local_dest: Path,
    date_str: str,
    dry_run: bool = False,
) -> bool:
    """Sync a single partition from GCS to local."""
    gcs_partition = f"{gcs_source}/date={date_str}/"
    local_partition = local_dest / f"date={date_str}"

    if dry_run:
        logger.info(f"[DRY RUN] Would sync: {gcs_partition} -> {local_partition}")
        return True

    # Create local directory
    local_partition.mkdir(parents=True, exist_ok=True)

    try:
        # Use gsutil rsync for efficient sync
        result = subprocess.run(
            [
                "gsutil", "-m", "rsync", "-r",
                gcs_partition,
                str(local_partition),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"Synced: {date_str}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to sync {date_str}: {e.stderr}")
        return False


def sync_lake(
    lake_path: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    hourly: bool = True,
    daily: bool = True,
    dry_run: bool = False,
    incremental: bool = True,
) -> dict:
    """
    Sync station data from GCS to local lake.

    Args:
        lake_path: Local lake directory path
        start_date: Start date filter (inclusive)
        end_date: End date filter (inclusive)
        hourly: Sync hourly partitions
        daily: Sync daily partitions
        dry_run: If True, only show what would be synced
        incremental: If True, skip existing partitions

    Returns:
        Summary dict with sync statistics
    """
    stats = {
        "hourly_synced": 0,
        "hourly_skipped": 0,
        "hourly_failed": 0,
        "daily_synced": 0,
        "daily_skipped": 0,
        "daily_failed": 0,
    }

    # Setup local paths
    local_hourly = lake_path / "station_hourly"
    local_daily = lake_path / "station_daily"

    if not dry_run:
        lake_path.mkdir(parents=True, exist_ok=True)

    # Sync hourly partitions
    if hourly:
        logger.info("Listing GCS hourly partitions...")
        gcs_hourly_dates = list_gcs_partitions(GCS_HOURLY)
        logger.info(f"Found {len(gcs_hourly_dates)} hourly partitions in GCS")

        if incremental:
            local_hourly_dates = list_local_partitions(local_hourly)
            logger.info(f"Found {len(local_hourly_dates)} hourly partitions locally")
        else:
            local_hourly_dates = set()

        # Filter by date range
        dates_to_sync = []
        for date_str in gcs_hourly_dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start_date and dt < start_date:
                    continue
                if end_date and dt > end_date:
                    continue
                if date_str in local_hourly_dates:
                    stats["hourly_skipped"] += 1
                    continue
                dates_to_sync.append(date_str)
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}")

        logger.info(f"Syncing {len(dates_to_sync)} hourly partitions...")
        for date_str in dates_to_sync:
            if sync_partition(GCS_HOURLY, local_hourly, date_str, dry_run):
                stats["hourly_synced"] += 1
            else:
                stats["hourly_failed"] += 1

    # Sync daily partitions
    if daily:
        logger.info("Listing GCS daily partitions...")
        gcs_daily_dates = list_gcs_partitions(GCS_DAILY)
        logger.info(f"Found {len(gcs_daily_dates)} daily partitions in GCS")

        if incremental:
            local_daily_dates = list_local_partitions(local_daily)
            logger.info(f"Found {len(local_daily_dates)} daily partitions locally")
        else:
            local_daily_dates = set()

        # Filter by date range
        dates_to_sync = []
        for date_str in gcs_daily_dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                if start_date and dt < start_date:
                    continue
                if end_date and dt > end_date:
                    continue
                if date_str in local_daily_dates:
                    stats["daily_skipped"] += 1
                    continue
                dates_to_sync.append(date_str)
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}")

        logger.info(f"Syncing {len(dates_to_sync)} daily partitions...")
        for date_str in dates_to_sync:
            if sync_partition(GCS_DAILY, local_daily, date_str, dry_run):
                stats["daily_synced"] += 1
            else:
                stats["daily_failed"] += 1

    return stats


def write_lake_manifest(lake_path: Path) -> None:
    """Write a manifest file with lake metadata."""
    manifest = {
        "synced_at": datetime.now().isoformat(),
        "source_bucket": GCS_BUCKET,
        "datasets": {},
    }

    for dataset in ["station_hourly", "station_daily"]:
        dataset_path = lake_path / dataset
        if dataset_path.exists():
            partitions = list_local_partitions(dataset_path)
            manifest["datasets"][dataset] = {
                "partition_count": len(partitions),
                "date_range": {
                    "min": min(partitions) if partitions else None,
                    "max": max(partitions) if partitions else None,
                },
            }

    manifest_path = lake_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync station data from GCS to local lake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--lake-path",
        type=Path,
        default=None,
        help=f"Local lake directory (default: <repo_root>/{DEFAULT_LAKE_PATH})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--hourly-only",
        action="store_true",
        help="Sync only hourly partitions",
    )
    parser.add_argument(
        "--daily-only",
        action="store_true",
        help="Sync only daily partitions",
    )
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help="Re-sync all partitions (ignore existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without downloading",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine lake path
    if args.lake_path:
        lake_path = args.lake_path
    else:
        lake_path = get_repo_root() / DEFAULT_LAKE_PATH

    logger.info(f"Lake path: {lake_path}")

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    # Determine what to sync
    sync_hourly = not args.daily_only
    sync_daily = not args.hourly_only

    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be downloaded")

    # Run sync
    stats = sync_lake(
        lake_path=lake_path,
        start_date=start_date,
        end_date=end_date,
        hourly=sync_hourly,
        daily=sync_daily,
        dry_run=args.dry_run,
        incremental=not args.full_sync,
    )

    # Write manifest (unless dry run)
    if not args.dry_run:
        write_lake_manifest(lake_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)
    print(f"Lake path: {lake_path}")
    if sync_hourly:
        print(f"Hourly: {stats['hourly_synced']} synced, {stats['hourly_skipped']} skipped, {stats['hourly_failed']} failed")
    if sync_daily:
        print(f"Daily:  {stats['daily_synced']} synced, {stats['daily_skipped']} skipped, {stats['daily_failed']} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
