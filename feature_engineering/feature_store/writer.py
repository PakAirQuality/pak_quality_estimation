"""
Feature Store Partition Writer
==============================

Takes a DataFrame and writes Parquet partitions by date.

Responsibilities:
- Ensure date column exists (extract from time if needed)
- Group by date
- Write part-000.parquet into date=YYYY-MM-DD/ directories
- Skip logic: if partition exists, skip (idempotent) unless --overwrite
"""

from __future__ import annotations

import logging
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .paths import StoreConfig, get_default_config, VALID_STAGES

logger = logging.getLogger(__name__)


class PartitionWriter:
    """
    Writer for date-partitioned Parquet files.

    Usage:
        writer = PartitionWriter(config, stage="met")
        stats = writer.write(df, overwrite=False)
        print(f"Wrote {stats['partitions_written']} partitions")
    """

    def __init__(
        self,
        config: Optional[StoreConfig] = None,
        stage: str = "met",
        date_column: str = "date",
        time_column: str = "time",
    ):
        """
        Initialize the partition writer.

        Args:
            config: Store configuration (uses default if None)
            stage: Feature stage name (met, aod, tropomi, geostatic)
            date_column: Column name for date (will be created if missing)
            time_column: Column name for timestamp (used to derive date if needed)
        """
        self.config = config or get_default_config()
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}")
        self.stage = stage
        self.date_column = date_column
        self.time_column = time_column

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has a date column.

        If date column doesn't exist, try to derive it from time column.
        """
        if self.date_column in df.columns:
            # Ensure it's date type
            if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
                df = df.copy()
                df[self.date_column] = pd.to_datetime(df[self.date_column]).dt.date
            elif hasattr(df[self.date_column].dtype, 'date'):
                pass  # Already date
            else:
                df = df.copy()
                df[self.date_column] = df[self.date_column].dt.date
            return df

        # Try to derive from time column
        if self.time_column in df.columns:
            df = df.copy()
            df[self.date_column] = pd.to_datetime(df[self.time_column]).dt.date
            logger.info(f"Created '{self.date_column}' column from '{self.time_column}'")
            return df

        # Try common time column names
        for col in ["time", "timestamp", "datetime", "date_utc", "time_utc"]:
            if col in df.columns:
                df = df.copy()
                df[self.date_column] = pd.to_datetime(df[col]).dt.date
                logger.info(f"Created '{self.date_column}' column from '{col}'")
                return df

        raise ValueError(
            f"Cannot find date or time column. "
            f"Expected '{self.date_column}' or '{self.time_column}' column. "
            f"Available columns: {list(df.columns)}"
        )

    def _write_partition(
        self,
        df: pd.DataFrame,
        partition_date: date,
        overwrite: bool = False,
    ) -> bool:
        """
        Write a single partition.

        Args:
            df: DataFrame to write (should already be filtered to this date)
            partition_date: Date for the partition
            overwrite: If True, overwrite existing partition

        Returns:
            True if written, False if skipped
        """
        partition_path = self.config.partition_path(self.stage, partition_date)
        parquet_path = self.config.parquet_path(self.stage, partition_date, part=0)

        # Check if partition exists
        if self.config.partition_exists(self.stage, partition_date):
            if not overwrite:
                logger.debug(f"Skipping existing partition: {partition_path}")
                return False
            else:
                logger.info(f"Overwriting partition: {partition_path}")
                self._delete_partition(partition_path)

        # Create directory and write
        if self.config.is_gcs:
            self._write_gcs(df, parquet_path)
        else:
            self._write_local(df, parquet_path)

        logger.debug(f"Wrote partition: {partition_path} ({len(df)} rows)")
        return True

    def _write_local(self, df: pd.DataFrame, path: str) -> None:
        """Write to local filesystem."""
        parquet_path = Path(path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

    def _write_gcs(self, df: pd.DataFrame, path: str) -> None:
        """Write to GCS."""
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(path.replace("gs://", ""), "wb") as f:
                df.to_parquet(f, index=False, engine="pyarrow")
        except ImportError:
            # Fall back to writing locally then gsutil cp
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                df.to_parquet(tmp.name, index=False, engine="pyarrow")
                # Ensure parent directory exists (gsutil handles this)
                subprocess.run(["gsutil", "cp", tmp.name, path], check=True)
                Path(tmp.name).unlink()

    def _delete_partition(self, partition_path: str) -> None:
        """Delete a partition directory."""
        if self.config.is_gcs:
            import subprocess
            subprocess.run(["gsutil", "-m", "rm", "-r", partition_path], check=True)
        else:
            shutil.rmtree(partition_path, ignore_errors=True)

    def write(
        self,
        df: pd.DataFrame,
        overwrite: bool = False,
        progress: bool = True,
    ) -> Dict[str, int]:
        """
        Write a DataFrame to date-partitioned Parquet files.

        Args:
            df: DataFrame with features (must have date or time column)
            overwrite: If True, overwrite existing partitions
            progress: If True, show progress

        Returns:
            Dict with statistics:
                - total_rows: Total rows in input
                - partitions_written: Number of partitions written
                - partitions_skipped: Number of partitions skipped (already exist)
                - dates: List of dates written
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to write")
            return {
                "total_rows": 0,
                "partitions_written": 0,
                "partitions_skipped": 0,
                "dates": [],
            }

        # Ensure date column exists
        df = self._ensure_date_column(df)

        # Group by date
        grouped = df.groupby(self.date_column)
        n_partitions = len(grouped)

        logger.info(f"Writing {len(df)} rows across {n_partitions} date partitions for stage '{self.stage}'")

        partitions_written = 0
        partitions_skipped = 0
        dates_written = []

        if progress:
            try:
                from tqdm import tqdm
                groups = tqdm(grouped, desc=f"Writing {self.stage}", unit="partition")
            except ImportError:
                groups = grouped
        else:
            groups = grouped

        for partition_date, group_df in groups:
            # Convert to date if it's a Timestamp
            if isinstance(partition_date, pd.Timestamp):
                partition_date = partition_date.date()

            written = self._write_partition(group_df, partition_date, overwrite=overwrite)
            if written:
                partitions_written += 1
                dates_written.append(partition_date)
            else:
                partitions_skipped += 1

        logger.info(
            f"Completed: {partitions_written} written, {partitions_skipped} skipped "
            f"({len(df)} total rows)"
        )

        return {
            "total_rows": len(df),
            "partitions_written": partitions_written,
            "partitions_skipped": partitions_skipped,
            "dates": dates_written,
        }

    def write_single_date(
        self,
        df: pd.DataFrame,
        partition_date: Union[date, datetime, str],
        overwrite: bool = False,
    ) -> bool:
        """
        Write a single date partition.

        Convenience method when you already have data for a specific date.

        Args:
            df: DataFrame for this date
            partition_date: Date for the partition
            overwrite: If True, overwrite if exists

        Returns:
            True if written, False if skipped
        """
        if isinstance(partition_date, str):
            partition_date = datetime.strptime(partition_date, "%Y-%m-%d").date()
        elif isinstance(partition_date, datetime):
            partition_date = partition_date.date()

        return self._write_partition(df, partition_date, overwrite=overwrite)


def write_stage_partitions(
    df: pd.DataFrame,
    stage: str,
    config: Optional[StoreConfig] = None,
    overwrite: bool = False,
    **kwargs,
) -> Dict[str, int]:
    """
    Convenience function to write partitions for a stage.

    Args:
        df: DataFrame with features
        stage: Stage name (met, aod, tropomi, geostatic)
        config: Store config (uses default if None)
        overwrite: If True, overwrite existing partitions
        **kwargs: Additional arguments for PartitionWriter

    Returns:
        Statistics dict
    """
    writer = PartitionWriter(config=config, stage=stage, **kwargs)
    return writer.write(df, overwrite=overwrite)
