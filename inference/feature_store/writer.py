"""
Grid Feature Store Partition Writer
===================================

Takes a grid DataFrame and writes Parquet partitions by date.

Responsibilities:
- Validate required columns (cell_id, lat, lon, date)
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

from .paths import GridStoreConfig, get_default_config, VALID_STAGES

logger = logging.getLogger(__name__)


class GridPartitionWriter:
    """
    Writer for grid-partitioned Parquet files.

    Grid features have one row per (cell_id, date) with ~24,675 rows per date.

    Usage:
        writer = GridPartitionWriter(config, stage="met")
        stats = writer.write(df, overwrite=False)
        print(f"Wrote {stats['partitions_written']} partitions")
    """

    # Expected grid size for Pakistan at 0.1 degree resolution
    EXPECTED_GRID_SIZE = 24675  # 141 x 175

    def __init__(
        self,
        config: Optional[GridStoreConfig] = None,
        stage: str = "met",
        date_column: str = "date",
    ):
        """
        Initialize the partition writer.

        Args:
            config: Store configuration (uses default if None)
            stage: Feature stage name (met, aod, tropomi)
            date_column: Column name for date
        """
        self.config = config or get_default_config()
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}")
        self.stage = stage
        self.date_column = date_column

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist.

        Required columns: cell_id (or lat+lon), date
        """
        has_cell_id = "cell_id" in df.columns
        has_coords = "lat" in df.columns and "lon" in df.columns
        has_date = self.date_column in df.columns

        if not has_date:
            raise ValueError(
                f"Missing required column '{self.date_column}'. "
                f"Available columns: {list(df.columns)[:20]}..."
            )

        if not has_cell_id and not has_coords:
            raise ValueError(
                "Missing required columns: need either 'cell_id' or ('lat', 'lon'). "
                f"Available columns: {list(df.columns)[:20]}..."
            )

    def _ensure_cell_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has a cell_id column.

        If cell_id doesn't exist, create it from lat/lon.
        """
        if "cell_id" in df.columns:
            return df

        if "lat" in df.columns and "lon" in df.columns:
            df = df.copy()
            df["cell_id"] = df.apply(
                lambda r: f"{r['lat']:.2f}_{r['lon']:.2f}", axis=1
            )
            logger.info("Created 'cell_id' column from 'lat' and 'lon'")
            return df

        raise ValueError("Cannot create cell_id: missing 'lat' and 'lon' columns")

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the date column is proper date type.
        """
        if self.date_column not in df.columns:
            raise ValueError(f"Missing date column: {self.date_column}")

        col = df[self.date_column]
        if pd.api.types.is_datetime64_any_dtype(col):
            df = df.copy()
            df[self.date_column] = col.dt.date
        elif not all(isinstance(v, date) for v in col.head(10) if pd.notna(v)):
            df = df.copy()
            df[self.date_column] = pd.to_datetime(col).dt.date

        return df

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

        # Validate row count
        if len(df) != self.EXPECTED_GRID_SIZE:
            logger.warning(
                f"Partition {partition_date} has {len(df)} rows, "
                f"expected {self.EXPECTED_GRID_SIZE}"
            )

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
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                df.to_parquet(tmp.name, index=False, engine="pyarrow")
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
            df: DataFrame with grid features (must have date, cell_id or lat/lon)
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

        # Validate and prepare
        self._validate_columns(df)
        df = self._ensure_cell_id(df)
        df = self._ensure_date_column(df)

        # Group by date
        grouped = df.groupby(self.date_column)
        n_partitions = len(grouped)

        logger.info(
            f"Writing {len(df)} rows across {n_partitions} date partitions "
            f"for stage '{self.stage}'"
        )

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

        # Ensure cell_id exists
        df = self._ensure_cell_id(df)

        return self._write_partition(df, partition_date, overwrite=overwrite)


def write_grid_partitions(
    df: pd.DataFrame,
    stage: str,
    config: Optional[GridStoreConfig] = None,
    overwrite: bool = False,
    **kwargs,
) -> Dict[str, int]:
    """
    Convenience function to write partitions for a stage.

    Args:
        df: DataFrame with grid features
        stage: Stage name (met, aod, tropomi)
        config: Store config (uses default if None)
        overwrite: If True, overwrite existing partitions
        **kwargs: Additional arguments for GridPartitionWriter

    Returns:
        Statistics dict
    """
    writer = GridPartitionWriter(config=config, stage=stage, **kwargs)
    return writer.write(df, overwrite=overwrite)
