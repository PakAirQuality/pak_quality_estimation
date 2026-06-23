"""
Feature Store Partition Reader
==============================

Read data back from the feature store efficiently.

Responsibilities:
- Read only partitions within a date range
- Return a concatenated DataFrame
- Support both local and GCS paths

So downstream training/joins never touch raw NetCDF/HDF again.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .paths import StoreConfig, get_default_config, VALID_STAGES

logger = logging.getLogger(__name__)


class PartitionReader:
    """
    Reader for date-partitioned Parquet files.

    Usage:
        reader = PartitionReader(config, stage="met")
        df = reader.read(start="2020-01-01", end="2025-07-01")
    """

    def __init__(
        self,
        config: Optional[StoreConfig] = None,
        stage: str = "met",
    ):
        """
        Initialize the partition reader.

        Args:
            config: Store configuration (uses default if None)
            stage: Feature stage name (met, aod, tropomi, geostatic)
        """
        self.config = config or get_default_config()
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}")
        self.stage = stage

    def _parse_partition_date(self, partition_path: str) -> Optional[date]:
        """
        Extract date from a partition path.

        Args:
            partition_path: Path like ".../date=2025-01-17/..."

        Returns:
            Date object or None if parsing fails
        """
        match = re.search(r"date=(\d{4}-\d{2}-\d{2})", partition_path)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        return None

    def _list_partition_dates(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[date]:
        """
        List available partition dates within a range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Sorted list of available dates
        """
        partitions = self.config.list_partitions(self.stage)
        dates = []

        for partition_path in partitions:
            partition_date = self._parse_partition_date(partition_path)
            if partition_date is None:
                continue
            if start and partition_date < start:
                continue
            if end and partition_date > end:
                continue
            dates.append(partition_date)

        return sorted(dates)

    def _read_partition(self, partition_date: date) -> pd.DataFrame:
        """
        Read a single partition.

        Args:
            partition_date: Date of the partition

        Returns:
            DataFrame with partition data
        """
        partition_path = self.config.partition_path(self.stage, partition_date)

        if self.config.is_gcs:
            return self._read_gcs_partition(partition_path)
        else:
            return self._read_local_partition(partition_path)

    def _read_local_partition(self, partition_path: str) -> pd.DataFrame:
        """Read from local filesystem."""
        partition_dir = Path(partition_path)
        parquet_files = list(partition_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No parquet files in partition: {partition_path}")
            return pd.DataFrame()

        # Read and concatenate all part files
        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _read_gcs_partition(self, partition_path: str) -> pd.DataFrame:
        """Read from GCS."""
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            gcs_path = partition_path.replace("gs://", "")

            # List parquet files
            try:
                files = [f for f in fs.ls(gcs_path) if f.endswith(".parquet")]
            except FileNotFoundError:
                logger.warning(f"Partition not found: {partition_path}")
                return pd.DataFrame()

            if not files:
                logger.warning(f"No parquet files in partition: {partition_path}")
                return pd.DataFrame()

            # Read and concatenate
            dfs = []
            for f in sorted(files):
                with fs.open(f, "rb") as fh:
                    dfs.append(pd.read_parquet(fh))

            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        except ImportError:
            # Fall back to gsutil + local read
            import subprocess
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["gsutil", "-m", "cp", f"{partition_path}/*.parquet", tmpdir],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to read partition: {partition_path}")
                    return pd.DataFrame()

                parquet_files = list(Path(tmpdir).glob("*.parquet"))
                dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
                return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def read(
        self,
        start: Optional[Union[date, datetime, str]] = None,
        end: Optional[Union[date, datetime, str]] = None,
        dates: Optional[List[Union[date, datetime, str]]] = None,
        columns: Optional[List[str]] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Read partitions within a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            dates: Specific dates to read (overrides start/end)
            columns: Specific columns to read (None = all)
            progress: Show progress bar

        Returns:
            Concatenated DataFrame with all data
        """
        # Parse dates
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d").date()
        elif isinstance(start, datetime):
            start = start.date()

        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d").date()
        elif isinstance(end, datetime):
            end = end.date()

        # Get list of dates to read
        if dates is not None:
            # Use specific dates
            target_dates = []
            for d in dates:
                if isinstance(d, str):
                    d = datetime.strptime(d, "%Y-%m-%d").date()
                elif isinstance(d, datetime):
                    d = d.date()
                target_dates.append(d)
        else:
            # Discover available partitions in range
            target_dates = self._list_partition_dates(start, end)

        if not target_dates:
            logger.warning(f"No partitions found for stage '{self.stage}' in date range")
            return pd.DataFrame()

        logger.info(
            f"Reading {len(target_dates)} partitions for stage '{self.stage}' "
            f"({target_dates[0]} to {target_dates[-1]})"
        )

        # Read partitions
        dfs = []
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(target_dates, desc=f"Reading {self.stage}", unit="partition")
            except ImportError:
                iterator = target_dates
        else:
            iterator = target_dates

        for partition_date in iterator:
            df = self._read_partition(partition_date)
            if not df.empty:
                if columns:
                    # Select only requested columns (if they exist)
                    available_cols = [c for c in columns if c in df.columns]
                    df = df[available_cols]
                dfs.append(df)

        if not dfs:
            logger.warning(f"All partitions were empty for stage '{self.stage}'")
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Read {len(result)} total rows from {len(dfs)} partitions")
        return result

    def read_single_date(self, partition_date: Union[date, datetime, str]) -> pd.DataFrame:
        """
        Read a single date partition.

        Args:
            partition_date: Date to read

        Returns:
            DataFrame with partition data
        """
        if isinstance(partition_date, str):
            partition_date = datetime.strptime(partition_date, "%Y-%m-%d").date()
        elif isinstance(partition_date, datetime):
            partition_date = partition_date.date()

        return self._read_partition(partition_date)

    def get_available_dates(self) -> List[date]:
        """Get all available partition dates for this stage."""
        return self._list_partition_dates()

    def get_date_range(self) -> tuple[Optional[date], Optional[date]]:
        """Get the min and max dates available."""
        dates = self.get_available_dates()
        if not dates:
            return None, None
        return min(dates), max(dates)


def read_stage(
    stage: str,
    start: Optional[Union[date, datetime, str]] = None,
    end: Optional[Union[date, datetime, str]] = None,
    config: Optional[StoreConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to read a stage.

    Args:
        stage: Stage name (met, aod, tropomi, geostatic)
        start: Start date
        end: End date
        config: Store config (uses default if None)
        **kwargs: Additional arguments for PartitionReader.read()

    Returns:
        DataFrame with stage data
    """
    reader = PartitionReader(config=config, stage=stage)
    return reader.read(start=start, end=end, **kwargs)


def read_all_stages(
    start: Optional[Union[date, datetime, str]] = None,
    end: Optional[Union[date, datetime, str]] = None,
    stages: List[str] = ["met", "aod", "tropomi"],
    config: Optional[StoreConfig] = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Read multiple stages into a dictionary.

    Args:
        start: Start date
        end: End date
        stages: List of stages to read
        config: Store config
        **kwargs: Additional arguments

    Returns:
        Dict mapping stage name to DataFrame
    """
    result = {}
    for stage in stages:
        logger.info(f"Reading stage: {stage}")
        result[stage] = read_stage(stage, start=start, end=end, config=config, **kwargs)
    return result
