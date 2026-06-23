"""
Grid Feature Store Partition Reader
===================================

Read grid data back from the feature store efficiently.

Responsibilities:
- Read only partitions within a date range
- Return a concatenated DataFrame
- Support both local and GCS paths
- Support column selection for efficient reads
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .paths import GridStoreConfig, get_default_config, VALID_STAGES

logger = logging.getLogger(__name__)


class GridPartitionReader:
    """
    Reader for grid-partitioned Parquet files.

    Usage:
        reader = GridPartitionReader(config, stage="met")
        df = reader.read(start="2024-01-01", end="2024-03-31")
    """

    def __init__(
        self,
        config: Optional[GridStoreConfig] = None,
        stage: str = "met",
    ):
        """
        Initialize the partition reader.

        Args:
            config: Store configuration (uses default if None)
            stage: Feature stage name (met, aod, tropomi)
        """
        self.config = config or get_default_config()
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}")
        self.stage = stage

    def _parse_partition_date(self, partition_path: str) -> Optional[date]:
        """
        Extract date from a partition path.

        Args:
            partition_path: Path like ".../date=2024-03-13/..."

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

        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _read_gcs_partition(self, partition_path: str) -> pd.DataFrame:
        """Read from GCS."""
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            gcs_path = partition_path.replace("gs://", "")

            try:
                files = [f for f in fs.ls(gcs_path) if f.endswith(".parquet")]
            except FileNotFoundError:
                logger.warning(f"Partition not found: {partition_path}")
                return pd.DataFrame()

            if not files:
                logger.warning(f"No parquet files in partition: {partition_path}")
                return pd.DataFrame()

            dfs = []
            for f in sorted(files):
                with fs.open(f, "rb") as fh:
                    dfs.append(pd.read_parquet(fh))

            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        except ImportError:
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
            target_dates = []
            for d in dates:
                if isinstance(d, str):
                    d = datetime.strptime(d, "%Y-%m-%d").date()
                elif isinstance(d, datetime):
                    d = d.date()
                target_dates.append(d)
        else:
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
                    available_cols = [c for c in columns if c in df.columns]
                    df = df[available_cols]
                dfs.append(df)

        if not dfs:
            logger.warning(f"All partitions were empty for stage '{self.stage}'")
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Read {len(result)} total rows from {len(dfs)} partitions")
        return result

    def read_single_date(
        self,
        partition_date: Union[date, datetime, str],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read a single date partition.

        Args:
            partition_date: Date to read
            columns: Specific columns to read (None = all)

        Returns:
            DataFrame with partition data
        """
        if isinstance(partition_date, str):
            partition_date = datetime.strptime(partition_date, "%Y-%m-%d").date()
        elif isinstance(partition_date, datetime):
            partition_date = partition_date.date()

        df = self._read_partition(partition_date)

        if columns and not df.empty:
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        return df

    def get_available_dates(self) -> List[date]:
        """Get all available partition dates for this stage."""
        return self._list_partition_dates()

    def get_date_range(self) -> Tuple[Optional[date], Optional[date]]:
        """Get the min and max dates available."""
        dates = self.get_available_dates()
        if not dates:
            return None, None
        return min(dates), max(dates)


def read_grid_stage(
    stage: str,
    start: Optional[Union[date, datetime, str]] = None,
    end: Optional[Union[date, datetime, str]] = None,
    config: Optional[GridStoreConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to read a grid stage.

    Args:
        stage: Stage name (met, aod, tropomi)
        start: Start date
        end: End date
        config: Store config (uses default if None)
        **kwargs: Additional arguments for GridPartitionReader.read()

    Returns:
        DataFrame with stage data
    """
    reader = GridPartitionReader(config=config, stage=stage)
    return reader.read(start=start, end=end, **kwargs)


def read_all_grid_stages(
    partition_date: Union[date, datetime, str],
    config: Optional[GridStoreConfig] = None,
    stages: List[str] = ["met", "aod", "tropomi"],
) -> Dict[str, pd.DataFrame]:
    """
    Read all stages for a single date.

    Args:
        partition_date: Date to read
        config: Store config
        stages: List of stages to read

    Returns:
        Dict mapping stage name to DataFrame
    """
    config = config or get_default_config()
    result = {}
    for stage in stages:
        reader = GridPartitionReader(config=config, stage=stage)
        result[stage] = reader.read_single_date(partition_date)
    return result


def join_grid_stages(
    df_met: pd.DataFrame,
    df_aod: pd.DataFrame,
    df_tropomi: pd.DataFrame,
    join_cols: List[str] = ["cell_id", "date"],
) -> pd.DataFrame:
    """
    Join grid stages on cell_id and date.

    Uses left join from MET (always complete) to AOD/TROPOMI.

    Args:
        df_met: MET stage DataFrame
        df_aod: AOD stage DataFrame
        df_tropomi: TROPOMI stage DataFrame
        join_cols: Columns to join on

    Returns:
        Joined DataFrame
    """
    if df_met.empty:
        logger.warning("MET DataFrame is empty, cannot join")
        return pd.DataFrame()

    # Determine available join columns
    actual_join_cols = [c for c in join_cols if c in df_met.columns]
    if not actual_join_cols:
        # Fall back to lat/lon/date
        actual_join_cols = ["lat", "lon", "date"]
        actual_join_cols = [c for c in actual_join_cols if c in df_met.columns]

    if not actual_join_cols:
        raise ValueError(f"No join columns available. MET columns: {list(df_met.columns)[:10]}...")

    logger.info(f"Joining stages on: {actual_join_cols}")

    # Start with MET as base
    result = df_met.copy()

    # Left join AOD
    if not df_aod.empty:
        aod_join_cols = [c for c in actual_join_cols if c in df_aod.columns]
        if aod_join_cols:
            # Get AOD columns that aren't join columns
            aod_feature_cols = [c for c in df_aod.columns if c not in aod_join_cols]
            result = result.merge(
                df_aod[aod_join_cols + aod_feature_cols],
                on=aod_join_cols,
                how="left",
                suffixes=("", "_aod"),
            )
            logger.info(f"Joined AOD: {len(aod_feature_cols)} feature columns")

    # Left join TROPOMI
    if not df_tropomi.empty:
        trop_join_cols = [c for c in actual_join_cols if c in df_tropomi.columns]
        if trop_join_cols:
            trop_feature_cols = [c for c in df_tropomi.columns if c not in trop_join_cols]
            result = result.merge(
                df_tropomi[trop_join_cols + trop_feature_cols],
                on=trop_join_cols,
                how="left",
                suffixes=("", "_trop"),
            )
            logger.info(f"Joined TROPOMI: {len(trop_feature_cols)} feature columns")

    return result
