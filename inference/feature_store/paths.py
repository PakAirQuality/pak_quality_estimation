"""
Grid Feature Store Paths Configuration
======================================

Defines where the grid feature store lives and how partitions are laid out.

Storage Layout:
    {base_path}/{stage}/date={YYYY-MM-DD}/part-000.parquet

Examples:
    Local:  derived/feature_store/grid/met/date=2024-03-13/part-000.parquet
    GCS:    gs://paqi-derived-hawanama-data/grid/met/date=2024-03-13/part-000.parquet

Stages:
    - met: Meteorological features
    - aod: Aerosol Optical Depth features
    - tropomi: TROPOMI satellite features

Static Grid:
    {base_path}/static/pakistan_grid_0p1.parquet
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Union


# Valid stage names for grid store
VALID_STAGES = {"met", "aod", "tropomi"}


@dataclass
class GridStoreConfig:
    """
    Configuration for grid feature store paths.

    Attributes:
        base_path: Root path for the store (local path or gs:// URI)
        partition_format: Date format for partition directories (default: date=%Y-%m-%d)
        grid_resolution: Grid resolution in degrees (default: 0.1)
    """
    base_path: str = "derived/feature_store/grid"
    partition_format: str = "date=%Y-%m-%d"

    # Default GCS paths
    gcs_bucket: str = "paqi-derived-hawanama-data"
    gcs_prefix: str = "grid"

    # Grid-specific attributes
    grid_resolution: float = 0.1
    static_grid_filename: str = "pakistan_grid_0p1.parquet"

    def __post_init__(self):
        """Normalize base path."""
        if not self.base_path.startswith("gs://"):
            self.base_path = str(Path(self.base_path).resolve())

    @classmethod
    def local(cls, base_path: str = "derived/feature_store/grid") -> "GridStoreConfig":
        """Create a local store config."""
        return cls(base_path=base_path)

    @classmethod
    def gcs(cls, bucket: str = "paqi-derived-hawanama-data", prefix: str = "grid") -> "GridStoreConfig":
        """Create a GCS store config."""
        return cls(
            base_path=f"gs://{bucket}/{prefix}",
            gcs_bucket=bucket,
            gcs_prefix=prefix
        )

    @property
    def is_gcs(self) -> bool:
        """Check if this is a GCS path."""
        return self.base_path.startswith("gs://")

    def stage_prefix(self, stage: str) -> str:
        """
        Get the prefix path for a stage.

        Args:
            stage: Feature stage (met, aod, tropomi)

        Returns:
            Full path prefix for the stage

        Example:
            >>> config.stage_prefix("met")
            "derived/feature_store/grid/met"
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}")
        return f"{self.base_path}/{stage}"

    def partition_path(self, stage: str, dt: Union[date, datetime, str]) -> str:
        """
        Get the partition directory path for a stage and date.

        Args:
            stage: Feature stage (met, aod, tropomi)
            dt: Date for the partition

        Returns:
            Full path to the partition directory

        Example:
            >>> config.partition_path("met", date(2024, 3, 13))
            "derived/feature_store/grid/met/date=2024-03-13"
        """
        if isinstance(dt, str):
            dt = datetime.strptime(dt, "%Y-%m-%d").date()
        elif isinstance(dt, datetime):
            dt = dt.date()

        partition_dir = dt.strftime(self.partition_format)
        return f"{self.stage_prefix(stage)}/{partition_dir}"

    def parquet_path(self, stage: str, dt: Union[date, datetime, str], part: int = 0) -> str:
        """
        Get the full path to a parquet file.

        Args:
            stage: Feature stage
            dt: Date for the partition
            part: Part number (default 0)

        Returns:
            Full path to the parquet file

        Example:
            >>> config.parquet_path("met", date(2024, 3, 13))
            "derived/feature_store/grid/met/date=2024-03-13/part-000.parquet"
        """
        return f"{self.partition_path(stage, dt)}/part-{part:03d}.parquet"

    def static_path(self) -> str:
        """
        Get the path to the static grid definition file.

        Returns:
            Full path to the static grid parquet file

        Example:
            >>> config.static_path()
            "derived/feature_store/grid/static/pakistan_grid_0p1.parquet"
        """
        return f"{self.base_path}/static/{self.static_grid_filename}"

    def list_partitions(self, stage: str) -> List[str]:
        """
        List all partition directories for a stage.

        Returns:
            List of partition directory paths
        """
        prefix = self.stage_prefix(stage)

        if self.is_gcs:
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                paths = fs.ls(prefix.replace("gs://", ""))
                return [f"gs://{p}" for p in paths if "/date=" in p]
            except ImportError:
                import subprocess
                result = subprocess.run(
                    ["gsutil", "ls", f"{prefix}/"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return [p.rstrip("/") for p in result.stdout.strip().split("\n") if p]
                return []
        else:
            stage_dir = Path(prefix)
            if not stage_dir.exists():
                return []
            return [str(p) for p in stage_dir.iterdir() if p.is_dir() and "date=" in p.name]

    def partition_exists(self, stage: str, dt: Union[date, datetime, str]) -> bool:
        """
        Check if a partition exists.

        Args:
            stage: Feature stage
            dt: Date to check

        Returns:
            True if the partition directory exists and has parquet files
        """
        partition_dir = self.partition_path(stage, dt)

        if self.is_gcs:
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                gcs_path = partition_dir.replace("gs://", "")
                return fs.exists(gcs_path) and any(
                    f.endswith(".parquet") for f in fs.ls(gcs_path)
                )
            except ImportError:
                import subprocess
                result = subprocess.run(
                    ["gsutil", "ls", f"{partition_dir}/*.parquet"],
                    capture_output=True, text=True
                )
                return result.returncode == 0
        else:
            partition_path = Path(partition_dir)
            return partition_path.exists() and any(partition_path.glob("*.parquet"))

    def static_exists(self) -> bool:
        """Check if the static grid file exists."""
        static_file = self.static_path()
        if self.is_gcs:
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                return fs.exists(static_file.replace("gs://", ""))
            except ImportError:
                import subprocess
                result = subprocess.run(
                    ["gsutil", "ls", static_file],
                    capture_output=True, text=True
                )
                return result.returncode == 0
        else:
            return Path(static_file).exists()


# Convenience functions using default config
_default_config: Optional[GridStoreConfig] = None


def get_default_config() -> GridStoreConfig:
    """Get the default store configuration."""
    global _default_config
    if _default_config is None:
        base_path = os.environ.get("GRID_FEATURE_STORE_PATH", "derived/feature_store/grid")
        _default_config = GridStoreConfig(base_path=base_path)
    return _default_config


def set_default_config(config: GridStoreConfig) -> None:
    """Set the default store configuration."""
    global _default_config
    _default_config = config


def get_stage_prefix(stage: str) -> str:
    """Get stage prefix using default config."""
    return get_default_config().stage_prefix(stage)


def get_partition_path(stage: str, dt: Union[date, datetime, str]) -> str:
    """Get partition path using default config."""
    return get_default_config().partition_path(stage, dt)


def get_parquet_path(stage: str, dt: Union[date, datetime, str], part: int = 0) -> str:
    """Get parquet file path using default config."""
    return get_default_config().parquet_path(stage, dt, part)
