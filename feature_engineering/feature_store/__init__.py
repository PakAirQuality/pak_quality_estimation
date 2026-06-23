"""
Feature Store ETL Layer
=======================

This module provides a clean ETL layer for managing feature data:

- paths.py: Canonical storage layout and path helpers
- writer.py: Write DataFrames to Parquet partitions (idempotent)
- reader.py: Read date-range partitions efficiently
- build_station_store.py: Build station-level feature stores from raw data
- build_master_from_store.py: Join stores into master dataset

Usage:
    # Build station stores from raw data
    python -m feature_store.build_station_store --stages met,aod,tropomi ...

    # Build master dataset from stores
    python -m feature_store.build_master_from_store --start 2020-01-01 --end 2025-07-01
"""

from .paths import StoreConfig, get_partition_path, get_stage_prefix
from .writer import PartitionWriter
from .reader import PartitionReader

__all__ = [
    "StoreConfig",
    "get_partition_path",
    "get_stage_prefix",
    "PartitionWriter",
    "PartitionReader",
]
