"""
Grid Feature Store ETL Layer
============================

Manages grid-level feature data for Pakistan PM2.5 inference:
- paths.py: Canonical storage layout
- writer.py: Write grid DataFrames to Parquet partitions
- reader.py: Read date-range partitions efficiently
- build_grid_store.py: Build grid stores from raw data (2020-2025)
- run_grid_inference_from_store.py: Parquet -> predict -> GeoTIFF

Storage Layout:
    {base_path}/{stage}/date={YYYY-MM-DD}/part-000.parquet

Examples:
    Local:  derived/feature_store/grid/met/date=2024-03-13/part-000.parquet
    GCS:    gs://your-derived-bucket/grid/met/date=2024-03-13/part-000.parquet
"""

from .paths import GridStoreConfig, VALID_STAGES
from .writer import GridPartitionWriter
from .reader import GridPartitionReader

__all__ = [
    "GridStoreConfig",
    "VALID_STAGES",
    "GridPartitionWriter",
    "GridPartitionReader",
]
