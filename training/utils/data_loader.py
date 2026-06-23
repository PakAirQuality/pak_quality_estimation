"""
Data Loader (Layer 1)
=====================

Fetch raw master data from local or GCS lake. Nothing else.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_master_from_lake(
    lake_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load master dataset from partitioned parquet (lake format).

    Args:
        lake_path: Path to the lake directory (local or gs:// URI)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Raw DataFrame with whatever columns exist in the lake
    """
    lake_path_str = str(lake_path)
    is_gcs = lake_path_str.startswith("gs://")

    if is_gcs:
        return _load_master_from_gcs(lake_path_str, start_date, end_date)
    else:
        return _load_master_from_local(Path(lake_path_str), start_date, end_date)


def _load_master_from_gcs(
    lake_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load master data from GCS path."""
    import gcsfs
    import pyarrow.parquet as pq

    fs = gcsfs.GCSFileSystem()
    gcs_path = lake_path.replace("gs://", "")

    # Try with /master subdirectory first
    master_path = f"{gcs_path}/master" if not gcs_path.endswith("/master") else gcs_path
    try:
        contents = fs.ls(master_path)
    except FileNotFoundError:
        master_path = gcs_path
        contents = fs.ls(master_path)

    # Find date partitions
    date_dirs = sorted([f"gs://{p}" for p in contents if "/date=" in p])

    if not date_dirs:
        raise FileNotFoundError(f"No date partitions found in: gs://{master_path}")

    print(f"Found {len(date_dirs)} date partitions in lake")

    # Filter partitions by date range
    filtered_dirs = []
    for d in date_dirs:
        date_part = d.split("/date=")[-1].split("/")[0]
        try:
            if start_date and date_part < start_date:
                continue
            if end_date and date_part > end_date:
                continue
            filtered_dirs.append(d)
        except ValueError:
            print(f"Warning: Skipping invalid date partition: {d}")
            continue

    if not filtered_dirs:
        raise FileNotFoundError(
            f"No partitions found in date range {start_date} to {end_date}"
        )

    print(f"Loading {len(filtered_dirs)} partitions after date filtering")

    # Load all parquet files
    dfs = []
    for partition_dir in filtered_dirs:
        gcs_partition = partition_dir.replace("gs://", "")
        try:
            parquet_files = [p for p in fs.ls(gcs_partition) if p.endswith(".parquet")]
            for pf in parquet_files:
                try:
                    with fs.open(pf, 'rb') as f:
                        table = pq.read_table(f)
                        df = table.to_pandas()
                        dfs.append(df)
                except Exception as e:
                    print(f"Warning: Failed to read {pf}: {e}")
        except Exception as e:
            print(f"Warning: Failed to list {partition_dir}: {e}")

    if not dfs:
        raise ValueError("No data loaded from lake partitions")

    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_combined)} rows from lake")

    return df_combined


def _load_master_from_local(
    lake_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load master data from local path."""
    master_dir = lake_path / "master"

    if not master_dir.exists():
        master_dir = lake_path

    if not master_dir.exists():
        raise FileNotFoundError(f"Lake directory not found: {lake_path}")

    # List all date partitions
    date_dirs = sorted([
        d for d in master_dir.iterdir()
        if d.is_dir() and d.name.startswith("date=")
    ])

    if not date_dirs:
        raise FileNotFoundError(f"No date partitions found in: {master_dir}")

    print(f"Found {len(date_dirs)} date partitions in lake")

    # Filter partitions by date range
    filtered_dirs = []
    for d in date_dirs:
        date_str = d.name.replace("date=", "")
        try:
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            filtered_dirs.append(d)
        except ValueError:
            print(f"Warning: Skipping invalid date partition: {d.name}")
            continue

    if not filtered_dirs:
        raise FileNotFoundError(
            f"No partitions found in date range {start_date} to {end_date}"
        )

    print(f"Loading {len(filtered_dirs)} partitions after date filtering")

    # Load all parquet files
    dfs = []
    for partition_dir in filtered_dirs:
        parquet_files = list(partition_dir.glob("*.parquet"))
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to read {pf}: {e}")

    if not dfs:
        raise ValueError("No data loaded from lake partitions")

    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_combined)} rows from lake")

    return df_combined
