"""
Climatological Baseline (Station-Month Median)
===============================================

Reference: Mason (2004) "On using climatology as a reference strategy
           in the Brier and ranked probability skill scores"
"""

import numpy as np
import pandas as pd


def compute_station_month_medians(
    train_data: pd.DataFrame,
    station_col: str = "sensor_id",
    target_col: str = "pm25",
) -> pd.DataFrame:
    """
    Compute median PM2.5 for each (station, month) from training data.

    This is the "climatological" baseline: What is PM2.5 typically like
    at this station in this month? Uses median (robust to spikes).
    """
    if station_col not in train_data.columns:
        if "grid_id" in train_data.columns:
            station_col = "grid_id"
        else:
            train_data = train_data.copy()
            train_data[station_col] = "all"

    medians = (
        train_data
        .groupby([station_col, "month"])[target_col]
        .median()
        .reset_index()
        .rename(columns={target_col: "pm25_median"})
    )
    return medians


def compute_climatology_baseline(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    station_col: str = "sensor_id",
    target_col: str = "pm25",
) -> np.ndarray:
    """
    Compute climatological baseline predictions for validation data.

    For each (station, month) in val_data, returns the median PM2.5
    from that station-month in training data. Falls back to global
    monthly median if station not seen in training.
    """
    if station_col not in val_data.columns:
        if "grid_id" in val_data.columns:
            station_col = "grid_id"
        else:
            val_data = val_data.copy()
            val_data[station_col] = "all"
            train_data = train_data.copy()
            train_data[station_col] = "all"

    station_month_medians = compute_station_month_medians(
        train_data, station_col, target_col
    )

    val_with_median = val_data.merge(
        station_month_medians,
        on=[station_col, "month"],
        how="left"
    )

    # Fallback to global monthly median for unseen stations
    global_month_medians = (
        train_data.groupby("month")[target_col].median().to_dict()
    )
    global_median = train_data[target_col].median()

    val_with_median["pm25_median"] = val_with_median.apply(
        lambda row: row["pm25_median"]
            if pd.notna(row["pm25_median"])
            else global_month_medians.get(row["month"], global_median),
        axis=1
    )

    return val_with_median["pm25_median"].values
