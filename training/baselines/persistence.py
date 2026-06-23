"""
Persistence Baseline (Yesterday's PM2.5)
========================================

Simple autoregressive baseline: ŷ_t = y_{t-1}
"""

import datetime as dt

import numpy as np
import pandas as pd


def compute_persistence_baseline(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    station_col: str = "sensor_id",
    target_col: str = "pm25",
) -> np.ndarray:
    """
    Compute persistence baseline: yesterday's PM2.5 for each station-day.

    Uses strict 1-day lag - gaps produce NaN rather than most recent
    observation (which would artificially strengthen persistence).
    """
    if station_col not in val_data.columns:
        if "grid_id" in val_data.columns:
            station_col = "grid_id"
        else:
            val_data = val_data.copy()
            val_data[station_col] = "all"
            train_data = train_data.copy()
            train_data[station_col] = "all"

    # Combine train + val to get complete observation lookup
    all_data = pd.concat([train_data, val_data], ignore_index=True)
    obs_lookup = (
        all_data[[station_col, "date", target_col]]
        .drop_duplicates(subset=[station_col, "date"])
        .rename(columns={target_col: "pm25_yesterday", "date": "_yesterday_date"})
    )

    val_yd = val_data.copy()
    val_yd["_val_idx"] = np.arange(len(val_yd))
    val_yd["_yesterday_date"] = val_yd["date"].apply(
        lambda d: d - dt.timedelta(days=1)
    )
    val_yd = val_yd.merge(
        obs_lookup, on=[station_col, "_yesterday_date"], how="left"
    )
    val_yd = val_yd.sort_values("_val_idx")

    return val_yd["pm25_yesterday"].values.astype(float)


def add_pm25_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PM2.5 lag features to the dataframe.

    Features added:
    - pm25_lag1d: Yesterday's PM2.5
    - pm25_lag2d: 2 days ago
    - pm25_lag3d: 3 days ago
    - pm25_lag7d: 7 days ago
    - pm25_rollmean_3d: 3-day rolling mean
    - pm25_rollmean_7d: 7-day rolling mean
    - pm25_rollstd_3d: 3-day rolling std
    """
    df = df.sort_values(["sensor_id", "date"]).copy()

    # Group by station for proper lag computation
    for lag in [1, 2, 3, 7]:
        df[f"pm25_lag{lag}d"] = df.groupby("sensor_id")["pm25"].shift(lag)

    # Rolling features (use min_periods=1 to get partial windows)
    df["pm25_rollmean_3d"] = (
        df.groupby("sensor_id")["pm25"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["pm25_rollmean_7d"] = (
        df.groupby("sensor_id")["pm25"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    df["pm25_rollstd_3d"] = (
        df.groupby("sensor_id")["pm25"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
    )

    return df
