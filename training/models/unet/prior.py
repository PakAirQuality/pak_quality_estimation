"""
PM2.5 Lag-1 Prior Builder
=========================

Gaussian-splatted "yesterday PM2.5" prior from station observations.

For each date, reads the previous day's station observations from master_lake,
then builds a smooth spatial field by splatting each station with a Gaussian
kernel (sigma_km=60, radius_km=180) onto the 0.1° grid.

Returns two channels:
  - pm25_lag1_prior: weighted mean PM2.5 (NaN where no coverage)
  - pm25_lag1_cov:   log1p(sum_weights), 0 where no coverage
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_pm25_lag1_prior_0p1(
    date_str: str,
    master_lake: Path,
    lat_max: float,
    lon_min: float,
    border_0p1: np.ndarray,   # shape (1, H, W), float32 0/1
    H: int,
    W: int,
    res_deg: float = 0.1,
    sigma_km: float = 60.0,
    radius_km: float = 180.0,
    eps: float = 1e-6,
):
    """Build lag-1 PM2.5 prior from yesterday's station observations.

    Parameters
    ----------
    date_str : str
        Current date (YYYY-MM-DD). Reads observations from date-1.
    master_lake : Path
        Root of master_lake hive-partitioned directory.
    lat_max, lon_min : float
        Grid origin (top-left corner).
    border_0p1 : ndarray (1, H, W)
        Pakistan boundary mask at 0.1° (1=inside, 0=outside).
    H, W : int
        Grid dimensions (141, 175).
    res_deg : float
        Grid resolution in degrees (0.1).
    sigma_km : float
        Gaussian kernel standard deviation in km.
    radius_km : float
        Hard cutoff radius in km.
    eps : float
        Minimum weight to consider a pixel covered.

    Returns
    -------
    prior : ndarray (H, W) float32
        Weighted mean PM2.5, NaN where no coverage.
    cov : ndarray (H, W) float32
        log1p(sum_weights), 0 where no coverage or outside border.
    """
    # Lag date
    d = pd.Timestamp(date_str) - pd.Timedelta(days=1)
    prev = d.strftime("%Y-%m-%d")

    def _empty():
        prior = np.full((H, W), np.nan, dtype=np.float32)
        cov = np.zeros((H, W), dtype=np.float32)
        m = border_0p1[0] > 0
        prior[~m] = np.nan
        cov[~m] = 0.0
        return prior, cov

    prev_path = master_lake / f"date={prev}"
    if not prev_path.exists():
        return _empty()

    try:
        obs = pd.read_parquet(prev_path, columns=["obs_lat", "obs_lon", "pm25"])
    except Exception:
        return _empty()

    obs = obs.dropna(subset=["pm25", "obs_lat", "obs_lon"])
    if obs.empty:
        return _empty()

    sum_w = np.zeros((H, W), dtype=np.float32)
    sum_wy = np.zeros((H, W), dtype=np.float32)

    # Row radius in grid cells (upper bound; cols adjusted per-station by cos(lat))
    rad_rows = int(np.ceil(radius_km / (111.0 * res_deg)))

    for lat, lon, y in zip(
        obs["obs_lat"].values, obs["obs_lon"].values, obs["pm25"].values
    ):
        # Continuous center in 0.1° grid
        r0 = (lat_max - lat) / res_deg
        c0 = (lon - lon_min) / res_deg

        ri = int(np.round(r0))
        ci = int(np.round(c0))
        if ri < -rad_rows or ri >= H + rad_rows:
            continue
        if ci < -rad_rows or ci >= W + rad_rows:
            continue

        coslat = np.cos(np.deg2rad(lat))
        coslat = max(coslat, 0.2)
        rad_cols = int(np.ceil(radius_km / (111.0 * coslat * res_deg)))

        r1 = max(0, ri - rad_rows)
        r2 = min(H, ri + rad_rows + 1)
        c1 = max(0, ci - rad_cols)
        c2 = min(W, ci + rad_cols + 1)

        dr = (np.arange(r1, r2) - ri).astype(np.float32)
        dc = (np.arange(c1, c2) - ci).astype(np.float32)

        dy_km = np.abs(dr) * (111.0 * res_deg)
        dx_km = np.abs(dc) * (111.0 * coslat * res_deg)

        # Distance grid → Gaussian weights
        dist = np.sqrt(dy_km[:, None] ** 2 + dx_km[None, :] ** 2)
        w = np.exp(-0.5 * (dist / sigma_km) ** 2).astype(np.float32)
        w[dist > radius_km] = 0.0

        sum_w[r1:r2, c1:c2] += w
        sum_wy[r1:r2, c1:c2] += w * float(y)

    prior = np.full((H, W), np.nan, dtype=np.float32)
    ok = sum_w > eps
    prior[ok] = sum_wy[ok] / sum_w[ok]

    cov = np.log1p(sum_w).astype(np.float32)

    # Apply Pakistan border: no signal outside
    m = border_0p1[0] > 0
    prior[~m] = np.nan
    cov[~m] = 0.0

    return prior, cov
