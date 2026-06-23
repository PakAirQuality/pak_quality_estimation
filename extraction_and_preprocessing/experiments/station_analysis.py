#!/usr/bin/env python3
"""
Station Network Analysis
========================

Queries the local station data lake to produce a summary of network
size, active stations, geographic coverage, and data availability.

A station is considered "active" if it has >= MIN_VALID_DAYS days with
a valid PM2.5 reading in the most recent ACTIVE_WINDOW_DAYS days.

Usage:
    python -m extraction_and_preprocessing.experiments.station_analysis
    python extraction_and_preprocessing/experiments/station_analysis.py

    # Custom activity threshold
    python extraction_and_preprocessing/experiments/station_analysis.py \
        --active_window 30 --min_valid_days 10

    # Specify a reference date
    python extraction_and_preprocessing/experiments/station_analysis.py \
        --reference_date 2025-06-01
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LAKE_PATH = (
    Path(__file__).resolve().parent.parent
    / "station_labels"
    / "lake"
    / "station_daily"
)


def load_lake_range(
    lake_path: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load all station-day records between start_date and end_date."""
    frames = []
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    current = start
    while current <= end:
        partition = lake_path / f"date={current.isoformat()}"
        if partition.exists():
            try:
                df = pd.read_parquet(partition)
                if "date" not in df.columns:
                    df["date"] = current.isoformat()
                frames.append(df)
            except Exception:
                pass
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def discover_lake_dates(lake_path: Path) -> list[str]:
    """Return sorted list of available date partition strings."""
    dates = []
    for p in lake_path.iterdir():
        if p.is_dir() and p.name.startswith("date="):
            dates.append(p.name.replace("date=", ""))
    return sorted(dates)


def run_analysis(
    lake_path: Path = LAKE_PATH,
    reference_date: Optional[str] = None,
    active_window_days: int = 30,
    min_valid_days: int = 10,
    min_pm25_hours: int = 18,
) -> dict:
    """
    Run a full station network analysis.

    Args:
        lake_path: Path to the station_daily lake.
        reference_date: Date to analyse from (default: latest available).
        active_window_days: Number of trailing days to evaluate activity.
        min_valid_days: Minimum days with valid PM2.5 data within the
            window for a station to be considered "active."
        min_pm25_hours: Minimum hourly readings in a day for PM2.5 to
            count as a valid observation.

    Returns:
        Dictionary of analysis results.
    """
    if not lake_path.exists():
        raise FileNotFoundError(f"Station lake not found: {lake_path}")

    # Discover available dates
    all_dates = discover_lake_dates(lake_path)
    if not all_dates:
        raise ValueError("No date partitions found in lake")

    earliest = all_dates[0]
    latest = all_dates[-1]

    if reference_date is None:
        reference_date = latest

    ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
    window_start = ref - timedelta(days=active_window_days)

    # ------------------------------------------------------------------
    # 1. Load activity window
    # ------------------------------------------------------------------
    print(f"Loading station data for activity window: "
          f"{window_start.isoformat()} to {reference_date} "
          f"({active_window_days} days) ...")

    df_window = load_lake_range(
        lake_path,
        start_date=window_start.isoformat(),
        end_date=reference_date,
    )

    if df_window.empty:
        raise ValueError(
            f"No data found in activity window "
            f"{window_start.isoformat()} to {reference_date}"
        )

    # Valid PM2.5 day: has a non-null mean AND meets hourly threshold
    df_window["pm25_valid_day"] = (
        df_window["pm25_ugm3_mean"].notna()
        & (df_window["pm25_ugm3_valid_hours"] >= min_pm25_hours)
    )

    # Include provider_name if available
    agg_dict = dict(
        station_name=("station_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        city_name=("city_name", "first"),
        state_name=("state_name", "first"),
        days_present=("date", "nunique"),
        valid_pm25_days=("pm25_valid_day", "sum"),
        mean_pm25=("pm25_ugm3_mean", "mean"),
    )
    has_provider = "provider_name" in df_window.columns
    if has_provider:
        agg_dict["provider_name"] = ("provider_name", "first")

    station_activity = (
        df_window.groupby("id")
        .agg(**agg_dict)
        .reset_index()
    )

    station_activity["active"] = (
        station_activity["valid_pm25_days"] >= min_valid_days
    )

    n_total = len(station_activity)
    n_active = int(station_activity["active"].sum())
    n_inactive = n_total - n_active

    active_stations = station_activity[station_activity["active"]].copy()
    inactive_stations = station_activity[~station_activity["active"]].copy()

    # ------------------------------------------------------------------
    # 2. All-time station count (load first + last year samples)
    # ------------------------------------------------------------------
    print("Scanning network growth ...")

    growth = []
    sample_dates = []
    for year in range(int(earliest[:4]), int(latest[:4]) + 1):
        for month in [1, 7]:
            d = f"{year}-{month:02d}-01"
            if d <= latest:
                sample_dates.append(d)

    for d in sample_dates:
        part = lake_path / f"date={d}"
        if part.exists():
            try:
                tmp = pd.read_parquet(part)
                growth.append({"date": d, "n_stations": tmp["id"].nunique()})
            except Exception:
                pass

    growth_df = pd.DataFrame(growth) if growth else pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Province and city breakdown (active stations)
    # ------------------------------------------------------------------
    province_counts = (
        active_stations.groupby("state_name")["id"]
        .count()
        .sort_values(ascending=False)
    )

    city_counts = (
        active_stations.groupby("city_name")["id"]
        .count()
        .sort_values(ascending=False)
    )

    # ------------------------------------------------------------------
    # 4. Print report
    # ------------------------------------------------------------------
    ref_month = ref.strftime("%B %Y")

    print()
    print("=" * 72)
    print("PAKISTAN AIR QUALITY (PAQi) STATION NETWORK ANALYSIS")
    print("=" * 72)

    print(f"\nData lake span:    {earliest}  to  {latest}")
    print(f"Reference date:    {reference_date}")
    print(f"Activity window:   {window_start.isoformat()} to {reference_date} "
          f"({active_window_days} days)")
    print(f"Active threshold:  >= {min_valid_days} valid PM2.5 days "
          f"(>= {min_pm25_hours} hourly readings/day)")

    print(f"\n--- Station Counts ---")
    print(f"Total stations reporting in window:  {n_total}")
    print(f"Active stations:                     {n_active}")
    print(f"Inactive / intermittent stations:    {n_inactive}")

    # ------------------------------------------------------------------
    # Provider breakdown
    # ------------------------------------------------------------------
    provider_summary = {}
    if has_provider:
        print(f"\n--- Provider Breakdown ---")
        print(f"  {'Provider':<28s} {'Total':>6s} {'Active':>7s} {'Inactive':>9s}"
              f"  {'Avg PM2.5 hrs/day':>18s}")
        print(f"  {'-'*28} {'-'*6} {'-'*7} {'-'*9}  {'-'*18}")

        for prov_name, grp in station_activity.groupby("provider_name"):
            p_total = len(grp)
            p_active = int(grp["active"].sum())
            p_inactive = p_total - p_active
            # Mean valid hours across all station-days for this provider
            prov_mask = df_window["id"].isin(grp["id"])
            avg_hrs = df_window.loc[prov_mask, "pm25_ugm3_valid_hours"].mean()
            avg_hrs_str = f"{avg_hrs:.1f}" if pd.notna(avg_hrs) else "N/A"
            print(f"  {str(prov_name)[:28]:<28s} {p_total:>6d} {p_active:>7d}"
                  f" {p_inactive:>9d}  {avg_hrs_str:>18s}")
            provider_summary[str(prov_name)] = {
                "total": p_total,
                "active": p_active,
                "inactive": p_inactive,
            }
        print()

    print(f"\n--- Citable Summary ---")
    if has_provider and "Punjab EPA" in provider_summary:
        epa = provider_summary["Punjab EPA"]
        lcs_active = n_active - epa["active"]
        lcs_total = n_total - epa["total"]
        print(f'As of {ref_month}, the ground-truth network comprises '
              f'{n_total} registered stations ({n_active} active): '
              f'{epa["total"]} Punjab EPA regulatory monitors '
              f'({epa["active"]} active) and {lcs_total} low-cost sensor '
              f'stations ({lcs_active} active) from the PAQi / Hawanama '
              f'network and affiliated sources.')
    else:
        print(f'As of {ref_month}, the network includes {n_total} '
              f'registered stations, of which {n_active} are considered '
              f'active (>= {min_valid_days} valid observation days in the '
              f'preceding {active_window_days}-day window).')

    if not growth_df.empty:
        print(f"\n--- Network Growth ---")
        for _, row in growth_df.iterrows():
            print(f"  {row['date']}:  {row['n_stations']:>4} stations")

    print(f"\n--- Provincial Breakdown (Active Stations) ---")
    for prov, cnt in province_counts.items():
        print(f"  {prov:<30s} {cnt:>3}")

    print(f"\n--- City Breakdown (Active Stations, top 15) ---")
    for city, cnt in city_counts.head(15).items():
        print(f"  {city:<30s} {cnt:>3}")

    # Geographic bounding box
    if n_active > 0:
        lat_min = active_stations["latitude"].min()
        lat_max = active_stations["latitude"].max()
        lon_min = active_stations["longitude"].min()
        lon_max = active_stations["longitude"].max()
        print(f"\n--- Geographic Extent (Active Stations) ---")
        print(f"  Latitude:   {lat_min:.2f} N  to  {lat_max:.2f} N")
        print(f"  Longitude:  {lon_min:.2f} E  to  {lon_max:.2f} E")

    # Top active stations by data completeness
    top = active_stations.nlargest(10, "valid_pm25_days")
    print(f"\n--- Most Complete Stations (top 10) ---")
    print(f"  {'Station':<35s} {'City':<18s} {'Valid Days':>10s} "
          f"{'Mean PM2.5':>10s}")
    print(f"  {'-'*35} {'-'*18} {'-'*10} {'-'*10}")
    for _, row in top.iterrows():
        pm = row["mean_pm25"]
        pm_str = f"{pm:.1f}" if pd.notna(pm) else "N/A"
        print(f"  {row['station_name'][:35]:<35s} "
              f"{str(row['city_name'])[:18]:<18s} "
              f"{int(row['valid_pm25_days']):>10d} "
              f"{pm_str:>10s}")

    print()
    print("=" * 72)

    return {
        "reference_date": reference_date,
        "lake_span": (earliest, latest),
        "active_window_days": active_window_days,
        "min_valid_days": min_valid_days,
        "min_pm25_hours": min_pm25_hours,
        "n_total": n_total,
        "n_active": n_active,
        "n_inactive": n_inactive,
        "provider_summary": provider_summary,
        "growth": growth_df.to_dict("records") if not growth_df.empty else [],
        "province_counts": province_counts.to_dict(),
        "city_counts": city_counts.to_dict(),
        "active_station_ids": active_stations["id"].tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PAQi station network analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lake_path",
        type=str,
        default=str(LAKE_PATH),
        help="Path to station_daily lake",
    )
    parser.add_argument(
        "--reference_date",
        type=str,
        default=None,
        help="Reference date (YYYY-MM-DD). Default: latest in lake.",
    )
    parser.add_argument(
        "--active_window",
        type=int,
        default=30,
        help="Number of trailing days for activity assessment (default: 30)",
    )
    parser.add_argument(
        "--min_valid_days",
        type=int,
        default=10,
        help="Minimum valid PM2.5 days to be 'active' (default: 10)",
    )
    parser.add_argument(
        "--min_pm25_hours",
        type=int,
        default=18,
        help="Minimum hourly readings for a day to count (default: 18)",
    )

    args = parser.parse_args()

    run_analysis(
        lake_path=Path(args.lake_path),
        reference_date=args.reference_date,
        active_window_days=args.active_window,
        min_valid_days=args.min_valid_days,
        min_pm25_hours=args.min_pm25_hours,
    )
