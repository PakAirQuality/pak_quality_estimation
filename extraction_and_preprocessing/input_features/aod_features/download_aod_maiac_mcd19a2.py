#!/usr/bin/env python3
"""
download_maiac_mcd19a2.py

Robust downloader for MODIS MAIAC AOD:

Product: MCD19A2
Version: 061

What this does:
- Authenticates with NASA Earthdata (via earthaccess)
- Searches granules for a date range + bounding box
- Chunks the search/download by year for stability
- Downloads to a structured local directory
- Writes a manifest CSV per year + a combined manifest

Why chunk by year?
- Avoids giant queries and timeouts
- Makes reruns/resume easy
- Keeps your 2024/2025 production window clean

Requirements:
  pip install earthaccess pandas tqdm

Auth options:
  1) Interactive: The script will open a login flow if needed.
  2) Env vars:
     export EARTHDATA_USERNAME="..."
     export EARTHDATA_PASSWORD="..."

Usage:
  python download_aod_maiac_mcd19a2.py \
      --start 2020-01-01 \
      --end 2025-12-31 \
      --bbox 60 23 78 38 \
      --out_dir raw_data/maiac_aod

Outputs:
  data/maiac_aod/
    MCD19A2.061/
      2020/
      2021/
      ...
      manifests/
        MCD19A2_061_manifest_2020.csv
        ...
        MCD19A2_061_manifest_ALL.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import earthaccess

SHORT_NAME = "MCD19A2"
VERSION = "061"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Download MODIS MAIAC AOD (MCD19A2 v061) from NASA Earthdata."
    )
    ap.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=[60, 23, 78, 38],
        help="Bounding box: west south east north",
    )
    ap.add_argument(
        "--out_dir",
        default="data/maiac_aod",
        help="Base output directory",
    )
    ap.add_argument(
        "--max_results",
        type=int,
        default=1000000,
        help="Safety cap for search results per year",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Only search and write manifests; do not download files.",
    )
    return ap.parse_args()


def daterange_years(start: str, end: str):
    s = datetime.fromisoformat(start).date()
    e = datetime.fromisoformat(end).date()
    years = list(range(s.year, e.year + 1))
    return years


def year_bounds(year: int, global_start: str, global_end: str):
    s = datetime.fromisoformat(global_start).date()
    e = datetime.fromisoformat(global_end).date()
    year_start = max(s, datetime(year, 1, 1).date())
    year_end = min(e, datetime(year, 12, 31).date())
    return year_start.isoformat(), year_end.isoformat()


def granule_to_row(g):
    """
    Convert an earthaccess granule object to a stable manifest row.
    We defensively access fields because metadata can vary slightly.
    """
    umm = getattr(g, "umm", {}) or {}
    title = umm.get("GranuleUR", None) or umm.get("Title", None)
    native_id = umm.get("NativeId", None)

    # Temporal info
    temporal = umm.get("TemporalExtent", {}) or {}
    rng = temporal.get("RangeDateTime", {}) or {}
    begin = rng.get("BeginningDateTime", None)
    end = rng.get("EndingDateTime", None)

    # Provider / collection info
    coll = umm.get("CollectionReference", {}) or {}
    short_name = coll.get("ShortName", SHORT_NAME)
    version = coll.get("Version", VERSION)

    # Online access lists
    urls = []
    for oa in (umm.get("RelatedUrls", []) or []):
        u = oa.get("URL")
        if u:
            urls.append(u)

    return {
        "title": title,
        "native_id": native_id,
        "short_name": short_name,
        "version": version,
        "begin_datetime": begin,
        "end_datetime": end,
        "n_related_urls": len(urls),
    }


def search_year(year: int, start: str, end: str, bbox, max_results: int):
    y_start, y_end = year_bounds(year, start, end)
    if y_start > y_end:
        return []

    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        temporal=(y_start, y_end),
        bounding_box=tuple(bbox),
        count=max_results,
    )
    return results


def ensure_dirs(base: Path, year: int):
    product_dir = base / f"{SHORT_NAME}.{VERSION}"
    year_dir = product_dir / str(year)
    manifest_dir = product_dir / "manifests"
    year_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return product_dir, year_dir, manifest_dir


def main():
    args = parse_args()
    out_base = Path(args.out_dir).resolve()
    years = daterange_years(args.start, args.end)

    # -------- Load environment variables --------
    load_dotenv()

    # -------- Auth --------
    # This will use env vars if set, otherwise interactive auth.
    earthaccess.login()

    all_rows = []

    for year in years:
        product_dir, year_dir, manifest_dir = ensure_dirs(out_base, year)

        print(f"\n=== {SHORT_NAME} v{VERSION} | Year {year} ===")
        print(
            f"Time window: {year_bounds(year, args.start, args.end)[0]} → "
            f"{year_bounds(year, args.start, args.end)[1]}"
        )
        print(f"BBOX: {args.bbox}")
        print(f"Output dir: {year_dir}")

        granules = search_year(year, args.start, args.end, args.bbox, args.max_results)
        print(f"Found {len(granules):,} granules for {year}")

        # Build manifest rows
        year_rows = [granule_to_row(g) for g in granules]
        year_df = pd.DataFrame(year_rows)
        year_manifest_path = manifest_dir / f"{SHORT_NAME}_{VERSION}_manifest_{year}.csv"
        year_df.to_csv(year_manifest_path, index=False)
        print(f"Manifest saved: {year_manifest_path}")

        all_rows.extend(year_rows)

        if args.dry_run:
            print("Dry run enabled: skipping download.")
            continue

        if len(granules) == 0:
            continue

        # -------- Download --------
        # earthaccess will handle URLs + auth + retries internally.
        # It returns local paths of downloaded files.
        print(f"Downloading {len(granules):,} granules to {year_dir} ...")

        # Some environments print a lot; keep a small progress feel here:
        # earthaccess.download doesn't expose per-file callbacks, so we just call it once.
        downloaded = earthaccess.download(granules, str(year_dir))

        # Save local file list for the year
        files_txt = year_dir / f"downloaded_files_{year}.txt"
        with open(files_txt, "w") as f:
            for p in (downloaded or []):
                f.write(str(p) + "\n")
        print(f"Downloaded file list saved: {files_txt}")

    # Combined manifest
    if all_rows:
        combined_df = pd.DataFrame(all_rows)
        combined_manifest_path = (
            Path(args.out_dir)
            / f"{SHORT_NAME}.{VERSION}"
            / "manifests"
            / f"{SHORT_NAME}_{VERSION}_manifest_ALL.csv"
        )
        combined_df.to_csv(combined_manifest_path, index=False)
        print(f"\nCombined manifest saved: {combined_manifest_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
