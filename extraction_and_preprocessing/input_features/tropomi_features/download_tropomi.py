#!/usr/bin/env python3
"""
download_atmos_products_pakistan_2020_2025.py

Single combined downloader for:
  TROPOMI (S5P OFFL L3): AAI, ALH, CH4, CLOUD, CO, HCHO, NO2, O3, SO2
  Optional (if requested): GEOS-CF NH3

Key points
- Uses Earth Engine getDownloadURL (synchronous) and streams GeoTIFF to disk.
- Auto-retries with coarser scales to fit Earth Engine ~48MB sync download cap.
- Exports "single" mode by default:
    band 1 = primary science band (per product)
    band 2 = qa_value if available
  So your extractor can safely use --data_band 1 and --qa_band 2 across products.
- If you choose --export_mode full, it exports extra bands for some products
  (CLOUD/CH4/ALH/O3), but that may increase file sizes; scale fallback helps.

One-time setup:
  earthengine authenticate
  earthengine set_project pak-climate-risk   # or your EE project id

Example:
  python download_atmos_products_pakistan_2020_2025.py \
    --start 2020-01-01 --end 2025-12-31 \
    --out_root tropomi_pakistan_2020_2025 \
    --products AAI,ALH,CH4,CLOUD,CO,HCHO,NO2,O3,SO2 \
    --buffer_km 30 \
    --region_mode bbox \
    --export_mode single \
    --scales 1113.2,1500,2000,3000,5000,8000 \
    --overwrite 0
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ee
import requests


# -----------------------------
# Product specifications
# -----------------------------
# NOTE: We deliberately define PRIMARY band candidates so band-1 is stable across exports.
# We also include qa_value when present so band-2 is QA in "single" mode.
PRODUCTS: Dict[str, Dict] = {
    # TROPOMI
    "AAI": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_AER_AI",
        "primary_candidates": ["absorbing_aerosol_index"],
        "qa_candidates": ["qa_value"],
        "extras_full": [],
        "prefix": "tropomi_aai_pakistan_",
    },
    "ALH": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_AER_LH",
        "primary_candidates": ["aerosol_height"],
        "qa_candidates": ["qa_value"],
        "extras_full": ["aerosol_pressure", "aerosol_optical_depth"],
        "prefix": "tropomi_alh_pakistan_",
    },
    "CH4": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_CH4",
        "primary_candidates": [
            "CH4_column_volume_mixing_ratio_dry_air_bias_corrected",
            "CH4_column_volume_mixing_ratio_dry_air",
        ],
        "qa_candidates": ["qa_value"],
        "extras_full": [
            "CH4_column_volume_mixing_ratio_dry_air_uncertainty",
            "aerosol_height",
            "aerosol_optical_depth",
        ],
        "prefix": "tropomi_ch4_pakistan_",
    },
    "CLOUD": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_CLOUD",
        "primary_candidates": ["cloud_fraction"],
        "qa_candidates": ["qa_value"],
        "extras_full": [
            "cloud_top_pressure",
            "cloud_top_height",
            "cloud_base_pressure",
            "cloud_base_height",
            "cloud_optical_depth",
            "surface_albedo",
        ],
        "prefix": "tropomi_cloud_pakistan_",
    },
    "CO": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "primary_candidates": ["CO_column_number_density"],
        "qa_candidates": ["qa_value"],
        "extras_full": [],
        "prefix": "tropomi_co_pakistan_",
    },
    "HCHO": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_HCHO",
        "primary_candidates": ["tropospheric_HCHO_column_number_density", "HCHO_column_number_density"],
        "qa_candidates": ["qa_value"],
        "extras_full": [],
        "prefix": "tropomi_hcho_pakistan_",
    },
    "NO2": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "primary_candidates": ["tropospheric_NO2_column_number_density", "NO2_column_number_density"],
        "qa_candidates": ["qa_value"],
        "extras_full": ["stratospheric_NO2_column_number_density"],
        "prefix": "tropomi_no2_pakistan_",
    },
    "O3": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_O3",
        "primary_candidates": ["O3_column_number_density"],
        "qa_candidates": ["qa_value"],
        "extras_full": ["O3_effective_temperature"],
        "prefix": "tropomi_o3_pakistan_",
    },
    "SO2": {
        "kind": "tropomi",
        "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
        "primary_candidates": ["SO2_column_number_density"],
        "qa_candidates": ["qa_value"],
        "extras_full": [],
        "prefix": "tropomi_so2_pakistan_",
    },
    # Optional GEOS-CF
    "NH3": {
        "kind": "geoscf",
        "collection": "NASA/GEOS-CF/v1/rpl/tavg1hr",
        "primary_candidates": ["NH3"],
        "qa_candidates": [],
        "extras_full": [],
        "prefix": "geos_cf_nh3_pakistan_",
        "default_scale_m": 27750.0,  # GEOS-CF native-ish pixel size
    },
}


# -----------------------------
# Helpers
# -----------------------------
def daterange(start: date, end: date):
    cur = start
    one = timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def _is_ee_size_limit_error(e: Exception) -> bool:
    msg = str(e)
    return ("Total request size" in msg) and ("must be less than or equal to" in msg)


def _pick_bands_for_product(
    prod: str,
    export_mode: str,
) -> Tuple[List[str], List[str]]:
    """
    Returns (selected_bands, available_bands).
    Uses EE to introspect bandNames() from a representative image.

    export_mode:
      - single: primary (+ qa_value if exists)
      - full  : primary + extras_full (+ qa_value if exists)
    """
    spec = PRODUCTS[prod]
    ic = ee.ImageCollection(spec["collection"])

    first = ic.limit(1).first()
    if first is None:
        return [], []

    available: List[str] = list(first.bandNames().getInfo() or [])

    def pick_first(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in available:
                return c
        return None

    primary = pick_first(spec.get("primary_candidates", []))
    if primary is None and available:
        primary = available[0]  # last-resort fallback

    qa = pick_first(spec.get("qa_candidates", []))

    bands: List[str] = []
    if primary:
        bands.append(primary)

    if export_mode == "full":
        for b in spec.get("extras_full", []):
            if b in available and b not in bands:
                bands.append(b)

    # include QA second if present (keeps extractor's assumption stable)
    if qa and qa not in bands:
        bands.append(qa)

    return bands, available


def _safe_daily_image_mean(
    ic: ee.ImageCollection, day_str: str, next_day_str: str
) -> Optional[ee.Image]:
    day_ic = ic.filterDate(day_str, next_day_str)
    n = int(day_ic.size().getInfo())
    if n == 0:
        return None
    return day_ic.mean()


def _download_ee_image_with_scale_fallback(
    img: ee.Image,
    out_path: Path,
    region: ee.Geometry,
    scales: List[float],
    crs: str = "EPSG:4326",
    max_stream_timeout_s: int = 900,
):
    if out_path.exists():
        return

    last_err: Optional[Exception] = None

    for scale in scales:
        try:
            url = img.getDownloadURL(
                {
                    "scale": float(scale),
                    "crs": crs,
                    "region": region,
                    "format": "GEO_TIFF",
                }
            )
            resp = requests.get(url, stream=True, timeout=(30, max_stream_timeout_s))
            resp.raise_for_status()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return

        except Exception as e:
            last_err = e
            if _is_ee_size_limit_error(e):
                continue
            raise

    raise RuntimeError(f"Still too large even at coarsest scale={scales[-1]}. Last error: {last_err}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project", default="pak-climate-risk", help="Earth Engine project id")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")

    ap.add_argument("--out_root", required=True, type=Path, help="Output root for all products")
    ap.add_argument(
        "--products",
        default="AAI,ALH,CH4,CLOUD,CO,HCHO,NO2,O3,SO2",
        help="Comma list. You can include NH3 to also download GEOS-CF NH3.",
    )

    ap.add_argument("--buffer_km", type=float, default=30.0, help="Buffer around Pakistan geometry")
    ap.add_argument(
        "--region_mode",
        choices=["bbox", "polygon"],
        default="bbox",
        help="bbox = download bounding box of buffered Pakistan (no masking). "
             "polygon = clip to buffered Pakistan polygon (masked outside).",
    )

    ap.add_argument(
        "--export_mode",
        choices=["single", "full"],
        default="single",
        help="single = primary (+ qa_value if present). full = primary + extras + qa_value.",
    )

    ap.add_argument(
        "--scales",
        default="1113.2,1500,2000,3000,5000,8000",
        help="Scale ladder (meters) used for size-limit fallback.",
    )

    ap.add_argument("--overwrite", type=int, default=0, help="1 to overwrite existing files")

    args = ap.parse_args()

    # EE init
    ee.Initialize(project=args.project)

    # Pakistan geometry (GAUL ADM0)
    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    pak = countries.filter(ee.Filter.eq("ADM0_NAME", "Pakistan")).geometry()
    region = pak.buffer(float(args.buffer_km) * 1000.0)

    # region param for download
    if args.region_mode == "bbox":
        region_param = region.bounds()
        clip_image = False
    else:
        region_param = region
        clip_image = True

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    products = [p.strip().upper() for p in args.products.split(",") if p.strip()]
    unknown = [p for p in products if p not in PRODUCTS]
    if unknown:
        raise SystemExit(f"Unknown products: {unknown}. Allowed: {sorted(PRODUCTS.keys())}")

    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    if not scales:
        raise SystemExit("No scales provided.")

    # For GEOS-CF NH3, if user uses the same ladder, it's okay but wasteful.
    # We'll override with a safer default if NH3 is requested and user kept only fine scales.
    def scales_for(prod: str) -> List[float]:
        if prod == "NH3":
            base = PRODUCTS["NH3"].get("default_scale_m", 27750.0)
            # ensure we include a coarse-enough ladder
            cand = [base, base * 1.5, base * 2.0, base * 3.0]
            return sorted(set([float(x) for x in cand + scales]))
        return scales

    print(f"[EE] project      : {args.project}")
    print(f"[dates]          : {start_d} .. {end_d}")
    print(f"[out_root]       : {args.out_root.resolve()}")
    print(f"[products]       : {', '.join(products)}")
    print(f"[buffer_km]      : {args.buffer_km}")
    print(f"[region_mode]    : {args.region_mode}")
    print(f"[export_mode]    : {args.export_mode}")
    print(f"[scale ladder]   : {scales}")

    # Build per-product ImageCollections with stable band ordering
    ic_by_prod: Dict[str, ee.ImageCollection] = {}
    bands_by_prod: Dict[str, List[str]] = {}

    print("\n[scan] Resolving bands per product (so band-1 stays primary)...")
    for prod in products:
        spec = PRODUCTS[prod]
        sel_bands, avail = _pick_bands_for_product(prod, args.export_mode)
        if not sel_bands:
            print(f"  - {prod}: could not resolve bands (available={avail}). Will likely skip.")
            continue
        bands_by_prod[prod] = sel_bands
        ic_by_prod[prod] = ee.ImageCollection(spec["collection"]).select(sel_bands)
        print(f"  - {prod}: bands={sel_bands}")

    # Download loop
    for d in daterange(start_d, end_d):
        day_str = d.isoformat()
        next_day_str = (d + timedelta(days=1)).isoformat()
        print(f"\n=== {day_str} ===")

        for prod in products:
            spec = PRODUCTS[prod]
            out_dir = args.out_root / prod
            out_name = f"{spec['prefix']}{day_str}.tif"
            out_path = out_dir / out_name

            if out_path.exists() and not args.overwrite:
                print(f"  -> {prod}: exists, skipping ({out_name})")
                continue
            if out_path.exists() and args.overwrite:
                out_path.unlink(missing_ok=True)

            ic = ic_by_prod.get(prod)
            if ic is None:
                print(f"  -> {prod}: no ImageCollection configured; skipping")
                continue

            try:
                img = _safe_daily_image_mean(ic, day_str, next_day_str)
                if img is None:
                    print(f"  -> {prod}: no granules (normal for some products), skipping")
                    continue

                if clip_image:
                    img = img.clip(region)

                print(f"  -> {prod}: downloading ...")
                _download_ee_image_with_scale_fallback(
                    img=img,
                    out_path=out_path,
                    region=region_param,
                    scales=scales_for(prod),
                )
                print(f"     saved: {out_path}")

            except Exception as e:
                print(f"  !! {prod} failed on {day_str}: {e}")

    print("\n✅ All downloads attempted.")


if __name__ == "__main__":
    main()
