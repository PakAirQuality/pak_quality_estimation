#!/usr/bin/env python3
"""
export_pm25_geotiff.py

Export PakistanPM25Predictor grid predictions to GeoTIFF(s) (EPSG:4326),
with correct orientation (north-up) and proper georeferencing.
Supports single day or date range exports.

Examples:
  # Single day
  python export_pm25_geotiff.py --date 2024-03-13 \
      --model best_model_weight/production_tuned_model_20251228_203856.joblib \
      --master_feature_dir ../inference \
      --out pm25_2024-03-13.tif
  
  # Date range (year-long)
  python export_pm25_geotiff.py --start_date 2024-01-01 --end_date 2024-12-31 \
      --model best_model_weight/production_tuned_model_20251228_203856.joblib \
      --master_feature_dir ../inference \
      --out_dir ./geotiffs_2024/
"""

import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from pakistan_daily_inference import PakistanPM25Predictor

warnings.filterwarnings("ignore")


def _grid_coords_flat(predictor: PakistanPM25Predictor) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback lat/lon if predictor day call fails. Supports multiple generator versions."""
    fg = predictor.feature_generator

    if hasattr(fg, "grid_lats") and hasattr(fg, "grid_lons"):
        lats = np.asarray(fg.grid_lats, dtype="float32").reshape(-1)
        lons = np.asarray(fg.grid_lons, dtype="float32").reshape(-1)
        return lats, lons

    if hasattr(fg, "lat_grid") and hasattr(fg, "lon_grid"):
        lats = np.asarray(fg.lat_grid, dtype="float32").reshape(-1)
        lons = np.asarray(fg.lon_grid, dtype="float32").reshape(-1)
        return lats, lons

    raise AttributeError(
        "Could not find grid coordinates on feature_generator. "
        "Expected grid_lats/grid_lons or lat_grid/lon_grid."
    )


def _infer_geotiff_transform(lat_grid: np.ndarray, lon_grid: np.ndarray):
    """
    Decide transpose/flips so output array is:
      - north-up (row 0 is max-lat)
      - west-left (col 0 is min-lon)
    and return (transpose, flipud, fliplr, bounds).
    
    Enhanced to minimize boundary artifacts and ensure consistent georeferencing.
    """
    eps = 1e-8  # Reduced epsilon for better precision
    latg = np.asarray(lat_grid, dtype=np.float64)  # Use higher precision
    long = np.asarray(lon_grid, dtype=np.float64)

    # More robust transpose detection
    lat_varies_by_col = np.nanstd(latg[0, :]) > eps
    lon_varies_by_row = np.nanstd(long[:, 0]) > eps
    transpose = lat_varies_by_col or lon_varies_by_row
    
    if transpose:
        latg = latg.T
        long = long.T

    # Extract axes with better edge handling
    lat_axis = latg[:, 0].copy()
    lon_axis = long[0, :].copy()
    
    # Remove any NaN values at edges
    lat_axis = lat_axis[np.isfinite(lat_axis)]
    lon_axis = lon_axis[np.isfinite(lon_axis)]

    # For GeoTIFF: want lon increasing left->right
    fliplr = len(lon_axis) > 1 and lon_axis[0] > lon_axis[-1]
    if fliplr:
        lon_axis = lon_axis[::-1]

    # For GeoTIFF: want north at top row (row0 = max lat)
    flipud = len(lat_axis) > 1 and lat_axis[0] < lat_axis[-1]
    
    # Calculate bounds with pixel-edge alignment
    if len(lon_axis) > 1:
        lon_res = np.median(np.abs(np.diff(lon_axis)))
        west = float(lon_axis.min() - lon_res/2)
        east = float(lon_axis.max() + lon_res/2)
    else:
        west = east = float(lon_axis[0]) if len(lon_axis) > 0 else 0.0
        
    if len(lat_axis) > 1:
        lat_res = np.median(np.abs(np.diff(lat_axis)))
        south = float(lat_axis.min() - lat_res/2)
        north = float(lat_axis.max() + lat_res/2)
    else:
        south = north = float(lat_axis[0]) if len(lat_axis) > 0 else 0.0

    return transpose, flipud, fliplr, (west, south, east, north)


def _apply_geotiff_ops(z2d: np.ndarray, transpose: bool, flipud: bool, fliplr: bool) -> np.ndarray:
    z = z2d
    if transpose:
        z = z.T
    if flipud:
        z = np.flipud(z)
    if fliplr:
        z = np.fliplr(z)
    return z


def export_geotiff(
    predictor: PakistanPM25Predictor,
    day: datetime,
    out_path: Path,
    nodata: float = -9999.0,
    compress: str = "deflate",
    verbose: bool = False,
) -> Path:
    """
    Run predictor for `day` and write a single-band GeoTIFF in EPSG:4326.
    """
    import rasterio
    from rasterio.transform import from_bounds

    date_str = day.strftime("%Y-%m-%d")
    out_path = Path(out_path)

    # Predict
    pred_vec, lat_vals, lon_vals, _ = predictor.predict(day, verbose=verbose)
    pred_vec = np.asarray(pred_vec, dtype="float32").reshape(-1)

    # Get grid shape + coords
    grid_shape = predictor.feature_generator.lat_grid.shape

    if lat_vals is None or lon_vals is None:
        lats, lons = _grid_coords_flat(predictor)
    else:
        lats = np.asarray(lat_vals, dtype="float32").reshape(-1)
        lons = np.asarray(lon_vals, dtype="float32").reshape(-1)

    # Safety
    n_expected = int(np.prod(grid_shape))
    if pred_vec.size != n_expected:
        raise ValueError(f"Prediction size mismatch: got {pred_vec.size}, expected {n_expected}")

    pred_grid = pred_vec.reshape(grid_shape)
    lat_grid = lats.reshape(grid_shape)
    lon_grid = lons.reshape(grid_shape)

    # Make north-up + west-left data for GeoTIFF
    transpose, flipud, fliplr, (west, south, east, north) = _infer_geotiff_transform(lat_grid, lon_grid)
    z = _apply_geotiff_ops(pred_grid, transpose, flipud, fliplr).astype("float32")

    # Replace NaN with nodata (GeoTIFF-friendly)
    z_out = z.copy()
    z_out[~np.isfinite(z_out)] = float(nodata)

    height, width = z_out.shape
    transform = from_bounds(west, south, east, north, width=width, height=height)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": float(nodata),
        "compress": compress,
        "tiled": True,
    }

    if verbose:
        print(f"[tiff] date={date_str}")
        print(f"[tiff] shape={height}x{width}")
        print(f"[tiff] bounds: west={west:.4f} south={south:.4f} east={east:.4f} north={north:.4f}")
        print(f"[tiff] transpose={transpose} flipud={flipud} fliplr={fliplr}")
        print(f"[tiff] writing: {out_path}")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(z_out, 1)
        dst.set_band_description(1, f"PM2.5 (ug/m3) prediction {date_str}")

        # Optional tags (nice for QGIS)
        dst.update_tags(
            TITLE="Pakistan Daily PM2.5 Prediction",
            DATE=date_str,
            UNITS="ug/m3",
            MODEL=str(getattr(predictor, "model_path", "")),
        )

    print(f"[tiff] ✓ Saved: {out_path}")
    return out_path


def export_date_range(
    predictor: PakistanPM25Predictor,
    start_date: datetime,
    end_date: datetime,
    out_dir: Path,
    nodata: float = -9999.0,
    compress: str = "deflate",
    verbose: bool = False,
):
    """
    Export PM2.5 predictions for a date range to individual GeoTIFF files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    current_date = start_date
    
    print(f"[batch] Exporting {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"[batch] Output directory: {out_dir}")
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        out_path = out_dir / f"pm25_{date_str}.tif"
        
        try:
            exported_path = export_geotiff(
                predictor=predictor,
                day=current_date,
                out_path=out_path,
                nodata=nodata,
                compress=compress,
                verbose=verbose,
            )
            exported_files.append(exported_path)
            
            if not verbose:
                print(f"[batch] ✓ {date_str}")
                
        except Exception as e:
            print(f"[batch] ✗ {date_str}: {e}")
            
        current_date += timedelta(days=1)
    
    print(f"[batch] Exported {len(exported_files)} files to {out_dir}")
    return exported_files


def main():
    p = argparse.ArgumentParser(description="Export PM2.5 predictions to GeoTIFF (EPSG:4326). Supports single day or date range.")
    
    # Date options - either single date or range
    date_group = p.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date", help="Single date (YYYY-MM-DD)")
    date_group.add_argument("--start_date", help="Start date for range (YYYY-MM-DD)")
    
    p.add_argument("--end_date", help="End date for range (YYYY-MM-DD), required with --start_date")
    
    # Output options
    output_group = p.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--out", help="Output .tif path for single date, e.g. pm25_2024-03-13.tif")
    output_group.add_argument("--out_dir", help="Output directory for date range exports")

    p.add_argument("--model", default="best_model_weight/production_tuned_model_20251228_203856.joblib")
    p.add_argument("--master_feature_dir", default="../inference")
    p.add_argument("--resolution", type=float, default=0.1)
    p.add_argument("--history_days", type=int, default=21)
    p.add_argument("--tropomi_window", type=int, default=5)

    p.add_argument("--nodata", type=float, default=-9999.0)
    p.add_argument("--compress", default="deflate", choices=["deflate", "lzw", "none"])
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    
    # Validate arguments
    if args.start_date and not args.end_date:
        p.error("--end_date is required when using --start_date")
    if args.start_date and not args.out_dir:
        p.error("--out_dir is required when using --start_date")
    if args.date and not args.out:
        p.error("--out is required when using --date")

    predictor = PakistanPM25Predictor(
        model_path=args.model,
        master_feature_dir=args.master_feature_dir,
        grid_resolution=args.resolution,
        tropomi_window=args.tropomi_window,
        history_days=args.history_days,
    )

    compress = None if args.compress == "none" else args.compress
    compress = compress if compress else "deflate"
    
    if args.date:
        # Single day export
        day = datetime.strptime(args.date, "%Y-%m-%d")
        export_geotiff(
            predictor=predictor,
            day=day,
            out_path=Path(args.out),
            nodata=args.nodata,
            compress=compress,
            verbose=args.verbose,
        )
    else:
        # Date range export
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        if start_date > end_date:
            p.error("start_date must be before or equal to end_date")
            
        export_date_range(
            predictor=predictor,
            start_date=start_date,
            end_date=end_date,
            out_dir=Path(args.out_dir),
            nodata=args.nodata,
            compress=compress,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()