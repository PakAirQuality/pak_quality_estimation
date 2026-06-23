#!/usr/bin/env python3
"""
Run Grid Inference from Feature Store
=====================================

Run PM2.5 inference using pre-built grid feature store instead of
processing raw data each time.

This provides much faster inference since features are already computed
and stored in Parquet format.

Usage:
    # Single date inference
    python run_grid_inference_from_store.py \
        --date 2024-03-13 \
        --model best_model_weight/model.joblib \
        --output_dir predictions/

    # Date range inference
    python run_grid_inference_from_store.py \
        --start 2024-03-01 --end 2024-03-31 \
        --model best_model_weight/model.joblib \
        --output_dir predictions/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Setup path for imports
_THIS_DIR = Path(__file__).resolve().parent
_INFERENCE_DIR = _THIS_DIR.parent
_REPO_ROOT = _INFERENCE_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

from .paths import GridStoreConfig, VALID_STAGES
from .reader import GridPartitionReader, join_grid_stages

warnings.filterwarnings("ignore")

# Pakistan boundary GeoJSON path (for proper masking)
_PAKISTAN_GEOJSON = _REPO_ROOT / "deployment" / "data" / "pakistan.geojson"


def _load_pakistan_boundary():
    """Load Pakistan boundary polygon for masking."""
    import json
    from shapely.geometry import shape

    if not _PAKISTAN_GEOJSON.exists():
        logger.warning(f"Pakistan GeoJSON not found: {_PAKISTAN_GEOJSON}")
        return None

    with open(_PAKISTAN_GEOJSON) as f:
        gj = json.load(f)

    if gj['type'] == 'FeatureCollection':
        geom = shape(gj['features'][0]['geometry'])
    elif gj['type'] == 'Feature':
        geom = shape(gj['geometry'])
    else:
        geom = shape(gj)

    return geom


def _create_pakistan_mask(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Create a boolean mask for points inside Pakistan using proper boundary."""
    from shapely.geometry import Point
    from shapely import vectorized

    geom = _load_pakistan_boundary()
    if geom is None:
        logger.warning("Using fallback: no Pakistan boundary mask")
        return np.ones(len(lats), dtype=bool)

    # Use vectorized contains for efficiency
    try:
        # shapely 2.0+ vectorized API
        mask = vectorized.contains(geom, lons, lats)
    except (AttributeError, ImportError):
        # Fallback to loop for older shapely
        mask = np.array([geom.contains(Point(lon, lat)) for lat, lon in zip(lats, lons)])

    return mask
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default features if model doesn't specify
DEFAULT_REQUIRED_FEATURES = [
    "VC", "blh", "WS10", "WS100", "RH", "theta", "u10", "v10", "u100", "v100",
    "tcc", "CLR", "msl", "sp", "t2m", "d2m", "q", "VPD", "dT", "MSLP_tend",
    "SP_tend", "BLH_tend", "VCi", "Stagnant", "HighRH", "WS10_lag1d", "WS10_lag3d",
    "WS10_rollmean_3d", "WS10_rollstd_3d", "WS10_rollmin_7d", "WS10_rollmax_7d",
    "calm3_count", "calm3_flag", "calm7_count", "calm7_flag", "blh_lag1d", "blh_lag3d",
    "blh_rollmean_3d", "blh_rollmin_7d", "blh_rollmean_7d", "blh_anom_7d",
    "doy_sin", "doy_cos", "heating_season_flag", "burning_season_flag",
    "optical_depth_047", "optical_depth_055", "aod_uncertainty",
    "no2_median", "no2_mean", "so2_median", "co_median",
]


def load_model(model_path: str) -> Tuple[object, List[str], Optional[pd.Series]]:
    """
    Load model and extract feature requirements.

    Args:
        model_path: Path to joblib model file (local or GCS path gs://...)

    Returns:
        (model, feature_cols, train_medians)
    """
    logger.info(f"Loading model: {model_path}")

    # Handle GCS paths by downloading to temp file
    if str(model_path).startswith("gs://"):
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ["gsutil", "cp", str(model_path), tmp_path],
            check=True,
            capture_output=True,
        )
        payload = joblib.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
    else:
        payload = joblib.load(model_path)

    if isinstance(payload, dict):
        model = payload.get("model")
        feature_cols = payload.get("feature_cols", DEFAULT_REQUIRED_FEATURES)
        train_medians = payload.get("train_medians")
    else:
        model = payload
        feature_cols = DEFAULT_REQUIRED_FEATURES
        train_medians = None

    logger.info(f"Model loaded: {len(feature_cols)} features required")
    return model, feature_cols, train_medians


def load_features_from_store(
    pred_date: date,
    config: GridStoreConfig,
    required_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load and join features from all stages for a single date.

    Args:
        pred_date: Prediction date
        config: Store configuration
        required_features: List of required feature columns

    Returns:
        (X, coords, stage_dfs) where X is model-ready features,
        coords has lat/lon/cell_id, and stage_dfs maps stage name to raw DataFrame
    """
    logger.info(f"Loading features from store for {pred_date}")

    # Read each stage
    readers = {
        stage: GridPartitionReader(config, stage)
        for stage in VALID_STAGES
    }

    stage_dfs = {}
    for stage, reader in readers.items():
        if config.partition_exists(stage, pred_date):
            df = reader.read_single_date(pred_date)
            stage_dfs[stage] = df
            logger.info(f"  {stage}: {len(df)} rows, {len(df.columns)} cols")
        else:
            logger.warning(f"  {stage}: partition missing")
            stage_dfs[stage] = pd.DataFrame()

    # Join stages
    df_met = stage_dfs.get("met", pd.DataFrame())
    df_aod = stage_dfs.get("aod", pd.DataFrame())
    df_tropomi = stage_dfs.get("tropomi", pd.DataFrame())

    if df_met.empty:
        raise ValueError(f"MET data missing for {pred_date}")

    master = join_grid_stages(df_met, df_aod, df_tropomi)
    logger.info(f"Joined features: {len(master)} rows, {len(master.columns)} cols")

    # Extract coordinates (including Pakistan mask if available)
    coord_cols = ["cell_id", "lat", "lon", "date", "__inpoly"]
    coords = master[[c for c in coord_cols if c in master.columns]].copy()

    # Select required features
    available_features = [f for f in required_features if f in master.columns]
    missing_features = [f for f in required_features if f not in master.columns]

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features: {missing_features[:10]}...")

    X = master[available_features].copy()
    logger.info(f"Feature matrix: {X.shape}")

    return X, coords, stage_dfs


def compute_quality_report(
    X: pd.DataFrame,
    coords: pd.DataFrame,
    required_features: List[str],
    available_features: List[str],
    stage_dfs: Dict[str, pd.DataFrame],
) -> Dict:
    """Compute data quality metrics before imputation.

    Measures NaN coverage within Pakistan cells to assess whether
    satellite/reanalysis data is sufficiently present for reliable inference.

    Reports per-stage NaN fraction (how much of each stage's data is missing)
    and per-stage coverage (fraction of cells with any data at all).
    """
    _META_COLS = {"cell_id", "lat", "lon", "date", "__inpoly"}

    # Use Pakistan mask if available
    if "__inpoly" in coords.columns:
        mask = coords["__inpoly"].values == 1
    else:
        mask = np.ones(len(X), dtype=bool)

    n_cells = int(mask.sum())
    n_features = len(required_features)
    n_missing = n_features - len(available_features)

    # NaN fraction across Pakistan cells only
    if n_cells > 0 and len(available_features) > 0:
        nan_frac = float(X.loc[mask].isna().values.mean())
    else:
        nan_frac = 1.0

    # Per-stage metrics
    stage_coverage = {}   # fraction of PK cells with at least one non-NaN feature
    stage_nan_frac = {}   # mean NaN fraction across PK cells × stage features
    stage_features = {}   # how many model features come from this stage
    required_set = set(required_features)

    for stage, df in stage_dfs.items():
        if df.empty or n_cells == 0:
            stage_coverage[stage] = 0.0
            stage_nan_frac[stage] = 1.0
            stage_features[stage] = 0
            continue

        # Feature columns in this stage (excluding metadata)
        feat_cols = [c for c in df.columns if c not in _META_COLS]
        # Features that the model actually uses from this stage
        model_feat_cols = [c for c in feat_cols if c in required_set]
        stage_features[stage] = len(model_feat_cols)

        if not feat_cols:
            stage_coverage[stage] = 0.0
            stage_nan_frac[stage] = 1.0
            continue

        stage_mask = mask[:len(df)] if len(df) == len(mask) else np.ones(len(df), dtype=bool)
        pk_data = df.loc[stage_mask, feat_cols]

        # Coverage: fraction of PK cells with at least one non-NaN
        stage_coverage[stage] = float(pk_data.notna().any(axis=1).mean())

        # NaN fraction: mean NaN across PK cells × model features from this stage
        if model_feat_cols:
            pk_model_data = df.loc[stage_mask, model_feat_cols]
            stage_nan_frac[stage] = float(pk_model_data.isna().values.mean())
        else:
            stage_nan_frac[stage] = float(pk_data.isna().values.mean())

    return {
        "total_features": n_features,
        "missing_features": n_missing,
        "available_features": len(available_features),
        "nan_fraction": round(nan_frac, 4),
        "pakistan_cells": n_cells,
        "stage_coverage": {k: round(v, 4) for k, v in stage_coverage.items()},
        "stage_nan_fraction": {k: round(v, 4) for k, v in stage_nan_frac.items()},
        "stage_feature_count": stage_features,
    }


def run_inference(
    pred_date: date,
    model,
    config: GridStoreConfig,
    required_features: List[str],
    train_medians: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    """
    Run inference for a single date.

    Args:
        pred_date: Prediction date
        model: Trained model
        config: Store configuration
        required_features: Required feature columns
        train_medians: Training medians for imputation

    Returns:
        (predictions, lats, lons, inpoly_mask, quality_report)
    """
    # Load features
    X, coords, stage_dfs = load_features_from_store(pred_date, config, required_features)

    # Compute quality report before imputation
    available_features = [f for f in required_features if f in X.columns]
    quality_report = compute_quality_report(X, coords, required_features, available_features, stage_dfs)
    snf = quality_report.get("stage_nan_fraction", {})
    sfc = quality_report.get("stage_feature_count", {})
    logger.info(
        f"Quality: {quality_report['available_features']}/{quality_report['total_features']} features, "
        f"NaN={quality_report['nan_fraction']:.1%}, "
        f"MET={sfc.get('met', 0)}f/{snf.get('met', 0):.0%}nan "
        f"AOD={sfc.get('aod', 0)}f/{snf.get('aod', 0):.0%}nan "
        f"TROPOMI={sfc.get('tropomi', 0)}f/{snf.get('tropomi', 0):.0%}nan"
    )

    # Impute missing values
    if train_medians is not None:
        # Handle both dict and Series formats
        if isinstance(train_medians, dict):
            for col in X.columns:
                if col in train_medians:
                    X[col] = X[col].fillna(train_medians[col])
        else:
            for col in X.columns:
                if col in train_medians.index:
                    X[col] = X[col].fillna(train_medians[col])

    # Fill remaining NaN with 0
    X = X.fillna(0.0)

    # Reorder columns to match required features
    X = X.reindex(columns=required_features, fill_value=0.0)

    # Run model
    logger.info(f"Running model prediction...")
    predictions = model.predict(X)

    # Create proper Pakistan mask using GeoJSON boundary (excludes ocean cells)
    lats = coords["lat"].values
    lons = coords["lon"].values
    pakistan_mask = _create_pakistan_mask(lats, lons)

    # Apply mask: set predictions outside Pakistan to NaN
    predictions = np.where(pakistan_mask, predictions, np.nan)
    n_inside = int(pakistan_mask.sum())
    logger.info(f"Applied Pakistan mask: {n_inside}/{len(predictions)} cells inside Pakistan")

    # Ensure predictions are non-negative
    predictions = np.where(np.isfinite(predictions), np.maximum(predictions, 0.0), predictions)

    return predictions, lats, lons, pakistan_mask, quality_report


def export_geotiff(
    predictions: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    pred_date: date,
    output_path: Path,
    grid_shape: Tuple[int, int] = (141, 175),
    nodata: float = -9999.0,
    compress: str = "deflate",
) -> Path:
    """
    Export predictions to GeoTIFF.

    Args:
        predictions: 1D array of predictions
        lats: 1D array of latitudes
        lons: 1D array of longitudes
        pred_date: Prediction date
        output_path: Output file path
        grid_shape: (n_lat, n_lon) grid dimensions
        nodata: NoData value
        compress: Compression method

    Returns:
        Output path
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
    except ImportError:
        logger.error("rasterio required for GeoTIFF export: pip install rasterio")
        raise

    n_lat, n_lon = grid_shape
    n_cells = n_lat * n_lon

    if len(predictions) != n_cells:
        logger.warning(
            f"Prediction count {len(predictions)} != expected {n_cells}, padding"
        )
        padded = np.full(n_cells, nodata, dtype=np.float32)
        padded[:len(predictions)] = predictions
        predictions = padded

    # Reshape to 2D grid
    pm25_grid = predictions.reshape(n_lat, n_lon).astype(np.float32)

    # Compute bounds from lat/lon
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Add half-cell padding for proper georeferencing
    res = 0.1  # Grid resolution
    west = lon_min - res / 2
    east = lon_max + res / 2
    south = lat_min - res / 2
    north = lat_max + res / 2

    transform = from_bounds(west, south, east, north, n_lon, n_lat)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write GeoTIFF
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": n_lon,
        "height": n_lat,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": nodata,
        "compress": compress if compress != "none" else None,
        "tiled": True,
    }

    # Replace NaN with nodata value for GeoTIFF
    pm25_grid = np.where(np.isnan(pm25_grid), nodata, pm25_grid)

    with rasterio.open(output_path, "w", **profile) as dst:
        # Grid is already north-up (row 0 = max lat = north), no flip needed
        dst.write(pm25_grid, 1)

    logger.info(f"Wrote GeoTIFF: {output_path}")
    return output_path


def export_csv(
    predictions: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    pred_date: date,
    output_path: Path,
) -> Path:
    """Export predictions to CSV."""
    df = pd.DataFrame({
        "date": pred_date,
        "lat": lats,
        "lon": lons,
        "pm25": predictions,
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote CSV: {output_path}")
    return output_path


def export_json(
    predictions: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    pred_date: date,
    output_path: Path,
    grid_shape: Tuple[int, int] = (141, 175),
) -> Path:
    """
    Export predictions to JSON format optimized for web display.

    Includes grid metadata, statistics, and flattened values for efficient
    frontend rendering with MapLibre or similar libraries.
    """
    import json

    valid = predictions[np.isfinite(predictions)]

    # Compute statistics
    stats = {
        "mean": float(np.mean(valid)) if len(valid) > 0 else None,
        "median": float(np.median(valid)) if len(valid) > 0 else None,
        "min": float(np.min(valid)) if len(valid) > 0 else None,
        "max": float(np.max(valid)) if len(valid) > 0 else None,
        "p05": float(np.percentile(valid, 5)) if len(valid) > 0 else None,
        "p95": float(np.percentile(valid, 95)) if len(valid) > 0 else None,
        "std": float(np.std(valid)) if len(valid) > 0 else None,
        "n_valid": int(len(valid)),
        "n_total": int(len(predictions)),
    }

    # Grid metadata
    grid = {
        "shape": list(grid_shape),  # [n_lat, n_lon]
        "bounds": {
            "west": float(np.min(lons)),
            "east": float(np.max(lons)),
            "south": float(np.min(lats)),
            "north": float(np.max(lats)),
        },
        "resolution": 0.1,
    }

    # Build output structure
    output = {
        "date": pred_date.isoformat(),
        "grid": grid,
        "stats": stats,
        # Flatten arrays for efficient transfer
        "lats": [round(float(x), 4) for x in lats],
        "lons": [round(float(x), 4) for x in lons],
        "pm25": [round(float(x), 2) if np.isfinite(x) else None for x in predictions],
        # Color scale recommendation (matches create_pm25_video_dynamic.py)
        "colorscale": {
            "name": "turbo",
            "vmin": 20,
            "vmax": 180,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f)

    logger.info(f"Wrote JSON: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run PM2.5 inference from grid feature store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Date selection
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date",
        help="Single prediction date YYYY-MM-DD",
    )
    date_group.add_argument(
        "--start",
        help="Start date for range YYYY-MM-DD (use with --end)",
    )

    parser.add_argument(
        "--end",
        help="End date for range YYYY-MM-DD",
    )

    # Model
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model file (joblib)",
    )

    # Paths
    parser.add_argument(
        "--store_path",
        default="derived/feature_store/grid",
        help="Path to grid feature store",
    )
    parser.add_argument(
        "--output_dir",
        default="predictions",
        help="Output directory for predictions",
    )

    # Output format
    parser.add_argument(
        "--format",
        choices=["geotiff", "csv", "json", "all"],
        default="geotiff",
        help="Output format (default: geotiff). 'all' exports geotiff, csv, and json.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="NoData value for GeoTIFF",
    )
    parser.add_argument(
        "--compress",
        choices=["deflate", "lzw", "none"],
        default="deflate",
        help="GeoTIFF compression",
    )

    # Options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dates
    if args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    else:
        if not args.end:
            parser.error("--end required when using --start")
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        if start > end:
            parser.error("start date must be <= end date")
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    # Setup
    config = GridStoreConfig(base_path=args.store_path)
    output_dir = Path(args.output_dir)
    # Keep model_path as string to preserve GCS URI scheme (gs://)
    model_path = args.model

    # Load model
    model, required_features, train_medians = load_model(model_path)

    logger.info(f"Processing {len(dates)} date(s)")
    logger.info(f"Store path: {config.base_path}")
    logger.info(f"Output dir: {output_dir}")

    # Grid shape (Pakistan at 0.1 deg)
    grid_shape = (141, 175)

    # Process each date
    results = []
    for pred_date in dates:
        try:
            # Check store has data
            missing_stages = [
                s for s in VALID_STAGES
                if not config.partition_exists(s, pred_date)
            ]
            if missing_stages:
                logger.warning(f"{pred_date}: Missing stages {missing_stages}, skipping")
                continue

            # Run inference (returns masked predictions with NaN outside Pakistan)
            predictions, lats, lons, _, quality_report = run_inference(
                pred_date, model, config, required_features, train_medians
            )

            # Write quality sidecar
            quality_path = output_dir / f"quality_{pred_date}.json"
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            with open(quality_path, "w") as f:
                json.dump(quality_report, f, indent=2)
            logger.info(f"Wrote quality report: {quality_path}")

            # Export
            if args.format in ("geotiff", "all"):
                tif_path = output_dir / f"pm25_{pred_date}.tif"
                export_geotiff(
                    predictions, lats, lons, pred_date, tif_path,
                    grid_shape=grid_shape,
                    nodata=args.nodata,
                    compress=args.compress,
                )

            if args.format in ("csv", "all"):
                csv_path = output_dir / f"pm25_{pred_date}.csv"
                export_csv(predictions, lats, lons, pred_date, csv_path)

            if args.format in ("json", "all"):
                json_path = output_dir / f"pm25_{pred_date}.json"
                export_json(
                    predictions, lats, lons, pred_date, json_path,
                    grid_shape=grid_shape,
                )

            # Stats
            valid = predictions[np.isfinite(predictions)]
            results.append({
                "date": pred_date,
                "mean": valid.mean(),
                "median": np.median(valid),
                "max": valid.max(),
                "min": valid.min(),
            })

            logger.info(
                f"{pred_date}: mean={valid.mean():.1f}, "
                f"median={np.median(valid):.1f}, max={valid.max():.1f}"
            )

        except Exception as e:
            logger.error(f"{pred_date}: Failed - {e}")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("INFERENCE COMPLETE")
        print("=" * 60)
        print(f"Dates processed: {len(results)}/{len(dates)}")
        print(f"Output directory: {output_dir}")

        all_means = [r["mean"] for r in results]
        print(f"Overall mean PM2.5: {np.mean(all_means):.1f} ug/m3")
        print("=" * 60)


if __name__ == "__main__":
    main()
