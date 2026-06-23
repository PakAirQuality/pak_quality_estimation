#!/usr/bin/env python3
"""
feature_analysis.py

Analyze feature grids to identify diagonal seams and boundary artifacts in PM2.5 GeoTIFF exports.
Tests specific features for seam patterns to determine if the issue comes from MAIAC tile boundaries
or TROPOMI swath boundaries.
"""

import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add inference dir to path
_REPO_ROOT = Path(__file__).resolve().parents[2]  # feature_engineering/visualization/../.. = repo root
sys.path.insert(0, str(_REPO_ROOT / "inference"))

from pakistan_daily_inference import PakistanPM25Predictor

def debug_feature_seams(date_str="2024-01-15"):
    """Debug diagonal seams in feature grids."""
    
    print(f"[debug] Testing features for {date_str}")
    print(f"[debug] Working from: {_REPO_ROOT}")
    
    # Initialize predictor with explicit correct paths
    predictor = PakistanPM25Predictor(
        model_path=str(_REPO_ROOT / "inference/best_model_weight/production_tuned_model_20251228_203856.joblib"),
        master_feature_dir=str(_REPO_ROOT / "inference"),
        aod_dir=str(_REPO_ROOT / "inference/datasets/MCD19A2.061"),  # Explicit AOD path
        tropomi_dir=str(_REPO_ROOT / "inference/datasets/tropomi_pakistan_2020_2025"),  # Explicit TROPOMI path
        geos_cf_dir=str(_REPO_ROOT / "inference/datasets/geos_cf_pakistan_2020_2025"),  # Explicit GEOS-CF path
        grid_resolution=0.1,
        tropomi_window=5,
        history_days=21,
    )
    
    # Generate features for the test date
    test_date = datetime.strptime(date_str, "%Y-%m-%d")
    print(f"[debug] Generating features for {test_date}")
    
    features_df = predictor.feature_generator.generate_daily_features(
        pred_date=test_date,
        history_days=21,
        verbose=True
    )
    
    print(f"[debug] Generated {len(features_df)} grid points with {len(features_df.columns)} features")
    
    # Get grid shape
    shape = predictor.feature_generator.lat_grid.shape  # (H,W)
    print(f"[debug] Grid shape: {shape}")
    
    def show_feature(col):
        """Display a feature grid with diagnostic info."""
        if col not in features_df.columns:
            print(f"[debug] Feature '{col}' not found in dataset")
            return
        
        # Skip non-numeric columns
        if features_df[col].dtype.kind in ['U', 'S', 'O', 'M']:  # string, object, or datetime
            print(f"[debug] Skipping {col} (dtype: {features_df[col].dtype})")
            return
            
        arr = features_df[col].values.reshape(shape)
        
        # Skip if all NaN
        if np.isnan(arr).all():
            print(f"[debug] Skipping {col} (all NaN)")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Raw values
        plt.subplot(2, 2, 1)
        plt.title(f"{col}\n(raw values)")
        im1 = plt.imshow(arr, origin="upper", aspect="auto")
        plt.colorbar(im1, shrink=0.8)
        
        # Deviation from mean to highlight seams
        plt.subplot(2, 2, 2)
        arr_norm = arr - np.nanmean(arr)
        plt.title(f"{col}\n(deviation from mean)")
        im2 = plt.imshow(arr_norm, origin="upper", aspect="auto", cmap="RdBu_r")
        plt.colorbar(im2, shrink=0.8)
        
        # Edge detection to highlight boundaries
        plt.subplot(2, 2, 3)
        from scipy import ndimage
        edges = ndimage.sobel(arr)
        plt.title(f"{col}\n(edge detection)")
        im3 = plt.imshow(edges, origin="upper", aspect="auto", cmap="hot")
        plt.colorbar(im3, shrink=0.8)
        
        # Histogram
        plt.subplot(2, 2, 4)
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) > 0:
            plt.hist(finite_vals, bins=50, alpha=0.7)
            plt.title(f"{col}\n(value distribution)")
            plt.xlabel("Value")
            plt.ylabel("Count")
        
        # Print diagnostics
        print(f"[debug] {col}: shape={arr.shape}, range=[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}], "
              f"mean={np.nanmean(arr):.3f}, std={np.nanstd(arr):.3f}")
        print(f"[debug] {col}: finite_count={np.sum(np.isfinite(arr))}, nan_count={np.sum(np.isnan(arr))}")
        
        plt.tight_layout()
        
        # Save plot instead of showing interactively
        output_dir = Path(__file__).parent / "seam_plots"
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / f"{col}_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[debug] Saved plot: {save_path}")
        plt.close()  # Close to free memory
    
    # Test suspect features in order of likelihood
    test_features = [
        # AOD features (MAIAC tile boundaries)
        "optical_depth_047",
        "optical_depth_055", 
        "aod_uncertainty",
        "qa_cloudmask",
        "qa_adjacency", 
        "qa_aod",
        "qa_n_pixels",
        "aod_total_valid_pixels",
        "aod_files_used",
        "aod_window_size_used",
        
        # TROPOMI features (satellite swath boundaries)
        "no2_median",
        "no2_mean", 
        "no2_std",
        "no2_min",
        "no2_max",
        "no2_n_pixels",
        "no2_window_size_used",
        "no2_window_coverage",
        "no2_radius_km_used",
        "no2_file_available",
        "no2_qa_pass_fraction",
        
        "so2_median",
        "so2_mean",
        "so2_std", 
        "so2_min",
        "so2_max",
        "so2_n_pixels",
        "so2_window_size_used",
        "so2_window_coverage",
        "so2_radius_km_used", 
        "so2_file_available",
        "so2_qa_pass_fraction",
        
        "co_median",
        "hcho_median",
        "o3_median",
    ]
    
    print(f"[debug] Available features ({len(features_df.columns)}): {sorted(features_df.columns)[:20]}...")
    
    # Check which features have actual data (not all NaN and numeric)
    features_with_data = []
    for col in features_df.columns:
        if (features_df[col].dtype.kind in ['i', 'f'] and  # numeric types
            not features_df[col].isna().all()):
            features_with_data.append(col)
    
    print(f"[debug] Features with data ({len(features_with_data)}): {sorted(features_with_data)[:20]}...")
    
    # First, show the suspect features if they exist and have data
    found_features = []
    for feature in test_features:
        if feature in features_df.columns:
            if not features_df[feature].isna().all():
                found_features.append(feature)
                print(f"[debug] Plotting {feature} (has data)")
                show_feature(feature)
            else:
                print(f"[debug] Feature '{feature}' exists but is all NaN")
        else:
            print(f"[debug] Feature '{feature}' not available")
    
    # If no suspect features have data, show any features that do have data
    if not found_features:
        print("[debug] No target features with data found! Showing available features with data:")
        
        # Show some meteorological features for reference
        met_features = ['t2m', 'RH', 'WS10', 'blh', 'sp']
        for col in met_features:
            if col in features_with_data:
                print(f"[debug] Plotting {col} (meteorological reference)")
                show_feature(col)
    
    print(f"[debug] Tested {len(found_features)} target features: {found_features}")
    print("[debug] Look for diagonal seams in the plots above.")
    print("[debug] If AOD features show seams: MAIAC tile boundary issue")
    print("[debug] If TROPOMI features show seams: satellite swath boundary issue")
    
    return features_df, found_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze feature grids for seam patterns")
    parser.add_argument("--date", default="2024-01-15", help="Date to analyze (YYYY-MM-DD)")
    args = parser.parse_args()
    
    features_df, found_features = debug_feature_seams(args.date)