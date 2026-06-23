#!/usr/bin/env python3
"""
Baseline Evaluation
===================

Compare our model against baseline methods:
1. Station-month median (climatological baseline)
2. GEOS-CF CTM (independent chemical transport model)
3. Persistence (yesterday's PM2.5) - with history-enabled model variant

Usage:
    python -m training.evaluate                    # Run all evaluations
    python -m training.evaluate --baselines        # Climatology + GEOS-CF only
    python -m training.evaluate --persistence      # Persistence only
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils.metrics import evaluate_predictions, compute_mae_skill, SEASONS
from training.baselines import (
    GEOSCFBaseline,
    compute_climatology_baseline,
    compute_persistence_baseline,
    add_pm25_lag_features,
)
from training.utils.trainer import load_master_from_lake

DEFAULT_LAKE_PATH = "feature_engineering/output/master_lake"
GCS_LAKE_PATH = "gs://paqi-features-hawanama-data/master_training/master"


def load_data():
    """Load and prepare master data."""
    lake_path = DEFAULT_LAKE_PATH
    if not Path(lake_path).exists():
        lake_path = GCS_LAKE_PATH
        print(f"   Using GCS lake: {lake_path}")
    else:
        print(f"   Using local lake: {lake_path}")

    master = load_master_from_lake(lake_path)
    print(f"   Total records: {len(master):,}")

    # Normalize columns
    if "obs_lat" in master.columns:
        master["lat"] = master["obs_lat"]
    if "obs_lon" in master.columns:
        master["lon"] = master["obs_lon"]
    master["date"] = pd.to_datetime(master["date"])

    return master


def get_feature_cols(df, include_pm25_lags=False):
    """Get feature columns, optionally including PM2.5 lags."""
    exclude_cols = [
        "pm25", "date", "date_utc", "sensor_id", "grid_id", "lat", "lon",
        "obs_lat", "obs_lon", "latitude", "longitude", "month", "year",
        "week", "City", "Name", "time",
    ]

    if not include_pm25_lags:
        exclude_cols += [
            "pm25_lag1d", "pm25_lag2d", "pm25_lag3d", "pm25_lag7d",
            "pm25_rollmean_3d", "pm25_rollmean_7d", "pm25_rollstd_3d",
        ]

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and (include_pm25_lags or not c.startswith("pm25_"))
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    return feature_cols


def train_model(train_data, test_data, feature_cols):
    """Train LightGBM model and return predictions."""
    X_train = train_data[feature_cols].values
    y_train = train_data["pm25"].values
    X_test = test_data[feature_cols].values

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        num_leaves=63, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred


def evaluate_baselines(master):
    """Evaluate model vs climatology and GEOS-CF baselines."""
    print("=" * 70)
    print("BASELINE EVALUATION: Climatology + GEOS-CF")
    print("=" * 70)
    print()
    print("Our model: Satellite-only (no PM2.5 lags)")
    print()

    # Split data
    train_data = master[master["date"].dt.year < 2025].copy()
    test_data = master[master["date"].dt.year == 2025].copy()
    train_data["month"] = train_data["date"].dt.month
    test_data["month"] = test_data["date"].dt.month

    print(f"Train (pre-2025): {len(train_data):,}")
    print(f"Test (2025): {len(test_data):,}")
    print()

    # Climatology baseline
    print("-" * 70)
    print("STATION-MONTH MEDIAN BASELINE")
    print("-" * 70)
    y_clim = compute_climatology_baseline(
        train_data, test_data,
        station_col="sensor_id", target_col="pm25"
    )
    print(f"Climatology predictions: {(~np.isnan(y_clim)).sum():,}")

    # GEOS-CF baseline
    print("-" * 70)
    print("GEOS-CF CTM BASELINE")
    print("-" * 70)
    geos = GEOSCFBaseline(cache_dir="data/geos_cf", use_gcs=True)
    print("Syncing from GCS...")
    sync_result = geos.sync_from_gcs(
        start_date=str(test_data["date"].min().date()),
        end_date=str(test_data["date"].max().date()),
        verbose=False
    )
    print(f"Downloaded: {sync_result['downloaded']}, Cached: {sync_result['already_cached']}")

    y_geoscf = geos.get_station_predictions(
        test_data, lat_col="lat", lon_col="lon", date_col="date", method="nearest"
    )
    print(f"GEOS-CF predictions: {(~np.isnan(y_geoscf)).sum():,}")
    print()

    # Our model (satellite-only)
    print("-" * 70)
    print("OUR MODEL (satellite-only)")
    print("-" * 70)
    feature_cols = get_feature_cols(test_data, include_pm25_lags=False)
    print(f"Features: {len(feature_cols)}")

    model, y_pred = train_model(train_data, test_data, feature_cols)
    y_test = test_data["pm25"].values
    print()

    # Evaluation
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    valid = ~np.isnan(y_test) & ~np.isnan(y_clim) & ~np.isnan(y_geoscf) & ~np.isnan(y_pred)
    print(f"Valid samples: {valid.sum():,}")
    print()

    y_obs = y_test[valid]
    y_m = y_pred[valid]
    y_c = y_clim[valid]
    y_g = y_geoscf[valid]

    print("OUR MODEL (satellite-only):")
    m_model = evaluate_predictions(y_obs, y_m, verbose=True)
    print()

    print("STATION-MONTH MEDIAN (climatological):")
    m_clim = evaluate_predictions(y_obs, y_c, verbose=True)
    print()

    print("GEOS-CF (CTM, 0.25 deg):")
    m_geos = evaluate_predictions(y_obs, y_g, verbose=True)
    print()

    # Skill scores
    print("-" * 70)
    print("SKILL SCORES")
    print("-" * 70)
    skill_vs_clim = compute_mae_skill(m_model["mae"], m_clim["mae"])
    skill_vs_geos = compute_mae_skill(m_model["mae"], m_geos["mae"])
    print(f"Skill vs Climatology: {skill_vs_clim:+.3f} ({skill_vs_clim:+.1%})")
    print(f"Skill vs GEOS-CF:     {skill_vs_geos:+.3f} ({skill_vs_geos:+.1%})")
    print()

    # Summary table
    print("-" * 70)
    print("SUMMARY TABLE")
    print("-" * 70)
    print()
    print("| Method                      | RMSE  | MAE   | R^2   |")
    print("|-----------------------------|-------|-------|-------|")
    print(f"| Our Model (satellite-only)  | {m_model['rmse']:>5.1f} | {m_model['mae']:>5.1f} | {m_model['r2']:>5.3f} |")
    print(f"| Station-month median        | {m_clim['rmse']:>5.1f} | {m_clim['mae']:>5.1f} | {m_clim['r2']:>5.3f} |")
    print(f"| GEOS-CF (CTM)               | {m_geos['rmse']:>5.1f} | {m_geos['mae']:>5.1f} | {m_geos['r2']:>5.3f} |")
    print()


def evaluate_persistence(master):
    """Evaluate model (with lags) vs persistence baseline."""
    print("=" * 70)
    print("PERSISTENCE BASELINE EVALUATION")
    print("=" * 70)
    print()
    print("Persistence baseline: y_hat_t = y_{t-1}")
    print("Our model: WITH PM2.5 lag features (fair comparison)")
    print()

    # Add lag features
    print("-" * 70)
    print("ADDING PM2.5 LAG FEATURES")
    print("-" * 70)
    master = add_pm25_lag_features(master)
    lag_cols = [c for c in master.columns if c.startswith("pm25_lag") or c.startswith("pm25_roll")]
    print(f"Added features: {lag_cols}")
    print()

    # Split data
    train_data = master[master["date"].dt.year < 2025].copy()
    test_data = master[master["date"].dt.year == 2025].copy()
    train_data["month"] = train_data["date"].dt.month
    test_data["month"] = test_data["date"].dt.month

    print(f"Train (pre-2025): {len(train_data):,}")
    print(f"Test (2025): {len(test_data):,}")
    print()

    # Persistence baseline
    print("-" * 70)
    print("PERSISTENCE BASELINE")
    print("-" * 70)
    y_persist = compute_persistence_baseline(
        train_data, test_data,
        station_col="sensor_id", target_col="pm25"
    )
    valid_persist = ~np.isnan(y_persist)
    print(f"Persistence predictions: {valid_persist.sum():,} / {len(y_persist):,}")
    print()

    # Our model (with PM2.5 lags)
    print("-" * 70)
    print("OUR MODEL (with PM2.5 lags)")
    print("-" * 70)
    feature_cols = get_feature_cols(test_data, include_pm25_lags=True)
    pm25_features = [c for c in feature_cols if "pm25" in c.lower()]
    print(f"Total features: {len(feature_cols)}")
    print(f"PM2.5 lag features: {pm25_features}")

    model, y_pred = train_model(train_data, test_data, feature_cols)
    y_test = test_data["pm25"].values
    print()

    # Evaluation
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    valid = ~np.isnan(y_test) & ~np.isnan(y_persist) & ~np.isnan(y_pred)
    print(f"Valid samples: {valid.sum():,}")
    print()

    y_obs = y_test[valid]
    y_m = y_pred[valid]
    y_p = y_persist[valid]

    print("OUR MODEL (with PM2.5 lags):")
    m_model = evaluate_predictions(y_obs, y_m, verbose=True)
    print()

    print("PERSISTENCE (yesterday's PM2.5):")
    m_persist = evaluate_predictions(y_obs, y_p, verbose=True)
    print()

    # Skill score
    print("-" * 70)
    print("SKILL VS PERSISTENCE")
    print("-" * 70)
    skill = compute_mae_skill(m_model["mae"], m_persist["mae"])
    print(f"MAE Skill: {skill:+.3f} ({skill:+.1%})")
    print()

    # Summary table
    print("-" * 70)
    print("SUMMARY TABLE")
    print("-" * 70)
    print()
    print("| Method                      | RMSE  | MAE   | R^2   |")
    print("|-----------------------------|-------|-------|-------|")
    print(f"| Our Model (with lags)       | {m_model['rmse']:>5.1f} | {m_model['mae']:>5.1f} | {m_model['r2']:>5.3f} |")
    print(f"| Persistence (y_{{t-1}})       | {m_persist['rmse']:>5.1f} | {m_persist['mae']:>5.1f} | {m_persist['r2']:>5.3f} |")
    print()

    # Feature importance
    print("-" * 70)
    print("PM2.5 LAG FEATURE IMPORTANCE")
    print("-" * 70)
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    pm25_imp = importances[importances["feature"].str.contains("pm25", case=False)]
    print()
    for _, row in pm25_imp.iterrows():
        pct = 100 * row["importance"] / importances["importance"].sum()
        print(f"  {row['feature']:<25} {pct:>5.1f}%")

    total_pm25_imp = pm25_imp["importance"].sum()
    total_imp = importances["importance"].sum()
    print()
    print(f"Total PM2.5 lag importance: {100*total_pm25_imp/total_imp:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against baselines")
    parser.add_argument("--baselines", action="store_true", help="Run climatology + GEOS-CF evaluation")
    parser.add_argument("--persistence", action="store_true", help="Run persistence evaluation")
    args = parser.parse_args()

    # Default: run all
    run_all = not args.baselines and not args.persistence

    print("-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    master = load_data()
    print()

    if run_all or args.baselines:
        evaluate_baselines(master.copy())
        print()

    if run_all or args.persistence:
        evaluate_persistence(master.copy())

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
