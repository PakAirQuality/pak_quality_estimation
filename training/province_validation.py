#!/usr/bin/env python3
"""
Spatial Generalization Analysis: Leave-One-Province-Out (LOPO) + Per-Station Metrics
====================================================================================

Demonstrates national spatial generalization by:
1. Assigning stations to provinces via ADM1 spatial join
2. Leave-one-province-out cross-validation (train on N-1 provinces, test on held-out)
3. Per-station metric distributions (worst-case analysis)
4. Province-level performance table for the paper

Uses the Trainer infrastructure and master lake.

Usage:
    python -m training.province_validation
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.utils.trainer import Trainer, load_master_from_lake


# ------------------------------------------------------------------ #
# Province assignment
# ------------------------------------------------------------------ #
def assign_provinces(df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    """Spatial-join stations to ADM1 provinces."""
    gdf_adm = gpd.read_file(geojson_path)
    print(f"Loaded ADM1 boundaries: {sorted(gdf_adm['shapeName'].unique())}")

    geometry = [Point(xy) for xy in zip(df["obs_lon"], df["obs_lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf_adm.crs)
    gdf = gpd.sjoin(gdf, gdf_adm[["shapeName", "geometry"]], how="left", predicate="within")
    gdf["province"] = gdf["shapeName"].fillna("Unknown")
    out = pd.DataFrame(gdf.drop(columns=["geometry", "index_right", "shapeName"]))
    print("Stations per province:")
    for prov, cnt in out.groupby("province")["sensor_id"].nunique().items():
        print(f"  {prov}: {cnt} stations")
    return out


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {}
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(yp - yt))
    extreme = yt >= 150
    n_ext = int(extreme.sum())
    if n_ext > 0:
        f1 = float(f1_score((yt >= 150).astype(int), (yp >= 150).astype(int), zero_division=0))
    else:
        f1 = float("nan")
    return dict(n=len(yt), rmse=rmse, mae=mae, r2=r2, bias=bias, f1_150=f1, n_extreme=n_ext)


# ------------------------------------------------------------------ #
# 1. Province-level metrics on the full-year 2025 held-out test
# ------------------------------------------------------------------ #
def province_test_metrics(trainer: Trainer, geojson_path: str, model_path: str):
    """Evaluate the benchmark model per province on 2025 held-out data."""
    print("\n" + "=" * 80)
    print("PROVINCE-LEVEL METRICS ON 2025 HELD-OUT TEST")
    print("=" * 80)

    # Load saved model
    payload = joblib.load(model_path)
    model = payload["model"]
    feat_cols = payload["feature_cols"]
    med = pd.Series(payload["train_medians"])

    # Get 2025 data
    test_data = trainer.get_year_data([2025])
    X_test, y_test, _, _ = trainer.prepare_features(
        test_data, feature_cols=feat_cols, train_medians=med, fit_medians=False
    )
    y_pred = model.predict(X_test)

    # Assign provinces
    test_data = test_data.copy()
    test_data["y_pred"] = y_pred
    test_data["y_true"] = y_test
    test_data = assign_provinces(test_data, geojson_path)

    # Per-province metrics
    results = {}
    print(f"\n{'Province':<30} {'n':>7} {'Stns':>5} {'MAE':>7} {'RMSE':>7} {'R2':>6} {'Bias':>7} {'F1@150':>7}")
    print("-" * 90)
    for prov in sorted(test_data["province"].unique()):
        mask = test_data["province"] == prov
        m = compute_metrics(test_data.loc[mask, "y_true"].values, test_data.loc[mask, "y_pred"].values)
        if not m:
            continue
        m["n_stations"] = int(test_data.loc[mask, "sensor_id"].nunique())
        results[prov] = m
        f1_str = f"{m['f1_150']:.3f}" if not np.isnan(m.get("f1_150", float("nan"))) else "---"
        print(f"  {prov:<28} {m['n']:>7,} {m['n_stations']:>5} {m['mae']:>7.1f} {m['rmse']:>7.1f} {m['r2']:>6.3f} {m['bias']:>+7.1f} {f1_str:>7}")

    return results, test_data


# ------------------------------------------------------------------ #
# 2. Per-station metric distribution (worst-case analysis)
# ------------------------------------------------------------------ #
def per_station_metrics(test_data: pd.DataFrame):
    """Compute metrics per station; report distribution."""
    print("\n" + "=" * 80)
    print("PER-STATION METRIC DISTRIBUTION (2025 HELD-OUT)")
    print("=" * 80)

    station_results = []
    for sid in test_data["sensor_id"].unique():
        mask = test_data["sensor_id"] == sid
        sub = test_data[mask]
        m = compute_metrics(sub["y_true"].values, sub["y_pred"].values)
        if not m or m["n"] < 30:
            continue
        m["sensor_id"] = sid
        m["province"] = sub["province"].iloc[0]
        station_results.append(m)

    sdf = pd.DataFrame(station_results)
    print(f"\nStations with >=30 obs: {len(sdf)}")
    print(f"\nMAE distribution across stations:")
    print(f"  Median: {sdf['mae'].median():.1f}")
    print(f"  Mean:   {sdf['mae'].mean():.1f}")
    print(f"  P10:    {sdf['mae'].quantile(0.10):.1f}")
    print(f"  P90:    {sdf['mae'].quantile(0.90):.1f}")
    print(f"  Worst:  {sdf['mae'].max():.1f}")
    print(f"\nR2 distribution across stations:")
    print(f"  Median: {sdf['r2'].median():.3f}")
    print(f"  Mean:   {sdf['r2'].mean():.3f}")
    print(f"  P10:    {sdf['r2'].quantile(0.10):.3f}")
    print(f"  P90:    {sdf['r2'].quantile(0.90):.3f}")
    print(f"  Worst:  {sdf['r2'].min():.3f}")

    # Worst 10 stations
    print(f"\nWorst 10 stations by MAE:")
    worst = sdf.nlargest(10, "mae")
    for _, row in worst.iterrows():
        print(f"  {row['province']:<28} station={row['sensor_id']}  MAE={row['mae']:.1f}  R2={row['r2']:.3f}  n={row['n']}")

    return sdf


# ------------------------------------------------------------------ #
# 3. Leave-one-province-out (LOPO) spatially-blocked CV
# ------------------------------------------------------------------ #
def leave_one_province_out_cv(trainer: Trainer, geojson_path: str):
    """
    Full LOPO: for each province with >=2 stations, hold it out entirely,
    train on all other stations (all years), evaluate on the held-out province.
    """
    print("\n" + "=" * 80)
    print("LEAVE-ONE-PROVINCE-OUT CROSS-VALIDATION (ALL YEARS)")
    print("=" * 80)

    full_data = trainer.load_and_prepare_data()
    full_data = assign_provinces(full_data, geojson_path)

    province_stations = full_data.groupby("province")["sensor_id"].nunique()
    valid_provinces = province_stations[province_stations >= 2].index.tolist()
    print(f"\nProvinces with >=2 stations for LOPO: {valid_provinces}")

    lopo_results = {}
    print(f"\n{'Held-out Province':<30} {'n_test':>7} {'Stns':>5} {'MAE':>7} {'RMSE':>7} {'R2':>6} {'Bias':>7} {'F1@150':>7}")
    print("-" * 90)

    for held_out_prov in valid_provinces:
        train_mask = full_data["province"] != held_out_prov
        test_mask = full_data["province"] == held_out_prov

        train_data = full_data[train_mask].copy()
        test_data = full_data[test_mask].copy()

        if len(test_data) < 30:
            continue

        X_train, y_train, feat_cols, med = trainer.prepare_features(train_data, fit_medians=True)
        X_test, y_test, _, _ = trainer.prepare_features(
            test_data, feature_cols=feat_cols, train_medians=med, fit_medians=False
        )

        # Use a small validation split from training data for early stopping
        n_val = min(int(len(y_train) * 0.1), 5000)
        X_val_es = X_train.iloc[-n_val:]
        y_val_es = y_train[-n_val:]

        model = trainer.train_lgbm_model(X_train, y_train, X_val_es, y_val_es)
        y_pred = model.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        if not m:
            continue
        m["n_stations"] = int(test_data["sensor_id"].nunique())
        lopo_results[held_out_prov] = m

        f1_str = f"{m['f1_150']:.3f}" if not np.isnan(m.get("f1_150", float("nan"))) else "---"
        print(f"  {held_out_prov:<28} {m['n']:>7,} {m['n_stations']:>5} {m['mae']:>7.1f} {m['rmse']:>7.1f} {m['r2']:>6.3f} {m['bias']:>+7.1f} {f1_str:>7}")

    # Summary
    if lopo_results:
        maes = [v["mae"] for v in lopo_results.values()]
        r2s = [v["r2"] for v in lopo_results.values()]
        print(f"\nLOPO Summary: mean MAE = {np.mean(maes):.1f}, mean R2 = {np.mean(r2s):.3f}")

    return lopo_results


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spatial generalization analysis")
    parser.add_argument("--lake_path", default="feature_engineering/output/master_lake")
    parser.add_argument("--registry_csv", default="training/data/features_registry.csv")
    parser.add_argument("--geojson", default="inference/output/PAK_ADM1.geojson")
    parser.add_argument("--model_path", default="training/results/benchmark_test/paper_model_20260206_045643.joblib")
    parser.add_argument("--results_dir", default="training/results/spatial_generalization")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        master_lake_path=args.lake_path,
        registry_csv=args.registry_csv,
        results_dir=args.results_dir,
        use_latlon=False,
        strict_registry=True,
    )

    # 1. Province-level metrics on 2025 test set
    prov_metrics, test_data = province_test_metrics(trainer, args.geojson, args.model_path)

    # 2. Per-station distribution
    station_df = per_station_metrics(test_data)
    station_df.to_csv(results_dir / "per_station_metrics_2025.csv", index=False)

    # 3. Leave-one-province-out CV
    lopo_results = leave_one_province_out_cv(trainer, args.geojson)

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "province_test_metrics_2025": prov_metrics,
        "per_station_summary": {
            "n_stations": len(station_df),
            "mae_median": float(station_df["mae"].median()),
            "mae_mean": float(station_df["mae"].mean()),
            "mae_p10": float(station_df["mae"].quantile(0.10)),
            "mae_p90": float(station_df["mae"].quantile(0.90)),
            "mae_worst": float(station_df["mae"].max()),
            "r2_median": float(station_df["r2"].median()),
            "r2_mean": float(station_df["r2"].mean()),
            "r2_p10": float(station_df["r2"].quantile(0.10)),
            "r2_p90": float(station_df["r2"].quantile(0.90)),
            "r2_worst": float(station_df["r2"].min()),
        },
        "lopo_cv": lopo_results,
    }

    # JSON-safe conversion
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    out_path = results_dir / f"spatial_generalization_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(to_jsonable(all_results), f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
