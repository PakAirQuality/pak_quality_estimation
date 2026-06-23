#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------
# Default paths (relative to script)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_SENSOR_CSV = DATA_DIR / "paqi_with_all_features.csv"
DEFAULT_PROVINCE_GEOJSON = DATA_DIR / "PAK_ADM1.geojson"
DEFAULT_PROVINCE_NAME_COL = "shapeName"

# -----------------------------
# Configured feature lists (from your Trainer)
# -----------------------------
SPATIAL_FEATURES = ["obs_lat", "obs_lon"]

METEOROLOGY_FEATURES = [
    "VC", "blh", "WS10", "WS100", "RH", "theta",
    "u10", "v10", "u100", "v100", "tcc", "CLR",
    "msl", "sp", "t2m", "d2m", "q", "VPD", "dT",
    "MSLP_tend", "SP_tend", "BLH_tend",

    "VCi", "Stagnant", "HighRH",

    "WS10_lag1d", "WS10_lag3d",
    "WS10_rollmean_3d", "WS10_rollstd_3d",
    "WS10_rollmin_7d", "WS10_rollmax_7d",
    "calm3_count", "calm3_flag",
    "calm7_count", "calm7_flag",

    "blh_lag1d", "blh_lag3d",
    "blh_rollmean_3d", "blh_rollmin_7d",
    "blh_rollmean_7d", "blh_anom_7d",
    "blh_rollmean_14d", "blh_anom_14d",

    "RH_rollmean_3d", "RH_rollmax_7d",
    "RH_rollmean_7d", "RH_anom_7d",
    "RH_rollstd_7d",

    "VPD_rollmean_3d", "VPD_rollmean_7d",

    "VC_rollmean_3d", "VC_rollmin_7d",
    "VC_rollmean_7d", "VC_anom_7d",
    "VC_rollmean_14d", "VC_anom_14d",

    "stagnant3_count", "stagnant3_flag",
    "stagnant7_count", "stagnant7_flag",

    "WD10", "WD10_sin", "WD10_cos",
    "WD10_sin_rm_7d", "WD10_cos_rm_7d", "WD10_var_7d",
    "WD10_sin_rm_14d", "WD10_cos_rm_14d", "WD10_var_14d",

    "dWS", "dWS_abs", "dWS_rollmean_3d", "dWS_rollstd_3d",
    "dWD", "dWD_abs", "dWD_rollstd_3d",

    "BLH_tend_rollmean_3d", "BLH_tend_rollstd_3d",
    "BLH_tend_rollmean_7d", "BLH_tend_rollstd_7d",
    "MSLP_tend_rollmean_3d", "MSLP_tend_rollstd_3d",
    "MSLP_tend_rollmean_7d", "MSLP_tend_rollstd_7d",
    "SP_tend_rollmean_3d", "SP_tend_rollstd_3d",
    "SP_tend_rollmean_7d", "SP_tend_rollstd_7d",
    "dT_rollmean_3d", "dT_rollstd_3d",
    "dT_rollmean_7d", "dT_rollstd_7d",

    "doy_sin", "doy_cos",
    "doy_sin_2", "doy_cos_2",
    "doy_sin_3", "doy_cos_3",
    "heating_season_flag", "burning_season_flag",
]

# Elevation and coast features removed

AOD_FEATURES = ["optical_depth_047", "optical_depth_055", "aod_uncertainty"]
QA_FEATURES = ["qa_cloudmask", "qa_adjacency", "qa_aod", "qa_n_pixels"]
AOD_QUALITY_FEATURES = ["aod_total_valid_pixels", "aod_files_used", "aod_window_size_used"]

NO2_FEATURES = ["no2_median", "no2_mean", "no2_std", "no2_min", "no2_max"]
NO2_QUALITY_FEATURES = [
    "no2_n_pixels", "no2_window_size_used", "no2_window_coverage",
    "no2_radius_km_used", "no2_file_available", "no2_qa_pass_fraction",
]

SO2_FEATURES = ["so2_median", "so2_mean", "so2_std", "so2_min", "so2_max"]
SO2_QUALITY_FEATURES = [
    "so2_n_pixels", "so2_window_size_used", "so2_window_coverage",
    "so2_radius_km_used", "so2_file_available", "so2_qa_pass_fraction",
]

CO_FEATURES = ["co_median", "co_mean", "co_std", "co_min", "co_max"]
CO_QUALITY_FEATURES = [
    "co_n_pixels", "co_window_size_used", "co_window_coverage",
    "co_radius_km_used", "co_file_available", "co_qa_pass_fraction",
]

HCHO_FEATURES = ["hcho_median", "hcho_mean", "hcho_std", "hcho_min", "hcho_max"]
HCHO_QUALITY_FEATURES = [
    "hcho_n_pixels", "hcho_window_size_used", "hcho_window_coverage",
    "hcho_radius_km_used", "hcho_file_available", "hcho_qa_pass_fraction",
]

AAI_FEATURES = ["aai_median", "aai_mean", "aai_std", "aai_min", "aai_max"]
AAI_QUALITY_FEATURES = [
    "aai_n_pixels", "aai_window_size_used", "aai_window_coverage",
    "aai_radius_km_used", "aai_file_available", "aai_qa_pass_fraction",
]

GAS_EXCLUDE_FEATURES = [
    "no2_units", "no2_scale_factor", "no2_add_offset", "no2_qa_threshold",
    "no2_qa_available", "no2_invalid_reason", "no2_file_used",
    "so2_units", "so2_scale_factor", "so2_add_offset", "so2_qa_threshold",
    "so2_qa_available", "so2_invalid_reason", "so2_file_used",
    "co_units", "co_scale_factor", "co_add_offset", "co_qa_threshold",
    "co_qa_available", "co_invalid_reason", "co_file_used",
    "hcho_units", "hcho_scale_factor", "hcho_add_offset", "hcho_qa_threshold",
    "hcho_qa_available", "hcho_invalid_reason", "hcho_file_used",
    "aai_units", "aai_scale_factor", "aai_add_offset", "aai_qa_threshold",
    "aai_qa_available", "aai_invalid_reason", "aai_file_used",
]

ALL_FEATURES_CONFIGURED = (
    METEOROLOGY_FEATURES
    + ELEVATION_FEATURES
    + DISTANCE_FEATURES
    + AOD_FEATURES
    + QA_FEATURES
    + AOD_QUALITY_FEATURES
    + NO2_FEATURES + NO2_QUALITY_FEATURES
    + SO2_FEATURES + SO2_QUALITY_FEATURES
    + CO_FEATURES + CO_QUALITY_FEATURES
    + HCHO_FEATURES + HCHO_QUALITY_FEATURES
    + AAI_FEATURES + AAI_QUALITY_FEATURES
)

# -----------------------------
# Helpers
# -----------------------------
def resolve_path_maybe_in_data(p: str | Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    alt = DATA_DIR / p.name
    if alt.exists():
        return alt
    return p

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))

def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() == 0:
        return float("nan")
    yt = y_true[m]
    yp = y_pred[m]
    ssr = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if sst == 0:
        return float("nan")
    return float(1.0 - ssr / sst)

def summarize(y, pred, label: str) -> None:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    print(f"\n{label}")
    print("-" * len(label))
    print(f"n={len(y):,} | R²={r2(y, pred):.4f} | RMSE={rmse(y, pred):.2f} | MAE={mae(y, pred):.2f}")
    print(f"y mean/median={np.mean(y):.3f}/{np.median(y):.3f} | pred mean/median={np.mean(pred):.3f}/{np.median(pred):.3f}")
    print(f"bias mean/median={np.mean(err):.3f}/{np.median(err):.3f}")

# -----------------------------
# Time encodings (optional)
# -----------------------------
def add_daily_encodings(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    if time_col not in df.columns:
        return df
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    df["day_of_year"] = df[time_col].dt.dayofyear.astype(int)
    df["month"] = df[time_col].dt.month.astype(int)
    df["year"] = df[time_col].dt.year.astype(int)

    # Sine/cos seasonality
    if "doy_sin" not in df.columns:
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    if "doy_cos" not in df.columns:
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    if "doy_sin_2" not in df.columns:
        df["doy_sin_2"] = np.sin(4 * np.pi * df["day_of_year"] / 366.0)
    if "doy_cos_2" not in df.columns:
        df["doy_cos_2"] = np.cos(4 * np.pi * df["day_of_year"] / 366.0)
    if "doy_sin_3" not in df.columns:
        df["doy_sin_3"] = np.sin(6 * np.pi * df["day_of_year"] / 366.0)
    if "doy_cos_3" not in df.columns:
        df["doy_cos_3"] = np.cos(6 * np.pi * df["day_of_year"] / 366.0)

    if "heating_season_flag" not in df.columns:
        df["heating_season_flag"] = df["month"].isin([11, 12, 1, 2]).astype("int8")
    if "burning_season_flag" not in df.columns:
        df["burning_season_flag"] = df["month"].isin([10, 11]).astype("int8")

    # Wind direction trig if WD10 exists
    if "WD10" in df.columns:
        if "WD10_sin" not in df.columns:
            df["WD10_sin"] = np.sin(np.deg2rad(df["WD10"]))
        if "WD10_cos" not in df.columns:
            df["WD10_cos"] = np.cos(np.deg2rad(df["WD10"]))
    return df

# -----------------------------
# Lat/lon detection + province assignment
# -----------------------------
def detect_latlon_cols(df: pd.DataFrame) -> Tuple[str, str]:
    if "obs_lat" in df.columns and "obs_lon" in df.columns:
        return "obs_lat", "obs_lon"
    for a, b in [("latitude", "longitude"), ("lat", "lon")]:
        if a in df.columns and b in df.columns:
            return a, b
    raise ValueError("Could not find lat/lon columns. Expected obs_lat/obs_lon or latitude/longitude or lat/lon.")

def assign_provinces_if_missing(
    df: pd.DataFrame,
    province_geojson: Path,
    province_name_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    if "province" in df.columns:
        return df

    if not province_geojson.exists():
        raise FileNotFoundError(f"Province GeoJSON not found: {province_geojson}")

    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("geopandas required to assign provinces. pip install geopandas") from e

    prov = gpd.read_file(province_geojson)
    if province_name_col not in prov.columns:
        raise ValueError(
            f"'{province_name_col}' not found in {province_geojson.name}. "
            f"Available: {list(prov.columns)}"
        )

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )

    if prov.crs is None:
        prov = prov.set_crs("EPSG:4326")
    else:
        prov = prov.to_crs("EPSG:4326")

    joined = gpd.sjoin(gdf, prov[[province_name_col, "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined = pd.DataFrame(joined)
    joined["province"] = joined[province_name_col].fillna("Unknown").astype(str)
    return joined

# -----------------------------
# Spatial blocking
# -----------------------------
def latlon_to_km(lat: np.ndarray, lon: np.ndarray, ref_lat: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if ref_lat is None:
        ref_lat = float(np.nanmean(lat))
    ref = np.deg2rad(ref_lat)
    x = lon * 111.32 * np.cos(ref)
    y = lat * 110.57
    return x, y

def _make_site_key(df: pd.DataFrame, lat_col: str, lon_col: str, round_decimals: int = 3) -> pd.Series:
    if "sensor_id" in df.columns:
        s = df["sensor_id"].astype(str)
        if s.nunique(dropna=True) > 1:
            return s.fillna("nan")
    latr = df[lat_col].astype(float).round(round_decimals)
    lonr = df[lon_col].astype(float).round(round_decimals)
    return (latr.astype(str) + "_" + lonr.astype(str)).astype(str)

def assign_blocks_grid_sites(site_df: pd.DataFrame, lat_col: str, lon_col: str, grid_km: float) -> pd.Series:
    x, y = latlon_to_km(site_df[lat_col].values, site_df[lon_col].values)
    bx = np.floor(x / float(grid_km)).astype(int)
    by = np.floor(y / float(grid_km)).astype(int)
    return pd.Series([f"{i}_{j}" for i, j in zip(bx, by)], index=site_df.index, name="block_id")

def assign_blocks_kmeans_sites(site_df: pd.DataFrame, lat_col: str, lon_col: str, n_blocks: int, seed: int) -> pd.Series:
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError("scikit-learn required for kmeans blocks. pip install scikit-learn") from e

    x, y = latlon_to_km(site_df[lat_col].values, site_df[lon_col].values)
    XY = np.vstack([x, y]).T
    n_sites = len(site_df)
    n_blocks = int(max(2, min(int(n_blocks), int(n_sites))))
    XY = XY + np.random.default_rng(seed).normal(scale=1e-6, size=XY.shape)
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    lab = km.fit_predict(XY)
    return pd.Series([f"c{int(i)}" for i in lab], index=site_df.index, name="block_id")

def make_spatial_block_folds_within_province(
    df: pd.DataFrame,
    province_name: str,
    k: int,
    seed: int,
    lat_col: str,
    lon_col: str,
    block_method: str,
    grid_km: float,
    grid_km_min: float,
    kmeans_blocks: int,
    site_round_decimals: int,
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame, Dict]]:
    sub = df[df["province"] == province_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for province={province_name}")

    sub["site_key"] = _make_site_key(sub, lat_col=lat_col, lon_col=lon_col, round_decimals=site_round_decimals)

    site_df = (
        sub[["site_key", lat_col, lon_col]]
        .dropna(subset=[lat_col, lon_col])
        .drop_duplicates(subset=["site_key"])
        .copy()
        .reset_index(drop=True)
    )

    n_sites = len(site_df)
    if n_sites < 2:
        raise ValueError(f"Only {n_sites} unique site in {province_name}. Need >=2 for within-province spatial blocks.")

    chosen_method = block_method
    chosen_grid = None

    if block_method == "grid":
        g = float(grid_km)
        while g >= float(grid_km_min):
            site_df["block_id"] = assign_blocks_grid_sites(site_df, lat_col=lat_col, lon_col=lon_col, grid_km=g)
            n_blocks = site_df["block_id"].nunique()
            if n_blocks >= 2:
                chosen_grid = g
                break
            g *= 0.7
        if chosen_grid is None:
            chosen_method = "kmeans"

    if chosen_method == "kmeans":
        site_df["block_id"] = assign_blocks_kmeans_sites(site_df, lat_col=lat_col, lon_col=lon_col, n_blocks=kmeans_blocks, seed=seed)

    n_blocks = int(site_df["block_id"].nunique())
    if n_blocks < 2:
        raise ValueError(f"Not enough blocks for {province_name} after assignment (n_blocks={n_blocks}).")

    # IMPORTANT: map blocks WITHOUT merge (preserve row identity)
    site_to_block = dict(zip(site_df["site_key"].astype(str), site_df["block_id"].astype(str)))
    sub["block_id"] = sub["site_key"].astype(str).map(site_to_block)

    blocks = sorted(sub["block_id"].astype(str).dropna().unique().tolist())
    if len(blocks) < 2:
        raise ValueError(f"Not enough block labels in rows for {province_name} (blocks={len(blocks)}).")

    k_eff = int(min(max(2, k), len(blocks)))

    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    folds = np.array_split(np.array(blocks, dtype=object), k_eff)

    out = []
    for i in range(k_eff):
        hold_blocks = set(folds[i].tolist())
        hold = sub[sub["block_id"].astype(str).isin(hold_blocks)].copy()

        # Exclude by row_id (NOT index)
        hold_ids = set(hold["row_id"].tolist())
        train = df[~df["row_id"].isin(hold_ids)].copy()

        meta = {
            "province": province_name,
            "method": chosen_method,
            "grid_km_used": chosen_grid,
            "n_sites": int(n_sites),
            "n_blocks": int(len(blocks)),
            "k_eff": int(k_eff),
        }

        out.append(
            (
                f"{province_name} spatial_{chosen_method} fold={i+1}/{k_eff} hold_blocks={len(hold_blocks)}",
                train,
                hold,
                meta,
            )
        )
    return out

# -----------------------------
# Features (STRICT: configured only)
# -----------------------------
def build_configured_feature_list(df: pd.DataFrame, y_col: str, use_latlon: bool) -> List[str]:
    feats = [f for f in ALL_FEATURES_CONFIGURED if f in df.columns and f not in GAS_EXCLUDE_FEATURES]
    if use_latlon:
        feats = feats + [f for f in SPATIAL_FEATURES if f in df.columns]
    # Hard safety: remove any column containing y_col substring (pm25_daily_mean, etc.)
    feats = [f for f in feats if (y_col.lower() not in f.lower()) or (f == y_col)]
    feats = [f for f in feats if f != y_col]
    feats = list(dict.fromkeys(feats))
    return feats

def fit_train_medians(X_train: pd.DataFrame) -> pd.Series:
    return X_train.median(numeric_only=True)

def apply_train_medians(X: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    X = X.copy()
    med_aligned = med.reindex(X.columns)
    return X.replace([np.inf, -np.inf], np.nan).fillna(med_aligned).fillna(0.0)

def filter_by_train_coverage(train_df: pd.DataFrame, features: List[str], min_non_nan_frac: float) -> List[str]:
    if min_non_nan_frac <= 0:
        return features
    nn = train_df[features].notna().mean()
    kept = [c for c in features if float(nn.get(c, 0.0)) >= float(min_non_nan_frac)]
    # prevent accidental collapse
    if len(kept) < 20:
        return features
    return kept

# -----------------------------
# Model: LightGBM
# -----------------------------
def train_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    y_col: str,
    seed: int,
    include_province_cat: bool,
    province_categories: List[str],
    median_impute: bool,
) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm required. pip install lightgbm") from e

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    if include_province_cat:
        prov_tr = pd.Categorical(train_df["province"].fillna("Unknown").astype(str), categories=province_categories)
        prov_te = pd.Categorical(test_df["province"].fillna("Unknown").astype(str), categories=province_categories)
        X_train["province"] = prov_tr
        X_test["province"] = prov_te

    y_train = train_df[y_col].values.astype(float)

    if median_impute:
        med = fit_train_medians(X_train)
        X_train = apply_train_medians(X_train, med)
        X_test = apply_train_medians(X_test, med)

    params = dict(
        objective="regression",
        learning_rate=0.05,
        n_estimators=4000,
        max_depth=8,
        num_leaves=127,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    model = lgb.LGBMRegressor(**params)

    cat_feats = ["province"] if include_province_cat else None
    model.fit(X_train, y_train, categorical_feature=cat_feats)

    pred = model.predict(X_test)
    return np.maximum(pred, 0.0)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sensor_csv", type=str, default=str(DEFAULT_SENSOR_CSV))
    ap.add_argument("--province_geojson", type=str, default=str(DEFAULT_PROVINCE_GEOJSON))
    ap.add_argument("--province_name_col", type=str, default=DEFAULT_PROVINCE_NAME_COL)

    ap.add_argument("--y_col", type=str, default="pm25")
    ap.add_argument("--time_col", type=str, default="time")
    ap.add_argument("--province_col", type=str, default="province")
    ap.add_argument("--province", type=str, default="all", help="Province name, comma-separated list, or 'all'")

    ap.add_argument("--block_method", type=str, default="grid", choices=["grid", "kmeans"])
    ap.add_argument("--grid_km", type=float, default=75.0)
    ap.add_argument("--grid_km_min", type=float, default=5.0)
    ap.add_argument("--kmeans_blocks", type=int, default=12)
    ap.add_argument("--site_round_decimals", type=int, default=3)

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_holdout_rows", type=int, default=150)

    ap.add_argument("--min_pm25", type=float, default=5.0)
    ap.add_argument("--use_latlon", type=int, default=0)
    ap.add_argument("--include_province_cat", type=int, default=0)

    ap.add_argument("--median_impute", type=int, default=1)
    ap.add_argument("--min_non_nan_frac", type=float, default=0.01)

    ap.add_argument("--per_fold_print", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="")

    args = ap.parse_args()

    sensor_csv = resolve_path_maybe_in_data(args.sensor_csv)
    province_geojson = resolve_path_maybe_in_data(args.province_geojson)

    if not sensor_csv.exists():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_csv}")
    if not province_geojson.exists():
        raise FileNotFoundError(f"Province GeoJSON not found: {province_geojson}")

    df = pd.read_csv(sensor_csv)

    # Stable row identity for leakage-free exclusion
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    if args.y_col not in df.columns:
        raise ValueError(f"Missing target column: {args.y_col}")

    df = df[df[args.y_col].notna()].copy()
    df = df[df[args.y_col] >= float(args.min_pm25)].copy()

    if args.time_col in df.columns:
        df = add_daily_encodings(df, time_col=args.time_col)

    lat_col, lon_col = detect_latlon_cols(df)

    if args.province_col in df.columns:
        df["province"] = df[args.province_col].fillna("Unknown").astype(str)
    else:
        df = assign_provinces_if_missing(
            df,
            province_geojson=province_geojson,
            province_name_col=args.province_name_col,
            lat_col=lat_col,
            lon_col=lon_col,
        )

    # Province list
    province_categories = sorted(df["province"].unique().tolist())
    if args.province.strip().lower() == "all":
        provinces = [p for p in province_categories if p != "Unknown"]
    else:
        provinces = [p.strip() for p in args.province.split(",") if p.strip()]

    # STRICT configured features (pre-filter)
    features_base = build_configured_feature_list(df, y_col=args.y_col, use_latlon=bool(args.use_latlon))
    if len(features_base) == 0:
        raise ValueError("No configured features found in this CSV (check column names / feature lists).")

    # Final safety: ensure NO pm25-derived feature sneaks in
    bad = [c for c in features_base if args.y_col.lower() in c.lower() and c != args.y_col]
    if bad:
        raise ValueError(f"LEAKY FEATURES DETECTED in feature list: {bad}")

    print("\n" + "=" * 110)
    print("Spatial-block CV (within-province): train on ALL data, test on held-out GEO blocks within each province")
    print("=" * 110)
    print(f"CSV: {sensor_csv}")
    print(f"GeoJSON: {province_geojson} (name_col={args.province_name_col})")
    print(f"Target: {args.y_col} | min_pm25={args.min_pm25}")
    print(f"Lat/Lon cols: {lat_col}, {lon_col}")
    print(f"Provinces: {provinces}")
    print(f"Requested block method: {args.block_method} | grid_km={args.grid_km} (min {args.grid_km_min}) | kmeans_blocks={args.kmeans_blocks}")
    print(f"Requested k-fold: k={args.k} | seed={args.seed} | min_holdout_rows={args.min_holdout_rows}")
    print(f"use_latlon_feature={bool(args.use_latlon)} | include_province_cat={bool(args.include_province_cat)}")
    print(f"median_impute(train-only)={bool(args.median_impute)}")
    print(f"min_non_nan_frac(train-only)={args.min_non_nan_frac}")
    print(f"Configured features present in CSV: {len(features_base)}")

    summary_rows: List[Dict] = []
    out_rows = []

    for prov in provinces:
        try:
            folds = make_spatial_block_folds_within_province(
                df=df,
                province_name=prov,
                k=args.k,
                seed=args.seed,
                lat_col=lat_col,
                lon_col=lon_col,
                block_method=args.block_method,
                grid_km=args.grid_km,
                grid_km_min=args.grid_km_min,
                kmeans_blocks=args.kmeans_blocks,
                site_round_decimals=args.site_round_decimals,
            )
        except Exception as e:
            print(f"\n[SKIP] {prov}: {e}")
            continue

        meta0 = folds[0][3]
        msg = f"{prov}: method={meta0['method']}, n_sites={meta0['n_sites']}, n_blocks={meta0['n_blocks']}, k_eff={meta0['k_eff']}"
        if meta0["method"] == "grid":
            msg += f", grid_km_used={meta0['grid_km_used']}"
        print("\n" + "-" * 110)
        print(msg)

        fold_scores = []
        for fold_name, train_df, hold_df, meta in folds:
            if len(hold_df) < args.min_holdout_rows:
                continue

            # Train-only coverage filtering for features
            features = filter_by_train_coverage(train_df, features_base, min_non_nan_frac=float(args.min_non_nan_frac))

            y_hold = hold_df[args.y_col].values.astype(float)
            preds = train_predict_lgbm(
                train_df=train_df,
                test_df=hold_df,
                features=features,
                y_col=args.y_col,
                seed=args.seed,
                include_province_cat=bool(args.include_province_cat),
                province_categories=province_categories,
                median_impute=bool(args.median_impute),
            )

            s = {
                "province": prov,
                "fold": fold_name,
                "method": meta["method"],
                "grid_km_used": meta["grid_km_used"],
                "n_sites": meta["n_sites"],
                "n_blocks": meta["n_blocks"],
                "k_eff": meta["k_eff"],
                "n_train": int(len(train_df)),
                "n_hold": int(len(hold_df)),
                "hold_blocks": int(hold_df["block_id"].nunique()),
                "n_features": int(len(features)),
                "r2": r2(y_hold, preds),
                "rmse": rmse(y_hold, preds),
                "mae": mae(y_hold, preds),
            }
            fold_scores.append(s)

            if args.per_fold_print:
                print("\n" + "-" * 110)
                print(f"Fold: {fold_name}")
                print(f"Train n={len(train_df):,} | Holdout n={len(hold_df):,} | holdout_blocks={hold_df['block_id'].nunique()}")
                if "sensor_id" in hold_df.columns:
                    print(f"Unique holdout sensors: {hold_df['sensor_id'].nunique()}")
                print(f"n_features_used={len(features)}")
                summarize(y_hold, preds, "LightGBM (spatial-block holdout)")

            if args.out_csv:
                cols_keep = []
                if args.time_col in hold_df.columns:
                    cols_keep.append(args.time_col)
                for c in [lat_col, lon_col, "province", args.y_col, "sensor_id", "block_id", "row_id"]:
                    if c in hold_df.columns:
                        cols_keep.append(c)
                tmp = hold_df[cols_keep].copy()
                tmp["fold"] = fold_name
                tmp["yhat"] = preds
                out_rows.append(tmp)

        if not fold_scores:
            print(f"\n[SKIP] {prov}: all folds too small (< min_holdout_rows={args.min_holdout_rows})")
            continue

        fs = pd.DataFrame(fold_scores)
        row = {
            "province": prov,
            "method": fs["method"].iloc[0],
            "grid_km_used": fs["grid_km_used"].iloc[0],
            "n_sites": int(fs["n_sites"].iloc[0]),
            "n_blocks": int(fs["n_blocks"].iloc[0]),
            "k_eff": int(fs["k_eff"].iloc[0]),
            "n_folds_used": int(len(fs)),
            "n_hold_total": int(fs["n_hold"].sum()),
            "n_features_mean": float(fs["n_features"].mean()),
            "r2_mean": float(fs["r2"].mean()),
            "rmse_mean": float(fs["rmse"].mean()),
            "mae_mean": float(fs["mae"].mean()),
            "r2_std": float(fs["r2"].std(ddof=0)),
            "rmse_std": float(fs["rmse"].std(ddof=0)),
            "mae_std": float(fs["mae"].std(ddof=0)),
        }
        summary_rows.append(row)

        print("\n" + "-" * 110)
        print(f"{prov} — spatial-block summary")
        print(f"  folds_used={row['n_folds_used']} | holdout_total={row['n_hold_total']:,} | n_features_mean={row['n_features_mean']:.1f}")
        print(f"  R²={row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
        print(f"  RMSE={row['rmse_mean']:.2f} ± {row['rmse_std']:.2f}")
        print(f"  MAE={row['mae_mean']:.2f} ± {row['mae_std']:.2f}")

    if not summary_rows:
        print("\nNo provinces evaluated.")
        return

    summary = pd.DataFrame(summary_rows).sort_values("r2_mean")
    print("\n" + "=" * 110)
    print("SUMMARY TABLE — Spatial-block CV (LightGBM)")
    print("=" * 110)
    with pd.option_context("display.width", 220, "display.max_rows", 300, "display.max_colwidth", 100):
        print(summary)

    if args.out_csv and out_rows:
        out = pd.concat(out_rows, axis=0, ignore_index=True)
        out.to_csv(args.out_csv, index=False)
        print(f"\nWrote holdout predictions to: {args.out_csv}")

    print("\nDone.")

if __name__ == "__main__":
    main()
