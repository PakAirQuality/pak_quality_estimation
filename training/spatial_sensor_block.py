import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# -----------------------------
# Feature pools (filtered to existing columns)
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
    # time encodings (computed)
    "doy_sin", "doy_cos",
    "doy_sin_2", "doy_cos_2",
    "doy_sin_3", "doy_cos_3",
    "heating_season_flag", "burning_season_flag",
]

# Elevation and topographic features
# Elevation and distance from coast features removed
ELEVATION_FEATURES = []
DISTANCE_FEATURES = []

# Core AOD features from MODIS MAIAC
AOD_FEATURES = [
    "optical_depth_047",    # AOD at 0.47 μm (Blue band)
    "optical_depth_055",    # AOD at 0.55 μm (Green band, most common)
    "aod_uncertainty",      # AOD uncertainty
]

# Decoded QA features from AOD_QA bitfield
QA_FEATURES = [
    "qa_cloudmask",   # bits 0-2: Cloud Mask
    "qa_adjacency",   # bits 5-7: Adjacency Mask  
    "qa_aod",         # bits 8-11: QA for AOD over land/water
    "qa_n_pixels",    # QA pixel count
]

# AOD quality metadata features
AOD_QUALITY_FEATURES = [
    "aod_total_valid_pixels",   # Number of pixels contributing to estimate
    "aod_files_used",          # Number of HDF files used
    "aod_window_size_used",    # Spatial window size needed (3, 5, or 7)
]

# Core NO2 features from TROPOMI
NO2_FEATURES = [
    "no2_median",               # Main NO2 measurement (median value)
    "no2_mean",                 # Mean NO2 in extraction window
    "no2_std",                  # Standard deviation of NO2 values
    "no2_min",                  # Minimum NO2 value in window
    "no2_max",                  # Maximum NO2 value in window
]

# NO2 quality metadata features (useful for ML)
NO2_QUALITY_FEATURES = [
    "no2_n_pixels",            # Number of valid pixels used
    "no2_window_size_used",    # Spatial window size used (pixels)
    "no2_window_coverage",     # Fraction of window with valid data
    "no2_radius_km_used",      # Search radius used (km)
    "no2_file_available",      # Whether NO2 file was available (0/1)
    "no2_qa_pass_fraction",    # Fraction of pixels passing QA (cloud/quality proxy)
]

# Core SO2 features from TROPOMI
SO2_FEATURES = [
    "so2_median",               # Main SO2 measurement (median value)
    "so2_mean",                 # Mean SO2 in extraction window
    "so2_std",                  # Standard deviation of SO2 values
    "so2_min",                  # Minimum SO2 value in window
    "so2_max",                  # Maximum SO2 value in window
]

# SO2 quality metadata features (useful for ML)
SO2_QUALITY_FEATURES = [
    "so2_n_pixels",            # Number of valid pixels used
    "so2_window_size_used",    # Spatial window size used (pixels)
    "so2_window_coverage",     # Fraction of window with valid data
    "so2_radius_km_used",      # Search radius used (km)
    "so2_file_available",      # Whether SO2 file was available (0/1)
    "so2_qa_pass_fraction",    # Fraction of pixels passing QA (cloud/quality proxy)
]

# Core CO features from TROPOMI
CO_FEATURES = [
    "co_median",                # Main CO measurement (median value)
    "co_mean",                  # Mean CO in extraction window
    "co_std",                   # Standard deviation of CO values
    "co_min",                   # Minimum CO value in window
    "co_max",                   # Maximum CO value in window
]

# CO quality metadata features (useful for ML)
CO_QUALITY_FEATURES = [
    "co_n_pixels",             # Number of valid pixels used
    "co_window_size_used",     # Spatial window size used (pixels)
    "co_window_coverage",      # Fraction of window with valid data
    "co_radius_km_used",       # Search radius used (km)
    "co_file_available",       # Whether CO file was available (0/1)
    "co_qa_pass_fraction",     # Fraction of pixels passing QA (cloud/quality proxy)
]

# Core HCHO features from TROPOMI
HCHO_FEATURES = [
    "hcho_median",              # Main HCHO measurement (median value)
    "hcho_mean",                # Mean HCHO in extraction window
    "hcho_std",                 # Standard deviation of HCHO values
    "hcho_min",                 # Minimum HCHO value in window
    "hcho_max",                 # Maximum HCHO value in window
]

# HCHO quality metadata features (useful for ML)
HCHO_QUALITY_FEATURES = [
    "hcho_n_pixels",           # Number of valid pixels used
    "hcho_window_size_used",   # Spatial window size used (pixels)
    "hcho_window_coverage",    # Fraction of window with valid data
    "hcho_radius_km_used",     # Search radius used (km)
    "hcho_file_available",     # Whether HCHO file was available (0/1)
    "hcho_qa_pass_fraction",   # Fraction of pixels passing QA (cloud/quality proxy)
]

# Core AAI features from TROPOMI
AAI_FEATURES = [
    "aai_median",               # Main AAI measurement (median value)
    "aai_mean",                 # Mean AAI in extraction window
    "aai_std",                  # Standard deviation of AAI values
    "aai_min",                  # Minimum AAI value in window
    "aai_max",                  # Maximum AAI value in window
]

# AAI quality metadata features (useful for ML)
AAI_QUALITY_FEATURES = [
    "aai_n_pixels",            # Number of valid pixels used
    "aai_window_size_used",    # Spatial window size used (pixels)
    "aai_window_coverage",     # Fraction of window with valid data
    "aai_radius_km_used",      # Search radius used (km)
    "aai_file_available",      # Whether AAI file was available (0/1)
    "aai_qa_pass_fraction",    # Fraction of pixels passing QA (cloud/quality proxy)
]

# Gas metadata features to EXCLUDE from training (not atmospheric signal)
GAS_EXCLUDE_FEATURES = [
    # NO2 metadata
    "no2_units", "no2_scale_factor", "no2_add_offset", "no2_qa_threshold", 
    "no2_qa_available", "no2_invalid_reason", "no2_file_used",
    # SO2 metadata
    "so2_units", "so2_scale_factor", "so2_add_offset", "so2_qa_threshold",
    "so2_qa_available", "so2_invalid_reason", "so2_file_used",
    # CO metadata
    "co_units", "co_scale_factor", "co_add_offset", "co_qa_threshold",
    "co_qa_available", "co_invalid_reason", "co_file_used",
    # HCHO metadata
    "hcho_units", "hcho_scale_factor", "hcho_add_offset", "hcho_qa_threshold",
    "hcho_qa_available", "hcho_invalid_reason", "hcho_file_used",
    # AAI metadata
    "aai_units", "aai_scale_factor", "aai_add_offset", "aai_qa_threshold",
    "aai_qa_available", "aai_invalid_reason", "aai_file_used",
]

# Combined feature list
ALL_FEATURES = (
    METEOROLOGY_FEATURES + 
    # Elevation and distance features removed
    AOD_FEATURES + 
    QA_FEATURES + 
    AOD_QUALITY_FEATURES +
    NO2_FEATURES +
    NO2_QUALITY_FEATURES +
    SO2_FEATURES +
    SO2_QUALITY_FEATURES +
    CO_FEATURES +
    CO_QUALITY_FEATURES +
    HCHO_FEATURES +
    HCHO_QUALITY_FEATURES +
    AAI_FEATURES +
    AAI_QUALITY_FEATURES
)


# -----------------------------
# Helpers
# -----------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def add_daily_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_year"] = df["time"].dt.dayofyear.astype(int)
    df["month"] = df["time"].dt.month.astype(int)
    df["year"] = df["time"].dt.year.astype(int)

    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    df["doy_sin_2"] = np.sin(4 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos_2"] = np.cos(4 * np.pi * df["day_of_year"] / 366.0)
    df["doy_sin_3"] = np.sin(6 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos_3"] = np.cos(6 * np.pi * df["day_of_year"] / 366.0)

    df["heating_season_flag"] = df["month"].isin([11, 12, 1, 2]).astype("int8")
    df["burning_season_flag"] = df["month"].isin([10, 11]).astype("int8")

    if "WD10" in df.columns:
        df["WD10_sin"] = np.sin(np.deg2rad(df["WD10"]))
        df["WD10_cos"] = np.cos(np.deg2rad(df["WD10"]))
    return df


def assign_provinces_if_needed(df: pd.DataFrame, province_geojson: Path, province_name_col: str) -> pd.DataFrame:
    if "province" in df.columns:
        return df
    if ("obs_lat" not in df.columns) or ("obs_lon" not in df.columns):
        raise ValueError("No 'province' column and missing obs_lat/obs_lon for GeoJSON assignment.")
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("geopandas required to assign provinces. pip install geopandas") from e

    provinces = gpd.read_file(province_geojson)
    if province_name_col not in provinces.columns:
        raise ValueError(f"'{province_name_col}' not found in {province_geojson}")

    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["obs_lon"], df["obs_lat"]), crs="EPSG:4326")
    if provinces.crs is None:
        provinces = provinces.set_crs("EPSG:4326")
    else:
        provinces = provinces.to_crs("EPSG:4326")

    joined = gpd.sjoin(gdf, provinces[[province_name_col, "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined = pd.DataFrame(joined)
    joined["province"] = joined[province_name_col].fillna("Unknown")
    return joined


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


def build_X_y(
    df: pd.DataFrame,
    numeric_features: List[str],
    include_province: bool,
    use_latlon: bool,
    province_categories: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    # Remove excluded metadata features
    feats = [f for f in numeric_features if f not in GAS_EXCLUDE_FEATURES]
    feats = pick_existing(df, feats)
    
    if use_latlon:
        feats += pick_existing(df, SPATIAL_FEATURES)
    feats = list(dict.fromkeys(feats))

    X = df[feats].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    if include_province:
        prov = df["province"].fillna("Unknown").astype(str)
        if province_categories is None:
            prov = prov.astype("category")
        else:
            prov = pd.Categorical(prov, categories=province_categories)
        X["province"] = prov

    y = df["pm25"].values.astype(float)
    return X, y


def train_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    include_province: bool,
    use_latlon: bool,
    seed: int,
    province_categories: List[str],
) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm required. pip install lightgbm") from e

    X_train, y_train = build_X_y(
        train_df, numeric_features,
        include_province=include_province,
        use_latlon=use_latlon,
        province_categories=province_categories,
    )
    X_test, _ = build_X_y(
        test_df, numeric_features,
        include_province=include_province,
        use_latlon=use_latlon,
        province_categories=province_categories,
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.03,
        n_estimators=1400,
        max_depth=8,
        num_leaves=127,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )

    cat_feats = ["province"] if include_province else None
    model.fit(X_train, y_train, categorical_feature=cat_feats)
    preds = np.maximum(model.predict(X_test), 0.0)
    return preds


def make_sensor_folds_within_province(
    df: pd.DataFrame,
    province_name: str,
    k: int,
    seed: int,
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    sub = df[df["province"] == province_name].copy()
    sensors = sorted(sub["sensor_id"].astype(str).unique().tolist())
    if len(sensors) < max(2, k):
        raise ValueError(f"Not enough sensors for k-fold. province={province_name}, sensors={len(sensors)}, k={k}")

    rng = np.random.default_rng(seed)
    rng.shuffle(sensors)
    folds = np.array_split(np.array(sensors, dtype=object), k)

    out = []
    for i in range(k):
        hold_sensors = set(folds[i].tolist())
        hold = sub[sub["sensor_id"].astype(str).isin(hold_sensors)].copy()

        train = df.copy()
        # remove ONLY the held-out sensors inside this province
        train = train[~((train["province"] == province_name) & (train["sensor_id"].astype(str).isin(hold_sensors)))].copy()

        out.append((f"{province_name} sensor_kfold fold={i+1}/{k} hold_sensors={len(hold_sensors)}", train, hold))
    return out


def make_time_block_within_province(
    df: pd.DataFrame,
    province_name: str,
    test_days: int,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    sub = df[df["province"] == province_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for province={province_name}")

    max_time = pd.to_datetime(sub["time"]).max()
    cutoff = max_time - pd.Timedelta(days=int(test_days))

    # Train on all provinces but only up to cutoff date
    train = df[pd.to_datetime(df["time"]) <= cutoff].copy()
    hold = sub[pd.to_datetime(sub["time"]) > cutoff].copy()

    name = f"{province_name} time_block test_days={test_days} cutoff={cutoff.date()}..{max_time.date()}"
    return name, train, hold


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    default_sensor_csv = data_dir / "paqi_with_all_features.csv"
    default_province_geojson = data_dir / "PAK_ADM1.geojson"

    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor_csv", type=str, default=str(default_sensor_csv))
    ap.add_argument("--province_geojson", type=str, default=str(default_province_geojson))
    ap.add_argument("--province_name_col", type=str, default="shapeName")

    ap.add_argument("--split_mode", type=str, default="sensor_kfold", choices=["sensor_kfold", "time_block"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--test_days", type=int, default=180)

    ap.add_argument("--feature_mode", type=str, default="all_features", choices=["met_time", "met_time_static", "all_features"])
    ap.add_argument("--use_latlon", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min_holdout_rows", type=int, default=500)
    ap.add_argument("--include_unknown", type=int, default=0)
    ap.add_argument("--per_fold_print", type=int, default=0, help="1 = print fold-by-fold detail, 0 = province summary only")
    ap.add_argument("--out_csv", type=str, default="", help="Optional: save all holdout rows + preds (can be large)")

    args = ap.parse_args()

    df = pd.read_csv(args.sensor_csv)
    if "time" not in df.columns or "pm25" not in df.columns:
        raise ValueError("CSV must contain at least: time, pm25")

    df["time"] = pd.to_datetime(df["time"])
    if "sensor_id" not in df.columns:
        df["sensor_id"] = "unknown"

    df = df[df["pm25"].notna() & (df["pm25"] >= 5.0)].copy()
    df = add_daily_encodings(df)

    # province assignment
    if "province" not in df.columns:
        if not args.province_geojson:
            raise ValueError("No 'province' column. Provide --province_geojson PAK_ADM1.geojson")
        gj = Path(args.province_geojson)
        if not gj.exists():
            raise FileNotFoundError(f"--province_geojson not found: {gj}")
        df = assign_provinces_if_needed(df, province_geojson=gj, province_name_col=args.province_name_col)

    df["province"] = df["province"].fillna("Unknown").astype(str)

    province_categories = sorted(df["province"].unique().tolist())
    provinces = province_categories.copy()
    if not args.include_unknown:
        provinces = [p for p in provinces if p != "Unknown"]

    # features - use ALL available features like temporal_cv.py
    all_available_features = pick_existing(df, ALL_FEATURES)
    
    # Get breakdown for reporting
    met_cols = pick_existing(df, METEOROLOGY_FEATURES)
    # Elevation and distance columns removed
    elev_cols = []
    dist_cols = []
    aod_cols = pick_existing(df, AOD_FEATURES)
    qa_cols = pick_existing(df, QA_FEATURES)
    aod_quality_cols = pick_existing(df, AOD_QUALITY_FEATURES)
    no2_cols = pick_existing(df, NO2_FEATURES)
    no2_quality_cols = pick_existing(df, NO2_QUALITY_FEATURES)
    so2_cols = pick_existing(df, SO2_FEATURES)
    so2_quality_cols = pick_existing(df, SO2_QUALITY_FEATURES)
    co_cols = pick_existing(df, CO_FEATURES)
    co_quality_cols = pick_existing(df, CO_QUALITY_FEATURES)
    hcho_cols = pick_existing(df, HCHO_FEATURES)
    hcho_quality_cols = pick_existing(df, HCHO_QUALITY_FEATURES)
    aai_cols = pick_existing(df, AAI_FEATURES)
    aai_quality_cols = pick_existing(df, AAI_QUALITY_FEATURES)

    if args.feature_mode == "met_time":
        numeric_features = met_cols
    else:
        # Use all available features like temporal_cv.py
        numeric_features = all_available_features
    numeric_features = list(dict.fromkeys(numeric_features))

    print("\n" + "=" * 110)
    print("CASE 2 — Partial pooling for ALL provinces: Train on all provinces, test WITHIN each province")
    print("=" * 110)
    print(f"CSV: {args.sensor_csv}")
    print(f"Split mode: {args.split_mode} | k={args.k} | test_days={args.test_days}")
    print(f"Feature mode: {args.feature_mode} | numeric_features={len(numeric_features)}")
    print(f"Feature breakdown: met={len(met_cols)}, elev={len(elev_cols)}, dist={len(dist_cols)}")
    print(f"                   aod={len(aod_cols)}, qa={len(qa_cols)}, aod_quality={len(aod_quality_cols)}")
    print(f"                   no2={len(no2_cols)}, no2_quality={len(no2_quality_cols)}")
    print(f"                   so2={len(so2_cols)}, so2_quality={len(so2_quality_cols)}")
    print(f"                   co={len(co_cols)}, co_quality={len(co_quality_cols)}")
    print(f"                   hcho={len(hcho_cols)}, hcho_quality={len(hcho_quality_cols)}")
    print(f"                   aai={len(aai_cols)}, aai_quality={len(aai_quality_cols)}")
    print(f"Use lat/lon: {bool(args.use_latlon)} | Seed: {args.seed}")
    print(f"Provinces evaluated: {provinces}")

    summary_rows: List[Dict] = []
    out_rows = []

    for prov in provinces:
        try:
            if args.split_mode == "sensor_kfold":
                folds = make_sensor_folds_within_province(df, province_name=prov, k=args.k, seed=args.seed)
            else:
                name, train_df, hold_df = make_time_block_within_province(df, province_name=prov, test_days=args.test_days)
                folds = [(name, train_df, hold_df)]
        except Exception as e:
            print(f"\n[SKIP] {prov}: cannot create folds ({e})")
            continue

        fold_scores = []
        hold_total = 0

        for fold_name, train_df, hold_df in folds:
            if len(hold_df) < args.min_holdout_rows:
                continue

            y_hold = hold_df["pm25"].values.astype(float)
            hold_total += len(hold_df)

            # A) Baseline
            pred_base = train_predict_lgbm(
                train_df=train_df,
                test_df=hold_df,
                numeric_features=numeric_features,
                include_province=False,
                use_latlon=bool(args.use_latlon),
                seed=args.seed,
                province_categories=province_categories,
            )

            # B) Partial pooling (province categorical)
            pred_pool = train_predict_lgbm(
                train_df=train_df,
                test_df=hold_df,
                numeric_features=numeric_features,
                include_province=True,
                use_latlon=bool(args.use_latlon),
                seed=args.seed,
                province_categories=province_categories,
            )

            s = {
                "province": prov,
                "fold": fold_name,
                "n_train": int(len(train_df)),
                "n_hold": int(len(hold_df)),
                "r2_base": r2(y_hold, pred_base),
                "rmse_base": rmse(y_hold, pred_base),
                "mae_base": mae(y_hold, pred_base),
                "r2_pool": r2(y_hold, pred_pool),
                "rmse_pool": rmse(y_hold, pred_pool),
                "mae_pool": mae(y_hold, pred_pool),
            }
            s["d_r2"] = s["r2_pool"] - s["r2_base"]
            s["d_rmse"] = s["rmse_pool"] - s["rmse_base"]
            s["d_mae"] = s["mae_pool"] - s["mae_base"]
            fold_scores.append(s)

            if args.per_fold_print:
                print("\n" + "-" * 110)
                print(f"Fold: {fold_name}")
                print(f"Train n={len(train_df):,} | Holdout ({prov}) n={len(hold_df):,} | unique holdout sensors={hold_df['sensor_id'].nunique()}")
                summarize(y_hold, pred_base, "A) Baseline (no province)")
                summarize(y_hold, pred_pool, "B) Partial pooling (province categorical)")

            if args.out_csv:
                tmp = hold_df[["time", "sensor_id", "province", "pm25"]].copy()
                tmp["fold"] = fold_name
                tmp["yhat_base"] = pred_base
                tmp["yhat_pool"] = pred_pool
                out_rows.append(tmp)

        if not fold_scores:
            print(f"\n[SKIP] {prov}: all folds too small (< min_holdout_rows={args.min_holdout_rows})")
            continue

        fs = pd.DataFrame(fold_scores)
        row = {
            "province": prov,
            "split_mode": args.split_mode,
            "n_folds": int(len(fs)),
            "n_hold_total": int(fs["n_hold"].sum()),
            "r2_base_mean": float(fs["r2_base"].mean()),
            "rmse_base_mean": float(fs["rmse_base"].mean()),
            "mae_base_mean": float(fs["mae_base"].mean()),
            "r2_pool_mean": float(fs["r2_pool"].mean()),
            "rmse_pool_mean": float(fs["rmse_pool"].mean()),
            "mae_pool_mean": float(fs["mae_pool"].mean()),
            "d_r2_mean": float(fs["d_r2"].mean()),
            "d_rmse_mean": float(fs["d_rmse"].mean()),
            "d_mae_mean": float(fs["d_mae"].mean()),
        }
        summary_rows.append(row)

        print("\n" + "-" * 110)
        print(f"{prov} — province summary (within-province holdout)")
        print(f"  folds={row['n_folds']} | holdout_total={row['n_hold_total']:,}")
        print(f"  baseline: R²={row['r2_base_mean']:.4f} | RMSE={row['rmse_base_mean']:.2f} | MAE={row['mae_base_mean']:.2f}")
        print(f"  pooled  : R²={row['r2_pool_mean']:.4f} | RMSE={row['rmse_pool_mean']:.2f} | MAE={row['mae_pool_mean']:.2f}")
        print(f"  delta   : dR²={row['d_r2_mean']:+.4f} | dRMSE={row['d_rmse_mean']:+.2f} | dMAE={row['d_mae_mean']:+.2f}")

    if not summary_rows:
        print("\nNo provinces evaluated.")
        return

    summary = pd.DataFrame(summary_rows).sort_values("r2_base_mean")
    print("\n" + "=" * 110)
    print("SUMMARY TABLE — All provinces (within-province holdout): baseline vs partial pooling")
    print("=" * 110)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.max_colwidth", 60):
        print(summary)

    print("\nOverall averages across provinces (simple mean over provinces):")
    print(f"  mean r2_base={summary['r2_base_mean'].mean():.4f} | mean r2_pool={summary['r2_pool_mean'].mean():.4f} | mean dR2={summary['d_r2_mean'].mean():+.4f}")
    print(f"  mean rmse_base={summary['rmse_base_mean'].mean():.2f} | mean rmse_pool={summary['rmse_pool_mean'].mean():.2f} | mean dRMSE={summary['d_rmse_mean'].mean():+.2f}")
    print(f"  mean mae_base={summary['mae_base_mean'].mean():.2f} | mean mae_pool={summary['mae_pool_mean'].mean():.2f} | mean dMAE={summary['d_mae_mean'].mean():+.2f}")

    if args.out_csv and out_rows:
        out = pd.concat(out_rows, axis=0, ignore_index=True)
        out.to_csv(args.out_csv, index=False)
        print(f"\nWrote holdout predictions to: {args.out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
