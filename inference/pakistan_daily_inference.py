import sys
import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import xarray as xr

from grids.grid_feature_generator import PakistanGridFeatureGenerator

warnings.filterwarnings("ignore")


# -----------------------------
# Default feature list (used only if model payload does NOT provide feature_cols)
# -----------------------------
DEFAULT_REQUIRED_FEATURES = [
    "VC", "blh", "WS10", "WS100", "RH", "theta", "u10", "v10", "u100", "v100",
    "tcc", "CLR", "msl", "sp", "t2m", "d2m", "q", "VPD", "dT", "MSLP_tend",
    "SP_tend", "BLH_tend", "VCi", "Stagnant", "HighRH", "WS10_lag1d", "WS10_lag3d",
    "WS10_rollmean_3d", "WS10_rollstd_3d", "WS10_rollmin_7d", "WS10_rollmax_7d",
    "calm3_count", "calm3_flag", "calm7_count", "calm7_flag", "blh_lag1d", "blh_lag3d",
    "blh_rollmean_3d", "blh_rollmin_7d", "blh_rollmean_7d", "blh_anom_7d",
    "blh_rollmean_14d", "blh_anom_14d", "RH_rollmean_3d", "RH_rollmax_7d",
    "RH_rollmean_7d", "RH_anom_7d", "RH_rollstd_7d", "VPD_rollmean_3d",
    "VPD_rollmean_7d", "VC_rollmean_3d", "VC_rollmin_7d", "VC_rollmean_7d",
    "VC_anom_7d", "VC_rollmean_14d", "VC_anom_14d", "stagnant3_count",
    "stagnant3_flag", "stagnant7_count", "stagnant7_flag", "WD10", "WD10_sin",
    "WD10_cos", "WD10_sin_rm_7d", "WD10_cos_rm_7d", "WD10_var_7d",
    "WD10_sin_rm_14d", "WD10_cos_rm_14d", "WD10_var_14d", "dWS", "dWS_abs",
    "dWS_rollmean_3d", "dWS_rollstd_3d", "dWD", "dWD_abs", "dWD_rollstd_3d",
    "BLH_tend_rollmean_3d", "BLH_tend_rollstd_3d", "BLH_tend_rollmean_7d",
    "BLH_tend_rollstd_7d", "MSLP_tend_rollmean_3d", "MSLP_tend_rollstd_3d",
    "MSLP_tend_rollmean_7d", "MSLP_tend_rollstd_7d", "SP_tend_rollmean_3d",
    "SP_tend_rollstd_3d", "SP_tend_rollmean_7d", "SP_tend_rollstd_7d",
    "dT_rollmean_3d", "dT_rollstd_3d", "dT_rollmean_7d", "dT_rollstd_7d",
    "doy_sin", "doy_cos", "doy_sin_2", "doy_cos_2", "doy_sin_3", "doy_cos_3",
    "heating_season_flag", "burning_season_flag", "optical_depth_047", "optical_depth_055",
    "aod_uncertainty", "qa_cloudmask", "qa_adjacency", "qa_aod", "qa_n_pixels",
    "aod_total_valid_pixels", "aod_files_used", "aod_window_size_used",
    "no2_median", "no2_mean", "no2_std", "no2_min", "no2_max", "no2_n_pixels",
    "no2_window_size_used", "no2_window_coverage", "no2_radius_km_used",
    "no2_file_available", "no2_qa_pass_fraction", "so2_median", "so2_mean",
    "so2_std", "so2_min", "so2_max", "so2_n_pixels", "so2_window_size_used",
    "so2_window_coverage", "so2_radius_km_used", "so2_file_available",
    "so2_qa_pass_fraction", "co_median", "co_mean", "co_std", "co_min", "co_max",
    "co_n_pixels", "co_window_size_used", "co_window_coverage", "co_radius_km_used",
    "co_file_available", "co_qa_pass_fraction", "hcho_median", "hcho_mean",
    "hcho_std", "hcho_min", "hcho_max", "hcho_n_pixels", "hcho_window_size_used",
    "hcho_window_coverage", "hcho_radius_km_used", "hcho_file_available",
    "hcho_qa_pass_fraction", "aai_median", "aai_mean", "aai_std", "aai_min",
    "aai_max", "aai_n_pixels", "aai_window_size_used", "aai_window_coverage",
    "aai_radius_km_used", "aai_file_available", "aai_qa_pass_fraction",
]


class PakistanPM25Predictor:
    """Daily PM2.5 prediction system for Pakistan using TRUE gridded feature engineering"""

    def __init__(
        self,
        model_path: str,
        master_feature_dir: str = "inference",
        grid_resolution: float = 0.1,
        aod_dir: str = None,
        tropomi_dir: str = None,
        geos_cf_dir: str = None,
        tropomi_window: int = 3,
        history_days: int = 21,
    ):
        self.model_path = Path(model_path)
        self.master_feature_dir = Path(master_feature_dir)
        self.grid_resolution = float(grid_resolution)
        self.history_days = int(history_days)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Default feature list (may be overridden by model payload)
        self.required_features = list(DEFAULT_REQUIRED_FEATURES)
        self.train_medians = None

        print(f"[predictor] Loading model: {self.model_path}")
        model_data = joblib.load(self.model_path)

        if isinstance(model_data, dict):
            if "model" not in model_data:
                raise ValueError(f"Model dict missing 'model' key. Keys={list(model_data.keys())}")
            self.model = model_data["model"]

            # Prefer training-time feature list if present
            if "feature_cols" in model_data and model_data["feature_cols"]:
                self.required_features = list(model_data["feature_cols"])
                print(f"[predictor] Using feature_cols from model payload ({len(self.required_features)})")
            else:
                print(f"[predictor] No feature_cols in payload; using DEFAULT_REQUIRED_FEATURES ({len(self.required_features)})")

            # Prefer training medians if present
            if "train_medians" in model_data and model_data["train_medians"]:
                tm = model_data["train_medians"]
                if isinstance(tm, pd.Series):
                    self.train_medians = tm.astype("float32")
                else:
                    self.train_medians = pd.Series(tm, dtype="float32")
                print(f"[predictor] Loaded train_medians ({len(self.train_medians)})")
        else:
            self.model = model_data
            print(f"[predictor] Loaded bare model (no payload dict). Using DEFAULT_REQUIRED_FEATURES ({len(self.required_features)})")

        # TRUE feature generator
        self.feature_generator = PakistanGridFeatureGenerator(
            master_feature_dir=str(self.master_feature_dir),
            grid_resolution=self.grid_resolution,
            aod_dir=aod_dir,
            tropomi_dir=tropomi_dir or "datasets/tropomi_pakistan_2020_2025",
            geos_cf_dir=geos_cf_dir or "datasets/geos_cf_pakistan_2020_2025",
            tropomi_window=tropomi_window,
        )

        print(f"[predictor] Model expects {len(self.required_features)} features")

    # -----------------------------
    # AQI helpers (PM2.5 24h, US EPA-style breakpoints)
    # -----------------------------
    @staticmethod
    def pm25_to_aqi(pm25_ugm3: np.ndarray) -> np.ndarray:
        pm = np.asarray(pm25_ugm3, dtype="float32")
        aqi = np.full(pm.shape, np.nan, dtype="float32")

        valid = np.isfinite(pm)
        if not valid.any():
            return aqi

        C = pm.copy()
        C[valid] = np.floor(C[valid] * 10.0) / 10.0

        bands = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]

        for C_lo, C_hi, I_lo, I_hi in bands:
            m = valid & (C >= C_lo) & (C <= C_hi)
            if not m.any():
                continue
            aqi[m] = ((I_hi - I_lo) / (C_hi - C_lo)) * (C[m] - C_lo) + I_lo

        aqi = np.clip(aqi, 0, 500).astype("float32")
        return aqi

    @staticmethod
    def aqi_discrete_style():
        colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
        bounds = [0, 50, 100, 150, 200, 300, 500]
        labels = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
        return colors, bounds, labels

    def prepare_features(self, pred_date: datetime, timings: dict = None, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        features_df = self.feature_generator.generate_daily_features(
            pred_date, 
            history_days=self.history_days,
            required_features=self.required_features,
            timings=timings,
            verbose=verbose
        )

        # Time the imputation/preparation steps separately
        t_prep = time.perf_counter()
        
        # CRITICAL FIX: Add temporal/seasonal encodings that training uses but inference was missing
        # These are created in training by temporal_cv.py::add_daily_encodings()
        pred_day = pd.Timestamp(pred_date).floor("D")
        doy = int(pred_day.dayofyear)
        month = int(pred_day.month)
        
        features_df["day_of_year"] = doy
        features_df["month"] = month
        
        # Day-of-year encodings (match training exactly with 366.0 denominator)
        features_df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
        features_df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
        features_df["doy_sin_2"] = np.sin(4 * np.pi * doy / 366.0)
        features_df["doy_cos_2"] = np.cos(4 * np.pi * doy / 366.0)
        features_df["doy_sin_3"] = np.sin(6 * np.pi * doy / 366.0)
        features_df["doy_cos_3"] = np.cos(6 * np.pi * doy / 366.0)
        
        # Seasonal flags (match training logic exactly)
        features_df["heating_season_flag"] = int(month in [11, 12, 1, 2])
        features_df["burning_season_flag"] = int(month in [10, 11])
        
        # CRITICAL FIX: Fail fast on missing required features to prevent silent failures
        missing_cols = [c for c in self.required_features if c not in features_df.columns]
        if missing_cols:
            print(f"[ERROR] Inference missing {len(missing_cols)} required features:")
            for c in missing_cols[:20]:  # Show first 20
                print(f"  - {c}")
            if len(missing_cols) > 20:
                print(f"  ... and {len(missing_cols) - 20} more")
            raise ValueError(f"Feature generation failed: {len(missing_cols)} required features missing from inference pipeline")
        
        X = features_df.reindex(columns=self.required_features)

        # Clean infs first
        X = X.replace([np.inf, -np.inf], np.nan)

        # Match training preprocessing as closely as possible
        if self.train_medians is not None:
            med = self.train_medians.reindex(self.required_features)
            
            # DEBUG: check how much we're imputing (only within Pakistan mask)
            if "__inpoly" in features_df.columns:
                pakistan_rows = features_df["__inpoly"] >= 0.5
                if pakistan_rows.any():
                    X_pak = X[pakistan_rows]
                    nan_rate = X_pak.isna().mean().sort_values(ascending=False)
                    print("[debug] top missing features (Pakistan only):\n", nan_rate.head(25))
                    print("[debug] rows with >50% missing (Pakistan only):", (X_pak.isna().mean(axis=1) > 0.5).mean())
                else:
                    print("[debug] No Pakistan pixels found for missingness analysis")
            else:
                # Fallback to full grid if no mask available
                nan_rate = X.isna().mean().sort_values(ascending=False)
                print("[debug] top missing features (full grid):\n", nan_rate.head(25))
                print("[debug] rows with >50% missing (full grid):", (X.isna().mean(axis=1) > 0.5).mean())
            
            X = X.fillna(med)
            X = X.fillna(0.0)
        else:
            # Fallback: day-wise median -> then guarantee no NaNs
            X = X.fillna(X.median(numeric_only=True))
            X = X.fillna(0.0)

        X = X.astype("float32")

        # Diagnose NaN rates after imputation
        nan_rates = (X.isna().sum() / len(X)).sort_values(ascending=False)
        top_nans = nan_rates[nan_rates > 0].head(20)
        if len(top_nans):
            print(f"[predictor] Top NaN rates after imputation:")
            for feat, rate in top_nans.items():
                print(f"  {feat}: {rate:.3f}")

        coords = features_df[["grid_lat", "grid_lon"]].copy()
        if "__inpoly" in features_df.columns:
            coords["__inpoly"] = features_df["__inpoly"].astype("float32").values

        if timings is not None:
            timings["impute+select"] = time.perf_counter() - t_prep

        return X, coords

    def predict(self, pred_date: datetime, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        print(f"\n[predictor] Predicting {pred_date.strftime('%Y-%m-%d')}")
        timings = {}

        t0 = time.perf_counter()
        X, coords = self.prepare_features(pred_date, timings, verbose=verbose)
        timings["features.total"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        # Robust predict (DataFrame vs ndarray)
        try:
            yhat = self.model.predict(X)
        except Exception:
            yhat = self.model.predict(X.values)
        timings["model.predict"] = time.perf_counter() - t1

        yhat = np.asarray(yhat, dtype="float32").reshape(-1)

        inpoly = coords["__inpoly"].values if "__inpoly" in coords.columns else None
        if inpoly is not None:
            yhat = np.where(inpoly >= 0.5, yhat, np.nan)

        # optional sanity: PM2.5 cannot be negative
        yhat = np.where(np.isfinite(yhat), np.maximum(yhat, 0.0), yhat)

        # Print sorted breakdown (top 10)
        items = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n[timing] {pd.Timestamp(pred_date).date()} breakdown:")
        for k, v in items[:10]:
            print(f"  {k:18s} {v:7.3f} s")
        print(f"  {'TOTAL':18s} {sum(timings.values()):7.3f} s\n")

        print(f"[predictor] Done. min={np.nanmin(yhat):.2f} max={np.nanmax(yhat):.2f} mean={np.nanmean(yhat):.2f}")
        return yhat, coords["grid_lat"].values, coords["grid_lon"].values, inpoly

    def save_aqi_map_png(
        self,
        aqi: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        pred_date: datetime,
        output_dir: str = "predictions",
        filename: Optional[str] = None,
    ) -> Path:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap, BoundaryNorm
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except Exception as e:
            raise RuntimeError(
                "Map output requires matplotlib + cartopy. "
                "Install e.g. `pip install matplotlib cartopy` (or conda). "
                f"Import error: {e}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        grid_shape = self.feature_generator.lat_grid.shape
        aqi_grid = aqi.reshape(grid_shape)
        lat_grid = lats.reshape(grid_shape)
        lon_grid = lons.reshape(grid_shape)

        colors, bounds, labels = self.aqi_discrete_style()
        cmap = ListedColormap(colors)
        cmap.set_bad((0, 0, 0, 0))
        norm = BoundaryNorm(bounds, cmap.N, clip=True)

        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # dynamic extent from grid (safer than hardcoding)
        ax.set_extent(
            [float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid)),
             float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))],
            crs=ccrs.PlateCarree()
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.2)

        im = ax.pcolormesh(
            lon_grid,
            lat_grid,
            aqi_grid,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
        cbar.set_label("AQI (PM2.5, 24h)", fontsize=12)
        cbar.set_ticks(bounds)
        cbar.set_ticklabels([str(b) for b in bounds])

        mu = float(np.nanmean(aqi))
        mx = float(np.nanmax(aqi))
        ax.set_title(
            f"Pakistan AQI Map (from PM2.5 daily mean)\n{pred_date.strftime('%Y-%m-%d')} | Mean AQI: {mu:.0f} | Max AQI: {mx:.0f}",
            fontsize=14,
            fontweight="bold",
        )

        ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0.3, alpha=0.5)
        plt.tight_layout()

        if filename is None:
            filename = f"aqi_pakistan_{pd.Timestamp(pred_date).strftime('%Y%m%d')}.png"

        out_png = output_dir / filename
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[save] AQI map -> {out_png}")
        return out_png

    def save_prediction(
        self,
        predictions: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        pred_date: datetime,
        output_dir: str = "predictions",
    ) -> Tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        aqi = self.pm25_to_aqi(predictions)

        grid_shape = self.feature_generator.lat_grid.shape
        pred_grid = predictions.reshape(grid_shape)
        aqi_grid = aqi.reshape(grid_shape)
        lat_grid = lats.reshape(grid_shape)
        lon_grid = lons.reshape(grid_shape)

        ds = xr.Dataset(
            {
                "pm25": (["lat", "lon"], pred_grid, {
                    "units": "μg/m³",
                    "long_name": "PM2.5 surface concentration",
                    "description": "Daily mean PM2.5 predictions from model",
                }),
                "aqi": (["lat", "lon"], aqi_grid, {
                    "units": "AQI",
                    "long_name": "Air Quality Index (from PM2.5)",
                    "description": "AQI computed from predicted PM2.5 using 24h PM2.5 breakpoints",
                }),
            },
            coords={
                "lat": (["lat"], lat_grid[:, 0]),
                "lon": (["lon"], lon_grid[0, :]),
                "time": pd.Timestamp(pred_date).floor("D").to_pydatetime(),
            },
            attrs={
                "title": "Daily PM2.5 predictions for Pakistan",
                "date_created": datetime.now().isoformat(),
                "grid_resolution_degrees": float(self.grid_resolution),
                "model_file": self.model_path.name,
                "feature_engineering_dir": str(self.master_feature_dir),
                "n_features": len(self.required_features),
                "prediction_date": pd.Timestamp(pred_date).strftime("%Y-%m-%d"),
                "conventions": "CF-1.8",
            },
        )

        netcdf_file = output_dir / f"pm25_pakistan_{pd.Timestamp(pred_date).strftime('%Y%m%d')}.nc"
        csv_file = output_dir / f"pm25_pakistan_{pd.Timestamp(pred_date).strftime('%Y%m%d')}.csv"

        print(f"[save] NetCDF -> {netcdf_file}")
        ds.to_netcdf(netcdf_file)

        print(f"[save] CSV   -> {csv_file}")
        pd.DataFrame({
            "lat": lats,
            "lon": lons,
            "pm25": predictions,
            "aqi": aqi,
            "date": pd.Timestamp(pred_date).strftime("%Y-%m-%d"),
        }).to_csv(csv_file, index=False)

        return netcdf_file, csv_file


def main():
    ap = argparse.ArgumentParser(description="Generate daily PM2.5 predictions for Pakistan (TRUE feature engineering)")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--model", default="best_model_weight/production_tuned_model_20251228_203856.joblib")
    ap.add_argument("--master_feature_dir", default="../inference")
    ap.add_argument("--output", default="predictions")
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--history_days", type=int, default=21)

    ap.add_argument("--aod_dir", default=None, help="Path to MCD19A2.061 base dir (contains year subdirs)")
    ap.add_argument("--tropomi_dir", default=None, help="Path to TROPOMI GeoTIFF base dir")
    ap.add_argument("--geos_cf_dir", default=None, help="Path to GEOS-CF base dir (for NH3 data)")
    ap.add_argument("--tropomi_window", type=int, default=3, help="Odd window size on 0.1° grid (e.g., 3 or 5)")

    ap.add_argument("--save_map", action="store_true", help="Also save an AQI PNG map")
    ap.add_argument("--map_name", default=None, help="Optional PNG filename (default auto)")

    args = ap.parse_args()

    try:
        pred_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print("Invalid --date, expected YYYY-MM-DD")
        sys.exit(1)

    master_dir = Path(args.master_feature_dir)
    if not master_dir.exists():
        print(f"inference directory not found: {master_dir}")
        sys.exit(1)

    predictor = PakistanPM25Predictor(
        model_path=args.model,
        master_feature_dir=args.master_feature_dir,
        grid_resolution=args.resolution,
        aod_dir=args.aod_dir,
        tropomi_dir=args.tropomi_dir,
        geos_cf_dir=args.geos_cf_dir,
        tropomi_window=args.tropomi_window,
        history_days=args.history_days,
    )

    yhat, lats, lons, _ = predictor.predict(pred_date)

    netcdf_file, csv_file = predictor.save_prediction(yhat, lats, lons, pred_date, args.output)

    map_file = None
    if args.save_map:
        aqi = predictor.pm25_to_aqi(yhat)
        try:
            map_file = predictor.save_aqi_map_png(aqi, lats, lons, pred_date, output_dir=args.output, filename=args.map_name)
        except Exception as e:
            print(f"[warn] Could not save map: {e}")
            print("[warn] Tip: install cartopy+matplotlib or run in the env where your video script works.")

    print("\n=== SUCCESS ===")
    print(f"Date   : {args.date}")
    print(f"NetCDF : {netcdf_file}")
    print(f"CSV    : {csv_file}")
    if map_file is not None:
        print(f"MAP    : {map_file}")


if __name__ == "__main__":
    main()
