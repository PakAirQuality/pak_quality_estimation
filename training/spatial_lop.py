"""
Spatial Leave-One-Province-Out (LOP) Training
==============================================

Extends the production Trainer with leave-one-province-out evaluation
for testing spatial generalization to unseen regions.

Usage:
    python -m training.spatial_lop
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.utils.trainer import Trainer


class TrainingLOP(Trainer):
    """
    Extends the optimized MET+ELEV+DIST+AOD+NO2+SO2 trainer with:
      - province assignment via GeoJSON
      - leave-one-province-out evaluation
      - better spatial generalization via:
          * stratified block val split
          * soft clipped province+sensor weights
          * missingness flags + no global AOD/gas imputation
    """

    def __init__(
        self,
        *args,
        province_geojson: str = "data/PAK_ADM1.geojson",
        province_name_col: str = "shapeName",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.province_geojson = Path(province_geojson)
        self.province_name_col = province_name_col

        if not self.province_geojson.exists():
            raise FileNotFoundError(
                f"Province GeoJSON not found at: {self.province_geojson.resolve()}"
            )

        print(f"LOP Configuration:")
        print(f"  Province GeoJSON: {self.province_geojson}")
        print(f"  Province name column: {self.province_name_col}")

    # ------------------------------------------------------------------
    # Province assignment
    # ------------------------------------------------------------------
    def assign_provinces(self, df: pd.DataFrame) -> pd.DataFrame:
        if "obs_lat" not in df.columns or "obs_lon" not in df.columns:
            raise ValueError("LOP requires 'obs_lat' and 'obs_lon' in the daily CSV.")

        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for leave-one-province-out.\n"
                "Install with:\n"
                "  pip install geopandas shapely pyproj rtree\n"
            )

        print(f"Loading province boundaries from: {self.province_geojson}")
        provinces = gpd.read_file(self.province_geojson)

        if self.province_name_col not in provinces.columns:
            raise ValueError(
                f"Expected province name column '{self.province_name_col}' "
                f"not found in GeoJSON. Available columns: {list(provinces.columns)}"
            )

        print(f"Found {len(provinces)} provinces in GeoJSON")
        print(f"Province names: {sorted(provinces[self.province_name_col].tolist())}")

        print(f"Creating point geometries for {len(df)} sensor observations...")
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["obs_lon"], df["obs_lat"]),
            crs="EPSG:4326",
        )

        if provinces.crs is None:
            provinces = provinces.set_crs("EPSG:4326")
        else:
            provinces = provinces.to_crs("EPSG:4326")

        print("Performing spatial join...")
        joined = gpd.sjoin(
            gdf,
            provinces[[self.province_name_col, "geometry"]],
            how="left",
            predicate="within",
        )

        joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
        joined = pd.DataFrame(joined)
        joined["province"] = joined[self.province_name_col].fillna("Unknown")

        counts = joined["province"].value_counts(dropna=False).to_dict()
        print(f"\nProvince assignment results:")
        total_assigned = sum(v for k, v in counts.items() if k != "Unknown")
        total_unknown = counts.get("Unknown", 0)

        for k, v in sorted(counts.items()):
            pct = v / len(joined) * 100
            print(f"  {k:25s}: {v:6,} ({pct:4.1f}%)")

        print(f"\nSummary: {total_assigned:,} assigned, {total_unknown:,} unknown")
        if total_unknown > len(joined) * 0.1:
            print(f"Warning: {total_unknown/len(joined)*100:.1f}% of points unassigned")

        return joined

    # ------------------------------------------------------------------
    # Override load to inject province assignment
    # ------------------------------------------------------------------
    def load_and_prepare_data(self) -> pd.DataFrame:
        df = super().load_and_prepare_data()
        if "province" not in df.columns:
            print("\nAssigning provinces using spatial join...")
            df = self.assign_provinces(df)
        else:
            print("Province column already present in data")
        return df

    # ------------------------------------------------------------------
    # Coverage diagnostics for all feature types
    # ------------------------------------------------------------------
    def analyze_feature_coverage_by_province(self, data: pd.DataFrame) -> dict:
        coverage_stats = {}
        for province in data["province"].unique():
            if province == "Unknown":
                continue
            prov_data = data[data["province"] == province]
            
            stats = {
                "total_samples": len(prov_data),
            }
            
            # AOD coverage
            aod_cols = [c for c in ["optical_depth_047", "optical_depth_055"] if c in prov_data.columns]
            if aod_cols:
                aod_available = (~prov_data[aod_cols].isna()).any(axis=1)
                stats["aod_coverage_pct"] = float(aod_available.sum() / len(prov_data) * 100)
                if "optical_depth_055" in prov_data.columns:
                    stats["mean_aod_055"] = float(prov_data["optical_depth_055"].mean())
                if "optical_depth_047" in prov_data.columns:
                    stats["mean_aod_047"] = float(prov_data["optical_depth_047"].mean())
            else:
                stats["aod_coverage_pct"] = 0.0
            
            # QA coverage
            qa_cols = [c for c in ["qa_cloudmask", "qa_adjacency", "qa_aod"] if c in prov_data.columns]
            if qa_cols:
                qa_available = (~prov_data[qa_cols].isna()).any(axis=1)
                stats["qa_coverage_pct"] = float(qa_available.sum() / len(prov_data) * 100)
                if "qa_cloudmask" in prov_data.columns:
                    stats["mean_qa_cloudmask"] = float(prov_data["qa_cloudmask"].mean())
                if "qa_aod" in prov_data.columns:
                    stats["mean_qa_aod"] = float(prov_data["qa_aod"].mean())
            else:
                stats["qa_coverage_pct"] = 0.0
            
            # TROPOMI product coverage
            tropomi_products = [
                ("NO2", "no2_median"),
                ("SO2", "so2_median"),
                ("CO", "co_median"),
                ("HCHO", "hcho_median"),
                ("AAI", "aai_median"),
            ]
            
            for product_name, median_col in tropomi_products:
                if median_col in prov_data.columns:
                    available = ~prov_data[median_col].isna()
                    stats[f"{product_name.lower()}_coverage_pct"] = float(available.sum() / len(prov_data) * 100)
                    stats[f"mean_{product_name.lower()}_column"] = float(prov_data[median_col].mean())
                else:
                    stats[f"{product_name.lower()}_coverage_pct"] = 0.0
            
            # Elevation coverage
            if "elevation_m" in prov_data.columns:
                elev_available = ~prov_data["elevation_m"].isna()
                stats["elev_coverage_pct"] = float(elev_available.sum() / len(prov_data) * 100)
                stats["mean_elevation"] = float(prov_data["elevation_m"].mean())
            else:
                stats["elev_coverage_pct"] = 0.0
            
            # Distance coverage
            if "dist_to_coast_km" in prov_data.columns:
                dist_available = ~prov_data["dist_to_coast_km"].isna()
                stats["dist_coverage_pct"] = float(dist_available.sum() / len(prov_data) * 100)
                stats["mean_distance_to_coast"] = float(prov_data["dist_to_coast_km"].mean())
            else:
                stats["dist_coverage_pct"] = 0.0

            coverage_stats[province] = stats
        return coverage_stats

    # ------------------------------------------------------------------
    # Spatial blocks (kept, but used with stratified split)
    # ------------------------------------------------------------------
    def make_spatial_block_groups(self, df, cell_deg=0.5):
        tmp = df.groupby("sensor_id")[["obs_lat", "obs_lon", "province"]].first().reset_index()
        lat_bin = np.floor(tmp["obs_lat"] / cell_deg).astype(int)
        lon_bin = np.floor(tmp["obs_lon"] / cell_deg).astype(int)
        tmp["spatial_block"] = tmp["province"].astype(str) + "_" + lat_bin.astype(str) + "_" + lon_bin.astype(str)
        return df.merge(tmp[["sensor_id", "spatial_block"]], on="sensor_id", how="left")

    # ------------------------------------------------------------------
    # NEW: stratified-by-province spatial-block split for early stopping
    # ------------------------------------------------------------------
    def stratified_block_split_minval(
        self,
        df: pd.DataFrame,
        group_col="spatial_block",
        strat_col="province",
        val_frac=0.20,
        seed=42,
        min_val_frac=0.18,
        min_val_samples=8000,
        max_val_frac=0.35,  # Hard cap to prevent huge validation sets
    ):
        """
        Province-aware selection of spatial blocks for validation, with guarantees:
          - tries ~val_frac per province (by samples, not just #blocks)
          - then "tops up" validation by adding more blocks until:
                val >= min_val_frac of all samples  AND  val >= min_val_samples
        """
        rng = np.random.default_rng(seed)

        # block sizes per province
        block_sizes = (
            df.groupby([strat_col, group_col])
              .size()
              .rename("n")
              .reset_index()
        )

        val_blocks = set()

        # 1) per-province choose blocks until reaching target samples
        for p, g in block_sizes.groupby(strat_col):
            blocks = g.sample(frac=1.0, random_state=seed)  # shuffle deterministically
            target = int(np.ceil(val_frac * df[df[strat_col] == p].shape[0]))
            acc = 0
            for _, row in blocks.iterrows():
                val_blocks.add((row[strat_col], row[group_col]))
                acc += int(row["n"])
                if acc >= target:
                    break

        # compute current val
        key = list(zip(df[strat_col].values, df[group_col].values))
        is_val = np.array([(k in val_blocks) for k in key], dtype=bool)
        val_n = int(is_val.sum())
        total_n = len(df)

        # 2) top-up: add more blocks (largest first) until minimums met, but respect max cap
        need_frac = int(np.ceil(min_val_frac * total_n))
        need_n = max(need_frac, int(min_val_samples))
        max_n = int(np.floor(max_val_frac * total_n))  # Hard cap
        
        # Don't add more if we're already over the max cap
        if val_n < need_n and val_n < max_n:
            # candidates = blocks not already in val, sorted by size descending
            candidates = block_sizes.copy()
            candidates["key"] = list(zip(candidates[strat_col], candidates[group_col]))
            candidates = candidates[~candidates["key"].isin(val_blocks)]
            candidates = candidates.sort_values("n", ascending=False)

            for _, row in candidates.iterrows():
                val_blocks.add((row[strat_col], row[group_col]))
                # update is_val incrementally
                is_val = np.array([(k in val_blocks) for k in key], dtype=bool)
                val_n = int(is_val.sum())
                if val_n >= need_n or val_n >= max_n:  # Stop at either minimum or maximum
                    break

        tr_idx = np.where(~is_val)[0]
        va_idx = np.where(is_val)[0]
        return tr_idx, va_idx

    # ------------------------------------------------------------------
    # NEW: soft + clipped province+sensor weights (sqrt balancing)
    # ------------------------------------------------------------------
    def province_sensor_balanced_weights_soft(self, df, clip_q=0.99):
        prov_counts = df["province"].value_counts()
        sensor_counts = df.groupby(["province", "sensor_id"]).size()

        w = []
        for p, s in zip(df["province"].values, df["sensor_id"].values):
            w.append((1.0 / np.sqrt(prov_counts[p])) * (1.0 / np.sqrt(sensor_counts[(p, s)])))

        w = np.array(w, dtype=float)
        cap = np.quantile(w, clip_q)
        w = np.clip(w, 0.0, cap)
        return w / np.mean(w)

    # ------------------------------------------------------------------
    # NEW: missingness flags + keep NaNs (no global median imputation)
    # ------------------------------------------------------------------
    def add_missing_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Add missingness flags for satellite-derived features that may have gaps
        miss_cols = [c for c in (
            self.AOD_FEATURES + 
            self.QA_FEATURES + 
            self.AOD_QUALITY_FEATURES + 
            self.NO2_FEATURES + 
            self.NO2_QUALITY_FEATURES + 
            self.SO2_FEATURES + 
            self.SO2_QUALITY_FEATURES +
            self.CO_FEATURES +
            self.CO_QUALITY_FEATURES +
            self.HCHO_FEATURES +
            self.HCHO_QUALITY_FEATURES +
            self.AAI_FEATURES +
            self.AAI_QUALITY_FEATURES
        ) if c in X.columns]
        
        for c in miss_cols:
            X[c + "_missing"] = X[c].isna().astype("int8")

        # Add combined availability flags
        satellite_products = [
            ("aod", self.AOD_FEATURES),
            ("no2", self.NO2_FEATURES),
            ("so2", self.SO2_FEATURES),
            ("co", self.CO_FEATURES),
            ("hcho", self.HCHO_FEATURES),
            ("aai", self.AAI_FEATURES),
        ]
        
        for product_name, feature_list in satellite_products:
            cols = [c for c in feature_list if c in X.columns]
            if cols:
                X[f"{product_name}_available"] = (~X[cols].isna()).any(axis=1).astype("int8")
            
        return X

    def prepare_features_log(self, data: pd.DataFrame):
        available_features = [f for f in self.ALL_FEATURES if f in data.columns]
        missing_features = [f for f in self.ALL_FEATURES if f not in data.columns]
        
        # Exclude problematic gas metadata features even if present in data
        excluded_present = [f for f in self.GAS_EXCLUDE_FEATURES if f in data.columns]
        if excluded_present:
            available_features = [f for f in available_features if f not in self.GAS_EXCLUDE_FEATURES]
        
        if missing_features:
            print(f"Info: {len(missing_features)} configured features not found (ignored).")

        print(f"Using {len(available_features)} DAILY MET+ELEV+DIST+AOD+TROPOMI-ALL features")
        X = data[available_features].copy()

        # Use direct PM2.5 target
        y = data["pm25"].values

        # preserve satellite-derived feature missingness signal
        X = self.add_missing_flags(X)

        # IMPORTANT: do NOT global-impute AOD/QA/NO2/SO2; let LGBM handle NaNs.
        # If you want to impute only MET/ELEV/DIST, do it explicitly (optional):
        # stable_cols = [c for c in (self.METEOROLOGY_FEATURES + self.ELEVATION_FEATURES + self.DISTANCE_FEATURES) if c in X.columns]
        # X[stable_cols] = X[stable_cols].fillna(X[stable_cols].median(numeric_only=True))

        return X, y

    # ------------------------------------------------------------------
    # Helper to create LGBM regressor with fixed iteration count
    # ------------------------------------------------------------------
    def make_lgbm_regressor(self, n_estimators: int, feature_names: list = None):
        import lightgbm as lgb
        
        params = self.LGBM_PARAMS.copy()
        params["n_estimators"] = n_estimators
        
        return lgb.LGBMRegressor(**params)

    # ------------------------------------------------------------------
    # train_lgbm_model: updated for new feature set
    # ------------------------------------------------------------------
    def train_lgbm_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        w_train: np.ndarray = None,
        w_val: np.ndarray = None,
        override_params: dict | None = None,
    ):
        import lightgbm as lgb

        params = self.LGBM_PARAMS.copy()
        if override_params is not None:
            params.update(override_params)

        print(f"Training LightGBM with {len(X_train.columns)} MET+ELEV+DIST+AOD+TROPOMI-ALL features...")
        if w_train is not None:
            print(f"  Using SOFT province+sensor weights (train: {len(w_train)} samples)")
        print(f"  Using {len(X_train.columns)} features")

        model = lgb.LGBMRegressor(**params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val] if w_val is not None else None,
            eval_metric="l1",
            callbacks=callbacks,
        )
        return model

    def reconstruct_pm25_from_log(self, data: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
        """Return PM2.5 predictions directly."""
        return y_pred

    # ------------------------------------------------------------------
    # LOP main loop with NEW split + NEW weights + ALL features
    # ------------------------------------------------------------------
    def run_leave_one_province_out(self, min_samples_per_province: int = 50):
        full_data = self.load_and_prepare_data()

        print("\nAnalyzing feature coverage by province...")
        coverage_stats = self.analyze_feature_coverage_by_province(full_data)

        provinces = sorted([p for p in full_data["province"].unique() if p != "Unknown"])
        results = {}
        all_metrics = []
        for holdout in provinces:
            prov_data = full_data[full_data["province"] == holdout].copy()
            n_prov = len(prov_data)
            if n_prov < min_samples_per_province:
                print(f"\nSkipping {holdout}: only {n_prov:,} samples < {min_samples_per_province}")
                continue

            train_pool = full_data[full_data["province"] != holdout].copy()
            # Much smaller blocks -> better early stopping stability + avoid huge validation fractions
            train_pool = self.make_spatial_block_groups(train_pool, cell_deg=0.10)

            tr_idx, va_idx = self.stratified_block_split_minval(
                train_pool,
                group_col="spatial_block",
                strat_col="province",
                val_frac=0.20,
                seed=42,
                min_val_frac=0.18,
                min_val_samples=8000,
                max_val_frac=0.35,  # Hard cap to prevent huge validation sets
            )
            internal_train = train_pool.iloc[tr_idx].copy()
            internal_val = train_pool.iloc[va_idx].copy()

            print(f"\n--- HOLDOUT PROVINCE: {holdout} ---")
            print(f"Holdout samples: {n_prov:,}")
            if holdout in coverage_stats:
                stats = coverage_stats[holdout]
                print(f"Holdout AOD coverage: {stats.get('aod_coverage_pct', 0.0):.1f}%")
                print(f"Holdout NO2 coverage: {stats.get('no2_coverage_pct', 0.0):.1f}%")
                print(f"Holdout SO2 coverage: {stats.get('so2_coverage_pct', 0.0):.1f}%")
                print(f"Holdout CO coverage: {stats.get('co_coverage_pct', 0.0):.1f}%")
                print(f"Holdout HCHO coverage: {stats.get('hcho_coverage_pct', 0.0):.1f}%")
                print(f"Holdout AAI coverage: {stats.get('aai_coverage_pct', 0.0):.1f}%")
                print(f"Holdout elevation coverage: {stats.get('elev_coverage_pct', 0.0):.1f}%")
                print(f"Holdout distance coverage: {stats.get('dist_coverage_pct', 0.0):.1f}%")

            print(f"Train pool samples (other provinces): {len(train_pool):,}")
            
            print(f"Internal early-stop split: "
                  f"train={len(internal_train):,} val={len(internal_val):,} "
                  f"(val_frac={len(internal_val)/len(train_pool):.3f})")
            print(f"Province coverage: train={internal_train['province'].nunique()} "
                  f"val={internal_val['province'].nunique()}")

            # EARLY STOP: unweighted (prevents underfit due to tiny/effective-val issues)
            X_tr, y_tr = self.prepare_features_log(internal_train)
            X_iv, y_iv = self.prepare_features_log(internal_val)

            model_es = self.train_lgbm_model(
                X_tr, y_tr, X_iv, y_iv,
                w_train=None, w_val=None,     # <--- important
                override_params={"learning_rate": 0.03}  # optional
            )

            best_iter = getattr(model_es, "best_iteration_", None)
            if best_iter is None:
                best_iter = getattr(model_es, "n_estimators_", None) or 3000
            print(f"Best iteration from early stopping: {best_iter}")

            # FINAL REFIT: with soft weights
            final_w = self.province_sensor_balanced_weights_soft(train_pool, clip_q=0.99)

            X_full, y_full = self.prepare_features_log(train_pool)
            final_model = self.make_lgbm_regressor(n_estimators=best_iter, feature_names=X_full.columns.tolist())
            final_model.fit(X_full, y_full, sample_weight=final_w)

            # Evaluate holdout
            X_hold, _ = self.prepare_features_log(prov_data)
            log_pred = final_model.predict(X_hold)
            hold_pred_pm25 = self.reconstruct_pm25_from_log(prov_data, log_pred)

            print(f"LOP metrics on {holdout}:")
            metrics = self.evaluate_predictions(prov_data["pm25"].values, hold_pred_pm25)

            results[holdout] = {
                "holdout_province": holdout,
                "holdout_samples": int(n_prov),
                "train_pool_samples": int(len(train_pool)),
                "best_iteration": int(best_iter),
                "coverage_stats": coverage_stats.get(holdout, {}),
                "metrics": metrics,
            }

            if metrics:
                all_metrics.append(metrics)

        # Aggregate summary
        summary = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                vals = [m[key] for m in all_metrics if key in m and m[key] is not None]
                if vals:
                    summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.results_dir / f"daily_met_elev_dist_aod_tropomi_all_lgbm_leave_one_province_{timestamp}.json"

        payload = {
            "model_type": "daily_met_elev_dist_aod_tropomi_all_lgbm_leave_one_province",
            "use_latlon": self.use_latlon,
            "province_geojson": str(self.province_geojson),
            "province_name_col": self.province_name_col,
            "min_samples_per_province": int(min_samples_per_province),
            "feature_coverage_by_province": coverage_stats,
            "results_by_province": results,
            "summary": summary,
            "timestamp": timestamp,
        }

        with open(out_path, "w") as f:
            json.dump(self.to_jsonable(payload), f, indent=2)

        print("\n" + "=" * 80)
        print("MET+ELEV+DIST+AOD+TROPOMI-ALL LOP SUMMARY (across evaluated provinces)")
        for k, stats in summary.items():
            print(f"{k:12s}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        print(f"\nSaved LOP results to: {out_path}")
        print("=" * 80)

        return results, summary, out_path


def main():

    trainer = TrainingLOP(
        sensor_csv="paqi_with_all_features.csv",  # Updated to use full TROPOMI dataset
        results_dir="results/met_elev_dist_aod_tropomi_all_daily_lgbm_lop",
        use_latlon=False,
        province_geojson="data/PAK_ADM1.geojson",
        province_name_col="shapeName",
    )

    trainer.run_leave_one_province_out(min_samples_per_province=50)

    print("\nLOP evaluation complete!")


if __name__ == "__main__":
    main()