"""
Registry-Authoritative LightGBM Training with Tail Weighting

This module implements PM2.5 prediction model training using feature registry as the 
single source of truth for feature selection. Includes tail weighting to improve 
performance on extreme PM2.5 events (≥100 μg/m³).

Key features:
- Registry-driven feature selection (no hardcoded feature lists)
- Tail weighting for rare high-PM2.5 events  
- Temporal cross-validation with rolling yearly folds
- Hyperparameter tuning via random search
- Clean duplicate column handling
- Fail-fast validation for missing registry features
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Import FeatureRegistry from your feature_engineering package.
# We keep this as a runtime path append so you can run from training.
# ---------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_MFE = _THIS.parent.parent / "feature_engineering"
if _MFE.exists():
    sys.path.append(str(_MFE))

from feature_registry import FeatureRegistry, TARGET_COL  # noqa: E402


class Trainer:

    def __init__(
        self,
        sensor_csv: str = "paqi_with_all_features.csv",
        registry_csv: str = "features_registry.csv",
        results_dir: str = "results/temporal_lgbm_upweighted",
        use_latlon: bool = False,
        strict_registry: bool = True,
        # NEW: tail-weighting knobs
        use_tail_weights: bool = True,
        tail_thresholds: tuple[float, float, float] = (100.0, 150.0, 200.0),
        tail_increments: tuple[float, float, float] = (1.0, 3.0, 6.0),
        tail_weight_cap: float = 12.0,
    ):
        # Hardcoded data paths
        self.master_path = Path("data") / sensor_csv
        self.registry_path = Path("data") / registry_csv
        
        if not self.master_path.exists():
            raise FileNotFoundError(f"Master dataset not found: {self.master_path}")
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Features registry not found: {self.registry_path}")

        # Initialize registry object, but override its paths to the resolved ones
        self.registry = FeatureRegistry()
        self.registry.master_path = self.master_path
        self.registry.registry_path = self.registry_path
        # keep metadata dir adjacent to registry file
        self.registry.metadata_dir = self.registry_path.parent / "metadata"
        self.registry.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.registry.registry = self.registry._load_or_init_registry()

        self.strict_registry = bool(strict_registry)
        self.use_latlon = bool(use_latlon)

        # NEW: tail-weighting config
        self.use_tail_weights = bool(use_tail_weights)
        self.tail_thresholds = tuple(float(x) for x in tail_thresholds)
        self.tail_increments = tuple(float(x) for x in tail_increments)
        self.tail_weight_cap = float(tail_weight_cap)

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Core model params (kept close to your current defaults)
        self.LGBM_PARAMS = {
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "n_estimators": 5000,
            "max_depth": 8,
            "num_leaves": 127,
            "min_child_samples": 25,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "force_col_wise": True,
        }

        # Optional spatial feature names
        self.SPATIAL_FEATURES = ["obs_lat", "obs_lon"]  # preferred
        self.SPATIAL_FEATURES_FALLBACK = ["latitude", "longitude"]

        # Derived encodings we compute in-code and always include if present
        self.DERIVED_FEATURES = [
            "doy_sin", "doy_cos", "doy_sin_2", "doy_cos_2", "doy_sin_3", "doy_cos_3",
            "heating_season_flag", "burning_season_flag",
            "WD10_sin", "WD10_cos",
        ]

        # Exclusion rules (safety: keep registry as source-of-truth, but still exclude obvious audit fields)
        def _is_excluded_feature(col: str) -> bool:
            c = str(col).lower()
            # string/audit columns that should never be features
            if "invalid_reason" in c:
                return True
            if c.endswith("_file_used") or "file_used" in c:
                return True
            # numeric metadata that's usually not useful
            if c.endswith("_scale_factor") or c.endswith("_add_offset"):
                return True
            if c.endswith("_qa_threshold") or c.endswith("_qa_available"):
                return True
            return False

        self._is_excluded_feature = _is_excluded_feature

        self.SEASONS = {
            "winter": [12, 1, 2],
            "pre_monsoon": [3, 4, 5],
            "monsoon": [6, 7, 8, 9],
            "post_monsoon": [10, 11],
        }

        self._full_data = None

    # ------------------------------------------------------------------
    # NEW: tail weights
    # ------------------------------------------------------------------
    def make_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Tail-aware weights for rare high-PM2.5 days.

        Default schedule:
          base = 1
          +1 if y >= 100
          +3 if y >= 150
          +6 if y >= 200
        capped at tail_weight_cap
        """
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y, dtype=float)

        if not self.use_tail_weights:
            return w

        t1, t2, t3 = self.tail_thresholds
        a1, a2, a3 = self.tail_increments

        w += a1 * (y >= t1)
        w += a2 * (y >= t2)
        w += a3 * (y >= t3)

        w = np.clip(w, 1.0, self.tail_weight_cap)
        return w

    # ------------------------------------------------------------------
    # Data loading / preparation
    # ------------------------------------------------------------------
    def load_and_prepare_data(self) -> pd.DataFrame:
        if self._full_data is not None:
            return self._full_data

        print(f"Loading clean master dataset: {self.master_path}")
        df = pd.read_csv(self.master_path)
        
        # Basic preprocessing
        if "time" not in df.columns:
            raise ValueError("Missing 'time' column in dataset")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing target column '{TARGET_COL}' in dataset")
            
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time", TARGET_COL])
        
        # Filter for reasonable PM2.5 values
        valid_pm25 = df[TARGET_COL].notna() & (df[TARGET_COL] >= 5.0)
        df = df[valid_pm25].copy()
        print(f"PM2.5 range: {df[TARGET_COL].min():.1f} to {df[TARGET_COL].max():.1f} μg/m³")
        
        # Add time-derived features
        df["day_of_year"] = df["time"].dt.dayofyear
        df["month"] = df["time"].dt.month
        df["year"] = df["time"].dt.year
        df = self.add_daily_encodings(df)

        df = df.sort_values(["sensor_id", "time"]).reset_index(drop=True)
        self._full_data = df
        return df

    def add_daily_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

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

        if "WD10" in df.columns:
            if "WD10_sin" not in df.columns:
                df["WD10_sin"] = np.sin(np.deg2rad(df["WD10"]))
            if "WD10_cos" not in df.columns:
                df["WD10_cos"] = np.cos(np.deg2rad(df["WD10"]))

        return df

    def get_year_data(self, years: list) -> pd.DataFrame:
        full_data = self.load_and_prepare_data()
        year_data = full_data[full_data["year"].isin(years)].copy()
        print(f"Data for years {years}: {len(year_data):,} samples")
        return year_data

    def prepare_features(
        self,
        data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        train_medians: Optional[pd.Series] = None,
        fit_medians: bool = False,
        min_non_nan_frac: float = 0.01,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], Optional[pd.Series]]:
        """
        Clean registry-authoritative feature selection.
        Uses registry as single source of truth for training features.
        """
        data = data.copy()

        if fit_medians or feature_cols is None:
            # Get features directly from registry
            registry_features = self.registry.get_feature_columns()
            if len(registry_features) == 0:
                raise ValueError("No features found in registry. Check features_registry.csv.")
            
            # Filter to features that exist in data
            available_features = [f for f in registry_features if f in data.columns]
            
            # Add spatial features if requested
            if self.use_latlon:
                for spatial_feat in self.SPATIAL_FEATURES + self.SPATIAL_FEATURES_FALLBACK:
                    if spatial_feat in data.columns and spatial_feat not in available_features:
                        available_features.append(spatial_feat)
            
            # Add derived features that exist in data
            for derived_feat in self.DERIVED_FEATURES:
                if derived_feat in data.columns and derived_feat not in available_features:
                    available_features.append(derived_feat)
            
            # Apply exclusion rules
            available_features = [c for c in available_features if not self._is_excluded_feature(c)]
            
            # Filter by data coverage if fitting medians
            if fit_medians and min_non_nan_frac > 0.0:
                X_temp = data.reindex(columns=available_features).copy()
                for c in X_temp.columns:
                    if not pd.api.types.is_numeric_dtype(X_temp[c]):
                        X_temp[c] = pd.to_numeric(X_temp[c], errors="coerce")
                        
                coverage = X_temp.notna().mean()
                available_features = [c for c in available_features if coverage.get(c, 0.0) >= min_non_nan_frac]
                print(f"After coverage filter ({min_non_nan_frac}): {len(available_features)} features")
            
            feature_cols = available_features
            
            # Report missing registry features
            missing_registry = [f for f in registry_features if f not in data.columns]
            if missing_registry:
                print(f"Warning: {len(missing_registry)} registry features missing from data")
                if self.strict_registry and len(missing_registry) > len(registry_features) * 0.1:
                    raise ValueError(f"Too many registry features missing: {missing_registry[:10]}...")
            
            print(f"Registry features available: {len(feature_cols)}")

        # Create feature matrix
        X = data.reindex(columns=feature_cols).copy()

        # Ensure all columns are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
        
        # Remove duplicate columns if any exist
        X = X.loc[:, ~X.columns.duplicated()]
        
        # Target vector
        y = data[TARGET_COL].values

        # Compute or use training medians for imputation
        if fit_medians:
            train_medians = X.median(numeric_only=True)
        
        if train_medians is not None:
            X = X.fillna(train_medians)
        else:
            X = X.fillna(X.median(numeric_only=True))

        # Final safety check
        X = X.fillna(0.0)
        
        # Update feature_cols to match final X columns
        feature_cols = list(X.columns)

        print(f"Final feature matrix: {len(feature_cols)} features, {len(X)} samples")
        return X, y, feature_cols, train_medians

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    @staticmethod
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {k: self.to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.to_jsonable(v) for v in obj]
        return self.json_safe(obj)

    # ------------------------------------------------------------------
    # Training (LightGBM)  --- MODIFIED: weights for train + val
    # ------------------------------------------------------------------
    def train_lgbm_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        override_params: dict | None = None,
        w_train: np.ndarray | None = None,
        w_val: np.ndarray | None = None,
    ) -> lgb.LGBMRegressor:

        params = self.LGBM_PARAMS.copy()
        if override_params is not None:
            params.update(override_params)

        if params.get("n_estimators", 0) < 6000:
            params["n_estimators"] = 9000

        model = lgb.LGBMRegressor(**params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=0)
        ]

        # NEW: if caller didn't pass weights, build them here
        if w_train is None:
            w_train = self.make_sample_weights(y_train)
        if w_val is None:
            w_val = self.make_sample_weights(y_val)

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="l1",
            callbacks=callbacks
        )
        return model

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True) -> dict:
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]

        if len(y_true_clean) == 0:
            return {}

        rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
        mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
        r2 = float(r2_score(y_true_clean, y_pred_clean))
        bias = float(np.mean(y_pred_clean - y_true_clean))

        extreme_mask = y_true_clean >= 150
        n_extreme_true = int(np.sum(extreme_mask))
        n_extreme_pred = int(np.sum(y_pred_clean >= 150))

        if n_extreme_true > 0:
            extreme_rmse = float(np.sqrt(mean_squared_error(
                y_true_clean[extreme_mask], y_pred_clean[extreme_mask]
            )))
            high_true_binary = (y_true_clean >= 150).astype(int)
            high_pred_binary = (y_pred_clean >= 150).astype(int)
            f1_150 = float(f1_score(high_true_binary, high_pred_binary, zero_division=0))
        else:
            extreme_rmse = float("nan")
            f1_150 = 0.0

        if verbose:
            print(f"  RMSE: {rmse:.1f} μg/m³")
            print(f"  MAE:  {mae:.1f} μg/m³")
            print(f"  R²:   {r2:.3f}")
            print(f"  Bias: {bias:.1f} μg/m³")
            print(f"  F1@150: {f1_150:.3f} ({n_extreme_true} extreme events)")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "bias": bias,
            "f1_150": f1_150,
            "extreme_rmse": extreme_rmse,
            "n_extreme_true": n_extreme_true,
            "n_extreme_pred": n_extreme_pred,
            "n_predictions": int(len(y_true_clean)),
        }

    # ------------------------------------------------------------------
    # NEW: Hyperparameter tuning via rolling CV (random search)
    # ------------------------------------------------------------------
    def tune_hyperparameters_rolling_cv(
        self,
        n_trials: int = 25,
        cv_val_years: tuple[int, ...] = (2022, 2023, 2024),
        min_non_nan_frac: float = 0.01,
        seed: int = 42,
    ):
        full_data = self.load_and_prepare_data()
        all_years = sorted(full_data["year"].unique())
        cv_val_years = tuple([y for y in cv_val_years if y in all_years])

        if len(cv_val_years) == 0:
            raise ValueError("No cv_val_years found in data. Check available years.")

        print("\n" + "=" * 80)
        print(f"HYPERPARAM TUNING (random search) | trials={n_trials} | folds={cv_val_years}")
        if self.use_tail_weights:
            print(f"(Tail weights ON: thresholds={self.tail_thresholds}, inc={self.tail_increments}, cap={self.tail_weight_cap})")
        print("=" * 80)

        param_space = {
            "max_depth": [4, 6, 8, 10, 12],
            "min_child_samples": [10, 25, 50, 100, 200],
            "min_child_weight": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
            "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
            "min_split_gain": [0.0, 0.01, 0.05, 0.1],
            "max_bin": [127, 255, 511],
            "extra_trees": [True, False],
        }

        sampler = ParameterSampler(param_space, n_iter=n_trials, random_state=seed)

        best_params = None
        best_mean_r2 = -1e9
        best_mean_mae = 1e9
        all_trials = []
        trial_idx = 0

        for sampled in sampler:
            trial_idx += 1

            max_depth = int(sampled["max_depth"])
            num_leaves = min((2 ** max_depth) - 1, 255)

            lr = float(sampled["learning_rate"])
            n_estimators = 12000 if lr <= 0.03 else (9000 if lr <= 0.05 else 7000)

            override = {
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "min_child_samples": int(sampled["min_child_samples"]),
                "reg_lambda": float(sampled["reg_lambda"]),
                "reg_alpha": float(sampled["reg_alpha"]),
                "subsample": float(sampled["subsample"]),
                "colsample_bytree": float(sampled["colsample_bytree"]),
                "learning_rate": lr,
                "min_split_gain": float(sampled["min_split_gain"]),
                "n_estimators": int(n_estimators),
                "random_state": seed,
            }

            fold_metrics = []
            for val_year in cv_val_years:
                train_years = [y for y in all_years if y < val_year]
                if len(train_years) == 0:
                    continue

                train_data = self.get_year_data(train_years)
                val_data = self.get_year_data([val_year])

                X_train, y_train, feat_cols, med = self.prepare_features(
                    train_data, fit_medians=True, min_non_nan_frac=min_non_nan_frac
                )
                X_val, y_val, _, _ = self.prepare_features(
                    val_data, feature_cols=feat_cols, train_medians=med, fit_medians=False
                )

                w_train = self.make_sample_weights(y_train)
                w_val = self.make_sample_weights(y_val)

                model = self.train_lgbm_model(
                    X_train, y_train, X_val, y_val,
                    override_params=override,
                    w_train=w_train,
                    w_val=w_val
                )

                val_pred = model.predict(X_val)
                metrics = self.evaluate_predictions(val_data["pm25"].values, val_pred, verbose=False)
                fold_metrics.append(metrics)

            if not fold_metrics:
                continue

            mean_r2 = float(np.nanmean([m.get("r2", np.nan) for m in fold_metrics]))
            mean_mae = float(np.nanmean([m.get("mae", np.nan) for m in fold_metrics]))

            all_trials.append({
                "trial": trial_idx,
                "override_params": override,
                "mean_r2": mean_r2,
                "mean_mae": mean_mae,
                "fold_metrics": fold_metrics,
                "tail_weighting": {
                    "use_tail_weights": self.use_tail_weights,
                    "thresholds": list(self.tail_thresholds),
                    "increments": list(self.tail_increments),
                    "cap": self.tail_weight_cap,
                }
            })

            print(f"[trial {trial_idx:02d}] mean_r2={mean_r2:.4f} | mean_mae={mean_mae:.3f}")

            is_better = (mean_r2 > best_mean_r2) or (np.isclose(mean_r2, best_mean_r2) and mean_mae < best_mean_mae)
            if is_better:
                best_mean_r2 = mean_r2
                best_mean_mae = mean_mae
                best_params = override.copy()

        print("\n" + "-" * 70)
        print("BEST TUNED PARAMS (rolling-year CV):")
        print(best_params)
        print(f"Best mean R²:  {best_mean_r2:.4f}")
        print(f"Best mean MAE: {best_mean_mae:.3f} μg/m³")

        if best_params is not None:
            self.LGBM_PARAMS.update(best_params)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_path = self.results_dir / f"tuning_randomsearch_{timestamp}.json"
        with open(tuning_path, "w") as f:
            json.dump(self.to_jsonable({
                "timestamp": timestamp,
                "n_trials": n_trials,
                "cv_val_years": list(cv_val_years),
                "best_params": best_params,
                "best_mean_r2": best_mean_r2,
                "best_mean_mae": best_mean_mae,
                "tail_weighting": {
                    "use_tail_weights": self.use_tail_weights,
                    "thresholds": list(self.tail_thresholds),
                    "increments": list(self.tail_increments),
                    "cap": self.tail_weight_cap,
                },
                "all_trials": all_trials,
            }), f, indent=2)

        print(f"Tuning report saved to: {tuning_path}")
        return best_params, tuning_path

    # ------------------------------------------------------------------
    # CV aggregations
    # ------------------------------------------------------------------
    def aggregate_cv_metrics(self, cv_metrics_list):
        if not cv_metrics_list:
            return {}
        keys = cv_metrics_list[0].keys()
        summary = {}
        for k in keys:
            vals = np.array([m[k] for m in cv_metrics_list if k in m and m[k] is not None], dtype=float)
            if vals.size == 0:
                continue
            summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        return summary

    def compute_monthly_metrics(self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
        monthly = {}
        for month in sorted(data["month"].unique()):
            mask = data["month"] == month
            monthly[str(int(month))] = self.evaluate_predictions(y_true[mask], y_pred[mask], verbose=False)
        return monthly

    def compute_seasonal_metrics(self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
        seasonal = {}
        for season_name, months in self.SEASONS.items():
            mask = data["month"].isin(months)
            if not mask.any():
                continue
            seasonal[season_name] = self.evaluate_predictions(y_true[mask], y_pred[mask], verbose=False)
        return seasonal

    def run_yearly_rolling_cv(self, cv_val_years=(2021, 2022, 2023, 2024)):
        full_data = self.load_and_prepare_data()
        all_years = sorted(full_data["year"].unique())

        cv_results = {}
        val_metrics_list = []

        print("\n" + "=" * 80)
        print("YEAR-WISE ROLLING CV FOR FIXED HYPERPARAMS")
        print("=" * 80)

        for val_year in cv_val_years:
            train_years = [y for y in all_years if y < val_year]
            if not train_years or val_year not in all_years:
                print(f"Skipping fold with val_year={val_year} (insufficient data)")
                continue

            print(f"\n--- CV Fold: Train years {train_years} -> Val year {val_year} ---")
            train_data = self.get_year_data(train_years)
            val_data = self.get_year_data([val_year])

            X_train, y_train, feat_cols, med = self.prepare_features(train_data, fit_medians=True)
            X_val, y_val, _, _ = self.prepare_features(val_data, feature_cols=feat_cols, train_medians=med)

            print(f"Train samples: {len(y_train):,}, Val samples: {len(y_val):,}")

            w_train = self.make_sample_weights(y_train)
            w_val = self.make_sample_weights(y_val)

            model = self.train_lgbm_model(X_train, y_train, X_val, y_val, w_train=w_train, w_val=w_val)

            val_pred = model.predict(X_val)
            val_metrics = self.evaluate_predictions(val_data["pm25"].values, val_pred, verbose=False)

            cv_results[str(val_year)] = {
                "train_years": train_years,
                "val_year": val_year,
                "val_metrics": val_metrics,
            }
            val_metrics_list.append(val_metrics)

        summary = self.aggregate_cv_metrics(val_metrics_list)
        print("\n=== CV Summary over validation years {} ===".format(cv_val_years))
        for k, stats in summary.items():
            print(f"{k:12s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

        return cv_results, summary

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run_production_training(
        self,
        final_train_years=(2020, 2021, 2022, 2023),
        dev_year=2024,
        test_year=2025,
        do_tuning: bool = True,
        tuning_trials: int = 25,
    ):
        full_data = self.load_and_prepare_data()
        all_years = sorted(full_data["year"].unique())

        cv_val_years = tuple(y for y in (2021, 2022, 2023, 2024) if y in all_years)
        cv_results, cv_summary = self.run_yearly_rolling_cv(cv_val_years=cv_val_years)

        if do_tuning:
            best_params, tuning_path = self.tune_hyperparameters_rolling_cv(
                n_trials=tuning_trials,
                cv_val_years=tuple(y for y in (2022, 2023, 2024) if y in all_years),
            )
        else:
            best_params, tuning_path = None, None

        print("\n" + "=" * 80)
        print(f"FINAL MODEL TRAINING (dev={dev_year} & test={test_year}) | tuned={do_tuning}")
        print("(DIRECT PM2.5 target)")
        if self.use_tail_weights:
            print(f"(Tail weights ON: thresholds={self.tail_thresholds}, inc={self.tail_increments}, cap={self.tail_weight_cap})")
        print("=" * 80)

        final_train_years = [y for y in final_train_years if y in all_years and y < dev_year]
        if not final_train_years:
            raise ValueError("No valid final_train_years found in data.")
        if dev_year not in all_years:
            raise ValueError(f"dev_year={dev_year} not found in data.")
        if test_year not in all_years:
            print(f"Warning: test_year={test_year} not found in data. Available years: {all_years}")
            print("Continuing without test evaluation.")
            test_year = None

        train_data = self.get_year_data(final_train_years)
        dev_data = self.get_year_data([dev_year])
        test_data = self.get_year_data([test_year]) if test_year is not None else None

        X_train, y_train, feat_cols, med = self.prepare_features(train_data, fit_medians=True)
        X_dev, y_dev, _, _ = self.prepare_features(dev_data, feature_cols=feat_cols, train_medians=med)

        if test_data is not None:
            X_test, y_test, _, _ = self.prepare_features(test_data, feature_cols=feat_cols, train_medians=med)
        else:
            X_test = y_test = None

        print(f"Train years: {final_train_years} | Train n={len(y_train):,} | Dev n={len(y_dev):,}")
        if test_data is not None:
            print(f"Test year: {test_year} | Test n={len(y_test):,}")

        w_train = self.make_sample_weights(y_train)
        w_dev = self.make_sample_weights(y_dev)

        final_model = self.train_lgbm_model(X_train, y_train, X_dev, y_dev, w_train=w_train, w_val=w_dev)

        # Feature importance
        feature_importance = final_model.feature_importances_
        feature_names = feat_cols
        importance_list = list(zip(feature_names, feature_importance))
        importance_list.sort(key=lambda x: x[1], reverse=True)

        sat_tags = {
            "no2_": "[NO2]",
            "so2_": "[SO2]",
            "co_": "[CO]",
            "hcho_": "[HCHO]",
            "aai_": "[AAI]",
            "alh_": "[ALH]",
            "ch4_": "[CH4]",
            "cloud_": "[CLOUD]",
            "o3_": "[O3]",
            "nh3_": "[NH3]",
        }

        def _feat_type(feature: str) -> str:
            """Categorize feature by registry stage or naming patterns"""
            f = feature.lower()
            
            # Check registry stage information first
            if hasattr(self.registry, 'registry') and not self.registry.registry.empty:
                reg_entry = self.registry.registry[self.registry.registry['column_name'] == feature]
                if not reg_entry.empty:
                    stage = reg_entry['stage'].iloc[0].upper()
                    return f"[{stage}]"
            
            # Fallback to naming patterns
            if "elevation" in f:
                return "[ELEVATION]"
            if any(x in f for x in ["dist_to_coast", "coastal"]):
                return "[DISTANCE]"
            if any(x in f for x in ["optical_depth", "aod_"]):
                return "[AOD]"
            for prefix, tag in sat_tags.items():
                if f.startswith(prefix):
                    return tag
            if any(x in f for x in ["doy_", "heating_season", "burning_season", "WD10_"]):
                return "[TEMPORAL]"
            if any(x in f for x in ["obs_lat", "obs_lon", "latitude", "longitude"]):
                return "[SPATIAL]"
            return "[MET]"

        print("\nTop 30 features (final model):")
        for i, (feature, importance) in enumerate(importance_list[:30]):
            print(f"{i + 1:2d}. {feature:30s} {_feat_type(feature):8s}: {float(importance):.4f}")

        print("\n" + "=" * 80)
        print(f"DEV ANALYSIS ON {dev_year}")
        print("=" * 80)

        dev_pred = final_model.predict(X_dev)
        print(f"Overall dev-year ({dev_year}) metrics:")
        dev_overall_metrics = self.evaluate_predictions(dev_data["pm25"].values, dev_pred, verbose=True)

        print(f"\nMonthly metrics ({dev_year}):")
        dev_monthly_metrics = self.compute_monthly_metrics(dev_data, dev_data["pm25"].values, dev_pred)

        print(f"\nSeasonal metrics ({dev_year}):")
        dev_seasonal_metrics = self.compute_seasonal_metrics(dev_data, dev_data["pm25"].values, dev_pred)

        test_overall_metrics = test_monthly_metrics = test_seasonal_metrics = None
        if test_data is not None:
            print("\n" + "=" * 80)
            print(f"FINAL TEST ON {test_year}")
            print("=" * 80)

            test_pred = final_model.predict(X_test)
            print(f"Overall test-year ({test_year}) metrics:")
            test_overall_metrics = self.evaluate_predictions(test_data["pm25"].values, test_pred, verbose=True)

            print(f"\nMonthly metrics ({test_year}):")
            test_monthly_metrics = self.compute_monthly_metrics(test_data, test_data["pm25"].values, test_pred)

            print(f"\nSeasonal metrics ({test_year}):")
            test_seasonal_metrics = self.compute_seasonal_metrics(test_data, test_data["pm25"].values, test_pred)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"production_tuned_results_{timestamp}.json"
        model_path = self.results_dir / f"production_tuned_model_{timestamp}.joblib"

        model_payload = {
            "model": final_model,
            "feature_cols": feat_cols,
            "train_medians": med.to_dict(),
            "use_latlon": self.use_latlon,
            "lgbm_params": self.LGBM_PARAMS,
            # NEW: save weighting config too (handy for reproducibility)
            "use_tail_weights": self.use_tail_weights,
            "tail_thresholds": list(self.tail_thresholds),
            "tail_increments": list(self.tail_increments),
            "tail_weight_cap": self.tail_weight_cap,
        }
        joblib.dump(model_payload, model_path)

        results = {
            "model_type": "registry_authoritative_lgbm_daily_tuned_weighted",
            "use_latlon": self.use_latlon,
            "use_tail_weights": self.use_tail_weights,
            "tail_thresholds": list(self.tail_thresholds),
            "tail_increments": list(self.tail_increments),
            "tail_weight_cap": self.tail_weight_cap,
            "final_train_years": list(final_train_years),
            "dev_year": dev_year,
            "test_year": test_year,
            "features_used_full": list(feat_cols),
            "n_features_full": len(feat_cols),
            "train_samples": int(len(train_data)),
            "dev_samples": int(len(dev_data)),
            "test_samples": int(len(test_data)) if test_data is not None else 0,
            "best_params_from_tuning": best_params,
            "tuning_report_path": str(tuning_path) if tuning_path is not None else None,
            "cv": {
                "folds": cv_results,
                "summary_val_metrics": cv_summary,
            },
            "final_model": {
                "dev_overall": dev_overall_metrics,
                "dev_monthly": dev_monthly_metrics,
                "dev_seasonal": dev_seasonal_metrics,
                "test_overall": test_overall_metrics,
                "test_monthly": test_monthly_metrics,
                "test_seasonal": test_seasonal_metrics,
                "feature_importance": importance_list,
            },
            "timestamp": timestamp,
        }

        with open(results_path, "w") as f:
            json.dump(self.to_jsonable(results), f, indent=2)

        print(f"\nResults saved to: {results_path}")
        print(f"Final model payload saved to: {model_path}")
        print("=" * 80)

        return {
            "final_model": final_model,
            "dev_overall_metrics": dev_overall_metrics,
            "test_overall_metrics": test_overall_metrics,
            "feature_importance": importance_list,
            "results_path": results_path,
            "model_path": model_path,
        }


def main():
    trainer = Trainer(
        sensor_csv="paqi_with_all_features.csv",
        registry_csv="features_registry.csv",
        results_dir="results/temporal_lgbm_upweighted",
        use_latlon=False,
        strict_registry=True,
        # Tail weighting for extreme PM2.5 events
        use_tail_weights=True,
        # Default thresholds: +1 weight at 100, +3 at 150, +6 at 200 μg/m³
        # tail_thresholds=(100.0, 150.0, 200.0),
        # tail_increments=(1.0, 3.0, 6.0),
        # tail_weight_cap=12.0,
    )

    trainer.run_production_training(
        final_train_years=(2020, 2021, 2022, 2023),
        dev_year=2024,
        test_year=2025,
        do_tuning=True,
        tuning_trials=25,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
