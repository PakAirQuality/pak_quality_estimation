"""
Training Pipeline (Layer 3)
===========================

ML process: model training, tuning, evaluation, artifact saving.

Public API:
    from training.utils.trainer import Trainer, load_master_from_lake
"""

import json
import sys
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Layer 1: Data loading
from .data_loader import load_master_from_lake

# Layer 2: Dataset preparation
from .dataset_prep import (
    prepare_master_data,
    prepare_features,
    select_candidate_features,
    add_daily_encodings,
    TARGET_COL,
    DERIVED_FEATURES,
    SPATIAL_FEATURES,
    SPATIAL_FEATURES_FALLBACK,
    is_excluded_feature,
)

# FeatureRegistry import
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent
_MFE = _REPO_ROOT / "feature_engineering"
if _MFE.exists():
    sys.path.append(str(_MFE))

from feature_family.feature_registry import FeatureRegistry


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Training pipeline for PM2.5 estimation models.

    Orchestrates:
    - Data loading (via Layer 1)
    - Dataset preparation (via Layer 2)
    - Model training (LightGBM)
    - Hyperparameter tuning (Optuna)
    - Evaluation and artifact saving
    """

    def __init__(
        self,
        master_lake_path: str = "../feature_engineering/output/master_lake",
        registry_csv: str = "data/features_registry.csv",
        results_dir: str = "results/production",
        use_latlon: bool = False,
        strict_registry: bool = True,
    ):
        self.master_lake_path = master_lake_path
        if not str(master_lake_path).startswith("gs://"):
            if not Path(master_lake_path).exists():
                raise FileNotFoundError(f"Master lake not found: {master_lake_path}")

        self.registry_path = Path(registry_csv)
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Features registry not found: {self.registry_path}")

        # Initialize registry
        self.registry = FeatureRegistry()
        self.registry.registry_path = self.registry_path
        self.registry.metadata_dir = self.registry_path.parent / "metadata"
        self.registry.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.registry.registry = self.registry._load_or_init_registry()

        self.strict_registry = bool(strict_registry)
        self.use_latlon = bool(use_latlon)

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Registry-driven feature list
        self.registry_feature_cols: List[str] = self.registry.get_feature_columns()
        if not self.registry_feature_cols:
            raise ValueError(
                f"Feature registry loaded from {self.registry_path} but contained 0 trainable features."
            )

        # For feature importance tagging
        reg_df = self.registry.registry.copy()
        self._col_to_stage = dict(zip(reg_df["column_name"].astype(str), reg_df["stage"].astype(str)))

        # Default LightGBM params
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

        self.SEASONS = {
            "winter": [12, 1, 2],
            "pre_monsoon": [3, 4, 5],
            "monsoon": [6, 7, 8, 9],
            "post_monsoon": [10, 11],
        }

        self._full_data: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # Data Loading + Preparation (delegates to Layer 1 & 2)
    # -------------------------------------------------------------------------

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load raw data and prepare for training."""
        if self._full_data is not None:
            return self._full_data

        # Layer 1: Load raw data
        df = load_master_from_lake(self.master_lake_path)

        # Layer 2: Prepare data
        df = prepare_master_data(
            df,
            registry_feature_cols=self.registry_feature_cols,
            strict_registry=self.strict_registry,
            lake_path=str(self.master_lake_path),
            registry_path=str(self.registry_path),
        )

        self._full_data = df
        return df

    def get_year_data(self, years: list) -> pd.DataFrame:
        """Get data for specific years."""
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
    ):
        """Prepare feature matrix - delegates to Layer 2."""
        return prepare_features(
            data=data,
            registry_feature_cols=self.registry_feature_cols,
            feature_cols=feature_cols,
            train_medians=train_medians,
            fit_medians=fit_medians,
            min_non_nan_frac=min_non_nan_frac,
            use_latlon=self.use_latlon,
        )

    # For backwards compatibility
    def add_daily_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_daily_encodings(df)

    def _select_candidate_features(self, df: pd.DataFrame) -> List[str]:
        return select_candidate_features(df, self.registry_feature_cols, self.use_latlon)

    def _is_excluded_feature(self, col: str) -> bool:
        return is_excluded_feature(col)

    # -------------------------------------------------------------------------
    # Model Training
    # -------------------------------------------------------------------------

    def train_lgbm_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        override_params: Optional[dict] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_sample_weight: Optional[np.ndarray] = None,
    ) -> lgb.LGBMRegressor:
        """Train LightGBM model with early stopping."""
        params = self.LGBM_PARAMS.copy()
        if override_params:
            params.update(override_params)

        if params.get("n_estimators", 0) < 6000:
            params["n_estimators"] = 9000

        model = lgb.LGBMRegressor(**params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "l1",
            "callbacks": callbacks,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_sample_weight is not None:
            fit_kwargs["eval_sample_weight"] = [eval_sample_weight]

        model.fit(X_train, y_train, **fit_kwargs)
        return model

    # -------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------

    def tune_hyperparameters_rolling_cv(
        self,
        train_data: pd.DataFrame,
        n_trials: int = 200,
        cv_val_years: Optional[Tuple[int, ...]] = None,
        min_non_nan_frac: float = 0.01,
        seed: int = 42,
    ) -> Tuple[Optional[Dict], Optional[Path]]:
        """
        Tune hyperparameters using Optuna TPE with rolling temporal CV.

        Returns:
            (best_params, tuning_report_path) - does NOT mutate self.LGBM_PARAMS
        """
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner

        all_years = sorted(train_data["year"].unique())

        if cv_val_years is None:
            if len(all_years) >= 4:
                cv_val_years = tuple(all_years[-3:])
            elif len(all_years) >= 3:
                cv_val_years = tuple(all_years[-2:])
            else:
                cv_val_years = tuple(all_years[-1:])

        cv_val_years = tuple([y for y in cv_val_years if y in all_years])
        if len(cv_val_years) == 0:
            print("WARNING: No valid CV years found. Skipping hyperparameter tuning.")
            return None, None

        print("\n" + "=" * 80)
        print(f"HYPERPARAM TUNING (Optuna TPE, MAE-primary) | trials={n_trials} | folds={cv_val_years}")
        print("=" * 80)

        # Pre-split data for each fold
        fold_data = {}
        for val_year in cv_val_years:
            train_years = [y for y in all_years if y < val_year]
            if not train_years:
                continue
            fold_train = train_data[train_data["year"].isin(train_years)].copy()
            fold_val = train_data[train_data["year"] == val_year].copy()
            if len(fold_train) == 0 or len(fold_val) == 0:
                continue
            X_tr, y_tr, feat_cols, med = self.prepare_features(
                fold_train, fit_medians=True, min_non_nan_frac=min_non_nan_frac
            )
            X_va, y_va, _, _ = self.prepare_features(
                fold_val, feature_cols=feat_cols, train_medians=med, fit_medians=False
            )
            fold_data[val_year] = (X_tr, y_tr, X_va, y_va)

        trainer_ref = self

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1])
            n_estimators = 15000 if lr <= 0.01 else (12000 if lr <= 0.03 else (9000 if lr <= 0.05 else 7000))

            override = {
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "num_leaves": trial.suggest_int("num_leaves", 15, 2047, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "learning_rate": lr,
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
                "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
                "n_estimators": n_estimators,
                "random_state": seed,
            }

            fold_maes = []
            for fold_idx, val_year in enumerate(sorted(fold_data.keys())):
                X_tr, y_tr, X_va, y_va = fold_data[val_year]
                model = trainer_ref.train_lgbm_model(
                    X_tr, y_tr, X_va, y_va, override_params=override
                )
                val_pred = model.predict(X_va)
                mae = float(np.nanmean(np.abs(y_va - val_pred)))
                fold_maes.append(mae)

                trial.report(float(np.mean(fold_maes)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_maes))

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=seed),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
            study_name="lgbm_mae_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_trial = study.best_trial
        best_params = best_trial.params.copy()

        lr = best_params["learning_rate"]
        best_params["n_estimators"] = 15000 if lr <= 0.01 else (12000 if lr <= 0.03 else (9000 if lr <= 0.05 else 7000))
        best_params["random_state"] = seed

        print("\n" + "-" * 70)
        print("BEST TUNED PARAMS (Optuna TPE, rolling-year CV):")
        for k, v in sorted(best_params.items()):
            print(f"  {k}: {v}")
        print(f"Best mean MAE: {best_trial.value:.3f} ug/m3")
        print(f"Completed trials: {len(study.trials)}")
        print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

        # Save tuning report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_path = self.results_dir / f"tuning_optuna_{timestamp}.json"
        with open(tuning_path, "w") as f:
            json.dump(self._to_jsonable({
                "timestamp": timestamp,
                "method": "optuna_tpe",
                "n_trials": n_trials,
                "completed_trials": len(study.trials),
                "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "cv_val_years": list(cv_val_years),
                "best_params": best_params,
                "best_mean_mae": best_trial.value,
                "all_trials": [
                    {
                        "number": t.number,
                        "value": t.value if t.value is not None else None,
                        "params": t.params,
                        "state": str(t.state),
                    }
                    for t in study.trials
                ],
            }), f, indent=2)

        print(f"Tuning report saved to: {tuning_path}")
        return best_params, tuning_path

    # -------------------------------------------------------------------------
    # Metrics (delegated to training.utils.metrics)
    # -------------------------------------------------------------------------

    def evaluate_predictions(self, y_true, y_pred, verbose=True):
        from training.utils.metrics import evaluate_predictions
        return evaluate_predictions(y_true, y_pred, verbose=verbose)

    def compute_monthly_metrics(self, data, y_true, y_pred):
        from training.utils.metrics import compute_monthly_metrics
        return compute_monthly_metrics(data, y_true, y_pred)

    def compute_seasonal_metrics(self, data, y_true, y_pred):
        from training.utils.metrics import compute_seasonal_metrics
        return compute_seasonal_metrics(data, y_true, y_pred, seasons=self.SEASONS)

    def compute_mae_skill(self, mae_model, mae_baseline):
        from training.utils.metrics import compute_mae_skill
        return compute_mae_skill(mae_model, mae_baseline)

    def compute_skill_metrics(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        station_col: str = "sensor_id",
        target_col: str = "pm25",
        verbose: bool = True,
    ) -> Dict:
        """Compute skill scores vs climatology and persistence baselines."""
        from training.baselines import compute_climatology_baseline, compute_persistence_baseline

        # Model metrics
        model_metrics = self.evaluate_predictions(y_true, y_pred, verbose=False)

        # Climatology baseline
        y_clim = compute_climatology_baseline(
            train_data, test_data, station_col=station_col, target_col=target_col
        )
        valid_clim = ~np.isnan(y_clim) & ~np.isnan(y_true)
        clim_metrics = self.evaluate_predictions(
            y_true[valid_clim], y_clim[valid_clim], verbose=False
        ) if valid_clim.sum() > 0 else {}

        # Persistence baseline
        y_persist = compute_persistence_baseline(
            train_data, test_data, station_col=station_col, target_col=target_col
        )
        valid_persist = ~np.isnan(y_persist) & ~np.isnan(y_true)
        persist_metrics = self.evaluate_predictions(
            y_true[valid_persist], y_persist[valid_persist], verbose=False
        ) if valid_persist.sum() > 0 else {}

        # Compute skill scores
        skill_vs_clim = self.compute_mae_skill(
            model_metrics.get("mae", np.nan),
            clim_metrics.get("mae", np.nan)
        ) if clim_metrics else np.nan

        skill_vs_persist = self.compute_mae_skill(
            model_metrics.get("mae", np.nan),
            persist_metrics.get("mae", np.nan)
        ) if persist_metrics else np.nan

        if verbose:
            print(f"Model MAE:       {model_metrics.get('mae', np.nan):.2f} ug/m3")
            print(f"Climatology MAE: {clim_metrics.get('mae', np.nan):.2f} ug/m3")
            print(f"Persistence MAE: {persist_metrics.get('mae', np.nan):.2f} ug/m3")
            print(f"Skill vs Climatology: {skill_vs_clim:+.3f} ({skill_vs_clim:+.1%})")
            print(f"Skill vs Persistence: {skill_vs_persist:+.3f} ({skill_vs_persist:+.1%})")

        return {
            "model_metrics": model_metrics,
            "clim_metrics": clim_metrics,
            "persist_metrics": persist_metrics,
            "skill_vs_climatology": skill_vs_clim,
            "skill_vs_persistence": skill_vs_persist,
        }

    # -------------------------------------------------------------------------
    # CV Aggregation
    # -------------------------------------------------------------------------

    def aggregate_cv_metrics(self, cv_metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across CV folds."""
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

    # -------------------------------------------------------------------------
    # JSON Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        return self._json_safe(obj)

    # Backwards compatibility aliases
    def json_safe(self, obj):
        return self._json_safe(obj)

    def to_jsonable(self, obj):
        return self._to_jsonable(obj)

    # -------------------------------------------------------------------------
    # Operational Training (Orchestrator)
    # -------------------------------------------------------------------------

    def run_operational_training(
        self,
        reference_date: Optional[str] = None,
        cutoff_days: int = 90,
        validation_days: int = 90,
        min_train_years: int = 2,
        do_tuning: bool = True,
        tuning_trials: int = 25,
    ):
        """
        Train a model aligned to the 90-day serving waterline.

        Args:
            reference_date: The "today" for computing waterline (YYYY-MM-DD)
            cutoff_days: Days before reference_date for waterline (default 90)
            validation_days: Size of validation window before waterline (default 90)
            min_train_years: Minimum years of training data required
            do_tuning: Whether to run hyperparameter tuning (default True)
            tuning_trials: Number of hyperparameter tuning trials (default 25)

        Returns:
            Dict with model, metrics, and paths
        """
        # Compute waterline
        if reference_date is None:
            t0 = date.today()
        else:
            t0 = datetime.strptime(reference_date, "%Y-%m-%d").date()

        waterline = t0 - timedelta(days=cutoff_days)
        val_end = waterline
        val_start = waterline - timedelta(days=validation_days)

        print("\n" + "=" * 80)
        print("OPERATIONAL TRAINING (waterline-aligned)")
        print("=" * 80)
        print(f"Reference date (t0):     {t0}")
        print(f"Cutoff days:             {cutoff_days}")
        print(f"Waterline (W):           {waterline}")
        print(f"Validation window:       {val_start} to {val_end} ({validation_days} days)")
        print(f"Train cutoff:            < {val_start}")
        print("=" * 80)

        # Load and prepare data
        full_data = self.load_and_prepare_data()

        if "year" not in full_data.columns:
            full_data["year"] = pd.to_datetime(full_data["time"]).dt.year

        time_col = "time" if "time" in full_data.columns else "timestamp"
        if time_col not in full_data.columns:
            raise ValueError(f"No time column found. Available: {list(full_data.columns)[:20]}")
        full_data["date"] = pd.to_datetime(full_data[time_col]).dt.date

        # Split by waterline
        train_mask = full_data["date"] < val_start
        val_mask = (full_data["date"] >= val_start) & (full_data["date"] <= val_end)

        train_data = full_data[train_mask].copy()
        val_data = full_data[val_mask].copy()

        if len(train_data) == 0:
            raise ValueError(f"No training data before validation start {val_start}")
        if len(val_data) == 0:
            raise ValueError(f"No validation data in window {val_start} to {val_end}")

        train_years = train_data["year"].nunique()
        if train_years < min_train_years:
            raise ValueError(
                f"Only {train_years} years of training data, need at least {min_train_years}"
            )

        train_date_range = (train_data["date"].min(), train_data["date"].max())
        val_date_range = (val_data["date"].min(), val_data["date"].max())

        print(f"\nTrain data: {len(train_data):,} samples")
        print(f"  Date range: {train_date_range[0]} to {train_date_range[1]}")
        print(f"  Years: {sorted(train_data['year'].unique())}")
        print(f"\nValidation data: {len(val_data):,} samples")
        print(f"  Date range: {val_date_range[0]} to {val_date_range[1]}")

        # Hyperparameter tuning
        tuning_path = None
        best_params = None
        if do_tuning:
            print(f"\n[Hyperparameter Tuning] Running {tuning_trials} trials...")
            best_params, tuning_path = self.tune_hyperparameters_rolling_cv(
                train_data=train_data,
                n_trials=tuning_trials,
                cv_val_years=None,
            )
            if best_params:
                self.LGBM_PARAMS.update(best_params)
                print(f"[Hyperparameter Tuning] Best params applied to model training")
            else:
                print(f"[Hyperparameter Tuning] Using default params (tuning found no improvement)")
        else:
            print("\n[Hyperparameter Tuning] Skipped (do_tuning=False)")

        # Prepare features
        X_train, y_train, feat_cols, med = self.prepare_features(train_data, fit_medians=True)
        X_val, y_val, _, _ = self.prepare_features(
            val_data, feature_cols=feat_cols, train_medians=med, fit_medians=False
        )

        # Train model
        print("\nTraining final model...")
        model = self.train_lgbm_model(X_train, y_train, X_val, y_val)

        # Evaluate
        val_pred = model.predict(X_val)
        print("\nValidation metrics:")
        val_metrics = self.evaluate_predictions(y_val, val_pred, verbose=True)

        # Monthly breakdown
        print("\nValidation monthly metrics:")
        val_monthly = self.compute_monthly_metrics(val_data, y_val, val_pred)

        # Feature importance
        feature_importance = model.feature_importances_
        importance_list = sorted(
            zip(feat_cols, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )

        # Save artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        waterline_str = waterline.strftime("%Y%m%d")

        model_path = self.results_dir / f"operational_model_W{waterline_str}_{timestamp}.joblib"
        results_path = self.results_dir / f"operational_results_W{waterline_str}_{timestamp}.json"

        model_payload = {
            "model": model,
            "feature_cols": feat_cols,
            "train_medians": med.to_dict(),
            "use_latlon": self.use_latlon,
            "lgbm_params": self.LGBM_PARAMS,
            "registry_csv": str(self.registry_path),
            "master_lake_path": str(self.master_lake_path),
            "training_mode": "operational_waterline",
            "reference_date": str(t0),
            "waterline": str(waterline),
            "cutoff_days": cutoff_days,
            "validation_days": validation_days,
            "train_date_range": [str(train_date_range[0]), str(train_date_range[1])],
            "val_date_range": [str(val_date_range[0]), str(val_date_range[1])],
        }
        joblib.dump(model_payload, model_path)

        results = {
            "training_mode": "operational_waterline",
            "reference_date": str(t0),
            "waterline": str(waterline),
            "cutoff_days": cutoff_days,
            "validation_days": validation_days,
            "train_date_range": [str(train_date_range[0]), str(train_date_range[1])],
            "val_date_range": [str(val_date_range[0]), str(val_date_range[1])],
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "train_years": sorted(train_data["year"].unique().tolist()),
            "n_features": len(feat_cols),
            "val_metrics": val_metrics,
            "val_monthly": val_monthly,
            "feature_importance_top30": importance_list[:30],
            "model_path": str(model_path),
            "timestamp": timestamp,
            "do_tuning": do_tuning,
            "tuning_trials": tuning_trials if do_tuning else 0,
            "best_params": best_params,
            "tuning_report_path": str(tuning_path) if tuning_path else None,
            "lgbm_params_used": self.LGBM_PARAMS,
        }

        with open(results_path, "w") as f:
            json.dump(self._to_jsonable(results), f, indent=2)

        print(f"\nModel saved to: {model_path}")
        print(f"Results saved to: {results_path}")
        print("=" * 80)

        return {
            "model": model,
            "model_path": model_path,
            "results_path": results_path,
            "val_metrics": val_metrics,
            "waterline": waterline,
            "feature_cols": feat_cols,
            "train_medians": med,
            "n_train": len(train_data),
            "n_val": len(val_data),
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Operational (waterline-aligned) model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lake_path", type=str, default="../feature_engineering/output/master_lake")
    parser.add_argument("--registry_csv", type=str, default="data/features_registry.csv")
    parser.add_argument("--results_dir", type=str, default="results/production")
    parser.add_argument("--use_latlon", action="store_true")
    parser.add_argument("--reference_date", type=str, default=None)
    parser.add_argument("--cutoff_days", type=int, default=90)
    parser.add_argument("--validation_days", type=int, default=90)
    parser.add_argument("--no_tuning", action="store_true")
    parser.add_argument("--do_tuning", action="store_true")
    parser.add_argument("--tuning_trials", type=int, default=25)

    args = parser.parse_args()

    do_tuning = True
    if args.no_tuning:
        do_tuning = False
    elif args.do_tuning:
        do_tuning = True

    trainer = Trainer(
        master_lake_path=args.lake_path,
        registry_csv=args.registry_csv,
        results_dir=args.results_dir,
        use_latlon=args.use_latlon,
        strict_registry=True,
    )

    trainer.run_operational_training(
        reference_date=args.reference_date,
        cutoff_days=args.cutoff_days,
        validation_days=args.validation_days,
        do_tuning=do_tuning,
        tuning_trials=args.tuning_trials,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
