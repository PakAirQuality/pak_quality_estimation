"""
Regulation Alignment Ablation Study
===================================

Ablation study comparing approaches to align LCS predictions with EPA
regulatory-grade reference stations.

Two-stage base model:
  Stage 1: Backbone LightGBM trained on all LCS (low-cost sensor) data.
  Stage 2: Residual-correction LightGBM trained on EPA anchor stations only.
  Final prediction: backbone(x) + residual(x)

Approaches compared:
  - backbone_only: No EPA correction (ablation baseline)
  - anchor_only: Train only on EPA, no LCS backbone (ablation baseline)
  - global: Joint training on LCS + EPA
  - global_weighted: Joint training with EPA upweighted
  - backbone_residual: Two-stage backbone + learned residual
  - daily_offset: Per-day median offset calibration
  - daily_affine: Per-day affine (scale + offset) calibration
  - residual_plus_offset: Hybrid - residual + daily offset
  - global_plus_offset: Hybrid - global + daily offset

Prerequisite:
  The master lake must include EPA data (2025-10-27 onward).

Usage:
    # Module-style invocation
    python -m training.ablation.regulation_alignment \\
        --station_lake_path extraction_and_preprocessing/station_labels/lake/station_daily \\
        --do_spatial_cv --do_temporal_cv --no_tuning

    # Import in code
    from training.ablation.regulation_alignment import AnchoredTrainer
"""

import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterSampler

warnings.filterwarnings("ignore")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.main import BenchmarkTrainer
from training.ablation.anchor_approaches import (
    fit_daily_offset,
    predict_daily_offset,
    fit_daily_affine,
    predict_daily_affine,
    train_two_stage,
    predict_two_stage,
    train_global_model,
    train_global_weighted_model,
    train_anchor_only_model,
)


# ---------------------------------------------------------------------------
# Deterministic sensor_id (mirrored from feature_family/met_features.py)
# ---------------------------------------------------------------------------
def _create_deterministic_sensor_id(lat: float, lon: float) -> int:
    lat_rounded = round(float(lat), 5)
    lon_rounded = round(float(lon), 5)
    id_string = f"{lat_rounded:.5f},{lon_rounded:.5f}"
    h = hashlib.sha1(id_string.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


class AnchoredTrainer(BenchmarkTrainer):
    """
    Two-stage Backbone + Anchor Residual trainer.

    Stage 1 trains a LightGBM backbone on all LCS data.
    Stage 2 trains a residual-correction LightGBM on EPA anchor stations.
    Final prediction = backbone(x) + residual(x).
    """

    EPA_PROVIDER = "Punjab EPA"

    def __init__(
        self,
        station_lake_path: str = "../extraction_and_preprocessing/station_labels/lake/station_daily",
        master_lake_path: str = "../feature_engineering/output/master_lake",
        registry_csv: str = "data/features_registry.csv",
        results_dir: str = "results/anchored",
        use_latlon: bool = False,
        strict_registry: bool = True,
    ):
        super().__init__(
            master_lake_path=master_lake_path,
            registry_csv=registry_csv,
            results_dir=results_dir,
            use_latlon=use_latlon,
            strict_registry=strict_registry,
        )
        self.station_lake_path = Path(station_lake_path)
        if not self.station_lake_path.exists():
            raise FileNotFoundError(
                f"Station daily lake not found: {self.station_lake_path}"
            )
        self._provider_lookup: Optional[Dict[int, str]] = None
        self.lambda_ = 1.0

    # ------------------------------------------------------------------
    # Provider lookup
    # ------------------------------------------------------------------
    def _build_provider_lookup(self) -> Dict[int, str]:
        """Build sensor_id -> provider_name lookup from station_daily lake."""
        if self._provider_lookup is not None:
            return self._provider_lookup

        print("\nBuilding provider lookup from station_daily lake...")
        date_dirs = sorted([
            d for d in self.station_lake_path.iterdir()
            if d.is_dir() and d.name.startswith("date=")
        ])
        if not date_dirs:
            raise FileNotFoundError(
                f"No date partitions in station_daily lake: {self.station_lake_path}"
            )

        # Collect unique (lat, lon, provider_name) tuples across all partitions
        records: Dict[Tuple[float, float], str] = {}
        for dd in date_dirs:
            for pf in dd.glob("*.parquet"):
                try:
                    df = pd.read_parquet(
                        pf, columns=["latitude", "longitude", "provider_name"]
                    )
                    for _, row in df.iterrows():
                        lat = row["latitude"]
                        lon = row["longitude"]
                        prov = row["provider_name"]
                        if pd.notna(lat) and pd.notna(lon) and pd.notna(prov):
                            records[(float(lat), float(lon))] = str(prov)
                except Exception:
                    continue

        # Compute sensor_id for each unique location
        lookup: Dict[int, str] = {}
        for (lat, lon), prov in records.items():
            sid = _create_deterministic_sensor_id(lat, lon)
            lookup[sid] = prov

        n_epa = sum(1 for v in lookup.values() if v == self.EPA_PROVIDER)
        print(f"  Provider lookup: {len(lookup)} stations total, {n_epa} EPA stations")
        self._provider_lookup = lookup
        return lookup

    # ------------------------------------------------------------------
    # Tag and split
    # ------------------------------------------------------------------
    def _tag_provider(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add provider_name and is_epa columns to master lake data."""
        lookup = self._build_provider_lookup()
        data = data.copy()
        data["provider_name"] = data["sensor_id"].map(lookup).fillna("Unknown")
        data["is_epa"] = data["provider_name"] == self.EPA_PROVIDER
        return data

    def _split_lcs_epa(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into LCS and EPA subsets. Raises if no EPA data found."""
        if "is_epa" not in data.columns:
            data = self._tag_provider(data)

        epa_data = data[data["is_epa"]].copy()
        lcs_data = data[~data["is_epa"]].copy()

        if len(epa_data) == 0:
            raise ValueError(
                "No EPA data found in master lake. The master lake likely needs to "
                "be extended through Dec 2025 via the feature pipeline. EPA data "
                "exists only from 2025-10-27 onward."
            )

        n_epa_stations = epa_data["sensor_id"].nunique()
        n_lcs_stations = lcs_data["sensor_id"].nunique()
        print(f"\n  LCS: {len(lcs_data):,} rows, {n_lcs_stations} stations")
        print(f"  EPA: {len(epa_data):,} rows, {n_epa_stations} stations")

        return lcs_data, epa_data

    # ------------------------------------------------------------------
    # Stage 1: Backbone hyperparameter tuning (rolling-year CV on LCS)
    # ------------------------------------------------------------------
    def tune_backbone_params(
        self,
        n_trials: int = 25,
        cv_val_years: Optional[Tuple[int, ...]] = None,
        seed: int = 42,
    ) -> Tuple[Optional[dict], Optional[Path]]:
        """
        Tune backbone hyperparameters via rolling-year CV on LCS data only.

        Uses the same search space as the paper module (11 hyperparameters).
        Selects best by R2 (tiebreaker: MAE).

        Returns (best_backbone_params, tuning_report_path).
        """
        print("\n" + "=" * 80)
        print(f"BACKBONE TUNING (rolling-year CV on LCS) | trials={n_trials}")
        print("=" * 80)

        full_data = self.load_and_prepare_data()
        full_data = self._tag_provider(full_data)
        lcs_data, _ = self._split_lcs_epa(full_data)

        all_years = sorted(lcs_data["year"].unique())
        if cv_val_years is None:
            cv_val_years = tuple(y for y in [2022, 2023, 2024] if y in all_years)
        else:
            cv_val_years = tuple(y for y in cv_val_years if y in all_years)

        if len(cv_val_years) == 0:
            raise ValueError("No cv_val_years found in LCS data.")

        print(f"  LCS years: {all_years}")
        print(f"  CV val years: {list(cv_val_years)}")

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

        for trial_idx, sampled in enumerate(sampler, start=1):
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
                if not train_years:
                    continue

                train_data = lcs_data[lcs_data["year"].isin(train_years)].copy()
                val_data = lcs_data[lcs_data["year"] == val_year].copy()

                if len(train_data) == 0 or len(val_data) == 0:
                    continue

                X_train, y_train, feat_cols, med = self.prepare_features(
                    train_data, fit_medians=True
                )
                X_val, y_val, _, _ = self.prepare_features(
                    val_data, feature_cols=feat_cols, train_medians=med,
                    fit_medians=False,
                )

                model = self.train_lgbm_model(
                    X_train, y_train, X_val, y_val, override_params=override,
                )
                val_pred = model.predict(X_val)
                metrics = self.evaluate_predictions(y_val, val_pred, verbose=False)
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
            })

            print(f"  [backbone trial {trial_idx:02d}] mean_r2={mean_r2:.4f} | mean_mae={mean_mae:.3f}")

            is_better = (mean_r2 > best_mean_r2) or (
                np.isclose(mean_r2, best_mean_r2) and mean_mae < best_mean_mae
            )
            if is_better:
                best_mean_r2 = mean_r2
                best_mean_mae = mean_mae
                best_params = override.copy()

        print("\n" + "-" * 70)
        print("BEST BACKBONE PARAMS (rolling-year CV on LCS):")
        print(best_params)
        print(f"Best mean R2:  {best_mean_r2:.4f}")
        print(f"Best mean MAE: {best_mean_mae:.3f} ug/m3")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_path = self.results_dir / f"backbone_tuning_{timestamp}.json"
        with open(tuning_path, "w") as f:
            json.dump(self.to_jsonable({
                "stage": "backbone",
                "timestamp": timestamp,
                "n_trials": n_trials,
                "cv_val_years": list(cv_val_years),
                "best_params": best_params,
                "best_mean_r2": best_mean_r2,
                "best_mean_mae": best_mean_mae,
                "all_trials": all_trials,
            }), f, indent=2)

        print(f"Backbone tuning report saved to: {tuning_path}")
        return best_params, tuning_path

    # ------------------------------------------------------------------
    # Stage 2: Residual hyperparameter tuning (spatial CV on EPA)
    # ------------------------------------------------------------------
    def tune_residual_params(
        self,
        backbone_params: Optional[dict] = None,
        n_trials: int = 25,
        k: int = 5,
        seed: int = 42,
    ) -> Tuple[Optional[dict], float, Optional[Path]]:
        """
        Tune residual hyperparameters via spatial leave-stations-out CV on EPA.

        Backbone is trained ONCE on all LCS data with fixed backbone_params.
        For each trial, the residual model is trained per fold, and all lambda
        values are evaluated on stored predictions (no re-training per lambda).

        Returns (best_residual_params, best_lambda, tuning_report_path).
        """
        print("\n" + "=" * 80)
        print(f"RESIDUAL TUNING (spatial CV on EPA) | trials={n_trials}")
        print("=" * 80)

        full_data = self.load_and_prepare_data()
        full_data = self._tag_provider(full_data)
        lcs_data, epa_data = self._split_lcs_epa(full_data)

        # Fit medians on LCS
        _, _, feat_cols, medians = self.prepare_features(lcs_data, fit_medians=True)

        # Train backbone ONCE on all LCS data
        print("  Training backbone once on all LCS data...")
        rng = np.random.RandomState(seed)
        X_lcs, y_lcs, _, _ = self.prepare_features(
            lcs_data, feature_cols=feat_cols, train_medians=medians, fit_medians=False,
        )
        n_lcs = len(X_lcs)
        lcs_val_idx = rng.choice(n_lcs, size=max(1, n_lcs // 5), replace=False)
        lcs_train_mask = np.ones(n_lcs, dtype=bool)
        lcs_train_mask[lcs_val_idx] = False
        backbone = self.train_lgbm_model(
            X_lcs[lcs_train_mask], y_lcs[lcs_train_mask],
            X_lcs[~lcs_train_mask], y_lcs[~lcs_train_mask],
            override_params=backbone_params,
        )

        # Prepare spatial CV folds on EPA stations
        epa_sensor_ids = np.array(sorted(epa_data["sensor_id"].unique()))
        n_stations = len(epa_sensor_ids)
        actual_k = min(k, n_stations)
        print(f"  EPA stations for residual tuning: {n_stations}, using {actual_k} folds")

        kf = KFold(n_splits=actual_k, shuffle=True, random_state=seed)
        folds = list(kf.split(epa_sensor_ids))

        # Residual search space
        residual_param_space = {
            "max_depth": [3, 4, 5, 6],
            "num_leaves": [31, 47, 63],
            "min_child_samples": [200, 400, 600, 800],
            "reg_lambda": [1, 5, 10, 20, 50],
            "learning_rate": [0.02, 0.05],
        }
        sampler = ParameterSampler(
            residual_param_space, n_iter=n_trials, random_state=seed,
        )

        lambda_grid = list(np.round(np.arange(0.0, 0.55, 0.05), 2))  # [0.0, 0.05, ..., 0.50]

        best_residual_params = None
        best_lambda = 1.0
        best_mae = 1e9
        all_trials = []

        for trial_idx, sampled in enumerate(sampler, start=1):
            override = {
                "max_depth": int(sampled["max_depth"]),
                "num_leaves": int(sampled["num_leaves"]),
                "min_child_samples": int(sampled["min_child_samples"]),
                "reg_lambda": float(sampled["reg_lambda"]),
                "learning_rate": float(sampled["learning_rate"]),
                "random_state": seed,
            }

            # Collect predictions across folds
            all_backbone_preds = []
            all_residual_preds = []
            all_y_true = []

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                train_sids = set(epa_sensor_ids[train_idx])
                test_sids = set(epa_sensor_ids[test_idx])

                epa_train = epa_data[epa_data["sensor_id"].isin(train_sids)].copy()
                epa_test = epa_data[epa_data["sensor_id"].isin(test_sids)].copy()

                X_epa_train, y_epa_train, _, _ = self.prepare_features(
                    epa_train, feature_cols=feat_cols, train_medians=medians,
                    fit_medians=False,
                )
                X_epa_test, y_epa_test, _, _ = self.prepare_features(
                    epa_test, feature_cols=feat_cols, train_medians=medians,
                    fit_medians=False,
                )

                # Compute residuals for training
                backbone_pred_train = backbone.predict(X_epa_train)
                y_residual = y_epa_train - backbone_pred_train

                X_epa_train_aug = X_epa_train.copy()
                X_epa_train_aug["backbone_pred"] = backbone_pred_train

                # Station-holdout early stopping for residual
                fold_rng = np.random.RandomState(seed + fold_idx)
                fold_unique_sids = np.array(sorted(epa_train["sensor_id"].unique()))
                n_val_sids = max(1, int(0.2 * len(fold_unique_sids)))
                fold_val_sids = set(fold_rng.choice(
                    fold_unique_sids, size=n_val_sids, replace=False,
                ))
                fold_val_mask = epa_train["sensor_id"].isin(fold_val_sids).values
                fold_train_mask = ~fold_val_mask

                residual_model = self.train_lgbm_model(
                    X_epa_train_aug[fold_train_mask], y_residual[fold_train_mask],
                    X_epa_train_aug[fold_val_mask], y_residual[fold_val_mask],
                    override_params=override,
                )

                # Store predictions on test fold
                backbone_pred_test = backbone.predict(X_epa_test)
                X_epa_test_aug = X_epa_test.copy()
                X_epa_test_aug["backbone_pred"] = backbone_pred_test
                residual_pred_test = residual_model.predict(X_epa_test_aug)

                all_backbone_preds.append(backbone_pred_test)
                all_residual_preds.append(residual_pred_test)
                all_y_true.append(y_epa_test)

            # Concatenate across folds
            all_backbone_preds = np.concatenate(all_backbone_preds)
            all_residual_preds = np.concatenate(all_residual_preds)
            all_y_true = np.concatenate(all_y_true)

            # Evaluate all lambda values (no re-training)
            best_trial_lambda = 1.0
            best_trial_mae = 1e9
            lambda_results = {}
            for lam in lambda_grid:
                combined_pred = all_backbone_preds + lam * all_residual_preds
                mae = float(np.mean(np.abs(all_y_true - combined_pred)))
                lambda_results[str(lam)] = mae
                if mae < best_trial_mae:
                    best_trial_mae = mae
                    best_trial_lambda = lam

            all_trials.append({
                "trial": trial_idx,
                "residual_params": override,
                "best_lambda": best_trial_lambda,
                "best_mae": best_trial_mae,
                "lambda_results": lambda_results,
            })

            print(
                f"  [residual trial {trial_idx:02d}] "
                f"best_lambda={best_trial_lambda:.1f} | mae={best_trial_mae:.3f}"
            )

            if best_trial_mae < best_mae:
                best_mae = best_trial_mae
                best_residual_params = override.copy()
                best_lambda = best_trial_lambda

        print("\n" + "-" * 70)
        print("BEST RESIDUAL PARAMS (spatial CV on EPA):")
        print(best_residual_params)
        print(f"Best lambda:   {best_lambda:.1f}")
        print(f"Best mean MAE: {best_mae:.3f} ug/m3")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_path = self.results_dir / f"residual_tuning_{timestamp}.json"
        with open(tuning_path, "w") as f:
            json.dump(self.to_jsonable({
                "stage": "residual",
                "timestamp": timestamp,
                "n_trials": n_trials,
                "k_folds": actual_k,
                "lambda_grid": lambda_grid,
                "best_residual_params": best_residual_params,
                "best_lambda": best_lambda,
                "best_mae": best_mae,
                "all_trials": all_trials,
            }), f, indent=2)

        print(f"Residual tuning report saved to: {tuning_path}")
        return best_residual_params, best_lambda, tuning_path

    # ------------------------------------------------------------------
    # Two-stage training (leakage-free)
    # ------------------------------------------------------------------
    def _train_two_stage(
        self,
        lcs_data: pd.DataFrame,
        epa_train: pd.DataFrame,
        feat_cols: List[str],
        medians: pd.Series,
        backbone_params: Optional[dict] = None,
        residual_params: Optional[dict] = None,
        val_split_mode: str = "random",
        seed: int = 42,
    ) -> Tuple:
        """
        Train backbone on LCS, then residual model on EPA training data.
        Delegates to anchor_approaches.two_stage.train_two_stage().
        """
        return train_two_stage(
            lcs_data=lcs_data,
            epa_data=epa_train,
            feat_cols=feat_cols,
            medians=medians,
            prepare_features_fn=self.prepare_features,
            train_lgbm_fn=self.train_lgbm_model,
            backbone_params=backbone_params,
            residual_params=residual_params,
            seed=seed,
            verbose=True,
        )

    def _predict_two_stage(
        self, backbone, residual_model, X: pd.DataFrame,
        lambda_: Optional[float] = None,
    ) -> np.ndarray:
        """Combined prediction. Delegates to anchor_approaches.two_stage."""
        lam = lambda_ if lambda_ is not None else self.lambda_
        return predict_two_stage(backbone, residual_model, X, lambda_=lam)

    # ------------------------------------------------------------------
    # Daily calibration (delegates to anchor_approaches.daily_calibration)
    # ------------------------------------------------------------------
    def _fit_daily_affine(
        self,
        epa_data: pd.DataFrame,
        backbone_preds: np.ndarray,
        y_true: np.ndarray,
        min_stations_per_day: int = 5,
        huber_epsilon: float = 1.35,
    ) -> Dict:
        """Fit per-day affine calibration. Delegates to anchor_approaches."""
        return fit_daily_affine(
            epa_data, backbone_preds, y_true,
            min_stations_per_day=min_stations_per_day,
            huber_epsilon=huber_epsilon,
        )

    def _predict_daily_affine(
        self,
        test_data: pd.DataFrame,
        backbone_preds: np.ndarray,
        affine_params: Dict,
    ) -> np.ndarray:
        """Apply per-day affine correction. Delegates to anchor_approaches."""
        return predict_daily_affine(test_data, backbone_preds, affine_params)

    def _fit_daily_offset(
        self,
        epa_data: pd.DataFrame,
        backbone_preds: np.ndarray,
        y_true: np.ndarray,
        min_stations_per_day: int = 3,
    ) -> Dict:
        """Fit per-day offset. Delegates to anchor_approaches."""
        return fit_daily_offset(
            epa_data, backbone_preds, y_true,
            min_stations_per_day=min_stations_per_day,
        )

    def _predict_daily_offset(
        self,
        test_data: pd.DataFrame,
        backbone_preds: np.ndarray,
        offset_params: Dict,
    ) -> np.ndarray:
        """Apply per-day offset. Delegates to anchor_approaches."""
        return predict_daily_offset(test_data, backbone_preds, offset_params)

    # ------------------------------------------------------------------
    # Baseline models for comparison
    # ------------------------------------------------------------------
    def _train_and_predict_baselines(
        self,
        lcs_data: pd.DataFrame,
        epa_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        feat_cols: List[str],
        medians: pd.Series,
        backbone,
        backbone_params: Optional[dict] = None,
        seed: int = 42,
        w_epa: float = 10.0,
    ) -> Dict[str, Dict]:
        """Train baseline models and evaluate. Delegates to anchor_approaches."""
        baselines = {}

        # 1. Backbone-only
        backbone_pred = backbone.predict(X_test)
        baselines["backbone_only"] = self.evaluate_predictions(
            y_test, backbone_pred, verbose=False
        )

        # 2. Global model (LCS + EPA)
        global_model = train_global_model(
            lcs_data, epa_train, feat_cols, medians,
            self.prepare_features, self.train_lgbm_model,
            backbone_params=backbone_params, seed=seed,
        )
        global_pred = global_model.predict(X_test)
        baselines["global"] = self.evaluate_predictions(
            y_test, global_pred, verbose=False
        )

        # 3. Global weighted (EPA upweighted)
        global_w_model = train_global_weighted_model(
            lcs_data, epa_train, feat_cols, medians,
            self.prepare_features, self.train_lgbm_model,
            backbone_params=backbone_params, w_epa=w_epa, seed=seed,
        )
        global_w_pred = global_w_model.predict(X_test)
        baselines["global_weighted"] = self.evaluate_predictions(
            y_test, global_w_pred, verbose=False
        )

        # 4. Anchor-only (EPA only)
        if len(epa_train) >= 10:
            anchor_model = train_anchor_only_model(
                epa_train, feat_cols, medians,
                self.prepare_features, self.train_lgbm_model,
                backbone_params=backbone_params, seed=seed,
            )
            anchor_pred = anchor_model.predict(X_test)
            baselines["anchor_only"] = self.evaluate_predictions(
                y_test, anchor_pred, verbose=False
            )
        else:
            baselines["anchor_only"] = {"error": "Too few EPA training samples"}

        # Stash for hybrid baselines
        baselines["_internal"] = {
            "global_model": global_model,
            "global_pred_test": global_pred,
        }

        return baselines

    # ------------------------------------------------------------------
    # Spatial CV (leave-k-stations-out)
    # ------------------------------------------------------------------
    def run_spatial_cv(
        self,
        k: int = 5,
        seed: int = 42,
        backbone_params: Optional[dict] = None,
        residual_params: Optional[dict] = None,
        lambda_: float = 1.0,
        w_epa: float = 10.0,
    ) -> Dict:
        """
        Spatial cross-validation: group EPA stations by sensor_id and
        perform K-fold splits on station groups.
        """
        print("\n" + "=" * 80)
        print(f"SPATIAL CV (leave-k-stations-out) | k={k} folds")
        print("=" * 80)

        full_data = self.load_and_prepare_data()
        full_data = self._tag_provider(full_data)
        lcs_data, epa_data = self._split_lcs_epa(full_data)

        # Prepare features on LCS data (fit medians on LCS)
        X_lcs_tmp, _, feat_cols, medians = self.prepare_features(
            lcs_data, fit_medians=True
        )

        # Group EPA stations
        epa_sensor_ids = np.array(sorted(epa_data["sensor_id"].unique()))
        n_stations = len(epa_sensor_ids)
        actual_k = min(k, n_stations)
        print(f"  EPA stations for spatial CV: {n_stations}, using {actual_k} folds")

        kf = KFold(n_splits=actual_k, shuffle=True, random_state=seed)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            kf.split(epa_sensor_ids), start=1
        ):
            train_sids = set(epa_sensor_ids[train_idx])
            test_sids = set(epa_sensor_ids[test_idx])

            epa_train = epa_data[epa_data["sensor_id"].isin(train_sids)].copy()
            epa_test = epa_data[epa_data["sensor_id"].isin(test_sids)].copy()

            X_test, y_test, _, _ = self.prepare_features(
                epa_test, feature_cols=feat_cols, train_medians=medians, fit_medians=False
            )

            print(f"\n  Fold {fold_idx}/{actual_k}: "
                  f"train_stations={len(train_sids)}, test_stations={len(test_sids)}, "
                  f"train_rows={len(epa_train)}, test_rows={len(epa_test)}")

            # Train two-stage (val split carved from training data, not test fold)
            backbone, residual_model = self._train_two_stage(
                lcs_data, epa_train, feat_cols, medians,
                backbone_params=backbone_params, residual_params=residual_params,
                seed=seed + fold_idx,
            )

            # Evaluate anchored model
            anchored_pred = self._predict_two_stage(
                backbone, residual_model, X_test, lambda_=lambda_,
            )
            anchored_metrics = self.evaluate_predictions(y_test, anchored_pred, verbose=False)

            # Evaluate baselines
            baselines = self._train_and_predict_baselines(
                lcs_data, epa_train, X_test, y_test, feat_cols, medians, backbone,
                backbone_params=backbone_params, w_epa=w_epa,
            )

            # Daily affine anchor baseline
            X_epa_train_da, y_epa_train_da, _, _ = self.prepare_features(
                epa_train, feature_cols=feat_cols, train_medians=medians,
                fit_medians=False,
            )
            backbone_pred_train = backbone.predict(X_epa_train_da)
            affine_params = self._fit_daily_affine(
                epa_train, backbone_pred_train, y_epa_train_da,
            )
            backbone_pred_test = backbone.predict(X_test)
            da_pred = self._predict_daily_affine(epa_test, backbone_pred_test, affine_params)
            baselines["daily_affine_anchor"] = self.evaluate_predictions(
                y_test, da_pred, verbose=False,
            )

            # Daily offset baseline (backbone + offset)
            offset_params = self._fit_daily_offset(
                epa_train, backbone_pred_train, y_epa_train_da,
            )
            do_pred = self._predict_daily_offset(epa_test, backbone_pred_test, offset_params)
            baselines["daily_offset"] = self.evaluate_predictions(
                y_test, do_pred, verbose=False,
            )

            # --- Hybrid baselines: base model + daily offset ---
            internal = baselines.pop("_internal", {})
            global_model = internal.get("global_model")
            global_pred_test = internal.get("global_pred_test")

            # Residual + daily offset: backbone + lambda*residual + b_t
            anchored_pred_train = self._predict_two_stage(
                backbone, residual_model, X_epa_train_da, lambda_=lambda_,
            )
            offset_res = self._fit_daily_offset(
                epa_train, anchored_pred_train, y_epa_train_da,
            )
            res_offset_pred = self._predict_daily_offset(
                epa_test, anchored_pred, offset_res,
            )
            baselines["residual_plus_offset"] = self.evaluate_predictions(
                y_test, res_offset_pred, verbose=False,
            )

            # Global + daily offset: global(x) + b_t
            if global_model is not None and global_pred_test is not None:
                global_pred_train = global_model.predict(X_epa_train_da)
                offset_glob = self._fit_daily_offset(
                    epa_train, global_pred_train, y_epa_train_da,
                )
                glob_offset_pred = self._predict_daily_offset(
                    epa_test, global_pred_test, offset_glob,
                )
                baselines["global_plus_offset"] = self.evaluate_predictions(
                    y_test, glob_offset_pred, verbose=False,
                )

            fold_results.append({
                "fold": fold_idx,
                "train_stations": len(train_sids),
                "test_stations": len(test_sids),
                "train_rows": len(epa_train),
                "test_rows": len(epa_test),
                "anchored": anchored_metrics,
                "baselines": baselines,
            })

            # Print comparison
            print(f"    Anchored R2={anchored_metrics.get('r2', float('nan')):.4f}, "
                  f"MAE={anchored_metrics.get('mae', float('nan')):.2f}")
            for name, bm in baselines.items():
                if isinstance(bm, dict) and "r2" in bm:
                    print(f"    {name:25s} R2={bm['r2']:.4f}, MAE={bm['mae']:.2f}")

        # Aggregate
        anchored_list = [f["anchored"] for f in fold_results if f["anchored"]]
        agg_anchored = self.aggregate_cv_metrics(anchored_list)

        agg_baselines = {}
        for bname in ["backbone_only", "global", "global_weighted", "anchor_only",
                      "daily_affine_anchor", "daily_offset",
                      "residual_plus_offset", "global_plus_offset"]:
            blist = [
                f["baselines"][bname]
                for f in fold_results
                if bname in f["baselines"] and isinstance(f["baselines"][bname], dict) and "r2" in f["baselines"][bname]
            ]
            if blist:
                agg_baselines[bname] = self.aggregate_cv_metrics(blist)

        self._print_comparison_table("Spatial CV", agg_anchored, agg_baselines)

        return {
            "cv_type": "spatial",
            "k": actual_k,
            "seed": seed,
            "folds": fold_results,
            "aggregated_anchored": agg_anchored,
            "aggregated_baselines": agg_baselines,
        }

    # ------------------------------------------------------------------
    # Temporal CV (rolling month splits)
    # ------------------------------------------------------------------
    def run_temporal_cv(
        self,
        backbone_params: Optional[dict] = None,
        residual_params: Optional[dict] = None,
        lambda_: float = 1.0,
        k: int = 5,
        seed: int = 42,
        w_epa: float = 10.0,
    ) -> Dict:
        """
        Nested spatial-temporal cross-validation.

        Outer loop: rolling month splits (temporal folds).
        Inner loop: K-fold on EPA stations (spatial folds).

        For each temporal fold the backbone is trained ONCE on all LCS data
        (cached across spatial folds). Each spatial fold partitions EPA
        stations into:
          - epa_train:  train_stations x train_months  (residual training)
          - epa_anchor: train_stations x test_month    (daily affine calibration)
          - epa_test:   test_stations  x test_month    (evaluation)

        Global baseline is omitted (not central to the daily-affine analysis).
        """
        print("\n" + "=" * 80)
        print(f"TEMPORAL CV (nested spatial-temporal) | k={k} spatial folds")
        print("=" * 80)

        full_data = self.load_and_prepare_data()
        full_data = self._tag_provider(full_data)
        lcs_data, epa_data = self._split_lcs_epa(full_data)

        # Prepare features (fit medians on LCS)
        _, _, feat_cols, medians = self.prepare_features(
            lcs_data, fit_medians=True
        )

        # Use year-month period keys to handle cross-year ordering correctly
        epa_data = epa_data.copy()
        date_col = "date_utc" if "date_utc" in epa_data.columns else "date"
        epa_data["month_key"] = pd.to_datetime(epa_data[date_col]).dt.to_period("M").astype(str)
        epa_months = sorted(epa_data["month_key"].unique())
        print(f"  EPA data months: {epa_months}")

        if len(epa_months) < 2:
            print("  WARNING: Need at least 2 months for temporal CV. Skipping.")
            return {"cv_type": "temporal", "folds": [], "error": "insufficient_months"}

        # EPA station list for spatial folds
        epa_sensor_ids = np.array(sorted(epa_data["sensor_id"].unique()))
        n_stations = len(epa_sensor_ids)
        actual_k = min(k, n_stations)
        print(f"  EPA stations: {n_stations}, spatial folds: {actual_k}")

        temporal_fold_results = []

        # Rolling temporal folds: train on months[:i], test on month[i]
        for t_idx in range(1, len(epa_months)):
            train_months = epa_months[:t_idx]
            test_month = epa_months[t_idx]

            epa_temporal_train = epa_data[epa_data["month_key"].isin(train_months)].copy()
            epa_temporal_test = epa_data[epa_data["month_key"] == test_month].copy()

            if len(epa_temporal_train) == 0 or len(epa_temporal_test) == 0:
                continue

            print(f"\n  Temporal fold {t_idx}: train_months={train_months}, "
                  f"test_month={test_month}")

            # Train backbone ONCE per temporal fold (LCS-only, independent of spatial split)
            rng = np.random.RandomState(seed + t_idx)
            X_lcs, y_lcs, _, _ = self.prepare_features(
                lcs_data, feature_cols=feat_cols, train_medians=medians,
                fit_medians=False,
            )
            n_lcs = len(X_lcs)
            lcs_val_idx = rng.choice(n_lcs, size=max(1, n_lcs // 5), replace=False)
            lcs_train_mask = np.ones(n_lcs, dtype=bool)
            lcs_train_mask[lcs_val_idx] = False
            backbone = self.train_lgbm_model(
                X_lcs[lcs_train_mask], y_lcs[lcs_train_mask],
                X_lcs[~lcs_train_mask], y_lcs[~lcs_train_mask],
                override_params=backbone_params,
            )

            # Inner spatial K-fold on EPA stations
            kf = KFold(n_splits=actual_k, shuffle=True, random_state=seed)
            spatial_fold_metrics = []

            for s_idx, (s_train_idx, s_test_idx) in enumerate(
                kf.split(epa_sensor_ids), start=1
            ):
                train_sids = set(epa_sensor_ids[s_train_idx])
                test_sids = set(epa_sensor_ids[s_test_idx])

                # epa_train: train_stations x train_months (for residual)
                epa_train = epa_temporal_train[
                    epa_temporal_train["sensor_id"].isin(train_sids)
                ].copy()
                # epa_anchor: train_stations x test_month (for daily affine)
                epa_anchor = epa_temporal_test[
                    epa_temporal_test["sensor_id"].isin(train_sids)
                ].copy()
                # epa_test: test_stations x test_month (evaluation)
                epa_test = epa_temporal_test[
                    epa_temporal_test["sensor_id"].isin(test_sids)
                ].copy()

                if len(epa_test) == 0:
                    continue

                X_test, y_test, _, _ = self.prepare_features(
                    epa_test, feature_cols=feat_cols, train_medians=medians,
                    fit_medians=False,
                )

                print(f"    Spatial fold {s_idx}/{actual_k}: "
                      f"train={len(epa_train)}, anchor={len(epa_anchor)}, "
                      f"test={len(epa_test)}")

                fold_baselines: Dict[str, Dict] = {}

                # 1. Backbone+Residual
                if len(epa_train) > 0:
                    X_epa_tr, y_epa_tr, _, _ = self.prepare_features(
                        epa_train, feature_cols=feat_cols, train_medians=medians,
                        fit_medians=False,
                    )
                    backbone_pred_tr = backbone.predict(X_epa_tr)
                    y_residual = y_epa_tr - backbone_pred_tr

                    X_epa_tr_aug = X_epa_tr.copy()
                    X_epa_tr_aug["backbone_pred"] = backbone_pred_tr

                    # Station-holdout early stopping for residual
                    fold_rng = np.random.RandomState(seed + t_idx * 100 + s_idx)
                    res_unique_sids = np.array(sorted(epa_train["sensor_id"].unique()))
                    n_res_val = max(1, int(0.2 * len(res_unique_sids)))
                    res_val_sids = set(fold_rng.choice(
                        res_unique_sids, size=n_res_val, replace=False,
                    ))
                    res_val_mask = epa_train["sensor_id"].isin(res_val_sids).values
                    res_train_mask = ~res_val_mask

                    residual_model = self.train_lgbm_model(
                        X_epa_tr_aug[res_train_mask], y_residual[res_train_mask],
                        X_epa_tr_aug[res_val_mask], y_residual[res_val_mask],
                        override_params=residual_params,
                    )
                    anchored_pred = self._predict_two_stage(
                        backbone, residual_model, X_test, lambda_=lambda_,
                    )
                    fold_baselines["backbone_residual"] = self.evaluate_predictions(
                        y_test, anchored_pred, verbose=False,
                    )

                # 2. Backbone-only
                backbone_pred_test = backbone.predict(X_test)
                fold_baselines["backbone_only"] = self.evaluate_predictions(
                    y_test, backbone_pred_test, verbose=False,
                )

                # 3. Daily affine anchor
                if len(epa_anchor) > 0:
                    X_anchor, y_anchor, _, _ = self.prepare_features(
                        epa_anchor, feature_cols=feat_cols, train_medians=medians,
                        fit_medians=False,
                    )
                    backbone_pred_anchor = backbone.predict(X_anchor)
                    affine_params = self._fit_daily_affine(
                        epa_anchor, backbone_pred_anchor, y_anchor,
                    )
                    da_pred = self._predict_daily_affine(
                        epa_test, backbone_pred_test, affine_params,
                    )
                    fold_baselines["daily_affine_anchor"] = self.evaluate_predictions(
                        y_test, da_pred, verbose=False,
                    )

                    # 3b. Daily offset
                    offset_params = self._fit_daily_offset(
                        epa_anchor, backbone_pred_anchor, y_anchor,
                    )
                    do_pred = self._predict_daily_offset(
                        epa_test, backbone_pred_test, offset_params,
                    )
                    fold_baselines["daily_offset"] = self.evaluate_predictions(
                        y_test, do_pred, verbose=False,
                    )

                # 4. Anchor-only (trained on EPA train stations x train months)
                if len(epa_train) >= 10:
                    X_epa_anch, y_epa_anch, _, _ = self.prepare_features(
                        epa_train, feature_cols=feat_cols, train_medians=medians,
                        fit_medians=False,
                    )
                    n_anch = len(X_epa_anch)
                    anch_rng = np.random.RandomState(seed + t_idx * 100 + s_idx + 50)
                    anch_val_idx = anch_rng.choice(
                        n_anch, size=max(1, n_anch // 5), replace=False,
                    )
                    anch_train_mask = np.ones(n_anch, dtype=bool)
                    anch_train_mask[anch_val_idx] = False
                    anchor_model = self.train_lgbm_model(
                        X_epa_anch[anch_train_mask], y_epa_anch[anch_train_mask],
                        X_epa_anch[~anch_train_mask], y_epa_anch[~anch_train_mask],
                        override_params=backbone_params,
                    )
                    anchor_pred = anchor_model.predict(X_test)
                    fold_baselines["anchor_only"] = self.evaluate_predictions(
                        y_test, anchor_pred, verbose=False,
                    )

                # 5. Global (LCS + EPA train stations x train months)
                if len(epa_train) > 0:
                    epa_train_clean = epa_train.drop(
                        columns=["month_key"], errors="ignore",
                    )
                    all_train = pd.concat([lcs_data, epa_train_clean], ignore_index=True)
                    X_all, y_all, _, _ = self.prepare_features(
                        all_train, feature_cols=feat_cols, train_medians=medians,
                        fit_medians=False,
                    )
                    n_all = len(X_all)
                    glob_rng = np.random.RandomState(seed + t_idx * 100 + s_idx + 70)
                    glob_val_idx = glob_rng.choice(
                        n_all, size=max(1, n_all // 5), replace=False,
                    )
                    glob_train_mask = np.ones(n_all, dtype=bool)
                    glob_train_mask[glob_val_idx] = False
                    global_model = self.train_lgbm_model(
                        X_all[glob_train_mask], y_all[glob_train_mask],
                        X_all[~glob_train_mask], y_all[~glob_train_mask],
                        override_params=backbone_params,
                    )
                    global_pred = global_model.predict(X_test)
                    fold_baselines["global"] = self.evaluate_predictions(
                        y_test, global_pred, verbose=False,
                    )

                    # 6. Global weighted (EPA upweighted)
                    is_epa_col = all_train["is_epa"].values if "is_epa" in all_train.columns else np.zeros(len(all_train), dtype=bool)
                    weights = np.where(is_epa_col, w_epa, 1.0).astype(float)
                    global_w_model = self.train_lgbm_model(
                        X_all[glob_train_mask], y_all[glob_train_mask],
                        X_all[~glob_train_mask], y_all[~glob_train_mask],
                        override_params=backbone_params,
                        sample_weight=weights[glob_train_mask],
                        eval_sample_weight=weights[~glob_train_mask],
                    )
                    global_w_pred = global_w_model.predict(X_test)
                    fold_baselines["global_weighted"] = self.evaluate_predictions(
                        y_test, global_w_pred, verbose=False,
                    )

                # --- Hybrid baselines: base model + daily offset ---
                # Residual + daily offset
                if (len(epa_anchor) > 0 and len(epa_train) > 0
                        and "backbone_residual" in fold_baselines):
                    anchored_pred_anchor = self._predict_two_stage(
                        backbone, residual_model, X_anchor, lambda_=lambda_,
                    )
                    offset_res = self._fit_daily_offset(
                        epa_anchor, anchored_pred_anchor, y_anchor,
                    )
                    res_offset_pred = self._predict_daily_offset(
                        epa_test, anchored_pred, offset_res,
                    )
                    fold_baselines["residual_plus_offset"] = self.evaluate_predictions(
                        y_test, res_offset_pred, verbose=False,
                    )

                # Global + daily offset
                if len(epa_anchor) > 0 and len(epa_train) > 0 and "global" in fold_baselines:
                    global_pred_anchor = global_model.predict(X_anchor)
                    offset_glob = self._fit_daily_offset(
                        epa_anchor, global_pred_anchor, y_anchor,
                    )
                    glob_offset_pred = self._predict_daily_offset(
                        epa_test, global_pred, offset_glob,
                    )
                    fold_baselines["global_plus_offset"] = self.evaluate_predictions(
                        y_test, glob_offset_pred, verbose=False,
                    )

                spatial_fold_metrics.append(fold_baselines)

                # Print spatial fold summary
                for name, bm in fold_baselines.items():
                    if isinstance(bm, dict) and "r2" in bm:
                        print(f"      {name:25s} R2={bm['r2']:.4f}, "
                              f"MAE={bm['mae']:.2f}, bias={bm.get('bias', float('nan')):.2f}")

            # Average across spatial folds for this temporal fold
            if not spatial_fold_metrics:
                continue

            temporal_fold_agg: Dict[str, Dict] = {}
            baseline_names = ["backbone_residual", "backbone_only",
                              "daily_affine_anchor", "daily_offset",
                              "anchor_only", "global", "global_weighted",
                              "residual_plus_offset", "global_plus_offset"]
            for bname in baseline_names:
                blist = [
                    sf[bname] for sf in spatial_fold_metrics
                    if bname in sf and isinstance(sf[bname], dict) and "r2" in sf[bname]
                ]
                if blist:
                    temporal_fold_agg[bname] = self.aggregate_cv_metrics(blist)

            temporal_fold_results.append({
                "temporal_fold": t_idx,
                "train_months": train_months,
                "test_month": test_month,
                "spatial_folds": spatial_fold_metrics,
                "aggregated": temporal_fold_agg,
            })

            # Print temporal fold summary
            print(f"    -- Temporal fold {t_idx} averages (across {len(spatial_fold_metrics)} spatial folds) --")
            for bname, bagg in temporal_fold_agg.items():
                r2_m = bagg.get("r2", {}).get("mean", float("nan"))
                mae_m = bagg.get("mae", {}).get("mean", float("nan"))
                bias_m = bagg.get("bias", {}).get("mean", float("nan"))
                print(f"      {bname:25s} R2={r2_m:.4f}, MAE={mae_m:.2f}, bias={bias_m:.2f}")

        # Final aggregation across temporal folds
        final_agg: Dict[str, Dict] = {}
        baseline_names = ["backbone_residual", "backbone_only",
                          "daily_affine_anchor", "daily_offset",
                          "anchor_only", "global", "global_weighted",
                          "residual_plus_offset", "global_plus_offset"]
        for bname in baseline_names:
            # Collect the per-temporal-fold mean values, then average them
            blist = [
                tf["aggregated"][bname]
                for tf in temporal_fold_results
                if bname in tf["aggregated"]
            ]
            if blist:
                # Each element is an aggregated dict with {metric: {mean, std, ...}}
                # We want the mean of means across temporal folds
                combined = {}
                for metric_name in blist[0]:
                    means = [b[metric_name]["mean"] for b in blist if "mean" in b[metric_name]]
                    if means:
                        combined[metric_name] = {
                            "mean": float(np.nanmean(means)),
                            "std": float(np.nanstd(means)),
                        }
                final_agg[bname] = combined

        # Use backbone_residual as the "anchored" model for comparison table
        agg_anchored = final_agg.pop("backbone_residual", {})
        # Rename daily_affine_anchor for display consistency
        agg_baselines = final_agg

        self._print_comparison_table("Temporal CV", agg_anchored, agg_baselines)

        return {
            "cv_type": "temporal",
            "k_spatial": actual_k,
            "seed": seed,
            "temporal_folds": temporal_fold_results,
            "aggregated_anchored": agg_anchored,
            "aggregated_baselines": agg_baselines,
        }

    # ------------------------------------------------------------------
    # Comparison table printer
    # ------------------------------------------------------------------
    def _print_comparison_table(
        self,
        cv_name: str,
        agg_anchored: Dict,
        agg_baselines: Dict[str, Dict],
    ):
        """Print a formatted comparison table."""
        print(f"\n{'=' * 80}")
        print(f"COMPARISON TABLE: {cv_name}")
        print(f"{'=' * 80}")

        header = f"{'Model':20s} {'R2':>12s} {'MAE':>12s} {'RMSE':>12s} {'Bias':>12s} {'F1@150':>12s}"
        print(header)
        print("-" * 80)

        def _fmt_row(name: str, agg: Dict):
            r2 = agg.get("r2", {})
            mae = agg.get("mae", {})
            rmse = agg.get("rmse", {})
            bias = agg.get("bias", {})
            f1 = agg.get("f1_150", {})
            print(
                f"{name:20s} "
                f"{r2.get('mean', float('nan')):7.4f}+/-{r2.get('std', 0):.4f} "
                f"{mae.get('mean', float('nan')):7.2f}+/-{mae.get('std', 0):.2f} "
                f"{rmse.get('mean', float('nan')):7.2f}+/-{rmse.get('std', 0):.2f} "
                f"{bias.get('mean', float('nan')):7.2f}+/-{bias.get('std', 0):.2f} "
                f"{f1.get('mean', float('nan')):7.3f}+/-{f1.get('std', 0):.3f}"
            )

        _fmt_row("Backbone+Residual", agg_anchored)
        display_names = {
            "daily_affine_anchor": "daily_affine",
            "residual_plus_offset": "residual+offset",
            "global_plus_offset": "global+offset",
        }
        for bname, bagg in agg_baselines.items():
            _fmt_row(display_names.get(bname, bname), bagg)

        print("=" * 80)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_anchored_training(
        self,
        do_spatial_cv: bool = True,
        do_temporal_cv: bool = True,
        do_tuning: bool = False,
        tune_backbone: bool = True,
        tune_residual: bool = True,
        backbone_trials: int = 25,
        residual_trials: int = 25,
        temporal_k: int = 5,
        w_epa: float = 10.0,
    ) -> Dict:
        """
        Main entry point for anchored training.

        1. Optionally tunes backbone and/or residual hyperparameters
        2. Optionally runs spatial and/or temporal CV
        3. Trains final two-stage model on all data
        4. Saves results and model payloads
        """
        print("\n" + "=" * 80)
        print("ANCHORED RESIDUAL TRAINING (Backbone + Anchor Residual)")
        print("=" * 80)

        results: Dict = {
            "model_type": "anchored_residual_lgbm",
            "training_mode": "backbone_anchor_residual",
        }

        # Two-stage hyperparameter tuning
        backbone_params = None
        residual_params = None
        best_lambda = 1.0

        if do_tuning:
            # Stage 1: Backbone tuning (rolling-year CV on LCS)
            if tune_backbone:
                backbone_params, bb_tuning_path = self.tune_backbone_params(
                    n_trials=backbone_trials,
                )
                results["backbone_tuning"] = {
                    "best_params": backbone_params,
                    "tuning_path": str(bb_tuning_path) if bb_tuning_path else None,
                }

            # Stage 2: Residual tuning (spatial CV on EPA)
            if tune_residual:
                residual_params, best_lambda, res_tuning_path = self.tune_residual_params(
                    backbone_params=backbone_params,
                    n_trials=residual_trials,
                )
                results["residual_tuning"] = {
                    "best_params": residual_params,
                    "best_lambda": best_lambda,
                    "tuning_path": str(res_tuning_path) if res_tuning_path else None,
                }

        # Cross-validation
        if do_spatial_cv:
            spatial_results = self.run_spatial_cv(
                backbone_params=backbone_params,
                residual_params=residual_params,
                lambda_=best_lambda,
                w_epa=w_epa,
            )
            results["spatial_cv"] = spatial_results

        if do_temporal_cv:
            temporal_results = self.run_temporal_cv(
                backbone_params=backbone_params,
                residual_params=residual_params,
                lambda_=best_lambda,
                k=temporal_k,
                w_epa=w_epa,
            )
            results["temporal_cv"] = temporal_results

        # Final model: train on all available data
        print("\n" + "=" * 80)
        print("FINAL TWO-STAGE MODEL (all data)")
        print("=" * 80)

        full_data = self.load_and_prepare_data()
        full_data = self._tag_provider(full_data)
        lcs_data, epa_data = self._split_lcs_epa(full_data)

        # Fit medians on LCS
        X_lcs_tmp, _, feat_cols, medians = self.prepare_features(
            lcs_data, fit_medians=True
        )

        # Train final two-stage (val carved from training data)
        backbone, residual_model = self._train_two_stage(
            lcs_data, epa_data, feat_cols, medians,
            backbone_params=backbone_params, residual_params=residual_params,
        )

        # Evaluate on EPA (in-sample — sanity check only, not held-out)
        X_epa, y_epa, _, _ = self.prepare_features(
            epa_data, feature_cols=feat_cols, train_medians=medians, fit_medians=False
        )
        final_pred = self._predict_two_stage(
            backbone, residual_model, X_epa, lambda_=best_lambda,
        )
        print("\nFinal model EPA in-sample metrics (SANITY CHECK - not held-out):")
        final_metrics = self.evaluate_predictions(y_epa, final_pred, verbose=True)
        results["final_model_insample"] = final_metrics

        # Also report backbone-only on EPA
        backbone_pred = backbone.predict(X_epa)
        print("\nBackbone-only EPA in-sample metrics (SANITY CHECK - not held-out):")
        backbone_metrics = self.evaluate_predictions(y_epa, backbone_pred, verbose=True)
        results["backbone_only_insample"] = backbone_metrics

        # Feature importance from backbone
        importance_list = sorted(
            zip(feat_cols, backbone.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        results["feature_importance_backbone_top30"] = importance_list[:30]

        # Save artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results["timestamp"] = timestamp
        results["lambda"] = best_lambda
        results["n_features"] = len(feat_cols)
        results["n_lcs_rows"] = len(lcs_data)
        results["n_epa_rows"] = len(epa_data)
        results["n_lcs_stations"] = int(lcs_data["sensor_id"].nunique())
        results["n_epa_stations"] = int(epa_data["sensor_id"].nunique())

        results_path = self.results_dir / f"anchored_results_{timestamp}.json"
        backbone_path = self.results_dir / f"anchored_backbone_{timestamp}.joblib"
        residual_path = self.results_dir / f"anchored_residual_{timestamp}.joblib"

        backbone_payload = {
            "model": backbone,
            "feature_cols": feat_cols,
            "train_medians": medians.to_dict(),
            "use_latlon": self.use_latlon,
            "lgbm_params": self.LGBM_PARAMS,
            "backbone_params": backbone_params,
            "training_mode": "anchored_backbone",
        }
        joblib.dump(backbone_payload, backbone_path)

        residual_payload = {
            "model": residual_model,
            "feature_cols": feat_cols,
            "train_medians": medians.to_dict(),
            "use_latlon": self.use_latlon,
            "lgbm_params": self.LGBM_PARAMS,
            "residual_params": residual_params,
            "lambda": best_lambda,
            "training_mode": "anchored_residual",
        }
        joblib.dump(residual_payload, residual_path)

        with open(results_path, "w") as f:
            json.dump(self.to_jsonable(results), f, indent=2)

        print(f"\nResults saved to: {results_path}")
        print(f"Backbone model saved to: {backbone_path}")
        print(f"Residual model saved to: {residual_path}")
        print("=" * 80)

        return {
            "backbone": backbone,
            "residual_model": residual_model,
            "results_path": results_path,
            "backbone_path": backbone_path,
            "residual_path": residual_path,
            "results": results,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Anchored residual training (Backbone + Anchor Residual)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two-stage PM2.5 estimation:
  Stage 1: Backbone LightGBM on all LCS data
  Stage 2: Residual correction LightGBM on EPA anchor stations
  Final:   backbone(x) + lambda * residual(x)

Examples:
  # Full training with both CV modes (no tuning)
  python -m training.ablation.regulation_alignment --do_spatial_cv --do_temporal_cv --no_tuning

  # Spatial CV only
  python -m training.ablation.regulation_alignment --do_spatial_cv --no_tuning

  # With full two-stage hyperparameter tuning
  python -m training.ablation.regulation_alignment --do_spatial_cv --do_tuning \\
      --backbone_trials 25 --residual_trials 25

  # Residual tuning only (skip backbone tuning)
  python -m training.ablation.regulation_alignment --do_spatial_cv --do_tuning \\
      --no_tune_backbone --residual_trials 25
        """,
    )
    parser.add_argument(
        "--lake_path",
        type=str,
        default="../feature_engineering/output/master_lake",
        help="Path to master_lake directory",
    )
    parser.add_argument(
        "--station_lake_path",
        type=str,
        default="../extraction_and_preprocessing/station_labels/lake/station_daily",
        help="Path to station_daily lake (for provider_name lookup)",
    )
    parser.add_argument(
        "--registry_csv",
        type=str,
        default="data/features_registry.csv",
        help="Path to features registry CSV",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/anchored",
        help="Directory to save results",
    )
    parser.add_argument(
        "--use_latlon",
        action="store_true",
        help="Include lat/lon as features",
    )
    parser.add_argument(
        "--do_spatial_cv",
        action="store_true",
        help="Run spatial (leave-k-stations-out) cross-validation",
    )
    parser.add_argument(
        "--do_temporal_cv",
        action="store_true",
        help="Run temporal (rolling month) cross-validation",
    )
    parser.add_argument(
        "--no_tuning",
        action="store_true",
        help="Skip hyperparameter tuning",
    )
    parser.add_argument(
        "--do_tuning",
        action="store_true",
        help="Enable two-stage hyperparameter tuning",
    )
    parser.add_argument(
        "--no_tune_backbone",
        action="store_true",
        help="Skip backbone tuning (use default params for backbone)",
    )
    parser.add_argument(
        "--no_tune_residual",
        action="store_true",
        help="Skip residual tuning (use default params and lambda=1.0)",
    )
    parser.add_argument(
        "--backbone_trials",
        type=int,
        default=25,
        help="Number of backbone hyperparameter tuning trials",
    )
    parser.add_argument(
        "--residual_trials",
        type=int,
        default=25,
        help="Number of residual hyperparameter tuning trials",
    )
    parser.add_argument(
        "--temporal_k",
        type=int,
        default=5,
        help="Spatial folds within temporal CV (default: 5)",
    )
    parser.add_argument(
        "--w_epa",
        type=float,
        default=10.0,
        help="EPA sample weight for global_weighted baseline (default: 10)",
    )

    args = parser.parse_args()

    do_tuning = False
    if args.do_tuning:
        do_tuning = True
    elif args.no_tuning:
        do_tuning = False

    trainer = AnchoredTrainer(
        station_lake_path=args.station_lake_path,
        master_lake_path=args.lake_path,
        registry_csv=args.registry_csv,
        results_dir=args.results_dir,
        use_latlon=args.use_latlon,
        strict_registry=True,
    )

    trainer.run_anchored_training(
        do_spatial_cv=args.do_spatial_cv,
        do_temporal_cv=args.do_temporal_cv,
        do_tuning=do_tuning,
        tune_backbone=not args.no_tune_backbone,
        tune_residual=not args.no_tune_residual,
        backbone_trials=args.backbone_trials,
        residual_trials=args.residual_trials,
        temporal_k=args.temporal_k,
        w_epa=args.w_epa,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
