"""
Benchmark Training Module (Paper Training)
==========================================

Paper/benchmark training with hyperparameter tuning, cross-validation,
skill scores, and fixed train/dev/test splits.

This module extends the production Trainer with research-oriented features.
For operational (deployment) training, use training.utils.trainer.

Usage:
    # Module-style invocation
    python -m training.main --do_tuning --tuning_trials 25

    # Paper training without tuning
    python -m training.main --no_tuning

    # Specific years
    python -m training.main --dev_year 2024 --test_year 2025

    # Import in code
    from training.main import BenchmarkTrainer
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings("ignore")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the base Trainer from utils module
from training.utils.trainer import Trainer, load_master_from_lake
from training.utils.dataset_prep import DERIVED_FEATURES


class BenchmarkTrainer(Trainer):
    """
    Extended Trainer with benchmark/paper training capabilities.

    Adds:
    - Hyperparameter tuning via rolling CV
    - Year-wise rolling cross-validation
    - Fixed train/dev/test split training (paper mode)
    """

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

    # ------------------------------------------------------------------
    # Hyperparameter tuning via rolling CV
    # ------------------------------------------------------------------
    def tune_hyperparameters_rolling_cv(
        self,
        n_trials: int = 25,
        cv_val_years: Tuple[int, ...] = (2022, 2023, 2024),
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

                train_data = self.get_year_data(train_years)
                val_data = self.get_year_data([val_year])

                X_train, y_train, feat_cols, med = self.prepare_features(
                    train_data, fit_medians=True, min_non_nan_frac=min_non_nan_frac
                )
                X_val, y_val, _, _ = self.prepare_features(
                    val_data, feature_cols=feat_cols, train_medians=med, fit_medians=False
                )

                model = self.train_lgbm_model(X_train, y_train, X_val, y_val, override_params=override)
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

            print(f"[trial {trial_idx:02d}] mean_r2={mean_r2:.4f} | mean_mae={mean_mae:.3f}")

            is_better = (mean_r2 > best_mean_r2) or (np.isclose(mean_r2, best_mean_r2) and mean_mae < best_mean_mae)
            if is_better:
                best_mean_r2 = mean_r2
                best_mean_mae = mean_mae
                best_params = override.copy()

        print("\n" + "-" * 70)
        print("BEST TUNED PARAMS (rolling-year CV):")
        print(best_params)
        print(f"Best mean R2:  {best_mean_r2:.4f}")
        print(f"Best mean MAE: {best_mean_mae:.3f} ug/m3")

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
                "all_trials": all_trials,
            }), f, indent=2)

        print(f"Tuning report saved to: {tuning_path}")
        return best_params, tuning_path

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

            model = self.train_lgbm_model(X_train, y_train, X_val, y_val)
            val_pred = model.predict(X_val)
            val_metrics = self.evaluate_predictions(y_val, val_pred, verbose=False)

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
    # Paper/benchmark training (fixed train/dev/test split)
    # ------------------------------------------------------------------
    def run_paper_training(
        self,
        final_train_years=(2020, 2021, 2022, 2023),
        dev_year=2024,
        test_year=2025,
        do_tuning: bool = True,
        tuning_trials: int = 25,
    ):
        """
        Train model with fixed train/dev/test split for paper/benchmark.

        This method:
        1. Runs yearly rolling CV to evaluate generalization
        2. Optionally tunes hyperparameters
        3. Trains final model on train years, evaluates on dev/test
        4. Saves comprehensive results for paper reporting
        """
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

        final_model = self.train_lgbm_model(X_train, y_train, X_dev, y_dev)

        # Feature importance
        feature_importance = final_model.feature_importances_
        importance_list = list(zip(feat_cols, feature_importance))
        importance_list.sort(key=lambda x: x[1], reverse=True)

        def _feat_type(feature: str) -> str:
            if feature in DERIVED_FEATURES:
                return "[TIME]"
            st = self._col_to_stage.get(feature, "")
            return {
                "met": "[MET]",
                "aod": "[AOD]",
                "tropomi": "[SAT]",
                "elevation": "[ELEV]",
                "coast": "[DIST]",
            }.get(st, "[UNK]")

        print("\nTop 30 features (final model):")
        for i, (feature, importance) in enumerate(importance_list[:30]):
            print(f"{i + 1:2d}. {feature:30s} {_feat_type(feature):8s}: {float(importance):.4f}")

        # Dev metrics
        print("\n" + "=" * 80)
        print(f"DEV ANALYSIS ON {dev_year}")
        print("=" * 80)

        dev_pred = final_model.predict(X_dev)
        print(f"Overall dev-year ({dev_year}) metrics:")
        dev_overall_metrics = self.evaluate_predictions(y_dev, dev_pred, verbose=True)

        print(f"\nMonthly metrics ({dev_year}):")
        dev_monthly_metrics = self.compute_monthly_metrics(dev_data, y_dev, dev_pred)

        print(f"\nSeasonal metrics ({dev_year}):")
        dev_seasonal_metrics = self.compute_seasonal_metrics(dev_data, y_dev, dev_pred)

        test_overall_metrics = test_monthly_metrics = test_seasonal_metrics = None
        if test_data is not None:
            print("\n" + "=" * 80)
            print(f"FINAL TEST ON {test_year}")
            print("=" * 80)

            test_pred = final_model.predict(X_test)
            print(f"Overall test-year ({test_year}) metrics:")
            test_overall_metrics = self.evaluate_predictions(y_test, test_pred, verbose=True)

            print(f"\nMonthly metrics ({test_year}):")
            test_monthly_metrics = self.compute_monthly_metrics(test_data, y_test, test_pred)

            print(f"\nSeasonal metrics ({test_year}):")
            test_seasonal_metrics = self.compute_seasonal_metrics(test_data, y_test, test_pred)

        # ----------------------------------------------------------
        # Skill scores (vs climatological & persistence baselines)
        # ----------------------------------------------------------
        _station_col = "sensor_id"
        _target_col = "pm25"

        # Ensure date column exists for persistence baseline
        for _df in [train_data, dev_data] + ([test_data] if test_data is not None else []):
            if "date" not in _df.columns:
                _df["date"] = pd.to_datetime(_df["time"]).dt.date

        print("\n" + "=" * 80)
        print(f"SKILL SCORES (DEV {dev_year})")
        print("=" * 80)
        dev_skill = self.compute_skill_metrics(
            train_data, dev_data, y_dev, dev_pred,
            station_col=_station_col, target_col=_target_col, verbose=True,
        )

        test_skill = None
        if test_data is not None:
            print("\n" + "=" * 80)
            print(f"SKILL SCORES (TEST {test_year})")
            print("=" * 80)
            test_skill = self.compute_skill_metrics(
                train_data, test_data, y_test, test_pred,
                station_col=_station_col, target_col=_target_col, verbose=True,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"paper_results_{timestamp}.json"
        model_path = self.results_dir / f"paper_model_{timestamp}.joblib"

        model_payload = {
            "model": final_model,
            "feature_cols": feat_cols,
            "train_medians": med.to_dict(),
            "use_latlon": self.use_latlon,
            "lgbm_params": self.LGBM_PARAMS,
            "registry_csv": str(self.registry_path),
            "master_lake_path": str(self.master_lake_path),
            "training_mode": "paper_benchmark",
        }
        joblib.dump(model_payload, model_path)

        results = {
            "model_type": "registry_authoritative_lgbm_daily_lake",
            "training_mode": "paper_benchmark",
            "use_latlon": self.use_latlon,
            "strict_registry": self.strict_registry,
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
                "dev_skill": dev_skill,
                "test_overall": test_overall_metrics,
                "test_monthly": test_monthly_metrics,
                "test_seasonal": test_seasonal_metrics,
                "test_skill": test_skill,
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper/benchmark training with hyperparameter tuning and CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Paper/Benchmark Training Mode:
  Fixed train/dev/test split for reproducible research results.
  Includes optional hyperparameter tuning via rolling CV.

Examples:
  # Full paper training with tuning
  python benchmark.py --do_tuning --tuning_trials 25

  # Paper training without tuning (use default hyperparams)
  python benchmark.py --no_tuning

  # Custom year splits
  python benchmark.py --dev_year 2024 --test_year 2025

For operational (deployment) training, use:
  python production.py
        """,
    )
    parser.add_argument(
        "--lake_path",
        type=str,
        default="../feature_engineering/output/master_lake",
        help="Path to master_lake directory",
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
        default="results/benchmark_model",
        help="Directory to save results",
    )
    parser.add_argument(
        "--use_latlon",
        action="store_true",
        help="Include lat/lon as features",
    )
    parser.add_argument(
        "--no_tuning",
        action="store_true",
        help="Skip hyperparameter tuning",
    )
    parser.add_argument(
        "--do_tuning",
        action="store_true",
        help="Enable hyperparameter tuning (default if neither specified)",
    )
    parser.add_argument(
        "--tuning_trials",
        type=int,
        default=25,
        help="Number of hyperparameter tuning trials",
    )
    parser.add_argument(
        "--dev_year",
        type=int,
        default=2024,
        help="Development/validation year",
    )
    parser.add_argument(
        "--test_year",
        type=int,
        default=2025,
        help="Test year (held out)",
    )

    args = parser.parse_args()

    # Determine tuning behavior
    do_tuning = True  # default
    if args.no_tuning:
        do_tuning = False
    elif args.do_tuning:
        do_tuning = True

    trainer = BenchmarkTrainer(
        master_lake_path=args.lake_path,
        registry_csv=args.registry_csv,
        results_dir=args.results_dir,
        use_latlon=args.use_latlon,
        strict_registry=True,
    )

    trainer.run_paper_training(
        final_train_years=(2020, 2021, 2022, 2023),
        dev_year=args.dev_year,
        test_year=args.test_year,
        do_tuning=do_tuning,
        tuning_trials=args.tuning_trials,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
