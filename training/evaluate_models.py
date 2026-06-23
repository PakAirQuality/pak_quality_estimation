#!/usr/bin/env python3
"""
Multi-Model Evaluation
======================

Tree-based model comparison for PM2.5 estimation:

1) Tree-boosting (LightGBM, XGBoost, CatBoost)
2) Bagged trees (RandomForest, ExtraTrees)

Usage:
    python -m training.evaluate_models
    python -m training.evaluate_models --do_tuning
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils.trainer import Trainer
from training.utils.metrics import evaluate_predictions


# =============================================================================
# Model Training Functions
# =============================================================================

def train_and_evaluate_lgbm(
    trainer: Trainer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Dict, object, np.ndarray]:
    """Train LightGBM and evaluate."""
    model = trainer.train_lgbm_model(X_train, y_train, X_val, y_val)
    pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, pred, verbose=True)
    return metrics, model, pred


def train_and_evaluate_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Dict, object, np.ndarray]:
    """Train XGBoost and evaluate."""
    from training.models.xgboost_model import train_xgboost

    model = train_xgboost(X_train, y_train, X_val, y_val)
    pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, pred, verbose=True)
    return metrics, model, pred


def train_and_evaluate_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Dict, object, np.ndarray]:
    """Train CatBoost and evaluate."""
    from training.models.catboost_model import train_catboost

    model = train_catboost(X_train, y_train, X_val, y_val)
    pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, pred, verbose=True)
    return metrics, model, pred


def train_and_evaluate_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Dict, object, np.ndarray]:
    """Train Random Forest and evaluate."""
    from training.models.random_forest import train_random_forest

    model = train_random_forest(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, pred, verbose=True)
    return metrics, model, pred


def train_and_evaluate_extratrees(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Dict, object, np.ndarray]:
    """Train ExtraTrees and evaluate."""
    from training.models.random_forest import train_extra_trees

    model = train_extra_trees(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, pred, verbose=True)
    return metrics, model, pred


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

class MultiModelTrainer(Trainer):
    """Extended Trainer for multi-model evaluation."""

    def run_multi_model_training(
        self,
        final_train_years: Tuple[int, ...] = (2020, 2021, 2022, 2023),
        dev_year: int = 2024,
        test_year: int = 2025,
        do_tuning: bool = False,
        tuning_trials: int = 25,
    ) -> Dict:
        """
        Train all tree-based models with fixed train/dev/test split.
        """
        full_data = self.load_and_prepare_data()
        all_years = sorted(full_data["year"].unique())

        # Validate years
        final_train_years = [y for y in final_train_years if y in all_years and y < dev_year]
        if not final_train_years:
            raise ValueError("No valid final_train_years found in data.")

        if dev_year not in all_years:
            raise ValueError(f"dev_year={dev_year} not found in data.")

        has_test = test_year in all_years
        if not has_test:
            print(f"Warning: test_year={test_year} not found. Evaluating on dev only.")

        # Load data splits
        print("\n" + "=" * 80)
        print("DATA SPLITS")
        print("=" * 80)

        train_data = self.get_year_data(final_train_years)
        dev_data = self.get_year_data([dev_year])
        test_data = self.get_year_data([test_year]) if has_test else None

        # Prepare features
        X_train, y_train, feat_cols, med = self.prepare_features(train_data, fit_medians=True)
        X_dev, y_dev, _, _ = self.prepare_features(dev_data, feature_cols=feat_cols, train_medians=med)

        if test_data is not None:
            X_test, y_test, _, _ = self.prepare_features(test_data, feature_cols=feat_cols, train_medians=med)
        else:
            X_test, y_test = X_dev, y_dev

        print(f"Train: {len(y_train):,} | Dev: {len(y_dev):,} | Test: {len(y_test):,}")
        print(f"Features: {len(feat_cols)}")

        # Optional tuning for LightGBM
        if do_tuning:
            print("\n" + "=" * 80)
            print("HYPERPARAMETER TUNING (LightGBM)")
            print("=" * 80)
            best_params, tuning_path = self.tune_hyperparameters_rolling_cv(
                train_data=train_data,
                n_trials=tuning_trials,
            )
            if best_params:
                self.LGBM_PARAMS.update(best_params)

        # Results storage
        all_results = {}
        model_comparison = []

        # =====================================================================
        # Tree Boosting Models
        # =====================================================================

        # LightGBM
        print("\n" + "=" * 80)
        print("1. LightGBM (baseline)")
        print("=" * 80)
        try:
            metrics, model, pred = train_and_evaluate_lgbm(
                self, X_train, y_train, X_dev, y_dev, X_test, y_test
            )
            all_results["lightgbm"] = {
                "metrics": metrics,
                "feature_importance": list(zip(feat_cols, model.feature_importances_.tolist())),
            }
            model_comparison.append({"model": "LightGBM", **metrics})
        except Exception as e:
            print(f"LightGBM failed: {e}")
            model_comparison.append({"model": "LightGBM", "error": str(e)})

        # XGBoost
        print("\n" + "=" * 80)
        print("2. XGBoost")
        print("=" * 80)
        try:
            metrics, model, pred = train_and_evaluate_xgboost(
                X_train, y_train, X_dev, y_dev, X_test, y_test
            )
            all_results["xgboost"] = {"metrics": metrics}
            model_comparison.append({"model": "XGBoost", **metrics})
        except Exception as e:
            print(f"XGBoost failed: {e}")
            model_comparison.append({"model": "XGBoost", "error": str(e)})

        # CatBoost
        print("\n" + "=" * 80)
        print("3. CatBoost")
        print("=" * 80)
        try:
            metrics, model, pred = train_and_evaluate_catboost(
                X_train, y_train, X_dev, y_dev, X_test, y_test
            )
            all_results["catboost"] = {"metrics": metrics}
            model_comparison.append({"model": "CatBoost", **metrics})
        except Exception as e:
            print(f"CatBoost failed: {e}")
            model_comparison.append({"model": "CatBoost", "error": str(e)})

        # =====================================================================
        # Bagged Tree Models
        # =====================================================================

        # Random Forest
        print("\n" + "=" * 80)
        print("4. Random Forest")
        print("=" * 80)
        try:
            metrics, model, pred = train_and_evaluate_rf(X_train, y_train, X_test, y_test)
            all_results["random_forest"] = {"metrics": metrics}
            model_comparison.append({"model": "RandomForest", **metrics})
        except Exception as e:
            print(f"RandomForest failed: {e}")
            model_comparison.append({"model": "RandomForest", "error": str(e)})

        # ExtraTrees
        print("\n" + "=" * 80)
        print("5. ExtraTrees")
        print("=" * 80)
        try:
            metrics, model, pred = train_and_evaluate_extratrees(X_train, y_train, X_test, y_test)
            all_results["extra_trees"] = {"metrics": metrics}
            model_comparison.append({"model": "ExtraTrees", **metrics})
        except Exception as e:
            print(f"ExtraTrees failed: {e}")
            model_comparison.append({"model": "ExtraTrees", "error": str(e)})

        # =====================================================================
        # Summary
        # =====================================================================

        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)

        df = pd.DataFrame(model_comparison)
        if "mae" in df.columns:
            df = df.sort_values("mae").reset_index(drop=True)

        # Print formatted table
        print()
        print("| Model           | RMSE  | MAE   | R²    | Bias   |")
        print("|-----------------|-------|-------|-------|--------|")
        for _, row in df.iterrows():
            if "error" in row and pd.notna(row.get("error")):
                print(f"| {row['model']:<15} | ERROR |       |       |        |")
            else:
                print(f"| {row['model']:<15} | {row.get('rmse', 0):>5.1f} | {row.get('mae', 0):>5.1f} | {row.get('r2', 0):>5.3f} | {row.get('bias', 0):>+6.1f} |")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"multi_model_results_{timestamp}.json"
        csv_path = self.results_dir / f"multi_model_comparison_{timestamp}.csv"

        df.to_csv(csv_path, index=False)

        final_results = {
            "timestamp": timestamp,
            "train_years": list(final_train_years),
            "dev_year": dev_year,
            "test_year": test_year if has_test else None,
            "n_features": len(feat_cols),
            "n_train": len(y_train),
            "n_dev": len(y_dev),
            "n_test": len(y_test) if has_test else 0,
            "model_comparison": model_comparison,
            "detailed_results": all_results,
            "do_tuning": do_tuning,
        }

        with open(results_path, "w") as f:
            json.dump(self._to_jsonable(final_results), f, indent=2)

        print(f"\nResults saved to: {results_path}")
        print(f"Comparison CSV: {csv_path}")

        return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Tree-based model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lake_path", type=str, default="feature_engineering/output/master_lake")
    parser.add_argument("--registry_csv", type=str, default="training/data/features_registry.csv")
    parser.add_argument("--results_dir", type=str, default="training/results/model_comparison")
    parser.add_argument("--dev_year", type=int, default=2024)
    parser.add_argument("--test_year", type=int, default=2025)
    parser.add_argument("--do_tuning", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--tuning_trials", type=int, default=25)

    args = parser.parse_args()

    trainer = MultiModelTrainer(
        master_lake_path=args.lake_path,
        registry_csv=args.registry_csv,
        results_dir=args.results_dir,
        strict_registry=False,
    )

    trainer.run_multi_model_training(
        final_train_years=(2020, 2021, 2022, 2023),
        dev_year=args.dev_year,
        test_year=args.test_year,
        do_tuning=args.do_tuning,
        tuning_trials=args.tuning_trials,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
