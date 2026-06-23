"""
Feature-Family Ablation Study
==============================

Ablates features by information source (family), not individual columns,
to quantify the contribution of each data stream to PM2.5 estimation.

Families
--------
  MET      : ERA5 + derived lags / rolls / VC / seasonality encodings
  AOD      : MAIAC AOD + QA / uncertainty + multi-scale deltas
  TROPOMI  : 9 Sentinel-5P products (NO2, SO2, CO, HCHO, O3, AAI, ALH, CH4, cloud)
  NH3      : GEOS-CF NH3 features (own block, separated from TROPOMI)

Experiments
-----------
  Leave-one-out : Full − {family}  for each family
  Additive      : MET → +AOD → +TROPOMI → +NH3

Usage:
    python -m training.ablation.feature_ablation
    python -m training.ablation.feature_ablation --results_dir results/ablation
"""

import json
import sys
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.utils.trainer import Trainer, load_master_from_lake


# ── family classification ────────────────────────────────────────────

def classify_features(
    feature_cols: List[str],
    col_to_stage: Dict[str, str],
    derived_features: List[str],
) -> Dict[str, List[str]]:
    """Assign every feature column to exactly one family."""
    families: Dict[str, List[str]] = {
        "MET": [],
        "AOD": [],
        "TROPOMI": [],
        "NH3": [],
    }
    for col in feature_cols:
        # derived temporal encodings → MET
        if col in derived_features:
            families["MET"].append(col)
            continue
        stage = col_to_stage.get(col, "")
        if stage == "met":
            families["MET"].append(col)
        elif stage == "aod":
            families["AOD"].append(col)
        elif stage == "tropomi":
            if col.startswith("nh3_"):
                families["NH3"].append(col)
            else:
                families["TROPOMI"].append(col)
        else:
            # fallback: assign to MET (shouldn't happen with clean registry)
            families["MET"].append(col)
    return families


# ── ablation runner ──────────────────────────────────────────────────

class AblationTrainer(Trainer):
    """Extends Trainer with family-level ablation experiments."""

    def _train_and_eval(
        self,
        train_data: pd.DataFrame,
        dev_data: pd.DataFrame,
        feature_subset: List[str],
        label: str,
    ) -> dict:
        """Train on *feature_subset* and return overall + seasonal metrics."""
        X_train, y_train, _, med = self.prepare_features(
            train_data, feature_cols=feature_subset, fit_medians=True
        )
        X_dev, y_dev, _, _ = self.prepare_features(
            dev_data, feature_cols=feature_subset, train_medians=med, fit_medians=False
        )

        model = self.train_lgbm_model(X_train, y_train, X_dev, y_dev)
        pred = model.predict(X_dev)

        overall = self.evaluate_predictions(y_dev, pred, verbose=False)
        seasonal = self.compute_seasonal_metrics(dev_data, y_dev, pred)

        print(
            f"  {label:40s} | n_feat={len(feature_subset):3d} "
            f"| MAE={overall['mae']:5.1f} | R2={overall['r2']:.3f} "
            f"| F1@150={overall.get('f1_150', 0) or 0:.3f}"
        )

        return {
            "label": label,
            "n_features": len(feature_subset),
            "overall": overall,
            "seasonal": seasonal,
        }

    # ─────────────────────────────────────────────────────────────────

    def run_ablation(
        self,
        train_years: Tuple[int, ...] = (2020, 2021, 2022, 2023),
        dev_year: int = 2024,
    ) -> dict:
        full_data = self.load_and_prepare_data()
        all_years = sorted(full_data["year"].unique())
        train_years = [y for y in train_years if y in all_years]

        train_data = self.get_year_data(train_years)
        dev_data = self.get_year_data([dev_year])

        # --- resolve full feature list (same logic as prepare_features) ---
        X_tmp, _, feat_cols_full, _ = self.prepare_features(
            train_data, fit_medians=True
        )

        families = classify_features(
            feat_cols_full, self._col_to_stage, self.DERIVED_FEATURES
        )

        print("\n" + "=" * 80)
        print("FEATURE FAMILY SIZES")
        print("=" * 80)
        for fam, cols in families.items():
            print(f"  {fam:10s}: {len(cols):3d} features")
        total = sum(len(v) for v in families.values())
        print(f"  {'TOTAL':10s}: {total:3d} features")

        results: Dict[str, dict] = OrderedDict()

        # ── 1. Full baseline ────────────────────────────────────────
        print("\n" + "=" * 80)
        print("FULL BASELINE")
        print("=" * 80)
        results["full"] = self._train_and_eval(
            train_data, dev_data, feat_cols_full, "Full (MET+AOD+TROPOMI+NH3)"
        )

        # ── 2. Leave-one-out ────────────────────────────────────────
        print("\n" + "=" * 80)
        print("LEAVE-ONE-OUT ABLATION  (Full − family)")
        print("=" * 80)
        for fam_name, fam_cols in families.items():
            subset = [c for c in feat_cols_full if c not in fam_cols]
            label = f"Full − {fam_name}"
            results[f"loo_{fam_name}"] = self._train_and_eval(
                train_data, dev_data, subset, label
            )

        # ── 3. Additive build-up ────────────────────────────────────
        print("\n" + "=" * 80)
        print("ADDITIVE ABLATION  (cumulative build-up)")
        print("=" * 80)
        cumulative: List[str] = []
        add_order = ["MET", "AOD", "TROPOMI", "NH3"]
        for i, fam_name in enumerate(add_order):
            cumulative = cumulative + families[fam_name]
            label = " + ".join(add_order[: i + 1])
            results[f"add_{fam_name}"] = self._train_and_eval(
                train_data, dev_data, list(cumulative), label
            )

        # ── 4. Compute deltas relative to full baseline ─────────────
        full_overall = results["full"]["overall"]
        print("\n" + "=" * 80)
        print("DELTAS vs FULL BASELINE  (negative MAE = better with family)")
        print("=" * 80)
        print(f"  {'Experiment':40s} | {'ΔMAE':>6s} | {'ΔR²':>7s} | {'ΔF1@150':>8s}")
        print("  " + "-" * 70)
        for key, res in results.items():
            if key == "full":
                continue
            d_mae = res["overall"]["mae"] - full_overall["mae"]
            d_r2 = res["overall"]["r2"] - full_overall["r2"]
            f1_full = full_overall.get("f1_150", 0) or 0
            f1_this = res["overall"].get("f1_150", 0) or 0
            d_f1 = f1_this - f1_full
            print(
                f"  {res['label']:40s} | {d_mae:+5.1f} | {d_r2:+6.4f} | {d_f1:+7.4f}"
            )

        # DJF deltas
        full_djf = (results["full"].get("seasonal") or {}).get("winter", {})
        if full_djf:
            print("\n  DJF (Winter) deltas:")
            print(f"  {'Experiment':40s} | {'ΔMAE':>6s} | {'ΔR²':>7s} | {'ΔF1@150':>8s}")
            print("  " + "-" * 70)
            for key, res in results.items():
                if key == "full":
                    continue
                djf = (res.get("seasonal") or {}).get("winter", {})
                if not djf:
                    continue
                d_mae = djf["mae"] - full_djf["mae"]
                d_r2 = djf["r2"] - full_djf["r2"]
                d_f1 = (djf.get("f1_150", 0) or 0) - (full_djf.get("f1_150", 0) or 0)
                print(
                    f"  {res['label']:40s} | {d_mae:+5.1f} | {d_r2:+6.4f} | {d_f1:+7.4f}"
                )

        # ── 5. Save ─────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.results_dir / f"ablation_results_{timestamp}.json"

        def jsonify(obj):
            if isinstance(obj, dict):
                return {k: jsonify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [jsonify(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        payload = {
            "timestamp": timestamp,
            "train_years": list(train_years),
            "dev_year": dev_year,
            "family_sizes": {k: len(v) for k, v in families.items()},
            "family_columns": {k: v for k, v in families.items()},
            "experiments": jsonify(results),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {out_path}")

        return results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature-family ablation study for PM2.5 estimation",
    )
    parser.add_argument(
        "--lake_path", type=str,
        default="../feature_engineering/output/master_lake",
        help="Path to master_lake directory",
    )
    parser.add_argument(
        "--registry_csv", type=str,
        default="data/features_registry.csv",
        help="Path to features registry CSV",
    )
    parser.add_argument(
        "--results_dir", type=str,
        default="results/ablation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--dev_year", type=int, default=2024,
        help="Development / validation year",
    )
    args = parser.parse_args()

    trainer = AblationTrainer(
        master_lake_path=args.lake_path,
        registry_csv=args.registry_csv,
        results_dir=args.results_dir,
        use_latlon=False,
        strict_registry=True,
    )
    trainer.run_ablation(dev_year=args.dev_year)
    print("\nDone.")


if __name__ == "__main__":
    main()
