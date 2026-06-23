#!/usr/bin/env python3
"""Build the held-out-2025 station-day validation table.

Loads the saved production support-aware model and runs inference twice over the
2025 test rows -- once with the monitor-history feature block present
(`pred_with_history`) and once with it masked (`pred_no_history`) -- mirroring the
feature preparation in `training/support_aware/train_production.py`. The output
reproduces the paper's headline 2025 numbers (with-history MAE 14.3 / F1@150 0.838;
no-history MAE 23.2).

Requirements (RESTRICTED -- not redistributed):
  - the master feature lake (`feature_engineering/output/master_lake`)
  - the feature registry (`training/data/features_registry.csv`)
  - the production model weights (`inference/best_model_weight/support_aware_*.joblib`)
Provided for transparency; it will not run without the above.

Output is anonymised before release: exact coordinates, device ids, and site names
are dropped and the station id is replaced by a salted, non-reversible hash.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from training.utils.trainer import Trainer
from training.support_aware.tier2 import build_temporal_features

MODEL = "inference/best_model_weight/support_aware_20260325_132406.joblib"
PROVINCE_CSV = "timeseries_plots_2024_2025/province/sensor_day_pm25_actual_predicted_2024_2025.csv"
SALT = "paqi-acp-2025-validation"
OUT = "validation/pm25_2025_station_day_validation.csv"


def main() -> None:
    art = joblib.load(MODEL)
    model = art["model"]
    all_cols = list(art["feature_cols"])
    base_cols = all_cols[: int(art["n_exogenous_features"])]
    temp_cols = list(art["temporal_features"])
    med = pd.Series(art["train_medians"])

    trainer = Trainer(
        master_lake_path="feature_engineering/output/master_lake",
        registry_csv="training/data/features_registry.csv",
        results_dir="training/results/support_aware_production",
        use_latlon=False, strict_registry=True,
    )
    full = trainer.load_and_prepare_data()
    if "date" not in full.columns:
        full["date"] = pd.to_datetime(full["time"]).dt.date
    awt, temp_feats = build_temporal_features(
        full.copy(), family=1, max_lag_days=3, include_own_lags=True,
    )
    test = awt[awt["year"] == 2025].copy().reset_index(drop=True)

    X_base, y, _, _ = trainer.prepare_features(
        test, feature_cols=base_cols, train_medians=med, fit_medians=False,
    )
    X_base = X_base.reset_index(drop=True)

    X_full = X_base.copy()
    for c in temp_feats:
        X_full[c] = test[c].values
    X_full = X_full[all_cols]

    X_nan = X_base.copy()
    for c in temp_feats:
        X_nan[c] = np.nan
    X_nan = X_nan[all_cols]

    pred_hist = np.expm1(model.predict(X_full))
    pred_nohist = np.expm1(model.predict(X_nan))

    prov = {}
    pc = Path(PROVINCE_CSV)
    if pc.exists():
        p = pd.read_csv(pc, usecols=lambda c: c in ("sensor_id", "province"))
        prov = p.dropna().drop_duplicates("sensor_id").set_index("sensor_id")["province"].to_dict()

    sid = test["sensor_id"].astype(str)
    out = pd.DataFrame({
        "date": pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d"),
        "station_id": sid.map(lambda s: "PK" + hashlib.sha1((SALT + s).encode()).hexdigest()[:10]),
        "city": test.get("City"),
        "province": test["sensor_id"].map(prov),
        "pm25_observed": np.asarray(y, float).round(2),
        "pred_with_history": pred_hist.round(2),
        "pred_no_history": pred_nohist.round(2),
    }).sort_values(["date", "station_id"]).reset_index(drop=True)

    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows")


if __name__ == "__main__":
    main()
