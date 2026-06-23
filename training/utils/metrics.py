"""
Core Evaluation Metrics
=======================

Standalone metric functions for PM2.5 model evaluation:
- RMSE, MAE, R², Bias, F1@150
- Monthly and seasonal breakdowns
- MAE skill score computation

These functions are used by both the Trainer (training.utils.trainer)
and benchmark (training.main) training pipelines.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score


# Pakistan seasons
SEASONS = {
    "winter": [12, 1, 2],
    "pre_monsoon": [3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "post_monsoon": [10, 11],
}


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Core evaluation metrics: RMSE, MAE, R², Bias, F1@150.

    Parameters
    ----------
    y_true : array-like
        Ground truth PM2.5 values (µg/m³)
    y_pred : array-like
        Predicted PM2.5 values (µg/m³)
    verbose : bool
        Print metrics to stdout

    Returns
    -------
    dict
        Dictionary with rmse, mae, r2, bias, f1_150, extreme_rmse, counts
    """
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
        print(f"  RMSE: {rmse:.1f} ug/m3")
        print(f"  MAE:  {mae:.1f} ug/m3")
        print(f"  R2:   {r2:.3f}")
        print(f"  Bias: {bias:.1f} ug/m3")
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


def compute_monthly_metrics(
    data: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Per-month evaluation metrics."""
    monthly = {}
    for month in sorted(data["month"].unique()):
        mask = data["month"] == month
        monthly[str(int(month))] = evaluate_predictions(y_true[mask], y_pred[mask], verbose=False)
    return monthly


def compute_seasonal_metrics(
    data: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasons: dict = None,
) -> dict:
    """Per-season evaluation metrics (Pakistan seasons by default)."""
    if seasons is None:
        seasons = SEASONS
    seasonal = {}
    for season_name, months in seasons.items():
        mask = data["month"].isin(months)
        if not mask.any():
            continue
        seasonal[season_name] = evaluate_predictions(y_true[mask], y_pred[mask], verbose=False)
    return seasonal


def compute_mae_skill(mae_model: float, mae_baseline: float) -> float:
    """
    Compute MAE skill score: 1 - MAE_model / MAE_baseline

    Interpretation:
    - skill > 0: Model beats baseline
    - skill = 0: Model same as baseline
    - skill < 0: Model worse than baseline
    - skill = 1: Perfect predictions (MAE_model = 0)

    Reference: Mason (2004) "On using climatology as a reference strategy"
    """
    if np.isnan(mae_baseline) or mae_baseline == 0:
        return float("nan")
    return 1.0 - (mae_model / mae_baseline)


def compute_rmse_skill(rmse_model: float, rmse_baseline: float) -> float:
    """
    Compute RMSE skill score: 1 - RMSE_model / RMSE_baseline

    Same interpretation as MAE skill.
    """
    if np.isnan(rmse_baseline) or rmse_baseline == 0:
        return float("nan")
    return 1.0 - (rmse_model / rmse_baseline)
