"""
Daily Calibration Approaches
============================

Per-day calibration methods that use same-day EPA reference data
to correct model predictions online.

Approaches:
- daily_offset: Additive offset b_t = median(y - y_hat) per day
- daily_affine: Affine correction y = a_t * y_hat + b_t per day
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor


def fit_daily_offset(
    anchor_data: pd.DataFrame,
    predictions: np.ndarray,
    y_true: np.ndarray,
    min_stations_per_day: int = 3,
) -> Dict:
    """
    Fit per-day additive offset: b_t = median(y - y_hat) for each date.

    Uses median for robustness to outliers. Simpler and often more stable
    than affine when anchors are few/noisy.

    Args:
        anchor_data: DataFrame with 'time' column for date extraction
        predictions: Model predictions on anchor data
        y_true: True PM2.5 values at anchor stations
        min_stations_per_day: Minimum observations to fit offset (else 0.0)

    Returns:
        Dict mapping date -> offset value
    """
    dates = pd.to_datetime(anchor_data["time"]).dt.date.values
    offset_params: Dict = {}
    residuals = y_true - predictions

    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < min_stations_per_day:
            offset_params[d] = 0.0
            continue
        offset_params[d] = float(np.median(residuals[mask]))

    return offset_params


def predict_daily_offset(
    test_data: pd.DataFrame,
    predictions: np.ndarray,
    offset_params: Dict,
) -> np.ndarray:
    """
    Apply per-day additive offset: y_hat + b_t.

    Falls back to 0.0 for dates not in offset_params.

    Args:
        test_data: DataFrame with 'time' column for date extraction
        predictions: Base model predictions
        offset_params: Dict from fit_daily_offset()

    Returns:
        Corrected predictions array
    """
    dates = pd.to_datetime(test_data["time"]).dt.date.values
    corrected = np.empty_like(predictions)
    for i, (d, pred) in enumerate(zip(dates, predictions)):
        corrected[i] = pred + offset_params.get(d, 0.0)
    return corrected


def fit_daily_affine(
    anchor_data: pd.DataFrame,
    predictions: np.ndarray,
    y_true: np.ndarray,
    min_stations_per_day: int = 5,
    huber_epsilon: float = 1.35,
) -> Dict:
    """
    Fit per-day robust affine calibration: y = a_t * y_hat + b_t.

    For each date with >= min_stations_per_day observations, fits a
    HuberRegressor for robustness to outliers.
    Falls back to identity (1.0, 0.0) if too few stations or fit fails.

    Args:
        anchor_data: DataFrame with 'time' column for date extraction
        predictions: Model predictions on anchor data
        y_true: True PM2.5 values at anchor stations
        min_stations_per_day: Minimum observations to fit affine
        huber_epsilon: Huber loss epsilon parameter (default 1.35)

    Returns:
        Dict mapping date -> (slope, intercept) tuple
    """
    dates = pd.to_datetime(anchor_data["time"]).dt.date.values
    affine_params: Dict = {}

    for d in np.unique(dates):
        mask = dates == d
        n_obs = mask.sum()
        if n_obs < min_stations_per_day:
            affine_params[d] = (1.0, 0.0)
            continue

        X_day = predictions[mask].reshape(-1, 1)
        y_day = y_true[mask]

        try:
            reg = HuberRegressor(epsilon=huber_epsilon)
            reg.fit(X_day, y_day)
            affine_params[d] = (float(reg.coef_[0]), float(reg.intercept_))
        except Exception:
            affine_params[d] = (1.0, 0.0)

    return affine_params


def predict_daily_affine(
    test_data: pd.DataFrame,
    predictions: np.ndarray,
    affine_params: Dict,
) -> np.ndarray:
    """
    Apply per-day affine correction: a_t * y_hat + b_t.

    Falls back to identity (1.0, 0.0) for dates not in affine_params.

    Args:
        test_data: DataFrame with 'time' column for date extraction
        predictions: Base model predictions
        affine_params: Dict from fit_daily_affine()

    Returns:
        Corrected predictions array
    """
    dates = pd.to_datetime(test_data["time"]).dt.date.values
    corrected = np.empty_like(predictions)
    for i, (d, pred) in enumerate(zip(dates, predictions)):
        a, b = affine_params.get(d, (1.0, 0.0))
        corrected[i] = a * pred + b
    return corrected
