"""
XGBoost Model
=============

Gradient boosting cousin to LightGBM. Often matches or slightly beats LGBM
depending on tuning and missingness patterns.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


DEFAULT_PARAMS = {
    "objective": "reg:absoluteerror",
    "learning_rate": 0.05,
    "n_estimators": 5000,
    "max_depth": 8,
    "min_child_weight": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 150,
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor with early stopping.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: Override default parameters
        early_stopping_rounds: Early stopping patience

    Returns:
        Trained XGBRegressor
    """
    model_params = DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)

    model = xgb.XGBRegressor(**model_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def train_xgboost_quantile(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    params: Optional[Dict] = None,
) -> Dict[float, xgb.XGBRegressor]:
    """
    Train XGBoost quantile regression models.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        quantiles: Quantile levels to fit
        params: Override default parameters

    Returns:
        Dict mapping quantile level to trained model
    """
    models = {}

    for q in quantiles:
        model_params = DEFAULT_PARAMS.copy()
        model_params["objective"] = "reg:quantileerror"
        model_params["quantile_alpha"] = q
        if params:
            model_params.update(params)

        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[q] = model

    return models


def predict_quantiles(
    models: Dict[float, xgb.XGBRegressor],
    X: pd.DataFrame,
) -> Dict[float, np.ndarray]:
    """
    Generate quantile predictions from trained models.

    Args:
        models: Dict of quantile models from train_xgboost_quantile
        X: Features

    Returns:
        Dict mapping quantile level to predictions
    """
    return {q: model.predict(X) for q, model in models.items()}
