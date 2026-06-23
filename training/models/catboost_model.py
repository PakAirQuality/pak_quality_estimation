"""
CatBoost Model
==============

Often more stable and robust than LightGBM/XGBoost.
Has excellent built-in quantile regression support.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


DEFAULT_PARAMS = {
    "loss_function": "MAE",
    "learning_rate": 0.05,
    "iterations": 5000,
    "depth": 8,
    "min_data_in_leaf": 25,
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
    "l2_leaf_reg": 2.0,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": False,
}


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 150,
) -> CatBoostRegressor:
    """
    Train CatBoost regressor with early stopping.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: Override default parameters
        early_stopping_rounds: Early stopping patience

    Returns:
        Trained CatBoostRegressor
    """
    model_params = DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)

    model = CatBoostRegressor(**model_params)

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    return model


def train_catboost_quantile(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 150,
) -> Dict[float, CatBoostRegressor]:
    """
    Train CatBoost quantile regression models.

    CatBoost has native multi-quantile support which is more efficient
    than training separate models.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        quantiles: Quantile levels to fit
        params: Override default parameters
        early_stopping_rounds: Early stopping patience

    Returns:
        Dict mapping quantile level to trained model
    """
    models = {}

    for q in quantiles:
        model_params = DEFAULT_PARAMS.copy()
        model_params["loss_function"] = f"Quantile:alpha={q}"
        if params:
            model_params.update(params)

        model = CatBoostRegressor(**model_params)

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        models[q] = model

    return models


def predict_quantiles(
    models: Dict[float, CatBoostRegressor],
    X: pd.DataFrame,
) -> Dict[float, np.ndarray]:
    """
    Generate quantile predictions from trained models.

    Args:
        models: Dict of quantile models from train_catboost_quantile
        X: Features

    Returns:
        Dict mapping quantile level to predictions
    """
    return {q: model.predict(X) for q, model in models.items()}
