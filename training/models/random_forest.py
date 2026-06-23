"""
Random Forest & ExtraTrees
==========================

Bagged tree ensembles with different bias/variance tradeoffs than boosting.
Often worse than boosting but can be better in noisy label settings (LCS noise, drift).
Good robustness baseline.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


RF_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 0,
}

ET_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 0,
}


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    params: Optional[Dict] = None,
) -> RandomForestRegressor:
    """
    Train Random Forest regressor.

    Note: RF doesn't support early stopping, so validation set is ignored
    but kept for API consistency.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (unused, for API consistency)
        y_val: Validation targets (unused)
        params: Override default parameters

    Returns:
        Trained RandomForestRegressor
    """
    model_params = RF_DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)

    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)

    return model


def train_extra_trees(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    params: Optional[Dict] = None,
) -> ExtraTreesRegressor:
    """
    Train ExtraTrees regressor.

    ExtraTrees uses random splits instead of optimal splits,
    which can reduce variance and improve generalization on noisy data.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (unused, for API consistency)
        y_val: Validation targets (unused)
        params: Override default parameters

    Returns:
        Trained ExtraTreesRegressor
    """
    model_params = ET_DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)

    model = ExtraTreesRegressor(**model_params)
    model.fit(X_train, y_train)

    return model


def predict_rf_quantiles(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
) -> Dict[float, np.ndarray]:
    """
    Estimate quantiles from Random Forest tree predictions.

    Uses individual tree predictions to estimate prediction intervals.

    Args:
        model: Trained RandomForestRegressor
        X: Features
        quantiles: Quantile levels

    Returns:
        Dict mapping quantile level to predictions
    """
    # Get predictions from all trees
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])

    # Compute quantiles across trees
    return {
        q: np.percentile(tree_preds, q * 100, axis=0)
        for q in quantiles
    }
