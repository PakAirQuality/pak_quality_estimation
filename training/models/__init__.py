"""
Model Zoo
=========

Tree-based models for PM2.5 estimation.

Model Categories:
-----------------

1) Tree-boosting:
   - XGBoost: Often matches or beats LightGBM
   - CatBoost: More stable, good quantile support

2) Bagged trees:
   - RandomForest: Robustness baseline
   - ExtraTrees: Better on noisy labels

Usage:
------
    from training.models.xgboost_model import train_xgboost
    from training.models.catboost_model import train_catboost
    from training.models.random_forest import train_random_forest, train_extra_trees

Note: Import models directly from submodules to avoid import errors
      if optional dependencies (catboost) are not installed.
"""

__all__ = [
    # XGBoost
    "train_xgboost",
    "train_xgboost_quantile",
    # CatBoost
    "train_catboost",
    "train_catboost_quantile",
    # Random Forest / ExtraTrees
    "train_random_forest",
    "train_extra_trees",
    "predict_rf_quantiles",
]


def __getattr__(name):
    """Lazy imports to avoid errors when optional dependencies are missing."""

    # XGBoost
    if name in ("train_xgboost", "train_xgboost_quantile"):
        from . import xgboost_model
        return getattr(xgboost_model, name)

    # CatBoost
    if name in ("train_catboost", "train_catboost_quantile"):
        from . import catboost_model
        return getattr(catboost_model, name)

    # Random Forest / ExtraTrees
    if name in ("train_random_forest", "train_extra_trees", "predict_rf_quantiles"):
        from . import random_forest
        return getattr(random_forest, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
