"""
Baseline Models for Ablation Comparison
=======================================

Baseline and ablation models to compare against the anchored approach:
- backbone_only: Just backbone, no EPA correction
- anchor_only: Train only on EPA, no LCS backbone
- global: Joint training on LCS + EPA
- global_weighted: Joint training with EPA upweighted
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def train_global_model(
    lcs_data: pd.DataFrame,
    epa_data: pd.DataFrame,
    feat_cols: List[str],
    medians: pd.Series,
    prepare_features_fn: Callable,
    train_lgbm_fn: Callable,
    backbone_params: Optional[dict] = None,
    seed: int = 42,
):
    """
    Train global model on combined LCS + EPA data.

    Args:
        lcs_data: Low-cost sensor training data
        epa_data: EPA anchor station training data
        feat_cols: Feature column names
        medians: Pre-computed feature medians for imputation
        prepare_features_fn: Function to prepare X, y from data
        train_lgbm_fn: Function to train LightGBM model
        backbone_params: Override params for model
        seed: Random seed

    Returns:
        Trained global model
    """
    rng = np.random.RandomState(seed)

    all_train = pd.concat([lcs_data, epa_data], ignore_index=True)
    X_all, y_all, _, _ = prepare_features_fn(
        all_train, feature_cols=feat_cols, train_medians=medians, fit_medians=False
    )

    n_all = len(X_all)
    val_idx = rng.choice(n_all, size=max(1, n_all // 5), replace=False)
    train_mask = np.ones(n_all, dtype=bool)
    train_mask[val_idx] = False

    global_model = train_lgbm_fn(
        X_all[train_mask], y_all[train_mask],
        X_all[~train_mask], y_all[~train_mask],
        override_params=backbone_params,
    )

    return global_model


def train_global_weighted_model(
    lcs_data: pd.DataFrame,
    epa_data: pd.DataFrame,
    feat_cols: List[str],
    medians: pd.Series,
    prepare_features_fn: Callable,
    train_lgbm_fn: Callable,
    backbone_params: Optional[dict] = None,
    w_epa: float = 10.0,
    seed: int = 42,
):
    """
    Train global model with EPA samples upweighted.

    Args:
        lcs_data: Low-cost sensor training data
        epa_data: EPA anchor station training data (must have 'is_epa' column)
        feat_cols: Feature column names
        medians: Pre-computed feature medians for imputation
        prepare_features_fn: Function to prepare X, y from data
        train_lgbm_fn: Function to train LightGBM model with sample_weight
        backbone_params: Override params for model
        w_epa: Weight multiplier for EPA samples (default 10.0)
        seed: Random seed

    Returns:
        Trained weighted global model
    """
    rng = np.random.RandomState(seed)

    all_train = pd.concat([lcs_data, epa_data], ignore_index=True)
    X_all, y_all, _, _ = prepare_features_fn(
        all_train, feature_cols=feat_cols, train_medians=medians, fit_medians=False
    )

    # Build sample weights
    is_epa = all_train["is_epa"].values if "is_epa" in all_train.columns else np.zeros(len(all_train), dtype=bool)
    weights = np.where(is_epa, w_epa, 1.0).astype(float)

    n_all = len(X_all)
    val_idx = rng.choice(n_all, size=max(1, n_all // 5), replace=False)
    train_mask = np.ones(n_all, dtype=bool)
    train_mask[val_idx] = False

    global_model = train_lgbm_fn(
        X_all[train_mask], y_all[train_mask],
        X_all[~train_mask], y_all[~train_mask],
        override_params=backbone_params,
        sample_weight=weights[train_mask],
        eval_sample_weight=weights[~train_mask],
    )

    return global_model


def train_anchor_only_model(
    epa_data: pd.DataFrame,
    feat_cols: List[str],
    medians: pd.Series,
    prepare_features_fn: Callable,
    train_lgbm_fn: Callable,
    backbone_params: Optional[dict] = None,
    seed: int = 42,
):
    """
    Train model only on EPA anchor data (no LCS backbone).

    This is an ablation baseline testing whether LCS pre-training helps.

    Args:
        epa_data: EPA anchor station training data
        feat_cols: Feature column names
        medians: Pre-computed feature medians for imputation
        prepare_features_fn: Function to prepare X, y from data
        train_lgbm_fn: Function to train LightGBM model
        backbone_params: Override params for model
        seed: Random seed

    Returns:
        Trained anchor-only model
    """
    rng = np.random.RandomState(seed)

    X_epa, y_epa, _, _ = prepare_features_fn(
        epa_data, feature_cols=feat_cols, train_medians=medians, fit_medians=False
    )

    n_epa = len(X_epa)
    val_idx = rng.choice(n_epa, size=max(1, n_epa // 5), replace=False)
    train_mask = np.ones(n_epa, dtype=bool)
    train_mask[val_idx] = False

    anchor_model = train_lgbm_fn(
        X_epa[train_mask], y_epa[train_mask],
        X_epa[~train_mask], y_epa[~train_mask],
        override_params=backbone_params,
    )

    return anchor_model


def evaluate_baselines(
    backbone,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    evaluate_fn: Callable,
    global_model=None,
    global_weighted_model=None,
    anchor_only_model=None,
) -> Dict[str, Dict]:
    """
    Evaluate all baseline models on the same test set.

    Args:
        backbone: Trained backbone model
        X_test: Test features
        y_test: Test labels
        evaluate_fn: Function to compute metrics (y_true, y_pred) -> dict
        global_model: Optional global model
        global_weighted_model: Optional weighted global model
        anchor_only_model: Optional anchor-only model

    Returns:
        Dict mapping baseline name -> metrics dict
    """
    baselines = {}

    # Backbone-only
    backbone_pred = backbone.predict(X_test)
    baselines["backbone_only"] = evaluate_fn(y_test, backbone_pred, verbose=False)

    # Global
    if global_model is not None:
        global_pred = global_model.predict(X_test)
        baselines["global"] = evaluate_fn(y_test, global_pred, verbose=False)

    # Global weighted
    if global_weighted_model is not None:
        global_w_pred = global_weighted_model.predict(X_test)
        baselines["global_weighted"] = evaluate_fn(y_test, global_w_pred, verbose=False)

    # Anchor-only
    if anchor_only_model is not None:
        anchor_pred = anchor_only_model.predict(X_test)
        baselines["anchor_only"] = evaluate_fn(y_test, anchor_pred, verbose=False)

    return baselines
