"""
Two-Stage Backbone + Residual Training
=======================================

Two-stage model that leverages both dense LCS data and sparse EPA anchors:
  Stage 1: Backbone LightGBM trained on all LCS data
  Stage 2: Residual-correction LightGBM trained on EPA anchor stations

Final prediction: backbone(x) + lambda * residual(x, backbone(x))

The residual model receives the backbone prediction as an additional feature,
allowing it to learn level-dependent corrections.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


def train_two_stage(
    lcs_data: pd.DataFrame,
    epa_data: pd.DataFrame,
    feat_cols: List[str],
    medians: pd.Series,
    prepare_features_fn: Callable,
    train_lgbm_fn: Callable,
    backbone_params: Optional[dict] = None,
    residual_params: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple:
    """
    Train two-stage backbone + residual model.

    Stage 1: Backbone on LCS with 20% random holdout for early stopping.
    Stage 2: Residual on EPA with station-holdout for early stopping.

    Args:
        lcs_data: Low-cost sensor training data
        epa_data: EPA anchor station training data
        feat_cols: Feature column names
        medians: Pre-computed feature medians for imputation
        prepare_features_fn: Function to prepare X, y from data
        train_lgbm_fn: Function to train LightGBM model
        backbone_params: Override params for backbone model
        residual_params: Override params for residual model
        seed: Random seed for reproducibility
        verbose: Print residual statistics

    Returns:
        (backbone_model, residual_model) tuple
    """
    rng = np.random.RandomState(seed)

    # Stage 1: Backbone on LCS (20% random holdout for early stopping)
    X_lcs, y_lcs, _, _ = prepare_features_fn(
        lcs_data, feature_cols=feat_cols, train_medians=medians, fit_medians=False
    )
    n_lcs = len(X_lcs)
    lcs_val_idx = rng.choice(n_lcs, size=max(1, n_lcs // 5), replace=False)
    lcs_train_mask = np.ones(n_lcs, dtype=bool)
    lcs_train_mask[lcs_val_idx] = False

    backbone = train_lgbm_fn(
        X_lcs[lcs_train_mask], y_lcs[lcs_train_mask],
        X_lcs[~lcs_train_mask], y_lcs[~lcs_train_mask],
        override_params=backbone_params,
    )

    # Stage 2: Residual on EPA (station-holdout for early stopping)
    X_epa, y_epa, _, _ = prepare_features_fn(
        epa_data, feature_cols=feat_cols, train_medians=medians, fit_medians=False
    )
    backbone_pred_epa = backbone.predict(X_epa)
    y_residual = y_epa - backbone_pred_epa

    # Add backbone prediction as feature for level-dependent corrections
    X_epa_aug = X_epa.copy()
    X_epa_aug["backbone_pred"] = backbone_pred_epa

    if verbose:
        print(
            f"    Residual stats: mean={np.mean(y_residual):.2f}, "
            f"std={np.std(y_residual):.2f}, "
            f"min={np.min(y_residual):.2f}, max={np.max(y_residual):.2f}"
        )

    # Station-holdout early stopping to avoid station-specific overfit
    unique_sids = np.array(sorted(epa_data["sensor_id"].unique()))
    n_val_sids = max(1, int(0.2 * len(unique_sids)))
    val_sids = set(rng.choice(unique_sids, size=n_val_sids, replace=False))
    val_mask = epa_data["sensor_id"].isin(val_sids).values
    train_mask = ~val_mask

    residual_model = train_lgbm_fn(
        X_epa_aug[train_mask], y_residual[train_mask],
        X_epa_aug[val_mask], y_residual[val_mask],
        override_params=residual_params,
    )

    return backbone, residual_model


def predict_two_stage(
    backbone,
    residual_model,
    X: pd.DataFrame,
    lambda_: float = 1.0,
) -> np.ndarray:
    """
    Combined prediction: backbone(x) + lambda * residual(x, backbone(x)).

    Args:
        backbone: Trained backbone model
        residual_model: Trained residual model
        X: Feature DataFrame (without backbone_pred column)
        lambda_: Residual weight (default 1.0)

    Returns:
        Final predictions array
    """
    backbone_pred = backbone.predict(X)
    X_aug = X.copy()
    X_aug["backbone_pred"] = backbone_pred
    return backbone_pred + lambda_ * residual_model.predict(X_aug)
