"""
Anchor Approaches Submodule
===========================

Submodules implementing different EPA/reference anchoring strategies
for the regulation_alignment ablation study.

Modules:
- daily_calibration: Per-day offset and affine calibration methods
- two_stage: Backbone + residual training and prediction
- baselines: Ablation baseline models (global, anchor_only, etc.)
"""

from training.ablation.anchor_approaches.daily_calibration import (
    fit_daily_offset,
    predict_daily_offset,
    fit_daily_affine,
    predict_daily_affine,
)
from training.ablation.anchor_approaches.two_stage import (
    train_two_stage,
    predict_two_stage,
)
from training.ablation.anchor_approaches.baselines import (
    train_global_model,
    train_global_weighted_model,
    train_anchor_only_model,
    evaluate_baselines,
)

__all__ = [
    # Daily calibration
    "fit_daily_offset",
    "predict_daily_offset",
    "fit_daily_affine",
    "predict_daily_affine",
    # Two-stage
    "train_two_stage",
    "predict_two_stage",
    # Baselines
    "train_global_model",
    "train_global_weighted_model",
    "train_anchor_only_model",
    "evaluate_baselines",
]
