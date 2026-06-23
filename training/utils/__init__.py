"""
Training Utilities
==================

3-Layer Architecture:
    Layer 1 (data_loader.py)   - Raw data fetching
    Layer 2 (dataset_prep.py)  - Schema normalization, features, filtering
    Layer 3 (trainer.py)       - Model training, tuning, evaluation
"""

from .metrics import (
    SEASONS,
    evaluate_predictions,
    compute_monthly_metrics,
    compute_seasonal_metrics,
    compute_mae_skill,
)

# Layer 1: Data loading
from .data_loader import load_master_from_lake

# Layer 2: Dataset preparation
from .dataset_prep import (
    prepare_master_data,
    prepare_features,
    add_daily_encodings,
)

# Layer 3: Training pipeline
from .trainer import Trainer

__all__ = [
    # Metrics
    "SEASONS",
    "evaluate_predictions",
    "compute_monthly_metrics",
    "compute_seasonal_metrics",
    "compute_mae_skill",
    # Layer 1
    "load_master_from_lake",
    # Layer 2
    "prepare_master_data",
    "prepare_features",
    "add_daily_encodings",
    # Layer 3
    "Trainer",
]
