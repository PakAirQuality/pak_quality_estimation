"""
Production Training Module
==========================

Production artifacts for operational PM2.5 model training.

Components:
- simulation: Waterline simulation across reference dates
- best_model_weight/: Saved model weights

Note: The core Trainer class has moved to training.utils.trainer
"""

from training.utils.trainer import Trainer, load_master_from_lake

__all__ = ["Trainer", "load_master_from_lake"]
