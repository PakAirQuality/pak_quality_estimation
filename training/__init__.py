"""
PM2.5 Estimation Training Package
=================================

Structure:
    training/
    ├── utils/              # Utilities
    │   ├── trainer.py      # Core Trainer class
    │   └── metrics.py      # Evaluation metrics (RMSE, MAE, R², F1@150)
    │
    ├── baselines/          # Baseline models
    │   ├── climatology.py  # Station-month median baseline
    │   ├── persistence.py  # Yesterday's PM2.5 baseline
    │   └── geos_cf.py      # GEOS-CF CTM baseline
    │
    ├── prod/               # Production artifacts
    │   ├── simulation.py   # Waterline simulation
    │   └── best_model_weight/
    │
    ├── ablation/           # Ablation studies
    │
    ├── main.py                  # Paper training
    ├── evaluate_benchmark.py    # Baseline evaluation script
    └── ...

Usage:
    from training.utils.trainer import Trainer, load_master_from_lake
    from training.utils.metrics import evaluate_predictions, compute_mae_skill
    from training.baselines import GEOSCFBaseline, compute_climatology_baseline
"""

from training.utils.trainer import Trainer, load_master_from_lake

__all__ = ["Trainer", "load_master_from_lake"]
