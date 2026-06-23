"""
Ablation Studies Module
=======================

Systematic ablation studies for PM2.5 estimation model components.

Components:
- feature_ablation: Feature importance and ablation analysis
- regulation_alignment: EPA/reference station alignment approaches
- anchor_approaches/: Submodules for different anchoring strategies
"""

from training.ablation.regulation_alignment import AnchoredTrainer

__all__ = ["AnchoredTrainer"]
