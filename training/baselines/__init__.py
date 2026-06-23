"""
Baseline Models for PM2.5 Evaluation
=====================================

Reference baselines for computing skill scores:
1. Climatological (station-month median) - Mason (2004)
2. Persistence (yesterday's PM2.5)
3. GEOS-CF CTM (independent chemical transport model)
"""

from .climatology import (
    compute_station_month_medians,
    compute_climatology_baseline,
)

from .persistence import (
    compute_persistence_baseline,
    add_pm25_lag_features,
)

from .geos_cf import GEOSCFBaseline

__all__ = [
    # Climatology
    "compute_station_month_medians",
    "compute_climatology_baseline",
    # Persistence
    "compute_persistence_baseline",
    "add_pm25_lag_features",
    # GEOS-CF
    "GEOSCFBaseline",
]
