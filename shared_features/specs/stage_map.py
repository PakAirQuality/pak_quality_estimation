from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd

from .met_specs import is_met_feature
from .aod_specs import is_aod_feature
from .tropomi_specs import is_tropomi_feature
from .static_specs import is_static_feature

# Treat these as "metadata/housekeeping" columns (never model features).
META_PREFIXES = ("grid_", "__", "snap_", "agg_")
META_EXACT = {
    "date", "date_utc", "latitude", "longitude",
    "grid_id", "row", "col",
    "snapped_flag", "dt_days",
}

STAGES = ("meta", "static", "met", "aod", "tropomi", "unknown")

def _looks_meta(name: str) -> bool:
    n = name.lower()
    return (n in META_EXACT) or n.startswith(META_PREFIXES)

def load_stage_map_from_registry(registry_csv: Optional[str | Path]) -> Dict[str, str]:
    """
    Optional: If your features_registry.csv has a 'stage' column, we use it.
    Otherwise returns empty dict and falls back to regex inference below.
    """
    if not registry_csv:
        return {}
    p = Path(registry_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "feature" not in df.columns or "stage" not in df.columns:
        return {}
    return dict(zip(df["feature"].astype(str), df["stage"].astype(str)))

def infer_stage(name: str, registry_map: Optional[Dict[str, str]] = None) -> str:
    """
    Single source of truth for "which stage owns this feature name".
    """
    if registry_map and name in registry_map:
        st = str(registry_map[name]).strip().lower()
        # Map registry stages to shared stages (for consistency)
        stage_mapping = {
            # Elevation and coast stages removed
            "met": "met",
            "aod": "aod", 
            "tropomi": "tropomi"
        }
        st = stage_mapping.get(st, st)
        return st if st in STAGES else "unknown"

    if _looks_meta(name):
        return "meta"
    if is_static_feature(name):
        return "static"
    if is_met_feature(name):
        return "met"
    if is_aod_feature(name):
        return "aod"
    if is_tropomi_feature(name):
        return "tropomi"
    return "unknown"

def split_by_stage(
    features: Iterable[str],
    registry_csv: Optional[str | Path] = None,
) -> Dict[str, List[str]]:
    """
    Returns {"met":[...], "aod":[...], ...}
    """
    registry_map = load_stage_map_from_registry(registry_csv)
    out: Dict[str, List[str]] = {k: [] for k in STAGES}
    for f in features:
        st = infer_stage(f, registry_map)
        out.setdefault(st, []).append(f)
    # drop empties for cleanliness
    return {k: v for k, v in out.items() if v}