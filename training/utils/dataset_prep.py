"""
Dataset Preparation (Layer 2)
=============================

Transform raw DataFrame into trainable dataset:
- Schema normalization
- Target detection/filtering
- Derived encodings
- Feature selection
- Imputation
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Import FeatureRegistry
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent
_MFE = _REPO_ROOT / "feature_engineering"
if _MFE.exists():
    sys.path.append(str(_MFE))

from feature_family.feature_registry import FeatureRegistry, TARGET_COL


# =============================================================================
# Schema Guards
# =============================================================================

def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'time' column exists, falling back to date_utc."""
    df = df.copy()
    if "time" not in df.columns and "date_utc" in df.columns:
        df["time"] = pd.to_datetime(df["date_utc"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def validate_required_columns(df: pd.DataFrame, required: List[str] = None):
    """Check that required columns exist."""
    if required is None:
        required = ["sensor_id", "time"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Master dataset missing required column: {c}")


def check_duplicates(df: pd.DataFrame, subset: List[str] = None):
    """Detect duplicates on key columns."""
    if subset is None:
        subset = ["sensor_id", "time"]
    dup = df.duplicated(subset=subset).sum()
    if dup > 0:
        raise ValueError(
            f"Master dataset has {dup} duplicate ({', '.join(subset)}) rows. "
            f"Fix upstream merges before training."
        )


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect target column name."""
    if TARGET_COL in df.columns:
        return TARGET_COL
    if "pm25" in df.columns:
        return "pm25"
    raise ValueError(f"Master dataset is missing target column '{TARGET_COL}' (and no 'pm25' fallback).")


def coerce_and_filter_target(df: pd.DataFrame, target_col: str, min_value: float = 5.0) -> pd.DataFrame:
    """Coerce target to numeric and filter invalid values."""
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    valid = df[target_col].notna() & (df[target_col] >= min_value)
    df = df[valid].copy()
    print(f"PM2.5 range: {df[target_col].min():.1f} to {df[target_col].max():.1f} ug/m3")
    return df


# =============================================================================
# Time Fields
# =============================================================================

def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_year, month, year columns."""
    df = df.dropna(subset=["time"]).copy()
    df["day_of_year"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    return df


# =============================================================================
# Derived Encodings
# =============================================================================

def add_daily_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and seasonal encodings."""
    df = df.copy()

    # Cyclical day-of-year encodings (1st, 2nd, 3rd harmonics)
    if "doy_sin" not in df.columns:
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    if "doy_cos" not in df.columns:
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    if "doy_sin_2" not in df.columns:
        df["doy_sin_2"] = np.sin(4 * np.pi * df["day_of_year"] / 366.0)
    if "doy_cos_2" not in df.columns:
        df["doy_cos_2"] = np.cos(4 * np.pi * df["day_of_year"] / 366.0)
    if "doy_sin_3" not in df.columns:
        df["doy_sin_3"] = np.sin(6 * np.pi * df["day_of_year"] / 366.0)
    if "doy_cos_3" not in df.columns:
        df["doy_cos_3"] = np.cos(6 * np.pi * df["day_of_year"] / 366.0)

    # Seasonal flags
    if "heating_season_flag" not in df.columns:
        df["heating_season_flag"] = df["month"].isin([11, 12, 1, 2]).astype("int8")
    if "burning_season_flag" not in df.columns:
        df["burning_season_flag"] = df["month"].isin([10, 11]).astype("int8")

    # Wind direction encoding
    if "WD10" in df.columns:
        if "WD10_sin" not in df.columns:
            df["WD10_sin"] = np.sin(np.deg2rad(pd.to_numeric(df["WD10"], errors="coerce")))
        if "WD10_cos" not in df.columns:
            df["WD10_cos"] = np.cos(np.deg2rad(pd.to_numeric(df["WD10"], errors="coerce")))

    return df


# =============================================================================
# Feature Selection
# =============================================================================

DERIVED_FEATURES = [
    "doy_sin", "doy_cos", "doy_sin_2", "doy_cos_2", "doy_sin_3", "doy_cos_3",
    "heating_season_flag", "burning_season_flag",
    "WD10_sin", "WD10_cos",
]

SPATIAL_FEATURES = ["obs_lat", "obs_lon"]
SPATIAL_FEATURES_FALLBACK = ["latitude", "longitude"]


def is_excluded_feature(col: str) -> bool:
    """Check if column should be excluded from features."""
    c = str(col).lower()
    if "invalid_reason" in c:
        return True
    if c.endswith("_file_used") or "file_used" in c:
        return True
    if c.endswith("_scale_factor") or c.endswith("_add_offset"):
        return True
    if c.endswith("_qa_threshold") or c.endswith("_qa_available"):
        return True
    return False


def select_candidate_features(
    df: pd.DataFrame,
    registry_feature_cols: List[str],
    use_latlon: bool = False,
) -> List[str]:
    """Select candidate feature columns based on registry and derived features."""
    candidates = [c for c in registry_feature_cols if c in df.columns]
    candidates += [c for c in DERIVED_FEATURES if c in df.columns]

    if use_latlon:
        if set(SPATIAL_FEATURES).issubset(df.columns):
            candidates += SPATIAL_FEATURES
        elif set(SPATIAL_FEATURES_FALLBACK).issubset(df.columns):
            candidates += SPATIAL_FEATURES_FALLBACK

    candidates = [c for c in candidates if not is_excluded_feature(c)]

    # Deduplicate while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# =============================================================================
# Feature Preparation (X, y matrices)
# =============================================================================

def prepare_features(
    data: pd.DataFrame,
    registry_feature_cols: List[str],
    feature_cols: Optional[List[str]] = None,
    train_medians: Optional[pd.Series] = None,
    fit_medians: bool = False,
    min_non_nan_frac: float = 0.01,
    use_latlon: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], Optional[pd.Series]]:
    """
    Prepare feature matrix X and target vector y.

    Args:
        data: Input DataFrame
        registry_feature_cols: Features from registry
        feature_cols: Override feature columns (if None, auto-select)
        train_medians: Pre-computed medians for imputation
        fit_medians: If True, compute medians from this data
        min_non_nan_frac: Minimum non-NaN fraction to keep a feature
        use_latlon: Include lat/lon features

    Returns:
        (X, y, feature_cols, train_medians)
    """
    data = data.copy()
    target_col = TARGET_COL if TARGET_COL in data.columns else "pm25"

    if feature_cols is None:
        feature_cols = select_candidate_features(data, registry_feature_cols, use_latlon)

    if not feature_cols:
        raise ValueError("No feature columns selected. Check registry and master dataset columns.")

    X = data.loc[:, feature_cols].copy()

    # Coerce to numeric
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Filter by non-NaN fraction if fitting
    if fit_medians:
        nn = X.notna().mean()
        feature_cols = [c for c in feature_cols if float(nn.get(c, 0.0)) >= float(min_non_nan_frac)]
        if len(feature_cols) < 20:
            feature_cols = list(X.columns)

        X = data.loc[:, feature_cols].copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

        print(f"Using {len(feature_cols)} features (registry-driven; min_non_nan_frac={min_non_nan_frac})")

    y = pd.to_numeric(data[target_col], errors="coerce").values

    # Compute or apply medians
    if fit_medians:
        train_medians = X.median(numeric_only=True)

    if train_medians is not None:
        X = X.fillna(train_medians)
    else:
        X = X.fillna(X.median(numeric_only=True))

    return X, y, feature_cols, train_medians


# =============================================================================
# Full Pipeline
# =============================================================================

def prepare_master_data(
    df: pd.DataFrame,
    registry_feature_cols: List[str],
    strict_registry: bool = True,
    lake_path: str = "",
    registry_path: str = "",
) -> pd.DataFrame:
    """
    Full preparation pipeline: schema -> time fields -> encodings -> filter.

    Args:
        df: Raw DataFrame from data loader
        registry_feature_cols: Feature columns from registry
        strict_registry: Raise error if registry features missing
        lake_path: For error messages
        registry_path: For error messages

    Returns:
        Prepared DataFrame ready for feature extraction
    """
    # Schema guards
    df = ensure_time_column(df)
    validate_required_columns(df)
    check_duplicates(df)

    # Target
    target_col = detect_target_column(df)
    df = coerce_and_filter_target(df, target_col)

    # Time fields
    df = add_time_fields(df)

    # Derived encodings
    df = add_daily_encodings(df)

    # Registry check
    missing = sorted(list(set(registry_feature_cols) - set(df.columns)))
    if missing:
        msg = (
            f"{len(missing)} registry features are missing from the master dataset.\n"
            f"Master lake: {lake_path}\nRegistry: {registry_path}\n"
            f"Examples: {missing[:30]}"
        )
        if strict_registry:
            raise ValueError(msg)
        else:
            print("WARNING:", msg)

    df = df.sort_values(["sensor_id", "time"]).reset_index(drop=True)
    return df
