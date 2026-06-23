#!/usr/bin/env python3
"""
Feature Registry System for Master Feature Engineering Pipeline

This module provides utilities for:
1. Maintaining a global feature registry
2. Managing the master dataset (paqi_with_all_features.csv)
3. Enforcing feature vs. metadata separation
4. Creating stage-specific metadata files

Each feature generation script should use these utilities to:
- Register all columns it adds
- Update the master dataset
- Export stage-specific metadata
- Maintain transparency and auditability
"""

from pathlib import Path
from typing import List, Dict, Optional, Literal, Set
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Configuration
MASTER_DATASET_PATH = Path("output/paqi_with_all_features.csv")
FEATURE_REGISTRY_PATH = Path("output/features_registry.csv")
METADATA_DIR = Path("output/metadata")

# Core split metadata (for grouping/splitting, not features)
CORE_SPLIT_METADATA = [
    "obs_lat", "obs_lon", "City", "Name", "date_utc", "latitude", "longitude"
]
DEFAULT_JOIN_KEYS = ["sensor_id", "time"]
TARGET_COL = "pm25"

class FeatureRegistry:
    """Manages the global feature registry and master dataset."""
    
    def __init__(self):
        self.registry_path = FEATURE_REGISTRY_PATH
        self.master_path = MASTER_DATASET_PATH
        self.metadata_dir = METADATA_DIR
        
        # Ensure output directories exist
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize registry
        self.registry = self._load_or_init_registry()
    
    def _save_dataframe_to_csv(self, df: pd.DataFrame, path: Path, **kwargs) -> None:
        """
        FIXED: Save DataFrame with boolean columns as 0/1 to prevent string vs bool issues.
        
        Args:
            df: DataFrame to save
            path: File path to save to
            **kwargs: Additional arguments passed to to_csv()
        """
        df_copy = df.copy()
        
        # Convert boolean columns to integers (0/1) to prevent "True"/"False" string issues
        bool_columns = df_copy.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df_copy[col] = df_copy[col].astype(int)
        
        # Save with proper boolean representation
        df_copy.to_csv(path, **kwargs)
    
    def _load_or_init_registry(self) -> pd.DataFrame:
        """
        Load existing registry or create new one.
        FIXED: Properly convert boolean fields to avoid string vs bool comparison issues.
        """
        if self.registry_path.exists():
            # Load with proper boolean conversion to prevent "True" (string) vs True (bool) mismatches
            converters = {
                "used_in_training": lambda x: str(x).lower() in ("1", "true", "yes", "t")
            }
            df = pd.read_csv(self.registry_path, converters=converters)
            
            # Ensure used_in_training is boolean dtype
            if "used_in_training" in df.columns:
                df["used_in_training"] = df["used_in_training"].astype(bool)
            
            return df
        else:
            return pd.DataFrame(columns=[
                "column_name", "stage", "category", "description", 
                "data_type", "date_added", "used_in_training"
            ])
    
    def register_columns(
        self,
        stage: str,
        columns: Dict[str, Dict],
        description: Optional[str] = None
    ) -> None:
        """
        Register columns added by a stage.
        
        Args:
            stage: Name of the feature generation stage (e.g., 'tropomi', 'aod', 'met')
            columns: Dict mapping column names to their properties
                    e.g., {"no2_median": {"category": "feature", "description": "...", "used_in_training": True}}
            description: Optional description of the stage
        """
        new_rows = []
        for col_name, props in columns.items():
            row = {
                "column_name": col_name,
                "stage": stage,
                "category": props.get("category", "feature"),  # feature, metadata, target
                "description": props.get("description", ""),
                "data_type": props.get("data_type", "float32"),
                "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "used_in_training": props.get("used_in_training", True)
            }
            new_rows.append(row)
        
        # Remove existing entries for these columns (update)
        existing_cols = set(self.registry["column_name"])
        new_cols = set(columns.keys())
        self.registry = self.registry[~self.registry["column_name"].isin(new_cols)]
        
        # Add new entries
        new_df = pd.DataFrame(new_rows)
        self.registry = pd.concat([self.registry, new_df], ignore_index=True)
        
        # Save registry (ensure booleans are saved as 0/1 instead of True/False strings)
        self._save_dataframe_to_csv(self.registry, self.registry_path, index=False)
        
        print(f"[registry] Registered {len(new_cols)} columns for stage '{stage}'")
        if existing_cols & new_cols:
            print(f"[registry] Updated {len(existing_cols & new_cols)} existing columns")
    
    def get_columns(self, category: Optional[str] = None, stage: Optional[str] = None, used_in_training: Optional[bool] = None) -> List[str]:
        """Get columns with optional filtering by category, stage, and training usage."""
        if self.registry.empty:
            return []
        
        mask = pd.Series([True] * len(self.registry))
        if category is not None:
            mask = mask & (self.registry["category"] == category)
        if stage is not None:
            mask = mask & (self.registry["stage"] == stage)
        if used_in_training is not None:
            mask = mask & (self.registry["used_in_training"] == used_in_training)
        
        return self.registry[mask]["column_name"].tolist()
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all columns marked as features for training."""
        return self.get_columns(category="feature", used_in_training=True)
    
    def get_metadata_columns(self) -> List[str]:
        """Get list of all metadata columns."""
        return self.get_columns(category="metadata")
    
    def get_stage_columns(self, stage: str, category: Optional[str] = None) -> List[str]:
        """Get columns for a specific stage, optionally filtered by category."""
        return self.get_columns(category=category, stage=stage)
    
    def load_master_dataset(self) -> Optional[pd.DataFrame]:
        """Load the master dataset."""
        if self.master_path.exists():
            return pd.read_csv(self.master_path)
        return None
    
    def update_master_dataset(
        self, 
        stage: str, 
        new_data: pd.DataFrame,
        on_keys: List[str] = None
    ) -> pd.DataFrame:
        """
        Update the master dataset enforcing clean-master contract.
        master = join_keys + core_split_metadata + target + training_features_only
        """
        if on_keys is None:
            on_keys = DEFAULT_JOIN_KEYS
            
        self.master_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Which columns are allowed in the master?
        allowed_features = self.get_columns(category="feature", used_in_training=True)
        allowed = []
        for c in on_keys + CORE_SPLIT_METADATA + [TARGET_COL] + allowed_features:
            if c in new_data.columns:
                allowed.append(c)

        # 2) If master exists, merge ONLY the stage feature cols (and target if needed)
        if self.master_path.exists():
            master = pd.read_csv(self.master_path)
            if not all(k in master.columns for k in on_keys):
                raise ValueError(f"Master missing join keys {on_keys}")

            stage_features = [
                c for c in self.get_columns(stage=stage, category="feature", used_in_training=True)
                if c in new_data.columns
            ]

            # only bring what we actually want to update/add
            update_cols = stage_features[:]
            if TARGET_COL in new_data.columns and TARGET_COL not in master.columns:
                update_cols.append(TARGET_COL)

            if update_cols:  # only merge if there are columns to update
                add_df = new_data[on_keys + update_cols].copy()
                
                # FIXED: Check for duplicate join keys to prevent row explosion
                duplicates = add_df.duplicated(subset=on_keys)
                if duplicates.any():
                    dup_count = duplicates.sum()
                    dup_examples = add_df[duplicates][on_keys].head(3)
                    raise ValueError(
                        f"Stage '{stage}' data contains {dup_count} duplicate join key(s) {on_keys}. "
                        f"This would cause row explosion in master merge. "
                        f"Examples: {dup_examples.to_dict('records')}"
                    )

                # avoid _x/_y suffixes
                master = master.drop(columns=[c for c in update_cols if c in master.columns], errors="ignore")
                master = master.merge(add_df, on=on_keys, how="left")

            # ensure core metadata exists (only if missing)
            missing_core = [c for c in CORE_SPLIT_METADATA if c not in master.columns and c in new_data.columns]
            if missing_core:
                core_df = new_data[on_keys + missing_core].copy()
                
                # FIXED: Check for duplicate join keys in core metadata
                core_duplicates = core_df.duplicated(subset=on_keys)
                if core_duplicates.any():
                    core_dup_count = core_duplicates.sum()
                    core_dup_examples = core_df[core_duplicates][on_keys].head(3)
                    raise ValueError(
                        f"Stage '{stage}' core metadata contains {core_dup_count} duplicate join key(s) {on_keys}. "
                        f"This would cause row explosion in master merge. "
                        f"Examples: {core_dup_examples.to_dict('records')}"
                    )
                
                master = master.merge(core_df, on=on_keys, how="left")

        else:
            # first stage creates master: but ONLY allowed columns
            master = new_data[allowed].copy()
            
            # FIXED: Check for duplicate join keys in initial master creation
            initial_duplicates = master.duplicated(subset=on_keys)
            if initial_duplicates.any():
                initial_dup_count = initial_duplicates.sum()
                initial_dup_examples = master[initial_duplicates][on_keys].head(3)
                raise ValueError(
                    f"Stage '{stage}' initial data contains {initial_dup_count} duplicate join key(s) {on_keys}. "
                    f"Master dataset cannot have duplicate keys. "
                    f"Examples: {initial_dup_examples.to_dict('records')}"
                )

        # 3) Hard-prune: keep only allowed columns (based on registry)
        allowed_features_now = self.get_columns(category="feature", used_in_training=True)
        keep = []
        for c in on_keys + CORE_SPLIT_METADATA + [TARGET_COL] + allowed_features_now:
            if c in master.columns:
                keep.append(c)

        master = master.loc[:, list(dict.fromkeys(keep))]  # dedupe keep list
        self._save_dataframe_to_csv(master, self.master_path, index=False)
        print(f"[registry] Updated master dataset (clean): {len(master):,} rows, {len(master.columns)} cols")
        return master
    
    def export_stage_metadata(
        self, 
        stage: str, 
        data: pd.DataFrame,
        join_keys: List[str] = None
    ) -> Path:
        """
        Export stage-specific metadata file.
        
        Args:
            stage: Name of the stage
            data: Full dataset
            join_keys: Minimal join keys to include
        
        Returns:
            Path to the exported metadata file
        """
        if join_keys is None:
            join_keys = ["sensor_id", "time", "date_utc", "obs_lat", "obs_lon", "City", "Name"]
        
        # Get metadata columns for this stage
        stage_metadata = self.get_stage_columns(stage, category="metadata")
        
        # Include join keys and stage-specific metadata
        metadata_cols = list(set(join_keys) & set(data.columns))
        metadata_cols.extend([col for col in stage_metadata if col in data.columns])
        
        if not metadata_cols:
            print(f"[metadata] No metadata columns found for stage '{stage}'")
            return None
        
        # Export metadata
        metadata_df = data[metadata_cols].drop_duplicates()
        metadata_path = self.metadata_dir / f"{stage}_metadata.csv"
        self._save_dataframe_to_csv(metadata_df, metadata_path, index=False)
        
        print(f"[metadata] Exported {len(metadata_df):,} rows, {len(metadata_cols)} columns to {metadata_path}")
        return metadata_path
    
    def print_summary(self) -> None:
        """Print a summary of the current registry."""
        if self.registry.empty:
            print("[registry] No features registered yet")
            return
        
        print("\n" + "=" * 60)
        print("FEATURE REGISTRY SUMMARY")
        print("=" * 60)
        
        # Summary by stage and category
        summary = self.registry.groupby(["stage", "category"]).size().unstack(fill_value=0)
        print(summary)
        
        # Training features count
        training_features = len(self.get_feature_columns())
        print(f"\nTotal features used in training: {training_features}")
        
        # Recent additions
        recent = self.registry.sort_values("date_added", ascending=False).head(10)
        print(f"\nRecent additions:")
        for _, row in recent.iterrows():
            print(f"  {row['column_name']} ({row['stage']}) - {row['category']}")


def validate_stage_output(
    data: pd.DataFrame, 
    stage: str, 
    expected_features: List[str],
    registry: FeatureRegistry
) -> bool:
    """
    Validate that a stage's output meets requirements.
    
    Args:
        data: Output DataFrame from the stage
        stage: Stage name
        expected_features: List of expected feature columns
        registry: FeatureRegistry instance
    
    Returns:
        True if validation passes
    """
    issues = []
    
    # Check required columns exist
    missing_features = set(expected_features) - set(data.columns)
    if missing_features:
        issues.append(f"Missing expected features: {missing_features}")
    
    # Check for join keys
    join_keys = ["sensor_id", "time"]
    missing_keys = set(join_keys) - set(data.columns)
    if missing_keys:
        issues.append(f"Missing join keys: {missing_keys}")
    
    # Check for target leakage
    feature_cols = registry.get_feature_columns()
    suspicious = [col for col in feature_cols if "pm25" in col.lower() and col != "pm25"]
    if suspicious:
        issues.append(f"Potential target leakage: {suspicious}")
    
    # Check data types
    for col in expected_features:
        if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
            if not col.endswith(("_reason", "_used", "_available")):  # Allow string columns for specific cases
                issues.append(f"Non-numeric feature column: {col}")
    
    if issues:
        print(f"[validation] FAILED for stage '{stage}':")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print(f"[validation] PASSED for stage '{stage}'")
    return True


# Convenience function for stages to use
def register_stage_features(
    stage: str,
    data: pd.DataFrame,
    feature_columns: List[str],
    metadata_columns: List[str] = None,
    descriptions: Dict[str, str] = None
) -> None:
    """
    Convenience function for stages to register their features and update datasets.
    
    Args:
        stage: Stage name
        data: Output DataFrame
        feature_columns: List of feature columns added by this stage
        metadata_columns: List of metadata columns added by this stage
        descriptions: Optional descriptions for columns
    """
    registry = FeatureRegistry()
    
    metadata_columns = metadata_columns or []
    descriptions = descriptions or {}
    
    # DUPLICATE KEY CHECK: Validate join keys before proceeding
    join_keys = ["sensor_id", "time"]
    missing_join_keys = [key for key in join_keys if key not in data.columns]
    if missing_join_keys:
        raise ValueError(
            f"Stage '{stage}' data is missing required join keys: {missing_join_keys}. "
            f"Available columns: {list(data.columns)}"
        )
    
    # Check for duplicate join keys early to prevent downstream issues
    duplicates = data.duplicated(subset=join_keys)
    if duplicates.any():
        dup_count = duplicates.sum()
        total_rows = len(data)
        dup_examples = data[duplicates][join_keys].head(5)
        
        raise ValueError(
            f"Stage '{stage}' contains {dup_count:,} duplicate rows out of {total_rows:,} total rows "
            f"based on join keys {join_keys}. This would cause data corruption in master dataset merge. "
            f"Duplicate examples:\n{dup_examples.to_string(index=False)}\n"
            f"Please ensure each (sensor_id, time) combination appears only once in your stage output."
        )
    
    # Register features
    for col in feature_columns:
        if col in data.columns:
            registry.register_columns(stage, {col: {
                "category": "feature",
                "used_in_training": True,
                "description": descriptions.get(col, f"{stage} feature: {col}"),
                "data_type": str(data[col].dtype)
            }})
    
    # Register metadata
    for col in metadata_columns:
        if col in data.columns:
            registry.register_columns(stage, {col: {
                "category": "metadata",
                "used_in_training": False,
                "description": descriptions.get(col, f"{stage} metadata: {col}"),
                "data_type": str(data[col].dtype)
            }})
    
    # Export stage metadata (this is your separate metadata artifact)
    if metadata_columns:
        registry.export_stage_metadata(stage, data)
    
    # Update master (feature-only; clamps columns)
    registry.update_master_dataset(stage, data, on_keys=["sensor_id", "time"])
    
    print(f"[{stage}] Registered {len([c for c in feature_columns if c in data.columns])} features and {len([c for c in metadata_columns if c in data.columns])} metadata columns")