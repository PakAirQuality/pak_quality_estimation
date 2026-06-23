from __future__ import annotations
import pandas as pd

def gap_safe_rolling(
    df: pd.DataFrame,
    group_cols: list[str],
    date_col: str,
    value_cols: list[str],
    window: int,
    agg: str = "mean",
    max_gap_days: int = 1,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Rolling window that breaks when gaps exceed max_gap_days.
    Works for point-based training features (per-station time series).
    """
    out = df.copy()
    out = out.sort_values(group_cols + [date_col])

    # detect gaps inside each group
    dt = out.groupby(group_cols)[date_col].diff().dt.days
    new_seg = (dt.isna()) | (dt > max_gap_days)
    seg_id = new_seg.groupby(out[group_cols].apply(tuple, axis=1)).cumsum()

    # roll inside (group, segment)
    key = list(group_cols) + [seg_id.rename("_seg")]

    for c in value_cols:
        s = out.groupby(key)[c]
        if agg == "mean":
            out[f"{c}_roll_mean_{window}d"] = s.rolling(window, min_periods=min_periods).mean().reset_index(level=key, drop=True)
        elif agg == "max":
            out[f"{c}_roll_max_{window}d"] = s.rolling(window, min_periods=min_periods).max().reset_index(level=key, drop=True)
        elif agg == "min":
            out[f"{c}_roll_min_{window}d"] = s.rolling(window, min_periods=min_periods).min().reset_index(level=key, drop=True)
        elif agg == "std":
            out[f"{c}_roll_std_{window}d"] = s.rolling(window, min_periods=min_periods).std().reset_index(level=key, drop=True)
        else:
            raise ValueError(f"Unsupported agg={agg}")
    return out