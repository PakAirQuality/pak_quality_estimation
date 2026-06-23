from __future__ import annotations
import numpy as np
from scipy.ndimage import generic_filter

def _nanfunc_factory(func):
    def f(values):
        v = values.astype(float)
        v = v[~np.isnan(v)]
        return func(v) if v.size else np.nan
    return f

def window_stat(arr: np.ndarray, radius: int, stat: str) -> np.ndarray:
    """
    Compute nan-aware window stats over a 2D array.
    radius=1 means 3x3, radius=2 means 5x5, etc.
    """
    size = 2 * radius + 1
    if stat == "mean":
        fn = _nanfunc_factory(np.mean)
    elif stat == "max":
        fn = _nanfunc_factory(np.max)
    elif stat == "min":
        fn = _nanfunc_factory(np.min)
    elif stat == "std":
        fn = _nanfunc_factory(np.std)
    else:
        raise ValueError(f"Unsupported stat={stat}")
    return generic_filter(arr, fn, size=(size, size), mode="nearest")