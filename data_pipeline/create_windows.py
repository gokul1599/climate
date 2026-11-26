"""Sliding-window index construction for 3-day input -> multi-step target sequences."""
from __future__ import annotations

import numpy as np


def build_window_indices(
    num_days: int,
    t_in: int = 3,
    horizon: int = 1,
) -> np.ndarray:
    """Return array of start indices i where inputs are [i, i+t_in) and targets are [i+t_in, i+t_in+horizon)."""
    max_start = num_days - t_in - horizon + 1
    if max_start <= 0:
        raise ValueError(f"Not enough days ({num_days}) for t_in={t_in}, horizon={horizon}")
    return np.arange(max_start, dtype=np.int64)
