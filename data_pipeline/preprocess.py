"""Preprocessing utilities: chronological split, normalisation, and day-of-year encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(
    num_days: int,
    train_frac: float = 0.8,
    gap_days: int = 10,
) -> tuple[slice, slice]:
    """Return non-overlapping chronological train and val slices with a gap to prevent window leakage."""
    split_idx = int(num_days * train_frac)
    train_slice = slice(0, split_idx)
    val_start = min(split_idx + gap_days, num_days)
    val_slice = slice(val_start, num_days)
    return train_slice, val_slice


def compute_doy_sincos(time_vals: np.ndarray) -> np.ndarray:
    """Compute [N, 2] sin/cos day-of-year encoding matching monsoon_r1.contract.doy_sincos."""
    dt_idx = pd.to_datetime(time_vals)
    doy = dt_idx.dayofyear.to_numpy(dtype=np.float32)
    angle = 2.0 * np.pi * doy / 365.0
    return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def normalise_dynamic(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalise [..., 6, 32, 36] array using per-channel mean/std."""
    m = np.asarray(mean, dtype=np.float32).reshape((1,) * (x.ndim - 3) + (6, 1, 1))
    s = np.asarray(std, dtype=np.float32).reshape((1,) * (x.ndim - 3) + (6, 1, 1))
    return ((x - m) / s).astype(np.float32)


def denormalise_dynamic(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert normalised [..., 6, 32, 36] array back to physical units."""
    m = np.asarray(mean, dtype=np.float32).reshape((1,) * (x_norm.ndim - 3) + (6, 1, 1))
    s = np.asarray(std, dtype=np.float32).reshape((1,) * (x_norm.ndim - 3) + (6, 1, 1))
    return (x_norm * s + m).astype(np.float32)
