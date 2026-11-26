"""Persistence baseline for multi-step climate forecasting."""
from __future__ import annotations

import numpy as np


def predict_persistence(init_states: np.ndarray, steps: int = 10) -> np.ndarray:
    """Repeat the latest observed state init_states[-1] for `steps` days."""
    last = np.asarray(init_states[-1], dtype=np.float32)
    return np.repeat(last[None, ...], steps, axis=0)
