"""Climatology baseline computed strictly from the chronological training split."""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

from monsoon_r1.stats import climatology as r1_climatology, clim_for


def fit_climatology(data_train: np.ndarray, time_train: np.ndarray, smooth_days: int = 15) -> np.ndarray:
    """Compute [366, 6, 32, 36] smoothed daily climatology from training data only."""
    dates_int = np.array(
        [int(pd.Timestamp(t).strftime("%Y%m%d")) for t in time_train],
        dtype=np.int64,
    )
    return r1_climatology(data_train, dates_int, smooth_days=smooth_days)


def predict_climatology(clim: np.ndarray, last_input_date: dt.date, steps: int = 10) -> np.ndarray:
    """Predict `steps` days starting at last_input_date + 1 day using training climatology."""
    out = [
        clim_for(clim, last_input_date + dt.timedelta(days=k + 1))
        for k in range(steps)
    ]
    return np.stack(out, axis=0).astype(np.float32)
