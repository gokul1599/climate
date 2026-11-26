"""Regional climate indicator calculations from model rollouts (indicative, not operational)."""
from __future__ import annotations

import numpy as np

LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0)

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Western Ghats & Konkan": (10.0, 20.0, 72.0, 77.0),
    "Central India Monsoon Core": (18.0, 26.0, 75.0, 85.0),
    "Indo-Gangetic Plain": (24.0, 31.0, 75.0, 88.0),
    "Northeast India": (22.0, 29.0, 88.0, 97.0),
    "Bay of Bengal": (8.0, 20.0, 82.0, 94.0),
    "Arabian Sea": (8.0, 20.0, 65.0, 75.0),
}


def region_mask(box: tuple[float, float, float, float]) -> np.ndarray:
    la0, la1, lo0, lo1 = box
    return ((LAT >= la0) & (LAT <= la1))[:, None] & ((LON >= lo0) & (LON <= lo1))[None, :]


def compute_regional_summary(rollout: np.ndarray, heavy_rain_thresh: float = 10.0) -> list[dict]:
    """Summarise mean rainfall, temperature, wind speed, moisture, and heavy-rain grid counts per region."""
    summaries = []
    for name, box in REGIONS.items():
        m = region_mask(box)
        tp = rollout[:, 1][:, m]
        t2m_c = rollout[:, 0][:, m] - 273.15
        u10 = rollout[:, 2][:, m]
        v10 = rollout[:, 3][:, m]
        tcwv = rollout[:, 5][:, m]
        wind = np.sqrt(u10 ** 2 + v10 ** 2)
        heavy_count = int((tp >= heavy_rain_thresh).sum())
        summaries.append(
            {
                "region": name,
                "grid_cells": int(m.sum()),
                "mean_tp_mm_day": float(tp.mean()),
                "max_tp_mm_day": float(tp.max()),
                "heavy_rain_cell_days": heavy_count,
                "mean_t2m_c": float(t2m_c.mean()),
                "mean_wind_ms": float(wind.mean()),
                "mean_tcwv_kg_m2": float(tcwv.mean()),
            }
        )
    return summaries
