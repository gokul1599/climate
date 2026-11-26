"""Regional climate and Monsoon Stress indicator calculations from model rollouts (indicative, not operational)."""
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

STRESS_FORMULAS: dict[str, str] = {
    "heavy_rain_exposure": "Cell-days with tp >= 10.0 mm/day and % = 100 * cell_days / (steps * grid_cells)",
    "persistent_rainfall_duration": "Max consecutive days across [0..steps-1] with tp >= 10.0 mm/day (or wet >= 2.5 mm/day) at any cell in region",
    "heat_stress_proxy": "Unvalidated apparent-temp proxy: T_c + 0.33*(0.65*tcwv) - 0.70*wind - 4.0 (deg C), plus cell-days T >= 35 C",
    "strong_wind_exposure": "Cell-days with wind = sqrt(u10^2 + v10^2) >= 10.0 m/s and regional max instantaneous daily wind",
}


def region_mask(box: tuple[float, float, float, float]) -> np.ndarray:
    la0, la1, lo0, lo1 = box
    return ((LAT >= la0) & (LAT <= la1))[:, None] & ((LON >= lo0) & (LON <= lo1))[None, :]


def _max_consecutive_days(bool_txc: np.ndarray) -> int:
    """Return longest run of consecutive True days for any spatial cell inside mask [T, N_cells]."""
    if bool_txc.size == 0:
        return 0
    best = 0
    t_steps, n_cells = bool_txc.shape
    for c in range(n_cells):
        cur = 0
        for t in range(t_steps):
            if bool_txc[t, c]:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
    return int(best)


def compute_heat_stress_proxy_c(t2m_c: np.ndarray, tcwv: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """Apparent-temperature humidity/wind proxy (deg C). Unvalidated proxy, not medical heat risk."""
    vapor_proxy_hpa = 0.65 * np.maximum(tcwv, 0.0)
    return t2m_c + 0.33 * vapor_proxy_hpa - 0.70 * np.maximum(wind, 0.0) - 4.0


def compute_regional_summary(
    rollout: np.ndarray,
    heavy_rain_thresh: float = 10.0,
    wet_day_thresh: float = 2.5,
    strong_wind_thresh: float = 10.0,
    hot_t2m_c_thresh: float = 35.0,
) -> list[dict]:
    """Summarise climate means and formula-backed Monsoon Stress indicators per Indian region."""
    steps = rollout.shape[0]
    summaries = []
    for name, box in REGIONS.items():
        m = region_mask(box)
        n_cells = int(m.sum())
        tp = rollout[:, 1][:, m]
        t2m_c = rollout[:, 0][:, m] - 273.15
        u10 = rollout[:, 2][:, m]
        v10 = rollout[:, 3][:, m]
        tcwv = rollout[:, 5][:, m]
        wind = np.sqrt(u10 ** 2 + v10 ** 2)

        heavy_mask = tp >= heavy_rain_thresh
        wet_mask = tp >= wet_day_thresh
        wind_mask = wind >= strong_wind_thresh
        hot_mask = t2m_c >= hot_t2m_c_thresh
        heat_proxy = compute_heat_stress_proxy_c(t2m_c, tcwv, wind)

        heavy_count = int(heavy_mask.sum())
        denom = max(steps * n_cells, 1)

        summaries.append(
            {
                "region": name,
                "grid_cells": n_cells,
                "mean_tp_mm_day": float(tp.mean()),
                "max_tp_mm_day": float(tp.max()),
                "heavy_rain_cell_days": heavy_count,
                "heavy_rain_exposure_pct": round(100.0 * heavy_count / denom, 2),
                "persistent_heavy_streak_days": _max_consecutive_days(heavy_mask),
                "persistent_wet_streak_days": _max_consecutive_days(wet_mask),
                "mean_t2m_c": float(t2m_c.mean()),
                "max_t2m_c": float(t2m_c.max()),
                "hot_day_cell_days": int(hot_mask.sum()),
                "heat_stress_proxy_c": round(float(heat_proxy.mean()), 2),
                "heat_stress_proxy_max_c": round(float(heat_proxy.max()), 2),
                "mean_wind_ms": float(wind.mean()),
                "max_wind_ms": float(wind.max()),
                "strong_wind_cell_days": int(wind_mask.sum()),
                "mean_tcwv_kg_m2": float(tcwv.mean()),
            }
        )
    return summaries
