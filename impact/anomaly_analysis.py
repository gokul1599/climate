"""Scenario-versus-normal anomaly calculations."""
from __future__ import annotations

import numpy as np
from impact.regional_analysis import REGIONS, STRESS_FORMULAS, compute_regional_summary, region_mask


def compute_scenario_anomalies(
    normal_rollout: np.ndarray,
    perturbed_rollout: np.ndarray,
) -> dict:
    """Compute perturbed - normal differences across regions, stress indicators, and grid cells."""
    diff = perturbed_rollout - normal_rollout
    norm_sum = {r["region"]: r for r in compute_regional_summary(normal_rollout)}
    pert_sum = {r["region"]: r for r in compute_regional_summary(perturbed_rollout)}
    regional = []
    for name, box in REGIONS.items():
        m = region_mask(box)
        nr = norm_sum[name]
        pr = pert_sum[name]
        regional.append(
            {
                "region": name,
                "delta_tp_mm_day": float(diff[:, 1][:, m].mean()),
                "delta_t2m_k": float(diff[:, 0][:, m].mean()),
                "delta_tcwv_kg_m2": float(diff[:, 5][:, m].mean()),
                "delta_u10_ms": float(diff[:, 2][:, m].mean()),
                "delta_v10_ms": float(diff[:, 3][:, m].mean()),
                "delta_heavy_rain_cell_days": pr["heavy_rain_cell_days"] - nr["heavy_rain_cell_days"],
                "delta_persistent_heavy_streak_days": pr["persistent_heavy_streak_days"] - nr["persistent_heavy_streak_days"],
                "delta_heat_stress_proxy_c": round(pr["heat_stress_proxy_c"] - nr["heat_stress_proxy_c"], 3),
                "delta_strong_wind_cell_days": pr["strong_wind_cell_days"] - nr["strong_wind_cell_days"],
            }
        )
    regional_sorted = sorted(regional, key=lambda r: abs(r["delta_tp_mm_day"]), reverse=True)
    return {
        "disclaimer": "Model-derived indicative scenario differences, not operational disaster or crop forecasts.",
        "formulas": STRESS_FORMULAS,
        "global_mean_delta_tp": float(diff[:, 1].mean()),
        "global_mean_delta_t2m": float(diff[:, 0].mean()),
        "global_mean_delta_tcwv": float(diff[:, 5].mean()),
        "regional_anomalies": regional_sorted,
    }
