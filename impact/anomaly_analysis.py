"""Scenario-versus-normal anomaly calculations."""
from __future__ import annotations

import numpy as np
from impact.regional_analysis import REGIONS, region_mask


def compute_scenario_anomalies(
    normal_rollout: np.ndarray,
    perturbed_rollout: np.ndarray,
) -> dict:
    """Compute perturbed - normal differences across regions and grid cells."""
    diff = perturbed_rollout - normal_rollout
    regional = []
    for name, box in REGIONS.items():
        m = region_mask(box)
        regional.append(
            {
                "region": name,
                "delta_tp_mm_day": float(diff[:, 1][:, m].mean()),
                "delta_t2m_k": float(diff[:, 0][:, m].mean()),
                "delta_tcwv_kg_m2": float(diff[:, 5][:, m].mean()),
                "delta_u10_ms": float(diff[:, 2][:, m].mean()),
                "delta_v10_ms": float(diff[:, 3][:, m].mean()),
                "delta_heavy_rain_cell_days": int((perturbed_rollout[:, 1][:, m] >= 10.0).sum())
                - int((normal_rollout[:, 1][:, m] >= 10.0).sum()),
            }
        )
    regional_sorted = sorted(regional, key=lambda r: abs(r["delta_tp_mm_day"]), reverse=True)
    return {
        "disclaimer": "Model-derived indicative scenario differences, not operational disaster or crop forecasts.",
        "global_mean_delta_tp": float(diff[:, 1].mean()),
        "global_mean_delta_t2m": float(diff[:, 0].mean()),
        "global_mean_delta_tcwv": float(diff[:, 5].mean()),
        "regional_anomalies": regional_sorted,
    }
