"""Combined impact indicator interface."""
from __future__ import annotations

import numpy as np
from impact.anomaly_analysis import compute_scenario_anomalies
from impact.regional_analysis import compute_regional_summary


def build_impact_report(
    normal_rollout: np.ndarray,
    perturbed_rollout: np.ndarray | None = None,
) -> dict:
    """Return model-derived regional indicators and optional scenario comparison."""
    report = {
        "note": "Model-derived indicators only; not guaranteed real-world outcomes.",
        "normal_regional_summary": compute_regional_summary(normal_rollout),
    }
    if perturbed_rollout is not None:
        report["perturbed_regional_summary"] = compute_regional_summary(perturbed_rollout)
        report["anomalies"] = compute_scenario_anomalies(normal_rollout, perturbed_rollout)
    return report
