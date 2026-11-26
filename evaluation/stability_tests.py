"""Physical validity and 10-day rollout stability checks."""
from __future__ import annotations

import numpy as np
from monsoon_r1 import scoring as S


def check_rollout_validity(rollout: np.ndarray, steps: int = 10) -> dict:
    """Check finite values, non-negative rainfall, temperature bounds, and channel ranges."""
    violations = S.validity(rollout, steps)
    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "tp_min": float(rollout[:, 1].min()),
        "tp_max": float(rollout[:, 1].max()),
        "t2m_min_c": float(rollout[:, 0].min() - 273.15),
        "t2m_max_c": float(rollout[:, 0].max() - 273.15),
        "all_finite": bool(np.isfinite(rollout).all()),
    }
