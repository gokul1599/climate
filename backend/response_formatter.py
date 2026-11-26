"""Format simulation outputs into canonical JSON-serialisable responses."""
from __future__ import annotations

import numpy as np


def format_simulation_response(rollout: np.ndarray, inference_ms: float) -> dict:
    """Return nested JSON lists of finite float32 values and inference_ms."""
    return {
        "rollout": np.asarray(rollout, dtype=np.float32).tolist(),
        "inference_ms": float(round(inference_ms, 3)),
    }
