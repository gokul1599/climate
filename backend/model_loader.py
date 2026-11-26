"""Singleton model loader for offline CPU inference."""
from __future__ import annotations

from pathlib import Path
from ml.inference import MonsoonPredictor


_PREDICTOR: MonsoonPredictor | None = None


def load_predictor() -> MonsoonPredictor:
    """Load model weights, stats.json, and static_fields.npz once at startup."""
    global _PREDICTOR
    if _PREDICTOR is None:
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
        _PREDICTOR = MonsoonPredictor(artifacts_dir=artifacts_dir)
    return _PREDICTOR
