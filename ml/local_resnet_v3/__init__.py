"""Isolated challenger model package: MonsoonVerse-LocalResNet-v3 (inference-safe exports)."""
from __future__ import annotations

from ml.local_resnet_v3.inference import LocalResNetV3Predictor, advance_doy_sincos, make_coord_grid
from ml.local_resnet_v3.model import LocalResBlock, MonsoonResNet

__all__ = [
    "LocalResBlock",
    "MonsoonResNet",
    "LocalResNetV3Predictor",
    "make_coord_grid",
    "advance_doy_sincos",
]
