"""Unit tests for MonsoonResCNN model loading and forward pass."""
import numpy as np
import torch
from ml.model import MonsoonResCNN
from ml.inference import MonsoonPredictor


def test_model_forward_shape():
    m = MonsoonResCNN(hidden_channels=48, num_blocks=5)
    hist = torch.randn(2, 3, 6, 32, 36)
    st = torch.randn(2, 2, 32, 36)
    doy = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    out = m(hist, st, doy)
    assert out.shape == (2, 6, 32, 36)
    assert torch.isfinite(out).all()


def test_predictor_loads_checkpoint():
    pred = MonsoonPredictor()
    assert pred.model_name == "MonsoonVerse"
    assert pred.static_norm.shape == (2, 32, 36)
