"""Tests for deterministic 10-day autoregressive rollout and DOY progression."""
import numpy as np
from ml.inference import MonsoonPredictor
from ml.rollout import advance_doy_sincos


def _dummy_init():
    base = np.array([298.0, 4.0, 2.0, 1.0, 100800.0, 38.0], dtype=np.float32)[:, None, None]
    return np.broadcast_to(base, (3, 6, 32, 36)).copy()


def test_doy_advances():
    s0 = [0.0, 1.0]
    s1 = advance_doy_sincos(s0, 1)
    assert not np.allclose(s0, s1)
    assert np.isclose(np.linalg.norm(s1), 1.0, atol=1e-5)


def test_rollout_deterministic_and_shape():
    pred = MonsoonPredictor()
    init = _dummy_init()
    out1 = pred.predict_rollout(init, [0.5, 0.866], steps=10)
    out2 = pred.predict_rollout(init, [0.5, 0.866], steps=10)
    assert out1.shape == (10, 6, 32, 36)
    assert np.allclose(out1, out2)
