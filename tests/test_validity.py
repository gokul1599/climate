"""Physical validity tests on rollouts."""
import numpy as np
from monsoon_r1 import scoring as S
from ml.inference import MonsoonPredictor


def test_physical_validity():
    pred = MonsoonPredictor()
    base = np.array([298.0, 0.0, 1.0, -1.0, 101000.0, 30.0], dtype=np.float32)[:, None, None]
    init = np.broadcast_to(base, (3, 6, 32, 36)).copy()
    out = pred.predict_rollout(init, [0.0, 1.0], steps=10)
    violations = S.validity(out, 10)
    assert violations == []
