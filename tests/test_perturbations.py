"""Tests for zero-perturbation consistency and physical shock responses."""
import numpy as np
from monsoon_r1 import contract as C, scoring as S
from ml.inference import MonsoonPredictor


def test_perturbations():
    pred = MonsoonPredictor()
    base = np.array([298.0, 5.0, 3.0, 2.0, 100600.0, 42.0], dtype=np.float32)[:, None, None]
    init = np.broadcast_to(base, (3, 6, 32, 36)).copy()
    doy = [0.2, 0.98]

    out_norm = pred.predict_rollout(init, doy, steps=10, perturbation=None)
    zeros = np.zeros((6, 32, 36), dtype=np.float32)
    out_zero = pred.predict_rollout(init, doy, steps=10, perturbation=zeros)
    assert S.null_consistent(out_norm, out_zero, pred.std)

    warm = S.make_shock("warm_ocean", init[-1], pred.std)
    out_warm = pred.predict_rollout(init, doy, steps=10, perturbation=warm)
    res_warm = S.causality_checks(out_norm, out_warm, C.BOX_BAY_OF_BENGAL, pred.std)
    assert res_warm["nonzero"] and res_warm["localized"] and res_warm["stable"]
