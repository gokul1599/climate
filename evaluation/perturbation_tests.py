"""Perturbation causality and zero-shock consistency tests."""
from __future__ import annotations

import numpy as np
from monsoon_r1 import contract as C, scoring as S


def run_perturbation_suite(
    predict_fn,
    init_states: np.ndarray,
    doy_sincos: list[float],
    std: np.ndarray,
    steps: int = 10,
) -> dict:
    """Run normal, zero-perturbation, warm_ocean, and monsoon_stall checks."""
    base = predict_fn(init_states, doy_sincos, steps=steps, perturbation=None)
    zeros = np.zeros((C.NC, C.NLAT, C.NLON), dtype=np.float32)
    zero_out = predict_fn(init_states, doy_sincos, steps=steps, perturbation=zeros)
    null_ok = S.null_consistent(base, zero_out, std)
    null_max_diff = float((np.abs(base - zero_out) / std[None, :, None, None]).max())

    shock_warm = S.make_shock("warm_ocean", init_states[-1], std)
    out_warm = predict_fn(init_states, doy_sincos, steps=steps, perturbation=shock_warm)
    res_warm = S.causality_checks(base, out_warm, C.BOX_BAY_OF_BENGAL, std)

    shock_stall = S.make_shock("monsoon_stall", init_states[-1], std)
    out_stall = predict_fn(init_states, doy_sincos, steps=steps, perturbation=shock_stall)
    res_stall = S.causality_checks(base, out_stall, C.BOX_ARABIAN_SEA, std)

    pts = S.causality_points([res_warm, res_stall], null_ok=null_ok)
    return {
        "null_consistent": bool(null_ok),
        "null_max_std_diff": null_max_diff,
        "warm_ocean": res_warm,
        "monsoon_stall": res_stall,
        "causality_points_out_of_15": float(pts),
    }
