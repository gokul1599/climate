"""Deterministic autoregressive rollout with day-of-year progression."""
from __future__ import annotations

import math
import numpy as np
import torch

from ml.normalisation import denormalise_np, normalise_np


def advance_doy_sincos(doy_sincos: list[float] | np.ndarray, step_offset: int) -> np.ndarray:
    """Advance [sin(2*pi*doy/365), cos(2*pi*doy/365)] by `step_offset` days using rotation."""
    s0 = float(doy_sincos[0])
    c0 = float(doy_sincos[1])
    dtheta = 2.0 * math.pi * step_offset / 365.0
    cd = math.cos(dtheta)
    sd = math.sin(dtheta)
    s_new = s0 * cd + c0 * sd
    c_new = c0 * cd - s0 * sd
    return np.array([s_new, c_new], dtype=np.float32)


@torch.no_grad()
def autoregressive_rollout(
    model: torch.nn.Module,
    init_states_phys: np.ndarray,
    doy_sincos: list[float] | np.ndarray,
    steps: int,
    mean: np.ndarray,
    std: np.ndarray,
    static_norm: np.ndarray,
    perturbation_phys: np.ndarray | None = None,
) -> np.ndarray:
    """Run deterministic `steps`-day rollout in physical units.

    init_states_phys: [3, 6, 32, 36] physical units (oldest first).
    doy_sincos: [sin, cos] of init_states_phys[-1].
    perturbation_phys: None or [6, 32, 36] physical delta added to init_states_phys[-1].
    Returns: [steps, 6, 32, 36] float32 in physical units.
    """
    model.eval()
    states = np.asarray(init_states_phys, dtype=np.float32).copy()
    if perturbation_phys is not None:
        states[-1] = states[-1] + np.asarray(perturbation_phys, dtype=np.float32)
        states[-1, 1] = np.maximum(states[-1, 1], 0.0)

    hist_norm = normalise_np(states, mean, std)  # [3, 6, 32, 36]
    st_t = torch.from_numpy(static_norm[None, ...])  # [1, 2, 32, 36]

    rollout_phys = np.zeros((steps, 6, 32, 36), dtype=np.float32)
    cur_hist = hist_norm.copy()

    for k in range(steps):
        doy_k = advance_doy_sincos(doy_sincos, k)
        hist_t = torch.from_numpy(cur_hist[None, ...])  # [1, 3, 6, 32, 36]
        doy_t = torch.from_numpy(doy_k[None, ...])      # [1, 2]

        pred_norm = model(hist_t, st_t, doy_t).squeeze(0).numpy()  # [6, 32, 36]
        pred_phys = denormalise_np(pred_norm, mean, std)
        # Physical validity guard: precipitation cannot be negative
        pred_phys[1] = np.maximum(pred_phys[1], 0.0)
        rollout_phys[k] = pred_phys

        # Re-normalise clamped physical state for next autoregressive step
        next_norm = normalise_np(pred_phys, mean, std)
        cur_hist = np.concatenate([cur_hist[1:], next_norm[None, ...]], axis=0)

    return rollout_phys
