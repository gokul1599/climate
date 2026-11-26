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


def doy_index_from_sincos(doy_sincos: list[float] | np.ndarray) -> int:
    """Map [sin, cos] to 0..364 day-of-year index."""
    ang = float(math.atan2(float(doy_sincos[0]), float(doy_sincos[1])))
    if ang < 0.0:
        ang += 2.0 * math.pi
    return int(round(ang * 365.0 / (2.0 * math.pi))) % 365


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
    clim_norm_366: np.ndarray | None = None,
) -> np.ndarray:
    """Run deterministic `steps`-day rollout in physical units."""
    model.eval()
    states = np.asarray(init_states_phys, dtype=np.float32).copy()
    if perturbation_phys is not None:
        states[-1] = states[-1] + np.asarray(perturbation_phys, dtype=np.float32)
        states[-1, 1] = np.maximum(states[-1, 1], 0.0)

    hist_norm = normalise_np(states, mean, std)
    st_t = torch.from_numpy(static_norm[None, ...])
    doy_idx0 = doy_index_from_sincos(doy_sincos)

    rollout_phys = np.zeros((steps, 6, 32, 36), dtype=np.float32)
    cur_hist = hist_norm.copy()

    for k in range(steps):
        doy_k = advance_doy_sincos(doy_sincos, k)
        hist_t = torch.from_numpy(cur_hist[None, ...])
        doy_t = torch.from_numpy(doy_k[None, ...])
        clim_t = None
        if getattr(model, "use_clim_cond", False) and clim_norm_366 is not None:
            cidx = (doy_idx0 + k + 1) % 365
            clim_t = torch.from_numpy(clim_norm_366[cidx : cidx + 1])

        pred_norm = model(hist_t, st_t, doy_t, clim_t).squeeze(0).numpy()
        pred_phys = denormalise_np(pred_norm, mean, std)
        pred_phys[1] = np.maximum(pred_phys[1], 0.0)
        rollout_phys[k] = pred_phys

        next_norm = normalise_np(pred_phys, mean, std)
        cur_hist = np.concatenate([cur_hist[1:], next_norm[None, ...]], axis=0)

    return rollout_phys
