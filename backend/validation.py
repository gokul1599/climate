"""Request and output validation for the MonsoonVerse API."""
from __future__ import annotations

import base64
from typing import Any

import numpy as np


EXPECTED_INIT_SHAPE = (3, 6, 32, 36)
EXPECTED_PERT_SHAPE = (6, 32, 36)


def decode_array(x: Any, expect_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Decode nested JSON lists or {'b64', 'shape'} dict into float32 numpy array."""
    if isinstance(x, dict):
        if "b64" not in x or "shape" not in x:
            raise ValueError("Array object must contain 'b64' and 'shape'")
        shape = tuple(int(s) for s in x["shape"])
        if any(s < 0 or s > 100000 for s in shape) or int(np.prod(shape)) > 20_000_000:
            raise ValueError("Array shape too large")
        raw = base64.b64decode(x["b64"], validate=True)
        if len(raw) != int(np.prod(shape)) * 4:
            raise ValueError("b64 byte length does not match shape")
        arr = np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)

    if expect_shape is not None and arr.shape != expect_shape:
        raise ValueError(f"Expected shape {expect_shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Input array contains NaN or Inf")
    return arr


def validate_simulate_request(body: dict[str, Any]) -> tuple[np.ndarray, list[float], int, np.ndarray | None]:
    """Validate /simulate JSON payload and return (init_states, doy_sincos, steps, perturbation)."""
    if "init_states" not in body:
        raise ValueError("Missing 'init_states'")
    if "doy_sincos" not in body:
        raise ValueError("Missing 'doy_sincos'")
    if "steps" not in body:
        raise ValueError("Missing 'steps'")

    init_states = decode_array(body["init_states"], EXPECTED_INIT_SHAPE)
    doy = [float(v) for v in body["doy_sincos"]]
    if len(doy) != 2 or not all(np.isfinite(doy)):
        raise ValueError("'doy_sincos' must be 2 finite floats")

    steps = int(body["steps"])
    if steps < 1 or steps > 30:
        raise ValueError(f"'steps' must be between 1 and 30, got {steps}")

    pert_raw = body.get("perturbation")
    pert = None if pert_raw is None else decode_array(pert_raw, EXPECTED_PERT_SHAPE)
    return init_states, doy, steps, pert


def validate_rollout_output(rollout: np.ndarray, steps: int) -> np.ndarray:
    """Ensure output rollout is finite, physically valid, and exact shape [steps, 6, 32, 36]."""
    out = np.asarray(rollout, dtype=np.float32)
    expected = (steps, 6, 32, 36)
    if out.shape != expected:
        raise RuntimeError(f"Invalid rollout shape {out.shape}, expected {expected}")
    if not np.isfinite(out).all():
        raise RuntimeError("Rollout contains NaN or Inf")
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
    return out
