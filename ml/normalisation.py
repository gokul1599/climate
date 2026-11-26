"""Normalisation helpers for physical <-> normalised state conversions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_norm_stats(stats_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load per-channel mean [6] and std [6] from stats.json."""
    with Path(stats_path).open("r", encoding="utf-8") as f:
        s = json.load(f)
    mean = np.asarray(s["mean"], dtype=np.float32)
    std = np.asarray(s["std"], dtype=np.float32)
    return mean, std


def normalise_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    m = mean.reshape((1,) * (x.ndim - 3) + (6, 1, 1))
    s = std.reshape((1,) * (x.ndim - 3) + (6, 1, 1))
    return ((x - m) / s).astype(np.float32)


def denormalise_np(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    m = mean.reshape((1,) * (x_norm.ndim - 3) + (6, 1, 1))
    s = std.reshape((1,) * (x_norm.ndim - 3) + (6, 1, 1))
    return (x_norm * s + m).astype(np.float32)


def normalise_torch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    shape = (1,) * (x.ndim - 3) + (6, 1, 1)
    return (x - mean.view(shape)) / std.view(shape)


def denormalise_torch(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    shape = (1,) * (x_norm.ndim - 3) + (6, 1, 1)
    return x_norm * std.view(shape) + mean.view(shape)
