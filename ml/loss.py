"""Latitude-weighted multi-step loss functions for MonsoonVerse."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_lat_weights(nlat: int = 32) -> torch.Tensor:
    lat = np.arange(6.5, 37.5 + 1e-9, 1.0, dtype=np.float32)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return torch.from_numpy(w).view(1, 1, nlat, 1)


class ClimateRolloutLoss(nn.Module):
    """Latitude- and channel-weighted SmoothL1 + MSE loss for multi-step rollout."""

    def __init__(
        self,
        channel_weights: tuple[float, ...] = (1.0, 1.25, 1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.register_buffer("lat_w", make_lat_weights())
        cw = torch.tensor(channel_weights, dtype=torch.float32).view(1, 6, 1, 1)
        self.register_buffer("chan_w", cw)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, H, 6, 32, 36] in normalised units."""
        b, h, c, lat, lon = pred.shape
        p = pred.reshape(b * h, c, lat, lon)
        t = target.reshape(b * h, c, lat, lon)
        w = self.lat_w * self.chan_w
        huber = F.smooth_l1_loss(p, t, reduction="none", beta=0.5)
        mse = (p - t) ** 2
        loss = (0.7 * huber + 0.3 * mse) * w
        return loss.mean()
