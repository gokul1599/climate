"""Compact CPU-efficient Residual CNN for MonsoonVerse spatiotemporal forecasting."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """3x3 Residual convolution block with replicate padding and GroupNorm."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(x + out)


class MonsoonResCNN(nn.Module):
    """Residual CNN predicting next-day normalised climate state from 3-day history + static + DOY.

    Input:
      hist_norm: [B, 3, 6, 32, 36]
      static_norm: [B, 2, 32, 36]
      doy_sincos: [B, 2]
    Output:
      next_state_norm: [B, 6, 32, 36] = latest_state_norm + predicted_delta
    """

    def __init__(self, hidden_channels: int = 48, num_blocks: int = 5) -> None:
        super().__init__()
        in_channels = 3 * 6 + 2 + 2  # 22
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden_channels, 6, kernel_size=3, padding=1, padding_mode="replicate")

        # Start near identity (zero-init head) so initial predictions are close to persistence
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        hist_norm: torch.Tensor,
        static_norm: torch.Tensor,
        doy_sincos: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c, h, w = hist_norm.shape
        hist_flat = hist_norm.reshape(b, t * c, h, w)
        doy_grid = doy_sincos.view(b, 2, 1, 1).expand(b, 2, h, w)
        x = torch.cat([hist_flat, static_norm, doy_grid], dim=1)
        feat = self.blocks(self.stem(x))
        delta = self.head(feat)
        return hist_norm[:, -1] + delta
