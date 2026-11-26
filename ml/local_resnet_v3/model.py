"""MonsoonVerse-LocalResNet-v3: pure-local physics-aware residual CNN (313,030 parameters).

Uses replicate padding, dilated local 3x3 convolutions, SiLU activations, and
learnable per-channel residual scaling (init=0.25), intentionally omitting
global pooling, spatial normalization, and future climatology channels.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class LocalResBlock(nn.Module):
    """Local 3x3 residual block with configurable dilation and learnable residual scale."""

    def __init__(self, channels: int, dilation: int = 1, init_scale: float = 0.25) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=self.dilation,
            dilation=self.dilation,
            padding_mode="replicate",
        )
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            padding_mode="replicate",
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.act(self.conv1(x)))
        return x + self.scale * out


class MonsoonResNet(nn.Module):
    """Predicts next-day normalized residual delta from 24 local/seasonal input channels."""

    def __init__(
        self,
        t_in: int = 3,
        n_dyn: int = 6,
        n_static: int = 2,
        n_time: int = 2,
        n_coord: int = 2,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dilations: Sequence[int] | None = (1, 2, 2, 1),
        init_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.t_in = int(t_in)
        self.n_dyn = int(n_dyn)
        self.n_static = int(n_static)
        self.n_time = int(n_time)
        self.n_coord = int(n_coord)
        self.hidden_dim = int(hidden_dim)

        in_channels = self.t_in * self.n_dyn + self.n_static + self.n_time + self.n_coord
        if dilations is None:
            dil_list = [1] * int(num_blocks)
        else:
            dil_list = [int(d) for d in dilations]
        if len(dil_list) != int(num_blocks):
            raise ValueError(f"Expected {num_blocks} dilations, got {len(dil_list)}")
        self.dilations = dil_list

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[LocalResBlock(self.hidden_dim, dilation=d, init_scale=init_scale) for d in dil_list]
        )
        self.head = nn.Conv2d(self.hidden_dim, self.n_dyn, kernel_size=3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        hist_norm: torch.Tensor,     # [B, 3, 6, 32, 36]
        static_norm: torch.Tensor,   # [B, 2, 32, 36]
        doy_sincos: torch.Tensor,    # [B, 2]
        coord_grid: torch.Tensor,    # [B, 2, 32, 36]
    ) -> torch.Tensor:
        b, _, _, h, w = hist_norm.shape
        dyn_flat = hist_norm.reshape(b, self.t_in * self.n_dyn, h, w)
        doy_map = doy_sincos[:, :, None, None].expand(b, self.n_time, h, w)
        x = torch.cat([dyn_flat, static_norm, doy_map, coord_grid], dim=1)
        feat = self.blocks(self.stem(x))
        delta = self.head(feat)
        return hist_norm[:, -1] + delta
