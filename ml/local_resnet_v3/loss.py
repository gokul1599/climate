"""Modular loss components adapted from the reference LocalResNet-v3 objective."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_lat_weights_tensor(lat_deg: np.ndarray | None = None) -> torch.Tensor:
    """Cosine-latitude weights normalized to unit mean with shape [1, 1, NLAT, 1]."""
    if lat_deg is None:
        lat_deg = np.arange(6.5, 37.5 + 1e-9, 1.0)
    w = np.cos(np.deg2rad(lat_deg))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32).view(1, 1, len(lat_deg), 1)


def heavy_rain_weight_map(
    target_seq: torch.Tensor,
    rain_thresh_norm: float | None,
    heavy_rain_boost: float = 2.0,
) -> torch.Tensor:
    """Return [B, H, 1, Lat, Lon] precipitation weight multipliers (>=10 mm/day boost)."""
    b, h, _, lat, lon = target_seq.shape
    mult = torch.ones((b, h, 1, lat, lon), dtype=target_seq.dtype, device=target_seq.device)
    if rain_thresh_norm is not None and heavy_rain_boost > 1.0:
        heavy = (target_seq[:, :, 1:2] >= rain_thresh_norm).to(target_seq.dtype)
        mult = 1.0 + (float(heavy_rain_boost) - 1.0) * heavy
    return mult


def lat_weighted_huber_loss(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    lat_w: torch.Tensor,
    ch_w: torch.Tensor,
    huber_beta: float = 1.0,
    rain_thresh_norm: float | None = None,
    heavy_rain_boost: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (total_huber_with_heavy_rain, extra_loss_from_heavy_rain_boost)."""
    huber = F.smooth_l1_loss(pred_seq, target_seq, reduction="none", beta=huber_beta)
    base_w = lat_w.unsqueeze(1) * ch_w.unsqueeze(1)  # [1, 1, 6, Lat, 1]
    base_loss = (huber * base_w).mean()

    if rain_thresh_norm is None or heavy_rain_boost <= 1.0:
        return base_loss, torch.zeros_like(base_loss)

    tp_mult = heavy_rain_weight_map(target_seq, rain_thresh_norm, heavy_rain_boost)
    w_full = base_w.expand_as(pred_seq).clone()
    w_full[:, :, 1:2] = w_full[:, :, 1:2] * tp_mult
    boosted_loss = (huber * w_full).mean()
    extra = boosted_loss - base_loss
    return boosted_loss, extra


def spatial_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_w: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted L1 finite-difference loss in zonal (x) and meridional (y) directions."""
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    loss_x = (torch.abs(dx_p - dx_t) * lat_w).mean()
    loss_y = (torch.abs(dy_p - dy_t) * lat_w[..., 1:, :]).mean()
    return 0.5 * (loss_x + loss_y)


def spatial_variance_floor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_ratio: float = 0.65,
) -> torch.Tensor:
    """One-sided hinge penalizing spatial variance collapse below `min_ratio * target_var`."""
    vp = pred.flatten(-2).var(dim=-1, unbiased=False)
    vt = target.flatten(-2).var(dim=-1, unbiased=False)
    deficit = F.relu(float(min_ratio) * vt - vp) / torch.clamp(vt, min=1e-4)
    return deficit.mean()


class MonsoonRolloutLoss(nn.Module):
    """Combined Huber + heavy-rain weighting + spatial gradient + spatial variance-floor loss."""

    def __init__(
        self,
        channel_weights: Sequence[float] | None = None,
        grad_weight: float = 0.14,
        var_weight: float = 0.08,
        var_min_ratio: float = 0.65,
        rain_thresh_norm: float | None = None,
        heavy_rain_boost: float = 2.2,
        huber_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("lat_w", make_lat_weights_tensor())
        cw = list(channel_weights) if channel_weights is not None else [1.0, 1.4, 1.0, 1.0, 1.0, 1.0]
        self.register_buffer("ch_w", torch.tensor(cw, dtype=torch.float32).view(1, 6, 1, 1))
        self.grad_weight = float(grad_weight)
        self.var_weight = float(var_weight)
        self.var_min_ratio = float(var_min_ratio)
        self.rain_thresh_norm = rain_thresh_norm
        self.heavy_rain_boost = float(heavy_rain_boost)
        self.huber_beta = float(huber_beta)

    def forward_with_components(
        self, pred_seq: torch.Tensor, target_seq: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss and decoupled scalar log values for each term."""
        b, h, c, lat, lon = pred_seq.shape
        weighted_huber, heavy_extra = lat_weighted_huber_loss(
            pred_seq,
            target_seq,
            self.lat_w,
            self.ch_w,
            huber_beta=self.huber_beta,
            rain_thresh_norm=self.rain_thresh_norm,
            heavy_rain_boost=self.heavy_rain_boost,
        )
        base_huber = weighted_huber - heavy_extra
        total = weighted_huber

        if self.grad_weight > 0.0:
            g_raw = spatial_gradient_loss(
                pred_seq.reshape(b * h, c, lat, lon),
                target_seq.reshape(b * h, c, lat, lon),
                self.lat_w,
            )
            g_term = self.grad_weight * g_raw
            total = total + g_term
        else:
            g_raw = torch.zeros_like(total)
            g_term = torch.zeros_like(total)

        if self.var_weight > 0.0:
            v_raw = spatial_variance_floor_loss(pred_seq, target_seq, min_ratio=self.var_min_ratio)
            v_term = self.var_weight * v_raw
            total = total + v_term
        else:
            v_raw = torch.zeros_like(total)
            v_term = torch.zeros_like(total)

        comps = {
            "base_huber": float(base_huber.detach()),
            "heavy_rain_extra": float(heavy_extra.detach()),
            "grad_raw": float(g_raw.detach()),
            "grad_weighted": float(g_term.detach()),
            "var_floor_raw": float(v_raw.detach()),
            "var_floor_weighted": float(v_term.detach()),
            "total": float(total.detach()),
        }
        return total, comps

    def forward(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        total, _ = self.forward_with_components(pred_seq, target_seq)
        return total
