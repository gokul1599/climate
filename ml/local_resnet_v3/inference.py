"""Pure-CPU deterministic inference adapter for MonsoonVerse-LocalResNet-v3 (repo_scan safe)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ml.local_resnet_v3.model import MonsoonResNet

NLAT, NLON, NC, T_IN = 32, 36, 6, 3
DTHETA = 2.0 * math.pi / 365.0
COS_D = math.cos(DTHETA)
SIN_D = math.sin(DTHETA)


def make_coord_grid() -> np.ndarray:
    """Normalized [-1, 1] lat/lon coordinate channels [2, 32, 36]."""
    lat = np.linspace(-1.0, 1.0, NLAT, dtype=np.float32)
    lon = np.linspace(-1.0, 1.0, NLON, dtype=np.float32)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    return np.stack([lat_g, lon_g], axis=0).astype(np.float32)


def advance_doy_sincos(sin_v: float, cos_v: float) -> tuple[float, float]:
    """Rotate [sin(theta), cos(theta)] forward by +1 day (2*pi/365)."""
    return (sin_v * COS_D + cos_v * SIN_D, cos_v * COS_D - sin_v * SIN_D)


class LocalResNetV3Predictor:
    """Loads isolated LocalResNet-v3 candidate artifacts once at startup."""

    def __init__(self, artifacts_dir: str | Path = "artifacts/local_resnet_v3") -> None:
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        adir = Path(artifacts_dir)
        if not adir.is_absolute() and not adir.is_dir():
            adir = Path(__file__).resolve().parents[2] / artifacts_dir

        norm_path = adir / "norm_stats.json"
        static_path = adir / "static_fields.npz"
        ckpt_path = adir / "model.pt"
        for req_p in (norm_path, static_path, ckpt_path):
            if not req_p.is_file():
                raise FileNotFoundError(f"Missing LocalResNet-v3 artifact: {req_p}")

        norm_cfg = json.loads(norm_path.read_text(encoding="utf-8"))
        self.mean = np.asarray(norm_cfg["mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(norm_cfg["std"], dtype=np.float32), 1e-6)
        self.static_mean = np.asarray(norm_cfg["static_mean"], dtype=np.float32)
        self.static_std = np.maximum(np.asarray(norm_cfg["static_std"], dtype=np.float32), 1e-6)
        self.log1p_tp = bool(norm_cfg.get("log1p_tp", False))

        self.tp_min_norm = float((0.0 - self.mean[1]) / self.std[1])
        self.t2m_min_norm = float((233.15 - self.mean[0]) / self.std[0])
        self.t2m_max_norm = float((333.15 - self.mean[0]) / self.std[0])

        static_raw = np.load(static_path)["static"].astype(np.float32)
        static_norm = (static_raw - self.static_mean[:, None, None]) / self.static_std[:, None, None]
        self.static_t = torch.from_numpy(static_norm[None])
        self.coord_t = torch.from_numpy(make_coord_grid()[None])

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = ckpt.get("model_config", {})
        self.model = MonsoonResNet(
            t_in=int(cfg.get("t_in", T_IN)),
            n_dyn=int(cfg.get("n_dyn", NC)),
            n_static=int(cfg.get("n_static", 2)),
            n_time=int(cfg.get("n_time", 2)),
            n_coord=int(cfg.get("n_coord", 2)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_blocks=int(cfg.get("num_blocks", 4)),
            dilations=cfg.get("dilations", [1, 2, 2, 1]),
            init_scale=float(cfg.get("init_scale", 0.25)),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.model_name = str(ckpt.get("model_name", "MonsoonVerse-LocalResNet-v3"))
        self.clim_norm_366 = None  # Explicitly not used by LocalResNet-v3

    def _normalize_phys(self, phys: np.ndarray) -> np.ndarray:
        arr = np.asarray(phys, dtype=np.float32).copy()
        if self.log1p_tp:
            arr[..., 1, :, :] = np.log1p(np.maximum(arr[..., 1, :, :], 0.0))
        view = (1,) * (arr.ndim - 3) + (NC, 1, 1)
        return ((arr - self.mean.reshape(view)) / self.std.reshape(view)).astype(np.float32)

    def _denormalize_norm(self, norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(norm, dtype=np.float32)
        view = (1,) * (arr.ndim - 3) + (NC, 1, 1)
        out = arr * self.std.reshape(view) + self.mean.reshape(view)
        if self.log1p_tp:
            out[..., 1, :, :] = np.expm1(np.clip(out[..., 1, :, :], 0.0, 6.5))
        else:
            out[..., 1, :, :] = np.maximum(out[..., 1, :, :], 0.0)
        out[..., 0, :, :] = np.clip(out[..., 0, :, :], 233.15, 333.15)
        return out.astype(np.float32)

    @torch.inference_mode()
    def predict_rollout(
        self,
        init_states: np.ndarray,
        doy_sincos: Sequence[float],
        steps: int,
        perturbation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run deterministic physical-unit autoregressive rollout [steps, 6, 32, 36]."""
        hist_phys = np.asarray(init_states, dtype=np.float32).copy()
        if hist_phys.shape != (T_IN, NC, NLAT, NLON):
            raise ValueError(f"init_states must have shape {(T_IN, NC, NLAT, NLON)}, got {hist_phys.shape}")
        if not np.isfinite(hist_phys).all():
            raise ValueError("init_states contains NaN or Inf")
        if float(hist_phys[:, 1].min()) < 0.0:
            raise ValueError("init_states contains negative precipitation")

        if doy_sincos is None or isinstance(doy_sincos, (str, bytes)):
            raise ValueError("doy_sincos must be a finite 2-element sequence")
        doy_arr = np.asarray(doy_sincos, dtype=np.float64)
        if doy_arr.shape != (2,) or not np.isfinite(doy_arr).all():
            raise ValueError(f"Invalid doy_sincos shape: {doy_arr.shape}")

        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
            raise ValueError(f"steps must be an int, got {type(steps).__name__}")
        n_steps = int(steps)
        if n_steps < 1 or n_steps > 14:
            raise ValueError(f"steps must be in [1, 14], got {n_steps}")

        if perturbation is not None:
            pert_arr = np.asarray(perturbation, dtype=np.float32)
            if pert_arr.shape != (NC, NLAT, NLON):
                raise ValueError(f"perturbation must have shape {(NC, NLAT, NLON)}, got {pert_arr.shape}")
            if not np.isfinite(pert_arr).all():
                raise ValueError("perturbation contains NaN or Inf")
            hist_phys[-1] = hist_phys[-1] + pert_arr
            hist_phys[-1, 1] = np.maximum(hist_phys[-1, 1], 0.0)

        hist_t = torch.from_numpy(self._normalize_phys(hist_phys)[None])
        s_val, c_val = float(doy_arr[0]), float(doy_arr[1])
        preds_norm = []

        for _ in range(n_steps):
            doy_t = torch.tensor([[s_val, c_val]], dtype=torch.float32)
            nxt = self.model(hist_t, self.static_t, doy_t, self.coord_t)
            nxt[:, 1] = torch.clamp(nxt[:, 1], min=self.tp_min_norm)
            nxt[:, 0] = torch.clamp(nxt[:, 0], min=self.t2m_min_norm, max=self.t2m_max_norm)
            preds_norm.append(nxt[0].numpy().copy())
            hist_t = torch.cat([hist_t[:, 1:], nxt.unsqueeze(1)], dim=1)
            s_val, c_val = advance_doy_sincos(s_val, c_val)

        rollout_phys = self._denormalize_norm(np.stack(preds_norm, axis=0))
        if not np.isfinite(rollout_phys).all():
            raise RuntimeError("Model produced non-finite rollout values")
        return rollout_phys
