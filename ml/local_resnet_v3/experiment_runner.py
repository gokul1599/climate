"""Inner-chronological experiment suite (2005-2017 train -> 2018-2019 inner-val) for Operation 70/70.

Prevents Protocol A (2020-2021) validation-selection leakage while screening loss, curriculum,
dilations, and wind-vector consistency hypotheses fairly via unmodified monsoon_r1.scoring.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.score_optimizer_70 import evaluate_exact_70
from ml.local_resnet_v3.dataset import (
    DEFAULT_NC_PATH,
    MonsoonWindowDataset,
    Normalizer,
    load_climate_archive,
)
from ml.local_resnet_v3.inference import LocalResNetV3Predictor, make_coord_grid
from ml.local_resnet_v3.loss import lat_weighted_huber_loss, make_lat_weights_tensor, spatial_gradient_loss, spatial_variance_floor_loss
from ml.local_resnet_v3.model import MonsoonResNet
from ml.local_resnet_v3.train import atomic_text_write, atomic_torch_save, build_train_climatology_366, train_stage


class EnhancedRolloutLoss(nn.Module):
    """Scientifically calibrated rollout loss combining Huber+MSE, channel balance, lead weighting, vorticity/div, gradient & variance floor."""

    def __init__(
        self,
        channel_weights=(1.28, 1.15, 1.10, 1.22, 1.05, 1.15),
        mse_mix: float = 0.35,
        grad_weight: float = 0.14,
        var_weight: float = 0.08,
        var_min_ratio: float = 0.65,
        vort_div_weight: float = 0.0,
        lead_weights: tuple[float, ...] | None = None,
        rain_thresh_norm: float | None = None,
        heavy_rain_boost: float = 1.65,
        huber_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("lat_w", make_lat_weights_tensor())
        self.register_buffer("ch_w", torch.tensor(list(channel_weights), dtype=torch.float32).view(1, 6, 1, 1))
        self.mse_mix = float(mse_mix)
        self.grad_weight = float(grad_weight)
        self.var_weight = float(var_weight)
        self.var_min_ratio = float(var_min_ratio)
        self.vort_div_weight = float(vort_div_weight)
        if lead_weights is not None:
            lw = torch.tensor(list(lead_weights), dtype=torch.float32)
            lw = lw / lw.mean()
            self.register_buffer("lead_w", lw.view(1, -1, 1, 1, 1))
        else:
            self.lead_w = None
        self.rain_thresh_norm = rain_thresh_norm
        self.heavy_rain_boost = float(heavy_rain_boost)
        self.huber_beta = float(huber_beta)

    def forward_with_components(self, pred_seq: torch.Tensor, target_seq: torch.Tensor):
        b, h, c, lat, lon = pred_seq.shape
        base_w = self.lat_w.unsqueeze(1) * self.ch_w.unsqueeze(1)
        if self.lead_w is not None:
            base_w = base_w * self.lead_w[:, :h]

        huber_raw = F.smooth_l1_loss(pred_seq, target_seq, reduction="none", beta=self.huber_beta)
        mse_raw = (pred_seq - target_seq) ** 2
        point_loss = (1.0 - self.mse_mix) * huber_raw + self.mse_mix * mse_raw

        if self.rain_thresh_norm is not None and self.heavy_rain_boost > 1.0:
            heavy = (target_seq[:, :, 1:2] >= self.rain_thresh_norm).to(pred_seq.dtype)
            w_full = base_w.expand_as(pred_seq).clone()
            w_full[:, :, 1:2] = w_full[:, :, 1:2] * (1.0 + (self.heavy_rain_boost - 1.0) * heavy)
            base_term = (point_loss * w_full).mean()
        else:
            base_term = (point_loss * base_w).mean()

        total = base_term
        g_val = 0.0
        if self.grad_weight > 0.0:
            g_loss = spatial_gradient_loss(pred_seq.reshape(b * h, c, lat, lon), target_seq.reshape(b * h, c, lat, lon), self.lat_w)
            g_term = self.grad_weight * g_loss
            total = total + g_term
            g_val = float(g_term.detach())

        v_val = 0.0
        if self.var_weight > 0.0:
            v_loss = spatial_variance_floor_loss(pred_seq, target_seq, min_ratio=self.var_min_ratio)
            v_term = self.var_weight * v_loss
            total = total + v_term
            v_val = float(v_term.detach())

        vd_val = 0.0
        if self.vort_div_weight > 0.0:
            up, vp = pred_seq[:, :, 2], pred_seq[:, :, 3]
            ut, vt = target_seq[:, :, 2], target_seq[:, :, 3]
            div_p = (up[:, :, 1:, 1:] - up[:, :, 1:, :-1]) + (vp[:, :, 1:, 1:] - vp[:, :, :-1, 1:])
            div_t = (ut[:, :, 1:, 1:] - ut[:, :, 1:, :-1]) + (vt[:, :, 1:, 1:] - vt[:, :, :-1, 1:])
            vort_p = (vp[:, :, 1:, 1:] - vp[:, :, 1:, :-1]) - (up[:, :, 1:, 1:] - up[:, :, :-1, 1:])
            vort_t = (vt[:, :, 1:, 1:] - vt[:, :, 1:, :-1]) - (ut[:, :, 1:, 1:] - ut[:, :, :-1, 1:])
            lw_vd = self.lat_w[:, :, 1:, :].squeeze(1)
            vd_loss = 0.5 * ((torch.abs(div_p - div_t) * lw_vd).mean() + (torch.abs(vort_p - vort_t) * lw_vd).mean())
            vd_term = self.vort_div_weight * vd_loss
            total = total + vd_term
            vd_val = float(vd_term.detach())

        return total, {
            "base_huber": float(base_term.detach()),
            "heavy_rain_extra": float(vd_val),
            "grad_weighted": g_val,
            "var_floor_weighted": v_val,
            "total": float(total.detach()),
        }

    def forward(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        t, _ = self.forward_with_components(pred_seq, target_seq)
        return t


def run_one_experiment(
    exp_id: str,
    hypothesis: str,
    archive,
    tr_start: dt.date,
    tr_end: dt.date,
    va_start: dt.date,
    va_end: dt.date,
    loss_cfg: dict,
    dilations: list[int] = (1, 2, 2, 1),
    hidden_dim: int = 64,
    epochs_s1: int = 8,
    epochs_s2: int = 5,
    s1_stride: int = 2,
    s2_stride: int = 4,
    lr: float = 1.5e-3,
    s2_lr_factor: float = 0.3,
    seed: int = 42,
    threads: int = 6,
) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_dir = ROOT_DIR / "artifacts" / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    tr_idx = np.where((archive.dates >= tr_start) & (archive.dates <= tr_end))[0]
    va_idx = np.where((archive.dates >= va_start) & (archive.dates <= va_end))[0]
    train_phys, train_dates = archive.data[tr_idx], archive.dates[tr_idx]
    val_phys, val_dates = archive.data[va_idx], archive.dates[va_idx]

    np.savez_compressed(exp_dir / "static_fields.npz", static=archive.static.astype(np.float32))
    clim_366 = build_train_climatology_366(train_phys, archive.dates_int[tr_idx], exp_dir / "climatology.npy")

    norm = Normalizer.fit(train_phys, archive.static, log1p_tp=False)
    norm.save(exp_dir / "norm_stats.json")
    train_norm = norm.normalize(train_phys)
    val_norm = norm.normalize(val_phys)

    static_t = torch.from_numpy(norm.normalize_static(archive.static)[None])
    coord_t = torch.from_numpy(make_coord_grid()[None])
    tp_min_norm = float((0.0 - norm.mean[1]) / norm.std[1])
    t2m_min_norm = float((233.15 - norm.mean[0]) / norm.std[0])
    t2m_max_norm = float((333.15 - norm.mean[0]) / norm.std[0])

    model_cfg = {
        "t_in": 3, "n_dyn": 6, "n_static": 2, "n_time": 2, "n_coord": 2,
        "hidden_dim": int(hidden_dim), "num_blocks": len(dilations), "dilations": list(dilations), "init_scale": 0.25,
    }
    model = MonsoonResNet(**model_cfg)
    rain_thresh_norm = float((10.0 - norm.mean[1]) / norm.std[1])
    loss_fn = EnhancedRolloutLoss(rain_thresh_norm=rain_thresh_norm, **loss_cfg)

    ckpt_path = exp_dir / "model.pt"
    ds_tr1 = MonsoonWindowDataset(train_phys, train_norm, train_dates, horizon=1, stride=s1_stride)
    ds_va1 = MonsoonWindowDataset(val_phys, val_norm, val_dates, horizon=1, stride=2)
    dl_tr1 = DataLoader(ds_tr1, batch_size=64, shuffle=True)
    dl_va1 = DataLoader(ds_va1, batch_size=64, shuffle=False)
    s1_res = train_stage(
        model, dl_tr1, dl_va1, loss_fn, static_t, coord_t,
        horizon=1, epochs=epochs_s1, lr=lr, weight_decay=1e-4, grad_clip=1.0,
        tp_min_norm=tp_min_norm, t2m_min_norm=t2m_min_norm, t2m_max_norm=t2m_max_norm,
        stage_name=f"{exp_id}-S1", ckpt_dest=ckpt_path, model_name=exp_id,
        model_cfg=model_cfg, device=torch.device("cpu"), grad_accum_steps=2,
    )
    del ds_tr1, ds_va1, dl_tr1, dl_va1
    gc.collect()

    s2_res = None
    if epochs_s2 > 0:
        ds_tr2 = MonsoonWindowDataset(train_phys, train_norm, train_dates, horizon=8, stride=s2_stride)
        ds_va2 = MonsoonWindowDataset(val_phys, val_norm, val_dates, horizon=8, stride=4)
        dl_tr2 = DataLoader(ds_tr2, batch_size=32, shuffle=True)
        dl_va2 = DataLoader(ds_va2, batch_size=32, shuffle=False)
        s2_res = train_stage(
            model, dl_tr2, dl_va2, loss_fn, static_t, coord_t,
            horizon=8, epochs=epochs_s2, lr=lr * s2_lr_factor, weight_decay=1e-4, grad_clip=1.0,
            tp_min_norm=tp_min_norm, t2m_min_norm=t2m_min_norm, t2m_max_norm=t2m_max_norm,
            stage_name=f"{exp_id}-S2", ckpt_dest=ckpt_path, model_name=exp_id,
            model_cfg=model_cfg, device=torch.device("cpu"), grad_accum_steps=4,
        )
        del ds_tr2, ds_va2, dl_tr2, dl_va2
        gc.collect()

    pred = LocalResNetV3Predictor(artifacts_dir=exp_dir)
    eval_inner = evaluate_exact_70(
        pred, archive, tr_start, tr_end, va_start, va_end,
        stride=14, shock_step=4, clim_366=clim_366, stats_std=norm.std,
    )

    rec = {
        "exp_id": exp_id,
        "hypothesis": hypothesis,
        "seed": seed,
        "model_cfg": model_cfg,
        "loss_cfg": loss_cfg,
        "params": sum(int(p.numel()) for p in model.parameters()),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "ckpt_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "s1_best_val": s1_res["best_val_loss"],
        "s2_best_val": s2_res["best_val_loss"] if s2_res else None,
        "eval": eval_inner,
    }
    atomic_text_write(json.dumps(rec, indent=2), exp_dir / "manifest_and_eval.json")
    print(
        f"=== [{exp_id}] SCORE={eval_inner['total_pts_70']:.2f}/70 | "
        f"RMSE={eval_inner['rmse_pts_30']:.2f} (ss={eval_inner['mean_ss']:.5f}) "
        f"CSI={eval_inner['csi_pts_15']:.2f} AB={eval_inner['anti_blur_pts_10']:.2f} "
        f"CAUS={eval_inner['causality_pts_15']:.2f} (warm={eval_inner['warm_prop_count']}/13, stall={eval_inner['stall_prop_count']}/13) ===",
        flush=True,
    )
    return rec


def main() -> None:
    archive = load_climate_archive(DEFAULT_NC_PATH)
    tr_s, tr_e = dt.date(2005, 1, 1), dt.date(2017, 12, 31)
    va_s, va_e = dt.date(2018, 1, 1), dt.date(2019, 12, 31)

    experiments = [
        (
            "inner_e0_ref_loss",
            "Inner-split baseline reproducing LocalResNet-v3 loss (Huber, cw=[1,1.4,1,1,1,1], boost=2.2, grad=0.14, var=0.08)",
            {"channel_weights": (1.0, 1.4, 1.0, 1.0, 1.0, 1.0), "mse_mix": 0.0, "grad_weight": 0.14, "var_weight": 0.08, "var_min_ratio": 0.65, "heavy_rain_boost": 2.2, "vort_div_weight": 0.0},
            [1, 2, 2, 1],
        ),
        (
            "inner_e1_mse_balanced",
            "RMSE Optimization: Mix 0.30 MSE + rebalance t2m/v10/tcwv channel weights [1.30, 1.35, 1.15, 1.25, 1.10, 1.15], heavy_rain=2.05 & lead [0.85..1.2]",
            {"channel_weights": (1.30, 1.35, 1.15, 1.25, 1.10, 1.15), "mse_mix": 0.30, "grad_weight": 0.14, "var_weight": 0.08, "var_min_ratio": 0.65, "heavy_rain_boost": 2.05, "vort_div_weight": 0.0, "lead_weights": (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)},
            [1, 2, 2, 1],
        ),
        (
            "inner_e2_dil1241_vort",
            "Dynamics & RMSE: Wider [1,2,4,1] local advection field + curl/divergence wind consistency (0.08) + var_min_ratio=0.72 + MSE/channel balance",
            {"channel_weights": (1.30, 1.35, 1.15, 1.25, 1.10, 1.15), "mse_mix": 0.30, "grad_weight": 0.14, "var_weight": 0.10, "var_min_ratio": 0.72, "heavy_rain_boost": 2.05, "vort_div_weight": 0.08, "lead_weights": (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)},
            [1, 2, 4, 1],
        ),
    ]

    summary = []
    for exp_id, hyp, lcfg, dils in experiments:
        summary.append(
            run_one_experiment(
                exp_id, hyp, archive, tr_s, tr_e, va_s, va_e,
                loss_cfg=lcfg, dilations=dils, epochs_s1=8, epochs_s2=5, s1_stride=2, s2_stride=4, seed=42, threads=6,
            )
        )

    out_p = ROOT_DIR / "reports" / "inner_experiments_stage1.json"
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("INNER SUITE SAVED ->", out_p)


if __name__ == "__main__":
    main()
