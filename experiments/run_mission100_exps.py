"""Mission 100 Phase 1 controlled scientific experiments strictly trained on tr_sl (chron split, gap=10)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.load_data import load_climate_dataset, load_stats
from data_pipeline.preprocess import chronological_split, compute_doy_sincos, denormalise_dynamic, normalise_dynamic
from evaluation.climatology import fit_climatology
from experiments.run_experiments import StridedRolloutDataset, _doy_idx_from_sincos, evaluate_candidate
from experiments.run_exp4 import eval_6_seasons, make_predict_fn
from ml.loss import make_lat_weights
from ml.model import MonsoonResCNN
from ml.rollout import advance_doy_sincos
from monsoon_r1 import contract as C, scoring as S


class AnomalyPreservingClimateLoss(nn.Module):
    """Lat/channel/lead SmoothL1+MSE plus anomaly spatial-std & finite-difference gradient conservation."""

    def __init__(
        self,
        cw_vals=(1.0, 1.35, 1.2, 1.2, 1.15, 1.0),
        grad_weight: float = 0.25,
        var_weight: float = 0.35,
        wind_var_boost: float = 2.0,
    ) -> None:
        super().__init__()
        self.register_buffer("lat_w", make_lat_weights())
        cw = torch.tensor(cw_vals, dtype=torch.float32).view(1, 1, 6, 1, 1)
        self.register_buffer("chan_w", cw)
        lw = torch.tensor([1.5, 1.3, 1.15, 1.05, 0.95, 0.9, 0.85, 0.85], dtype=torch.float32).view(1, 8, 1, 1, 1)
        self.register_buffer("lead_w", lw)
        vw = torch.tensor([1.0, 1.0, wind_var_boost, wind_var_boost, 1.0, 1.0], dtype=torch.float32).view(1, 1, 6)
        self.register_buffer("var_chan_w", vw)
        self.grad_weight = grad_weight
        self.var_weight = var_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, clim_future: torch.Tensor) -> torch.Tensor:
        h = pred.shape[1]
        w = self.lat_w.unsqueeze(1) * self.chan_w * self.lead_w[:, :h]
        huber = F.smooth_l1_loss(pred, target, reduction="none", beta=0.5)
        mse = (pred - target) ** 2
        base_loss = ((0.6 * huber + 0.4 * mse) * w).mean()

        anom_p = pred - clim_future[:, :h]
        anom_t = target - clim_future[:, :h]

        # Spatial anomaly std conservation (prevents wave/eddy energy collapse toward climatology)
        std_p = torch.sqrt(anom_p.var(dim=(-2, -1), unbiased=False) + 1e-6)
        std_t = torch.sqrt(anom_t.var(dim=(-2, -1), unbiased=False) + 1e-6)
        var_l = (F.smooth_l1_loss(std_p, std_t, reduction="none", beta=0.2) * self.var_chan_w).mean()

        # Spatial gradient matching of anomaly fields (preserves sharpness and advection fronts)
        dy_p = anom_p[:, :, :, 1:, :] - anom_p[:, :, :, :-1, :]
        dy_t = anom_t[:, :, :, 1:, :] - anom_t[:, :, :, :-1, :]
        dx_p = anom_p[:, :, :, :, 1:] - anom_p[:, :, :, :, :-1]
        dx_t = anom_t[:, :, :, :, 1:] - anom_t[:, :, :, :, :-1]
        grad_l = 0.5 * (F.l1_loss(dy_p, dy_t) + F.l1_loss(dx_p, dx_t))

        return base_loss + self.var_weight * var_l + self.grad_weight * grad_l


class AnomalySpaceResCNN(nn.Module):
    """Predicts next-day synoptic anomaly relative to climatology: X_{t+1} = C_{t+1} + F(A_{t-2..t}, static, doy)."""

    def __init__(self, hidden_channels: int = 32, num_blocks: int = 5) -> None:
        super().__init__()
        self.core = MonsoonResCNN(hidden_channels=hidden_channels, num_blocks=num_blocks, use_clim_cond=False)

    def forward(
        self,
        anom_hist: torch.Tensor,
        static_norm: torch.Tensor,
        doy_sincos: torch.Tensor,
    ) -> torch.Tensor:
        return self.core(anom_hist, static_norm, doy_sincos)


def make_anom_predict_fn(model: AnomalySpaceResCNN, mean, std, static_norm, clim_norm_366):
    @torch.no_grad()
    def predict_fn(init_states, doy_sincos, steps=10, perturbation=None):
        st = np.asarray(init_states, dtype=np.float32).copy()
        if perturbation is not None:
            st[-1] = st[-1] + np.asarray(perturbation, dtype=np.float32)
            st[-1, 1] = np.maximum(st[-1, 1], 0.0)
        cur_hist = normalise_dynamic(st, mean, std)
        doy_idx0 = _doy_idx_from_sincos(doy_sincos)
        # Convert 3-frame input history to anomalies at (t-2, t-1, t)
        hist_cidx = np.array([(doy_idx0 - 2 + j) % 365 for j in range(3)], dtype=np.int64)
        cur_anom = cur_hist - clim_norm_366[hist_cidx]
        st_t = torch.from_numpy(static_norm[None, ...])
        out = np.zeros((steps, 6, 32, 36), dtype=np.float32)
        for k in range(steps):
            dk = advance_doy_sincos(doy_sincos, k)
            at = torch.from_numpy(cur_anom[None, ...])
            dt_t = torch.from_numpy(dk[None, ...])
            anext = model(at, st_t, dt_t).squeeze(0).numpy()
            cidx = (doy_idx0 + k + 1) % 365
            pn = anext + clim_norm_366[cidx]
            pp = denormalise_dynamic(pn, mean, std)
            pp[1] = np.maximum(pp[1], 0.0)
            out[k] = pp
            anext_clipped = normalise_dynamic(pp, mean, std) - clim_norm_366[cidx]
            cur_anom = np.concatenate([cur_anom[1:], anext_clipped[None, ...]], axis=0)
        return out
    return predict_fn


def summarize_eval(ev: dict, s6: dict) -> dict:
    w_ok = sum(1 for r in s6["warm_6"] if r["propagates"])
    st_ok = sum(1 for r in s6["stall_6"] if r["propagates"])
    return {
        "fc_pts": round(float(ev["skill"]["summary"]["forecast_points_out_of_45"]["points"]), 4),
        "rmse_pts": round(float(ev["skill"]["summary"]["forecast_points_out_of_45"]["rmse_pts"]), 4),
        "csi_pts": round(float(ev["skill"]["summary"]["forecast_points_out_of_45"]["csi_pts"]), 4),
        "anti_blur": round(float(ev["skill"]["summary"]["anti_blur_points_out_of_10"]), 4),
        "beats_persist_d4": bool(ev["skill"]["summary"]["beats_persistence_day4"]),
        "caus_pts": round(float(ev["perturbations"]["causality_points_out_of_15"]), 4),
        "warm_prop": bool(ev["perturbations"]["warm_ocean"]["propagates"]),
        "stall_prop": bool(ev["perturbations"]["monsoon_stall"]["propagates"]),
        "warm6": f"{w_ok}/6",
        "stall6": f"{st_ok}/6",
        "warm6_ok": w_ok,
        "stall6_ok": st_ok,
        "validity_failures": int(ev["validity_failures"]),
        "cpu_ms_mean": round(float(ev["cpu_inference_ms_mean"]), 2),
        "null_max_abs": float(ev["perturbations"].get("null_check", {}).get("max_abs_diff", 0.0)),
        "stall_seasons": {r["season"]: {"prop": r["propagates"], "far": round(r.get("far_frac_max", 0.0), 5)} for r in s6["stall_6"]},
    }


def run_exp5_varcons_finetune(data_norm, clim_seq_norm, static_norm, doy_sc, tr_sl, va_sl, val_data, val_times, val_doy, clim_phys, clim_norm_366, mean, std):
    exp_id = "exp5_varcons_ft"
    out_dir = ROOT / "experiments" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    ck = torch.load(ROOT / "artifacts" / "model.pt", map_location="cpu", weights_only=True)
    cfg = dict(ck["config"])
    model = MonsoonResCNN(int(cfg["hidden_channels"]), int(cfg["num_blocks"]), use_clim_cond=True)
    model.load_state_dict(ck["model_state_dict"])

    horizon = 8
    epochs = 2
    tr_ds = StridedRolloutDataset(data_norm[tr_sl], clim_seq_norm[tr_sl], static_norm, doy_sc[tr_sl], horizon=horizon, stride=5)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    crit = AnomalyPreservingClimateLoss(grad_weight=0.25, var_weight=0.50, wind_var_boost=2.5)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)

    t0 = time.perf_counter()
    results = []
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)
            cur_hist = batch["hist"]
            clim_b = batch["clim_future"]
            preds = []
            for k in range(horizon):
                pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k], clim_b[:, k])
                preds.append(pk)
                cur_hist = torch.cat([cur_hist[:, 1:], pk.unsqueeze(1)], dim=1)
            loss = crit(torch.stack(preds, dim=1), batch["target"], clim_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()); nb += 1
        model.eval()
        pfn = make_predict_fn(model, mean, std, static_norm, clim_norm_366)
        ev = evaluate_candidate(pfn, val_data, val_times, val_doy, clim_phys, std)
        s6 = eval_6_seasons(pfn, val_data, val_times, val_doy, std)
        summ = summarize_eval(ev, s6)
        print(f"[{exp_id}] Ep {ep}/{epochs} loss={tr_loss/nb:.4f} | {summ}", flush=True)
        ckpt_p = out_dir / f"model_ep{ep}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg, "epoch": ep, "summary": summ}, ckpt_p)
        results.append({"epoch": ep, "train_loss": round(tr_loss / nb, 5), **summ})

    return {
        "id": exp_id,
        "hypothesis": "Preserve exp3 climatology-conditioned forecast skill while penalizing spatial anomaly variance collapse (var_weight=0.50, wind_var_boost=2.5, grad_weight=0.25) on tr_sl.",
        "seed": 42,
        "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "epochs": results,
    }


def run_exp6_anom_space(data_norm, clim_seq_norm, static_norm, doy_sc, tr_sl, va_sl, val_data, val_times, val_doy, clim_phys, clim_norm_366, mean, std):
    exp_id = "exp6_anom_space"
    out_dir = ROOT / "experiments" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    np.random.seed(123)

    model = AnomalySpaceResCNN(hidden_channels=32, num_blocks=5)
    horizon = 8
    epochs = 4
    tr_ds = StridedRolloutDataset(data_norm[tr_sl], clim_seq_norm[tr_sl], static_norm, doy_sc[tr_sl], horizon=horizon, stride=5)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    crit = AnomalyPreservingClimateLoss(grad_weight=0.20, var_weight=0.35, wind_var_boost=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-4)

    t0 = time.perf_counter()
    results = []
    # Reconstruct lagged climatology for the 3-frame input window inside the batch
    anom_norm = data_norm - clim_seq_norm
    tr_anom_ds = StridedRolloutDataset(anom_norm[tr_sl], clim_seq_norm[tr_sl], static_norm, doy_sc[tr_sl], horizon=horizon, stride=5)
    tr_anom_loader = DataLoader(tr_anom_ds, batch_size=64, shuffle=True)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for batch in tr_anom_loader:
            opt.zero_grad(set_to_none=True)
            cur_anom = batch["hist"]
            clim_b = batch["clim_future"]
            target_full = batch["target"] + clim_b
            preds_full = []
            for k in range(horizon):
                ak = model(cur_anom, batch["static"], batch["doy_seq"][:, k])
                preds_full.append(ak + clim_b[:, k])
                cur_anom = torch.cat([cur_anom[:, 1:], ak.unsqueeze(1)], dim=1)
            loss = crit(torch.stack(preds_full, dim=1), target_full, clim_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()); nb += 1
        sched.step()
        model.eval()
        pfn = make_anom_predict_fn(model, mean, std, static_norm, clim_norm_366)
        ev = evaluate_candidate(pfn, val_data, val_times, val_doy, clim_phys, std)
        s6 = eval_6_seasons(pfn, val_data, val_times, val_doy, std)
        summ = summarize_eval(ev, s6)
        print(f"[{exp_id}] Ep {ep}/{epochs} loss={tr_loss/nb:.4f} | {summ}", flush=True)
        ckpt_p = out_dir / f"model_ep{ep}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": {"hidden_channels": 32, "num_blocks": 5, "anom_space": True}, "epoch": ep, "summary": summ}, ckpt_p)
        results.append({"epoch": ep, "train_loss": round(tr_loss / nb, 5), **summ})

    return {
        "id": exp_id,
        "hypothesis": "Predict synoptic anomalies relative to daily climatology X_{t+1}=C_{t+1}+F(A_{t-2..t},static,doy) with anomaly variance conservation so future C does not directly damp winds.",
        "seed": 123,
        "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "epochs": results,
    }


def main() -> None:
    ds = load_climate_dataset()
    stats = load_stats(ROOT / "artifacts" / "stats.json")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    data_phys = ds["data"]
    data_norm = normalise_dynamic(data_phys, mean, std)
    static = ds["static"]
    oro_std = float(static[1].std()) or 1.0
    static_norm = np.stack([static[0], (static[1] - float(static[1].mean())) / oro_std], axis=0).astype(np.float32)
    times = pd.to_datetime(ds["time"])
    doy_sc = compute_doy_sincos(ds["time"])

    tr_sl, va_sl = chronological_split(len(data_phys), train_frac=0.8, gap_days=10)
    clim_phys = fit_climatology(data_phys[tr_sl], ds["time"][tr_sl])
    clim_norm_366 = normalise_dynamic(clim_phys, mean, std)
    doy_idx_all = np.array([_doy_idx_from_sincos(d) for d in doy_sc], dtype=np.int64)
    clim_seq_norm = clim_norm_366[doy_idx_all]
    val_data, val_times, val_doy = data_phys[va_sl], times[va_sl], doy_sc[va_sl]

    r5 = run_exp5_varcons_finetune(data_norm, clim_seq_norm, static_norm, doy_sc, tr_sl, va_sl, val_data, val_times, val_doy, clim_phys, clim_norm_366, mean, std)
    r6 = run_exp6_anom_space(data_norm, clim_seq_norm, static_norm, doy_sc, tr_sl, va_sl, val_data, val_times, val_doy, clim_phys, clim_norm_366, mean, std)

    out_json = Path(r"C:\Users\gokul\OneDrive\Documents\projects\Yugaantar-Track2\test-results\mission_100_exp5_6.json")
    out_json.write_text(json.dumps({"exp5": r5, "exp6": r6}, indent=2), encoding="utf-8")
    print("SAVED", out_json)


if __name__ == "__main__":
    main()
