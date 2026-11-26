"""Experiment 4: Fine-tune ClimResCNN strictly on training split (tr_sl) with wind-climatology dropout & spatial gradient loss to eliminate spurious wind-shock damping."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline.load_data import load_climate_dataset, load_stats
from data_pipeline.preprocess import chronological_split, compute_doy_sincos, denormalise_dynamic, normalise_dynamic
from evaluation.climatology import fit_climatology
from experiments.run_experiments import (
    StridedRolloutDataset,
    _doy_idx_from_sincos,
    evaluate_candidate,
)
from ml.loss import make_lat_weights
from ml.model import MonsoonResCNN
from ml.rollout import advance_doy_sincos
from monsoon_r1 import contract as C, scoring as S


class GradientLeadClimateLoss(nn.Module):
    """Lat-, channel-, and lead-weighted SmoothL1+MSE plus spatial gradient preservation loss."""

    def __init__(self, cw_vals=(1.0, 1.35, 1.2, 1.2, 1.15, 1.0), grad_weight: float = 0.25) -> None:
        super().__init__()
        self.register_buffer("lat_w", make_lat_weights())
        cw = torch.tensor(cw_vals, dtype=torch.float32).view(1, 1, 6, 1, 1)
        self.register_buffer("chan_w", cw)
        lw = torch.tensor([1.5, 1.3, 1.15, 1.05, 0.95, 0.9, 0.85, 0.85], dtype=torch.float32).view(1, 8, 1, 1, 1)
        self.register_buffer("lead_w", lw)
        self.grad_weight = grad_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h = pred.shape[1]
        w = self.lat_w.unsqueeze(1) * self.chan_w * self.lead_w[:, :h]
        huber = F.smooth_l1_loss(pred, target, reduction="none", beta=0.5)
        mse = (pred - target) ** 2
        base_loss = ((0.6 * huber + 0.4 * mse) * w).mean()

        # Spatial finite-difference gradient loss preserves advection & eddy sharpness
        dy_p = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dy_t = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        dx_p = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        dx_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        grad_l = (F.l1_loss(dy_p, dy_t) + F.l1_loss(dx_p, dx_t)) * 0.5
        return base_loss + self.grad_weight * grad_l


def make_predict_fn(model, mean, std, static_norm, clim_norm_366):
    @torch.no_grad()
    def predict_fn(init_states, doy_sincos, steps=10, perturbation=None):
        st = np.asarray(init_states, dtype=np.float32).copy()
        if perturbation is not None:
            st[-1] = st[-1] + np.asarray(perturbation, dtype=np.float32)
            st[-1, 1] = np.maximum(st[-1, 1], 0.0)
        cur_hist = normalise_dynamic(st, mean, std)
        st_t = torch.from_numpy(static_norm[None, ...])
        doy_idx0 = _doy_idx_from_sincos(doy_sincos)
        out = np.zeros((steps, 6, 32, 36), dtype=np.float32)
        for k in range(steps):
            dk = advance_doy_sincos(doy_sincos, k)
            ht = torch.from_numpy(cur_hist[None, ...])
            dt_t = torch.from_numpy(dk[None, ...])
            cidx = (doy_idx0 + k + 1) % 365
            ct = torch.from_numpy(clim_norm_366[cidx : cidx + 1])
            pn = model(ht, st_t, dt_t, ct).squeeze(0).numpy()
            pp = denormalise_dynamic(pn, mean, std)
            pp[1] = np.maximum(pp[1], 0.0)
            out[k] = pp
            cur_hist = np.concatenate([cur_hist[1:], normalise_dynamic(pp, mean, std)[None, ...]], axis=0)
        return out
    return predict_fn


def eval_6_seasons(predict_fn, val_data, val_times, val_doy, std) -> dict:
    valid_idx = np.arange(2, len(val_times) - 10)
    t_sub = val_times[valid_idx]
    tp_mean = val_data[valid_idx, 1].mean(axis=(1, 2))
    warm_list, stall_list = [], []
    for target_m, lbl in [(5, "May"), (6, "Jun"), (7, "Jul"), (8, "Aug"), (9, "Sep"), (1, "Jan")]:
        idxs = valid_idx[t_sub.month == target_m]
        sub_scores = tp_mean[t_sub.month == target_m]
        m_idx = int(idxs[np.argmax(sub_scores)])
        init_m = val_data[m_idx - 2 : m_idx + 1]
        doy_m = val_doy[m_idx].tolist()
        base_m = predict_fn(init_m, doy_m, 10, None)
        out_w = predict_fn(init_m, doy_m, 10, S.make_shock("warm_ocean", init_m[-1], std))
        out_s = predict_fn(init_m, doy_m, 10, S.make_shock("monsoon_stall", init_m[-1], std))
        warm_list.append({"season": lbl, **S.causality_checks(base_m, out_w, C.BOX_BAY_OF_BENGAL, std)})
        stall_list.append({"season": lbl, **S.causality_checks(base_m, out_s, C.BOX_ARABIAN_SEA, std)})
    return {"warm_6": warm_list, "stall_6": stall_list}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "experiments" / "exp4_wind_adv"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_climate_dataset()
    stats = load_stats(root / "artifacts" / "stats.json")
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

    val_data = data_phys[va_sl]
    val_times = times[va_sl]
    val_doy = doy_sc[va_sl]

    torch.manual_seed(42)
    np.random.seed(42)

    # Load frozen baseline exp3 weights cleanly (read-only) to initialize fine-tuning on tr_sl only
    ck = torch.load(root / "artifacts" / "model.pt", map_location="cpu", weights_only=True)
    cfg = dict(ck["config"])
    model = MonsoonResCNN(
        hidden_channels=int(cfg["hidden_channels"]),
        num_blocks=int(cfg["num_blocks"]),
        use_clim_cond=True,
    )
    model.load_state_dict(ck["model_state_dict"])

    horizon = 8
    epochs = 2
    tr_ds = StridedRolloutDataset(data_norm[tr_sl], clim_seq_norm[tr_sl], static_norm, doy_sc[tr_sl], horizon=horizon, stride=6)
    va_ds = StridedRolloutDataset(data_norm[va_sl], clim_seq_norm[va_sl], static_norm, doy_sc[va_sl], horizon=horizon, stride=15)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    crit = GradientLeadClimateLoss(grad_weight=0.25)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    t0 = time.perf_counter()
    logs = []
    best_score = -1.0
    ckpt_path = out_dir / "model.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for b_idx, batch in enumerate(tr_loader):
            opt.zero_grad(set_to_none=True)
            cur_hist = batch["hist"]
            # Zero out u10/v10 clim conditioning 85% of the time per batch to stop spurious clim wind relaxation
            clim_b = batch["clim_future"].clone()
            mask = (torch.rand(clim_b.shape[0], 1, 1, 1, 1) < 0.85)
            clim_b[:, :, 2:4] = torch.where(mask, torch.zeros_like(clim_b[:, :, 2:4]), clim_b[:, :, 2:4])
            preds = []
            for k in range(horizon):
                pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k], clim_b[:, k])
                preds.append(pk)
                cur_hist = torch.cat([cur_hist[:, 1:], pk.unsqueeze(1)], dim=1)
            loss = crit(torch.stack(preds, dim=1), batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item())
            nb += 1
            if (b_idx + 1) % 5 == 0 or (b_idx + 1) == len(tr_loader):
                print(f"[exp4] Ep {ep} batch {b_idx+1}/{len(tr_loader)} loss={loss.item():.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
        sched.step()
        tr_loss /= max(nb, 1)

        model.eval()
        pfn = make_predict_fn(model, mean, std, static_norm, clim_norm_366)
        ev = evaluate_candidate(pfn, val_data, val_times, val_doy, clim_phys, std)
        fc = ev["skill"]["summary"]["forecast_points_out_of_45"]["points"]
        ca = ev["perturbations"]["causality_points_out_of_15"]
        wp = ev["perturbations"]["warm_ocean"]["propagates"]
        sp = ev["perturbations"]["monsoon_stall"]["propagates"]
        tot = fc + ev["skill"]["summary"]["anti_blur_points_out_of_10"] + ca
        line = f"[exp4] Ep {ep}/{epochs} | tr={tr_loss:.4f} | fc={fc:.2f}/45 caus={ca:.2f}/15 (warm={wp}, stall={sp}) total={tot:.2f}/70 | {time.perf_counter()-t0:.1f}s"
        print(line, flush=True)
        logs.append(line)

        ep_ck = out_dir / f"model_ep{ep}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg, "epoch": ep, "score": tot}, ep_ck)
        if tot > best_score:
            best_score = tot
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "epoch": ep, "score": tot}, ckpt_path)

    best_ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_ck["model_state_dict"]); model.eval()
    pfn = make_predict_fn(model, mean, std, static_norm, clim_norm_366)
    final_eval = evaluate_candidate(pfn, val_data, val_times, val_doy, clim_phys, std)
    seasons = eval_6_seasons(pfn, val_data, val_times, val_doy, std)
    summary = {
        "name": "exp4_wind_adv",
        "best_epoch": int(best_ck["epoch"]),
        "train_time_s": round(time.perf_counter() - t0, 1),
        "eval": final_eval,
        "seasons_6": seasons,
    }
    (out_dir / "train.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    (out_dir / "validation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE exp4:", round(final_eval["skill"]["summary"]["forecast_points_out_of_45"]["points"], 2), "caus=", final_eval["perturbations"]["causality_points_out_of_15"], "warm6=", sum(x["propagates"] for x in seasons["warm_6"]), "stall6=", sum(x["propagates"] for x in seasons["stall_6"]), flush=True)


if __name__ == "__main__":
    main()
