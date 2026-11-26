"""Experiment 3: Tuned Climatology-Conditioned ResCNN (stride=2, 7 epochs, lead-weighted loss, 5 ResBlocks)."""
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
    ClimCondResCNN,
    StridedRolloutDataset,
    _doy_idx_from_sincos,
    evaluate_candidate,
)
from ml.loss import make_lat_weights
from ml.rollout import advance_doy_sincos


class LeadWeightedClimateLoss(nn.Module):
    def __init__(self, tp_weight: float = 1.35) -> None:
        super().__init__()
        self.register_buffer("lat_w", make_lat_weights())
        cw = torch.tensor((1.0, tp_weight, 1.0, 1.0, 1.0, 1.0), dtype=torch.float32).view(1, 1, 6, 1, 1)
        self.register_buffer("chan_w", cw)
        lw = torch.tensor([1.6, 1.3, 1.1, 1.0, 0.9, 0.85, 0.8, 0.8], dtype=torch.float32).view(1, 8, 1, 1, 1)
        self.register_buffer("lead_w", lw)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h = pred.shape[1]
        w = self.lat_w.unsqueeze(1) * self.chan_w * self.lead_w[:, :h]
        huber = F.smooth_l1_loss(pred, target, reduction="none", beta=0.5)
        mse = (pred - target) ** 2
        return ((0.6 * huber + 0.4 * mse) * w).mean()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "experiments" / "exp3_clim_tuned"
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

    horizon = 8
    epochs = 7
    tr_ds = StridedRolloutDataset(data_norm[tr_sl], clim_seq_norm[tr_sl], static_norm, doy_sc[tr_sl], horizon=horizon, stride=2)
    va_ds = StridedRolloutDataset(data_norm[va_sl], clim_seq_norm[va_sl], static_norm, doy_sc[va_sl], horizon=horizon, stride=15)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    model = ClimCondResCNN(hidden_channels=32, num_blocks=5)
    crit = LeadWeightedClimateLoss(tp_weight=1.35)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val = float("inf")
    best_ep = -1
    logs = []
    t0 = time.perf_counter()
    ckpt_path = out_dir / "model.pt"

    for ep in range(1, epochs + 1):
        cur_h = 1 if ep <= 2 else (4 if ep <= 4 else 8)
        model.train()
        tr_loss, nb = 0.0, 0
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)
            cur_hist = batch["hist"]
            preds = []
            for k in range(cur_h):
                pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k], batch["clim_future"][:, k])
                preds.append(pk)
                cur_hist = torch.cat([cur_hist[:, 1:], pk.unsqueeze(1)], dim=1)
            loss = crit(torch.stack(preds, dim=1), batch["target"][:, :cur_h])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item())
            nb += 1
        sched.step()
        tr_loss /= max(nb, 1)

        model.eval()
        va_loss, vnb = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                cur_hist = batch["hist"]
                preds = []
                for k in range(horizon):
                    pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k], batch["clim_future"][:, k])
                    preds.append(pk)
                    cur_hist = torch.cat([cur_hist[:, 1:], pk.unsqueeze(1)], dim=1)
                va_loss += float(crit(torch.stack(preds, dim=1), batch["target"]).item())
                vnb += 1
        va_loss /= max(vnb, 1)
        line = f"[exp3_clim_tuned] Ep {ep:02d}/{epochs} (h={cur_h}) | tr={tr_loss:.5f} | va={va_loss:.5f} | {time.perf_counter()-t0:.1f}s"
        print(line, flush=True)
        logs.append(line)

        if cur_h == horizon and va_loss < best_val:
            best_val = va_loss
            best_ep = ep
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model_name": "MonsoonVerse-ClimResCNN",
                        "use_clim_cond": True,
                        "hidden_channels": 32,
                        "num_blocks": 5,
                        "rollout_horizon": horizon,
                        "epochs": epochs,
                        "seed": 42,
                    },
                    "epoch": ep,
                    "val_loss": va_loss,
                },
                ckpt_path,
            )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    eval_res = evaluate_candidate(predict_fn, val_data, val_times, val_doy, clim_phys, std)
    summary = {
        "name": "exp3_clim_tuned",
        "use_clim_cond": True,
        "best_epoch": best_ep,
        "best_val_loss": best_val,
        "train_time_s": round(time.perf_counter() - t0, 1),
        "eval": eval_res,
    }
    (out_dir / "train.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    (out_dir / "validation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sk = eval_res["skill"]
    print(
        f"SUMMARY exp3_clim_tuned: D1={sk['leads']['day_1']['mean_ss']:.4f}, "
        f"D4={sk['leads']['day_4']['mean_ss']:.4f}, D8={sk['leads']['day_8']['mean_ss']:.4f}, "
        f"FC_PTS={sk['summary']['forecast_points_out_of_45']['points']:.2f}, "
        f"AB={sk['summary']['anti_blur_points_out_of_10']:.2f}, "
        f"CAUS={eval_res['perturbations']['causality_points_out_of_15']:.2f}"
    )


if __name__ == "__main__":
    main()
