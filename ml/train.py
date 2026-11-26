"""Reproducible training script for MonsoonVerse Climatology-Conditioned Residual CNN."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_pipeline.load_data import load_climate_dataset, load_stats
from data_pipeline.preprocess import chronological_split, compute_doy_sincos, normalise_dynamic
from evaluation.climatology import fit_climatology
from ml.loss import make_lat_weights
from ml.model import MonsoonResCNN
from ml.rollout import doy_index_from_sincos


class LeadWeightedClimateLoss(nn.Module):
    """Latitude-, channel-, and lead-weighted SmoothL1 + MSE loss for 8-step rollouts."""

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


class RolloutWindowDataset(Dataset):
    def __init__(
        self,
        data_norm: np.ndarray,
        clim_seq_norm: np.ndarray,
        static_norm: np.ndarray,
        doy_sincos: np.ndarray,
        t_in: int = 3,
        horizon: int = 8,
        stride: int = 2,
    ) -> None:
        self.data_norm = data_norm
        self.clim_seq_norm = clim_seq_norm
        self.static_norm = static_norm
        self.doy_sincos = doy_sincos
        self.t_in = t_in
        self.horizon = horizon
        max_s = len(data_norm) - t_in - horizon + 1
        self.starts = np.arange(0, max_s, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        return {
            "hist": torch.from_numpy(self.data_norm[s : s + self.t_in]),
            "static": torch.from_numpy(self.static_norm),
            "doy_seq": torch.from_numpy(self.doy_sincos[s + self.t_in - 1 : s + self.t_in - 1 + self.horizon]),
            "clim_future": torch.from_numpy(self.clim_seq_norm[s + self.t_in : s + self.t_in + self.horizon]),
            "target": torch.from_numpy(self.data_norm[s + self.t_in : s + self.t_in + self.horizon]),
        }


def train() -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "ml" / "config.json").read_text(encoding="utf-8"))
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    ds = load_climate_dataset()
    stats = load_stats(root / "artifacts" / "stats.json")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    data_phys = ds["data"]
    data_norm = normalise_dynamic(data_phys, mean, std)
    static = ds["static"]
    oro_std = float(static[1].std()) or 1.0
    static_norm = np.stack([static[0], (static[1] - float(static[1].mean())) / oro_std], axis=0).astype(np.float32)
    doy_sc = compute_doy_sincos(ds["time"])

    tr_sl, va_sl = chronological_split(len(data_phys), train_frac=float(cfg["train_frac"]), gap_days=int(cfg["gap_days"]))
    clim_phys = fit_climatology(data_phys[tr_sl], ds["time"][tr_sl])
    clim_norm_366 = normalise_dynamic(clim_phys, mean, std)
    np.savez_compressed(root / "artifacts" / "climatology.npz", clim_norm=clim_norm_366.astype(np.float32))

    doy_idx_all = np.array([doy_index_from_sincos(d) for d in doy_sc], dtype=np.int64)
    clim_seq_norm = clim_norm_366[doy_idx_all]

    horizon = int(cfg["rollout_horizon"])
    epochs = int(cfg["epochs"])
    tr_ds = RolloutWindowDataset(data_norm[tr_sl], clim_seq_norm[tr_sl], static_norm, doy_sc[tr_sl], t_in=int(cfg["t_in"]), horizon=horizon, stride=2)
    va_ds = RolloutWindowDataset(data_norm[va_sl], clim_seq_norm[va_sl], static_norm, doy_sc[va_sl], t_in=int(cfg["t_in"]), horizon=horizon, stride=15)
    tr_loader = DataLoader(tr_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    model = MonsoonResCNN(
        hidden_channels=int(cfg["hidden_channels"]),
        num_blocks=int(cfg["num_blocks"]),
        use_clim_cond=bool(cfg.get("use_clim_cond", True)),
    )
    crit = LeadWeightedClimateLoss(tp_weight=1.35)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val = float("inf")
    best_ep = -1
    history = []
    t0 = time.perf_counter()
    ckpt_path = root / "artifacts" / "model.pt"

    for ep in range(1, epochs + 1):
        cur_h = 1 if ep <= 2 else (4 if ep <= 4 else horizon)
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
        history.append({"epoch": ep, "horizon": cur_h, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"Epoch {ep:02d}/{epochs} (h={cur_h}) | tr={tr_loss:.5f} | va={va_loss:.5f} | {time.perf_counter()-t0:.1f}s", flush=True)

        if cur_h == horizon and va_loss < best_val:
            best_val = va_loss
            best_ep = ep
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "epoch": ep, "val_loss": va_loss}, ckpt_path)

    meta = {
        "model_name": cfg["model_name"],
        "epoch": best_ep,
        "val_loss": best_val,
        "checkpoint_size_bytes": int(ckpt_path.stat().st_size),
        "train_time_s": round(time.perf_counter() - t0, 2),
        "train_samples": len(tr_ds),
        "val_samples": len(va_ds),
        "config": cfg,
        "history": history,
    }
    (root / "artifacts" / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    train()
