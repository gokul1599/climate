"""Controlled forecast skill improvement experiments (Exp 1: 8-step rollout, Exp 2: Clim-conditioned ResCNN)."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_pipeline.load_data import load_climate_dataset, load_stats
from data_pipeline.preprocess import chronological_split, compute_doy_sincos, denormalise_dynamic, normalise_dynamic
from evaluation.climatology import fit_climatology, predict_climatology
from evaluation.metrics import evaluate_lead_metrics
from evaluation.persistence import predict_persistence
from evaluation.perturbation_tests import run_perturbation_suite
from evaluation.stability_tests import check_rollout_validity
from ml.loss import ClimateRolloutLoss
from ml.model import MonsoonResCNN, ResBlock
from ml.rollout import advance_doy_sincos


class ClimCondResCNN(nn.Module):
    """28-channel Residual CNN conditioned on training-only daily climatology for the target day."""

    def __init__(self, hidden_channels: int = 32, num_blocks: int = 4) -> None:
        super().__init__()
        in_channels = 3 * 6 + 2 + 2 + 6  # 28
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden_channels, 6, kernel_size=3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        hist_norm: torch.Tensor,
        static_norm: torch.Tensor,
        doy_sincos: torch.Tensor,
        clim_next_norm: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c, h, w = hist_norm.shape
        hist_flat = hist_norm.reshape(b, t * c, h, w)
        doy_grid = doy_sincos.view(b, 2, 1, 1).expand(b, 2, h, w)
        x = torch.cat([hist_flat, static_norm, doy_grid, clim_next_norm], dim=1)
        delta = self.head(self.blocks(self.stem(x)))
        return hist_norm[:, -1] + delta


class StridedRolloutDataset(Dataset):
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


def _doy_idx_from_sincos(doy_sincos: list[float] | np.ndarray) -> int:
    ang = float(np.arctan2(doy_sincos[0], doy_sincos[1]))
    if ang < 0:
        ang += 2.0 * np.pi
    doy = int(round(ang * 365.0 / (2.0 * np.pi))) % 365
    return doy  # 0..364


@torch.no_grad()
def evaluate_candidate(
    predict_fn,
    val_data: np.ndarray,
    val_times: pd.DatetimeIndex,
    val_doy: np.ndarray,
    clim_phys: np.ndarray,
    std: np.ndarray,
) -> dict:
    t_in, steps, stride = 3, 10, 15
    starts = list(range(0, len(val_data) - t_in - steps + 1, stride))
    preds, trues, persists, clims, lat_ms = [], [], [], [], []
    invalid = 0
    for s in starts:
        init = val_data[s : s + t_in]
        doy = val_doy[s + t_in - 1]
        t0 = time.perf_counter()
        p = predict_fn(init, doy, steps=steps, perturbation=None)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        if not check_rollout_validity(p, steps=steps)["valid"]:
            invalid += 1
        preds.append(p)
        trues.append(val_data[s + t_in : s + t_in + steps])
        persists.append(predict_persistence(init, steps=steps))
        clims.append(predict_climatology(clim_phys, val_times[s + t_in - 1].date(), steps=steps))

    skill = evaluate_lead_metrics(
        np.stack(preds), np.stack(trues), np.stack(persists), np.stack(clims)
    )
    ms_idx = next((i for i, s in enumerate(starts) if val_times[s + t_in - 1].month in (7, 8)), len(starts) // 2)
    ms_s = starts[ms_idx]
    pert = run_perturbation_suite(
        predict_fn,
        val_data[ms_s : ms_s + t_in],
        val_doy[ms_s + t_in - 1].tolist(),
        std=std,
        steps=steps,
    )
    return {
        "validity_failures": invalid,
        "cpu_inference_ms_mean": float(np.mean(lat_ms)),
        "skill": skill,
        "perturbations": pert,
    }


def run_experiment(
    name: str,
    use_clim_cond: bool,
    data_norm: np.ndarray,
    clim_seq_norm: np.ndarray,
    clim_norm_366: np.ndarray,
    static_norm: np.ndarray,
    doy_sc: np.ndarray,
    tr_sl: slice,
    va_sl: slice,
    val_data: np.ndarray,
    val_times: pd.DatetimeIndex,
    val_doy: np.ndarray,
    clim_phys: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    out_dir: Path,
    epochs: int = 8,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    horizon = 8
    tr_ds = StridedRolloutDataset(data_norm[tr_sl], clim_seq_norm[tr_sl], static_norm, doy_sc[tr_sl], horizon=horizon, stride=4)
    va_ds = StridedRolloutDataset(data_norm[va_sl], clim_seq_norm[va_sl], static_norm, doy_sc[va_sl], horizon=horizon, stride=15)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    model = ClimCondResCNN(32, 4) if use_clim_cond else MonsoonResCNN(32, 4)
    crit = ClimateRolloutLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val = float("inf")
    best_ep = -1
    logs = []
    t0 = time.perf_counter()
    ckpt_path = out_dir / "model.pt"

    for ep in range(1, epochs + 1):
        cur_h = 1 if ep == 1 else (4 if ep == 2 else 8)
        model.train()
        tr_loss, nb = 0.0, 0
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)
            cur_hist = batch["hist"]
            preds = []
            for k in range(cur_h):
                if use_clim_cond:
                    pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k], batch["clim_future"][:, k])
                else:
                    pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k])
                preds.append(pk)
                cur_hist = torch.cat([cur_hist[:, 1:], pk.unsqueeze(1)], dim=1)
            pred_t = torch.stack(preds, dim=1)
            loss = crit(pred_t, batch["target"][:, :cur_h])
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
                    if use_clim_cond:
                        pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k], batch["clim_future"][:, k])
                    else:
                        pk = model(cur_hist, batch["static"], batch["doy_seq"][:, k])
                    preds.append(pk)
                    cur_hist = torch.cat([cur_hist[:, 1:], pk.unsqueeze(1)], dim=1)
                va_loss += float(crit(torch.stack(preds, dim=1), batch["target"]).item())
                vnb += 1
        va_loss /= max(vnb, 1)
        line = f"[{name}] Ep {ep:02d}/{epochs} (h={cur_h}) | tr={tr_loss:.5f} | va={va_loss:.5f} | {time.perf_counter()-t0:.1f}s"
        print(line, flush=True)
        logs.append(line)

        if cur_h == horizon and va_loss < best_val:
            best_val = va_loss
            best_ep = ep
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model_name": f"MonsoonVerse-{name}",
                        "use_clim_cond": use_clim_cond,
                        "hidden_channels": 32,
                        "num_blocks": 4,
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
            if use_clim_cond:
                cidx = (doy_idx0 + k + 1) % 365
                ct = torch.from_numpy(clim_norm_366[cidx : cidx + 1])
                pn = model(ht, st_t, dt_t, ct).squeeze(0).numpy()
            else:
                pn = model(ht, st_t, dt_t).squeeze(0).numpy()
            pp = denormalise_dynamic(pn, mean, std)
            pp[1] = np.maximum(pp[1], 0.0)
            out[k] = pp
            cur_hist = np.concatenate([cur_hist[1:], normalise_dynamic(pp, mean, std)[None, ...]], axis=0)
        return out

    eval_res = evaluate_candidate(predict_fn, val_data, val_times, val_doy, clim_phys, std)
    summary = {
        "name": name,
        "use_clim_cond": use_clim_cond,
        "best_epoch": best_ep,
        "best_val_loss": best_val,
        "train_time_s": round(time.perf_counter() - t0, 1),
        "eval": eval_res,
    }
    (out_dir / "train.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    (out_dir / "validation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
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
    clim_phys = fit_climatology(data_phys[tr_sl], ds["time"][tr_sl])  # [366, 6, 32, 36]
    clim_norm_366 = normalise_dynamic(clim_phys, mean, std)
    doy_idx_all = np.array([_doy_idx_from_sincos(d) for d in doy_sc], dtype=np.int64)
    clim_seq_norm = clim_norm_366[doy_idx_all]

    val_data = data_phys[va_sl]
    val_times = times[va_sl]
    val_doy = doy_sc[va_sl]

    res1 = run_experiment(
        "exp1_rollout8", False, data_norm, clim_seq_norm, clim_norm_366, static_norm, doy_sc,
        tr_sl, va_sl, val_data, val_times, val_doy, clim_phys, mean, std,
        root / "experiments" / "exp1_rollout8", epochs=5,
    )
    res2 = run_experiment(
        "exp2_clim_cond", True, data_norm, clim_seq_norm, clim_norm_366, static_norm, doy_sc,
        tr_sl, va_sl, val_data, val_times, val_doy, clim_phys, mean, std,
        root / "experiments" / "exp2_clim_cond", epochs=5,
    )

    for r in (res1, res2):
        sk = r["eval"]["skill"]
        print(
            f"SUMMARY {r['name']}: D1={sk['leads']['day_1']['mean_ss']:.4f}, "
            f"D4={sk['leads']['day_4']['mean_ss']:.4f}, D8={sk['leads']['day_8']['mean_ss']:.4f}, "
            f"FC_PTS={sk['summary']['forecast_points_out_of_45']['points']:.2f}, "
            f"AB={sk['summary']['anti_blur_points_out_of_10']:.2f}, "
            f"CAUS={r['eval']['perturbations']['causality_points_out_of_15']:.2f}"
        )


if __name__ == "__main__":
    main()
