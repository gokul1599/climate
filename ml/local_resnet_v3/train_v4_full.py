"""Full 2005-2019 training of Operation 70/70 promoted architecture: LocalResNet-v4.

Incorporates inner-split-verified winners (inner_e2_dil1241_vort: +6.83/70 over reference loss):
- dilations = [1, 2, 4, 1], hidden_dim = 64, init_scale = 0.25 (313,030 params)
- 0.30 MSE + 0.70 Huber + balanced channel weights [1.30, 1.35, 1.15, 1.25, 1.10, 1.15]
- Lead progression [0.85..1.20], heavy_rain_boost=2.05, grad=0.14, var=0.10 (min_ratio=0.72), vort_div=0.08
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
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.score_optimizer_70 import evaluate_exact_70
from ml.local_resnet_v3.dataset import DEFAULT_NC_PATH, MonsoonWindowDataset, Normalizer, chronological_split_indices, load_climate_archive
from ml.local_resnet_v3.experiment_runner import EnhancedRolloutLoss
from ml.local_resnet_v3.inference import LocalResNetV3Predictor, make_coord_grid
from ml.local_resnet_v3.model import MonsoonResNet
from ml.local_resnet_v3.train import atomic_text_write, build_train_climatology_366, train_stage


def train_v4_full(
    seed: int = 42,
    out_dir_name: str = "local_resnet_v4_s42",
    epochs_s1: int = 14,
    epochs_s2: int = 9,
    lyap_weight: float = 0.0,
    threads: int = 6,
) -> dict:
    archive = load_climate_archive(DEFAULT_NC_PATH)
    train_idx, val_idx = chronological_split_indices(archive.dates, mode="reference")
    train_phys, train_dates = archive.data[train_idx], archive.dates[train_idx]
    val_phys, val_dates = archive.data[val_idx], archive.dates[val_idx]

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    adir = ROOT_DIR / "artifacts" / "experiments" / out_dir_name
    adir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    np.savez_compressed(
        adir / "static_fields.npz",
        static=archive.static.astype(np.float32),
        channels=np.array(["land_sea_mask", "orography_m"]),
    )
    clim_366 = build_train_climatology_366(train_phys, archive.dates_int[train_idx], adir / "climatology.npy")

    norm = Normalizer.fit(train_phys, archive.static, log1p_tp=False)
    norm.validate_round_trip(train_phys[:16])
    norm.save(adir / "norm_stats.json")
    atomic_text_write(
        json.dumps({"channels": norm.to_dict()["channels"], "mean": norm.to_dict()["mean"], "std": norm.to_dict()["std"]}, indent=2),
        adir / "stats.json",
    )

    train_norm = norm.normalize(train_phys)
    val_norm = norm.normalize(val_phys)
    static_t = torch.from_numpy(norm.normalize_static(archive.static)[None])
    coord_t = torch.from_numpy(make_coord_grid()[None])

    tp_min_norm = float((0.0 - norm.mean[1]) / norm.std[1])
    t2m_min_norm = float((233.15 - norm.mean[0]) / norm.std[0])
    t2m_max_norm = float((333.15 - norm.mean[0]) / norm.std[0])

    model_cfg = {
        "t_in": 3, "n_dyn": 6, "n_static": 2, "n_time": 2, "n_coord": 2,
        "hidden_dim": 64, "num_blocks": 4, "dilations": [1, 2, 4, 1], "init_scale": 0.25,
    }
    model = MonsoonResNet(**model_cfg)
    loss_cfg = {
        "channel_weights": (1.30, 1.35, 1.15, 1.25, 1.10, 1.15),
        "mse_mix": 0.30,
        "grad_weight": 0.14,
        "var_weight": 0.10,
        "var_min_ratio": 0.72,
        "heavy_rain_boost": 2.05,
        "vort_div_weight": 0.08,
        "lead_weights": (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2),
    }
    rain_thresh_norm = float((10.0 - norm.mean[1]) / norm.std[1])
    loss_fn = EnhancedRolloutLoss(rain_thresh_norm=rain_thresh_norm, **loss_cfg)

    ckpt_path = adir / "model.pt"
    ds_tr1 = MonsoonWindowDataset(train_phys, train_norm, train_dates, horizon=1, stride=1)
    ds_va1 = MonsoonWindowDataset(val_phys, val_norm, val_dates, horizon=1, stride=1)
    dl_tr1 = DataLoader(ds_tr1, batch_size=64, shuffle=True)
    dl_va1 = DataLoader(ds_va1, batch_size=64, shuffle=False)
    s1_res = train_stage(
        model, dl_tr1, dl_va1, loss_fn, static_t, coord_t,
        horizon=1, epochs=epochs_s1, lr=1.5e-3, weight_decay=1e-4, grad_clip=1.0,
        tp_min_norm=tp_min_norm, t2m_min_norm=t2m_min_norm, t2m_max_norm=t2m_max_norm,
        stage_name=f"v4-s{seed}-S1", ckpt_dest=ckpt_path,
        model_name="MonsoonVerse-LocalResNet-v4", model_cfg=model_cfg, device=torch.device("cpu"), grad_accum_steps=2,
    )
    del ds_tr1, ds_va1, dl_tr1, dl_va1
    gc.collect()

    ds_tr2 = MonsoonWindowDataset(train_phys, train_norm, train_dates, horizon=8, stride=3)
    ds_va2 = MonsoonWindowDataset(val_phys, val_norm, val_dates, horizon=8, stride=3)
    dl_tr2 = DataLoader(ds_tr2, batch_size=32, shuffle=True)
    dl_va2 = DataLoader(ds_va2, batch_size=32, shuffle=False)
    s2_res = train_stage(
        model, dl_tr2, dl_va2, loss_fn, static_t, coord_t,
        horizon=8, epochs=epochs_s2, lr=4.5e-4, weight_decay=1e-4, grad_clip=1.0,
        tp_min_norm=tp_min_norm, t2m_min_norm=t2m_min_norm, t2m_max_norm=t2m_max_norm,
        stage_name=f"v4-s{seed}-S2", ckpt_dest=ckpt_path,
        model_name="MonsoonVerse-LocalResNet-v4", model_cfg=model_cfg, device=torch.device("cpu"), grad_accum_steps=4,
    )
    del ds_tr2, dl_tr2
    gc.collect()

    pred = LocalResNetV3Predictor(artifacts_dir=adir)
    ev = evaluate_exact_70(
        pred, archive, dt.date(2005, 1, 1), dt.date(2019, 12, 31),
        dt.date(2020, 1, 1), dt.date(2021, 12, 31),
        stride=14, shock_step=4, clim_366=clim_366, stats_std=norm.std,
    )

    # Empirical residual uncertainty bands
    from ml.local_resnet_v3.dataset import date_to_doy_sincos
    starts = list(range(0, len(val_phys) - 3 - 10 + 1, 14))
    resids = [pred.predict_rollout(val_phys[s : s + 3], date_to_doy_sincos(val_dates[s + 2]).tolist(), steps=10) - val_phys[s + 3 : s + 13] for s in starts]
    res_arr = np.stack(resids, axis=0)
    chs = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
    bands = [
        {
            "day": k + 1,
            "channels": {
                cn: {
                    "bias": round(float(np.mean(res_arr[:, k, c])), 4),
                    "mae": round(float(np.mean(np.abs(res_arr[:, k, c]))), 4),
                    "rmse": round(float(np.sqrt(np.mean(res_arr[:, k, c] ** 2))), 4),
                    "p05": round(float(np.percentile(res_arr[:, k, c], 5)), 4),
                    "p95": round(float(np.percentile(res_arr[:, k, c], 95)), 4),
                }
                for c, cn in enumerate(chs)
            },
        }
        for k in range(10)
    ]
    atomic_text_write(
        json.dumps({
            "model_name": "MonsoonVerse-LocalResNet-v4",
            "is_probabilistic_ensemble": False,
            "deterministic_note": "Deterministic rollout; intervals reflect empirical 2020-2021 validation residuals.",
            "num_windows": len(starts),
            "val_date_start": "2020-01-01",
            "val_date_end": "2021-12-31",
            "leads": bands,
        }, indent=2),
        adir / "uncertainty_bands.json",
    )

    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    man = {
        "model_name": "MonsoonVerse-LocalResNet-v4",
        "model_version": f"local-resnet-v4-seed{seed}",
        "seed": seed,
        "architecture": {**model_cfg, "param_count": sum(int(p.numel()) for p in model.parameters())},
        "loss_cfg": loss_cfg,
        "dataset": {"train_range": "2005-01-01..2019-12-31", "val_range": "2020-01-01..2021-12-31"},
        "duration_s": round(time.perf_counter() - t0, 1),
        "ckpt_sha256": sha,
        "s1": s1_res,
        "s2": s2_res,
        "protocol_a_eval": ev,
    }
    atomic_text_write(json.dumps(man, indent=2), adir / "manifest.json")
    print(f"=== [LocalResNet-v4 seed={seed}] SCORE={ev['total_pts_70']:.2f}/70 | RMSE={ev['rmse_pts_30']:.2f} (ss={ev['mean_ss']:.5f}) CSI={ev['csi_pts_15']:.2f} AB={ev['anti_blur_pts_10']:.2f} CAUS={ev['causality_pts_15']:.2f} (warm={ev['warm_prop_count']}/13, stall={ev['stall_prop_count']}/13) ===", flush=True)
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--epochs-s1", type=int, default=14)
    ap.add_argument("--epochs-s2", type=int, default=9)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    out_n = args.out_dir or f"local_resnet_v4_s{args.seed}"
    train_v4_full(seed=args.seed, out_dir_name=out_n, epochs_s1=args.epochs_s1, epochs_s2=args.epochs_s2, threads=args.threads)
