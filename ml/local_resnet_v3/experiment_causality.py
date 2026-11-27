"""Stage 5 / Phase 4 Causality Experiment: Bounded Lyapunov Perturbation Sensitivity Regularization.

Strictly trained on inner split (2005-2017 -> 2018-2019), initialized from inner_e2_dil1241_vort.
Uses generic random Gaussian patches (zero knowledge of BoB/Arabian Sea or official shocks)
to keep the model's multi-day tangent sensitivity bounded in a non-dissipative, stable band.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.score_optimizer_70 import evaluate_exact_70
from ml.local_resnet_v3.dataset import DEFAULT_NC_PATH, MonsoonWindowDataset, Normalizer, load_climate_archive
from ml.local_resnet_v3.experiment_runner import EnhancedRolloutLoss
from ml.local_resnet_v3.inference import LocalResNetV3Predictor, make_coord_grid
from ml.local_resnet_v3.model import MonsoonResNet
from ml.local_resnet_v3.train import atomic_text_write, atomic_torch_save, build_train_climatology_366, rollout_forward


def sample_random_local_patch(b: int, lat: int = 32, lon: int = 36) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate generic localized wind-vector synoptic patches on random coordinates (no official boxes)."""
    mask = torch.zeros((b, 1, lat, lon), dtype=torch.float32)
    amp = torch.zeros((b, 6, lat, lon), dtype=torch.float32)
    for i in range(b):
        la0 = int(torch.randint(2, lat - 10, (1,)).item())
        lo0 = int(torch.randint(2, lon - 10, (1,)).item())
        h_box = int(torch.randint(6, 12, (1,)).item())
        w_box = int(torch.randint(6, 12, (1,)).item())
        mask[i, 0, la0 : la0 + h_box, lo0 : lo0 + w_box] = 1.0
        amp[i, 2] = mask[i, 0] * float(torch.randn(1).item() * 0.5)
        amp[i, 3] = mask[i, 0] * float(torch.randn(1).item() * 0.5)
    return amp, mask


def main() -> None:
    archive = load_climate_archive(DEFAULT_NC_PATH)
    tr_s, tr_e = dt.date(2005, 1, 1), dt.date(2017, 12, 31)
    va_s, va_e = dt.date(2018, 1, 1), dt.date(2019, 12, 31)

    parent_dir = ROOT_DIR / "artifacts" / "experiments" / "inner_e2_dil1241_vort"
    exp_id = "inner_e3_lyapunov_ft"
    exp_dir = ROOT_DIR / "artifacts" / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)
    t0 = time.perf_counter()

    tr_idx = np.where((archive.dates >= tr_s) & (archive.dates <= tr_e))[0]
    va_idx = np.where((archive.dates >= va_s) & (archive.dates <= va_e))[0]
    train_phys, train_dates = archive.data[tr_idx], archive.dates[tr_idx]
    val_phys, val_dates = archive.data[va_idx], archive.dates[va_idx]

    np.savez_compressed(exp_dir / "static_fields.npz", static=archive.static.astype(np.float32))
    clim_366 = build_train_climatology_366(train_phys, archive.dates_int[tr_idx], exp_dir / "climatology.npy")
    norm = Normalizer.load(parent_dir / "norm_stats.json")
    norm.save(exp_dir / "norm_stats.json")

    train_norm = norm.normalize(train_phys)
    static_t = torch.from_numpy(norm.normalize_static(archive.static)[None])
    coord_t = torch.from_numpy(make_coord_grid()[None])
    tp_min_norm = float((0.0 - norm.mean[1]) / norm.std[1])
    t2m_min_norm = float((233.15 - norm.mean[0]) / norm.std[0])
    t2m_max_norm = float((333.15 - norm.mean[0]) / norm.std[0])

    ck = torch.load(parent_dir / "model.pt", map_location="cpu", weights_only=True)
    mcfg = dict(ck["model_config"])
    model = MonsoonResNet(**mcfg)
    model.load_state_dict(ck["state_dict"])

    rain_thresh_norm = float((10.0 - norm.mean[1]) / norm.std[1])
    base_crit = EnhancedRolloutLoss(
        channel_weights=(1.30, 1.35, 1.15, 1.25, 1.10, 1.15),
        mse_mix=0.30, grad_weight=0.14, var_weight=0.10, var_min_ratio=0.72,
        heavy_rain_boost=2.05, vort_div_weight=0.08,
        lead_weights=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2),
        rain_thresh_norm=rain_thresh_norm,
    )

    ds_tr = MonsoonWindowDataset(train_phys, train_norm, train_dates, horizon=8, stride=6)
    dl_tr = DataLoader(ds_tr, batch_size=24, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1e-4)

    ckpt_path = exp_dir / "model.pt"
    epochs = 1
    for ep in range(1, epochs + 1):
        model.train()
        ep_t0 = time.perf_counter()
        sums = {"base": 0.0, "lyap": 0.0, "total": 0.0}
        n = 0
        opt.zero_grad(set_to_none=True)
        for b_idx, (init_n, doy_sc, tgt_n, _) in enumerate(dl_tr):
            pred_clean = rollout_forward(model, init_n, doy_sc, static_t, coord_t, 8, tp_min_norm, t2m_min_norm, t2m_max_norm)
            l_base, _ = base_crit.forward_with_components(pred_clean, tgt_n)

            # Bounded Lyapunov sensitivity & outside-patch propagation on random spatial patches
            amp, patch_mask = sample_random_local_patch(init_n.shape[0])
            init_pert = init_n.clone()
            init_pert[:, -1] = init_pert[:, -1] + amp
            pred_pert = rollout_forward(model, init_pert, doy_sc, static_t, coord_t, 8, tp_min_norm, t2m_min_norm, t2m_max_norm)

            diff_seq = pred_pert - pred_clean  # [B, 8, 6, 32, 36]
            e_init = (amp ** 2).sum(dim=(1, 2, 3)) + 1e-6  # [B]
            e_tot_k = (diff_seq ** 2).sum(dim=(2, 3, 4))  # [B, 8]
            outside_m = (1.0 - patch_mask).unsqueeze(1)
            e_out_late = ((diff_seq[:, 3:8] ** 2) * outside_m).sum(dim=(2, 3, 4)).mean(dim=1)  # [B]
            # Encourage slow-decaying multi-day outward sensitivity (~2.5% of random initial patch energy) & forbid blow-up (>2.5x)
            loss_floor = F.relu(0.025 * e_init - e_out_late) / e_init
            loss_ceil = F.relu(e_tot_k.max(dim=1).values - 2.5 * e_init) / e_init
            l_lyap = (loss_floor + 0.25 * loss_ceil).mean()

            tot = l_base + 0.60 * l_lyap
            (tot / 4.0).backward()
            if ((b_idx + 1) % 4 == 0) or ((b_idx + 1) == len(dl_tr)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            bs = int(init_n.shape[0])
            sums["base"] += float(l_base.detach()) * bs
            sums["lyap"] += float(l_lyap.detach()) * bs
            sums["total"] += float(tot.detach()) * bs
            n += bs
        print(f"[{exp_id}] ep {ep}/{epochs} | base={sums['base']/n:.4f} lyap={sums['lyap']/n:.4f} | {time.perf_counter()-ep_t0:.1f}s", flush=True)

    atomic_torch_save({"model_name": exp_id, "model_config": mcfg, "state_dict": model.state_dict()}, ckpt_path)
    pred = LocalResNetV3Predictor(artifacts_dir=exp_dir)
    ev = evaluate_exact_70(pred, archive, tr_s, tr_e, va_s, va_e, stride=14, shock_step=4, clim_366=clim_366, stats_std=norm.std)
    rec = {
        "exp_id": exp_id,
        "hypothesis": "Generic random-patch bounded Lyapunov sensitivity regularizer (0.025*e_init outward floor, 2.5x ceiling)",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "ckpt_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "eval": ev,
    }
    atomic_text_write(json.dumps(rec, indent=2), exp_dir / "manifest_and_eval.json")
    print(f"=== [{exp_id}] SCORE={ev['total_pts_70']:.2f}/70 | RMSE={ev['rmse_pts_30']:.2f} (ss={ev['mean_ss']:.5f}) CSI={ev['csi_pts_15']:.2f} AB={ev['anti_blur_pts_10']:.2f} CAUS={ev['causality_pts_15']:.2f} (warm={ev['warm_prop_count']}/13, stall={ev['stall_prop_count']}/13) ===", flush=True)


if __name__ == "__main__":
    main()
