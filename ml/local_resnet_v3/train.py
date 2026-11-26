"""Reproducible two-stage training pipeline for MonsoonVerse-LocalResNet-v3 challenger."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.local_resnet_v3.dataset import (
    DEFAULT_NC_PATH,
    MonsoonWindowDataset,
    Normalizer,
    chronological_split_indices,
    load_climate_archive,
)
from ml.local_resnet_v3.inference import COS_D, SIN_D, LocalResNetV3Predictor, make_coord_grid
from ml.local_resnet_v3.loss import MonsoonRolloutLoss
from ml.local_resnet_v3.model import MonsoonResNet


def atomic_torch_save(payload: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, dest)


def atomic_text_write(text: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dest)


def build_train_climatology_366(train_data: np.ndarray, train_dates_int: np.ndarray, out_path: Path) -> np.ndarray:
    """Compute 15-day smoothed day-of-year climatology strictly from training split."""
    dates = [dt.datetime.strptime(str(d), "%Y%m%d").date() for d in train_dates_int]
    doys = np.array([min(d.timetuple().tm_yday - 1, 364) for d in dates], dtype=np.int64)
    _, c, h, w = train_data.shape
    raw = np.zeros((365, c, h, w), dtype=np.float64)
    cnt = np.zeros((365,), dtype=np.int64)
    for t_idx in range(len(train_data)):
        raw[doys[t_idx]] += train_data[t_idx]
        cnt[doys[t_idx]] += 1
    raw /= np.maximum(cnt[:, None, None, None], 1)

    clim_365 = np.zeros_like(raw)
    offsets = np.arange(-7, 8)
    for d in range(365):
        idxs = (d + offsets) % 365
        clim_365[d] = raw[idxs].mean(axis=0)
    clim_366 = np.concatenate([clim_365, clim_365[364:365]], axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, clim_366)
    return clim_366


def rollout_forward(
    model: MonsoonResNet,
    init_norm: torch.Tensor,
    doy_sc: torch.Tensor,
    static_t: torch.Tensor,
    coord_t: torch.Tensor,
    horizon: int,
    tp_min_norm: float,
    t2m_min_norm: float,
    t2m_max_norm: float,
) -> torch.Tensor:
    b = init_norm.shape[0]
    st_b = static_t.expand(b, -1, -1, -1)
    co_b = coord_t.expand(b, -1, -1, -1)
    hist = init_norm
    s_v = doy_sc[:, 0]
    c_v = doy_sc[:, 1]
    preds = []

    for _ in range(horizon):
        d_cur = torch.stack([s_v, c_v], dim=-1)
        nxt = model(hist, st_b, d_cur, co_b)
        tp_clamped = torch.clamp(nxt[:, 1:2], min=tp_min_norm)
        t2m_clamped = torch.clamp(nxt[:, 0:1], min=t2m_min_norm, max=t2m_max_norm)
        nxt_safe = torch.cat([t2m_clamped, tp_clamped, nxt[:, 2:]], dim=1)
        preds.append(nxt_safe)
        hist = torch.cat([hist[:, 1:], nxt_safe.unsqueeze(1)], dim=1)
        s_next = s_v * COS_D + c_v * SIN_D
        c_next = c_v * COS_D - s_v * SIN_D
        s_v, c_v = s_next, c_next

    return torch.stack(preds, dim=1)


def _check_finite_grads(model: MonsoonResNet) -> None:
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise FloatingPointError(f"Non-finite gradient detected in {name}")


def train_stage(
    model: MonsoonResNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: MonsoonRolloutLoss,
    static_t: torch.Tensor,
    coord_t: torch.Tensor,
    horizon: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    tp_min_norm: float,
    t2m_min_norm: float,
    t2m_max_norm: float,
    stage_name: str,
    ckpt_dest: Path,
    model_name: str,
    model_cfg: dict,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)

    best_val = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    static_dev = static_t.to(device)
    coord_dev = coord_t.to(device)
    accum = max(1, int(grad_accum_steps))

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        cur_lr = float(optimizer.param_groups[0]["lr"])
        model.train()
        tr_sums: dict[str, float] = {}
        tr_n = 0
        n_batches = len(train_loader)

        try:
            optimizer.zero_grad(set_to_none=True)
            for b_idx, (init_norm, doy_sc, target_norm, _) in enumerate(train_loader):
                init_norm = init_norm.to(device)
                doy_sc = doy_sc.to(device)
                target_norm = target_norm.to(device)

                pred_seq = rollout_forward(
                    model, init_norm, doy_sc, static_dev, coord_dev, horizon, tp_min_norm, t2m_min_norm, t2m_max_norm
                )
                loss, comps = loss_fn.forward_with_components(pred_seq, target_norm)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at {stage_name} ep={ep}: {comps}")

                # Scale loss cleanly for gradient accumulation equivalence to larger batch
                step_scale = accum if (b_idx + accum <= n_batches) else max(1, n_batches - (b_idx // accum) * accum)
                (loss / float(step_scale)).backward()

                if ((b_idx + 1) % accum == 0) or ((b_idx + 1) == n_batches):
                    _check_finite_grads(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                bs = int(init_norm.shape[0])
                for k_c, v_c in comps.items():
                    tr_sums[k_c] = tr_sums.get(k_c, 0.0) + v_c * bs
                tr_n += bs
        except KeyboardInterrupt:
            print(f"[{stage_name}] Interrupted cleanly at ep={ep}; restoring best checkpoint.", flush=True)
            break

        scheduler.step()
        model.eval()
        va_sums: dict[str, float] = {}
        va_n = 0
        with torch.no_grad():
            for init_norm, doy_sc, target_norm, _ in val_loader:
                init_norm = init_norm.to(device)
                doy_sc = doy_sc.to(device)
                target_norm = target_norm.to(device)
                pred_seq = rollout_forward(
                    model, init_norm, doy_sc, static_dev, coord_dev, horizon, tp_min_norm, t2m_min_norm, t2m_max_norm
                )
                loss, comps = loss_fn.forward_with_components(pred_seq, target_norm)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite val loss at {stage_name} ep={ep}")
                bs = int(init_norm.shape[0])
                for k_c, v_c in comps.items():
                    va_sums[k_c] = va_sums.get(k_c, 0.0) + v_c * bs
                va_n += bs

        tr_avg = {k: round(v / max(tr_n, 1), 6) for k, v in tr_sums.items()}
        va_avg = {k: round(v / max(va_n, 1), 6) for k, v in va_sums.items()}
        va_loss = va_avg["total"]
        dt_s = round(time.perf_counter() - t0, 2)

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_torch_save(
                {
                    "model_name": model_name,
                    "model_config": model_cfg,
                    "stage": stage_name,
                    "epoch": ep,
                    "val_loss": best_val,
                    "state_dict": best_state,
                },
                ckpt_dest,
            )

        rec = {
            "epoch": ep,
            "lr": cur_lr,
            "train_loss": tr_avg["total"],
            "val_loss": va_loss,
            "train_components": tr_avg,
            "val_components": va_avg,
            "sec": dt_s,
            "is_best": is_best,
        }
        history.append(rec)
        print(
            f"[{stage_name}] ep {ep:02d}/{epochs} lr={cur_lr:.2e} | tr={tr_avg['total']:.4f} "
            f"(huber={tr_avg['base_huber']:.4f}, rain+={tr_avg['heavy_rain_extra']:.4f}, "
            f"grad={tr_avg['grad_weighted']:.4f}, var={tr_avg['var_floor_weighted']:.4f}) | "
            f"val={va_loss:.4f} | {dt_s:.1f}s {'*' if is_best else ''}",
            flush=True,
        )

    model.load_state_dict(best_state)
    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "train_samples_per_epoch": len(train_loader.dataset),
        "val_samples_per_epoch": len(val_loader.dataset),
        "history": history,
    }


def main() -> None:
    default_cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description="Train MonsoonVerse-LocalResNet-v3 challenger cleanly into isolated dir.")
    ap.add_argument("--nc", default=str(DEFAULT_NC_PATH))
    ap.add_argument("--artifacts", default=str(ROOT_DIR / "artifacts" / "local_resnet_v3"))
    ap.add_argument("--split-mode", choices=["reference", "incumbent"], default=default_cfg["split_mode"])
    ap.add_argument("--hidden-dim", type=int, default=default_cfg["hidden_dim"])
    ap.add_argument("--dilations", default=",".join(str(x) for x in default_cfg["dilations"]))
    ap.add_argument("--epochs-s1", type=int, default=default_cfg["epochs_s1"])
    ap.add_argument("--epochs-s2", type=int, default=default_cfg["epochs_s2"])
    ap.add_argument("--s1-stride", type=int, default=default_cfg["s1_stride"])
    ap.add_argument("--s2-stride", type=int, default=default_cfg["s2_stride"])
    ap.add_argument("--rollout-horizon", type=int, default=default_cfg["rollout_horizon"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--s2-batch-size", type=int, default=32)
    ap.add_argument("--effective-batch-size", type=int, default=default_cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=default_cfg["lr"])
    ap.add_argument("--s2-lr-factor", type=float, default=default_cfg["s2_lr_factor"])
    ap.add_argument("--grad-weight", type=float, default=default_cfg["grad_weight"])
    ap.add_argument("--var-weight", type=float, default=default_cfg["var_weight"])
    ap.add_argument("--var-min-ratio", type=float, default=default_cfg["var_min_ratio"])
    ap.add_argument("--heavy-rain-boost", type=float, default=default_cfg["heavy_rain_boost"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=default_cfg["seed"])
    ap.add_argument("--resume", default="")
    ap.add_argument("--smoke-test", action="store_true", help="Run fast 1+1 epoch subsampled reliability check.")
    args = ap.parse_args()

    if args.smoke_test:
        args.epochs_s1 = 1
        args.epochs_s2 = 1
        args.s1_stride = 14
        args.s2_stride = 21

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(max(1, int(args.threads)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    adir = Path(args.artifacts).resolve()
    adir.mkdir(parents=True, exist_ok=True)
    t_run0 = time.perf_counter()

    archive = load_climate_archive(args.nc)
    np.savez_compressed(
        adir / "static_fields.npz",
        static=archive.static.astype(np.float32),
        channels=np.array(["land_sea_mask", "orography_m"]),
    )

    train_idx, val_idx = chronological_split_indices(archive.dates, mode=args.split_mode)
    train_phys, train_dates = archive.data[train_idx], archive.dates[train_idx]
    val_phys, val_dates = archive.data[val_idx], archive.dates[val_idx]

    build_train_climatology_366(train_phys, archive.dates_int[train_idx], adir / "climatology.npy")

    norm = Normalizer.fit(train_phys, archive.static, log1p_tp=False)
    norm.validate_round_trip(train_phys[:16])
    norm.save(adir / "norm_stats.json")
    # Also save stats.json compatible with official tooling
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

    dilations = [int(d) for d in args.dilations.split(",")]
    model_cfg = {
        "t_in": 3,
        "n_dyn": 6,
        "n_static": 2,
        "n_time": 2,
        "n_coord": 2,
        "hidden_dim": int(args.hidden_dim),
        "num_blocks": len(dilations),
        "dilations": dilations,
        "init_scale": float(default_cfg["init_scale"]),
    }
    model = MonsoonResNet(**model_cfg).to(device)
    if args.resume:
        r_ck = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(r_ck["state_dict"])
        print(f"Resumed weights from {args.resume}", flush=True)

    n_params = sum(int(p.numel()) for p in model.parameters())
    print(
        f"[LocalResNet-v3] params={n_params:,} | split={args.split_mode} "
        f"({archive.dates[train_idx[0]]}..{archive.dates[train_idx[-1]]} -> "
        f"{archive.dates[val_idx[0]]}..{archive.dates[val_idx[-1]]}) | "
        f"device={device} | threads={args.threads} | smoke={args.smoke_test}",
        flush=True,
    )

    rain_thresh_norm = float((float(default_cfg["heavy_rain_thresh_mm"]) - norm.mean[1]) / norm.std[1])
    loss_fn = MonsoonRolloutLoss(
        channel_weights=default_cfg["channel_weights"],
        grad_weight=args.grad_weight,
        var_weight=args.var_weight,
        var_min_ratio=args.var_min_ratio,
        rain_thresh_norm=rain_thresh_norm,
        heavy_rain_boost=args.heavy_rain_boost,
        huber_beta=default_cfg["huber_beta"],
    ).to(device)

    ckpt_path = adir / "model.pt"
    s1_accum = max(1, int(args.effective_batch_size) // max(1, int(args.batch_size)))
    ds_tr1 = MonsoonWindowDataset(train_phys, train_norm, train_dates, horizon=1, stride=args.s1_stride)
    ds_va1 = MonsoonWindowDataset(val_phys, val_norm, val_dates, horizon=1, stride=args.s1_stride)
    dl_tr1 = DataLoader(ds_tr1, batch_size=args.batch_size, shuffle=True)
    dl_va1 = DataLoader(ds_va1, batch_size=args.batch_size, shuffle=False)
    s1_res = train_stage(
        model, dl_tr1, dl_va1, loss_fn, static_t, coord_t,
        horizon=1, epochs=args.epochs_s1, lr=args.lr,
        weight_decay=default_cfg["weight_decay"], grad_clip=default_cfg["grad_clip"],
        tp_min_norm=tp_min_norm, t2m_min_norm=t2m_min_norm, t2m_max_norm=t2m_max_norm,
        stage_name="S1-1step", ckpt_dest=ckpt_path,
        model_name=default_cfg["model_name"], model_cfg=model_cfg, device=device,
        grad_accum_steps=s1_accum,
    )
    del dl_tr1, dl_va1, ds_tr1
    gc.collect()

    s2_res = None
    if args.epochs_s2 > 0 and args.rollout_horizon > 1:
        s2_accum = max(1, int(args.effective_batch_size) // max(1, int(args.s2_batch_size)))
        ds_tr2 = MonsoonWindowDataset(train_phys, train_norm, train_dates, horizon=args.rollout_horizon, stride=args.s2_stride)
        ds_va2 = MonsoonWindowDataset(val_phys, val_norm, val_dates, horizon=args.rollout_horizon, stride=args.s2_stride)
        dl_tr2 = DataLoader(ds_tr2, batch_size=args.s2_batch_size, shuffle=True)
        dl_va2 = DataLoader(ds_va2, batch_size=args.s2_batch_size, shuffle=False)
        s2_res = train_stage(
            model, dl_tr2, dl_va2, loss_fn, static_t, coord_t,
            horizon=args.rollout_horizon, epochs=args.epochs_s2, lr=args.lr * args.s2_lr_factor,
            weight_decay=default_cfg["weight_decay"], grad_clip=default_cfg["grad_clip"],
            tp_min_norm=tp_min_norm, t2m_min_norm=t2m_min_norm, t2m_max_norm=t2m_max_norm,
            stage_name=f"S2-{args.rollout_horizon}step", ckpt_dest=ckpt_path,
            model_name=default_cfg["model_name"], model_cfg=model_cfg, device=device,
            grad_accum_steps=s2_accum,
        )
        del dl_tr2, dl_va2, ds_tr2, ds_va2
        gc.collect()

    total_dur = round(time.perf_counter() - t_run0, 2)
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    try:
        commit_sha = subprocess.check_output(["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_sha = "unknown"

    # Derive empirical out-of-sample residual uncertainty bands on validation slice
    predictor = LocalResNetV3Predictor(artifacts_dir=adir)
    val_starts = list(range(0, len(val_phys) - 3 - 10 + 1, 14 if args.split_mode == "reference" else 15))
    resids = []
    for s in val_starts:
        doy_sc = ds_va1.doy_sc[s + 2].tolist()
        out_r = predictor.predict_rollout(val_phys[s : s + 3], doy_sc, steps=10, perturbation=None)
        resids.append(out_r - val_phys[s + 3 : s + 13])
    res_arr = np.stack(resids, axis=0)
    chs = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
    bands = []
    for k in range(10):
        rk = res_arr[:, k]
        bands.append({
            "day": k + 1,
            "channels": {
                cname: {
                    "bias": round(float(np.mean(rk[:, c])), 4),
                    "mae": round(float(np.mean(np.abs(rk[:, c]))), 4),
                    "rmse": round(float(np.sqrt(np.mean(rk[:, c] ** 2))), 4),
                    "p05": round(float(np.percentile(rk[:, c], 5)), 4),
                    "p95": round(float(np.percentile(rk[:, c], 95)), 4),
                }
                for c, cname in enumerate(chs)
            },
        })
    atomic_text_write(
        json.dumps({
            "model_name": default_cfg["model_name"],
            "is_probabilistic_ensemble": False,
            "deterministic_note": "Deterministic rollout; intervals reflect empirical out-of-sample validation residuals.",
            "num_windows": len(val_starts),
            "val_date_start": str(archive.dates[val_idx[0]]),
            "val_date_end": str(archive.dates[val_idx[-1]]),
            "leads": bands,
        }, indent=2),
        adir / "uncertainty_bands.json",
    )

    manifest = {
        "model_name": default_cfg["model_name"],
        "model_version": "local-resnet-v3-challenger" if not args.smoke_test else "local-resnet-v3-smoke",
        "is_smoke_test": bool(args.smoke_test),
        "architecture": {**model_cfg, "param_count": n_params, "norm": "none", "activation": "SiLU"},
        "dataset": {
            "identifier": "india_climate_train.nc (ERA5 via WeatherBench 2)",
            "dataset_date_range": f"{archive.dates[0]}..{archive.dates[-1]}",
            "split_mode": args.split_mode,
            "train_date_range": f"{archive.dates[train_idx[0]]}..{archive.dates[train_idx[-1]]}",
            "val_date_range": f"{archive.dates[val_idx[0]]}..{archive.dates[val_idx[-1]]}",
            "train_days": int(len(train_idx)),
            "val_days": int(len(val_idx)),
        },
        "normalization": norm.to_dict(),
        "training": {
            "seed": args.seed,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "lr_s1": args.lr,
            "lr_s2": args.lr * args.s2_lr_factor,
            "batch_size": args.batch_size,
            "epochs_s1": args.epochs_s1,
            "epochs_s2": args.epochs_s2,
            "s1_stride": args.s1_stride,
            "s2_stride": args.s2_stride,
            "rollout_horizon": args.rollout_horizon,
            "loss_weights": {
                "grad_weight": args.grad_weight,
                "var_weight": args.var_weight,
                "var_min_ratio": args.var_min_ratio,
                "heavy_rain_boost": args.heavy_rain_boost,
                "channel_weights": default_cfg["channel_weights"],
            },
            "duration_s": total_dur,
            "stage1": s1_res,
            "stage2": s2_res,
        },
        "provenance": {
            "checkpoint_path": str(ckpt_path.resolve().relative_to(ROOT_DIR.resolve())),
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_bytes": ckpt_path.stat().st_size,
            "code_commit": commit_sha,
        },
        "deterministic": True,
    }
    atomic_text_write(json.dumps(manifest, indent=2), adir / "manifest.json")
    print(f"[LocalResNet-v3] DONE in {total_dur}s | sha256={ckpt_sha[:16]} | manifest={adir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
