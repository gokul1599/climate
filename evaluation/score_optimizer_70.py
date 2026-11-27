"""Phase 5: Exact Official 70/70 Evaluator using unmodified monsoon_r1.scoring."""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.compare_incumbent_vs_local_resnet_v3 import climatology_rollout, persistence_rollout
from ml.local_resnet_v3.dataset import DEFAULT_NC_PATH, date_to_doy_sincos, load_climate_archive
from ml.local_resnet_v3.train import build_train_climatology_366
from monsoon_r1 import contract as C, scoring as S


def evaluate_exact_70(
    predictor,
    archive,
    train_start: dt.date,
    train_end: dt.date,
    val_start: dt.date,
    val_end: dt.date,
    stride: int = 14,
    shock_step: int = 4,
    clim_366: np.ndarray | None = None,
    stats_std: np.ndarray | None = None,
) -> dict:
    """Evaluate a predictor on an exact [val_start..val_end] chronological slice using official monsoon_r1.scoring."""
    dates = archive.dates
    tr_idx = np.where((dates >= train_start) & (dates <= train_end))[0]
    va_idx = np.where((dates >= val_start) & (dates <= val_end))[0]
    if len(tr_idx) == 0 or len(va_idx) == 0 or int(tr_idx[-1]) >= int(va_idx[0]):
        raise ValueError(f"Invalid split: {train_start}..{train_end} vs {val_start}..{val_end}")

    train_data, train_dates_int = archive.data[tr_idx], archive.dates_int[tr_idx]
    val_data, val_dates = archive.data[va_idx], archive.dates[va_idx]

    if clim_366 is None:
        clim_366 = build_train_climatology_366(
            train_data, train_dates_int, ROOT_DIR / "reports" / f"_clim_{train_end:%Y%m%d}.npy"
        )
    if stats_std is None:
        stats_std = np.maximum(train_data.astype(np.float64).std(axis=(0, 2, 3)), 1e-6).astype(np.float32)

    steps = C.SCORE_STEPS
    max_start = len(val_dates) - C.T_IN - steps
    win_starts = list(range(0, max_start + 1, stride))

    preds, trues, persists, clims = [], [], [], []
    timings = []
    validity_errs = 0
    for s in win_starts:
        init_s = val_data[s : s + C.T_IN]
        last_d = val_dates[s + C.T_IN - 1]
        true_r = val_data[s + C.T_IN : s + C.T_IN + steps]
        doy_sc = date_to_doy_sincos(last_d).tolist()

        t0 = time.perf_counter()
        out_r = predictor.predict_rollout(init_s, doy_sc, steps, perturbation=None)
        timings.append((time.perf_counter() - t0) * 1000.0)
        if S.validity(out_r, steps):
            validity_errs += 1

        preds.append(out_r)
        trues.append(true_r)
        persists.append(persistence_rollout(init_s, steps))
        clims.append(climatology_rollout(clim_366, last_d, steps))

    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    PE = np.stack(persists, axis=0)
    CL = np.stack(clims, axis=0)

    ss_all, csi_ss_all = [], []
    lead_summary = {}
    d4_beat_persist = True
    for lead in C.LEAD_DAYS:
        k = lead - 1
        sk = S.skill_vs_best_ref(P[:, k], T[:, k], PE[:, k], CL[:, k])
        csi_m = S.csi(P[:, k, 1], T[:, k, 1])
        csi_p = S.csi(PE[:, k, 1], T[:, k, 1])
        csi_c = S.csi(CL[:, k, 1], T[:, k, 1])
        csi_ss = S.csi_skill(csi_m, csi_p, csi_c)
        ss_all.extend([float(v) for v in sk["ss"]])
        if np.isfinite(csi_ss):
            csi_ss_all.append(float(csi_ss))
        if lead == 4:
            d4_beat_persist = bool(np.nanmean(sk["ss_persist"]) > 0.0)
        lead_summary[f"day_{lead}"] = {
            "mean_ss_best": round(float(np.nanmean(sk["ss"])), 5),
            "per_ch_ss_best": {C.CHANNELS[ci]: round(float(sk["ss"][ci]), 4) for ci in range(6)},
            "csi_10mm": round(float(csi_m), 4),
        }

    day8_ratios = np.stack([S.variance_ratio(P[i, 7], T[i, 7]) for i in range(len(win_starts))])
    ab_pts = float(S.anti_blur_points(day8_ratios))
    fc = S.forecast_points(ss_all, csi_ss_all)

    shock_starts = win_starts[:: max(1, shock_step)]
    warm_chks, stall_chks, null_chks = [], [], []
    caus_pts_list = []
    for s in shock_starts:
        init_s = val_data[s : s + C.T_IN]
        last_d = val_dates[s + C.T_IN - 1]
        doy_sc = date_to_doy_sincos(last_d).tolist()
        base = predictor.predict_rollout(init_s, doy_sc, steps, perturbation=None)
        w_out = predictor.predict_rollout(init_s, doy_sc, steps, perturbation=S.make_shock("warm_ocean", init_s[-1], stats_std))
        s_out = predictor.predict_rollout(init_s, doy_sc, steps, perturbation=S.make_shock("monsoon_stall", init_s[-1], stats_std))
        n_out = predictor.predict_rollout(init_s, doy_sc, steps, perturbation=np.zeros_like(init_s[-1]))
        wc = S.causality_checks(base, w_out, C.BOX_BAY_OF_BENGAL, stats_std)
        sc = S.causality_checks(base, s_out, C.BOX_ARABIAN_SEA, stats_std)
        nc = bool(S.null_consistent(base, n_out, stats_std))
        warm_chks.append(wc)
        stall_chks.append(sc)
        null_chks.append(nc)
        caus_pts_list.append(float(S.causality_points([wc, sc], nc)))

    caus_mean = float(np.mean(caus_pts_list))
    total = float(fc["points"] + ab_pts + caus_mean)
    return {
        "split": f"{val_start}..{val_end} ({len(win_starts)}w, {len(shock_starts)} shocks)",
        "rmse_pts_30": round(float(fc["rmse_pts"]), 4),
        "csi_pts_15": round(float(fc["csi_pts"]), 4),
        "anti_blur_pts_10": round(ab_pts, 4),
        "causality_pts_15": round(caus_mean, 4),
        "total_pts_70": round(total, 4),
        "mean_ss": round(float(fc["mean_ss"]), 5),
        "mean_csi_skill": round(float(fc["mean_csi_skill"]), 5),
        "warm_prop_count": int(sum(1 for x in warm_chks if x["propagates"])),
        "stall_prop_count": int(sum(1 for x in stall_chks if x["propagates"])),
        "warm_rate": round(float(np.mean([x["propagates"] for x in warm_chks])), 4),
        "stall_rate": round(float(np.mean([x["propagates"] for x in stall_chks])), 4),
        "null_rate": round(float(np.mean(null_chks)), 4),
        "all_no_teleport": bool(all(x["no_teleport"] for x in warm_chks + stall_chks)),
        "all_stable": bool(all(x["stable"] for x in warm_chks + stall_chks)),
        "all_localized": bool(all(x["localized"] for x in warm_chks + stall_chks)),
        "validity_failures": validity_errs,
        "day4_beats_persist": d4_beat_persist,
        "latency_mean_ms": round(float(np.mean(timings)), 2),
        "latency_max_ms": round(float(np.max(timings)), 2),
        "day8_var_ratios": [round(float(v), 4) for v in np.nanmean(day8_ratios, axis=0)],
        "leads": lead_summary,
    }


def main() -> None:
    archive = load_climate_archive(DEFAULT_NC_PATH)
    # Smoke CLI verification on Protocol A for promoted LocalResNet-v3
    from ml.local_resnet_v3.inference import LocalResNetV3Predictor

    pred = LocalResNetV3Predictor(artifacts_dir=ROOT_DIR / "artifacts" / "local_resnet_v3")
    st = json.loads((ROOT_DIR / "artifacts" / "local_resnet_v3" / "norm_stats.json").read_text(encoding="utf-8"))
    res = evaluate_exact_70(
        pred,
        archive,
        dt.date(2005, 1, 1),
        dt.date(2019, 12, 31),
        dt.date(2020, 1, 1),
        dt.date(2021, 12, 31),
        stats_std=np.asarray(st["std"], dtype=np.float32),
    )
    print("Protocol A Verified Check:", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
