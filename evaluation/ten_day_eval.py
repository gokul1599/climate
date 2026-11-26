"""Complete Day 1-10 evaluation pipeline across all 6 variables and regions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from monsoon_r1 import contract as C, scoring as S

from data_pipeline.load_data import load_climate_dataset, load_stats
from data_pipeline.preprocess import chronological_split, compute_doy_sincos
from evaluation.climatology import fit_climatology, predict_climatology
from evaluation.persistence import predict_persistence
from impact.regional_analysis import REGIONS, region_mask
from ml.inference import MonsoonPredictor


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = np.sqrt((af ** 2).sum() * (bf ** 2).sum())
    return float((af * bf).sum() / denom) if denom > 1e-12 else 0.0


def run_ten_day_evaluation(stride: int = 15, steps: int = 10) -> dict:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_climate_dataset()
    data = ds["data"]
    times = pd.to_datetime(ds["time"])
    doy_sc = compute_doy_sincos(ds["time"])
    tr_sl, va_sl = chronological_split(len(data), train_frac=0.8, gap_days=10)
    clim = fit_climatology(data[tr_sl], ds["time"][tr_sl])

    val_data = data[va_sl]
    val_times = times[va_sl]
    val_doy = doy_sc[va_sl]

    predictor = MonsoonPredictor(root / "artifacts")
    t_in = 3
    starts = list(range(0, len(val_data) - t_in - steps + 1, stride))

    preds, trues, persists, clims = [], [], [], []
    for s in starts:
        init = val_data[s : s + t_in]
        doy = val_doy[s + t_in - 1]
        preds.append(predictor.predict_rollout(init, doy, steps=steps, perturbation=None))
        trues.append(val_data[s + t_in : s + t_in + steps])
        persists.append(predict_persistence(init, steps=steps))
        clims.append(predict_climatology(clim, val_times[s + t_in - 1].date(), steps=steps))

    p_arr = np.stack(preds, axis=0)    # [W, 10, 6, 32, 36]
    t_arr = np.stack(trues, axis=0)
    pe_arr = np.stack(persists, axis=0)
    cl_arr = np.stack(clims, axis=0)

    days_report = []
    csv_rows = []

    for d_idx in range(steps):
        day_num = d_idx + 1
        p = p_arr[:, d_idx]
        t = t_arr[:, d_idx]
        pe = pe_arr[:, d_idx]
        cl = cl_arr[:, d_idx]

        sk = S.skill_vs_best_ref(p, t, pe, cl)
        rmse_p = S.rmse_lat(pe, t)
        rmse_c = S.rmse_lat(cl, t)

        # Precipitation (channel 1) metrics at 10 mm/day
        ptp = p[:, 1] > S.RAIN_THRESH
        ttp = t[:, 1] > S.RAIN_THRESH
        hits = int((ptp & ttp).sum())
        misses = int((~ptp & ttp).sum())
        fa = int((ptp & ~ttp).sum())
        csi_m = hits / (hits + misses + fa) if (hits + misses + fa) > 0 else float("nan")
        prec = hits / (hits + fa) if (hits + fa) > 0 else float("nan")
        rec = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
        csi_p = S.csi(pe[:, 1], t[:, 1], S.RAIN_THRESH)
        csi_c = S.csi(cl[:, 1], t[:, 1], S.RAIN_THRESH)
        csi_sk = S.csi_skill(csi_m, csi_p, csi_c)

        ch_list = []
        for c_idx, cname in enumerate(C.CHANNELS):
            pc = p[:, c_idx]
            tc = t[:, c_idx]
            err = pc - tc
            mae = float(np.abs(err).mean())
            bias = float(err.mean())
            corr = _pearson(pc, tc)
            sp_corrs = [_pearson(pc[w], tc[w]) for w in range(len(pc))]
            vr = [
                float(pc[w].var() / max(float(tc[w].var()), 1e-30))
                for w in range(len(pc))
            ]
            invalid_cnt = int((~np.isfinite(pc)).sum())
            if c_idx == 1:
                invalid_cnt += int((pc < -1e-6).sum())

            ch_entry = {
                "channel": cname,
                "rmse": float(sk["rmse"][c_idx]),
                "mae": mae,
                "bias": bias,
                "corr": corr,
                "spatial_corr_mean": float(np.mean(sp_corrs)),
                "spatial_var_ratio": float(np.mean(vr)),
                "rmse_persist": float(rmse_p[c_idx]),
                "rmse_clim": float(rmse_c[c_idx]),
                "ss_vs_persist": float(sk["ss_persist"][c_idx]),
                "ss_vs_clim": float(sk["ss_clim"][c_idx]),
                "ss_vs_best": float(sk["ss"][c_idx]),
                "valid_samples": int(len(pc)),
                "invalid_count": invalid_cnt,
            }
            ch_list.append(ch_entry)
            csv_rows.append(
                {
                    "day": day_num,
                    **ch_entry,
                    "tp_csi_model": float(csi_m) if c_idx == 1 else "",
                    "tp_csi_persist": float(csi_p) if c_idx == 1 else "",
                    "tp_csi_clim": float(csi_c) if c_idx == 1 else "",
                    "tp_csi_skill": float(csi_sk) if c_idx == 1 else "",
                    "tp_precision": float(prec) if c_idx == 1 else "",
                    "tp_recall": float(rec) if c_idx == 1 else "",
                }
            )

        days_report.append(
            {
                "day": day_num,
                "mean_ss_vs_best": float(np.mean(sk["ss"])),
                "mean_ss_vs_persist": float(np.mean(sk["ss_persist"])),
                "mean_ss_vs_clim": float(np.mean(sk["ss_clim"])),
                "precip_10mm": {
                    "csi_model": float(csi_m),
                    "csi_persist": float(csi_p),
                    "csi_clim": float(csi_c),
                    "csi_skill": float(csi_sk),
                    "precision": float(prec),
                    "recall": float(rec),
                    "pred_event_freq": float(ptp.mean()),
                    "true_event_freq": float(ttp.mean()),
                },
                "channels": ch_list,
            }
        )

    # Regional RMSE breakdown at Day 1, Day 4, Day 8, Day 10
    regional_rmse = {}
    for rname, box in REGIONS.items():
        m = region_mask(box)
        regional_rmse[rname] = {}
        for d in (1, 4, 8, 10):
            diff = p_arr[:, d - 1][:, :, m] - t_arr[:, d - 1][:, :, m]
            regional_rmse[rname][f"day_{d}"] = {
                cname: float(np.sqrt((diff[:, ci] ** 2).mean()))
                for ci, cname in enumerate(C.CHANNELS)
            }

    out = {
        "model_name": "MonsoonVerse-ClimResCNN",
        "val_date_start": str(val_times[0].date()),
        "val_date_end": str(val_times[-1].date()),
        "num_windows": len(starts),
        "days": days_report,
        "regional_rmse": regional_rmse,
    }
    (out_dir / "ten_day_forecast_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    with (out_dir / "ten_day_forecast_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("Saved 10-day metrics to evaluation/results/")
    for d in days_report:
        print(
            f"Day {d['day']:2d}: SS_best={d['mean_ss_vs_best']:+.4f} | "
            f"SS_pers={d['mean_ss_vs_persist']:+.4f} | SS_clim={d['mean_ss_vs_clim']:+.4f} | "
            f"CSI={d['precip_10mm']['csi_model']:.4f}"
        )
    return out


if __name__ == "__main__":
    run_ten_day_evaluation()
