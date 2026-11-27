"""Phase 1: Deep Forecast Error Attribution over Protocol A (2020-2021, 52 windows)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.compare_incumbent_vs_local_resnet_v3 import climatology_rollout, persistence_rollout
from impact.regional_analysis import REGIONS, region_mask
from ml.local_resnet_v3.dataset import DEFAULT_NC_PATH, chronological_split_indices, date_to_doy_sincos, load_climate_archive
from ml.local_resnet_v3.inference import LocalResNetV3Predictor
from ml.local_resnet_v3.train import build_train_climatology_366
from monsoon_r1 import contract as C, scoring as S


def _season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF_Winter"
    if month in (3, 4, 5):
        return "MAM_PreMonsoon"
    if month in (6, 7, 8, 9):
        return "JJAS_Monsoon"
    return "OND_PostMonsoon"


def main() -> None:
    archive = load_climate_archive(DEFAULT_NC_PATH)
    train_idx, val_idx = chronological_split_indices(archive.dates, mode="reference")
    train_data, train_dates_int = archive.data[train_idx], archive.dates_int[train_idx]
    val_data, val_dates = archive.data[val_idx], archive.dates[val_idx]

    clim_p = ROOT_DIR / "artifacts" / "local_resnet_v3" / "climatology.npy"
    clim_366 = np.load(clim_p) if clim_p.is_file() else build_train_climatology_366(train_data, train_dates_int, clim_p)
    pred = LocalResNetV3Predictor(artifacts_dir=ROOT_DIR / "artifacts" / "local_resnet_v3")

    steps = C.SCORE_STEPS
    stride = 14
    max_start = len(val_dates) - C.T_IN - steps
    win_starts = list(range(0, max_start + 1, stride))

    preds, trues, persists, clims, win_meta = [], [], [], [], []
    for s in win_starts:
        init_s = val_data[s : s + C.T_IN]
        last_d = val_dates[s + C.T_IN - 1]
        true_r = val_data[s + C.T_IN : s + C.T_IN + steps]
        doy_sc = date_to_doy_sincos(last_d).tolist()
        pred_r = pred.predict_rollout(init_s, doy_sc, steps, perturbation=None)
        preds.append(pred_r)
        trues.append(true_r)
        persists.append(persistence_rollout(init_s, steps))
        clims.append(climatology_rollout(clim_366, last_d, steps))
        win_meta.append({"window_idx": len(win_meta), "init_end_date": str(last_d), "month": last_d.month, "season": _season_of(last_d.month)})

    P = np.stack(preds, axis=0)      # [52, 10, 6, 32, 36]
    T = np.stack(trues, axis=0)
    PE = np.stack(persists, axis=0)
    CL = np.stack(clims, axis=0)
    lat_w = S.lat_weights()[None, :, None]  # [1, 32, 1]

    # 1. Exact 3x6 Skill Attribution Matrix for D1, D4, D8
    matrix_3x6 = {}
    all_18_ss = []
    for lead in C.LEAD_DAYS:
        k = lead - 1
        sk = S.skill_vs_best_ref(P[:, k], T[:, k], PE[:, k], CL[:, k])
        vr = np.nanmean([S.variance_ratio(P[w, k], T[w, k]) for w in range(len(win_starts))], axis=0)
        ch_rows = {}
        for ci, cname in enumerate(C.CHANNELS):
            p_flat = P[:, k, ci].reshape(-1)
            t_flat = T[:, k, ci].reshape(-1)
            bias = float(np.mean(p_flat - t_flat))
            corr = float(np.corrcoef(p_flat, t_flat)[0, 1])
            winner_ref = "persist" if float(np.sqrt(np.mean(((PE[:, k, ci] - T[:, k, ci]) ** 2) * lat_w))) < float(np.sqrt(np.mean(((CL[:, k, ci] - T[:, k, ci]) ** 2) * lat_w))) else "clim"
            ss_b = float(sk["ss"][ci])
            all_18_ss.append(ss_b)
            ch_rows[cname] = {
                "rmse_model": round(float(sk["rmse"][ci]), 4),
                "rmse_persist": round(float(np.sqrt(np.mean(((PE[:, k, ci] - T[:, k, ci]) ** 2) * lat_w))), 4),
                "rmse_clim": round(float(np.sqrt(np.mean(((CL[:, k, ci] - T[:, k, ci]) ** 2) * lat_w))), 4),
                "best_baseline": winner_ref,
                "ss_vs_best_ref": round(ss_b, 5),
                "ss_vs_persist": round(float(sk["ss_persist"][ci]), 5),
                "ss_vs_clim": round(float(sk["ss_clim"][ci]), 5),
                "ss_gap_to_012": round(ss_b - 0.12, 5),
                "bias": round(bias, 4),
                "corr": round(corr, 4),
                "spatial_var_ratio": round(float(vr[ci]), 4),
            }
        matrix_3x6[f"day_{lead}"] = ch_rows

    # 2. All 10 leads drift & bias curve
    drift_10d = []
    for k in range(steps):
        sk = S.skill_vs_best_ref(P[:, k], T[:, k], PE[:, k], CL[:, k])
        drift_10d.append({
            "day": k + 1,
            "mean_ss_best": round(float(np.nanmean(sk["ss"])), 5),
            "per_channel_ss_best": {C.CHANNELS[ci]: round(float(sk["ss"][ci]), 4) for ci in range(6)},
            "per_channel_bias": {C.CHANNELS[ci]: round(float(np.mean(P[:, k, ci] - T[:, k, ci])), 4) for ci in range(6)},
        })

    # 3. Seasonal & Monthly breakdown (on aggregate 18-cell SS or D4/D8)
    seasonal = {}
    for s_name in ("DJF_Winter", "MAM_PreMonsoon", "JJAS_Monsoon", "OND_PostMonsoon"):
        idxs = [w for w, m in enumerate(win_meta) if m["season"] == s_name]
        ss_cells = []
        ch_means = {c: [] for c in C.CHANNELS}
        for lead in C.LEAD_DAYS:
            k = lead - 1
            sk = S.skill_vs_best_ref(P[idxs, k], T[idxs, k], PE[idxs, k], CL[idxs, k])
            for ci, cname in enumerate(C.CHANNELS):
                ss_cells.append(float(sk["ss"][ci]))
                ch_means[cname].append(float(sk["ss"][ci]))
        seasonal[s_name] = {
            "num_windows": len(idxs),
            "mean_ss_18": round(float(np.mean(ss_cells)), 5),
            "channel_mean_ss": {c: round(float(np.mean(v)), 4) for c, v in ch_means.items()},
        }

    # 4. Geographic Regional RMSE (6 Indian climate boxes across D1/D4/D8)
    regional = {}
    for rname, box in REGIONS.items():
        m = region_mask(box)
        reg_leads = {}
        for lead in C.LEAD_DAYS:
            k = lead - 1
            reg_leads[f"day_{lead}"] = {
                C.CHANNELS[ci]: {
                    "rmse": round(float(np.sqrt(np.mean((P[:, k, ci][:, m] - T[:, k, ci][:, m]) ** 2))), 4),
                    "clim_rmse": round(float(np.sqrt(np.mean((CL[:, k, ci][:, m] - T[:, k, ci][:, m]) ** 2))), 4),
                    "bias": round(float(np.mean(P[:, k, ci][:, m] - T[:, k, ci][:, m])), 4),
                }
                for ci in range(6)
            }
        regional[rname] = reg_leads

    # 5. Per-window 52 scores (sorted worst to best)
    per_win = []
    for w, meta in enumerate(win_meta):
        ss_w = []
        for lead in C.LEAD_DAYS:
            k = lead - 1
            sk_w = S.skill_vs_best_ref(P[w : w + 1, k], T[w : w + 1, k], PE[w : w + 1, k], CL[w : w + 1, k])
            ss_w.extend([float(x) for x in sk_w["ss"]])
        per_win.append({**meta, "mean_ss_best": round(float(np.nanmean(ss_w)), 4), "d4_tp_rmse": round(float(S.rmse_lat(P[w : w + 1, 3], T[w : w + 1, 3])[1]), 3)})
    per_win_sorted = sorted(per_win, key=lambda r: r["mean_ss_best"])

    out = {
        "model_name": "MonsoonVerse-LocalResNet-v3",
        "protocol_a_windows": len(win_starts),
        "aggregate_mean_ss": round(float(np.mean(all_18_ss)), 5),
        "required_ss_for_30_of_30": S.SS_REF,
        "ss_gap_to_30": round(S.SS_REF - float(np.mean(all_18_ss)), 5),
        "matrix_3x6": matrix_3x6,
        "drift_10d": drift_10d,
        "seasonal_attribution": seasonal,
        "regional_attribution": regional,
        "worst_10_windows": per_win_sorted[:10],
        "per_window_all_52": per_win,
    }
    dest_json = ROOT_DIR / "reports" / "forecast_error_attribution.json"
    dest_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Readable report
    lines = [
        "# MonsoonVerse Forecast Error Attribution (`LocalResNet-v3`, Protocol A: 52 Windows)",
        "",
        f"- **Aggregate Mean Skill (`18` lead×channel pairs):** `{out['aggregate_mean_ss']:.5f}` (Target for `30/30`: `{S.SS_REF:.5f}`, Deficit: `-{out['ss_gap_to_30']:.5f}`)",
        "",
        "## Exact `(Lead × Channel)` Skill Attribution Table (`SS vs Best Reference`)",
        "| Lead | `t2m` SS (bias) | `tp` SS (bias) | `u10` SS (bias) | `v10` SS (bias) | `msl` SS (bias) | `tcwv` SS (bias) | Lead Mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lead in C.LEAD_DAYS:
        row = matrix_3x6[f"day_{lead}"]
        vals = [f"{row[c]['ss_vs_best_ref']:+.4f} ({row[c]['bias']:+.2f})" for c in C.CHANNELS]
        lm = float(np.mean([row[c]["ss_vs_best_ref"] for c in C.CHANNELS]))
        lines.append(f"| Day {lead} | " + " | ".join(vals) + f" | **{lm:+.4f}** |")
    lines += [
        "",
        "## Seasonal Attribution (`SS vs Best Reference`)",
        "| Season | Windows | Mean SS (18) | `t2m` | `tp` | `u10` | `v10` | `msl` | `tcwv` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sn, sd in seasonal.items():
        cm = sd["channel_mean_ss"]
        lines.append(f"| {sn} | {sd['num_windows']} | {sd['mean_ss_18']:+.4f} | {cm['t2m']:+.4f} | {cm['tp']:+.4f} | {cm['u10']:+.4f} | {cm['v10']:+.4f} | {cm['msl']:+.4f} | {cm['tcwv']:+.4f} |")

    (ROOT_DIR / "reports" / "forecast_error_attribution.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT_DIR / "docs" / "FORECAST_ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE forecast_error_attribution.{json,md} & docs/FORECAST_ERROR_ANALYSIS.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
