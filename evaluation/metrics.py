"""Evaluation metrics aligned with monsoon_r1.scoring."""
from __future__ import annotations

import numpy as np
from monsoon_r1 import contract as C, scoring as S


def evaluate_lead_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
    persists: np.ndarray,
    clims: np.ndarray,
    leads: tuple[int, ...] = (1, 4, 8),
) -> dict:
    """Compute per-lead latitude-weighted RMSE, skill vs best ref, CSI (10 mm/day), and variance ratio.

    Arrays have shape [W, steps, 6, 32, 36].
    """
    results: dict = {"leads": {}}
    ss_all, csi_ss_all, var_ratios_day8 = [], [], []

    for lead in leads:
        idx = lead - 1
        p = preds[:, idx]
        t = trues[:, idx]
        pe = persists[:, idx]
        cl = clims[:, idx]

        sk = S.skill_vs_best_ref(p, t, pe, cl)
        csi_m = S.csi(p[:, 1], t[:, 1], S.RAIN_THRESH)
        csi_p = S.csi(pe[:, 1], t[:, 1], S.RAIN_THRESH)
        csi_c = S.csi(cl[:, 1], t[:, 1], S.RAIN_THRESH)
        csi_sk = S.csi_skill(csi_m, csi_p, csi_c)

        vr = np.stack([S.variance_ratio(p[w], t[w]) for w in range(len(p))], axis=0)
        vr_mean = np.nanmean(vr, axis=0)

        if lead == 8:
            var_ratios_day8 = vr

        ss_all.append(float(np.nanmean(sk["ss"])))
        if np.isfinite(csi_sk):
            csi_ss_all.append(float(csi_sk))

        results["leads"][f"day_{lead}"] = {
            "channels": list(C.CHANNELS),
            "rmse_model": [float(x) for x in sk["rmse"]],
            "rmse_persist": [float(x) for x in S.rmse_lat(pe, t)],
            "rmse_clim": [float(x) for x in S.rmse_lat(cl, t)],
            "ss_vs_best": [float(x) for x in sk["ss"]],
            "ss_vs_persist": [float(x) for x in sk["ss_persist"]],
            "ss_vs_clim": [float(x) for x in sk["ss_clim"]],
            "mean_ss": float(np.nanmean(sk["ss"])),
            "mean_ss_vs_persist": float(np.nanmean(sk["ss_persist"])),
            "csi_model": float(csi_m) if np.isfinite(csi_m) else None,
            "csi_persist": float(csi_p) if np.isfinite(csi_p) else None,
            "csi_clim": float(csi_c) if np.isfinite(csi_c) else None,
            "csi_skill": float(csi_sk) if np.isfinite(csi_sk) else None,
            "spatial_var_ratio": [float(x) for x in vr_mean],
        }

    fc_pts = S.forecast_points(ss_all, csi_ss_all)
    ab_pts = S.anti_blur_points(var_ratios_day8) if len(var_ratios_day8) else 0.0
    results["summary"] = {
        "forecast_points_out_of_45": fc_pts,
        "anti_blur_points_out_of_10": float(ab_pts),
        "beats_persistence_day4": bool(results["leads"]["day_4"]["mean_ss_vs_persist"] > 0.0),
    }
    return results
