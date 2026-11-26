"""Controlled side-by-side comparison: Incumbent (ClimResCNN) vs Challenger (LocalResNet-v3).

Strictly separates Protocol A (Reference 2020-2021, 52 windows, stride 14) from
Protocol B (Incumbent 2018-08-18..2021-12-31, 82 windows, stride 15) and compares both
against Persistence and Train-Only Climatology using official monsoon_r1.scoring.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.inference import MonsoonPredictor
from ml.local_resnet_v3.dataset import (
    DEFAULT_NC_PATH,
    chronological_split_indices,
    date_to_doy_sincos,
    load_climate_archive,
)
from ml.local_resnet_v3.inference import LocalResNetV3Predictor
from ml.local_resnet_v3.train import build_train_climatology_366
from monsoon_r1 import contract as C, scoring as S


def persistence_rollout(init_states: np.ndarray, steps: int = 10) -> np.ndarray:
    last = init_states[-1:]
    return np.repeat(last, steps, axis=0).astype(np.float32)


def climatology_rollout(clim_366: np.ndarray, last_date: dt.date, steps: int = 10) -> np.ndarray:
    doy0 = last_date.timetuple().tm_yday - 1
    out = [clim_366[min((doy0 + k + 1) % 365, 365)] for k in range(steps)]
    return np.stack(out, axis=0).astype(np.float32)


def contingency_counts(pred_tp: np.ndarray, true_tp: np.ndarray, thr: float) -> dict:
    p = pred_tp >= thr
    t = true_tp >= thr
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    denom = tp + fp + fn
    csi = float(tp / denom) if denom > 0 else 0.0
    return {"thr_mm_day": float(thr), "tp_hits": tp, "fp_false_alarms": fp, "fn_misses": fn, "csi": round(csi, 5)}


def eval_predictor_on_protocol(
    predictor,
    archive,
    split_mode: str,
    stride: int,
    steps: int = C.SCORE_STEPS,
    shock_step: int = 4,
    clim_cache: np.ndarray | None = None,
    stats_std: np.ndarray | None = None,
) -> dict:
    train_idx, val_idx = chronological_split_indices(archive.dates, mode=split_mode)
    train_data, train_dates_int = archive.data[train_idx], archive.dates_int[train_idx]
    val_data, val_dates = archive.data[val_idx], archive.dates[val_idx]

    if clim_cache is None:
        clim_366 = build_train_climatology_366(
            train_data, train_dates_int, ROOT_DIR / "reports" / f"_tmp_clim_{split_mode}.npy"
        )
    else:
        clim_366 = clim_cache

    std = (
        np.asarray(stats_std, dtype=np.float32)
        if stats_std is not None
        else np.maximum(train_data.astype(np.float64).std(axis=(0, 2, 3)), 1e-6).astype(np.float32)
    )

    max_start = len(val_dates) - C.T_IN - steps
    win_starts = list(range(0, max_start + 1, stride))

    preds, trues, persists, clims = [], [], [], []
    timings_ms = []
    validity_failures = []
    per_window_d4_rmse_tp = []

    for s in win_starts:
        init_states = val_data[s : s + C.T_IN]
        last_date = val_dates[s + C.T_IN - 1]
        true_rollout = val_data[s + C.T_IN : s + C.T_IN + steps]
        doy_sc = date_to_doy_sincos(last_date).tolist()

        t0 = time.perf_counter()
        pred_rollout = predictor.predict_rollout(init_states, doy_sc, steps, perturbation=None)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

        bad = S.validity(pred_rollout, steps)
        if bad:
            validity_failures.append({"window_start": str(val_dates[s]), "errors": bad})

        preds.append(pred_rollout)
        trues.append(true_rollout)
        persists.append(persistence_rollout(init_states, steps))
        clims.append(climatology_rollout(clim_366, last_date, steps))
        per_window_d4_rmse_tp.append(
            {
                "init_end_date": str(last_date),
                "d1_tp_rmse": round(float(S.rmse_lat(pred_rollout[0:1], true_rollout[0:1])[1]), 4),
                "d4_tp_rmse": round(float(S.rmse_lat(pred_rollout[3:4], true_rollout[3:4])[1]), 4),
                "d8_tp_rmse": round(float(S.rmse_lat(pred_rollout[7:8], true_rollout[7:8])[1]), 4),
            }
        )

    preds_arr = np.stack(preds, axis=0)
    trues_arr = np.stack(trues, axis=0)
    pers_arr = np.stack(persists, axis=0)
    clim_arr = np.stack(clims, axis=0)

    lead_metrics = {}
    ss_all = []
    csi_ss_all = []
    day4_ss_persist = None

    for lead in C.LEAD_DAYS:
        idx = lead - 1
        p_d, t_d, pe_d, cl_d = preds_arr[:, idx], trues_arr[:, idx], pers_arr[:, idx], clim_arr[:, idx]
        sk = S.skill_vs_best_ref(p_d, t_d, pe_d, cl_d)
        r_pers = S.rmse_lat(pe_d, t_d)
        r_clim = S.rmse_lat(cl_d, t_d)
        csi_m = S.csi(p_d[:, 1], t_d[:, 1])
        csi_p = S.csi(pe_d[:, 1], t_d[:, 1])
        csi_c = S.csi(cl_d[:, 1], t_d[:, 1])
        csi_ss = S.csi_skill(csi_m, csi_p, csi_c)

        ss_all.extend([float(v) for v in sk["ss"]])
        if np.isfinite(csi_ss):
            csi_ss_all.append(float(csi_ss))
        if lead == 4:
            day4_ss_persist = [float(v) for v in sk["ss_persist"]]

        lead_metrics[f"day_{lead}"] = {
            "channels": list(C.CHANNELS),
            "rmse_model": [round(float(v), 5) for v in sk["rmse"]],
            "rmse_persist": [round(float(v), 5) for v in r_pers],
            "rmse_clim": [round(float(v), 5) for v in r_clim],
            "ss_vs_best_ref": [round(float(v), 5) for v in sk["ss"]],
            "ss_vs_persist": [round(float(v), 5) for v in sk["ss_persist"]],
            "ss_vs_clim": [round(float(v), 5) for v in sk["ss_clim"]],
            "mean_ss_vs_best_ref": round(float(np.nanmean(sk["ss"])), 5),
            "mean_ss_vs_persist": round(float(np.nanmean(sk["ss_persist"])), 5),
            "csi_10mm": float(csi_m),
            "csi_persist": float(csi_p),
            "csi_clim": float(csi_c),
            "csi_skill": float(csi_ss),
            "contingency": [
                contingency_counts(p_d[:, 1], t_d[:, 1], thr) for thr in (5.0, 10.0, 20.0)
            ],
        }

    day8_ratios = np.stack([S.variance_ratio(preds_arr[i, 7], trues_arr[i, 7]) for i in range(len(win_starts))])
    anti_blur_pts = float(S.anti_blur_points(day8_ratios))

    shock_starts = win_starts[:: max(1, shock_step)]
    warm_checks, stall_checks, null_checks = [], [], []
    shock_dates = []
    for s in shock_starts:
        init_states = val_data[s : s + C.T_IN]
        last_date = val_dates[s + C.T_IN - 1]
        shock_dates.append(str(last_date))
        doy_sc = date_to_doy_sincos(last_date).tolist()
        base = predictor.predict_rollout(init_states, doy_sc, steps, perturbation=None)
        p_w = S.make_shock("warm_ocean", init_states[-1], std)
        out_w = predictor.predict_rollout(init_states, doy_sc, steps, perturbation=p_w)
        warm_checks.append(S.causality_checks(base, out_w, C.BOX_BAY_OF_BENGAL, std))
        p_st = S.make_shock("monsoon_stall", init_states[-1], std)
        out_st = predictor.predict_rollout(init_states, doy_sc, steps, perturbation=p_st)
        stall_checks.append(S.causality_checks(base, out_st, C.BOX_ARABIAN_SEA, std))
        out_null = predictor.predict_rollout(init_states, doy_sc, steps, perturbation=np.zeros_like(init_states[-1]))
        null_checks.append(bool(S.null_consistent(base, out_null, std)))

    def _agg(c_list):
        return {
            "n_windows": len(c_list),
            "nonzero_rate": round(float(np.mean([c["nonzero"] for c in c_list])), 4),
            "localized_rate": round(float(np.mean([c["localized"] for c in c_list])), 4),
            "propagates_rate": round(float(np.mean([c["propagates"] for c in c_list])), 4),
            "no_teleport_rate": round(float(np.mean([c["no_teleport"] for c in c_list])), 4),
            "stable_rate": round(float(np.mean([c["stable"] for c in c_list])), 4),
            "mean_near_frac_day1": round(float(np.mean([c.get("near_frac_day1", 0.0) for c in c_list])), 4),
            "mean_far_frac_max": round(float(np.mean([c.get("far_frac_max", 0.0) for c in c_list])), 5),
        }

    caus_list = [S.causality_points([warm_checks[i], stall_checks[i]], bool(null_checks[i])) for i in range(len(shock_starts))]
    fc_pts = S.forecast_points(ss_all, csi_ss_all)

    return {
        "model_name": getattr(predictor, "model_name", "unknown"),
        "protocol": {
            "split_mode": split_mode,
            "train_range": f"{archive.dates[train_idx[0]]}..{archive.dates[train_idx[-1]]}",
            "val_range": f"{archive.dates[val_idx[0]]}..{archive.dates[val_idx[-1]]}",
            "num_windows": len(win_starts),
            "stride_days": stride,
            "num_shock_windows": len(shock_starts),
            "steps": steps,
        },
        "latency_ms": {
            "mean": round(float(np.mean(timings_ms)), 2),
            "p95": round(float(np.percentile(timings_ms, 95)), 2),
            "max": round(float(np.max(timings_ms)), 2),
        },
        "validity_failures": len(validity_failures),
        "leads": lead_metrics,
        "anti_blur_day8": {
            "channels": list(C.CHANNELS),
            "mean_var_ratio": [round(float(v), 4) for v in np.nanmean(day8_ratios, axis=0)],
            "anti_blur_pts_10": round(anti_blur_pts, 4),
        },
        "causality": {
            "shock_dates": shock_dates,
            "warm_ocean": _agg(warm_checks),
            "monsoon_stall": _agg(stall_checks),
            "null_consistent_rate": round(float(np.mean(null_checks)), 4),
            "mean_causality_pts_15": round(float(np.mean(caus_list)), 4),
        },
        "score_summary": {
            "forecast_rmse_pts_30": round(float(fc_pts["rmse_pts"]), 4),
            "forecast_csi_pts_15": round(float(fc_pts["csi_pts"]), 4),
            "forecast_pts_45": round(float(fc_pts["points"]), 4),
            "anti_blur_pts_10": round(anti_blur_pts, 4),
            "causality_pts_15": round(float(np.mean(caus_list)), 4),
            "total_model_pts_70": round(float(fc_pts["points"] + anti_blur_pts + np.mean(caus_list)), 4),
            "day4_beats_persistence": bool(np.nanmean(day4_ss_persist) > 0.0),
        },
        "per_window_summary": per_window_d4_rmse_tp,
    }


def build_markdown(rep: dict) -> str:
    p_a_inc = rep["protocol_a_reference_52w"]["incumbent"]
    p_a_chg = rep["protocol_a_reference_52w"]["local_resnet_v3"]
    p_b_inc = rep["protocol_b_incumbent_82w"]["incumbent"]
    p_b_chg = rep["protocol_b_incumbent_82w"]["local_resnet_v3"]
    promo = rep["promotion_decision"]

    def _row(lbl, a, b, fmt=".2f"):
        return f"| {lbl} | {format(a, fmt)} | {format(b, fmt)} |"

    lines = [
        "# MonsoonVerse: Controlled Incumbent vs LocalResNet-v3 Comparison",
        "",
        f"**Promotion Decision:** **`{promo['decision']}`** — {promo['reason']}",
        "",
        "## Protocol A — Reference Benchmark (`2020-01-01..2021-12-31`, `52` windows, stride `14`, strictly out-of-sample for BOTH models)",
        "",
        "| Metric | Incumbent (`ClimResCNN`) | Challenger (`LocalResNet-v3`) |",
        "|---|---:|---:|",
        f"| Train Split of Checkpoint | 2005-01-01..2018-08-07 | 2005-01-01..2019-12-31 |",
        _row("Forecast RMSE Skill (/30)", p_a_inc["score_summary"]["forecast_rmse_pts_30"], p_a_chg["score_summary"]["forecast_rmse_pts_30"]),
        _row("Heavy-Rain CSI Skill (/15)", p_a_inc["score_summary"]["forecast_csi_pts_15"], p_a_chg["score_summary"]["forecast_csi_pts_15"]),
        _row("Anti-Blur Day-8 (/10)", p_a_inc["score_summary"]["anti_blur_pts_10"], p_a_chg["score_summary"]["anti_blur_pts_10"]),
        _row("Perturbation Causality (/15)", p_a_inc["score_summary"]["causality_pts_15"], p_a_chg["score_summary"]["causality_pts_15"]),
        _row("**Total Model Score (/70)**", p_a_inc["score_summary"]["total_model_pts_70"], p_a_chg["score_summary"]["total_model_pts_70"]),
        _row("Day-1 CSI (>10mm)", p_a_inc["leads"]["day_1"]["csi_10mm"], p_a_chg["leads"]["day_1"]["csi_10mm"], ".4f"),
        _row("Day-4 CSI (>10mm)", p_a_inc["leads"]["day_4"]["csi_10mm"], p_a_chg["leads"]["day_4"]["csi_10mm"], ".4f"),
        _row("Day-8 CSI (>10mm)", p_a_inc["leads"]["day_8"]["csi_10mm"], p_a_chg["leads"]["day_8"]["csi_10mm"], ".4f"),
        _row("Warm BoB Propagates Rate", p_a_inc["causality"]["warm_ocean"]["propagates_rate"], p_a_chg["causality"]["warm_ocean"]["propagates_rate"], ".2%"),
        _row("Arabian Sea Stall Propagates Rate", p_a_inc["causality"]["monsoon_stall"]["propagates_rate"], p_a_chg["causality"]["monsoon_stall"]["propagates_rate"], ".2%"),
        _row("Null Consistency Rate", p_a_inc["causality"]["null_consistent_rate"], p_a_chg["causality"]["null_consistent_rate"], ".2%"),
        _row("CPU Latency Mean (ms)", p_a_inc["latency_ms"]["mean"], p_a_chg["latency_ms"]["mean"]),
        f"| Validity Failures | {p_a_inc['validity_failures']} | {p_a_chg['validity_failures']} |",
        "",
        "## Protocol B — Incumbent Project Benchmark (`2018-08-18..2021-12-31`, `82` windows, stride `15`)",
        "",
        "> Note: Windows in `2018-08-18..2019-12-31` are out-of-sample for `ClimResCNN` (`<=2018-08-07`) but inside `LocalResNet-v3`'s training period (`<=2019-12-31`). Protocol A above provides the leakage-free side-by-side benchmark.",
        "",
        "| Metric | Incumbent (`ClimResCNN`) | Challenger (`LocalResNet-v3`) |",
        "|---|---:|---:|",
        _row("Forecast Score (/45)", p_b_inc["score_summary"]["forecast_pts_45"], p_b_chg["score_summary"]["forecast_pts_45"]),
        _row("Anti-Blur (/10)", p_b_inc["score_summary"]["anti_blur_pts_10"], p_b_chg["score_summary"]["anti_blur_pts_10"]),
        _row("Causality (/15)", p_b_inc["score_summary"]["causality_pts_15"], p_b_chg["score_summary"]["causality_pts_15"]),
        _row("Total Model Score (/70)", p_b_inc["score_summary"]["total_model_pts_70"], p_b_chg["score_summary"]["total_model_pts_70"]),
        _row("Warm BoB Propagates Rate", p_b_inc["causality"]["warm_ocean"]["propagates_rate"], p_b_chg["causality"]["warm_ocean"]["propagates_rate"], ".2%"),
        _row("Arabian Sea Stall Propagates Rate", p_b_inc["causality"]["monsoon_stall"]["propagates_rate"], p_b_chg["causality"]["monsoon_stall"]["propagates_rate"], ".2%"),
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=str(DEFAULT_NC_PATH))
    ap.add_argument("--challenger-dir", default=str(ROOT_DIR / "artifacts" / "local_resnet_v3"))
    ap.add_argument("--out-json", default=str(ROOT_DIR / "reports" / "incumbent_vs_local_resnet_v3.json"))
    ap.add_argument("--out-md", default=str(ROOT_DIR / "reports" / "incumbent_vs_local_resnet_v3.md"))
    args = ap.parse_args()

    archive = load_climate_archive(args.nc)
    incumbent = MonsoonPredictor(artifacts_dir=ROOT_DIR / "artifacts")
    challenger = LocalResNetV3Predictor(artifacts_dir=args.challenger_dir)

    inc_stats = json.loads((ROOT_DIR / "artifacts" / "stats.json").read_text(encoding="utf-8"))
    chg_stats = json.loads((Path(args.challenger_dir) / "norm_stats.json").read_text(encoding="utf-8"))

    # Protocol A: 2020-2021, 52 windows, stride 14
    a_inc = eval_predictor_on_protocol(incumbent, archive, split_mode="reference", stride=14, shock_step=4, stats_std=chg_stats["std"])
    a_chg = eval_predictor_on_protocol(challenger, archive, split_mode="reference", stride=14, shock_step=4, stats_std=chg_stats["std"])

    # Protocol B: 2018-08-18..2021-12-31, 82 windows, stride 15
    b_inc = eval_predictor_on_protocol(incumbent, archive, split_mode="incumbent", stride=15, shock_step=5, stats_std=inc_stats["std"])
    b_chg = eval_predictor_on_protocol(challenger, archive, split_mode="incumbent", stride=15, shock_step=5, stats_std=inc_stats["std"])

    # Promotion gate on leakage-free Protocol A (both models out-of-sample)
    chg_tot = a_chg["score_summary"]["total_model_pts_70"]
    inc_tot = a_inc["score_summary"]["total_model_pts_70"]
    chg_fc = a_chg["score_summary"]["forecast_pts_45"]
    inc_fc = a_inc["score_summary"]["forecast_pts_45"]
    chg_warm = a_chg["causality"]["warm_ocean"]["propagates_rate"]
    inc_warm = a_inc["causality"]["warm_ocean"]["propagates_rate"]
    no_regressions = (
        a_chg["validity_failures"] == 0
        and a_chg["score_summary"]["day4_beats_persistence"]
        and a_chg["latency_ms"]["max"] < 5000.0
        and a_chg["causality"]["null_consistent_rate"] == 1.0
        and a_chg["score_summary"]["anti_blur_pts_10"] >= a_inc["score_summary"]["anti_blur_pts_10"]
        and chg_warm >= inc_warm - 0.10
    )
    promote = (chg_fc > inc_fc) and (chg_tot > inc_tot) and no_regressions

    if promote:
        decision = "PROMOTE_CHALLENGER"
        reason = f"LocalResNet-v3 outperforms incumbent cleanly on Protocol A ({chg_tot:.2f} vs {inc_tot:.2f}/70; fc {chg_fc:.2f} vs {inc_fc:.2f}/45) without dynamics/validity regressions."
    elif chg_fc > inc_fc and chg_tot > inc_tot:
        decision = "RETAIN_INCUMBENT_DEFAULT_KEEP_CHALLENGER"
        reason = (
            f"LocalResNet-v3 improves Protocol A forecast skill ({chg_fc:.2f} vs {inc_fc:.2f}/45, total {chg_tot:.2f} vs {inc_tot:.2f}/70), "
            f"but warm_ocean propagation rate ({chg_warm:.0%} vs {inc_warm:.0%}) is lower. Retaining incumbent as default and exposing challenger cleanly."
        )
    else:
        decision = "RETAIN_INCUMBENT"
        reason = f"Incumbent retains advantage ({inc_tot:.2f} vs {chg_tot:.2f}/70) on Protocol A under full promotion gate."

    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_a_reference_52w": {"incumbent": a_inc, "local_resnet_v3": a_chg},
        "protocol_b_incumbent_82w": {"incumbent": b_inc, "local_resnet_v3": b_chg},
        "promotion_decision": {"decision": decision, "promote": promote, "reason": reason},
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(build_markdown(report), encoding="utf-8")
    print(f"Comparison written -> {args.out_json} & {args.out_md} | Decision={decision}", flush=True)


if __name__ == "__main__":
    main()
