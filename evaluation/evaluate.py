"""Independent evaluation runner for MonsoonVerse vs persistence and climatology."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.load_data import load_climate_dataset, load_stats
from data_pipeline.preprocess import chronological_split, compute_doy_sincos
from evaluation.climatology import fit_climatology, predict_climatology
from evaluation.metrics import evaluate_lead_metrics
from evaluation.persistence import predict_persistence
from evaluation.perturbation_tests import run_perturbation_suite
from evaluation.stability_tests import check_rollout_validity
from ml.inference import MonsoonPredictor


def run_full_evaluation(stride: int = 15, steps: int = 10) -> dict:
    root = Path(__file__).resolve().parents[1]
    ds = load_climate_dataset()
    stats = load_stats(root / "artifacts" / "stats.json")
    std = np.asarray(stats["std"], dtype=np.float32)

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
    max_start = len(val_data) - t_in - steps + 1
    starts = list(range(0, max_start, stride))

    preds, trues, persists, clims = [], [], [], []
    inf_times_ms = []
    validity_failures = 0

    for s in starts:
        init = val_data[s : s + t_in]
        doy = val_doy[s + t_in - 1]
        true_w = val_data[s + t_in : s + t_in + steps]

        t0 = time.perf_counter()
        pred_w = predictor.predict_rollout(init, doy, steps=steps, perturbation=None)
        inf_times_ms.append((time.perf_counter() - t0) * 1000.0)

        v = check_rollout_validity(pred_w, steps=steps)
        if not v["valid"]:
            validity_failures += 1

        pers_w = predict_persistence(init, steps=steps)
        last_date = val_times[s + t_in - 1].date()
        clim_w = predict_climatology(clim, last_date, steps=steps)

        preds.append(pred_w)
        trues.append(true_w)
        persists.append(pers_w)
        clims.append(clim_w)

    preds_arr = np.stack(preds, axis=0)
    trues_arr = np.stack(trues, axis=0)
    pers_arr = np.stack(persists, axis=0)
    clim_arr = np.stack(clims, axis=0)

    skill_report = evaluate_lead_metrics(preds_arr, trues_arr, pers_arr, clim_arr)

    # Pick a monsoon-season validation window (July/August) for perturbation causality
    monsoon_idx = next(
        (i for i, s in enumerate(starts) if val_times[s + t_in - 1].month in (7, 8)),
        len(starts) // 2,
    )
    ms_start = starts[monsoon_idx]
    pert_report = run_perturbation_suite(
        predictor.predict_rollout,
        val_data[ms_start : ms_start + t_in],
        val_doy[ms_start + t_in - 1].tolist(),
        std=std,
        steps=steps,
    )

    # Save 3 representative validation windows (winter, monsoon, post-monsoon) for dashboard
    sample_indices = []
    for target_month in (7, 1, 10):
        idx = next((s for s in starts if val_times[s + t_in - 1].month == target_month), starts[0])
        sample_indices.append(idx)
    sample_init = np.stack([val_data[s : s + t_in] for s in sample_indices], axis=0).astype(np.float32)
    sample_doy = np.stack([val_doy[s + t_in - 1] for s in sample_indices], axis=0).astype(np.float32)
    np.savez(root / "artifacts" / "sample_windows.npz", init_states=sample_init, doy_sincos=sample_doy)

    # Baseline-only report
    baseline_report = {
        "val_date_start": str(val_times[0].date()),
        "val_date_end": str(val_times[-1].date()),
        "num_eval_windows": len(starts),
        "day_1": {
            "rmse_persist": skill_report["leads"]["day_1"]["rmse_persist"],
            "rmse_clim": skill_report["leads"]["day_1"]["rmse_clim"],
            "csi_persist": skill_report["leads"]["day_1"]["csi_persist"],
            "csi_clim": skill_report["leads"]["day_1"]["csi_clim"],
        },
        "day_4": {
            "rmse_persist": skill_report["leads"]["day_4"]["rmse_persist"],
            "rmse_clim": skill_report["leads"]["day_4"]["rmse_clim"],
            "csi_persist": skill_report["leads"]["day_4"]["csi_persist"],
            "csi_clim": skill_report["leads"]["day_4"]["csi_clim"],
        },
        "day_8": {
            "rmse_persist": skill_report["leads"]["day_8"]["rmse_persist"],
            "rmse_clim": skill_report["leads"]["day_8"]["rmse_clim"],
            "csi_persist": skill_report["leads"]["day_8"]["csi_persist"],
            "csi_clim": skill_report["leads"]["day_8"]["csi_clim"],
        },
    }
    (root / "reports" / "baseline_results.json").write_text(json.dumps(baseline_report, indent=2), encoding="utf-8")

    full_report = {
        "model_name": "MonsoonVerse-ResCNN",
        "val_date_start": str(val_times[0].date()),
        "val_date_end": str(val_times[-1].date()),
        "num_eval_windows": len(starts),
        "validity_failures": validity_failures,
        "cpu_inference_ms_mean": float(np.mean(inf_times_ms)),
        "cpu_inference_ms_max": float(np.max(inf_times_ms)),
        "skill": skill_report,
        "perturbations": pert_report,
    }
    (root / "reports" / "validation_results.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    # Write markdown summary
    md = [
        "# MonsoonVerse Evaluation Report",
        "",
        f"- **Validation period**: {full_report['val_date_start']} to {full_report['val_date_end']} ({len(starts)} windows)",
        f"- **Validity failures**: {validity_failures} / {len(starts)}",
        f"- **CPU 10-day inference**: {full_report['cpu_inference_ms_mean']:.1f} ms mean ({full_report['cpu_inference_ms_max']:.1f} ms max)",
        f"- **Beats persistence at Day 4**: {skill_report['summary']['beats_persistence_day4']}",
        f"- **Forecast skill points (est /45)**: {skill_report['summary']['forecast_points_out_of_45']['points']:.2f}",
        f"- **Anti-blur points (est /10)**: {skill_report['summary']['anti_blur_points_out_of_10']:.2f}",
        f"- **Causality points (est /15)**: {pert_report['causality_points_out_of_15']:.2f}",
        "",
        "## Lead-Day Skill Summary",
        "",
        "| Lead | Mean SS vs Best | Mean SS vs Persist | Heavy-Rain CSI (Model / Persist / Clim) |",
        "|---|---:|---:|---|",
    ]
    for ld in ("day_1", "day_4", "day_8"):
        d = skill_report["leads"][ld]
        md.append(
            f"| {ld} | {d['mean_ss']:.4f} | {d['mean_ss_vs_persist']:.4f} | "
            f"{d['csi_model']:.4f} / {d['csi_persist']:.4f} / {d['csi_clim']:.4f} |"
        )
    md += [
        "",
        "## Perturbation Causality",
        "",
        f"- **Null shock consistent**: {pert_report['null_consistent']} (max diff = {pert_report['null_max_std_diff']:.2e} std)",
        f"- **Warm Bay of Bengal**: {pert_report['warm_ocean']}",
        f"- **Arabian Sea Stall**: {pert_report['monsoon_stall']}",
    ]
    (root / "reports" / "evaluation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return full_report


if __name__ == "__main__":
    rep = run_full_evaluation()
    print(json.dumps(rep["skill"]["summary"], indent=2))
    print(json.dumps(rep["perturbations"], indent=2))
