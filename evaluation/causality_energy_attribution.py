"""Phase 3: Deep Causality & Perturbation Energy Attribution over Protocol A 13 Shock Windows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.local_resnet_v3.dataset import DEFAULT_NC_PATH, chronological_split_indices, date_to_doy_sincos, load_climate_archive
from ml.local_resnet_v3.inference import LocalResNetV3Predictor
from monsoon_r1 import contract as C, scoring as S


def analyze_single_window(predictor, init_s, doy_sc, kind: str, box, std: np.ndarray, date_str: str) -> dict:
    base = predictor.predict_rollout(init_s, doy_sc, C.SCORE_STEPS, perturbation=None)
    shock = S.make_shock(kind, init_s[-1], std)
    init_e = float(((shock / std[:, None, None]) ** 2).sum())
    out = predictor.predict_rollout(init_s, doy_sc, C.SCORE_STEPS, perturbation=shock)

    chk = S.causality_checks(base, out, box, std)
    diff = out - base
    e = (diff / std[None, :, None, None]) ** 2  # [10, 6, 32, 36]
    near = S.near_mask(box)
    boxm = S.box_mask(box)

    tot_k = [round(float(e[k].sum()), 4) for k in range(C.SCORE_STEPS)]
    in_k = [round(float(e[k].sum(axis=0)[near].sum()), 4) for k in range(C.SCORE_STEPS)]
    out_k = [round(float(e[k].sum(axis=0)[~near].sum()), 4) for k in range(C.SCORE_STEPS)]
    late_out = float(np.mean(out_k[3:8]))
    req_ratio = 1.5 * out_k[0]
    req_floor = 0.02 * max(tot_k)

    # Spatial centroid of energy per day & distance from box center
    lat_c_box = float(C.LAT[boxm.any(axis=1)].mean())
    lon_c_box = float(C.LON[boxm.any(axis=0)].mean())
    ii, jj = np.meshgrid(C.LAT, C.LON, indexing="ij")
    centroids = []
    for k in range(C.SCORE_STEPS):
        ek = e[k].sum(axis=0)
        esum = max(float(ek.sum()), 1e-12)
        clat = float((ek * ii).sum() / esum)
        clon = float((ek * jj).sum() / esum)
        shift_deg = float(np.hypot(clat - lat_c_box, clon - lon_c_box))
        centroids.append({"day": k + 1, "lat": round(clat, 2), "lon": round(clon, 2), "dist_deg": round(shift_deg, 2)})

    fail_reasons = []
    if not chk["nonzero"]:
        fail_reasons.append("nonzero")
    if not chk["localized"]:
        fail_reasons.append("localized")
    if not chk["no_teleport"]:
        fail_reasons.append("no_teleport")
    if not chk["stable"]:
        fail_reasons.append("stable")
    if not (late_out > req_ratio):
        fail_reasons.append(f"growth_ratio({late_out:.3f}<=1.5*out0={req_ratio:.3f})")
    if not (late_out > req_floor):
        fail_reasons.append(f"energy_floor({late_out:.3f}<=0.02*max={req_floor:.3f})")

    return {
        "date": date_str,
        "month": int(date_str.split("-")[1]),
        "init_shock_energy": round(init_e, 3),
        "max_tot_energy": round(max(tot_k), 3),
        "day1_near_frac": round(chk["near_frac_day1"], 4),
        "out_e_day1": out_k[0],
        "out_e_day4": out_k[3],
        "out_e_day8": out_k[7],
        "late_out_d4_d8": round(late_out, 4),
        "req_1_5_out0": round(req_ratio, 4),
        "req_0_02_max": round(req_floor, 4),
        "far_frac_max": round(chk["far_frac_max"], 6),
        "max_abs_std": round(float(np.abs(diff / std[None, :, None, None]).max()), 4),
        "propagates": chk["propagates"],
        "localized": chk["localized"],
        "stable": chk["stable"],
        "no_teleport": chk["no_teleport"],
        "fail_reasons": fail_reasons,
        "tot_energy_d1_d10": tot_k,
        "in_near_energy_d1_d10": in_k,
        "out_energy_d1_d10": out_k,
        "per_channel_d1_d4_d8": {
            f"day_{d+1}": {cname: round(float(e[d, ci].sum()), 3) for ci, cname in enumerate(C.CHANNELS)}
            for d in (0, 3, 7)
        },
        "centroid_d1_d4_d8": [centroids[0], centroids[3], centroids[7]],
    }


def main() -> None:
    archive = load_climate_archive(DEFAULT_NC_PATH)
    _, val_idx = chronological_split_indices(archive.dates, mode="reference")
    val_data, val_dates = archive.data[val_idx], archive.dates[val_idx]

    stats = json.loads((ROOT_DIR / "artifacts" / "local_resnet_v3" / "norm_stats.json").read_text(encoding="utf-8"))
    std = np.asarray(stats["std"], dtype=np.float32)
    pred = LocalResNetV3Predictor(artifacts_dir=ROOT_DIR / "artifacts" / "local_resnet_v3")

    max_start = len(val_dates) - C.T_IN - C.SCORE_STEPS
    win_starts = list(range(0, max_start + 1, 14))
    shock_starts = win_starts[::4]  # Exact 13 shock windows of Protocol A

    warm_rows, stall_rows = [], []
    for s in shock_starts:
        init_s = val_data[s : s + C.T_IN]
        last_d = val_dates[s + C.T_IN - 1]
        doy_sc = date_to_doy_sincos(last_d).tolist()
        warm_rows.append(analyze_single_window(pred, init_s, doy_sc, "warm_ocean", C.BOX_BAY_OF_BENGAL, std, str(last_d)))
        stall_rows.append(analyze_single_window(pred, init_s, doy_sc, "monsoon_stall", C.BOX_ARABIAN_SEA, std, str(last_d)))

    out = {
        "model_name": "MonsoonVerse-LocalResNet-v3",
        "num_shock_windows": len(shock_starts),
        "warm_ocean_pass_count": sum(1 for r in warm_rows if r["propagates"]),
        "monsoon_stall_pass_count": sum(1 for r in stall_rows if r["propagates"]),
        "warm_ocean_windows": warm_rows,
        "monsoon_stall_windows": stall_rows,
    }
    dest = ROOT_DIR / "reports" / "causality_energy_attribution.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"WARM_OCEAN pass={out['warm_ocean_pass_count']}/13 | MONSOON_STALL pass={out['monsoon_stall_pass_count']}/13")
    print("--- WARM_OCEAN PER WINDOW ---")
    for r in warm_rows:
        print(f"  {r['date']} prop={str(r['propagates']):5s} | init={r['init_shock_energy']:5.1f} out0={r['out_e_day1']:5.2f} late={r['late_out_d4_d8']:6.2f} | 1.5*out0={r['req_1_5_out0']:5.2f} 0.02*max={r['req_0_02_max']:4.2f} | fails={r['fail_reasons']}")
    print("--- MONSOON_STALL PER WINDOW ---")
    for r in stall_rows:
        print(f"  {r['date']} prop={str(r['propagates']):5s} | init={r['init_shock_energy']:5.1f} out0={r['out_e_day1']:5.2f} late={r['late_out_d4_d8']:6.2f} | 1.5*out0={r['req_1_5_out0']:5.2f} 0.02*max={r['req_0_02_max']:4.2f} | fails={r['fail_reasons']}")


if __name__ == "__main__":
    main()
