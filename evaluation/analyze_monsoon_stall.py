"""Phase 6 diagnostic: Compare warm_ocean vs monsoon_stall decay & propagation across Incumbent and LocalResNet-v3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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
from monsoon_r1 import contract as C, scoring as S


def inspect_shock_curve(predictor, init_states, doy_sc, kind: str, box, std: np.ndarray) -> dict:
    base = predictor.predict_rollout(init_states, doy_sc, C.SCORE_STEPS, perturbation=None)
    shock = S.make_shock(kind, init_states[-1], std)
    init_e = float(((shock / std[:, None, None]) ** 2).sum())
    pert_out = predictor.predict_rollout(init_states, doy_sc, C.SCORE_STEPS, perturbation=shock)

    chk = S.causality_checks(base, pert_out, box, std)
    diff = pert_out - base
    e = (diff / std[None, :, None, None]) ** 2  # [10, 6, 32, 36]
    near = S.near_mask(box)
    tot_k = [round(float(e[k].sum()), 4) for k in range(C.SCORE_STEPS)]
    out_k = [round(float(e[k].sum(axis=0)[~near].sum()), 4) for k in range(C.SCORE_STEPS)]
    ch_k = {
        k_idx: {cname: round(float(e[k_idx, ci].sum()), 4) for ci, cname in enumerate(C.CHANNELS)}
        for k_idx in (0, 3, 7, 9)
    }
    late_out = float(np.mean(out_k[3:8]))
    thresh_out = 0.02 * max(tot_k)
    return {
        "causality_checks": chk,
        "initial_shock_energy": round(init_e, 3),
        "max_total_energy": round(max(tot_k), 3),
        "required_0_02_threshold": round(thresh_out, 3),
        "late_out_mean_d4_to_d8": round(late_out, 3),
        "growth_ratio_vs_day1_out": round(late_out / max(out_k[0], 1e-9), 2),
        "tot_energy_d1_to_d10": tot_k,
        "out_energy_d1_to_d10": out_k,
        "channel_energy_snapshots": ch_k,
    }


def main() -> None:
    archive = load_climate_archive(DEFAULT_NC_PATH)
    _, val_idx = chronological_split_indices(archive.dates, mode="reference")
    val_data, val_dates = archive.data[val_idx], archive.dates[val_idx]

    # Pick peak July monsoon day in 2020-2021 validation slice
    jul_mask = np.array([d.month == 7 for d in val_dates[:-10]])
    jul_indices = np.where(jul_mask)[0]
    jul_indices = jul_indices[jul_indices >= 2]
    best_i = int(jul_indices[np.argmax(val_data[jul_indices, 1].mean(axis=(1, 2)))])
    init_states = val_data[best_i - 2 : best_i + 1]
    doy_sc = date_to_doy_sincos(val_dates[best_i]).tolist()

    stats = json.loads((ROOT_DIR / "artifacts" / "local_resnet_v3" / "norm_stats.json").read_text(encoding="utf-8"))
    std = np.asarray(stats["std"], dtype=np.float32)

    inc = MonsoonPredictor(artifacts_dir=ROOT_DIR / "artifacts")
    chg = LocalResNetV3Predictor(artifacts_dir=ROOT_DIR / "artifacts" / "local_resnet_v3")

    out = {
        "sample_date": str(val_dates[best_i]),
        "incumbent": {
            "model_name": "MonsoonVerse-ClimResCNN",
            "warm_ocean": inspect_shock_curve(inc, init_states, doy_sc, "warm_ocean", C.BOX_BAY_OF_BENGAL, std),
            "monsoon_stall": inspect_shock_curve(inc, init_states, doy_sc, "monsoon_stall", C.BOX_ARABIAN_SEA, std),
        },
        "local_resnet_v3": {
            "model_name": "MonsoonVerse-LocalResNet-v3",
            "warm_ocean": inspect_shock_curve(chg, init_states, doy_sc, "warm_ocean", C.BOX_BAY_OF_BENGAL, std),
            "monsoon_stall": inspect_shock_curve(chg, init_states, doy_sc, "monsoon_stall", C.BOX_ARABIAN_SEA, std),
        },
    }
    dest = ROOT_DIR / "reports" / "monsoon_stall_comparison.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", dest)
    for m_key in ("incumbent", "local_resnet_v3"):
        for sk in ("warm_ocean", "monsoon_stall"):
            d = out[m_key][sk]
            print(
                f"{m_key:16s} | {sk:13s} | prop={d['causality_checks']['propagates']} "
                f"late_out={d['late_out_mean_d4_to_d8']:.3f} vs thresh={d['required_0_02_threshold']:.3f} "
                f"ratio={d['growth_ratio_vs_day1_out']:.1f}x far={d['causality_checks']['far_frac_max']:.4f}"
            )


if __name__ == "__main__":
    main()
