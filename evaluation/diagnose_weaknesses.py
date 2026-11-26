"""Diagnose Day-8 t2m/tcwv errors and monsoon_stall spatial propagation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from monsoon_r1 import contract as C, scoring as S
from ml.inference import MonsoonPredictor


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    d = json.loads((root / "evaluation" / "results" / "ten_day_forecast_metrics.json").read_text(encoding="utf-8"))
    for day in (1, 4, 8, 10):
        dd = d["days"][day - 1]
        print(f"=== Day {day} ===")
        for ch in dd["channels"]:
            print(
                f"  {ch['channel']:4s}: rmse={ch['rmse']:7.3f} mae={ch['mae']:6.3f} "
                f"bias={ch['bias']:+7.3f} corr={ch['corr']:.3f} vr={ch['spatial_var_ratio']:.3f} ss_best={ch['ss_vs_best']:+.4f}"
            )

    sw = np.load(root / "artifacts" / "sample_windows.npz")
    init = sw["init_states"][0]
    doy = sw["doy_sincos"][0].tolist()
    pred = MonsoonPredictor(root / "artifacts")
    base = pred.predict_rollout(init, doy, steps=10, perturbation=None)
    stall = S.make_shock("monsoon_stall", init[-1], pred.std)
    out_s = pred.predict_rollout(init, doy, steps=10, perturbation=stall)
    diff = out_s - base
    e = ((diff / pred.std[None, :, None, None]) ** 2).sum(axis=1)
    near = S.near_mask(C.BOX_ARABIAN_SEA)
    print("=== MONSOON_STALL ENERGY (Jul Sample Window) ===")
    for k in range(10):
        print(
            f"  Day {k+1:2d}: tot={e[k].sum():8.4f}  near={e[k][near].sum():8.4f}  "
            f"out={e[k][~near].sum():8.4f}  max_z={np.abs(diff[k]/pred.std[:,None,None]).max():.4f}"
        )


if __name__ == "__main__":
    main()
