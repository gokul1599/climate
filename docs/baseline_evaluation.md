# MonsoonVerse Baseline Evaluation (Phase 1)

## 1. Reproduction Command & Split Integrity
- **Command**: `python evaluation/evaluate.py` (using `monsoon_r1.scoring` v1.0.0)
- **Validation Period**: `2018-08-18` to `2021-12-31` (`1,232` days, `82` windows at `stride=15`, `t_in=3`, `steps=10`).
- **Climatology Integrity**: Fit strictly on training split (`2005-01-01` to `2018-08-07`, `4,967` days) with 10-day gap (`2018-08-08` to `2018-08-17`). No validation leakage.

## 2. Official Skill Breakdown (Original Epoch-4 Checkpoint)

| Lead | Mean SS vs Best | Mean SS vs Persist | Mean SS vs Clim | Heavy-Rain CSI (Model / Persist / Clim) | CSI Skill |
|---|---:|---:|---:|---|---:|
| **Day 1** | **+0.1583** | +0.1583 | +0.3877 | **0.4339** / 0.3841 / 0.2576 | +0.0808 |
| **Day 4** | **-0.0184** | +0.1856 | -0.0184 | **0.3048** / 0.1897 / 0.2702 | +0.0473 |
| **Day 8** | **-0.1461** | +0.1544 | -0.1461 | **0.3207** / 0.1896 / 0.2861 | +0.0484 |

### Per-Channel `ss_vs_best`
| Channel | Day 1 | Day 4 | Day 8 |
|---|---:|---:|---:|
| `t2m` (K) | +0.0997 | **-0.0920** | **-0.3135** |
| `tp` (mm/day) | +0.2291 | +0.0159 | -0.0119 |
| `u10` (m/s) | +0.1311 | +0.0068 | -0.0970 |
| `v10` (m/s) | +0.1810 | -0.0122 | -0.1257 |
| `msl` (Pa) | +0.1788 | **-0.0510** | **-0.1149** |
| `tcwv` (kg/m²) | +0.1303 | +0.0220 | **-0.2137** |

## 3. Root-Cause Diagnosis
1. **Climatology overtakes persistence by Day 4** on all 6 channels. Because the model only trained to `horizon=3` (4 epochs) and only sees scalar `[sin(doy), cos(doy)]`, it has no spatial seasonal baseline to relax toward at Days 4–8.
2. **`t2m` (-31.35% at Day 8) and `tcwv` (-21.37% at Day 8)** drive most of the negative official RMSE skill.
3. **Summary**: `rmse_pts = 0.00 / 30.0`, `csi_pts = 8.83 / 15.0`, `anti_blur = 10.00 / 10.0`, `causality = 15.00 / 15.0`.
