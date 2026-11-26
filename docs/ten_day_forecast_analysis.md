# Ten-Day Forecast Analysis (`MonsoonVerse-ClimResCNN`)

Evaluated on `82` chronological validation windows (`2018-08-18` to `2021-12-31`, `stride=15`, `t_in=3`, `steps=10`), using `evaluation/ten_day_eval.py` and `monsoon_r1.scoring`. All `82 × 10` predictions are finite with `0` invalid outputs (`tp >= 0`, `t2m` in `[-40°C, +60°C]`).

## 1. Day 1–10 Summary Table

| Lead | Mean SS vs Best | Mean SS vs Persist | Mean SS vs Clim | Heavy-Rain CSI (Model / Persist / Clim) | Precision | Recall |
|---:|---:|---:|---:|---|---:|---:|
| **Day 1** | **+0.1564** | +0.1564 | +0.3868 | **0.4374** / 0.3841 / 0.2576 | 0.6742 | 0.5546 |
| **Day 2** | **+0.1341** | +0.2042 | +0.1633 | **0.3210** / 0.2788 / 0.2557 | 0.6251 | 0.3976 |
| **Day 3** | **+0.0800** | +0.2271 | +0.0800 | **0.3033** / 0.2225 / 0.2591 | 0.6085 | 0.3769 |
| **Day 4** | **+0.0441** | +0.2363 | +0.0441 | **0.3023** / 0.1897 / 0.2702 | 0.6048 | 0.3767 |
| **Day 5** | **+0.0148** | +0.2408 | +0.0148 | **0.3007** / 0.1823 / 0.2747 | 0.5944 | 0.3782 |
| **Day 6** | **-0.0077** | +0.2471 | -0.0077 | **0.2811** / 0.1859 / 0.2827 | 0.5797 | 0.3531 |
| **Day 7** | **-0.0207** | +0.2456 | -0.0207 | **0.2912** / 0.1870 / 0.2856 | 0.5778 | 0.3698 |
| **Day 8** | **-0.0292** | +0.2410 | -0.0292 | **0.2835** / 0.1896 / 0.2861 | 0.5698 | 0.3608 |
| **Day 9** | **-0.0395** | +0.2388 | -0.0395 | **0.2843** / 0.1930 / 0.2816 | 0.5653 | 0.3638 |
| **Day 10** | **-0.0492** | +0.2370 | -0.0492 | **0.2854** / 0.1904 / 0.2833 | 0.5661 | 0.3654 |

## 2. Six-Variable Breakdown (Days 1, 4, 8, 10)

| Channel | Day 1 RMSE (SS_best) | Day 4 RMSE (SS_best) | Day 8 RMSE (SS_best) | Day 10 RMSE (SS_best) | Day-8 Bias | Day-8 Var Ratio |
|---|---|---|---|---|---:|---:|
| `t2m` (K) | 0.982 (+0.0951) | 1.771 (-0.0228) | 2.098 (-0.1906) | 2.207 (-0.2715) | +0.109 | 0.975 |
| `tp` (mm/d) | 4.790 (+0.2335) | 6.232 (+0.0297) | 6.786 (+0.0058) | 7.192 (+0.0049) | -0.317 | 0.478 |
| `u10` (m/s) | 0.963 (+0.1325) | 1.483 (+0.0641) | 1.632 (+0.0125) | 1.721 (+0.0044) | +0.055 | 0.799 |
| `v10` (m/s) | 0.954 (+0.1907) | 1.441 (+0.0376) | 1.487 (+0.0012) | 1.475 (-0.0027) | -0.079 | 0.696 |
| `msl` (Pa) | 168.99 (+0.1618) | 243.54 (+0.0886) | 276.50 (+0.0240) | 292.97 (+0.0018) | -42.79 | 0.879 |
| `tcwv` (kg/m²) | 3.372 (+0.1247) | 6.059 (+0.0674) | 6.517 (-0.0281) | 6.750 (-0.0324) | -0.101 | 0.922 |

## 3. Error Growth & Day-8 Weakness Diagnosis
1. **Days 1–5 Positive Skill**: Conditioning on training-only daily climatology enables `MonsoonVerse-ClimResCNN` to beat **both** persistence and climatology continuously through **Day 5** (`+0.1564` → `+0.0148`).
2. **Why `t2m` (`-19.06%`) and `tcwv` (`-2.81%`) trail climatology at Day 8**: Both variables preserve sharp spatial structure (`vr = 0.975` for `t2m`, `0.922` for `tcwv`, `10.0/10` anti-blur) and have minimal mean bias (`+0.109 K`, `-0.101 kg/m²`). Because deterministic synoptic weather features drift slightly out-of-phase after Day 5, squared-error RMSE penalizes sharp predictions relative to the smooth climatological mean (double-penalty effect).
3. **Heavy-Rain CSI**: Beats both persistence and climatology at Days 1–5, 7, 9, and 10, with `precision ~0.57–0.67` and `recall ~0.35–0.55`.
4. **Files**: Full per-day, per-variable, and regional metrics are saved in `evaluation/results/ten_day_forecast_metrics.json` and `evaluation/results/ten_day_forecast_metrics.csv`.
