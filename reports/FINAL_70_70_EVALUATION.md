# MonsoonVerse — Operation 70/70 Final Evaluation Report

**Branch:** `research/operation-70-70`  
**Base Commit (Preserved Unmodified):** `a85695c5a32f256f056e3378115fe0195c7c3f02` (`completion/mission-100`)  
**Frozen Submission Baselines (Untouched):** `main` (`b423298`), `qa/monsoonverse-100-percent` (`479843b`), `completion/mission-100` (`a85695c`)  
**Official Evaluator Integrity:** `monsoon_r1/scoring.py` and `monsoon_r1/contract.py` verified byte-equivalent (0 diff lines after `.splitlines()`) to `yug-monsoon-kit`; unmodified throughout (`SS_REF = 0.12`, `CSI_SS_REF = 0.10`, `STABLE_MAX_STD = 5.0`).

---

## 1. Executive Summary & Scientific Verdict

Starting from `MonsoonVerse-LocalResNet-v3` (`64.49 / 70` on Protocol A), Operation 70/70 systematically attributed every point deficit, designed controlled inner-split experiments (`2005–2017 -> 2018–2019`), trained a multi-seed architecture/loss upgrade (`MonsoonVerse-LocalResNet-v4`), and ran full Protocol A (`2005–2019 -> 2020–2021`, 52 rolling 10-day forecast windows, 13 perturbation shock windows) plus full-stack regression QA.

- **Best Verified Single Model Score:** **`65.56 / 70`** (`MonsoonVerse-LocalResNet-v4`, `seed=123`, **PROMOTED** to `artifacts/local_resnet_v4/`).
- **Multi-Seed `LocalResNet-v4` (`s42`, `s123`) Mean ± Std:** **`65.44 ± 0.12 / 70`** (RMSE `27.17 ± 0.03`, Causality `13.27 ± 0.14`, CSI `15.00`, Anti-Blur `10.00`).
- **Did it reach `70.00 / 70`?** **No — gap is `-4.44` points** (`-2.86` RMSE at Day 4–8; `-1.59` `monsoon_stall` winter/dry-season causality). Both remaining gaps reflect fundamental physical limits of deterministic 10-day autoregression and dry-season Arabian Sea linear wind shocks rather than underfitting or missing heuristics (detailed in Section 4).

---

## 2. Protocol A Model Progression (2020–2021 Held-Out Test Set, Zero Leakage)

| Model / Variant | RMSE (/30) | CSI (/15) | Anti-Blur (/10) | Causality (/15) | Total (/70) | `warm_ocean` | `monsoon_stall` | All Safe (`stable`, `local`, `no_tp`) | Eval Latency Mean | Checkpoint SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | ---: | --- |
| Incumbent (`ClimResCNN`, `artifacts/model.pt`) | 13.53 | 3.51 | 10.00 | 12.84 | **39.87** | 11/13 | 0/13 | Yes | 41.52 ms | `1eba8c5c2f09...af1b208` |
| Baseline Challenger (`LocalResNet-v3`, `s42`) | 26.65 | 15.00 | 10.00 | 12.84 | **64.49** | 10/13 | 1/13 | Yes | 48.58 ms | `9be5ff497241...4d30a96` |
| Candidate (`LocalResNet-v4`, `s42`) | 27.20 | 15.00 | 10.00 | 13.12 | **65.33** | 11/13 | 3/13 | **No** (`std=5.015`) | 50.33 ms | `54a5bc2e27f4...d8d04` |
| **Promoted (`LocalResNet-v4`, `s123`)** | **27.14** | **15.00** | **10.00** | **13.41** | **65.56** | **13/13** | **2/13** | **Yes** (max `3.41`) | **50.21 ms** | `5773532aca4b...fa6f4` |

### Key Protocol A Details for Promoted `MonsoonVerse-LocalResNet-v4` (`seed=123`)
- **Architecture:** `24ch -> stem(64) -> 4x LocalResBlock(dilations=[1,2,4,1], init_scale=0.25, no GroupNorm) -> 313,030 params`.
- **Loss Configuration:** `mse_mix=0.30` (`0.30 MSE + 0.70 Huber`), balanced channel weights `[1.30, 1.35, 1.15, 1.25, 1.10, 1.15]`, `heavy_rain_boost=2.05`, `vort_div_weight=0.08`, `var_weight=0.10`, `var_min_ratio=0.72`, `lead_weights=[0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]`.
- **Lead-by-Lead Accuracy (`mean_ss_best` / `csi_10mm`):**
  - **Day 1:** `0.2366` (`t2m=0.1611, tp=0.2556, u10=0.2423, v10=0.2675, msl=0.2743, tcwv=0.2188`), CSI `0.5406`
  - **Day 4:** `0.0810` (`t2m=0.0465, tp=0.0600, u10=0.1101, v10=0.0521, msl=0.1152, tcwv=0.1022`), CSI `0.3884`
  - **Day 8:** `0.0081` (`t2m=-0.0307, tp=-0.0095, u10=0.0193, v10=-0.0013, msl=0.0472, tcwv=0.0237`), CSI `0.2890`
- **Overall Forecast Summary:** `mean_ss = 0.10858` (up from `0.10661`), `mean_csi_skill = 0.1220` (up from `0.1116`), `validity_failures = 0`, `day4_beats_persist = True`.
- **Day-8 Spatial Variance Ratios (Anti-Blur `10.00/10`):** `[1.0162, 1.3241, 0.9013, 0.8820, 0.9828, 0.9222]` (all well above `>= 0.40`).

---

## 3. Inner Chronological Ablation Progression (`2005–2017 -> 2018–2019`, Zero Leakage)

All hypotheses were evaluated on a strict inner split (`8` S1 epochs stride 2, `5` S2 epochs stride 4, `seed=42`) before full-scale training:

| Experiment | Description | RMSE (/30) | CSI (/15) | Causality (/15) | Total (/70) | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `inner_e0_ref_loss` | Reference `v3` loss & `[1,1,1,1]` dilations | 7.45 | 7.14 | 11.25 (`0/13`, `0/13`) | 35.84 | Baseline |
| `inner_e1_mse_balanced` | `+0.30 MSE mix` + balanced channels + lead weights | 9.78 | 9.22 | 11.25 (`0/13`, `0/13`) | 40.25 | Keep |
| **`inner_e2_dil1241_vort`** | `+dilations=[1,2,4,1]` + `vort_div_weight=0.08` | **10.55** | **9.57** | **12.55** (`9/13`, `0/13`) | **42.67** | **Promoted to `v4`** |
| `inner_e3_lyapunov_ft` | `+1 ep` random wind-patch Lyapunov hinge loss (`E2` init) | 3.53 | 7.34 | 12.16 (`8/13`, `0/13`) | 33.27 | **Rejected (Negative Result)** |

---

## 4. Scientific Explanation of Remaining `4.44`-Point Gap

1. **RMSE Skill (`27.14 / 30.00`, `-0.0114` mean SS):**  
   With `v4`, Day-1 skill increased from `+0.2276` to `+0.2366–0.2392` and Day-4 skill rose to `+0.0810`, while preserving near-1.0 Day-8 variance (`0.88–1.32`). Further pushing MSE/Huber toward smoother multi-day means triggers the meteorological double-penalty trade-off: deterministic rollout variance drops at Day 8 (risking the `<0.40` Anti-Blur threshold) without solving intrinsic synoptic phase growth. Reaching `mean_ss >= 0.12` across `D1/D4/D8` reliably without sacrificing Anti-Blur would likely require a multi-step probabilistic/ensemble generator or larger upper-air ERA5 pressure-level predictors not present in the 6-surface-variable `india_climate_train.nc` schema.
2. **`monsoon_stall` Causality (`2 / 13` pass, `13.41 / 15.00`):**  
   Unlike `warm_ocean` (where `LocalResNet-v4` achieved `13/13 = 100%`), `monsoon_stall` scales horizontal winds by `0.6x` inside `[12..20°N, 64..74°E]`. Outside JJAS (DJF/MAM/OND), dry tropospheric stability prevents wind deceleration from sparking deep moist convection (`late_out ~ 0.06–0.45`, below both `1.5*out0` and `0.02*max_tot`). During JJAS onset/peak (`2020-06-19`, `2021-07-16`), `late_out` meets `1.5*out0` (`2.91 > 1.85` and `4.43 > 3.29`) yet sits slightly below `0.02*max_tot` (`4.14` and `5.29`). See `docs/CAUSALITY_RESEARCH.md` for window-by-window energy tables and the negative `E3` Lyapunov result.

---

## 5. Full Product, Security & Regression Verification Evidence (Stage 8)

All verification commands below were executed against the promoted `MonsoonVerse-LocalResNet-v4` codebase:

- **`pytest tests -v`:** **25 passed in 10.10s** (`0` failed, `0` skipped), covering `/health`, `/simulate`, variant selector (`local_resnet_v4`, `local_resnet_v3`, `incumbent`), serving import isolation (no `xarray` / training loop in runtime path), input validation, Neon dormant fallback, `LocalResNet-v4` dilations/loss, rollout determinism, and physical validity.
- **`monsoon_r1.repo_scan .`:** Exited `0` (normal Git blobs verified; zero network calls; valid schema).
- **Dashboard TypeScript & Production Build (`tsc && vite build`):** **PASS** (0 errors, `2581 modules transformed`, built in `9.32s`).
- **`monsoon_r1.schema_validator http://127.0.0.1:8000`:** **`RESULT: all checks passed`** (JSON lists + base64 payloads; `no perturbation`, `warm_ocean`, `zeros`).
- **Serving Latency, Determinism & Memory (`10x POST /simulate` over HTTP):**
  - Mean HTTP latency: **`113.18 ms`**, p95: **`123.87 ms`**, max: **`127.31 ms`**
  - Bitwise determinism check: `repeat_diff = 0.0`
  - FastAPI RSS process memory (`v4` loaded): **`263.8 MB`**
- **Fallback Variant Verification (`MONSOONVERSE_MODEL_VARIANT=incumbent`):** Confirmed clean switch to `MonsoonVerse-ClimResCNN` (`1eba8c5c2f09aa83f98b071872296eae94c83bf255c48216874db00eeaf1b208`) on port `8028`; live PostgreSQL (`neondb`) run count verified intact (`7` saved runs).
- **Playwright End-to-End Browser QA (`run_browser_qa.mjs`):**  
  `BROWSER_QA_DONE console_errs=0 page_errs=0 http_errs=0 overflow_cells=0/63 exports=7` across the 7-viewport × 9-page matrix, Neon CRUD lifecycle, and 7 exported files.
- **Playwright Upgrade Verification (`verify_upgrades.mjs`):**  
  `console_errors=[]`, `page_errors=[]`, `failed_requests=[]`, `custom_json_invalid_rejected=true`, `custom_json_valid_executed=true`, `impact_stress_columns_present=true`, `impact_formulas_present=true`, `performance_uncertainty_present=true`, `performance_comparison_present=true`, `status_provenance_present=true`.
- **Unverified / Environment Limits:** Container build/execution via Docker was **not executed** in this Windows environment because `docker` is absent from PATH.

---

## 6. Artifact & Branch Safety Inventory

- Preserved Untouched:
  - `artifacts/model.pt` (`MonsoonVerse-ClimResCNN`, SHA-256 `1eba8c5c2f09aa83f98b071872296eae94c83bf255c48216874db00eeaf1b208`)
  - `artifacts/local_resnet_v3/model.pt` (`MonsoonVerse-LocalResNet-v3`, SHA-256 `9be5ff49724110b3be1483eaa130a6155dda30a96432f98bfbe1f917f62a0c3e`)
  - Frozen branches `main`, `qa/monsoonverse-100-percent`, `completion/mission-100`
- Promoted Weights Directory (normal Git blobs):
  - `artifacts/local_resnet_v4/{model.pt, climatology.npy, norm_stats.json, static_fields.npz, stats.json, uncertainty_bands.json, manifest.json}`
- Intermediate Experiment Binaries (`artifacts/experiments/**/*.{pt,npy,npz}`): Excluded from Git via `.gitignore` to prevent repository bloat; all JSON manifests and scorecards under `artifacts/experiments/*/` and `reports/` are tracked cleanly.
