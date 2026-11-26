# MONSOONVERSE — SCIENTIFIC LIMITATIONS & MONSOON STALL ANALYSIS

## 1. Purpose & Non-Operational Scope
MonsoonVerse (`MonsoonVerse-LocalResNet-v3` promoted default and `MonsoonVerse-ClimResCNN` preserved incumbent) is an exploratory 1°×1° deterministic autoregressive climate digital twin trained on historical Copernicus ERA5 reanalysis (`india_climate_train.nc`, 2005–2021). It is **not** an operational weather forecasting or early-warning disaster service.

## 2. Deterministic Point Output vs Uncertainty
- Both `MonsoonResCNN` (`incumbent`) and `MonsoonResNet` (`local_resnet_v3`) produce **single deterministic trajectories** (`model.eval()`, `torch.inference_mode()`).
- Neither emits calibrated multi-member ensemble probability distributions. Displayed error bands (`p05`–`p95`, MAE, RMSE in `artifacts/incumbent_uncertainty_bands.json` and `artifacts/local_resnet_v3/uncertainty_bands.json`) are **historical out-of-sample validation residual quantiles**, not forecast-specific probabilities.
- Forecast skill decreases with lead time; in long horizons (`Day 8+`), certain smooth channels such as `t2m` may trail 15-day smoothed climatology on RMSE.

## 3. Formal `monsoon_stall` Outward-Propagation Limitation (`Phase 6`)
Under `monsoon_r1.scoring.causality_checks`, `monsoon_stall` perturbs `u10, v10` by `-40%` over the Arabian Sea box (`8°–20°N, 65°–75°E`) at `init_states[-1]` and requires `mean(out_e[3:8]) > max(1.5 * out_e[0], 0.02 * max(tot_e))`.

Side-by-side step analysis on `2021-07-09` ([reports/monsoon_stall_comparison.json](file:///C:/Users/gokul/OneDrive/Documents/projects/Yugaantar-Track2/MonsoonVerse/reports/monsoon_stall_comparison.json)):

| Metric | `ClimResCNN` (Incumbent) | `LocalResNet-v3` (Promoted) |
|---|---:|---:|
| Initial `monsoon_stall` energy | `205.611` | `205.611` |
| Peak total energy (`max_tot`) | `177.450` | `251.868` |
| Required `0.02 * max_tot` outside-box floor | `3.549` | `5.037` |
| Mean Days 4–8 outside energy (`late_out`) | `1.446` | **`4.163`** (3× stronger, reaches `5.67` on Day 10) |
| Total energy retained by Day 10 | `8.990` | **`77.354`** (8.6× higher) |
| Far-field max leak (`far_frac_max`, limit `< 0.05`) | `0.00236` | **`0.0000001`** (`no_teleport = True`) |
| Protocol A (`52`w / `13` shock windows) pass rate | `0.00%` (`0/13`) | **`7.69%` (`1/13`)** |

**Physical Root Cause:**
1. Empirical ERA5 `tr_sl` synoptic wind anomalies decorrelate with a ~2.5-day timescale (`u10 r(4)=0.211, r(7)=0.067`; `v10 r(4)=0.112, r(7)=0.027`). Because `-40%` wind perturbations start inside a single box with high total kinetic energy (`205.6`), natural synoptic damping reduces interior and advected energy across Days 4–8.
2. Notice `LocalResNet-v3` (omitting `GroupNorm` and future climatology restoring pull, using `[1, 2, 2, 1]` dilations + `spatial_variance_floor_loss`) retains `8.6x` more dynamical energy by Day 10 and lifts July `late_out` from `1.446` to `4.163`, raising Protocol A `monsoon_stall` pass rate from `0%` (`0/13`) to `7.69%` (`1/13`). However, forcing higher Day 1–8 wind-shock propagation across all seasons conflicts mathematically with 8-day deterministic RMSE skill (`exp5..exp8` Pareto curve).
3. Furthermore, deterministic impulse perturbation responses (`warm_ocean`, `monsoon_stall`, custom JSON shocks) reflect model sensitivity to initial-state perturbations, not validated counterfactual physical experiments.

## 4. Heat-Stress Proxy & Derived Stress Disclaimer
The Regional Monsoon Stress Explorer computes apparent temperature as `T_c + 0.33*(0.65*tcwv) - 0.70*|V| - 4.0`. This is an **unvalidated model-derived indicator** intended purely for comparative baseline-vs-scenario screening, never a clinical or statutory heat advisory.
