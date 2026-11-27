# MonsoonVerse Operation 70/70 — Official Experiment Matrix

## A. Protocol A Out-of-Sample Evaluation (`2020-01-01..2021-12-31`, 52 windows, unmodified `monsoon_r1.scoring`)

| Candidate | RMSE /30 | CSI /15 | Anti-blur /10 | Causality /15 | **Total /70** | Warm BoB | Monsoon Stall | All Stable | Mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Incumbent (ClimResCNN) | 13.53 | 3.51 | 10.00 | 12.84 | **39.87** | 11/13 | 0/13 | True | 41.52 |
| LocalResNet-v3 (baseline) | 26.65 | 15.00 | 10.00 | 12.84 | **64.49** | 10/13 | 1/13 | True | 48.58 |
| LocalResNet-v4 (seed=42) | 27.20 | 15.00 | 10.00 | 13.12 | **65.33** | 11/13 | 3/13 | False | 50.33 |
| LocalResNet-v4 (seed=123, PROMOTED) | 27.14 | 15.00 | 10.00 | 13.41 | **65.56** | 13/13 | 2/13 | True | 50.21 |

- **Multi-Seed Reproducibility (`LocalResNet-v4`, seeds 42 & 123):** Total = **65.44 ± 0.12 / 70** | RMSE = `27.17 ± 0.03` | Causality = `13.27 ± 0.14`.

## B. Inner Chronological Holdout Screening (`2005-2017 -> 2018-2019`, 52 windows, zero Protocol A leakage)

| Inner Experiment | RMSE /30 | CSI /15 | Anti-blur /10 | Causality /15 | **Total /70** | Warm BoB | Stall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `inner_e0_ref_loss` | 7.45 | 7.14 | 10.00 | 11.25 | **35.84** | 0/13 | 0/13 |
| `inner_e1_mse_balanced` | 9.78 | 9.22 | 10.00 | 11.25 | **40.25** | 0/13 | 0/13 |
| `inner_e2_dil1241_vort` | 10.55 | 9.57 | 10.00 | 12.55 | **42.67** | 9/13 | 0/13 |
| `inner_e3_lyapunov_ft` | 3.53 | 7.34 | 10.00 | 12.40 | **33.27** | 8/13 | 0/13 |