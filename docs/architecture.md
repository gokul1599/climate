# MonsoonVerse Architecture

## Overview
MonsoonVerse is an offline-capable, CPU-efficient regional climate digital twin for India built on the 1°×1° ERA5-derived daily dataset (`india_climate_train.nc`, 32×36 grid, 6.5°–37.5°N, 65.5°–100.5°E).

## Pipeline
1. **Data Pipeline (`data_pipeline/`)**:
   - Loads `india_climate_train.nc` and `artifacts/stats.json`.
   - Extracts and saves static geographic fields (`land_sea_mask`, `orography_m`) into `artifacts/static_fields.npz`.
   - Splits chronologically (80% train, 10-day gap, 20% validation) to prevent temporal window leakage.
   - Constructs 3-day input windows with day-of-year `sin/cos` encoding.

2. **Machine Learning (`ml/`)**:
   - `MonsoonResCNN`: Compact 2D Residual CNN (`hidden_channels=48`, `num_blocks=5`, replicate padding, GroupNorm, SiLU) taking `[B, 22, 32, 36]` (3×6 dynamic + 2 static + 2 DOY) and predicting the next-day residual `latest_state + delta` in normalised space.
   - Trained with a curriculum multi-step autoregressive rollout loss (`ClimateRolloutLoss`, latitude-weighted Huber + MSE) up to 4-day rollouts.
   - `autoregressive_rollout`: Deterministic 10-day forecast advancing `doy_sincos` each step, enforcing physical `tp >= 0` in physical space between steps.

3. **Backend (`serve.py`, `backend/`)**:
   - FastAPI + Uvicorn server on `0.0.0.0:8000` implementing `POST /health` and `POST /simulate`.
   - Applies physical-unit perturbations exclusively to `init_states[-1]` prior to internal normalisation.
   - Validates all inputs and outputs (finite, `tp >= 0`, `t2m` in `[-40°C, +60°C]`).

4. **Evaluation (`evaluation/`) & Impact (`impact/`)**:
   - Benchmarks against persistence and training-only climatology at lead days 1, 4, and 8.
   - Evaluates heavy-rain CSI (≥10 mm/day), Day-8 spatial variance ratio, perturbation causality, and CPU inference latency.
   - Computes model-derived regional rainfall, temperature, and moisture anomaly indicators.

5. **Dashboard (`dashboard/`)**:
   - React + TypeScript + Tailwind + Recharts interface connected live to `/health`, `/simulate`, and `/api/sample-windows`.
