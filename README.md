# MonsoonVerse — Multiverse of Monsoon

**India's Interactive Climate Digital Twin** for 10-day autoregressive weather forecasting, physical-unit perturbation experiments, and regional climate impact exploration.

---

## 1. Overview & Architecture
MonsoonVerse learns spatiotemporal atmospheric dynamics from 17 years of daily ERA5-derived observations (`2005–2021`, `32 × 36` grid over `6.5°–37.5°N, 65.5°–100.5°E`) and serves deterministic 10-day forecasts over an offline, CPU-only FastAPI backend (`serve.py`) paired with an interactive React dashboard (`dashboard/`).

- **`data_pipeline/`**: NetCDF inspection, chronological split (`80%` train / `10`-day gap / `20%` validation), sliding-window construction (`3` input days), and static field extraction (`land_sea_mask`, `orography_m`).
- **`ml/`**: Compact 2D Residual CNN (`MonsoonResCNN`, `32` hidden channels, `4` residual blocks, replicate padding, GroupNorm, SiLU) with curriculum multi-step rollout training (`ClimateRolloutLoss`) and deterministic autoregressive inference.
- **`backend/` & `serve.py`**: FastAPI + Uvicorn server implementing `POST /health` and `POST /simulate` in physical units (`K`, `mm/day`, `m/s`, `m/s`, `Pa`, `kg/m²`).
- **`evaluation/`**: Independent verification against persistence and training-only climatology at lead days 1, 4, and 8, heavy-rain CSI (`10 mm/day`), Day-8 spatial variance ratio, perturbation causality, and CPU latency.
- **`impact/`**: Model-derived regional rainfall, temperature, moisture, and heavy-rain cell-day indicators.
- **`dashboard/`**: React + TypeScript + Tailwind + Recharts UI connected live to `/health`, `/simulate`, and `/api/sample-windows`.

## 2. Dataset & Preprocessing
- **Source**: `india_climate_train.nc` (`6,209` daily steps, `2005-01-01` to `2021-12-31`, `0` gaps, `0` NaNs/Infs).
- **Dynamic Channels (exact order)**:
  0. `t2m` — 2m temperature (`K`)
  1. `tp` — daily total precipitation (`mm/day`)
  2. `u10` — 10m eastward wind (`m/s`)
  3. `v10` — 10m northward wind (`m/s`)
  4. `msl` — mean sea-level pressure (`Pa`)
  5. `tcwv` — total column water vapour (`kg/m²`)
- **Static Channels**: `land_sea_mask`, `orography_m` (`artifacts/static_fields.npz`).
- **Normalisation**: Internal z-score normalisation using `artifacts/stats.json`; all API inputs and outputs remain in physical units.

## 3. Model & Training Summary
- **Input**: `[B, 22, 32, 36]` (`3 × 6` dynamic history + `2` static + `2` broadcast `doy_sincos`).
- **Output**: Residual prediction `latest_input_norm + delta_norm` for `[B, 6, 32, 36]`.
- **Training**: Curriculum multi-step rollout (`h = 1 → 2 → 3`) with latitude- and channel-weighted `0.7 * SmoothL1 + 0.3 * MSE` loss. Saved checkpoint: `artifacts/model.pt` (`epoch 4`, `val_loss = 0.1632`).

## 4. Validation Results (`2018-08-18` to `2021-12-31`, `82` windows)
- **Validity failures**: `0 / 82` (no NaNs/Infs, `tp >= 0`, `t2m` in `[-40°C, +60°C]`).
- **CPU 10-day inference**: `69.7 ms` mean (`220.6 ms` max) — well under the `5,000 ms` limit.
- **Day-1 Skill**: `Mean SS = +0.1583` vs best baseline; Heavy-rain CSI `0.4339` (vs `0.3841` persistence, `0.2576` climatology).
- **Day-4 Skill**: Beats persistence by `+18.56%` (`Mean SS vs Persist = +0.1856`); Heavy-rain CSI `0.3048` (vs `0.1897` persistence, `0.2702` climatology).
- **Day-8 Skill**: Beats persistence by `+15.44%`; Heavy-rain CSI `0.3207` (vs `0.1896` persistence, `0.2861` climatology).
- **Anti-blur**: `10.0 / 10.0` (Day-8 spatial variance ratio within `[0.40, 2.50]`).
- **Perturbation Causality**: `15.0 / 15.0` (`warm_ocean` Day-1 near-box energy `99.7%`, `monsoon_stall` `99.9%`, outward propagation Days 4–8 without teleportation, `null` shock max diff `0.00` std).

## 5. Installation & Usage

### Python Environment
```bash
pip install -r requirements.txt
pip install monsoon_r1_kit-1.0.0-py3-none-any.whl
```

### Start the Inference Server (Offline CPU)
```bash
python serve.py --port 8000
```
- `POST /health` → `{"status": "ready", "model_name": "MonsoonVerse"}`
- `POST /simulate` → `{"init_states": [3,6,32,36], "doy_sincos": [2], "steps": 10, "perturbation": null|[6,32,36]}`

### Start the Dashboard
```bash
cd dashboard
npm install
npm run dev
```

### Run Tests & Official Validators
```bash
pytest tests -v
python -m monsoon_r1.schema_validator http://localhost:8000
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=2))"
```

## 6. Known Limitations
- Coarse `1° × 1°` (~111 km) spatial resolution smooths local convective extremes and steep orographic gradients.
- Trained on `2005–2021` ERA5-derived data; multi-day RMSE beyond Day 4 approaches climatology as predictability decays.
- Impact Explorer indicators are model-derived scenario comparisons, **not** operational flood, crop-loss, or disaster predictions.

## 7. Attribution & License
Code is licensed under the MIT License (`LICENSE`). Climate data (`india_climate_train.nc`, `stats.json`) contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al. 2020) accessed via WeatherBench 2 (Rasp et al. 2024), licensed under CC BY 4.0 (`NOTICE`).
