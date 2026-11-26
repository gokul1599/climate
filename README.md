# MonsoonVerse — Multiverse of Monsoon

**India's Interactive Climate Digital Twin** for 10-day autoregressive weather forecasting, physical-unit perturbation experiments, and regional climate impact exploration.

---

## 1. Overview & Architecture
MonsoonVerse learns spatiotemporal atmospheric dynamics from 17 years of daily ERA5-derived observations (`2005–2021`, `32 × 36` grid over `6.5°–37.5°N, 65.5°–100.5°E`) and serves deterministic 10-day forecasts over an offline, CPU-only FastAPI backend (`serve.py`) paired with an interactive React dashboard (`dashboard/`).

- **`data_pipeline/`**: NetCDF inspection, chronological split (`80%` train / `10`-day gap / `20%` validation), sliding-window construction (`3` input days), and static field extraction (`land_sea_mask`, `orography_m`).
- **`ml/`**: Compact 2D Residual CNN (`MonsoonResCNN`, `32` hidden channels, `5` residual blocks, replicate padding, GroupNorm, SiLU) conditioned on training-only daily climatology (`28` input channels), trained with curriculum multi-step rollout (`h = 1 → 4 → 8`).
- **`backend/` & `serve.py`**: FastAPI + Uvicorn server implementing `POST /health` and `POST /simulate` in physical units (`K`, `mm/day`, `m/s`, `m/s`, `Pa`, `kg/m²`).
- **`evaluation/`**: Independent verification against persistence and training-only climatology at lead days 1, 4, and 8, heavy-rain CSI (`10 mm/day`), Day-8 spatial variance ratio, perturbation causality, and CPU latency.
- **`impact/`**: Model-derived regional rainfall, temperature, moisture, and heavy-rain cell-day indicators.
- **`dashboard/`**: React + TypeScript + Tailwind + Recharts UI connected live to `/health`, `/simulate`, and `/api/sample-windows`.

## 2. Dataset & Preprocessing
- **Source**: `india_climate_train.nc` (`6,209` daily steps, `2005-01-01` to `2021-12-31`, `0` gaps, `0` NaNs/Infs; placed at `../official-kit/release/india_climate_train.nc` for training, excluded from inference repo).
- **Dynamic Channels (exact order)**: `0: t2m` (`K`), `1: tp` (`mm/day`), `2: u10` (`m/s`), `3: v10` (`m/s`), `4: msl` (`Pa`), `5: tcwv` (`kg/m²`).
- **Static & Climatology Assets**: `artifacts/static_fields.npz` (`land_sea_mask`, `orography_m`), `artifacts/stats.json`, and training-only `artifacts/climatology.npz` (`[366, 6, 32, 36]`).

## 3. Validation Results (`2018-08-18` to `2021-12-31`, `82` windows)
- **Validity failures**: `0 / 82` (no NaNs/Infs, `tp >= 0`, `t2m` in `[-40°C, +60°C]`).
- **CPU 10-day inference**: `73.5 ms` pure rollout mean (~`2.1 s` HTTP JSON roundtrip, well under `5,000 ms`).
- **Day-1 Skill**: `Mean SS vs Best = +0.1564`; Heavy-rain CSI `0.4374` (vs `0.3841` persistence, `0.2576` climatology).
- **Day-4 Skill (Hard Gate)**: `Mean SS vs Best = +0.0441`, `Mean SS vs Persist = +0.2363`; Heavy-rain CSI `0.3023` (vs `0.1897` persistence, `0.2702` climatology).
- **Day-8 Skill**: `Mean SS vs Best = -0.0292`, `Mean SS vs Persist = +0.2410`; Heavy-rain CSI `0.2835`.
- **Anti-blur**: `10.0 / 10.0` (Day-8 spatial variance ratio within `[0.40, 2.50]`).
- **Perturbation Causality**: `13.13 / 15.0` (`warm_ocean` Day-1 near-box energy `99.85%` + propagates Days 4–8; `monsoon_stall` `99.93%` localized; `null` shock `0.00` std diff).

## 4. Installation & Usage

### Python Environment
```bash
pip install -r requirements.txt
pip install monsoon_r1_kit-1.0.0-py3-none-any.whl
```

### Start the Offline CPU Inference Server
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

## 5. Known Limitations & Attribution
- Coarse `1° × 1°` (~111 km) spatial resolution smooths local convective extremes and steep orographic gradients.
- `t2m` (`-19.1%`) and `tcwv` (`-2.8%`) still trail climatology at Day 8; hidden post-`2023-01-10` judge windows are not available locally.
- Impact Explorer indicators are model-derived scenario comparisons, **not** operational flood, crop-loss, or disaster predictions.
- Technical Brief: [brief.pdf](brief.pdf). Code: MIT (`LICENSE`). Data: modified Copernicus ERA5 via WeatherBench 2, CC BY 4.0 (`NOTICE`).
