# MonsoonVerse Engineering Audit (Phase 0)

## 1. Repository Structure & Module Responsibilities
- `serve.py`: Root FastAPI/Uvicorn entrypoint (`0.0.0.0:8000`). Calls `backend.api:app`.
- `backend/`: `api.py` (`POST /health`, `POST /simulate`, `GET /api/sample-windows`), `model_loader.py` (singleton loader), `validation.py` (input/output shape & physical bounds), `response_formatter.py` (JSON list formatting).
- `data_pipeline/`: `load_data.py` (`h5netcdf` NetCDF loader), `inspect_data.py` (data quality & `static_fields.npz`), `preprocess.py` (chronological 80/20 split with 10-day gap, DOY sin/cos, z-score), `create_windows.py`, `dataset.py` (`ClimateWindowDataset`).
- `ml/`: `model.py` (`MonsoonResCNN`, 32ch, 4 ResBlocks, 82,662 params), `loss.py` (`ClimateRolloutLoss`), `normalisation.py`, `rollout.py` (`autoregressive_rollout`), `train.py`, `inference.py` (`MonsoonPredictor`), `config.json`.
- `evaluation/`: `persistence.py`, `climatology.py`, `metrics.py` (wraps `monsoon_r1.scoring`), `perturbation_tests.py`, `stability_tests.py`, `evaluate.py`.
- `impact/`: `regional_analysis.py`, `anomaly_analysis.py`, `impact_metrics.py`.
- `dashboard/`: React 18 + TypeScript + Vite + Tailwind + Recharts UI.
- `artifacts/`: `model.pt` (341,401 B), `model_metadata.json`, `stats.json`, `static_fields.npz`, `sample_windows.npz`.

## 2. Verified Completed vs Incomplete Components
- **Completed**: Data pipeline (6,209 days, 0 NaNs/gaps), FastAPI backend (`schema_validator` passed), perturbation causality (`15/15`), anti-blur (`10/10`), `repo_scan` (`15/15`), React dashboard build + Vite proxy.
- **Incomplete / Gaps**:
  1. **Training interrupted**: `artifacts/model.pt` is from Epoch 4 of 7 (`val_loss = 0.16322`); `model_metadata.json` lacks `history`, `training_time_s`, and sample counts.
  2. **Negative Day-4 / Day-8 official skill**: `Mean SS vs Best` is `-1.84%` at Day 4 and `-14.61%` at Day 8 because the model was only trained up to `horizon=3` and lacks seasonal climatology conditioning.
  3. **PDF brief**: Currently 1 page; specification requires an exact **two-page** PDF (`Page 1: Model Design & Validation`, `Page 2: Visualization & Impact`) visually inspected as rendered images.
  4. **Dashboard browser E2E**: Tested via Vite build and HTTP proxy, not yet Playwright/browser-tested.
  5. **Clean-env offline test & toolkit packaging**: `monsoon_r1` wheel is in `official-kit/release/`, not yet tested in a clean isolated checkout.

## 3. Prioritized Improvement Plan
1. **P0**: Reproduce baseline & diagnose per-channel/per-lead errors (`docs/baseline_evaluation.md`).
2. **P0**: Run controlled experiments in `experiments/` to eliminate negative Day-4 and Day-8 skill while preserving Day-1 skill, CSI, anti-blur, causality, and `<5s` CPU latency.
3. **P1**: Clean-environment offline test, Playwright browser test of dashboard, and exact 2-page `brief.pdf` with visual verification.
4. **P1**: Final `README.md` update, full test suite, official `schema_validator` and `repo_scan`, and push `improve/forecast-skill-and-readiness`.
