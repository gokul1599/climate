# MONSOONVERSE — TRAINING & REPRODUCIBILITY GUIDE

## 1. Strict Split & Normalization Governance
- **Raw Dataset (`india_climate_train.nc`):** Loaded strictly for training/offline evaluation (`2005-01-01` to `2021-12-31`, `6,209` continuous daily steps on `32×36` grid). **Never** committed to Git and **never** imported by the FastAPI serving graph (`backend/api.py`, `ml/inference.py`, `ml/local_resnet_v3/inference.py`).
- **Incumbent Split (`80/20`, 10-day gap):** Train `2005-01-01..2018-08-07`, gap `10` days, validation `2018-08-18..2021-12-31`.
- **Reference Protocol Split (`local_resnet_v3`):** Train `2005-01-01..2019-12-31` (`5,478` days), validation `2020-01-01..2021-12-31` (`731` days). Zero windows cross split boundaries.
- **Train-Only Statistics:** Both `artifacts/stats.json` (incumbent) and `artifacts/local_resnet_v3/norm_stats.json` (challenger) are fit strictly on training slices (`std >= 1e-6` clamp) and verified via physical round-trip checks.

## 2. Reproducing `MonsoonVerse-LocalResNet-v3` Challenger Training
```powershell
$env:PYTHONPATH = "."
python -m ml.local_resnet_v3.train `
  --split-mode reference `
  --artifacts artifacts/local_resnet_v3 `
  --epochs-s1 14 --epochs-s2 8 `
  --s1-stride 1 --s2-stride 3 `
  --batch-size 64 --s2-batch-size 32 --effective-batch-size 128 `
  --rollout-horizon 8 --grad-weight 0.14 --var-weight 0.08 `
  --var-min-ratio 0.65 --heavy-rain-boost 2.2 --threads 6
```
- Notice: Stage 2 uses `s2_batch_size=32` with 4-step gradient accumulation (`effective_batch_size=128`), keeping gradient updates mathematically equivalent to `batch=128` while preventing CPU virtual memory allocation spikes during 8-step backpropagation.
- Smoke-test command (1+1 epochs, subsampled stride): `python -m ml.local_resnet_v3.train --smoke-test --artifacts artifacts/local_resnet_v3_smoke`.

## 3. Reproducing Controlled Model Evaluation
```powershell
$env:PYTHONPATH = "."
python evaluation/compare_incumbent_vs_local_resnet_v3.py
```
Outputs side-by-side Protocol A (`2020–2021`, `52` windows, stride `14`) and Protocol B (`2018–2021`, `82` windows, stride `15`) results into `reports/incumbent_vs_local_resnet_v3.{json,md}`.

## 4. Deterministic Behavior & Empirical Residual Intervals
Both models run deterministically (`model.eval()`, `torch.inference_mode()`). Neither emits calibrated ensemble probabilities; out-of-sample `p05`/`p95` error intervals in `artifacts/incumbent_uncertainty_bands.json` and `artifacts/local_resnet_v3/uncertainty_bands.json` are empirical validation residual quantiles.
