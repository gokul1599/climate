# MonsoonVerse Technical Brief Content

1. **Model Architecture**: Compact 2D Residual CNN (`MonsoonResCNN`, 32 hidden channels, 4 residual blocks, replicate padding, GroupNorm, SiLU) predicting next-day normalised residual `latest_state + delta` from `[B, 22, 32, 36]`.
2. **Dataset & Preprocessing**: `india_climate_train.nc` (2005–2021, 6,209 daily steps, 0 NaNs/gaps, 32×36 grid 6.5°–37.5°N, 65.5°–100.5°E). Chronological 80/20 split with 10-day anti-leakage gap. Internal z-score normalisation via `stats.json`; physical units preserved on API.
3. **Physical Assumptions**: Daily means (00/06/12/18 UTC) for `t2m, u10, v10, msl, tcwv`; 24h accumulated `tp >= 0`; static `land_sea_mask` and `orography_m` bundled locally.
4. **Forecasting & Perturbations**: Deterministic 10-day autoregressive rollout advancing `doy_sincos` each step. Physical-unit perturbations applied exclusively to `init_states[-1]` prior to normalisation.
5. **Dynamics, Drift, Blurring & Stability**: Curriculum multi-step rollout training (`h=1→2→3`) prevents drift (0/82 validity failures) and preserves spatial variance (10.0/10.0 anti-blur). Both `warm_ocean` and `monsoon_stall` show >99.6% Day-1 localization and outward propagation by Days 4–8.
6. **Dashboard & Impact Explorer**: Live React + Recharts dashboard displaying normal, perturbed, and difference maps plus indicative regional rainfall/temperature/moisture indicators.
7. **Limitations**: 1° resolution cannot resolve mesoscale convection; multi-day skill decays toward climatology; impact indicators are exploratory, not operational warnings.
