# Impact Explorer Methodology

## Purpose
The MonsoonVerse Impact Explorer translates 10-day model rollouts into regional summary indicators to help users compare normal and perturbed climate scenarios.

## Indicators
- **Regional Mean Rainfall (mm/day)**: Spatial and temporal mean of `tp` across predefined geographic boxes (Western Ghats & Konkan, Central India Monsoon Core, Indo-Gangetic Plain, Northeast India, Bay of Bengal, Arabian Sea).
- **Rainfall Anomaly (mm/day)**: `Scenario Mean Rain − Normal Mean Rain`.
- **Heavy-Rain Cell-Days (≥10 mm/day)**: Count of `(day, lat, lon)` grid points within a region where simulated `tp >= 10.0 mm/day`.
- **Temperature Difference (K)** & **Moisture Difference (kg/m²)**: Regional mean `Perturbed − Normal` for `t2m` and `tcwv`.

## Scientific Limitations
- All indicators are **model-derived scenario comparisons** from a compact 1°×1° regional CNN, not operational weather forecasts.
- They must **not** be interpreted as guaranteed flood events, crop yields, or economic loss predictions.
- Coarse 1° resolution (~111 km) smooths local convective extremes and orographic rainfall peaks.
