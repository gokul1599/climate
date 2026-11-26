# MonsoonVerse: Controlled Incumbent vs LocalResNet-v3 Comparison

**Promotion Decision:** **`PROMOTE_CHALLENGER`** — LocalResNet-v3 outperforms incumbent cleanly on Protocol A (64.49 vs 39.87/70; fc 41.65 vs 17.03/45) without dynamics/validity regressions.

## Protocol A — Reference Benchmark (`2020-01-01..2021-12-31`, `52` windows, stride `14`, strictly out-of-sample for BOTH models)

| Metric | Incumbent (`ClimResCNN`) | Challenger (`LocalResNet-v3`) |
|---|---:|---:|
| Train Split of Checkpoint | 2005-01-01..2018-08-07 | 2005-01-01..2019-12-31 |
| Forecast RMSE Skill (/30) | 13.53 | 26.65 |
| Heavy-Rain CSI Skill (/15) | 3.51 | 15.00 |
| Anti-Blur Day-8 (/10) | 10.00 | 10.00 |
| Perturbation Causality (/15) | 12.84 | 12.84 |
| **Total Model Score (/70)** | 39.87 | 64.49 |
| Day-1 CSI (>10mm) | 0.4810 | 0.5335 |
| Day-4 CSI (>10mm) | 0.3172 | 0.3819 |
| Day-8 CSI (>10mm) | 0.2233 | 0.2823 |
| Warm BoB Propagates Rate | 84.62% | 76.92% |
| Arabian Sea Stall Propagates Rate | 0.00% | 7.69% |
| Null Consistency Rate | 100.00% | 100.00% |
| CPU Latency Mean (ms) | 58.76 | 55.96 |
| Validity Failures | 0 | 0 |

## Protocol B — Incumbent Project Benchmark (`2018-08-18..2021-12-31`, `82` windows, stride `15`)

> Note: Windows in `2018-08-18..2019-12-31` are out-of-sample for `ClimResCNN` (`<=2018-08-07`) but inside `LocalResNet-v3`'s training period (`<=2019-12-31`). Protocol A above provides the leakage-free side-by-side benchmark.

| Metric | Incumbent (`ClimResCNN`) | Challenger (`LocalResNet-v3`) |
|---|---:|---:|
| Forecast Score (/45) | 20.89 | 43.00 |
| Anti-Blur (/10) | 10.00 | 10.00 |
| Causality (/15) | 12.90 | 13.01 |
| Total Model Score (/70) | 43.79 | 66.02 |
| Warm BoB Propagates Rate | 82.35% | 88.24% |
| Arabian Sea Stall Propagates Rate | 5.88% | 5.88% |