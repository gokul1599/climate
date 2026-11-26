# Forecast Skill Improvement Report (Phase 2)

## 1. Controlled Experiment Comparison (`82` Chronological Validation Windows)

| Candidate | Architecture & Training | Day-1 SS | Day-4 SS | Day-8 SS | Forecast Pts (/45) | Anti-Blur (/10) | Causality (/15) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `exp0_baseline` | 22-ch ResCNN (32ch, 4 blk), `h=3`, Ep 4 | +0.1583 | -0.0184 | -0.1461 | 8.83 | 10.00 | 15.00 | Replaced |
| `exp1_rollout8` | 22-ch ResCNN (32ch, 4 blk), `h=8`, Ep 5 (`stride=4`) | +0.0801 | -0.0537 | -0.1338 | 0.00 | 9.60 | 11.25 | **Rejected** |
| `exp2_clim_cond` | 28-ch + Clim (32ch, 4 blk), `h=8`, Ep 5 (`stride=4`) | +0.0908 | +0.0141 | -0.0401 | 5.65 | 10.00 | 11.25 | **Rejected** |
| **`exp3_clim_tuned`** | **28-ch + Clim (32ch, 5 blk), `h=8`, Ep 7 (`stride=2`, lead-weighted)** | **+0.1564** | **+0.0441** | **-0.0292** | **20.61** | **10.00** | **13.13** | **ACCEPTED** |

## 2. Why `exp3_clim_tuned` Succeeded
1. **Training-only climatology conditioning**: Providing `clim_norm(doy + k + 1)` gives the network the exact local seasonal cycle so synoptic anomalies relax smoothly toward climatology at Days 4–8 instead of drifting.
2. **Lead-weighted 8-step rollout loss**: Weighting early leads (`1.6, 1.3, 1.1, 1.0, 0.9, ...`) and `tp` (`1.35×`) preserved Day-1 skill (`+0.1564`) and Day-1 heavy-rain CSI (`0.4374`) while turning Day-4 official skill **positive** (`+4.41%`, beating climatology on 5/6 channels) and improving Day-8 skill by **+11.69 percentage points**.
3. **Why `exp1` and `exp2` were rejected**: `exp1` lacked seasonal conditioning and degraded Day 1; `exp2` underfit Day 1 and spatial propagation due to `stride=4` and equal 8-step loss weights.
