# Perturbation Dynamics & Scientific Validation (Phase 3)

## 1. Official Contract Compliance
- Perturbations are added exclusively to `init_states[-1]` in physical units prior to internal z-score normalisation (`ml/rollout.py`).
- Tested on a monsoon-season validation window (`2018–2021` split) using `monsoon_r1.scoring.causality_checks`.

## 2. Official Scenario Results (`MonsoonVerse-ClimResCNN`)

| Scenario | Finite | Nonzero | Localized (Day-1 Near Frac ≥ 0.60) | Propagates (Days 4–8) | No Teleport (Far Frac < 0.05) | Stable (≤ 5 std) |
|---|---|---|---|---|---|---|
| `warm_ocean` (+3 K `t2m`, +15% `tcwv`, BoB) | True | True | **True** (`0.9985`) | **True** | **True** (`0.00054`) | True |
| `monsoon_stall` (-40% `u10/v10`, Arabian Sea) | True | True | **True** (`0.9993`) | False | **True** (`0.00096`) | True |
| `null` / `zeros` | True | — | `null_consistent = True` (`max_std_diff = 0.0`) | — | — | True |

- **Official Causality Score**: **`13.125 / 15.0`**.
- **Quantitative `monsoon_stall` Diagnosis (`evaluation/diagnose_weaknesses.py`)**: On the July validation window, out-of-box energy grows **6.5×** from Day 1 (`0.1916`) to Days 4–8 mean (`1.2456`, peaking at `1.3388` on Day 4), while total energy relaxes toward climatology (`150.47 → 24.10 → 9.53`). Because `1.2456 < 0.02 × 150.47 = 3.009`, `monsoon_r1.scoring.causality_checks` marks `propagates=False` despite outward growth. This reflects physically plausible dissipative relaxation rather than an implementation defect, so the accepted `exp3_clim_tuned` checkpoint is preserved.
