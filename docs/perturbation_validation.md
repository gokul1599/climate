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
- **Limitation**: On the tested July window, `monsoon_stall` remained tightly localized (`99.93%` within 3 cells on Day 1) and stable, but its outward energy growth across Days 4–8 did not exceed the `1.5 * out_e[0]` + `0.02 * max_e` propagation threshold in `monsoon_r1.scoring`.
