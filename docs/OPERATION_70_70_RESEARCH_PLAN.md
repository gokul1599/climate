# OPERATION 70/70 — SCIENTIFIC RESEARCH PLAN

**Branch:** `research/operation-70-70` (branched cleanly from `completion/mission-100` @ `a85695c5a32f256f056e3378115fe0195c7c3f02`)
**Incumbent Checkpoint:** [artifacts/model.pt](file:///C:/Users/gokul/OneDrive/Documents/projects/Yugaantar-Track2/MonsoonVerse/artifacts/model.pt) (`SHA-256: 1eba8c5c2f09aa83f98b071872296eae94c83bf255c48216874db00eeaf1b208`)
**Current Promoted Challenger:** [artifacts/local_resnet_v3/model.pt](file:///C:/Users/gokul/OneDrive/Documents/projects/Yugaantar-Track2/MonsoonVerse/artifacts/local_resnet_v3/model.pt) (`SHA-256: 9be5ff49724110b3be1483eaa130a6155dda30a96432f98bfbe1f917f62a0c3e`, `64.49 / 70` on Protocol A)
**Official Scorer Verification:** `monsoon_r1.scoring` and `monsoon_r1.contract` match `yug-monsoon-kit` with **0 diff lines** (`CRLF` vs `LF` line-ending difference only; constants `SS_REF=0.12`, `CSI_SS_REF=0.10`, `LOCAL_MIN_FRAC=0.6`, `STABLE_MAX_STD=5.0`, `DILATE=3`, `MAX_SPEED=8.0`).

---

## Baseline Gap Decomposition
| Score Component | Current (`LocalResNet-v3`) | Maximum | Gap | Scientific Target |
|---|---:|---:|---:|---|
| Forecast RMSE Skill | `26.65` (`mean_ss = 0.1066`) | `30.00` | `-3.35` | Lift `mean_ss` by `+0.0134` (`>= 0.1200`) across `(D1, D4, D8) × 6 channels` |
| Heavy-Rain CSI (`>=10 mm/d`) | `15.00` (`mean_csi_skill >= 0.10`) | `15.00` | `0.00` | Preserve `15.00 / 15.00` |
| Anti-Blur Day-8 | `10.00` (`ratios in [0.40, 2.50]`) | `10.00` | `0.00` | Preserve `10.00 / 10.00` |
| Perturbation Causality | `12.84` (`warm=10/13`, `stall=1/13`, `null=13/13`) | `15.00` | `-2.16` | Diagnose & improve both `warm_ocean` and `monsoon_stall` propagation without hurting RMSE/CSI/Anti-Blur |
| **Total** | **`64.49`** | **`70.00`** | **`-5.51`** | Scientifically defensible optimum |

## Validation-Selection Bias Prevention (Inner Chronological Discipline)
Because `2020-01-01..2021-12-31` (`52` windows) has already been used for `LocalResNet-v3` early stopping and comparison, hyperparameter / objective screening MUST happen on an **inner chronological holdout inside `2005–2019`** (`inner_train: 2005-01-01..2017-12-31`, `inner_val: 2018-01-01..2019-12-31`, matching 2 full years with zero overlap) before retraining final surviving candidates cleanly on `2005–2019` and evaluating `2020–2021` Protocol A (plus multi-seed checks).
