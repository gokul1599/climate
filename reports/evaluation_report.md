# MonsoonVerse Evaluation Report

- **Validation period**: 2018-08-18 to 2021-12-31 (82 windows)
- **Validity failures**: 0 / 82
- **CPU 10-day inference**: 75.9 ms mean (191.1 ms max)
- **Beats persistence at Day 4**: True
- **Forecast skill points (est /45)**: 20.61
- **Anti-blur points (est /10)**: 10.00
- **Causality points (est /15)**: 13.12

## Lead-Day Skill Summary

| Lead | Mean SS vs Best | Mean SS vs Persist | Heavy-Rain CSI (Model / Persist / Clim) |
|---|---:|---:|---|
| day_1 | 0.1564 | 0.1564 | 0.4374 / 0.3841 / 0.2576 |
| day_4 | 0.0441 | 0.2363 | 0.3023 / 0.1897 / 0.2702 |
| day_8 | -0.0292 | 0.2410 | 0.2835 / 0.1896 / 0.2861 |

## Perturbation Causality

- **Null shock consistent**: True (max diff = 0.00e+00 std)
- **Warm Bay of Bengal**: {'finite': True, 'nonzero': True, 'stable': True, 'near_frac_day1': 0.9985190629959106, 'localized': True, 'far_frac_max': 0.0005389602738432586, 'no_teleport': True, 'propagates': True}
- **Arabian Sea Stall**: {'finite': True, 'nonzero': True, 'stable': True, 'near_frac_day1': 0.999273955821991, 'localized': True, 'far_frac_max': 0.0009640591451898217, 'no_teleport': True, 'propagates': False}
