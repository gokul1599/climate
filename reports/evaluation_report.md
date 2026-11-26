# MonsoonVerse Evaluation Report

- **Validation period**: 2018-08-18 to 2021-12-31 (82 windows)
- **Validity failures**: 0 / 82
- **CPU 10-day inference**: 69.7 ms mean (220.6 ms max)
- **Beats persistence at Day 4**: True
- **Forecast skill points (est /45)**: 8.83
- **Anti-blur points (est /10)**: 10.00
- **Causality points (est /15)**: 15.00

## Lead-Day Skill Summary

| Lead | Mean SS vs Best | Mean SS vs Persist | Heavy-Rain CSI (Model / Persist / Clim) |
|---|---:|---:|---|
| day_1 | 0.1583 | 0.1583 | 0.4339 / 0.3841 / 0.2576 |
| day_4 | -0.0184 | 0.1856 | 0.3048 / 0.1897 / 0.2702 |
| day_8 | -0.1461 | 0.1544 | 0.3207 / 0.1896 / 0.2861 |

## Perturbation Causality

- **Null shock consistent**: True (max diff = 0.00e+00 std)
- **Warm Bay of Bengal**: {'finite': True, 'nonzero': True, 'stable': True, 'near_frac_day1': 0.9969852566719055, 'localized': True, 'far_frac_max': 0.00047905524843372405, 'no_teleport': True, 'propagates': True}
- **Arabian Sea Stall**: {'finite': True, 'nonzero': True, 'stable': True, 'near_frac_day1': 0.9989002346992493, 'localized': True, 'far_frac_max': 0.0021669252309948206, 'no_teleport': True, 'propagates': True}
