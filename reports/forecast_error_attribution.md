# MonsoonVerse Forecast Error Attribution (`LocalResNet-v3`, Protocol A: 52 Windows)

- **Aggregate Mean Skill (`18` lead×channel pairs):** `0.10661` (Target for `30/30`: `0.12000`, Deficit: `-0.01339`)

## Exact `(Lead × Channel)` Skill Attribution Table (`SS vs Best Reference`)
| Lead | `t2m` SS (bias) | `tp` SS (bias) | `u10` SS (bias) | `v10` SS (bias) | `msl` SS (bias) | `tcwv` SS (bias) | Lead Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Day 1 | +0.1790 (-0.00) | +0.2536 (+0.07) | +0.2304 (+0.05) | +0.2422 (+0.07) | +0.2788 (-7.37) | +0.1814 (-0.14) | **+0.2276** |
| Day 4 | +0.0474 (-0.11) | +0.0587 (+0.12) | +0.1007 (+0.09) | +0.0379 (+0.08) | +0.1191 (-30.95) | +0.1043 (-0.41) | **+0.0780** |
| Day 8 | -0.0232 (-0.12) | +0.0027 (+0.20) | +0.0259 (+0.18) | -0.0017 (+0.10) | +0.0436 (-51.40) | +0.0384 (-1.03) | **+0.0143** |

## Seasonal Attribution (`SS vs Best Reference`)
| Season | Windows | Mean SS (18) | `t2m` | `tp` | `u10` | `v10` | `msl` | `tcwv` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DJF_Winter | 14 | +0.0840 | +0.0614 | +0.1133 | +0.0732 | +0.0614 | +0.1084 | +0.0866 |
| MAM_PreMonsoon | 12 | +0.1081 | +0.0309 | +0.1160 | +0.1197 | +0.0973 | +0.1694 | +0.1153 |
| JJAS_Monsoon | 18 | +0.0864 | +0.0085 | +0.0934 | +0.0996 | +0.0977 | +0.1176 | +0.1018 |
| OND_PostMonsoon | 8 | +0.1604 | +0.1913 | +0.1120 | +0.1938 | +0.1119 | +0.2147 | +0.1387 |