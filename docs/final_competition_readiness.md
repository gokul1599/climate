# Final Competition Readiness Report

## Official Scoring Breakdown (82 Local Validation Windows)

| Category | Max Points | Measured Score | Evidence |
|---|---:|---:|---|
| **A. Forecast Skill** | **45** | **20.61** | `rmse_pts = 14.27 / 30`, `csi_pts = 6.34 / 15`; Day 1 `+15.64%`, Day 4 `+4.41%` (beats persistence `+23.63%`), Day 8 `-2.92%` |
| **B. Dynamics & Causality** | **25** | **23.13** | Anti-blur `10.00 / 10` (Day-8 var ratio in `[0.40, 2.50]`); Causality `13.125 / 15` (`warm_ocean` passes all, `monsoon_stall` localized & stable, `null` diff `0.0`) |
| **C. Repository Integrity** | **15** | **15.00** | `monsoon_r1.repo_scan` = `15.0 / 15.0` (`anti_cheat = 8.0`, `code_health = 7.0`, `flags = []`) |
| **D. Scientific Brief** | **15** | **15.00 (est.)** | Exact 2-page `brief.pdf` visually inspected via rendered PNGs; covers all required sections honestly |
| **Total (Local Benchmark)** | **100** | **73.74** | Passes all hard qualification gates (`0 / 82` validity failures, `< 5 s` CPU latency) |

## Untested Items
- **5 Hidden Judge Windows** (post-`2023-01-10` ERA5): `NOT TESTED` locally.
