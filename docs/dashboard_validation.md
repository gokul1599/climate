# Dashboard Validation (Phase 7)

Verified via production build (`npm --prefix dashboard run build` → `tsc && vite build` exited `0`) AND live interactive browser testing (`http://127.0.0.1:5173/` proxied to `serve.py --port 8000`):

1. **Dashboard loads & live API status**: **PASS** (`MonsoonVerse: ready`, `Live CPU Inference: 102.6 ms`).
2. **Normal, Warm Bay of Bengal, and Difference maps + Charts + Impact Explorer**: **PASS** (all 3 grids, 2 Recharts SVG charts, and 6-region table rendered from live `/simulate` output).
3. **Day slider / selector (Day 4, Day 8)**: **PASS** (map titles and ranges updated dynamically).
4. **Variable selector (`2m Temperature (K)`)**: **PASS** (maps and charts updated units and ranges).
5. **Arabian Sea Monsoon Stall simulation**: **PASS** (`Live CPU Inference: 107.6 ms`, difference map `-0.547 to 0.600 K`).
6. **Zero Perturbation Check**: **PASS** (`Live CPU Inference: 67.8 ms`, difference map `0.000 to 0.000 K`, all anomalies `+0.000`).
7. **Console & Network Errors**: **PASS** (`0` console errors, `0` failed requests across `29` network requests).
