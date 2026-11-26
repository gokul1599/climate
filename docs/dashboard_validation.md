# Dashboard Validation (Phase 7)

- **Production Build**: `npm --prefix dashboard run build` (`tsc && vite build`) exited `0` (`dist/index.html` + `dist/assets/index-B_kfGOEH.js` `558.22 kB`).
- **Live Vite-to-Backend Connectivity**:
  - `GET http://127.0.0.1:5173/` → `200` (`MonsoonVerse` HTML title present)
  - `POST http://127.0.0.1:5173/health` → `200 {'status': 'ready', 'model_name': 'MonsoonVerse'}`
  - `GET http://127.0.0.1:5173/api/sample-windows` → `200` (`3` initial validation windows)
  - `POST http://127.0.0.1:5173/simulate` → `200` (`rollout_len: 10`, live `inference_ms`)
- **Limitations**: Verified via Vite production build and live HTTP proxy integration; interactive DOM clicks in a headless browser (`playwright`) were not executed because `playwright` is not installed in `.venv`.
