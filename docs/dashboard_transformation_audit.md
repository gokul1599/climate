# MonsoonVerse Dashboard Transformation — Pre-Implementation Audit

## 1. Git & Branch State
- **Current branch**: `feat/neon-persistence` (created from `improve/forecast-skill-and-readiness` at `07bf094`).
- **Protected branches untouched**: `main` (`7a9c9bc`), `improve/forecast-skill-and-readiness` (`07bf094`).
- **Local Neon changes preserved**: `.gitignore`, `.env.example`, `.env` (gitignored), `db/schema.sql`, `backend/db.py`, `backend/api.py`, `pyproject.toml`, `tests/test_db.py`, `dashboard/src/services/api.ts`, `dashboard/src/components/SavedSimulations.tsx`, `dashboard/src/App.tsx`.

## 2. Existing Working Features
- **Inference & ML (`ml/`, `artifacts/`)**: `MonsoonVerse-ClimResCNN` (`model.pt`, `climatology.npz`, `static_fields.npz`, `stats.json`, `sample_windows.npz`) passing `schema_validator` (`2.08s–2.24s`), `repo_scan` (`15.0/15.0`), and `pytest` (`9/9`).
- **Neon PostgreSQL**: Live `neondb` connection verified via `pg8000`, `simulation_runs` schema initialized, `2` runs persisted in DB.
- **Dashboard (`dashboard/src/`)**: Single-page React + TypeScript + Tailwind + Recharts view (`App.tsx`, `IndiaGridMap.tsx`, `ForecastCharts.tsx`, `ImpactExplorer.tsx`, `SavedSimulations.tsx`).

## 3. Missing Features Targeted in Transformation
- Multi-page `AppShell` with collapsible `Sidebar` (9 sections) and independent status `Topbar`.
- Dedicated pages: `Overview`, `Forecast Studio`, `Scenario Lab`, `Impact Explorer`, `Saved Simulations`, `Model Performance`, `Data & Methodology`, `System Status`, `Settings`.
- Full Neon CRUD (`GET /{id}`, `PATCH /{id}` rename, search/filter/pagination, compare view, delete confirmation) + read-only `/api/evaluation` and `/api/system-status` endpoints.
- Upgraded `ForecastMap` (variable-specific palettes, zoom/pan, India geographic reference boundaries, clickable grid-cell inspection) & expanded `ForecastCharts` (min/max envelope, anomaly, regional/grid-cell time series).
- Client-side `Exports` (CSV forecast, JSON metadata, Regional CSV, Chart PNG, Simulation PDF report).

## 4. Risks & Safeguards
- **Judge Offline Compatibility**: `/health` and `/simulate` remain 100% independent of Neon, internet, and external packages; `backend/db.py` stays dormant when `DATABASE_URL` is unset.
- **Secret Hygiene**: `DATABASE_URL` resides strictly in gitignored `.env` and backend runtime; never exposed via `VITE_*` variables.
- **Static Scan (`repo_scan.py`)**: All new backend endpoints avoid hardcoded dates or external weather URLs.
