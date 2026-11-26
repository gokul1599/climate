-- MonsoonVerse Neon PostgreSQL Schema
-- Stores simulation run summaries, configuration, scenario metadata, and regional impact indicators.

CREATE TABLE IF NOT EXISTS simulation_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scenario TEXT NOT NULL,
    window_id INTEGER NOT NULL,
    window_label TEXT NOT NULL,
    doy_sincos DOUBLE PRECISION[] NOT NULL,
    steps INTEGER NOT NULL,
    inference_ms DOUBLE PRECISION NOT NULL,
    regional_summary JSONB NOT NULL,
    anomalies JSONB
);

ALTER TABLE simulation_runs ADD COLUMN IF NOT EXISTS name TEXT;
ALTER TABLE simulation_runs ADD COLUMN IF NOT EXISTS model_version TEXT DEFAULT 'MonsoonVerse-ClimResCNN';
ALTER TABLE simulation_runs ADD COLUMN IF NOT EXISTS configuration JSONB DEFAULT '{}'::jsonb;
ALTER TABLE simulation_runs ADD COLUMN IF NOT EXISTS notes TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_simulation_runs_created_at
    ON simulation_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_simulation_runs_scenario_created
    ON simulation_runs (scenario, created_at DESC);
