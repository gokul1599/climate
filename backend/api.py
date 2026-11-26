"""FastAPI application implementing the exact MonsoonVerse R1 API contract."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.db import (
    delete_simulation_run,
    get_db_status,
    insert_simulation_run,
    is_db_configured,
    list_simulation_runs,
)
from backend.model_loader import load_predictor
from backend.response_formatter import format_simulation_response
from backend.validation import validate_rollout_output, validate_simulate_request

app = FastAPI(title="MonsoonVerse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and inference assets eagerly at startup so /health reports ready only when loaded
_PREDICTOR = load_predictor()


@app.post("/health")
def health() -> dict[str, str]:
    if _PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_name": "MonsoonVerse"}


@app.post("/simulate")
async def simulate_endpoint(request: Request) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        body = await request.json()
        init_states, doy_sincos, steps, perturbation = validate_simulate_request(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Malformed request: {exc}") from exc

    raw_out = _PREDICTOR.predict_rollout(
        init_states=init_states,
        doy_sincos=doy_sincos,
        steps=steps,
        perturbation=perturbation,
    )
    out = validate_rollout_output(raw_out, steps=steps)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return format_simulation_response(out, elapsed_ms)


@app.get("/api/sample-windows")
def sample_windows() -> dict[str, Any]:
    """Return bundled 3-day validation sample windows for the interactive dashboard."""
    p = Path(__file__).resolve().parents[1] / "artifacts" / "sample_windows.npz"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Sample windows not generated yet")
    z = np.load(p)
    labels = ["Monsoon Season (Jul)", "Winter Season (Jan)", "Post-Monsoon (Oct)"]
    windows = []
    for i, label in enumerate(labels):
        if i < len(z["init_states"]):
            windows.append(
                {
                    "id": i,
                    "label": label,
                    "init_states": z["init_states"][i].tolist(),
                    "doy_sincos": z["doy_sincos"][i].tolist(),
                }
            )
    return {"windows": windows}


@app.get("/api/db-status")
def db_status() -> dict[str, Any]:
    """Report Neon PostgreSQL connectivity status (offline-disabled when no DATABASE_URL)."""
    return get_db_status()


@app.get("/api/simulations")
def get_simulations(limit: int = 20) -> dict[str, Any]:
    """List saved simulation summaries from Neon PostgreSQL."""
    if not is_db_configured():
        return {"enabled": False, "runs": []}
    try:
        runs = list_simulation_runs(limit=limit)
        return {"enabled": True, "runs": runs}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neon query failed: {exc}") from exc


@app.post("/api/simulations", status_code=201)
async def save_simulation(request: Request) -> dict[str, Any]:
    """Persist a simulation run summary and regional impact indicators to Neon PostgreSQL."""
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Neon DATABASE_URL not configured")
    try:
        body = await request.json()
        row = insert_simulation_run(
            scenario=str(body["scenario"]),
            window_id=int(body["window_id"]),
            window_label=str(body["window_label"]),
            doy_sincos=[float(x) for x in body["doy_sincos"]],
            steps=int(body["steps"]),
            inference_ms=float(body["inference_ms"]),
            regional_summary=list(body["regional_summary"]),
            anomalies=body.get("anomalies"),
        )
        return {"saved": True, "run": row}
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist simulation: {exc}") from exc


@app.delete("/api/simulations/{run_id}")
def remove_simulation(run_id: int) -> dict[str, Any]:
    """Delete a saved simulation run by ID from Neon PostgreSQL."""
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Neon DATABASE_URL not configured")
    deleted = delete_simulation_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return {"deleted": True, "id": run_id}
