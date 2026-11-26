"""FastAPI application implementing the exact MonsoonVerse R1 API contract + product routes."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.db import (
    delete_simulation_run,
    get_db_status,
    get_simulation_run,
    insert_simulation_run,
    is_db_configured,
    list_simulation_runs,
    update_simulation_run,
)
from backend.model_loader import load_predictor
from backend.response_formatter import format_simulation_response
from backend.validation import validate_rollout_output, validate_simulate_request

ROOT_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(title="MonsoonVerse API", version="1.0.0")

_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
_extra_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _extra_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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
    p = ROOT_DIR / "artifacts" / "sample_windows.npz"
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


@app.get("/api/evaluation")
def evaluation_data() -> dict[str, Any]:
    """Serve verified evaluation outputs and model metadata read-only from local artifacts."""
    ten_day_path = ROOT_DIR / "evaluation" / "results" / "ten_day_forecast_metrics.json"
    val_path = ROOT_DIR / "reports" / "validation_results.json"
    meta_path = ROOT_DIR / "artifacts" / "model_metadata.json"
    data_meta_path = ROOT_DIR / "data_pipeline" / "metadata.json"

    return {
        "ten_day": json.loads(ten_day_path.read_text(encoding="utf-8")) if ten_day_path.is_file() else None,
        "validation": json.loads(val_path.read_text(encoding="utf-8")) if val_path.is_file() else None,
        "model_metadata": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None,
        "dataset_metadata": json.loads(data_meta_path.read_text(encoding="utf-8")) if data_meta_path.is_file() else None,
    }


@app.get("/api/system-status")
def system_status() -> dict[str, Any]:
    """Independent live health and asset diagnostics for the System Status dashboard page."""
    db_st = get_db_status()
    art = ROOT_DIR / "artifacts"
    assets_ok = all(
        (art / fname).is_file()
        for fname in ("model.pt", "stats.json", "static_fields.npz", "climatology.npz")
    )
    model_ready = _PREDICTOR is not None and assets_ok
    return {
        "fastapi": "operational",
        "model_loaded": bool(model_ready),
        "model_name": "MonsoonVerse-ClimResCNN",
        "inference_mode": "cpu-offline",
        "assets_present": assets_ok,
        "climatology_loaded": bool(_PREDICTOR is not None and _PREDICTOR.clim_norm_366 is not None),
        "neon": db_st,
    }


@app.get("/api/db-status")
def db_status() -> dict[str, Any]:
    """Report Neon PostgreSQL connectivity status (offline-disabled when no DATABASE_URL)."""
    return get_db_status()


@app.get("/api/simulations")
def get_simulations(
    limit: int = 20,
    offset: int = 0,
    search: str | None = None,
    scenario: str | None = None,
) -> dict[str, Any]:
    """List saved simulation summaries from Neon PostgreSQL with filtering and pagination."""
    if not is_db_configured():
        return {"enabled": False, "runs": [], "total": 0}
    try:
        runs, total = list_simulation_runs(
            limit=limit, offset=offset, search=search, scenario=scenario
        )
        return {"enabled": True, "runs": runs, "total": total}
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Database unavailable ({type(exc).__name__})"
        ) from exc


@app.post("/api/simulations", status_code=201)
async def save_simulation(request: Request) -> dict[str, Any]:
    """Persist a simulation run summary and regional impact indicators to Neon PostgreSQL."""
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Neon DATABASE_URL not configured")
    try:
        body = await request.json()
        row = insert_simulation_run(
            name=body.get("name"),
            scenario=str(body["scenario"]),
            window_id=int(body["window_id"]),
            window_label=str(body["window_label"]),
            doy_sincos=[float(x) for x in body["doy_sincos"]],
            steps=int(body["steps"]),
            inference_ms=float(body["inference_ms"]),
            model_version=str(body.get("model_version", "MonsoonVerse-ClimResCNN")),
            configuration=body.get("configuration"),
            regional_summary=list(body["regional_summary"]),
            anomalies=body.get("anomalies"),
            notes=str(body.get("notes", "")),
        )
        return {"saved": True, "run": row}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid simulation payload: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Failed to persist simulation ({type(exc).__name__})"
        ) from exc


@app.get("/api/simulations/{run_id}")
def get_simulation(run_id: int) -> dict[str, Any]:
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Neon DATABASE_URL not configured")
    row = get_simulation_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return {"run": row}


@app.patch("/api/simulations/{run_id}")
async def patch_simulation(run_id: int, request: Request) -> dict[str, Any]:
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Neon DATABASE_URL not configured")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    name = body.get("name")
    notes = body.get("notes")
    if name is not None and not str(name).strip():
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    row = update_simulation_run(
        run_id,
        name=str(name) if name is not None else None,
        notes=str(notes) if notes is not None else None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return {"updated": True, "run": row}


@app.delete("/api/simulations/{run_id}")
def remove_simulation(run_id: int) -> dict[str, Any]:
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Neon DATABASE_URL not configured")
    deleted = delete_simulation_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return {"deleted": True, "id": run_id}
