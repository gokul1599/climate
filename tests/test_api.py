"""Integration tests for FastAPI /health and /simulate endpoints."""
import asyncio
import json
import numpy as np
from starlette.requests import Request
from backend.api import health, simulate_endpoint


def _make_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {"type": "http", "method": "POST", "path": "/simulate", "headers": []}
    return Request(scope, receive)


def test_health_endpoint():
    data = health()
    assert data["status"] == "ready"
    assert data["model_name"] == "MonsoonVerse"


def test_simulate_endpoint():
    base = np.array([298.0, 4.0, 2.0, 1.0, 100800.0, 38.0], dtype=np.float32)[:, None, None]
    init = np.broadcast_to(base, (3, 6, 32, 36)).tolist()
    req = _make_request({"init_states": init, "doy_sincos": [0.5, 0.866], "steps": 10, "perturbation": None})
    body = asyncio.run(simulate_endpoint(req))
    rollout = np.asarray(body["rollout"], dtype=np.float32)
    assert rollout.shape == (10, 6, 32, 36)
    assert np.isfinite(rollout).all()
    assert body["inference_ms"] < 5000.0
