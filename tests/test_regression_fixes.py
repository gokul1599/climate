"""Regression tests for Phase 01/05 backend validation and database failure handling fixes."""
import asyncio
import json

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend import api
from backend.validation import validate_simulate_request


def _make_req(payload, method: str = "POST", path: str = "/simulate") -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": method, "path": path, "headers": []}, receive)


def test_validate_simulate_rejects_non_dict_and_bool_steps():
    with pytest.raises(ValueError, match="JSON object"):
        validate_simulate_request([1, 2, 3])  # type: ignore[arg-type]

    init = np.zeros((3, 6, 32, 36), dtype=np.float32).tolist()
    with pytest.raises(ValueError, match="integer"):
        validate_simulate_request({"init_states": init, "doy_sincos": [0.0, 1.0], "steps": True})
    with pytest.raises(ValueError, match="integer"):
        validate_simulate_request({"init_states": init, "doy_sincos": [0.0, 1.0], "steps": 4.5})


def test_validate_simulate_rejects_negative_init_tp():
    init = np.zeros((3, 6, 32, 36), dtype=np.float32)
    init[1, 1, 5, 5] = -2.0
    with pytest.raises(ValueError, match="negative"):
        validate_simulate_request({"init_states": init.tolist(), "doy_sincos": [0.0, 1.0], "steps": 10})


def test_get_simulations_invalid_pagination():
    with pytest.raises(HTTPException) as e1:
        api.get_simulations(limit=0, offset=0)
    assert e1.value.status_code == 422

    with pytest.raises(HTTPException) as e2:
        api.get_simulations(limit=10, offset=-1)
    assert e2.value.status_code == 422


def test_run_id_endpoints_return_503_on_db_outage(monkeypatch):
    monkeypatch.setattr(api, "is_db_configured", lambda: True)

    def _boom(*args, **kwargs):
        raise RuntimeError("Simulated Neon outage")

    monkeypatch.setattr(api, "get_simulation_run", _boom)
    monkeypatch.setattr(api, "update_simulation_run", _boom)
    monkeypatch.setattr(api, "delete_simulation_run", _boom)

    with pytest.raises(HTTPException) as g_err:
        api.get_simulation(1)
    assert g_err.value.status_code == 503
    assert "Database unavailable" in str(g_err.value.detail)

    with pytest.raises(HTTPException) as p_err:
        asyncio.run(api.patch_simulation(1, _make_req({"name": "valid"}, method="PATCH")))
    assert p_err.value.status_code == 503

    with pytest.raises(HTTPException) as d_err:
        api.remove_simulation(1)
    assert d_err.value.status_code == 503


def test_patch_simulation_rejects_non_dict_and_blank_name(monkeypatch):
    monkeypatch.setattr(api, "is_db_configured", lambda: True)

    with pytest.raises(HTTPException) as e1:
        asyncio.run(api.patch_simulation(1, _make_req(["bad"], method="PATCH")))
    assert e1.value.status_code == 422

    with pytest.raises(HTTPException) as e2:
        asyncio.run(api.patch_simulation(1, _make_req({"name": "   "}, method="PATCH")))
    assert e2.value.status_code == 422
