"""Tests for optional Neon PostgreSQL persistence, offline dormancy, and API CRUD."""
from backend import db
from backend.api import evaluation_data, get_simulations, system_status


def test_db_dormant_when_unconfigured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_ENV_LOADED", True)
    assert db.is_db_configured() is False
    st = db.get_db_status()
    assert st["connected"] is False
    assert st["mode"] == "offline-disabled"
    runs, total = db.list_simulation_runs()
    assert runs == []
    assert total == 0


def test_evaluation_and_system_status_endpoints():
    ev = evaluation_data()
    assert ev["ten_day"] is not None
    assert len(ev["ten_day"]["days"]) == 10
    ss = system_status()
    assert ss["fastapi"] == "operational"
    assert ss["model_loaded"] is True
    assert ss["assets_present"] is True


def test_get_simulations_dormant_offline(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_ENV_LOADED", True)
    res = get_simulations()
    assert res == {"enabled": False, "runs": [], "total": 0}
