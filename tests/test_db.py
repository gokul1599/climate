"""Tests for optional Neon PostgreSQL persistence and offline dormancy."""
from unittest.mock import patch
from backend import db


def test_db_dormant_when_unconfigured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_ENV_LOADED", True)
    assert db.is_db_configured() is False
    st = db.get_db_status()
    assert st["connected"] is False
    assert st["mode"] == "offline-disabled"
    assert db.list_simulation_runs() == []
