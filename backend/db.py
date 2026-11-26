"""Optional Neon PostgreSQL persistence for MonsoonVerse simulation summaries.

Designed for dual-mode operation:
- When DATABASE_URL / NEON_DATABASE_URL is configured (and pg8000 or psycopg is installed),
  persists and retrieves simulation runs in Neon serverless PostgreSQL.
- When unset (e.g., offline judge container with --network none), remains completely
  dormant with zero external network calls or hard dependencies.
"""
from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_SCHEMA_READY = False
_ENV_LOADED = False


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def get_database_url() -> str | None:
    _load_env_once()
    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
    return url.strip() if url and url.strip() else None


def is_db_configured() -> bool:
    return get_database_url() is not None


class _CompatConn:
    """Thin wrapper unifying pg8000.dbapi and psycopg connections with dict rows."""

    def __init__(self, conn: Any, driver: str) -> None:
        self._conn = conn
        self._driver = driver

    def __enter__(self) -> "_CompatConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is not None:
                self._conn.rollback()
        finally:
            self._conn.close()

    def execute_ddl(self, sql: str) -> None:
        cur = self._conn.cursor()
        try:
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                cur.execute(stmt)
            self._conn.commit()
        finally:
            cur.close()

    def fetch_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row is None:
                return None
            if isinstance(row, dict):
                return row
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
        finally:
            cur.close()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                return []
            if isinstance(rows[0], dict):
                return list(rows)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]
        finally:
            cur.close()

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
            rc = cur.rowcount
            self._conn.commit()
            return rc
        finally:
            cur.close()


def _connect() -> _CompatConn:
    url = get_database_url()
    if not url:
        raise RuntimeError("Neon database URL is not configured")

    try:
        import psycopg
        from psycopg.rows import dict_row

        return _CompatConn(psycopg.connect(url, connect_timeout=10, row_factory=dict_row), "psycopg")
    except Exception:
        pass

    import pg8000.dbapi

    parsed = urlparse(url)
    ssl_ctx = ssl.create_default_context()
    conn = pg8000.dbapi.connect(
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/") or "neondb",
        ssl_context=ssl_ctx,
        timeout=10,
    )
    return _CompatConn(conn, "pg8000")


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    schema_sql = (Path(__file__).resolve().parents[1] / "db" / "schema.sql").read_text(encoding="utf-8")
    with _connect() as conn:
        conn.execute_ddl(schema_sql)
    _SCHEMA_READY = True


def get_db_status() -> dict[str, Any]:
    if not is_db_configured():
        return {"connected": False, "provider": "neon", "mode": "offline-disabled"}
    try:
        ensure_schema()
        with _connect() as conn:
            row = conn.fetch_one("SELECT 1 AS ok, current_database() AS db")
        return {
            "connected": True,
            "provider": "neon",
            "mode": "live",
            "database": row["db"] if row else "neondb",
        }
    except Exception as exc:
        return {
            "connected": False,
            "provider": "neon",
            "mode": "error",
            "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
        }


def insert_simulation_run(
    scenario: str,
    window_id: int,
    window_label: str,
    doy_sincos: list[float],
    steps: int,
    inference_ms: float,
    regional_summary: list[dict[str, Any]],
    anomalies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ensure_schema()
    sql = """
        INSERT INTO simulation_runs (
            scenario, window_id, window_label, doy_sincos, steps,
            inference_ms, regional_summary, anomalies
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        RETURNING id, created_at, scenario, window_id, window_label,
                  doy_sincos, steps, inference_ms, regional_summary, anomalies
    """
    with _connect() as conn:
        row = conn.fetch_one(
            sql,
            (
                scenario,
                int(window_id),
                window_label,
                [float(x) for x in doy_sincos],
                int(steps),
                float(inference_ms),
                json.dumps(regional_summary),
                json.dumps(anomalies) if anomalies is not None else None,
            ),
        )
        conn._conn.commit()
    return _serialise_row(row or {})


def list_simulation_runs(limit: int = 20) -> list[dict[str, Any]]:
    if not is_db_configured():
        return []
    ensure_schema()
    sql = """
        SELECT id, created_at, scenario, window_id, window_label,
               doy_sincos, steps, inference_ms, regional_summary, anomalies
        FROM simulation_runs
        ORDER BY created_at DESC
        LIMIT %s
    """
    with _connect() as conn:
        rows = conn.fetch_all(sql, (max(1, min(int(limit), 100)),))
    return [_serialise_row(r) for r in rows]


def delete_simulation_run(run_id: int) -> bool:
    if not is_db_configured():
        return False
    ensure_schema()
    with _connect() as conn:
        rc = conn.execute_write("DELETE FROM simulation_runs WHERE id = %s", (int(run_id),))
    return rc > 0


def _serialise_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("created_at") is not None and hasattr(out["created_at"], "isoformat"):
        out["created_at"] = out["created_at"].isoformat()
    if isinstance(out.get("regional_summary"), str):
        out["regional_summary"] = json.loads(out["regional_summary"])
    if isinstance(out.get("anomalies"), str):
        out["anomalies"] = json.loads(out["anomalies"])
    return out
