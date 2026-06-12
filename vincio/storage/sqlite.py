"""SQLite metadata store (MVP storage, core tables)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..core.errors import StorageError

__all__ = ["SQLiteMetadataStore"]

# Core tables get first-class columns; everything else lives in
# a generic records table with a JSON payload.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  app_id TEXT NOT NULL DEFAULT '',
  user_id TEXT,
  tenant_id TEXT,
  objective TEXT,
  status TEXT,
  started_at TEXT,
  ended_at TEXT,
  cost_usd REAL,
  latency_ms INTEGER,
  json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_app ON runs(app_id);
CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id);

CREATE TABLE IF NOT EXISTS context_packets (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL DEFAULT '',
  spec_hash TEXT NOT NULL DEFAULT '',
  token_count INTEGER,
  json TEXT,
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_packets_run ON context_packets(run_id);

CREATE TABLE IF NOT EXISTS eval_results (
  id TEXT PRIMARY KEY,
  app_id TEXT NOT NULL DEFAULT '',
  dataset_id TEXT NOT NULL DEFAULT '',
  run_id TEXT,
  metric_name TEXT,
  metric_value REAL,
  details TEXT,
  created_at TEXT,
  json TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_app ON eval_results(app_id);

CREATE TABLE IF NOT EXISTS records (
  kind TEXT NOT NULL,
  id TEXT NOT NULL,
  json TEXT NOT NULL,
  PRIMARY KEY (kind, id)
);
CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind);
"""

_CORE_COLUMNS = {
    "runs": ("app_id", "user_id", "tenant_id", "objective", "status", "started_at", "ended_at", "cost_usd", "latency_ms"),
    "context_packets": ("run_id", "spec_hash", "token_count", "created_at"),
    "eval_results": ("app_id", "dataset_id", "run_id", "metric_name", "metric_value", "details", "created_at"),
}


class SQLiteMetadataStore:
    def __init__(self, path: str | Path = ".vincio/vincio.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save(self, kind: str, record: dict[str, Any]) -> None:
        record_id = record.get("id")
        if not record_id:
            raise StorageError(f"record for {kind!r} has no id")
        payload = json.dumps(record, default=str)
        with self._lock:
            if kind in _CORE_COLUMNS:
                columns = _CORE_COLUMNS[kind]
                values = []
                for column in columns:
                    value = record.get(column)
                    if column == "details" and value is not None and not isinstance(value, str):
                        value = json.dumps(value, default=str)
                    values.append(value)
                placeholders = ",".join("?" * (len(columns) + 2))
                self._conn.execute(
                    f"INSERT OR REPLACE INTO {kind} (id, {', '.join(columns)}, json) VALUES ({placeholders})",
                    (str(record_id), *values, payload),
                )
            else:
                self._conn.execute(
                    "INSERT OR REPLACE INTO records (kind, id, json) VALUES (?,?,?)",
                    (kind, str(record_id), payload),
                )
            self._conn.commit()

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        if kind in _CORE_COLUMNS:
            row = self._conn.execute(f"SELECT json FROM {kind} WHERE id = ?", (record_id,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT json FROM records WHERE kind = ? AND id = ?", (kind, record_id)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def query(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        where = where or {}
        if kind in _CORE_COLUMNS:
            clauses, params = [], []
            for key, value in where.items():
                if key in _CORE_COLUMNS[kind] or key == "id":
                    clauses.append(f"{key} = ?")
                    params.append(value)
            sql = f"SELECT json FROM {kind}"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
            rows = self._conn.execute(sql, (*params, limit, offset)).fetchall()
            records = [json.loads(r[0]) for r in rows]
            # Residual filters not covered by columns.
            residual = {k: v for k, v in where.items() if k not in _CORE_COLUMNS[kind] and k != "id"}
            if residual:
                records = [r for r in records if all(r.get(k) == v for k, v in residual.items())]
            return records
        rows = self._conn.execute(
            "SELECT json FROM records WHERE kind = ? ORDER BY rowid DESC", (kind,)
        ).fetchall()
        records = [json.loads(r[0]) for r in rows]
        if where:
            records = [r for r in records if all(r.get(k) == v for k, v in where.items())]
        return records[offset : offset + limit]

    def delete(self, kind: str, record_id: str) -> bool:
        with self._lock:
            if kind in _CORE_COLUMNS:
                cursor = self._conn.execute(f"DELETE FROM {kind} WHERE id = ?", (record_id,))
            else:
                cursor = self._conn.execute(
                    "DELETE FROM records WHERE kind = ? AND id = ?", (kind, record_id)
                )
            self._conn.commit()
            return cursor.rowcount > 0

    def count(self, kind: str) -> int:
        if kind in _CORE_COLUMNS:
            (count,) = self._conn.execute(f"SELECT COUNT(*) FROM {kind}").fetchone()
        else:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM records WHERE kind = ?", (kind,)
            ).fetchone()
        return count
