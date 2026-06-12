"""Memory stores: in-memory and SQLite persistence for MemoryItems."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ..core.types import MemoryItem, MemoryScope

__all__ = ["MemoryStore", "InMemoryMemoryStore", "SQLiteMemoryStore"]


class MemoryStore(Protocol):
    def put(self, item: MemoryItem) -> None: ...

    def get(self, memory_id: str) -> MemoryItem | None: ...

    def delete(self, memory_id: str) -> bool: ...

    def all_items(
        self,
        *,
        scope: MemoryScope | None = None,
        owner_id: str | None = None,
        statuses: tuple[str, ...] = ("active", "validated"),
    ) -> list[MemoryItem]: ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._lock = threading.Lock()

    def put(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.id] = item

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            return self._items.pop(memory_id, None) is not None

    def all_items(
        self,
        *,
        scope: MemoryScope | None = None,
        owner_id: str | None = None,
        statuses: tuple[str, ...] = ("active", "validated"),
    ) -> list[MemoryItem]:
        items = []
        for item in self._items.values():
            if statuses and item.status not in statuses:
                continue
            if scope is not None and item.scope != scope:
                continue
            if owner_id is not None and item.owner_id not in (None, owner_id):
                continue
            items.append(item)
        return items


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  owner_id TEXT,
  type TEXT NOT NULL,
  content TEXT NOT NULL,
  confidence REAL,
  privacy_class TEXT,
  status TEXT,
  source_trace_id TEXT,
  supersedes TEXT,
  usage_count INTEGER DEFAULT 0,
  confirmations INTEGER DEFAULT 0,
  entities TEXT,
  metadata TEXT,
  created_at TEXT,
  updated_at TEXT,
  expires_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_scope_owner ON memory_items(scope, owner_id);
CREATE INDEX IF NOT EXISTS idx_memory_status ON memory_items(status);
"""


class SQLiteMemoryStore:
    def __init__(self, path: str | Path = ".vincio/memory.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def put(self, item: MemoryItem) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO memory_items
                   (id, scope, owner_id, type, content, confidence, privacy_class, status,
                    source_trace_id, supersedes, usage_count, confirmations, entities,
                    metadata, created_at, updated_at, expires_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    item.id,
                    item.scope.value,
                    item.owner_id,
                    item.type.value,
                    item.content,
                    item.confidence,
                    item.privacy_class.value,
                    item.status,
                    item.source_trace_id,
                    item.supersedes,
                    item.usage_count,
                    item.confirmations,
                    json.dumps(item.entities),
                    json.dumps(item.metadata, default=str),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                    item.expires_at.isoformat() if item.expires_at else None,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_item(row: tuple) -> MemoryItem:
        return MemoryItem(
            id=row[0],
            scope=row[1],
            owner_id=row[2],
            type=row[3],
            content=row[4],
            confidence=row[5],
            privacy_class=row[6],
            status=row[7],
            source_trace_id=row[8],
            supersedes=row[9],
            usage_count=row[10],
            confirmations=row[11],
            entities=json.loads(row[12] or "[]"),
            metadata=json.loads(row[13] or "{}"),
            created_at=datetime.fromisoformat(row[14]),
            updated_at=datetime.fromisoformat(row[15]),
            expires_at=datetime.fromisoformat(row[16]) if row[16] else None,
        )

    def get(self, memory_id: str) -> MemoryItem | None:
        cursor = self._conn.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        return self._row_to_item(row) if row else None

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def all_items(
        self,
        *,
        scope: MemoryScope | None = None,
        owner_id: str | None = None,
        statuses: tuple[str, ...] = ("active", "validated"),
    ) -> list[MemoryItem]:
        query = "SELECT * FROM memory_items WHERE 1=1"
        params: list = []
        if statuses:
            query += f" AND status IN ({','.join('?' * len(statuses))})"
            params.extend(statuses)
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope.value)
        if owner_id is not None:
            query += " AND (owner_id IS NULL OR owner_id = ?)"
            params.append(owner_id)
        cursor = self._conn.execute(query, params)
        return [self._row_to_item(row) for row in cursor.fetchall()]
