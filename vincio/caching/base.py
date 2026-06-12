"""Cache backends: TTL + LRU in-memory and SQLite-persistent.

Entries carry **tags** so the invalidation manager can clear precisely the
entries affected by a change (document updated, prompt version changed...).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

__all__ = ["CacheBackend", "InMemoryCache", "SQLiteCache"]


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, *, ttl_s: float | None = None, tags: list[str] | None = None) -> None: ...

    def delete(self, key: str) -> bool: ...

    def invalidate_tag(self, tag: str) -> int: ...

    def clear(self) -> int: ...

    def stats(self) -> dict[str, Any]: ...


class InMemoryCache:
    """Thread-safe LRU cache with TTL and tag invalidation."""

    def __init__(self, *, max_entries: int = 10_000, default_ttl_s: float | None = 3600.0) -> None:
        self.max_entries = max_entries
        self.default_ttl_s = default_ttl_s
        self._data: OrderedDict[str, tuple[float | None, Any, frozenset[str]]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value, _tags = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, *, ttl_s: float | None = None, tags: list[str] | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        expires_at = time.monotonic() + ttl if ttl is not None else None
        with self._lock:
            self._data[key] = (expires_at, value, frozenset(tags or ()))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def invalidate_tag(self, tag: str) -> int:
        with self._lock:
            keys = [k for k, (_, _, tags) in self._data.items() if tag in tags]
            for key in keys:
                del self._data[key]
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class SQLiteCache:
    """Persistent cache (values must be JSON-serializable)."""

    def __init__(self, path: str | Path = ".vincio/cache.db", *, default_ttl_s: float | None = 3600.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl_s = default_ttl_s
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              expires_at REAL,
              tags TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
            """
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        self._conn.close()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            value, expires_at = row
            if expires_at is not None and time.time() > expires_at:
                self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self._conn.commit()
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(value)

    def set(self, key: str, value: Any, *, ttl_s: float | None = None, tags: list[str] | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        expires_at = time.time() + ttl if ttl is not None else None
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_entries (key, value, expires_at, tags) VALUES (?,?,?,?)",
                (key, json.dumps(value, default=str), expires_at, json.dumps(sorted(tags or []))),
            )
            self._conn.commit()

    def delete(self, key: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0

    def invalidate_tag(self, tag: str) -> int:
        with self._lock:
            rows = self._conn.execute("SELECT key, tags FROM cache_entries").fetchall()
            keys = [key for key, tags in rows if tag in json.loads(tags or "[]")]
            for key in keys:
                self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self._conn.commit()
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()
            return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        with self._lock:
            (entries,) = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        total = self.hits + self.misses
        return {
            "entries": entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
