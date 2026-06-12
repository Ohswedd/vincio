"""Storage abstractions.

- :class:`MetadataStore` — runs, context packets, eval results, documents,
  chunks, tool calls (protocol + in-memory implementation).
- :class:`BlobStore` — document artifacts on the filesystem.
- URL-based factory: ``sqlite:///path``, ``memory://``, ``duckdb:///path``,
  ``postgres://…``, ``qdrant://host:port``, ``redis://host:port``,
  ``neo4j://host``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from ..core.errors import StorageError

__all__ = ["MetadataStore", "InMemoryMetadataStore", "BlobStore", "FileBlobStore", "create_metadata_store", "parse_storage_url"]

RECORD_KINDS = ("runs", "context_packets", "eval_results", "documents", "chunks", "tool_calls", "traces")


class MetadataStore(Protocol):
    def save(self, kind: str, record: dict[str, Any]) -> None: ...

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None: ...

    def query(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]: ...

    def delete(self, kind: str, record_id: str) -> bool: ...

    def count(self, kind: str) -> int: ...


class InMemoryMetadataStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {kind: {} for kind in RECORD_KINDS}
        self._lock = threading.Lock()

    def _table(self, kind: str) -> dict[str, dict[str, Any]]:
        if kind not in self._data:
            self._data[kind] = {}
        return self._data[kind]

    def save(self, kind: str, record: dict[str, Any]) -> None:
        record_id = record.get("id")
        if not record_id:
            raise StorageError(f"record for {kind!r} has no id")
        with self._lock:
            self._table(kind)[str(record_id)] = dict(record)

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        record = self._table(kind).get(record_id)
        return dict(record) if record else None

    def query(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        records = list(self._table(kind).values())
        if where:
            records = [r for r in records if all(r.get(k) == v for k, v in where.items())]
        return [dict(r) for r in records[offset : offset + limit]]

    def delete(self, kind: str, record_id: str) -> bool:
        with self._lock:
            return self._table(kind).pop(record_id, None) is not None

    def count(self, kind: str) -> int:
        return len(self._table(kind))


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...

    def get(self, key: str) -> bytes | None: ...

    def delete(self, key: str) -> bool: ...


class FileBlobStore:
    """Document artifacts on the local filesystem."""

    def __init__(self, directory: str | Path = ".vincio/documents") -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        safe = key.replace("..", "_").lstrip("/")
        return self.directory / safe

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.is_file():
            path.unlink()
            return True
        return False


def parse_storage_url(url: str) -> tuple[str, str]:
    """Returns (scheme, location)."""
    if url.startswith("memory://"):
        return "memory", ""
    parsed = urlparse(url)
    if not parsed.scheme:
        raise StorageError(f"invalid storage url: {url!r}")
    if parsed.scheme in ("sqlite", "duckdb"):
        # sqlite:///relative/path or sqlite:////absolute/path
        location = (parsed.netloc + parsed.path).lstrip("/")
        if url.count("/") >= 4 and parsed.path.startswith("//"):
            location = "/" + location
        return parsed.scheme, location
    return parsed.scheme, url


def create_metadata_store(url: str) -> MetadataStore:
    scheme, location = parse_storage_url(url)
    if scheme == "memory":
        return InMemoryMetadataStore()
    if scheme == "sqlite":
        from .sqlite import SQLiteMetadataStore

        return SQLiteMetadataStore(location or ".vincio/vincio.db")
    if scheme in ("postgres", "postgresql"):
        from .postgres import PostgresMetadataStore

        return PostgresMetadataStore(location)
    raise StorageError(f"unsupported metadata storage scheme: {scheme!r}")
