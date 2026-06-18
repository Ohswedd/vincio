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

__all__ = [
    "MetadataStore",
    "AsyncMetadataStore",
    "InMemoryMetadataStore",
    "BlobStore",
    "FileBlobStore",
    "create_metadata_store",
    "parse_storage_url",
    "asave",
    "aget",
    "aquery",
    "adelete",
    "acount",
]

RECORD_KINDS = ("runs", "context_packets", "eval_results", "documents", "chunks", "tool_calls", "traces")


class MetadataStore(Protocol):
    def save(self, kind: str, record: dict[str, Any]) -> None: ...

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None: ...

    def query(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]: ...

    def delete(self, kind: str, record_id: str) -> bool: ...

    def count(self, kind: str) -> int: ...


class AsyncMetadataStore(Protocol):
    """The canonical async store contract (2.0).

    Async is the contract the run path binds to — the module-level
    :func:`asave` / :func:`aget` / :func:`aquery` / :func:`adelete` /
    :func:`acount` helpers dispatch to these coroutines when a store implements
    them (e.g. the psycopg3-async Postgres pool), and otherwise run the
    synchronous :class:`MetadataStore` methods in a worker thread. So a sync
    store keeps working unchanged, while an async-native store never touches the
    event loop with blocking I/O. Implementing the sync :class:`MetadataStore`
    is the thin shim; implementing this is the fast path.
    """

    async def asave(self, kind: str, record: dict[str, Any]) -> None: ...

    async def aget(self, kind: str, record_id: str) -> dict[str, Any] | None: ...

    async def aquery(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]: ...

    async def adelete(self, kind: str, record_id: str) -> bool: ...

    async def acount(self, kind: str) -> int: ...


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

    # Async-native methods (3.0): the in-memory store is the canonical async
    # contract's reference implementation — its operations are non-blocking, so
    # the module-level asave/aquery helpers take the native fast path (no
    # worker-thread hop) rather than the sync shim.

    async def asave(self, kind: str, record: dict[str, Any]) -> None:
        self.save(kind, record)

    async def aget(self, kind: str, record_id: str) -> dict[str, Any] | None:
        return self.get(kind, record_id)

    async def aquery(
        self, kind: str, *, where: dict[str, Any] | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        return self.query(kind, where=where, limit=limit, offset=offset)

    async def adelete(self, kind: str, record_id: str) -> bool:
        return self.delete(kind, record_id)

    async def acount(self, kind: str) -> int:
        return self.count(kind)


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


_DISCOVERED_STORES: dict[str, Any] | None = None


def _discovered_stores() -> dict[str, Any]:
    """Third-party metadata stores advertised under the ``vincio.stores``
    entry-point group (discovered once, then cached). A factory takes the
    parsed *location* string and returns a :class:`MetadataStore`."""
    global _DISCOVERED_STORES
    if _DISCOVERED_STORES is None:
        from ..providers.registry import discover_entry_points

        _DISCOVERED_STORES = discover_entry_points("vincio.stores")
    return _DISCOVERED_STORES


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
    discovered = _discovered_stores()
    if scheme in discovered:
        return discovered[scheme](location)
    raise StorageError(f"unsupported metadata storage scheme: {scheme!r}")


async def asave(store: MetadataStore, kind: str, record: dict[str, Any]) -> None:
    """Persist a record off the event loop.

    Prefers a store-native ``asave`` coroutine; otherwise runs the synchronous
    :meth:`MetadataStore.save` in a worker thread so disk/network writes never
    block the run pipeline. (1.7)
    """
    native = getattr(store, "asave", None)
    if native is not None:
        await native(kind, record)
        return
    import asyncio

    await asyncio.to_thread(store.save, kind, record)


async def aget(store: MetadataStore, kind: str, record_id: str) -> dict[str, Any] | None:
    """Fetch one record off the event loop (native ``aget`` or threaded ``get``)."""
    native = getattr(store, "aget", None)
    if native is not None:
        return await native(kind, record_id)
    import asyncio

    return await asyncio.to_thread(store.get, kind, record_id)


async def aquery(
    store: MetadataStore,
    kind: str,
    *,
    where: dict[str, Any] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Query records off the event loop (native ``aquery`` or threaded ``query``)."""
    native = getattr(store, "aquery", None)
    if native is not None:
        return await native(kind, where=where, limit=limit, offset=offset)
    import asyncio

    return await asyncio.to_thread(
        lambda: store.query(kind, where=where, limit=limit, offset=offset)
    )


async def adelete(store: MetadataStore, kind: str, record_id: str) -> bool:
    """Delete one record off the event loop (native ``adelete`` or threaded ``delete``)."""
    native = getattr(store, "adelete", None)
    if native is not None:
        return await native(kind, record_id)
    import asyncio

    return await asyncio.to_thread(store.delete, kind, record_id)


async def acount(store: MetadataStore, kind: str) -> int:
    """Count records off the event loop (native ``acount`` or threaded ``count``)."""
    native = getattr(store, "acount", None)
    if native is not None:
        return await native(kind)
    import asyncio

    return await asyncio.to_thread(store.count, kind)
