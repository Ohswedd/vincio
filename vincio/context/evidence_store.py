"""Content-addressed evidence store backing slim Context Packets.

A *slim* packet holds evidence by content hash instead of duplicating the text,
and materializes it lazily from the in-process Context IR. That works within one
process, but a packet shipped to another worker has no IR to read from. Backing
slim packets with a content-addressed store closes that gap: the text is written
once under its hash, and ``ContextPacket.materialize(store=...)`` resolves it
after deserialization — true zero-copy cross-process packet shipping.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

__all__ = ["content_hash", "EvidenceStore", "InMemoryEvidenceStore", "BlobEvidenceStore"]


def content_hash(content: str) -> str:
    """Stable 16-hex content address (matches the packet's evidence ref)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class EvidenceStore(Protocol):
    """Maps a content hash to its text. Implementations may be in-memory or
    persistent (a blob store, an object store)."""

    def put(self, content: str) -> str: ...

    def get(self, content_hash: str) -> str | None: ...


class InMemoryEvidenceStore:
    """Process-local content-addressed store (a dedup cache by hash)."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def put(self, content: str) -> str:
        digest = content_hash(content)
        self._data.setdefault(digest, content)
        return digest

    def get(self, content_hash: str) -> str | None:
        return self._data.get(content_hash)

    def __len__(self) -> int:
        return len(self._data)


class BlobEvidenceStore:
    """Content-addressed store over a :class:`~vincio.storage.base.BlobStore`,
    so evidence text persists across processes for cross-worker materialization.
    """

    def __init__(self, blob_store: Any, *, prefix: str = "evidence/") -> None:
        self._blobs = blob_store
        self._prefix = prefix

    def put(self, content: str) -> str:
        digest = content_hash(content)
        self._blobs.put(f"{self._prefix}{digest}", content.encode("utf-8"))
        return digest

    def get(self, content_hash: str) -> str | None:
        data = self._blobs.get(f"{self._prefix}{content_hash}")
        return data.decode("utf-8") if data is not None else None
