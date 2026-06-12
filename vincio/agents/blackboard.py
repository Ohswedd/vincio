"""Shared blackboard (agents/blackboard).

A :class:`Blackboard` is the shared working memory of a multi-agent crew:
agents post findings under keys, every post keeps its author and version
history, and the whole board snapshots to plain JSON so crew runs can be
persisted, replayed, and diffed. Posts optionally emit ``blackboard.posted``
events on the app event bus so other components (traces, evals, memory
write-back) can observe coordination as it happens.
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, Field

from ..core.events import EventBus
from ..core.utils import new_id, utcnow

__all__ = ["BlackboardEntry", "Blackboard"]


class BlackboardEntry(BaseModel):
    id: str = Field(default_factory=lambda: new_id("bb"))
    key: str
    value: Any = None
    author: str = ""
    version: int = 1
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class Blackboard:
    """Versioned, author-attributed shared memory for agent teams."""

    def __init__(self, *, event_bus: EventBus | None = None) -> None:
        self._entries: dict[str, BlackboardEntry] = {}
        self._history: dict[str, list[BlackboardEntry]] = {}
        self._lock = threading.Lock()
        self.event_bus = event_bus

    def post(
        self, key: str, value: Any, *, author: str = "", **metadata: Any
    ) -> BlackboardEntry:
        """Write ``value`` under ``key``; prior versions are kept in history."""
        with self._lock:
            previous = self._entries.get(key)
            entry = BlackboardEntry(
                key=key,
                value=value,
                author=author,
                version=(previous.version + 1) if previous else 1,
                metadata=metadata,
            )
            self._entries[key] = entry
            self._history.setdefault(key, []).append(entry)
        if self.event_bus is not None:
            self.event_bus.emit(
                "blackboard.posted", {"key": key, "author": author, "version": entry.version}
            )
        return entry

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._entries.get(key)
        return entry.value if entry else default

    def entry(self, key: str) -> BlackboardEntry | None:
        with self._lock:
            return self._entries.get(key)

    def history(self, key: str) -> list[BlackboardEntry]:
        with self._lock:
            return list(self._history.get(key, []))

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def entries(self) -> list[BlackboardEntry]:
        """Current entries in post order (latest version per key)."""
        with self._lock:
            return list(self._entries.values())

    def as_context(self, *, max_chars_per_entry: int = 600) -> str:
        """Render the board for inclusion in an agent's working context."""
        lines = []
        for entry in self.entries():
            value = str(entry.value)[:max_chars_per_entry]
            author = f" (by {entry.author})" if entry.author else ""
            lines.append(f"- {entry.key}{author}: {value}")
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable snapshot: latest entries plus full history."""
        with self._lock:
            return {
                "entries": {k: e.model_dump(mode="json") for k, e in self._entries.items()},
                "history": {
                    k: [e.model_dump(mode="json") for e in items]
                    for k, items in self._history.items()
                },
            }

    @classmethod
    def restore(cls, snapshot: dict[str, Any], *, event_bus: EventBus | None = None) -> Blackboard:
        board = cls(event_bus=event_bus)
        board._entries = {
            k: BlackboardEntry.model_validate(v) for k, v in snapshot.get("entries", {}).items()
        }
        board._history = {
            k: [BlackboardEntry.model_validate(e) for e in items]
            for k, items in snapshot.get("history", {}).items()
        }
        return board
