"""Warm candidate arena: reuse the prepared candidate set across compiles.

Collecting and normalizing candidates — building a validated
:class:`~vincio.context.scoring.ContextCandidate` for every evidence, memory,
and tool item, collapsing whitespace, and screening privacy scope — is
query-independent: it depends only on the inputs and the run's privacy scope,
not on the user's question or the token budget. The steady-state pattern of a
session (the same retrieved corpus, a new turn each time) re-does all of it on
every compile.

The arena memoizes that prepared set, keyed by a content fingerprint of the
inputs. When the candidate set is unchanged since a recent compile, the
compiler skips collection, normalization, and the privacy screen and works from
the cached set, so a warm recompile's cost is dominated by the query-dependent
scoring and selection rather than by re-allocating the candidate set.

The arena is safe under the compiler's concurrent use: stored entries are
immutable snapshots, every reuse hands back fresh per-compile copies (their own
``scores`` and ``metadata``, so in-place compression during selection never
touches the cache), and the bounded LRU is mutated under a lock.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..core.utils import stable_hash
from .scoring import ContextCandidate, ContextScores

__all__ = ["CandidateArena", "PreparedCandidates"]


@dataclass
class PreparedCandidates:
    """A query-independent prepared candidate set: the normalized candidates
    plus the privacy-screen exclusions they produced (so a warm compile's
    excluded report is identical to a cold one)."""

    candidates: list[ContextCandidate]
    excluded: list[dict[str, Any]] = field(default_factory=list)


def _fresh(candidate: ContextCandidate) -> ContextCandidate:
    """A compile-private copy: a new scores object and a new metadata dict.

    The source / image / table payloads are immutable references and are shared;
    only the fields selection mutates in place (``scores``, ``metadata``) are
    given fresh objects so concurrent compiles never alias them. The
    query-independent ``memory_value`` (a memory item's confidence, set at
    collection) is preserved — every other score is recomputed per query.
    """
    return candidate.model_copy(
        update={
            "scores": ContextScores(memory_value=candidate.scores.memory_value),
            "metadata": dict(candidate.metadata),
        }
    )


class CandidateArena:
    """Bounded, content-addressed cache of prepared candidate sets."""

    def __init__(self, *, max_entries: int = 32) -> None:
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, PreparedCandidates] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def fingerprint(signature: object) -> str:
        """Stable key for a query-independent candidate signature."""
        return stable_hash(signature)

    def get(self, key: str) -> PreparedCandidates | None:
        """Fresh copies of the prepared candidate set for *key*, or ``None``."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            snapshot = entry
        return PreparedCandidates(
            candidates=[_fresh(candidate) for candidate in snapshot.candidates],
            excluded=[dict(item) for item in snapshot.excluded],
        )

    def put(self, key: str, prepared: PreparedCandidates) -> None:
        """Snapshot the prepared candidate set under *key* (LRU-bounded)."""
        snapshot = PreparedCandidates(
            candidates=[_fresh(candidate) for candidate in prepared.candidates],
            excluded=[dict(item) for item in prepared.excluded],
        )
        with self._lock:
            self._entries[key] = snapshot
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
