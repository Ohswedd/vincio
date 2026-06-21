"""Reasoning-trace-aware caching (caching/reasoning).

A reasoning model spends real tokens "thinking" before it answers. When a
re-ask shares a stable thinking prefix with an earlier call — the same system
prompt, role, rules, and output contract, differing only in the volatile task —
that thinking was already paid for once. :class:`ReasoningTraceCache` records
each reasoning trace keyed by the **thinking prefix** (the compiled stable-prefix
hash + model + effort), so a warm prefix is recognized and reused instead of
re-thought.

The cache is the reasoning-side analogue of the prompt compiler's
``ProgramCache`` and the context compiler's ``CandidateArena``: a bounded LRU
that lives under the same resident-memory budget the rest of the platform holds.
It evicts by both an entry count and an optional byte ceiling, lowest-recency
first, and tracks its hit rate so the saving is observable. It captures only the
reasoning *token cost* and an optional provider state handle — never the raw
chain-of-thought unless a caller explicitly supplies it — so it is safe to keep
resident.

The cache is deterministic and dependency-free: insertion order is tracked by a
monotonic counter (not wall-clock), so a seeded run is fully reproducible.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from pydantic import BaseModel, Field

from ..core.utils import stable_hash

__all__ = [
    "ReasoningTrace",
    "ReasoningTraceCache",
    "reasoning_prefix_key",
]

# Per-entry structural overhead charged on top of any captured trace text:
# the key, prefix hash, model id, effort, token count, and bookkeeping. Mirrors
# the context footprint estimator's per-entry charge so the two budgets compose.
_ENTRY_OVERHEAD_BYTES = 256


def reasoning_prefix_key(prefix_hash: str, model: str, effort: str | None) -> str:
    """Content-address a thinking prefix.

    Two calls share a reasoning trace when they share the stable prefix
    (``prefix_hash`` — the compiled prompt's ``prompt_spec_hash``), run on the
    same ``model``, and were asked to think at the same ``effort``. The effort is
    part of the key because a ``high``-effort trace is not interchangeable with a
    ``low``-effort one.
    """
    return "rtrace:" + stable_hash(
        {"prefix": prefix_hash, "model": model, "effort": effort or "default"}
    )


class ReasoningTrace(BaseModel):
    """One cached reasoning trace: how much thinking a warm prefix already cost.

    ``trace_text`` is empty unless a caller opts in to capturing the raw thinking
    (the cache never reaches into a provider's chain-of-thought itself).
    ``response_id`` carries a provider state handle (e.g. an OpenAI Responses
    ``previous_response_id``) so a continuation can resume the reasoning without
    resending it. ``nbytes`` is the entry's estimated resident footprint.
    """

    key: str
    prefix_hash: str
    model: str
    effort: str | None = None
    reasoning_tokens: int = 0
    trace_text: str = ""
    response_id: str | None = None
    nbytes: int = 0
    # Monotonic insert sequence — deterministic ordering without a wall clock.
    seq: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)

    def estimate_bytes(self) -> int:
        """Deterministic resident-byte estimate for this trace."""
        return _ENTRY_OVERHEAD_BYTES + len(self.trace_text.encode("utf-8"))


class ReasoningTraceCache:
    """Bounded LRU of reasoning traces under a resident-memory budget.

    ``max_entries`` caps the number of cached prefixes; ``max_resident_bytes``,
    when set, caps their total estimated footprint. On overflow the cache evicts
    the least-recently-used entries first — the same discipline ``ProgramCache``
    uses for compiled prefixes — recording nothing it cannot hold. Every access
    updates recency and the hit/miss counters, so :meth:`stats` reports the
    realized saving.
    """

    def __init__(self, *, max_entries: int = 128, max_resident_bytes: int | None = None) -> None:
        if max_entries < 1:
            raise ValueError("ReasoningTraceCache requires max_entries >= 1")
        if max_resident_bytes is not None and max_resident_bytes < 0:
            raise ValueError("max_resident_bytes must be non-negative")
        self.max_entries = max_entries
        self.max_resident_bytes = max_resident_bytes
        self._entries: OrderedDict[str, ReasoningTrace] = OrderedDict()
        self._lock = threading.Lock()
        self._seq = 0
        self.resident_bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(prefix_hash: str, model: str, effort: str | None = None) -> str:
        """The cache key for a thinking prefix (see :func:`reasoning_prefix_key`)."""
        return reasoning_prefix_key(prefix_hash, model, effort)

    def get(self, key: str) -> ReasoningTrace | None:
        """Return the cached trace for ``key``, marking it most-recently-used."""
        with self._lock:
            trace = self._entries.get(key)
            if trace is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return trace.model_copy()

    def lookup(
        self, prefix_hash: str, model: str, effort: str | None = None
    ) -> ReasoningTrace | None:
        """Convenience: :meth:`get` by prefix/model/effort instead of a raw key."""
        return self.get(self.key(prefix_hash, model, effort))

    def put(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Insert or refresh a trace, then evict to fit both ceilings.

        A re-insert under the same key replaces the old entry (its bytes are
        swapped, not double-counted). The stamped monotonic ``seq`` makes the
        insertion order reproducible without a wall clock.
        """
        with self._lock:
            self._seq += 1
            trace.seq = self._seq
            trace.nbytes = trace.estimate_bytes()
            existing = self._entries.pop(trace.key, None)
            if existing is not None:
                self.resident_bytes -= existing.nbytes
            self._entries[trace.key] = trace
            self.resident_bytes += trace.nbytes
            self._evict()
            return trace.model_copy()

    def record(
        self,
        *,
        prefix_hash: str,
        model: str,
        effort: str | None = None,
        reasoning_tokens: int = 0,
        trace_text: str = "",
        response_id: str | None = None,
    ) -> ReasoningTrace:
        """Record a reasoning trace from a completed call's outcome."""
        trace = ReasoningTrace(
            key=self.key(prefix_hash, model, effort),
            prefix_hash=prefix_hash,
            model=model,
            effort=effort,
            reasoning_tokens=max(0, reasoning_tokens),
            trace_text=trace_text,
            response_id=response_id,
        )
        return self.put(trace)

    def _evict(self) -> None:
        """Evict LRU-first until both the entry and byte ceilings are met."""
        while len(self._entries) > self.max_entries:
            _, victim = self._entries.popitem(last=False)
            self.resident_bytes -= victim.nbytes
        if self.max_resident_bytes is not None:
            # Keep at least one entry: a single oversized trace is still a hit.
            while self.resident_bytes > self.max_resident_bytes and len(self._entries) > 1:
                _, victim = self._entries.popitem(last=False)
                self.resident_bytes -= victim.nbytes

    def clear(self) -> None:
        """Drop every cached trace and reset the footprint (counters persist)."""
        with self._lock:
            self._entries.clear()
            self.resident_bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, object]:
        """Hit/miss counters, entry count, and resident footprint."""
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "resident_bytes": self.resident_bytes,
            "max_entries": self.max_entries,
            "max_resident_bytes": self.max_resident_bytes,
        }
