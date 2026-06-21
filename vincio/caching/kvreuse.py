"""Cross-request KV-prefix reuse (caching/kvreuse).

The prompt compiler lays out a *stable prefix* — system role, rules, output
contract — before the volatile task, and the provider-aware cache strategy keeps
that prefix warm for one request's TTL. The rung above is **cross-request
reuse**: a family of requests that share the same stable head (the same compiled
``prompt_spec_hash`` on the same model) share the head's decode KV-cache, so the
serving engine computes it once and the rest of the family skips it.

In-process Vincio cannot reach into a serving engine's KV cache, so
:class:`KVPrefixPool` does the two things it *can* do without one: it **accounts**
the reuse — recognising when a request shares a head already seen and reporting
the KV bytes the shared head avoids recomputing — and it holds that accounting
under the resident-memory budget, the same bounded-LRU discipline the reasoning
and semantic caches use. The actual warming is performed by the existing
provider-aware prompt-cache strategy; the pool turns a one-request optimisation
into an observable, budgeted, family-wide one.

KV footprint is estimated from the prefix's token count (the decode KV cache
grows linearly in context length) via ``kv_bytes_per_token`` — the same model the
long-horizon :class:`~vincio.context.ContextBudget` uses, so the two compose.
Insertion order is a monotonic counter, not wall-clock, so a seeded run is
reproducible.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from pydantic import BaseModel

from ..core.utils import stable_hash

__all__ = [
    "kv_prefix_key",
    "KVPrefixEntry",
    "KVPrefixObservation",
    "KVReuseReport",
    "KVPrefixPool",
]

# Per-entry metadata footprint of the *pool* (not the KV cache itself, which
# lives in the serving engine): the key, prefix hash, model, and counters.
_ENTRY_OVERHEAD_BYTES = 128


def kv_prefix_key(prefix_hash: str, model: str) -> str:
    """Content-address a shared stable prefix.

    Two requests share a KV prefix when they share the compiled stable-prefix
    hash (``prefix_hash`` — the ``prompt_spec_hash``) on the same ``model``; the
    model is part of the key because KV cache is not portable across models.
    """
    return "kvprefix:" + stable_hash({"prefix": prefix_hash, "model": model})


class KVPrefixEntry(BaseModel):
    """One stable prefix tracked across the requests that share it.

    ``prefix_tokens`` is the head's token length; ``kv_bytes`` is its estimated
    serving-engine KV footprint. ``reuses`` counts the requests after the first
    that landed on this head — each one avoids recomputing ``kv_bytes``.
    """

    key: str
    prefix_hash: str
    model: str
    prefix_tokens: int = 0
    kv_bytes: int = 0
    reuses: int = 0
    seq: int = 0
    nbytes: int = 0


class KVPrefixObservation(BaseModel):
    """The result of observing one request's stable prefix.

    ``reused`` is ``True`` when the head was already in the pool, in which case
    ``kv_bytes_reused`` is the serving-engine KV the shared head avoids
    recomputing. ``family_size`` is how many requests have now shared this head
    (1 on first sight).
    """

    key: str
    prefix_hash: str
    model: str
    prefix_tokens: int
    reused: bool
    kv_bytes_reused: int
    family_size: int


class KVReuseReport(BaseModel):
    """Pool-wide reuse accounting, the residency analogue of a savings report.

    ``families`` is the number of distinct stable heads tracked;
    ``observations`` the total requests seen; ``reuses`` how many landed on a
    warm head; ``kv_bytes_reused`` the cumulative serving-engine KV those reuses
    avoided recomputing. ``resident_bytes`` is the pool's own bounded footprint.
    """

    families: int
    observations: int
    reuses: int
    kv_bytes_reused: int
    reuse_rate: float
    resident_bytes: int
    max_entries: int
    max_resident_bytes: int | None


class KVPrefixPool:
    """Bounded tracker of cross-request shared stable-prefix KV reuse.

    Each :meth:`observe` records a request's stable prefix and reports whether it
    reused a warm head and the KV bytes that reuse saved. The pool evicts
    lowest-recency-first to fit both an entry-count and an optional resident-byte
    ceiling, recording nothing it cannot hold — so it stays inside the
    resident-memory budget the rest of the platform honours.
    """

    def __init__(
        self,
        *,
        kv_bytes_per_token: int = 2048,
        max_entries: int = 256,
        max_resident_bytes: int | None = None,
    ) -> None:
        if kv_bytes_per_token < 0:
            raise ValueError("kv_bytes_per_token must be non-negative")
        if max_entries < 1:
            raise ValueError("KVPrefixPool requires max_entries >= 1")
        if max_resident_bytes is not None and max_resident_bytes < 0:
            raise ValueError("max_resident_bytes must be non-negative")
        self.kv_bytes_per_token = kv_bytes_per_token
        self.max_entries = max_entries
        self.max_resident_bytes = max_resident_bytes
        self._entries: OrderedDict[str, KVPrefixEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._seq = 0
        self.resident_bytes = 0
        self.observations = 0
        self.reuses = 0
        self.kv_bytes_reused = 0

    def observe(self, *, prefix_hash: str, model: str, prefix_tokens: int) -> KVPrefixObservation:
        """Record a request's stable prefix and report its reuse.

        First sight of a head is a miss (``reused=False``); every later request
        on the same head is a reuse that avoids recomputing the head's KV.
        """
        prefix_tokens = max(0, prefix_tokens)
        key = kv_prefix_key(prefix_hash, model)
        kv_bytes = prefix_tokens * self.kv_bytes_per_token
        with self._lock:
            self.observations += 1
            self._seq += 1
            existing = self._entries.get(key)
            if existing is not None:
                existing.reuses += 1
                # A later request may carry a longer prefix; track the largest.
                if prefix_tokens > existing.prefix_tokens:
                    existing.prefix_tokens = prefix_tokens
                    existing.kv_bytes = kv_bytes
                existing.seq = self._seq
                self._entries.move_to_end(key)
                self.reuses += 1
                reused_bytes = existing.kv_bytes
                self.kv_bytes_reused += reused_bytes
                return KVPrefixObservation(
                    key=key,
                    prefix_hash=prefix_hash,
                    model=model,
                    prefix_tokens=existing.prefix_tokens,
                    reused=True,
                    kv_bytes_reused=reused_bytes,
                    family_size=existing.reuses + 1,
                )
            entry = KVPrefixEntry(
                key=key,
                prefix_hash=prefix_hash,
                model=model,
                prefix_tokens=prefix_tokens,
                kv_bytes=kv_bytes,
                reuses=0,
                seq=self._seq,
                nbytes=_ENTRY_OVERHEAD_BYTES,
            )
            self._entries[key] = entry
            self.resident_bytes += entry.nbytes
            self._evict()
            return KVPrefixObservation(
                key=key,
                prefix_hash=prefix_hash,
                model=model,
                prefix_tokens=prefix_tokens,
                reused=False,
                kv_bytes_reused=0,
                family_size=1,
            )

    def clear(self) -> int:
        """Drop every tracked prefix and reclaim the footprint (counters persist)."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self.resident_bytes = 0
            return count

    def __len__(self) -> int:
        return len(self._entries)

    def report(self) -> KVReuseReport:
        """Pool-wide reuse accounting and resident footprint."""
        with self._lock:
            return KVReuseReport(
                families=len(self._entries),
                observations=self.observations,
                reuses=self.reuses,
                kv_bytes_reused=self.kv_bytes_reused,
                reuse_rate=round(self.reuses / self.observations, 4) if self.observations else 0.0,
                resident_bytes=self.resident_bytes,
                max_entries=self.max_entries,
                max_resident_bytes=self.max_resident_bytes,
            )

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries:
            _, victim = self._entries.popitem(last=False)
            self.resident_bytes -= victim.nbytes
        if self.max_resident_bytes is not None:
            while self.resident_bytes > self.max_resident_bytes and len(self._entries) > 1:
                _, victim = self._entries.popitem(last=False)
                self.resident_bytes -= victim.nbytes
