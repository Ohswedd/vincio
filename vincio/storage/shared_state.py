"""Shared server state: rate limits, idempotency, and tenant quotas (2.1).

A single-process uvicorn worker can keep rate-limit counters and idempotency
records in memory, but the moment you scale to multiple workers (the whole
point of ``vincio serve --workers N``) that in-process state is incoherent —
each worker enforces its own limit and dedups its own writes. These protocols
let the server hold that state *shared*: the in-memory implementations here are
the single-process default, and the Redis-backed implementations
(:mod:`vincio.storage.redis`) make the same contract coherent across workers.

* :class:`RateLimiter` — fixed-window request limiting per key.
* :class:`IdempotencyStore` — remember a write's result so a retried request
  with the same idempotency key replays instead of re-executing.
* :class:`TenantQuotaManager` — per-tenant request budgets over a rolling
  window, layered on a rate limiter.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

__all__ = [
    "RateLimitDecision",
    "RateLimiter",
    "InMemoryRateLimiter",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "TenantQuotaManager",
]


class RateLimitDecision(BaseModel):
    """The verdict for one rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float = 0.0


@runtime_checkable
class RateLimiter(Protocol):
    def check(self, key: str, *, limit: int, window_s: float, cost: int = 1) -> RateLimitDecision:
        ...


class InMemoryRateLimiter:
    """Process-local fixed-window rate limiter (deterministic, clock-injectable)."""

    def __init__(self, *, clock: Any = None) -> None:
        self._clock = clock or time.monotonic
        self._windows: dict[str, list[float]] = {}  # key -> [window_index, count]

    def check(self, key: str, *, limit: int, window_s: float, cost: int = 1) -> RateLimitDecision:
        now = self._clock()
        window_index = float(int(now // window_s))
        state = self._windows.get(key)
        if state is None or state[0] != window_index:
            state = [window_index, 0.0]
            self._windows[key] = state
        used = int(state[1])
        retry_after = window_s - (now % window_s)
        if used + cost > limit:
            return RateLimitDecision(
                allowed=False, limit=limit, remaining=max(0, limit - used),
                retry_after_s=round(retry_after, 3),
            )
        state[1] = used + cost
        return RateLimitDecision(
            allowed=True, limit=limit, remaining=max(0, limit - int(state[1])), retry_after_s=0.0
        )


@runtime_checkable
class IdempotencyStore(Protocol):
    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any, *, ttl_s: float | None = None) -> None: ...


class InMemoryIdempotencyStore:
    """Process-local idempotency record store with TTL + bounded size."""

    def __init__(self, *, max_entries: int = 10_000, clock: Any = None) -> None:
        self._clock = clock or time.monotonic
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float | None, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and self._clock() >= expires_at:
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: Any, *, ttl_s: float | None = None) -> None:
        expires_at = (self._clock() + ttl_s) if ttl_s else None
        self._entries[key] = (expires_at, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


class TenantQuotaManager:
    """Per-tenant request quotas over a rolling window, on a rate limiter.

    Each tenant gets ``default_limit`` requests per ``window_s`` (overridable per
    tenant). Backed by any :class:`RateLimiter`, so a Redis limiter makes the
    quota coherent across workers.
    """

    def __init__(
        self,
        limiter: RateLimiter | None = None,
        *,
        default_limit: int,
        window_s: float = 60.0,
        per_tenant: dict[str, int] | None = None,
    ) -> None:
        self.limiter = limiter or InMemoryRateLimiter()
        self.default_limit = default_limit
        self.window_s = window_s
        self.per_tenant = dict(per_tenant or {})

    def limit_for(self, tenant_id: str | None) -> int:
        return self.per_tenant.get(tenant_id or "", self.default_limit)

    def check(self, tenant_id: str | None, *, cost: int = 1) -> RateLimitDecision:
        key = f"quota:{tenant_id or 'anonymous'}"
        return self.limiter.check(
            key, limit=self.limit_for(tenant_id), window_s=self.window_s, cost=cost
        )
