"""Redis cache backend + shared server state (rate limit / idempotency).

The :class:`RedisCache` implements the ``CacheBackend`` protocol; the 2.1
:class:`RedisRateLimiter` and :class:`RedisIdempotencyStore` implement the
shared-state protocols (:mod:`vincio.storage.shared_state`) so a multi-worker
``vincio serve`` deployment enforces one coherent rate limit and dedups writes
across every worker. Requires ``pip install "vincio[redis]"`` (or inject a
client for tests)."""

from __future__ import annotations

import json
from typing import Any

from ..core.errors import StorageError
from .shared_state import RateLimitDecision

__all__ = ["RedisCache", "RedisRateLimiter", "RedisIdempotencyStore"]


def _redis_client(url: str, client: Any | None) -> Any:
    if client is not None:
        return client
    try:
        import redis
    except ImportError as exc:
        raise StorageError('Redis support requires: pip install "vincio[redis]"') from exc
    return redis.Redis.from_url(url, decode_responses=True)


class RedisCache:
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "vincio:",
        default_ttl_s: float | None = 3600.0,
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise StorageError('Redis support requires: pip install "vincio[redis]"') from exc
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix
        self.default_ttl_s = default_ttl_s
        self.hits = 0
        self.misses = 0

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _tag_key(self, tag: str) -> str:
        return f"{self.prefix}tag:{tag}"

    def get(self, key: str) -> Any | None:
        value = self._redis.get(self._key(key))
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(value)

    def set(self, key: str, value: Any, *, ttl_s: float | None = None, tags: list[str] | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        full_key = self._key(key)
        self._redis.set(full_key, json.dumps(value, default=str), ex=int(ttl) if ttl else None)
        for tag in tags or []:
            self._redis.sadd(self._tag_key(tag), full_key)
            if ttl:
                self._redis.expire(self._tag_key(tag), int(ttl) * 2)

    def delete(self, key: str) -> bool:
        return bool(self._redis.delete(self._key(key)))

    def invalidate_tag(self, tag: str) -> int:
        tag_key = self._tag_key(tag)
        members = self._redis.smembers(tag_key)
        removed = 0
        if members:
            removed = self._redis.delete(*members)
        self._redis.delete(tag_key)
        return removed

    def clear(self) -> int:
        keys = list(self._redis.scan_iter(match=f"{self.prefix}*"))
        if not keys:
            return 0
        return self._redis.delete(*keys)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": sum(1 for _ in self._redis.scan_iter(match=f"{self.prefix}*")),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class RedisRateLimiter:
    """Cross-worker fixed-window rate limiter (``INCRBY`` + ``EXPIRE``).

    The counter for ``(key, window)`` lives in Redis, so every uvicorn worker
    sees the same count — the coherence a multi-worker deployment needs. Window
    boundaries are derived from wall-clock so all workers agree on the bucket.
    """

    def __init__(
        self, url: str = "redis://localhost:6379/0", *, prefix: str = "vincio:rl:", client: Any | None = None
    ) -> None:
        self._redis = _redis_client(url, client)
        self.prefix = prefix

    def check(self, key: str, *, limit: int, window_s: float, cost: int = 1) -> RateLimitDecision:
        import time

        window = int(time.time() // window_s)
        full = f"{self.prefix}{key}:{window}"
        used = int(self._redis.incrby(full, cost))
        if used == cost:
            self._redis.expire(full, int(window_s) + 1)
        if used > limit:
            ttl = self._redis.ttl(full)
            return RateLimitDecision(
                allowed=False, limit=limit, remaining=0,
                retry_after_s=float(ttl if ttl and ttl > 0 else window_s),
            )
        return RateLimitDecision(
            allowed=True, limit=limit, remaining=max(0, limit - used), retry_after_s=0.0
        )


class RedisIdempotencyStore:
    """Cross-worker idempotency record store (``SET`` with TTL)."""

    def __init__(
        self, url: str = "redis://localhost:6379/0", *, prefix: str = "vincio:idem:", client: Any | None = None
    ) -> None:
        self._redis = _redis_client(url, client)
        self.prefix = prefix

    def get(self, key: str) -> Any | None:
        raw = self._redis.get(f"{self.prefix}{key}")
        return json.loads(raw) if raw is not None else None

    def put(self, key: str, value: Any, *, ttl_s: float | None = None) -> None:
        self._redis.set(
            f"{self.prefix}{key}", json.dumps(value, default=str), ex=int(ttl_s) if ttl_s else None
        )
