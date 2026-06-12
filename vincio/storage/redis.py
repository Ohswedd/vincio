"""Redis cache backend implementing the CacheBackend protocol.
Requires ``pip install "vincio[redis]"``."""

from __future__ import annotations

import json
from typing import Any

from ..core.errors import StorageError

__all__ = ["RedisCache"]


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
