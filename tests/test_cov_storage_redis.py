"""Real-behavior coverage for vincio.storage.redis.

The redis-absent contract (``RedisCache.__init__``, ``_redis_client`` without
an injected client raise ``StorageError``) is exercised deterministically by
poisoning ``sys.modules["redis"]`` with ``monkeypatch`` — the lazy ``import
redis`` then raises ``ImportError`` whether or not redis-py is installed on
the host, so the suite passes identically everywhere. Everything else is
exercised against a small deterministic in-memory fake that honours the real
redis-py call surface (``get``/``set``/``sadd``/``expire``/``smembers``/
``scan_iter``/``delete``/``incrby``/``ttl``). ``RedisCache`` has no client
injection, so we build it with ``object.__new__`` and wire the fake into
``_redis`` — every method below runs the real ``RedisCache`` code.

No network, no API key, no unittest.mock — a real object graph throughout.
"""

from __future__ import annotations

import fnmatch
import json
import sys

import pytest

from vincio.core.errors import StorageError
from vincio.storage.redis import (
    RedisCache,
    RedisIdempotencyStore,
    RedisRateLimiter,
)
from vincio.storage.shared_state import RateLimitDecision


class FakeRedis:
    """Deterministic in-memory stand-in for a ``decode_responses=True`` client.

    Models exactly the operations the storage adapter calls. ``incrby`` /
    ``ttl`` / ``expire`` track an explicit TTL so rate-limiter logic is real.
    """

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.ttls: dict[str, int] = {}

    # --- string ops -----------------------------------------------------
    def get(self, key: str):
        return self.kv.get(key)

    def set(self, key: str, value, ex=None) -> None:
        self.kv[key] = value
        if ex is not None:
            self.ttls[key] = ex

    def incrby(self, key: str, amount: int) -> int:
        new = int(self.kv.get(key, 0)) + amount
        self.kv[key] = str(new)
        return new

    # --- key ops --------------------------------------------------------
    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.kv:
                del self.kv[key]
                removed += 1
            if key in self.sets:
                del self.sets[key]
                removed += 1
            self.ttls.pop(key, None)
        return removed

    def expire(self, key: str, seconds: int) -> bool:
        if key in self.kv or key in self.sets:
            self.ttls[key] = seconds
            return True
        return False

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)

    # --- set ops --------------------------------------------------------
    def sadd(self, key: str, *members: str) -> int:
        bucket = self.sets.setdefault(key, set())
        added = 0
        for m in members:
            if m not in bucket:
                bucket.add(m)
                added += 1
        return added

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    # --- scan -----------------------------------------------------------
    def scan_iter(self, match: str = "*"):
        all_keys = set(self.kv) | set(self.sets)
        for key in sorted(all_keys):
            if fnmatch.fnmatch(key, match):
                yield key


def make_cache(**kwargs) -> RedisCache:
    """Build a real RedisCache wired to a FakeRedis, bypassing the redis import."""
    cache = object.__new__(RedisCache)
    cache._redis = FakeRedis()
    cache.prefix = kwargs.get("prefix", "vincio:")
    cache.default_ttl_s = kwargs.get("default_ttl_s", 3600.0)
    cache.hits = 0
    cache.misses = 0
    return cache


# ---------------------------------------------------------------------------
# Not-installed branches (lines 24-28, 39-47)
# ---------------------------------------------------------------------------
class TestRedisNotInstalled:
    """The redis-absent contract, forced deterministically on any host.

    ``sys.modules["redis"] = None`` makes the lazy ``import redis`` raise
    ``ImportError`` even when redis-py is installed locally, so these tests
    assert the same contract in every environment instead of depending on
    what happens to be installed.
    """

    def test_redis_client_helper_raises_without_client(self, monkeypatch):
        from vincio.storage.redis import _redis_client

        monkeypatch.setitem(sys.modules, "redis", None)
        with pytest.raises(StorageError, match=r'pip install "vincio\[redis\]"'):
            _redis_client("redis://localhost:6379/0", None)

    def test_redis_client_helper_returns_injected_client(self):
        from vincio.storage.redis import _redis_client

        fake = FakeRedis()
        assert _redis_client("redis://ignored", fake) is fake

    def test_rediscache_init_raises_without_redis_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "redis", None)
        with pytest.raises(StorageError, match=r'pip install "vincio\[redis\]"'):
            RedisCache()

    def test_storage_error_chains_importerror_cause(self, monkeypatch):
        from vincio.storage.redis import _redis_client

        monkeypatch.setitem(sys.modules, "redis", None)
        with pytest.raises(StorageError) as exc:
            _redis_client("redis://x", None)
        assert isinstance(exc.value.__cause__, ImportError)


# ---------------------------------------------------------------------------
# RedisCache.get / set / miss / hit accounting (lines 55-70)
# ---------------------------------------------------------------------------
class TestRedisCacheGetSet:
    def test_get_miss_returns_none_and_counts(self):
        cache = make_cache()
        assert cache.get("absent") is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_set_then_get_roundtrips_value_and_counts_hit(self):
        cache = make_cache()
        cache.set("k", {"a": 1, "b": [2, 3]})
        assert cache.get("k") == {"a": 1, "b": [2, 3]}
        assert cache.hits == 1
        assert cache.misses == 0

    def test_set_applies_prefix_to_stored_key(self):
        cache = make_cache(prefix="t:")
        cache.set("k", 5)
        assert "t:k" in cache._redis.kv
        assert json.loads(cache._redis.kv["t:k"]) == 5

    def test_set_default_ttl_is_used_when_ttl_s_omitted(self):
        cache = make_cache(default_ttl_s=120.0)
        cache.set("k", 1)
        assert cache._redis.ttls["vincio:k"] == 120

    def test_set_explicit_ttl_overrides_default(self):
        cache = make_cache(default_ttl_s=3600.0)
        cache.set("k", 1, ttl_s=10.0)
        assert cache._redis.ttls["vincio:k"] == 10

    def test_set_with_no_ttl_stores_without_expiry(self):
        cache = make_cache(default_ttl_s=None)
        cache.set("k", 1)
        assert "vincio:k" not in cache._redis.ttls

    def test_set_serialises_non_json_types_via_default_str(self):
        from datetime import date

        cache = make_cache()
        cache.set("k", {"d": date(2020, 1, 2)})
        # default=str means the date round-trips as a string, not an error.
        assert cache.get("k") == {"d": "2020-01-02"}


# ---------------------------------------------------------------------------
# Tag tracking + invalidate_tag (lines 67-70, 75-82)
# ---------------------------------------------------------------------------
class TestRedisCacheTags:
    def test_set_with_tags_registers_key_under_each_tag_set(self):
        cache = make_cache()
        cache.set("a", 1, tags=["x", "y"])
        assert cache._redis.smembers("vincio:tag:x") == {"vincio:a"}
        assert cache._redis.smembers("vincio:tag:y") == {"vincio:a"}

    def test_set_with_tags_expires_tag_set_at_double_ttl(self):
        cache = make_cache()
        cache.set("a", 1, ttl_s=30.0, tags=["x"])
        assert cache._redis.ttls["vincio:tag:x"] == 60

    def test_set_with_tags_but_no_ttl_does_not_expire_tag_set(self):
        cache = make_cache(default_ttl_s=None)
        cache.set("a", 1, tags=["x"])
        assert "vincio:tag:x" not in cache._redis.ttls
        assert cache._redis.smembers("vincio:tag:x") == {"vincio:a"}

    def test_invalidate_tag_removes_members_and_returns_count(self):
        cache = make_cache()
        cache.set("a", 1, tags=["grp"])
        cache.set("b", 2, tags=["grp"])
        removed = cache.invalidate_tag("grp")
        assert removed == 2
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_invalidate_tag_deletes_the_tag_set_itself(self):
        cache = make_cache()
        cache.set("a", 1, tags=["grp"])
        cache.invalidate_tag("grp")
        assert cache._redis.smembers("vincio:tag:grp") == set()

    def test_invalidate_unknown_tag_returns_zero(self):
        cache = make_cache()
        assert cache.invalidate_tag("never") == 0


# ---------------------------------------------------------------------------
# delete / clear / stats (lines 72-73, 84-97)
# ---------------------------------------------------------------------------
class TestRedisCacheLifecycle:
    def test_delete_present_key_returns_true(self):
        cache = make_cache()
        cache.set("k", 1)
        assert cache.delete("k") is True
        assert cache.get("k") is None

    def test_delete_absent_key_returns_false(self):
        cache = make_cache()
        assert cache.delete("ghost") is False

    def test_clear_empty_cache_returns_zero(self):
        cache = make_cache()
        assert cache.clear() == 0

    def test_clear_removes_only_prefixed_keys(self):
        cache = make_cache(prefix="p:")
        cache.set("a", 1)
        cache.set("b", 2)
        cache._redis.kv["other:c"] = "3"  # foreign key, must survive
        removed = cache.clear()
        assert removed == 2
        assert "other:c" in cache._redis.kv

    def test_stats_hit_rate_zero_when_no_lookups(self):
        cache = make_cache()
        stats = cache.stats()
        assert stats == {"entries": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}

    def test_stats_counts_entries_and_computes_hit_rate(self):
        cache = make_cache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("z")  # miss
        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == round(2 / 3, 4)


# ---------------------------------------------------------------------------
# RedisRateLimiter (lines 114-130)
# ---------------------------------------------------------------------------
class TestRedisRateLimiter:
    def test_allows_until_limit_then_blocks(self):
        rl = RedisRateLimiter(client=FakeRedis())
        d1 = rl.check("u", limit=2, window_s=60)
        d2 = rl.check("u", limit=2, window_s=60)
        d3 = rl.check("u", limit=2, window_s=60)
        assert (d1.allowed, d2.allowed, d3.allowed) == (True, True, False)
        assert isinstance(d3, RateLimitDecision)

    def test_remaining_decrements_with_each_allowed_call(self):
        rl = RedisRateLimiter(client=FakeRedis())
        assert rl.check("u", limit=3, window_s=60).remaining == 2
        assert rl.check("u", limit=3, window_s=60).remaining == 1
        assert rl.check("u", limit=3, window_s=60).remaining == 0

    def test_blocked_decision_reports_zero_remaining(self):
        rl = RedisRateLimiter(client=FakeRedis())
        rl.check("u", limit=1, window_s=60)
        blocked = rl.check("u", limit=1, window_s=60)
        assert blocked.allowed is False
        assert blocked.remaining == 0

    def test_first_call_sets_window_expiry(self):
        fake = FakeRedis()
        rl = RedisRateLimiter(client=fake)
        rl.check("u", limit=5, window_s=60)
        # Exactly one window key created, given a TTL of window_s + 1.
        assert any(v == 61 for v in fake.ttls.values())

    def test_cost_consumes_multiple_units_at_once(self):
        rl = RedisRateLimiter(client=FakeRedis())
        d = rl.check("u", limit=10, window_s=60, cost=4)
        assert d.allowed is True
        assert d.remaining == 6

    def test_blocked_retry_after_uses_positive_ttl(self):
        fake = FakeRedis()
        rl = RedisRateLimiter(client=fake)
        rl.check("u", limit=1, window_s=60)
        blocked = rl.check("u", limit=1, window_s=60)
        # FakeRedis recorded ttl=61 on the window key; that is surfaced.
        assert blocked.retry_after_s == 61.0

    def test_blocked_retry_after_falls_back_to_window_when_ttl_missing(self):
        class NoTTL(FakeRedis):
            def ttl(self, key):  # type: ignore[override]
                return -2  # redis sentinel: key has no TTL

        rl = RedisRateLimiter(client=NoTTL())
        rl.check("u", limit=1, window_s=45)
        blocked = rl.check("u", limit=1, window_s=45)
        assert blocked.retry_after_s == 45.0

    def test_distinct_keys_have_independent_counters(self):
        rl = RedisRateLimiter(client=FakeRedis())
        rl.check("a", limit=1, window_s=60)
        # "b" is untouched, so it is still allowed.
        assert rl.check("b", limit=1, window_s=60).allowed is True

    def test_custom_prefix_namespaces_the_window_key(self):
        fake = FakeRedis()
        rl = RedisRateLimiter(client=fake, prefix="myrl:")
        rl.check("u", limit=5, window_s=60)
        assert all(k.startswith("myrl:u:") for k in fake.kv)


# ---------------------------------------------------------------------------
# RedisIdempotencyStore (lines 142-149)
# ---------------------------------------------------------------------------
class TestRedisIdempotencyStore:
    def test_get_missing_key_returns_none(self):
        store = RedisIdempotencyStore(client=FakeRedis())
        assert store.get("nope") is None

    def test_put_then_get_roundtrips_payload(self):
        store = RedisIdempotencyStore(client=FakeRedis())
        store.put("k", {"status": "ok", "n": 7})
        assert store.get("k") == {"status": "ok", "n": 7}

    def test_put_applies_prefix(self):
        fake = FakeRedis()
        store = RedisIdempotencyStore(client=fake, prefix="idem:")
        store.put("abc", 1)
        assert "idem:abc" in fake.kv

    def test_put_with_ttl_records_integer_expiry(self):
        fake = FakeRedis()
        store = RedisIdempotencyStore(client=fake)
        store.put("k", 1, ttl_s=90.5)
        assert fake.ttls["vincio:idem:k"] == 90

    def test_put_without_ttl_stores_without_expiry(self):
        fake = FakeRedis()
        store = RedisIdempotencyStore(client=fake)
        store.put("k", 1)
        assert "vincio:idem:k" not in fake.ttls

    def test_put_serialises_non_json_via_default_str(self):
        from datetime import date

        store = RedisIdempotencyStore(client=FakeRedis())
        store.put("k", {"when": date(2021, 5, 6)})
        assert store.get("k") == {"when": "2021-05-06"}
