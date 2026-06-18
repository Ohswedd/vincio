"""Redis-backed shared server state + content-capture controls + vincio serve
. Shared state and content gating are exercised offline; the server
endpoints use FastAPI's TestClient; the Redis classes use an injected fake."""

from __future__ import annotations

import pytest

from vincio.observability import ContentCapturePolicy
from vincio.storage.shared_state import (
    InMemoryIdempotencyStore,
    InMemoryRateLimiter,
    TenantQuotaManager,
)

# ---------------------------------------------------------------------------
# Shared state: rate limiting, idempotency, quotas
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_fixed_window_allows_then_denies(self):
        clock = {"t": 0.0}
        rl = InMemoryRateLimiter(clock=lambda: clock["t"])
        for _ in range(3):
            assert rl.check("k", limit=3, window_s=60).allowed
        denied = rl.check("k", limit=3, window_s=60)
        assert not denied.allowed and denied.remaining == 0 and denied.retry_after_s > 0

    def test_window_resets(self):
        clock = {"t": 0.0}
        rl = InMemoryRateLimiter(clock=lambda: clock["t"])
        assert rl.check("k", limit=1, window_s=60).allowed
        assert not rl.check("k", limit=1, window_s=60).allowed
        clock["t"] = 61.0
        assert rl.check("k", limit=1, window_s=60).allowed

    def test_keys_isolated(self):
        rl = InMemoryRateLimiter()
        assert rl.check("a", limit=1, window_s=60).allowed
        assert rl.check("b", limit=1, window_s=60).allowed


class TestIdempotencyStore:
    def test_get_put_roundtrip(self):
        store = InMemoryIdempotencyStore()
        assert store.get("k") is None
        store.put("k", {"result": 42})
        assert store.get("k") == {"result": 42}

    def test_ttl_expiry(self):
        clock = {"t": 0.0}
        store = InMemoryIdempotencyStore(clock=lambda: clock["t"])
        store.put("k", "v", ttl_s=10)
        assert store.get("k") == "v"
        clock["t"] = 11.0
        assert store.get("k") is None

    def test_bounded_size_evicts_oldest(self):
        store = InMemoryIdempotencyStore(max_entries=2)
        store.put("a", 1)
        store.put("b", 2)
        store.put("c", 3)
        assert store.get("a") is None and store.get("c") == 3


class TestTenantQuota:
    def test_per_tenant_limits(self):
        mgr = TenantQuotaManager(default_limit=2, window_s=60, per_tenant={"vip": 5})
        for _ in range(2):
            assert mgr.check("free").allowed
        assert not mgr.check("free").allowed
        for _ in range(5):
            assert mgr.check("vip").allowed
        assert not mgr.check("vip").allowed


class TestRedisSharedState:
    """Redis rate limiter/idempotency against an injected fake client."""

    class _FakeRedis:
        def __init__(self):
            self.kv = {}
            self.counts = {}

        def incrby(self, key, amount):
            self.counts[key] = self.counts.get(key, 0) + amount
            return self.counts[key]

        def expire(self, key, seconds):
            return True

        def ttl(self, key):
            return 42

        def get(self, key):
            return self.kv.get(key)

        def set(self, key, value, ex=None):
            self.kv[key] = value

    def test_redis_rate_limiter(self):
        from vincio.storage.redis import RedisRateLimiter

        rl = RedisRateLimiter(client=self._FakeRedis())
        assert rl.check("k", limit=2, window_s=60).allowed
        assert rl.check("k", limit=2, window_s=60).allowed
        assert not rl.check("k", limit=2, window_s=60).allowed

    def test_redis_idempotency_store(self):
        from vincio.storage.redis import RedisIdempotencyStore

        store = RedisIdempotencyStore(client=self._FakeRedis())
        assert store.get("k") is None
        store.put("k", {"x": 1}, ttl_s=60)
        assert store.get("k") == {"x": 1}


# ---------------------------------------------------------------------------
# Content-capture policy
# ---------------------------------------------------------------------------


class TestContentCapturePolicy:
    def test_default_drops_content(self):
        policy = ContentCapturePolicy()
        assert policy.apply("my secret prompt") is None
        scrubbed = policy.scrub_attributes(
            {"output": "hello", "model": "gpt-5.2", "cost_usd": 0.01}
        )
        assert "output" not in scrubbed  # content dropped
        assert scrubbed["model"] == "gpt-5.2" and scrubbed["cost_usd"] == 0.01

    def test_opt_in_truncates_and_redacts(self):
        policy = ContentCapturePolicy(capture=True, max_chars=20, redact_pii=True)
        out = policy.apply("contact me at alice@example.com please " + "x" * 50)
        assert out is not None
        assert "alice@example.com" not in out  # redacted
        assert out.endswith("…[truncated]")

    def test_opt_in_no_redaction_preserves_text(self):
        policy = ContentCapturePolicy(capture=True, redact_pii=False, max_chars=1000)
        assert policy.apply("plain output") == "plain output"


# ---------------------------------------------------------------------------
# Server: health/readiness/metrics + rate limiting
# ---------------------------------------------------------------------------


def _client(**server_overrides):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vincio import ContextApp
    from vincio.core.config import VincioConfig
    from vincio.providers import MockProvider
    from vincio.server import create_app

    config = VincioConfig()
    for key, value in server_overrides.items():
        setattr(config.server, key, value)
    app = ContextApp(name="demo", provider=MockProvider(default_text="hi"))
    api = create_app(config, apps={"demo": app})
    return TestClient(api)


class TestServerEndpoints:
    def test_readiness(self):
        with _client() as client:
            resp = client.get("/v1/health/ready")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ready"

    def test_metrics_endpoint(self):
        with _client() as client:
            client.get("/v1/health")
            resp = client.get("/v1/metrics")
            assert resp.status_code == 200
            assert "vincio_requests_total" in resp.text

    def test_rate_limit_returns_429(self):
        with _client(rate_limit_per_min=2) as client:
            assert client.get("/v1/health").status_code == 200
            assert client.get("/v1/health").status_code == 200
            blocked = client.get("/v1/health")
            assert blocked.status_code == 429
            assert "Retry-After" in blocked.headers
