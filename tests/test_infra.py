"""Providers, observability, caching, storage, config, events tests."""

import json

import pytest

from vincio.caching import (
    InMemoryCache,
    InvalidationManager,
    ResponseCache,
    SemanticCache,
    SQLiteCache,
)
from vincio.core.config import load_config
from vincio.core.events import EventBus
from vincio.core.tokens import HeuristicTokenCounter
from vincio.core.types import Message, ModelRequest, ModelResponse, TokenUsage
from vincio.core.utils import stable_hash, to_jsonable
from vincio.observability import (
    CostTracker,
    InMemoryExporter,
    JSONLExporter,
    Tracer,
    trace_diff,
)
from vincio.providers import (
    AnthropicProvider,
    FailoverChain,
    GoogleProvider,
    MockProvider,
    OpenAIProvider,
    RetryingProvider,
    build_provider,
    instance_from_schema,
)
from vincio.retrieval import LocalHashEmbedder
from vincio.storage import FileBlobStore, InMemoryMetadataStore, SQLiteMetadataStore


class TestCore:
    def test_token_counter(self):
        counter = HeuristicTokenCounter()
        assert counter.count("") == 0
        assert 8 <= counter.count("The quick brown fox jumps over the lazy dog today.") <= 16

    def test_stable_hash_deterministic(self):
        assert stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})

    def test_to_jsonable(self):
        from vincio.core.types import Objective

        payload = to_jsonable({"o": Objective("x"), "s": {1, 2}})
        assert payload["o"]["text"] == "x"
        assert payload["s"] == ["1", "2"] or payload["s"] == [1, 2]

    def test_config_env_override(self, monkeypatch, tmp_cwd):
        monkeypatch.setenv("VINCIO_RETRIEVAL__TOP_K", "13")
        monkeypatch.setenv("VINCIO_SECURITY__TENANT_ISOLATION", "false")
        config = load_config()
        assert config.retrieval.top_k == 13
        assert config.security.tenant_isolation is False

    def test_config_file_discovery(self, tmp_cwd):
        (tmp_cwd / "vincio.yaml").write_text("project: from_file\nprovider:\n  model: m-1\n")
        config = load_config()
        assert config.project == "from_file"
        assert config.provider.model == "m-1"

    def test_event_bus_wildcards(self):
        bus = EventBus()
        seen = []
        bus.subscribe("tool.*", lambda e: seen.append(e.name))
        bus.emit("tool.called", {})
        bus.emit("other.event", {})
        assert seen == ["tool.called"]


class TestProviders:
    @pytest.mark.asyncio
    async def test_mock_schema_valid_output(self):
        schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["a", "b"]},
                "confidence": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["label", "confidence", "tags"],
        }
        provider = MockProvider()
        response = await provider.generate(
            ModelRequest(model="m", messages=[Message(role="user", content="q")], output_schema=schema)
        )
        from vincio.tools.runtime import validate_against_schema

        assert validate_against_schema(response.structured, schema) == []

    def test_instance_from_schema_refs(self):
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            items: list[Inner]
            count: int

        schema = Outer.model_json_schema()
        instance = instance_from_schema(schema)
        assert isinstance(instance["items"][0]["name"], str)

    @pytest.mark.asyncio
    async def test_retry_on_retryable(self):
        from vincio.core.errors import ProviderUnavailableError

        class Flaky(MockProvider):
            name = "flaky"

            def __init__(self):
                super().__init__()
                self.attempts = 0

            async def generate(self, request):
                self.attempts += 1
                if self.attempts < 3:
                    raise ProviderUnavailableError("down", provider="flaky")
                return await super().generate(request)

        flaky = Flaky()
        provider = RetryingProvider(flaky, max_retries=3, base_delay_s=0.001)
        response = await provider.generate(ModelRequest(model="m", messages=[Message(role="user", content="x")]))
        assert flaky.attempts == 3
        assert response.text

    @pytest.mark.asyncio
    async def test_failover(self):
        from vincio.core.errors import ProviderUnavailableError

        class Down(MockProvider):
            name = "down"

            async def generate(self, request):
                raise ProviderUnavailableError("down", provider="down")

        chain = FailoverChain([(Down(), None), (MockProvider(default_text="fallback"), None)])
        response = await chain.generate(ModelRequest(model="m", messages=[Message(role="user", content="x")]))
        assert response.text == "fallback"

    def test_payload_rendering(self):
        request = ModelRequest(
            model="m",
            messages=[
                Message(role="system", content="rules", cache_hint=True),
                Message(role="user", content="hi"),
            ],
            output_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        openai_payload = OpenAIProvider(api_key="k")._payload(request, stream=True)
        assert openai_payload["response_format"]["type"] == "json_schema"
        assert openai_payload["stream"] is True

        system_blocks, _messages = AnthropicProvider(api_key="k")._render(request)
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
        anthropic_payload = AnthropicProvider(api_key="k")._payload(request)
        assert any(t["name"] == "emit_structured_output" for t in anthropic_payload["tools"])

        google_payload = GoogleProvider(api_key="k")._payload(request)
        assert google_payload["generationConfig"]["responseMimeType"] == "application/json"
        assert "additionalProperties" not in json.dumps(google_payload)

    def test_build_provider_registry(self):
        provider = build_provider("mock", with_retries=False)
        assert isinstance(provider, MockProvider)
        wrapped = build_provider("openai", api_key="k")
        assert isinstance(wrapped, RetryingProvider)


class TestObservability:
    def test_span_nesting_and_export(self):
        exporter = InMemoryExporter()
        tracer = Tracer("app", exporter)
        with tracer.trace(run_id="r1"):
            with tracer.span("outer", type="retrieval"):
                with tracer.span("inner", type="retrieval"):
                    tracer.event("scored", n=3)
        trace = exporter.traces[0]
        inner = next(s for s in trace.spans if s.name == "inner")
        outer = next(s for s in trace.spans if s.name == "outer")
        assert inner.parent_id == outer.id
        assert inner.events[0].name == "scored"

    def test_error_recorded(self):
        exporter = InMemoryExporter()
        tracer = Tracer("app", exporter)
        with pytest.raises(ValueError):
            with tracer.trace():
                with tracer.span("bad"):
                    raise ValueError("boom")
        trace = exporter.traces[0]
        assert trace.status == "error"
        assert trace.spans[0].status == "error"

    def test_jsonl_roundtrip(self, tmp_path):
        exporter = JSONLExporter(tmp_path)
        tracer = Tracer("app", exporter)
        with tracer.trace(run_id="r1") as trace:
            with tracer.span("s"):
                pass
        loaded = exporter.load(trace.id)
        assert loaded is not None and loaded.spans[0].name == "s"

    def test_cost_tracking(self):
        tracker = CostTracker()
        cost = tracker.record_model_call(
            "claude-sonnet-4-6",
            TokenUsage(input_tokens=10_000, output_tokens=2_000, cached_input_tokens=4_000),
        )
        assert cost == pytest.approx(0.0492)
        assert tracker.summary()["total_usd"] == pytest.approx(0.0492)

    def test_trace_diff(self):
        exporter = InMemoryExporter()
        tracer = Tracer("app", exporter)
        for spans in (["a"], ["a", "b"]):
            with tracer.trace():
                for name in spans:
                    with tracer.span(name):
                        pass
        diff = trace_diff(exporter.traces[0], exporter.traces[1])
        assert diff["spans_only_in_b"] == ["custom:b"]


class TestCaching:
    def test_lru_ttl_tags(self):
        cache = InMemoryCache(max_entries=2, default_ttl_s=None)
        cache.set("a", 1, tags=["t"])
        cache.set("b", 2)
        cache.set("c", 3)  # evicts a (LRU)
        assert cache.get("a") is None and cache.get("c") == 3

    def test_sqlite_cache(self, tmp_path):
        cache = SQLiteCache(tmp_path / "c.db")
        cache.set("k", {"v": 1}, tags=["doc:d1"])
        assert cache.get("k") == {"v": 1}
        assert cache.invalidate_tag("doc:d1") == 1
        assert cache.get("k") is None

    def test_response_cache_and_invalidation(self):
        backend = InMemoryCache()
        response_cache = ResponseCache(backend)
        request = ModelRequest(model="m", messages=[Message(role="user", content="hi")])
        response_cache.set(request, ModelResponse(text="cached"), prompt_version="v1")
        assert response_cache.get(request).text == "cached"
        manager = InvalidationManager([backend])
        assert manager.prompt_version_changed("v1") >= 1
        assert response_cache.get(request) is None

    @pytest.mark.asyncio
    async def test_semantic_cache_scoping(self):
        cache = SemanticCache(LocalHashEmbedder(), threshold=0.8)
        await cache.set("what is the refund policy", {"answer": "30 days"}, policy_scope="t:acme", schema_ref="R")
        assert await cache.get("what is the refund policy?", policy_scope="t:acme", schema_ref="R")
        assert await cache.get("what is the refund policy?", policy_scope="t:other", schema_ref="R") is None
        assert await cache.get("how do whales sleep", policy_scope="t:acme", schema_ref="R") is None


class TestStorage:
    def test_sqlite_core_tables(self, tmp_path):
        store = SQLiteMetadataStore(tmp_path / "s.db")
        store.save("runs", {"id": "r1", "app_id": "a", "tenant_id": "acme", "status": "ok", "extra": {"x": 1}})
        store.save("custom", {"id": "k1", "foo": "bar"})
        assert store.get("runs", "r1")["extra"] == {"x": 1}
        assert store.query("runs", where={"tenant_id": "acme"})
        assert not store.query("runs", where={"tenant_id": "zzz"})
        assert store.get("custom", "k1")["foo"] == "bar"
        assert store.delete("custom", "k1")
        store.close()

    def test_in_memory_store(self):
        store = InMemoryMetadataStore()
        store.save("runs", {"id": "1", "status": "ok"})
        assert store.count("runs") == 1

    def test_blob_store(self, tmp_path):
        blobs = FileBlobStore(tmp_path)
        blobs.put("a/b.txt", b"data")
        assert blobs.get("a/b.txt") == b"data"
        assert blobs.delete("a/b.txt")
