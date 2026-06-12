"""0.2 performance & core hardening: concurrency, caches, streaming,
throughput primitives, zero-copy packets."""

from __future__ import annotations

import asyncio
import json

import pytest

from vincio import ContextApp
from vincio.caching import ChunkCache, ContextCompileCache, InMemoryCache, PromptCompileCache
from vincio.context import ContextCompiler, ContextCompilerOptions
from vincio.core.concurrency import gather_bounded, map_bounded
from vincio.core.types import (
    Budget,
    Document,
    EvidenceItem,
    Message,
    ModelRequest,
    ModelResponse,
    Objective,
    RunStatus,
    ToolCallRequest,
    UserInput,
)
from vincio.prompts import CompilerOptions, PromptCompiler, PromptSpec
from vincio.providers import CoalescingProvider, MockProvider
from vincio.providers.transport import build_pooled_client
from vincio.retrieval.chunking import chunk_document
from vincio.retrieval.embeddings import (
    BatchingEmbedder,
    CachedEmbedder,
    ProviderEmbedder,
)

# ---------------------------------------------------------------------------
# Bounded concurrency primitives
# ---------------------------------------------------------------------------


class TestGatherBounded:
    async def test_preserves_order(self):
        async def job(i: int) -> int:
            await asyncio.sleep(0.001 * (5 - i))
            return i

        assert await gather_bounded((job(i) for i in range(5)), limit=2) == [0, 1, 2, 3, 4]

    async def test_respects_limit(self):
        active = 0
        peak = 0

        async def job() -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1

        await gather_bounded((job() for _ in range(10)), limit=3)
        assert peak <= 3

    async def test_first_failure_cancels_siblings(self):
        cancelled = asyncio.Event()

        async def failing() -> None:
            raise ValueError("boom")

        async def slow() -> None:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(ValueError):
            await gather_bounded([failing(), slow()], limit=2)
        assert cancelled.is_set()

    async def test_map_bounded(self):
        async def double(x: int) -> int:
            return x * 2

        assert await map_bounded(double, [1, 2, 3], limit=2) == [2, 4, 6]

    async def test_return_exceptions(self):
        async def ok() -> str:
            return "ok"

        async def bad() -> str:
            raise RuntimeError("bad")

        results = await gather_bounded([ok(), bad()], limit=2, return_exceptions=True)
        assert results[0] == "ok"
        assert isinstance(results[1], RuntimeError)


# ---------------------------------------------------------------------------
# Embedding caches & batching
# ---------------------------------------------------------------------------


class _CountingEmbedder:
    dim = 8

    def __init__(self):
        self.calls = 0
        self.texts_seen: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_seen.append(list(texts))
        return [[float(len(t))] * self.dim for t in texts]


class TestEmbedders:
    async def test_cached_embedder_content_addressed(self):
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner)
        first = await cached.embed(["alpha", "beta"])
        second = await cached.embed(["beta", "alpha"])  # same content, new order
        assert inner.calls == 1
        assert second == [first[1], first[0]]
        assert cached.hits == 2

    async def test_cached_embedder_dedupes_within_call(self):
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner)
        await cached.embed(["same", "same", "same"])
        assert inner.texts_seen == [["same"]]

    async def test_cached_embedder_persistent_backend(self):
        backend = InMemoryCache(default_ttl_s=None)
        inner_a = _CountingEmbedder()
        await CachedEmbedder(inner_a, backend=backend).embed(["persisted"])
        inner_b = _CountingEmbedder()
        fresh = CachedEmbedder(inner_b, backend=backend)  # new in-memory layer
        await fresh.embed(["persisted"])
        assert inner_b.calls == 0  # served from the shared backend

    async def test_provider_embedder_splits_batches(self):
        provider = MockProvider()
        calls: list[int] = []
        original = provider.embed

        async def counting_embed(texts, model=None):
            calls.append(len(texts))
            return await original(texts, model)

        provider.embed = counting_embed
        embedder = ProviderEmbedder(provider, batch_size=10, concurrency=2)
        vectors = await embedder.embed([f"text {i}" for i in range(25)])
        assert len(vectors) == 25
        assert calls == [10, 10, 5]

    async def test_batching_embedder_coalesces(self):
        inner = _CountingEmbedder()
        batcher = BatchingEmbedder(inner, max_batch=64, window_ms=20.0)
        results = await asyncio.gather(
            batcher.embed(["a"]), batcher.embed(["b"]), batcher.embed(["a", "c"])
        )
        await batcher.aclose()
        assert inner.calls == 1  # one provider call for three concurrent embeds
        assert results[0][0] == results[2][0]  # duplicate text, one vector
        assert {len(v) for r in results for v in r} == {8}

    async def test_batching_embedder_full_batch_flushes_immediately(self):
        inner = _CountingEmbedder()
        batcher = BatchingEmbedder(inner, max_batch=2, window_ms=10_000.0)
        vectors = await batcher.embed(["x", "y"])  # fills the batch: no timer wait
        await batcher.aclose()
        assert len(vectors) == 2
        assert inner.calls == 1


# ---------------------------------------------------------------------------
# Request coalescing & pooled transport
# ---------------------------------------------------------------------------


class TestCoalescing:
    async def test_identical_inflight_requests_share_one_call(self):
        gate = asyncio.Event()

        class SlowMock(MockProvider):
            async def generate(self, request):
                await gate.wait()
                return await super().generate(request)

        inner = SlowMock()
        provider = CoalescingProvider(inner)
        request = ModelRequest(model="m", messages=[Message(role="user", content="hi")])

        async def fire():
            return await provider.generate(request)

        tasks = [asyncio.create_task(fire()) for _ in range(5)]
        await asyncio.sleep(0.01)
        gate.set()
        responses = await asyncio.gather(*tasks)
        assert inner.call_count == 1
        assert provider.coalesced_count == 4
        assert len({id(r) for r in responses}) == 5  # independent copies
        assert len({r.text for r in responses}) == 1

    async def test_different_requests_not_coalesced(self):
        inner = MockProvider()
        provider = CoalescingProvider(inner)
        a = ModelRequest(model="m", messages=[Message(role="user", content="a")])
        b = ModelRequest(model="m", messages=[Message(role="user", content="b")])
        await asyncio.gather(provider.generate(a), provider.generate(b))
        assert inner.call_count == 2

    async def test_pooled_client_limits(self):
        client = build_pooled_client(max_connections=7, max_keepalive_connections=3)
        try:
            assert client is not None
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Content-addressed compilation caches
# ---------------------------------------------------------------------------


class TestChunkCache:
    def test_rechunk_is_cache_hit_with_correct_provenance(self):
        cache = ChunkCache()
        doc_a = Document(text="# Title\n\n" + "Sentence about policies. " * 80, title="a")
        first = chunk_document(doc_a, strategy="recursive", size=80, cache=cache)
        doc_b = Document(text=doc_a.text, title="b", tenant_id="t2")
        second = chunk_document(doc_b, strategy="recursive", size=80, cache=cache)
        assert [c.text for c in first] == [c.text for c in second]
        assert all(c.document_id == doc_b.id for c in second)
        assert all(c.tenant_id == "t2" for c in second)
        assert second[0].citation_ref.startswith(doc_b.id)

    def test_different_options_miss(self):
        cache = ChunkCache()
        doc = Document(text="words " * 300, title="x")
        chunk_document(doc, strategy="recursive", size=80, cache=cache)
        backend_entries = cache.backend.stats()["entries"]
        chunk_document(doc, strategy="recursive", size=120, cache=cache)
        assert cache.backend.stats()["entries"] == backend_entries + 1


class TestPromptCompileCache:
    def test_unchanged_inputs_hit(self):
        compiler = PromptCompiler(CompilerOptions(), cache=PromptCompileCache())
        spec = PromptSpec(name="t", role="assistant", objective="answer", rules=["Be brief"])
        first = compiler.compile(spec, user_task="What is the refund window?")
        second = compiler.compile(spec, user_task="What is the refund window?")
        assert compiler.cache_hits == 1
        assert second.rendered_hash == first.rendered_hash
        assert second.messages[0].cache_hint is True

    def test_changed_task_misses(self):
        compiler = PromptCompiler(CompilerOptions(), cache=PromptCompileCache())
        spec = PromptSpec(name="t", role="assistant", objective="answer")
        compiler.compile(spec, user_task="q1")
        compiler.compile(spec, user_task="q2")
        assert compiler.cache_hits == 0


class TestContextCompileCache:
    @pytest.fixture()
    def compile_kwargs(self, sample_evidence):
        return dict(
            objective=Objective("Summarize renewal terms"),
            user_input=UserInput(text="What are the renewal terms?"),
            evidence=sample_evidence,
            budget=Budget(max_input_tokens=4000),
        )

    async def test_unchanged_inputs_hit(self, compile_kwargs):
        compiler = ContextCompiler(ContextCompilerOptions(), cache=ContextCompileCache())
        first = await compiler.compile(**compile_kwargs)
        second = await compiler.compile(**compile_kwargs)
        assert compiler.cache_hits == 1
        assert second.from_cache is True
        assert second.packet.spec_hash == first.packet.spec_hash
        assert second.packet.id != first.packet.id  # fresh per-run identity
        assert [e.id for e in second.ir.evidence] == [e.id for e in first.ir.evidence]

    async def test_changed_evidence_misses(self, compile_kwargs):
        compiler = ContextCompiler(ContextCompilerOptions(), cache=ContextCompileCache())
        await compiler.compile(**compile_kwargs)
        changed = dict(compile_kwargs)
        changed["evidence"] = compile_kwargs["evidence"][:1]
        await compiler.compile(**changed)
        assert compiler.cache_hits == 0

    async def test_recompile_partial_edit(self, compile_kwargs):
        compiler = ContextCompiler(ContextCompilerOptions())
        first = await compiler.compile(**compile_kwargs)
        extra = EvidenceItem(
            id="e9",
            source_id="D9",
            text="Renewal requires written notice sent 60 days in advance.",
            authority=0.9,
            relevance=0.9,
        )
        edited = await compiler.recompile(first, add_evidence=[extra], remove_evidence_ids=["e2"])
        kept_ids = {e.id for e in edited.ir.evidence}
        assert "e9" in kept_ids
        assert "e2" not in kept_ids
        assert edited.ir.objective.text == first.ir.objective.text


# ---------------------------------------------------------------------------
# Zero-copy Context Packet
# ---------------------------------------------------------------------------


class TestSlimPacket:
    async def _compile(self, sample_evidence, *, slim: bool):
        compiler = ContextCompiler(ContextCompilerOptions(slim_packets=slim))
        return await compiler.compile(
            objective=Objective("Summarize renewal terms"),
            user_input=UserInput(text="What are the renewal terms?"),
            evidence=sample_evidence,
            budget=Budget(max_input_tokens=4000),
        )

    async def test_slim_packet_references_text_by_hash(self, sample_evidence):
        compiled = await self._compile(sample_evidence, slim=True)
        packet = compiled.packet
        assert packet.slim is True
        assert all("text" not in entry for entry in packet.evidence_items)
        assert all(entry.get("text_hash") for entry in packet.evidence_items)

    async def test_lazy_materialization(self, sample_evidence):
        compiled = await self._compile(sample_evidence, slim=True)
        packet = compiled.packet
        item = compiled.ir.evidence[0]
        assert packet.evidence_text(item.id) == item.text
        packet.materialize()
        assert packet.slim is False
        assert packet.evidence_items[0]["text"] == compiled.ir.evidence[0].text

    async def test_slim_packet_is_smaller(self, sample_evidence):
        slim = (await self._compile(sample_evidence, slim=True)).packet
        full = (await self._compile(sample_evidence, slim=False)).packet
        assert slim.approx_size_bytes() < full.approx_size_bytes()

    async def test_iter_json_equals_full_dump(self, sample_evidence):
        packet = (await self._compile(sample_evidence, slim=False)).packet
        streamed = json.loads("".join(packet.iter_json()))
        assert streamed == packet.model_dump(mode="json")

    async def test_iter_json_streams_in_chunks(self, sample_evidence):
        packet = (await self._compile(sample_evidence, slim=False)).packet
        chunks = list(packet.iter_json())
        assert len(chunks) > len(packet.evidence_items)


# ---------------------------------------------------------------------------
# End-to-end streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_astream_text_deltas_reconstruct_output(self, rag_app):
        events = [e async for e in rag_app.astream("What is the refund window for the Pro plan?")]
        types = [e.type for e in events]
        assert types[-1] == "done"
        assert "stage" in types
        deltas = [e.text for e in events if e.type == "text_delta"]
        assert len(deltas) > 1  # real chunked streaming, not one blob
        result = events[-1].result
        assert result is not None and result.status == RunStatus.SUCCEEDED
        assert "".join(deltas) == result.raw_text

    async def test_astream_partial_json_for_structured_output(self, offline_config, tmp_cwd):
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "confidence", "items"],
        }
        app = ContextApp(
            name="stream_structured",
            provider=MockProvider(),
            model="mock-1",
            output_schema=schema,
            config=offline_config,
        )
        events = [e async for e in app.astream("Extract the items")]
        partials = [e for e in events if e.type == "partial_output"]
        assert partials, "expected incremental partial_output events"
        assert all(p.partial_output is not None for p in partials)
        done = events[-1]
        assert done.result is not None and done.result.status == RunStatus.SUCCEEDED

    async def test_astream_records_ttft_span(self, rag_app):
        [e async for e in rag_app.astream("What is the refund window for the Pro plan?")]
        # The model span carries first-token latency for the streamed call.
        trace = rag_app.tracer.exporter.traces[-1]
        model_spans = [s for s in trace.spans if s.type == "model_call"]
        assert model_spans and "ttft_ms" in model_spans[0].attributes

    def test_sync_stream_wrapper(self, rag_app):
        events = list(rag_app.stream("What is the refund window for the Pro plan?"))
        assert events[-1].type == "done"


# ---------------------------------------------------------------------------
# Concurrent tool fan-out, cancellation, deadlines
# ---------------------------------------------------------------------------


class TestRunHardening:
    def _tooled_app(self, offline_config, *, provider) -> ContextApp:
        app = ContextApp(name="tools", provider=provider, model="mock-1", config=offline_config)

        def lookup_a(key: str) -> dict:
            """Lookup A."""
            return {"a": key}

        def lookup_b(key: str) -> dict:
            """Lookup B."""
            return {"b": key}

        app.add_tool(lookup_a)
        app.add_tool(lookup_b)
        return app

    async def test_tool_round_fans_out_concurrently(self, offline_config, tmp_cwd):
        first = ModelResponse(
            text="",
            tool_calls=[
                ToolCallRequest(name="lookup_a", arguments={"key": "k1"}),
                ToolCallRequest(name="lookup_b", arguments={"key": "k2"}),
            ],
            finish_reason="tool_calls",
        )
        provider = MockProvider(script=[first, "final answer"])
        app = self._tooled_app(offline_config, provider=provider)
        result = await app.arun("look both up")
        assert result.status == RunStatus.SUCCEEDED
        assert [t.tool_name for t in result.tool_results] == ["lookup_a", "lookup_b"]
        assert all(t.status == "ok" for t in result.tool_results)

    async def test_cancellation_propagates(self, offline_config, tmp_cwd):
        class SlowProvider(MockProvider):
            async def generate(self, request):
                await asyncio.sleep(30)
                return await super().generate(request)

        app = ContextApp(name="slow", provider=SlowProvider(), model="mock-1", config=offline_config)
        task = asyncio.create_task(app.arun("hello"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_latency_budget_enforced(self, offline_config, tmp_cwd):
        class SlowProvider(MockProvider):
            async def generate(self, request):
                await asyncio.sleep(30)
                return await super().generate(request)

        app = ContextApp(
            name="deadline",
            provider=SlowProvider(),
            model="mock-1",
            config=offline_config,
            budget=Budget(max_latency_ms=100),
        )
        result = await app.arun("hello")
        assert result.status == RunStatus.FAILED
        assert "max_latency_ms" in (result.error or "")

    async def test_provider_instances_reused_across_runs(self, rag_app):
        first = rag_app.resolve_provider()
        second = rag_app.resolve_provider()
        assert first is second
