"""Performance & core hardening: concurrency, caches, streaming,
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


# ---------------------------------------------------------------------------
# Vectorized candidate scoring
# ---------------------------------------------------------------------------


def _scoring_candidates(n: int) -> list:
    from vincio.context.scoring import ContextCandidate

    pool = []
    for i in range(n):
        pool.append(
            ContextCandidate(
                id=f"c{i}",
                type="evidence" if i % 3 else "memory",
                content=(
                    f"Refund window clause {i}: customers on the Pro plan may request "
                    f"a refund within {30 + i} days of purchase for ${i}.00."
                ),
                token_cost=10 + (i % 7),
                authority=0.3 + (i % 5) / 10,
                provenance=0.4 + (i % 4) / 10,
                leakage_risk=0.0 if i % 4 else 0.2,
                metadata={"upstream_relevance": (i % 10) / 10},
            )
        )
    return pool


class TestVectorizedScoring:
    QUERY = "What is the refund window for the Pro plan?"

    def test_batch_equals_per_candidate_loop(self):
        from vincio.context.scoring import ContextScorer

        pool = _scoring_candidates(40)
        loop_scorer = ContextScorer()
        for candidate in pool:
            loop_scorer.score(candidate, query=self.QUERY, selected=[])
        loop_totals = [c.scores.total for c in pool]

        batch_pool = _scoring_candidates(40)
        ContextScorer().score_batch(batch_pool, self.QUERY)
        batch_totals = [c.scores.total for c in batch_pool]

        assert len(batch_totals) == len(loop_totals)
        for a, b in zip(loop_totals, batch_totals, strict=True):
            assert a == pytest.approx(b, abs=1e-12)

    def test_batch_fills_every_component(self):
        from vincio.context.scoring import ContextScorer

        pool = _scoring_candidates(5)
        ContextScorer().score_batch(pool, self.QUERY)
        for candidate in pool:
            assert candidate.scores.novelty == 1.0
            assert candidate.scores.duplication == 0.0
            assert 0.0 <= candidate.scores.relevance <= 1.0

    def test_empty_pool_is_noop(self):
        from vincio.context.scoring import ContextScorer

        ContextScorer().score_batch([], self.QUERY)  # must not raise

    async def test_compile_selection_unchanged(self, sample_evidence):
        # The batched path must not change which evidence the compiler selects.
        compiler = ContextCompiler(ContextCompilerOptions())
        compiled = await compiler.compile(
            objective=Objective("Summarize renewal terms"),
            user_input=UserInput(text="What are the renewal terms?"),
            evidence=sample_evidence,
            budget=Budget(max_input_tokens=4000),
        )
        kept = {e.id for e in compiled.ir.evidence}
        assert "e1" in kept  # high-relevance renewal clause kept
        assert "e3" not in kept  # irrelevant "bananas" evidence excluded

    def test_weighted_totals_fallback_matches_numpy_contract(self):
        from vincio.context.vectorized import weighted_totals

        columns = [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]
        weights = [2.0, -1.0]
        assert weighted_totals(columns, weights) == [1.5, 3.5, 5.5]
        assert weighted_totals([], [1.0]) == []

    def test_semantic_batch_equals_per_candidate_loop(self):
        # Exercises the embedding-cosine relevance path (matrix product under
        # NumPy, per-item cosine otherwise) and the reranker-verdict blend.
        from vincio.context.scoring import ContextScorer

        def vec(seed: int) -> list[float]:
            return [((seed * (k + 3)) % 11) / 10.0 for k in range(8)]

        vectors = {self.QUERY: vec(1)}
        pool = _scoring_candidates(12)
        for i, candidate in enumerate(pool):
            vectors[candidate.content] = vec(i + 2)

        loop = ContextScorer()
        loop.set_embeddings(dict(vectors))
        for candidate in pool:
            loop.score(candidate, query=self.QUERY, selected=[])
        loop_totals = [c.scores.total for c in pool]

        batch_pool = _scoring_candidates(12)
        for i, candidate in enumerate(batch_pool):
            vectors[candidate.content] = vec(i + 2)
        batch = ContextScorer()
        batch.set_embeddings(dict(vectors))
        batch.score_batch(batch_pool, self.QUERY)
        batch_totals = [c.scores.total for c in batch_pool]

        for a, b in zip(loop_totals, batch_totals, strict=True):
            assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# Warm candidate arena
# ---------------------------------------------------------------------------


class TestWarmCandidateArena:
    @staticmethod
    def _kwargs(evidence, query):
        return dict(
            objective=Objective("Summarize renewal terms"),
            user_input=UserInput(text=query),
            evidence=evidence,
            budget=Budget(max_input_tokens=4000),
        )

    async def test_reuse_on_unchanged_candidate_set(self, sample_evidence):
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        await compiler.compile(**self._kwargs(sample_evidence, "What are the renewal terms?"))
        assert compiler.arena_hits == 0  # cold: nothing to reuse yet
        # A new query over the same candidate set reuses the prepared candidates.
        await compiler.compile(**self._kwargs(sample_evidence, "When does it auto-renew?"))
        assert compiler.arena_hits == 1

    async def test_reuse_is_equivalent_to_cold(self, sample_evidence):
        # Warming the arena must not change the compiled output for a new query.
        warm = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        await warm.compile(**self._kwargs(sample_evidence, "What are the renewal terms?"))
        warm_compiled = await warm.compile(**self._kwargs(sample_evidence, "auto-renew window?"))
        assert warm.arena_hits == 1

        cold = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=False))
        cold_compiled = await cold.compile(**self._kwargs(sample_evidence, "auto-renew window?"))

        assert [e.id for e in warm_compiled.ir.evidence] == [
            e.id for e in cold_compiled.ir.evidence
        ]
        assert warm_compiled.token_count == cold_compiled.token_count
        assert warm_compiled.excluded_report == cold_compiled.excluded_report
        assert warm_compiled.packet.spec_hash == cold_compiled.packet.spec_hash

    async def test_changed_candidate_set_misses(self, sample_evidence):
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        await compiler.compile(**self._kwargs(sample_evidence, "q1"))
        await compiler.compile(**self._kwargs(sample_evidence[:1], "q1"))  # fewer items
        assert compiler.arena_hits == 0

    async def test_kept_memory_item_equivalent_across_reuse(self, sample_evidence):
        # A kept memory item carries a query-independent memory_value (its
        # confidence); the arena must preserve it so warm scoring matches cold.
        from vincio.core.types import MemoryItem, MemoryScope, PolicySet

        memory = [
            MemoryItem(
                id="m1",
                content="The customer prefers refunds processed to the original card.",
                scope=MemoryScope.USER,
                owner_id="u1",
                confidence=0.95,
            )
        ]
        kwargs = dict(
            objective=Objective("renewal"),
            user_input=UserInput(text="refund preferences?", user_id="u1"),
            evidence=sample_evidence,
            memory=memory,
            policies=PolicySet(privacy="open"),  # keep the memory item
            budget=Budget(max_input_tokens=4000),
        )
        warm = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        await warm.compile(**kwargs)
        warm_compiled = await warm.compile(**dict(kwargs, user_input=UserInput(text="card refunds?", user_id="u1")))
        assert warm.arena_hits == 1

        cold = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=False))
        cold_compiled = await cold.compile(**dict(kwargs, user_input=UserInput(text="card refunds?", user_id="u1")))

        assert [m.id for m in warm_compiled.ir.memory] == [m.id for m in cold_compiled.ir.memory]
        assert [e.id for e in warm_compiled.ir.evidence] == [e.id for e in cold_compiled.ir.evidence]
        assert "m1" in {m.id for m in warm_compiled.ir.memory}  # the memory survived selection
        assert warm_compiled.token_count == cold_compiled.token_count

    async def test_privacy_exclusions_preserved_across_reuse(self):
        from vincio.core.types import MemoryItem, MemoryScope, PolicySet

        foreign = MemoryItem(
            id="m_foreign",
            content="Acme's internal renewal note.",
            scope=MemoryScope.TENANT,
            owner_id="other_corp",
        )
        evidence = [
            EvidenceItem(id="e1", source_id="D1", text="Renews automatically after 60 days notice.")
        ]
        policies = PolicySet(privacy="tenant_isolated")
        kwargs = dict(
            objective=Objective("renewal"),
            user_input=UserInput(text="renewal?", tenant_id="acme"),
            evidence=evidence,
            memory=[foreign],
            policies=policies,
            budget=Budget(max_input_tokens=4000),
        )
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        first = await compiler.compile(**kwargs)
        kwargs2 = dict(kwargs, user_input=UserInput(text="renew window?", tenant_id="acme"))
        second = await compiler.compile(**kwargs2)
        assert compiler.arena_hits == 1

        def reasons(c):
            return sorted(e["reason"] for e in c.excluded_report)

        assert "privacy_scope_mismatch" in reasons(first)
        assert reasons(first) == reasons(second)
        assert all(m.id != "m_foreign" for m in second.ir.memory)

    async def test_page_change_misses(self):
        # page feeds citation_ref, so evidence differing only by page must not
        # reuse a cached candidate with a stale citation.
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        base = dict(objective=Objective("refunds"), budget=Budget(max_input_tokens=4000))
        query = UserInput(text="When are refunds available?")
        ev1 = [EvidenceItem(id="e1", source_id="D1", text="Refunds are available within 30 days.", page=1, relevance=0.9)]
        ev2 = [EvidenceItem(id="e1", source_id="D1", text="Refunds are available within 30 days.", page=2, relevance=0.9)]
        await compiler.compile(user_input=query, evidence=ev1, **base)
        compiled = await compiler.compile(user_input=query, evidence=ev2, **base)
        assert compiler.arena_hits == 0  # page changed → distinct candidate set
        assert compiled.packet.evidence_items[0]["citation_ref"] == "D1:p2"

    async def test_concurrent_compiles_share_arena_safely(self, sample_evidence):
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        results = await asyncio.gather(
            *(compiler.compile(**self._kwargs(sample_evidence, f"q{i}")) for i in range(16))
        )
        ids = [tuple(e.id for e in r.ir.evidence) for r in results]
        assert all(i == ids[0] for i in ids)  # identical selection under concurrency


# ---------------------------------------------------------------------------
# Streaming-first compilation
# ---------------------------------------------------------------------------


class TestStreamingCompilation:
    @staticmethod
    def _kwargs(evidence):
        from vincio.core.types import Constraint, Instruction

        return dict(
            objective=Objective("Summarize renewal terms"),
            user_input=UserInput(text="What are the renewal terms?"),
            instructions=[Instruction("Answer only from the provided evidence.")],
            constraints=[Constraint("Do not speculate beyond the documents.")],
            evidence=evidence,
            budget=Budget(max_input_tokens=4000),
        )

    async def test_prefix_emitted_before_scoring(self, sample_evidence):
        compiler = ContextCompiler(ContextCompilerOptions())
        state = {"compiled": False}
        original = compiler.compile

        async def flagged(**kw):
            state["compiled"] = True
            return await original(**kw)

        compiler.compile = flagged  # type: ignore[method-assign]
        gen = compiler.compile_streaming(**self._kwargs(sample_evidence))
        first = await gen.__anext__()
        assert first.type == "prefix"
        # The compile (scoring + selection) has not run when the prefix arrives.
        assert state["compiled"] is False
        assert "renewal" in first.text.lower()
        assert first.blocks["instructions"] == ["Answer only from the provided evidence."]

        rest = [event async for event in gen]
        assert state["compiled"] is True
        assert [event.type for event in rest] == ["evidence", "done"]

    async def test_done_equals_direct_compile(self, sample_evidence):
        streamed = None
        async for event in ContextCompiler(ContextCompilerOptions()).compile_streaming(
            **self._kwargs(sample_evidence)
        ):
            if event.type == "evidence":
                assert event.evidence  # selected evidence surfaced mid-stream
            if event.type == "done":
                streamed = event.result
        assert streamed is not None

        direct = await ContextCompiler(ContextCompilerOptions()).compile(
            **self._kwargs(sample_evidence)
        )
        assert [e.id for e in streamed.ir.evidence] == [e.id for e in direct.ir.evidence]
        assert streamed.token_count == direct.token_count
        assert streamed.packet.spec_hash == direct.packet.spec_hash

    async def test_backpressure_consumer_paced(self, sample_evidence):
        # The async generator only advances when pulled — a consumer can stop
        # after the prefix without forcing the rest of the compile.
        compiler = ContextCompiler(ContextCompilerOptions())
        gen = compiler.compile_streaming(**self._kwargs(sample_evidence))
        first = await gen.__anext__()
        assert first.type == "prefix"
        await gen.aclose()  # closing mid-stream must not raise


# ---------------------------------------------------------------------------
# Speculative retrieval prefetch
# ---------------------------------------------------------------------------


class TestSpeculativePrefetch:
    async def test_warm_makes_retrieval_embed_a_cache_hit(self):
        from vincio.retrieval.embeddings import embed_texts
        from vincio.retrieval.prefetch import SpeculativePrefetcher

        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner)
        prefetcher = SpeculativePrefetcher(cached)

        warmed = await prefetcher.warm("what is the refund window?").result()
        assert warmed == 1
        assert prefetcher.warmed == 1
        calls_after_warm = inner.calls

        # Retrieval embeds the query the same way (input_type="query"); it now
        # hits the warm cache instead of calling the embedder again.
        await embed_texts(cached, ["what is the refund window?"], input_type="query")
        assert inner.calls == calls_after_warm

    def test_predict_queries(self):
        from vincio.core.types import TaskType
        from vincio.retrieval.prefetch import SpeculativePrefetcher

        prefetcher = SpeculativePrefetcher(_CountingEmbedder())
        assert prefetcher.predict_queries("  refund? ") == ["refund?"]
        assert prefetcher.predict_queries("") == []
        assert prefetcher.predict_queries("anything", TaskType.CLASSIFICATION) == []
        assert prefetcher.predict_queries("docs?", TaskType.DOCUMENT_QA) == ["docs?"]

    async def test_cancellation_is_clean(self):
        from vincio.retrieval.prefetch import SpeculativePrefetcher

        gate = asyncio.Event()

        class SlowEmbedder:
            dim = 8
            calls = 0

            async def embed(self, texts, **_):
                type(self).calls += 1
                await gate.wait()
                return [[0.0] * self.dim for _ in texts]

        prefetcher = SpeculativePrefetcher(SlowEmbedder())
        handle = prefetcher.warm("slow query")
        handle.cancel()
        assert await handle.result() == 0  # cancelled, no exception
        assert prefetcher.warmed == 0
        gate.set()

    async def test_failed_warm_never_raises(self):
        from vincio.retrieval.prefetch import SpeculativePrefetcher

        class BrokenEmbedder:
            dim = 8

            async def embed(self, texts, **_):
                raise RuntimeError("embedding backend down")

        prefetcher = SpeculativePrefetcher(BrokenEmbedder())
        assert await prefetcher.warm("q").result() == 0  # swallowed

    async def test_base_exception_stays_contained(self):
        # Even a pathological non-Exception failure stays in the warming task —
        # the run is never broken by prefetch.
        from vincio.retrieval.prefetch import SpeculativePrefetcher

        class FatalEmbedder:
            dim = 8

            async def embed(self, texts, **_):
                raise SystemExit("pathological embedder")

        assert await SpeculativePrefetcher(FatalEmbedder()).warm("q").result() == 0

    async def test_app_run_with_prefetch_enabled(self, sample_docs_dir, offline_config, tmp_cwd):
        offline_config.performance.speculative_prefetch = True
        app = ContextApp(
            name="prefetch", provider=MockProvider(), model="mock-1", config=offline_config
        )
        app.add_source("docs", path=str(sample_docs_dir), retrieval="hybrid")
        assert app._prefetcher is not None
        result = await app.arun("What is the refund window for the Pro plan?")
        assert result.status == RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Per-app memory-footprint budget
# ---------------------------------------------------------------------------


def _sized_evidence(n: int, *, chars: int = 200) -> list:
    out = []
    for i in range(n):
        body = f"renewal clause {i} " + "term notice renewal window " * (chars // 26)
        out.append(
            EvidenceItem(
                id=f"ev{i}",
                source_id=f"D{i}",
                text=body,
                authority=0.5,
                relevance=0.9 - i * 0.05,
            )
        )
    return out


class TestMemoryFootprintBudget:
    def test_estimator_is_monotonic_and_slim_helps(self):
        from vincio.context.footprint import estimate_resident_bytes

        small = estimate_resident_bytes(["abc"], [], slim=False)
        large = estimate_resident_bytes(["abc", "defgh"], [], slim=False)
        assert large > small  # more/larger evidence → larger estimate
        slim = estimate_resident_bytes(["a" * 1000], [], slim=True)
        full = estimate_resident_bytes(["a" * 1000], [], slim=False)
        assert slim < full  # slim references text by hash instead of inlining it

    async def test_resident_bytes_reported_by_default(self, sample_evidence):
        compiled = await ContextCompiler(ContextCompilerOptions()).compile(
            objective=Objective("renewal"),
            user_input=UserInput(text="renewal terms?"),
            evidence=sample_evidence,
            budget=Budget(max_input_tokens=4000),
        )
        assert compiled.resident_bytes > 0  # always surfaced

    async def test_budget_slims_and_evicts(self):
        evidence = _sized_evidence(4)
        unbounded = await ContextCompiler(ContextCompilerOptions()).compile(
            objective=Objective("renewal"),
            user_input=UserInput(text="renewal term notice window?"),
            evidence=evidence,
            budget=Budget(max_input_tokens=8000),
        )
        assert len(unbounded.ir.evidence) == 4  # all fit without a ceiling
        ceiling = 1200
        bounded = await ContextCompiler(
            ContextCompilerOptions(max_resident_bytes=ceiling)
        ).compile(
            objective=Objective("renewal"),
            user_input=UserInput(text="renewal term notice window?"),
            evidence=evidence,
            budget=Budget(max_input_tokens=8000),
        )
        assert bounded.packet.slim is True  # slimmed first
        assert bounded.resident_bytes <= ceiling
        assert len(bounded.ir.evidence) < 4  # lowest-utility evidence evicted
        evicted = [e for e in bounded.excluded_report if e["reason"] == "memory_budget_exceeded"]
        assert evicted
        # The highest-utility evidence is the one that survives.
        assert "ev0" in {e.id for e in bounded.ir.evidence}

    async def test_app_surfaces_memory_in_result_and_cost(
        self, sample_docs_dir, offline_config, tmp_cwd
    ):
        offline_config.performance.memory_budget_mb = 0.001  # 1000 bytes
        app = ContextApp(
            name="footprint", provider=MockProvider(), model="mock-1", config=offline_config
        )
        app.add_source("docs", path=str(sample_docs_dir), retrieval="hybrid")
        result = await app.arun("What is the refund window for the Pro plan?")
        assert result.status == RunStatus.SUCCEEDED
        assert result.memory_bytes > 0
        assert result.memory_bytes <= 1000
        assert app.cost_tracker.summary()["peak_resident_bytes"] >= result.memory_bytes
