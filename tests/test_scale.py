"""Cost, reliability & scale: batch, circuit breaking, key pooling,
cascades, cost attribution, budgets, prompt-cache strategy, incremental and
sharded indexing. All offline via the mock provider and httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from vincio import ContextApp
from vincio.core.errors import CircuitOpenError, ProviderError, ProviderUnavailableError
from vincio.core.events import EventBus
from vincio.core.types import (
    Chunk,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from vincio.observability.costs import ModelPrice
from vincio.observability.finops import BudgetManager, CostBudget, CostEvent, CostLedger
from vincio.optimize.routing import ModelCascade, response_confidence
from vincio.providers import (
    AnthropicBatchBackend,
    AnthropicProvider,
    BatchRequest,
    BatchRunner,
    CircuitBreaker,
    CircuitState,
    HealthAwareFailover,
    InProcessBatchBackend,
    KeyPool,
    MockProvider,
    OpenAIBatchBackend,
    OpenAIProvider,
    PromptCacheStrategy,
    RateLimiter,
)
from vincio.providers.batch import BatchStatus
from vincio.retrieval import BM25Index, LiveIndex, ShardedIndex, VectorIndex
from vincio.retrieval.embeddings import LocalHashEmbedder


def _req(model: str = "mock") -> ModelRequest:
    return ModelRequest(model=model, messages=[Message(role="user", content="hi")])


class _Flaky(MockProvider):
    """A provider that errors for the first ``fail`` calls, then succeeds."""

    def __init__(self, *, fail: int = 0, error: type[ProviderError] = ProviderUnavailableError, **kw):
        super().__init__(**kw)
        self._fail = fail
        self._error = error
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        if self.calls <= self._fail:
            raise self._error("boom", provider="flaky")
        return await super().generate(request)


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


class TestBatch:
    @pytest.mark.asyncio
    async def test_in_process_reconciles_by_custom_id(self):
        runner = BatchRunner(MockProvider(default_text="ok"), poll_interval_s=0.0)
        requests = [BatchRequest(custom_id=f"c{i}", request=_req()) for i in range(5)]
        result = await runner.run(requests)
        assert result.job.status is BatchStatus.COMPLETED
        assert [r.custom_id for r in result.results] == ["c0", "c1", "c2", "c3", "c4"]
        assert all(r.ok for r in result.results)

    @pytest.mark.asyncio
    async def test_partial_failure_surfaced(self):
        backend = InProcessBatchBackend(
            MockProvider(default_text="ok"),
            fail_if=lambda item: "bad input" if item.custom_id == "c2" else None,
        )
        runner = BatchRunner(backend, poll_interval_s=0.0)
        requests = [BatchRequest(custom_id=f"c{i}", request=_req()) for i in range(4)]
        result = await runner.run(requests)
        assert [r.custom_id for r in result.failed] == ["c2"]
        assert len(result.succeeded) == 3
        assert result.by_id()["c2"].error == "bad input"

    @pytest.mark.asyncio
    async def test_missing_result_reconciled_as_failure(self):
        # A backend that drops one request must not silently lose it.
        class Dropping(InProcessBatchBackend):
            async def results(self, job, requests):
                out = await super().results(job, requests)
                return [r for r in out if r.custom_id != "c1"]

        runner = BatchRunner(Dropping(MockProvider(default_text="ok")), poll_interval_s=0.0)
        result = await runner.run([BatchRequest(custom_id=f"c{i}", request=_req()) for i in range(3)])
        assert result.by_id()["c1"].error == "missing from batch output"
        assert len(result.failed) == 1

    @pytest.mark.asyncio
    async def test_batch_cost_is_discounted(self):
        from vincio.observability.costs import PriceTable

        table = PriceTable()
        table.set("gpt-5.2", ModelPrice(input_per_mtok=1_000_000, output_per_mtok=1_000_000))

        def responder(req):
            return ModelResponse(text="ok", usage=TokenUsage(input_tokens=1, output_tokens=1))

        runner = BatchRunner(
            MockProvider(responder=responder), price_table=table, discount=0.5, poll_interval_s=0.0
        )
        result = await runner.run([BatchRequest(custom_id="c0", request=_req("gpt-5.2"))])
        # Full price would be (1+1) tokens * $1/token = $2; batch halves it.
        assert result.cost_usd == pytest.approx(1.0)
        assert result.results[0].response.cost_usd == pytest.approx(1.0)

    def test_app_batch_returns_run_results(self, offline_config, tmp_cwd):
        app = ContextApp(name="b", provider=MockProvider(default_text="batched"), config=offline_config)
        results = app.batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(r.status.value == "succeeded" for r in results)
        assert all(r.raw_text == "batched" for r in results)

    def test_app_batch_exempt_from_cost_cap(self, offline_config, tmp_cwd):
        app = ContextApp(name="b", provider=MockProvider(default_text="ok"), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e9))
        app.set_cost_budget(scope="global", limit_usd=1e-9, period="total", on_breach="cap")
        # Interactive run is capped; batch is the queue target, so it runs.
        results = app.batch(["x", "y"])
        assert all(r.status.value == "succeeded" for r in results)


class TestBatchWire:
    @pytest.mark.asyncio
    async def test_openai_batch_backend_roundtrip(self):
        state = {"polls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/files"):
                return httpx.Response(200, json={"id": "file_in"})
            if path.endswith("/batches") and request.method == "POST":
                return httpx.Response(
                    200, json={"id": "batch_1", "status": "in_progress", "request_counts": {"total": 2}}
                )
            if path.endswith("/batches/batch_1"):
                state["polls"] += 1
                return httpx.Response(
                    200,
                    json={
                        "id": "batch_1",
                        "status": "completed",
                        "output_file_id": "file_out",
                        "request_counts": {"total": 2, "completed": 2, "failed": 0},
                    },
                )
            if "file_out/content" in path:
                lines = [
                    {
                        "custom_id": cid,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "model": "gpt-5.2",
                                "choices": [
                                    {"message": {"content": f"answer {cid}"}, "finish_reason": "stop"}
                                ],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                            },
                        },
                        "error": None,
                    }
                    for cid in ("c0", "c1")
                ]
                return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))
            return httpx.Response(404, json={"error": "not found"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="x", client=client)
        runner = BatchRunner(OpenAIBatchBackend(provider), poll_interval_s=0.0)
        result = await runner.run(
            [BatchRequest(custom_id=f"c{i}", request=_req("gpt-5.2")) for i in range(2)]
        )
        assert state["polls"] >= 1  # it actually polled the job
        assert {r.custom_id for r in result.succeeded} == {"c0", "c1"}
        assert result.by_id()["c0"].response.text == "answer c0"

    @pytest.mark.asyncio
    async def test_openai_batch_error_file_and_failed_status(self):
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/files"):
                return httpx.Response(200, json={"id": "file_in"})
            if path.endswith("/batches") and request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "id": "b2",
                        "status": "completed",
                        "output_file_id": "ok_file",
                        "error_file_id": "err_file",
                        "request_counts": {"total": 2, "completed": 1, "failed": 1},
                    },
                )
            if "ok_file/content" in path:
                line = {
                    "custom_id": "c0",
                    "response": {"status_code": 200, "body": {
                        "model": "gpt-5.2",
                        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                    }},
                    "error": None,
                }
                return httpx.Response(200, text=json.dumps(line))
            if "err_file/content" in path:
                line = {"custom_id": "c1", "error": {"code": "rate_limit", "message": "slow down"}}
                return httpx.Response(200, text=json.dumps(line))
            return httpx.Response(404, json={"error": "nf"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(api_key="x", client=client)
        runner = BatchRunner(OpenAIBatchBackend(provider), poll_interval_s=0.0)
        result = await runner.run(
            [BatchRequest(custom_id=f"c{i}", request=_req("gpt-5.2")) for i in range(2)]
        )
        assert result.by_id()["c0"].ok
        assert not result.by_id()["c1"].ok and "rate_limit" in result.by_id()["c1"].error

    @pytest.mark.asyncio
    async def test_anthropic_batch_errored_result_and_cancel(self):
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/messages/batches") and request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "id": "mb2",
                        "processing_status": "ended",
                        "results_url": "https://api.anthropic.com/v1/messages/batches/mb2/results",
                        "request_counts": {"succeeded": 1, "errored": 1},
                    },
                )
            if path.endswith("/results"):
                lines = [
                    {"custom_id": "c0", "result": {"type": "succeeded", "message": {
                        "model": "claude-sonnet-4-6",
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }}},
                    {"custom_id": "c1", "result": {"type": "errored", "error": {"type": "overloaded"}}},
                ]
                return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))
            if path.endswith("/cancel"):
                return httpx.Response(200, json={"id": "mb2", "processing_status": "canceling"})
            return httpx.Response(404, json={"error": "nf"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(api_key="x", client=client)
        backend = AnthropicBatchBackend(provider)
        runner = BatchRunner(backend, poll_interval_s=0.0)
        result = await runner.run(
            [BatchRequest(custom_id=f"c{i}", request=_req("claude-sonnet-4-6")) for i in range(2)]
        )
        assert result.by_id()["c0"].ok
        assert not result.by_id()["c1"].ok and "overloaded" in result.by_id()["c1"].error
        # cancel returns a non-terminal-mapped job without raising.
        job = await backend.cancel(result.job)
        assert job.id == "mb2"

    @pytest.mark.asyncio
    async def test_anthropic_batch_backend_roundtrip(self):
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/messages/batches") and request.method == "POST":
                return httpx.Response(
                    200, json={"id": "mb_1", "processing_status": "in_progress"}
                )
            if path.endswith("/messages/batches/mb_1"):
                return httpx.Response(
                    200,
                    json={
                        "id": "mb_1",
                        "processing_status": "ended",
                        "results_url": "https://api.anthropic.com/v1/messages/batches/mb_1/results",
                        "request_counts": {"succeeded": 2, "errored": 0},
                    },
                )
            if path.endswith("/results"):
                lines = [
                    {
                        "custom_id": cid,
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "model": "claude-sonnet-4-6",
                                "content": [{"type": "text", "text": f"answer {cid}"}],
                                "stop_reason": "end_turn",
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                            },
                        },
                    }
                    for cid in ("c0", "c1")
                ]
                return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))
            return httpx.Response(404, json={"error": "not found"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(api_key="x", client=client)
        runner = BatchRunner(AnthropicBatchBackend(provider), poll_interval_s=0.0)
        result = await runner.run(
            [BatchRequest(custom_id=f"c{i}", request=_req("claude-sonnet-4-6")) for i in range(2)]
        )
        assert {r.custom_id for r in result.succeeded} == {"c0", "c1"}
        assert result.by_id()["c1"].response.text == "answer c1"


# ---------------------------------------------------------------------------
# Circuit breaking & health-aware failover
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_trips_after_failure_threshold_and_fast_fails(self):
        clock = {"t": 0.0}
        events = EventBus()
        seen: list[str] = []
        events.subscribe("circuit.opened", lambda e: seen.append("opened"))
        cb = CircuitBreaker(
            _Flaky(fail=100), failure_threshold=0.5, min_calls=3, cooldown_s=10,
            events=events, clock=lambda: clock["t"],
        )
        errors = 0
        fast_failed = False
        for _ in range(6):
            try:
                await cb.generate(_req())
            except CircuitOpenError:
                fast_failed = True
                break
            except ProviderError:
                errors += 1
        assert cb.state is CircuitState.OPEN
        assert fast_failed and seen == ["opened"]
        # The breaker's inner provider stopped being called once open.
        assert cb.inner.calls == 3

    @pytest.mark.asyncio
    async def test_half_open_probe_closes_on_success(self):
        clock = {"t": 0.0}
        cb = CircuitBreaker(
            _Flaky(fail=3, default_text="ok"), failure_threshold=0.5, min_calls=3,
            cooldown_s=10, clock=lambda: clock["t"],
        )
        for _ in range(3):
            with pytest.raises(ProviderError):
                await cb.generate(_req())
        assert cb.state is CircuitState.OPEN
        clock["t"] = 20.0  # cooldown elapsed -> half-open admits a probe
        resp = await cb.generate(_req())
        assert resp.text == "ok"
        assert cb.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_probe_freed_on_cancellation(self):
        # A cancelled probe is not a health verdict; it must not consume the
        # single probe slot, or the breaker would never recover.
        clock = {"t": 0.0}
        cb = CircuitBreaker(
            _Flaky(fail=3), failure_threshold=0.5, min_calls=3, cooldown_s=10,
            half_open_max=1, clock=lambda: clock["t"],
        )
        for _ in range(3):
            with pytest.raises(ProviderError):
                await cb.generate(_req())
        assert cb.state is CircuitState.OPEN
        clock["t"] = 20.0  # half-open

        class Cancel(MockProvider):
            async def generate(self, request):
                raise asyncio.CancelledError()

        cb.inner = Cancel()
        with pytest.raises(asyncio.CancelledError):
            await cb.generate(_req())
        # Probe slot was returned, so a fresh probe is admitted (not rejected).
        cb.inner = MockProvider(default_text="ok")
        assert (await cb.generate(_req())).text == "ok"
        assert cb.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_latency_trips_breaker(self):
        clock = {"t": 0.0}

        class Slow(MockProvider):
            async def generate(self, request):
                clock["t"] += 5.0  # 5s per call
                return await super().generate(request)

        cb = CircuitBreaker(
            Slow(default_text="ok"), failure_threshold=0.5, min_calls=3,
            latency_threshold_ms=1000, cooldown_s=10, clock=lambda: clock["t"],
        )
        for _ in range(3):
            await cb.generate(_req())
        assert cb.state is CircuitState.OPEN  # all slow -> unhealthy


class TestHealthAwareFailover:
    @pytest.mark.asyncio
    async def test_steers_to_healthy_entry(self):
        clock = {"t": 0.0}
        bad = CircuitBreaker(_Flaky(fail=100), failure_threshold=0.5, min_calls=2,
                             cooldown_s=100, clock=lambda: clock["t"])
        good = CircuitBreaker(MockProvider(default_text="healthy"), clock=lambda: clock["t"])
        # Trip the bad breaker.
        for _ in range(2):
            try:
                await bad.generate(_req())
            except ProviderError:
                pass
        assert bad.state is CircuitState.OPEN
        chain = HealthAwareFailover([(bad, None), (good, None)])
        resp = await chain.generate(_req())
        assert resp.text == "healthy"
        # bad was not called again (still open, ranked last and skipped).
        assert bad.inner.calls == 2


# ---------------------------------------------------------------------------
# Key pooling & rate limiting
# ---------------------------------------------------------------------------


class TestKeyPool:
    @pytest.mark.asyncio
    async def test_round_robin_across_keys(self):
        a, b = MockProvider(default_text="a"), MockProvider(default_text="b")
        pool = KeyPool([a, b], breaker=False, seed=0)
        for _ in range(4):
            await pool.generate(_req())
        assert a.call_count == 2 and b.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_429_backoff_then_next_key(self):
        # First key always rate-limits; pool must fall through to the second.
        from vincio.core.errors import ProviderRateLimitError

        class Limited(MockProvider):
            async def generate(self, request):
                raise ProviderRateLimitError("429", retry_after_s=0.0, provider="limited")

        pool = KeyPool(
            [Limited(), MockProvider(default_text="ok")], breaker=False, seed=0,
            base_backoff_s=0.0, max_backoff_s=0.0,
        )
        resp = await pool.generate(_req())
        assert resp.text == "ok"

    def test_rate_limiter_buckets(self):
        clock = {"t": 0.0}
        rl = RateLimiter(rpm=60, tpm=600, clock=lambda: clock["t"])
        assert rl.available(tokens=10)
        # Drain one request slot.
        assert rl.wait_time(tokens=10) == 0.0

    @pytest.mark.asyncio
    async def test_stream_does_not_call_open_breaker(self):
        # When every key's breaker is open, stream() must NOT fall back to
        # calling a known-open breaker (the pre-fix bug); it exhausts attempts
        # and raises, leaving the dead provider untouched.
        clock = {"t": 0.0}
        inner = _Flaky(fail=10**9)
        pool = KeyPool(
            [inner], breaker=True, max_attempts=2, base_backoff_s=0.0, max_backoff_s=0.0,
            clock=lambda: clock["t"], seed=0,
        )
        breaker = pool.keys[0].provider
        breaker.state = CircuitState.OPEN  # forced open; cooldown not elapsed at t=0
        breaker._opened_at = 0.0
        with pytest.raises(ProviderUnavailableError):
            async for _ in pool.stream(_req()):
                pass
        assert inner.calls == 0  # the open breaker's provider was never called


# ---------------------------------------------------------------------------
# Runtime model cascades
# ---------------------------------------------------------------------------


class TestCascade:
    def test_default_confidence_signal(self):
        assert response_confidence(ModelResponse(text="x", finish_reason="stop")) == 1.0
        assert response_confidence(ModelResponse(text="x", finish_reason="length")) == 0.0
        assert response_confidence(ModelResponse(finish_reason="stop"), expects_schema=True) == 0.2

    def test_next_rung_logic(self):
        casc = ModelCascade.from_models(["cheap", "mid", "strong"], min_confidence=0.5)
        assert casc.next_rung("cheap", confidence=0.2).model == "mid"
        assert casc.next_rung("cheap", confidence=0.9) is None  # confident: stay
        assert casc.next_rung("strong", confidence=0.0) is None  # top rung

    def test_escalates_on_low_confidence(self, offline_config, tmp_cwd):
        def responder(req):
            if "mini" in req.model:
                return ModelResponse(text="truncated", finish_reason="length")
            return ModelResponse(text="full", finish_reason="stop")

        app = ContextApp(name="c", provider=MockProvider(responder=responder), config=offline_config)
        app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
        result = app.run("hi")
        assert result.metadata["cascade"]["model"] == "gpt-5.2"
        assert result.metadata["cascade"]["escalations"] == 1

    def test_no_escalation_when_confident(self, offline_config, tmp_cwd):
        app = ContextApp(name="c", provider=MockProvider(default_text="ok"), config=offline_config)
        app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
        result = app.run("hi")
        assert result.metadata["cascade"]["model"] == "gpt-5.2-mini"
        assert result.metadata["cascade"]["escalations"] == 0

    def test_custom_confidence_fn(self, offline_config, tmp_cwd):
        app = ContextApp(name="c", provider=MockProvider(default_text="ok"), config=offline_config)
        app.use_cascade(["gpt-5.2-mini", "gpt-5.2"], confidence=lambda r: 0.0)  # always escalate
        result = app.run("hi")
        assert result.metadata["cascade"]["escalations"] == 1

    def test_duplicate_rung_models_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            ModelCascade.from_models(["gpt-5.2", "gpt-5.2"])

    def test_explicit_run_model_overrides_cascade(self, offline_config, tmp_cwd):
        seen: list[str] = []
        app = ContextApp(
            name="c",
            provider=MockProvider(responder=lambda r: seen.append(r.model) or ModelResponse(text="ok")),
            config=offline_config,
        )
        app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
        from vincio.core.types import RunConfig

        result = app.run("hi", config=RunConfig(model="gpt-4o"))
        assert seen == ["gpt-4o"]  # pinned model wins; cascade not engaged
        assert "cascade" not in result.metadata

    @pytest.mark.asyncio
    async def test_streaming_starts_on_first_rung(self, offline_config, tmp_cwd):
        app = ContextApp(name="c", provider=MockProvider(default_text="ok"), config=offline_config)
        app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
        async for _ in app.astream("hi"):
            pass
        trace = app.tracer.exporter.traces[-1]
        model_span = next(s for s in trace.spans if s.type == "model_call")
        assert model_span.attributes["model"] == "gpt-5.2-mini"  # cheap rung, no mid-stream escalation


# ---------------------------------------------------------------------------
# Cost attribution & budgets
# ---------------------------------------------------------------------------


class TestCostAttribution:
    def test_ledger_rolls_up_by_dimension(self):
        ledger = CostLedger()
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        ledger.record_model_call(model="m", usage=usage, cost_usd=1.0, tenant_id="a", feature="chat")
        ledger.record_model_call(model="m", usage=usage, cost_usd=2.0, tenant_id="a", feature="search")
        ledger.record_model_call(model="m", usage=usage, cost_usd=4.0, tenant_id="b", feature="chat")
        by_tenant = {r.key: r.cost_usd for r in ledger.report("tenant").rows}
        assert by_tenant == {"a": 3.0, "b": 4.0}
        by_feature = {r.key: r.cost_usd for r in ledger.report("feature").rows}
        assert by_feature == {"chat": 5.0, "search": 2.0}
        assert ledger.report("tenant").total_usd == 7.0

    def test_attribution_through_app_run(self, offline_config, tmp_cwd):
        app = ContextApp(name="a", provider=MockProvider(default_text="ok"), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e6, output_per_mtok=1e6))
        app.run("hello world", tenant_id="acme", feature="chat", user_id="u1")
        report = app.cost_report(by="tenant")
        assert any(r.key == "acme" and r.cost_usd > 0 for r in report.rows)
        assert any(r.key == "chat" for r in app.cost_report(by="feature").rows)

    def test_ledger_persists_to_store_and_reloads(self, offline_config, tmp_cwd):
        app = ContextApp(name="a", provider=MockProvider(default_text="ok"), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e6))
        app.run("hello", tenant_id="acme")
        reloaded = CostLedger.from_store(app.store)
        assert reloaded.report("tenant").total_usd == pytest.approx(app.cost_report(by="tenant").total_usd)
        assert reloaded.report("tenant").total_usd > 0

    def test_agent_calls_are_attributed(self, offline_config, tmp_cwd):
        app = ContextApp(name="a", provider=MockProvider(default_text="ok"), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e6, output_per_mtok=1e6))

        def tool(x: str) -> str:
            """echo"""
            return x

        app.agent(tools=[tool], planner="react", max_steps=3).run(
            "do something", tenant_id="acme", feature="research"
        )
        report = app.cost_report(by="tenant")
        assert any(r.key == "acme" and r.cost_usd > 0 for r in report.rows)
        assert any(r.key == "research" for r in app.cost_report(by="feature").rows)

    def test_crew_calls_are_attributed(self, offline_config, tmp_cwd):
        app = ContextApp(name="c", provider=MockProvider(default_text="done"), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e6, output_per_mtok=1e6))
        crew = app.crew(
            members=[{"name": "r", "goal": "research"}, {"name": "w", "goal": "write"}],
            process="sequential",
        )
        crew.run("summarize", tenant_id="globex", feature="report")
        assert any(r.key == "globex" and r.cost_usd > 0 for r in app.cost_report(by="tenant").rows)

    def test_response_cache_hit_is_free(self, tmp_cwd):
        from vincio.core.config import CacheConfig

        config = offline_config_with(tmp_cwd, cache=CacheConfig(response_cache=True))
        app = ContextApp(name="cache", provider=MockProvider(default_text="cached"), config=config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e6, output_per_mtok=1e6))
        first = app.run("same question")
        second = app.run("same question")
        assert first.cost_usd > 0 and second.cost_usd == 0.0  # cache hit billed nothing
        # The ledger reflects the free hit too.
        events = app.cost_ledger.events
        assert events[-1].cost_usd == 0.0


def offline_config_with(tmp_path, **overrides):
    from vincio import VincioConfig

    config = VincioConfig()
    config.storage.metadata = "memory://"
    config.observability.exporter = "memory"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestStreamingCascade:
    @pytest.mark.asyncio
    async def test_streaming_run_escalates_and_streams_accepted_answer(self, offline_config, tmp_cwd):
        def responder(req):
            if "mini" in req.model:
                return ModelResponse(text="partial", finish_reason="length")
            return ModelResponse(text="full answer", finish_reason="stop")

        app = ContextApp(name="s", provider=MockProvider(responder=responder), config=offline_config)
        app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
        deltas: list[str] = []
        result = None
        async for ev in app.astream("hi"):
            if ev.type == "text_delta":
                deltas.append(ev.text or "")
            elif ev.type == "done":
                result = ev.result
        streamed = "".join(deltas)
        assert result.metadata["cascade"]["escalations"] == 1
        assert "full answer" in streamed and "partial" not in streamed  # discarded cheap answer never streamed


class TestBudgets:
    def test_hard_cap_denies(self, offline_config, tmp_cwd):
        app = ContextApp(name="b", provider=MockProvider(default_text="ok"), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e9))
        # This test exercises the per-tenant cost SLO, a distinct layer from the
        # per-run hard cap; raise the run cap so the SLO is what denies.
        app.budget = app.budget.model_copy(update={"max_cost_usd": 1e12})
        app.set_cost_budget(scope="tenant", id="acme", limit_usd=0.001, period="total")
        first = app.run("a", tenant_id="acme")
        second = app.run("b", tenant_id="acme")
        assert first.status.value == "succeeded"
        assert second.status.value == "denied" and "budget" in second.error

    def test_degrade_swaps_model(self, offline_config, tmp_cwd):
        seen: list[str] = []

        def responder(req):
            seen.append(req.model)
            return ModelResponse(text="ok")

        app = ContextApp(name="b", provider=MockProvider(responder=responder), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e9))
        app.budget = app.budget.model_copy(update={"max_cost_usd": 1e12})
        app.set_cost_budget(
            scope="tenant", id="acme", limit_usd=0.001, period="total",
            on_breach="degrade", degrade_model="gpt-5.2-nano",
        )
        app.run("a", tenant_id="acme")  # over budget after this
        result = app.run("b", tenant_id="acme")
        assert result.status.value == "succeeded"
        assert seen[-1] == "gpt-5.2-nano"  # degraded model used

    def test_queue_to_batch_denies_with_hint(self, offline_config, tmp_cwd):
        app = ContextApp(name="b", provider=MockProvider(default_text="ok"), config=offline_config)
        app.cost_tracker.price_table.set("gpt-5.2", ModelPrice(input_per_mtok=1e9))
        app.budget = app.budget.model_copy(update={"max_cost_usd": 1e12})
        app.set_cost_budget(scope="global", limit_usd=1e-9, period="total", on_breach="queue_to_batch")
        app.run("a")
        result = app.run("b")
        assert result.status.value == "denied" and "batch" in result.error

    def test_anomaly_event_raised(self):
        events = EventBus()
        fired: list[dict] = []
        events.subscribe("cost.anomaly", lambda e: fired.append(e.payload))
        ledger = CostLedger()
        mgr = BudgetManager(ledger, events=events)
        mgr.add(CostBudget(scope="global", limit_usd=1e9, anomaly_factor=3.0))
        # Establish a baseline of cheap calls, then a spike.
        for _ in range(6):
            mgr.observe(CostEvent(cost_usd=0.001))
        mgr.observe(CostEvent(cost_usd=1.0))
        assert fired and fired[0]["factor"] == 3.0


# ---------------------------------------------------------------------------
# Provider-aware prompt caching
# ---------------------------------------------------------------------------


class TestPromptCache:
    def test_strategy_applies_ttl_to_long_prefix(self):
        caps = ModelCapabilities(prompt_caching=True)
        strat = PromptCacheStrategy(ttl="1h", min_prefix_tokens=5)
        messages = [
            Message(role="system", content="system instructions " * 50, cache_hint=True),
            Message(role="user", content="question"),
        ]
        out, info = strat.apply(messages, capabilities=caps)
        assert info["applied"] and info["breakpoints"] == 1
        assert out[0].cache_ttl == "1h"

    def test_strategy_skips_short_prefix(self):
        caps = ModelCapabilities(prompt_caching=True)
        strat = PromptCacheStrategy(min_prefix_tokens=10_000)
        messages = [Message(role="system", content="short", cache_hint=True), Message(role="user", content="q")]
        out, info = strat.apply(messages, capabilities=caps)
        assert not info["applied"] and out[0].cache_ttl is None

    def test_strategy_noop_for_non_caching_provider(self):
        caps = ModelCapabilities(prompt_caching=False)
        strat = PromptCacheStrategy()
        messages = [Message(role="system", content="x " * 500, cache_hint=True)]
        _, info = strat.apply(messages, capabilities=caps)
        assert not info["applied"] and not info["supported"]

    @pytest.mark.asyncio
    async def test_anthropic_emits_cache_control_ttl(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            seen["beta"] = request.headers.get("anthropic-beta")
            return httpx.Response(
                200,
                json={
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 4},
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(api_key="x", client=client)
        request = ModelRequest(
            model="claude-sonnet-4-6",
            messages=[
                Message(role="system", content="stable", cache_hint=True, cache_ttl="1h"),
                Message(role="user", content="q"),
            ],
        )
        resp = await provider.generate(request)
        assert seen["payload"]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "extended-cache-ttl" in seen["beta"]
        assert resp.usage.cached_input_tokens == 4

    def test_anthropic_caches_multipart_message(self):
        # A multi-part message marked cache_hint must still get a cache breakpoint
        # on its last content block (not only single-string messages).
        from vincio.core.types import ContentPart

        provider = AnthropicProvider(api_key="x")
        request = ModelRequest(
            model="claude-sonnet-4-6",
            messages=[
                Message(
                    role="user",
                    content=[ContentPart(type="text", text="a"), ContentPart(type="text", text="b")],
                    cache_hint=True,
                    cache_ttl="1h",
                )
            ],
        )
        _system, messages = provider._render(request)
        assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "cache_control" not in messages[0]["content"][0]  # only the last block is the breakpoint

    def test_cache_hit_rate_telemetry_on_span(self, offline_config, tmp_cwd):
        def responder(req):
            return ModelResponse(
                text="ok", usage=TokenUsage(input_tokens=100, cached_input_tokens=40, output_tokens=5)
            )

        app = ContextApp(name="c", provider=MockProvider(responder=responder), config=offline_config)
        app.run("hi")
        trace = app.tracer.exporter.traces[-1]
        model_span = next(s for s in trace.spans if s.type == "model_call")
        assert model_span.attributes["cache_hit_rate"] == pytest.approx(0.4)
        assert model_span.attributes["cached_input_tokens"] == 40


# ---------------------------------------------------------------------------
# Incremental & sharded indexing
# ---------------------------------------------------------------------------


def _chunks(n: int, doc: str = "d1", prefix: str = "text") -> list[Chunk]:
    return [Chunk(id=f"{doc}-c{i}", document_id=doc, text=f"{prefix} {i}", index=i) for i in range(n)]


class TestIncrementalIndexing:
    @pytest.mark.asyncio
    async def test_content_hash_skips_unchanged(self):
        live = LiveIndex(VectorIndex(LocalHashEmbedder()))
        chunks = _chunks(5)
        first = await live.upsert(chunks)
        assert first.added == 5 and first.reembedded == 5
        again = await live.upsert([c.model_copy(deep=True) for c in chunks])
        assert again.unchanged == 5 and again.reembedded == 0
        edited = [c.model_copy(deep=True) for c in chunks]
        edited[2].text = "text 2 EDITED"
        third = await live.upsert(edited)
        assert third.updated == 1 and third.unchanged == 4
        assert len(live) == 5

    @pytest.mark.asyncio
    async def test_unchanged_chunk_keeps_indexed_at(self):
        # Re-upserting unchanged content must not rewrite it, so the inner
        # index's freshness stamp stays consistent with the original index time.
        inner = VectorIndex(LocalHashEmbedder())
        live = LiveIndex(inner)
        chunks = _chunks(3)
        await live.upsert(chunks)
        stamped = inner.chunks["d1-c0"].metadata["indexed_at"]
        again = await live.upsert([c.model_copy(deep=True) for c in chunks])
        assert again.unchanged == 3 and again.reembedded == 0
        assert inner.chunks["d1-c0"].metadata["indexed_at"] == stamped  # not re-stamped

    @pytest.mark.asyncio
    async def test_streaming_ingestion(self):
        live = LiveIndex(VectorIndex(LocalHashEmbedder()))

        async def stream():
            for chunk in _chunks(10, doc="s"):
                yield chunk

        stats = await live.upsert_stream(stream(), batch_size=4)
        assert stats.added == 10 and len(live) == 10


class TestShardedIndex:
    @pytest.mark.asyncio
    async def test_add_search_delete_len(self):
        shards = [VectorIndex(LocalHashEmbedder()) for _ in range(3)]
        index = ShardedIndex(shards)
        chunks = [
            Chunk(id=f"k{i}", document_id=f"doc{i % 4}", text=f"apple banana {i}", index=i)
            for i in range(20)
        ]
        await index.add(chunks)
        assert len(index) == 20
        assert sum(len(s) for s in shards) == 20
        hits = await index.search("apple", top_k=5)
        assert len(hits) == 5
        removed = await index.delete(["k0", "k1"])
        assert removed == 2 and len(index) == 18

    @pytest.mark.asyncio
    async def test_document_chunks_colocate(self):
        shards = [VectorIndex(LocalHashEmbedder()) for _ in range(3)]
        index = ShardedIndex(shards)
        await index.add(_chunks(6, doc="onedoc"))
        # All chunks of one document route to the same shard.
        non_empty = [s for s in shards if len(s) > 0]
        assert len(non_empty) == 1 and len(non_empty[0]) == 6

    @pytest.mark.asyncio
    async def test_live_over_sharded_composes(self):
        index = LiveIndex(ShardedIndex([BM25Index() for _ in range(2)]))
        await index.upsert(_chunks(8))
        assert len(index) == 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cost_report_command(self, tmp_path, capsys):
        from vincio.cli.main import main
        from vincio.storage.base import create_metadata_store

        db = tmp_path / "vincio.db"
        store = create_metadata_store(f"sqlite:///{db}")
        ledger = CostLedger(store=store)
        ledger.record_model_call(
            model="m", usage=TokenUsage(input_tokens=10), cost_usd=1.5, tenant_id="acme"
        )
        code = main(["cost", "report", "--by", "tenant", "--db", str(db)])
        assert code == 0
        assert "acme" in capsys.readouterr().out

    def test_batch_command(self, tmp_path, capsys):
        from vincio.cli.main import main

        app_file = tmp_path / "app.py"
        app_file.write_text(
            "from vincio import ContextApp\n"
            "from vincio.providers import MockProvider\n"
            'app = ContextApp(name="cli", provider=MockProvider(default_text="ok"))\n',
            encoding="utf-8",
        )
        code = main(["batch", str(app_file), "--input", "a", "--input", "b"])
        assert code == 0
        assert "2/2 succeeded" in capsys.readouterr().out
