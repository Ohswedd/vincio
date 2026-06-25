"""Real-behavior coverage for the circuit breaker + health-aware failover.

Drives the open/half-open/closed state machine, the failure-threshold trip,
the cooldown-to-half-open transition, probe budgeting, latency-based health,
stream health accounting, and the health-ordered failover chain — all offline
with deterministic fake providers and a manually advanced clock.
"""

from __future__ import annotations

import pytest

from vincio.core.errors import (
    CircuitOpenError,
    ModelRetiredError,
    ProviderError,
    ProviderUnavailableError,
)
from vincio.core.types import (
    Message,
    ModelCapabilities,
    ModelEvent,
    ModelProfile,
    ModelRequest,
    ModelResponse,
)
from vincio.providers import MockProvider
from vincio.providers.base import ModelProvider
from vincio.providers.circuit import CircuitBreaker, CircuitState, HealthAwareFailover


def _req(text: str = "hi", model: str = "x") -> ModelRequest:
    return ModelRequest(model=model, messages=[Message(role="user", content=text)])


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FlakyProvider(ModelProvider):
    """Provider whose every call raises a chosen error (or succeeds)."""

    name = "flaky"

    def __init__(self, *, fail: bool = True, error: Exception | None = None) -> None:
        self.fail = fail
        self.error = error or ProviderError("boom", provider="flaky", retryable=True)
        self.calls = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.fail:
            raise self.error
        return ModelResponse(text="ok", model=request.model, provider=self.name)

    async def stream(self, request: ModelRequest):
        self.calls += 1
        if self.fail:
            raise self.error
        yield ModelEvent(type="text_delta", text="ok")
        yield ModelEvent(type="done")


class MidStreamProvider(ModelProvider):
    """Yields one token, then raises — exercises the mid-stream branch."""

    name = "midstream"

    async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
        return ModelResponse(text="ok", model=request.model, provider=self.name)

    async def stream(self, request: ModelRequest):
        yield ModelEvent(type="text_delta", text="partial")
        raise ProviderError("died mid-stream", provider=self.name)


class CancelProvider(ModelProvider):
    """Raises a non-ProviderError (cancellation/abort)."""

    name = "cancel"

    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise asyncio_cancel()

    async def stream(self, request: ModelRequest):
        raise asyncio_cancel()
        yield  # pragma: no cover  (unreachable, makes this an async generator)


def asyncio_cancel() -> BaseException:
    import asyncio

    return asyncio.CancelledError()


async def _drain(provider: ModelProvider, request: ModelRequest) -> list[ModelEvent]:
    return [event async for event in provider.stream(request)]


# -- construction & validation ----------------------------------------------


def test_failure_threshold_zero_rejected() -> None:
    with pytest.raises(ValueError, match=r"failure_threshold must be in \(0, 1\]"):
        CircuitBreaker(MockProvider(), failure_threshold=0.0)


def test_failure_threshold_above_one_rejected() -> None:
    with pytest.raises(ValueError, match=r"failure_threshold must be in \(0, 1\]"):
        CircuitBreaker(MockProvider(), failure_threshold=1.5)


def test_min_calls_and_window_floored() -> None:
    # min_calls floored to 1; window floored to min_calls.
    breaker = CircuitBreaker(MockProvider(), min_calls=0, window=0)
    assert breaker.min_calls == 1
    assert breaker.window == 1


def test_window_never_below_min_calls() -> None:
    breaker = CircuitBreaker(MockProvider(), min_calls=10, window=3)
    assert breaker.window == 10


def test_inherits_inner_name() -> None:
    breaker = CircuitBreaker(FlakyProvider())
    assert breaker.name == "flaky"


# -- failure_rate / snapshot ------------------------------------------------


def test_failure_rate_empty_is_zero() -> None:
    breaker = CircuitBreaker(MockProvider())
    assert breaker.failure_rate() == 0.0


def test_snapshot_fields() -> None:
    breaker = CircuitBreaker(FlakyProvider(), min_calls=2)
    snap = breaker.snapshot()
    assert snap == {
        "name": "flaky",
        "state": "closed",
        "failure_rate": 0.0,
        "samples": 0,
        "trips": 0,
    }


# -- closed -> open: failure threshold --------------------------------------


@pytest.mark.asyncio
async def test_trips_open_after_threshold() -> None:
    breaker = CircuitBreaker(
        FlakyProvider(fail=True), failure_threshold=0.5, min_calls=4, window=10
    )
    req = _req()
    # 4 failures: failure_rate reaches 1.0 >= 0.5 once min_calls met.
    for _ in range(4):
        with pytest.raises(ProviderError):
            await breaker.generate(req)
    assert breaker.state is CircuitState.OPEN
    assert breaker.trip_count == 1
    # Now fails fast without touching inner.
    inner_calls = breaker.inner.calls
    with pytest.raises(CircuitOpenError, match=r"circuit open for provider 'flaky'"):
        await breaker.generate(req)
    assert breaker.inner.calls == inner_calls  # inner not called when open


@pytest.mark.asyncio
async def test_does_not_trip_below_min_calls() -> None:
    breaker = CircuitBreaker(FlakyProvider(fail=True), failure_threshold=0.5, min_calls=5)
    req = _req()
    for _ in range(4):
        with pytest.raises(ProviderError):
            await breaker.generate(req)
    # 4 < min_calls=5, so still closed despite 100% failure.
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_rate() == 1.0


@pytest.mark.asyncio
async def test_mixed_outcomes_below_threshold_stay_closed() -> None:
    inner = FlakyProvider(fail=True)
    breaker = CircuitBreaker(inner, failure_threshold=0.75, min_calls=4, window=4)
    req = _req()
    # 2 failures then 2 successes -> rate 0.5 < 0.75.
    for _ in range(2):
        with pytest.raises(ProviderError):
            await breaker.generate(req)
    inner.fail = False
    for _ in range(2):
        await breaker.generate(req)
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_rate() == 0.5


# -- latency-based health ----------------------------------------------------


@pytest.mark.asyncio
async def test_slow_calls_count_unhealthy_and_trip() -> None:
    clock = FakeClock()

    class SlowProvider(ModelProvider):
        name = "slow"

        async def generate(self, request: ModelRequest) -> ModelResponse:
            clock.advance(0.2)  # 200 ms
            return ModelResponse(text="ok", model=request.model, provider=self.name)

    breaker = CircuitBreaker(
        SlowProvider(),
        latency_threshold_ms=100.0,
        failure_threshold=0.5,
        min_calls=2,
        clock=clock,
    )
    req = _req()
    for _ in range(2):
        await breaker.generate(req)
    assert breaker.failure_rate() == 1.0
    assert breaker.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_fast_calls_stay_healthy() -> None:
    clock = FakeClock()

    class FastProvider(ModelProvider):
        name = "fast"

        async def generate(self, request: ModelRequest) -> ModelResponse:
            clock.advance(0.01)  # 10 ms < threshold
            return ModelResponse(text="ok", model=request.model, provider=self.name)

    breaker = CircuitBreaker(
        FastProvider(), latency_threshold_ms=100.0, min_calls=2, clock=clock
    )
    req = _req()
    for _ in range(5):
        await breaker.generate(req)
    assert breaker.failure_rate() == 0.0
    assert breaker.state is CircuitState.CLOSED


# -- cooldown -> half-open -> closed/open -----------------------------------


@pytest.mark.asyncio
async def test_open_cooldown_goes_half_open_then_closes_on_probe_success() -> None:
    clock = FakeClock()
    inner = FlakyProvider(fail=True)
    breaker = CircuitBreaker(
        inner, failure_threshold=0.5, min_calls=2, cooldown_s=30.0, clock=clock
    )
    req = _req()
    for _ in range(2):
        with pytest.raises(ProviderError):
            await breaker.generate(req)
    assert breaker.state is CircuitState.OPEN
    assert not breaker.healthy

    # Before cooldown elapses: still open, fast-fail.
    clock.advance(10)
    assert not breaker.healthy
    with pytest.raises(CircuitOpenError):
        await breaker.generate(req)

    # After cooldown: effective state is half-open and admits a probe.
    clock.advance(25)  # total 35 >= 30
    assert breaker.healthy
    assert breaker._effective_state() is CircuitState.HALF_OPEN
    inner.fail = False  # the probe will succeed
    resp = await breaker.generate(req)
    assert resp.text == "ok"
    # Successful probe closes the breaker and clears the window.
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_rate() == 0.0


@pytest.mark.asyncio
async def test_half_open_probe_failure_reopens() -> None:
    clock = FakeClock()
    inner = FlakyProvider(fail=True)
    breaker = CircuitBreaker(
        inner, failure_threshold=0.5, min_calls=2, cooldown_s=30.0, clock=clock
    )
    req = _req()
    for _ in range(2):
        with pytest.raises(ProviderError):
            await breaker.generate(req)
    assert breaker.trip_count == 1
    clock.advance(31)
    # Probe fails -> re-opens, trip_count increments, _opened_at reset.
    with pytest.raises(ProviderError):
        await breaker.generate(req)
    assert breaker.state is CircuitState.OPEN
    assert breaker.trip_count == 2
    assert breaker._opened_at == clock.t


@pytest.mark.asyncio
async def test_half_open_probe_budget_enforced() -> None:
    clock = FakeClock()
    inner = FlakyProvider(fail=True)
    breaker = CircuitBreaker(
        inner,
        failure_threshold=0.5,
        min_calls=2,
        cooldown_s=30.0,
        half_open_max=1,
        clock=clock,
    )
    req = _req()
    for _ in range(2):
        with pytest.raises(ProviderError):
            await breaker.generate(req)
    clock.advance(31)
    inner.fail = False
    breaker._before_call()  # claim the single probe slot without resolving it
    assert breaker.state is CircuitState.HALF_OPEN
    # A second concurrent probe is rejected: budget reached.
    with pytest.raises(CircuitOpenError, match=r"probe limit reached"):
        breaker._before_call()


# -- abort/cancel does not consume health or probe budget --------------------


@pytest.mark.asyncio
async def test_cancellation_not_counted_as_failure() -> None:
    import asyncio

    breaker = CircuitBreaker(CancelProvider(), failure_threshold=0.5, min_calls=2)
    req = _req()
    for _ in range(5):
        with pytest.raises(asyncio.CancelledError):
            await breaker.generate(req)
    # No outcome recorded; never trips.
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_rate() == 0.0


@pytest.mark.asyncio
async def test_abort_releases_half_open_probe_slot() -> None:
    import asyncio

    clock = FakeClock()
    inner = CancelProvider()
    breaker = CircuitBreaker(inner, cooldown_s=30.0, clock=clock)
    # Force the breaker open by hand, then cool down to half-open.
    breaker._transition(CircuitState.OPEN)
    clock.advance(31)
    assert breaker._effective_state() is CircuitState.HALF_OPEN
    with pytest.raises(asyncio.CancelledError):
        await breaker.generate(_req())
    # The probe slot was freed (abort is not a health signal).
    assert breaker._half_open_calls == 0


def test_abort_probe_noop_when_closed() -> None:
    breaker = CircuitBreaker(FlakyProvider())
    breaker._half_open_calls = 0
    breaker._abort_probe()  # closed state: nothing to release
    assert breaker._half_open_calls == 0


# -- _transition no-op -------------------------------------------------------


def test_transition_same_state_is_noop() -> None:
    breaker = CircuitBreaker(FlakyProvider())
    assert breaker.state is CircuitState.CLOSED
    breaker._transition(CircuitState.CLOSED)  # no-op, no trip
    assert breaker.trip_count == 0
    assert breaker.state is CircuitState.CLOSED


# -- event emission ----------------------------------------------------------


@pytest.mark.asyncio
async def test_events_emitted_on_state_changes() -> None:
    events: list[tuple[str, dict]] = []

    class Bus:
        def emit(self, name: str, payload: dict) -> None:
            events.append((name, payload))

    clock = FakeClock()
    inner = FlakyProvider(fail=True)
    breaker = CircuitBreaker(
        inner, failure_threshold=0.5, min_calls=2, cooldown_s=10.0, events=Bus(), clock=clock
    )
    req = _req()
    for _ in range(2):
        with pytest.raises(ProviderError):
            await breaker.generate(req)
    clock.advance(11)
    inner.fail = False
    await breaker.generate(req)  # probe closes
    names = [name for name, _ in events]
    # opened on trip, half_open when the cooled-down probe is admitted, closed
    # when that probe succeeds.
    assert names == ["circuit.opened", "circuit.half_open", "circuit.closed"]
    # Payload carries the provider name and snapshot.
    assert events[0][1]["provider"] == "flaky"
    assert events[0][1]["state"] == "open"


# -- stream health accounting ------------------------------------------------


@pytest.mark.asyncio
async def test_stream_pre_token_failure_counts_unhealthy() -> None:
    breaker = CircuitBreaker(
        FlakyProvider(fail=True), failure_threshold=0.5, min_calls=2
    )
    req = _req()
    for _ in range(2):
        with pytest.raises(ProviderError):
            await _drain(breaker, req)
    assert breaker.failure_rate() == 1.0
    assert breaker.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_stream_midstream_failure_not_counted() -> None:
    breaker = CircuitBreaker(MidStreamProvider(), failure_threshold=0.5, min_calls=2)
    req = _req()
    for _ in range(3):
        events: list[ModelEvent] = []
        with pytest.raises(ProviderError, match="died mid-stream"):
            async for event in breaker.stream(req):
                events.append(event)
        assert events[0].text == "partial"  # a token did arrive
    # Mid-stream breaks recorded as healthy (not yielded -> False here is False).
    assert breaker.failure_rate() == 0.0
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_stream_success_records_healthy() -> None:
    breaker = CircuitBreaker(FlakyProvider(fail=False), min_calls=2)
    req = _req()
    events = await _drain(breaker, req)
    assert [e.type for e in events] == ["text_delta", "done"]
    assert breaker.failure_rate() == 0.0


@pytest.mark.asyncio
async def test_stream_slow_counts_unhealthy() -> None:
    clock = FakeClock()

    class SlowStream(ModelProvider):
        name = "slowstream"

        async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
            return ModelResponse(text="ok", model=request.model)

        async def stream(self, request: ModelRequest):
            yield ModelEvent(type="text_delta", text="ok")
            clock.advance(0.5)  # 500 ms after first token

    breaker = CircuitBreaker(
        SlowStream(), latency_threshold_ms=100.0, failure_threshold=0.5, min_calls=2, clock=clock
    )
    req = _req()
    for _ in range(2):
        await _drain(breaker, req)
    assert breaker.failure_rate() == 1.0
    assert breaker.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_stream_fast_fails_when_open() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(FlakyProvider(fail=True), cooldown_s=100.0, clock=clock)
    breaker._transition(CircuitState.OPEN)
    with pytest.raises(CircuitOpenError):
        await _drain(breaker, _req())


@pytest.mark.asyncio
async def test_stream_cancellation_releases_probe() -> None:
    import asyncio

    clock = FakeClock()
    breaker = CircuitBreaker(CancelProvider(), cooldown_s=30.0, clock=clock)
    breaker._transition(CircuitState.OPEN)
    clock.advance(31)
    with pytest.raises(asyncio.CancelledError):
        await _drain(breaker, _req())
    assert breaker._half_open_calls == 0


# -- delegation methods ------------------------------------------------------


def test_capabilities_delegates_to_inner() -> None:
    class CapProvider(ModelProvider):
        name = "cap"

        async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
            return ModelResponse(text="x", model=request.model)

        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(vision=True, max_context_tokens=12345)

    breaker = CircuitBreaker(CapProvider())
    caps = breaker.capabilities("any")
    assert caps.vision is True
    assert caps.max_context_tokens == 12345


@pytest.mark.asyncio
async def test_embed_delegates_to_inner() -> None:
    breaker = CircuitBreaker(MockProvider(embedding_dim=8))
    vectors = await breaker.embed(["alpha beta", "alpha"])
    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)


@pytest.mark.asyncio
async def test_list_models_and_aclose_delegate() -> None:
    closed: list[bool] = []

    class ListProvider(ModelProvider):
        name = "lister"

        async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
            return ModelResponse(text="x", model=request.model)

        async def list_models(self) -> list[ModelProfile]:
            return [ModelProfile(name="m1", provider="lister", model="m1")]

        async def aclose(self) -> None:
            closed.append(True)

    breaker = CircuitBreaker(ListProvider())
    models = await breaker.list_models()
    assert [m.model for m in models] == ["m1"]
    await breaker.aclose()
    assert closed == [True]


# -- HealthAwareFailover -----------------------------------------------------


def test_failover_requires_entries() -> None:
    with pytest.raises(ValueError, match="requires at least one provider"):
        HealthAwareFailover([])


@pytest.mark.asyncio
async def test_failover_uses_first_healthy() -> None:
    good = FlakyProvider(fail=False)
    chain = HealthAwareFailover([(good, None)], guard_capabilities=False)
    resp = await chain.generate(_req())
    assert resp.text == "ok"
    assert good.calls == 1


@pytest.mark.asyncio
async def test_failover_falls_through_to_second() -> None:
    bad = FlakyProvider(fail=True)
    good = FlakyProvider(fail=False)
    chain = HealthAwareFailover([(bad, None), (good, None)], guard_capabilities=False)
    resp = await chain.generate(_req())
    assert resp.text == "ok"
    assert bad.calls == 1
    assert good.calls == 1


@pytest.mark.asyncio
async def test_failover_all_fail_raises_aggregate() -> None:
    bad1 = FlakyProvider(fail=True, error=ProviderError("e1", provider="flaky"))
    bad2 = FlakyProvider(fail=True, error=ProviderError("e2", provider="flaky"))
    chain = HealthAwareFailover([(bad1, None), (bad2, None)], guard_capabilities=False)
    with pytest.raises(ProviderUnavailableError, match="all providers failed") as exc:
        await chain.generate(_req())
    assert "e1" in exc.value.message
    assert "e2" in exc.value.message


@pytest.mark.asyncio
async def test_failover_classifies_lifecycle_error() -> None:
    retired = FlakyProvider(
        fail=True, error=ProviderError("model_not_found: gone", provider="flaky")
    )
    chain = HealthAwareFailover([(retired, None)], guard_capabilities=False)
    with pytest.raises(ProviderUnavailableError, match="rotate now") as exc:
        await chain.generate(_req())
    assert "model_not_found" in exc.value.message


@pytest.mark.asyncio
async def test_failover_orders_open_breaker_last() -> None:
    clock = FakeClock()
    # First entry is an OPEN breaker; second is healthy.
    tripped_inner = FlakyProvider(fail=False)
    tripped = CircuitBreaker(tripped_inner, cooldown_s=10_000.0, clock=clock)
    tripped._transition(CircuitState.OPEN)
    healthy = FlakyProvider(fail=False)
    chain = HealthAwareFailover(
        [(tripped, None), (healthy, None)], guard_capabilities=False
    )
    resp = await chain.generate(_req())
    assert resp.text == "ok"
    # The healthy provider was tried first despite being second in declaration;
    # the open breaker was never invoked.
    assert healthy.calls == 1
    assert tripped_inner.calls == 0


@pytest.mark.asyncio
async def test_failover_applies_model_override() -> None:
    seen: list[str] = []

    class RecordingProvider(ModelProvider):
        name = "rec"

        async def generate(self, request: ModelRequest) -> ModelResponse:
            seen.append(request.model)
            return ModelResponse(text="ok", model=request.model, provider=self.name)

    chain = HealthAwareFailover(
        [(RecordingProvider(), "override-model")], guard_capabilities=False
    )
    await chain.generate(_req(model="original"))
    assert seen == ["override-model"]


@pytest.mark.asyncio
async def test_failover_stream_falls_through() -> None:
    bad = FlakyProvider(fail=True)
    good = FlakyProvider(fail=False)
    chain = HealthAwareFailover([(bad, None), (good, None)], guard_capabilities=False)
    events = [e async for e in chain.stream(_req())]
    assert [e.type for e in events] == ["text_delta", "done"]


@pytest.mark.asyncio
async def test_failover_stream_midstream_break_propagates() -> None:
    chain = HealthAwareFailover(
        [(MidStreamProvider(), None), (FlakyProvider(fail=False), None)],
        guard_capabilities=False,
    )
    events: list[ModelEvent] = []
    with pytest.raises(ProviderError, match="died mid-stream"):
        async for event in chain.stream(_req()):
            events.append(event)
    # The partial token surfaced before the break; no failover after first token.
    assert [e.text for e in events] == ["partial"]


@pytest.mark.asyncio
async def test_failover_stream_all_fail_raises() -> None:
    chain = HealthAwareFailover(
        [(FlakyProvider(fail=True), None)], guard_capabilities=False
    )
    with pytest.raises(ProviderUnavailableError, match="all providers failed"):
        async for _ in chain.stream(_req()):
            pass


@pytest.mark.asyncio
async def test_failover_stream_lifecycle_classified() -> None:
    retired = FlakyProvider(
        fail=True, error=ProviderError("model has been deprecated", provider="flaky")
    )
    chain = HealthAwareFailover([(retired, None)], guard_capabilities=False)
    with pytest.raises(ProviderUnavailableError, match="rotate now"):
        async for _ in chain.stream(_req()):
            pass


def test_failover_capabilities_delegates_to_first_entry() -> None:
    class CapProvider(ModelProvider):
        name = "cap"

        async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
            return ModelResponse(text="x", model=request.model)

        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(audio=True)

    chain = HealthAwareFailover([(CapProvider(), None)], guard_capabilities=False)
    assert chain.capabilities("any").audio is True


@pytest.mark.asyncio
async def test_failover_list_models_merges_entries() -> None:
    class P1(ModelProvider):
        name = "p1"

        async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
            return ModelResponse(text="x", model=request.model)

        async def list_models(self) -> list[ModelProfile]:
            return [ModelProfile(name="a", provider="p1", model="a")]

    class P2(ModelProvider):
        name = "p2"

        async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
            return ModelResponse(text="x", model=request.model)

        async def list_models(self) -> list[ModelProfile]:
            return [ModelProfile(name="b", provider="p2", model="b")]

    chain = HealthAwareFailover([(P1(), None), (P2(), None)], guard_capabilities=False)
    models = await chain.list_models()
    assert {m.model for m in models} == {"a", "b"}


@pytest.mark.asyncio
async def test_failover_aclose_closes_all_entries() -> None:
    closed: list[str] = []

    class Closer(ModelProvider):
        def __init__(self, tag: str) -> None:
            self.name = tag

        async def generate(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
            return ModelResponse(text="x", model=request.model)

        async def aclose(self) -> None:
            closed.append(self.name)

    chain = HealthAwareFailover(
        [(Closer("a"), None), (Closer("b"), None)], guard_capabilities=False
    )
    await chain.aclose()
    assert closed == ["a", "b"]


@pytest.mark.asyncio
async def test_failover_default_registry_when_guarding() -> None:
    # guard_capabilities=True with no registry triggers the default-registry
    # lazy load (_reg branch). MockProvider serves a registry-known model.
    chain = HealthAwareFailover([(MockProvider(), None)])
    resp = await chain.generate(_req(model="gpt-5.2"))
    assert resp.provider == "mock"
    # Registry was lazily materialized.
    assert chain._registry is not None


@pytest.mark.asyncio
async def test_failover_guard_skips_retired_model_only() -> None:
    # A registry-retired model with no attemptable entry yields a retire-now error.
    from vincio.providers.registry import ModelRegistry

    registry = ModelRegistry(
        [ModelProfile(name="old", provider="mock", model="old-x", retirement_date="2020-01-01")]
    )
    chain = HealthAwareFailover(
        [(MockProvider(), "old-x")], guard_capabilities=True, registry=registry
    )
    with pytest.raises(ModelRetiredError, match="rotate now"):
        await chain.generate(_req(model="whatever"))
