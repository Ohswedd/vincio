"""Circuit breaking and health-aware failover (1.3).

A :class:`CircuitBreaker` wraps any :class:`~vincio.providers.base.ModelProvider`
and tracks its recent outcomes (errors and slow calls) over a rolling window.
When the unhealthy fraction crosses a threshold it *opens*: calls fail fast with
a non-retryable :class:`~vincio.core.errors.CircuitOpenError` instead of waiting
on a request that is expected to fail. After a cooldown it goes *half-open* and
admits a few probes; a success closes it, a failure re-opens it.

:class:`HealthAwareFailover` is a failover chain that consults the breakers on
its entries and tries healthy providers first, so a systemic provider outage is
routed around in microseconds rather than one slow timeout per request.

The documented reliability pattern, made explicit: **retries** for transient
errors (``RetryingProvider``), **fallback** for persistent ones
(``FailoverChain`` / ``HealthAwareFailover``), and **circuit-breaking** for
systemic ones (``CircuitBreaker``). Compose them inner-to-outer as
``CircuitBreaker(RetryingProvider(provider))`` so retries absorb transient
blips while the breaker counts the failures that survive them.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Any

from ..core.errors import CircuitOpenError, ProviderError
from ..core.types import ModelCapabilities, ModelEvent, ModelProfile, ModelRequest, ModelResponse
from .base import (
    ModelProvider,
    _merge_model_lists,
    failover_failure,
    is_lifecycle_error,
    screen_entries,
)

__all__ = ["CircuitState", "CircuitBreaker", "HealthAwareFailover"]


class CircuitState(StrEnum):
    CLOSED = "closed"  # healthy: calls pass through
    OPEN = "open"  # tripped: calls fail fast until cooldown elapses
    HALF_OPEN = "half_open"  # probing: a limited number of calls are admitted


class CircuitBreaker(ModelProvider):
    """Per-provider circuit breaker with half-open probing.

    A call is counted *unhealthy* if it raises a :class:`ProviderError` or, when
    ``latency_threshold_ms`` is set, exceeds it. The breaker trips once the
    window holds at least ``min_calls`` samples and the unhealthy fraction
    reaches ``failure_threshold``. State changes raise ``circuit.opened`` /
    ``circuit.half_open`` / ``circuit.closed`` on the event bus when one is
    provided.
    """

    def __init__(
        self,
        inner: ModelProvider,
        *,
        failure_threshold: float = 0.5,
        min_calls: int = 5,
        window: int = 20,
        latency_threshold_ms: float | None = None,
        cooldown_s: float = 30.0,
        half_open_max: int = 1,
        events: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < failure_threshold <= 1.0:
            raise ValueError("failure_threshold must be in (0, 1]")
        self.inner = inner
        self.name = inner.name
        self.failure_threshold = failure_threshold
        self.min_calls = max(1, min_calls)
        self.window = max(self.min_calls, window)
        self.latency_threshold_ms = latency_threshold_ms
        self.cooldown_s = cooldown_s
        self.half_open_max = max(1, half_open_max)
        self._events = events
        self._clock = clock
        self._outcomes: deque[bool] = deque(maxlen=self.window)  # True == unhealthy
        self.state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._half_open_calls = 0
        self.trip_count = 0

    # -- state ---------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """True when the breaker would admit a call right now (closed, probing,
        or open-but-cooled-down)."""
        return self._effective_state() is not CircuitState.OPEN

    def failure_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self._effective_state().value,
            "failure_rate": round(self.failure_rate(), 4),
            "samples": len(self._outcomes),
            "trips": self.trip_count,
        }

    def _effective_state(self) -> CircuitState:
        if self.state is CircuitState.OPEN and (self._clock() - self._opened_at) >= self.cooldown_s:
            return CircuitState.HALF_OPEN
        return self.state

    def _emit(self, name: str) -> None:
        if self._events is not None:
            self._events.emit(name, {"provider": self.name, **self.snapshot()})

    def _transition(self, state: CircuitState) -> None:
        if state is self.state:
            return
        self.state = state
        if state is CircuitState.OPEN:
            self._opened_at = self._clock()
            self.trip_count += 1
            self._half_open_calls = 0
            self._emit("circuit.opened")
        elif state is CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._emit("circuit.half_open")
        else:  # CLOSED
            self._outcomes.clear()
            self._half_open_calls = 0
            self._emit("circuit.closed")

    def _before_call(self) -> None:
        """Admit or fast-fail; raises :class:`CircuitOpenError` when open."""
        effective = self._effective_state()
        if effective is CircuitState.HALF_OPEN:
            if self.state is CircuitState.OPEN:
                self._transition(CircuitState.HALF_OPEN)
            if self._half_open_calls >= self.half_open_max:
                raise CircuitOpenError(
                    f"circuit half-open for {self.name!r}: probe limit reached",
                    provider=self.name,
                )
            self._half_open_calls += 1
        elif effective is CircuitState.OPEN:
            raise CircuitOpenError(
                f"circuit open for provider {self.name!r} "
                f"(failure_rate={self.failure_rate():.2f})",
                provider=self.name,
            )

    def _abort_probe(self) -> None:
        """Release a half-open probe slot claimed by a call that ended in a
        non-provider error (cancellation, abort) — it is not a health signal,
        so it must not consume the limited probe budget."""
        if self.state is CircuitState.HALF_OPEN and self._half_open_calls > 0:
            self._half_open_calls -= 1

    def _record(self, *, unhealthy: bool) -> None:
        if self.state is CircuitState.HALF_OPEN:
            # A probe decides the outcome immediately.
            self._transition(CircuitState.OPEN if unhealthy else CircuitState.CLOSED)
            return
        self._outcomes.append(unhealthy)
        if (
            self.state is CircuitState.CLOSED
            and len(self._outcomes) >= self.min_calls
            and self.failure_rate() >= self.failure_threshold
        ):
            self._transition(CircuitState.OPEN)

    # -- model provider interface -------------------------------------------

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self._before_call()
        started = self._clock()
        try:
            response = await self.inner.generate(request)
        except ProviderError:
            self._record(unhealthy=True)
            raise
        except BaseException:
            # Cancellation/abort is not a provider-health signal; free the probe.
            self._abort_probe()
            raise
        elapsed_ms = (self._clock() - started) * 1000
        slow = self.latency_threshold_ms is not None and elapsed_ms > self.latency_threshold_ms
        self._record(unhealthy=slow)
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self._before_call()
        started = self._clock()
        yielded = False
        try:
            async for event in self.inner.stream(request):
                yielded = True
                yield event
        except ProviderError:
            # Only pre-first-token failures reflect on provider health; a
            # mid-stream break is the run's problem, not a systemic outage.
            self._record(unhealthy=not yielded)
            raise
        except BaseException:
            self._abort_probe()
            raise
        elapsed_ms = (self._clock() - started) * 1000
        slow = self.latency_threshold_ms is not None and elapsed_ms > self.latency_threshold_ms
        self._record(unhealthy=slow)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.inner.capabilities(model)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.inner.embed(texts, model)

    async def list_models(self) -> list[ModelProfile]:
        return await self.inner.list_models()

    async def aclose(self) -> None:
        await self.inner.aclose()


class HealthAwareFailover(ModelProvider):
    """Failover chain that tries healthy providers first.

    Entries are ``(provider, model_override)`` tuples, exactly like
    :class:`~vincio.providers.base.FailoverChain`; wrap each provider in a
    :class:`CircuitBreaker` to get health awareness. On each call the chain
    orders entries by breaker state (closed, then half-open, then open) while
    preserving the original order within a tier, so a tripped provider is
    skipped in microseconds and only retried once everything healthier has
    failed.
    """

    name = "health_aware_failover"

    def __init__(
        self,
        entries: list[tuple[ModelProvider, str | None]],
        *,
        guard_capabilities: bool = True,
        registry: Any | None = None,
    ) -> None:
        if not entries:
            raise ValueError("HealthAwareFailover requires at least one provider")
        self.entries = entries
        self.guard_capabilities = guard_capabilities
        self._registry = registry

    def _reg(self) -> Any:
        if self._registry is None:
            from .registry import default_model_registry

            self._registry = default_model_registry()
        return self._registry

    @staticmethod
    def _rank(provider: ModelProvider) -> int:
        state = getattr(provider, "_effective_state", None)
        if state is None:
            return 0
        return {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}[state()]

    def _ordered(self) -> list[tuple[int, tuple[ModelProvider, str | None]]]:
        # Stable sort keeps original priority within each health tier. Returns
        # ``(original_index, (provider, model_override))`` pairs.
        return sorted(enumerate(self.entries), key=lambda iv: (self._rank(iv[1][0]), iv[0]))

    def _screen(
        self, request: ModelRequest
    ) -> tuple[list[tuple[ModelProvider, str | None, str]], list[str], list[str]]:
        registry = self._reg() if self.guard_capabilities else None
        ordered = [entry for _, entry in self._ordered()]
        return screen_entries(
            ordered, request, guard=self.guard_capabilities, registry=registry
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        attemptable, lifecycle, incapable = self._screen(request)
        attempt_errors: list[str] = []
        for provider, model_override, model in attemptable:
            attempt = request if not model_override else request.model_copy(update={"model": model_override})
            try:
                return await provider.generate(attempt)
            except ProviderError as exc:
                if is_lifecycle_error(exc):
                    lifecycle.append(f"{provider.name}/{model}: {exc.message}")
                else:
                    attempt_errors.append(f"{provider.name}: {exc.message}")
        raise failover_failure(self.name, len(attemptable), attempt_errors, lifecycle, incapable)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        attemptable, lifecycle, incapable = self._screen(request)
        attempt_errors: list[str] = []
        for provider, model_override, model in attemptable:
            attempt = request if not model_override else request.model_copy(update={"model": model_override})
            yielded = False
            try:
                async for event in provider.stream(attempt):
                    yielded = True
                    yield event
                return
            except ProviderError as exc:
                if yielded:
                    raise
                if is_lifecycle_error(exc):
                    lifecycle.append(f"{provider.name}/{model}: {exc.message}")
                else:
                    attempt_errors.append(f"{provider.name}: {exc.message}")
        raise failover_failure(self.name, len(attemptable), attempt_errors, lifecycle, incapable)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.entries[0][0].capabilities(model)

    async def list_models(self) -> list[ModelProfile]:
        return _merge_model_lists([await p.list_models() for p, _ in self.entries])

    async def aclose(self) -> None:
        for provider, _ in self.entries:
            await provider.aclose()
