"""Health-aware key/region pooling with rate-limit queueing.

:class:`KeyPool` spreads load across several API keys or regions of the same
provider. It round-robins **health-aware** (skipping keys whose circuit breaker
is open), enforces per-key **dual RPM + TPM** token buckets so a free-tier limit
self-heals instead of erroring, and applies **full-jitter** exponential backoff
when a key returns 429 — honoring any ``retry_after`` the provider reports.

It implements the :class:`~vincio.providers.base.ModelProvider` interface, so it
drops in wherever a single provider would: the compiler, evals, guardrails, and
cost tracking are unchanged.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from ..core.errors import ProviderError, ProviderRateLimitError, ProviderUnavailableError
from ..core.tokens import count_tokens
from ..core.types import ModelCapabilities, ModelEvent, ModelRequest, ModelResponse
from .base import ModelProvider
from .circuit import CircuitBreaker

__all__ = ["RateLimiter", "KeyPool"]


class RateLimiter:
    """A dual requests-per-minute / tokens-per-minute token bucket.

    Buckets refill continuously. :meth:`acquire` awaits until one request slot
    (and, when ``tpm`` is set, ``tokens`` token slots) are free, then debits
    them — so concurrent callers queue instead of overrunning the limit.
    """

    def __init__(
        self,
        *,
        rpm: float | None = None,
        tpm: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._clock = clock
        self._req = float(rpm) if rpm else 0.0
        self._tok = float(tpm) if tpm else 0.0
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        dt = max(0.0, now - self._last)
        self._last = now
        if self.rpm:
            self._req = min(float(self.rpm), self._req + dt * self.rpm / 60.0)
        if self.tpm:
            self._tok = min(float(self.tpm), self._tok + dt * self.tpm / 60.0)

    def wait_time(self, tokens: int = 0) -> float:
        """Seconds until one request and ``tokens`` tokens would be available."""
        self._refill()
        need = 0.0
        if self.rpm and self._req < 1.0:
            need = max(need, (1.0 - self._req) * 60.0 / self.rpm)
        if self.tpm and tokens and self._tok < tokens:
            need = max(need, (tokens - self._tok) * 60.0 / self.tpm)
        return need

    def available(self, tokens: int = 0) -> bool:
        return self.wait_time(tokens) <= 0.0

    async def acquire(self, tokens: int = 0) -> None:
        while True:
            # Hold the lock across refill→check→debit so concurrent callers
            # cannot both claim the same slot and drive a bucket negative.
            async with self._lock:
                need = self.wait_time(tokens)
                if need <= 0.0:
                    if self.rpm:
                        self._req -= 1.0
                    if self.tpm:
                        self._tok -= tokens
                    return
            await asyncio.sleep(need)


@dataclass(eq=False)
class _Key:
    provider: ModelProvider
    limiter: RateLimiter
    label: str


class KeyPool(ModelProvider):
    """Round-robin pool over multiple keys/regions of one logical provider."""

    name = "key_pool"

    def __init__(
        self,
        providers: list[ModelProvider],
        *,
        rpm: float | None = None,
        tpm: float | None = None,
        breaker: bool = True,
        labels: list[str] | None = None,
        max_attempts: int | None = None,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
        seed: int | None = None,
        events: object | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("KeyPool requires at least one provider")
        self._clock = clock
        self.keys: list[_Key] = []
        for i, provider in enumerate(providers):
            wrapped = (
                CircuitBreaker(provider, events=events, clock=clock) if breaker else provider
            )
            label = labels[i] if labels and i < len(labels) else f"{provider.name}#{i}"
            self.keys.append(_Key(wrapped, RateLimiter(rpm=rpm, tpm=tpm, clock=clock), label))
        self.name = f"pool[{providers[0].name}x{len(providers)}]"
        self.max_attempts = max_attempts or max(len(self.keys) * 3, 3)
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self._rng = random.Random(seed)
        self._cursor = 0
        self.dispatch_count = 0

    # -- selection -----------------------------------------------------------

    def _healthy_keys(self) -> list[_Key]:
        ordered = [self.keys[(self._cursor + i) % len(self.keys)] for i in range(len(self.keys))]
        return [k for k in ordered if getattr(k.provider, "healthy", True)]

    def _select(self, tokens: int) -> _Key | None:
        """Pick the next healthy key, preferring one with immediate rate budget."""
        healthy = self._healthy_keys()
        if not healthy:
            return None
        ready = [k for k in healthy if k.limiter.available(tokens)]
        chosen = ready[0] if ready else min(healthy, key=lambda k: k.limiter.wait_time(tokens))
        # Advance the round-robin cursor past the chosen key.
        self._cursor = (self.keys.index(chosen) + 1) % len(self.keys)
        return chosen

    def _backoff(self, attempt: int, retry_after_s: float | None) -> float:
        capped = min(self.max_backoff_s, self.base_backoff_s * (2**attempt))
        jittered = self._rng.uniform(0.0, capped)  # full jitter
        if retry_after_s:
            return min(self.max_backoff_s, max(retry_after_s, jittered))
        return jittered

    @staticmethod
    def _estimate_tokens(request: ModelRequest) -> int:
        text = "\n".join(m.text for m in request.messages)
        return count_tokens(text) + (request.max_output_tokens or 0)

    # -- model provider interface -------------------------------------------

    async def generate(self, request: ModelRequest) -> ModelResponse:
        tokens = self._estimate_tokens(request)
        errors: list[str] = []
        for attempt in range(self.max_attempts):
            key = self._select(tokens)
            if key is None:
                # Every key is circuit-open: wait a jittered beat for cooldown.
                await asyncio.sleep(self._backoff(attempt, None))
                continue
            await key.limiter.acquire(tokens)
            try:
                self.dispatch_count += 1
                return await key.provider.generate(request)
            except ProviderRateLimitError as exc:
                errors.append(f"{key.label}: {exc.message}")
                await asyncio.sleep(self._backoff(attempt, exc.retry_after_s))
            except ProviderError as exc:
                errors.append(f"{key.label}: {exc.message}")
        raise ProviderUnavailableError(
            f"key pool exhausted after {self.max_attempts} attempts: " + " | ".join(errors[-4:]),
            provider=self.name,
            retryable=False,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        tokens = self._estimate_tokens(request)
        key: _Key | None = None
        for attempt in range(self.max_attempts):
            key = self._select(tokens)
            if key is not None:
                break
            # Every key is circuit-open: wait a jittered beat rather than
            # streaming from a known-open breaker.
            await asyncio.sleep(self._backoff(attempt, None))
        if key is None:
            raise ProviderUnavailableError(
                "key pool exhausted: all keys unhealthy", provider=self.name, retryable=False
            )
        await key.limiter.acquire(tokens)
        self.dispatch_count += 1
        async for event in key.provider.stream(request):
            yield event

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        key = self._select(sum(count_tokens(t) for t in texts)) or self.keys[0]
        return await key.provider.embed(texts, model)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.keys[0].provider.capabilities(model)

    async def list_models(self):  # type: ignore[override]
        # All keys front the same provider/region set; the first is representative.
        return await self.keys[0].provider.list_models()

    def snapshot(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for key in self.keys:
            entry: dict[str, object] = {"label": key.label}
            snap = getattr(key.provider, "snapshot", None)
            if callable(snap):
                entry.update(snap())
            out.append(entry)
        return out

    async def aclose(self) -> None:
        for key in self.keys:
            await key.provider.aclose()
