"""Provider abstraction.

Providers translate the provider-neutral :class:`ModelRequest` into vendor
APIs and back. All providers are async-first; ``generate_sync`` is provided
for synchronous callers. Retry/backoff and failover are implemented once,
provider-neutrally, in :class:`RetryingProvider` and :class:`FailoverChain`.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from ..core.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ..core.types import (
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
)

__all__ = [
    "ModelProvider",
    "HTTPProvider",
    "RetryingProvider",
    "FailoverChain",
    "ProviderRegistry",
    "parse_sse_lines",
    "run_sync",
]


def run_sync(coro):
    """Run *coro* to completion from sync code, inside or outside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # We're inside a running loop (e.g. Jupyter): execute in a fresh thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class ModelProvider(ABC):
    """Abstract model provider."""

    name: str = "base"

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute a single model call."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Stream a model call. Default: emulate streaming from generate()."""
        response = await self.generate(request)
        if response.text:
            yield ModelEvent(type="text_delta", text=response.text)
        for tool_call in response.tool_calls:
            yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        yield ModelEvent(type="usage", usage=response.usage)
        yield ModelEvent(type="done", response=response)

    def capabilities(self, model: str) -> ModelCapabilities:
        """Capability matrix for *model*."""
        return ModelCapabilities()

    def generate_sync(self, request: ModelRequest) -> ModelResponse:
        return run_sync(self.generate(request))

    def stream_sync(self, request: ModelRequest) -> Iterator[ModelEvent]:
        """Synchronous streaming: collects events from the async iterator."""

        async def collect() -> list[ModelEvent]:
            return [event async for event in self.stream(request)]

        yield from run_sync(collect())

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed texts. Providers that support embeddings override this."""
        raise ProviderError(
            f"provider {self.name!r} does not support embeddings", provider=self.name
        )

    async def aclose(self) -> None:
        """Release underlying resources (HTTP clients)."""
        return None


class HTTPProvider(ModelProvider):
    """Shared httpx plumbing for HTTP API providers."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    default_base_url: str = ""
    requires_api_key: bool = True

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    def _check_key(self) -> None:
        if self.requires_api_key and not self.api_key:
            raise ProviderAuthError(
                f"missing API key for provider {self.name!r}", provider=self.name
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = {"raw": response.text[:2000]}
        message = json.dumps(body)[:2000]
        kw: dict[str, Any] = {"provider": self.name, "details": {"status": response.status_code}}
        if response.status_code in (401, 403):
            raise ProviderAuthError(f"authentication failed: {message}", **kw)
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimitError(
                f"rate limited: {message}",
                retry_after_s=float(retry_after) if retry_after else None,
                **kw,
            )
        if response.status_code in (408, 504):
            raise ProviderTimeoutError(f"timeout: {message}", **kw)
        if response.status_code >= 500 or response.status_code == 529:
            raise ProviderUnavailableError(f"provider unavailable: {message}", **kw)
        raise ProviderResponseError(f"provider error {response.status_code}: {message}", **kw)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._check_key()
        try:
            response = await self.client.post(
                f"{self.base_url}{path}", json=payload, headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc), provider=self.name) from exc
        self._raise_for_status(response)
        return response.json()

    async def _post_stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[str]:
        self._check_key()
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}{path}", json=payload, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_status(response)
                async for line in response.aiter_lines():
                    yield line
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(str(exc), provider=self.name) from exc


def parse_sse_lines(line: str) -> str | None:
    """Extract the data payload from an SSE line; returns None for non-data lines."""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data:"):
        return line[len("data:") :].strip()
    return None


class RetryingProvider(ModelProvider):
    """Wraps a provider with bounded exponential backoff on retryable errors."""

    def __init__(
        self,
        inner: ModelProvider,
        *,
        max_retries: int = 2,
        base_delay_s: float = 0.5,
        max_delay_s: float = 20.0,
        jitter: float = 0.2,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.jitter = jitter
        self.retry_count = 0

    def _delay(self, attempt: int, error: ProviderError) -> float:
        if isinstance(error, ProviderRateLimitError) and error.retry_after_s:
            return min(self.max_delay_s, error.retry_after_s)
        delay = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return delay * (1.0 + random.uniform(-self.jitter, self.jitter))

    async def generate(self, request: ModelRequest) -> ModelResponse:
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.inner.generate(request)
            except ProviderError as exc:
                if not exc.retryable or attempt == self.max_retries:
                    raise
                last_error = exc
                self.retry_count += 1
                await asyncio.sleep(self._delay(attempt, exc))
        raise last_error or ProviderError("retries exhausted", provider=self.name)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        # Retry only before first event is yielded; mid-stream errors propagate.
        for attempt in range(self.max_retries + 1):
            yielded = False
            try:
                async for event in self.inner.stream(request):
                    yielded = True
                    yield event
                return
            except ProviderError as exc:
                if yielded or not exc.retryable or attempt == self.max_retries:
                    raise
                self.retry_count += 1
                await asyncio.sleep(self._delay(attempt, exc))

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.inner.capabilities(model)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.inner.embed(texts, model)

    async def aclose(self) -> None:
        await self.inner.aclose()


class FailoverChain(ModelProvider):
    """Provider failover: try providers/models in order.

    Each entry is ``(provider, model_override | None)``. On non-retryable
    auth errors or exhausted retries, the next entry is attempted.
    """

    name = "failover"

    def __init__(self, entries: list[tuple[ModelProvider, str | None]]) -> None:
        if not entries:
            raise ConfigError("FailoverChain requires at least one provider")
        self.entries = entries

    async def generate(self, request: ModelRequest) -> ModelResponse:
        errors: list[str] = []
        for provider, model_override in self.entries:
            attempt_request = request
            if model_override:
                attempt_request = request.model_copy(update={"model": model_override})
            try:
                return await provider.generate(attempt_request)
            except ProviderError as exc:
                errors.append(f"{provider.name}: {exc.message}")
        raise ProviderUnavailableError(
            "all providers failed: " + " | ".join(errors), provider=self.name, retryable=False
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        errors: list[str] = []
        for provider, model_override in self.entries:
            attempt_request = request
            if model_override:
                attempt_request = request.model_copy(update={"model": model_override})
            yielded = False
            try:
                async for event in provider.stream(attempt_request):
                    yielded = True
                    yield event
                return
            except ProviderError as exc:
                if yielded:
                    raise
                errors.append(f"{provider.name}: {exc.message}")
        raise ProviderUnavailableError(
            "all providers failed: " + " | ".join(errors), provider=self.name, retryable=False
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.entries[0][0].capabilities(model)

    async def aclose(self) -> None:
        for provider, _ in self.entries:
            await provider.aclose()


class ProviderRegistry:
    """Named provider construction + instance cache."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}
        self._instances: dict[str, ModelProvider] = {}

    def register(self, name: str, factory: Any) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> ModelProvider:
        if name not in self._factories:
            raise ConfigError(
                f"unknown provider {name!r}; known: {sorted(self._factories)}"
            )
        return self._factories[name](**kwargs)

    def get_or_create(self, name: str, **kwargs: Any) -> ModelProvider:
        key = f"{name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        if key not in self._instances:
            self._instances[key] = self.create(name, **kwargs)
        return self._instances[key]

    @property
    def names(self) -> list[str]:
        return sorted(self._factories)


def measure_latency_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)
