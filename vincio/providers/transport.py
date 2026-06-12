"""Throughput primitives for the provider transport (0.2).

- :class:`CoalescingProvider` — in-flight request coalescing: concurrent,
  identical ``generate`` calls share a single provider call instead of
  hitting the API N times.
- :func:`build_pooled_client` — a shared, connection-pooled
  ``httpx.AsyncClient`` for providers that accept an external client.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from ..core.types import ModelCapabilities, ModelEvent, ModelRequest, ModelResponse
from .base import ModelProvider

__all__ = ["CoalescingProvider", "build_pooled_client"]


def build_pooled_client(
    *,
    timeout_s: float = 120.0,
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
) -> httpx.AsyncClient:
    """A connection-pooled HTTP client suitable for sharing across providers."""
    return httpx.AsyncClient(
        timeout=timeout_s,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
    )


class CoalescingProvider(ModelProvider):
    """Deduplicates identical in-flight ``generate`` calls.

    Keyed by the full request hash, so only byte-identical requests coalesce.
    The first caller executes; concurrent identical callers await the same
    future and receive an independent copy of the response. Streaming and
    embedding pass straight through — only complete generations coalesce.
    """

    def __init__(self, inner: ModelProvider) -> None:
        self.inner = inner
        self.name = inner.name
        self.coalesced_count = 0
        self._in_flight: dict[str, asyncio.Future[ModelResponse]] = {}

    async def generate(self, request: ModelRequest) -> ModelResponse:
        key = request.hash
        existing = self._in_flight.get(key)
        if existing is not None:
            self.coalesced_count += 1
            response = await asyncio.shield(existing)
            return response.model_copy(deep=True)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ModelResponse] = loop.create_future()
        self._in_flight[key] = future
        try:
            response = await self.inner.generate(request)
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                # Followers consume the exception; avoid "never retrieved".
                future.exception()
            raise
        else:
            if not future.done():
                future.set_result(response)
            return response
        finally:
            self._in_flight.pop(key, None)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        async for event in self.inner.stream(request):
            yield event

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.inner.embed(texts, model)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.inner.capabilities(model)

    async def aclose(self) -> None:
        await self.inner.aclose()
