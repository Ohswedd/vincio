"""Connector protocol, registry, and factory."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import httpx

from ..core.errors import ConfigError
from ..core.types import Document

__all__ = ["Connector", "CONNECTORS", "register_connector", "connect", "managed_client"]


@runtime_checkable
class Connector(Protocol):
    name: str

    async def load(self) -> list[Document]:  # pragma: no cover
        ...


CONNECTORS: dict[str, Callable[..., Any]] = {}

# Built-in connectors import lazily so optional dependencies stay optional.
_BUILTIN_MODULES = {
    "web": "vincio.connectors.web",
    "github": "vincio.connectors.github",
    "sql": "vincio.connectors.sql",
    "s3": "vincio.connectors.s3",
    "gcs": "vincio.connectors.gcs",
    "notion": "vincio.connectors.notion",
    "confluence": "vincio.connectors.confluence",
    "slack": "vincio.connectors.slack",
}


def register_connector(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a connector factory under ``name`` (plugin extension point)."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        CONNECTORS[name] = factory
        return factory

    return decorator


def connect(kind: str, **options: Any) -> Connector:
    """Instantiate a connector by kind, e.g. ``connect("web", urls=[...])``."""
    if kind not in CONNECTORS and kind in _BUILTIN_MODULES:
        importlib.import_module(_BUILTIN_MODULES[kind])
    if kind not in CONNECTORS:
        known = sorted(set(CONNECTORS) | set(_BUILTIN_MODULES))
        raise ConfigError(f"unknown connector {kind!r}; known: {known}")
    return CONNECTORS[kind](**options)


@asynccontextmanager
async def managed_client(
    client: httpx.AsyncClient | None, **client_kwargs: Any
) -> AsyncIterator[httpx.AsyncClient]:
    """Use the injected client (kept open, enables offline test transports)
    or create one for the duration of the load."""
    if client is not None:
        yield client
        return
    owned = httpx.AsyncClient(**client_kwargs)
    try:
        yield owned
    finally:
        await owned.aclose()
