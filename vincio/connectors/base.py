"""Connector protocol, registry, and factory."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import httpx

from ..core.errors import ConfigError
from ..core.types import Document

__all__ = [
    "Connector",
    "CONNECTORS",
    "register_connector",
    "connect",
    "managed_client",
    "row_text",
    "sampled_rows",
]


@runtime_checkable
class Connector(Protocol):
    name: str

    async def load(self) -> list[Document]:  # pragma: no cover
        ...


CONNECTORS: dict[str, Callable[..., Any]] = {}

# Built-in connectors import lazily so optional dependencies stay optional.
_BUILTIN_MODULES = {
    "web": "vincio.connectors.web",
    "websearch": "vincio.connectors.websearch",
    "github": "vincio.connectors.github",
    "sql": "vincio.connectors.sql",
    "s3": "vincio.connectors.s3",
    "gcs": "vincio.connectors.gcs",
    "notion": "vincio.connectors.notion",
    "confluence": "vincio.connectors.confluence",
    "slack": "vincio.connectors.slack",
    "jira": "vincio.connectors.jira",
    "linear": "vincio.connectors.linear",
    "gdrive": "vincio.connectors.gdrive",
    "sharepoint": "vincio.connectors.sharepoint",
    "salesforce": "vincio.connectors.salesforce",
    "zendesk": "vincio.connectors.zendesk",
    "bigquery": "vincio.connectors.bigquery",
    "snowflake": "vincio.connectors.snowflake",
}


def row_text(row: dict[str, Any], text_columns: list[str] | None = None) -> str:
    """Render a result-set row as ``"column: value"`` lines.

    Shared by the SQL-family connectors (``sql``, ``bigquery``, ``snowflake``).
    With ``text_columns`` unset, every string-valued column is included.
    """
    columns = text_columns or [c for c, v in row.items() if isinstance(v, str)]
    return "\n".join(f"{c}: {row[c]}" for c in columns if row.get(c) is not None)


def sampled_rows(
    rows: Any, *, max_rows: int, sample: int | None = None, seed: int = 0
) -> list[tuple[int, Any]]:
    """Materialize result-set rows as ``(original_index, row)`` pairs.

    With ``sample`` unset, this takes the first ``max_rows`` rows (the legacy
    cutoff). With ``sample`` set, it draws a uniform reservoir sample of that many
    rows from the *entire* result set in a single bounded pass — a representative
    sample stands in for the whole instead of an order-biased prefix. The original
    row index is preserved so downstream document ids stay stable.
    """
    if sample is None:
        out: list[tuple[int, Any]] = []
        for index, row in enumerate(rows):
            if index >= max_rows:
                break
            out.append((index, row))
        return out
    from ..data.sampling import reservoir_sample

    return reservoir_sample(enumerate(rows), sample, seed=seed)


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
        # An installed third-party connector registers via the ``vincio.connectors``
        # entry-point group on first miss.
        from ..plugins import ensure_loaded

        ensure_loaded("vincio.connectors")
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
