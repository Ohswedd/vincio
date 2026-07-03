"""Open-web search backends: DuckDuckGo by default, pluggable by contract.

The search layer is deliberately thin and honest about what it is: an HTTP
client for a public results page plus a tolerant, dependency-free parser. It
returns typed :class:`SearchResult` rows — never raw HTML — so everything
downstream (tools, connectors, the research agent) consumes one stable shape
regardless of which engine produced it.

* :class:`DuckDuckGoBackend` — the default engine. Queries the keyless
  ``html.duckduckgo.com/html`` endpoint (falling back to
  ``lite.duckduckgo.com/lite``), decodes the ``uddg=`` redirect wrappers to the
  real target URLs, drops ads and internal links, and detects the
  rate-limit/anomaly challenge page as a typed
  :class:`~vincio.core.errors.WebSearchError` instead of silently returning
  nothing.
* :class:`StaticSearchBackend` — a deterministic in-memory engine for tests,
  benchmarks, and air-gapped runs; the same contract, zero network.
* :class:`SearchBackend` — the :class:`~typing.Protocol` any third-party engine
  (SearXNG, Brave, an intranet index) implements to plug in.

An injected ``httpx.AsyncClient`` (e.g. one built on ``httpx.MockTransport``)
makes the live backend fully testable offline, the same pattern the providers
use.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, quote_plus, urlsplit

import httpx
from pydantic import BaseModel

from ..core.errors import WebSearchError

__all__ = [
    "SearchBackend",
    "SearchResult",
    "DuckDuckGoBackend",
    "StaticSearchBackend",
]

#: Browser-like default agent: the public endpoints serve the plain HTML page
#: (not the JS app) to any client, but reject empty/robot-default agents.
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; VincioWeb/1.0; +https://github.com/Ohswedd/vincio)"

_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

# Markers of the rate-limit / bot-challenge interstitial (no results markup).
_BLOCKED_RE = re.compile(
    r"anomaly-modal|detected an anomaly|challenge-form|robot|unusual traffic", re.IGNORECASE
)


class SearchResult(BaseModel):
    """One search hit: the stable row every backend returns."""

    rank: int
    title: str
    url: str
    snippet: str = ""
    source: str = "duckduckgo"


@runtime_checkable
class SearchBackend(Protocol):
    """The contract a pluggable search engine implements."""

    name: str

    async def search(
        self, query: str, *, max_results: int = 8
    ) -> list[SearchResult]:  # pragma: no cover - protocol
        ...


def _decode_result_href(href: str) -> str | None:
    """The real target URL behind a results-page anchor, or ``None``.

    DuckDuckGo wraps every organic result as ``//duckduckgo.com/l/?uddg=<url>``;
    ads route through ``y.js`` and internal links stay on the engine's host —
    both are dropped.
    """
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    host = parts.netloc.lower()
    if host.endswith("duckduckgo.com"):
        if "y.js" in parts.path:  # ad redirector
            return None
        target = parse_qs(parts.query).get("uddg", [""])[0]
        if target.startswith(("http://", "https://")):
            return target
        return None
    if parts.scheme in ("http", "https") and host:
        return href
    return None


class _ResultsParser(HTMLParser):
    """Tolerant parser for both DuckDuckGo results layouts.

    Anchors whose href decodes to an external URL become results (anchor text =
    title); the class-tagged snippet element that follows attaches to the most
    recent result. Layout drift degrades to fewer fields, never to a crash.
    """

    _SNIPPET_CLASSES = ("result__snippet", "result-snippet")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._title_parts: list[str] | None = None
        self._snippet_parts: list[str] | None = None
        self._snippet_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        classes = attr_map.get("class", "")
        if self._snippet_parts is not None:
            self._snippet_depth += 1
        elif any(marker in classes for marker in self._SNIPPET_CLASSES):
            self._snippet_parts = []
            self._snippet_depth = 1
            return
        if tag == "a" and self._title_parts is None:
            url = _decode_result_href(attr_map.get("href", ""))
            if url is not None and not any(marker in classes for marker in self._SNIPPET_CLASSES):
                self.results.append({"url": url, "title": "", "snippet": ""})
                self._title_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._snippet_parts is not None:
            self._snippet_depth -= 1
            if self._snippet_depth <= 0:
                if self.results:
                    text = " ".join(" ".join(self._snippet_parts).split())
                    self.results[-1]["snippet"] = text
                self._snippet_parts = None
        if tag == "a" and self._title_parts is not None:
            if self.results:
                self.results[-1]["title"] = " ".join(" ".join(self._title_parts).split())
            self._title_parts = None

    def handle_data(self, data: str) -> None:
        if self._title_parts is not None:
            self._title_parts.append(data)
        elif self._snippet_parts is not None:
            self._snippet_parts.append(data)


def parse_results_html(html: str, *, max_results: int = 8) -> list[SearchResult]:
    """Parse a DuckDuckGo results page into deduplicated :class:`SearchResult` rows."""
    parser = _ResultsParser()
    parser.feed(html)
    parser.close()
    seen: set[str] = set()
    results: list[SearchResult] = []
    for row in parser.results:
        url, title = row["url"], row["title"]
        if not title or url in seen:
            continue
        seen.add(url)
        results.append(
            SearchResult(rank=len(results) + 1, title=title, url=url, snippet=row["snippet"])
        )
        if len(results) >= max_results:
            break
    return results


class DuckDuckGoBackend:
    """DuckDuckGo over the keyless HTML endpoints.

    Region pins results to a locale (``kl``, e.g. ``us-en``); ``recency``
    narrows to the last day/week/month/year (``d``/``w``/``m``/``y``). An
    injected ``client`` (e.g. over ``httpx.MockTransport``) runs the whole
    backend offline; without one, a client is created per call.
    """

    name = "duckduckgo"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 15.0,
        user_agent: str = DEFAULT_USER_AGENT,
        region: str | None = None,
        endpoint: str = _HTML_ENDPOINT,
        fallback_endpoint: str | None = _LITE_ENDPOINT,
    ) -> None:
        self.client = client
        self.timeout = timeout
        self.user_agent = user_agent
        self.region = region
        self.endpoint = endpoint
        self.fallback_endpoint = fallback_endpoint

    def _url(self, endpoint: str, query: str, recency: str | None) -> str:
        params = [f"q={quote_plus(query)}"]
        if self.region:
            params.append(f"kl={quote_plus(self.region)}")
        if recency:
            params.append(f"df={quote_plus(recency)}")
        return endpoint + "?" + "&".join(params)

    async def _get(self, client: httpx.AsyncClient, url: str) -> str:
        try:
            response = await client.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
        except httpx.HTTPError as exc:
            raise WebSearchError(
                f"search endpoint unreachable: {exc}", details={"url": url}
            ) from exc
        if response.status_code >= 400:
            raise WebSearchError(
                f"search endpoint returned HTTP {response.status_code}",
                details={"url": url, "status": response.status_code},
            )
        return response.text

    async def search(
        self, query: str, *, max_results: int = 8, recency: str | None = None
    ) -> list[SearchResult]:
        """Run one query and return up to *max_results* organic hits."""
        if not query.strip():
            raise WebSearchError("empty search query")
        endpoints = [self.endpoint]
        if self.fallback_endpoint and self.fallback_endpoint != self.endpoint:
            endpoints.append(self.fallback_endpoint)
        blocked: list[str] = []
        async with _managed_client(self.client) as client:
            for endpoint in endpoints:
                html = await self._get(client, self._url(endpoint, query, recency))
                results = parse_results_html(html, max_results=max_results)
                if results:
                    return results
                if _BLOCKED_RE.search(html):
                    blocked.append(endpoint)
                    continue
                # A parseable page with zero hits is a valid empty answer.
                return []
        raise WebSearchError(
            "search blocked by the engine's rate-limit/anomaly challenge",
            details={"query": query, "endpoints": blocked},
        )


class StaticSearchBackend:
    """Deterministic in-memory results: tests, benchmarks, air-gapped runs.

    ``results`` maps a query to its rows; unknown queries return the ``default``
    rows (empty by default), so a scripted session never touches the network.
    """

    name = "static"

    def __init__(
        self,
        results: dict[str, list[SearchResult]] | None = None,
        *,
        default: list[SearchResult] | None = None,
    ) -> None:
        self.results = dict(results or {})
        self.default = list(default or [])
        self.queries: list[str] = []

    async def search(self, query: str, *, max_results: int = 8) -> list[SearchResult]:
        self.queries.append(query)
        rows = self.results.get(query, self.default)
        return [row.model_copy(update={"rank": i + 1}) for i, row in enumerate(rows[:max_results])]


class _managed_client:
    """Use the injected client (kept open) or create one for the call."""

    def __init__(self, client: httpx.AsyncClient | None, **kwargs: Any) -> None:
        self._injected = client
        self._own: httpx.AsyncClient | None = None
        self._kwargs = kwargs

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._injected is not None:
            return self._injected
        self._own = httpx.AsyncClient(**self._kwargs)
        return self._own

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._own is not None:
            await self._own.aclose()
