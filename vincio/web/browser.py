"""The governed browsing session: search, read, remember, prove.

:class:`WebBrowser` is the one object the rest of the platform touches. It
threads every operation through the :class:`~vincio.web.WebPolicy` rails
(pre-egress, typed refusal), keeps the session honest (per-session search and
fetch budgets, duplicate-query suppression), and makes the result *provable*:
every page read is snapshotted, content-hashed, and recorded as a
:class:`WebEvidence` whose excerpts re-derive offline from the snapshot bytes —
the same honesty contract charts and narratives carry.

It also exposes itself to models as two ordinary Vincio tools —

* ``web_search(query, max_results)`` — typed search hits, never raw HTML;
* ``web_read(url, query, budget_tokens)`` — only the passages of one page
  relevant to *query*, packed under a token budget;

registered via :meth:`tool_handlers` on the standard tool registry, so web
access rides the same RBAC, approval, budget, cache, and audit path as every
other tool, for **every** provider. Nothing here is provider-aware; pairing it
with :class:`~vincio.providers.ToolProtocolProvider` extends the same two
tools to models without native function calling.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import BaseModel, Field

from ..core.errors import WebFetchError, WebPolicyError
from ..core.types import ToolSpec
from .extract import PageExcerpt, PageExtract, extract_page
from .policy import WebPolicy
from .search import DuckDuckGoBackend, SearchBackend, SearchResult, _managed_client

__all__ = ["WebBrowser", "WebEvidence", "SearchRecord", "WebSessionReport"]

_TEXTUAL_TYPES = (
    "text/",
    "application/xhtml",
    "application/xml",
    "application/json",
    "application/rss",
    "application/atom",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SearchRecord(BaseModel):
    """One audited search: the query, what came back, and whether it was replayed."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    cached: bool = False
    at: str = ""


class WebEvidence(BaseModel):
    """One page read, content-bound to its snapshot.

    ``content_hash`` is the SHA-256 of the snapshot (the page body's canonical
    UTF-8 decoding), and the excerpts are a pure function of (snapshot, query,
    budget), so :meth:`verify` re-derives the whole record offline from bytes.
    """

    url: str
    title: str = ""
    query: str = ""
    fetched_at: str = ""
    content_hash: str = ""
    budget_tokens: int = 0
    max_excerpts: int = 12
    excerpts: list[PageExcerpt] = Field(default_factory=list)
    excerpt_tokens: int = 0
    page_tokens: int = 0

    def verify(self, snapshot: str | bytes) -> bool:
        """True iff *snapshot* hashes to ``content_hash`` and re-extraction
        reproduces these excerpts exactly."""
        text = snapshot.decode("utf-8", errors="replace") if isinstance(snapshot, bytes) else snapshot
        if _content_digest(text) != self.content_hash:
            return False
        rerun = extract_page(
            text,
            url=self.url,
            query=self.query,
            budget_tokens=self.budget_tokens,
            max_excerpts=self.max_excerpts,
        )
        return [e.model_dump() for e in rerun.excerpts] == [e.model_dump() for e in self.excerpts]


class WebSessionReport(BaseModel):
    """Everything one browsing session did, in order, offline-verifiable."""

    searches: list[SearchRecord] = Field(default_factory=list)
    reads: list[WebEvidence] = Field(default_factory=list)
    searches_used: int = 0
    fetches_used: int = 0

    def verify(self, snapshots: dict[str, str | bytes]) -> bool:
        """True iff every read re-derives from its snapshot (keyed by content hash)."""
        return all(
            evidence.content_hash in snapshots
            and evidence.verify(snapshots[evidence.content_hash])
            for evidence in self.reads
        )


class WebBrowser:
    """A governed, evidence-keeping browsing session.

    ``backend`` defaults to :class:`~vincio.web.DuckDuckGoBackend`; ``client``
    (an ``httpx.AsyncClient``, e.g. over a mock transport) is shared by search
    and fetch so the whole session can run offline in tests. ``audit`` accepts
    the app's hash-chained :class:`~vincio.security.audit.AuditLog` (or any
    object with a compatible ``record``); ``clock`` pins timestamps for
    deterministic replay.
    """

    def __init__(
        self,
        backend: SearchBackend | None = None,
        *,
        policy: WebPolicy | None = None,
        client: httpx.AsyncClient | None = None,
        audit: object | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.policy = policy or WebPolicy()
        self.backend = backend if backend is not None else DuckDuckGoBackend(
            client=client, timeout=self.policy.timeout_s, user_agent=self.policy.user_agent
        )
        self.client = client
        self.audit = audit
        self.clock = clock or _utc_now
        self.searches: list[SearchRecord] = []
        self.reads: list[WebEvidence] = []
        #: content_hash -> snapshot text; the bytes WebEvidence verifies against.
        self.snapshots: dict[str, str] = {}
        self._search_cache: dict[tuple[str, int], list[SearchResult]] = {}
        self._robots: dict[str, RobotFileParser | None] = {}

    # -- session accounting ------------------------------------------------------------------

    @property
    def searches_used(self) -> int:
        return sum(1 for record in self.searches if not record.cached)

    @property
    def fetches_used(self) -> int:
        return len(self.reads)

    def report(self) -> WebSessionReport:
        """The session's auditable transcript."""
        return WebSessionReport(
            searches=list(self.searches),
            reads=list(self.reads),
            searches_used=self.searches_used,
            fetches_used=self.fetches_used,
        )

    def _record(self, action: str, *, decision: str, **details: object) -> None:
        record = getattr(self.audit, "record", None)
        if callable(record):
            record(action, decision=decision, details=dict(details))

    # -- search --------------------------------------------------------------------------------

    async def search(
        self, query: str, *, max_results: int | None = None, recency: str | None = None
    ) -> list[SearchResult]:
        """Run one governed search; identical repeat queries replay from cache."""
        query = " ".join(query.split())
        limit = max_results or self.policy.max_results
        key = (query.lower(), limit)
        if key in self._search_cache:
            results = self._search_cache[key]
            self.searches.append(
                SearchRecord(query=query, results=results, cached=True, at=self.clock())
            )
            return results
        if self.searches_used >= self.policy.max_searches:
            self._record("web_search", decision="deny", query=query, reason="budget")
            raise WebPolicyError(
                f"search budget exhausted ({self.policy.max_searches}); "
                "answer from what you have",
                details={"query": query, "max_searches": self.policy.max_searches},
            )
        kwargs: dict[str, object] = {"max_results": limit}
        if recency is not None and "recency" in inspect.signature(self.backend.search).parameters:
            kwargs["recency"] = recency
        results = await self.backend.search(query, **kwargs)  # type: ignore[arg-type]
        self._search_cache[key] = results
        self.searches.append(
            SearchRecord(query=query, results=results, cached=False, at=self.clock())
        )
        self._record(
            "web_search", decision="allow", query=query, results=len(results),
            backend=getattr(self.backend, "name", type(self.backend).__name__),
        )
        return results

    # -- read ----------------------------------------------------------------------------------

    async def _robots_allows(self, client: httpx.AsyncClient, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = None
            try:
                response = await client.get(
                    origin + "/robots.txt",
                    timeout=self.policy.timeout_s,
                    follow_redirects=True,
                    headers={"User-Agent": self.policy.user_agent},
                )
                # An unreadable robots.txt (4xx/5xx or network error) leaves
                # parser None: everything is allowed, per convention.
                if response.status_code < 400:
                    fetched = RobotFileParser()
                    fetched.parse(response.text.splitlines())
                    parser = fetched
            except httpx.HTTPError:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(self.policy.user_agent, url)

    async def read(
        self,
        url: str,
        *,
        query: str = "",
        budget_tokens: int | None = None,
        max_excerpts: int = 12,
    ) -> PageExtract:
        """Fetch one page and return only its *query*-relevant passages.

        The full body is snapshotted and hashed; the returned
        :class:`~vincio.web.PageExtract` carries the ``content_hash`` that keys
        the snapshot, and the matching :class:`WebEvidence` lands on
        :attr:`reads`.
        """
        budget = budget_tokens or self.policy.excerpt_budget_tokens
        self.policy.check_url(url)
        if self.fetches_used >= self.policy.max_fetches:
            self._record("web_read", decision="deny", url=url, reason="budget")
            raise WebPolicyError(
                f"fetch budget exhausted ({self.policy.max_fetches}); "
                "answer from what you have",
                details={"url": url, "max_fetches": self.policy.max_fetches},
            )
        async with _managed_client(self.client) as client:
            if self.policy.respect_robots and not await self._robots_allows(client, url):
                self._record("web_read", decision="deny", url=url, reason="robots")
                raise WebPolicyError(
                    "robots.txt disallows fetching this URL", details={"url": url}
                )
            try:
                response = await client.get(
                    url,
                    timeout=self.policy.timeout_s,
                    follow_redirects=True,
                    headers={"User-Agent": self.policy.user_agent},
                )
            except httpx.HTTPError as exc:
                raise WebFetchError(f"fetch failed: {exc}", details={"url": url}) from exc
        if response.status_code >= 400:
            raise WebFetchError(
                f"fetch returned HTTP {response.status_code}",
                details={"url": url, "status": response.status_code},
            )
        content_type = response.headers.get("content-type", "text/html").split(";")[0].strip()
        if content_type and not content_type.startswith(_TEXTUAL_TYPES):
            raise WebFetchError(
                f"unsupported content type {content_type!r}",
                details={"url": url, "content_type": content_type},
            )
        if len(response.content) > self.policy.max_page_bytes:
            raise WebFetchError(
                f"page body exceeds the {self.policy.max_page_bytes}-byte ceiling",
                details={"url": url, "bytes": len(response.content)},
            )
        text = response.text
        digest = _content_digest(text)
        self.snapshots[digest] = text
        extract = extract_page(
            text, url=url, query=query, budget_tokens=budget, max_excerpts=max_excerpts
        )
        extract.content_hash = digest
        evidence = WebEvidence(
            url=url,
            title=extract.title,
            query=query,
            fetched_at=self.clock(),
            content_hash=digest,
            budget_tokens=budget,
            max_excerpts=max_excerpts,
            excerpts=list(extract.excerpts),
            excerpt_tokens=extract.excerpt_tokens,
            page_tokens=extract.page_tokens,
        )
        self.reads.append(evidence)
        self._record(
            "web_read", decision="allow", url=url, content_hash=digest,
            page_tokens=extract.page_tokens, excerpt_tokens=extract.excerpt_tokens,
        )
        return extract

    # -- sync front door -------------------------------------------------------------------------

    def search_sync(
        self, query: str, *, max_results: int | None = None, recency: str | None = None
    ) -> list[SearchResult]:
        return _run_sync(self.search(query, max_results=max_results, recency=recency))

    def read_sync(
        self,
        url: str,
        *,
        query: str = "",
        budget_tokens: int | None = None,
        max_excerpts: int = 12,
    ) -> PageExtract:
        return _run_sync(
            self.read(url, query=query, budget_tokens=budget_tokens, max_excerpts=max_excerpts)
        )

    # -- the model-facing tools --------------------------------------------------------------

    def tool_handlers(self) -> list[tuple[ToolSpec, Callable]]:
        """The ``web_search`` / ``web_read`` tools, ready for the registry.

        Register each pair with
        ``app.tool_registry.register_spec(spec, handler=handler)`` (that is what
        :meth:`~vincio.core.app.ContextApp.use_web_search` does). Outputs are
        compact and JSON-ready — typed hits and budgeted excerpts, never HTML.
        """

        async def web_search(query: str, max_results: int = 0) -> dict[str, object]:
            results = await self.search(query, max_results=max_results or None)
            return {
                "results": [
                    {"rank": r.rank, "title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ]
            }

        async def web_read(url: str, query: str = "", budget_tokens: int = 0) -> dict[str, object]:
            extract = await self.read(url, query=query, budget_tokens=budget_tokens or None)
            return {
                "url": extract.url,
                "title": extract.title,
                "excerpts": [
                    {"section": e.section, "text": e.text} for e in extract.excerpts
                ],
                "content_hash": extract.content_hash,
            }

        search_spec = ToolSpec(
            name="web_search",
            description=(
                "Search the open web. Use for facts that are recent, volatile, niche, "
                "or need an exact citation — not for stable knowledge or anything "
                "already in context. Query: 2-5 significant keywords, not a sentence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "2-5 keyword search terms"},
                    "max_results": {
                        "type": "integer",
                        "description": "How many hits to return (default: policy)",
                    },
                },
                "required": ["query"],
            },
            permissions=["web:search"],
            side_effects="external",
            cacheable=True,
            metadata={"plane": "web"},
        )
        read_spec = ToolSpec(
            name="web_read",
            description=(
                "Read one web page from a web_search result, returning only the "
                "passages relevant to `query` under a token budget. Read the most "
                "promising 1-2 results, then answer and cite the URLs you used."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The page URL to read"},
                    "query": {
                        "type": "string",
                        "description": "The fact you need from this page",
                    },
                    "budget_tokens": {
                        "type": "integer",
                        "description": "Max tokens of excerpts to return (default: policy)",
                    },
                },
                "required": ["url"],
            },
            permissions=["web:read"],
            side_effects="external",
            cacheable=True,
            metadata={"plane": "web"},
        )
        return [(search_spec, web_search), (read_spec, web_read)]


def _run_sync(coro):  # noqa: ANN001, ANN202 - mirror of providers.base.run_sync
    """Run *coro* from sync code, inside or outside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
