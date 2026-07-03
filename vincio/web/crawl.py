"""Bounded, governed crawling — a site into a verifiable collection.

Search-and-read answers a question; sometimes you want the *corpus*: a library's
whole documentation, a set of pages to index for RAG, or a table of records to
analyze. :class:`WebCrawler` walks outward from seed URLs through the same
:class:`~vincio.web.WebBrowser` — so every fetch keeps the SSRF rails, robots,
per-host pacing, size caps, and snapshotting — and returns a
:class:`WebCollection` that converts to retrieval :class:`Document`\\ s or a
tabular :class:`~vincio.data.Dataset`, and re-derives offline from its snapshots.

The walk is *bounded on every axis* a hostile or accidental trap could exploit:
total pages, depth, **per-host** pages, total bytes, wall-clock seconds, and
links-followed-per-page, plus canonical-URL dedup and a repeating-path-template
guard that stops calendar/pagination traps that mint infinite distinct URLs. It
is deterministic: a breadth-first, lexicographically-ordered frontier means the
same seeds and limits always visit the same pages in the same order (no
``asyncio.gather`` nondeterminism), so the crawl is replayable and gate-able.
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, Field

from ..core.diagnostics import note_suppressed
from ..core.errors import VincioError, WebError, WebPolicyError
from ..core.types import Document, TrustLevel
from .browser import WebBrowser, _content_digest
from .extract import ExtractMode, PageLink
from .policy import WebPolicy

__all__ = ["WebCrawler", "WebCollection", "CrawledPage", "CrawlScope"]

CrawlScope = str  # "page" | "subtree" | "domain"

# Collapse a URL to a path *template* so ?page=1, ?page=2, /2024/01, /2024/02 all
# map to one shape; a template seen too many times is a pagination/calendar trap.
_DIGITS_RE = re.compile(r"\d+")
_TRAP_TEMPLATE_LIMIT = 8


def _path_template(url: str) -> str:
    parts = urlsplit(url)
    path = _DIGITS_RE.sub("#", parts.path)
    keys = ",".join(sorted(k for k, _ in (kv.split("=", 1) if "=" in kv else (kv, "")
                                          for kv in parts.query.split("&") if kv)))
    return f"{parts.netloc}{path}?{keys}"


def _in_scope(url: str, seed: str, scope: CrawlScope) -> bool:
    u, s = urlsplit(url), urlsplit(seed)
    if u.scheme not in ("http", "https"):
        return False
    if scope == "domain":
        return u.hostname == s.hostname or (
            (u.hostname or "").endswith("." + (s.hostname or "")) if s.hostname else False
        )
    if scope == "subtree":
        seed_dir = s.path.rsplit("/", 1)[0] + "/"
        return u.hostname == s.hostname and u.path.startswith(seed_dir)
    return u.hostname == s.hostname and u.path == s.path  # "page"


class CrawledPage(BaseModel):
    """One page visited during a crawl, content-bound to its snapshot."""

    url: str
    title: str = ""
    depth: int = 0
    content_hash: str = ""
    text: str = ""  # the extracted, budgeted reading (not the raw HTML)
    page_tokens: int = 0


class WebCollection(BaseModel):
    """The result of a crawl: the visited pages plus what stopped the walk.

    Convert to :meth:`to_documents` for retrieval or :meth:`to_dataset` for the
    data plane; :meth:`verify` re-derives every page offline from its snapshot.
    """

    seeds: list[str] = Field(default_factory=list)
    scope: CrawlScope = "subtree"
    pages: list[CrawledPage] = Field(default_factory=list)
    stopped_reason: str = ""
    pages_fetched: int = 0
    bytes_fetched: int = 0

    def to_documents(self) -> list[Document]:
        """One :class:`~vincio.core.types.Document` per page, web-trust-tagged
        and carrying its content hash — ready for ``app.add_source``."""
        return [
            Document(
                source_uri=page.url,
                title=page.title or page.url,
                media_type="text/plain",
                text=page.text,
                trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
                metadata={
                    "connector": "webcrawl",
                    "content_hash": page.content_hash,
                    "depth": page.depth,
                    "page_tokens": page.page_tokens,
                },
            )
            for page in self.pages
        ]

    def to_dataset(self, *, name: str = "web_collection"):  # noqa: ANN201 - Dataset return
        """A tabular :class:`~vincio.data.Dataset` — one row per page (``url``,
        ``title``, ``depth``, ``content_hash``, ``page_tokens``, ``text``) — so a
        scrape flows straight into the governed data/analysis plane."""
        from ..data import Dataset

        records = [
            {
                "url": page.url,
                "title": page.title,
                "depth": page.depth,
                "content_hash": page.content_hash,
                "page_tokens": page.page_tokens,
                "text": page.text,
            }
            for page in self.pages
        ]
        return Dataset.from_records(records, name=name, source="webcrawl")

    def verify(self, snapshots: dict[str, str | bytes]) -> bool:
        """True iff every page's snapshot is present and hashes to its record."""
        for page in self.pages:
            snap = snapshots.get(page.content_hash)
            if snap is None:
                return False
            text = snap.decode("utf-8", errors="replace") if isinstance(snap, bytes) else snap
            if _content_digest(text) != page.content_hash:
                return False
        return True


class WebCrawler:
    """A bounded, deterministic, governed crawl over a :class:`WebBrowser`.

    Reuses the browser's rails and snapshot store, so a crawl cannot exceed the
    browser's own fetch budget and every page it keeps is offline-verifiable.
    """

    def __init__(
        self,
        browser: WebBrowser | None = None,
        *,
        policy: WebPolicy | None = None,
        client: object | None = None,
        mode: ExtractMode = "full",
    ) -> None:
        self.browser = browser or WebBrowser(policy=policy, client=client)  # type: ignore[arg-type]
        self.policy = self.browser.policy
        self.mode = mode

    async def crawl(
        self,
        seeds: str | list[str],
        *,
        scope: CrawlScope = "subtree",
        query: str = "",
        max_pages: int | None = None,
        max_depth: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> WebCollection:
        """Walk outward from *seeds*, bounded on every axis, into a collection.

        ``scope`` is ``"page"`` (just the seeds), ``"subtree"`` (URLs under a
        seed's directory), or ``"domain"`` (same host). ``clock`` (a monotonic
        seconds source) enforces the wall-clock budget; injected as ``lambda: 0``
        it makes the walk deterministic offline.
        """
        _require_seed(seeds)
        seed_list = [seeds] if isinstance(seeds, str) else list(seeds)
        pol = self.policy
        pages_cap = max_pages if max_pages is not None else pol.max_crawl_pages
        depth_cap = max_depth if max_depth is not None else pol.max_crawl_depth
        # A real monotonic clock by default so the wall-clock budget actually
        # fires in production; tests inject ``lambda: 0.0`` for determinism.
        now = clock or time.monotonic
        start = now()

        # BFS with a lexicographically-ordered frontier for determinism.
        frontier: deque[tuple[str, int, str]] = deque()  # (url, depth, seed)
        seen: set[str] = set()
        for seed in seed_list:
            canon = pol.canonicalize(seed)
            if canon not in seen:
                seen.add(canon)
                frontier.append((seed, 0, seed))

        collection = WebCollection(seeds=seed_list, scope=scope)
        per_host: dict[str, int] = {}
        templates: dict[str, int] = {}
        stopped = ""

        while frontier:
            if len(collection.pages) >= pages_cap:
                stopped = "max_pages"
                break
            if collection.bytes_fetched >= pol.max_crawl_bytes:
                stopped = "max_bytes"
                break
            if now() - start >= pol.max_crawl_seconds:
                stopped = "max_seconds"
                break
            url, depth, seed = frontier.popleft()
            host = (urlsplit(url).hostname or "").lower()
            if per_host.get(host, 0) >= pol.max_crawl_pages_per_host:
                continue
            template = _path_template(url)
            if templates.get(template, 0) >= _TRAP_TEMPLATE_LIMIT:
                note_suppressed("web.crawl_trap_template_skipped")
                continue
            try:
                extract = await self.browser.read(url, query=query, mode=self.mode)
            except WebPolicyError as exc:
                # A fetch-budget refusal is terminal for the whole walk (the
                # shared browser budget is spent), not a per-page skip; report it
                # distinctly rather than as "frontier_exhausted".
                if (exc.details or {}).get("reason") == "budget" or "budget" in str(exc):
                    stopped = "fetch_budget"
                    break
                note_suppressed("web.crawl_page_skipped")
                continue
            except VincioError:
                note_suppressed("web.crawl_page_skipped")
                continue
            per_host[host] = per_host.get(host, 0) + 1
            templates[template] = templates.get(template, 0) + 1
            snapshot = self.browser.snapshots.get(extract.content_hash, "")
            collection.bytes_fetched += len(snapshot.encode("utf-8"))
            collection.pages.append(
                CrawledPage(
                    url=extract.url or url,
                    title=extract.title,
                    depth=depth,
                    content_hash=extract.content_hash,
                    text=extract.as_context(),
                    page_tokens=extract.page_tokens,
                )
            )
            if depth < depth_cap:
                self._enqueue_links(url, seed, depth, scope, extract.links, seen, frontier)

        collection.stopped_reason = stopped or "frontier_exhausted"
        collection.pages_fetched = len(collection.pages)
        return collection

    def _enqueue_links(
        self,
        base: str,
        seed: str,
        depth: int,
        scope: CrawlScope,
        links: list[PageLink],
        seen: set[str],
        frontier: deque[tuple[str, int, str]],
    ) -> None:
        # Resolve, scope-filter, dedupe, and add in sorted order (determinism),
        # bounded by the per-page link cap so one hub page cannot flood the walk.
        candidates: list[str] = []
        for link in links[: self.policy.max_links_per_page]:
            absolute = urljoin(base, link.url)
            if not _in_scope(absolute, seed, scope):
                continue
            if not self.policy.allows_url(absolute):
                continue
            canon = self.policy.canonicalize(absolute)
            if canon in seen:
                continue
            seen.add(canon)
            candidates.append(absolute)
        for url in sorted(candidates):
            frontier.append((url, depth + 1, seed))

    async def crawl_to_documents(
        self, seeds: str | list[str], **kwargs: object
    ) -> list[Document]:
        """Convenience: crawl and return retrieval documents directly."""
        collection = await self.crawl(seeds, **kwargs)  # type: ignore[arg-type]
        return collection.to_documents()


def _require_seed(seeds: str | list[str]) -> None:
    if not seeds:
        raise WebError("a crawl needs at least one seed URL")
