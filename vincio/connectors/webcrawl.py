"""Web-crawl connector: a bounded site walk into retrieval Documents.

Where :class:`~vincio.connectors.websearch.WebSearchConnector` turns queries
into documents, this turns *seeds* into a corpus — a library's documentation,
a section of a site — via a governed :class:`~vincio.web.WebCrawler`. Every
page keeps the SSRF rails, robots, size caps, and snapshotting, and the walk is
bounded on every axis (pages, depth, per-host, bytes, wall-clock) with
trap-template defense. Register it through ``connect("webcrawl", ...)`` or
``app.add_source("docs", connector=connect("webcrawl", seeds=[...]))``.
"""

from __future__ import annotations

import httpx

from ..core.types import Document
from ..web.crawl import WebCrawler
from ..web.policy import WebPolicy
from ..web.search import SearchBackend
from .base import register_connector

__all__ = ["WebCrawlConnector"]


@register_connector("webcrawl")
class WebCrawlConnector:
    name = "webcrawl"

    def __init__(
        self,
        seeds: str | list[str],
        *,
        scope: str = "subtree",
        query: str = "",
        max_pages: int | None = None,
        max_depth: int | None = None,
        policy: WebPolicy | None = None,
        backend: SearchBackend | None = None,
        client: httpx.AsyncClient | None = None,
        mode: str = "full",
    ) -> None:
        self.seeds = seeds
        self.scope = scope
        self.query = query
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.crawler = WebCrawler(
            policy=policy or WebPolicy.preset("scrape"),
            client=client,
            mode=mode,  # type: ignore[arg-type]
        )

    async def load(self) -> list[Document]:
        collection = await self.crawler.crawl(
            self.seeds,
            scope=self.scope,  # type: ignore[arg-type]
            query=self.query,
            max_pages=self.max_pages,
            max_depth=self.max_depth,
        )
        return collection.to_documents()
