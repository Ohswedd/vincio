"""Web-search connector: search queries in, cited Documents out.

Where :class:`~vincio.connectors.web.WebConnector` fetches URLs you already
know, this connector *finds* them: each query runs through a governed
:class:`~vincio.web.WebBrowser` session (DuckDuckGo by default, any
:class:`~vincio.web.SearchBackend`), the top hits are read token-efficiently,
and each page's relevant passages become a :class:`~vincio.core.types.Document`
whose metadata carries the query, rank, and content hash of the snapshot it
was derived from.

This is what makes the existing deep-research agent web-backed with zero new
agent code::

    app.add_source("web", connector=connect("websearch", queries=[question]))
    report = app.research(question)

With ``fetch_pages=False`` only the result snippets are indexed — one request
per query, no page fetches at all.
"""

from __future__ import annotations

import httpx

from ..core.errors import VincioError
from ..core.types import Document
from ..web.browser import WebBrowser
from ..web.policy import WebPolicy
from ..web.search import SearchBackend
from .base import register_connector

__all__ = ["WebSearchConnector"]


@register_connector("websearch")
class WebSearchConnector:
    name = "websearch"

    def __init__(
        self,
        queries: list[str],
        *,
        backend: SearchBackend | None = None,
        policy: WebPolicy | None = None,
        client: httpx.AsyncClient | None = None,
        max_results: int = 3,
        fetch_pages: bool = True,
        budget_tokens: int | None = None,
    ) -> None:
        self.queries = list(queries)
        self.max_results = max_results
        self.fetch_pages = fetch_pages
        self.budget_tokens = budget_tokens
        self.browser = WebBrowser(backend, policy=policy, client=client)

    async def load(self) -> list[Document]:
        documents: list[Document] = []
        for query in self.queries:
            results = await self.browser.search(query, max_results=self.max_results)
            for result in results:
                if not self.fetch_pages:
                    documents.append(
                        Document(
                            source_uri=result.url,
                            title=result.title,
                            media_type="text/plain",
                            text=result.snippet or result.title,
                            metadata={
                                "connector": self.name, "query": query, "rank": result.rank,
                            },
                        )
                    )
                    continue
                try:
                    extract = await self.browser.read(
                        result.url, query=query, budget_tokens=self.budget_tokens
                    )
                except VincioError:
                    # A page the policy refuses or the fetch loses is skipped, not
                    # fatal: the query's other hits still index. The refusal is
                    # already on the browser's audit trail.
                    continue
                documents.append(
                    Document(
                        source_uri=result.url,
                        title=extract.title or result.title,
                        media_type="text/plain",
                        text=extract.as_context(),
                        metadata={
                            "connector": self.name,
                            "query": query,
                            "rank": result.rank,
                            "content_hash": extract.content_hash,
                            "page_tokens": extract.page_tokens,
                            "excerpt_tokens": extract.excerpt_tokens,
                        },
                    )
                )
        return documents
