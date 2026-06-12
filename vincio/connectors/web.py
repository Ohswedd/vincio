"""Web connector: fetch URLs into Documents."""

from __future__ import annotations

import re

import httpx

from ..core.concurrency import gather_bounded
from ..core.errors import LoaderError
from ..core.types import Document
from ..documents.parsers import extract_markdown_sections, strip_html
from .base import managed_client, register_connector

__all__ = ["WebConnector"]

_TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")


@register_connector("web")
class WebConnector:
    name = "web"

    def __init__(
        self,
        urls: list[str],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 15.0,
        max_concurrency: int = 4,
    ) -> None:
        self.urls = list(urls)
        self.client = client
        self.timeout = timeout
        self.max_concurrency = max_concurrency

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> Document:
        response = await client.get(url, timeout=self.timeout, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        raw = response.text
        if "html" in content_type or raw.lstrip()[:1] == "<":
            match = _TITLE_RE.search(raw)
            title = strip_html(match.group(1)).strip() if match else url
            text = strip_html(raw)
            media_type = "text/html"
            sections: list[dict] = []
        else:
            title = url.rsplit("/", 1)[-1] or url
            text = raw
            media_type = content_type.split(";")[0].strip() or "text/plain"
            sections = (
                [s.model_dump(mode="json") for s in extract_markdown_sections(text)]
                if media_type == "text/markdown"
                else []
            )
        return Document(
            source_uri=url,
            title=title,
            media_type=media_type,
            text=text,
            sections=sections,
            metadata={"connector": self.name, "status_code": response.status_code},
        )

    async def load(self) -> list[Document]:
        if not self.urls:
            return []
        async with managed_client(self.client) as client:
            try:
                return await gather_bounded(
                    (self._fetch(client, url) for url in self.urls),
                    limit=self.max_concurrency,
                )
            except httpx.HTTPError as exc:
                raise LoaderError(f"web connector failed: {exc}") from exc
