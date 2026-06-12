"""Confluence connector: space pages via the Confluence REST API."""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from ..documents.parsers import strip_html
from .base import managed_client, register_connector

__all__ = ["ConfluenceConnector"]


@register_connector("confluence")
class ConfluenceConnector:
    name = "confluence"

    def __init__(
        self,
        base_url: str,
        *,
        space: str | None = None,
        token: str | None = None,
        username: str | None = None,
        client: httpx.AsyncClient | None = None,
        max_pages: int = 100,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.space = space
        self.token = token
        self.username = username
        self.client = client
        self.max_pages = max_pages

    def _auth_kwargs(self) -> dict[str, Any]:
        if self.token and self.username:
            return {"auth": (self.username, self.token)}
        if self.token:
            return {"headers": {"Authorization": f"Bearer {self.token}"}}
        return {}

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                documents: list[Document] = []
                start = 0
                while len(documents) < self.max_pages:
                    params: dict[str, Any] = {
                        "expand": "body.storage,version,space",
                        "limit": min(50, self.max_pages),
                        "start": start,
                        "type": "page",
                    }
                    if self.space:
                        params["spaceKey"] = self.space
                    response = await client.get(
                        f"{self.base_url}/rest/api/content",
                        params=params,
                        **self._auth_kwargs(),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    results = payload.get("results", [])
                    for page in results:
                        body_html = page.get("body", {}).get("storage", {}).get("value", "")
                        links = page.get("_links", {})
                        extra: dict[str, Any] = {}
                        when = page.get("version", {}).get("when")
                        if when:
                            extra["created_at"] = when
                        documents.append(
                            Document(
                                source_uri=f"{self.base_url}{links.get('webui', '/' + page.get('id', ''))}",
                                title=page.get("title", page.get("id", "untitled")),
                                media_type="text/html",
                                text=strip_html(body_html),
                                metadata={
                                    "connector": self.name,
                                    "page_id": page.get("id"),
                                    "space": page.get("space", {}).get("key", self.space),
                                },
                                **extra,
                            )
                        )
                        if len(documents) >= self.max_pages:
                            break
                    if len(results) < params["limit"]:
                        break
                    start += len(results)
                return documents
            except httpx.HTTPError as exc:
                raise LoaderError(f"confluence connector failed: {exc}") from exc
