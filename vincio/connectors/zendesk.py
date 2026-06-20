"""Zendesk connector: Help Center articles via the REST API.

Each published article becomes one :class:`~vincio.core.types.Document`
(HTML body stripped to text). Runs on the core ``httpx`` dependency with
email + API token (Basic auth) or a bearer OAuth token; accepts an injected
client for tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from ..documents.parsers import strip_html
from .base import managed_client, register_connector

__all__ = ["ZendeskConnector"]


@register_connector("zendesk")
class ZendeskConnector:
    name = "zendesk"

    def __init__(
        self,
        subdomain: str,
        *,
        email: str | None = None,
        token: str | None = None,
        oauth_token: str | None = None,
        locale: str = "en-us",
        client: httpx.AsyncClient | None = None,
        max_articles: int = 200,
        page_size: int = 100,
    ) -> None:
        self.subdomain = subdomain
        self.email = email
        self.token = token
        self.oauth_token = oauth_token
        self.locale = locale
        self.base_url = f"https://{subdomain}.zendesk.com"
        self.client = client
        self.max_articles = max_articles
        self.page_size = page_size

    def _auth_kwargs(self) -> dict[str, Any]:
        if self.oauth_token:
            return {"headers": {"Authorization": f"Bearer {self.oauth_token}"}}
        if self.email and self.token:
            # Zendesk API-token auth: "<email>/token" as the basic-auth username.
            return {"auth": (f"{self.email}/token", self.token)}
        return {}

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                documents: list[Document] = []
                # First page carries query params; Zendesk returns an absolute
                # ``next_page`` URL with paging baked in for each page after it.
                url: str | None = f"{self.base_url}/api/v2/help_center/{self.locale}/articles.json"
                params: dict[str, Any] | None = {"page[size]": self.page_size}
                while url and len(documents) < self.max_articles:
                    response = await client.get(url, params=params, **self._auth_kwargs())
                    response.raise_for_status()
                    payload = response.json()
                    for article in payload.get("articles", []):
                        if article.get("draft"):
                            continue
                        extra: dict[str, Any] = {}
                        if article.get("updated_at"):
                            extra["created_at"] = article["updated_at"]
                        documents.append(
                            Document(
                                source_uri=article.get("html_url"),
                                title=article.get("title", str(article.get("id", "untitled"))),
                                media_type="text/html",
                                text=strip_html(article.get("body") or ""),
                                metadata={
                                    "connector": self.name,
                                    "article_id": article.get("id"),
                                    "section_id": article.get("section_id"),
                                    "locale": article.get("locale", self.locale),
                                },
                                **extra,
                            )
                        )
                        if len(documents) >= self.max_articles:
                            break
                    url = payload.get("next_page")
                    params = None
                return documents
            except httpx.HTTPError as exc:
                raise LoaderError(f"zendesk connector failed: {exc}") from exc
