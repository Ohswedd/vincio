"""SharePoint connector: document-library files via the Microsoft Graph API.

Lists files in a site's default drive (optionally under a folder path) and
downloads text content into :class:`~vincio.core.types.Document`\\ s. Runs on
the core ``httpx`` dependency with a bearer access token; accepts an injected
client for tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from ..documents.parsers import strip_html
from .base import managed_client, register_connector

__all__ = ["SharePointConnector"]

_TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".csv", ".json", ".xml", ".html", ".htm", ".rst")


@register_connector("sharepoint")
class SharePointConnector:
    name = "sharepoint"

    def __init__(
        self,
        site_id: str,
        access_token: str,
        *,
        folder_path: str = "",
        api_base: str = "https://graph.microsoft.com/v1.0",
        client: httpx.AsyncClient | None = None,
        max_files: int = 100,
    ) -> None:
        self.site_id = site_id
        self.access_token = access_token
        self.folder_path = folder_path.strip("/")
        self.api_base = api_base.rstrip("/")
        self.client = client
        self.max_files = max_files

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _children_url(self) -> str:
        root = f"{self.api_base}/sites/{self.site_id}/drive/root"
        if self.folder_path:
            return f"{root}:/{self.folder_path}:/children"
        return f"{root}/children"

    def _wanted(self, item: dict[str, Any]) -> bool:
        if "file" not in item:  # folders and other facets
            return False
        name = str(item.get("name", "")).lower()
        return name.endswith(_TEXT_SUFFIXES)

    async def _item_text(self, client: httpx.AsyncClient, item: dict[str, Any]) -> str:
        response = await client.get(
            f"{self.api_base}/sites/{self.site_id}/drive/items/{item['id']}/content",
            headers=self._headers(),
        )
        response.raise_for_status()
        text = response.text
        name = str(item.get("name", "")).lower()
        return strip_html(text) if name.endswith((".html", ".htm")) else text

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                documents: list[Document] = []
                url: str | None = self._children_url()
                while url and len(documents) < self.max_files:
                    response = await client.get(url, headers=self._headers())
                    response.raise_for_status()
                    payload = response.json()
                    for item in payload.get("value", []):
                        if not self._wanted(item):
                            continue
                        text = await self._item_text(client, item)
                        extra: dict[str, Any] = {}
                        if item.get("lastModifiedDateTime"):
                            extra["created_at"] = item["lastModifiedDateTime"]
                        documents.append(
                            Document(
                                source_uri=item.get("webUrl"),
                                title=item.get("name", item.get("id", "untitled")),
                                text=text,
                                metadata={
                                    "connector": self.name,
                                    "item_id": item.get("id"),
                                    "site_id": self.site_id,
                                },
                                **extra,
                            )
                        )
                        if len(documents) >= self.max_files:
                            break
                    url = payload.get("@odata.nextLink")
                return documents
            except httpx.HTTPError as exc:
                raise LoaderError(f"sharepoint connector failed: {exc}") from exc
