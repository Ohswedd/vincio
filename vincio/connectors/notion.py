"""Notion connector: pages (optionally from a database) via the Notion API."""

from __future__ import annotations

from typing import Any

import httpx

from ..core.concurrency import gather_bounded
from ..core.errors import LoaderError
from ..core.types import Document
from .base import managed_client, register_connector

__all__ = ["NotionConnector"]

_TEXT_BLOCK_TYPES = (
    "paragraph", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "quote", "callout", "toggle", "to_do",
)


def _rich_text(block_payload: dict[str, Any]) -> str:
    return "".join(part.get("plain_text", "") for part in block_payload.get("rich_text", []))


@register_connector("notion")
class NotionConnector:
    name = "notion"

    def __init__(
        self,
        token: str,
        *,
        database_id: str | None = None,
        page_ids: list[str] | None = None,
        api_base: str = "https://api.notion.com/v1",
        notion_version: str = "2022-06-28",
        client: httpx.AsyncClient | None = None,
        max_pages: int = 100,
        max_concurrency: int = 4,
    ) -> None:
        if database_id is None and not page_ids:
            raise LoaderError("notion connector needs a database_id or page_ids")
        self.token = token
        self.database_id = database_id
        self.page_ids = list(page_ids or [])
        self.api_base = api_base.rstrip("/")
        self.notion_version = notion_version
        self.client = client
        self.max_pages = max_pages
        self.max_concurrency = max_concurrency

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": self.notion_version,
        }

    @staticmethod
    def _page_title(page: dict[str, Any]) -> str:
        for prop in page.get("properties", {}).values():
            if prop.get("type") == "title":
                return "".join(part.get("plain_text", "") for part in prop.get("title", []))
        return page.get("id", "untitled")

    async def _page_text(self, client: httpx.AsyncClient, page_id: str) -> str:
        lines: list[str] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            response = await client.get(
                f"{self.api_base}/blocks/{page_id}/children",
                params=params,
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            for block in payload.get("results", []):
                block_type = block.get("type", "")
                if block_type in _TEXT_BLOCK_TYPES:
                    text = _rich_text(block.get(block_type, {}))
                    if text.strip():
                        lines.append(text)
            if not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")
        return "\n".join(lines)

    async def _load_page(self, client: httpx.AsyncClient, page: dict[str, Any]) -> Document:
        page_id = page["id"]
        text = await self._page_text(client, page_id)
        extra: dict[str, Any] = {}
        if page.get("last_edited_time"):
            extra["created_at"] = page["last_edited_time"]
        return Document(
            source_uri=page.get("url") or f"notion://{page_id}",
            title=self._page_title(page),
            text=text,
            metadata={"connector": self.name, "page_id": page_id},
            **extra,
        )

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                pages: list[dict[str, Any]] = []
                if self.database_id:
                    cursor: str | None = None
                    while len(pages) < self.max_pages:
                        body: dict[str, Any] = {"page_size": min(100, self.max_pages)}
                        if cursor:
                            body["start_cursor"] = cursor
                        response = await client.post(
                            f"{self.api_base}/databases/{self.database_id}/query",
                            json=body,
                            headers=self._headers(),
                        )
                        response.raise_for_status()
                        payload = response.json()
                        pages.extend(payload.get("results", []))
                        if not payload.get("has_more"):
                            break
                        cursor = payload.get("next_cursor")
                for page_id in self.page_ids:
                    response = await client.get(
                        f"{self.api_base}/pages/{page_id}", headers=self._headers()
                    )
                    response.raise_for_status()
                    pages.append(response.json())
                return await gather_bounded(
                    (self._load_page(client, page) for page in pages[: self.max_pages]),
                    limit=self.max_concurrency,
                )
            except httpx.HTTPError as exc:
                raise LoaderError(f"notion connector failed: {exc}") from exc
