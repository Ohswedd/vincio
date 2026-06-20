"""Linear connector: issues via the Linear GraphQL API.

Each Linear issue becomes one :class:`~vincio.core.types.Document`
(``IDENT: title`` + Markdown description). Runs on the core ``httpx``
dependency and accepts an injected client for tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from .base import managed_client, register_connector

__all__ = ["LinearConnector"]

_ISSUES_QUERY = """
query Issues($first: Int!, $after: String) {
  issues(first: $first, after: $after) {
    nodes {
      id identifier title description url updatedAt
      state { name }
      team { key }
    }
    pageInfo { hasNextPage endCursor }
  }
}
""".strip()


@register_connector("linear")
class LinearConnector:
    name = "linear"

    def __init__(
        self,
        api_key: str,
        *,
        query: str = _ISSUES_QUERY,
        api_url: str = "https://api.linear.app/graphql",
        client: httpx.AsyncClient | None = None,
        max_issues: int = 100,
        page_size: int = 50,
    ) -> None:
        self.api_key = api_key
        self.query = query
        self.api_url = api_url
        self.client = client
        self.max_issues = max_issues
        self.page_size = page_size

    def _headers(self) -> dict[str, str]:
        # Linear accepts the personal/API key directly in Authorization.
        return {"Authorization": self.api_key, "Content-Type": "application/json"}

    def _issue_document(self, node: dict[str, Any]) -> Document:
        identifier = str(node.get("identifier") or node.get("id", ""))
        title = str(node.get("title") or "").strip()
        description = str(node.get("description") or "").strip()
        body = f"{title}\n\n{description}".strip() if description else title
        return Document(
            source_uri=node.get("url"),
            title=f"{identifier}: {title}" if title else identifier,
            media_type="text/markdown",
            text=body,
            metadata={
                "connector": self.name,
                "identifier": identifier,
                "state": (node.get("state") or {}).get("name"),
                "team": (node.get("team") or {}).get("key"),
                "id": node.get("id"),
            },
        )

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                documents: list[Document] = []
                cursor: str | None = None
                while len(documents) < self.max_issues:
                    variables: dict[str, Any] = {
                        "first": min(self.page_size, self.max_issues - len(documents)),
                        "after": cursor,
                    }
                    response = await client.post(
                        self.api_url,
                        json={"query": self.query, "variables": variables},
                        headers=self._headers(),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("errors"):
                        raise LoaderError(f"linear API error: {payload['errors']}")
                    issues = (payload.get("data", {}) or {}).get("issues", {}) or {}
                    nodes = issues.get("nodes", [])
                    for node in nodes:
                        documents.append(self._issue_document(node))
                        if len(documents) >= self.max_issues:
                            break
                    page_info = issues.get("pageInfo", {}) or {}
                    if not page_info.get("hasNextPage") or not nodes:
                        break
                    cursor = page_info.get("endCursor")
                return documents
            except httpx.HTTPError as exc:
                raise LoaderError(f"linear connector failed: {exc}") from exc
