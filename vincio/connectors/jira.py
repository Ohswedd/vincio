"""Jira connector: issues via the Jira Cloud REST API (v3).

Each issue becomes one :class:`~vincio.core.types.Document` (``KEY: summary`` +
description), so a backlog chunks, indexes, budgets, and cites like any source.
Runs on the core ``httpx`` dependency and accepts an injected client for tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from .base import managed_client, register_connector

__all__ = ["JiraConnector", "adf_to_text"]


def adf_to_text(node: Any) -> str:
    """Flatten an Atlassian Document Format (ADF) value to plain text.

    Descriptions on Jira Cloud are ADF (a nested ``{type, content, text}``
    document); a plain string is returned unchanged.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(child) for child in node)
    if isinstance(node, dict):
        parts: list[str] = []
        if isinstance(node.get("text"), str):
            parts.append(node["text"])
        parts.append(adf_to_text(node.get("content")))
        rendered = "".join(parts)
        # Block-level nodes read better separated by newlines.
        if node.get("type") in ("paragraph", "heading", "blockquote", "listItem", "codeBlock"):
            rendered += "\n"
        return rendered
    return ""


@register_connector("jira")
class JiraConnector:
    name = "jira"

    def __init__(
        self,
        base_url: str,
        *,
        jql: str = "order by updated DESC",
        email: str | None = None,
        token: str | None = None,
        fields: tuple[str, ...] = ("summary", "description", "status", "issuetype", "project"),
        api_path: str = "/rest/api/3/search",
        client: httpx.AsyncClient | None = None,
        max_issues: int = 100,
        page_size: int = 50,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.jql = jql
        self.email = email
        self.token = token
        self.fields = fields
        self.api_path = "/" + api_path.strip("/")
        self.client = client
        self.max_issues = max_issues
        self.page_size = page_size

    def _auth_kwargs(self) -> dict[str, Any]:
        # Atlassian Cloud uses email + API token over Basic auth; self-managed
        # instances accept a bearer Personal Access Token.
        if self.token and self.email:
            return {"auth": (self.email, self.token)}
        if self.token:
            return {"headers": {"Authorization": f"Bearer {self.token}"}}
        return {}

    def _issue_document(self, issue: dict[str, Any]) -> Document:
        key = str(issue.get("key", issue.get("id", "")))
        fields = issue.get("fields", {}) or {}
        summary = str(fields.get("summary") or "").strip()
        description = adf_to_text(fields.get("description")).strip()
        status = (fields.get("status") or {}).get("name")
        issuetype = (fields.get("issuetype") or {}).get("name")
        project = (fields.get("project") or {}).get("key")
        body = f"{summary}\n\n{description}".strip() if description else summary
        return Document(
            source_uri=f"{self.base_url}/browse/{key}",
            title=f"{key}: {summary}" if summary else key,
            text=body,
            metadata={
                "connector": self.name,
                "key": key,
                "status": status,
                "issue_type": issuetype,
                "project": project,
            },
        )

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                documents: list[Document] = []
                start_at = 0
                while len(documents) < self.max_issues:
                    params: dict[str, Any] = {
                        "jql": self.jql,
                        "startAt": start_at,
                        "maxResults": min(self.page_size, self.max_issues - len(documents)),
                        "fields": ",".join(self.fields),
                    }
                    response = await client.get(
                        f"{self.base_url}{self.api_path}", params=params, **self._auth_kwargs()
                    )
                    response.raise_for_status()
                    payload = response.json()
                    issues = payload.get("issues", [])
                    if not issues:
                        break
                    for issue in issues:
                        documents.append(self._issue_document(issue))
                        if len(documents) >= self.max_issues:
                            break
                    total = payload.get("total")
                    start_at += len(issues)
                    if total is not None and start_at >= total:
                        break
                    if len(issues) < params["maxResults"]:
                        break
                return documents
            except httpx.HTTPError as exc:
                raise LoaderError(f"jira connector failed: {exc}") from exc
