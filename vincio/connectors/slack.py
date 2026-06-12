"""Slack connector: channel history via the Slack Web API.

Each channel becomes one Document of "user: message" lines (newest last),
so conversations chunk and retrieve as coherent threads of context.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from .base import managed_client, register_connector

__all__ = ["SlackConnector"]


@register_connector("slack")
class SlackConnector:
    name = "slack"

    def __init__(
        self,
        token: str,
        channels: list[str],
        *,
        api_base: str = "https://slack.com/api",
        client: httpx.AsyncClient | None = None,
        max_messages: int = 200,
    ) -> None:
        self.token = token
        self.channels = list(channels)
        self.api_base = api_base.rstrip("/")
        self.client = client
        self.max_messages = max_messages

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def _channel_document(self, client: httpx.AsyncClient, channel: str) -> Document:
        messages: list[dict[str, Any]] = []
        cursor: str | None = None
        while len(messages) < self.max_messages:
            params: dict[str, Any] = {"channel": channel, "limit": min(200, self.max_messages)}
            if cursor:
                params["cursor"] = cursor
            response = await client.get(
                f"{self.api_base}/conversations.history",
                params=params,
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok", False):
                raise LoaderError(f"slack API error for {channel}: {payload.get('error')}")
            messages.extend(payload.get("messages", []))
            cursor = payload.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        messages = messages[: self.max_messages]
        # History arrives newest-first; render oldest-first for readability.
        lines = [
            f"{m.get('user') or m.get('bot_id') or 'unknown'}: {m.get('text', '')}"
            for m in reversed(messages)
            if m.get("text")
        ]
        extra: dict[str, Any] = {}
        timestamps = [float(m["ts"]) for m in messages if m.get("ts")]
        if timestamps:
            extra["created_at"] = datetime.fromtimestamp(max(timestamps), tz=UTC)
        return Document(
            source_uri=f"slack://channel/{channel}",
            title=f"#{channel}",
            text="\n".join(lines),
            metadata={"connector": self.name, "channel": channel, "message_count": len(lines)},
            **extra,
        )

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                return [await self._channel_document(client, channel) for channel in self.channels]
            except httpx.HTTPError as exc:
                raise LoaderError(f"slack connector failed: {exc}") from exc
