"""Memory summarizers (layered memory; session → episodic compaction).

Compacts conversation/session text into summary memories. The default is
extractive (deterministic, offline); an LLM summarizer is used when a
provider is supplied.
"""

from __future__ import annotations

import json

from ..context.compression import extractive_compress, split_sentences
from ..context.scoring import lexical_similarity
from ..core.types import MemoryItem, MemoryScope, MemoryType, Message, ModelRequest
from ..providers.base import ModelProvider

__all__ = ["extractive_summary", "SessionSummarizer"]


def extractive_summary(text: str, *, max_tokens: int = 150, focus: str = "") -> str:
    """Pick the most central sentences (similarity to the whole text or to
    *focus*) within a token budget."""
    if not text.strip():
        return ""
    target = focus or " ".join(split_sentences(text)[:3])
    return extractive_compress(text, target, max_tokens).text


_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "open_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_facts", "decisions", "open_items"],
    "additionalProperties": False,
}


class SessionSummarizer:
    def __init__(
        self,
        provider: ModelProvider | None = None,
        *,
        model: str | None = None,
        max_tokens: int = 200,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens

    async def summarize(
        self,
        session_text: str,
        *,
        scope: MemoryScope = MemoryScope.SESSION,
        owner_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        """Produce summary + decision memories from a session transcript."""
        if not session_text.strip():
            return []
        if self.provider is not None and self.model is not None:
            return await self._summarize_llm(session_text, scope=scope, owner_id=owner_id, session_id=session_id)
        summary = extractive_summary(session_text, max_tokens=self.max_tokens)
        if not summary:
            return []
        return [
            MemoryItem(
                scope=scope,
                type=MemoryType.SUMMARY,
                content=summary,
                owner_id=owner_id,
                confidence=0.6,
                metadata={"session_id": session_id, "method": "extractive"},
            )
        ]

    async def _summarize_llm(
        self,
        session_text: str,
        *,
        scope: MemoryScope,
        owner_id: str | None,
        session_id: str | None,
    ) -> list[MemoryItem]:
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(
                    role="system",
                    content=(
                        "Summarize this session for long-term memory. Extract only "
                        "durable information: stable facts, explicit decisions, and "
                        "open items. Exclude small talk and transient state."
                    ),
                ),
                Message(role="user", content=session_text[:24_000]),
            ],
            output_schema=_SUMMARY_SCHEMA,
            output_schema_name="session_summary",
            temperature=0.0,
        )
        response = await self.provider.generate(request)  # type: ignore[union-attr]
        payload = response.structured
        if payload is None:
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError:
                return [
                    MemoryItem(
                        scope=scope,
                        type=MemoryType.SUMMARY,
                        content=extractive_summary(session_text, max_tokens=self.max_tokens),
                        owner_id=owner_id,
                        confidence=0.5,
                        metadata={"session_id": session_id, "method": "extractive_fallback"},
                    )
                ]
        items: list[MemoryItem] = []
        base_meta = {"session_id": session_id, "method": "llm"}
        if payload.get("summary"):
            items.append(
                MemoryItem(scope=scope, type=MemoryType.SUMMARY, content=payload["summary"],
                           owner_id=owner_id, confidence=0.7, metadata=dict(base_meta))
            )
        for fact in payload.get("key_facts", []):
            # Facts that restate the summary add tokens without information.
            if items and lexical_similarity(fact, items[0].content) > 0.8:
                continue
            items.append(
                MemoryItem(scope=scope, type=MemoryType.FACT, content=fact,
                           owner_id=owner_id, confidence=0.7, metadata=dict(base_meta))
            )
        for decision in payload.get("decisions", []):
            items.append(
                MemoryItem(scope=scope, type=MemoryType.DECISION, content=decision,
                           owner_id=owner_id, confidence=0.75, metadata=dict(base_meta))
            )
        for open_item in payload.get("open_items", []):
            items.append(
                MemoryItem(scope=scope, type=MemoryType.GOAL, content=open_item,
                           owner_id=owner_id, confidence=0.6, metadata=dict(base_meta))
            )
        return items
