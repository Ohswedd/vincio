"""Hierarchical retrieval: auto-merging parents and contextual prefixes.

:class:`AutoMergingIndex` wraps any ``Index`` for parent-document /
auto-merging retrieval over chunks produced by the ``hierarchical``
chunking strategy: small children are indexed for precision; when enough
siblings of one parent are retrieved, the children merge back into the
parent so the model sees one coherent unit instead of fragments.

:func:`contextualize_chunks` implements LLM-written chunk prefixes
("contextual retrieval"): each chunk gets a short model-written line
situating it within its document before indexing. Without a provider it
falls back to the same heuristic prefix as the ``contextual`` chunker.
"""

from __future__ import annotations

import json
from collections import defaultdict

from ..core.concurrency import gather_bounded
from ..core.tokens import count_tokens
from ..core.types import Chunk, Document, Message, ModelRequest
from ..providers.base import ModelProvider
from .indexes import Index, SearchHit, Where

__all__ = ["AutoMergingIndex", "contextualize_chunks"]


class AutoMergingIndex:
    """Index children, return parents when enough siblings match.

    ``add`` routes chunks by ``metadata["level"]``: parents go to an
    in-memory store, children to the inner index. ``search`` retrieves
    children, and any parent with at least ``merge_threshold`` of its
    children (and at least two) among the hits replaces them with one
    parent hit carrying the best child score.
    """

    name = "auto_merge"

    def __init__(self, inner: Index, *, merge_threshold: float = 0.5) -> None:
        self.inner = inner
        self.merge_threshold = merge_threshold
        self.parents: dict[str, Chunk] = {}
        self._children_per_parent: dict[str, set[str]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self.inner)

    async def add(self, chunks: list[Chunk]) -> None:
        children: list[Chunk] = []
        for chunk in chunks:
            if chunk.metadata.get("level") == "parent":
                self.parents[chunk.id] = chunk
                continue
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                self._children_per_parent[parent_id].add(chunk.id)
            children.append(chunk)
        if children:
            await self.inner.add(children)

    async def delete(self, chunk_ids: list[str]) -> int:
        removed = await self.inner.delete(chunk_ids)
        for chunk_id in chunk_ids:
            removed += int(self.parents.pop(chunk_id, None) is not None)
            for siblings in self._children_per_parent.values():
                siblings.discard(chunk_id)
        return removed

    async def search(
        self, query: str, *, top_k: int = 10, where: Where | None = None
    ) -> list[SearchHit]:
        # Over-fetch children so sibling co-occurrence is observable.
        hits = await self.inner.search(query, top_k=top_k * 3, where=where)
        by_parent: dict[str, list[SearchHit]] = defaultdict(list)
        for hit in hits:
            parent_id = hit.chunk.metadata.get("parent_id")
            if parent_id and parent_id in self.parents:
                by_parent[parent_id].append(hit)
        merged_children: set[str] = set()
        parent_hits: list[SearchHit] = []
        for parent_id, child_hits in by_parent.items():
            siblings = self._children_per_parent.get(parent_id, set())
            if len(siblings) < 2 or len(child_hits) / len(siblings) < self.merge_threshold:
                continue
            if len(child_hits) < 2:
                continue
            merged_children.update(h.chunk.id for h in child_hits)
            best = max(h.score for h in child_hits)
            parent = self.parents[parent_id]
            parent_hits.append(SearchHit(chunk=parent, score=best, source=self.name))
        kept = [h for h in hits if h.chunk.id not in merged_children]
        results = parent_hits + kept
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k]


_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {"context": {"type": "string"}},
    "required": ["context"],
    "additionalProperties": False,
}


async def contextualize_chunks(
    document: Document,
    chunks: list[Chunk],
    *,
    provider: ModelProvider | None = None,
    model: str | None = None,
    max_document_chars: int = 6000,
    max_concurrency: int = 4,
) -> list[Chunk]:
    """Prefix every chunk with a short line situating it in its document.

    With a provider, the prefix is model-written (Anthropic-style contextual
    retrieval); otherwise the heuristic prefix from the ``contextual``
    chunking strategy applies. Original text is preserved in
    ``metadata["original_text"]``; chunks already contextualized are skipped.
    """
    pending = [c for c in chunks if "original_text" not in c.metadata]
    if not pending:
        return chunks
    if provider is None or model is None:
        _apply_heuristic(document, pending)
        return chunks

    doc_excerpt = document.text[:max_document_chars]

    async def annotate(chunk: Chunk) -> None:
        request = ModelRequest(
            model=model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "Write one or two short sentences situating the chunk within "
                        "the document for search: what it is about and where it sits. "
                        "Be specific; never invent facts."
                    ),
                ),
                Message(
                    role="user",
                    content=f"<document>\n{doc_excerpt}\n</document>\n\n<chunk>\n{chunk.text}\n</chunk>",
                ),
            ],
            output_schema=_CONTEXT_SCHEMA,
            output_schema_name="chunk_context",
            temperature=0.0,
        )
        try:
            response = await provider.generate(request)
            payload = response.structured or json.loads(response.text)
            context = str(payload.get("context", "")).strip()
        except Exception:  # noqa: BLE001 - contextualization is best-effort
            context = ""
        if not context:
            _apply_heuristic(document, [chunk])
            return
        chunk.metadata = {**chunk.metadata, "original_text": chunk.text, "contextualized": "llm"}
        chunk.text = f"[{context}]\n{chunk.text}"
        chunk.token_count = count_tokens(chunk.text)

    await gather_bounded((annotate(chunk) for chunk in pending), limit=max_concurrency)
    return chunks


def _apply_heuristic(document: Document, chunks: list[Chunk]) -> None:
    from .chunking import _document_context

    doc_context = _document_context(document)
    for chunk in chunks:
        scope = " > ".join(chunk.section_path)
        prefix = " — ".join(p for p in (doc_context, scope) if p)
        if not prefix or chunk.text.startswith(f"[{prefix}]"):
            continue
        chunk.metadata = {**chunk.metadata, "original_text": chunk.text, "contextualized": "heuristic"}
        chunk.text = f"[{prefix}]\n{chunk.text}"
        chunk.token_count = count_tokens(chunk.text)
