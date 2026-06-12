"""GraphRAG: community detection and hierarchical summaries over the
entity graph, with global vs local query routing.

Local questions ("What did Acme agree with Beta?") walk entity paths via
:class:`~vincio.retrieval.graph_retrieval.EntityGraph`. Global questions
("What are the main themes across these contracts?") cannot be answered by
any single chunk — they need corpus-level structure. :class:`GraphRAG`
clusters the entity graph into communities (deterministic label
propagation), summarizes each community (extractive offline, LLM-written
when a provider is configured), builds a second hierarchy level by
collapsing the community graph, and routes each query to the right mode.

Community summaries become evidence with full provenance back to the
member chunks, so they budget and cite like any other evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict

from pydantic import BaseModel, Field

from ..context.compression import split_sentences
from ..context.scoring import lexical_similarity
from ..core.concurrency import gather_bounded
from ..core.tokens import count_tokens
from ..core.types import EvidenceItem, Message, ModelRequest, TrustLevel
from ..providers.base import ModelProvider
from .graph_retrieval import EntityGraph

__all__ = ["Community", "GraphRAG", "detect_communities"]

_GLOBAL_CUES = frozenset(
    "overall themes theme summarize summary main key trends pattern patterns "
    "common across all everything entire corpus general landscape compare".split()
)


class Community(BaseModel):
    id: str
    level: int = 0
    entities: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)


def detect_communities(graph: EntityGraph, *, max_iters: int = 10) -> dict[str, str]:
    """Label propagation over entity co-occurrence edges.

    Deterministic: entities are visited in sorted order and ties resolve to
    the lexicographically smallest label, so the same graph always yields
    the same communities. Returns ``entity -> community label``.
    """
    entities = sorted(graph.entity_chunks)
    labels = {entity: entity for entity in entities}
    for _ in range(max_iters):
        changed = False
        for entity in entities:
            neighbors = graph.edges.get(entity, {})
            if not neighbors:
                continue
            tally: dict[str, float] = defaultdict(float)
            for neighbor, weight in neighbors.items():
                if neighbor in labels:
                    tally[labels[neighbor]] += weight
            best = min(sorted(tally), key=lambda label: (-tally[label], label), default=None)
            if best is not None and best != labels[entity]:
                labels[entity] = best
                changed = True
        if not changed:
            break
    return labels


_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


class GraphRAG:
    def __init__(
        self,
        graph: EntityGraph,
        *,
        provider: ModelProvider | None = None,
        model: str | None = None,
        summary_sentences: int = 3,
        min_community_size: int = 2,
    ) -> None:
        self.graph = graph
        self.provider = provider
        self.model = model
        self.summary_sentences = summary_sentences
        self.min_community_size = min_community_size
        self.communities: dict[str, Community] = {}

    # -- build ---------------------------------------------------------------------

    async def build(self, *, levels: int = 2) -> list[Community]:
        """Detect communities, summarize them, and stack hierarchy levels."""
        labels = detect_communities(self.graph)
        members: dict[str, list[str]] = defaultdict(list)
        for entity, label in labels.items():
            members[label].append(entity)
        self.communities = {}
        level0: list[Community] = []
        for index, label in enumerate(sorted(members)):
            entities = sorted(members[label])
            if len(entities) < self.min_community_size:
                continue
            chunk_ids = sorted({cid for e in entities for cid in self.graph.entity_chunks[e]})
            community = Community(
                id=f"com_0_{index}",
                level=0,
                entities=[self.graph.display_names.get(e, e) for e in entities],
                chunk_ids=chunk_ids,
            )
            level0.append(community)
            self.communities[community.id] = community
        await gather_bounded((self._summarize(c) for c in level0), limit=4)

        # Higher levels: collapse the community graph and summarize groups
        # of communities (community-of-communities).
        current = level0
        for level in range(1, max(1, levels)):
            if len(current) < 2:
                break
            grouped = self._group_communities(current)
            if len(grouped) >= len(current):
                break
            parents: list[Community] = []
            for index, group in enumerate(grouped):
                parent = Community(
                    id=f"com_{level}_{index}",
                    level=level,
                    entities=sorted({e for c in group for e in c.entities}),
                    chunk_ids=sorted({cid for c in group for cid in c.chunk_ids}),
                    children=[c.id for c in group],
                )
                for child in group:
                    child.parent_id = parent.id
                parents.append(parent)
                self.communities[parent.id] = parent
            await gather_bounded((self._summarize(c) for c in parents), limit=4)
            current = parents
        return list(self.communities.values())

    def _group_communities(self, communities: list[Community]) -> list[list[Community]]:
        """Merge communities that share entities or co-occurrence edges."""
        normalized = {c.id: {e.lower() for e in c.entities} for c in communities}
        parent_of: dict[str, str] = {c.id: c.id for c in communities}

        def find(x: str) -> str:
            while parent_of[x] != x:
                parent_of[x] = parent_of[parent_of[x]]
                x = parent_of[x]
            return x

        ordered = sorted(communities, key=lambda c: c.id)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                connected = bool(normalized[a.id] & normalized[b.id]) or any(
                    self.graph.edges.get(ea, {}).get(eb)
                    for ea in normalized[a.id]
                    for eb in normalized[b.id]
                )
                if connected:
                    parent_of[find(a.id)] = find(b.id)
        groups: dict[str, list[Community]] = defaultdict(list)
        for community in ordered:
            groups[find(community.id)].append(community)
        return [groups[root] for root in sorted(groups)]

    async def _summarize(self, community: Community) -> None:
        if community.children:
            corpus = " ".join(
                self.communities[child].summary
                for child in community.children
                if self.communities.get(child) and self.communities[child].summary
            )
        else:
            corpus = " ".join(
                self.graph.chunks[cid].text for cid in community.chunk_ids if cid in self.graph.chunks
            )
        if not corpus.strip():
            community.summary = ", ".join(community.entities[:8])
            return
        if self.provider is not None and self.model is not None:
            try:
                community.summary = await self._llm_summary(community, corpus)
                if community.summary:
                    return
            except Exception:  # noqa: BLE001 - summary falls back to extractive
                pass
        community.summary = self._extractive_summary(community, corpus)

    def _extractive_summary(self, community: Community, corpus: str) -> str:
        anchor = " ".join(community.entities)
        sentences = split_sentences(corpus)
        scored = sorted(
            ((lexical_similarity(sentence, anchor), index, sentence) for index, sentence in enumerate(sentences)),
            key=lambda item: (-item[0], item[1]),
        )[: self.summary_sentences]
        # Restore document order so the summary reads coherently.
        return " ".join(sentence for _score, _index, sentence in sorted(scored, key=lambda item: item[1]))

    async def _llm_summary(self, community: Community, corpus: str) -> str:
        request = ModelRequest(
            model=self.model or "",
            messages=[
                Message(
                    role="system",
                    content=(
                        "Summarize what connects these passages in 2-4 sentences: the "
                        "entities involved, their relationships, and the key facts. "
                        "Use only the provided text."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Entities: {', '.join(community.entities[:12])}\n\n{corpus[:6000]}",
                ),
            ],
            output_schema=_SUMMARY_SCHEMA,
            output_schema_name="community_summary",
            temperature=0.0,
        )
        response = await self.provider.generate(request)  # type: ignore[union-attr]
        payload = response.structured or json.loads(response.text)
        return str(payload.get("summary", "")).strip()

    # -- query ---------------------------------------------------------------------

    def route(self, query: str) -> str:
        """``local`` when the query names known entities; ``global`` for
        corpus-level questions (themes, comparisons, summaries)."""
        words = {w.strip("?.,!").lower() for w in query.split()}
        if words & _GLOBAL_CUES:
            return "global"
        entities = self.graph.find_entities(query, limit=3)
        if any(entity in query.lower() for entity in entities):
            return "local"
        return "global" if self.communities else "local"

    async def retrieve(self, query: str, *, top_k: int = 8, mode: str | None = None) -> list[EvidenceItem]:
        mode = mode or self.route(query)
        if mode == "local":
            evidence = self.graph.retrieve(query, top_k=top_k)
            for item in evidence:
                item.metadata["graphrag_mode"] = "local"
            return evidence
        if not self.communities:
            await self.build()
        scored = sorted(
            (
                (lexical_similarity(f"{community.summary} {' '.join(community.entities)}", query), community)
                for community in self.communities.values()
                if community.summary
            ),
            key=lambda item: (-item[0], item[1].id),
        )
        evidence: list[EvidenceItem] = []
        for score, community in scored[:top_k]:
            if score <= 0.0 and evidence:
                break
            evidence.append(
                EvidenceItem(
                    id=f"{community.id}:S0",
                    source_id=community.id,
                    source_type="document",
                    text=community.summary,
                    trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
                    relevance=min(1.0, score * 2),
                    provenance=0.7,
                    token_cost=count_tokens(community.summary),
                    metadata={
                        "graphrag_mode": "global",
                        "community_id": community.id,
                        "community_level": community.level,
                        "entities": community.entities[:12],
                        "member_chunk_ids": community.chunk_ids[:32],
                    },
                )
            )
        return evidence
