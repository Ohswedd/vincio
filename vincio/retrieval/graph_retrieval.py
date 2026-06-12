"""Graph retrieval.

Builds an entity/claim graph from chunks: entities are nodes, chunks are
evidence nodes, edges connect co-occurring entities and link entities to the
chunks mentioning them. Queries walk entity paths (BFS) to surface evidence
that pure similarity search misses (multi-entity questions, relationship
questions). Pure-python; the Neo4j storage adapter offers the same API at
scale.
"""

from __future__ import annotations

from collections import defaultdict, deque

from pydantic import BaseModel, Field

from ..context.scoring import lexical_similarity
from ..core.types import Chunk, EvidenceItem, TrustLevel

__all__ = ["EntityGraph", "GraphPath"]


class GraphPath(BaseModel):
    entities: list[str]
    chunk_ids: list[str] = Field(default_factory=list)
    score: float = 0.0


def _normalize_entity(entity: str) -> str:
    return " ".join(entity.lower().split())


class EntityGraph:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        # entity -> chunk ids mentioning it
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        # entity -> co-occurring entities with weight
        self.edges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.display_names: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.entity_chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            normalized = [_normalize_entity(e) for e in chunk.entities if e.strip()]
            for raw, norm in zip(chunk.entities, normalized, strict=False):
                self.display_names.setdefault(norm, raw)
                self.entity_chunks[norm].add(chunk.id)
            for i in range(len(normalized)):
                for j in range(i + 1, len(normalized)):
                    a, b = normalized[i], normalized[j]
                    if a == b:
                        continue
                    self.edges[a][b] += 1.0
                    self.edges[b][a] += 1.0

    def find_entities(self, query: str, *, limit: int = 5) -> list[str]:
        """Entities best matching the query text."""
        query_lower = query.lower()
        scored: list[tuple[float, str]] = []
        for entity in self.entity_chunks:
            if entity in query_lower:
                scored.append((1.0 + len(entity) / 50, entity))
                continue
            similarity = lexical_similarity(entity, query)
            if similarity > 0.2:
                scored.append((similarity, entity))
        scored.sort(reverse=True)
        return [entity for _, entity in scored[:limit]]

    def paths_between(
        self, start: str, end: str, *, max_depth: int = 4
    ) -> list[GraphPath]:
        """BFS paths between two entities (Customer → Plan → Policy → Evidence)."""
        start, end = _normalize_entity(start), _normalize_entity(end)
        if start not in self.entity_chunks or end not in self.entity_chunks:
            return []
        queue: deque[list[str]] = deque([[start]])
        paths: list[GraphPath] = []
        visited_paths: set[tuple[str, ...]] = set()
        while queue and len(paths) < 8:
            path = queue.popleft()
            current = path[-1]
            if current == end and len(path) > 1:
                chunk_ids: list[str] = []
                for a, b in zip(path, path[1:], strict=False):
                    shared = self.entity_chunks[a] & self.entity_chunks[b]
                    chunk_ids.extend(sorted(shared))
                weight = sum(
                    self.edges[a].get(b, 0.0) for a, b in zip(path, path[1:], strict=False)
                )
                paths.append(
                    GraphPath(
                        entities=[self.display_names.get(e, e) for e in path],
                        chunk_ids=list(dict.fromkeys(chunk_ids)),
                        score=weight / max(1, len(path) - 1),
                    )
                )
                continue
            if len(path) >= max_depth:
                continue
            for neighbor in sorted(self.edges[current], key=self.edges[current].get, reverse=True)[:8]:
                if neighbor in path:
                    continue
                key = tuple(path + [neighbor])
                if key in visited_paths:
                    continue
                visited_paths.add(key)
                queue.append(path + [neighbor])
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths

    def retrieve(self, query: str, *, top_k: int = 8) -> list[EvidenceItem]:
        """Graph-first retrieval: find query entities, walk their
        neighborhoods (and inter-entity paths), collect linked chunks."""
        entities = self.find_entities(query)
        if not entities:
            return []
        chunk_scores: dict[str, float] = defaultdict(float)
        # Direct mentions.
        for rank, entity in enumerate(entities):
            for chunk_id in self.entity_chunks[entity]:
                chunk_scores[chunk_id] += 1.0 / (rank + 1)
        # Paths between top query entities.
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                for path in self.paths_between(entities[i], entities[j])[:3]:
                    for chunk_id in path.chunk_ids:
                        chunk_scores[chunk_id] += 0.5 * path.score / max(1.0, path.score + 1)
        ranked = sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        evidence: list[EvidenceItem] = []
        for chunk_id, score in ranked:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            evidence.append(
                EvidenceItem(
                    id=chunk.citation_ref,
                    source_id=chunk.document_id,
                    source_type="document",
                    text=chunk.text,
                    page=chunk.page,
                    section_path=chunk.section_path,
                    trust_level=TrustLevel.UNTRUSTED_DOCUMENT,
                    relevance=min(1.0, score / (ranked[0][1] or 1.0)),
                    provenance=0.9 if chunk.source_uri else 0.5,
                    token_cost=chunk.token_count,
                    metadata={"chunk_id": chunk.id, "graph_score": score, "retrieval": "graph"},
                )
            )
        return evidence
