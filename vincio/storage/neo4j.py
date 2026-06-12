"""Neo4j graph store mirroring the MemoryGraph API at scale.
Requires ``pip install "vincio[graph]"``."""

from __future__ import annotations

from typing import Any

from ..core.errors import StorageError
from ..core.types import MemoryItem

__all__ = ["Neo4jGraphStore"]


class Neo4jGraphStore:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        *,
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise StorageError(
                'Neo4j support requires: pip install "vincio[graph]"'
            ) from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self._driver.close()

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def upsert_node(self, kind: str, label: str, **attributes: Any) -> None:
        self._run(
            "MERGE (n:VincioNode {kind: $kind, label: $label}) SET n += $attrs",
            kind=kind,
            label=label,
            attrs=attributes,
        )

    def add_edge(
        self, kind: str, source_label: str, target_label: str, *, weight: float = 1.0
    ) -> None:
        self._run(
            "MATCH (a:VincioNode {label: $src}), (b:VincioNode {label: $dst}) "
            "MERGE (a)-[r:RELATES {kind: $kind}]->(b) SET r.weight = $weight",
            src=source_label,
            dst=target_label,
            kind=kind,
            weight=weight,
        )

    def add_memory(self, item: MemoryItem) -> None:
        """Project a memory item: owner → typed edge → content, plus entities."""
        owner_kind = "tenant" if item.scope.value in ("tenant", "organization") else "user"
        owner_label = item.owner_id or f"anonymous_{owner_kind}"
        content_label = item.content[:160]
        self.upsert_node(owner_kind, owner_label)
        self.upsert_node(item.type.value, content_label, memory_id=item.id, confidence=item.confidence)
        self.add_edge("related_to", owner_label, content_label, weight=item.confidence)
        for entity in item.entities:
            self.upsert_node("entity", entity)
            self.add_edge("related_to", content_label, entity)

    def memories_about(self, entity_label: str) -> list[str]:
        rows = self._run(
            "MATCH (e:VincioNode {kind: 'entity', label: $label})--(m:VincioNode) "
            "WHERE m.memory_id IS NOT NULL RETURN m.memory_id AS memory_id",
            label=entity_label,
        )
        return [row["memory_id"] for row in rows]

    def paths_between(self, start: str, end: str, *, max_depth: int = 4) -> list[list[str]]:
        rows = self._run(
            f"MATCH p = shortestPath((a:VincioNode {{label: $start}})-[*..{max_depth}]-(b:VincioNode {{label: $end}})) "
            "RETURN [n IN nodes(p) | n.label] AS labels LIMIT 8",
            start=start,
            end=end,
        )
        return [row["labels"] for row in rows]
