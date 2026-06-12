"""Memory graph: typed nodes and edges over memory items.

Nodes: User, Tenant, Project, Entity, Preference, Goal, Decision, Fact,
Event. Edges: prefers, owns, decided, works_on, related_to, supersedes,
contradicts. Pure-python adjacency; the Neo4j storage adapter mirrors the
API for production graph workloads.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.types import MemoryItem, MemoryType
from ..core.utils import new_id

__all__ = ["MemoryNode", "MemoryEdge", "MemoryGraph", "NodeKind", "EdgeKind"]

NodeKind = Literal[
    "user", "agent", "tenant", "project", "entity", "preference", "goal", "decision", "fact", "event"
]
EdgeKind = Literal[
    "prefers", "owns", "decided", "works_on", "related_to", "supersedes", "contradicts"
]

_TYPE_TO_NODE: dict[MemoryType, NodeKind] = {
    MemoryType.FACT: "fact",
    MemoryType.PREFERENCE: "preference",
    MemoryType.GOAL: "goal",
    MemoryType.DECISION: "decision",
    MemoryType.SUMMARY: "event",
    MemoryType.ENTITY: "entity",
    MemoryType.RELATIONSHIP: "fact",
}

_TYPE_TO_EDGE: dict[MemoryType, EdgeKind] = {
    MemoryType.PREFERENCE: "prefers",
    MemoryType.GOAL: "works_on",
    MemoryType.DECISION: "decided",
    MemoryType.FACT: "related_to",
    MemoryType.SUMMARY: "related_to",
    MemoryType.ENTITY: "related_to",
    MemoryType.RELATIONSHIP: "related_to",
}


class MemoryNode(BaseModel):
    id: str = Field(default_factory=lambda: new_id("mnode"))
    kind: NodeKind
    label: str
    memory_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class MemoryEdge(BaseModel):
    id: str = Field(default_factory=lambda: new_id("medge"))
    kind: EdgeKind
    source: str
    target: str
    weight: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)


class MemoryGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[str, MemoryEdge] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)  # node -> edge ids
        self._label_index: dict[tuple[str, str], str] = {}  # (kind, label) -> node id

    def upsert_node(self, kind: NodeKind, label: str, **attributes: Any) -> MemoryNode:
        key = (kind, label.lower())
        if key in self._label_index:
            node = self.nodes[self._label_index[key]]
            node.attributes.update(attributes)
            return node
        node = MemoryNode(kind=kind, label=label, attributes=attributes)
        self.nodes[node.id] = node
        self._label_index[key] = node.id
        return node

    def add_edge(self, kind: EdgeKind, source: MemoryNode, target: MemoryNode, *, weight: float = 1.0, **attributes: Any) -> MemoryEdge:
        edge = MemoryEdge(kind=kind, source=source.id, target=target.id, weight=weight, attributes=attributes)
        self.edges[edge.id] = edge
        self._adjacency[source.id].add(edge.id)
        self._adjacency[target.id].add(edge.id)
        return edge

    def add_memory(self, item: MemoryItem) -> MemoryNode:
        """Project a memory item into the graph: owner node → typed edge →
        content node, plus related_to edges to its entities."""
        node_kind = _TYPE_TO_NODE.get(item.type, "fact")
        content_node = self.upsert_node(node_kind, item.content[:160], memory_id=item.id, confidence=item.confidence)
        content_node.memory_id = item.id
        if item.scope.value in ("tenant", "organization"):
            owner_kind: NodeKind = "tenant"
        elif item.scope.value == "agent":
            owner_kind = "agent"
        else:
            owner_kind = "user"
        owner_label = item.owner_id or f"anonymous_{owner_kind}"
        owner = self.upsert_node(owner_kind, owner_label)
        self.add_edge(_TYPE_TO_EDGE.get(item.type, "related_to"), owner, content_node, weight=item.confidence)
        for entity in item.entities:
            entity_node = self.upsert_node("entity", entity)
            self.add_edge("related_to", content_node, entity_node)
        if item.supersedes:
            for node in self.nodes.values():
                if node.memory_id == item.supersedes:
                    self.add_edge("supersedes", content_node, node)
                    break
        return content_node

    def neighbors(self, node_id: str, *, kinds: tuple[EdgeKind, ...] | None = None) -> list[tuple[MemoryEdge, MemoryNode]]:
        results: list[tuple[MemoryEdge, MemoryNode]] = []
        for edge_id in self._adjacency.get(node_id, ()):
            edge = self.edges[edge_id]
            if kinds and edge.kind not in kinds:
                continue
            other_id = edge.target if edge.source == node_id else edge.source
            results.append((edge, self.nodes[other_id]))
        return results

    def memories_about(self, entity_label: str) -> list[str]:
        """Memory ids linked to an entity."""
        key = ("entity", entity_label.lower())
        node_id = self._label_index.get(key)
        if node_id is None:
            return []
        memory_ids = []
        for _edge, node in self.neighbors(node_id):
            if node.memory_id:
                memory_ids.append(node.memory_id)
        return memory_ids

    def memories_for_owner(self, owner_label: str, *, edge_kind: EdgeKind | None = None) -> list[str]:
        for kind in ("user", "agent", "tenant"):
            node_id = self._label_index.get((kind, owner_label.lower()))
            if node_id:
                return [
                    node.memory_id
                    for edge, node in self.neighbors(node_id, kinds=(edge_kind,) if edge_kind else None)
                    if node.memory_id
                ]
        return []
