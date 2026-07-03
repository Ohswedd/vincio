"""Prompt AST.

Prompts are structured trees, not strings. Each node carries stability
information used by the compiler to build cache-friendly layouts: stable
nodes form the prefix, volatile nodes go to the suffix.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.types import Example
from ..core.utils import stable_hash
from ..stability import deprecated

__all__ = [
    "NodeKind",
    "PromptNode",
    "SystemRoleNode",
    "ObjectiveNode",
    "RuleNode",
    "DefinitionNode",
    "SafetyPolicyNode",
    "OutputContractNode",
    "ExampleNode",
    "MemoryBlockNode",
    "EvidenceBlockNode",
    "ToolResultBlockNode",
    "UserTaskNode",
    "PromptAST",
]

NodeKind = Literal[
    "system_role",
    "objective",
    "rule",
    "definition",
    "safety_policy",
    "output_contract",
    "example",
    "memory_block",
    "evidence_block",
    "tool_result_block",
    "user_task",
]


class PromptNode(BaseModel):
    kind: NodeKind
    text: str = ""
    stable: bool = True  # part of the cacheable prefix?
    priority: int = 100  # ordering within its section (lower first)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def digest(self) -> str:
        """Stable content hash of the node (kind + text)."""
        return stable_hash({"kind": self.kind, "text": self.text})

    @property
    @deprecated(since="7.5", removed_in="8.0", alternative="digest()")
    def content_hash(self) -> str:
        """Deprecated name for :meth:`digest`."""
        return self.digest()


class SystemRoleNode(PromptNode):
    kind: NodeKind = "system_role"
    priority: int = 0


class ObjectiveNode(PromptNode):
    kind: NodeKind = "objective"
    priority: int = 10


class RuleNode(PromptNode):
    kind: NodeKind = "rule"
    priority: int = 20
    hard: bool = True
    source: str | None = None


class DefinitionNode(PromptNode):
    kind: NodeKind = "definition"
    priority: int = 30
    term: str = ""


class SafetyPolicyNode(PromptNode):
    kind: NodeKind = "safety_policy"
    priority: int = 25


class OutputContractNode(PromptNode):
    kind: NodeKind = "output_contract"
    priority: int = 40
    schema_def: dict[str, Any] | None = None
    format: str = "json"


class ExampleNode(PromptNode):
    kind: NodeKind = "example"
    priority: int = 50
    example: Example | None = None


class MemoryBlockNode(PromptNode):
    kind: NodeKind = "memory_block"
    stable: bool = False
    priority: int = 60
    items: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceBlockNode(PromptNode):
    kind: NodeKind = "evidence_block"
    stable: bool = False
    priority: int = 70
    items: list[dict[str, Any]] = Field(default_factory=list)


class ToolResultBlockNode(PromptNode):
    kind: NodeKind = "tool_result_block"
    stable: bool = False
    priority: int = 80
    items: list[dict[str, Any]] = Field(default_factory=list)


class UserTaskNode(PromptNode):
    kind: NodeKind = "user_task"
    stable: bool = False
    priority: int = 100


class PromptAST(BaseModel):
    """Ordered collection of prompt nodes with stable/volatile partitioning."""

    nodes: list[PromptNode] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add(self, node: PromptNode) -> PromptAST:
        self.nodes.append(node)
        return self

    def by_kind(self, kind: NodeKind) -> list[PromptNode]:
        return [n for n in self.nodes if n.kind == kind]

    @property
    def stable_nodes(self) -> list[PromptNode]:
        return [n for n in self.nodes if n.stable]

    @property
    def volatile_nodes(self) -> list[PromptNode]:
        return [n for n in self.nodes if not n.stable]

    def ordered(self) -> list[PromptNode]:
        """Cache-aware order: stable prefix (by priority), then volatile suffix."""
        stable = sorted(self.stable_nodes, key=lambda n: n.priority)
        volatile = sorted(self.volatile_nodes, key=lambda n: n.priority)
        return stable + volatile

    @property
    def spec_hash(self) -> str:
        return stable_hash([n.digest() for n in self.ordered()])
