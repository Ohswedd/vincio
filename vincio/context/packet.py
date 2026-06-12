"""Context Packet: the universal unit passed to models, tools,
agents, evaluators, and traces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.types import Budget, Objective, PolicySet, UserInput
from ..core.utils import new_id, stable_hash, utcnow
from .ir import ContextIR

__all__ = ["ContextPacket"]


class ContextPacket(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ctx"))
    version: int = 1
    objective: Objective
    user_input: UserInput = Field(default_factory=UserInput)
    constraints: list[str] = Field(default_factory=list)
    memory_included: list[dict[str, Any]] = Field(default_factory=list)
    memory_excluded: list[dict[str, Any]] = Field(default_factory=list)
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ledger: list[dict[str, Any]] = Field(default_factory=list)
    tools_allowed: list[str] = Field(default_factory=list)
    tools_denied: list[str] = Field(default_factory=list)
    output_schema_ref: str | None = None
    budgets: Budget = Field(default_factory=Budget)
    policies: PolicySet = Field(default_factory=PolicySet)
    trace_parent_id: str | None = None
    token_count: int = 0
    spec_hash: str = ""
    created_at: Any = Field(default_factory=utcnow)
    excluded_report: list[dict[str, Any]] = Field(default_factory=list)
    budget_report: dict[str, Any] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_ir(
        cls,
        ir: ContextIR,
        *,
        excluded_report: list[dict[str, Any]] | None = None,
        budget_report: dict[str, Any] | None = None,
        conflicts: list[dict[str, Any]] | None = None,
        memory_excluded: list[dict[str, Any]] | None = None,
        trace_parent_id: str | None = None,
        token_count: int = 0,
    ) -> ContextPacket:
        packet = cls(
            objective=ir.objective,
            user_input=ir.input,
            constraints=[c.text for c in ir.constraints],
            memory_included=[
                {"id": m.id, "content": m.content, "scope": m.scope, "confidence": m.confidence}
                for m in ir.memory
            ],
            memory_excluded=memory_excluded or [],
            evidence_items=[
                {
                    "id": e.id,
                    "citation_ref": e.citation_ref,
                    "source_id": e.source_id,
                    "source_type": e.source_type,
                    "text": e.text,
                    "page": e.page,
                    "relevance": e.relevance,
                }
                for e in ir.evidence
            ],
            evidence_ledger=list(ir.evidence_ledger),
            tools_allowed=[t.name for t in ir.tool_specs],
            output_schema_ref=ir.output_contract.schema_ref,
            budgets=ir.budgets,
            policies=ir.policies,
            trace_parent_id=trace_parent_id,
            token_count=token_count,
            excluded_report=excluded_report or [],
            budget_report=budget_report or {},
            conflicts=conflicts or [],
            metadata=dict(ir.metadata),
        )
        packet.spec_hash = stable_hash(
            {
                "objective": packet.objective.text,
                "constraints": packet.constraints,
                "evidence": [e.get("id") for e in packet.evidence_items],
                "memory": [m.get("id") for m in packet.memory_included],
                "tools": packet.tools_allowed,
                "schema": packet.output_schema_ref,
            }
        )
        return packet
