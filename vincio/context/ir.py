"""Context Intermediate Representation.

The CIR is the provider-neutral representation of everything that will be
rendered into a model request: objective, instructions, constraints,
examples, input, memory, evidence, tools, output contract, budgets, policies.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.types import (
    Budget,
    Constraint,
    EvidenceItem,
    Example,
    Instruction,
    MemoryItem,
    Objective,
    PolicySet,
    ToolSpec,
    UserInput,
)
from ..core.utils import stable_hash

__all__ = ["OutputContractRef", "ContextIR"]


class OutputContractRef(BaseModel):
    """Reference to the output contract carried inside the CIR."""

    schema_ref: str | None = None  # name of the registered schema
    schema_def: dict[str, Any] | None = None
    format: str = "text"  # json | markdown | text | tool | native
    instructions: str = ""


class ContextIR(BaseModel):
    objective: Objective
    instructions: list[Instruction] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    input: UserInput = Field(default_factory=UserInput)
    memory: list[MemoryItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_specs: list[ToolSpec] = Field(default_factory=list)
    output_contract: OutputContractRef = Field(default_factory=OutputContractRef)
    budgets: Budget = Field(default_factory=Budget)
    policies: PolicySet = Field(default_factory=PolicySet)
    evidence_ledger: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ir_hash(self) -> str:
        return stable_hash(
            {
                "objective": self.objective.text,
                "instructions": [i.text for i in self.instructions],
                "constraints": [c.text for c in self.constraints],
                "examples": [(e.input, e.output) for e in self.examples],
                "input": self.input.text,
                "memory": [m.content for m in self.memory],
                "evidence": [e.text for e in self.evidence],
                "tools": [t.name for t in self.tool_specs],
                "schema": self.output_contract.schema_def,
            }
        )

    def evidence_as_items(self) -> list[dict[str, Any]]:
        """Evidence rendered for prompt blocks, ledger-style when available."""
        if self.evidence_ledger:
            return [
                {
                    "id": entry.get("id"),
                    "text": f"{entry.get('claim')} (source: {entry.get('source')}, confidence: {entry.get('confidence')})",
                }
                for entry in self.evidence_ledger
            ]
        return [
            {"id": e.citation_ref, "text": e.text or ""}
            for e in self.evidence
        ]

    def memory_as_items(self) -> list[dict[str, Any]]:
        return [{"id": m.id, "text": m.content} for m in self.memory]
