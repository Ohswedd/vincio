"""PromptSpec: the typed, declarative prompt definition.

A PromptSpec is compiled into a PromptAST and then rendered by the prompt
compiler. Specs support ``${variable}`` interpolation with declared,
type-checked variables — undeclared or missing variables are compile errors,
not silent empty strings.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import PromptError
from ..core.types import Example
from ..core.utils import stable_hash
from .ast import (
    DefinitionNode,
    EvidenceBlockNode,
    ExampleNode,
    MemoryBlockNode,
    ObjectiveNode,
    OutputContractNode,
    PromptAST,
    RuleNode,
    SafetyPolicyNode,
    SystemRoleNode,
    ToolResultBlockNode,
    UserTaskNode,
)

__all__ = ["PromptVariable", "PromptSpec"]

_VAR_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptVariable(BaseModel):
    name: str
    type: str = "str"  # str | int | float | bool | list | dict
    required: bool = True
    default: Any = None
    description: str | None = None

    def check(self, value: Any) -> Any:
        expected = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }.get(self.type, str)
        if value is None:
            if self.required and self.default is None:
                raise PromptError(f"missing required prompt variable {self.name!r}")
            return self.default
        if not isinstance(value, expected):  # type: ignore[arg-type]
            raise PromptError(
                f"prompt variable {self.name!r} expected {self.type}, got {type(value).__name__}"
            )
        return value


class PromptSpec(BaseModel):
    """Declarative prompt definition compiled to an AST."""

    name: str = "prompt"
    role: str = ""
    objective: str = ""
    rules: list[str] = Field(default_factory=list)
    soft_rules: list[str] = Field(default_factory=list)
    definitions: dict[str, str] = Field(default_factory=dict)
    safety_policies: list[str] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    output_format: str = "text"  # json | markdown | text | tool | native
    output_instructions: str = ""
    citation_policy: str = ""  # e.g. "Cite evidence IDs in square brackets."
    insufficient_evidence_behavior: str = ""  # e.g. "Say you cannot answer."
    reasoning_mode: str = "direct"  # direct | plan | evidence_first
    variables: list[PromptVariable] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- variables ---------------------------------------------------------------

    def _declared_names(self) -> set[str]:
        return {v.name for v in self.variables}

    def _all_text_fields(self) -> list[str]:
        texts = [self.role, self.objective, self.output_instructions, self.citation_policy,
                 self.insufficient_evidence_behavior]
        texts.extend(self.rules)
        texts.extend(self.soft_rules)
        texts.extend(self.definitions.values())
        texts.extend(self.safety_policies)
        return texts

    def referenced_variables(self) -> set[str]:
        names: set[str] = set()
        for text in self._all_text_fields():
            names.update(_VAR_RE.findall(text or ""))
        return names

    def substitute(self, values: dict[str, Any] | None = None) -> PromptSpec:
        """Return a copy with ``${var}`` placeholders resolved and type-checked."""
        values = values or {}
        referenced = self.referenced_variables()
        declared = self._declared_names()
        undeclared = referenced - declared
        if undeclared:
            raise PromptError(
                f"undeclared prompt variables referenced: {sorted(undeclared)}; "
                "declare them in PromptSpec.variables"
            )
        resolved: dict[str, str] = {}
        for variable in self.variables:
            value = variable.check(values.get(variable.name))
            resolved[variable.name] = "" if value is None else str(value)

        def sub(text: str) -> str:
            return _VAR_RE.sub(lambda m: resolved.get(m.group(1), ""), text or "")

        return self.model_copy(
            update={
                "role": sub(self.role),
                "objective": sub(self.objective),
                "rules": [sub(r) for r in self.rules],
                "soft_rules": [sub(r) for r in self.soft_rules],
                "definitions": {k: sub(v) for k, v in self.definitions.items()},
                "safety_policies": [sub(p) for p in self.safety_policies],
                "output_instructions": sub(self.output_instructions),
                "citation_policy": sub(self.citation_policy),
                "insufficient_evidence_behavior": sub(self.insufficient_evidence_behavior),
            }
        )

    # -- AST ---------------------------------------------------------------------

    def build_stable_ast(self) -> PromptAST:
        """The cacheable prefix: the nodes that depend only on the spec.

        Role, objective, rules, safety policies, definitions, the output
        contract, and examples — none of which vary with the per-call task or
        context. Rendering these once and reusing them is what the compiler's
        render program does.
        """
        ast = PromptAST(metadata={"spec_name": self.name})
        if self.role:
            ast.add(SystemRoleNode(text=self.role))
        if self.objective:
            ast.add(ObjectiveNode(text=self.objective))
        for rule in self.rules:
            ast.add(RuleNode(text=rule, hard=True))
        for rule in self.soft_rules:
            ast.add(RuleNode(text=rule, hard=False))
        if self.citation_policy:
            ast.add(RuleNode(text=self.citation_policy, hard=True, metadata={"citation": True}))
        if self.insufficient_evidence_behavior:
            ast.add(
                RuleNode(
                    text=self.insufficient_evidence_behavior,
                    hard=True,
                    metadata={"insufficient_evidence": True},
                )
            )
        for policy in self.safety_policies:
            ast.add(SafetyPolicyNode(text=policy))
        for term, definition in self.definitions.items():
            ast.add(DefinitionNode(term=term, text=f"{term}: {definition}"))
        if self.output_schema is not None or self.output_instructions or self.output_format != "text":
            ast.add(
                OutputContractNode(
                    text=self.output_instructions,
                    schema_def=self.output_schema,
                    format=self.output_format,
                )
            )
        for example in self.examples:
            ast.add(ExampleNode(example=example, text=f"{example.input} -> {example.output}"))
        return ast

    def build_volatile_ast(
        self,
        *,
        user_task: str = "",
        memory_items: list[dict[str, Any]] | None = None,
        evidence_items: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> PromptAST:
        """The per-call suffix: memory, evidence, tool results, and the task."""
        ast = PromptAST(metadata={"spec_name": self.name})
        if memory_items:
            ast.add(MemoryBlockNode(items=memory_items))
        if evidence_items:
            ast.add(EvidenceBlockNode(items=evidence_items))
        if tool_results:
            ast.add(ToolResultBlockNode(items=tool_results))
        if user_task:
            ast.add(UserTaskNode(text=user_task))
        return ast

    @property
    def spec_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"metadata"}))
