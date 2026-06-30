"""Prompt linting: rules PROMPT001–PROMPT009."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from .ast import PromptAST
from .templates import PromptSpec

__all__ = ["LintFinding", "lint_spec", "lint_ast", "LINT_RULES"]


class LintFinding(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    hint: str = ""
    location: str = ""


_VAGUE_ROLES = {
    "assistant",
    "ai",
    "helpful assistant",
    "you are an ai",
    "chatbot",
    "helper",
    "ai assistant",
    "you are a helpful assistant",
}

_NEGATION_RE = re.compile(r"\b(never|do not|don't|must not|avoid|forbid(?:den)?)\b", re.IGNORECASE)
_AFFIRM_RE = re.compile(r"\b(always|must|shall|required to)\b", re.IGNORECASE)
_SCHEMA_PROSE_RE = re.compile(
    r"\b(respond|reply|answer|output|return)\b.{0,40}\b(json|valid json|a json object)\b",
    re.IGNORECASE,
)
_GROUNDED_RE = re.compile(
    r"\b(use only|based only on|provided (documents|sources|context)|cite|citation|grounded)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _conflicts(rule_a: str, rule_b: str) -> bool:
    """Detect a likely conflict: same subject with opposite polarity."""
    a_neg, b_neg = bool(_NEGATION_RE.search(rule_a)), bool(_NEGATION_RE.search(rule_b))
    if a_neg == b_neg:
        return False
    stripped_a = set(_normalize(_NEGATION_RE.sub(" ", _AFFIRM_RE.sub(" ", rule_a))).split())
    stripped_b = set(_normalize(_NEGATION_RE.sub(" ", _AFFIRM_RE.sub(" ", rule_b))).split())
    stop = {"the", "a", "an", "to", "of", "in", "and", "or", "for", "with", "be", "is", "are", "you"}
    stripped_a -= stop
    stripped_b -= stop
    if not stripped_a or not stripped_b:
        return False
    overlap = len(stripped_a & stripped_b) / min(len(stripped_a), len(stripped_b))
    return overlap >= 0.75


def lint_spec(spec: PromptSpec, *, grounded_task: bool | None = None) -> list[LintFinding]:
    findings: list[LintFinding] = []

    # PROMPT001: vague role
    role_norm = _normalize(spec.role)
    if not spec.role or role_norm in _VAGUE_ROLES:
        findings.append(
            LintFinding(
                code="PROMPT001",
                severity="warning",
                message="vague or missing role",
                hint="Give the model a specific operating role, e.g. 'insurance_claim_decision_engine'.",
                location="role",
            )
        )

    # PROMPT002: duplicate instruction
    seen: dict[str, int] = {}
    all_rules = spec.rules + spec.soft_rules
    for index, rule in enumerate(all_rules):
        key = _normalize(rule)
        if key in seen:
            findings.append(
                LintFinding(
                    code="PROMPT002",
                    severity="warning",
                    message=f"duplicate instruction: {rule!r}",
                    hint="Remove the duplicate; repeated rules waste tokens and dilute attention.",
                    location=f"rules[{index}]",
                )
            )
        seen.setdefault(key, index)

    # PROMPT003: conflicting constraints
    for i in range(len(all_rules)):
        for j in range(i + 1, len(all_rules)):
            if _conflicts(all_rules[i], all_rules[j]):
                findings.append(
                    LintFinding(
                        code="PROMPT003",
                        severity="error",
                        message=f"likely conflicting constraints: {all_rules[i]!r} vs {all_rules[j]!r}",
                        hint="Resolve the contradiction or scope each rule explicitly.",
                        location=f"rules[{i}]~rules[{j}]",
                    )
                )

    # PROMPT004: missing insufficient-evidence behavior for grounded tasks
    is_grounded = grounded_task if grounded_task is not None else bool(
        _GROUNDED_RE.search(" ".join(all_rules) + " " + spec.objective)
    )
    if is_grounded and not spec.insufficient_evidence_behavior:
        findings.append(
            LintFinding(
                code="PROMPT004",
                severity="warning",
                message="grounded task without insufficient-evidence behavior",
                hint="Set insufficient_evidence_behavior, e.g. 'If evidence is missing, report missing_information.'",
                location="insufficient_evidence_behavior",
            )
        )

    # PROMPT005: schema requested in prose while structured output is available
    if spec.output_schema is not None:
        for index, rule in enumerate(all_rules):
            if _SCHEMA_PROSE_RE.search(rule):
                findings.append(
                    LintFinding(
                        code="PROMPT005",
                        severity="warning",
                        message=f"JSON format requested in prose while a structured output schema is set: {rule!r}",
                        hint="Drop the prose rule; the schema is enforced natively.",
                        location=f"rules[{index}]",
                    )
                )

    # PROMPT007: no citation policy for grounded task
    if is_grounded and not spec.citation_policy:
        findings.append(
            LintFinding(
                code="PROMPT007",
                severity="warning",
                message="grounded task without a citation policy",
                hint="Set citation_policy, e.g. 'Cite evidence IDs for every claim.'",
                location="citation_policy",
            )
        )

    # PROMPT008: excessive examples
    if len(spec.examples) > 8:
        findings.append(
            LintFinding(
                code="PROMPT008",
                severity="warning",
                message=f"{len(spec.examples)} examples; more than 8 rarely improves quality and inflates cost",
                hint="Keep the most informative examples or use dynamic example selection.",
                location="examples",
            )
        )

    return findings


def lint_ast(ast: PromptAST) -> list[LintFinding]:
    """AST-level lints: cache layout (PROMPT006) and hidden rules (PROMPT009)."""
    findings: list[LintFinding] = []
    ordered = ast.ordered()

    # PROMPT006: dynamic content placed before cacheable prefix content.
    seen_volatile = False
    for node in ordered:
        if not node.stable:
            seen_volatile = True
        elif seen_volatile:
            findings.append(
                LintFinding(
                    code="PROMPT006",
                    severity="error",
                    message=f"stable node {node.kind!r} ordered after dynamic content; breaks prompt cache",
                    hint="Keep all stable content in the prefix, dynamic content in the suffix.",
                    location=node.kind,
                )
            )

    # PROMPT009: business rules embedded in the user task instead of developer rules.
    for node in ast.by_kind("user_task"):
        if _AFFIRM_RE.search(node.text) and len(node.text.split()) > 12 and (
            "must" in node.text.lower() or "always" in node.text.lower()
        ):
            findings.append(
                LintFinding(
                    code="PROMPT009",
                    severity="warning",
                    message="business rule appears only in the user message",
                    hint="Move durable rules to PromptSpec.rules so they live in the developer prefix.",
                    location="user_task",
                )
            )
    return findings


LINT_RULES: dict[str, str] = {
    "PROMPT001": "vague role",
    "PROMPT002": "duplicate instruction",
    "PROMPT003": "conflicting constraints",
    "PROMPT004": "missing insufficient-evidence behavior",
    "PROMPT005": "schema requested in prose while structured output available",
    "PROMPT006": "dynamic content placed before cacheable prefix",
    "PROMPT007": "no citation policy for grounded task",
    "PROMPT008": "excessive examples",
    "PROMPT009": "hidden business rule only in user message",
}
