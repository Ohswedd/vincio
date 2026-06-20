"""Code-review vertical pack (grounded diff review with security findings).

A full-stack starting point for an automated reviewer: grounded, severity-ranked
findings over a diff or file with code-aware chunking, a secrets rail so a review
never echoes a leaked credential, team-scoped memory for repository conventions,
and a golden eval set. The fuller counterpart to the lightweight ``engineering``
domain pack.
"""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="code_review",
    description="Grounded, severity-ranked code review with security findings over a diff.",
    role="staff software engineer reviewing a change",
    objective=(
        "Review the change, surface real defects and security issues with a concrete fix, "
        "and decide whether it is safe to approve."
    ),
    rules=[
        "Ground every finding in the provided diff or code; cite the file and line when you can.",
        "Rank severity honestly — do not inflate style nits into blockers.",
        "Propose the smallest correct fix; never invent an API that is not in the context.",
        "Flag any hardcoded secret, injection sink, or unsafe deserialization as a security risk.",
    ],
    soft_rules=[
        "Prefer standard-library and existing-dependency solutions.",
        "Call out missing tests when behavior changes.",
        "Respect remembered repository conventions.",
    ],
    definitions={
        "severity": "blocker > critical > major > minor > trivial",
    },
    output_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["blocker", "critical", "major", "minor", "trivial"],
                        },
                        "file": {"type": "string"},
                        "issue": {"type": "string"},
                        "fix": {"type": "string"},
                    },
                    "required": ["severity", "issue", "fix"],
                    "additionalProperties": False,
                },
            },
            "security_risk": {"type": "boolean"},
            "needs_tests": {"type": "boolean"},
            "approve": {"type": "boolean"},
        },
        "required": ["summary", "findings", "security_risk", "approve"],
        "additionalProperties": False,
    },
    output_schema_name="code_review",
    policies={"answer_only_from_sources": True},
    rails=[
        {"name": "no_secrets", "kind": "safety", "direction": "output", "detectors": ["secrets"]},
    ],
    evaluators=["groundedness", "schema_validity"],
    retrieval={"mode": "hybrid", "chunking": "code_aware", "top_k": 12},
    memory={"scope": "team", "strategy": "semantic"},
    purpose="legitimate_interest",
    eval_cases=[
        EvalCase(
            id="cr_nplus1",
            input="Review: the orders endpoint queries the DB inside a for-loop over line items.",
            expected={"approve": False},
            tags=["performance"],
            difficulty="medium",
        ),
        EvalCase(
            id="cr_secret",
            input="Review: the diff adds an AWS access key as a default constant in settings.py.",
            expected={"security_risk": True, "approve": False},
            tags=["security"],
            difficulty="easy",
        ),
        EvalCase(
            id="cr_race",
            input="Review: two coroutines increment a shared counter without a lock.",
            expected={"security_risk": False},
            tags=["concurrency"],
            difficulty="hard",
        ),
    ],
    tags=["engineering", "code-review", "security", "vertical"],
)
