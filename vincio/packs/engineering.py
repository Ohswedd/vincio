"""Software-engineering domain pack (code review / bug triage)."""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="engineering",
    description="Code review and bug triage over a codebase or diff.",
    role="senior software engineer reviewing code",
    objective="Review the change or report, surface real defects, and propose a concrete fix.",
    rules=[
        "Ground every finding in the provided code or logs; cite the file and line when you can.",
        "Rank severity honestly — do not inflate style nits to bugs.",
        "Propose the smallest correct fix; never invent APIs that are not in the context.",
    ],
    soft_rules=[
        "Prefer standard-library and existing-dependency solutions.",
        "Call out missing tests when behavior changes.",
    ],
    definitions={
        "severity": "blocker > critical > major > minor > trivial",
    },
    output_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["blocker", "critical", "major", "minor", "trivial"],
            },
            "files": {"type": "array", "items": {"type": "string"}},
            "suggested_fix": {"type": "string"},
            "needs_tests": {"type": "boolean"},
        },
        "required": ["summary", "severity", "suggested_fix"],
        "additionalProperties": False,
    },
    output_schema_name="engineering_review",
    policies={"answer_only_from_sources": True},
    evaluators=["groundedness", "schema_validity"],
    eval_cases=[
        EvalCase(
            id="eng_nplus1",
            input="Review: the orders endpoint queries the DB inside a for-loop over line items.",
            expected={"severity": "major"},
            tags=["performance"],
            difficulty="medium",
        ),
        EvalCase(
            id="eng_null_deref",
            input="Bug: NullPointerException in CartService.total() when the cart is empty.",
            expected={"needs_tests": True},
            tags=["bug"],
            difficulty="easy",
        ),
        EvalCase(
            id="eng_race",
            input="Review: two coroutines increment a shared counter without a lock.",
            expected={"severity": "critical"},
            tags=["concurrency"],
            difficulty="hard",
        ),
    ],
    tags=["engineering", "code-review", "bug-triage"],
)
