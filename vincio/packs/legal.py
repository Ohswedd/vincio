"""Legal domain pack (contract review)."""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="legal",
    description="Grounded contract-clause review and risk flagging.",
    role="contracts analyst",
    objective="Identify clauses, assess risk, and recommend action using only the contract text.",
    rules=[
        "Quote or cite the exact clause text; never paraphrase a term into something it does not say.",
        "Assess risk only from what the document states.",
        "This is not legal advice — recommend review by qualified counsel for high-risk clauses.",
        "If the contract is silent on an issue, say so explicitly.",
    ],
    soft_rules=[
        "Flag auto-renewal, liability caps, indemnity, and termination terms.",
        "Note unusual or one-sided language.",
    ],
    output_schema={
        "type": "object",
        "properties": {
            "clause": {"type": "string"},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string"},
            "recommendation": {"type": "string"},
        },
        "required": ["clause", "risk_level", "rationale", "recommendation"],
        "additionalProperties": False,
    },
    output_schema_name="legal_review",
    policies={"answer_only_from_sources": True, "require_citations": True},
    evaluators=["groundedness", "schema_validity"],
    eval_cases=[
        EvalCase(
            id="legal_autorenew",
            input="Does this agreement renew automatically, and on what notice?",
            expected={"clause": "auto-renewal"},
            tags=["renewal"],
            difficulty="easy",
        ),
        EvalCase(
            id="legal_liability",
            input="Is there a limitation-of-liability cap?",
            expected={"risk_level": "medium"},
            tags=["liability"],
            difficulty="medium",
        ),
        EvalCase(
            id="legal_silent",
            input="What does the contract say about data-breach notification timelines?",
            expected={"recommendation": "review by counsel"},
            tags=["silent"],
            difficulty="hard",
        ),
    ],
    tags=["legal", "contracts", "review"],
)
