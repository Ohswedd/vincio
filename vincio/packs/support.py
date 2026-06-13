"""Customer-support domain pack."""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="support",
    description="Customer-support triage and grounded resolution.",
    role="customer support assistant",
    objective="Classify the ticket, then resolve it using only the provided knowledge base.",
    rules=[
        "Answer using only the provided sources; never invent policy details.",
        "Classify every ticket into exactly one category.",
        "If the sources do not contain the answer, set needs_human to true and explain why.",
    ],
    soft_rules=[
        "Be concise and empathetic.",
        "Quote the relevant policy when you cite it.",
    ],
    output_schema={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["billing", "technical", "account", "feature_request", "other"],
            },
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "response": {"type": "string"},
            "needs_human": {"type": "boolean"},
        },
        "required": ["category", "priority", "response", "needs_human"],
        "additionalProperties": False,
    },
    output_schema_name="support_resolution",
    policies={"answer_only_from_sources": True, "require_citations": True},
    evaluators=["groundedness", "schema_validity", "semantic_similarity"],
    eval_cases=[
        EvalCase(
            id="support_billing_dup",
            input="I was charged twice for my subscription this month.",
            expected={"category": "billing", "needs_human": False},
            tags=["billing"],
            difficulty="easy",
        ),
        EvalCase(
            id="support_login",
            input="I can't log in even after resetting my password.",
            expected={"category": "technical"},
            tags=["technical"],
            difficulty="medium",
        ),
        EvalCase(
            id="support_unknown_policy",
            input="Do you offer student discounts in Brazil?",
            expected={"needs_human": True},
            tags=["escalation"],
            difficulty="hard",
        ),
    ],
    tags=["support", "cx", "triage"],
)
