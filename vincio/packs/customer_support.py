"""Customer-support vertical pack (grounded resolution with personalization).

A full-stack starting point for a production support assistant: grounded triage
and resolution over a knowledge base, PII-redacting output rails so one
customer's data never leaks to another, user-scoped memory so the assistant
remembers a customer's plan and history across turns and sessions, and a golden
eval set. The fuller counterpart to the lightweight ``support`` domain pack.
"""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="customer_support",
    description="Grounded, personalized customer-support triage and resolution.",
    role="customer support specialist",
    objective=(
        "Classify the request, resolve it from the knowledge base, and lay out the next "
        "steps — using what you remember about the customer where it helps."
    ),
    rules=[
        "Answer using only the provided knowledge base; never invent policy or pricing.",
        "Classify every request into exactly one category.",
        "If the knowledge base does not contain the answer, set needs_human to true and say why.",
        "Never disclose another customer's data.",
    ],
    soft_rules=[
        "Be concise, specific, and empathetic.",
        "Personalize using remembered context (plan, prior issues) when it is relevant.",
        "Offer concrete next steps the customer can take.",
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
            "resolution_steps": {"type": "array", "items": {"type": "string"}},
            "needs_human": {"type": "boolean"},
        },
        "required": ["category", "priority", "response", "needs_human"],
        "additionalProperties": False,
    },
    output_schema_name="support_resolution",
    policies={"answer_only_from_sources": True, "require_citations": True},
    rails=[
        {"name": "pii_redact", "kind": "safety", "direction": "output",
         "detectors": ["pii"], "action": "redact"},
        {"name": "no_secrets", "kind": "safety", "direction": "output", "detectors": ["secrets"]},
    ],
    evaluators=["groundedness", "schema_validity", "answer_relevance", "lexical_overlap"],
    retrieval={"mode": "hybrid", "chunking": "sentence_window", "top_k": 8},
    memory={"scope": "user", "strategy": "semantic"},
    purpose="contract",
    eval_cases=[
        EvalCase(
            id="cs_billing_dup",
            input="I was charged twice for my subscription this month.",
            expected={"category": "billing", "needs_human": False},
            tags=["billing"],
            difficulty="easy",
        ),
        EvalCase(
            id="cs_login",
            input="I can't log in even after resetting my password.",
            expected={"category": "technical"},
            tags=["technical"],
            difficulty="medium",
        ),
        EvalCase(
            id="cs_unknown",
            input="Do you offer student discounts in Brazil?",
            expected={"needs_human": True},
            tags=["escalation"],
            difficulty="hard",
        ),
    ],
    tags=["support", "cx", "personalization", "vertical"],
)
