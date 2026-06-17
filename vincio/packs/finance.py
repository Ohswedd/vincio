"""Finance domain pack (document analysis / metric extraction)."""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="finance",
    description="Grounded financial-document analysis and metric extraction.",
    role="financial analyst",
    objective="Extract and explain financial metrics strictly from the provided documents.",
    rules=[
        "Use only figures present in the sources; never estimate or extrapolate.",
        "Always report the currency and the reporting period for every figure.",
        "Cite the source for each extracted value.",
        "If a figure is not stated, return null for it rather than guessing.",
    ],
    soft_rules=[
        "Prefer the most recent period when multiple are present.",
        "Note any restatements or one-off items that affect comparability.",
    ],
    output_schema={
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "value": {"type": ["number", "null"]},
            "currency": {"type": "string"},
            "period": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["metric", "value", "currency", "period", "confidence"],
        "additionalProperties": False,
    },
    output_schema_name="finance_metric",
    policies={
        "answer_only_from_sources": True,
        "require_citations": True,
        "redact_pii_in_context": True,
    },
    rails=[{"name": "no_pii_leak", "kind": "safety", "direction": "output", "detectors": ["pii"]}],
    evaluators=["groundedness", "schema_validity", "lexical_overlap"],
    eval_cases=[
        EvalCase(
            id="fin_revenue",
            input="What was total revenue in the latest fiscal year?",
            expected={"metric": "total_revenue"},
            tags=["extraction"],
            difficulty="easy",
        ),
        EvalCase(
            id="fin_margin",
            input="What is the gross margin for Q3?",
            expected={"metric": "gross_margin"},
            tags=["ratio"],
            difficulty="medium",
        ),
        EvalCase(
            id="fin_missing",
            input="What is the customer churn rate?",
            expected={"value": None},
            tags=["not_stated"],
            difficulty="hard",
        ),
    ],
    tags=["finance", "extraction", "analysis"],
)
