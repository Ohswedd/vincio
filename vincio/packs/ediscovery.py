"""Legal e-discovery vertical pack (responsiveness + privilege review).

A full-stack starting point for document review in litigation: grounded
responsiveness and privilege calls over produced documents, parent-document
retrieval so long records stay coherent, team-scoped memory for consistent
coding decisions, an in-jurisdiction residency posture, and a golden eval set.
Privilege calls are advisory — the schema records the basis so counsel can
confirm.
"""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="ediscovery",
    description="Responsiveness and privilege review over a litigation document set.",
    role="e-discovery review analyst",
    objective=(
        "Decide whether each document is responsive to the request and whether it is "
        "privileged, citing the exact passages that drive the call."
    ),
    rules=[
        "Base every call only on the document text; never speculate about facts not present.",
        "Mark a document privileged only when the text shows a basis (attorney-client "
        "communication or work product); record that basis.",
        "Quote the key passages that drive a responsiveness or privilege call.",
        "This is an attorney-supervised determination, not a final legal conclusion.",
    ],
    soft_rules=[
        "Prefer over-inclusion of privilege candidates for counsel review when uncertain.",
        "Note custodian and date when the document states them.",
    ],
    definitions={
        "responsive": "within the scope of a discovery request.",
        "work_product": "material prepared in anticipation of litigation.",
    },
    output_schema={
        "type": "object",
        "properties": {
            "responsive": {"type": "boolean"},
            "privileged": {"type": "boolean"},
            "privilege_basis": {
                "type": "string",
                "enum": ["attorney_client", "work_product", "none"],
            },
            "key_passages": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["responsive", "privileged", "privilege_basis", "confidence"],
        "additionalProperties": False,
    },
    output_schema_name="ediscovery_review",
    policies={"answer_only_from_sources": True, "require_citations": True},
    rails=[
        {"name": "no_secrets", "kind": "safety", "direction": "output", "detectors": ["secrets"]},
    ],
    evaluators=["groundedness", "schema_validity", "citation_accuracy"],
    retrieval={"mode": "hybrid", "chunking": "parent_document", "top_k": 12},
    memory={"scope": "team", "strategy": "semantic"},
    residency=["us"],
    purpose="legal_obligation",
    eval_cases=[
        EvalCase(
            id="ed_responsive",
            input="Is this email about the Q3 pricing change responsive to the pricing-practices request?",
            expected={"responsive": True},
            tags=["responsiveness"],
            difficulty="easy",
        ),
        EvalCase(
            id="ed_privilege",
            input="Is this memo from outside counsel advising on the merger privileged?",
            expected={"privileged": True, "privilege_basis": "attorney_client"},
            tags=["privilege"],
            difficulty="medium",
        ),
        EvalCase(
            id="ed_nonresponsive",
            input="Is this cafeteria-menu announcement responsive to the pricing-practices request?",
            expected={"responsive": False},
            tags=["non_responsive"],
            difficulty="hard",
        ),
    ],
    tags=["legal", "ediscovery", "privilege", "vertical"],
)
