"""Financial KYC / AML vertical pack (customer due diligence and risk rating).

A full-stack starting point for know-your-customer and anti-money-laundering
review: grounded risk rating over onboarding documents, screening notes, and
transaction summaries, with PII-redacting output rails, customer-scoped memory
for case continuity, an in-jurisdiction residency posture, and a golden eval
set. Decisions are advisory inputs to a compliance officer — the schema records
a SAR recommendation, never files one.
"""

from __future__ import annotations

from ..evals.datasets import EvalCase
from .base import Pack

PACK = Pack(
    name="kyc",
    description="Customer due diligence and AML risk rating with a compliance officer in the loop.",
    role="financial crime compliance analyst",
    objective=(
        "Assess customer risk from the provided due-diligence material, flag sanctions / PEP "
        "exposure, and recommend whether a suspicious-activity report is warranted."
    ),
    rules=[
        "Base the risk rating only on the provided documents and screening results; never "
        "fabricate a sanctions or PEP match.",
        "Cite the source for every risk factor.",
        "A SAR recommendation is advisory input for a compliance officer, not a filing.",
        "If the material is insufficient to rate risk, say so and recommend escalation.",
    ],
    soft_rules=[
        "Weight recent adverse media and transaction anomalies most heavily.",
        "Distinguish a confirmed match from a possible name collision.",
    ],
    definitions={
        "PEP": "politically exposed person.",
        "SAR": "suspicious activity report.",
    },
    output_schema={
        "type": "object",
        "properties": {
            "risk_rating": {"type": "string", "enum": ["low", "medium", "high"]},
            "sanctions_hit": {"type": "boolean"},
            "pep": {"type": "boolean"},
            "sar_recommended": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": ["risk_rating", "sanctions_hit", "pep", "sar_recommended", "rationale"],
        "additionalProperties": False,
    },
    output_schema_name="kyc_assessment",
    policies={
        "answer_only_from_sources": True,
        "require_citations": True,
        "redact_pii_in_context": True,
    },
    rails=[
        {"name": "pii_redact", "kind": "safety", "direction": "output",
         "detectors": ["pii"], "action": "redact"},
        {"name": "no_secrets", "kind": "safety", "direction": "output", "detectors": ["secrets"]},
    ],
    evaluators=["groundedness", "schema_validity", "citation_accuracy"],
    retrieval={"mode": "hybrid", "chunking": "recursive", "top_k": 10},
    memory={"scope": "user", "strategy": "semantic"},
    residency=["us"],
    purpose="legal_obligation",
    eval_cases=[
        EvalCase(
            id="kyc_clean",
            input="Rate the risk for a domestic retail customer with consistent payroll deposits.",
            expected={"risk_rating": "low", "sar_recommended": False},
            tags=["low_risk"],
            difficulty="easy",
        ),
        EvalCase(
            id="kyc_sanctions",
            input="Screening returned a confirmed match against an OFAC-listed entity.",
            expected={"sanctions_hit": True, "risk_rating": "high"},
            tags=["sanctions"],
            difficulty="medium",
        ),
        EvalCase(
            id="kyc_structuring",
            input="The account shows repeated cash deposits just under the reporting threshold.",
            expected={"sar_recommended": True},
            tags=["structuring", "aml"],
            difficulty="hard",
        ),
    ],
    tags=["finance", "kyc", "aml", "compliance", "vertical"],
)
