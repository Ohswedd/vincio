"""Vincio governance & compliance (1.6).

Enterprise governance evidence generated *in the library, from the running
system* — no hosted compliance program, just the artifacts and controls as
files you own:

* **Model & system cards** (:mod:`~vincio.governance.cards`) — machine-readable
  documentation generated from the live config and eval evidence.
* **Compliance-framework mapping** (:mod:`~vincio.governance.frameworks`) —
  OWASP LLM Top 10 (2025), OWASP Agentic, NIST AI RMF, and MITRE ATLAS coverage
  backed by red-team and eval evidence.
* **AI-BOM** (:mod:`~vincio.governance.aibom`) — an AI bill of materials with
  SHA-256 model-hash verification, extending the shipped CycloneDX SBOM.
* **EU AI Act transparency** (:mod:`~vincio.governance.transparency`) —
  synthetic-content marking, AI-interaction disclosure, grounding-data summary.
* **Lineage & erasure** (:mod:`~vincio.governance.lineage`) — source → chunk →
  evidence → output, with right-to-erasure-by-source.
* **Data-residency routing** (:mod:`~vincio.governance.residency`) — refuse
  egress to disallowed provider regions, deterministically.
* **Tokenizer fertility** (:mod:`~vincio.governance.fertility`) — the non-English
  token tax, per language and tenant.

Everything is additive and reads from data Vincio already holds (the audit
chain, evidence ledger, eval reports, price table), so governance is a view over
the running system, not a parallel bookkeeping burden.
"""

from .aibom import AIBOM, AIComponent, generate_aibom, sha256_file, sha256_text
from .cards import (
    CardFormat,
    ModelCard,
    SystemCard,
    generate_model_card,
    generate_system_card,
)
from .fertility import FertilityTracker, LanguageFertility
from .frameworks import (
    CONTROL_CATALOG,
    ComplianceFramework,
    ComplianceMapper,
    ComplianceReport,
    Control,
    ControlCoverage,
    map_compliance,
)
from .lineage import ErasureResult, LineageIndex, LineageRecord
from .residency import ResidencyPolicy, residency_violation
from .transparency import (
    ProvenanceManifest,
    ai_disclosure,
    data_summary,
    mark_synthetic_content,
)

__all__ = [
    # cards
    "CardFormat",
    "ModelCard",
    "SystemCard",
    "generate_model_card",
    "generate_system_card",
    # frameworks
    "ComplianceFramework",
    "Control",
    "ControlCoverage",
    "ComplianceReport",
    "ComplianceMapper",
    "map_compliance",
    "CONTROL_CATALOG",
    # aibom
    "AIComponent",
    "AIBOM",
    "generate_aibom",
    "sha256_file",
    "sha256_text",
    # transparency
    "ProvenanceManifest",
    "mark_synthetic_content",
    "ai_disclosure",
    "data_summary",
    # lineage
    "LineageRecord",
    "LineageIndex",
    "ErasureResult",
    # residency
    "ResidencyPolicy",
    "residency_violation",
    # fertility
    "LanguageFertility",
    "FertilityTracker",
]
