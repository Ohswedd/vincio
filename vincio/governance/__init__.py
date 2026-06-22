"""Vincio governance & compliance.

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
* **Lineage & provable erasure** (:mod:`~vincio.governance.lineage`) — source →
  chunk → evidence → output, with right-to-erasure-by-source and a signed,
  content-bound :class:`~vincio.governance.lineage.ErasureProof`.
* **Consent & purpose** (:mod:`~vincio.governance.consent`) — a
  :class:`~vincio.governance.consent.ConsentLedger` binding data to a GDPR
  purpose and lawful basis, consulted by access decisions and memory recall.
* **Data-residency routing** (:mod:`~vincio.governance.residency`) — refuse
  egress to disallowed provider regions, deterministically.
* **Invariant verification** (:mod:`~vincio.governance.verification`) — a
  deterministic, in-process verifier that *proves* the governance invariants
  (containment, residency, budget, erasure) hold across their whole bounded,
  typed state space ahead of any run, and yields a minimal counterexample on a
  violation.
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
from .consent import (
    ConsentDecision,
    ConsentLedger,
    ConsentRecord,
    LawfulBasis,
    Purpose,
)
from .eu_ai_act import (
    AnnexIVBuilder,
    FRIAGenerator,
    RiskAssessment,
    RiskTier,
    RiskTierClassifier,
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
from .lineage import (
    ErasureProof,
    ErasureResult,
    LineageIndex,
    LineageRecord,
    build_erasure_proof,
    verify_erasure_proof,
)
from .privacy import (
    PrivacyAccountant,
    PrivacyBudget,
    PrivacyBudgetError,
    PrivacyDecision,
    PrivacyMechanism,
    PrivacyReport,
    PrivacyRow,
    PrivacySpend,
    gaussian_rdp,
    rdp_to_epsilon,
)
from .residency import ResidencyPolicy, infer_region_from_url, residency_violation
from .transparency import (
    ContentSigner,
    HmacSigner,
    ProvenanceManifest,
    ai_disclosure,
    data_summary,
    embed_provenance,
    extract_embedded_manifest,
    mark_synthetic_content,
    verify_embedded_manifest,
    verify_manifest,
    write_sidecar_manifest,
)
from .verification import (
    Counterexample,
    GovernanceVerifier,
    Invariant,
    InvariantResult,
    StateVariable,
    VerificationReport,
    budget_invariant,
    containment_invariant,
    default_invariants,
    erasure_invariant,
    residency_invariant,
    within_budget,
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
    # EU AI Act conformity pack
    "RiskTier",
    "RiskAssessment",
    "RiskTierClassifier",
    "AnnexIVBuilder",
    "FRIAGenerator",
    # aibom
    "AIComponent",
    "AIBOM",
    "generate_aibom",
    "sha256_file",
    "sha256_text",
    # transparency
    "ProvenanceManifest",
    "ContentSigner",
    "HmacSigner",
    "mark_synthetic_content",
    "verify_manifest",
    "embed_provenance",
    "extract_embedded_manifest",
    "verify_embedded_manifest",
    "write_sidecar_manifest",
    "ai_disclosure",
    "data_summary",
    # lineage & provable erasure
    "LineageRecord",
    "LineageIndex",
    "ErasureResult",
    "ErasureProof",
    "build_erasure_proof",
    "verify_erasure_proof",
    # consent & purpose
    "ConsentLedger",
    "ConsentRecord",
    "ConsentDecision",
    "Purpose",
    "LawfulBasis",
    # differential-privacy accounting
    "PrivacyAccountant",
    "PrivacyBudget",
    "PrivacyBudgetError",
    "PrivacyDecision",
    "PrivacyMechanism",
    "PrivacyReport",
    "PrivacyRow",
    "PrivacySpend",
    "gaussian_rdp",
    "rdp_to_epsilon",
    # residency
    "ResidencyPolicy",
    "residency_violation",
    "infer_region_from_url",
    # formal verification of governance invariants
    "GovernanceVerifier",
    "VerificationReport",
    "InvariantResult",
    "Counterexample",
    "Invariant",
    "StateVariable",
    "containment_invariant",
    "residency_invariant",
    "budget_invariant",
    "erasure_invariant",
    "default_invariants",
    "within_budget",
    # fertility
    "LanguageFertility",
    "FertilityTracker",
]
