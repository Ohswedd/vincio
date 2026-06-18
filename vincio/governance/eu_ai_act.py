"""EU AI Act conformity pack — Annex IV docs as generated, cited artifacts.

The document-generation engine's first governance application. A
:class:`RiskTierClassifier` places a configured app into the Act's risk tiers;
an :class:`AnnexIVBuilder` renders the **Annex IV technical documentation** and a
:class:`FRIAGenerator` the Article 27 **fundamental-rights impact assessment** —
both as cited documents through the
:class:`~vincio.generation.builder.DocumentBuilder`, every field drawn from the
live config, the model/system cards, the compliance matrix, and the eval/red-team
evidence Vincio already holds. The pack is a *view* over the running system,
regenerated on every config change and recorded as a ``conformity_doc`` audit
event. Deadline-agnostic and pluggable, like the existing card formats.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .cards import generate_model_card, generate_system_card
from .frameworks import ComplianceMapper

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.app import ContextApp
    from ..core.config import VincioConfig
    from ..evals.redteam import RedTeamReport
    from ..evals.reports import EvalReport
    from ..generation.render import DocumentArtifact, RenderFormat

__all__ = [
    "RiskTier",
    "RiskAssessment",
    "RiskTierClassifier",
    "AnnexIVBuilder",
    "FRIAGenerator",
    "ANNEX_III_HIGH_RISK_DOMAINS",
    "PROHIBITED_PRACTICES",
]


class RiskTier(StrEnum):
    PROHIBITED = "prohibited"
    HIGH_RISK = "high_risk"
    LIMITED = "limited"
    MINIMAL = "minimal"


# Annex III high-risk domain cues (Art. 6 + Annex III). Matched against the
# declared purpose/domains — advisory, the org makes the final classification.
ANNEX_III_HIGH_RISK_DOMAINS = {
    "biometric identification",
    "critical infrastructure",
    "education",
    "employment",
    "recruitment",
    "creditworthiness",
    "credit scoring",
    "essential services",
    "insurance",
    "law enforcement",
    "migration",
    "asylum",
    "border control",
    "administration of justice",
    "democratic processes",
}

# Article 5 prohibited practices (cues).
PROHIBITED_PRACTICES = {
    "social scoring",
    "subliminal manipulation",
    "exploiting vulnerabilities",
    "real-time remote biometric identification",
    "untargeted facial scraping",
    "emotion recognition in the workplace",
    "biometric categorization of sensitive attributes",
    "predictive policing of individuals",
}


class RiskAssessment(BaseModel):
    """An advisory EU AI Act risk-tier classification with its rationale."""

    tier: RiskTier
    rationale: str
    drivers: list[str] = Field(default_factory=list)
    matched_high_risk_domains: list[str] = Field(default_factory=list)
    matched_prohibited_practices: list[str] = Field(default_factory=list)
    human_oversight: bool = False
    interacts_with_humans: bool = False
    generates_content: bool = False
    obligations: list[str] = Field(default_factory=list)


_OBLIGATIONS = {
    RiskTier.PROHIBITED: [
        "Practice is prohibited under Article 5 — do not deploy.",
    ],
    RiskTier.HIGH_RISK: [
        "Risk-management system (Art. 9)",
        "Data governance (Art. 10)",
        "Technical documentation — Annex IV (Art. 11)",
        "Record-keeping / logging (Art. 12)",
        "Transparency & information to deployers (Art. 13)",
        "Human oversight (Art. 14)",
        "Accuracy, robustness, cybersecurity (Art. 15)",
        "Fundamental-rights impact assessment for deployers (Art. 27)",
    ],
    RiskTier.LIMITED: [
        "Transparency: disclose AI interaction (Art. 50(1))",
        "Mark synthetic content as AI-generated (Art. 50(2))",
    ],
    RiskTier.MINIMAL: [
        "No mandatory obligations; voluntary codes of conduct encouraged.",
    ],
}


class RiskTierClassifier:
    """Place an app into the EU AI Act risk tiers from its declared profile.

    Advisory by design — it infers a tier from the declared ``purpose``,
    ``domains``, ``prohibited_practices``, and oversight controls (and a
    :class:`~vincio.core.app.ContextApp`'s live config), and records the
    assessment as metadata; the final classification is the operator's.
    """

    def __init__(
        self,
        *,
        purpose: str = "",
        domains: list[str] | None = None,
        prohibited_practices: list[str] | None = None,
        human_oversight: bool | None = None,
        interacts_with_humans: bool = True,
        generates_content: bool = True,
    ) -> None:
        self.purpose = purpose
        self.domains = [d.lower() for d in (domains or [])]
        self.declared_prohibited = [p.lower() for p in (prohibited_practices or [])]
        self.human_oversight = human_oversight
        self.interacts_with_humans = interacts_with_humans
        self.generates_content = generates_content

    def classify(self, target: ContextApp | VincioConfig | None = None) -> RiskAssessment:
        haystack = " ".join([self.purpose, *self.domains, *self.declared_prohibited]).lower()
        # Canonical cues found in the declared profile, plus any explicitly
        # declared prohibited practice (canonical or custom) — an operator's
        # explicit prohibition is always honored, never silently dropped.
        prohibited = sorted(
            {p for p in PROHIBITED_PRACTICES if p in haystack} | set(self.declared_prohibited)
        )
        high_risk = sorted({d for d in ANNEX_III_HIGH_RISK_DOMAINS if d in haystack})

        oversight = self.human_oversight
        if oversight is None:
            oversight = _config_has_oversight(target)

        drivers: list[str] = []
        if self.purpose:
            drivers.append(f"declared purpose: {self.purpose}")
        if self.domains:
            drivers.append(f"declared domains: {', '.join(self.domains)}")

        if prohibited:
            tier = RiskTier.PROHIBITED
            rationale = "Declared use matches an Article 5 prohibited practice."
        elif high_risk:
            tier = RiskTier.HIGH_RISK
            rationale = f"Declared domain falls under Annex III high-risk: {', '.join(high_risk)}."
        elif self.interacts_with_humans or self.generates_content:
            tier = RiskTier.LIMITED
            rationale = "Interacts with humans and/or generates content — transparency duties apply."
        else:
            tier = RiskTier.MINIMAL
            rationale = "No high-risk domain, prohibited practice, or transparency trigger detected."

        return RiskAssessment(
            tier=tier,
            rationale=rationale,
            drivers=drivers,
            matched_high_risk_domains=high_risk,
            matched_prohibited_practices=prohibited,
            human_oversight=bool(oversight),
            interacts_with_humans=self.interacts_with_humans,
            generates_content=self.generates_content,
            obligations=list(_OBLIGATIONS[tier]),
        )


def _config_has_oversight(target: ContextApp | VincioConfig | None) -> bool:
    if target is None:
        return False
    cfg = getattr(target, "config", None) or target
    policies = getattr(cfg, "policies", None)
    return bool(policies and (policies.require_citations or policies.safety != "minimal"))


def _matrix_table(report: Any) -> dict[str, Any]:
    """A compliance coverage matrix as a generation-engine table dict."""
    rows = [
        [c.framework.value, c.control_id, c.title, c.status, "; ".join(c.evidence) or "—"]
        for c in report.coverage
    ]
    return {"title": "Control coverage", "columns": ["Framework", "Control", "Title", "Status", "Evidence"], "rows": rows}


class AnnexIVBuilder:
    """Render EU AI Act **Annex IV** technical documentation as a cited document.

    Every section is drawn from the live system: the model/system cards, the
    risk-tier assessment, the compliance matrix, and (when supplied) eval and
    red-team evidence — so the document is grounded by construction, not
    hand-written, and regenerates on every config change.
    """

    def __init__(self, *, classifier: RiskTierClassifier | None = None) -> None:
        self.classifier = classifier or RiskTierClassifier()

    def build(
        self,
        app: ContextApp | VincioConfig,
        *,
        format: RenderFormat = "markdown",
        eval_report: EvalReport | None = None,
        redteam: RedTeamReport | None = None,
    ) -> DocumentArtifact:
        from ..generation.builder import DocumentBuilder

        assessment = self.classifier.classify(app)
        system_card = generate_system_card(app, eval_report=eval_report)
        model_card = generate_model_card(app, eval_report=eval_report)
        report = ComplianceMapper().map(redteam=redteam, eval_report=eval_report, target=app)

        sections = self._sections(app, assessment, system_card, model_card, report)
        title = f"EU AI Act — Annex IV Technical Documentation: {system_card.name}"
        builder = DocumentBuilder(audit_log=getattr(app, "audit", None))
        artifact = builder.build({"title": title, "sections": sections}, format=format)
        _record_conformity(app, "annex_iv", assessment, report)
        return artifact

    @staticmethod
    def _sections(
        app: Any, assessment: RiskAssessment, system_card: Any, model_card: Any, report: Any
    ) -> list[dict[str, Any]]:
        data = system_card.data_handling
        return [
            {
                "heading": "1. General description of the AI system",
                "body": (
                    f"System: {system_card.name}. Provider/model: {model_card.provider}/"
                    f"{model_card.model_id}. Intended purpose: {model_card.intended_use}. "
                    f"EU AI Act risk tier (advisory): {assessment.tier.value} — {assessment.rationale}"
                ),
                "items": [f"Obligation: {o}" for o in assessment.obligations],
            },
            {
                "heading": "2. Detailed description of elements and development process",
                "body": "Architecture and components composing the system:",
                "items": [
                    f"Retrieval: {system_card.retrieval}",
                    f"Memory: {system_card.memory}",
                    f"Model capabilities: {model_card.capabilities}",
                    f"Safety filters: {', '.join(system_card.safety_filters) or 'none configured'}",
                ],
            },
            {
                "heading": "3. Monitoring, functioning and control",
                "body": "Operational monitoring and human-oversight controls.",
                "items": list(system_card.human_oversight) + list(system_card.governance_controls),
            },
            {
                "heading": "4. Risk-management system (Art. 9)",
                "body": (
                    f"Risk drivers: {'; '.join(assessment.drivers) or 'none declared'}. "
                    f"Coverage across mapped frameworks: {int(report.coverage_rate * 100)}% "
                    f"({report.summary()['controls_covered']}/{report.summary()['controls_total']} controls)."
                ),
                "table": _matrix_table(report),
            },
            {
                "heading": "5. Accuracy, robustness and cybersecurity (Art. 15)",
                "body": "Measured evidence backing accuracy/robustness/security claims.",
                "items": [f"{k} = {v}" for k, v in (model_card.evaluation or {}).items()]
                or ["No eval report supplied; attach one to populate measured evidence."],
            },
            {
                "heading": "6. Data governance and provenance (Art. 10)",
                "body": "Data-handling controls applied to inputs and grounding data.",
                "items": [f"{k}: {v}" for k, v in data.items()],
            },
            {
                "heading": "7. Record-keeping and post-market monitoring (Art. 12, 72)",
                "body": (
                    "Every run, retrieval, tool call, memory write, and generated artifact is recorded on "
                    "a hash-chained, offline-verifiable audit log; generated media carries C2PA provenance."
                ),
            },
        ]


class FRIAGenerator:
    """Generate an Article 27 **fundamental-rights impact assessment** (FRIA).

    A cited document drawn from the risk-tier assessment, the system card, and
    the bias/fairness eval evidence — the deployer-facing companion to Annex IV.
    """

    def __init__(self, *, classifier: RiskTierClassifier | None = None) -> None:
        self.classifier = classifier or RiskTierClassifier()

    def generate(
        self,
        app: ContextApp | VincioConfig,
        *,
        format: RenderFormat = "markdown",
        affected_groups: list[str] | None = None,
        eval_report: EvalReport | None = None,
    ) -> DocumentArtifact:
        from ..generation.builder import DocumentBuilder

        assessment = self.classifier.classify(app)
        system_card = generate_system_card(app, eval_report=eval_report)
        groups = affected_groups or ["End users", "Data subjects whose data is processed"]
        bias = (system_card.evaluation or {}).get("bias")
        toxicity = (system_card.evaluation or {}).get("toxicity")

        sections = [
            {
                "heading": "1. Purpose and context of use",
                "body": f"{system_card.name}: {assessment.rationale} (tier: {assessment.tier.value}).",
                "items": assessment.drivers or ["No purpose/domain declared."],
            },
            {
                "heading": "2. Categories of natural persons and groups affected",
                "items": groups,
            },
            {
                "heading": "3. Risks of harm to fundamental rights",
                "body": "Identified risk areas and measured indicators.",
                "items": [
                    f"Bias / non-discrimination indicator (eval bias rate): {bias if bias is not None else 'not measured'}",
                    f"Dignity / safety (eval toxicity rate): {toxicity if toxicity is not None else 'not measured'}",
                    "Privacy: governed by data-handling controls (see measures below).",
                ],
            },
            {
                "heading": "4. Human-oversight measures",
                "items": list(system_card.human_oversight),
            },
            {
                "heading": "5. Measures on materialization of risks",
                "items": list(system_card.safety_filters) + list(system_card.governance_controls),
            },
        ]
        title = f"EU AI Act Art. 27 — Fundamental-Rights Impact Assessment: {system_card.name}"
        builder = DocumentBuilder(audit_log=getattr(app, "audit", None))
        artifact = builder.build({"title": title, "sections": sections}, format=format)
        report = ComplianceMapper().map(eval_report=eval_report, target=app)
        _record_conformity(app, "fria", assessment, report)
        return artifact


def _record_conformity(app: Any, kind: str, assessment: RiskAssessment, report: Any) -> None:
    audit = getattr(app, "audit", None)
    if audit is None:
        return
    audit.record(
        "conformity_doc",
        resource=kind,
        details={
            "framework": "eu_ai_act",
            "document": kind,
            "risk_tier": assessment.tier.value,
            "obligations": assessment.obligations,
            "compliance_coverage": report.coverage_rate,
        },
    )
