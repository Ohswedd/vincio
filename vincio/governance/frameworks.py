"""Compliance-framework mapping: red-team + audit + eval as control evidence.

Regulated buyers ask "which OWASP / NIST / MITRE control does this address, and
what is your evidence?". Vincio answers mechanically: it maps the controls of
four frameworks —

* **OWASP LLM Top 10 (2025)**
* **OWASP Agentic AI — Threats & Mitigations**
* **NIST AI RMF (Generative AI Profile)**
* **MITRE ATLAS**

— to the capabilities it actually ships, then back the mapping with *measured*
evidence: red-team probe outcomes (:class:`~vincio.evals.redteam.RedTeamReport`),
the security configuration, and evaluation results
(:class:`~vincio.evals.reports.EvalReport`). The result is a coverage matrix you
can hand to an auditor — every "covered" claim names the evidence that supports
it, and an uncovered control is reported honestly rather than hidden in an
aggregate.

The mapping is data-driven (the control catalog below) so it can track the
standards as they evolve without touching the engine.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from ..core.utils import utcnow
from ..stability import experimental

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.app import ContextApp
    from ..core.config import VincioConfig
    from ..evals.redteam import RedTeamReport
    from ..evals.reports import EvalReport

__all__ = [
    "ComplianceFramework",
    "Control",
    "ControlCoverage",
    "ComplianceReport",
    "ComplianceMapper",
    "map_compliance",
    "CONTROL_CATALOG",
]


class ComplianceFramework(StrEnum):
    """A governance framework whose controls Vincio maps onto."""

    OWASP_LLM_2025 = "owasp_llm_2025"
    OWASP_AGENTIC = "owasp_agentic"
    NIST_AI_RMF = "nist_ai_rmf"
    MITRE_ATLAS = "mitre_atlas"
    ISO_42001 = "iso_42001"
    EU_AI_ACT = "eu_ai_act"


class Control(BaseModel):
    """One framework control and the Vincio capabilities that address it."""

    framework: ComplianceFramework
    control_id: str
    title: str
    # Capability keys (see ``_CAPABILITIES``) required to consider it covered.
    capabilities: list[str] = Field(default_factory=list)


# Capability keys describe what Vincio does; controls reference them. Keeping
# this layer between "control" and "evidence" means each capability's evidence
# is gathered once and reused across every framework that needs it.
_CAPABILITIES = {
    "injection_defense",
    "output_handling",
    "pii_protection",
    "secret_protection",
    "system_prompt_protection",
    "poisoning_defense",
    "tool_governance",
    "excessive_agency_bounds",
    "memory_governance",
    "resource_bounds",
    "supply_chain",
    "grounding",
    "bias_fairness",
    "toxicity",
    "residency",
    "transparency",
    "human_oversight",
    "audit",
}


CONTROL_CATALOG: list[Control] = [
    # ---- OWASP LLM Top 10 (2025) -------------------------------------------------
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM01",
            title="Prompt Injection", capabilities=["injection_defense"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM02",
            title="Sensitive Information Disclosure",
            capabilities=["pii_protection", "secret_protection"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM03",
            title="Supply Chain", capabilities=["supply_chain"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM04",
            title="Data and Model Poisoning", capabilities=["poisoning_defense"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM05",
            title="Improper Output Handling", capabilities=["output_handling"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM06",
            title="Excessive Agency",
            capabilities=["excessive_agency_bounds", "tool_governance", "human_oversight"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM07",
            title="System Prompt Leakage", capabilities=["system_prompt_protection"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM08",
            title="Vector and Embedding Weaknesses",
            capabilities=["poisoning_defense", "pii_protection"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM09",
            title="Misinformation", capabilities=["grounding"]),
    Control(framework=ComplianceFramework.OWASP_LLM_2025, control_id="LLM10",
            title="Unbounded Consumption", capabilities=["resource_bounds"]),
    # ---- OWASP Agentic AI (Threats & Mitigations) --------------------------------
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T1",
            title="Memory Poisoning", capabilities=["memory_governance", "poisoning_defense"]),
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T2",
            title="Tool Misuse", capabilities=["tool_governance"]),
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T3",
            title="Privilege Compromise", capabilities=["tool_governance", "audit"]),
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T4",
            title="Resource Overload", capabilities=["resource_bounds"]),
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T5",
            title="Cascading Hallucination", capabilities=["grounding"]),
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T6",
            title="Intent Breaking & Goal Manipulation", capabilities=["injection_defense"]),
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T7",
            title="Misaligned & Deceptive Behaviors",
            capabilities=["human_oversight", "excessive_agency_bounds"]),
    Control(framework=ComplianceFramework.OWASP_AGENTIC, control_id="T8",
            title="Repudiation & Untraceability", capabilities=["audit"]),
    # ---- NIST AI RMF (Generative AI Profile) -------------------------------------
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="GOVERN-1.1",
            title="Governance: policies, roles, and accountability", capabilities=["audit"]),
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="MAP-2.3",
            title="System purpose & capabilities documented",
            capabilities=["transparency"]),
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="MEASURE-2.3",
            title="Safety of the GenAI system is evaluated",
            capabilities=["toxicity", "injection_defense"]),
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="MEASURE-2.6",
            title="Security & resilience are evaluated",
            capabilities=["injection_defense", "poisoning_defense", "resource_bounds"]),
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="MEASURE-2.7",
            title="Privacy is evaluated", capabilities=["pii_protection"]),
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="MEASURE-2.9",
            title="Output validity & reliability are evaluated", capabilities=["grounding"]),
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="MEASURE-2.11",
            title="Fairness & bias are evaluated", capabilities=["bias_fairness"]),
    Control(framework=ComplianceFramework.NIST_AI_RMF, control_id="MANAGE-4.1",
            title="Post-deployment monitoring & response", capabilities=["audit"]),
    # ---- MITRE ATLAS -------------------------------------------------------------
    Control(framework=ComplianceFramework.MITRE_ATLAS, control_id="AML.T0051",
            title="LLM Prompt Injection", capabilities=["injection_defense"]),
    Control(framework=ComplianceFramework.MITRE_ATLAS, control_id="AML.T0054",
            title="LLM Jailbreak", capabilities=["injection_defense"]),
    Control(framework=ComplianceFramework.MITRE_ATLAS, control_id="AML.T0057",
            title="LLM Data Leakage",
            capabilities=["pii_protection", "secret_protection", "system_prompt_protection"]),
    Control(framework=ComplianceFramework.MITRE_ATLAS, control_id="AML.T0070",
            title="RAG Poisoning", capabilities=["poisoning_defense"]),
    Control(framework=ComplianceFramework.MITRE_ATLAS, control_id="AML.T0024",
            title="Exfiltration via ML Inference API",
            capabilities=["secret_protection", "residency"]),
    Control(framework=ComplianceFramework.MITRE_ATLAS, control_id="AML.T0048",
            title="Societal Harm", capabilities=["toxicity", "bias_fairness"]),
    Control(framework=ComplianceFramework.MITRE_ATLAS, control_id="AML.T0029",
            title="Denial of ML Service", capabilities=["resource_bounds"]),
    # ---- ISO/IEC 42001:2023 (AI management system — Annex A controls) ------------
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.2.2",
            title="AI policy", capabilities=["audit", "transparency"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.5.2",
            title="AI system impact assessment process",
            capabilities=["transparency", "human_oversight"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.5.4",
            title="Assessing impacts on individuals and groups",
            capabilities=["bias_fairness", "human_oversight"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.6.2.4",
            title="AI system verification and validation", capabilities=["grounding"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.6.2.6",
            title="AI system operation and monitoring", capabilities=["audit"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.7.4",
            title="Quality of data for AI systems",
            capabilities=["poisoning_defense", "grounding"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.7.5",
            title="Data provenance", capabilities=["audit", "transparency"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.8.2",
            title="System documentation and information for users",
            capabilities=["transparency"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.9.2",
            title="Processes for responsible use of AI systems",
            capabilities=["human_oversight", "excessive_agency_bounds"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.9.3",
            title="Privacy and protection of personal data",
            capabilities=["pii_protection", "residency"]),
    Control(framework=ComplianceFramework.ISO_42001, control_id="A.10.2",
            title="Allocating responsibilities across the AI supply chain",
            capabilities=["supply_chain"]),
]


class _CapabilityEvidence(BaseModel):
    status: str = "not_covered"  # covered | partial | not_covered
    evidence: list[str] = Field(default_factory=list)


class ControlCoverage(BaseModel):
    """The coverage verdict for one control, with its evidence."""

    framework: ComplianceFramework
    control_id: str
    title: str
    status: str  # covered | partial | not_covered
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    """A coverage matrix across the mapped frameworks."""

    generated_at: datetime = Field(default_factory=utcnow)
    vincio_version: str = ""
    coverage: list[ControlCoverage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def frameworks(self) -> list[str]:
        return sorted({c.framework.value for c in self.coverage})

    def by_framework(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for fw in self.frameworks:
            controls = [c for c in self.coverage if c.framework.value == fw]
            covered = sum(1 for c in controls if c.status == "covered")
            partial = sum(1 for c in controls if c.status == "partial")
            out[fw] = {
                "controls": len(controls),
                "covered": covered,
                "partial": partial,
                "not_covered": len(controls) - covered - partial,
                "coverage_rate": round((covered + 0.5 * partial) / len(controls), 4) if controls else 0.0,
            }
        return out

    @property
    def coverage_rate(self) -> float:
        """Weighted coverage across all controls (partial counts as 0.5)."""
        if not self.coverage:
            return 0.0
        score = sum(1.0 if c.status == "covered" else 0.5 if c.status == "partial" else 0.0 for c in self.coverage)
        return round(score / len(self.coverage), 4)

    def gaps(self) -> list[ControlCoverage]:
        return [c for c in self.coverage if c.status != "covered"]

    def summary(self) -> dict[str, Any]:
        return {
            "frameworks": len(self.frameworks),
            "controls_total": len(self.coverage),
            "controls_covered": sum(1 for c in self.coverage if c.status == "covered"),
            "coverage_rate": self.coverage_rate,
            "by_framework": self.by_framework(),
            "gaps": [f"{c.framework.value}:{c.control_id}" for c in self.gaps()],
        }

    def to_json(self, *, indent: int = 2) -> str:
        import json

        return json.dumps(self.model_dump(mode="json"), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# Compliance coverage matrix", ""]
        for fw, stats in self.by_framework().items():
            lines.append(f"## {fw} — {stats['covered']}/{stats['controls']} covered "
                         f"({int(stats['coverage_rate'] * 100)}%)")
            lines.append("")
            lines.append("| Control | Title | Status | Evidence |")
            lines.append("|---|---|---|---|")
            for c in [c for c in self.coverage if c.framework.value == fw]:
                ev = "; ".join(c.evidence) if c.evidence else "—"
                lines.append(f"| {c.control_id} | {c.title} | {c.status} | {ev} |")
            lines.append("")
        return "\n".join(lines)


@experimental(since="1.6")
class ComplianceMapper:
    """Map Vincio's measured controls onto governance frameworks.

    Pass any combination of a :class:`RedTeamReport` (behavioural evidence), an
    :class:`EvalReport` (measured quality/safety), and a
    :class:`~vincio.core.app.ContextApp` or :class:`VincioConfig` (configured
    controls). The more evidence supplied, the stronger the coverage claims.
    """

    def __init__(self, *, catalog: list[Control] | None = None) -> None:
        self.catalog = list(catalog) if catalog is not None else list(CONTROL_CATALOG)

    # -- evidence gathering ----------------------------------------------------

    def _redteam_status(self, report: RedTeamReport | None, *categories: str) -> _CapabilityEvidence:
        if report is None:
            return _CapabilityEvidence()
        by_cat = report.by_category()
        relevant = {c: by_cat[c] for c in categories if c in by_cat}
        if not relevant:
            return _CapabilityEvidence()
        total = sum(int(stats["probes"]) for stats in relevant.values())
        passed = sum(int(stats["passed"]) for stats in relevant.values())
        if total == 0:
            return _CapabilityEvidence()
        rate = passed / total
        ev = [f"red-team {cat}: {int(stats['passed'])}/{int(stats['probes'])} probes defended"
              for cat, stats in sorted(relevant.items())]
        status = "covered" if rate >= 1.0 else "partial" if rate > 0 else "not_covered"
        return _CapabilityEvidence(status=status, evidence=ev)

    @staticmethod
    def _merge(*parts: _CapabilityEvidence) -> _CapabilityEvidence:
        order = {"covered": 2, "partial": 1, "not_covered": 0}
        best = "not_covered"
        evidence: list[str] = []
        for part in parts:
            evidence.extend(part.evidence)
            if order[part.status] > order[best]:
                best = part.status
        return _CapabilityEvidence(status=best, evidence=evidence)

    def _collect(
        self,
        *,
        redteam: RedTeamReport | None,
        eval_report: EvalReport | None,
        cfg: VincioConfig | None,
        target: ContextApp | VincioConfig | None = None,
    ) -> dict[str, _CapabilityEvidence]:
        ev: dict[str, _CapabilityEvidence] = {k: _CapabilityEvidence() for k in _CAPABILITIES}

        def by_construction(*evidence: str) -> _CapabilityEvidence:
            # A structural / deterministic guarantee that holds for every run by
            # construction (audit chain, residency check, bounded executors) —
            # legitimately "covered" because it is enforced, not merely claimed.
            return _CapabilityEvidence(status="covered", evidence=list(evidence))

        def cfg_flag(*evidence: str) -> _CapabilityEvidence:
            # A *measurable* control that config enables but no test has yet
            # exercised. Honestly "partial" until red-team/eval evidence
            # corroborates it (1.7): a config flag alone can no longer reach
            # "covered" — the auditor matrix reflects defense actually exercised.
            return _CapabilityEvidence(status="partial", evidence=list(evidence))

        def cfg_available(*evidence: str) -> _CapabilityEvidence:
            # A capability the library ships but does not auto-apply to a run —
            # honestly "partial" (available) until wired or evidenced, never an
            # unconditional "covered".
            return _CapabilityEvidence(status="partial", evidence=list(evidence))

        # --- configured controls: enabled-but-unmeasured ⇒ partial until the
        # behavioural/measured evidence below elevates them ---
        if cfg is not None:
            if cfg.security.injection_detection:
                ev["injection_defense"] = self._merge(
                    ev["injection_defense"], cfg_flag("injection detector enabled (trust-tagged)"))
                ev["system_prompt_protection"] = self._merge(
                    ev["system_prompt_protection"], cfg_flag("untrusted-content instruction blocking"))
            if cfg.security.pii_detection:
                ev["pii_protection"] = self._merge(ev["pii_protection"], cfg_flag("PII detection enabled"))
                ev["secret_protection"] = self._merge(ev["secret_protection"], cfg_flag("secret scanner enabled"))
            # Audit and residency are enforced by construction when configured.
            if cfg.security.audit_log:
                ev["audit"] = by_construction("hash-chained, offline-verifiable audit log enabled")
            if cfg.policies.require_citations or cfg.policies.answer_only_from_sources:
                ev["grounding"] = self._merge(ev["grounding"], cfg_flag("citations/grounding required by policy"))
            if cfg.policies.block_untrusted_instructions:
                ev["injection_defense"] = self._merge(
                    ev["injection_defense"], cfg_flag("block_untrusted_instructions policy on"))
            # Residency: enforced at the choke point when an allowed-region set is configured.
            gov = getattr(cfg, "governance", None)
            if gov is not None and getattr(gov, "allowed_regions", None):
                ev["residency"] = by_construction(
                    f"residency policy: allowed regions {sorted(gov.allowed_regions)}")
            if gov is not None and getattr(gov, "content_marking", False):
                ev["transparency"] = self._merge(
                    ev["transparency"], by_construction("synthetic-content marking enabled"))

        # --- always-on-by-construction guarantees (true for every run) ---
        ev["tool_governance"] = self._merge(ev["tool_governance"], by_construction(
            "permissioned, sandboxed, idempotency-keyed tool runtime (approvals when a callback is set)"))
        ev["excessive_agency_bounds"] = self._merge(ev["excessive_agency_bounds"], by_construction(
            "bounded executors: max_steps/tool_calls/cost, termination guarantees"))
        ev["resource_bounds"] = self._merge(ev["resource_bounds"], by_construction(
            "hard Budget limits (tokens/cost/latency/steps) + cost SLOs"))
        ev["memory_governance"] = self._merge(ev["memory_governance"], by_construction(
            "guarded memory writes: no-secrets, confidence/provenance, contradiction supersede"))
        ev["supply_chain"] = self._merge(ev["supply_chain"], by_construction(
            "CycloneDX SBOM + SLSA provenance + AI-BOM with model-hash verification"))
        ev["human_oversight"] = self._merge(ev["human_oversight"], by_construction(
            "human-in-the-loop interrupts on durable graphs/workflows (write approvals when a callback is set)"))
        ev["output_handling"] = self._merge(ev["output_handling"], by_construction(
            "schema-validated structured output; repair never invents facts"))
        ev["transparency"] = self._merge(ev["transparency"], by_construction(
            "model/system cards generated from live config"))
        # --- available-but-opt-in capabilities (partial until wired/evidenced) ---
        # RAG-poisoning detection ships as a utility but is not auto-applied to
        # the retrieval flow, so it is honestly "partial" unless the app wires a
        # detector (then it is an active control).
        poison_active = getattr(target, "poisoning_detector", None) is not None
        ev["poisoning_defense"] = self._merge(ev["poisoning_defense"], (
            by_construction("authority/provenance RAG-poisoning detector wired into the app")
            if poison_active else
            cfg_available("authority/provenance RAG-poisoning detector available "
                          "(vincio.security.PoisoningDetector); enable in your retrieval flow")))

        # --- behavioural evidence (red-team) ---
        ev["injection_defense"] = self._merge(
            ev["injection_defense"], self._redteam_status(redteam, "jailbreak", "injection"))
        ev["system_prompt_protection"] = self._merge(
            ev["system_prompt_protection"], self._redteam_status(redteam, "injection"))
        ev["pii_protection"] = self._merge(ev["pii_protection"], self._redteam_status(redteam, "pii_leak"))
        ev["secret_protection"] = self._merge(ev["secret_protection"], self._redteam_status(redteam, "pii_leak"))
        ev["bias_fairness"] = self._merge(ev["bias_fairness"], self._redteam_status(redteam, "bias"))
        ev["toxicity"] = self._merge(ev["toxicity"], self._redteam_status(redteam, "toxicity"))

        # --- measured quality evidence (eval report) ---
        if eval_report is not None:
            ev["grounding"] = self._merge(ev["grounding"], self._eval_status(
                eval_report, ("faithfulness", "answer_relevance", "context_precision"), floor=0.7))
            ev["grounding"] = self._merge(ev["grounding"], self._eval_status(
                eval_report, ("hallucination",), floor=0.0, lower_is_better=True))
            ev["bias_fairness"] = self._merge(ev["bias_fairness"], self._eval_status(
                eval_report, ("bias",), floor=0.0, lower_is_better=True))
            ev["toxicity"] = self._merge(ev["toxicity"], self._eval_status(
                eval_report, ("toxicity",), floor=0.0, lower_is_better=True))

        return ev

    @staticmethod
    def _eval_status(
        report: EvalReport,
        metrics: tuple[str, ...],
        *,
        floor: float,
        lower_is_better: bool = False,
    ) -> _CapabilityEvidence:
        evidence: list[str] = []
        statuses: list[str] = []
        for metric in metrics:
            values = report.metric_values(metric)
            if not values:
                continue
            mean = sum(values) / len(values)
            ok = mean <= floor if lower_is_better else mean >= floor
            statuses.append("covered" if ok else "partial")
            evidence.append(f"eval {metric}={round(mean, 3)} ({'≤' if lower_is_better else '≥'} {floor})")
        if not statuses:
            return _CapabilityEvidence()
        status = "covered" if all(s == "covered" for s in statuses) else "partial"
        return _CapabilityEvidence(status=status, evidence=evidence)

    # -- mapping ---------------------------------------------------------------

    def map(
        self,
        *,
        redteam: RedTeamReport | None = None,
        eval_report: EvalReport | None = None,
        target: ContextApp | VincioConfig | None = None,
    ) -> ComplianceReport:
        import vincio

        cfg: VincioConfig | None = None
        if target is not None:
            cfg = cast("VincioConfig", getattr(target, "config", None) or target)
        evidence = self._collect(redteam=redteam, eval_report=eval_report, cfg=cfg, target=target)

        order = {"covered": 2, "partial": 1, "not_covered": 0}
        coverage: list[ControlCoverage] = []
        for control in self.catalog:
            cap_status = [evidence[c] for c in control.capabilities if c in evidence]
            if not cap_status:
                status = "not_covered"
                ev_strings: list[str] = []
                gaps = list(control.capabilities)
            else:
                worst = min(order[c.status] for c in cap_status)
                status = {2: "covered", 1: "partial", 0: "not_covered"}[worst]
                ev_strings = [e for c in cap_status for e in c.evidence]
                # Iterate the control's required capabilities directly (not a
                # filtered, possibly-shorter list) so an unevidenced capability
                # is never silently dropped from the gaps report.
                gaps = [cap for cap in control.capabilities
                        if evidence.get(cap, _CapabilityEvidence()).status != "covered"]
            coverage.append(ControlCoverage(
                framework=control.framework, control_id=control.control_id, title=control.title,
                status=status, evidence=ev_strings, gaps=gaps))
        return ComplianceReport(vincio_version=vincio.__version__, coverage=coverage)


def map_compliance(
    *,
    redteam: RedTeamReport | None = None,
    eval_report: EvalReport | None = None,
    target: ContextApp | VincioConfig | None = None,
) -> ComplianceReport:
    """Convenience wrapper around :class:`ComplianceMapper`."""
    return ComplianceMapper().map(redteam=redteam, eval_report=eval_report, target=target)
