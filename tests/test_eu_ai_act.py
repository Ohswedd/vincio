"""Tests for the 1.9 EU AI Act conformity pack and ISO/IEC 42001 mapping."""

from __future__ import annotations

import pytest

from vincio import ContextApp, VincioConfig
from vincio.governance import (
    AnnexIVBuilder,
    ComplianceFramework,
    ComplianceMapper,
    RiskTier,
    RiskTierClassifier,
)
from vincio.providers import MockProvider

pytestmark = pytest.mark.filterwarnings("ignore::vincio.VincioExperimentalWarning")


@pytest.fixture()
def app(offline_config):
    return ContextApp(name="loan_helper", provider=MockProvider(), model="mock-1", config=offline_config)


class TestRiskTier:
    def test_high_risk_domain(self, app):
        a = RiskTierClassifier(purpose="credit scoring", domains=["creditworthiness"]).classify(app)
        assert a.tier == RiskTier.HIGH_RISK
        assert a.matched_high_risk_domains
        assert any("Annex IV" in o for o in a.obligations)

    def test_prohibited_practice(self, app):
        a = RiskTierClassifier(purpose="real-time remote biometric identification").classify(app)
        assert a.tier == RiskTier.PROHIBITED

    def test_limited_for_chatbot(self, app):
        a = RiskTierClassifier(purpose="general assistant", interacts_with_humans=True).classify(app)
        assert a.tier == RiskTier.LIMITED

    def test_minimal(self, app):
        a = RiskTierClassifier(
            purpose="internal log summarizer", interacts_with_humans=False, generates_content=False
        ).classify(app)
        assert a.tier == RiskTier.MINIMAL

    def test_app_method(self, app):
        assert app.risk_tier(purpose="employment screening").tier == RiskTier.HIGH_RISK


class TestAnnexIV:
    def test_builds_cited_document(self, app):
        classifier = RiskTierClassifier(purpose="credit scoring", domains=["creditworthiness"])
        art = AnnexIVBuilder(classifier=classifier).build(app, format="markdown")
        text = art.text
        assert "Annex IV" in text
        assert "Control coverage" in text  # compliance matrix table
        assert "risk tier" in text.lower()
        assert "Risk-management system" in text

    def test_records_conformity_event(self, app):
        app.annex_iv(purpose="employment screening", domains=["employment"])
        events = [e for e in app.audit.entries if e.action == "conformity_doc"]
        assert events and events[-1].details["framework"] == "eu_ai_act"
        assert events[-1].details["risk_tier"] == "high_risk"

    def test_grounded_by_eval_evidence(self, app):
        from vincio.evals.reports import EvalReport

        # An empty-but-present eval report still flows through without error.
        art = app.annex_iv(purpose="x", eval_report=EvalReport())
        assert art.content


class TestFRIA:
    def test_generates_with_affected_groups(self, app):
        art = app.fria(purpose="credit scoring", domains=["creditworthiness"],
                       affected_groups=["Loan applicants"])
        assert "Fundamental-Rights" in art.text and "Loan applicants" in art.text

    def test_records_conformity_event(self, app):
        before = len([e for e in app.audit.entries if e.action == "conformity_doc"])
        app.fria(purpose="x")
        after = len([e for e in app.audit.entries if e.action == "conformity_doc"])
        assert after == before + 1


class TestISO42001:
    def test_iso_controls_mapped(self):
        report = ComplianceMapper().map(target=VincioConfig())
        assert ComplianceFramework.ISO_42001.value in report.frameworks
        iso = [c for c in report.coverage if c.framework == ComplianceFramework.ISO_42001]
        assert len(iso) >= 8
        # At least the by-construction audit/provenance controls are covered.
        assert any(c.status == "covered" for c in iso)
