"""1.10 — the deep-research agent: budgeted search→read→reflect→verify→synthesize,
with cited, grounded, eval-scored output."""

import re
import warnings

import pytest

from vincio import ContextApp, ResearchAgent, ResearchBudget, VincioConfig
from vincio.providers import MockProvider

warnings.simplefilter("ignore")


@pytest.fixture()
def research_app(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text(
        "# Refund Policy\n\n"
        "The refund window for the Pro plan is 30 days from purchase. "
        "Enterprise customers get a 60 day refund window. "
        "Refunds are processed within 5 business days.\n",
        encoding="utf-8",
    )
    (docs / "billing.md").write_text(
        "# Billing\n\n"
        "Invoices are issued monthly. Late payments accrue 1.5% monthly interest. "
        "Annual plans are billed up front.\n",
        encoding="utf-8",
    )
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")

    def responder(request):
        # Cite the first evidence ref present in the prompt, so synthesis is cited.
        text = "\n".join(m.text for m in request.messages)
        match = re.search(r"\[([\w.:-]+)\]", text)
        ref = match.group(1) if match else "E1"
        return f"The Pro plan refund window is 30 days. [{ref}]"

    app = ContextApp(name="research", provider=MockProvider(responder=responder),
                     model="mock-1", config=config)
    app.add_source("docs", path=str(docs), retrieval="hybrid")
    return app


class TestResearchAgent:
    def test_produces_cited_grounded_report(self, research_app):
        report = research_app.research("What is the refund window for the Pro plan?")
        assert report.answer
        assert report.sources, "research should collect evidence"
        assert report.cited_report is not None
        # The answer carries resolved citations and a non-zero coverage.
        assert report.metrics["citation_coverage"] > 0.0

    def test_decomposes_into_subquestions(self, research_app):
        report = research_app.research("What are the refund and billing policies?")
        assert len(report.sub_questions) >= 1
        assert report.sub_questions[0]  # original question is first probe

    def test_budget_caps_sources(self, research_app):
        budget = ResearchBudget(breadth=2, depth=1, max_sources=2, top_k=4)
        agent = ResearchAgent(research_app, budget=budget)
        bounded = agent.run("refund policy")
        assert len(bounded.sources) <= 2

    def test_depth_zero_single_round(self, research_app):
        agent = ResearchAgent(research_app, budget=ResearchBudget(depth=0, breadth=2))
        report = agent.run("refund window")
        assert report.rounds == 1

    def test_offline_synthesis_is_cited_without_provider_citations(self, tmp_path):
        # A provider that never cites forces the deterministic grounded synthesis.
        docs = tmp_path / "d"
        docs.mkdir()
        (docs / "p.md").write_text("The Pro plan refund window is 30 days.", encoding="utf-8")
        config = VincioConfig()
        config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
        config.observability.exporter = "memory"
        app = ContextApp(name="r", provider=MockProvider(responder=lambda r: "no citation here"),
                         model="mock-1", config=config)
        app.add_source("d", path=str(docs), retrieval="hybrid")
        report = app.research("refund window")
        # Deterministic synthesis must still attach citation markers.
        assert "[" in report.answer and "]" in report.answer
        assert report.metrics["citation_coverage"] > 0.0

    def test_requires_a_source(self, tmp_path):
        from vincio.core.errors import RetrievalError

        config = VincioConfig()
        config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
        config.observability.exporter = "memory"
        app = ContextApp(name="r", provider=MockProvider(), model="mock-1", config=config)
        with pytest.raises(RetrievalError):
            app.research("anything")

    def test_records_audit_and_event(self, research_app):
        seen = []
        research_app.events.subscribe("research.completed", lambda e: seen.append(e))
        research_app.research("refund window")
        assert seen and seen[0].payload["sources"] >= 0
        assert research_app.audit.verify_chain()

    def test_render_cited_report(self, research_app):
        report = research_app.research("refund window for Pro plan")
        artifact = report.render("markdown")
        assert artifact.text or artifact.content

    def test_judge_verification_metric(self, research_app):
        from vincio.evals.judges import DeterministicJudge
        from vincio.evals.metrics import METRICS

        judge = DeterministicJudge(METRICS["groundedness"], name="faithfulness")
        agent = ResearchAgent(research_app, judge=judge)
        report = agent.run("refund window")
        assert "verification" in report.metrics
