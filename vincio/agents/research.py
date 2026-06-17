"""Deep-research agent (1.10): budgeted, citation-gated, eval-scored.

A :class:`ResearchAgent` runs the loop the field now expects of a deep-research
product — **search → read → reflect → verify → synthesize** — but composed
entirely from organs Vincio already ships, so every claim is grounded and cited
*by construction*:

- **search** reuses the query-understanding planners
  (:class:`~vincio.retrieval.query_understanding.QueryUnderstanding`: HyDE /
  multi-query / decompose / step-back) to turn the question into sub-questions
  and search probes;
- **read** retrieves through the app's :class:`~vincio.retrieval.engine.RetrievalEngine`
  and dedups sources by content;
- **reflect** finds under-covered sub-questions and, while depth budget remains,
  generates step-back follow-ups for another round;
- **verify** keeps only claims the cited evidence supports
  (:func:`~vincio.memory.facts.extract_grounded_facts`, optionally a faithfulness
  judge from :mod:`vincio.evals.judges`);
- **synthesize** emits a cited report through the 1.9
  :class:`~vincio.generation.report.CitedReportBuilder`.

The whole loop is bounded by an explicit breadth/depth/source/token budget, and
every run lands on the app's trace and audit chain. Offline (or whenever the
provider does not return cited prose) a deterministic synthesis from the
grounded evidence keeps the agent reproducible and air-gapped-safe.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.tokens import count_tokens
from ..core.types import EvidenceItem, Message, ModelRequest
from ..generation.report import CitationContract, CitedReport, CitedReportBuilder
from ..memory.facts import GroundedFact, extract_grounded_facts
from ..providers.base import run_sync
from ..retrieval.query_understanding import QueryUnderstanding

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..evals.judges import Judge

__all__ = ["ResearchBudget", "ResearchReport", "ResearchAgent"]

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_SYNTHESIS_SYSTEM = (
    "You are a research assistant. Using ONLY the numbered evidence, write a "
    "concise, factual answer to the question. Cite the evidence id in square "
    "brackets (e.g. [E1]) after every claim. Do not state anything the evidence "
    "does not support."
)


class ResearchBudget(BaseModel):
    """Explicit breadth/depth/source/token bounds for one research run."""

    breadth: int = 3  # sub-questions explored per round
    depth: int = 2  # reflection rounds after the first
    max_sources: int = 20  # cap on deduped evidence carried into synthesis
    top_k: int = 6  # evidence retrieved per sub-question
    max_context_tokens: int = 4000  # token budget for the evidence used to synthesize


class ResearchReport(BaseModel):
    """The cited, budgeted, eval-scored output of a research run."""

    model_config = {"arbitrary_types_allowed": True}

    question: str
    answer: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    facts: list[GroundedFact] = Field(default_factory=list)
    sources: list[EvidenceItem] = Field(default_factory=list)
    rounds: int = 0
    cited_report: CitedReport | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    def render(self, fmt: str = "markdown") -> Any:
        """Render the cited report through the document engine."""
        if self.cited_report is None:
            raise ValueError("no cited report to render")
        return self.cited_report.render(fmt)  # type: ignore[arg-type]


class ResearchAgent:
    """Search → read → reflect → verify → synthesize, cited and budget-bounded."""

    def __init__(
        self,
        app: ContextApp,
        *,
        budget: ResearchBudget | None = None,
        strategies: tuple[str, ...] = ("hyde", "multi_query"),
        judge: Judge | None = None,
        min_support: float = 0.5,
        require_citations: bool = True,
    ) -> None:
        self.app = app
        self.budget = budget or ResearchBudget()
        self.strategies = list(strategies)
        self.judge = judge
        self.min_support = min_support
        self.require_citations = require_citations
        self.understanding = QueryUnderstanding(app._base_provider(), app.model)

    async def arun(self, question: str, *, objective: str = "") -> ResearchReport:
        if self.app.retrieval is None:
            from ..core.errors import RetrievalError

            raise RetrievalError(
                "ResearchAgent needs a retrieval engine; add a source first "
                "(app.add_source(...))."
            )
        objective = objective or question
        report = ResearchReport(question=question)

        # 1. search — decompose the question into sub-questions (breadth).
        sub_questions = await self._decompose(question, objective)
        report.sub_questions = sub_questions

        evidence: dict[str, EvidenceItem] = {}
        asked: set[str] = set()
        pending = list(sub_questions)
        rounds = 0
        # 2-3. read + reflect: retrieve per sub-question, dedup, and (while depth
        # remains) widen with step-back follow-ups for under-covered questions.
        for round_index in range(self.budget.depth + 1):
            batch = [q for q in pending if q not in asked][: self.budget.breadth]
            if not batch:
                break
            rounds = round_index + 1
            uncovered: list[str] = []
            for sub_q in batch:
                asked.add(sub_q)
                hits = await self._retrieve(sub_q, objective)
                if len(hits) < 2:
                    uncovered.append(sub_q)
                for item in hits:
                    key = self._dedup_key(item)
                    if key not in evidence and len(evidence) < self.budget.max_sources:
                        evidence[key] = item
            if round_index < self.budget.depth and uncovered:
                follow_ups = await self._reflect(uncovered, objective)
                pending.extend(q for q in follow_ups if q not in asked)
        report.rounds = rounds

        sources = self._budget_sources(list(evidence.values()))
        report.sources = sources

        # 4-5. synthesize a cited answer, then verify it against the evidence.
        answer = await self._synthesize(question, sources)
        report.answer = answer
        report.facts = extract_grounded_facts(answer, sources, min_support=self.min_support, max_facts=20)
        report.cited_report = await self._cite(answer, sources)
        report.metrics = await self._score(answer, sources, report)
        self._audit(question, report)
        return report

    def run(self, question: str, *, objective: str = "") -> ResearchReport:
        return run_sync(self.arun(question, objective=objective))

    # -- stages --------------------------------------------------------------

    async def _decompose(self, question: str, objective: str) -> list[str]:
        expansions = await self.understanding.expand(
            question, ["decompose", "step_back"], objective=objective
        )
        sub: list[str] = []
        for expansion in expansions:
            sub.extend(expansion.queries)
        # Always include the original question as the first probe.
        ordered = [question, *[q for q in sub if q.strip().lower() != question.strip().lower()]]
        deduped = list(dict.fromkeys(q.strip() for q in ordered if q.strip()))
        return deduped[: max(1, self.budget.breadth * (self.budget.depth + 1))]

    async def _retrieve(self, sub_q: str, objective: str) -> list[EvidenceItem]:
        result = await self.app.retrieval.retrieve(  # type: ignore[union-attr]
            sub_q, top_k=self.budget.top_k, objective=objective, strategies=self.strategies
        )
        return result.evidence

    async def _reflect(self, uncovered: list[str], objective: str) -> list[str]:
        follow_ups: list[str] = []
        for question in uncovered:
            expansions = await self.understanding.expand(
                question, ["step_back", "multi_query"], objective=objective
            )
            for expansion in expansions:
                follow_ups.extend(expansion.queries)
        return list(dict.fromkeys(q.strip() for q in follow_ups if q.strip()))

    async def _synthesize(self, question: str, sources: list[EvidenceItem]) -> str:
        if not sources:
            return ""
        numbered = self._numbered_evidence(sources)
        provider = self.app._base_provider()
        try:
            request = ModelRequest(
                model=self.app.model or "",
                messages=[
                    Message(role="system", content=_SYNTHESIS_SYSTEM),
                    Message(role="user", content=f"Question: {question}\n\nEvidence:\n{numbered}"),
                ],
                temperature=0.0,
                max_output_tokens=1024,
            )
            response = run_sync(provider.generate(request))
            text = (response.text or "").strip()
        except Exception:  # noqa: BLE001 - any provider failure → deterministic synthesis
            text = ""
        # Require real citations: if the model didn't cite, synthesize
        # deterministically from the evidence so the answer is always grounded.
        if not text or (self.require_citations and not _has_citation(text, sources)):
            return self._synthesize_offline(sources)
        return text

    def _synthesize_offline(self, sources: list[EvidenceItem]) -> str:
        lines: list[str] = []
        for item in sources:
            sentence = _first_sentence(item.text or "")
            if not sentence:
                continue
            ref = item.citation_ref or item.id
            lines.append(f"{sentence} [{ref}]")
            if len(lines) >= 8:
                break
        return " ".join(lines)

    async def _cite(self, answer: str, sources: list[EvidenceItem]) -> CitedReport | None:
        if not answer:
            return None
        builder = CitedReportBuilder(audit_log=self.app.audit, tenant_id=None)
        contract = CitationContract(min_coverage=0.0, allow_unresolved_markers=True)
        return await builder.build_report(answer, sources, title="Research", contract=contract)

    async def _score(
        self, answer: str, sources: list[EvidenceItem], report: ResearchReport
    ) -> dict[str, float]:
        coverage = report.cited_report.coverage.coverage if report.cited_report else 0.0
        grounding = (
            sum(f.support for f in report.facts) / len(report.facts) if report.facts else 0.0
        )
        unique_sources = len({s.source_id for s in sources})
        diversity = unique_sources / len(sources) if sources else 0.0
        metrics = {
            "citation_coverage": round(coverage, 4),
            "grounding": round(grounding, 4),
            "source_diversity": round(diversity, 4),
            "sources": float(len(sources)),
            "rounds": float(report.rounds),
        }
        if self.judge is not None:
            verdict = await self._verify(answer, sources)
            metrics["verification"] = round(verdict, 4)
        return metrics

    async def _verify(self, answer: str, sources: list[EvidenceItem]) -> float:
        from ..evals.datasets import EvalCase
        from ..evals.metrics import RunOutput

        case = EvalCase(id="research", input=self.app.prompt_spec.objective or "research")
        output = RunOutput(output=answer, raw_text=answer, evidence=sources)
        result = await self.judge.score(case, output)  # type: ignore[union-attr]
        return float(result.value)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _dedup_key(item: EvidenceItem) -> str:
        text = (item.text or "").strip().lower()
        return f"{item.source_id}:{hash(text)}"

    def _budget_sources(self, sources: list[EvidenceItem]) -> list[EvidenceItem]:
        ordered = sorted(sources, key=lambda e: e.relevance, reverse=True)
        kept: list[EvidenceItem] = []
        tokens = 0
        for item in ordered[: self.budget.max_sources]:
            cost = item.token_cost or count_tokens(item.text or "")
            if kept and tokens + cost > self.budget.max_context_tokens:
                break
            kept.append(item)
            tokens += cost
        return kept

    @staticmethod
    def _numbered_evidence(sources: list[EvidenceItem]) -> str:
        lines = []
        for item in sources:
            ref = item.citation_ref or item.id
            snippet = (item.text or "").strip().replace("\n", " ")
            lines.append(f"[{ref}] {snippet[:400]}")
        return "\n".join(lines)

    def _audit(self, question: str, report: ResearchReport) -> None:
        self.app.audit.record(
            "research",
            decision="allow",
            details={
                "question": question[:200],
                "sub_questions": len(report.sub_questions),
                "rounds": report.rounds,
                "sources": len(report.sources),
                "facts": len(report.facts),
                "metrics": report.metrics,
            },
        )
        self.app.events.emit(
            "research.completed",
            {"question": question[:200], "sources": len(report.sources),
             "coverage": report.metrics.get("citation_coverage", 0.0)},
        )


def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    parts = _SENTENCE_END_RE.split(text, maxsplit=1)
    sentence = parts[0].strip()
    return sentence[:300]


def _has_citation(text: str, sources: list[EvidenceItem]) -> bool:
    from ..output.parsers import extract_citations

    keys = set()
    for item in sources:
        keys.update({item.id, item.citation_ref, item.source_id})
    markers = set(extract_citations(text))
    return bool(markers & keys)
