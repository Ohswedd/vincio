"""Cited-report assembly with resolved citations, bibliography & entailment.

:class:`CitedReportBuilder` takes a validated answer plus the
:class:`~vincio.core.types.EvidenceItem`\\ s that grounded it and renders a
shippable report: inline ``[E1]``-style markers resolved to numbered
footnotes/endnotes, a generated source bibliography with per-claim provenance
(trust level, ``source_uri``, page/section), sentence-level citation-coverage
metrics, and an optional NLI/entailment check that the cited evidence actually
*supports* each claim. This replaces the flat "one valid citation anywhere"
membership check with a per-claim contract: every claim cited and supported.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import CitationValidationError
from ..core.types import EvidenceItem
from ..documents.parsers import TableData
from ..evals.metrics import _supported_strict, _verifiable_claims
from ..output.parsers import extract_citations
from .builder import DocumentBuilder, markdown_to_model
from .model import DocumentModel
from .render import DocumentArtifact, RenderFormat, render

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data import DataCatalog
    from ..security.audit import AuditLog

__all__ = [
    "ResolvedCitation",
    "ClaimCheck",
    "CitationCoverage",
    "CitationContract",
    "Figure",
    "FigureBinding",
    "CitedReport",
    "CitedReportBuilder",
]

# Entailment backend: given a claim and the evidence it cites, return whether the
# evidence supports it. Sync or async; defaults to the strict lexical check.
EntailmentFn = (
    Callable[[str, list[EvidenceItem]], bool]
    | Callable[[str, list[EvidenceItem]], Awaitable[bool]]
)

_MARKER_RE = re.compile(r"\[((?:[A-Za-z]+[\w.-]*:)?[A-Za-z0-9][\w.:-]*)\]")


def _fmt_seconds(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


class ResolvedCitation(BaseModel):
    marker: str
    number: int
    evidence_id: str
    source_id: str
    source_uri: str | None = None
    page: int | None = None
    # Temporal locator (start, end in seconds) for a clip-grounded citation, so a
    # video/audio answer points at the moment it came from, not just the source.
    time_range: tuple[float, float] | None = None
    section_path: list[str] = Field(default_factory=list)
    trust_level: str = "untrusted_document"
    excerpt: str = ""

    def footnote(self) -> str:
        if self.time_range is not None:
            start, end = self.time_range
            locator = f", t{_fmt_seconds(start)}–{_fmt_seconds(end)}s"
        elif self.page is not None:
            locator = f", p{self.page}"
        else:
            locator = ""
        if self.section_path:
            locator += " §" + " › ".join(self.section_path)
        uri = f" — {self.source_uri}" if self.source_uri else ""
        snippet = f': "{self.excerpt}"' if self.excerpt else ""
        return f"{self.source_id} [{self.trust_level}]{locator}{uri}{snippet}"


class ClaimCheck(BaseModel):
    claim: str
    citations: list[str] = Field(default_factory=list)
    cited: bool = False
    entailed: bool | None = None  # None when no entailment check ran
    support_score: float = 0.0


class CitationCoverage(BaseModel):
    claims: int = 0
    cited_claims: int = 0
    entailed_claims: int = 0
    coverage: float = 1.0  # fraction of verifiable claims carrying a citation
    entailment_rate: float | None = None  # of cited claims, fraction supported
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    # Per-figure data binding (charts/tables that re-derive from their source).
    figures: int = 0
    data_bound_figures: int = 0
    figure_binding_rate: float | None = None  # of figures, fraction data-bound


class CitationContract(BaseModel):
    """Field/claim-level citation requirements for a cited report."""

    min_coverage: float = 1.0  # min fraction of verifiable claims that must cite
    require_entailment: bool = False  # cited evidence must support the claim
    min_entailment_rate: float = 1.0
    allow_unresolved_markers: bool = False  # markers not matching any evidence
    # Per-figure data binding: every embedded chart/table must re-derive from its
    # source (the data-plane analogue of per-claim entailment).
    require_figure_binding: bool = False
    min_figure_binding_rate: float = 1.0


class Figure(BaseModel):
    """A chart or table embedded in a cited report, **data-bound** to its source.

    Wraps a :class:`~vincio.data.Chart` or a cited
    :class:`~vincio.data.QueryResult` so the cited-report builder can resolve a
    ``[F1]``-style marker in the narrative to the figure, render its data into the
    deliverable, carry its source-cell citations, and **verify it re-derives from
    the bytes** — the per-figure analogue of a claim's per-claim entailment. Build
    one with :meth:`from_chart` or :meth:`from_table`."""

    model_config = {"arbitrary_types_allowed": True}

    marker: str = ""
    caption: str = ""
    kind: str = "chart"  # "chart" | "table"
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    cite_refs: list[str] = Field(default_factory=list)
    coverage: str = "result"
    artifact: Any = None  # the Chart / QueryResult, retained for verify()

    @staticmethod
    def _refs(provenance: Any) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for prov in provenance or []:
            for ref in prov.refs:
                if ref not in seen:
                    seen.add(ref)
                    out.append(ref)
        return out

    @classmethod
    def from_chart(cls, chart: Any, *, marker: str = "", caption: str = "") -> Figure:
        """Wrap a :class:`~vincio.data.Chart` as a report figure."""
        cols = list(chart.spec.columns)
        rows = [[rec.get(c) for c in cols] for rec in chart.spec.values]
        return cls(
            marker=marker,
            caption=caption or chart.spec.title or "Chart",
            kind="chart",
            columns=cols,
            rows=rows,
            cite_refs=chart.cite_refs(),
            coverage=str(chart.coverage),
            artifact=chart,
        )

    @classmethod
    def from_table(cls, result: Any, *, marker: str = "", caption: str = "") -> Figure:
        """Wrap a cited :class:`~vincio.data.QueryResult` as a report figure (a
        table)."""
        return cls(
            marker=marker,
            caption=caption or "Table",
            kind="table",
            columns=list(result.columns),
            rows=[list(r) for r in result.rows],
            cite_refs=cls._refs(getattr(result, "provenance", [])),
            coverage=str(getattr(result, "coverage", "result")),
            artifact=result,
        )

    def evidence_item(self) -> EvidenceItem:
        """The figure as an :class:`~vincio.core.types.EvidenceItem` whose id is the
        marker, so a ``[marker]`` reference in the narrative resolves to it."""
        item = self.artifact.to_evidence(source_id=self.marker, caption=self.caption).to_evidence_item()
        item.id = self.marker
        return item

    def table_data(self) -> TableData:
        """The figure's plotted/queried data as a renderable table block."""
        rows = [["" if cell is None else str(cell) for cell in row] for row in self.rows]
        return TableData(title=self.caption, columns=list(self.columns), rows=rows)

    def verify(self, catalog: DataCatalog | None) -> bool | None:
        """Whether the figure re-derives from its source against *catalog*; ``None``
        when no catalog is given (binding unchecked) or the artifact cannot verify."""
        if catalog is None or self.artifact is None or not hasattr(self.artifact, "verify"):
            return None
        return bool(self.artifact.verify(catalog))


class FigureBinding(BaseModel):
    """A figure's per-figure data-binding verdict in a cited report."""

    marker: str
    kind: str
    caption: str
    cite_refs: list[str] = Field(default_factory=list)
    coverage: str = "result"
    data_bound: bool | None = None  # None when no catalog was supplied to check


class CitedReport(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    document: DocumentModel
    citations: list[ResolvedCitation] = Field(default_factory=list)
    coverage: CitationCoverage = Field(default_factory=CitationCoverage)
    unresolved_markers: list[str] = Field(default_factory=list)
    figures: list[FigureBinding] = Field(default_factory=list)

    def render(self, fmt: RenderFormat = "markdown") -> DocumentArtifact:
        return render(self.document, fmt)


def _evidence_keys(item: EvidenceItem) -> set[str]:
    keys = {item.id, item.citation_ref, item.source_id}
    return {k for k in keys if k}


class CitedReportBuilder:
    """Resolve citations, verify per-claim support, render a cited report."""

    def __init__(
        self,
        *,
        entailment: EntailmentFn | None = None,
        audit_log: AuditLog | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.entailment = entailment
        self.audit_log = audit_log
        self.tenant_id = tenant_id

    def _answer_text(self, answer: Any) -> str:
        if isinstance(answer, str):
            return answer
        if hasattr(answer, "raw_text"):  # RunResult
            out = getattr(answer, "output", None)
            if isinstance(out, str) and out:
                return out
            return answer.raw_text
        if hasattr(answer, "model_dump_json"):
            return answer.model_dump_json()
        import json

        try:
            return json.dumps(answer, default=str)
        except (TypeError, ValueError):
            return str(answer)

    def _resolve(
        self, text: str, evidence: list[EvidenceItem]
    ) -> tuple[str, list[ResolvedCitation], list[str]]:
        lookup: dict[str, EvidenceItem] = {}
        for item in evidence:
            for key in _evidence_keys(item):
                lookup.setdefault(key, item)
        numbering: dict[str, int] = {}
        resolved: list[ResolvedCitation] = []
        unresolved: list[str] = []

        for marker in extract_citations(text):
            matched = lookup.get(marker)
            if matched is None:
                if marker not in unresolved:
                    unresolved.append(marker)
                continue
            item = matched
            if marker not in numbering:
                number = len(numbering) + 1
                numbering[marker] = number
                excerpt = (item.text or "").strip().replace("\n", " ")
                resolved.append(
                    ResolvedCitation(
                        marker=marker,
                        number=number,
                        evidence_id=item.id,
                        source_id=item.source_id,
                        source_uri=item.metadata.get("source_uri") or item.media_ref,
                        page=item.page,
                        time_range=item.time_range,
                        section_path=list(item.section_path),
                        trust_level=getattr(item.trust_level, "value", str(item.trust_level)),
                        excerpt=excerpt[:200] + ("…" if len(excerpt) > 200 else ""),
                    )
                )

        # Rewrite each resolved marker to its numbered form; leave unresolved
        # markers in place so the gap is visible, not silently dropped.
        def rewrite(match: re.Match[str]) -> str:
            marker = match.group(1)
            return f"[{numbering[marker]}]" if marker in numbering else match.group(0)

        rewritten = _MARKER_RE.sub(rewrite, text)
        return rewritten, resolved, unresolved

    async def _entailed(self, claim: str, cited: list[EvidenceItem]) -> bool:
        if self.entailment is not None:
            result = self.entailment(claim, cited)
            if inspect.isawaitable(result):
                return bool(await result)
            return bool(result)
        return _supported_strict(claim, cited)

    async def _coverage(
        self, text: str, evidence: list[EvidenceItem], *, check_entailment: bool
    ) -> CitationCoverage:
        lookup: dict[str, EvidenceItem] = {}
        for item in evidence:
            for key in _evidence_keys(item):
                lookup.setdefault(key, item)
        claims = _verifiable_claims(text)
        checks: list[ClaimCheck] = []
        cited_count = 0
        entailed_count = 0
        for claim in claims:
            markers = extract_citations(claim)
            cited_items = [lookup[m] for m in markers if m in lookup]
            cited = bool(cited_items)
            if cited:
                cited_count += 1
            entailed: bool | None = None
            score = 0.0
            if cited and check_entailment:
                entailed = await self._entailed(claim, cited_items)
                if entailed:
                    entailed_count += 1
                score = 1.0 if entailed else 0.0
            checks.append(
                ClaimCheck(
                    claim=claim[:200],
                    citations=markers,
                    cited=cited,
                    entailed=entailed,
                    support_score=score,
                )
            )
        total = len(claims)
        coverage = cited_count / total if total else 1.0
        entailment_rate = (entailed_count / cited_count) if (check_entailment and cited_count) else None
        return CitationCoverage(
            claims=total,
            cited_claims=cited_count,
            entailed_claims=entailed_count,
            coverage=round(coverage, 4),
            entailment_rate=round(entailment_rate, 4) if entailment_rate is not None else None,
            claim_checks=checks,
        )

    async def build_report(
        self,
        answer: Any,
        evidence: list[EvidenceItem],
        *,
        title: str = "",
        contract: CitationContract | None = None,
        figures: list[Figure] | None = None,
        catalog: DataCatalog | None = None,
    ) -> CitedReport:
        """Resolve citations and assemble a :class:`CitedReport`.

        Enforces ``contract`` when given: coverage floor, no unresolved markers,
        per-claim entailment, and (with ``figures``) per-figure data binding —
        raising :class:`~vincio.core.errors.CitationValidationError` on a breach.

        ``figures`` embeds charts/tables (:class:`Figure`) as **data-bound** parts of
        the deliverable: each gets a ``[F1]``-style marker the narrative can
        reference, is rendered into the document, and — when a ``catalog`` is given —
        is verified to re-derive from its source, the per-figure analogue of a
        claim's entailment.
        """
        figs = self._number_figures(figures or [])
        figure_evidence = [f.evidence_item() for f in figs]
        pool = list(evidence) + figure_evidence

        text = self._answer_text(answer)
        check_entailment = bool(contract and contract.require_entailment)
        rewritten, resolved, unresolved = self._resolve(text, pool)
        coverage = await self._coverage(text, pool, check_entailment=check_entailment)
        bindings = self._bind_figures(figs, catalog, coverage)

        if contract is not None:
            self._enforce(contract, coverage, unresolved)

        model = markdown_to_model(rewritten, title=title)
        model.footnotes = [f"[{c.number}] {c.footnote()}" for c in resolved]
        model.bibliography = _bibliography(resolved)
        model.source_evidence_ids = [c.evidence_id for c in resolved]
        self._render_figures(model, figs, bindings)
        model.metadata.update(
            {
                "citation_coverage": coverage.coverage,
                "entailment_rate": coverage.entailment_rate,
                "unresolved_markers": unresolved,
                "figure_binding_rate": coverage.figure_binding_rate,
            }
        )
        report = CitedReport(
            document=model,
            citations=resolved,
            coverage=coverage,
            unresolved_markers=unresolved,
            figures=bindings,
        )
        self._audit(report)
        return report

    @staticmethod
    def _number_figures(figures: list[Figure]) -> list[Figure]:
        """Assign ``F1``-style markers to any figure that did not declare one."""
        out: list[Figure] = []
        for i, fig in enumerate(figures, start=1):
            out.append(fig if fig.marker else fig.model_copy(update={"marker": f"F{i}"}))
        return out

    def _bind_figures(
        self, figures: list[Figure], catalog: DataCatalog | None, coverage: CitationCoverage
    ) -> list[FigureBinding]:
        bindings: list[FigureBinding] = []
        bound = 0
        checked = 0
        for fig in figures:
            verdict = fig.verify(catalog)
            if verdict is not None:
                checked += 1
                bound += int(verdict)
            bindings.append(
                FigureBinding(
                    marker=fig.marker,
                    kind=fig.kind,
                    caption=fig.caption,
                    cite_refs=list(fig.cite_refs),
                    coverage=fig.coverage,
                    data_bound=verdict,
                )
            )
        coverage.figures = len(figures)
        coverage.data_bound_figures = bound
        coverage.figure_binding_rate = round(bound / checked, 4) if checked else None
        return bindings

    @staticmethod
    def _render_figures(
        model: DocumentModel, figures: list[Figure], bindings: list[FigureBinding]
    ) -> None:
        if not figures:
            return
        model.heading("Figures", level=2)
        for fig, binding in zip(figures, bindings, strict=True):
            if binding.data_bound is None:
                status = ""
            elif binding.data_bound:
                status = " · data-bound"
            else:
                status = " · UNVERIFIED"
            model.paragraph(f"**[{fig.marker}]** {fig.caption}{status}")
            model.add_table(fig.table_data())
            if fig.cite_refs:
                model.paragraph("Sources: " + ", ".join(fig.cite_refs))

    @staticmethod
    def _enforce(
        contract: CitationContract, coverage: CitationCoverage, unresolved: list[str]
    ) -> None:
        if not contract.allow_unresolved_markers and unresolved:
            raise CitationValidationError(
                f"citation markers reference no evidence: {unresolved}"
            )
        if coverage.coverage < contract.min_coverage:
            raise CitationValidationError(
                f"citation coverage {coverage.coverage} below required {contract.min_coverage}"
            )
        if contract.require_entailment:
            rate = coverage.entailment_rate if coverage.entailment_rate is not None else 1.0
            if rate < contract.min_entailment_rate:
                raise CitationValidationError(
                    f"claim entailment rate {rate} below required {contract.min_entailment_rate}"
                )
        if contract.require_figure_binding:
            rate = coverage.figure_binding_rate if coverage.figure_binding_rate is not None else 1.0
            if rate < contract.min_figure_binding_rate:
                raise CitationValidationError(
                    f"figure data-binding rate {rate} below required "
                    f"{contract.min_figure_binding_rate} (pass catalog= to check, and confirm "
                    "every figure re-derives from its source)"
                )

    async def build(
        self,
        answer: Any,
        evidence: list[EvidenceItem],
        *,
        format: RenderFormat = "markdown",
        title: str = "",
        contract: CitationContract | None = None,
        figures: list[Figure] | None = None,
        catalog: DataCatalog | None = None,
    ) -> DocumentArtifact:
        """Build a cited report and render it via the document engine."""
        report = await self.build_report(
            answer, evidence, title=title, contract=contract, figures=figures, catalog=catalog
        )
        builder = DocumentBuilder(audit_log=self.audit_log, tenant_id=self.tenant_id)
        return builder.build(report.document, format=format, title=title)

    def _audit(self, report: CitedReport) -> None:
        if self.audit_log is None:
            return
        self.audit_log.record(
            "cited_report",
            tenant_id=self.tenant_id,
            resource=report.document.title or "cited_report",
            details={
                "citations": len(report.citations),
                "coverage": report.coverage.coverage,
                "entailment_rate": report.coverage.entailment_rate,
                "unresolved_markers": report.unresolved_markers,
                "source_evidence_ids": report.document.source_evidence_ids,
                "figures": len(report.figures),
                "data_bound_figures": report.coverage.data_bound_figures,
                "figure_binding_rate": report.coverage.figure_binding_rate,
            },
        )


def _bibliography(citations: list[ResolvedCitation]) -> list[str]:
    seen: dict[str, str] = {}
    for citation in citations:
        if citation.source_id in seen:
            continue
        uri = f" — {citation.source_uri}" if citation.source_uri else ""
        seen[citation.source_id] = f"{citation.source_id} [{citation.trust_level}]{uri}"
    return list(seen.values())
