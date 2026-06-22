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
from ..evals.metrics import _supported_strict, _verifiable_claims
from ..output.parsers import extract_citations
from .builder import DocumentBuilder, markdown_to_model
from .model import DocumentModel
from .render import DocumentArtifact, RenderFormat, render

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.audit import AuditLog

__all__ = [
    "ResolvedCitation",
    "ClaimCheck",
    "CitationCoverage",
    "CitationContract",
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


class CitationContract(BaseModel):
    """Field/claim-level citation requirements for a cited report."""

    min_coverage: float = 1.0  # min fraction of verifiable claims that must cite
    require_entailment: bool = False  # cited evidence must support the claim
    min_entailment_rate: float = 1.0
    allow_unresolved_markers: bool = False  # markers not matching any evidence


class CitedReport(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    document: DocumentModel
    citations: list[ResolvedCitation] = Field(default_factory=list)
    coverage: CitationCoverage = Field(default_factory=CitationCoverage)
    unresolved_markers: list[str] = Field(default_factory=list)

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
    ) -> CitedReport:
        """Resolve citations and assemble a :class:`CitedReport`.

        Enforces ``contract`` when given: coverage floor, no unresolved markers,
        and (optionally) per-claim entailment — raising
        :class:`~vincio.core.errors.CitationValidationError` on a breach.
        """
        text = self._answer_text(answer)
        check_entailment = bool(contract and contract.require_entailment)
        rewritten, resolved, unresolved = self._resolve(text, evidence)
        coverage = await self._coverage(text, evidence, check_entailment=check_entailment)

        if contract is not None:
            self._enforce(contract, coverage, unresolved)

        model = markdown_to_model(rewritten, title=title)
        model.footnotes = [f"[{c.number}] {c.footnote()}" for c in resolved]
        model.bibliography = _bibliography(resolved)
        model.source_evidence_ids = [c.evidence_id for c in resolved]
        model.metadata.update(
            {
                "citation_coverage": coverage.coverage,
                "entailment_rate": coverage.entailment_rate,
                "unresolved_markers": unresolved,
            }
        )
        report = CitedReport(
            document=model, citations=resolved, coverage=coverage, unresolved_markers=unresolved
        )
        self._audit(report)
        return report

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

    async def build(
        self,
        answer: Any,
        evidence: list[EvidenceItem],
        *,
        format: RenderFormat = "markdown",
        title: str = "",
        contract: CitationContract | None = None,
    ) -> DocumentArtifact:
        """Build a cited report and render it via the document engine."""
        report = await self.build_report(answer, evidence, title=title, contract=contract)
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
