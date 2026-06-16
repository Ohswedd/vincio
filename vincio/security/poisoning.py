"""RAG-poisoning detection on retrieved evidence.

A handful of crafted documents inserted into a corpus can flip a large fraction
of answers — the retrieval-time analogue of prompt injection. Vincio already
tags trust and scores authority/provenance/freshness on every
:class:`~vincio.core.types.EvidenceItem`; this detector turns those signals into
a deterministic poisoning verdict *before* poisoned evidence reaches the model:

* **Embedded instructions** — the evidence text carries injection signals
  (reuses the :class:`~vincio.security.injection.InjectionDetector`).
* **Low authority/provenance, high promotion** — an untrusted, low-provenance
  source that nonetheless ranks highly is a classic poisoning shape.
* **Consensus outlier** — when scanning a set, an item whose authority is far
  below its peers is flagged as a likely injected document.

An optional async **classifier hook** (PromptArmor-class) blends in, exactly
like the injection detector — the deterministic layers never depend on it. The
:class:`PoisoningReport` carries **FP/FN telemetry** against labelled evidence
so the detector's own precision/recall is measurable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .injection import InjectionDetector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import EvidenceItem

__all__ = ["PoisonSignal", "PoisonVerdict", "PoisoningReport", "PoisoningDetector"]

ClassifierFn = Callable[[str], Awaitable[float]]


class PoisonSignal(BaseModel):
    name: str
    weight: float
    detail: str = ""


class PoisonVerdict(BaseModel):
    evidence_id: str
    source_id: str = ""
    poisoned: bool
    risk: float
    signals: list[PoisonSignal] = Field(default_factory=list)


class PoisoningReport(BaseModel):
    verdicts: list[PoisonVerdict] = Field(default_factory=list)

    @property
    def flagged(self) -> list[PoisonVerdict]:
        return [v for v in self.verdicts if v.poisoned]

    @property
    def flagged_ids(self) -> list[str]:
        return [v.evidence_id for v in self.flagged]

    def telemetry(self, poisoned_ids: set[str]) -> dict[str, float]:
        """Precision/recall/FP/FN of the detector against known labels."""
        flagged = {v.evidence_id for v in self.flagged}
        all_ids = {v.evidence_id for v in self.verdicts}
        clean = all_ids - poisoned_ids
        tp = len(flagged & poisoned_ids)
        fp = len(flagged & clean)
        fn = len(poisoned_ids - flagged)
        tn = len(clean - flagged)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        return {
            "true_positives": float(tp),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_negatives": float(tn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
            "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
        }


def _is_untrusted(item: EvidenceItem) -> bool:
    trust = getattr(getattr(item, "trust_level", None), "value", "")
    return trust.startswith("untrusted")


class PoisoningDetector:
    """Flag likely-poisoned retrieved evidence from authority/provenance signals."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_authority: float = 0.35,
        min_provenance: float = 0.35,
        classifier: ClassifierFn | None = None,
        injection_detector: InjectionDetector | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_authority = min_authority
        self.min_provenance = min_provenance
        self.classifier = classifier
        self.injection = injection_detector or InjectionDetector()

    @staticmethod
    def _combine(signals: list[PoisonSignal]) -> float:
        # Noisy-or over signal weights (same combiner the injection detector uses).
        risk = 1.0
        for signal in signals:
            risk *= 1.0 - signal.weight
        return round(1.0 - risk, 4)

    def inspect(
        self, item: EvidenceItem, *, peer_authority_mean: float | None = None
    ) -> PoisonVerdict:
        """Verdict for a single evidence item (optionally vs. its peers' mean)."""
        signals: list[PoisonSignal] = []
        text = item.text or ""

        verdict = self.injection.detect(text)
        if verdict.detected:
            signals.append(PoisonSignal(
                name="embedded_instruction", weight=min(0.9, max(0.6, verdict.risk)),
                detail=f"injection risk {verdict.risk} in retrieved text"))

        if _is_untrusted(item) and item.authority < self.min_authority and item.relevance >= 0.6:
            signals.append(PoisonSignal(
                name="low_authority_high_promotion", weight=0.55,
                detail=f"authority={item.authority} but relevance={item.relevance}"))

        if item.provenance < self.min_provenance:
            signals.append(PoisonSignal(
                name="weak_provenance", weight=0.4,
                detail=f"provenance={item.provenance} below {self.min_provenance}"))

        if peer_authority_mean is not None and item.authority < peer_authority_mean - 0.3:
            signals.append(PoisonSignal(
                name="consensus_outlier", weight=0.5,
                detail=f"authority={item.authority} far below peer mean {round(peer_authority_mean, 3)}"))

        risk = self._combine(signals)
        return PoisonVerdict(
            evidence_id=item.id, source_id=item.source_id,
            poisoned=risk >= self.threshold, risk=risk, signals=signals)

    def scan(self, evidence: list[EvidenceItem]) -> PoisoningReport:
        """Scan a set of evidence, using the set's authority distribution."""
        if not evidence:
            return PoisoningReport()
        mean_authority = sum(e.authority for e in evidence) / len(evidence)
        verdicts = [self.inspect(item, peer_authority_mean=mean_authority) for item in evidence]
        return PoisoningReport(verdicts=verdicts)

    async def ascan(self, evidence: list[EvidenceItem]) -> PoisoningReport:
        """Like :meth:`scan` but blends an optional async classifier per item."""
        report = self.scan(evidence)
        if self.classifier is None:
            return report
        by_id = {item.id: item for item in evidence}
        for verdict in report.verdicts:
            if verdict.poisoned:
                continue
            item = by_id.get(verdict.evidence_id)
            text = (item.text or "") if item is not None else ""
            if not text:
                continue
            model_risk = await self.classifier(text)
            blended = max(verdict.risk, model_risk)
            if blended != verdict.risk:
                verdict.risk = round(blended, 4)
                verdict.poisoned = blended >= self.threshold
                verdict.signals.append(PoisonSignal(
                    name="classifier", weight=model_risk, detail="external poisoning classifier"))
        return report
