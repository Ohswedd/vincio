"""Human-in-the-loop annotation and judge calibration.

A local :class:`AnnotationQueue` records human labels next to LLM-judge scores
on the metadata store (kind ``annotation_labels``) and tracks **Cohen's κ**
between human and judge. A judge only earns CI-gating weight once agreement
clears a threshold — so you trust an automated judge as a gate exactly when it
has demonstrably agreed with people. Everything is local; no labels leave the
process.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import new_id, utcnow
from ..storage.base import InMemoryMetadataStore, MetadataStore

__all__ = ["cohens_kappa", "AnnotationItem", "AnnotationQueue"]


def _binify(value: float, bins: int) -> int:
    """Bin a score in [0, 1] into one of ``bins`` ordinal categories."""
    return min(bins - 1, max(0, int(float(value) * bins)))


def cohens_kappa(pairs: list[tuple[float, float]], *, bins: int = 2) -> float:
    """Cohen's κ between two raters over (rater_a, rater_b) score pairs.

    Scores are expected in [0, 1] and binned into ``bins`` ordinal categories.
    κ corrects observed agreement for agreement expected by chance:
    ``κ = (p_o − p_e) / (1 − p_e)``. Returns 1.0 when there is no chance
    disagreement to correct for (e.g. one rater is constant and they agree)."""
    if len(pairs) < 2:
        raise ValueError("Cohen's kappa requires at least 2 pairs")
    a = [_binify(x, bins) for x, _ in pairs]
    b = [_binify(y, bins) for _, y in pairs]
    n = len(pairs)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum((count_a[k] / n) * (count_b[k] / n) for k in set(a) | set(b))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 4)


class AnnotationItem(BaseModel):
    id: str = Field(default_factory=lambda: new_id("anno"))
    queue: str = "default"
    run_id: str = ""
    case_id: str = ""
    input: str = ""
    output: str = ""
    judge_score: float | None = None
    human_score: float | None = None
    label: str = ""
    annotator: str = ""
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())


class AnnotationQueue:
    """A local queue of items to label, tracking human↔judge agreement."""

    KIND = "annotation_labels"

    def __init__(
        self,
        store: MetadataStore | None = None,
        *,
        name: str = "default",
        app_name: str = "",
        bins: int = 2,
    ) -> None:
        self.store: MetadataStore = store or InMemoryMetadataStore()
        self.name = name
        self.app_name = app_name
        self.bins = bins

    # -- queue operations ----------------------------------------------------

    def add(
        self,
        *,
        run_id: str = "",
        case_id: str = "",
        input: str = "",
        output: str = "",
        judge_score: float | None = None,
    ) -> AnnotationItem:
        item = AnnotationItem(
            queue=self.name, run_id=run_id, case_id=case_id, input=input,
            output=output, judge_score=judge_score,
        )
        self._save(item)
        return item

    def label(
        self, item_id: str, human_score: float, *, label: str = "", annotator: str = ""
    ) -> AnnotationItem:
        record = self.store.get(self.KIND, item_id)
        if record is None:
            raise KeyError(f"annotation item {item_id!r} not found")
        item = AnnotationItem.model_validate(record)
        item.human_score = float(human_score)
        item.label = label
        item.annotator = annotator
        self._save(item)
        return item

    def items(self) -> list[AnnotationItem]:
        rows = self.store.query(self.KIND, where={"queue": self.name}, limit=100_000)
        items = [AnnotationItem.model_validate(r) for r in rows]
        return sorted(items, key=lambda i: i.created_at)

    def pending(self) -> list[AnnotationItem]:
        return [i for i in self.items() if i.human_score is None]

    def labeled(self) -> list[AnnotationItem]:
        return [i for i in self.items() if i.human_score is not None]

    # -- agreement -----------------------------------------------------------

    def pairs(self) -> list[tuple[float, float]]:
        return [
            (i.judge_score, i.human_score)
            for i in self.items()
            if i.judge_score is not None and i.human_score is not None
        ]

    def cohens_kappa(self, *, bins: int | None = None) -> float:
        return cohens_kappa(self.pairs(), bins=bins or self.bins)

    def agreement(self, *, bins: int | None = None) -> dict[str, Any]:
        pairs = self.pairs()
        if len(pairs) < 2:
            return {"cohens_kappa": None, "n": len(pairs), "exact_agreement": None}
        bin_count = bins or self.bins
        a = [_binify(j, bin_count) for j, _ in pairs]
        b = [_binify(h, bin_count) for _, h in pairs]
        exact = sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(pairs)
        return {
            "cohens_kappa": cohens_kappa(pairs, bins=bin_count),
            "exact_agreement": round(exact, 4),
            "n": len(pairs),
            "bins": bin_count,
        }

    def judge_trusted(self, *, threshold: float = 0.6, bins: int | None = None) -> bool:
        """Whether the judge has earned CI-gating weight (κ ≥ threshold)."""
        pairs = self.pairs()
        if len(pairs) < 2:
            return False
        return self.cohens_kappa(bins=bins) >= threshold

    def gating_weight(self, *, threshold: float = 0.6, bins: int | None = None) -> float:
        """1.0 if the judge clears the agreement bar, else 0.0 — multiply a
        judge's gate weight by this so an uncalibrated judge cannot block CI."""
        return 1.0 if self.judge_trusted(threshold=threshold, bins=bins) else 0.0

    # -- internals -----------------------------------------------------------

    def _save(self, item: AnnotationItem) -> None:
        record = item.model_dump(mode="json")
        record["app_id"] = self.app_name
        self.store.save(self.KIND, record)
