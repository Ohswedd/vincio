"""Result models for the open evaluation plane — one run's verifiable record.

Every number a report or a leaderboard prints comes from here, and every number
carries its **provenance tier**. The ``determinism_digest`` is a content hash over
the sorted per-item outcomes (and *not* over wall-clock, run id, or duration), so
two Tier-S / Tier-R runs on different machines produce the same digest — the
property the determinism gate checks.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from ...core.utils import compact_hash, utcnow
from .tiers import ProvenanceTier

__all__ = ["ItemResult", "BenchmarkRun", "SuiteRun"]


class ItemResult(BaseModel):
    """One scored task: the verifiable outcome the report cites."""

    task_id: str
    success: bool
    score: float = 0.0
    tier: ProvenanceTier = ProvenanceTier.STATIC
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRun(BaseModel):
    """One benchmark's run over one model at one tier."""

    benchmark_id: str
    niche: str = ""
    title: str = ""
    tier: ProvenanceTier = ProvenanceTier.STATIC
    requested_tier: ProvenanceTier = ProvenanceTier.STATIC
    primary_metric: str = "accuracy"
    primary: float = 0.0
    success_rate: float = 0.0
    mean_score: float = 0.0
    n: int = 0
    task_set_hash: str = ""
    replayed: bool = True
    source: str = "fabricated"
    items: list[ItemResult] = Field(default_factory=list)
    # long-context uplift: {"base": float, "governed": float, "uplift": float}
    governed: dict[str, float] | None = None
    duration_ms: int = 0

    @property
    def determinism_digest(self) -> str:
        """A content hash over the sorted per-item outcomes — wall-clock excluded."""
        canonical = [
            [i.task_id, i.success, round(i.score, 6), i.tier.value]
            for i in sorted(self.items, key=lambda r: r.task_id)
        ]
        payload = {
            "benchmark": self.benchmark_id,
            "tier": self.tier.value,
            "task_set_hash": self.task_set_hash,
            "items": canonical,
            "governed": self.governed,
        }
        return compact_hash(payload)

    @property
    def failures(self) -> list[ItemResult]:
        """The scored items the model got wrong (cited in the report's detail)."""
        return [i for i in self.items if not i.success]


class SuiteRun(BaseModel):
    """A whole suite run: one model over a set of benchmarks at one tier."""

    run_id: str
    model: str = "mock"
    provider: str = ""
    tier: ProvenanceTier = ProvenanceTier.STATIC
    created_at: Any = Field(default_factory=utcnow)
    environment: dict[str, Any] = Field(default_factory=dict)
    runs: list[BenchmarkRun] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- aggregation ----------------------------------------------------------

    def by_niche(self) -> dict[str, list[BenchmarkRun]]:
        """Benchmark runs grouped by niche, in stable order."""
        grouped: dict[str, list[BenchmarkRun]] = {}
        for run in self.runs:
            grouped.setdefault(run.niche, []).append(run)
        return grouped

    def niche_scores(self) -> dict[str, float]:
        """The mean primary score per niche (the radar axes)."""
        scores: dict[str, float] = {}
        for niche, runs in self.by_niche().items():
            if runs:
                scores[niche] = round(sum(r.primary for r in runs) / len(runs), 4)
        return scores

    def overall(self) -> float:
        """The unweighted mean primary score across every benchmark."""
        if not self.runs:
            return 0.0
        return round(sum(r.primary for r in self.runs) / len(self.runs), 4)

    @property
    def determinism_digest(self) -> str:
        """A content hash over every benchmark's digest — the suite-level pin."""
        parts = sorted(f"{r.benchmark_id}:{r.determinism_digest}" for r in self.runs)
        # Historical spaced-separator form — persisted determinism pins depend
        # on these exact bytes; do NOT switch to core.utils.compact_hash.
        blob = json.dumps({"tier": self.tier.value, "runs": parts}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def gated(self) -> bool:
        """Whether this run's tier is allowed to gate CI (Static / Recorded)."""
        return self.tier.gates_ci
