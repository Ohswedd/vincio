"""Versioned index/retrieval regression artifacts.

A re-embed, a chunking tweak, or an index swap can silently regress retrieval
quality. This module persists a **regression artifact** for each retrieval
configuration — keyed on ``(embedder, chunker, corpus hash)`` — so a later run on
the *same* golden query set is comparable against a stable baseline, and a drop
in recall/nDCG is caught instead of shipped.

Artifacts persist through the standard :class:`~vincio.storage.base.MetadataStore`
(in-memory by default; SQLite/Postgres when wired), so the regression history is
stored exactly like runs and traces. The comparison/gating logic lives in
:mod:`vincio.evals.retrieval_eval`, which reuses the same significance machinery
a model swap is gated on — this module is the durable record, not the judge.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["IndexRegressionArtifact", "IndexRegressionStore", "config_key"]

# The metadata-store table these artifacts live in.
_KIND = "index_regression"


def config_key(embedder: str, chunker: str, corpus_hash: str, *, reranker: str = "", index: str = "") -> str:
    """The stable key for a retrieval configuration over a fixed corpus.

    Keyed on ``(embedder, chunker, corpus hash)`` — plus reranker/index when set —
    so two runs of the same pipeline over the same corpus compare, and any change
    to those dimensions starts a fresh regression lineage.
    """
    blob = "|".join([embedder, chunker, reranker, index, corpus_hash])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class IndexRegressionArtifact(BaseModel):
    """One recorded measurement of a retrieval config against a golden set."""

    id: str = ""  # set on record() to "<key>:v<version>"
    key: str
    embedder: str = ""
    chunker: str = ""
    reranker: str = ""
    index: str = ""
    corpus_hash: str = ""
    n_queries: int = 0
    version: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)  # aggregate means
    report: dict[str, Any] = Field(default_factory=dict)  # full EvalReport (per-case)

    @classmethod
    def key_for(
        cls, embedder: str, chunker: str, corpus_hash: str, *, reranker: str = "", index: str = ""
    ) -> str:
        return config_key(embedder, chunker, corpus_hash, reranker=reranker, index=index)


class IndexRegressionStore:
    """Persist and look up :class:`IndexRegressionArtifact`\\ s by config key.

    Each :meth:`record` appends a new version under the artifact's key, so the
    history is preserved and :meth:`baseline` returns the most recent *prior*
    measurement to compare a fresh run against.
    """

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from .base import InMemoryMetadataStore

            store = InMemoryMetadataStore()
        self.store = store

    def history(self, key: str) -> list[IndexRegressionArtifact]:
        rows = self.store.query(_KIND, where={"key": key}, limit=10_000)
        artifacts = [IndexRegressionArtifact.model_validate(r) for r in rows]
        return sorted(artifacts, key=lambda a: a.version)

    def baseline(self, key: str) -> IndexRegressionArtifact | None:
        """The latest recorded artifact for ``key`` (the regression baseline)."""
        history = self.history(key)
        return history[-1] if history else None

    def record(self, artifact: IndexRegressionArtifact) -> IndexRegressionArtifact:
        """Append ``artifact`` as the next version under its key."""
        history = self.history(artifact.key)
        artifact.version = (history[-1].version + 1) if history else 1
        artifact.id = f"{artifact.key}:v{artifact.version}"
        self.store.save(_KIND, artifact.model_dump(mode="json"))
        return artifact

    def latest_metrics(self, key: str) -> dict[str, float]:
        artifact = self.baseline(key)
        return dict(artifact.metrics) if artifact else {}
