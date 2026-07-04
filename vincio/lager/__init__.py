"""LAGER — Lazy Graph Evidence Retrieval: reasoning-driven retrieval (RDR).

A retrieval plane where reasoning drives retrieval instead of retrieval
driving reasoning. The corpus becomes **Evidence Objects** — atomic claims
lifted byte-exactly from their sources, with provenance, entities, typed
relations, confidence, and temporal validity — connected in a lightweight
knowledge graph. Retrieval is **lazy**: evidence is acquired incrementally,
guided by explicit information needs, expanding the graph only while the
marginal information gain justifies it — no fixed top-k, no chunk windows, no
prompt stuffing. The model then receives the verified minimum, every answer
traceable to its objects and source spans, and unanswerable queries abstain
honestly instead of guessing.

    from vincio.lager import LagerEngine

    engine = LagerEngine()
    engine.ingest(documents)
    pack = engine.retrieve("how does the outage relate to the TLS rotation?")
    for line in pack.gain_trace:      # every retrieval decision, explainable
        print(line)

Deterministic by default (same bytes in → same objects, ids, edges, and trace
out), offline-verifiable end to end, and modular — extractor, embedder,
planner, and controller are all independently replaceable.
"""

from .answer import (
    LagerAnswer,
    build_context,
    cited_ids,
    estimate_confidence,
    generate_answer,
    verify_answer,
)
from .controller import LazyOptions, LazyRetriever
from .engine import LagerEngine
from .extract import ClaimExtractor, DeterministicClaimExtractor, normalize_entities
from .graph import EvidenceGraph, claims_contradict
from .index import EvidenceIndex, fuse_ranked
from .objects import (
    EvidenceObject,
    EvidencePack,
    EvidenceRelation,
    canonical_text,
    document_key,
)
from .planner import InformationNeed, QueryPlanner

__all__ = [
    "ClaimExtractor",
    "DeterministicClaimExtractor",
    "EvidenceGraph",
    "EvidenceIndex",
    "EvidenceObject",
    "EvidencePack",
    "EvidenceRelation",
    "InformationNeed",
    "LagerAnswer",
    "LagerEngine",
    "LazyOptions",
    "LazyRetriever",
    "QueryPlanner",
    "build_context",
    "canonical_text",
    "cited_ids",
    "claims_contradict",
    "document_key",
    "estimate_confidence",
    "fuse_ranked",
    "generate_answer",
    "normalize_entities",
    "verify_answer",
]
