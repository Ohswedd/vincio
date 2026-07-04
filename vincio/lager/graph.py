"""The evidence knowledge graph — typed edges between Evidence Objects.

Construction is deterministic and bounded. Edges:

* ``follows`` — adjacent claims from the same document (narrative order),
* ``depends_on`` — a claim opening with a referring form points at its
  antecedent, so packing can keep pronoun-opening claims self-contained,
* ``supports`` — same-bucket claims with high lexical affinity that do not
  contradict,
* ``contradicts`` — a **gated** detector: the shared memory heuristic
  (:func:`~vincio.memory.policies.detect_contradiction`) fires on ordinary
  temporal/scope variation ("raised prices in March" vs "… in June"), so LAGER
  applies its own guards — temporal-scope alignment, a qualifier-aware
  value-divergence test, and a containment requirement on the negation branch —
  and records the detector basis on every edge so downstream confidence
  penalties stay traceable.

Pairwise work is bounded: ubiquitous entities (document frequency above the
cut) are skipped for pair generation, and buckets are hard-capped with
deterministic member selection — both observably (``note_suppressed``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from ..context.scoring import lexical_similarity, near_duplicate_score
from ..core.diagnostics import note_suppressed
from .extract import content_terms, is_referring
from .objects import EvidenceObject, EvidenceRelation

__all__ = ["EvidenceGraph", "claims_contradict"]

_NEGATION_RE = re.compile(
    r"\b(not|no|never|cannot|can't|won't|isn't|doesn't|don't|didn't|wasn't|weren't|without)\b",
    re.IGNORECASE,
)
_QUALIFIER_RE = re.compile(
    r"\b(?:\d{4}(?:-\d{2}(?:-\d{2})?)?|q[1-4]|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_MIN_CONTENT_TERMS = 6
_SUPPORT_THRESHOLD = 0.5
_BUCKET_CAP = 128
_UBIQUITY_FRACTION = 0.10
_UBIQUITY_MIN_CORPUS = 50


def _strip_qualifiers(terms: set[str], a: EvidenceObject, b: EvidenceObject) -> set[str]:
    """Remove temporal qualifiers and one-sided entity names from a divergence
    set — a different quarter or a different product is a different *slot*, not
    a different value for the same slot."""
    kept = {t for t in terms if not _QUALIFIER_RE.fullmatch(t)}
    one_sided_entities: set[str] = set()
    for entity in [*a.entities, *b.entities]:
        if (entity in a.entities) != (entity in b.entities):
            one_sided_entities.update(entity.split())
    return {t for t in kept if t not in one_sided_entities}


def claims_contradict(a: EvidenceObject, b: EvidenceObject) -> str | None:
    """The gated contradiction test: the basis string when *a* and *b* genuinely
    conflict, ``None`` otherwise.

    Guards (each suppresses a measured false-positive class of the raw memory
    heuristic): differing temporal scope, qualifier-only divergence, short
    claims, and a containment requirement when only negation differs."""
    if a.observed_at and b.observed_at and a.observed_at != b.observed_at:
        return None  # different time scopes can both be true
    terms_a, terms_b = content_terms(a.claim), content_terms(b.claim)
    if len(terms_a) < _MIN_CONTENT_TERMS or len(terms_b) < _MIN_CONTENT_TERMS:
        return None
    similarity = near_duplicate_score(a.claim, b.claim)
    if similarity < 0.45:
        return None  # unrelated
    negation_differs = bool(_NEGATION_RE.search(a.claim)) != bool(_NEGATION_RE.search(b.claim))
    set_a, set_b = set(terms_a), set(terms_b)
    if negation_differs:
        # Opposite polarity is a contradiction only when the claims talk about
        # the same thing: the negation-stripped smaller side must be ~contained
        # in the larger. (Checked BEFORE the restatement cut — "X was enabled"
        # vs "X was not enabled" is near-identical lexically and is exactly the
        # contradiction, never a restatement.)
        stripped_a = {t for t in set_a if not _NEGATION_RE.fullmatch(t)}
        stripped_b = {t for t in set_b if not _NEGATION_RE.fullmatch(t)}
        smaller, larger = sorted((stripped_a, stripped_b), key=len)
        if smaller and len(smaller & larger) / len(smaller) >= 0.8:
            return "negation"
        return None
    if similarity >= 0.95:
        return None  # same polarity, near-identical text: a restatement
    diverge_a = _strip_qualifiers(set_a - set_b, a, b)
    diverge_b = _strip_qualifiers(set_b - set_a, a, b)
    # A genuine value conflict swaps exactly ONE slot in the same frame
    # ("returns JSON" vs "returns XML"). Two-plus differing tokens per side
    # means independent slots differ ("advanced … enterprise" vs "basic …
    # individual") — different statements, both can be true.
    if len(diverge_a) == 1 and len(diverge_b) == 1 and similarity >= 0.55:
        return "value-divergence"
    return None


class EvidenceGraph:
    """Typed adjacency over Evidence Objects plus an entity co-occurrence layer.

    Every accessor is sorted-stable (weight descending, id ascending) — no set
    ordering ever reaches an output."""

    def __init__(self) -> None:
        self.objects: dict[str, EvidenceObject] = {}
        #: bucket key (entity or fallback term) → sorted EO ids
        self.buckets: dict[str, list[str]] = {}
        #: eo id → outgoing relations (sorted on build)
        self.edges: dict[str, list[EvidenceRelation]] = defaultdict(list)
        #: entity → co-occurring entity → shared-object count
        self.entity_edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        #: doc_key → the document's EO ids in narrative order (document
        #: coherence: an anchored document's other claims are frontier
        #: candidates even when they share no words with the query)
        self.doc_objects: dict[str, list[str]] = {}
        self.contradictions: list[tuple[str, str, str]] = []

    def __len__(self) -> int:
        return len(self.objects)

    # -- construction ----------------------------------------------------------------

    def build(self, objects: list[EvidenceObject]) -> None:
        """(Re)build the graph from *objects* — deterministic for a fixed input."""
        ordered = sorted(objects, key=lambda o: o.id)
        self.objects = {o.id: o for o in ordered}
        self._build_buckets(ordered)
        self._sequential_edges(objects)  # narrative order needs input order
        self._bucket_edges()
        self._entity_cooccurrence(ordered)
        for relations in self.edges.values():
            relations.sort(key=lambda r: (-r.weight, r.kind, r.target))
        for obj in self.objects.values():
            obj.relations = list(self.edges.get(obj.id, []))

    def _build_buckets(self, ordered: list[EvidenceObject]) -> None:
        raw: dict[str, list[str]] = defaultdict(list)
        for obj in ordered:
            for key in dict.fromkeys((*obj.entities, *obj.terms)):
                raw[key].append(obj.id)
        self.buckets = {key: sorted(ids) for key, ids in sorted(raw.items())}

    def _sequential_edges(self, objects: list[EvidenceObject]) -> None:
        """``follows`` between adjacent same-document claims; ``depends_on``
        from a referring claim to its immediate antecedent."""
        by_doc: dict[str, list[EvidenceObject]] = defaultdict(list)
        for obj in objects:  # input order == document order
            by_doc[obj.doc_key].append(obj)
        self.doc_objects = {key: [o.id for o in seq] for key, seq in sorted(by_doc.items())}
        for sequence in by_doc.values():
            for i, obj in enumerate(sequence):
                for distance in (1, 2):
                    if i + distance < len(sequence):
                        self.edges[obj.id].append(EvidenceRelation(
                            kind="follows", target=sequence[i + distance].id,
                            weight=1.0 / distance, basis="adjacent",
                        ))
                if i > 0 and obj.kind == "claim" and is_referring(obj.claim):
                    self.edges[obj.id].append(EvidenceRelation(
                        kind="depends_on", target=sequence[i - 1].id,
                        weight=1.0, basis="referring-opener",
                    ))

    def _bucket_edges(self) -> None:
        """``supports`` / ``contradicts`` inside bounded entity/term buckets."""
        total = max(len(self.objects), 1)
        ubiquity_cut = max(int(total * _UBIQUITY_FRACTION), 4)
        seen_pairs: set[tuple[str, str]] = set()
        for key, ids in self.buckets.items():
            if total >= _UBIQUITY_MIN_CORPUS and len(ids) > ubiquity_cut:
                note_suppressed("lager.bucket_ubiquitous")
                continue  # a term in most claims relates nothing specific
            members = ids
            if len(members) > _BUCKET_CAP:
                note_suppressed("lager.bucket_truncated")
                members = members[:_BUCKET_CAP]  # ids sorted → deterministic cut
            for i, id_a in enumerate(members):
                for id_b in members[i + 1:]:
                    pair = (id_a, id_b)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    self._relate(self.objects[id_a], self.objects[id_b])
            _ = key

    def _relate(self, a: EvidenceObject, b: EvidenceObject) -> None:
        basis = claims_contradict(a, b)
        if basis is not None:
            self.edges[a.id].append(EvidenceRelation(
                kind="contradicts", target=b.id, weight=1.0, basis=basis))
            self.edges[b.id].append(EvidenceRelation(
                kind="contradicts", target=a.id, weight=1.0, basis=basis))
            self.contradictions.append((a.id, b.id, basis))
            return
        similarity = lexical_similarity(a.claim, b.claim)
        if similarity >= _SUPPORT_THRESHOLD:
            self.edges[a.id].append(EvidenceRelation(
                kind="supports", target=b.id, weight=round(similarity, 4), basis="affinity"))
            self.edges[b.id].append(EvidenceRelation(
                kind="supports", target=a.id, weight=round(similarity, 4), basis="affinity"))

    def _entity_cooccurrence(self, ordered: list[EvidenceObject]) -> None:
        for obj in ordered:
            keys = list(dict.fromkeys((*obj.entities, *obj.terms)))
            for i, entity_a in enumerate(keys):
                for entity_b in keys[i + 1:]:
                    if entity_a == entity_b:
                        continue
                    self.entity_edges[entity_a][entity_b] += 1
                    self.entity_edges[entity_b][entity_a] += 1

    # -- navigation ------------------------------------------------------------------

    def neighbors(
        self,
        eo_id: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int = 8,
    ) -> list[EvidenceRelation]:
        """Outgoing edges, optionally filtered by kind — (weight desc, id asc)."""
        wanted = set(kinds) if kinds is not None else None
        relations = [r for r in self.edges.get(eo_id, [])
                     if wanted is None or r.kind in wanted]
        return relations[:limit]  # already sorted on build

    def bucket_objects(self, key: str, *, limit: int = 16) -> list[EvidenceObject]:
        """Objects in an entity/term bucket, id-ordered."""
        return [self.objects[i] for i in self.buckets.get(key, [])[:limit]]

    def entity_path_exists(self, entity_a: str, entity_b: str, *, max_depth: int = 4) -> bool:
        """True when the two entities connect through co-occurrence within
        *max_depth* hops — the structural check behind relation-need coverage."""
        start, goal = entity_a.lower(), entity_b.lower()
        if start == goal:
            return start in self.buckets
        if start not in self.entity_edges or goal not in self.entity_edges:
            return False
        frontier = [start]
        visited = {start}
        for _ in range(max_depth):
            next_frontier: list[str] = []
            for node in frontier:
                for neighbor in sorted(self.entity_edges[node]):
                    if neighbor == goal:
                        return True
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            if not next_frontier:
                return False
            frontier = next_frontier
        return False
