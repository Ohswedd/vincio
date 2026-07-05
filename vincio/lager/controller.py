"""The lazy retrieval controller — reasoning drives retrieval.

Evidence is acquired incrementally: round 0 is one hybrid seed (so an easy
query costs exactly one round), later rounds expand the typed graph outward
from what is already acquired, prioritized ``depends_on`` > ``contradicts`` >
``supports`` > ``follows`` > entity buckets. After every round the controller
re-judges **structural** need coverage — entity anchoring AND a kind-specific
test (a relation need requires an entity path, a temporal need a dated claim,
an aggregate need a quantity) AND a similarity floor — never bare lexical
overlap, which scores 0.0 on paraphrase and 0.7 on lexically-parallel wrong
answers.

Termination is provable: both coverage denominators are frozen at plan time
(needs from the planner, entity targets from the query/needs only — evidence-
discovered entities never enter the denominator), the acquired set is
append-only, so gain is monotone and bounded. Every loop has exactly five
exits, each recorded on the trace:

``E0`` empty frontier · ``E1`` sufficient · ``E2`` diminishing gain ·
``E3`` token budget · ``E4`` max rounds

An unanswerable query exits honestly: ``sufficient=False`` with the uncovered
required needs named, so downstream abstains instead of guessing.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from ..context.scoring import lexical_similarity, near_duplicate_score
from ..core.diagnostics import note_suppressed
from ..core.tokens import count_tokens
from .extract import is_referring
from .graph import EvidenceGraph
from .index import EvidenceIndex
from .objects import EvidenceObject, EvidencePack
from .planner import InformationNeed, QueryPlanner

__all__ = ["LazyRetriever", "LazyOptions"]

_EDGE_PRIORITY = {"depends_on": 4.0, "contradicts": 3.0, "supports": 2.0, "follows": 1.0}
_CAUSAL_MARKER_RE = re.compile(
    r"\b(?:because|due to|caused?|causing|root cause|stemmed from|resulted?\s+(?:in|from)|"
    r"led to|reason|assigned the .{0,24}cause|attributed to|owing to|triggered)\b",
    re.IGNORECASE,
)


class LazyOptions(BaseModel):
    """Controller knobs — constructor-only, all deterministic."""

    max_rounds: int = 6
    batch_size: int = 4
    per_need_seeds: int = 3  # round 0 may admit len(needs) × this
    max_evidence_tokens: int = 1200
    gain_epsilon: float = 0.01
    patience: int = 2
    similarity_floor: float = 0.4  # coverage tau on lexical overlap (always applied)
    #: When an embedder is configured, a claim that misses the lexical
    #: ``similarity_floor`` still covers if its dense cosine to the need's
    #: ENTITY-NEUTRALIZED topic clears this (higher) floor — the paraphrase-recall
    #: residual. Set above ``similarity_floor`` on purpose: a lexical/hash
    #: embedder rarely reaches it, so only a genuinely semantic embedder lifts a
    #: topic paraphrase over it. This is a TIGHTENING, not a proven guarantee —
    #: on a real (anisotropic) embedder an entity-sharing but topically-wrong
    #: cause can score high, so the exact safe margin is embedder-specific and
    #: must be validated per model (``benchmarks/lager_residuals.py --embedder
    #: <name>``, whose decoy row is the control). The entity-neutralized topic
    #: probe (:meth:`LazyRetriever._topic_text`) widens that margin by denying the
    #: decoy the shared entity. No embedder → never consulted → the coverage
    #: decision is byte-identical to the lexical path.
    dense_rescue_floor: float = 0.55
    #: Opt-in residual-1 tightening: when True *and* a dense signal is present,
    #: an entity-less causal claim that shares a document with a query match must
    #: also clear ``bridge_similarity_floor`` against the need, so a semantically
    #: off-topic causal sentence adrift in a query-matching document no longer
    #: covers. Default False and a strict no-op without an embedder. REQUIRES a
    #: genuinely semantic embedder (``embedder="auto"`` with fastembed present,
    #: NOT the lexical ``"local"`` hash): the flagship same-document bridge shares
    #: no surface words with the need, so its dense cosine is a thin, model-noisy
    #: signal (a hash embedder scores it barely above the floor — near enough to
    #: the ``bridge_similarity_floor`` noise floor that a legitimate bridge can be
    #: wrongly rejected). A bridge-floor rejection is recorded via
    #: ``note_suppressed`` so a mis-calibrated run's over-abstentions are
    #: observable rather than silent. Calibrate the floor against the flagship
    #: control row of ``lager_residuals.py`` for the target embedder before use.
    reject_same_doc_causal_decoys: bool = False
    bridge_similarity_floor: float = 0.25
    duplicate_threshold: float = 0.95


class LazyRetriever:
    """Runs the lazy loop over an :class:`EvidenceIndex` + :class:`EvidenceGraph`."""

    def __init__(
        self,
        index: EvidenceIndex,
        graph: EvidenceGraph,
        *,
        options: LazyOptions | None = None,
        planner: QueryPlanner | None = None,
    ) -> None:
        self.index = index
        self.graph = graph
        self.options = options or LazyOptions()
        self.planner = planner or QueryPlanner()

    # -- coverage --------------------------------------------------------------------

    def _anchored(self, need: InformationNeed, obj: EvidenceObject) -> bool:
        """The claim talks about what the need is about: entity intersection,
        or (for entity-less needs) at least two shared content terms."""
        if need.entities:
            keys = set(obj.entities) | set(obj.terms)
            if any(entity in keys for entity in need.entities):
                return True
            return False
        obj_terms = set(obj.terms) | {t for t in need.terms if t in obj.claim.lower()}
        return len([t for t in need.terms if t in obj.claim.lower()]) >= 2 or len(obj_terms) >= 2

    def _term_anchored(self, need: InformationNeed, obj: EvidenceObject) -> bool:
        """The strict, non-vacuous core of :meth:`_anchored`: a shared entity or
        at least two genuinely shared content terms — never the entity-less
        fallback that admits any object. Used to gate document-coherence coverage
        so 'in the same document as *something* that matches the query' means a
        real match, not a reachable neighbour."""
        if need.entities:
            keys = set(obj.entities) | set(obj.terms)
            return any(entity in keys for entity in need.entities)
        return len([t for t in need.terms if t in obj.claim.lower()]) >= 2

    def _linked_to_anchored(
        self,
        obj: EvidenceObject,
        need: InformationNeed,
        anchored_ids: set[str],
        acquired: list[EvidenceObject],
    ) -> bool:
        """The claim connects (typed edge or shared content) to a DIFFERENT
        acquired claim that IS anchored to the need — the structural bridge
        across a lexical gap: 'the root cause was the gateway' shares no word
        with 'why did the outage happen', but it is graph-linked to the outage
        claim that does, and its own subject ('gateway') is elaborated elsewhere
        in the acquired evidence. The need's own anchor entities are subtracted
        from the shared-content test, so merely re-sharing the query entity is
        not a bridge (that is topical overlap, not a connection)."""
        if not anchored_ids:
            return False
        for relation in obj.relations:
            if relation.target in anchored_ids and relation.target != obj.id:
                return True
        need_entities = set(need.entities)
        keys = (set(obj.entities) | set(obj.terms)) - need_entities
        if not keys:
            return False
        for held in acquired:
            if held.id == obj.id or held.id not in anchored_ids:
                continue
            if keys & ((set(held.entities) | set(held.terms)) - need_entities):
                return True
        return False

    def _topic_text(self, need: InformationNeed) -> str:
        """The need text with its query entities removed — what the answer must
        be ABOUT, stripped of the entity a decoy could merely share. The dense
        rescue compares the claim to THIS, not the raw need: a decoy that shares
        only the query entity ('ACME revenue fell because of tariffs' for a
        why-ACME-outage need) then cannot ride the shared entity over the rescue
        floor on a real, anisotropic embedder — its remaining content (revenue,
        tariffs) is genuinely far from the topic (outage), while a true paraphrase
        ('the ACME downtime was caused by …') stays close because it renames the
        topic noun, not the entity. Falls back to the full text when stripping
        would leave nothing to embed."""
        if not need.entities:
            return need.text
        topic = need.text
        for entity in need.entities:
            topic = re.sub(re.escape(entity), " ", topic, flags=re.IGNORECASE)
        topic = " ".join(topic.split())
        return topic if len(topic) >= 3 else need.text

    def _dense_rescue(self, need: InformationNeed, obj: EvidenceObject) -> bool:
        """The dense coverage rescue: the claim clears ``dense_rescue_floor``
        against the need's entity-neutralized topic. False (no rescue) whenever
        no embedder is configured (``semantic_similarity`` is then ``None``), so
        the embedder-off coverage decision is byte-identical to the lexical path.

        The floor is a deliberate TIGHTENING, not a proven guarantee: it is set
        above ``similarity_floor`` so a lexical/hash embedder cannot reach it, and
        the entity-neutralized topic probe is what keeps the entity-sharing decoy
        below it — but the exact margin between a genuine paraphrase and an
        adjacent wrong cause is embedder-specific. Validate/tune it for a given
        embedder with ``benchmarks/lager_residuals.py --embedder <name>``, whose
        decoy row is the safety control."""
        sim = self.index.semantic_similarity(self._topic_text(need), obj)
        return sim is not None and sim >= self.options.dense_rescue_floor

    def _similarity_ok(self, need: InformationNeed, obj: EvidenceObject) -> bool:
        """The coverage similarity gate: the lexical floor, OR — when an embedder
        is configured — a genuine dense topic match above the higher
        ``dense_rescue_floor``. Byte-identical to the pure-lexical test when no
        embedder is present (``_dense_rescue`` is then always False), so the
        embedder-off path abstains on a paraphrase exactly as before."""
        if lexical_similarity(need.text, obj.claim) >= self.options.similarity_floor:
            return True
        return self._dense_rescue(need, obj)

    def _covers(
        self,
        need: InformationNeed,
        obj: EvidenceObject,
        *,
        anchored_ids: set[str],
        acquired: list[EvidenceObject],
    ) -> bool:
        anchored = self._anchored(need, obj)
        if need.kind == "causal":
            # A why-question is answered only by a claim carrying a causal
            # marker; a decoy that merely mentions the topic never covers it.
            if not _CAUSAL_MARKER_RE.search(obj.claim):
                return False
            # The causal claim may share no words with the query (the bridge
            # case) — a graph link to a DIFFERENT anchored claim carries it.
            if self._linked_to_anchored(obj, need, anchored_ids, acquired):
                return True
            if need.entities:
                # An entity-anchored causal claim must clear the coverage
                # similarity gate: sharing the query entity is topical overlap,
                # not answeredness ('revenue fell because of tariffs' shares the
                # entity with a why-outage need but never answers it). The gate
                # is lexical by default; a dense embedder additionally recalls a
                # topic paraphrase ('the ACME downtime was caused by …' for a
                # why-ACME-outage need) that the lexical floor alone would miss —
                # WITHOUT re-admitting the entity-sharing decoy, whose dense
                # cosine to the need stays below the rescue floor.
                return anchored and self._similarity_ok(need, obj)
            # An entity-less why-need has no entity anchor, so coverage rides on
            # document coherence: the causal claim must share its source document
            # with a claim that GENUINELY matches the query (real term overlap),
            # never merely sit in some graph-reachable document. This preserves
            # the same-document multi-hop bridge while refusing a bare causal
            # sentence adrift in an unrelated document.
            same_document = any(
                held.id != obj.id
                and held.doc_key == obj.doc_key
                and self._term_anchored(need, held)
                for held in acquired
            )
            if not same_document:
                return False
            if self.options.reject_same_doc_causal_decoys:
                # Opt-in residual-1 tightening: with a genuine dense signal, a
                # same-document causal claim that is semantically off-topic for
                # the need no longer covers — closing the case where a decoy
                # cause merely co-habits a query-matching document. Gated on the
                # signal being PRESENT (sim is not None): with no embedder the
                # flagship same-document bridge, which shares no words with the
                # need, must still cover, so the tightening is a strict no-op off
                # the dense path. The rejection is observable — a mis-calibrated
                # floor (e.g. a hash embedder, see the option caveat) over-abstains
                # rather than dropping the bridge silently.
                sim = self.index.semantic_similarity(need.text, obj)
                if sim is not None and sim < self.options.bridge_similarity_floor:
                    note_suppressed("lager.bridge_floor_rejected")
                    return False
            return True
        if not anchored:
            return False
        if need.kind == "temporal" and obj.observed_at is None:
            return False
        if need.kind == "aggregate" and not any(ch.isdigit() for ch in obj.claim):
            return False
        if need.kind == "relation":
            endpoints = need.entities[:2]
            if len(endpoints) == 2 and not self.graph.entity_path_exists(
                endpoints[0], endpoints[1]
            ):
                return False  # never a confident stop on a relation with no path
        return self._similarity_ok(need, obj)

    def _coverage(
        self, needs: list[InformationNeed], acquired: list[EvidenceObject]
    ) -> dict[str, list[str]]:
        coverage: dict[str, list[str]] = {}
        for need in needs:
            anchored_ids = {obj.id for obj in acquired if self._anchored(need, obj)}
            coverage[need.text] = sorted(
                obj.id for obj in acquired
                if self._covers(need, obj, anchored_ids=anchored_ids, acquired=acquired)
            )
        return coverage

    # -- the loop --------------------------------------------------------------------

    def retrieve(self, query: str) -> EvidencePack:
        options = self.options
        needs = self.planner.plan(query)  # denominator frozen here
        entity_targets = sorted({e for need in needs for e in need.entities})
        acquired: list[EvidenceObject] = []
        acquired_ids: set[str] = set()
        trace: list[dict[str, Any]] = []
        tokens = 0
        exit_reason = "E4:max_rounds"
        previous_score = 0.0
        stalled = 0

        for round_number in range(options.max_rounds):
            frontier = self._frontier(query, needs, acquired, acquired_ids, round_number)
            if not frontier:
                exit_reason = "E0:empty_frontier"
                self._trace(trace, round_number, [], 0.0, needs, acquired, len(frontier))
                break
            coverage = self._coverage(needs, acquired)
            batch_cap = (
                max(len(needs) * options.per_need_seeds, options.batch_size)
                if round_number == 0 else options.batch_size
            )
            batch = self._pick(frontier, needs, coverage, acquired, batch_cap)
            round_start = len(acquired)
            for obj in batch:
                if obj.id in acquired_ids:
                    continue  # already force-added as an antecedent this round
                cost = count_tokens(obj.claim)
                if tokens + cost > options.max_evidence_tokens and acquired:
                    exit_reason = "E3:token_budget"
                    break
                acquired.append(obj)
                acquired_ids.add(obj.id)
                tokens += cost
                tokens += self._force_antecedent(
                    obj, acquired, acquired_ids,
                    tokens=tokens, budget=options.max_evidence_tokens,
                )
            score = self._score(needs, entity_targets, acquired)
            gain = score - previous_score
            previous_score = score
            # Trace only what was ACTUALLY acquired this round (batch objects and
            # any forced antecedents), so an E3 break mid-batch never records an
            # id the pack does not contain — the explainability contract holds
            # exactly on the budget-pressure runs.
            self._trace(trace, round_number, [o.id for o in acquired[round_start:]], gain,
                        needs, acquired, len(frontier))
            if exit_reason == "E3:token_budget":
                break
            coverage = self._coverage(needs, acquired)
            if all(coverage[n.text] for n in needs if n.required):
                exit_reason = "E1:sufficient"
                break
            stalled = stalled + 1 if gain < options.gain_epsilon else 0
            if stalled >= options.patience:
                exit_reason = "E2:diminishing_gain"
                break
        else:
            exit_reason = "E4:max_rounds"

        coverage = self._coverage(needs, acquired)
        uncovered = [n.text for n in needs if n.required and not coverage[n.text]]
        packed = self._collapse(acquired)
        return EvidencePack(
            query=query,
            objects=packed,
            coverage=coverage,
            contradictions=self._pack_contradictions(packed),
            rounds=len(trace),
            gain_trace=trace,
            exit_reason=exit_reason,
            sufficient=not uncovered,
            uncovered_needs=uncovered,
            token_cost=sum(count_tokens(o.claim) for o in packed),
        )

    # -- steps -----------------------------------------------------------------------

    def _frontier(
        self,
        query: str,
        needs: list[InformationNeed],
        acquired: list[EvidenceObject],
        acquired_ids: set[str],
        round_number: int,
    ) -> list[EvidenceObject]:
        if round_number == 0 or not acquired:
            entities = sorted({e for need in needs for e in need.entities})
            seeds = self.index.seed(query, entities=entities, limit=32)
            return [o for o in seeds if o.id not in acquired_ids]
        ranked: dict[str, float] = {}
        for obj in acquired:
            for relation in self.graph.neighbors(obj.id, limit=8):
                if relation.target in acquired_ids:
                    continue
                weight = _EDGE_PRIORITY.get(relation.kind, 0.5) + relation.weight
                ranked[relation.target] = max(ranked.get(relation.target, 0.0), weight)
            # Document coherence: an anchored document's remaining claims are
            # candidates even when they share no words with the query — the
            # root-cause paragraph of a long incident report belongs to the
            # frontier once the report itself is acquired.
            for sibling in self.graph.doc_objects.get(obj.doc_key, []):
                if sibling not in acquired_ids:
                    ranked.setdefault(sibling, 1.0)
            for key in (obj.entities or obj.terms):
                for neighbor in self.graph.bucket_objects(key, limit=8):
                    if neighbor.id not in acquired_ids:
                        ranked.setdefault(neighbor.id, 0.5)
        ordered = sorted(ranked.items(), key=lambda kv: (-kv[1], kv[0]))
        return [self.graph.objects[eo_id] for eo_id, _ in ordered[:32]]

    def _pick(
        self,
        frontier: list[EvidenceObject],
        needs: list[InformationNeed],
        coverage: dict[str, list[str]],
        acquired: list[EvidenceObject],
        cap: int,
    ) -> list[EvidenceObject]:
        """Utility against UNCOVERED needs, novelty against acquired —
        (score desc, id asc)."""
        uncovered = [n for n in needs if not coverage.get(n.text)]
        targets = uncovered or needs
        scored: list[tuple[float, str, EvidenceObject]] = []
        for obj in frontier:
            affinity = max(
                (lexical_similarity(need.text, obj.claim)
                 + (0.5 if self._anchored(need, obj) else 0.0)
                 + self._kind_bonus(need, obj))
                for need in targets
            )
            novelty_penalty = max(
                (lexical_similarity(obj.claim, held.claim) for held in acquired),
                default=0.0,
            )
            utility = affinity + 0.2 * obj.confidence - 0.5 * novelty_penalty
            scored.append((utility, obj.id, obj))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [obj for _, _, obj in scored[:cap]]

    @staticmethod
    def _kind_bonus(need: InformationNeed, obj: EvidenceObject) -> float:
        """Reasoning drives retrieval: a candidate that carries the KIND of
        evidence an uncovered need requires (a causal marker for a why-need, a
        date for a when-need, a quantity for a how-many-need) outranks a
        candidate that merely shares the query's words."""
        if need.kind == "causal" and _CAUSAL_MARKER_RE.search(obj.claim):
            return 0.6
        if need.kind == "temporal" and obj.observed_at is not None:
            return 0.4
        if need.kind == "aggregate" and any(ch.isdigit() for ch in obj.claim):
            return 0.4
        return 0.0

    def _force_antecedent(
        self,
        obj: EvidenceObject,
        acquired: list[EvidenceObject],
        acquired_ids: set[str],
        *,
        tokens: int,
        budget: int,
    ) -> int:
        """A referring claim ("It reports to…") gets its antecedent packed with
        it — self-containment is a pack-level guarantee, not a hope. The
        antecedent is charged against the SAME hard token budget as any acquired
        object: an antecedent that will not fit is skipped rather than appended,
        so an arbitrarily large antecedent (a whole code block or table) can
        never push ``token_cost`` unboundedly past ``max_evidence_tokens``."""
        if obj.kind != "claim" or not is_referring(obj.claim):
            return 0
        for relation in obj.relations:
            if relation.kind == "depends_on" and relation.target not in acquired_ids:
                antecedent = self.graph.objects.get(relation.target)
                if antecedent is not None:
                    cost = count_tokens(antecedent.claim)
                    if tokens + cost > budget:
                        return 0  # will not fit the hard budget; do not overshoot
                    acquired.append(antecedent)
                    acquired_ids.add(antecedent.id)
                    return cost
        return 0

    def _score(
        self,
        needs: list[InformationNeed],
        entity_targets: list[str],
        acquired: list[EvidenceObject],
    ) -> float:
        """Monotone progress score over FROZEN denominators."""
        coverage = self._coverage(needs, acquired)
        need_score = sum(1 for n in needs if coverage[n.text]) / max(len(needs), 1)
        if entity_targets:
            held = {key for obj in acquired for key in (*obj.entities, *obj.terms)}
            entity_score = sum(1 for e in entity_targets if e in held) / len(entity_targets)
        else:
            entity_score = 0.0
        return need_score + 0.5 * entity_score

    def _collapse(self, acquired: list[EvidenceObject]) -> list[EvidenceObject]:
        """Near-duplicates collapse at pack level: keep the highest-authority
        copy, carry the rest as corroborating ids — confidence benefits without
        spending tokens."""
        kept: list[EvidenceObject] = []
        for obj in acquired:
            duplicate_of = None
            for held in kept:
                if near_duplicate_score(obj.claim, held.claim) >= self.options.duplicate_threshold:
                    duplicate_of = held
                    break
            if duplicate_of is None:
                kept.append(obj.model_copy(deep=True))
            elif obj.authority > duplicate_of.authority:
                index = kept.index(duplicate_of)
                replacement = obj.model_copy(deep=True)
                corroborating = [duplicate_of.id,
                                 *duplicate_of.metadata.get("corroborated_by", [])]
                replacement.metadata["corroborated_by"] = corroborating
                kept[index] = replacement
            else:
                corroborating = duplicate_of.metadata.setdefault("corroborated_by", [])
                if obj.id not in corroborating:
                    corroborating.append(obj.id)
        return kept

    def _pack_contradictions(self, packed: list[EvidenceObject]) -> list[tuple[str, str, str]]:
        ids = {obj.id for obj in packed}
        return sorted(
            (a, b, basis) for a, b, basis in self.graph.contradictions
            if a in ids and b in ids
        )

    @staticmethod
    def _trace(
        trace: list[dict[str, Any]],
        round_number: int,
        added: list[str],
        gain: float,
        needs: list[InformationNeed],
        acquired: list[EvidenceObject],
        frontier_size: int,
    ) -> None:
        trace.append({
            "round": round_number,
            "added": added,
            "gain": round(gain, 4),
            "acquired": len(acquired),
            "needs": len(needs),
            "frontier": frontier_size,
        })
