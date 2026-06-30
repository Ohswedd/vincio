"""Cross-org / federated analytics — the data plane across a trust boundary.

The single-org capstone :class:`~vincio.data.DataEngagement` threads the whole
analytics plane behind one governed, audited call-path and seals it into a signed,
data-bound :class:`~vincio.data.DataNarrative`. But an analytical question often
spans **more than one organization's** data — total revenue across a partnership,
a benchmark over a cohort of independent operators — and the answer must be
computed **without pooling the raw rows into a shared warehouse**. This module is
that reach: the analytics analogue of federated self-improvement
(:meth:`~vincio.core.app.ContextApp.federated_improvement`) and the data-plane
twin of the cross-org :class:`~vincio.settlement.CrossOrgEngagement`.

A :class:`FederatedDataEngagement` (built by
:meth:`~vincio.core.app.ContextApp.federated_data_engagement`) spans the data
planes of several :class:`FederatedMember`\\ s over the *existing* cross-org
fabric:

* A :class:`FederatedQuery` is the **shape** of one governed metric run everywhere
  — the measures and dimensions, the source columns it touches, the residency
  posture it must respect, and the budget — bound, by its content digest, into a
  negotiated :class:`~vincio.negotiation.Contract` (the digest is the contract's
  hashed ``scope``, so the agreed query shape is tamper-evident).
* The query is **choreographed as a** :class:`~vincio.choreography.Saga`: one
  contract-governed step per member, each running *that org's own governed query
  plane locally* (:meth:`~vincio.core.app.ContextApp.query_metric`) and returning
  **only the typed, aggregated, cell-cited** :class:`~vincio.data.MetricResult`.
  The raw rows stay home — they are never serialized into the dispatch, the
  journal, or the narrative, and the only thing that crosses a trust boundary is a
  group-by aggregate. Because the metric runs through each member's
  :class:`~vincio.data.SemanticLayer`, the grouping attributes are the
  analyst-sanctioned governed *dimensions*, never arbitrary raw identifiers, and
  every member must compute the metric by the **same** layer definitions (an
  org whose layer digest differs is refused — the metric is computed one way
  everywhere or not at all).
* The members' aggregates are **reconciled** into one :class:`FederatedFinding`
  per metric and group. Reconciliation is exact for the **partition-decomposable**
  aggregations — ``SUM`` and ``COUNT`` add, ``MIN`` / ``MAX`` take the extremum
  across orgs — so a federated total is the true total, not an estimate. ``AVG``,
  ``COUNT_DISTINCT``, and ratio measures are *not* exactly decomposable across
  partitions and are refused at construction with guidance (federate the
  decomposable components — e.g. a ``SUM`` and a ``COUNT`` — and combine them).
* The whole engagement seals into a signed, hash-chained, offline-verifiable
  :class:`FederatedNarrative` whose every finding **re-derives from each org's
  content-hashed source**: :meth:`FederatedDataEngagement.verify` re-executes each
  member's :class:`~vincio.data.MetricResult` against that member's catalog and
  re-derives every reconciled value, so a tamper to any org's source — or to the
  reconciliation — is caught from the bytes alone.

Governance crosses the boundary intact. **Residency-aware egress refusal**
(reusing :class:`~vincio.governance.ResidencyPolicy`), the **consent ledger**'s
``ANALYTICS`` purpose, and the **differential-privacy accountant** all apply to a
member's contribution exactly as they would to a local query — a member outside
the query's residency posture, without analytics consent, or over its privacy
budget is **refused and audited** (the round raises, never silently dropping a
non-compliant contribution) — and a round with fewer than the ``min_members``
k-anonymity contributor floor is refused so a single org's aggregate is never
singled out.

Everything here is deterministic, dependency-free, and offline. It composes the
primitives the platform already ships — negotiation, choreography, the governed
metric, the audit chain, and the governance rails — into a federated analytics
engagement, never a hosted query federation, a shared warehouse, or a data
clean-room service.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import DataError, ResidencyViolationError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .semantic import Aggregation, MetricQuery, MetricResult, SemanticLayer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.audit import ChainSigner

__all__ = [
    "FederatedQuery",
    "FederatedMember",
    "FederatedContribution",
    "FederatedFinding",
    "FederatedStage",
    "FederatedSignature",
    "FederatedVerification",
    "FederatedNarrative",
    "FederatedDataEngagement",
]

# The audit action a sealed federated-analytics narrative is recorded under, and
# the per-member governance decisions the coordinator records on its own chain.
ENGAGEMENT_ACTION = "federated_data_engagement"
GOVERNANCE_ACTION = "federated_query_governance"

# How each governed aggregation reconciles across partitions (orgs). Only these
# are exactly decomposable: a global SUM/COUNT is the sum of per-partition
# SUM/COUNTs, and a global MIN/MAX is the extremum of per-partition extrema. AVG
# and COUNT_DISTINCT are deliberately absent — they cannot be computed exactly
# from per-partition results alone (an average needs the per-partition counts; a
# distinct count needs the per-partition sets), so a federated query that asks for
# one is refused with guidance rather than silently approximated.
_COMBINE_OP: dict[Aggregation, str] = {
    Aggregation.SUM: "sum",
    Aggregation.COUNT: "sum",
    Aggregation.MIN: "min",
    Aggregation.MAX: "max",
}


# --------------------------------------------------------------------------- #
# Artifact projection helpers (the integrity anchors of a stage)              #
# --------------------------------------------------------------------------- #


def _artifact_wire(artifact: Any) -> Any:
    """A deterministic, JSON-safe projection of any captured artifact.

    Prefers the artifact's own ``to_wire`` (when a primitive publishes one), then a
    pydantic ``model_dump``, falling back to a JSON-safe coercion. A list projects
    element-wise, so a multi-artifact stage (e.g. the reconciled findings) digests
    faithfully.
    """
    if isinstance(artifact, (list, tuple)):
        return [_artifact_wire(item) for item in artifact]
    to_wire = getattr(artifact, "to_wire", None)
    if callable(to_wire):
        try:
            return to_wire()
        except Exception:  # pragma: no cover - defensive; fall through to dump
            pass
    dump = getattr(artifact, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # pragma: no cover - non-serializable; coerce below
            pass
    return to_jsonable(artifact)


def _artifact_digest(artifact: Any) -> str:
    """A content digest of an artifact's bytes — the integrity anchor of a stage."""
    return stable_hash(_artifact_wire(artifact), length=32)


def _artifact_kind(artifact: Any) -> str:
    """The artifact's type name, for a human-legible, audit-friendly stage label."""
    if isinstance(artifact, list):
        inner = _artifact_kind(artifact[0]) if artifact else "object"
        return f"list[{inner}]"
    return type(artifact).__name__


def _artifact_id(artifact: Any) -> str:
    """The artifact's own id, when it carries one (a list/scalar has none)."""
    if isinstance(artifact, (list, tuple)):
        return ""
    return str(getattr(artifact, "id", "") or "")


def _artifact_hash(artifact: Any) -> str:
    """The artifact's own published commitment, best-effort (a digest binds it regardless)."""
    if isinstance(artifact, (list, tuple)):
        return ""
    for attr in ("content_hash", "result_hash", "layer_hash", "head_hash"):
        value = getattr(artifact, attr, None)
        if value:
            return str(value)
    return ""


# --------------------------------------------------------------------------- #
# The federated query shape                                                    #
# --------------------------------------------------------------------------- #


class FederatedQuery(BaseModel):
    """The shape of one governed metric run across organizations.

    A federated query is *not* the data — it is the **agreement** about what to
    compute: the governed :class:`~vincio.data.MetricQuery` (measures and
    dimensions) to run at every member, the source ``columns_touched`` it reads,
    the ``residency`` posture every contribution must respect, the ``budget``, and
    the ``min_members`` k-anonymity contributor floor. Its :meth:`digest` content-
    binds the shape, and :meth:`scope` produces the string the negotiated
    :class:`~vincio.negotiation.Contract` carries as its (hashed) scope, so the
    agreed query shape is tamper-evident in the signed contract.

    The requested measures must be **partition-decomposable** (``SUM`` / ``COUNT`` /
    ``MIN`` / ``MAX``); :meth:`validate_against` (called when the layer is known)
    refuses an ``AVG``, ``COUNT_DISTINCT``, or ratio measure, which cannot be
    reconciled exactly across orgs without exchanging counts or sets.
    """

    model_config = ConfigDict(frozen=True)

    metric: MetricQuery
    table: str
    columns_touched: tuple[str, ...] = ()
    residency: tuple[str, ...] = ()
    purpose: str = "analytics"
    budget_usd: float = 0.0
    sla_seconds: float = 0.0
    quality_floor: float = 0.0
    min_members: int = 2
    dp_noise_multiplier: float = 1.0

    @classmethod
    def of(
        cls,
        metrics: str | MetricQuery | Sequence[str],
        *,
        table: str,
        by: Sequence[str] | None = None,
        where: Sequence[str] | None = None,
        order_by: str = "",
        descending: bool = False,
        limit: int | None = None,
        columns_touched: Sequence[str] | None = None,
        residency: Sequence[str] | None = None,
        purpose: str = "analytics",
        budget_usd: float = 0.0,
        sla_seconds: float = 0.0,
        quality_floor: float = 0.0,
        min_members: int = 2,
        dp_noise_multiplier: float = 1.0,
    ) -> FederatedQuery:
        """Build a :class:`FederatedQuery` from a metric name, list, or
        :class:`~vincio.data.MetricQuery`, plus the federation posture."""
        if isinstance(metrics, MetricQuery):
            spec = metrics
        else:
            names = [metrics] if isinstance(metrics, str) else list(metrics)
            spec = MetricQuery(
                metrics=names,
                dimensions=list(by or []),
                filters=list(where or []),
                order_by=order_by,
                descending=descending,
                limit=limit,
            )
        return cls(
            metric=spec,
            table=table,
            columns_touched=tuple(columns_touched or ()),
            residency=tuple(residency or ()),
            purpose=purpose,
            budget_usd=budget_usd,
            sla_seconds=sla_seconds,
            quality_floor=quality_floor,
            min_members=min_members,
            dp_noise_multiplier=dp_noise_multiplier,
        )

    def query_facts(self) -> dict[str, Any]:
        """The facts the query digest binds (deliberately order-stable)."""
        return {
            "metrics": list(self.metric.metrics),
            "dimensions": list(self.metric.dimensions),
            "filters": list(self.metric.filters),
            "table": self.table,
            "columns_touched": sorted(self.columns_touched),
            "residency": sorted(self.residency),
            "purpose": self.purpose,
            "budget_usd": self.budget_usd,
            "min_members": self.min_members,
        }

    def digest(self) -> str:
        """A stable content hash of the query shape — bound into the contract scope."""
        return stable_hash(self.query_facts(), length=32)

    def scope(self) -> str:
        """The negotiation scope string — human-legible and digest-bound.

        The metric names and table read at a glance; the trailing digest makes the
        agreed shape tamper-evident once the contract that carries this scope is
        signed (the scope *is* part of the contract's content hash).
        """
        metrics = ",".join(self.metric.metrics)
        return f"federated-metric:{metrics}@{self.table}#{self.digest()}"

    def validate_against(self, layer: SemanticLayer) -> FederatedQuery:
        """Refuse a query whose measures are not exactly federable over *layer*.

        Each requested metric must resolve to a governed measure whose aggregation
        is partition-decomposable (``SUM`` / ``COUNT`` / ``MIN`` / ``MAX``). An
        ``AVG``, ``COUNT_DISTINCT``, or ratio measure is refused with guidance,
        because it cannot be reconciled exactly from per-org aggregates alone.
        """
        by_name = {m.name: m for m in layer.measures}
        for name in self.metric.metrics:
            measure = by_name.get(name)
            if measure is None:
                raise DataError(
                    f"federated metric {name!r} is not a governed measure of layer "
                    f"{layer.name or layer.table!r}",
                    details={"metric": name, "table": self.table},
                )
            if measure.is_ratio:
                raise DataError(
                    f"federated metric {name!r} is a ratio measure, which cannot be "
                    "reconciled exactly across organizations; federate its decomposable "
                    "components (e.g. the numerator SUM and the denominator COUNT) and "
                    "combine the reconciled values",
                    details={"metric": name, "reason": "ratio_not_decomposable"},
                )
            agg = measure.agg
            if agg is None or agg not in _COMBINE_OP:
                shown = agg.value if agg is not None else "a non-aggregate"
                raise DataError(
                    f"federated metric {name!r} aggregates with {shown!r}, which "
                    "is not partition-decomposable; only sum / count / min / max federate "
                    "exactly (an average federates as a SUM divided by a COUNT)",
                    details={"metric": name, "agg": shown},
                )
        return self

    def combine_op(self, metric: str, layer: SemanticLayer) -> str:
        """The reconciliation operator (``sum`` / ``min`` / ``max``) for *metric*."""
        for measure in layer.measures:
            if measure.name == metric and measure.agg is not None:
                return _COMBINE_OP[measure.agg]
        raise DataError(f"metric {metric!r} is not a decomposable measure of the layer")


# --------------------------------------------------------------------------- #
# A participating organization                                                 #
# --------------------------------------------------------------------------- #


class FederatedMember:
    """One organization participating in a federated analytics engagement.

    A member binds an org id to its *own* governed :class:`~vincio.core.app.ContextApp`
    — its catalog, its :class:`~vincio.data.SemanticLayer`, its consent ledger and
    privacy accountant, its audit chain — and the ``region`` its data resides in
    (matched against a :class:`FederatedQuery`'s residency posture) and the data
    ``subject`` its contribution is governed under (the consent / privacy key,
    defaulting to the org id). The member's raw dataset never leaves its app; only
    the aggregated :class:`~vincio.data.MetricResult` its query plane produces
    crosses the boundary.
    """

    def __init__(
        self,
        org: str,
        app: Any,
        *,
        table: str | None = None,
        layer: Any | None = None,
        region: str = "",
        subject: str = "",
    ) -> None:
        if not org:
            raise DataError("a federated member needs a non-empty org id")
        self.org = org
        self.app = app
        self.table = table
        self._layer = layer
        self.region = region
        self.subject = subject or org

    def resolve_layer(self) -> SemanticLayer:
        """The member's governed semantic layer (explicit, else resolved by table)."""
        if self._layer is not None:
            return cast(SemanticLayer, self._layer)
        resolver = getattr(self.app, "_resolve_layer", None)
        if callable(resolver):
            return cast(SemanticLayer, resolver(None, self.table))
        raise DataError(f"member {self.org!r} has no resolvable semantic layer")

    def catalog(self) -> Any:
        """The member's live data catalog (the content-hashed source of its answers)."""
        return self.app.data_catalog()

    def source_hash(self, table: str) -> str:
        """The content hash of the member's source table, bound into a finding."""
        try:
            return str(self.catalog().content_hashes().get(table, ""))
        except Exception:  # pragma: no cover - defensive
            return ""


# --------------------------------------------------------------------------- #
# The reconciled cross-org finding                                             #
# --------------------------------------------------------------------------- #


class FederatedContribution(BaseModel):
    """One organization's aggregated, source-bound contribution to a finding."""

    org: str
    value: Any = None
    region: str = ""
    source_hash: str = ""
    result_hash: str = ""


class FederatedFinding(BaseModel):
    """A reconciled cross-org answer for one metric and one dimension group.

    The federated ``value`` is the exact combination of each member's aggregate by
    the metric's partition-decomposable rule (``op``: ``sum`` / ``min`` / ``max``).
    Each :class:`FederatedContribution` records the org, its aggregate, and the
    content hash of the source the aggregate rests on, so a finding re-derives from
    every org's bytes — and :meth:`recompute` re-combines the contributions, the
    check :meth:`FederatedDataEngagement.verify` runs against the live members.
    """

    metric: str
    op: str
    group: dict[str, Any] = Field(default_factory=dict)
    value: Any = None
    members: list[str] = Field(default_factory=list)
    contributions: list[FederatedContribution] = Field(default_factory=list)

    def recompute(self) -> Any:
        """Re-combine the recorded contributions by :attr:`op` (the reconciliation)."""
        values = [c.value for c in self.contributions if c.value is not None]
        return _combine(self.op, values)

    @property
    def group_label(self) -> str:
        """A compact ``k=v`` label for the group (``"*"`` for the ungrouped total)."""
        if not self.group:
            return "*"
        return ", ".join(f"{k}={v}" for k, v in self.group.items())

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        wire: dict[str, Any] = to_jsonable(self.model_dump(mode="json"))
        return wire


def _combine(op: str, values: Sequence[Any]) -> Any:
    """Combine partition aggregates into a global aggregate by the decomposable rule."""
    numeric = [v for v in values if v is not None]
    if not numeric:
        return None
    if op == "sum":
        return sum(numeric)
    if op == "min":
        return min(numeric)
    if op == "max":
        return max(numeric)
    raise DataError(f"unknown reconciliation operator {op!r}")


# --------------------------------------------------------------------------- #
# The hash-chained narrative                                                   #
# --------------------------------------------------------------------------- #


class FederatedStage(BaseModel):
    """One step of a federated engagement, bound into the narrative's hash chain.

    Each stage records the lifecycle ``stage`` verb (``negotiate``,
    ``choreograph``, ``query``, ``reconcile``), an optional ``member`` (the org a
    per-org query stage ran at), the captured artifact's ``kind`` / ``artifact_id``
    / ``artifact_hash`` (its own published commitment), a deterministic ``digest``
    of its bytes (the integrity anchor), and a compact ``summary``. ``prev_hash``
    links it to the preceding stage and ``entry_hash`` binds all of the above, so
    the stages form a tamper-evident chain.
    """

    index: int = 0
    stage: str
    member: str = ""
    kind: str = ""
    artifact_id: str = ""
    artifact_hash: str = ""
    digest: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=utcnow)
    prev_hash: str = ""
    entry_hash: str = ""

    def link_facts(self) -> dict[str, Any]:
        """The fields the link hash binds (deliberately excludes the timestamp)."""
        return {
            "index": self.index,
            "stage": self.stage,
            "member": self.member,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "digest": self.digest,
            "summary": to_jsonable(self.summary),
            "prev_hash": self.prev_hash,
        }

    def compute_entry_hash(self) -> str:
        """Recompute this stage's chain link from its current fields."""
        return stable_hash(self.link_facts(), length=32)

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        wire: dict[str, Any] = to_jsonable(self.model_dump(mode="json"))
        return wire

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> FederatedStage:
        return cls.model_validate(data)


class FederatedSignature(BaseModel):
    """One party's signature over a federated narrative's content hash."""

    party: str
    signature: str
    key_id: str = ""


class FederatedVerification(BaseModel):
    """The (non-raising) outcome of verifying a federated engagement offline.

    ``intact`` is whether the stage chain links cleanly, ``head_ok`` whether the
    recorded head matches the chain, ``hash_ok`` whether the content hash
    recomputes, ``digests_ok`` whether the live artifacts (when supplied) still
    match the bound digests, and ``signatures_ok`` whether the required signatures
    verify. ``data_bound`` is whether every member's aggregate re-executed against
    its content-hashed source *and* every reconciled finding re-derived from those
    aggregates (``None`` when no members were available to re-check). ``valid`` is
    the conjunction. ``broken_at`` pinpoints the first stage that fails to chain.
    """

    valid: bool
    intact: bool
    head_ok: bool
    hash_ok: bool
    digests_ok: bool
    signatures_ok: bool
    data_bound: bool | None = None
    signed_by: list[str] = Field(default_factory=list)
    stages: int = 0
    broken_at: int | None = None
    reason: str | None = None


class FederatedNarrative(BaseModel):
    """A signed, content-bound, hash-chained narrative of a federated engagement.

    The capstone artifact: the ordered chain of :class:`FederatedStage`\\ s a
    :class:`FederatedDataEngagement` produced as it threaded the cross-org analytics
    pipeline — negotiate → choreograph → per-org governed query → reconcile —
    sealed into a single content hash the coordinator signs. It is offline-verifiable
    the way a :class:`~vincio.data.DataNarrative` is — :meth:`verify` recomputes the
    whole chain from the bytes alone, so a re-ordered or edited stage, a broken
    link, a tampered head, or a forged signature is caught; pass the live artifacts
    and a tamper to any *underlying* artifact (a member aggregate, the reconciled
    findings) is caught too. The reconciled :attr:`findings` travel on the narrative
    so the federated answer is carried, cited, and re-derivable as data.
    """

    id: str = Field(default_factory=lambda: new_id("federated"))
    coordinator: str
    table: str = ""
    metrics: list[str] = Field(default_factory=list)
    member_ids: list[str] = Field(default_factory=list)
    stages: list[FederatedStage] = Field(default_factory=list)
    findings: list[FederatedFinding] = Field(default_factory=list)
    head_hash: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    sealed_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[FederatedSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- sealing & hashing ----------------------------------------------------

    def narrative_facts(self) -> dict[str, Any]:
        """The facts the content hash binds: the coordinator, table, members, chain."""
        return {
            "coordinator": self.coordinator,
            "table": self.table,
            "metrics": list(self.metrics),
            "member_ids": list(self.member_ids),
            "stage_count": len(self.stages),
            "entries": [s.entry_hash for s in self.stages],
            "head_hash": self.head_hash,
        }

    def compute_hash(self) -> str:
        """Recompute the narrative's content hash from the current chain."""
        return stable_hash(self.narrative_facts(), length=32)

    def seal(self) -> FederatedNarrative:
        """Re-link every stage in order and stamp the head and content hash (idempotent)."""
        prev = ""
        for i, stage in enumerate(self.stages):
            stage.index = i
            stage.prev_hash = prev
            stage.entry_hash = stage.compute_entry_hash()
            prev = stage.entry_hash
        self.head_hash = prev
        self.content_hash = self.compute_hash()
        return self

    # -- signing & verification -----------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> FederatedNarrative:
        """Add ``party``'s signature over the content hash (sealing first).

        Re-signing for the same party replaces its prior signature, so a narrative
        cannot accumulate stale signatures for one identity.
        """
        if not self.content_hash:
            self.seal()
        sig = FederatedSignature(
            party=party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != party]
        self.signatures.append(sig)
        return self

    def verify(
        self,
        verifier: ChainSigner | None = None,
        *,
        require: list[str] | None = None,
        artifacts: list[Any] | None = None,
    ) -> FederatedVerification:
        """Verify the narrative offline: the chain links, the hash recomputes, signatures check.

        Walks the stage chain recomputing each link, confirms the head and content
        hash, and (when ``verifier`` is supplied) checks each signature — ``require``
        names the parties that must have a verified signature (defaults to the
        coordinator; pass ``[]`` to check the binding alone). Pass the live
        ``artifacts`` the engagement captured (aligned to the stages) to additionally
        re-digest each and confirm it still matches its bound digest, so a tamper to
        any underlying artifact is caught from the bytes alone. (Data-binding —
        re-executing each member's aggregate against the live source — is layered on
        by :meth:`FederatedDataEngagement.verify`, which holds the members.)
        """
        prev = ""
        intact = True
        broken_at: int | None = None
        for i, stage in enumerate(self.stages):
            if (
                stage.index != i
                or stage.prev_hash != prev
                or stage.entry_hash != stage.compute_entry_hash()
            ):
                intact = False
                broken_at = i
                break
            prev = stage.entry_hash
        head_ok = intact and self.head_hash == prev
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()

        digests_ok = True
        if artifacts is not None:
            if len(artifacts) != len(self.stages):
                digests_ok = False
                if broken_at is None:
                    broken_at = min(len(artifacts), len(self.stages))
            else:
                for i, (stage, artifact) in enumerate(zip(self.stages, artifacts, strict=True)):
                    if _artifact_digest(artifact) != stage.digest:
                        digests_ok = False
                        broken_at = i if broken_at is None else broken_at
                        break

        required = [self.coordinator] if require is None else require
        verified: list[str] = []
        signatures_ok = True
        for sig in self.signatures:
            if verifier is not None:
                if verifier.verify(self.content_hash, sig.signature):
                    verified.append(sig.party)
                else:
                    signatures_ok = False
            else:
                verified.append(sig.party)
        missing = [p for p in required if p not in verified]
        if missing:
            signatures_ok = False

        valid = (
            intact
            and head_ok
            and hash_ok
            and digests_ok
            and signatures_ok
            and (verifier is not None or not required)
        )
        reason: str | None = None
        if not intact:
            reason = f"chain broken at stage {broken_at}"
        elif not head_ok:
            reason = "head hash mismatch"
        elif not hash_ok:
            reason = "content hash mismatch"
        elif not digests_ok:
            reason = f"artifact digest mismatch at stage {broken_at}"
        elif not signatures_ok:
            reason = (
                f"missing/invalid signatures for {missing}" if missing else "signature mismatch"
            )
        elif verifier is None and required:
            reason = "no verifier supplied — signatures present but not authenticated"
        return FederatedVerification(
            valid=valid,
            intact=intact,
            head_ok=head_ok,
            hash_ok=hash_ok,
            digests_ok=digests_ok,
            signatures_ok=signatures_ok,
            signed_by=verified,
            stages=len(self.stages),
            broken_at=broken_at,
            reason=reason,
        )

    def require_valid(
        self,
        verifier: ChainSigner,
        *,
        require: list[str] | None = None,
        artifacts: list[Any] | None = None,
    ) -> FederatedNarrative:
        """Verify and raise :class:`~vincio.core.errors.DataError` if not valid."""
        result = self.verify(verifier, require=require, artifacts=artifacts)
        if not result.valid:
            raise DataError(
                f"federated engagement {self.id} failed verification: {result.reason}",
                details={"engagement_id": self.id, "reason": result.reason},
            )
        return self

    # -- views ----------------------------------------------------------------

    @property
    def stage_names(self) -> list[str]:
        """The lifecycle verbs threaded, in order."""
        return [s.stage for s in self.stages]

    def stage(self, name: str) -> FederatedStage | None:
        """The first recorded stage with the given verb, if any."""
        for stage in self.stages:
            if stage.stage == name:
                return stage
        return None

    def finding(self, metric: str, **group: Any) -> FederatedFinding | None:
        """The reconciled finding for *metric* and an optional dimension group."""
        for f in self.findings:
            if f.metric == metric and (not group or f.group == group):
                return f
        return None

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the engagement for the audit chain."""
        details: dict[str, Any] = to_jsonable(
            {
                "engagement_id": self.id,
                "coordinator": self.coordinator,
                "table": self.table,
                "metrics": list(self.metrics),
                "members": list(self.member_ids),
                "stages": self.stage_names,
                "stage_count": len(self.stages),
                "findings": len(self.findings),
                "head_hash": self.head_hash,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )
        return details

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        wire: dict[str, Any] = to_jsonable(self.model_dump(mode="json"))
        return wire

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> FederatedNarrative:
        return cls.model_validate(data)

    def print_summary(self) -> None:
        """Print a one-line-per-stage view, then the reconciled findings."""
        print(
            f"Federated engagement {self.id} — {self.table or '?'}"
            f" · {','.join(self.metrics) or 'metrics'} across {len(self.member_ids)} orgs"
        )
        for stage in self.stages:
            label = f"{stage.stage}@{stage.member}" if stage.member else stage.stage
            facts = ", ".join(f"{k}={v}" for k, v in stage.summary.items())
            print(f"  {stage.index:>2}. {label:<22} {stage.kind:<18} {facts}")
        for f in self.findings:
            print(f"     · {f.metric}[{f.group_label}] = {f.value} ({f.op} of {f.members})")
        print(f"  head={self.head_hash} signed_by={self.signed_by}")


# --------------------------------------------------------------------------- #
# The federated engagement facade                                             #
# --------------------------------------------------------------------------- #


class FederatedDataEngagement:
    """A governed, compositional facade for analytics across organizations.

    Build one with :meth:`~vincio.core.app.ContextApp.federated_data_engagement`,
    add each participating org with :meth:`add_member`, then call the lifecycle
    verbs in order — :meth:`negotiate` (bind the :class:`FederatedQuery` into a
    signed :class:`~vincio.negotiation.Contract`), :meth:`dispatch` (choreograph the
    contract-governed saga so each org runs the metric *locally* and returns only
    its aggregated :class:`~vincio.data.MetricResult`), and :meth:`reconcile`
    (combine the aggregates into :class:`FederatedFinding`\\ s) — or call
    :meth:`run` to thread all three at once.

    Every governance rail a local query passes applies at the boundary:
    :meth:`dispatch` refuses a member outside the query's residency posture
    (:class:`~vincio.governance.ResidencyPolicy`), without ``ANALYTICS`` consent, or
    over its differential-privacy budget, and refuses a round below the
    ``min_members`` k-anonymity floor — each decision audited on the coordinator's
    chain, the member's own query plane audited on its chain. **The raw rows never
    cross**: only the aggregated metric outputs are dispatched, journaled, and
    sealed.

    Call :meth:`seal` to mint the content-bound, signed :class:`FederatedNarrative`,
    and :meth:`verify` to prove the whole chain — every member's aggregate re-executes
    against its content-hashed source and every reconciled value re-derives — offline.
    """

    def __init__(
        self,
        app: Any,
        *,
        query: FederatedQuery | None = None,
        coordinator: str | None = None,
        layer: Any | None = None,
    ) -> None:
        self.app = app
        self.coordinator: str = str(coordinator or getattr(app, "name", "coordinator"))
        self.query = query
        self._layer = layer
        self._started_at = utcnow()
        self._members: list[FederatedMember] = []
        self._stages: list[FederatedStage] = []
        self._artifacts: list[Any] = []
        self._binders: list[Callable[[Mapping[str, Any] | None], bool] | None] = []
        self.narrative: FederatedNarrative | None = None

        # Captured artifacts, by lifecycle stage (None / empty until that stage runs).
        self.contract: Any = None
        self.negotiation: Any = None
        self.delivery: Any = None
        self.member_results: list[tuple[FederatedMember, MetricResult]] = []
        self.findings: list[FederatedFinding] = []

    # -- members --------------------------------------------------------------

    def add_member(
        self,
        org: str,
        app: Any,
        *,
        table: str | None = None,
        layer: Any | None = None,
        region: str = "",
        subject: str = "",
    ) -> FederatedMember:
        """Register a participating organization and return its :class:`FederatedMember`.

        ``table`` defaults to the engagement's query table; ``region`` is matched
        against the query's residency posture and ``subject`` keys the org's consent
        and privacy budget (defaulting to the org id).
        """
        member = FederatedMember(
            org,
            app,
            table=table if table is not None else (self.query.table if self.query else None),
            layer=layer,
            region=region,
            subject=subject,
        )
        self._members.append(member)
        return member

    @property
    def members(self) -> list[FederatedMember]:
        """The registered members, in registration order."""
        return list(self._members)

    # -- stage recording ------------------------------------------------------

    def _record(
        self,
        stage: str,
        artifact: Any,
        *,
        member: str = "",
        artifact_hash: str | None = None,
        binder: Callable[[Mapping[str, Any] | None], bool] | None = None,
        **summary: Any,
    ) -> None:
        """Append a stage for ``artifact`` and invalidate any cached narrative."""
        self._artifacts.append(artifact)
        self._binders.append(binder)
        self._stages.append(
            FederatedStage(
                index=len(self._stages),
                stage=stage,
                member=member,
                kind=_artifact_kind(artifact),
                artifact_id=_artifact_id(artifact),
                artifact_hash=artifact_hash
                if artifact_hash is not None
                else _artifact_hash(artifact),
                digest=_artifact_digest(artifact),
                summary=to_jsonable({k: v for k, v in summary.items() if v is not None}),
            )
        )
        self.narrative = None

    @property
    def stages(self) -> list[FederatedStage]:
        """The stages recorded so far, in order (a live view, copied)."""
        return [s.model_copy(deep=True) for s in self._stages]

    def _resolve_query(self, query: FederatedQuery | None) -> FederatedQuery:
        query = query if query is not None else self.query
        if query is None:
            raise DataError(
                "this engagement has no federated query; pass query= to "
                "federated_data_engagement(...) / negotiate(...) / run(...)"
            )
        self.query = query
        return query

    def _reference_layer(self, query: FederatedQuery) -> SemanticLayer:
        """The governed layer the metric is defined by — explicit, else a member's.

        All members must compute the metric by the *same* definitions; this is the
        reference every member's layer digest is checked against.
        """
        if self._layer is not None:
            return cast(SemanticLayer, self._layer)
        resolver = getattr(self.app, "_resolve_layer", None)
        if callable(resolver):
            try:
                return cast(SemanticLayer, resolver(None, query.table))
            except Exception:
                pass
        if self._members:
            return self._members[0].resolve_layer()
        raise DataError(
            "no semantic layer available; register the layer on the coordinator app, "
            "pass layer= to federated_data_engagement(...), or add a member first"
        )

    # -- negotiate ------------------------------------------------------------

    def negotiate(
        self,
        *,
        query: FederatedQuery | None = None,
        buyer: Any = None,
        seller: Any = None,
        require_agreement: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Bind the federated query into a signed contract and record the opening stage.

        Negotiates a :class:`~vincio.negotiation.Contract` whose hashed ``scope`` is
        the query's :meth:`~FederatedQuery.scope` (so the agreed query shape is
        tamper-evident), defaulting the buyer/seller positions from the query's
        budget when not supplied. With ``require_agreement`` (the default) a no-deal
        raises :class:`~vincio.core.errors.DataError`; pass ``False`` to record the
        no-deal and return the result. Returns the :class:`~vincio.negotiation.Contract`.
        """
        from ..negotiation import buyer_position, seller_position

        query = self._resolve_query(query)
        reference = self._reference_layer(query)
        query.validate_against(reference)

        budget = query.budget_usd or 1.0
        sla = query.sla_seconds or 60.0
        if buyer is None:
            buyer = buyer_position(
                max_price_usd=budget,
                ideal_price_usd=0.0,
                max_sla_seconds=sla,
                min_quality=query.quality_floor,
            )
        if seller is None:
            seller = seller_position(
                min_price_usd=0.0,
                ideal_price_usd=budget,
                min_sla_seconds=0.0,
                max_quality=1.0,
            )
        result = self.app.negotiate(
            query.scope(),
            buyer=buyer,
            seller=seller,
            buyer_id=self.coordinator,
            seller_id="federation",
            **kwargs,
        )
        self.negotiation = result
        contract = getattr(result, "contract", None)
        status = getattr(result, "status", "unknown")
        if contract is None:
            if require_agreement:
                raise DataError(
                    f"federated query negotiation reached no agreement (status={status!r})",
                    details={"scope": query.scope(), "status": status},
                )
            self._record("negotiate", result, status=status)
            return result
        self.contract = contract
        self._record(
            "negotiate",
            contract,
            status=status,
            scope=contract.terms.scope,
            metrics=list(query.metric.metrics),
            price_usd=contract.terms.price_usd,
        )
        return contract

    # -- dispatch (choreograph the governed saga) -----------------------------

    def dispatch(
        self,
        *,
        query: FederatedQuery | None = None,
        contract: Any | None = None,
    ) -> list[tuple[FederatedMember, MetricResult]]:
        """Choreograph the contract-governed saga across members, governing the boundary.

        For each member, **before** its aggregate may cross, the coordinator applies
        the query's governance exactly as a local query would: residency egress
        refusal, the consent ledger's analytics purpose, and the differential-privacy
        budget — a non-compliant member raises (audited and refused). The approved
        members must clear the ``min_members`` k-anonymity floor; then a
        :class:`~vincio.choreography.Saga` (one contract-governed step per member)
        runs each org's :meth:`~vincio.core.app.ContextApp.query_metric` **locally**
        and returns only the aggregated :class:`~vincio.data.MetricResult`. Returns
        the per-member ``(member, result)`` pairs and records the ``choreograph``
        stage plus one ``query`` stage per member.
        """
        from ..choreography import Saga, StepOutcome

        query = self._resolve_query(query)
        contract = contract if contract is not None else self.contract
        reference = self._reference_layer(query)
        query.validate_against(reference)
        reference_digest = reference.digest()

        if not self._members:
            raise DataError(
                "a federated engagement needs at least one member; call add_member(...)"
            )

        approved = self._govern_members(query)
        if len(approved) < query.min_members:
            self._audit_governance(
                "deny",
                reason="contributor_floor",
                approved=[m.org for m in approved],
                min_members=query.min_members,
            )
            raise DataError(
                f"federated round has {len(approved)} eligible member(s) but the "
                f"k-anonymity contributor floor is {query.min_members}; a round below "
                "the floor is refused so no single org's aggregate is singled out",
                details={"approved": len(approved), "min_members": query.min_members},
            )

        # Build a contract-governed saga: one step per approved member. The handler
        # runs that member's OWN governed query plane locally and returns only the
        # aggregated MetricResult (its wire form) — the raw rows never leave the org.
        saga = Saga(name=f"federated-query:{query.table}")
        participants: dict[str, Any] = {}
        for member in approved:
            step = f"query@{member.org}"
            saga = saga.step(step, participant=member.org, action="run_metric", contract=contract)
            participants[member.org] = {
                "run_metric": self._member_handler(member, query, StepOutcome)
            }

        delivery = self.app.choreograph(
            saga, participants=participants, input={"scope": query.scope()}
        )
        self.delivery = delivery
        journal = getattr(delivery, "journal", None)
        self._record(
            "choreograph",
            delivery,
            artifact_hash=getattr(journal, "head_hash", "") if journal is not None else "",
            status=getattr(delivery, "status", None),
            members=[m.org for m in approved],
        )
        if getattr(delivery, "status", None) != "completed":
            raise DataError(
                f"federated dispatch did not complete (status={getattr(delivery, 'status', None)!r})",
                details={"failed_step": getattr(delivery, "failed_step", None)},
            )

        results: list[tuple[FederatedMember, MetricResult]] = []
        for member in approved:
            output = delivery.output_of(f"query@{member.org}")
            wire = output.get("metric")
            if not wire:
                raise DataError(f"member {member.org!r} returned no metric result")
            mr = MetricResult.model_validate(wire)
            # The member's metric must be computed by the SAME governed layer
            # definitions as everyone else's — one metric, one meaning, federation-wide.
            if mr.layer_hash != reference_digest:
                self._audit_governance("deny", member=member.org, reason="layer_mismatch")
                raise DataError(
                    f"member {member.org!r} computed the metric by a different semantic "
                    "layer (layer digest mismatch); a federated metric must be defined "
                    "identically at every org",
                    details={"member": member.org, "reason": "layer_mismatch"},
                )
            results.append((member, mr))
            self._record(
                "query",
                mr,
                member=member.org,
                binder=self._member_binder(member),
                metrics=list(mr.metrics),
                rows=mr.row_count,
                region=member.region or None,
                source_hash=output.get("source_hash", ""),
            )
        self.member_results = results
        return results

    def _member_handler(
        self, member: FederatedMember, query: FederatedQuery, step_outcome_cls: type
    ) -> Callable[[dict[str, Any]], Any]:
        """A saga handler that runs the member's governed metric locally.

        Returns only the aggregated :class:`~vincio.data.MetricResult` (its wire
        form) plus the source content hash — never the raw rows. The closure runs in
        the member's app, so the member's own query plane audits the read on its own
        chain.
        """

        def _run(_payload: dict[str, Any]) -> Any:
            mr = member.app.query_metric(query.metric, table=member.table)
            return step_outcome_cls(
                ok=True,
                cost_usd=0.0,
                latency_ms=0.0,
                quality=1.0,
                output={
                    "org": member.org,
                    "region": member.region,
                    "metric": mr.model_dump(mode="json"),
                    "source_hash": member.source_hash(member.table or query.table),
                },
            )

        return _run

    def _member_binder(self, member: FederatedMember) -> Callable[[Mapping[str, Any] | None], bool]:
        """A data-binder that re-executes a member's aggregate against its live source."""

        def _bind(catalogs: Mapping[str, Any] | None) -> bool:
            mr = next((r for m, r in self.member_results if m.org == member.org), None)
            if mr is None:
                return False
            catalog = (catalogs or {}).get(member.org) if catalogs else None
            if catalog is None:
                catalog = member.catalog()
            return bool(mr.verify(member.resolve_layer(), catalog))

        return _bind

    # -- governance at the boundary -------------------------------------------

    def _govern_members(self, query: FederatedQuery) -> list[FederatedMember]:
        """Apply residency / consent / privacy to each member; return the approved set.

        A non-compliant member is refused (raises) and audited — the cross-org
        analogue of a local query's governance. The differential-privacy budget is
        charged here, so a member over budget refuses before its aggregate is run.
        """
        from ..governance.consent import Purpose
        from ..governance.privacy import PrivacyMechanism
        from ..governance.residency import ResidencyPolicy

        approved: list[FederatedMember] = []
        residency = ResidencyPolicy(
            allowed_regions=list(query.residency),
            provider_regions={m.org: m.region for m in self._members if m.region},
            deny_on_unknown=True,
        )
        for member in self._members:
            # Residency-aware egress refusal: a member whose data region is outside
            # the query's posture may not let even an aggregate cross the boundary.
            if residency.enforced:
                violation = residency.check(provider=member.org)
                if violation is not None:
                    self._audit_governance(
                        "deny", member=member.org, reason="residency", region=member.region
                    )
                    raise ResidencyViolationError(
                        f"federated query forbids {member.org!r} (region {member.region or '?'}) "
                        f"to contribute; allowed regions are {sorted(query.residency)}",
                        region=member.region or None,
                        allowed=list(query.residency),
                        details={"member": member.org, "region": member.region},
                    )

            # Consent: the org must permit the ANALYTICS purpose for its subject. Only
            # enforced when the member has configured a ledger (a local query with no
            # ledger is likewise unblocked).
            ledger = getattr(member.app, "consent_ledger", None)
            if ledger is not None:
                decision = ledger.check(member.subject, Purpose.ANALYTICS)
                if not decision.allowed:
                    self._audit_governance("deny", member=member.org, reason="consent")
                    raise DataError(
                        f"member {member.org!r} has no ANALYTICS consent for subject "
                        f"{member.subject!r}; the contribution is refused",
                        details={"member": member.org, "subject": member.subject},
                    )

            # Differential privacy: charge the contribution against the member's
            # budget exactly as a federated training round is; over-budget refuses.
            accountant = getattr(member.app, "privacy_accountant", None)
            if accountant is not None:
                mechanism = PrivacyMechanism(
                    label="federated_query",
                    noise_multiplier=query.dp_noise_multiplier,
                    sample_rate=1.0,
                    steps=1,
                )
                accountant.charge(
                    member.subject,
                    mechanism,
                    operation="federated_query",
                    round_id=query.digest(),
                    details={"coordinator": self.coordinator, "table": query.table},
                )

            self._audit_governance("allow", member=member.org, region=member.region)
            approved.append(member)
        return approved

    def _audit_governance(self, decision: str, **details: Any) -> None:
        """Record a per-member governance decision on the coordinator's audit chain."""
        audit = getattr(self.app, "audit", None)
        if audit is None:
            return
        audit.record(
            GOVERNANCE_ACTION,
            decision=decision,
            resource=str(details.get("member", self.coordinator)),
            details=to_jsonable({k: v for k, v in details.items() if v is not None}),
        )

    # -- reconcile ------------------------------------------------------------

    def reconcile(self, *, query: FederatedQuery | None = None) -> list[FederatedFinding]:
        """Combine the members' aggregates into one finding per metric and group.

        Reconciliation is exact: a ``SUM`` / ``COUNT`` adds across orgs, a ``MIN`` /
        ``MAX`` takes the extremum, group by group, binding each contribution to the
        org's content-hashed source. Records the ``reconcile`` stage and returns the
        :class:`FederatedFinding`\\ s.
        """
        query = self._resolve_query(query)
        if not self.member_results:
            raise DataError("nothing to reconcile; call dispatch(...) first")
        reference = self._reference_layer(query)
        findings = self._reconcile_results(self.member_results, query, reference)
        self.findings = findings
        self._record(
            "reconcile",
            findings,
            binder=self._reconcile_binder(query),
            metrics=list(query.metric.metrics),
            groups=len({f.group_label for f in findings}),
            findings=len(findings),
        )
        return findings

    def _reconcile_results(
        self,
        member_results: list[tuple[FederatedMember, MetricResult]],
        query: FederatedQuery,
        layer: SemanticLayer,
    ) -> list[FederatedFinding]:
        """Pure reconciliation: combine per-org aggregates into findings (no recording)."""
        dims = list(query.metric.dimensions)
        metrics = list(query.metric.metrics)
        # accum[(metric, group_key)] -> list[FederatedContribution]
        accum: dict[tuple[str, tuple[Any, ...]], list[FederatedContribution]] = {}
        order: list[tuple[str, tuple[Any, ...]]] = []
        for member, mr in member_results:
            source_hash = member.source_hash(member.table or query.table)
            columns = mr.columns
            col_index = {c: i for i, c in enumerate(columns)}
            for row in mr.rows:
                group_key = tuple(row[col_index[d]] for d in dims if d in col_index)
                for metric in metrics:
                    if metric not in col_index:
                        continue
                    value = row[col_index[metric]]
                    key = (metric, group_key)
                    if key not in accum:
                        accum[key] = []
                        order.append(key)
                    accum[key].append(
                        FederatedContribution(
                            org=member.org,
                            value=value,
                            region=member.region,
                            source_hash=source_hash,
                            result_hash=mr.result.result_hash,
                        )
                    )
        findings: list[FederatedFinding] = []
        for metric, group_key in order:
            contributions = accum[(metric, group_key)]
            op = query.combine_op(metric, layer)
            group = {d: v for d, v in zip(dims, group_key, strict=False)}
            findings.append(
                FederatedFinding(
                    metric=metric,
                    op=op,
                    group=group,
                    value=_combine(op, [c.value for c in contributions]),
                    members=[c.org for c in contributions],
                    contributions=contributions,
                )
            )
        return findings

    def _reconcile_binder(
        self, query: FederatedQuery
    ) -> Callable[[Mapping[str, Any] | None], bool]:
        """A binder that re-derives the findings from the live members and compares."""

        def _bind(_catalogs: Mapping[str, Any] | None) -> bool:
            reference = self._reference_layer(query)
            fresh = self._reconcile_results(self.member_results, query, reference)
            if len(fresh) != len(self.findings):
                return False
            recorded = {(f.metric, f.group_label): f.value for f in self.findings}
            for f in fresh:
                if recorded.get((f.metric, f.group_label)) != f.value:
                    return False
            return True

        return _bind

    # -- run, record, seal, verify --------------------------------------------

    def run(
        self,
        *,
        query: FederatedQuery | None = None,
        buyer: Any = None,
        seller: Any = None,
    ) -> list[FederatedFinding]:
        """Thread the whole federated lifecycle — negotiate, dispatch, reconcile.

        A convenience over the granular methods; returns the reconciled findings.
        """
        query = self._resolve_query(query)
        self.negotiate(query=query, buyer=buyer, seller=seller)
        self.dispatch(query=query)
        return self.reconcile(query=query)

    def record_stage(
        self,
        stage: str,
        artifact: Any,
        *,
        member: str = "",
        binder: Callable[[Mapping[str, Any] | None], bool] | None = None,
        **summary: Any,
    ) -> Any:
        """Record an arbitrary already-produced artifact as a custom engagement stage."""
        self._record(stage, artifact, member=member, binder=binder, **summary)
        return artifact

    def seal(self, *, sign: bool = True, record_audit: bool = True) -> FederatedNarrative:
        """Mint the content-bound, signed :class:`FederatedNarrative` of the engagement.

        Hash-links every recorded stage, carries the reconciled findings, signs the
        narrative as the coordinator, and — unless ``record_audit`` is off — lands the
        sealed engagement on the app's hash-chained audit log under
        ``federated_data_engagement``. Returns the narrative; re-sealing after more
        stages run produces a fresh one.
        """
        query = self.query
        narrative = FederatedNarrative(
            id=new_id("federated"),
            coordinator=self.coordinator,
            table=query.table if query else "",
            metrics=list(query.metric.metrics) if query else [],
            member_ids=[m.org for m in self._members],
            started_at=self._started_at,
            sealed_at=utcnow(),
            stages=[s.model_copy(deep=True) for s in self._stages],
            findings=[f.model_copy(deep=True) for f in self.findings],
        )
        narrative.seal()
        if sign:
            signer = self._signer()
            if signer is not None:
                narrative.sign(signer, party=self.coordinator)
        if record_audit and getattr(self.app, "audit", None) is not None:
            entry = self.app.audit.record(
                ENGAGEMENT_ACTION,
                resource=narrative.id,
                decision="sealed",
                details=narrative.audit_details(),
            )
            narrative.audit_id = getattr(entry, "id", None)
        self.narrative = narrative
        return narrative

    def verify(
        self,
        verifier: Any | None = None,
        *,
        require: list[str] | None = None,
        catalogs: Mapping[str, Any] | None = None,
    ) -> FederatedVerification:
        """Verify the whole engagement offline — the chain, every digest, *and* every finding.

        Seals the narrative if needed, verifies its hash chain from the bytes alone,
        and re-digests every captured artifact against its bound digest. Then it
        **re-executes** every member's aggregate against its content-hashed source
        (each member's own catalog, or an override in ``catalogs`` keyed by org) and
        re-derives every reconciled finding from those aggregates — surfaced as
        ``data_bound``. Pass the contract ``verifier`` to authenticate the
        coordinator's signature too.
        """
        narrative = self.narrative or self.seal(sign=verifier is not None, record_audit=False)
        base = narrative.verify(verifier, require=require, artifacts=list(self._artifacts))

        data_bound: bool | None = None
        binders = [b for b in self._binders if b is not None]
        if binders:
            data_bound = all(self._safe_bind(b, catalogs) for b in binders)

        valid = base.valid and data_bound is not False
        reason = base.reason
        if data_bound is False and reason is None:
            reason = "a member aggregate or reconciled finding failed to re-derive from its source"
        return base.model_copy(update={"data_bound": data_bound, "valid": valid, "reason": reason})

    @staticmethod
    def _safe_bind(
        binder: Callable[[Mapping[str, Any] | None], bool], catalogs: Mapping[str, Any] | None
    ) -> bool:
        """Run a data-binder, treating any failure to re-execute as not-bound."""
        try:
            return bool(binder(catalogs))
        except Exception:
            return False

    def _signer(self) -> Any:
        """The app's contract signer, when one is resolvable."""
        resolver = getattr(self.app, "_resolve_contract_signer", None)
        if callable(resolver):
            return resolver(None, True)
        return getattr(self.app, "contract_signer", None)
