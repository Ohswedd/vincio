"""The cross-org engagement lifecycle facade — the fabric as one system.

Twenty rungs delivered the cross-org *primitives* — negotiation and contracting,
choreographed delivery, metering and settlement, multilateral netting, dispute
arbitration, portable and revocable reputation, reputation-gated admission,
collateral escrow / pooling / rehypothecation guards, proof-of-reserves,
proof-of-solvency, liability completeness / non-equivocation / history
consistency, and insolvency resolution by seniority waterfall with close-out
set-off — each signed, content-bound, offline-verifiable, and reputation- and
audit-closing on its own. What was missing is not a forty-first primitive but the
**whole**: nothing yet presented the fabric as one coherent system or proved it
composes end-to-end. This module is that capstone.

A :class:`CrossOrgEngagement` (built by
:meth:`~vincio.core.app.ContextApp.cross_org_engagement`) threads the entire
pipeline behind one governed, audited call-path — discover → negotiate → contract
→ choreograph delivery → meter → settle → net → arbitrate → attest and port
reputation → admit on earned standing → post and pool collateral under a
rehypothecation guard → prove reserves, solvency, completeness, non-equivocation,
and history → and, on default, resolve the insolvency by seniority waterfall with
close-out set-off. It is **purely compositional**: every lifecycle method delegates
to the *same* :class:`~vincio.core.app.ContextApp` entry point a caller would use
directly (each unchanged and still usable on its own), captures the artifact it
produced, and records it as a stage in a single hash-linked narrative.

The :class:`EngagementNarrative` is that narrative: an ordered chain of
:class:`EngagementStage`\\ s, each binding the stage's verb, the artifact's own
content hash, and a deterministic digest of the artifact's bytes into a link that
chains to the previous one. It is content-bound and offline-verifiable the way a
:class:`~vincio.settlement.SettlementRecord` is — :meth:`EngagementNarrative.verify`
recomputes the whole chain from the bytes alone, so a tamper introduced anywhere
(a re-ordered stage, an edited digest, a forged signature) is caught, and re-digesting
the live artifacts against the bound digests proves a tamper to any *underlying*
artifact is caught too. This is the proof that the fabric is a *system*, not a pile
of primitives: one continuous, signed, audited narrative from the first offer to the
final distribution, every hash linking.

Everything here is dependency-free, deterministic, and offline — never a hosted
marketplace, clearing house, arbitration service, reputation bureau, underwriting
service, escrow custodian, receiver, or payment processor, only a mechanical,
verifiable composition of the primitives the fabric already ships.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "EngagementStage",
    "EngagementSignature",
    "EngagementVerification",
    "EngagementNarrative",
    "CrossOrgEngagement",
]

# The audit action a sealed engagement narrative is recorded under — the single
# key a fabric-wide engagement roll-up reads back from the chain.
ENGAGEMENT_ACTION = "cross_org_engagement"


def _artifact_wire(artifact: Any) -> Any:
    """A deterministic, JSON-safe projection of any captured artifact.

    Prefers the artifact's own ``to_wire`` (the canonical bytes the primitive
    publishes), then a pydantic ``model_dump``, falling back to a JSON-safe
    coercion. A list of artifacts projects element-wise, so a multi-record stage
    (e.g. :meth:`CrossOrgEngagement.settle_saga`) digests faithfully.
    """
    if isinstance(artifact, (list, tuple)):
        return [_artifact_wire(item) for item in artifact]
    to_wire = getattr(artifact, "to_wire", None)
    if callable(to_wire):
        return to_wire()
    dump = getattr(artifact, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
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
    """The artifact's own content hash, best-effort (a digest binds it regardless).

    Reads the artifact's published commitment — a ``content_hash`` (most signed
    artifacts and a :class:`~vincio.negotiation.Contract`), else the ``head_hash``
    of a hash-chained ledger or of a result's durable journal — so the narrative
    binds the *same* hash the primitive signs, not only an opaque digest.
    """
    if isinstance(artifact, (list, tuple)):
        return ""
    for attr in ("content_hash", "head_hash"):
        value = getattr(artifact, attr, None)
        if value:
            return str(value)
    journal = getattr(artifact, "journal", None)
    if journal is not None:
        head = getattr(journal, "head_hash", None)
        if head:
            return str(head)
    return ""


class EngagementStage(BaseModel):
    """One step of a cross-org engagement, bound into the narrative's hash chain.

    Each stage records the lifecycle ``stage`` verb (``negotiate``, ``settle``,
    ``resolve_insolvency`` …), the captured artifact's ``kind`` / ``artifact_id`` /
    ``artifact_hash`` (its own published commitment), a deterministic ``digest`` of
    its bytes (the integrity anchor — a tamper to the artifact changes it), and a
    compact ``summary`` of the economic facts. ``prev_hash`` links it to the
    preceding stage and ``entry_hash`` binds all of the above, so the stages form a
    tamper-evident chain the way a :class:`~vincio.settlement.SettlementBook`'s
    records do.
    """

    index: int = 0
    stage: str
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
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> EngagementStage:
        return cls.model_validate(data)


class EngagementSignature(BaseModel):
    """One party's signature over an engagement narrative's content hash."""

    party: str
    signature: str
    key_id: str = ""


class EngagementVerification(BaseModel):
    """The (non-raising) outcome of verifying an engagement narrative offline.

    ``intact`` is whether the stage chain links cleanly, ``head_ok`` whether the
    recorded head matches the chain, ``hash_ok`` whether the content hash recomputes,
    ``digests_ok`` whether the live artifacts (when supplied) still match the bound
    digests, and ``signatures_ok`` whether the required signatures verify. ``valid``
    is the conjunction. ``broken_at`` pinpoints the first stage that fails to chain.
    """

    valid: bool
    intact: bool
    head_ok: bool
    hash_ok: bool
    digests_ok: bool
    signatures_ok: bool
    signed_by: list[str] = Field(default_factory=list)
    stages: int = 0
    broken_at: int | None = None
    reason: str | None = None


class EngagementNarrative(BaseModel):
    """A signed, content-bound, hash-chained narrative of a whole cross-org engagement.

    The capstone artifact: the ordered chain of :class:`EngagementStage`\\ s a
    :class:`CrossOrgEngagement` produced as it threaded the fabric end-to-end, sealed
    into a single content hash the coordinator signs. It is offline-verifiable the
    way a :class:`~vincio.settlement.SettlementRecord` is — :meth:`verify` recomputes
    the entire chain from the bytes alone, so a re-ordered or edited stage, a broken
    link, a tampered head, or a forged signature is caught; pass the live artifacts to
    :meth:`verify` and a tamper to any *underlying* artifact is caught too. One
    narrative, one continuous proof that every primitive composed.
    """

    id: str = Field(default_factory=lambda: new_id("engagement"))
    coordinator: str
    buyer: str = ""
    seller: str = ""
    scope: str = ""
    stages: list[EngagementStage] = Field(default_factory=list)
    head_hash: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    sealed_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[EngagementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- sealing & hashing ----------------------------------------------------

    def narrative_facts(self) -> dict[str, Any]:
        """The facts the content hash binds: the parties, scope, and chained stages."""
        return {
            "coordinator": self.coordinator,
            "buyer": self.buyer,
            "seller": self.seller,
            "scope": self.scope,
            "stage_count": len(self.stages),
            "entries": [s.entry_hash for s in self.stages],
            "head_hash": self.head_hash,
        }

    def compute_hash(self) -> str:
        """Recompute the narrative's content hash from the current chain."""
        return stable_hash(self.narrative_facts(), length=32)

    def seal(self) -> EngagementNarrative:
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

    def sign(self, signer: ChainSigner, *, party: str) -> EngagementNarrative:
        """Add ``party``'s signature over the content hash (sealing first).

        Re-signing for the same party replaces its prior signature, so a narrative
        cannot accumulate stale signatures for one identity.
        """
        if not self.content_hash:
            self.seal()
        sig = EngagementSignature(
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
    ) -> EngagementVerification:
        """Verify the narrative offline: the chain links, the hash recomputes, signatures check.

        Walks the stage chain recomputing each link, confirms the head and content
        hash, and (when ``verifier`` is supplied) checks each signature — ``require``
        names the parties that must have a verified signature (defaults to the
        coordinator; pass ``[]`` to check the binding alone). Pass the live
        ``artifacts`` the engagement captured (aligned to the stages) to additionally
        re-digest each and confirm it still matches its bound digest, so a tamper to
        any underlying artifact is caught from the bytes alone.
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
            reason = f"missing/invalid signatures for {missing}" if missing else "signature mismatch"
        elif verifier is None and required:
            reason = "no verifier supplied — signatures present but not authenticated"
        return EngagementVerification(
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
    ) -> EngagementNarrative:
        """Verify and raise :class:`SettlementError` if the narrative is not valid."""
        result = self.verify(verifier, require=require, artifacts=artifacts)
        if not result.valid:
            raise SettlementError(
                f"engagement {self.id} failed verification: {result.reason}",
                details={"engagement_id": self.id, "reason": result.reason},
            )
        return self

    # -- views ----------------------------------------------------------------

    @property
    def stage_names(self) -> list[str]:
        """The lifecycle verbs threaded, in order."""
        return [s.stage for s in self.stages]

    def stage(self, name: str) -> EngagementStage | None:
        """The first recorded stage with the given verb, if any."""
        for stage in self.stages:
            if stage.stage == name:
                return stage
        return None

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the engagement for the audit chain."""
        return to_jsonable(
            {
                "engagement_id": self.id,
                "coordinator": self.coordinator,
                "buyer": self.buyer,
                "seller": self.seller,
                "scope": self.scope,
                "stages": self.stage_names,
                "stage_count": len(self.stages),
                "head_hash": self.head_hash,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> EngagementNarrative:
        return cls.model_validate(data)

    def print_summary(self) -> None:
        """Print a one-line-per-stage view of the engagement narrative."""
        print(
            f"Engagement {self.id} — {self.buyer or '?'} ⇄ {self.seller or '?'}"
            f" · {self.scope or 'cross-org'}"
        )
        for stage in self.stages:
            facts = ", ".join(f"{k}={v}" for k, v in stage.summary.items())
            print(f"  {stage.index:>2}. {stage.stage:<18} {stage.kind:<22} {facts}")
        print(f"  head={self.head_hash} signed_by={self.signed_by}")


class CrossOrgEngagement:
    """A purely-compositional facade threading the whole cross-org fabric in one call-path.

    Build one with :meth:`~vincio.core.app.ContextApp.cross_org_engagement` and call
    the lifecycle verbs in the order the engagement runs — :meth:`negotiate`,
    :meth:`choreograph`, :meth:`settle` / :meth:`settle_saga`, :meth:`net`,
    :meth:`arbitrate`, :meth:`attest_reputation` / :meth:`import_reputation`,
    :meth:`admit`, :meth:`post_escrow` / :meth:`post_collateral_pool` /
    :meth:`guard_collateral`, :meth:`attest_custody` / :meth:`attest_liabilities` /
    :meth:`prove_solvency` / :meth:`check_completeness` / :meth:`check_root_consistency`
    / :meth:`check_history_consistency`, and :meth:`resolve_insolvency`. Each delegates
    to the *same* :class:`~vincio.core.app.ContextApp` method a caller would use
    directly — the primitives are unchanged and still usable on their own — captures
    the artifact it produced (exposed as an attribute, e.g. :attr:`contract`,
    :attr:`delivery`, :attr:`netting`, :attr:`insolvency`), returns it, and records it
    as a stage in the engagement's hash-linked narrative.

    Call :meth:`seal` to mint the content-bound, signed :class:`EngagementNarrative`,
    and :meth:`verify` to prove the whole chain — and every captured artifact — verifies
    offline. The facade adds no new economic logic; it composes and *narrates* the
    primitives, so the fabric reads as one system.
    """

    def __init__(
        self,
        app: Any,
        *,
        buyer: str = "",
        seller: str = "",
        scope: str = "",
        coordinator: str | None = None,
    ) -> None:
        self.app = app
        self.coordinator: str = str(coordinator or getattr(app, "name", "coordinator"))
        self.buyer = buyer
        self.seller = seller
        self.scope = scope
        self._started_at = utcnow()
        self._stages: list[EngagementStage] = []
        self._artifacts: list[Any] = []
        self.narrative: EngagementNarrative | None = None

        # Captured artifacts, by lifecycle stage (None until that stage runs).
        self.negotiation: Any = None
        self.contract: Any = None
        self.admission: Any = None
        self.delivery: Any = None
        self.settlements: list[Any] = []
        self.netting: Any = None
        self.arbitration: Any = None
        self.attestation: Any = None
        self.reputation: Any = None
        self.escrow: Any = None
        self.pool: Any = None
        self.ledger: Any = None
        self.reserves: Any = None
        self.liabilities: Any = None
        self.solvency: Any = None
        self.completeness: Any = None
        self.root_consistency: Any = None
        self.history: Any = None
        self.insolvency: Any = None

        # A facade threading the whole pipeline needs a durable book and a
        # reputation ledger to settle, net, and close the reputation loop onto;
        # ensure both without disturbing an already-configured one.
        if getattr(app, "settlement_book", None) is None and hasattr(app, "use_settlement_book"):
            app.use_settlement_book(owner=self.coordinator)
        if getattr(app, "reputation_ledger", None) is None and hasattr(app, "use_reputation_ledger"):
            app.use_reputation_ledger()

    # -- stage recording ------------------------------------------------------

    def _record(self, stage: str, artifact: Any, *, artifact_hash: str | None = None, **summary: Any) -> None:
        """Append a stage for ``artifact`` and invalidate any cached narrative."""
        self._artifacts.append(artifact)
        self._stages.append(
            EngagementStage(
                index=len(self._stages),
                stage=stage,
                kind=_artifact_kind(artifact),
                artifact_id=_artifact_id(artifact),
                artifact_hash=artifact_hash if artifact_hash is not None else _artifact_hash(artifact),
                digest=_artifact_digest(artifact),
                summary=to_jsonable({k: v for k, v in summary.items() if v is not None}),
            )
        )
        self.narrative = None

    @property
    def stages(self) -> list[EngagementStage]:
        """The stages recorded so far, in order (a live view, copied)."""
        return [s.model_copy(deep=True) for s in self._stages]

    # -- discover / negotiate / contract --------------------------------------

    def negotiate(
        self,
        *,
        buyer: Any,
        seller: Any,
        require_agreement: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Negotiate the engagement's contract and record it as the opening stage.

        Delegates to :meth:`~vincio.core.app.ContextApp.negotiate` for this
        engagement's ``scope`` between :attr:`buyer` and :attr:`seller`, stores the
        :class:`~vincio.negotiation.NegotiationResult` on :attr:`negotiation`, and —
        on agreement — captures the minted :class:`~vincio.negotiation.Contract` on
        :attr:`contract` and returns it. With ``require_agreement`` (the default) a
        no-deal raises :class:`SettlementError`; pass ``False`` to record the no-deal
        outcome and return the result instead.
        """
        result = self.app.negotiate(
            self.scope, buyer=buyer, seller=seller, buyer_id=self.buyer, seller_id=self.seller, **kwargs
        )
        self.negotiation = result
        contract = getattr(result, "contract", None)
        status = getattr(result, "status", "unknown")
        if contract is None:
            if require_agreement:
                raise SettlementError(
                    f"engagement negotiation reached no agreement (status={status!r})",
                    details={"scope": self.scope, "status": status},
                )
            self._record("negotiate", result, status=status)
            return result
        self.contract = contract
        self._record(
            "negotiate",
            contract,
            status=status,
            scope=contract.terms.scope,
            price_usd=contract.terms.price_usd,
        )
        return contract

    def admit(self, subject: str | None = None, **kwargs: Any) -> Any:
        """Decide the counterparty's admitted exposure and record the decision.

        Delegates to :meth:`~vincio.core.app.ContextApp.admit` (defaulting
        ``subject`` to the engagement's :attr:`seller`), stores the
        :class:`~vincio.settlement.AdmissionDecision` on :attr:`admission`, and returns
        it.
        """
        subject = subject or self.seller
        decision = self.app.admit(subject, **kwargs)
        self.admission = decision
        self._record(
            "admit",
            decision,
            subject=getattr(decision, "subject", subject),
            max_contract_value_usd=getattr(decision, "max_contract_value_usd", None),
        )
        return decision

    def choreograph(self, saga: Any, *, participants: Any, **kwargs: Any) -> Any:
        """Run the contracted delivery as a durable, compensating cross-org saga.

        Delegates to :meth:`~vincio.core.app.ContextApp.choreograph` (pass
        ``directory=`` to resolve a step's participant by capability at dispatch time —
        the *discover* half of the pipeline), stores the
        :class:`~vincio.choreography.SagaResult` on :attr:`delivery`, and returns it.
        The stage binds the saga journal's head hash, so the engagement narrative
        chains onto the delivery's own tamper-evident journal.
        """
        result = self.app.choreograph(saga, participants=participants, **kwargs)
        self.delivery = result
        journal = getattr(result, "journal", None)
        self._record(
            "choreograph",
            result,
            artifact_hash=getattr(journal, "head_hash", "") if journal is not None else "",
            status=getattr(result, "status", None),
            completed=list(getattr(result, "completed_steps", []) or []),
            discovered=bool(getattr(result, "bindings", {}) or {}),
        )
        return result

    # -- meter / settle -------------------------------------------------------

    def settle(self, contract: Any | None = None, **kwargs: Any) -> Any:
        """Close the books on one contract and record the settlement.

        Delegates to :meth:`~vincio.core.app.ContextApp.settle` (defaulting
        ``contract`` to the engagement's negotiated :attr:`contract`), appends the
        :class:`~vincio.settlement.SettlementRecord` to :attr:`settlements`, and returns
        it.
        """
        contract = contract if contract is not None else self.contract
        if contract is None:
            raise SettlementError("settle() needs a contract; negotiate one first or pass contract=")
        record = self.app.settle(contract, **kwargs)
        self.settlements.append(record)
        self._record(
            "settle", record, status=getattr(record, "status", None), balance_usd=getattr(record, "balance_usd", None)
        )
        return record

    def settle_saga(self, result: Any | None = None, *, contracts: dict[str, Any], **kwargs: Any) -> list[Any]:
        """Close the books on every contract a saga ran under and record the batch.

        Delegates to :meth:`~vincio.core.app.ContextApp.settle_saga` (defaulting
        ``result`` to the engagement's :attr:`delivery`), extends :attr:`settlements`
        with the per-contract :class:`~vincio.settlement.SettlementRecord`\\ s, and
        returns them.
        """
        result = result if result is not None else self.delivery
        if result is None:
            raise SettlementError("settle_saga() needs a saga result; choreograph one first or pass result=")
        records = self.app.settle_saga(result, contracts=contracts, **kwargs)
        self.settlements.extend(records)
        self._record(
            "settle_saga",
            records,
            count=len(records),
            statuses=[getattr(r, "status", None) for r in records],
        )
        return records

    def net(self, **kwargs: Any) -> Any:
        """Net the fleet's settlement books into one minimal cleared set and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.clear_settlements`, stores the
        :class:`~vincio.settlement.NettingSet` on :attr:`netting`, and returns it.
        """
        netting = self.app.clear_settlements(**kwargs)
        self.netting = netting
        self._record(
            "net",
            netting,
            clean=getattr(netting, "clean", None),
            obligations=len(getattr(netting, "obligations", []) or []),
        )
        return netting

    def arbitrate(self, records: list[Any], **kwargs: Any) -> Any:
        """Adjudicate a disputed contract from the records its parties submit and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.arbitrate`, stores the
        :class:`~vincio.settlement.Resolution` on :attr:`arbitration`, and returns it.
        """
        resolution = self.app.arbitrate(records, **kwargs)
        self.arbitration = resolution
        self._record(
            "arbitrate",
            resolution,
            status=getattr(resolution, "status", None),
            contract_id=getattr(resolution, "contract_id", None),
        )
        return resolution

    # -- reputation portability & admission -----------------------------------

    def attest_reputation(self, subject: str | None = None, **kwargs: Any) -> Any:
        """Issue a portable attestation of the counterparty's earned standing and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.attest_reputation` (defaulting
        ``subject`` to :attr:`seller`), stores the
        :class:`~vincio.settlement.ReputationAttestation` on :attr:`attestation`, and
        returns it.
        """
        subject = subject or self.seller
        attestation = self.app.attest_reputation(subject, **kwargs)
        self.attestation = attestation
        self._record(
            "attest_reputation",
            attestation,
            subject=getattr(attestation, "subject", subject),
            successes=getattr(attestation, "successes", None),
            failures=getattr(attestation, "failures", None),
        )
        return attestation

    def import_reputation(self, attestations: list[Any], **kwargs: Any) -> Any:
        """Combine issuers' attestations into a portable prior and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.import_reputation`, stores the
        :class:`~vincio.settlement.PortableReputation` on :attr:`reputation`, and
        returns it.
        """
        prior = self.app.import_reputation(attestations, **kwargs)
        self.reputation = prior
        self._record(
            "import_reputation",
            prior,
            subject=getattr(prior, "subject", None),
            issuers=len(getattr(prior, "standings", []) or getattr(prior, "issuers", []) or []),
        )
        return prior

    # -- collateral -----------------------------------------------------------

    def post_escrow(self, contract: Any | None = None, **kwargs: Any) -> Any:
        """Post collateral against a contract as a signed escrow and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.post_escrow` (defaulting
        ``contract`` to :attr:`contract`), stores the
        :class:`~vincio.settlement.Escrow` on :attr:`escrow`, and returns it.
        """
        contract = contract if contract is not None else self.contract
        if contract is None:
            raise SettlementError("post_escrow() needs a contract; negotiate one first or pass contract=")
        escrow = self.app.post_escrow(contract, **kwargs)
        self.escrow = escrow
        self._record("post_escrow", escrow, held_usd=getattr(escrow, "held_usd", None))
        return escrow

    def post_collateral_pool(self, contracts: Any, **kwargs: Any) -> Any:
        """Post one stake backing many contracts as a margin account and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.post_collateral_pool`, stores
        the :class:`~vincio.settlement.CollateralPool` on :attr:`pool`, and returns it.
        """
        pool = self.app.post_collateral_pool(contracts, **kwargs)
        self.pool = pool
        self._record("post_collateral_pool", pool, posted_usd=getattr(pool, "posted_usd", None))
        return pool

    def guard_collateral(self, pools: list[Any], **kwargs: Any) -> Any:
        """Fold a counterparty's pools into a rehypothecation re-use guard and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.guard_collateral`, stores the
        :class:`~vincio.settlement.CollateralLedger` on :attr:`ledger`, and returns it.
        """
        ledger = self.app.guard_collateral(pools, **kwargs)
        self.ledger = ledger
        self._record("guard_collateral", ledger, status=getattr(ledger, "status", None))
        return ledger

    # -- solvency & insolvency ------------------------------------------------

    def attest_custody(self, poster: str, reserves: Any, **kwargs: Any) -> Any:
        """Attest a poster's proven reserves (proof-of-reserves) and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.attest_custody`, stores the
        :class:`~vincio.settlement.CustodyAttestation` on :attr:`reserves`, and returns
        it.
        """
        attestation = self.app.attest_custody(poster, reserves, **kwargs)
        self.reserves = attestation
        self._record(
            "attest_custody", attestation, poster=poster, reserves_usd=getattr(attestation, "reserves_usd", None)
        )
        return attestation

    def attest_liabilities(self, poster: str, liabilities: Any, **kwargs: Any) -> Any:
        """Attest a poster's total obligations (proof-of-liabilities) and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.attest_liabilities`, stores the
        :class:`~vincio.settlement.LiabilityAttestation` on :attr:`liabilities`, and
        returns it.
        """
        attestation = self.app.attest_liabilities(poster, liabilities, **kwargs)
        self.liabilities = attestation
        self._record(
            "attest_liabilities",
            attestation,
            poster=poster,
            liabilities_usd=getattr(attestation, "liabilities_usd", None),
        )
        return attestation

    def prove_solvency(self, custody: Any, liabilities: Any, **kwargs: Any) -> Any:
        """Fold a reserve proof against a liability proof into a proof-of-solvency and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.prove_solvency`, stores the
        :class:`~vincio.settlement.SolvencyProof` on :attr:`solvency`, and returns it.
        """
        proof = self.app.prove_solvency(custody, liabilities, **kwargs)
        self.solvency = proof
        self._record(
            "prove_solvency", proof, status=getattr(proof, "status", None), margin_usd=getattr(proof, "margin_usd", None)
        )
        return proof

    def check_completeness(self, liabilities: Any, claims: Any, **kwargs: Any) -> Any:
        """Fold creditor claims against a liability attestation into a completeness check and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.check_completeness`, stores the
        :class:`~vincio.settlement.CompletenessProof` on :attr:`completeness`, and
        returns it.
        """
        proof = self.app.check_completeness(liabilities, claims, **kwargs)
        self.completeness = proof
        self._record("check_completeness", proof, status=getattr(proof, "status", None))
        return proof

    def check_root_consistency(self, attestations: Any, **kwargs: Any) -> Any:
        """Compare liability attestations across creditors for non-equivocation and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.check_root_consistency`, stores
        the :class:`~vincio.settlement.RootConsistencyReport` on :attr:`root_consistency`,
        and returns it.
        """
        report = self.app.check_root_consistency(attestations, **kwargs)
        self.root_consistency = report
        self._record(
            "check_root_consistency",
            report,
            consistent=getattr(report, "consistent", None),
            equivocations=len(getattr(report, "equivocations", []) or []),
        )
        return report

    def check_history_consistency(self, attestations: Any, **kwargs: Any) -> Any:
        """Walk a poster's liability snapshots for cross-time monotonicity and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.check_history_consistency`,
        stores the :class:`~vincio.settlement.HistoryConsistencyReport` on
        :attr:`history`, and returns it.
        """
        report = self.app.check_history_consistency(attestations, **kwargs)
        self.history = report
        self._record(
            "check_history_consistency", report, consistent=getattr(report, "consistent", None)
        )
        return report

    def resolve_insolvency(self, custody: Any, liabilities: Any, schedule: Any | None = None, **kwargs: Any) -> Any:
        """Distribute proven reserves across ranked liabilities into a resolution and record it.

        Delegates to :meth:`~vincio.core.app.ContextApp.resolve_insolvency` (pass
        ``set_off=`` to close-out net each creditor first), stores the
        :class:`~vincio.settlement.InsolvencyResolution` on :attr:`insolvency`, and
        returns it.
        """
        resolution = self.app.resolve_insolvency(custody, liabilities, schedule, **kwargs)
        self.insolvency = resolution
        self._record(
            "resolve_insolvency",
            resolution,
            status=getattr(resolution, "status", None),
            solvent=getattr(resolution, "solvent", None),
            distributed_usd=getattr(resolution, "distributed_usd", None),
        )
        return resolution

    # -- closure --------------------------------------------------------------

    def record_stage(self, stage: str, artifact: Any, **summary: Any) -> Any:
        """Record an arbitrary already-produced artifact as a custom engagement stage.

        An escape hatch for a primitive without a dedicated facade method (or a
        caller-built artifact): binds ``artifact``'s digest into the narrative under
        the ``stage`` label, exactly as the lifecycle methods do, and returns it.
        """
        self._record(stage, artifact, **summary)
        return artifact

    def seal(self, *, sign: bool = True, record_audit: bool = True) -> EngagementNarrative:
        """Mint the content-bound, signed :class:`EngagementNarrative` of the engagement.

        Hash-links every recorded stage, signs the narrative as the coordinator, and
        — unless ``record_audit`` is off — lands the sealed engagement on the app's
        hash-chained audit log under ``cross_org_engagement``. Returns the narrative;
        re-sealing after more stages run produces a fresh one.
        """
        narrative = EngagementNarrative(
            id=new_id("engagement"),
            coordinator=self.coordinator,
            buyer=self.buyer,
            seller=self.seller,
            scope=self.scope,
            started_at=self._started_at,
            sealed_at=utcnow(),
            stages=[s.model_copy(deep=True) for s in self._stages],
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

    def verify(self, verifier: Any | None = None, *, require: list[str] | None = None) -> EngagementVerification:
        """Verify the whole engagement offline — the narrative chain *and* every live artifact.

        Seals the narrative if needed, then verifies its hash chain from the bytes
        alone and re-digests every captured artifact against its bound digest, so a
        tamper anywhere in the engagement — a re-ordered stage or an edited underlying
        artifact — is caught. Pass the fabric ``verifier`` to additionally authenticate
        the coordinator's signature.
        """
        narrative = self.narrative or self.seal(sign=verifier is not None, record_audit=False)
        return narrative.verify(verifier, require=require, artifacts=list(self._artifacts))

    def _signer(self) -> Any:
        """The app's contract signer, when one is resolvable."""
        resolver = getattr(self.app, "_resolve_contract_signer", None)
        if callable(resolver):
            return resolver(None, True)
        return getattr(self.app, "contract_signer", None)
