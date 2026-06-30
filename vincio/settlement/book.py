"""The settlement engine and the durable, hash-chained book of settlements.

:func:`settle_contract` reconciles one contract against its metered delivery into
a :class:`~vincio.settlement.record.SettlementRecord`; :func:`settle_saga` does it
for every contract a cross-org saga ran under, from the durable journal. A
:class:`SettlementBook` is an org's tamper-evident ledger of those records — the
settlement analogue of the :class:`~vincio.choreography.SagaJournal`: each record
links to the previous by an entry hash, so :meth:`SettlementBook.verify`
recomputes the whole book offline and pinpoints any edited record, while the
party signatures stay over the per-record reconciliation hash so two orgs' books
co-sign the same economic facts.

Closing the books on a contract also closes the reputation loop: a settlement that
fulfils its terms credits the seller and one that overruns or falls short debits
it, so reliability earned in *delivery* weights the next negotiation — the same
:class:`~vincio.optimize.reputation.ReputationLedger`
:meth:`~vincio.core.app.ContextApp.enforce_contract` already feeds.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.diagnostics import note_suppressed
from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .collateral import COLLATERAL_ACTION, CollateralPool, post_collateral_pool
from .custody import CUSTODY_ACTION, CustodyAttestation, attest_custody
from .escrow import ESCROW_ACTION, Escrow, post_escrow
from .meter import Meter, MeterReading
from .record import (
    SETTLEMENT_ACTION,
    Reconciliation,
    SettlementLine,
    SettlementRecord,
    reconcile,
)
from .rehypothecation import REHYPOTHECATION_ACTION, CollateralLedger, guard_collateral
from .setoff import (
    SETOFF_ACTION,
    SetOffStatement,
    build_set_off_statement,
    set_off_from_records,
)
from .solvency import (
    COMPLETENESS_ACTION,
    DISCHARGE_ACTION,
    EQUIVOCATION_ACTION,
    HISTORY_ACTION,
    LIABILITY_ACTION,
    SOLVENCY_ACTION,
    CompletenessProof,
    Discharge,
    EquivocationProof,
    HistoryConsistencyProof,
    HistoryConsistencyReport,
    InclusionProof,
    LiabilityAttestation,
    RootConsistencyReport,
    SolvencyProof,
    attest_liabilities,
    check_completeness,
    check_history_consistency,
    check_root_consistency,
    discharge_liability,
    prove_solvency,
)
from .waterfall import (
    INSOLVENCY_ACTION,
    SENIORITY_ACTION,
    InsolvencyResolution,
    SenioritySchedule,
    build_seniority_schedule,
    resolve_insolvency,
)

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "SettlementBook",
    "SettlementRow",
    "SettlementReport",
    "BookVerification",
    "settle_contract",
    "settle_saga",
]

# The metadata-store record kind a settlement book is persisted under (generic
# ``records`` table on backed stores; no RECORD_KINDS registration needed).
SETTLEMENT_STORE_KIND = "settlement_books"


def _delivered_from(
    reading: MeterReading | None,
    cost_usd: float | None,
    latency_ms: float | None,
    quality: float | None,
) -> tuple[float | None, float | None, float | None, float, int]:
    """Resolve delivered metrics from a reading or explicit figures."""
    if reading is not None:
        return (
            reading.cost_usd if cost_usd is None else cost_usd,
            reading.latency_ms if latency_ms is None else latency_ms,
            reading.quality if quality is None else quality,
            reading.units,
            reading.events,
        )
    units = 0.0
    events = 0
    if cost_usd is not None or latency_ms is not None or quality is not None:
        units = 1.0
        events = 1
    return cost_usd, latency_ms, quality, units, events


def _build_lines(
    *,
    price_usd: float,
    sla_seconds: float,
    quality_floor: float,
    cost_usd: float | None,
    latency_ms: float | None,
    quality: float | None,
    breaches: list[str],
) -> list[SettlementLine]:
    """One reconciliation line per dimension the contract constrains."""
    lines: list[SettlementLine] = []
    breached = {b.split(":", 1)[0] for b in breaches}
    if price_usd > 0:
        lines.append(
            SettlementLine(
                dimension="price",
                owed=round(price_usd, 9),
                delivered=None if cost_usd is None else round(cost_usd, 9),
                within="price" not in breached,
                delta=None if cost_usd is None else round(price_usd - cost_usd, 9),
                note="" if cost_usd is not None else "not metered",
            )
        )
    if sla_seconds > 0:
        owed_ms = round(sla_seconds * 1000.0, 6)
        lines.append(
            SettlementLine(
                dimension="sla",
                owed=owed_ms,
                delivered=None if latency_ms is None else round(latency_ms, 6),
                within="sla" not in breached,
                delta=None if latency_ms is None else round(owed_ms - latency_ms, 6),
                note="" if latency_ms is not None else "not metered",
            )
        )
    if quality_floor > 0:
        lines.append(
            SettlementLine(
                dimension="quality",
                owed=round(quality_floor, 9),
                delivered=None if quality is None else round(quality, 9),
                within="quality" not in breached,
                delta=None if quality is None else round(quality - quality_floor, 9),
                note="" if quality is not None else "not metered",
            )
        )
    return lines


def settle_contract(
    contract: Any,
    *,
    reading: MeterReading | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
    quality: float | None = None,
    run_id: str | None = None,
    saga_id: str | None = None,
) -> SettlementRecord:
    """Reconcile delivery against a contract into an (unsigned) settlement record.

    Delivered metrics come from a :class:`~vincio.settlement.meter.MeterReading`
    (``reading``) or explicit ``cost_usd`` / ``latency_ms`` / ``quality`` figures;
    the breach verdict reuses :meth:`~vincio.negotiation.Contract.check`, so a
    settlement holds delivery to exactly the same terms the runtime enforces as a
    budget. The agreed price is what is owed for the scope, and the
    :attr:`~vincio.settlement.record.SettlementRecord.balance_usd` is that price
    minus the delivered cost (a credit when under, an overrun when over). The
    returned record is sealed but unsigned — sign it with each party's key (or let
    a :class:`SettlementBook` do it on :meth:`~SettlementBook.settle`).
    """
    terms = contract.terms
    d_cost, d_latency, d_quality, units, events = _delivered_from(
        reading, cost_usd, latency_ms, quality
    )
    fulfillment = contract.check(cost_usd=d_cost, latency_ms=d_latency, quality=d_quality)
    breaches = list(fulfillment.breaches)
    amount_owed = round(float(terms.price_usd), 9)
    balance = round(amount_owed - (d_cost or 0.0), 9)
    lines = _build_lines(
        price_usd=float(terms.price_usd),
        sla_seconds=float(terms.sla_seconds),
        quality_floor=float(terms.quality_floor),
        cost_usd=d_cost,
        latency_ms=d_latency,
        quality=d_quality,
        breaches=breaches,
    )
    record = SettlementRecord(
        contract_id=contract.id,
        buyer=contract.buyer,
        seller=contract.seller,
        scope=terms.scope,
        run_id=run_id or (reading.run_id if reading is not None else None),
        saga_id=saga_id,
        price_usd=round(float(terms.price_usd), 9),
        sla_seconds=round(float(terms.sla_seconds), 9),
        quality_floor=round(float(terms.quality_floor), 9),
        delivered_cost_usd=None if d_cost is None else round(d_cost, 9),
        delivered_latency_ms=None if d_latency is None else round(d_latency, 9),
        delivered_quality=None if d_quality is None else round(d_quality, 9),
        metered_units=round(units, 9),
        metered_events=events,
        lines=lines,
        amount_owed_usd=amount_owed,
        balance_usd=balance,
        fulfilled=fulfillment.fulfilled,
        status="settled" if fulfillment.fulfilled else "breached",
        breaches=breaches,
    )
    return record.seal()


def settle_saga(
    result: Any,
    *,
    contracts: dict[str, Any],
    run_id: str | None = None,
) -> list[SettlementRecord]:
    """Settle every contract a cross-org saga ran under, from its durable journal.

    Meters each completed forward step that ran under a contract
    (:meth:`~vincio.settlement.meter.Meter.from_saga`) and reconciles the per-step
    delivery against the matching contract in ``contracts`` (keyed by contract id).
    Returns one sealed, unsigned :class:`~vincio.settlement.record.SettlementRecord`
    per contract, in contract-id order. A contracted step whose contract is missing
    from ``contracts`` raises :class:`SettlementError` — the books cannot close on
    terms the caller did not supply.
    """
    journal = getattr(result, "journal", result)
    meters = Meter.from_saga(result, run_id=run_id)
    records: list[SettlementRecord] = []
    for contract_id in sorted(meters):
        contract = contracts.get(contract_id)
        if contract is None:
            raise SettlementError(
                f"saga ran under contract {contract_id!r} but no matching contract was "
                f"supplied to settle it",
                details={"contract_id": contract_id, "saga_id": journal.id},
            )
        records.append(
            settle_contract(
                contract,
                reading=meters[contract_id].reading(),
                run_id=run_id or journal.id,
                saga_id=journal.id,
            )
        )
    return records


class SettlementRow(BaseModel):
    """One counterparty's line in a :class:`SettlementReport`."""

    counterparty: str
    settlements: int = 0
    settled: int = 0
    breached: int = 0
    total_owed_usd: float = 0.0
    total_delivered_usd: float = 0.0
    net_balance_usd: float = 0.0


class SettlementReport(BaseModel):
    """Per-counterparty settlement roll-up, alongside the cost report.

    Each row totals what was owed, what was delivered, and the net balance across
    every settlement with that counterparty, with the settled / breached tally
    behind it, so a cross-org engagement's books are a mechanical, auditable figure.
    """

    owner: str = ""
    rows: list[SettlementRow] = Field(default_factory=list)

    @property
    def net_balance_usd(self) -> float:
        """Net balance across every counterparty (credit positive, overrun negative)."""
        return round(sum(r.net_balance_usd for r in self.rows), 9)

    @property
    def breached(self) -> int:
        """Total breached settlements across every counterparty."""
        return sum(r.breached for r in self.rows)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print a compact per-counterparty settlement table."""
        print(f"Settlement report ({self.owner})" if self.owner else "Settlement report")
        for row in self.rows:
            print(
                f"  {row.counterparty}: owed=${row.total_owed_usd:.4f} "
                f"delivered=${row.total_delivered_usd:.4f} balance=${row.net_balance_usd:+.4f} "
                f"({row.settled} settled / {row.breached} breached)"
            )


class BookVerification(BaseModel):
    """The (non-raising) outcome of verifying a settlement book offline."""

    intact: bool
    entries: int
    broken_at: int | None = None
    reason: str | None = None


def _entry_hash(record: SettlementRecord) -> str:
    """The book-chain link for a record: its reconciliation hash plus its position."""
    return stable_hash(
        {
            "content_hash": record.content_hash,
            "prev_hash": record.prev_hash,
            "seq": record.seq,
            "run_id": record.run_id,
            "saga_id": record.saga_id,
            "settled_at": record.settled_at.isoformat(),
        },
        length=32,
    )


class SettlementBook:
    """An org's durable, hash-chained, offline-verifiable ledger of settlements.

    Append a reconciled settlement with :meth:`settle` (which reconciles, signs as
    the owner's side, links the record into the chain, records it on the audit
    chain, and closes the reputation loop) or :meth:`append` a pre-built record.
    Each record links to the previous by an entry hash, so :meth:`verify`
    recomputes the whole book from the bytes alone and pinpoints any tampered
    record; :meth:`report` rolls the books up per counterparty. The party
    signatures stay over each record's reconciliation hash, so the buyer's book and
    the seller's book co-sign the *same* economic facts and :func:`reconcile` ties
    them out.
    """

    def __init__(
        self,
        owner: str,
        *,
        signer: ChainSigner | None = None,
        audit: Any | None = None,
        events: Any | None = None,
        store: Any | None = None,
        reputation: Any | None = None,
        book_id: str | None = None,
    ) -> None:
        self.owner = owner
        self.signer = signer
        self.audit = audit
        self.events = events
        self.store = store
        self.reputation = reputation
        # A unique id by default so an org's books never collide on a shared store;
        # pass a stable ``book_id`` to keep one durable ledger across restarts (the
        # settlement analogue of a saga's id).
        self.id = book_id or new_id("settlement-book")
        self.records: list[SettlementRecord] = []
        self.head_hash = ""
        self.created_at = utcnow()
        self.updated_at = utcnow()
        if store is not None:
            self._load()

    # -- settlement ---------------------------------------------------------

    def settle(
        self,
        contract: Any,
        *,
        reading: MeterReading | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        quality: float | None = None,
        run_id: str | None = None,
        saga_id: str | None = None,
        party: str | None = None,
        sign: bool = True,
        record_reputation: bool = True,
        escrow: Escrow | None = None,
        escrow_config: Any | None = None,
        pool: CollateralPool | None = None,
    ) -> SettlementRecord:
        """Reconcile delivery against a contract and close the books on it.

        Builds the reconciled :class:`~vincio.settlement.record.SettlementRecord`,
        signs it as ``party`` (defaulting to whichever side this book's owner is on
        the contract), links it into the book's hash chain, records the verdict on
        the audit chain, and — unless ``record_reputation`` is off — credits the
        seller on fulfilment or debits it on a breach, so reliability earned in
        delivery weights the next negotiation. Returns the appended record.

        Pass an ``escrow`` posted against the contract to settle the collateral in the
        same call: it is resolved against the record (the whole stake released on a
        fulfilled delivery, a bounded proportional slice forfeited on a breach), signed,
        and audited in place — so the collateral closes the same loop the settlement
        does. ``escrow_config`` overrides the forfeiture policy for that resolution.

        Pass a ``pool`` the contract is backed by to draw the same settlement against a
        shared :class:`~vincio.settlement.collateral.CollateralPool` instead: the
        forfeiture is drawn from the pooled stake and the rest released back to the
        available balance, re-signed and audited in place.
        """
        record = settle_contract(
            contract,
            reading=reading,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality=quality,
            run_id=run_id,
            saga_id=saga_id,
        )
        resolved_party = self._resolve_party(party, record)
        if sign and self.signer is not None and resolved_party is not None:
            record.sign(self.signer, party=resolved_party)
        self.append(record)
        self._record_reputation(record, record_reputation)
        if escrow is not None:
            self.settle_escrow(escrow, record, party=party, sign=sign, config=escrow_config)
        if pool is not None:
            self.draw_pool(pool, record, party=party, sign=sign, config=escrow_config)
        return record

    def settle_saga(
        self,
        result: Any,
        *,
        contracts: dict[str, Any],
        run_id: str | None = None,
        party: str | None = None,
        sign: bool = True,
        record_reputation: bool = True,
    ) -> list[SettlementRecord]:
        """Close the books on every contract a cross-org saga ran under.

        Meters each contracted forward step from the saga's durable journal
        (:func:`settle_saga`) and signs, links, audits, and reputation-closes one
        record per contract on this book, in contract-id order.
        """
        records = settle_saga(result, contracts=contracts, run_id=run_id)
        for record in records:
            resolved_party = self._resolve_party(party, record)
            if sign and self.signer is not None and resolved_party is not None:
                record.sign(self.signer, party=resolved_party)
            self.append(record)
            self._record_reputation(record, record_reputation)
        return records

    def append(self, record: SettlementRecord) -> SettlementRecord:
        """Link a (sealed) record into the chain, audit it, and checkpoint."""
        if not record.content_hash:
            record.seal()
        record.seq = len(self.records)
        record.prev_hash = self.head_hash
        record.entry_hash = _entry_hash(record)
        self.records.append(record)
        self.head_hash = record.entry_hash
        self.updated_at = utcnow()
        self._audit(record)
        self._checkpoint()
        self._emit(record)
        return record

    def _resolve_party(self, party: str | None, record: Any) -> str | None:
        if party is not None:
            return party
        if self.owner in (record.buyer, record.seller):
            return self.owner
        return None

    # -- collateral / escrow ------------------------------------------------

    def post_escrow(
        self,
        contract: Any,
        *,
        decision: Any | None = None,
        fraction: float | None = None,
        amount: float | None = None,
        poster: str | None = None,
        beneficiary: str | None = None,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Escrow:
        """Post collateral against a contract, signed and audited.

        Builds the :class:`~vincio.settlement.escrow.Escrow`
        (:func:`~vincio.settlement.escrow.post_escrow`) holding the admission-required
        collateral — read from an :class:`~vincio.settlement.admission.AdmissionDecision`
        (``decision``), an explicit ``fraction`` / ``amount``, or the admission posture
        stamped onto the contract's terms — signs it as this book's owner (when a side of
        the contract), and records the posting on the audit chain. Returns the escrow.
        """
        escrow = post_escrow(
            contract,
            decision=decision,
            fraction=fraction,
            amount=amount,
            poster=poster,
            beneficiary=beneficiary,
            config=config,
        )
        resolved_party = self._resolve_party(party, escrow)
        if sign and self.signer is not None and resolved_party is not None:
            escrow.sign(self.signer, party=resolved_party)
        self._audit_escrow(escrow)
        return escrow

    def settle_escrow(
        self,
        escrow: Escrow,
        record: SettlementRecord,
        *,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Escrow:
        """Resolve a posted escrow against a settlement record, signed and audited.

        Releases the whole stake on a fulfilled delivery and forfeits a bounded,
        proportional slice on a breach (:meth:`Escrow.resolve`), re-signs the resolved
        escrow as this book's owner, and records the release / forfeiture on the audit
        chain — so the collateral's whole lifecycle is on the same tamper-evident log the
        settlement is. Returns the resolved escrow.
        """
        escrow.resolve(record, config=config)
        resolved_party = self._resolve_party(party, escrow)
        if sign and self.signer is not None and resolved_party is not None:
            escrow.sign(self.signer, party=resolved_party)
        self._audit_escrow(escrow)
        return escrow

    def _audit_escrow(self, escrow: Escrow) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            ESCROW_ACTION,
            resource=escrow.contract_id,
            decision=escrow.state,
            details=escrow.audit_details(),
        )
        escrow.audit_id = getattr(entry, "id", None)

    # -- collateral pooling -------------------------------------------------

    def post_collateral_pool(
        self,
        contracts: Any,
        *,
        poster: str | None = None,
        posted: float | None = None,
        decisions: Any | None = None,
        fraction: float | None = None,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> CollateralPool:
        """Post one stake backing many contracts, signed and audited.

        Builds the :class:`~vincio.settlement.collateral.CollateralPool`
        (:func:`~vincio.settlement.collateral.post_collateral_pool`) holding a single stake
        against the admission-required collateral of each contract — read from a matching
        :class:`~vincio.settlement.admission.AdmissionDecision` in ``decisions``, a uniform
        ``fraction``, or the admission posture stamped onto each contract's terms — signs it
        as this book's owner (when a party to the pool), and records the posting on the
        audit chain. Returns the pool.
        """
        pool = post_collateral_pool(
            contracts,
            poster=poster,
            posted=posted,
            decisions=decisions,
            fraction=fraction,
            config=config,
        )
        resolved_party = self._resolve_pool_party(party, pool)
        if sign and self.signer is not None and resolved_party is not None:
            pool.sign(self.signer, party=resolved_party)
        self._audit_pool(pool)
        return pool

    def draw_pool(
        self,
        pool: CollateralPool,
        record: SettlementRecord,
        *,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> CollateralPool:
        """Draw one backed contract's settlement against a collateral pool, signed and audited.

        Draws a bounded forfeiture from the shared stake on a breach and releases the rest
        back to the available balance on a clean delivery
        (:meth:`~vincio.settlement.collateral.CollateralPool.draw`), re-signs the pool as
        this book's owner, and records the draw on the audit chain — so the pooled
        collateral's whole lifecycle is on the same tamper-evident log the settlement is.
        Returns the pool.
        """
        pool.draw(record, config=config)
        resolved_party = self._resolve_pool_party(party, pool)
        if sign and self.signer is not None and resolved_party is not None:
            pool.sign(self.signer, party=resolved_party)
        self._audit_pool(pool)
        return pool

    def _resolve_pool_party(self, party: str | None, pool: CollateralPool) -> str | None:
        if party is not None:
            return party
        return self.owner if self.owner in pool.parties else None

    def _audit_pool(self, pool: CollateralPool) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            COLLATERAL_ACTION,
            resource=pool.id,
            decision=pool.status,
            details=pool.audit_details(),
        )
        pool.audit_id = getattr(entry, "id", None)

    # -- custody attestation (proof-of-reserves) ----------------------------

    def attest_custody(
        self,
        poster: str,
        reserves: Any,
        *,
        custodian: str | None = None,
        as_of: Any | None = None,
        sign: bool = True,
    ) -> CustodyAttestation:
        """Attest a poster's proven reserves into a signed, audited proof-of-reserves.

        Builds the :class:`~vincio.settlement.custody.CustodyAttestation`
        (:func:`~vincio.settlement.custody.attest_custody`) over the capital ``poster`` holds
        — itemized ``reserves`` whose total re-derives on every verify — signs it as the
        custodian (this book's owner by default, i.e. self-custody when the owner is the
        poster), and records the issuance on the audit chain. The attestation reads as the
        ``held`` figure of :meth:`guard_collateral` (``custody=``). Returns it.
        """
        attestation = attest_custody(
            poster, reserves, custodian=custodian or self.owner, as_of=as_of
        )
        if sign and self.signer is not None and self.owner == attestation.custodian:
            attestation.sign(self.signer, party=attestation.custodian)
        self._audit_custody(attestation)
        return attestation

    def _audit_custody(self, attestation: CustodyAttestation) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            CUSTODY_ACTION,
            resource=attestation.poster,
            decision="self_custody" if attestation.self_custody else "custodied",
            details=attestation.audit_details(),
        )
        attestation.audit_id = getattr(entry, "id", None)

    # -- liability attestation & proof-of-solvency --------------------------

    def attest_liabilities(
        self,
        poster: str,
        liabilities: Any,
        *,
        attestor: str | None = None,
        as_of: Any | None = None,
        prior: LiabilityAttestation | None = None,
        sign: bool = True,
    ) -> LiabilityAttestation:
        """Attest a poster's total obligations into a signed, audited proof-of-liabilities.

        Builds the :class:`~vincio.settlement.solvency.LiabilityAttestation`
        (:func:`~vincio.settlement.solvency.attest_liabilities`) over the obligations ``poster``
        owes — itemized ``liabilities`` whose total re-derives on every verify — signs it as the
        attestor (this book's owner by default, i.e. self-attested when the owner is the poster),
        and records the issuance on the audit chain. It folds against a proof-of-reserves in
        :meth:`prove_solvency`. Pass ``prior`` (the preceding snapshot) to link it into a
        hash-linked history :meth:`check_history_consistency` can walk. Returns it.
        """
        attestation = attest_liabilities(
            poster, liabilities, attestor=attestor or self.owner, as_of=as_of, prior=prior
        )
        if sign and self.signer is not None and self.owner == attestation.attestor:
            attestation.sign(self.signer, party=attestation.attestor)
        self._audit_liabilities(attestation)
        return attestation

    def _audit_liabilities(self, attestation: LiabilityAttestation) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            LIABILITY_ACTION,
            resource=attestation.poster,
            decision="self_attested" if attestation.self_attested else "attested",
            details=attestation.audit_details(),
        )
        attestation.audit_id = getattr(entry, "id", None)

    def inclusion_proof(self, liabilities: LiabilityAttestation, creditor: str) -> InclusionProof:
        """Build an offline-verifiable inclusion proof for one creditor's claim.

        Thin wrapper over :meth:`~vincio.settlement.solvency.LiabilityAttestation.inclusion_proof`:
        the proof shows ``creditor``'s obligation is a leaf of the attestation's signed Merkle
        root, so the creditor confirms its claim was counted in the attested total. Returns it.
        """
        return liabilities.inclusion_proof(creditor)

    def check_completeness(
        self,
        liabilities: LiabilityAttestation,
        claims: Any | None = None,
        *,
        verify_with: ChainSigner | None = None,
        as_of: Any | None = None,
        sign: bool = True,
    ) -> CompletenessProof:
        """Fold creditor claims against a liability attestation into a signed completeness check.

        Builds the :class:`~vincio.settlement.solvency.CompletenessProof`
        (:func:`~vincio.settlement.solvency.check_completeness`), pinpointing every claim the
        attestation omits or under-states as an
        :class:`~vincio.settlement.solvency.OmissionBreach` and raising the attested figure to a
        completed total :meth:`prove_solvency` reads (``completeness=``). When ``claims`` is
        omitted they are derived from this book's own settled records against the attestation's
        poster — what this book's owner, as a creditor, can prove it is owed. Signs the check as
        this book's owner and records it on the audit chain. A tampered attestation is refused;
        with ``verify_with`` a forged attestor signature is too. Returns the check.
        """
        resolved_claims = self.claims_against(liabilities.poster) if claims is None else claims
        proof = check_completeness(liabilities, resolved_claims, verifier=verify_with, as_of=as_of)
        if sign and self.signer is not None:
            proof.sign(self.signer, party=self.owner)
        self._audit_completeness(proof)
        return proof

    def claims_against(self, poster: str) -> dict[str, float]:
        """What this book's owner, as a creditor, can prove ``poster`` owes it from its records.

        Sums ``amount_owed_usd`` over every settled record where this book's owner is the seller
        and ``poster`` is the buyer — the obligations the owner delivered against and is owed.
        Folded by :meth:`check_completeness` to detect a liability attestation that omits or
        under-states what this creditor is owed.
        """
        owed = 0.0
        for record in self.records:
            if record.seller == self.owner and record.buyer == poster:
                owed += float(record.amount_owed_usd)
        return {self.owner: round(owed, 6)} if owed > 0.0 else {}

    def _audit_completeness(self, proof: CompletenessProof) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            COMPLETENESS_ACTION,
            resource=proof.poster,
            decision=proof.status,
            details=proof.audit_details(),
        )
        proof.audit_id = getattr(entry, "id", None)

    def prove_solvency(
        self,
        custody: CustodyAttestation,
        liabilities: LiabilityAttestation,
        *,
        poster: str | None = None,
        completeness: CompletenessProof | None = None,
        as_of: Any | None = None,
        verify_with: ChainSigner | None = None,
        sign: bool = True,
    ) -> SolvencyProof:
        """Fold a reserve proof against a liability proof into a signed, audited proof-of-solvency.

        Builds the :class:`~vincio.settlement.solvency.SolvencyProof`
        (:func:`~vincio.settlement.solvency.prove_solvency`) reconciling the proven reserves
        against the proven liabilities into a bounded solvency margin, pinpointing an
        :class:`~vincio.settlement.solvency.InsolvencyBreach` when the liabilities exceed the
        reserves. Pass ``completeness`` (a
        :class:`~vincio.settlement.solvency.CompletenessProof` over this attestation) to bound
        the margin against the *completed* liability total — the attestor's figure raised by
        every obligation a creditor proved it omitted. Signs the proof as this book's owner and
        records it on the audit chain. A tampered or wrong-poster attestation (or completeness
        check) is refused; with ``verify_with`` a forged signature is too. The proof's
        solvency-adjusted held figure reads into :meth:`guard_collateral` (``solvency=``).
        Returns the proof.
        """
        proof = prove_solvency(
            custody,
            liabilities,
            poster=poster,
            completeness=completeness,
            as_of=as_of,
            verifier=verify_with,
        )
        if sign and self.signer is not None:
            proof.sign(self.signer, party=self.owner)
        self._audit_solvency(proof)
        return proof

    def _audit_solvency(self, proof: SolvencyProof) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            SOLVENCY_ACTION,
            resource=proof.poster,
            decision=proof.status,
            details=proof.audit_details(),
        )
        proof.audit_id = getattr(entry, "id", None)

    def check_root_consistency(
        self,
        attestations: Any,
        *,
        verify_with: ChainSigner | None = None,
        record_reputation: bool = True,
    ) -> RootConsistencyReport:
        """Compare liability attestations across creditors for cross-org non-equivocation.

        Folds a set of liability attestations a group of creditors hold — each the attestation a
        poster signed *for it* — into a
        :class:`~vincio.settlement.solvency.RootConsistencyReport`
        (:func:`~vincio.settlement.solvency.check_root_consistency`), surfacing every poster that
        signed **different** roots for the same ``(poster, attestor, as_of)`` as a non-repudiable
        :class:`~vincio.settlement.solvency.EquivocationProof`. Completeness catches an omission
        only when the omitted creditor folds its own claim; this catches the counterparty that
        equivocates across the set. ``attestations`` items may be bare attestations or
        ``(creditor, attestation)`` pairs to record which creditor saw each root. With
        ``verify_with`` only attestor-signed roots are admitted as evidence, so a forged root
        cannot manufacture a false accusation. Records each proven equivocation on the audit chain
        and — unless ``record_reputation`` is off — credits a failure against the equivocating
        poster on the bound reputation ledger. Returns the report.
        """
        report = check_root_consistency(attestations, verifier=verify_with)
        dinged: set[str] = set()
        for proof in report.equivocations:
            self._audit_equivocation(proof)
            # Every distinct pairwise conflict is audited, but a poster's reputation is debited
            # once per check — three conflicting roots are one equivocating counterparty.
            if proof.poster not in dinged:
                self._record_equivocation_reputation(proof, record_reputation)
                dinged.add(proof.poster)
        return report

    def _audit_equivocation(self, proof: EquivocationProof) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            EQUIVOCATION_ACTION,
            resource=proof.poster,
            decision="equivocation",
            details=proof.audit_details(),
        )
        proof.audit_id = getattr(entry, "id", None)

    def _record_equivocation_reputation(self, proof: EquivocationProof, enabled: bool) -> None:
        if not enabled or self.reputation is None:
            return
        self.reputation.record_outcome(
            proof.poster,
            passed=False,
            round_id=proof.id,
            details={
                "kind": "liability_equivocation",
                "proof_id": proof.id,
                "attestor": proof.attestor,
                "liabilities_gap_usd": proof.liabilities_gap_usd,
            },
        )

    # -- liability history consistency --------------------------------------

    def discharge_liability(
        self,
        poster: str,
        amount_usd: float,
        *,
        creditor: str | None = None,
        as_of: Any | None = None,
        note: str = "",
        sign: bool = True,
    ) -> Discharge:
        """Issue a signed, audited :class:`~vincio.settlement.solvency.Discharge` of what ``poster`` owes.

        Builds the discharge (:func:`~vincio.settlement.solvency.discharge_liability`) releasing
        ``amount_usd`` of the obligation ``poster`` owes this book's owner — the **creditor** issues
        it, so the owner is the creditor (``creditor`` defaults to the owner) and the book signs it
        with the owner's key. Folded into :meth:`check_history_consistency` (``discharges=``) to
        explain a legitimate reduction in the poster's liabilities between two snapshots, so the
        matching drop is not treated as a debt that silently vanished. Records the issuance on the
        audit chain. Returns it.
        """
        discharge = discharge_liability(
            poster, creditor or self.owner, amount_usd, as_of=as_of, note=note
        )
        if sign and self.signer is not None and self.owner == discharge.creditor:
            discharge.sign(self.signer, party=discharge.creditor)
        self._audit_discharge(discharge)
        return discharge

    def _audit_discharge(self, discharge: Discharge) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            DISCHARGE_ACTION,
            resource=discharge.poster,
            decision=discharge.status,
            details=discharge.audit_details(),
        )
        discharge.audit_id = getattr(entry, "id", None)

    def check_history_consistency(
        self,
        attestations: Any,
        *,
        discharges: Any | None = None,
        verify_with: ChainSigner | None = None,
        record_reputation: bool = True,
    ) -> HistoryConsistencyReport:
        """Walk a poster's liability snapshots for cross-time monotonicity, signed and audited.

        Folds a set of liability snapshots into a
        :class:`~vincio.settlement.solvency.HistoryConsistencyReport`
        (:func:`~vincio.settlement.solvency.check_history_consistency`), surfacing every poster that
        let a creditor's obligation **drop** between snapshots without a signed
        :class:`~vincio.settlement.solvency.Discharge` (``discharges``) explaining the release as a
        pinpointed :class:`~vincio.settlement.solvency.MonotonicityBreach`. Where
        :meth:`check_root_consistency` catches a counterparty signing different roots for the *same*
        instant, this catches one dropping a past obligation in a *later* snapshot. With
        ``verify_with`` only attestor-signed snapshots and creditor-signed discharges are admitted as
        evidence. Signs each chain's proof as this book's owner, records every inconsistent history
        on the audit chain, and — unless ``record_reputation`` is off — credits a failure against the
        breaching poster on the bound reputation ledger. Returns the report.
        """
        report = check_history_consistency(
            attestations, discharges=discharges, verifier=verify_with
        )
        for proof in report.proofs:
            if self.signer is not None:
                proof.sign(self.signer, party=self.owner)
            self._audit_history(proof)
            if not proof.monotone:
                self._record_history_reputation(proof, record_reputation)
        return report

    def _audit_history(self, proof: HistoryConsistencyProof) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            HISTORY_ACTION,
            resource=proof.poster,
            decision=proof.status,
            details=proof.audit_details(),
        )
        proof.audit_id = getattr(entry, "id", None)

    def _record_history_reputation(self, proof: HistoryConsistencyProof, enabled: bool) -> None:
        if not enabled or self.reputation is None:
            return
        self.reputation.record_outcome(
            proof.poster,
            passed=False,
            round_id=proof.id,
            details={
                "kind": "liability_history",
                "proof_id": proof.id,
                "attestor": proof.attestor,
                "unexplained_usd": proof.unexplained_usd,
            },
        )

    # -- insolvency resolution & seniority waterfall ------------------------

    def build_set_off_statement(
        self,
        poster: str,
        creditor: str,
        owed_usd: float | None = None,
        owing_usd: float | None = None,
        *,
        liabilities: LiabilityAttestation | None = None,
        references: Any | None = None,
        as_of: Any | None = None,
        verify_with: ChainSigner | None = None,
        sign: bool = True,
    ) -> SetOffStatement:
        """Collapse the mutual obligations between a poster and one creditor, signed and audited.

        Builds the :class:`~vincio.settlement.setoff.SetOffStatement`
        (:func:`~vincio.settlement.setoff.build_set_off_statement`) stating the obligations running
        *both ways* between ``poster`` and ``creditor`` — what the poster owes the creditor and what
        the creditor owes back — and computing the poster's bounded net liability. Pass explicit
        ``owed_usd`` / ``owing_usd`` figures, or pass a ``liabilities`` attestation to derive the
        statement straight from the signed artifacts (:func:`~vincio.settlement.setoff.set_off_from_records`):
        ``owed_usd`` is read from the attestation's lines for the creditor and ``owing_usd`` from
        this book's own settlement records where the poster is the seller and the creditor the
        buyer. Signs it as this book's owner (one side of the mutually-agreed close-out — the
        counterparty co-signs its copy) and records the issuance on the audit chain. Returns it.
        """
        if liabilities is not None:
            statement = set_off_from_records(
                poster,
                creditor,
                liabilities,
                self.records,
                as_of=as_of,
                verifier=verify_with,
            )
        else:
            statement = build_set_off_statement(
                poster,
                creditor,
                owed_usd or 0.0,
                owing_usd or 0.0,
                references=references,
                as_of=as_of,
            )
        if sign and self.signer is not None:
            statement.sign(self.signer, party=self.owner)
        self._audit_setoff(statement)
        return statement

    def _audit_setoff(self, statement: SetOffStatement) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            SETOFF_ACTION,
            resource=statement.poster,
            decision=statement.direction,
            details=statement.audit_details(),
        )
        statement.audit_id = getattr(entry, "id", None)

    def build_seniority_schedule(
        self,
        poster: str,
        tranches: Any,
        *,
        as_of: Any | None = None,
        sign: bool = True,
    ) -> SenioritySchedule:
        """Rank a poster's obligations into a signed, audited seniority schedule.

        Builds the :class:`~vincio.settlement.waterfall.SenioritySchedule`
        (:func:`~vincio.settlement.waterfall.build_seniority_schedule`) ranking the creditors
        ``poster`` owes into priority tranches — rank ``0`` most senior — an
        :meth:`resolve_insolvency` waterfall pays out in. ``tranches`` is an ordered spec (a list
        of creditor-name lists where position is priority, or
        :class:`~vincio.settlement.waterfall.SeniorityTranche` items). Signs it as this book's
        owner (the poster declaring its own subordination, or a creditor party to an inter-creditor
        agreement) and records the issuance on the audit chain. Returns it.
        """
        schedule = build_seniority_schedule(poster, tranches, as_of=as_of)
        if sign and self.signer is not None:
            schedule.sign(self.signer, party=self.owner)
        self._audit_seniority(schedule)
        return schedule

    def _audit_seniority(self, schedule: SenioritySchedule) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            SENIORITY_ACTION,
            resource=schedule.poster,
            decision="self_ranked" if schedule.poster == self.owner else "ranked",
            details=schedule.audit_details(),
        )
        schedule.audit_id = getattr(entry, "id", None)

    def resolve_insolvency(
        self,
        custody: CustodyAttestation,
        liabilities: LiabilityAttestation,
        schedule: SenioritySchedule | None = None,
        *,
        poster: str | None = None,
        completeness: CompletenessProof | None = None,
        solvency: SolvencyProof | None = None,
        set_off: list[SetOffStatement] | None = None,
        as_of: Any | None = None,
        verify_with: ChainSigner | None = None,
        sign: bool = True,
        record_reputation: bool = True,
    ) -> InsolvencyResolution:
        """Distribute a poster's proven reserves across its ranked liabilities, signed and audited.

        Builds the :class:`~vincio.settlement.waterfall.InsolvencyResolution`
        (:func:`~vincio.settlement.waterfall.resolve_insolvency`) folding a proven
        :class:`~vincio.settlement.custody.CustodyAttestation` against a proven
        :class:`~vincio.settlement.solvency.LiabilityAttestation` and distributing the reserves
        across the obligations **by seniority then pari-passu within a tranche** (``schedule``),
        pinpointing each creditor's bounded :class:`~vincio.settlement.waterfall.CreditorRecovery`.
        With no ``schedule`` the whole liability set is one pari-passu tranche; pass
        ``completeness`` to distribute against the *completed* liability set, and ``set_off`` (a
        list of mutually-signed :class:`~vincio.settlement.setoff.SetOffStatement`\\ s) to
        **close-out net** each creditor to its net claim before the waterfall. Reuses
        :func:`~vincio.settlement.solvency.prove_solvency`, so a tampered, forged, or wrong-poster
        attestation (or a malformed/wrong-poster schedule, or a one-sided/over-stated set-off) is
        refused; with ``verify_with`` a forged signature is too. Signs the resolution as this book's
        owner, records it on the audit chain, and — unless ``record_reputation`` is off — credits a
        failure against the poster on the bound reputation ledger when the reserves could not make
        every creditor whole. Returns the resolution.
        """
        resolution = resolve_insolvency(
            custody,
            liabilities,
            schedule,
            poster=poster,
            completeness=completeness,
            solvency=solvency,
            set_off=set_off,
            as_of=as_of,
            verifier=verify_with,
        )
        if sign and self.signer is not None:
            resolution.sign(self.signer, party=self.owner)
        self._audit_insolvency(resolution)
        if not resolution.solvent:
            self._record_insolvency_reputation(resolution, record_reputation)
        return resolution

    def _audit_insolvency(self, resolution: InsolvencyResolution) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            INSOLVENCY_ACTION,
            resource=resolution.poster,
            decision=resolution.status,
            details=resolution.audit_details(),
        )
        resolution.audit_id = getattr(entry, "id", None)

    def _record_insolvency_reputation(
        self, resolution: InsolvencyResolution, enabled: bool
    ) -> None:
        if not enabled or self.reputation is None:
            return
        self.reputation.record_outcome(
            resolution.poster,
            passed=False,
            round_id=resolution.id,
            details={
                "kind": "insolvency_resolution",
                "resolution_id": resolution.id,
                "attestor": resolution.attestor,
                "shortfall_usd": resolution.shortfall_usd,
            },
        )

    # -- rehypothecation guard ----------------------------------------------

    def guard_collateral(
        self,
        pools: Any,
        *,
        poster: str | None = None,
        held: float | None = None,
        custody: CustodyAttestation | None = None,
        solvency: SolvencyProof | None = None,
        verify_with: ChainSigner | None = None,
        sign: bool = True,
    ) -> CollateralLedger:
        """Fold a counterparty's collateral pools into a re-use guard, signed and audited.

        Builds the :class:`~vincio.settlement.rehypothecation.CollateralLedger`
        (:func:`~vincio.settlement.rehypothecation.guard_collateral`) reconciling what the
        ``pools`` collectively pledge against the capital the poster holds — solvency-adjusted by
        a ``solvency`` :class:`~vincio.settlement.solvency.SolvencyProof`, proven by a ``custody``
        :class:`~vincio.settlement.custody.CustodyAttestation`, asserted via ``held``, or
        defaulted — pinpointing a contract pledged across more than one pool as a re-use breach,
        bounding each beneficiary's claim to its deterministic share, and surfacing an
        under-reserved breach when the held capital falls below the pledges. Signs the ledger as
        this book's owner and records the guard on the audit chain. A tampered pool, custody
        attestation, or solvency proof is refused; with ``verify_with`` a forged signature is
        too. Returns the ledger.
        """
        ledger = guard_collateral(
            pools,
            poster=poster,
            held=held,
            custody=custody,
            solvency=solvency,
            verify_with=verify_with,
        )
        if sign and self.signer is not None:
            ledger.sign(self.signer, party=self.owner)
        self._audit_ledger(ledger)
        return ledger

    def _audit_ledger(self, ledger: CollateralLedger) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            REHYPOTHECATION_ACTION,
            resource=ledger.id,
            decision=ledger.status,
            details=ledger.audit_details(),
        )
        ledger.audit_id = getattr(entry, "id", None)

    # -- reads --------------------------------------------------------------

    def counterparties(self) -> list[str]:
        """Every counterparty the book has a settlement with, sorted."""
        seen: set[str] = set()
        for record in self.records:
            seen.add(self._counterparty(record))
        return sorted(seen)

    def records_with(self, counterparty: str) -> list[SettlementRecord]:
        """Every settlement record with a given counterparty, in book order."""
        return [r for r in self.records if self._counterparty(r) == counterparty]

    def record_by_id(self, settlement_id: str) -> SettlementRecord | None:
        return next((r for r in self.records if r.id == settlement_id), None)

    def _counterparty(self, record: SettlementRecord) -> str:
        """The other party to a settlement from this book's owner's point of view."""
        if self.owner == record.buyer:
            return record.seller
        if self.owner == record.seller:
            return record.buyer
        # A neutral book (a coordinator's): attribute to the seller it pays out to.
        return record.seller

    # -- verification -------------------------------------------------------

    def verify(self, verifier: ChainSigner | None = None) -> BookVerification:
        """Recompute the book's hash chain (and, with a verifier, signatures).

        The settlement-integrity check: a record whose reconciliation hash no
        longer recomputes (a tampered economic figure), whose chain link is wrong
        (an inserted, dropped, or reordered record), or — with a ``verifier`` —
        whose signature no longer checks, breaks the chain at ``broken_at``.
        """
        previous = ""
        for record in self.records:
            if record.content_hash != record.compute_hash():
                return BookVerification(
                    intact=False,
                    entries=len(self.records),
                    broken_at=record.seq,
                    reason="reconciliation hash mismatch",
                )
            if record.prev_hash != previous or record.entry_hash != _entry_hash(record):
                return BookVerification(
                    intact=False,
                    entries=len(self.records),
                    broken_at=record.seq,
                    reason="entry chain broken",
                )
            if verifier is not None and record.signatures:
                check = record.verify(verifier, require=[])
                if not check.signatures_ok:
                    return BookVerification(
                        intact=False,
                        entries=len(self.records),
                        broken_at=record.seq,
                        reason="signature mismatch",
                    )
            previous = record.entry_hash
        if self.head_hash != previous:
            return BookVerification(
                intact=False,
                entries=len(self.records),
                reason="head hash does not match chain",
            )
        return BookVerification(intact=True, entries=len(self.records))

    def require_intact(self, verifier: ChainSigner | None = None) -> SettlementBook:
        """Verify and raise :class:`SettlementError` if the book is not intact."""
        result = self.verify(verifier)
        if not result.intact:
            raise SettlementError(
                f"settlement book {self.id} failed verification: {result.reason}",
                details={"book_id": self.id, "broken_at": result.broken_at},
            )
        return self

    def reconcile_with(self, counterparty_record: SettlementRecord) -> Reconciliation:
        """Tie a counterparty's record out against this book's own for it.

        Looks up this book's settlement for the same contract and reconciles the
        two (:func:`reconcile`), so two orgs confirm their books agree on the
        delivered figures and the balance. Raises :class:`SettlementError` if this
        book has no settlement for that contract to reconcile against.
        """
        ours = next(
            (r for r in self.records if r.contract_id == counterparty_record.contract_id),
            None,
        )
        if ours is None:
            raise SettlementError(
                f"book {self.id} has no settlement for contract "
                f"{counterparty_record.contract_id!r} to reconcile against",
                details={"book_id": self.id, "contract_id": counterparty_record.contract_id},
            )
        return reconcile(ours, counterparty_record)

    def net(self, *, sign: bool = True) -> Any:
        """Net this book's own records into the owner's cleared positions.

        The single-org view of :func:`~vincio.settlement.netting.net_settlements`:
        an org that is a buyer to some counterparties and a seller to others folds
        its whole book into one :class:`~vincio.settlement.netting.NettingSet` — its
        net position against each counterparty and the minimal cleared set of
        transfers. Signs the set as this book's owner when a signer is attached, so
        a cleared balance is offline-verifiable the way a record is.
        """
        from .netting import net_settlements

        netting = net_settlements(self.records, owner=self.owner)
        if sign and self.signer is not None:
            netting.sign(self.signer, party=self.owner)
        return netting

    def arbitrate(
        self,
        *counterparty_records: SettlementRecord,
        contract_id: str | None = None,
        sign: bool = True,
        verify_with: ChainSigner | None = None,
    ) -> Any:
        """Adjudicate a dispute between this book's record and a counterparty's claims.

        The dispute counterpart of :meth:`reconcile_with`: combines this book's own
        record(s) for the disputed contract with the ``counterparty_records`` the
        other side submits and resolves them with
        :func:`~vincio.settlement.arbitration.arbitrate` — so an org settles which
        figure stands from the signed records alone. ``contract_id`` selects the
        disputed contract (inferred from the counterparty's records when omitted);
        ``verify_with`` authenticates the submitted signatures. Signs the resulting
        :class:`~vincio.settlement.arbitration.Resolution` as this book's owner when a
        signer is attached, so a settled dispute is offline-verifiable the way a
        record is.
        """
        from .arbitration import arbitrate

        target = contract_id
        if target is None:
            ids = sorted({r.contract_id for r in counterparty_records})
            if len(ids) == 1:
                target = ids[0]
        pool: list[SettlementRecord] = list(counterparty_records)
        if target is not None:
            pool += [r for r in self.records if r.contract_id == target]
        else:
            pool += list(self.records)
        resolution = arbitrate(
            pool, contract_id=target, arbiter=self.owner, verify_with=verify_with
        )
        if sign and self.signer is not None:
            resolution.sign(self.signer, party=self.owner)
        return resolution

    def attest(
        self,
        subject: str,
        *,
        resolutions: Any = None,
        config: Any = None,
        sign: bool = True,
        verify_with: ChainSigner | None = None,
        horizon_days: float | None = None,
        note: str = "",
    ) -> Any:
        """Issue a portable attestation of a counterparty's earned standing.

        Reads this book's own signed records (and any arbitration ``resolutions``)
        for ``subject`` and summarizes how its delivery fared — fulfilled settlements
        as successes, breaches and arbitration dissents as failures — into a signed,
        offline-verifiable :class:`~vincio.settlement.attestation.ReputationAttestation`
        (:func:`~vincio.settlement.attestation.attest_reputation`). A prospective
        counterparty verifies it from the bytes alone and folds several issuers'
        attestations into a bounded prior that weights the next negotiation.
        ``horizon_days`` optionally declares a validity window after which an
        as-of-aware combination treats the attestation as stale. Signs the attestation
        as this book's owner (the issuer) when a signer is attached, so an attested
        standing is offline-verifiable the way a record is. Raises
        :class:`SettlementError` when this book has no admissible history with the
        subject to attest.
        """
        from .attestation import attest_reputation

        attestation = attest_reputation(
            self.records,
            subject,
            issuer=self.owner,
            resolutions=resolutions,
            config=config,
            verify_with=verify_with,
            horizon_days=horizon_days,
            note=note,
        )
        if sign and self.signer is not None:
            attestation.sign(self.signer, party=self.owner)
        return attestation

    def revoke(
        self,
        attestation: Any,
        *,
        replacement: Any = None,
        reason: str = "",
        sign: bool = True,
    ) -> Any:
        """Withdraw a prior attestation this book issued, by its hash.

        Builds a signed, offline-verifiable
        :class:`~vincio.settlement.attestation.AttestationRevocation`
        (:func:`~vincio.settlement.attestation.revoke_attestation`) that supersedes or
        withdraws ``attestation`` — which this book must have issued, since an issuer
        can revoke only its own claim. ``replacement`` optionally names the attestation
        that supersedes it. A prospective counterparty folds the revocation into a
        combination so the withdrawn claim is excluded, pinpointed, never silently
        honored. Signs the revocation as this book's owner (the issuer) when a signer
        is attached. Raises :class:`SettlementError` when the attestation was not
        issued by this book.
        """
        from .attestation import revoke_attestation

        if getattr(attestation, "issuer", self.owner) != self.owner:
            raise SettlementError(
                f"{self.owner!r} cannot revoke an attestation issued by "
                f"{getattr(attestation, 'issuer', None)!r}",
                details={"owner": self.owner, "issuer": getattr(attestation, "issuer", None)},
            )
        revocation = revoke_attestation(
            attestation, issuer=self.owner, replacement=replacement, reason=reason
        )
        if sign and self.signer is not None:
            revocation.sign(self.signer, party=self.owner)
        return revocation

    # -- reporting ----------------------------------------------------------

    def report(self, counterparty: str | None = None) -> SettlementReport:
        """Per-counterparty settlement roll-up — beside the cost report."""
        parties = [counterparty] if counterparty is not None else self.counterparties()
        rows: list[SettlementRow] = []
        for party in parties:
            recs = self.records_with(party)
            if not recs:
                rows.append(SettlementRow(counterparty=party))
                continue
            owed = round(sum(r.amount_owed_usd for r in recs), 9)
            delivered = round(sum(r.delivered_cost_usd or 0.0 for r in recs), 9)
            rows.append(
                SettlementRow(
                    counterparty=party,
                    settlements=len(recs),
                    settled=sum(1 for r in recs if r.status == "settled"),
                    breached=sum(1 for r in recs if r.status == "breached"),
                    total_owed_usd=owed,
                    total_delivered_usd=delivered,
                    net_balance_usd=round(sum(r.balance_usd for r in recs), 9),
                )
            )
        return SettlementReport(owner=self.owner, rows=rows)

    # -- persistence --------------------------------------------------------

    def to_record(self) -> dict[str, Any]:
        """A JSON-safe projection for the metadata store (keyed by ``id``)."""
        return to_jsonable(
            {
                "id": self.id,
                "owner": self.owner,
                "head_hash": self.head_hash,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "records": [r.model_dump(mode="json") for r in self.records],
            }
        )

    def load_record(self, data: dict[str, Any]) -> SettlementBook:
        """Populate this book from a persisted projection (replaces its state)."""
        self.id = data.get("id", self.id)
        self.head_hash = data.get("head_hash", "")
        self.records = [SettlementRecord.model_validate(r) for r in data.get("records", [])]
        created = data.get("created_at")
        if isinstance(created, str):
            self.created_at = datetime.fromisoformat(created)
        return self

    def _load(self) -> None:
        if self.store is None:
            return
        try:
            data = self.store.get(SETTLEMENT_STORE_KIND, self.id)
        except Exception:
            note_suppressed("settlement.book.load")
            return
        if data:
            self.load_record(data)

    def _checkpoint(self) -> None:
        if self.store is None:
            return
        try:
            self.store.save(SETTLEMENT_STORE_KIND, self.to_record())
        except Exception:
            note_suppressed("settlement.book.persist")
            return

    def _audit(self, record: SettlementRecord) -> None:
        if self.audit is None:
            return
        entry = self.audit.record(
            SETTLEMENT_ACTION,
            resource=record.contract_id,
            decision=record.status,
            details=record.audit_details(),
        )
        record.audit_id = getattr(entry, "id", None)

    def _emit(self, record: SettlementRecord) -> None:
        if self.events is None:
            return
        try:
            self.events.emit(
                "settlement.recorded",
                {
                    "settlement_id": record.id,
                    "contract_id": record.contract_id,
                    "seller": record.seller,
                    "status": record.status,
                    "balance_usd": record.balance_usd,
                },
            )
        except Exception:
            note_suppressed("settlement.book.event_delivery")

    def _record_reputation(self, record: SettlementRecord, enabled: bool) -> None:
        if not enabled or self.reputation is None:
            return
        self.reputation.record_outcome(
            record.seller,
            passed=record.fulfilled,
            round_id=record.contract_id,
            details={
                "kind": "settlement",
                "settlement_id": record.id,
                "balance_usd": record.balance_usd,
                "breaches": record.breaches,
            },
        )
