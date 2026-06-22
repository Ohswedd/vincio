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

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .meter import Meter, MeterReading
from .record import (
    SETTLEMENT_ACTION,
    Reconciliation,
    SettlementLine,
    SettlementRecord,
    reconcile,
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
    """Per-counterparty settlement roll-up — alongside the cost report.

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
    ) -> SettlementRecord:
        """Reconcile delivery against a contract and close the books on it.

        Builds the reconciled :class:`~vincio.settlement.record.SettlementRecord`,
        signs it as ``party`` (defaulting to whichever side this book's owner is on
        the contract), links it into the book's hash chain, records the verdict on
        the audit chain, and — unless ``record_reputation`` is off — credits the
        seller on fulfilment or debits it on a breach, so reliability earned in
        delivery weights the next negotiation. Returns the appended record.
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

    def _resolve_party(self, party: str | None, record: SettlementRecord) -> str | None:
        if party is not None:
            return party
        if self.owner in (record.buyer, record.seller):
            return self.owner
        return None

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

    def reconcile_with(
        self, counterparty_record: SettlementRecord
    ) -> Reconciliation:
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
            delivered = round(
                sum(r.delivered_cost_usd or 0.0 for r in recs), 9
            )
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
        self.records = [
            SettlementRecord.model_validate(r) for r in data.get("records", [])
        ]
        created = data.get("created_at")
        if isinstance(created, str):
            self.created_at = datetime.fromisoformat(created)
        return self

    def _load(self) -> None:
        if self.store is None:
            return
        try:
            data = self.store.get(SETTLEMENT_STORE_KIND, self.id)
        except Exception:  # noqa: BLE001 - a store without the kind is simply empty
            return
        if data:
            self.load_record(data)

    def _checkpoint(self) -> None:
        if self.store is None:
            return
        try:
            self.store.save(SETTLEMENT_STORE_KIND, self.to_record())
        except Exception:  # noqa: BLE001 - persistence is best-effort
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
        except Exception:  # noqa: BLE001 - event delivery is best-effort
            pass

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
