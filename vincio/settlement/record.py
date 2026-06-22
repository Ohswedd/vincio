"""The typed, signed, offline-verifiable settlement record.

A :class:`SettlementRecord` closes the books on work delivered under a
:class:`~vincio.negotiation.Contract`: it reconciles the metered delivery (a
:class:`~vincio.settlement.meter.MeterReading`) against the agreed price / SLA /
quality, lining up what was owed beside what was delivered, and binds the result
into a content hash both parties sign. It is the settlement analogue of the
contract itself — the same dependency-free, deterministic guarantees:

* **Typed reconciliation.** A :class:`SettlementLine` per dimension (price, SLA,
  quality) records owed vs delivered and whether the term held, and the record
  carries the net :attr:`SettlementRecord.balance_usd` (the agreed price minus the
  delivered cost — a credit when delivery came in under, an overrun when it went
  over), so a settlement is debuggable, not just a boolean.
* **Signed & offline-verifiable.** The reconciliation hash binds the economic
  facts — contract, parties, agreed terms, delivered metrics, balance — and both
  parties sign *that* hash with the same :class:`~vincio.security.audit.ChainSigner`
  the audit chain uses. :meth:`SettlementRecord.verify` recomputes it from the
  stored fields, so a tampered figure or a forged signature is caught from the
  bytes alone, without the live parties.
* **Reconciled across the boundary.** Because both sides compute the *same*
  deterministic reconciliation hash from the same contract and delivery, two
  independently-produced records co-sign one hash when the books agree;
  :func:`reconcile` compares two records' economic facts and flags any discrepancy
  as a dispute — the cross-org analogue of two ledgers that must tie out.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "SettlementStatus",
    "SettlementLine",
    "SettlementSignature",
    "SettlementVerification",
    "SettlementRecord",
    "Reconciliation",
    "reconcile",
]

SettlementStatus = Literal["settled", "breached"]

# The audit action a settlement record is recorded under — the single key a
# settlement roll-up reads back from the chain.
SETTLEMENT_ACTION = "settlement"

_TOLERANCE = 1e-6


class SettlementLine(BaseModel):
    """One dimension of a settlement, reconciling what was owed against delivery.

    ``owed`` is the agreed term (price, SLA in ms, or quality floor); ``delivered``
    is the metered figure; ``within`` is whether the term held; ``delta`` is the
    signed slack (positive = within by that margin, negative = over/under by it).
    A line with no delivered figure (the dimension was not metered) is recorded as
    ``within=True`` with a ``None`` delivered value — an un-metered term is not a
    breach.
    """

    dimension: Literal["price", "sla", "quality"]
    owed: float
    delivered: float | None = None
    within: bool = True
    delta: float | None = None
    note: str = ""


class SettlementSignature(BaseModel):
    """One party's signature over a settlement's reconciliation hash."""

    party: str
    signature: str
    key_id: str = ""


class SettlementVerification(BaseModel):
    """The (non-raising) outcome of verifying a settlement record offline."""

    valid: bool
    hash_ok: bool
    signatures_ok: bool
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class SettlementRecord(BaseModel):
    """A signed, offline-verifiable reconciliation of delivery against a contract.

    Produced by :func:`~vincio.settlement.book.settle_contract` (or
    :meth:`~vincio.settlement.book.SettlementBook.settle`) from a contract and a
    metered delivery. Both parties :meth:`sign` the reconciliation hash;
    :meth:`verify` checks it from the bytes alone; :func:`reconcile` ties two
    parties' records out. When carried in a :class:`~vincio.settlement.book.SettlementBook`
    the engine additionally links it into the book's hash chain (``seq`` /
    ``prev_hash`` / ``entry_hash``) for a tamper-evident ledger, while the
    party signatures stay over the reconciliation hash so both books co-sign the
    same economic facts.
    """

    id: str = Field(default_factory=lambda: new_id("settlement"))
    contract_id: str
    buyer: str
    seller: str
    scope: str = ""
    run_id: str | None = None
    saga_id: str | None = None

    # Agreed terms (copied from the contract at settlement time).
    price_usd: float = 0.0
    sla_seconds: float = 0.0
    quality_floor: float = 0.0

    # Delivered, metered figures reconciled against the terms.
    delivered_cost_usd: float | None = None
    delivered_latency_ms: float | None = None
    delivered_quality: float | None = None
    metered_units: float = 0.0
    metered_events: int = 0

    # The reconciliation.
    lines: list[SettlementLine] = Field(default_factory=list)
    amount_owed_usd: float = 0.0
    balance_usd: float = 0.0
    fulfilled: bool = True
    status: SettlementStatus = "settled"
    breaches: list[str] = Field(default_factory=list)

    settled_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # Book-chain linkage (set by a SettlementBook on append; not part of the
    # reconciliation hash so both parties' books co-sign the same economic facts).
    seq: int | None = None
    prev_hash: str = ""
    entry_hash: str = ""

    def reconciliation_facts(self) -> dict[str, Any]:
        """The economic facts the reconciliation hash binds (and both sides sign)."""
        return {
            "contract_id": self.contract_id,
            "buyer": self.buyer,
            "seller": self.seller,
            "scope": self.scope,
            "price_usd": round(float(self.price_usd), 9),
            "sla_seconds": round(float(self.sla_seconds), 9),
            "quality_floor": round(float(self.quality_floor), 9),
            "delivered_cost_usd": _r(self.delivered_cost_usd),
            "delivered_latency_ms": _r(self.delivered_latency_ms),
            "delivered_quality": _r(self.delivered_quality),
            "metered_units": round(float(self.metered_units), 9),
            "metered_events": self.metered_events,
            "amount_owed_usd": round(float(self.amount_owed_usd), 9),
            "balance_usd": round(float(self.balance_usd), 9),
            "fulfilled": self.fulfilled,
            "status": self.status,
            "breaches": list(self.breaches),
        }

    def compute_hash(self) -> str:
        """The reconciliation hash binding the economic facts (run-id-independent).

        Deliberately excludes book position, run/saga id, the settlement id, and
        the timestamp: those are local metadata, not economic terms. Two parties
        that reconcile the same contract and delivery therefore compute the *same*
        hash and can co-sign it, while a tampered figure changes it.
        """
        return stable_hash(self.reconciliation_facts(), length=32)

    def seal(self) -> SettlementRecord:
        """Stamp the reconciliation hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    @property
    def fully_signed(self) -> bool:
        """Both buyer and seller have signed the reconciliation."""
        signers = set(self.signed_by)
        return self.buyer in signers and self.seller in signers

    @property
    def overrun_usd(self) -> float:
        """How far delivered cost exceeded the agreed price (0 when within)."""
        return round(max(0.0, -self.balance_usd), 9)

    @property
    def credit_usd(self) -> float:
        """How far delivered cost came in under the agreed price (0 when over)."""
        return round(max(0.0, self.balance_usd), 9)

    def sign(self, signer: ChainSigner, *, party: str) -> SettlementRecord:
        """Add ``party``'s signature over the reconciliation hash (sealing first).

        Re-signing for the same party replaces its prior signature, so a record
        cannot accumulate stale signatures for one identity.
        """
        if party not in (self.buyer, self.seller):
            raise SettlementError(
                f"party {party!r} is neither the buyer nor the seller of settlement "
                f"{self.id}",
                details={"settlement_id": self.id, "party": party},
            )
        if not self.content_hash:
            self.seal()
        sig = SettlementSignature(
            party=party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != party]
        self.signatures.append(sig)
        return self

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> SettlementVerification:
        """Verify the record offline: the hash recomputes and signatures check.

        ``verifier`` checks each signature against the reconciliation hash;
        ``require`` names the parties that must have a verified signature (defaults
        to both buyer and seller). Pass ``[]`` to check the hash binding alone.
        """
        expected = self.compute_hash()
        hash_ok = bool(self.content_hash) and self.content_hash == expected
        if not hash_ok:
            return SettlementVerification(
                valid=False, hash_ok=False, signatures_ok=False, reason="content hash mismatch"
            )
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
        required = [self.buyer, self.seller] if require is None else require
        missing = [p for p in required if p not in verified]
        if missing:
            signatures_ok = False
        valid = hash_ok and signatures_ok and (verifier is not None or not required)
        reason = None
        if not signatures_ok:
            reason = (
                f"missing/invalid signatures for {missing}" if missing else "signature mismatch"
            )
        elif verifier is None and required:
            reason = "no verifier supplied — signatures present but not authenticated"
        return SettlementVerification(
            valid=valid,
            hash_ok=hash_ok,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner, *, require: list[str] | None = None
    ) -> SettlementRecord:
        """Verify and raise :class:`SettlementError` if the record is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"settlement {self.id} failed verification: {result.reason}",
                details={"settlement_id": self.id, "reason": result.reason},
            )
        return self

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the settlement for the audit chain."""
        return to_jsonable(
            {
                "settlement_id": self.id,
                "contract_id": self.contract_id,
                "buyer": self.buyer,
                "seller": self.seller,
                "scope": self.scope,
                "run_id": self.run_id,
                "amount_owed_usd": self.amount_owed_usd,
                "balance_usd": self.balance_usd,
                "fulfilled": self.fulfilled,
                "status": self.status,
                "breaches": self.breaches,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> SettlementRecord:
        return cls.model_validate(data)


class Reconciliation(BaseModel):
    """Whether two parties' settlement records tie out — the dispute verdict.

    Produced by :func:`reconcile`. ``agrees`` is true when both records describe
    the same contract, parties, agreed terms, and delivered metrics (within
    tolerance) and therefore share a reconciliation hash; ``discrepancies`` lists
    every field that differs, so a dispute is pinpointed rather than merely flagged.
    """

    agrees: bool
    contract_id: str
    hashes_match: bool = False
    balance_a: float | None = None
    balance_b: float | None = None
    discrepancies: list[str] = Field(default_factory=list)
    reason: str | None = None


def reconcile(
    a: SettlementRecord, b: SettlementRecord, *, tolerance: float = _TOLERANCE
) -> Reconciliation:
    """Tie two independently-produced settlement records out against each other.

    The cross-org reconciliation: the buyer's record and the seller's record must
    describe the same economic outcome. They agree when they reference the same
    contract and parties and their agreed terms, delivered metrics, and balance
    match within ``tolerance`` — in which case they share a reconciliation hash and
    can co-sign it. Any difference is a discrepancy (a dispute), reported field by
    field. Raises :class:`SettlementError` only when the two records are not even
    for the same contract — a category error, not a dispute.
    """
    if a.contract_id != b.contract_id:
        raise SettlementError(
            f"cannot reconcile settlements for different contracts "
            f"({a.contract_id!r} vs {b.contract_id!r})",
            details={"a": a.contract_id, "b": b.contract_id},
        )
    discrepancies: list[str] = []

    def _cmp_str(field: str, x: Any, y: Any) -> None:
        if x != y:
            discrepancies.append(f"{field}: {x!r} != {y!r}")

    def _cmp_num(field: str, x: float | None, y: float | None) -> None:
        if x is None and y is None:
            return
        if x is None or y is None or abs(float(x) - float(y)) > tolerance:
            discrepancies.append(f"{field}: {x} != {y}")

    _cmp_str("buyer", a.buyer, b.buyer)
    _cmp_str("seller", a.seller, b.seller)
    _cmp_str("scope", a.scope, b.scope)
    _cmp_num("price_usd", a.price_usd, b.price_usd)
    _cmp_num("sla_seconds", a.sla_seconds, b.sla_seconds)
    _cmp_num("quality_floor", a.quality_floor, b.quality_floor)
    _cmp_num("delivered_cost_usd", a.delivered_cost_usd, b.delivered_cost_usd)
    _cmp_num("delivered_latency_ms", a.delivered_latency_ms, b.delivered_latency_ms)
    _cmp_num("delivered_quality", a.delivered_quality, b.delivered_quality)
    _cmp_num("balance_usd", a.balance_usd, b.balance_usd)
    _cmp_str("status", a.status, b.status)

    hashes_match = a.compute_hash() == b.compute_hash()
    agrees = not discrepancies and hashes_match
    reason = None
    if not agrees:
        reason = (
            "; ".join(discrepancies)
            if discrepancies
            else "reconciliation hashes differ"
        )
    return Reconciliation(
        agrees=agrees,
        contract_id=a.contract_id,
        hashes_match=hashes_match,
        balance_a=a.balance_usd,
        balance_b=b.balance_usd,
        discrepancies=discrepancies,
        reason=reason,
    )


def _r(value: float | None) -> float | None:
    return None if value is None else round(float(value), 9)
