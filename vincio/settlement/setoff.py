"""Cross-org insolvency set-off & close-out netting.

The insolvency waterfall (:func:`~vincio.settlement.waterfall.resolve_insolvency`) distributes a
poster's proven reserves across the creditors it owes — but a creditor is often *also* a debtor of
the same counterparty across a web of contracts, and the waterfall pays it on its **gross** claim
while it still owes the insolvent estate the other side. Real insolvency law resolves this first
with **set-off** (close-out netting): mutual obligations between the same two parties collapse to a
single net claim *before* any distribution, so a creditor that owes more than it is owed is not paid
at all, and one owed more recovers only its *net* position. The fabric already nets bilateral
*settlements* multilaterally (:func:`~vincio.settlement.netting.net_settlements`); the liability side
needs the same applied *before* the waterfall. This module is that reach.

* **Signed set-off statement.** A :class:`SetOffStatement` (:func:`build_set_off_statement`) is a
  content-bound statement of the obligations running *both ways* between a ``poster`` and one
  ``creditor`` — ``owed_usd`` the poster owes the creditor, ``owing_usd`` the creditor owes the
  poster back — collapsed to the poster's bounded **net** liability (``max(0, owed − owing)``). It
  is signed with the same :class:`~vincio.security.audit.ChainSigner` a contract uses, by **both**
  parties (a mutually-agreed close-out), so the netting is itself an auditable, non-repudiable
  artifact rather than one side's assertion. :meth:`SetOffStatement.verify` recomputes the content
  hash and re-derives the net from the two gross figures — an over-stated set-off or a tampered net
  is caught from the bytes alone, even after re-sealing — and (with a ``verifier``)
  ``require_mutual`` refuses a one-sided claim only one party signed.
* **Close-out netting into the waterfall.** :func:`~vincio.settlement.waterfall.resolve_insolvency`
  takes ``set_off=`` — a list of statements for the poster — and reduces each creditor's proven
  liability to its **net** claim *before* distributing the reserves: a creditor in debit (it owes
  the estate more than it is owed) recovers nothing, and the estate's distributable claims shrink to
  the true net exposure. The reduction reads only signed, content-bound statements, reconciles each
  against the attested gross it nets (an over-stated set-off — one claiming a different gross than
  the attestation — is refused), and is bound into the resolution by hash so
  :meth:`~vincio.settlement.waterfall.InsolvencyResolution.verify` re-derives every net claim from
  the recorded gross and the applied set-off.
* **Auditable & offline.** The statements read only signed, content-bound artifacts — the poster's
  :class:`~vincio.settlement.solvency.LiabilityAttestation` (the obligations the poster owes) and the
  fabric's :class:`~vincio.settlement.record.SettlementRecord`\\ s (what the creditor owes the
  poster back) — and assert nothing they cannot recompute. :func:`set_off_from_records` derives a
  statement straight from those artifacts (a tampered record refused, a forged signature too with a
  verifier). The :class:`~vincio.settlement.book.SettlementBook` /
  :class:`~vincio.core.app.ContextApp` path signs each statement and lands it on the hash-chained
  audit log. Never a hosted clearing house, a bankruptcy court, or a trusted third party — a
  mechanical, reconstructable close-out over the obligations the fabric already attests.

Everything is dependency-free, deterministic, and offline.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .record import SettlementRecord, SettlementSignature

if TYPE_CHECKING:
    from ..security.audit import ChainSigner
    from .solvency import LiabilityAttestation

__all__ = [
    "SetOffStatement",
    "SetOffVerification",
    "build_set_off_statement",
    "set_off_from_records",
]

# The audit action a set-off statement is recorded under; the decision field carries the net
# direction (``poster_owes`` when the poster still owes net, ``creditor_in_debit`` when the
# creditor owes the estate more than it is owed, ``eliminated`` when the two sides cancel).
SETOFF_ACTION = "liability_set_off"

_TOLERANCE = 1e-6


def _r6(value: float) -> float:
    """Round a money figure to six places so float drift never breaks a net or its hash."""
    return round(float(value), 6)


class SetOffVerification(BaseModel):
    """The (non-raising) outcome of verifying a set-off statement offline.

    A statement is **valid** when its content hash recomputes (``hash_ok``), it is well-formed —
    the poster and creditor are distinct named parties and both gross figures are non-negative
    (``well_formed``) — the net re-derives from the two gross figures (``net_sound`` — an
    over-stated set-off or a tampered net is caught), and — with a ``verifier`` — every signature
    checks (``signatures_ok``). With ``require_mutual`` a one-sided statement only one party signed
    is refused.
    """

    valid: bool
    hash_ok: bool
    well_formed: bool
    net_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class SetOffStatement(BaseModel):
    """A signed, offline-verifiable statement of the obligations running both ways.

    Produced by :func:`build_set_off_statement` / :func:`set_off_from_records` (or
    :meth:`~vincio.settlement.book.SettlementBook.build_set_off_statement` /
    :meth:`~vincio.core.app.ContextApp.build_set_off_statement`): a content-bound statement of the
    mutual obligations between a ``poster`` and one ``creditor`` — ``owed_usd`` the poster owes the
    creditor (the liability side), ``owing_usd`` the creditor owes the poster back — collapsed to
    the poster's bounded **net** liability ``net_usd = max(0, owed − owing)``. It binds the parties,
    the two gross figures, the references, and the instant onto a content hash, so the close-out is
    a mechanical number anyone recomputes.

    A statement is signed by **both** parties (a mutually-agreed close-out): the signature records
    *who agreed to the netting*, and ``require_mutual`` refuses a one-sided claim. :meth:`verify`
    recomputes the hash and re-derives the net from the two gross figures, so an over-stated set-off
    (a tampered ``owing_usd`` inflating what the creditor is said to owe back, wiping out its
    recovery) or a tampered net is caught from the bytes alone, even after re-sealing.
    :func:`~vincio.settlement.waterfall.resolve_insolvency` reduces the creditor's proven liability
    to :attr:`poster_net_claim_usd` *before* distributing the reserves.
    """

    id: str = Field(default_factory=lambda: new_id("setoff"))
    poster: str
    creditor: str
    owed_usd: float = 0.0
    owing_usd: float = 0.0
    net_debtor: str = ""
    net_creditor: str = ""
    net_usd: float = 0.0
    references: list[str] = Field(default_factory=list)

    as_of: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def poster_net_claim_usd(self) -> float:
        """The poster's bounded net liability to the creditor (``max(0, owed − owing)``).

        The claim that survives close-out into the waterfall: what the poster still owes the
        creditor once the creditor's own obligation back is set off, floored at zero. ``0`` when
        the creditor owes the estate at least as much as it is owed (a creditor in debit recovers
        nothing).
        """
        return _r6(max(0.0, self.owed_usd - self.owing_usd))

    @property
    def set_off_usd(self) -> float:
        """The portion of the poster's gross liability the creditor's obligation cancels.

        ``min(owed, owing)`` — the amount netted out of the creditor's gross claim. The waterfall
        records this so :meth:`~vincio.settlement.waterfall.InsolvencyResolution.verify` re-derives
        the net claim from the gross.
        """
        return _r6(min(self.owed_usd, self.owing_usd))

    @property
    def creditor_in_debit(self) -> bool:
        """Whether the creditor owes the estate more than it is owed (it recovers nothing)."""
        return self.owing_usd > self.owed_usd + _TOLERANCE

    @property
    def eliminated(self) -> bool:
        """Whether the two sides cancel exactly (the net liability is zero)."""
        return self.poster_net_claim_usd <= _TOLERANCE

    @property
    def direction(self) -> str:
        """``poster_owes`` / ``creditor_in_debit`` / ``eliminated`` — the net direction."""
        if self.creditor_in_debit:
            return "creditor_in_debit"
        if self.eliminated:
            return "eliminated"
        return "poster_owes"

    @property
    def mutual(self) -> bool:
        """Whether both the poster and the creditor have signed the close-out."""
        signers = set(self.signed_by)
        return self.poster in signers and self.creditor in signers

    def _expected_net(self) -> tuple[str, str, float]:
        """The ``(net_debtor, net_creditor, net_amount)`` the two gross figures imply."""
        diff = _r6(self.owed_usd - self.owing_usd)
        if diff > _TOLERANCE:
            return self.poster, self.creditor, diff
        if diff < -_TOLERANCE:
            return self.creditor, self.poster, _r6(-diff)
        return "", "", 0.0

    # -- hashing ------------------------------------------------------------

    def setoff_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the parties, the gross figures, and the instant.

        Excludes the id, signatures, and audit linkage (local metadata, not the close-out), so the
        same mutual obligations as of the same instant hash identically wherever recomputed.
        References are sorted so the order they were listed in never changes the hash. The net is
        bound too, so a tampered net that no longer re-derives is caught even after re-sealing.
        """
        return {
            "poster": self.poster,
            "creditor": self.creditor,
            "owed_usd": _r6(self.owed_usd),
            "owing_usd": _r6(self.owing_usd),
            "net_debtor": self.net_debtor,
            "net_creditor": self.net_creditor,
            "net_usd": _r6(self.net_usd),
            "references": sorted(self.references),
            "as_of": self.as_of.isoformat(),
        }

    def compute_hash(self) -> str:
        """The content hash binding the parties, the gross figures, and the net."""
        return stable_hash(self.setoff_facts(), length=32)

    def seal(self) -> SetOffStatement:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> SetOffStatement:
        """Add ``party``'s signature over the content hash (sealing first).

        A set-off statement is signed by **both** the poster and the creditor — each agreeing to
        the netting it commits. Re-signing for the same party replaces its prior signature, so a
        statement cannot accumulate stale signatures for one identity.
        """
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

    # -- verification -------------------------------------------------------

    def _well_formed(self) -> bool:
        """Distinct named parties and non-negative gross figures."""
        if not self.poster or not self.creditor or self.poster == self.creditor:
            return False
        return self.owed_usd >= -_TOLERANCE and self.owing_usd >= -_TOLERANCE

    def _net_sound(self) -> bool:
        """The recorded net re-derives from the two gross figures."""
        debtor, creditor, amount = self._expected_net()
        return (
            self.net_debtor == debtor
            and self.net_creditor == creditor
            and abs(self.net_usd - amount) <= _TOLERANCE
        )

    def verify(
        self,
        verifier: ChainSigner | None = None,
        *,
        require: list[str] | None = None,
        require_mutual: bool = False,
    ) -> SetOffVerification:
        """Verify the statement offline: the hash recomputes and the net re-derives.

        Recomputes the content hash, checks the statement is well-formed (distinct named parties,
        non-negative gross figures), and re-derives the net from the two gross figures — so an
        over-stated set-off or a tampered net is caught even when the hash was recomputed to match.
        ``verifier`` additionally checks each signature against the content hash; ``require`` names
        parties that must have a verified signature, and ``require_mutual`` shorthand requires both
        the poster and the creditor (a one-sided close-out is refused).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        well_formed = self._well_formed()
        net_sound = self._net_sound()
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
        required = list(require or [])
        if require_mutual:
            required.extend([self.poster, self.creditor])
        missing = [p for p in required if p not in verified]
        if missing:
            signatures_ok = False
        valid = hash_ok and well_formed and net_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "statement is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the statement facts"
            elif not well_formed:
                reason = "statement is malformed (same party on both sides, or a negative figure)"
            elif not net_sound:
                reason = "the net does not re-derive from the gross figures"
            elif missing:
                reason = f"missing/invalid signatures for {sorted(set(missing))}"
            else:
                reason = "signature mismatch"
        return SetOffVerification(
            valid=valid,
            hash_ok=hash_ok,
            well_formed=well_formed,
            net_sound=net_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self,
        verifier: ChainSigner | None = None,
        *,
        require: list[str] | None = None,
        require_mutual: bool = False,
    ) -> SetOffStatement:
        """Verify and raise :class:`SettlementError` if the statement is not valid."""
        result = self.verify(verifier, require=require, require_mutual=require_mutual)
        if not result.valid:
            raise SettlementError(
                f"set-off statement {self.id} failed verification: {result.reason}",
                details={"setoff_id": self.id, "reason": result.reason},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the statement for the audit chain."""
        return to_jsonable(
            {
                "setoff_id": self.id,
                "poster": self.poster,
                "creditor": self.creditor,
                "owed_usd": _r6(self.owed_usd),
                "owing_usd": _r6(self.owing_usd),
                "net_usd": _r6(self.net_usd),
                "net_debtor": self.net_debtor,
                "net_creditor": self.net_creditor,
                "direction": self.direction,
                "references": sorted(self.references),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> SetOffStatement:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the mutual obligations and the resulting net liability."""
        print(
            f"Set-off ({self.poster} ↔ {self.creditor}): owed ${self.owed_usd:,.2f}, "
            f"owing ${self.owing_usd:,.2f} → {self.direction}"
        )
        if self.net_debtor:
            print(f"  {self.net_debtor} owes {self.net_creditor} ${self.net_usd:,.2f} net")
        else:
            print("  the two sides cancel exactly")


def build_set_off_statement(
    poster: str,
    creditor: str,
    owed_usd: float,
    owing_usd: float,
    *,
    references: Iterable[str] | None = None,
    as_of: datetime | None = None,
) -> SetOffStatement:
    """Collapse the mutual obligations between a poster and one creditor into a statement.

    The close-out analogue of :func:`~vincio.settlement.waterfall.build_seniority_schedule`: it
    states the obligations running *both ways* between ``poster`` and ``creditor`` — ``owed_usd``
    the poster owes the creditor, ``owing_usd`` the creditor owes the poster back — and computes the
    poster's bounded net liability (``max(0, owed − owing)``). ``references`` is free-text
    provenance for the figures (e.g. the contract ids that fed ``owing_usd``).

    Returns a sealed, unsigned :class:`SetOffStatement` — sign it with **both** parties' keys (a
    mutually-agreed close-out), or let a :class:`~vincio.settlement.book.SettlementBook` sign it as
    this book's owner. Raises :class:`SettlementError` when the parties are not distinct named
    parties or a figure is negative.
    """
    statement = SetOffStatement(
        poster=poster,
        creditor=creditor,
        owed_usd=_r6(owed_usd),
        owing_usd=_r6(owing_usd),
        references=sorted(references or []),
        as_of=as_of or utcnow(),
    )
    debtor, net_creditor, amount = statement._expected_net()
    statement.net_debtor = debtor
    statement.net_creditor = net_creditor
    statement.net_usd = amount
    statement.seal()
    if not statement._well_formed():
        raise SettlementError(
            f"set-off statement for {poster!r}/{creditor!r} is malformed: the poster and creditor "
            "must be distinct named parties and both gross figures must be non-negative",
            details={"poster": poster, "creditor": creditor},
        )
    return statement


def _owing_from_records(
    poster: str,
    creditor: str,
    records: Iterable[SettlementRecord],
    *,
    verifier: ChainSigner | None,
) -> tuple[float, list[str]]:
    """Sum what ``creditor`` owes ``poster`` across the settlement records, deduped by contract.

    The reverse direction of a close-out: in a settlement record the buyer owes the seller, so the
    obligations the ``creditor`` owes the ``poster`` are the records where the poster is the seller
    and the creditor the buyer. The same bilateral settlement appears in both books co-signing one
    reconciliation hash, so records are deduped by content hash. A tampered record (its hash no
    longer recomputes) is refused; with a ``verifier`` a forged signature is too.
    """
    owing = 0.0
    contracts: set[str] = set()
    seen: set[str] = set()
    for record in records:
        if record.seller != poster or record.buyer != creditor:
            continue
        if record.content_hash != record.compute_hash():
            raise SettlementError(
                f"settlement {record.id} for contract {record.contract_id!r} is tampered "
                "(reconciliation hash does not recompute); refusing to set it off",
                details={"settlement_id": record.id, "contract_id": record.contract_id},
            )
        if verifier is not None and record.signatures:
            if not record.verify(verifier, require=[]).signatures_ok:
                raise SettlementError(
                    f"settlement {record.id} for contract {record.contract_id!r} has an invalid "
                    "signature; refusing to set it off",
                    details={"settlement_id": record.id, "contract_id": record.contract_id},
                )
        if record.content_hash in seen:
            continue
        seen.add(record.content_hash)
        amount = _r6(record.amount_owed_usd)
        if amount <= 0:
            continue
        owing = _r6(owing + amount)
        contracts.add(record.contract_id)
    return owing, sorted(contracts)


def set_off_from_records(
    poster: str,
    creditor: str,
    liabilities: LiabilityAttestation,
    records: Iterable[SettlementRecord],
    *,
    as_of: datetime | None = None,
    verifier: ChainSigner | None = None,
) -> SetOffStatement:
    """Derive a set-off statement straight from the existing signed, content-bound artifacts.

    Reads only what the fabric already attests: the poster's
    :class:`~vincio.settlement.solvency.LiabilityAttestation` gives ``owed_usd`` (the obligations the
    ``poster`` owes the ``creditor``, summed across its line items), and the
    :class:`~vincio.settlement.record.SettlementRecord`\\ s give ``owing_usd`` (what the creditor
    owes the poster back — the records where the poster is the seller and the creditor the buyer,
    deduped by reconciliation hash). A tampered liability attestation or settlement record is
    refused; with a ``verifier`` a forged signature is too.

    Returns a sealed, unsigned :class:`SetOffStatement` — sign it with both parties' keys. Raises
    :class:`SettlementError` when the attestation is for a different poster, or an artifact is
    tampered or forged.
    """
    if liabilities.poster != poster:
        raise SettlementError(
            f"liability attestation {liabilities.id} is for {liabilities.poster!r}, not the poster "
            f"{poster!r} the set-off is about; refusing it",
            details={"attestation_id": liabilities.id, "poster": poster},
        )
    if liabilities.content_hash != liabilities.compute_hash():
        raise SettlementError(
            f"liability attestation {liabilities.id} is tampered (content hash does not recompute); "
            "refusing to set it off",
            details={"attestation_id": liabilities.id},
        )
    if verifier is not None and liabilities.signatures:
        if not liabilities.verify(verifier).signatures_ok:
            raise SettlementError(
                f"liability attestation {liabilities.id} has an invalid signature; refusing to set "
                "it off",
                details={"attestation_id": liabilities.id},
            )
    owed = _r6(
        sum(line.amount_usd for line in liabilities.liabilities if line.creditor == creditor)
    )
    owing, contracts = _owing_from_records(poster, creditor, records, verifier=verifier)
    return build_set_off_statement(poster, creditor, owed, owing, references=contracts, as_of=as_of)
