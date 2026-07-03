"""Cross-org dispute resolution & arbitration.

With settlements netted and a disagreement *pinpointed* as a
:class:`~vincio.settlement.netting.NettingDispute`, the next reach is **resolving**
it: a typed, signed, offline-verifiable adjudication that two orgs run over their
own signed records to settle which figure stands, so a dispute closes the way a
settlement does. It is a library-side protocol — the dispute analogue of the
settlement record — never a hosted arbitration service or a court of record.

* **Typed adjudication.** Each party submits its signed
  :class:`~vincio.settlement.record.SettlementRecord`\\ s for the disputed contract,
  and the deterministic :func:`arbitrate` decides **which records verify and
  reconcile**. The decision rests on nothing it cannot recompute: a reconciliation
  hash that **both** the buyer and the seller signed (each on their own record, the
  two co-signing one hash exactly as :func:`~vincio.settlement.record.reconcile`
  describes) is incontrovertible mutual agreement and **stands**; a unilateral claim
  contradicting it is **rejected** and pinpointed. When neither side's figure is
  corroborated the dispute is honestly left **unresolved** rather than decided by
  fiat.
* **Still offline-verifiable.** The resulting :class:`Resolution` is content-bound
  the way a record is: a resolution hash binds the contract, the parties, the
  outcome, and every adjudicated claim, and the same
  :class:`~vincio.security.audit.ChainSigner` co-signs *that* hash.
  :meth:`Resolution.verify` recomputes the whole adjudication from the bytes alone —
  the hash matches and the decision re-derives from the recorded claims — so a
  settled dispute verifies without the arbiter, and a rejected claim is pinpointed
  rather than merely overruled.
* **Same discipline.** Arbitration reads only the existing signed records and
  asserts nothing it cannot recompute. Unlike netting, which *refuses* to clear over
  a tampered book, arbitration is the venue where a bad claim is adjudicated: a
  tampered or forged claim is marked **inadmissible** and pinpointed, never silently
  dropped and never crashing the resolution. A settled dispute also **closes the
  reputation loop** on the party whose claim did not stand.

:func:`arbitrate` resolves one disputed contract from a pool of submitted records;
:meth:`~vincio.settlement.book.SettlementBook.arbitrate` resolves an org's own
record against a counterparty's submitted claims. Everything is dependency-free,
deterministic, and offline.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .record import SettlementRecord, SettlementSignature, _resolve_verifier

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "ResolutionStatus",
    "ClaimVerdict",
    "ResolutionVerification",
    "Resolution",
    "arbitrate",
]

# The audit action a resolution is recorded under — the single key a dispute
# roll-up reads back from the chain.
ARBITRATION_ACTION = "arbitration"

# An adjudication either upholds a single corroborated record or, when no figure is
# corroborated, honestly leaves the dispute unresolved.
ResolutionStatus = Literal["upheld", "unresolved"]

_TOLERANCE = 1e-6


def _r(value: float | None) -> float | None:
    return None if value is None else round(float(value), 9)


class ClaimVerdict(BaseModel):
    """One submitted record's standing in the adjudication — pinpointed, not summed.

    A claim is **admissible** when its reconciliation hash recomputes and (with a
    verifier) its signatures check; a tampered or forged claim is inadmissible and
    its ``reason`` says why. Among the admissible claims, the one(s) carrying the
    corroborated reconciliation hash **stand**; a claim that contradicts the
    corroborated figure is admissible but does not stand, and ``reason`` pinpoints
    that too. ``signed_by`` is the parties whose signatures the claim carries — the
    corroboration the decision is recomputed from.
    """

    settlement_id: str
    contract_id: str
    reconciliation_hash: str
    signed_by: list[str] = Field(default_factory=list)
    balance_usd: float | None = None
    settlement_status: str = ""
    admissible: bool = True
    stands: bool = False
    reason: str | None = None


class ResolutionVerification(BaseModel):
    """The (non-raising) outcome of verifying a resolution offline."""

    valid: bool
    hash_ok: bool
    decision_sound: bool
    signatures_ok: bool
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class Resolution(BaseModel):
    """A content-bound, offline-verifiable adjudication of a disputed contract.

    Produced by :func:`arbitrate` (or
    :meth:`~vincio.settlement.book.SettlementBook.arbitrate`) from the signed
    settlement records the disputing parties submit. It carries every submitted
    record's :class:`ClaimVerdict`, the :attr:`status` of the dispute, and — when a
    figure was corroborated — the :attr:`upheld_hash` that stands and its
    :attr:`upheld_balance_usd`.

    The resolution hash (:meth:`compute_hash`) binds the contract, the parties, the
    outcome, and every claim verdict, so :meth:`verify` recomputes the whole
    adjudication from the bytes alone: the hash matches and the decision re-derives
    from the recorded claims (the same corroborated figure stands, the same claims
    are rejected). A :class:`~vincio.security.audit.ChainSigner` co-signs that hash,
    exactly as a settlement record's parties co-sign theirs — and because the hash
    excludes the arbiter, two arbiters that read the same submitted records compute
    the *same* hash and can co-sign it.
    """

    id: str = Field(default_factory=lambda: new_id("resolution"))
    contract_id: str
    buyer: str = ""
    seller: str = ""
    arbiter: str = ""

    status: ResolutionStatus = "unresolved"
    upheld_hash: str = ""
    upheld_balance_usd: float | None = None
    upheld_status: str = ""

    claims: list[ClaimVerdict] = Field(default_factory=list)
    corroborated_by: list[str] = Field(default_factory=list)
    source_hashes: list[str] = Field(default_factory=list)
    reason: str | None = None

    resolved_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived figures ----------------------------------------------------

    @property
    def upheld(self) -> bool:
        """A corroborated figure stands — the dispute closed with a winner."""
        return self.status == "upheld"

    @property
    def resolved(self) -> bool:
        """The dispute reached an outcome (a figure stands)."""
        return self.status == "upheld"

    @property
    def standing_claims(self) -> list[ClaimVerdict]:
        """The claims whose reconciliation hash stands."""
        return [c for c in self.claims if c.stands]

    @property
    def rejected_claims(self) -> list[ClaimVerdict]:
        """The admissible claims that contradicted the corroborated figure."""
        return [c for c in self.claims if c.admissible and not c.stands]

    @property
    def inadmissible_claims(self) -> list[ClaimVerdict]:
        """The submitted records that did not even verify (tampered or forged)."""
        return [c for c in self.claims if not c.admissible]

    @property
    def dissenters(self) -> list[str]:
        """The parties whose admissible claim did not stand — the reputation loss."""
        if not self.upheld:
            return []
        out: list[str] = []
        for c in self.rejected_claims:
            for party in c.signed_by:
                if party in (self.buyer, self.seller) and party not in out:
                    out.append(party)
        return sorted(out)

    def claim_for(self, settlement_id: str) -> ClaimVerdict | None:
        """The verdict for a submitted record by id, or ``None``."""
        return next((c for c in self.claims if c.settlement_id == settlement_id), None)

    # -- hashing & signing --------------------------------------------------

    def resolution_facts(self) -> dict[str, Any]:
        """The facts the resolution hash binds (and a signer signs).

        Deliberately excludes the resolution id, the arbiter, and the timestamp:
        those are local metadata, not the adjudicated outcome. Two arbiters that
        read the same submitted records therefore compute the *same* hash and can
        co-sign it, while a tampered verdict changes it. The claims are bound by the
        recomputable facts the decision rests on — the reconciliation hash, the
        corroborating signers, admissibility, and whether the claim stands — not by
        record id, so the same economic claim submitted from both sides binds once.
        """
        return {
            "contract_id": self.contract_id,
            "buyer": self.buyer,
            "seller": self.seller,
            "status": self.status,
            "upheld_hash": self.upheld_hash,
            "upheld_balance_usd": _r(self.upheld_balance_usd),
            "upheld_status": self.upheld_status,
            "source_hashes": sorted(self.source_hashes),
            "claims": sorted(
                (
                    {
                        "reconciliation_hash": c.reconciliation_hash,
                        "signed_by": sorted(c.signed_by),
                        "balance_usd": _r(c.balance_usd),
                        "admissible": c.admissible,
                        "stands": c.stands,
                    }
                    for c in self.claims
                ),
                key=lambda c: (
                    c["reconciliation_hash"],
                    tuple(c["signed_by"]),
                    c["admissible"],
                    c["stands"],
                ),
            ),
        }

    def compute_hash(self) -> str:
        """The resolution hash binding the adjudicated outcome (arbiter-independent)."""
        return stable_hash(self.resolution_facts(), length=32)

    def seal(self) -> Resolution:
        """Stamp the resolution hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> Resolution:
        """Add ``party``'s signature over the resolution hash (sealing first).

        A resolution is signed by whoever adjudicated it — an arbiter or either
        disputing party that independently recomputes it. Re-signing for the same
        party replaces its prior signature, so a resolution cannot accumulate stale
        signatures for one identity.
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

    def _decision_sound(self) -> bool:
        """The recorded outcome re-derives from the recorded claims.

        Re-runs the adjudication over the stored claim verdicts and checks the
        recomputed status, upheld hash, and per-claim standing match what was
        recorded — so a flipped verdict, a swapped winner, or a smuggled-in standing
        claim is caught even when the hash was recomputed to match.
        """
        status, upheld_hash, stands = _decide(self.claims, self.buyer, self.seller)
        if status != self.status or upheld_hash != self.upheld_hash:
            return False
        for c in self.claims:
            if c.stands != (c.settlement_id in stands):
                return False
        # A standing claim must be admissible and carry the upheld hash.
        for c in self.standing_claims:
            if not c.admissible or c.reconciliation_hash != self.upheld_hash:
                return False
        # The upheld balance must be the figure a standing claim actually carries.
        if self.upheld:
            balances = {_r(c.balance_usd) for c in self.standing_claims}
            if _r(self.upheld_balance_usd) not in balances:
                return False
        return True

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> ResolutionVerification:
        """Verify the resolution offline: hash, decision, and signatures.

        Recomputes the resolution hash from the stored fields, re-derives the
        adjudication from the recorded claims to confirm the same figure stands, and
        — with a ``verifier`` — checks that each signature checks. ``require`` names
        parties that must have a verified signature (none by default: a resolution's
        authenticity is that anyone can recompute it and co-sign the same hash). A
        tampered verdict breaks the hash and, almost always, the decision too.
        """
        expected = self.compute_hash()
        hash_ok = bool(self.content_hash) and self.content_hash == expected
        decision_sound = self._decision_sound()
        if not hash_ok:
            return ResolutionVerification(
                valid=False,
                hash_ok=False,
                decision_sound=decision_sound,
                signatures_ok=False,
                reason="content hash mismatch",
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
        required = require or []
        missing = [p for p in required if p not in verified]
        if missing:
            signatures_ok = False
        valid = hash_ok and decision_sound and signatures_ok
        reason = None
        if not decision_sound:
            reason = "recorded outcome does not re-derive from the claims"
        elif not signatures_ok:
            reason = (
                f"missing/invalid signatures for {missing}" if missing else "signature mismatch"
            )
        return ResolutionVerification(
            valid=valid,
            hash_ok=hash_ok,
            decision_sound=decision_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> Resolution:
        """Verify and raise :class:`SettlementError` if the resolution is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"resolution {self.id} failed verification: {result.reason}",
                details={"resolution_id": self.id, "reason": result.reason},
            )
        return self

    def require_resolved(self) -> Resolution:
        """Raise :class:`SettlementError` if the dispute was left unresolved.

        The strict-mode counterpart to inspecting :attr:`status`: a dispute whose
        figures neither side corroborated cannot be closed without one party
        co-signing the other's record (or escalating outside the library), and this
        pinpoints that no admissible claim stood.
        """
        if not self.resolved:
            raise SettlementError(
                f"resolution {self.id} left contract {self.contract_id!r} unresolved: "
                f"{self.reason or 'no corroborated claim'}",
                details={"resolution_id": self.id, "contract_id": self.contract_id},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the adjudication for the audit chain."""
        return to_jsonable(
            {
                "resolution_id": self.id,
                "contract_id": self.contract_id,
                "buyer": self.buyer,
                "seller": self.seller,
                "arbiter": self.arbiter,
                "status": self.status,
                "upheld_hash": self.upheld_hash,
                "upheld_balance_usd": self.upheld_balance_usd,
                "corroborated_by": self.corroborated_by,
                "dissenters": self.dissenters,
                "claims": len(self.claims),
                "rejected": len(self.rejected_claims),
                "inadmissible": len(self.inadmissible_claims),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Resolution:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the outcome and how each submitted claim was adjudicated."""
        title = f"Resolution ({self.arbiter})" if self.arbiter else "Resolution"
        outcome = (
            f"upheld ${self.upheld_balance_usd:+.4f} corroborated by {self.corroborated_by}"
            if self.upheld and self.upheld_balance_usd is not None
            else self.status
        )
        print(f"{title} for contract {self.contract_id}: {outcome}")
        for c in self.claims:
            if c.stands:
                mark = "✓ stands"
            elif not c.admissible:
                mark = "✗ inadmissible"
            else:
                mark = "✗ rejected"
            note = f" ({c.reason})" if c.reason else ""
            print(f"  {mark}: {c.settlement_id} signed by {c.signed_by}{note}")


# -- the adjudication ---------------------------------------------------------


def _decide(
    claims: list[ClaimVerdict], buyer: str, seller: str
) -> tuple[ResolutionStatus, str, set[str]]:
    """The deterministic decision over the admissible claims.

    Pure over the recorded claim verdicts (so :func:`arbitrate` and
    :meth:`Resolution.verify` agree): a reconciliation hash is **corroborated** when
    the parties that signed an admissible claim carrying it include *both* the buyer
    and the seller — the two sides co-signing one figure. Exactly one corroborated
    hash stands; a single uncontested admissible figure stands too; anything else
    (no admissible claim, conflicting figures neither side corroborates, or — only
    under forgery — two contradictory corroborated hashes) is left unresolved.
    Returns ``(status, upheld_hash, {settlement_id of every standing claim})``.
    """
    admissible = [c for c in claims if c.admissible]
    if not admissible:
        return "unresolved", "", set()

    corroborators: dict[str, set[str]] = {}
    for c in admissible:
        corr = corroborators.setdefault(c.reconciliation_hash, set())
        for party in c.signed_by:
            if party in (buyer, seller):
                corr.add(party)

    parties = {buyer, seller}
    mutual = sorted(h for h, corr in corroborators.items() if parties <= corr)
    distinct = sorted(corroborators)

    if len(mutual) == 1:
        upheld_hash = mutual[0]
    elif len(mutual) > 1:
        return "unresolved", "", set()  # contradictory mutual agreements (forgery)
    elif len(distinct) == 1:
        upheld_hash = distinct[0]  # a single uncontested admissible figure stands
    else:
        return "unresolved", "", set()  # a genuine standoff neither side corroborates

    stands = {c.settlement_id for c in admissible if c.reconciliation_hash == upheld_hash}
    return "upheld", upheld_hash, stands


def _claim_verdict(
    record: SettlementRecord, *, verifier: ChainSigner | None
) -> ClaimVerdict:
    """Admissibility for one submitted record — pinpointed, never raised.

    A claim is inadmissible when its reconciliation hash no longer recomputes (a
    tampered economic figure), when it carries no signature (an unauthenticated
    claim), or — with a ``verifier`` — when a signature is forged. Unlike netting,
    which refuses to clear over a tampered book, arbitration is the venue where a
    bad claim is adjudicated, so it is recorded as inadmissible and pinpointed rather
    than crashing the resolution.
    """
    signed_by = [s.party for s in record.signatures]
    admissible = True
    reason: str | None = None
    if record.content_hash != record.compute_hash():
        admissible, reason = False, "tampered: reconciliation hash does not recompute"
    elif not record.signatures:
        admissible, reason = False, "unsigned: claim carries no signature"
    elif verifier is not None:
        check = record.verify(verifier, require=[])
        if not check.signatures_ok:
            admissible, reason = False, "forged: a signature does not verify"
    return ClaimVerdict(
        settlement_id=record.id,
        contract_id=record.contract_id,
        reconciliation_hash=record.content_hash,
        signed_by=signed_by,
        balance_usd=_r(record.balance_usd),
        settlement_status=record.status,
        admissible=admissible,
        reason=reason,
    )


def arbitrate(
    records: Iterable[SettlementRecord],
    *,
    contract_id: str | None = None,
    arbiter: str = "",
    verifier: ChainSigner | None = None,
    verify_with: ChainSigner | None = None,
) -> Resolution:
    """Adjudicate a disputed contract from the records its parties submit.

    Reads the signed :class:`~vincio.settlement.record.SettlementRecord`\\ s the
    disputing parties submit for one contract and decides — deterministically and
    from nothing it cannot recompute — which figure **stands**: a reconciliation hash
    that both the buyer and the seller signed (each on their own record, co-signing
    one figure) is mutually corroborated and upheld, a unilateral claim contradicting
    it is rejected and pinpointed, and a single uncontested figure stands on its own.
    When neither side's figure is corroborated the dispute is left **unresolved**
    rather than decided by fiat. A tampered or forged claim is marked inadmissible
    and pinpointed, never silently dropped. The returned :class:`Resolution` is
    sealed but unsigned — sign it with the arbiter's key.

    ``contract_id`` selects the disputed contract when the pool mixes several
    (inferred when every record is for one contract); ``arbiter`` labels the
    resolution; ``verifier`` authenticates each submitted record's signatures.
    ``verify_with`` is a deprecated alias for ``verifier`` (since 7.5, removed in
    8.0). Raises :class:`SettlementError` only on a category error — no records,
    or a pool spanning several contracts with no ``contract_id`` to pick one.
    """
    verifier = _resolve_verifier(verifier, verify_with, "arbitrate")
    pool = list(records)
    if contract_id is not None:
        pool = [r for r in pool if r.contract_id == contract_id]
        if not pool:
            raise SettlementError(
                f"no submitted records for contract {contract_id!r} to arbitrate",
                details={"contract_id": contract_id},
            )
    else:
        ids = sorted({r.contract_id for r in pool})
        if not ids:
            raise SettlementError("no records submitted to arbitrate", details={})
        if len(ids) > 1:
            raise SettlementError(
                f"cannot arbitrate records spanning several contracts {ids}; "
                f"pass contract_id= to pick the disputed one",
                details={"contracts": ids},
            )
        contract_id = ids[0]

    # The disputed parties are the contract's buyer/seller, taken deterministically.
    buyer = sorted({r.buyer for r in pool})[0]
    seller = sorted({r.seller for r in pool})[0]

    claims = [_claim_verdict(r, verifier=verifier) for r in pool]
    claims.sort(key=lambda c: (c.reconciliation_hash, tuple(sorted(c.signed_by)), c.settlement_id))

    status, upheld_hash, stands = _decide(claims, buyer, seller)
    for c in claims:
        c.stands = c.settlement_id in stands

    corroborated_by: list[str] = []
    upheld_balance: float | None = None
    upheld_status = ""
    if status == "upheld":
        signers: set[str] = set()
        for c in claims:
            if c.stands:
                signers.update(p for p in c.signed_by if p in (buyer, seller))
                if upheld_balance is None:
                    upheld_balance = c.balance_usd
                    upheld_status = c.settlement_status
        corroborated_by = sorted(signers)

    reason = _verdict_reason(status, upheld_hash, claims, buyer, seller)
    for c in claims:
        if c.admissible and not c.stands:
            c.reason = (
                "rejected: contradicts the corroborated figure"
                if status == "upheld"
                else "unresolved: no corroborated figure"
            )

    resolution = Resolution(
        contract_id=contract_id,
        buyer=buyer,
        seller=seller,
        arbiter=arbiter,
        status=status,
        upheld_hash=upheld_hash,
        upheld_balance_usd=upheld_balance,
        upheld_status=upheld_status,
        claims=claims,
        corroborated_by=corroborated_by,
        source_hashes=sorted({c.reconciliation_hash for c in claims if c.admissible}),
        reason=reason,
    )
    return resolution.seal()


def _verdict_reason(
    status: ResolutionStatus,
    upheld_hash: str,
    claims: list[ClaimVerdict],
    buyer: str,
    seller: str,
) -> str | None:
    """A human-readable summary of why the dispute resolved as it did."""
    if status == "upheld":
        return None
    admissible = [c for c in claims if c.admissible]
    if not admissible:
        return "no admissible claim: every submitted record was tampered or forged"
    if len({c.reconciliation_hash for c in admissible}) <= 1:
        return None
    return (
        "unresolved: the submitted figures disagree and neither was corroborated by "
        "both parties"
    )
