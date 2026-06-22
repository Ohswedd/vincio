"""Cross-org collateral custody attestation & proof-of-reserves.

A :class:`~vincio.settlement.rehypothecation.CollateralLedger` now bounds a counterparty's
pledges across its :class:`~vincio.settlement.collateral.CollateralPool`\\ s against the
capital it actually ``held`` — but that holdings figure is the one input the guard
*trusts*: it is **asserted**, not proven. A counterparty over-stating its real reserves
still passes the guard, the way a self-asserted reputation score passed before a
:class:`~vincio.settlement.attestation.ReputationAttestation` made standing verifiable.
This module is the next reach: it makes the held capital itself **evidence-backed** — a
signed, offline-verifiable **proof-of-reserves** that the reserves the guard bounds against
actually exist, so a re-use bound rests on a proven figure rather than a promise.

* **Signed proof-of-reserves.** A custodian (or the poster's own signed reserve record)
  issues a :class:`CustodyAttestation` over the capital actually held — itemized into one
  :class:`ReserveLine` per custodied holding, so the attested ``reserves_usd`` total
  re-derives from the components the way an escrow's forfeiture re-derives from the
  shortfall. It is content-bound and signed with the same
  :class:`~vincio.security.audit.ChainSigner` a contract uses, so the proof is a mechanical,
  reconstructable artifact, never a custodian's say-so.
* **Read by the guard.** :func:`~vincio.settlement.rehypothecation.guard_collateral`
  (with ``custody=``) reads the attested reserves as the ``held`` figure instead of the
  asserted default — a pledge bounded against **proven** reserves. When the proven reserves
  fall below what the pools pledge, the shortfall surfaces as a bounded, pinpointed
  under-reserved breach, the way an over-commitment does today, rather than passing on an
  inflated holdings claim. A custody attestation for a different poster, a tampered reserve
  figure, or (with a verifier) a forged custodian is **refused**, never silently honored.
* **Auditable & offline.** The attestation reads only what it can recompute:
  :meth:`CustodyAttestation.verify` recomputes the content hash and re-derives the reserve
  total from the line items, so a tampered figure is caught even after re-sealing, and a
  forged custodian signature is caught with the verifier — and the
  :class:`~vincio.settlement.book.SettlementBook` / :class:`~vincio.core.app.ContextApp`
  path lands each issuance on the hash-chained audit log. Never a hosted custodian or a
  trusted third party — a verifiable proof-of-reserves over the collateral the fabric
  already pools.

The attestation folds into the *existing* guard path: :func:`attest_custody` builds one over
a poster's reserves, and :func:`~vincio.settlement.rehypothecation.guard_collateral` reads it
as the held figure in one call (:meth:`~vincio.core.app.ContextApp.attest_custody` /
:meth:`~vincio.core.app.ContextApp.guard_collateral`). Everything is dependency-free,
deterministic, and offline.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .record import SettlementSignature

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "ReserveLine",
    "CustodyAttestationVerification",
    "CustodyAttestation",
    "attest_custody",
]

# The single audit action every custody attestation is recorded under; the decision field
# carries whether the attestation is self-custody (``self_custody`` / ``custodied``).
CUSTODY_ACTION = "custody_attestation"

_TOLERANCE = 1e-6


def _r6(value: float) -> float:
    """Round a money figure to six places so float drift never breaks a balance."""
    return round(float(value), 6)


class ReserveLine(BaseModel):
    """One custodied holding backing the poster's proven reserves.

    A proof-of-reserves is itemized into its components — each ``account`` (a custody
    account, wallet, or escrow id) holding ``amount_usd`` — so the attested total
    re-derives from the line items and a tampered total is caught even after re-sealing,
    the way a :class:`~vincio.settlement.meter.MeterReading`'s total is exactly the sum of
    its events. The ``note`` is free-text provenance (the custodian's reference for the
    holding) and is bound into the attestation hash like every other field.
    """

    account: str
    amount_usd: float = 0.0
    note: str = ""

    def facts(self) -> dict[str, Any]:
        """The per-line facts the attestation's content hash binds."""
        return {
            "account": self.account,
            "amount_usd": _r6(self.amount_usd),
            "note": self.note,
        }


class CustodyAttestationVerification(BaseModel):
    """The (non-raising) outcome of verifying a custody attestation offline.

    An attestation is **valid** when its content hash recomputes (``hash_ok``), the attested
    reserve total re-derives from the line items and every holding is non-negative
    (``reserves_sound``), and — with a ``verifier`` — every signature checks
    (``signatures_ok``). A tampered reserve figure or a forged custodian is caught from the
    bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    reserves_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class CustodyAttestation(BaseModel):
    """A signed, offline-verifiable proof-of-reserves over a poster's held capital.

    Produced by :func:`attest_custody` (or
    :meth:`~vincio.settlement.book.SettlementBook.attest_custody` /
    :meth:`~vincio.core.app.ContextApp.attest_custody`): a ``custodian`` attests the capital a
    ``poster`` actually holds, itemized into :class:`ReserveLine`\\ s whose amounts sum to
    ``reserves_usd``. It binds the custodian, the poster, the reserve line items, and the
    total onto a content hash, so the proof is a mechanical number anyone recomputes.

    When ``custodian == poster`` the attestation is **self-custody** — the poster's own
    signed reserve record, still content-bound and non-repudiable rather than a bare number;
    otherwise an independent custodian vouches for the reserves.
    :func:`~vincio.settlement.rehypothecation.guard_collateral` reads ``reserves_usd`` as the
    ``held`` figure it bounds the pledges against. :meth:`verify` re-derives the total from
    the line items and checks the custodian signature from the bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("custody"))
    custodian: str
    poster: str
    reserves: list[ReserveLine] = Field(default_factory=list)
    reserves_usd: float = 0.0

    as_of: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def self_custody(self) -> bool:
        """Whether the poster attests its own reserves (custodian is the poster)."""
        return self.custodian == self.poster

    @property
    def accounts(self) -> list[str]:
        """The custody accounts the reserves are held in, sorted."""
        return sorted(line.account for line in self.reserves)

    def _reserves_total(self) -> float:
        """The reserve total re-derived from the line items."""
        return _r6(sum(line.amount_usd for line in self.reserves))

    # -- hashing ------------------------------------------------------------

    def attestation_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the custodian, poster, lines, and total.

        Excludes the id, signatures, and audit linkage (local metadata, not the proof), so
        the same custodian attesting the same reserves for the same poster as of the same
        instant hashes identically wherever it is recomputed — the way two parties co-sign
        one reconciliation hash. Lines are sorted by account so the order they were listed in
        never changes the hash.
        """
        return {
            "custodian": self.custodian,
            "poster": self.poster,
            "reserves_usd": _r6(self.reserves_usd),
            "as_of": self.as_of.isoformat(),
            "reserves": [
                line.facts() for line in sorted(self.reserves, key=lambda line: line.account)
            ],
        }

    def compute_hash(self) -> str:
        """The content hash binding the custodian, the poster, and the reserves."""
        return stable_hash(self.attestation_facts(), length=32)

    def seal(self) -> CustodyAttestation:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str | None = None) -> CustodyAttestation:
        """Add the custodian's signature over the content hash (sealing first).

        A custody attestation is the *custodian's* claim, so only the custodian signs it
        (``party`` defaults to the custodian; passing a different party is refused). Re-signing
        replaces the prior signature, so an attestation cannot accumulate stale signatures.
        """
        resolved = party or self.custodian
        if resolved != self.custodian:
            raise SettlementError(
                f"a custody attestation is signed by its custodian {self.custodian!r}, "
                f"not {resolved!r}",
                details={"attestation_id": self.id, "custodian": self.custodian, "party": resolved},
            )
        if not self.content_hash:
            self.seal()
        sig = SettlementSignature(
            party=resolved,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != resolved]
        self.signatures.append(sig)
        return self

    # -- verification -------------------------------------------------------

    def _reserves_sound(self) -> bool:
        """The attested total re-derives from the line items and no holding is negative."""
        if any(line.amount_usd < -_TOLERANCE for line in self.reserves):
            return False
        return abs(self.reserves_usd - self._reserves_total()) <= _TOLERANCE

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CustodyAttestationVerification:
        """Verify the attestation offline: the hash recomputes and the reserves re-derive.

        Recomputes the content hash and re-derives the reserve total from the line items
        (checking every holding is non-negative) — so a tampered reserve figure is caught
        even when the hash was recomputed to match. ``verifier`` additionally checks each
        signature against the content hash; ``require`` names parties that must have a
        verified signature (defaults to none — pass ``[custodian]`` to demand the custodian's
        signature).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        reserves_sound = self._reserves_sound()
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
        valid = hash_ok and reserves_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "attestation is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the attestation facts"
            elif not reserves_sound:
                reason = "reserve total does not re-derive from the line items"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return CustodyAttestationVerification(
            valid=valid,
            hash_ok=hash_ok,
            reserves_sound=reserves_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CustodyAttestation:
        """Verify and raise :class:`SettlementError` if the attestation is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"custody attestation {self.id} failed verification: {result.reason}",
                details={"attestation_id": self.id, "reason": result.reason},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the attestation for the audit chain."""
        return to_jsonable(
            {
                "attestation_id": self.id,
                "custodian": self.custodian,
                "poster": self.poster,
                "self_custody": self.self_custody,
                "reserves_usd": _r6(self.reserves_usd),
                "accounts": len(self.reserves),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> CustodyAttestation:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the attested reserves and the custody accounts."""
        kind = "self-custody" if self.self_custody else f"custodied by {self.custodian}"
        print(
            f"Custody attestation ({self.poster}): ${self.reserves_usd:,.2f} in reserves "
            f"across {len(self.reserves)} account(s) — {kind}"
        )
        for line in sorted(self.reserves, key=lambda line: line.account):
            note = f" ({line.note})" if line.note else ""
            print(f"  {line.account}: ${line.amount_usd:,.2f}{note}")


# -- module-level builder -----------------------------------------------------


def _coerce_reserves(reserves: Any) -> list[ReserveLine]:
    """Normalize a reserves spec into :class:`ReserveLine`\\ s.

    Accepts a single number (one unnamed line), a mapping of ``account -> amount``, or an
    iterable of :class:`ReserveLine` / ``(account, amount)`` pairs / dicts. Raises
    :class:`SettlementError` on a negative holding so a proof-of-reserves can never net a
    real shortfall against a fictitious negative balance.
    """
    lines: list[ReserveLine] = []
    if isinstance(reserves, (int, float)):
        lines = [ReserveLine(account="reserves", amount_usd=float(reserves))]
    elif isinstance(reserves, dict):
        lines = [ReserveLine(account=str(k), amount_usd=float(v)) for k, v in reserves.items()]
    else:
        for item in reserves:
            if isinstance(item, ReserveLine):
                lines.append(item)
            elif isinstance(item, dict):
                lines.append(ReserveLine.model_validate(item))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                lines.append(ReserveLine(account=str(item[0]), amount_usd=float(item[1])))
            else:
                raise SettlementError(
                    "attest_custody reserves must be a number, a mapping, or ReserveLine / "
                    f"(account, amount) items; got {item!r}",
                    details={"item": repr(item)},
                )
    for line in lines:
        if line.amount_usd < 0.0:
            raise SettlementError(
                f"reserve account {line.account!r} holds a negative amount {line.amount_usd}; "
                "a proof-of-reserves cannot net a real shortfall against a fictitious balance",
                details={"account": line.account, "amount_usd": line.amount_usd},
            )
    return lines


def attest_custody(
    poster: str,
    reserves: Any,
    *,
    custodian: str | None = None,
    as_of: datetime | None = None,
) -> CustodyAttestation:
    """Attest a poster's proven reserves into an (unsigned) :class:`CustodyAttestation`.

    The proof-of-reserves analogue of
    :func:`~vincio.settlement.attestation.attest_reputation`: ``custodian`` (defaulting to
    the ``poster`` itself — self-custody) vouches for the capital ``poster`` actually holds.
    ``reserves`` is a single number (one unnamed holding), a mapping of ``account ->
    amount``, or an iterable of :class:`ReserveLine` / ``(account, amount)`` items; the
    attested ``reserves_usd`` is their sum, re-derived on every verify.

    Returns a sealed, unsigned attestation — sign it with the custodian's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.attest_custody`). Raises
    :class:`SettlementError` when a holding is negative.
    """
    lines = _coerce_reserves(reserves)
    attestation = CustodyAttestation(
        custodian=custodian or poster,
        poster=poster,
        reserves=lines,
        reserves_usd=_r6(sum(line.amount_usd for line in lines)),
        as_of=as_of or utcnow(),
    )
    return attestation.seal()
