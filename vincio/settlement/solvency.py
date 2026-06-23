"""Cross-org custody liability attestation & proof-of-solvency.

A :class:`~vincio.settlement.custody.CustodyAttestation` now proves the capital a
counterparty *holds*, so :func:`~vincio.settlement.rehypothecation.guard_collateral`
bounds its pledges against a **proven** reserve figure rather than a self-reported one.
But reserves are only one side of the ledger. A counterparty solvent against *one*
buyer's pledges may be deeply **under-water** once *every* obligation it owes is counted:
the guard sees the reserves and this buyer's pledges, never the counterparty's other
liabilities. A counterparty could prove the same reserves against many buyers while
quietly insolvent across all of them — the canonical gap the proof-of-reserves literature
closes next with a **proof-of-solvency** (reserves ≥ total liabilities). This module is
that reach: it makes the *liability* side evidence-backed too, and folds the two proofs
into a bounded, offline-verifiable solvency margin the guard bounds pledges against.

* **Signed liability attestation.** A counterparty (or its custodian) issues a
  :class:`LiabilityAttestation` over the total obligations it owes — itemized into one
  :class:`LiabilityLine` per creditor, so the attested ``liabilities_usd`` total re-derives
  from the components the way a reserve attestation's total does. It is the liability
  analogue of the proof-of-reserves: content-bound and signed with the same
  :class:`~vincio.security.audit.ChainSigner` a contract uses, so the claim is a mechanical,
  reconstructable artifact, never a counterparty's say-so.
* **Proof-of-solvency.** :func:`prove_solvency` folds a
  :class:`~vincio.settlement.custody.CustodyAttestation` against a
  :class:`LiabilityAttestation` into a :class:`SolvencyProof` — a bounded, offline-verifiable
  solvency margin (``reserves − liabilities``). When the proven reserves cannot cover the
  proven liabilities the shortfall surfaces as a pinpointed :class:`InsolvencyBreach` rather
  than passing on a one-sided reserve claim, and the proof exposes a *solvency-adjusted* held
  figure (the unencumbered capital, ``max(0, reserves − liabilities)``) that
  :func:`~vincio.settlement.rehypothecation.guard_collateral` reads (``solvency=``) as the
  held figure — so a pledge is bounded against capital **not already owed elsewhere**.
* **Auditable & offline.** Both attestations read only signed, content-bound artifacts:
  :meth:`LiabilityAttestation.verify` recomputes the content hash and re-derives the
  liability total from the line items, so a tampered figure is caught even after re-sealing,
  and a forged issuer signature is caught with the verifier. :func:`prove_solvency` refuses a
  tampered attestation, a forged issuer, or a custody / liability pair for *different* posters,
  and :meth:`SolvencyProof.verify` re-derives the margin and the insolvency breach from the
  bytes alone. The :class:`~vincio.settlement.book.SettlementBook` /
  :class:`~vincio.core.app.ContextApp` path lands each issuance and proof on the hash-chained
  audit log. Never a hosted solvency auditor or a trusted third party — a verifiable
  proof-of-solvency over the obligations and reserves the fabric already attests.

The proof folds into the *existing* guard path: :func:`attest_liabilities` builds a liability
proof, :func:`prove_solvency` folds it against a reserve proof, and
:func:`~vincio.settlement.rehypothecation.guard_collateral` reads the result as the held figure
in one call (:meth:`~vincio.core.app.ContextApp.attest_liabilities` /
:meth:`~vincio.core.app.ContextApp.prove_solvency`). Everything is dependency-free,
deterministic, and offline.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .custody import CustodyAttestation
from .record import SettlementSignature

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "LiabilityLine",
    "LiabilityAttestationVerification",
    "LiabilityAttestation",
    "attest_liabilities",
    "InsolvencyBreach",
    "SolvencyProofVerification",
    "SolvencyProof",
    "prove_solvency",
]

# The audit action a liability attestation is recorded under; the decision field carries
# whether the attestation is self-attested (``self_attested`` / ``attested``).
LIABILITY_ACTION = "liability_attestation"

# The audit action a solvency proof is recorded under; the decision field carries whether the
# counterparty is solvent (``solvent`` / ``insolvent``).
SOLVENCY_ACTION = "solvency_proof"

_TOLERANCE = 1e-6


def _r6(value: float) -> float:
    """Round a money figure to six places so float drift never breaks a balance."""
    return round(float(value), 6)


# -- liability attestation ----------------------------------------------------


class LiabilityLine(BaseModel):
    """One obligation owed, backing the poster's attested total liabilities.

    A proof-of-liabilities is itemized into its components — each obligation owed to a
    ``creditor`` for ``amount_usd`` — so the attested total re-derives from the line items
    and a tampered total is caught even after re-sealing, the way a
    :class:`~vincio.settlement.custody.ReserveLine`'s holding does for reserves. The ``note``
    is free-text provenance (the obligation's reference) and is bound into the attestation
    hash like every other field.
    """

    creditor: str
    amount_usd: float = 0.0
    note: str = ""

    def facts(self) -> dict[str, Any]:
        """The per-line facts the attestation's content hash binds."""
        return {
            "creditor": self.creditor,
            "amount_usd": _r6(self.amount_usd),
            "note": self.note,
        }


class LiabilityAttestationVerification(BaseModel):
    """The (non-raising) outcome of verifying a liability attestation offline.

    An attestation is **valid** when its content hash recomputes (``hash_ok``), the attested
    liability total re-derives from the line items and every obligation is non-negative
    (``liabilities_sound``), and — with a ``verifier`` — every signature checks
    (``signatures_ok``). A tampered liability figure or a forged issuer is caught from the
    bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    liabilities_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class LiabilityAttestation(BaseModel):
    """A signed, offline-verifiable proof-of-liabilities over a poster's total obligations.

    Produced by :func:`attest_liabilities` (or
    :meth:`~vincio.settlement.book.SettlementBook.attest_liabilities` /
    :meth:`~vincio.core.app.ContextApp.attest_liabilities`): an ``attestor`` attests the total
    obligations a ``poster`` owes, itemized into :class:`LiabilityLine`\\ s whose amounts sum
    to ``liabilities_usd``. It binds the attestor, the poster, the liability line items, and
    the total onto a content hash, so the claim is a mechanical number anyone recomputes —
    the liability analogue of a :class:`~vincio.settlement.custody.CustodyAttestation`.

    When ``attestor == poster`` the attestation is **self-attested** — the poster's own signed
    liability record, still content-bound and non-repudiable rather than a bare number;
    otherwise an independent attestor (e.g. an auditor or custodian) vouches for the
    obligations. :func:`prove_solvency` folds ``liabilities_usd`` against a proven reserve
    figure into a solvency margin. :meth:`verify` re-derives the total from the line items and
    checks the attestor signature from the bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("liability"))
    attestor: str
    poster: str
    liabilities: list[LiabilityLine] = Field(default_factory=list)
    liabilities_usd: float = 0.0

    as_of: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def self_attested(self) -> bool:
        """Whether the poster attests its own liabilities (attestor is the poster)."""
        return self.attestor == self.poster

    @property
    def creditors(self) -> list[str]:
        """The creditors the obligations are owed to, sorted."""
        return sorted(line.creditor for line in self.liabilities)

    def _liabilities_total(self) -> float:
        """The liability total re-derived from the line items."""
        return _r6(sum(line.amount_usd for line in self.liabilities))

    # -- hashing ------------------------------------------------------------

    def attestation_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the attestor, poster, lines, and total.

        Excludes the id, signatures, and audit linkage (local metadata, not the proof), so the
        same attestor attesting the same obligations for the same poster as of the same instant
        hashes identically wherever it is recomputed. Lines are sorted by creditor so the order
        they were listed in never changes the hash.
        """
        return {
            "attestor": self.attestor,
            "poster": self.poster,
            "liabilities_usd": _r6(self.liabilities_usd),
            "as_of": self.as_of.isoformat(),
            "liabilities": [
                line.facts() for line in sorted(self.liabilities, key=lambda line: line.creditor)
            ],
        }

    def compute_hash(self) -> str:
        """The content hash binding the attestor, the poster, and the liabilities."""
        return stable_hash(self.attestation_facts(), length=32)

    def seal(self) -> LiabilityAttestation:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str | None = None) -> LiabilityAttestation:
        """Add the attestor's signature over the content hash (sealing first).

        A liability attestation is the *attestor's* claim, so only the attestor signs it
        (``party`` defaults to the attestor; passing a different party is refused). Re-signing
        replaces the prior signature, so an attestation cannot accumulate stale signatures.
        """
        resolved = party or self.attestor
        if resolved != self.attestor:
            raise SettlementError(
                f"a liability attestation is signed by its attestor {self.attestor!r}, "
                f"not {resolved!r}",
                details={"attestation_id": self.id, "attestor": self.attestor, "party": resolved},
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

    def _liabilities_sound(self) -> bool:
        """The attested total re-derives from the line items and no obligation is negative."""
        if any(line.amount_usd < -_TOLERANCE for line in self.liabilities):
            return False
        return abs(self.liabilities_usd - self._liabilities_total()) <= _TOLERANCE

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> LiabilityAttestationVerification:
        """Verify the attestation offline: the hash recomputes and the liabilities re-derive.

        Recomputes the content hash and re-derives the liability total from the line items
        (checking every obligation is non-negative) — so a tampered liability figure is caught
        even when the hash was recomputed to match. ``verifier`` additionally checks each
        signature against the content hash; ``require`` names parties that must have a verified
        signature (defaults to none — pass ``[attestor]`` to demand the attestor's signature).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        liabilities_sound = self._liabilities_sound()
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
        valid = hash_ok and liabilities_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "attestation is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the attestation facts"
            elif not liabilities_sound:
                reason = "liability total does not re-derive from the line items"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return LiabilityAttestationVerification(
            valid=valid,
            hash_ok=hash_ok,
            liabilities_sound=liabilities_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> LiabilityAttestation:
        """Verify and raise :class:`SettlementError` if the attestation is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"liability attestation {self.id} failed verification: {result.reason}",
                details={"attestation_id": self.id, "reason": result.reason},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the attestation for the audit chain."""
        return to_jsonable(
            {
                "attestation_id": self.id,
                "attestor": self.attestor,
                "poster": self.poster,
                "self_attested": self.self_attested,
                "liabilities_usd": _r6(self.liabilities_usd),
                "creditors": len(self.liabilities),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> LiabilityAttestation:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the attested liabilities and the creditors."""
        kind = "self-attested" if self.self_attested else f"attested by {self.attestor}"
        print(
            f"Liability attestation ({self.poster}): ${self.liabilities_usd:,.2f} owed "
            f"across {len(self.liabilities)} creditor(s) — {kind}"
        )
        for line in sorted(self.liabilities, key=lambda line: line.creditor):
            note = f" ({line.note})" if line.note else ""
            print(f"  {line.creditor}: ${line.amount_usd:,.2f}{note}")


def _coerce_liabilities(liabilities: Any) -> list[LiabilityLine]:
    """Normalize a liabilities spec into :class:`LiabilityLine`\\ s.

    Accepts a single number (one unnamed obligation), a mapping of ``creditor -> amount``, or
    an iterable of :class:`LiabilityLine` / ``(creditor, amount)`` pairs / dicts. Raises
    :class:`SettlementError` on a negative obligation so a proof-of-liabilities can never net a
    real debt against a fictitious negative balance.
    """
    lines: list[LiabilityLine] = []
    if isinstance(liabilities, (int, float)):
        lines = [LiabilityLine(creditor="liabilities", amount_usd=float(liabilities))]
    elif isinstance(liabilities, dict):
        lines = [
            LiabilityLine(creditor=str(k), amount_usd=float(v)) for k, v in liabilities.items()
        ]
    else:
        for item in liabilities:
            if isinstance(item, LiabilityLine):
                lines.append(item)
            elif isinstance(item, dict):
                lines.append(LiabilityLine.model_validate(item))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                lines.append(LiabilityLine(creditor=str(item[0]), amount_usd=float(item[1])))
            else:
                raise SettlementError(
                    "attest_liabilities liabilities must be a number, a mapping, or "
                    f"LiabilityLine / (creditor, amount) items; got {item!r}",
                    details={"item": repr(item)},
                )
    for line in lines:
        if line.amount_usd < 0.0:
            raise SettlementError(
                f"liability to {line.creditor!r} owes a negative amount {line.amount_usd}; a "
                "proof-of-liabilities cannot net a real debt against a fictitious balance",
                details={"creditor": line.creditor, "amount_usd": line.amount_usd},
            )
    return lines


def attest_liabilities(
    poster: str,
    liabilities: Any,
    *,
    attestor: str | None = None,
    as_of: datetime | None = None,
) -> LiabilityAttestation:
    """Attest a poster's total obligations into an (unsigned) :class:`LiabilityAttestation`.

    The proof-of-liabilities analogue of
    :func:`~vincio.settlement.custody.attest_custody`: ``attestor`` (defaulting to the
    ``poster`` itself — self-attested) vouches for the total obligations ``poster`` owes.
    ``liabilities`` is a single number (one unnamed obligation), a mapping of ``creditor ->
    amount``, or an iterable of :class:`LiabilityLine` / ``(creditor, amount)`` items; the
    attested ``liabilities_usd`` is their sum, re-derived on every verify.

    Returns a sealed, unsigned attestation — sign it with the attestor's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.attest_liabilities`). Raises
    :class:`SettlementError` when an obligation is negative.
    """
    lines = _coerce_liabilities(liabilities)
    attestation = LiabilityAttestation(
        attestor=attestor or poster,
        poster=poster,
        liabilities=lines,
        liabilities_usd=_r6(sum(line.amount_usd for line in lines)),
        as_of=as_of or utcnow(),
    )
    return attestation.seal()


# -- proof-of-solvency --------------------------------------------------------


class InsolvencyBreach(BaseModel):
    """A proven shortfall — the obligations owed exceed the reserves actually held.

    Surfaced by :func:`prove_solvency` when the proven reserves cannot cover the proven
    liabilities: the ``poster`` holds ``reserves_usd`` (the custody attestation pinned by
    ``custody_hash``) but owes ``liabilities_usd`` (the liability attestation pinned by
    ``liability_hash``, attested by ``attestor``), so it is insolvent by ``shortfall_usd``
    (``liabilities − reserves``). The reserve proof alone could not catch this: it proves the
    capital exists, not that it exceeds every obligation the counterparty owes elsewhere.
    """

    poster: str
    custodian: str = ""
    attestor: str = ""
    custody_hash: str = ""
    liability_hash: str = ""
    reserves_usd: float = 0.0
    liabilities_usd: float = 0.0
    shortfall_usd: float = 0.0


class SolvencyProofVerification(BaseModel):
    """The (non-raising) outcome of verifying a solvency proof offline.

    A proof is **valid** when its content hash recomputes (``hash_ok``), the solvency margin
    and the solvency-adjusted held figure re-derive from the proven reserves and liabilities
    and the insolvency breach re-derives from the margin (``margin_sound``), and — with a
    ``verifier`` — every signature checks (``signatures_ok``). A tampered margin or a flipped
    solvency verdict is caught from the bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    margin_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class SolvencyProof(BaseModel):
    """A signed, offline-verifiable proof-of-solvency over a poster's reserves and liabilities.

    Produced by :func:`prove_solvency` (or
    :meth:`~vincio.settlement.book.SettlementBook.prove_solvency` /
    :meth:`~vincio.core.app.ContextApp.prove_solvency`): it folds a proven
    :class:`~vincio.settlement.custody.CustodyAttestation` (``reserves_usd``, by ``custodian``)
    against a proven :class:`LiabilityAttestation` (``liabilities_usd``, by ``attestor``) for
    the same ``poster`` into a solvency ``margin_usd`` (``reserves − liabilities``). It binds
    both attestation hashes, the proven figures, and the margin onto a content hash, so the
    proof is a mechanical number anyone recomputes.

    The proof is **solvent** when ``margin_usd >= 0``; otherwise the shortfall surfaces as a
    pinpointed :class:`InsolvencyBreach`. :attr:`solvency_adjusted_held` is the unencumbered
    capital — ``max(0, margin)`` — that :func:`~vincio.settlement.rehypothecation.guard_collateral`
    reads (``solvency=``) as the held figure, so a pledge is bounded against capital **not
    already owed elsewhere**. :meth:`verify` re-derives the margin and the breach from the
    bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("solvency"))
    poster: str
    custodian: str = ""
    attestor: str = ""
    custody_hash: str = ""
    liability_hash: str = ""

    reserves_usd: float = 0.0
    liabilities_usd: float = 0.0
    margin_usd: float = 0.0

    as_of: datetime = Field(default_factory=utcnow)
    breach: InsolvencyBreach | None = None
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def solvent(self) -> bool:
        """Whether the proven reserves cover the proven liabilities (margin ≥ 0)."""
        return self.margin_usd >= -_TOLERANCE

    @property
    def insolvent(self) -> bool:
        """Whether the proven liabilities exceed the proven reserves (a shortfall)."""
        return not self.solvent

    @property
    def solvency_adjusted_held(self) -> float:
        """The unencumbered capital the guard bounds pledges against (``max(0, margin)``).

        The capital left once every proven obligation is covered — what the counterparty can
        back a *new* pledge with. Floored at zero, since an insolvent counterparty has no free
        capital to pledge (its shortfall is pinpointed separately as an
        :class:`InsolvencyBreach`).
        """
        return _r6(max(0.0, self.margin_usd))

    @property
    def status(self) -> str:
        """``solvent`` (reserves cover liabilities) or ``insolvent`` (a proven shortfall)."""
        return "solvent" if self.solvent else "insolvent"

    def _derive_breach(self) -> InsolvencyBreach | None:
        """The insolvency breach when the proven liabilities exceed the proven reserves."""
        shortfall = _r6(max(0.0, self.liabilities_usd - self.reserves_usd))
        if shortfall <= _TOLERANCE:
            return None
        return InsolvencyBreach(
            poster=self.poster,
            custodian=self.custodian,
            attestor=self.attestor,
            custody_hash=self.custody_hash,
            liability_hash=self.liability_hash,
            reserves_usd=_r6(self.reserves_usd),
            liabilities_usd=_r6(self.liabilities_usd),
            shortfall_usd=shortfall,
        )

    # -- hashing ------------------------------------------------------------

    def solvency_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the parties, the proofs, and the margin.

        Excludes the id, signatures, and audit linkage (local metadata), so the same reserve
        and liability proofs folded for the same poster hash identically wherever they are
        recomputed — the way two parties co-sign one reconciliation hash. The insolvency breach
        is bound in (when present) so a flipped verdict is caught even after re-sealing.
        """
        return {
            "poster": self.poster,
            "custodian": self.custodian,
            "attestor": self.attestor,
            "custody_hash": self.custody_hash,
            "liability_hash": self.liability_hash,
            "reserves_usd": _r6(self.reserves_usd),
            "liabilities_usd": _r6(self.liabilities_usd),
            "margin_usd": _r6(self.margin_usd),
            "solvency_adjusted_held": self.solvency_adjusted_held,
            "as_of": self.as_of.isoformat(),
            "breach": (
                {
                    "poster": self.breach.poster,
                    "custodian": self.breach.custodian,
                    "attestor": self.breach.attestor,
                    "custody_hash": self.breach.custody_hash,
                    "liability_hash": self.breach.liability_hash,
                    "reserves_usd": _r6(self.breach.reserves_usd),
                    "liabilities_usd": _r6(self.breach.liabilities_usd),
                    "shortfall_usd": _r6(self.breach.shortfall_usd),
                }
                if self.breach is not None
                else None
            ),
        }

    def compute_hash(self) -> str:
        """The content hash binding the folded proofs and the solvency margin."""
        return stable_hash(self.solvency_facts(), length=32)

    def seal(self) -> SolvencyProof:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> SolvencyProof:
        """Add ``party``'s signature over the content hash (sealing first).

        A proof is signed by whoever folded it — the poster proving its own solvency, or a
        counterparty that independently re-folds the two attestations it was handed. Re-signing
        for the same party replaces its prior signature, so a proof cannot accumulate stale
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

    def _margin_sound(self) -> bool:
        """The margin re-derives from the proven figures and the breach re-derives from it."""
        if self.reserves_usd < -_TOLERANCE or self.liabilities_usd < -_TOLERANCE:
            return False
        if abs(self.margin_usd - _r6(self.reserves_usd - self.liabilities_usd)) > _TOLERANCE:
            return False
        expected = self._derive_breach()
        if (expected is None) != (self.breach is None):
            return False
        if expected is not None and self.breach is not None:
            if (
                self.breach.poster != expected.poster
                or self.breach.custodian != expected.custodian
                or self.breach.attestor != expected.attestor
                or self.breach.custody_hash != expected.custody_hash
                or self.breach.liability_hash != expected.liability_hash
                or abs(self.breach.reserves_usd - expected.reserves_usd) > _TOLERANCE
                or abs(self.breach.liabilities_usd - expected.liabilities_usd) > _TOLERANCE
                or abs(self.breach.shortfall_usd - expected.shortfall_usd) > _TOLERANCE
            ):
                return False
        return True

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> SolvencyProofVerification:
        """Verify the proof offline: the hash recomputes and the margin re-derives.

        Recomputes the content hash and re-derives the solvency margin from the proven
        reserves and liabilities and the insolvency breach from the margin — so a tampered
        margin or a flipped solvency verdict is caught even when the hash was recomputed to
        match. ``verifier`` additionally checks each signature against the content hash;
        ``require`` names parties that must have a verified signature (defaults to none).
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        margin_sound = self._margin_sound()
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
        valid = hash_ok and margin_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "proof is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the solvency facts"
            elif not margin_sound:
                reason = "solvency margin or insolvency breach does not re-derive"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return SolvencyProofVerification(
            valid=valid,
            hash_ok=hash_ok,
            margin_sound=margin_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> SolvencyProof:
        """Verify and raise :class:`SettlementError` if the proof is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"solvency proof {self.id} failed verification: {result.reason}",
                details={"proof_id": self.id, "reason": result.reason},
            )
        return self

    def require_solvent(self) -> SolvencyProof:
        """Raise :class:`SettlementError` if the proven liabilities exceed the reserves.

        The strict-mode counterpart to inspecting :attr:`solvent`: a counterparty whose
        liability attestation proves more debt than its custody attestation proves reserves
        cannot be admitted to a new deal without resolving the insolvency first, and this
        pinpoints the custodian, the attestor, and the shortfall.
        """
        if self.breach is not None:
            raise SettlementError(
                f"solvency proof {self.id} is insolvent by ${self.breach.shortfall_usd:,.2f}: "
                f"{self.poster!r} owes ${self.liabilities_usd:,.2f} against "
                f"${self.reserves_usd:,.2f} proven reserves",
                details={
                    "proof_id": self.id,
                    "poster": self.poster,
                    "reserves_usd": self.reserves_usd,
                    "liabilities_usd": self.liabilities_usd,
                    "shortfall_usd": self.breach.shortfall_usd,
                },
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the proof for the audit chain."""
        return to_jsonable(
            {
                "proof_id": self.id,
                "poster": self.poster,
                "custodian": self.custodian,
                "attestor": self.attestor,
                "status": self.status,
                "reserves_usd": _r6(self.reserves_usd),
                "liabilities_usd": _r6(self.liabilities_usd),
                "margin_usd": _r6(self.margin_usd),
                "solvency_adjusted_held": self.solvency_adjusted_held,
                "shortfall_usd": (_r6(self.breach.shortfall_usd) if self.breach else 0.0),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> SolvencyProof:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the proven reserves, liabilities, and the solvency margin."""
        print(
            f"Solvency proof ({self.poster}): ${self.reserves_usd:,.2f} reserves − "
            f"${self.liabilities_usd:,.2f} liabilities = ${self.margin_usd:,.2f} margin "
            f"({self.solvency_adjusted_held:,.2f} free) — {self.status}"
        )
        if self.breach is not None:
            print(
                f"  ! insolvent by ${self.breach.shortfall_usd:,.2f}: proven liabilities "
                f"exceed the reserves"
            )


def _proven_reserves(
    custody: CustodyAttestation, *, poster: str, verifier: ChainSigner | None
) -> float:
    """Read a custody attestation's proven reserves, refusing a tampered or mismatched one.

    Reads only what it can recompute: an attestation whose content hash no longer recomputes
    or whose total no longer re-derives — a tampered reserve figure — is refused outright, and
    with a ``verifier`` a forged custodian signature is too. An attestation for a *different*
    poster cannot stand in for this poster's reserves and is refused. Returns ``reserves_usd``.
    """
    result = custody.verify(verifier)
    if not result.hash_ok or not result.reserves_sound:
        raise SettlementError(
            f"custody attestation {custody.id} is tampered ({result.reason}); refusing to read "
            "it as proof-of-reserves",
            details={"attestation_id": custody.id, "reason": result.reason},
        )
    if verifier is not None and custody.signatures and not result.signatures_ok:
        raise SettlementError(
            f"custody attestation {custody.id} has an invalid custodian signature; refusing to "
            "read it as proof-of-reserves",
            details={"attestation_id": custody.id},
        )
    if custody.poster != poster:
        raise SettlementError(
            f"custody attestation {custody.id} attests reserves for {custody.poster!r}, not the "
            f"poster {poster!r} the solvency proof folds; refusing it",
            details={"attestation_id": custody.id, "attests": custody.poster, "poster": poster},
        )
    return _r6(custody.reserves_usd)


def _proven_liabilities(
    liabilities: LiabilityAttestation, *, poster: str, verifier: ChainSigner | None
) -> float:
    """Read a liability attestation's proven total, refusing a tampered or mismatched one.

    The liability analogue of :func:`_proven_reserves`: an attestation whose hash no longer
    recomputes or whose total no longer re-derives is refused, with a ``verifier`` a forged
    attestor signature is too, and an attestation for a *different* poster is refused. Returns
    ``liabilities_usd``.
    """
    result = liabilities.verify(verifier)
    if not result.hash_ok or not result.liabilities_sound:
        raise SettlementError(
            f"liability attestation {liabilities.id} is tampered ({result.reason}); refusing to "
            "read it as proof-of-liabilities",
            details={"attestation_id": liabilities.id, "reason": result.reason},
        )
    if verifier is not None and liabilities.signatures and not result.signatures_ok:
        raise SettlementError(
            f"liability attestation {liabilities.id} has an invalid attestor signature; refusing "
            "to read it as proof-of-liabilities",
            details={"attestation_id": liabilities.id},
        )
    if liabilities.poster != poster:
        raise SettlementError(
            f"liability attestation {liabilities.id} attests liabilities for "
            f"{liabilities.poster!r}, not the poster {poster!r} the solvency proof folds; "
            "refusing it",
            details={
                "attestation_id": liabilities.id,
                "attests": liabilities.poster,
                "poster": poster,
            },
        )
    return _r6(liabilities.liabilities_usd)


def prove_solvency(
    custody: CustodyAttestation,
    liabilities: LiabilityAttestation,
    *,
    poster: str | None = None,
    as_of: datetime | None = None,
    verifier: ChainSigner | None = None,
) -> SolvencyProof:
    """Fold a reserve proof against a liability proof into a proof-of-solvency.

    The proof-of-solvency the literature pairs with a proof-of-reserves: ``reserves ≥
    liabilities``. Reads a poster's :class:`~vincio.settlement.custody.CustodyAttestation`
    (proven reserves) and its :class:`LiabilityAttestation` (proven obligations), refusing
    either if its content hash no longer recomputes or its total no longer re-derives (a forged
    signature too, with ``verifier``), and reconciles them into a bounded solvency
    ``margin_usd`` (``reserves − liabilities``). When the liabilities exceed the reserves the
    shortfall is pinpointed as an :class:`InsolvencyBreach`. Returns a sealed, unsigned
    :class:`SolvencyProof` whose :attr:`~SolvencyProof.solvency_adjusted_held` the guard reads
    (``solvency=``) as the held figure — a pledge bounded against capital not already owed.

    ``poster`` is the counterparty both attestations are about (defaults to the poster they
    share; an explicit poster is required when they differ, and both must attest it). Raises
    :class:`SettlementError` when the two attestations attest different posters and none is
    given, or when either is tampered, forged, or for the wrong poster.
    """
    resolved_poster = poster
    if resolved_poster is None:
        if custody.poster != liabilities.poster:
            raise SettlementError(
                "prove_solvency needs an explicit poster: the custody attestation attests "
                f"{custody.poster!r} but the liability attestation attests "
                f"{liabilities.poster!r}",
                details={"custody_poster": custody.poster, "liability_poster": liabilities.poster},
            )
        resolved_poster = custody.poster
    reserves_usd = _proven_reserves(custody, poster=resolved_poster, verifier=verifier)
    liabilities_usd = _proven_liabilities(liabilities, poster=resolved_poster, verifier=verifier)
    proof = SolvencyProof(
        poster=resolved_poster,
        custodian=custody.custodian,
        attestor=liabilities.attestor,
        custody_hash=custody.content_hash,
        liability_hash=liabilities.content_hash,
        reserves_usd=reserves_usd,
        liabilities_usd=liabilities_usd,
        margin_usd=_r6(reserves_usd - liabilities_usd),
        as_of=as_of or utcnow(),
    )
    proof.breach = proof._derive_breach()
    return proof.seal()
