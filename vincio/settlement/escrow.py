"""Cross-org collateralized settlement & escrow.

Admission now sets a required collateral / escrow fraction on a thin or low-trust
counterparty's contract — but the fraction is still only a *number stamped on the
terms*; nothing **holds** it, releases it on a clean delivery, or forfeits a bounded
slice of it on a breach. A counterparty admitted on conservative terms posts no actual
collateral, so the escrow the admission policy asked for has no teeth, and a breach is
still only debited to reputation after the fact. This module makes the posted collateral
a **verifiable, offline escrow bound to the contract** — held against delivery and
settled deterministically — so the conservative terms a thin standing is admitted on are
backed by something, not merely recorded.

* **Posted, content-bound escrow.** An :class:`Escrow` binds an admission-required
  collateral amount to a *specific* :class:`~vincio.negotiation.Contract` and
  counterparty into a signed, offline-verifiable artifact — the escrow analogue of a
  :class:`~vincio.settlement.record.SettlementRecord`. The amount it holds re-derives
  from the admission-required ``escrow_fraction`` and the contract price, the contract it
  backs is pinned by hash, and the whole thing recomputes from the bytes alone, so what
  was posted is a mechanical, reconstructable number, never a custodian's ledger entry.
* **Deterministic release & forfeiture.** Settling the contract **releases** the escrow
  on a fulfilled delivery (the whole stake back to the poster) and **forfeits** a
  bounded, pinpointed slice on a breach — proportional to the shortfall the settlement
  measured, never the whole stake, never punitive — with the remainder released. The
  outcome is driven by the *same* :class:`~vincio.settlement.record.SettlementRecord`
  verdict the books already close on, so the collateral closes the same loop the
  settlement does rather than judging the delivery a second time.
* **Auditable & offline.** Every post, release, and forfeiture binds the contract, the
  amount, and the verdict onto a content hash; :meth:`Escrow.verify` recomputes it and
  re-derives the disposition (a tampered forfeiture is caught even after re-sealing,
  exactly as a tampered admission ceiling is), and
  :meth:`~vincio.core.app.ContextApp.post_escrow` /
  :meth:`~vincio.core.app.ContextApp.settle_escrow` land each transition on the
  hash-chained audit log — so an escrow's whole lifecycle is reconstructable offline,
  never a trusted third party.

The escrow folds into the *existing* settlement path: :func:`post_escrow` reads the
collateral straight off the :class:`~vincio.settlement.admission.AdmissionDecision`
(or the admission posture :meth:`AdmissionDecision.apply_to_terms` stamped onto a
contract's terms), and :meth:`~vincio.settlement.book.SettlementBook.settle` resolves an
attached escrow against the record it produces in one call. Everything is
dependency-free, deterministic, and offline — a verifiable escrow over the standing the
fabric already earns, never a hosted escrow service.
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
    "EscrowState",
    "EscrowConfig",
    "EscrowSignature",
    "EscrowVerification",
    "Escrow",
    "post_escrow",
    "settle_escrow",
]

# The single audit action every escrow transition is recorded under; the decision
# field carries the state (``posted`` / ``released`` / ``forfeited``).
ESCROW_ACTION = "escrow"

EscrowState = Literal["posted", "released", "forfeited"]

# The contract dimensions a breach is measured against, mapped to their reconciliation
# line so a forfeiture pinpoints which term the delivery missed.
_BREACH_DIMENSIONS = ("price", "sla", "quality")

_TOLERANCE = 1e-6


class EscrowConfig(BaseModel):
    """How a breach's measured shortfall maps to a bounded forfeiture.

    Forfeiture is **proportional** to the shortfall the settlement measured and
    **bounded** so it is never punitive: the forfeited fraction of the posted stake is
    the per-dimension shortfall (how far delivery missed the worst breached term, as a
    fraction of that term), clamped into ``[0, max_forfeit_fraction]``. A clean delivery
    forfeits nothing (the whole stake is released); a partial breach forfeits only the
    missed proportion and releases the rest; ``max_forfeit_fraction`` caps the slice so a
    catastrophic breach still leaves a residual when set below ``1``.

    * ``max_forfeit_fraction`` is the most of the posted stake a single breach can
      forfeit. The default ``1`` keeps forfeiture purely proportional (a total miss
      forfeits the whole stake); set it below ``1`` to guarantee a residual is always
      released to the poster.
    """

    max_forfeit_fraction: float = 1.0

    def validate_coherent(self) -> EscrowConfig:
        """Raise :class:`SettlementError` unless the configuration is coherent."""
        if not 0.0 < self.max_forfeit_fraction <= 1.0:
            raise SettlementError(
                f"max_forfeit_fraction must be in (0, 1]; got {self.max_forfeit_fraction}",
                details={"max_forfeit_fraction": self.max_forfeit_fraction},
            )
        return self

    def canonical(self) -> dict[str, float]:
        """A stable, hashable projection of the policy the escrow binds."""
        return {"max_forfeit_fraction": round(float(self.max_forfeit_fraction), 9)}


class EscrowSignature(BaseModel):
    """One party's signature over an escrow's content hash."""

    party: str
    signature: str
    key_id: str = ""


class EscrowVerification(BaseModel):
    """The (non-raising) outcome of verifying an escrow offline.

    An escrow is **valid** when its content hash recomputes (``hash_ok``), its
    disposition re-derives from the posted amount under its policy (``terms_sound`` —
    the held amount matches the admission-required collateral and the released /
    forfeited split matches the bound shortfall), and, with a ``verifier``, every
    signature checks (``signatures_ok``). A tampered amount or forfeiture is caught from
    the bytes alone, even after the hash was recomputed to match.
    """

    valid: bool
    hash_ok: bool
    terms_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class Escrow(BaseModel):
    """Posted collateral bound to a contract — held, released, or forfeited.

    Produced by :func:`post_escrow` (or
    :meth:`~vincio.core.app.ContextApp.post_escrow`) from a contract and the
    admission-required collateral fraction. It binds the contract it backs (by id and
    content hash), the parties, the posted ``amount_usd``, and the admission posture that
    asked for it onto a content hash, so the collateral is a mechanical number anyone
    recomputes. :meth:`resolve` settles it against the contract's
    :class:`~vincio.settlement.record.SettlementRecord` — releasing the whole stake on a
    fulfilled delivery and forfeiting a bounded, proportional slice on a breach — and
    :meth:`verify` re-derives the disposition from the bytes alone.

    The poster (the counterparty admitted on conservative terms — by default the
    contract's seller) backs its own delivery; the beneficiary (by default the buyer) is
    made whole up to the forfeited slice on a breach. No money moves: the escrow is a
    verifiable record of what was posted and how it settled, not a payment rail.
    """

    id: str = Field(default_factory=lambda: new_id("escrow"))
    contract_id: str
    contract_hash: str = ""
    buyer: str
    seller: str
    poster: str
    beneficiary: str
    scope: str = ""

    # The posted collateral and how it was derived (the admission posture that asked
    # for it), bound so the amount re-derives from the fraction and the contract price.
    price_usd: float = 0.0
    escrow_fraction: float = 0.0
    amount_usd: float = 0.0
    decision_id: str | None = None
    decision_hash: str = ""
    config: EscrowConfig = Field(default_factory=EscrowConfig)

    # The disposition (set on resolution; ``posted`` holds the whole stake).
    state: EscrowState = "posted"
    shortfall_fraction: float = 0.0
    forfeited_usd: float = 0.0
    released_usd: float = 0.0
    breaches: list[str] = Field(default_factory=list)

    # The settlement the disposition was driven by (bound on resolution; the economic
    # verdict, not the local id, anchors the hash).
    settlement_id: str | None = None
    settlement_hash: str = ""

    posted_at: datetime = Field(default_factory=utcnow)
    resolved_at: datetime | None = None
    content_hash: str = ""
    signatures: list[EscrowSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def is_posted(self) -> bool:
        """Whether the collateral is still held (not yet resolved)."""
        return self.state == "posted"

    @property
    def is_released(self) -> bool:
        """Whether the whole stake was released (a clean delivery)."""
        return self.state == "released"

    @property
    def is_forfeited(self) -> bool:
        """Whether a slice of the stake was forfeited (a breach)."""
        return self.state == "forfeited"

    @property
    def is_resolved(self) -> bool:
        """Whether the escrow has settled (released or forfeited)."""
        return self.state != "posted"

    @property
    def parties(self) -> tuple[str, str]:
        """The two parties to the escrow (buyer and seller of the contract)."""
        return (self.buyer, self.seller)

    # -- the disposition map (the escrow's mechanical core) -----------------

    def _expected_disposition(self) -> tuple[float, float]:
        """The (forfeited, released) split a sound escrow must carry, re-derived.

        A held escrow forfeits and releases nothing. A resolved one forfeits the bound
        shortfall (clamped into ``[0, max_forfeit_fraction]``) of the posted amount and
        releases the remainder — so :meth:`verify` recomputes the split from the bytes.
        """
        amount = round(float(self.amount_usd), 6)
        if self.state == "posted":
            return 0.0, 0.0
        fraction = min(max(0.0, self.shortfall_fraction), self.config.max_forfeit_fraction)
        forfeited = round(amount * fraction, 6)
        released = round(amount - forfeited, 6)
        return forfeited, released

    def _amount_sound(self) -> bool:
        """The posted amount re-derives from the admission-required collateral.

        When the escrow was posted from a fraction (an
        :class:`~vincio.settlement.admission.AdmissionDecision` or an explicit one) the
        amount must equal that fraction of the contract price, so a tampered amount is
        caught even after re-sealing. A flat amount (no fraction) is authoritative as
        posted — bound by the hash, with nothing to re-derive against.
        """
        if self.escrow_fraction <= 0.0:
            return True
        expected = round(self.escrow_fraction * self.price_usd, 6)
        return abs(self.amount_usd - expected) <= _TOLERANCE

    # -- hashing ------------------------------------------------------------

    def escrow_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the collateral and its disposition.

        Excludes the id, timestamps, the decision / settlement local ids, the
        signatures, and the audit linkage (local metadata, not the escrow), so the same
        collateral settled the same way against the same contract hashes identically
        wherever it is recomputed — the way two parties co-sign one reconciliation hash.
        """
        return {
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "buyer": self.buyer,
            "seller": self.seller,
            "poster": self.poster,
            "beneficiary": self.beneficiary,
            "scope": self.scope,
            "price_usd": round(float(self.price_usd), 9),
            "escrow_fraction": round(float(self.escrow_fraction), 9),
            "amount_usd": round(float(self.amount_usd), 6),
            "decision_hash": self.decision_hash,
            "config": self.config.canonical(),
            "state": self.state,
            "shortfall_fraction": round(float(self.shortfall_fraction), 9),
            "forfeited_usd": round(float(self.forfeited_usd), 6),
            "released_usd": round(float(self.released_usd), 6),
            "breaches": list(self.breaches),
            "settlement_hash": self.settlement_hash,
        }

    def compute_hash(self) -> str:
        """The content hash binding the collateral, the contract, and the disposition."""
        return stable_hash(self.escrow_facts(), length=32)

    def seal(self) -> Escrow:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    @property
    def fully_signed(self) -> bool:
        """Both the buyer and the seller have signed the escrow."""
        signers = set(self.signed_by)
        return self.buyer in signers and self.seller in signers

    def sign(self, signer: ChainSigner, *, party: str) -> Escrow:
        """Add ``party``'s signature over the content hash (sealing first).

        Only the buyer or the seller of the backed contract can sign. Re-signing for the
        same party replaces its prior signature, so an escrow cannot accumulate stale
        signatures for one identity.
        """
        if party not in (self.buyer, self.seller):
            raise SettlementError(
                f"party {party!r} is neither the buyer nor the seller of escrow {self.id}",
                details={"escrow_id": self.id, "party": party},
            )
        if not self.content_hash:
            self.seal()
        sig = EscrowSignature(
            party=party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != party]
        self.signatures.append(sig)
        return self

    # -- verification -------------------------------------------------------

    def _terms_sound(self) -> bool:
        """The posted amount and disposition re-derive from the bytes under the policy."""
        exp_forfeited, exp_released = self._expected_disposition()
        return (
            self._amount_sound()
            and abs(self.forfeited_usd - exp_forfeited) <= _TOLERANCE
            and abs(self.released_usd - exp_released) <= _TOLERANCE
        )

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> EscrowVerification:
        """Verify the escrow offline: the hash recomputes and the disposition re-derives.

        Recomputes the content hash and re-derives the disposition — the held amount
        matches the admission-required collateral, and the released / forfeited split
        matches the bound shortfall — so a tampered amount or forfeiture is caught even
        when the hash was recomputed to match. ``verifier`` additionally checks each
        signature against the content hash; ``require`` names the parties that must have
        a verified signature (defaults to none — pass ``[poster]`` to demand the poster's
        signature). Pass no verifier to check the binding and disposition alone.
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        terms_sound = self._terms_sound()
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
        valid = hash_ok and terms_sound and signatures_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "escrow is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the escrow facts"
            elif not terms_sound:
                reason = "amount or disposition does not re-derive from the bound collateral"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return EscrowVerification(
            valid=valid,
            hash_ok=hash_ok,
            terms_sound=terms_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> Escrow:
        """Verify and raise :class:`SettlementError` if the escrow is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"escrow {self.id} failed verification: {result.reason}",
                details={"escrow_id": self.id, "reason": result.reason},
            )
        return self

    # -- resolution ---------------------------------------------------------

    def resolve(self, record: Any, *, config: EscrowConfig | None = None) -> Escrow:
        """Settle the escrow against the contract's settlement record.

        Releases the whole stake on a fulfilled delivery and forfeits a bounded,
        proportional slice on a breach — driven by the *same*
        :class:`~vincio.settlement.record.SettlementRecord` verdict the books close on,
        so the collateral never judges the delivery a second time. The forfeited slice is
        the shortfall the settlement measured (how far delivery missed the worst breached
        term), clamped by ``max_forfeit_fraction``; the remainder is released. Clears any
        signatures (the content hash changes on resolution) and re-seals — the caller
        signs the resolved escrow. Raises :class:`SettlementError` if the record is for a
        different contract or the escrow is already resolved.
        """
        if self.is_resolved:
            raise SettlementError(
                f"escrow {self.id} is already resolved ({self.state})",
                details={"escrow_id": self.id, "state": self.state},
            )
        record_contract = getattr(record, "contract_id", None)
        if record_contract != self.contract_id:
            raise SettlementError(
                f"cannot resolve escrow for contract {self.contract_id!r} against a "
                f"settlement for {record_contract!r}",
                details={"escrow_id": self.id, "contract_id": self.contract_id},
            )
        if config is not None:
            self.config = config.validate_coherent()
        fulfilled = bool(getattr(record, "fulfilled", True))
        shortfall, breaches = _shortfall_from_record(record)
        if fulfilled:
            shortfall, breaches = 0.0, []
        self.shortfall_fraction = round(shortfall, 9)
        self.breaches = breaches
        forfeit_fraction = min(shortfall, self.config.max_forfeit_fraction)
        amount = round(float(self.amount_usd), 6)
        self.forfeited_usd = round(amount * forfeit_fraction, 6)
        self.released_usd = round(amount - self.forfeited_usd, 6)
        self.state = "forfeited" if self.forfeited_usd > 0.0 else "released"
        self.settlement_id = getattr(record, "id", None)
        self.settlement_hash = getattr(record, "content_hash", "") or ""
        self.resolved_at = utcnow()
        self.signatures = []
        return self.seal()

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the escrow for the audit chain."""
        return to_jsonable(
            {
                "escrow_id": self.id,
                "contract_id": self.contract_id,
                "buyer": self.buyer,
                "seller": self.seller,
                "poster": self.poster,
                "beneficiary": self.beneficiary,
                "amount_usd": round(float(self.amount_usd), 6),
                "escrow_fraction": round(float(self.escrow_fraction), 9),
                "state": self.state,
                "shortfall_fraction": round(float(self.shortfall_fraction), 9),
                "forfeited_usd": round(float(self.forfeited_usd), 6),
                "released_usd": round(float(self.released_usd), 6),
                "breaches": list(self.breaches),
                "settlement_id": self.settlement_id,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Escrow:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the posted collateral and how it settled."""
        if self.is_posted:
            print(
                f"Escrow {self.poster}→{self.contract_id}: ${self.amount_usd:,.2f} held "
                f"({self.escrow_fraction:.0%} of ${self.price_usd:,.2f}) — posted"
            )
            return
        verb = "forfeited" if self.is_forfeited else "released"
        detail = (
            f"${self.forfeited_usd:,.2f} forfeited to {self.beneficiary}, "
            f"${self.released_usd:,.2f} released to {self.poster}"
            if self.is_forfeited
            else f"${self.released_usd:,.2f} released to {self.poster}"
        )
        pinned = f" ({', '.join(self.breaches)})" if self.breaches else ""
        print(f"Escrow {self.poster}→{self.contract_id}: {verb} — {detail}{pinned}")


# -- module-level builders -----------------------------------------------------


def _resolve_collateral(
    contract: Any,
    *,
    decision: Any | None,
    fraction: float | None,
    amount: float | None,
    source: str = "post_escrow",
) -> tuple[float, float, float, str | None, str]:
    """Resolve the collateral a contract must post, from its admission posture.

    The shared collateral-source resolver for :func:`post_escrow` and the
    :class:`~vincio.settlement.collateral.CollateralPool`: returns the contract
    ``price``, the resolved ``fraction`` and ``amount`` of collateral, and the
    admission decision id / hash it was read from. The amount comes from, in order: an
    explicit ``amount`` (a flat stake), an explicit ``fraction`` of the contract price,
    an :class:`~vincio.settlement.admission.AdmissionDecision`'s ``escrow_fraction``
    (``decision``), or the admission posture stamped onto the contract's terms metadata.
    Raises :class:`SettlementError` when no source is given and the terms carry no
    admission posture, or when the resolved fraction / amount is negative.
    """
    terms = contract.terms
    price = round(float(getattr(terms, "price_usd", 0.0)), 9)
    decision_id: str | None = None
    decision_hash = ""
    resolved_fraction = 0.0

    if amount is not None:
        resolved_amount = round(float(amount), 6)
    elif fraction is not None:
        resolved_fraction = float(fraction)
        resolved_amount = round(resolved_fraction * price, 6)
    elif decision is not None:
        resolved_fraction = float(getattr(decision, "escrow_fraction", 0.0))
        resolved_amount = round(resolved_fraction * price, 6)
        decision_id = getattr(decision, "id", None)
        decision_hash = getattr(decision, "content_hash", "") or ""
    else:
        stamp = dict(getattr(terms, "metadata", {}) or {}).get("admission") or {}
        if "escrow_fraction" not in stamp:
            raise SettlementError(
                f"{source} needs a collateral source: pass amount=, fraction=, "
                "decision=, or a contract whose terms carry an admission posture",
                details={"contract_id": getattr(contract, "id", None)},
            )
        resolved_fraction = float(stamp["escrow_fraction"])
        resolved_amount = round(resolved_fraction * price, 6)
        decision_id = stamp.get("decision_id")

    if resolved_fraction < 0.0:
        raise SettlementError(
            f"collateral fraction must be non-negative; got {resolved_fraction}",
            details={"contract_id": getattr(contract, "id", None)},
        )
    if resolved_amount < 0.0:
        raise SettlementError(
            f"collateral amount must be non-negative; got {resolved_amount}",
            details={"contract_id": getattr(contract, "id", None)},
        )
    return price, resolved_fraction, resolved_amount, decision_id, decision_hash


def post_escrow(
    contract: Any,
    *,
    decision: Any | None = None,
    fraction: float | None = None,
    amount: float | None = None,
    poster: str | None = None,
    beneficiary: str | None = None,
    config: EscrowConfig | None = None,
) -> Escrow:
    """Post collateral against a contract into an (unsigned) :class:`Escrow`.

    The escrow analogue of :func:`~vincio.settlement.book.settle_contract`: resolves the
    collateral to hold and binds it to the contract. The amount comes from, in order: an
    explicit ``amount`` (a flat stake), an explicit ``fraction`` of the contract price, a
    :class:`~vincio.settlement.admission.AdmissionDecision`'s ``escrow_fraction``
    (``decision``), or the admission posture
    :meth:`~vincio.settlement.admission.AdmissionDecision.apply_to_terms` stamped onto
    the contract's terms metadata. The poster (the counterparty backing its delivery)
    defaults to the contract's seller and the beneficiary to the buyer. Returns a sealed,
    unsigned escrow — sign it with each party's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.post_escrow`). Raises
    :class:`SettlementError` when no collateral source is given and the contract carries
    no admission posture to read one from.
    """
    price, resolved_fraction, resolved_amount, decision_id, decision_hash = _resolve_collateral(
        contract, decision=decision, fraction=fraction, amount=amount, source="post_escrow"
    )
    terms = contract.terms
    escrow = Escrow(
        contract_id=contract.id,
        contract_hash=getattr(contract, "content_hash", "") or "",
        buyer=contract.buyer,
        seller=contract.seller,
        poster=poster or contract.seller,
        beneficiary=beneficiary or contract.buyer,
        scope=getattr(terms, "scope", ""),
        price_usd=price,
        escrow_fraction=round(resolved_fraction, 9),
        amount_usd=resolved_amount,
        decision_id=decision_id,
        decision_hash=decision_hash,
        config=(config or EscrowConfig()).validate_coherent(),
    )
    return escrow.seal()


def settle_escrow(escrow: Escrow, record: Any, *, config: EscrowConfig | None = None) -> Escrow:
    """Resolve a posted escrow against a settlement record (release or forfeit).

    The module-level convenience over :meth:`Escrow.resolve`: settles ``escrow`` against
    the contract's :class:`~vincio.settlement.record.SettlementRecord`, releasing the
    whole stake on a fulfilled delivery and forfeiting a bounded, proportional slice on a
    breach. Returns the re-sealed (unsigned) escrow.
    """
    return escrow.resolve(record, config=config)


def _shortfall_from_record(record: Any) -> tuple[float, list[str]]:
    """The breach shortfall ``∈ [0, 1]`` and pinpointed dimensions from a record.

    Reads the record's reconciliation lines: for each breached, metered dimension the
    shortfall is how far delivery missed the term as a fraction of it (a cost overrun, a
    latency over the SLA, a quality under the floor), clamped into ``[0, 1]``. The
    forfeiture is proportional to the **worst** breach (the maximum across dimensions),
    and the breached dimensions are returned so a forfeiture pinpoints which terms the
    delivery missed. A fulfilled or un-metered delivery has no shortfall.
    """
    severities: list[float] = []
    breached: list[str] = []
    for line in getattr(record, "lines", []) or []:
        if getattr(line, "within", True):
            continue
        dimension = getattr(line, "dimension", "")
        if dimension in _BREACH_DIMENSIONS:
            breached.append(dimension)
        owed = float(getattr(line, "owed", 0.0) or 0.0)
        delta = getattr(line, "delta", None)
        if owed <= 0.0 or delta is None:
            continue
        # ``delta`` is the signed slack: negative when the term was missed, by exactly
        # the overrun (price/SLA) or the shortfall below the floor (quality).
        miss = max(0.0, -float(delta))
        severities.append(min(1.0, miss / owed))
    shortfall = max(severities) if severities else 0.0
    return shortfall, sorted(set(breached))
