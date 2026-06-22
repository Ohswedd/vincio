"""Cross-org collateral rehypothecation guards & re-use bounds.

A :class:`~vincio.settlement.collateral.CollateralPool` lets a counterparty back many
contracts with one posted stake, freeing capital as clean deliveries release it — but a
pool only ever *re-allocates* the freed capital **within the same pool**. The next reach
is the dual risk: when a counterparty pledges the *same* collateral across more than one
pool (or re-pledges collateral a beneficiary already has a claim on), nothing bounds the
**re-use**. A stake double-counted across pools over-states what actually backs each deal —
the collateral analogue of a :class:`~vincio.settlement.record.SettlementRecord`
double-counted before :func:`~vincio.settlement.netting.net_settlements` deduplicated it.
This module is the **rehypothecation guard**: a bounded, offline-verifiable check that a
posted stake is not committed beyond what it holds across the pools that draw on it.

* **Cross-pool re-use bound.** A :class:`CollateralLedger` folds a counterparty's
  :class:`~vincio.settlement.collateral.CollateralPool`\\ s into one view of what its
  posted capital is committed to across pools. It surfaces an over-commitment — the same
  capital pledged twice — as a bounded, pinpointed :class:`ReuseBreach` (a contract backed
  by more than one pool, its collateral provably double-pledged) rather than silently
  over-stating coverage, and reconciles the total pledged against the capital the poster
  actually ``held`` (``reuse_usd`` is what the pledges exceed the holdings by).
* **Beneficiary-claim priority.** When a stake backs deals for more than one beneficiary,
  the guard bounds each beneficiary's claim to its deterministic share of the held capital
  (pari passu, proportional to the capital pledged to it), so a forfeiture cannot pay one
  beneficiary out of capital another has first claim on. Each
  :class:`BeneficiaryClaim` carries the ``secured_usd`` it is actually covered for and the
  ``unsecured_usd`` the over-commitment leaves it exposed to.
* **Proof-of-reserves.** The ``held`` figure was the one input the guard *trusted* — it was
  **asserted**, not proven, so a counterparty over-stating its reserves still passed. A
  :class:`~vincio.settlement.custody.CustodyAttestation` (``custody=``) makes the held capital
  itself **evidence-backed**: a signed, content-bound proof-of-reserves the guard reads as
  the ``held`` figure instead of the asserted default. When the proven reserves fall below
  what the pools pledge, the shortfall surfaces as a bounded, pinpointed
  :class:`UnderReservedBreach`, the way an over-commitment does — and a custody attestation
  for a different poster, a tampered reserve figure, or (with a verifier) a forged custodian
  is **refused**.
* **Auditable & offline.** The ledger reads only the existing signed, content-bound pools
  (and the signed custody attestation) and asserts nothing it cannot recompute: a tampered
  pool (its content hash no longer recomputes) is **refused** at fold time, and
  :meth:`CollateralLedger.verify` re-derives the pledged total, the re-use breaches, the
  beneficiary apportionment, and the under-reserved breach from the bytes alone (a tampered
  figure is caught even after re-sealing). The settlement path lands each guard on the
  hash-chained audit log — never a trusted third party.

The guard folds over the *existing* collateral machinery:
:func:`guard_collateral` reads a counterparty's
:class:`~vincio.settlement.collateral.CollateralPool`\\ s the way
:func:`~vincio.settlement.netting.net_settlements` reads a fleet's books, and
:meth:`~vincio.settlement.book.SettlementBook.guard_collateral` /
:meth:`~vincio.core.app.ContextApp.guard_collateral` build, sign, and audit one in a single
call. Everything is dependency-free, deterministic, and offline — a verifiable re-use bound
over the collateral the fabric already pools, never a hosted custodian or a clearing house.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .collateral import CollateralPool
from .custody import CustodyAttestation
from .record import SettlementSignature

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "LedgerContract",
    "LedgerPool",
    "ReuseBreach",
    "BeneficiaryClaim",
    "UnderReservedBreach",
    "CollateralLedgerVerification",
    "CollateralLedger",
    "guard_collateral",
]

# The single audit action a rehypothecation guard is recorded under; the decision field
# carries whether the ledger is over-committed (``over_committed`` / ``within_bounds``).
REHYPOTHECATION_ACTION = "rehypothecation"

_TOLERANCE = 1e-6


def _r6(value: float) -> float:
    """Round a money figure to six places so float drift never breaks a balance."""
    return round(float(value), 6)


class LedgerContract(BaseModel):
    """One open contract a pooled stake is pledged to, as the ledger reads it.

    A contract is *open* in a :class:`~vincio.settlement.collateral.CollateralPool` when it
    has not yet settled, so the pool still earmarks ``allocated_usd`` of its stake to back
    it for ``beneficiary``. The ledger folds these across pools: a contract that appears in
    more than one pool is a re-pledge (its collateral counted twice), and a beneficiary's
    claim is the capital pledged to it across every pool.
    """

    contract_id: str
    beneficiary: str
    allocated_usd: float = 0.0
    required_usd: float = 0.0


class LedgerPool(BaseModel):
    """One :class:`~vincio.settlement.collateral.CollateralPool` folded into the ledger.

    Carries the per-pool figures the ledger reconciles — the posted stake, the live
    ``balance_usd`` (what the pool still holds: posted minus drawn forfeitures, the capital
    pledged to this pool that cannot simultaneously back another), the ``committed_usd``
    earmarked to its open contracts, and those open contracts — plus the pool's
    ``content_hash`` so the exact bytes the ledger read are bound into it (a tampered pool
    is refused at fold time, the way :func:`~vincio.settlement.netting.net_settlements`
    refuses a tampered settlement record).
    """

    pool_id: str
    poster: str
    posted_usd: float = 0.0
    balance_usd: float = 0.0
    committed_usd: float = 0.0
    status: str = "posted"
    content_hash: str = ""
    open_contracts: list[LedgerContract] = Field(default_factory=list)

    def facts(self) -> dict[str, Any]:
        """The per-pool facts the ledger's content hash binds."""
        return {
            "pool_id": self.pool_id,
            "poster": self.poster,
            "posted_usd": _r6(self.posted_usd),
            "balance_usd": _r6(self.balance_usd),
            "committed_usd": _r6(self.committed_usd),
            "status": self.status,
            "content_hash": self.content_hash,
            "open_contracts": [
                {
                    "contract_id": c.contract_id,
                    "beneficiary": c.beneficiary,
                    "allocated_usd": _r6(c.allocated_usd),
                    "required_usd": _r6(c.required_usd),
                }
                for c in sorted(self.open_contracts, key=lambda c: c.contract_id)
            ],
        }


class ReuseBreach(BaseModel):
    """A contract pledged across more than one pool — the same collateral, twice.

    The pinpointed evidence of re-use: ``contract_id`` is backed by every pool in ``pools``
    (two or more), so ``pledged_usd`` (the capital they collectively earmark to it)
    over-states the ``secured_usd`` the deal actually needs once — by ``excess_usd``, the
    provably double-pledged amount. A forfeiture against this contract can be honored only
    once, so the excess is capital the poster has committed but does not separately hold.
    """

    contract_id: str
    beneficiary: str
    pools: list[str] = Field(default_factory=list)
    secured_usd: float = 0.0
    pledged_usd: float = 0.0
    excess_usd: float = 0.0


class BeneficiaryClaim(BaseModel):
    """One beneficiary's bounded claim on the poster's held capital.

    ``claim_usd`` is the capital pledged to ``beneficiary`` across the poster's pools (its
    distinct backed contracts, counted once each). Under scarcity the guard bounds it to
    ``secured_usd`` — the beneficiary's deterministic, pari-passu share of the held capital
    (proportional to its claim) — leaving ``unsecured_usd`` exposed, so a forfeiture cannot
    pay one beneficiary out of capital another has first claim on. ``share`` is
    ``secured_usd / claim_usd`` (``1.0`` when fully covered).
    """

    beneficiary: str
    claim_usd: float = 0.0
    secured_usd: float = 0.0
    unsecured_usd: float = 0.0
    share: float = 1.0
    pools: list[str] = Field(default_factory=list)
    contracts: list[str] = Field(default_factory=list)

    @property
    def is_secured(self) -> bool:
        """Whether the beneficiary's whole claim is covered by held capital."""
        return self.unsecured_usd <= _TOLERANCE


class UnderReservedBreach(BaseModel):
    """A proven-reserves shortfall — the pools pledge more than the custodian attests.

    Surfaced only when the held capital is **proven** by a
    :class:`~vincio.settlement.custody.CustodyAttestation` (``custody=``) rather than
    asserted: the ``custodian`` attests ``reserves_usd`` of capital (the attestation pinned
    by ``attestation_hash``), but the pools collectively pledge ``pledged_usd``, so the
    poster is under-reserved by ``shortfall_usd`` — capital it has committed but the proof
    does not cover. The re-use guard's over-commitment was about the same contract pledged
    twice; this is the orthogonal risk the proof closes: the reserves simply do not back the
    pledges, double-pledged or not.
    """

    custodian: str
    attestation_hash: str = ""
    reserves_usd: float = 0.0
    pledged_usd: float = 0.0
    shortfall_usd: float = 0.0


class CollateralLedgerVerification(BaseModel):
    """The (non-raising) outcome of verifying a collateral ledger offline.

    A ledger is **valid** when its content hash recomputes (``hash_ok``), the pledged total,
    the re-use breaches, the held-capital reconciliation, the beneficiary apportionment, and
    the under-reserved breach re-derive from the per-pool figures and the proven reserves
    (``terms_sound``), and — with a ``verifier`` — every signature checks (``signatures_ok``).
    A tampered figure is caught from the bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    terms_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class CollateralLedger(BaseModel):
    """A poster's cross-pool rehypothecation view — a bounded re-use guard.

    Produced by :func:`guard_collateral` (or
    :meth:`~vincio.settlement.book.SettlementBook.guard_collateral` /
    :meth:`~vincio.core.app.ContextApp.guard_collateral`) from a counterparty's
    :class:`~vincio.settlement.collateral.CollateralPool`\\ s and the capital it actually
    ``held_usd``. It binds the ``poster``, the folded per-pool figures, and the reconciled
    totals onto a content hash, so the re-use bound is a mechanical number anyone recomputes.

    The ledger reconciles what the pools collectively pledge (``pledged_usd`` — the sum of
    their live balances) against the capital the poster holds (``held_usd``): the excess is
    ``reuse_usd``, surfaced when the same capital is pledged across pools. Every contract
    pledged by more than one pool is pinpointed as a :class:`ReuseBreach`, and each
    :class:`BeneficiaryClaim` is bounded to its deterministic share of the held capital, so
    a scarce stake is apportioned by priority rather than over-promised. When the held figure
    is **proven** by a :class:`~vincio.settlement.custody.CustodyAttestation`
    (``reserves_proven``) rather than asserted, a shortfall of the proven reserves below the
    pledges surfaces as an :class:`UnderReservedBreach`. :meth:`verify` re-derives all of it
    from the bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("collateral-ledger"))
    poster: str
    pools: list[LedgerPool] = Field(default_factory=list)
    pool_hashes: list[str] = Field(default_factory=list)

    # The reconciled totals (set on every recompute).
    posted_usd: float = 0.0
    pledged_usd: float = 0.0
    held_usd: float = 0.0
    available_usd: float = 0.0
    reuse_usd: float = 0.0
    duplicate_pledge_usd: float = 0.0

    # Proof-of-reserves: when the held figure is backed by a signed CustodyAttestation, the
    # custodian and the attestation hash are bound in, and a shortfall surfaces as a breach.
    reserves_proven: bool = False
    custodian: str = ""
    custody_hash: str = ""
    reserves_usd: float = 0.0

    breaches: list[ReuseBreach] = Field(default_factory=list)
    claims: list[BeneficiaryClaim] = Field(default_factory=list)
    reserve_breach: UnderReservedBreach | None = None

    folded_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def pool_ids(self) -> list[str]:
        """The ids of the pools this ledger folds, sorted."""
        return sorted(p.pool_id for p in self.pools)

    @property
    def beneficiaries(self) -> list[str]:
        """Every beneficiary a pooled stake is pledged to, sorted."""
        return sorted(c.beneficiary for c in self.claims)

    @property
    def over_committed(self) -> bool:
        """Whether the pools pledge more capital than the poster actually holds."""
        return self.reuse_usd > _TOLERANCE

    @property
    def within_bounds(self) -> bool:
        """Whether every pledge is covered by held capital (no re-use)."""
        return not self.over_committed

    @property
    def under_reserved(self) -> bool:
        """Whether the proven reserves fall below what the pools pledge.

        Meaningful only when the held figure is backed by a
        :class:`~vincio.settlement.custody.CustodyAttestation` (:attr:`reserves_proven`): an
        asserted holdings figure is not *proven*, so it cannot under-reserve — it can only
        over-commit. ``True`` exactly when a :attr:`reserve_breach` was surfaced.
        """
        return self.reserve_breach is not None

    @property
    def status(self) -> str:
        """``over_committed`` (re-use detected) or ``within_bounds`` (fully held)."""
        return "over_committed" if self.over_committed else "within_bounds"

    def breach(self, contract_id: str) -> ReuseBreach | None:
        """The re-use breach for a given contract, or ``None``."""
        return next((b for b in self.breaches if b.contract_id == contract_id), None)

    def claim(self, beneficiary: str) -> BeneficiaryClaim | None:
        """The bounded claim for a given beneficiary, or ``None``."""
        return next((c for c in self.claims if c.beneficiary == beneficiary), None)

    # -- the reconciliation (the ledger's mechanical core) ------------------

    def _recompute(self) -> None:
        """Re-derive the pledged total, the re-use breaches, and the beneficiary claims.

        The ledger's deterministic core, run once at fold time against the already-resolved
        :attr:`held_usd` (which it never overwrites — the holdings are an input the guard
        bounds the pledges by, not a figure it derives): the pledged total is the sum of the
        pools' live balances, a contract earmarked by more than one pool is a double-pledge
        (its excess the capital committed beyond the one claim it can honor), the held
        capital reconciles to the available balance and the re-use, and each beneficiary's
        claim is bounded to its pari-passu share of the held capital.
        """
        self.posted_usd = _r6(sum(p.posted_usd for p in self.pools))
        self.pledged_usd = _r6(sum(p.balance_usd for p in self.pools))

        breaches, duplicate_pledge = _reuse_breaches(self.pools)
        self.breaches = breaches
        self.duplicate_pledge_usd = _r6(duplicate_pledge)

        self.available_usd = _r6(self.held_usd - self.pledged_usd)
        self.reuse_usd = _r6(max(0.0, -self.available_usd))

        self.claims = _beneficiary_claims(self.pools, self.held_usd)
        self.reserve_breach = self._derive_reserve_breach()

    def _derive_reserve_breach(self) -> UnderReservedBreach | None:
        """The under-reserved breach when proven reserves fall below the pledges.

        Surfaced only when the held figure is **proven** (:attr:`reserves_proven`): an
        asserted holdings figure can over-commit but cannot under-*reserve*, because nothing
        proves the reserves exist. When a custody attestation backs the held figure and the
        proven reserves are below what the pools pledge, the shortfall is pinpointed against
        the custodian and the attestation that vouched for it.
        """
        if not self.reserves_proven:
            return None
        shortfall = _r6(max(0.0, self.pledged_usd - self.reserves_usd))
        if shortfall <= _TOLERANCE:
            return None
        return UnderReservedBreach(
            custodian=self.custodian,
            attestation_hash=self.custody_hash,
            reserves_usd=_r6(self.reserves_usd),
            pledged_usd=_r6(self.pledged_usd),
            shortfall_usd=shortfall,
        )

    # -- hashing ------------------------------------------------------------

    def ledger_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the pools, the totals, the apportionment.

        Excludes the id, timestamps, signatures, and audit linkage (local metadata), so the
        same pools reconciled against the same holdings hash identically wherever they are
        recomputed — the way two clearers co-sign one netting hash. Pools, breaches, and
        claims are sorted so the order they were folded in never changes the hash.
        """
        return {
            "poster": self.poster,
            "pool_hashes": sorted(self.pool_hashes),
            "posted_usd": _r6(self.posted_usd),
            "pledged_usd": _r6(self.pledged_usd),
            "held_usd": _r6(self.held_usd),
            "available_usd": _r6(self.available_usd),
            "reuse_usd": _r6(self.reuse_usd),
            "duplicate_pledge_usd": _r6(self.duplicate_pledge_usd),
            "reserves_proven": self.reserves_proven,
            "custodian": self.custodian,
            "custody_hash": self.custody_hash,
            "reserves_usd": _r6(self.reserves_usd),
            "reserve_breach": (
                {
                    "custodian": self.reserve_breach.custodian,
                    "attestation_hash": self.reserve_breach.attestation_hash,
                    "reserves_usd": _r6(self.reserve_breach.reserves_usd),
                    "pledged_usd": _r6(self.reserve_breach.pledged_usd),
                    "shortfall_usd": _r6(self.reserve_breach.shortfall_usd),
                }
                if self.reserve_breach is not None
                else None
            ),
            "pools": [p.facts() for p in sorted(self.pools, key=lambda p: p.pool_id)],
            "breaches": [
                {
                    "contract_id": b.contract_id,
                    "beneficiary": b.beneficiary,
                    "pools": sorted(b.pools),
                    "secured_usd": _r6(b.secured_usd),
                    "pledged_usd": _r6(b.pledged_usd),
                    "excess_usd": _r6(b.excess_usd),
                }
                for b in sorted(self.breaches, key=lambda b: b.contract_id)
            ],
            "claims": [
                {
                    "beneficiary": c.beneficiary,
                    "claim_usd": _r6(c.claim_usd),
                    "secured_usd": _r6(c.secured_usd),
                    "unsecured_usd": _r6(c.unsecured_usd),
                    "share": round(float(c.share), 9),
                }
                for c in sorted(self.claims, key=lambda c: c.beneficiary)
            ],
        }

    def compute_hash(self) -> str:
        """The content hash binding the folded pools and the reconciled re-use bound."""
        return stable_hash(self.ledger_facts(), length=32)

    def seal(self) -> CollateralLedger:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> CollateralLedger:
        """Add ``party``'s signature over the content hash (sealing first).

        A ledger is signed by whoever computed the guard — the poster attesting its own
        capital, or a beneficiary that independently re-folds the pools it is owed against.
        Re-signing for the same party replaces its prior signature, so a ledger cannot
        accumulate stale signatures for one identity.
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

    def _terms_sound(self) -> bool:
        """The totals reconcile and the breaches / claims re-derive from the bytes."""
        if abs(self.posted_usd - _r6(sum(p.posted_usd for p in self.pools))) > _TOLERANCE:
            return False
        pledged = _r6(sum(p.balance_usd for p in self.pools))
        if abs(self.pledged_usd - pledged) > _TOLERANCE:
            return False
        # Each pool's committed figure must equal the capital earmarked to its open contracts.
        for p in self.pools:
            committed = _r6(sum(c.allocated_usd for c in p.open_contracts))
            if abs(p.committed_usd - committed) > _TOLERANCE:
                return False
        if abs(self.available_usd - _r6(self.held_usd - pledged)) > _TOLERANCE:
            return False
        if abs(self.reuse_usd - _r6(max(0.0, -(self.held_usd - pledged)))) > _TOLERANCE:
            return False
        expected_breaches, duplicate_pledge = _reuse_breaches(self.pools)
        if abs(self.duplicate_pledge_usd - _r6(duplicate_pledge)) > _TOLERANCE:
            return False
        if not _breaches_match(self.breaches, expected_breaches):
            return False
        expected_claims = _beneficiary_claims(self.pools, self.held_usd)
        if not _claims_match(self.claims, expected_claims):
            return False
        if not self._reserves_sound():
            return False
        return True

    def _reserves_sound(self) -> bool:
        """The proven-reserves fields reconcile and the under-reserved breach re-derives.

        When the held figure is proven, the proven reserves are exactly the held figure (the
        custodian's attested total is what the guard bounds against) and the under-reserved
        breach re-derives from the pledged total. When it is not proven, no reserve fields are
        set and no breach is surfaced — so a fabricated breach is caught from the bytes.
        """
        if self.reserves_proven:
            if abs(self.reserves_usd - self.held_usd) > _TOLERANCE:
                return False
        else:
            if abs(self.reserves_usd) > _TOLERANCE or self.custodian or self.custody_hash:
                return False
        expected = self._derive_reserve_breach()
        if (expected is None) != (self.reserve_breach is None):
            return False
        if expected is not None and self.reserve_breach is not None:
            if (
                self.reserve_breach.custodian != expected.custodian
                or self.reserve_breach.attestation_hash != expected.attestation_hash
                or abs(self.reserve_breach.reserves_usd - expected.reserves_usd) > _TOLERANCE
                or abs(self.reserve_breach.pledged_usd - expected.pledged_usd) > _TOLERANCE
                or abs(self.reserve_breach.shortfall_usd - expected.shortfall_usd) > _TOLERANCE
            ):
                return False
        return True

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CollateralLedgerVerification:
        """Verify the ledger offline: the hash recomputes and the re-use bound re-derives.

        Recomputes the content hash and re-derives the pledged total, the re-use breaches,
        the held-capital reconciliation, and the beneficiary apportionment from the per-pool
        figures — so a tampered total, breach, or claim is caught even when the hash was
        recomputed to match. ``verifier`` additionally checks each signature against the
        content hash; ``require`` names parties that must have a verified signature
        (defaults to none — pass ``[poster]`` to demand the poster's signature).
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
                reason = "ledger is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the ledger facts"
            elif not terms_sound:
                reason = "re-use bound or beneficiary apportionment does not re-derive"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return CollateralLedgerVerification(
            valid=valid,
            hash_ok=hash_ok,
            terms_sound=terms_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CollateralLedger:
        """Verify and raise :class:`SettlementError` if the ledger is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"collateral ledger {self.id} failed verification: {result.reason}",
                details={"ledger_id": self.id, "reason": result.reason},
            )
        return self

    def require_within_bounds(self) -> CollateralLedger:
        """Raise :class:`SettlementError` if the poster has over-committed its capital.

        The strict-mode counterpart to inspecting :attr:`over_committed`: a poster whose
        pools pledge more than it holds cannot be admitted to a new deal without resolving
        the re-use first, and this pinpoints exactly which contracts are double-pledged and
        by how much the holdings fall short.
        """
        if self.over_committed:
            ids = [b.contract_id for b in self.breaches]
            raise SettlementError(
                f"collateral ledger {self.id} is over-committed by ${self.reuse_usd:,.2f} "
                f"(double-pledged contracts: {ids})",
                details={
                    "ledger_id": self.id,
                    "reuse_usd": self.reuse_usd,
                    "breaches": ids,
                },
            )
        return self

    def require_reserved(self) -> CollateralLedger:
        """Raise :class:`SettlementError` if the proven reserves fall below the pledges.

        The strict-mode counterpart to inspecting :attr:`under_reserved`: a poster whose
        custody attestation proves less capital than its pools pledge cannot be admitted to a
        new deal without resolving the under-reservation first. Unlike
        :meth:`require_within_bounds`, this fires only when the held figure is **proven** by a
        :class:`~vincio.settlement.custody.CustodyAttestation` — a self-asserted holdings
        figure cannot under-reserve.
        """
        if self.reserve_breach is not None:
            raise SettlementError(
                f"collateral ledger {self.id} is under-reserved by "
                f"${self.reserve_breach.shortfall_usd:,.2f}: {self.custodian!r} attests only "
                f"${self.reserves_usd:,.2f} against ${self.pledged_usd:,.2f} pledged",
                details={
                    "ledger_id": self.id,
                    "custodian": self.custodian,
                    "reserves_usd": self.reserves_usd,
                    "pledged_usd": self.pledged_usd,
                    "shortfall_usd": self.reserve_breach.shortfall_usd,
                },
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the guard for the audit chain."""
        return to_jsonable(
            {
                "ledger_id": self.id,
                "poster": self.poster,
                "status": self.status,
                "pools": len(self.pools),
                "posted_usd": _r6(self.posted_usd),
                "pledged_usd": _r6(self.pledged_usd),
                "held_usd": _r6(self.held_usd),
                "available_usd": _r6(self.available_usd),
                "reuse_usd": _r6(self.reuse_usd),
                "breaches": [b.contract_id for b in self.breaches],
                "reserves_proven": self.reserves_proven,
                "custodian": self.custodian,
                "reserves_usd": _r6(self.reserves_usd),
                "under_reserved_usd": (
                    _r6(self.reserve_breach.shortfall_usd) if self.reserve_breach else 0.0
                ),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> CollateralLedger:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the pledged total, the held capital, and the re-use breaches."""
        held_label = (
            f"${self.reserves_usd:,.2f} proven by {self.custodian}"
            if self.reserves_proven
            else f"${self.held_usd:,.2f} held"
        )
        print(
            f"Collateral ledger ({self.poster}): {len(self.pools)} pool(s) pledge "
            f"${self.pledged_usd:,.2f} against {held_label} — {self.status}"
        )
        if self.reserve_breach is not None:
            print(
                f"  ! under-reserved ${self.reserve_breach.shortfall_usd:,.2f}: proven "
                f"reserves below the pledges"
            )
        if self.over_committed:
            print(f"  re-use ${self.reuse_usd:,.2f} over the held stake")
        for b in self.breaches:
            print(
                f"  ! {b.contract_id}: ${b.pledged_usd:,.2f} pledged across {len(b.pools)} "
                f"pools, ${b.excess_usd:,.2f} double-pledged"
            )
        for c in self.claims:
            if not c.is_secured:
                print(
                    f"  {c.beneficiary}: ${c.secured_usd:,.2f} secured of ${c.claim_usd:,.2f} "
                    f"claimed (${c.unsecured_usd:,.2f} exposed)"
                )


# -- the reconciliation primitives --------------------------------------------


def _reuse_breaches(pools: list[LedgerPool]) -> tuple[list[ReuseBreach], float]:
    """The contracts pledged across more than one pool, and the double-pledged total.

    Folds the open contracts across pools by id: a contract backed by two or more pools is
    a re-pledge — its collateral earmarked once per pool but honorable only once, so the
    excess over the single claim it can satisfy is provably double-pledged capital. The
    single claim is the **largest** allocation any one pool earmarks (the most that deal
    could ever forfeit). Returns the pinpointed breaches (sorted by contract id) and the
    sum of their excesses.
    """
    by_contract: dict[str, list[tuple[str, LedgerContract]]] = {}
    for pool in pools:
        for c in pool.open_contracts:
            by_contract.setdefault(c.contract_id, []).append((pool.pool_id, c))

    breaches: list[ReuseBreach] = []
    duplicate_pledge = 0.0
    for contract_id in sorted(by_contract):
        entries = by_contract[contract_id]
        if len(entries) < 2:
            continue
        pledged = _r6(sum(c.allocated_usd for _pid, c in entries))
        secured = _r6(max(c.allocated_usd for _pid, c in entries))
        excess = _r6(pledged - secured)
        # The beneficiary is consistent across honest pools; pick it deterministically.
        beneficiary = sorted(c.beneficiary for _pid, c in entries)[0]
        breaches.append(
            ReuseBreach(
                contract_id=contract_id,
                beneficiary=beneficiary,
                pools=sorted(pid for pid, _c in entries),
                secured_usd=secured,
                pledged_usd=pledged,
                excess_usd=excess,
            )
        )
        duplicate_pledge += excess
    return breaches, _r6(duplicate_pledge)


def _beneficiary_claims(pools: list[LedgerPool], held_usd: float) -> list[BeneficiaryClaim]:
    """Each beneficiary's claim on the held capital, bounded to its pari-passu share.

    A beneficiary's claim is the capital pledged to it across the poster's pools — each of
    its distinct backed contracts counted once (the **largest** allocation any one pool
    earmarks, since a contract can forfeit only once however many pools list it). When the
    held capital covers every claim each is fully secured; when it falls short the held
    capital is apportioned proportionally (pari passu), so no beneficiary is promised
    capital another has first claim on. Sorted by beneficiary for a reproducible result.
    """
    # Reduce to one claim per distinct contract (its largest single pledge), tracking which
    # beneficiary and pools it belongs to.
    per_contract: dict[str, dict[str, Any]] = {}
    for pool in pools:
        for c in pool.open_contracts:
            slot = per_contract.setdefault(
                c.contract_id,
                {"beneficiary": c.beneficiary, "claim": 0.0, "pools": set()},
            )
            slot["claim"] = max(slot["claim"], c.allocated_usd)
            slot["pools"].add(pool.pool_id)
            # Keep the beneficiary deterministic if pools ever disagree.
            slot["beneficiary"] = min(slot["beneficiary"], c.beneficiary)

    agg: dict[str, dict[str, Any]] = {}
    for contract_id in sorted(per_contract):
        slot = per_contract[contract_id]
        ben = slot["beneficiary"]
        entry = agg.setdefault(ben, {"claim": 0.0, "pools": set(), "contracts": []})
        entry["claim"] = _r6(entry["claim"] + slot["claim"])
        entry["pools"].update(slot["pools"])
        entry["contracts"].append(contract_id)

    total_claims = _r6(sum(e["claim"] for e in agg.values()))
    fully_covered = total_claims <= _TOLERANCE or held_usd + _TOLERANCE >= total_claims
    coverage = 1.0 if fully_covered else max(0.0, held_usd) / total_claims

    claims: list[BeneficiaryClaim] = []
    for ben in sorted(agg):
        entry = agg[ben]
        claim = _r6(entry["claim"])
        secured = claim if fully_covered else _r6(claim * coverage)
        secured = min(secured, claim)
        unsecured = _r6(claim - secured)
        share = 1.0 if claim <= _TOLERANCE else round(min(1.0, secured / claim), 9)
        claims.append(
            BeneficiaryClaim(
                beneficiary=ben,
                claim_usd=claim,
                secured_usd=secured,
                unsecured_usd=unsecured,
                share=share,
                pools=sorted(entry["pools"]),
                contracts=sorted(entry["contracts"]),
            )
        )
    return claims


def _breaches_match(have: list[ReuseBreach], want: list[ReuseBreach]) -> bool:
    """Whether two breach lists agree, regardless of order."""
    if len(have) != len(want):
        return False
    have_by_id = {b.contract_id: b for b in have}
    for w in want:
        h = have_by_id.get(w.contract_id)
        if h is None:
            return False
        if (
            sorted(h.pools) != sorted(w.pools)
            or abs(h.pledged_usd - w.pledged_usd) > _TOLERANCE
            or abs(h.secured_usd - w.secured_usd) > _TOLERANCE
            or abs(h.excess_usd - w.excess_usd) > _TOLERANCE
        ):
            return False
    return True


def _claims_match(have: list[BeneficiaryClaim], want: list[BeneficiaryClaim]) -> bool:
    """Whether two claim lists agree, regardless of order."""
    if len(have) != len(want):
        return False
    have_by_ben = {c.beneficiary: c for c in have}
    for w in want:
        h = have_by_ben.get(w.beneficiary)
        if h is None:
            return False
        if (
            abs(h.claim_usd - w.claim_usd) > _TOLERANCE
            or abs(h.secured_usd - w.secured_usd) > _TOLERANCE
            or abs(h.unsecured_usd - w.unsecured_usd) > _TOLERANCE
        ):
            return False
    return True


# -- module-level builder -----------------------------------------------------


def _ledger_pool(pool: CollateralPool, *, verifier: ChainSigner | None) -> LedgerPool:
    """Fold one :class:`CollateralPool` into a :class:`LedgerPool`, refusing a tampered one.

    Reads only what it can recompute: a pool whose content hash no longer recomputes — a
    tampered allocation or balance — is refused outright, and with a ``verifier`` a forged
    signature is too. The pool's still-open contracts are bound as the capital it pledges.
    """
    if not pool.content_hash or pool.content_hash != pool.compute_hash():
        raise SettlementError(
            f"collateral pool {pool.id} is tampered (content hash does not recompute); "
            "refusing to fold it into the ledger",
            details={"pool_id": pool.id},
        )
    if verifier is not None and pool.signatures:
        check = pool.verify(verifier, require=[])
        if not check.signatures_ok:
            raise SettlementError(
                f"collateral pool {pool.id} has an invalid signature; refusing to fold it",
                details={"pool_id": pool.id},
            )
    open_contracts = [
        LedgerContract(
            contract_id=c.contract_id,
            beneficiary=c.beneficiary,
            allocated_usd=_r6(c.allocated_usd),
            required_usd=_r6(c.required_usd),
        )
        for c in pool.open_contracts
    ]
    return LedgerPool(
        pool_id=pool.id,
        poster=pool.poster,
        posted_usd=_r6(pool.posted_usd),
        balance_usd=_r6(pool.balance_usd),
        committed_usd=_r6(sum(c.allocated_usd for c in open_contracts)),
        status=pool.status,
        content_hash=pool.content_hash,
        open_contracts=open_contracts,
    )


def _reserves_from_custody(
    custody: CustodyAttestation,
    *,
    poster: str,
    verifier: ChainSigner | None,
) -> float:
    """Read a custody attestation's proven reserves, refusing a tampered or mismatched one.

    Reads only what it can recompute: an attestation whose content hash no longer recomputes
    or whose total no longer re-derives from the line items — a tampered reserve figure — is
    refused outright, and with a ``verifier`` a forged custodian signature is too. An
    attestation that vouches for a *different* poster cannot stand in for this poster's
    reserves and is refused. Returns the attested ``reserves_usd``.
    """
    result = custody.verify(verifier)
    if not result.hash_ok or not result.reserves_sound:
        raise SettlementError(
            f"custody attestation {custody.id} is tampered ({result.reason}); "
            "refusing to read it as proof-of-reserves",
            details={"attestation_id": custody.id, "reason": result.reason},
        )
    if verifier is not None and custody.signatures and not result.signatures_ok:
        raise SettlementError(
            f"custody attestation {custody.id} has an invalid custodian signature; "
            "refusing to read it as proof-of-reserves",
            details={"attestation_id": custody.id},
        )
    if custody.poster != poster:
        raise SettlementError(
            f"custody attestation {custody.id} attests reserves for {custody.poster!r}, "
            f"not the poster {poster!r} the guard bounds; refusing it",
            details={"attestation_id": custody.id, "attests": custody.poster, "poster": poster},
        )
    return _r6(custody.reserves_usd)


def guard_collateral(
    pools: Iterable[CollateralPool],
    *,
    poster: str | None = None,
    held: float | None = None,
    custody: CustodyAttestation | None = None,
    verify_with: ChainSigner | None = None,
) -> CollateralLedger:
    """Fold a counterparty's collateral pools into a bounded, offline-verifiable re-use guard.

    The rehypothecation analogue of :func:`~vincio.settlement.netting.net_settlements`:
    reads a counterparty's :class:`~vincio.settlement.collateral.CollateralPool`\\ s,
    refuses any whose content hash no longer recomputes (a forged signature too, with
    ``verify_with``), and reconciles what they collectively pledge against the capital the
    poster actually holds. A contract pledged across more than one pool is pinpointed as a
    :class:`ReuseBreach`, and each beneficiary's claim is bounded to its deterministic share
    of the held capital. Returns a sealed, unsigned :class:`CollateralLedger`.

    ``poster`` is the counterparty whose stake the ledger views (defaults to the poster
    every pool shares; an explicit poster is required when they differ). The capital that
    poster holds — the figure the guard bounds the pledges by — comes from one of:

    * ``custody`` — a signed :class:`~vincio.settlement.custody.CustodyAttestation`
      **proving** the reserves. The guard reads its ``reserves_usd`` as the held figure,
      marks the bound :attr:`~CollateralLedger.reserves_proven`, and surfaces an
      :class:`UnderReservedBreach` when the proven reserves fall below the pledges. A
      tampered reserve figure, a forged custodian (with ``verify_with``), or an attestation
      for a different poster is **refused**.
    * ``held`` — an explicit, *asserted* holdings figure (the legacy input). It can
      over-commit but never under-*reserves*, because nothing proves it.
    * neither — defaults to the gross pledge minus the provably double-pledged capital, so a
      re-pledged contract surfaces as an over-commitment while genuinely separately-funded
      pools do not.

    Raises :class:`SettlementError` when the set is empty, the posters differ and none is
    given, ``held`` is negative, or both ``held`` and ``custody`` are passed (the held figure
    has one source).
    """
    pool_list = list(pools)
    if not pool_list:
        raise SettlementError(
            "guard_collateral needs at least one collateral pool to fold",
            details={},
        )
    if held is not None and custody is not None:
        raise SettlementError(
            "guard_collateral takes either an asserted held= figure or a proven custody= "
            "attestation, not both; the held figure has one source",
            details={},
        )
    posters = {p.poster for p in pool_list}
    resolved_poster = poster
    if resolved_poster is None:
        if len(posters) != 1:
            raise SettlementError(
                "guard_collateral needs an explicit poster: the pools do not share one "
                f"poster (posters={sorted(posters)})",
                details={"posters": sorted(posters)},
            )
        resolved_poster = next(iter(posters))
    mismatched = sorted(p.id for p in pool_list if p.poster != resolved_poster)
    if mismatched:
        raise SettlementError(
            f"pools {mismatched} are not posted by {resolved_poster!r}; a collateral "
            "ledger views one counterparty's stake",
            details={"poster": resolved_poster, "pools": mismatched},
        )
    if held is not None and held < 0.0:
        raise SettlementError(
            f"held capital must be non-negative; got {held}",
            details={"poster": resolved_poster, "held": held},
        )

    ledger_pools = [_ledger_pool(p, verifier=verify_with) for p in pool_list]
    # The held figure: proven by a custody attestation, asserted via held=, or defaulted to
    # the gross pledge minus the provably double-pledged capital (the same contract backed by
    # more than one pool), so duplicates surface as an over-commitment by default while
    # genuinely separately-funded pools do not.
    pledged = _r6(sum(p.balance_usd for p in ledger_pools))
    _breaches, duplicate_pledge = _reuse_breaches(ledger_pools)
    reserves_proven = custody is not None
    custodian = ""
    custody_hash = ""
    reserves_usd = 0.0
    if custody is not None:
        resolved_held = _reserves_from_custody(
            custody, poster=resolved_poster, verifier=verify_with
        )
        custodian = custody.custodian
        custody_hash = custody.content_hash
        reserves_usd = resolved_held
    elif held is None:
        resolved_held = _r6(pledged - duplicate_pledge)
    else:
        resolved_held = _r6(held)
    ledger = CollateralLedger(
        poster=resolved_poster,
        pools=ledger_pools,
        pool_hashes=sorted(p.content_hash for p in ledger_pools),
        held_usd=resolved_held,
        reserves_proven=reserves_proven,
        custodian=custodian,
        custody_hash=custody_hash,
        reserves_usd=reserves_usd,
    )
    ledger._recompute()
    return ledger.seal()
