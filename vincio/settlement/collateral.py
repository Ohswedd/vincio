"""Cross-org collateral pooling & cross-contract margin.

An :class:`~vincio.settlement.escrow.Escrow` now backs *one* contract with collateral
held against its delivery — but a counterparty running many concurrent contracts must
lock **separate** collateral per deal, even though its breaches and clean deliveries
across those contracts net out. Capital is stranded contract-by-contract the way
bilateral settlements were stranded book-by-book before
:func:`~vincio.settlement.netting.net_settlements` folded them. This module is the next
reach: a **bounded collateral pool** — a margin account a counterparty posts once that
backs many contracts at a deterministic, offline-verifiable allocation, the collateral
analogue of the :class:`~vincio.settlement.netting.NettingSet`.

* **Posted, content-bound pool.** A :class:`CollateralPool` binds a counterparty's single
  posted stake to the set of contracts it backs into a signed, offline-verifiable
  artifact, allocating a per-contract share **deterministically** — proportional to each
  contract's admission-required collateral — so what backs each deal is a mechanical,
  reconstructable number, never a custodian's omnibus account. The required collateral of
  each backed contract re-derives from its admission posture (an
  :class:`~vincio.settlement.admission.AdmissionDecision`, an explicit fraction / amount,
  or the posture stamped onto the contract's terms) exactly as a standalone escrow's does.
* **Deterministic draw & top-up.** Settling a contract **draws** a forfeiture from the
  pool (a bounded, pinpointed slice proportional to the shortfall the settlement measured,
  driven by the *same* :class:`~vincio.settlement.record.SettlementRecord` verdict the
  books close on) and **releases** the rest of its requirement back to the available
  balance — so a clean delivery frees capital for the next contract and a breach is
  covered from the shared stake. A pool committed below the collateral its still-open
  contracts require surfaces a bounded, pinpointed **top-up** obligation rather than
  silently over-committing.
* **Auditable & offline.** Every post, draw, release, and top-up binds the pool, the
  contracts, and the balances onto a content hash; :meth:`CollateralPool.verify`
  recomputes it and re-derives the allocations and reconciles the balance (a tampered
  allocation or balance is caught even after re-sealing, exactly as a tampered escrow
  forfeiture is), and the settlement path lands each transition on the hash-chained audit
  log — so a pool's whole lifecycle is reconstructable offline, never a trusted third
  party.

The pool folds into the *existing* settlement path over the
:class:`~vincio.settlement.escrow.Escrow` / :class:`~vincio.settlement.admission.AdmissionDecision`
/ :class:`~vincio.settlement.book.SettlementBook` machinery: :func:`post_collateral_pool`
reads each contract's required collateral the way :func:`~vincio.settlement.escrow.post_escrow`
does, and :meth:`~vincio.settlement.book.SettlementBook.settle` (with ``pool=``) draws an
open contract's settlement against the pool in one call. Everything is dependency-free,
deterministic, and offline — a verifiable margin account over the collateral the fabric
already requires, never a hosted clearing house or a payment rail.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .escrow import EscrowConfig, _resolve_collateral, _shortfall_from_record
from .record import SettlementSignature

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "PooledContractState",
    "PoolStatus",
    "PooledContract",
    "CollateralPoolVerification",
    "CollateralPool",
    "post_collateral_pool",
    "draw_pool",
]

# The single audit action every collateral-pool transition is recorded under; the
# decision field carries the pool status (``posted`` / ``active`` / ``settled``).
COLLATERAL_ACTION = "collateral_pool"

PooledContractState = Literal["open", "released", "forfeited"]
PoolStatus = Literal["posted", "active", "settled"]

_TOLERANCE = 1e-6


def _r6(value: float) -> float:
    """Round a money figure to six places so float drift never breaks a balance."""
    return round(float(value), 6)


class PooledContract(BaseModel):
    """One contract a :class:`CollateralPool` backs, with its share and disposition.

    Carries the contract it backs (by id and content hash), the admission-required
    ``required_usd`` collateral it demands (re-deriving from ``escrow_fraction`` × the
    contract ``price_usd`` when posted from a fraction), the ``allocated_usd`` share of
    the pool earmarked to back it (proportional to its requirement, pro-rated down when
    the pool cannot cover every open contract), and — once it settles — the disposition:
    the ``forfeited_usd`` slice drawn from the pool on a breach and the ``released_usd``
    remainder freed back to the available balance.
    """

    contract_id: str
    contract_hash: str = ""
    buyer: str
    seller: str
    beneficiary: str
    scope: str = ""

    price_usd: float = 0.0
    escrow_fraction: float = 0.0
    required_usd: float = 0.0
    decision_id: str | None = None
    decision_hash: str = ""

    allocated_usd: float = 0.0

    state: PooledContractState = "open"
    shortfall_fraction: float = 0.0
    forfeited_usd: float = 0.0
    released_usd: float = 0.0
    breaches: list[str] = Field(default_factory=list)

    settlement_id: str | None = None
    settlement_hash: str = ""
    resolved_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        """Whether the contract is still open (not yet settled)."""
        return self.state == "open"

    @property
    def is_resolved(self) -> bool:
        """Whether the contract has settled (released or forfeited)."""
        return self.state != "open"

    def _required_sound(self) -> bool:
        """The required collateral re-derives from the admission-required fraction.

        When the contract was posted from a fraction the required amount must equal that
        fraction of the contract price, so a tampered requirement is caught even after
        re-sealing. A flat requirement (no fraction) is authoritative as posted.
        """
        if self.escrow_fraction <= 0.0:
            return True
        expected = round(self.escrow_fraction * self.price_usd, 6)
        return abs(self.required_usd - expected) <= _TOLERANCE

    def _disposition_sound(self, config: EscrowConfig) -> bool:
        """The forfeited / released split re-derives from the bound shortfall.

        An open contract draws and releases nothing. A resolved one forfeits the bound
        shortfall (clamped into ``[0, max_forfeit_fraction]``) of its requirement and
        releases the remainder — so :meth:`CollateralPool.verify` recomputes the split
        from the bytes.
        """
        required = _r6(self.required_usd)
        if self.state == "open":
            return self.forfeited_usd == 0.0 and self.released_usd == 0.0
        fraction = min(max(0.0, self.shortfall_fraction), config.max_forfeit_fraction)
        forfeited = _r6(required * fraction)
        released = _r6(required - forfeited)
        return (
            abs(self.forfeited_usd - forfeited) <= _TOLERANCE
            and abs(self.released_usd - released) <= _TOLERANCE
        )

    def facts(self) -> dict[str, Any]:
        """The per-contract facts the pool's content hash binds."""
        return {
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "buyer": self.buyer,
            "seller": self.seller,
            "beneficiary": self.beneficiary,
            "scope": self.scope,
            "price_usd": round(float(self.price_usd), 9),
            "escrow_fraction": round(float(self.escrow_fraction), 9),
            "required_usd": _r6(self.required_usd),
            "decision_hash": self.decision_hash,
            "allocated_usd": _r6(self.allocated_usd),
            "state": self.state,
            "shortfall_fraction": round(float(self.shortfall_fraction), 9),
            "forfeited_usd": _r6(self.forfeited_usd),
            "released_usd": _r6(self.released_usd),
            "breaches": list(self.breaches),
            "settlement_hash": self.settlement_hash,
        }


class CollateralPoolVerification(BaseModel):
    """The (non-raising) outcome of verifying a collateral pool offline.

    A pool is **valid** when its content hash recomputes (``hash_ok``), the per-contract
    requirements and dispositions re-derive and the balances reconcile (``terms_sound`` —
    the balance equals the posted stake minus the drawn forfeitures, the allocations
    re-derive proportional to the open requirements, and the top-up reconciles), and,
    with a ``verifier``, every signature checks (``signatures_ok``). A tampered allocation,
    balance, or forfeiture is caught from the bytes alone, even after re-sealing.
    """

    valid: bool
    hash_ok: bool
    terms_sound: bool
    signatures_ok: bool = True
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class CollateralPool(BaseModel):
    """A counterparty's single posted stake backing many contracts, a margin account.

    Produced by :func:`post_collateral_pool` (or
    :meth:`~vincio.core.app.ContextApp.post_collateral_pool`) from a set of contracts and
    the admission-required collateral each demands. It binds the ``poster`` (the
    counterparty backing its delivery across the deals), the ``posted_usd`` single stake,
    and every :class:`PooledContract` it backs onto a content hash, so the margin account
    is a mechanical number anyone recomputes.

    The pool allocates a per-contract share **proportional to each contract's required
    collateral** — the collateral analogue of a
    :class:`~vincio.settlement.netting.NettingSet`'s net positions. :meth:`draw` settles
    one contract against its :class:`~vincio.settlement.record.SettlementRecord` (drawing a
    bounded forfeiture from the shared stake on a breach, releasing the rest back to the
    available balance on a clean delivery), :meth:`back` adds another contract to the open
    pool, :meth:`top_up` adds capital to a pool that has fallen below its open contracts'
    requirement, and :meth:`verify` re-derives the allocations and reconciles the balance
    from the bytes alone.
    """

    id: str = Field(default_factory=lambda: new_id("pool"))
    poster: str
    contracts: list[PooledContract] = Field(default_factory=list)
    config: EscrowConfig = Field(default_factory=EscrowConfig)

    # The single posted stake and the balances it nets to (set on every recompute).
    posted_usd: float = 0.0
    drawn_usd: float = 0.0
    balance_usd: float = 0.0
    required_open_usd: float = 0.0
    available_usd: float = 0.0
    topup_usd: float = 0.0
    coverage: float = 1.0

    posted_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime | None = None
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def parties(self) -> list[str]:
        """Every party to a backed contract (the poster and its counterparties)."""
        seen: set[str] = {self.poster}
        for c in self.contracts:
            seen.add(c.buyer)
            seen.add(c.seller)
        return sorted(seen)

    @property
    def beneficiaries(self) -> list[str]:
        """The counterparties a breach forfeiture is paid out to, sorted."""
        return sorted({c.beneficiary for c in self.contracts})

    @property
    def open_contracts(self) -> list[PooledContract]:
        """The contracts the pool still backs (not yet settled)."""
        return [c for c in self.contracts if c.is_open]

    @property
    def status(self) -> PoolStatus:
        """``posted`` (nothing settled), ``active`` (some open), or ``settled`` (all closed)."""
        resolved = [c for c in self.contracts if c.is_resolved]
        if not resolved:
            return "posted"
        return "settled" if all(c.is_resolved for c in self.contracts) else "active"

    @property
    def needs_topup(self) -> bool:
        """Whether the pool is committed below the collateral its open contracts require."""
        return self.topup_usd > _TOLERANCE

    @property
    def residual_usd(self) -> float:
        """The stake returnable to the poster once every contract has settled.

        The balance left after the forfeitures are drawn — meaningful once the pool is
        fully settled, when nothing remains committed to an open contract.
        """
        return self.balance_usd

    def contract(self, contract_id: str) -> PooledContract | None:
        """The backed contract with a given id, or ``None``."""
        return next((c for c in self.contracts if c.contract_id == contract_id), None)

    # -- the balance recompute (the pool's mechanical core) -----------------

    def _recompute(self) -> None:
        """Re-derive the balances and per-contract allocations from the current state.

        The pool's deterministic core, called after every post, draw, and top-up: the
        balance is the posted stake minus the drawn forfeitures, the open contracts' total
        requirement is what the pool is committed to, the available balance is the rest
        (negative when over-committed, surfacing a top-up obligation), and each open
        contract is allocated a share proportional to its requirement, pro-rated by the
        coverage the balance affords.
        """
        drawn = _r6(sum(c.forfeited_usd for c in self.contracts))
        self.drawn_usd = drawn
        self.balance_usd = _r6(self.posted_usd - drawn)
        open_contracts = self.open_contracts
        required_open = _r6(sum(c.required_usd for c in open_contracts))
        self.required_open_usd = required_open
        self.available_usd = _r6(self.balance_usd - required_open)
        self.topup_usd = _r6(max(0.0, -self.available_usd))
        coverage = (
            1.0
            if required_open <= _TOLERANCE
            else min(1.0, max(0.0, self.balance_usd) / required_open)
        )
        self.coverage = round(coverage, 9)
        for c in self.contracts:
            c.allocated_usd = _r6(c.required_usd * coverage) if c.is_open else 0.0

    # -- hashing ------------------------------------------------------------

    def pool_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the stake, the contracts, the balances.

        Excludes the id, timestamps, signatures, and audit linkage (local metadata, not
        the pool), so the same stake backing the same contracts settled the same way
        hashes identically wherever it is recomputed — the way two parties co-sign one
        reconciliation hash. Contracts are sorted by id so the order they were added in
        never changes the hash.
        """
        return {
            "poster": self.poster,
            "config": self.config.canonical(),
            "posted_usd": _r6(self.posted_usd),
            "drawn_usd": _r6(self.drawn_usd),
            "balance_usd": _r6(self.balance_usd),
            "required_open_usd": _r6(self.required_open_usd),
            "available_usd": _r6(self.available_usd),
            "topup_usd": _r6(self.topup_usd),
            "contracts": [c.facts() for c in sorted(self.contracts, key=lambda c: c.contract_id)],
        }

    def compute_hash(self) -> str:
        """The content hash binding the stake, the contracts, and the balances."""
        return stable_hash(self.pool_facts(), length=32)

    def seal(self) -> CollateralPool:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- signing ------------------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    def sign(self, signer: ChainSigner, *, party: str) -> CollateralPool:
        """Add ``party``'s signature over the content hash (sealing first).

        Only the poster or a counterparty to a backed contract can sign. Re-signing for
        the same party replaces its prior signature, so a pool cannot accumulate stale
        signatures for one identity.
        """
        if party not in self.parties:
            raise SettlementError(
                f"party {party!r} is neither the poster nor a counterparty of pool {self.id}",
                details={"pool_id": self.id, "party": party},
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

    # -- verification -------------------------------------------------------

    def _terms_sound(self) -> bool:
        """The balances reconcile and every contract's terms re-derive from the bytes."""
        drawn = _r6(sum(c.forfeited_usd for c in self.contracts))
        if abs(self.drawn_usd - drawn) > _TOLERANCE:
            return False
        if abs(self.balance_usd - _r6(self.posted_usd - drawn)) > _TOLERANCE:
            return False
        required_open = _r6(sum(c.required_usd for c in self.open_contracts))
        if abs(self.required_open_usd - required_open) > _TOLERANCE:
            return False
        if abs(self.available_usd - _r6(self.balance_usd - required_open)) > _TOLERANCE:
            return False
        if abs(self.topup_usd - _r6(max(0.0, -(self.balance_usd - required_open)))) > _TOLERANCE:
            return False
        coverage = (
            1.0
            if required_open <= _TOLERANCE
            else min(1.0, max(0.0, self.balance_usd) / required_open)
        )
        for c in self.contracts:
            if not c._required_sound() or not c._disposition_sound(self.config):
                return False
            expected_alloc = _r6(c.required_usd * coverage) if c.is_open else 0.0
            if abs(c.allocated_usd - expected_alloc) > _TOLERANCE:
                return False
        return True

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CollateralPoolVerification:
        """Verify the pool offline: the hash recomputes and the balances reconcile.

        Recomputes the content hash and re-derives every per-contract requirement,
        allocation, and disposition while reconciling the balance (posted minus drawn) and
        the top-up — so a tampered allocation, balance, or forfeiture is caught even when
        the hash was recomputed to match. ``verifier`` additionally checks each signature
        against the content hash; ``require`` names the parties that must have a verified
        signature (defaults to none — pass ``[poster]`` to demand the poster's signature).
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
                reason = "pool is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the pool facts"
            elif not terms_sound:
                reason = "allocations or balance do not re-derive from the posted stake"
            elif missing:
                reason = f"missing/invalid signatures for {missing}"
            else:
                reason = "signature mismatch"
        return CollateralPoolVerification(
            valid=valid,
            hash_ok=hash_ok,
            terms_sound=terms_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> CollateralPool:
        """Verify and raise :class:`SettlementError` if the pool is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"collateral pool {self.id} failed verification: {result.reason}",
                details={"pool_id": self.id, "reason": result.reason},
            )
        return self

    # -- mutation -----------------------------------------------------------

    def back(
        self,
        contract: Any,
        *,
        decision: Any | None = None,
        fraction: float | None = None,
        amount: float | None = None,
        beneficiary: str | None = None,
    ) -> PooledContract:
        """Add another contract to the open pool, re-allocating the shared stake.

        Reads the contract's admission-required collateral the way
        :func:`~vincio.settlement.escrow.post_escrow` does (from a ``decision``, an
        explicit ``fraction`` / ``amount``, or the admission posture stamped onto the
        terms), binds it as a new :class:`PooledContract`, and re-derives the allocations
        — so backing a new deal may surface a top-up obligation when the pool can no
        longer cover every open contract. The poster must be a party to the new contract.
        Clears the signatures (the content hash changes) and re-seals.
        """
        pooled = _pool_contract(
            contract,
            poster=self.poster,
            decision=decision,
            fraction=fraction,
            amount=amount,
            beneficiary=beneficiary,
        )
        if self.contract(pooled.contract_id) is not None:
            raise SettlementError(
                f"pool {self.id} already backs contract {pooled.contract_id!r}",
                details={"pool_id": self.id, "contract_id": pooled.contract_id},
            )
        self.contracts.append(pooled)
        self.updated_at = utcnow()
        self.signatures = []
        self._recompute()
        self.seal()
        return pooled

    def draw(self, record: Any, *, config: EscrowConfig | None = None) -> PooledContract:
        """Settle one backed contract against its settlement record (draw or release).

        Draws a bounded, pinpointed forfeiture from the shared stake on a breach
        (proportional to the shortfall the settlement measured, clamped by
        ``max_forfeit_fraction``) and releases the rest of the contract's requirement back
        to the available balance on a clean delivery — driven by the *same*
        :class:`~vincio.settlement.record.SettlementRecord` verdict the books close on, so
        the collateral never judges the delivery a second time. Re-derives the balances and
        re-allocates the still-open contracts, clears the signatures (the content hash
        changes), and re-seals. Raises :class:`SettlementError` if the record is for a
        contract the pool does not back or that has already settled.
        """
        record_contract = getattr(record, "contract_id", None)
        pooled = self.contract(record_contract) if record_contract is not None else None
        if pooled is None:
            raise SettlementError(
                f"pool {self.id} does not back contract {record_contract!r}",
                details={"pool_id": self.id, "contract_id": record_contract},
            )
        if pooled.is_resolved:
            raise SettlementError(
                f"contract {pooled.contract_id!r} in pool {self.id} is already settled "
                f"({pooled.state})",
                details={"pool_id": self.id, "contract_id": pooled.contract_id},
            )
        cfg = (config or self.config).validate_coherent()
        fulfilled = bool(getattr(record, "fulfilled", True))
        shortfall, breaches = _shortfall_from_record(record)
        if fulfilled:
            shortfall, breaches = 0.0, []
        pooled.shortfall_fraction = round(shortfall, 9)
        pooled.breaches = breaches
        forfeit_fraction = min(shortfall, cfg.max_forfeit_fraction)
        required = _r6(pooled.required_usd)
        pooled.forfeited_usd = _r6(required * forfeit_fraction)
        pooled.released_usd = _r6(required - pooled.forfeited_usd)
        pooled.state = "forfeited" if pooled.forfeited_usd > 0.0 else "released"
        pooled.settlement_id = getattr(record, "id", None)
        pooled.settlement_hash = getattr(record, "content_hash", "") or ""
        pooled.resolved_at = utcnow()
        self.updated_at = utcnow()
        self.signatures = []
        self._recompute()
        self.seal()
        return pooled

    def top_up(self, amount: float) -> CollateralPool:
        """Add capital to the pool, raising the posted stake by ``amount``.

        The response to a :attr:`needs_topup` obligation: increasing the posted stake
        lifts the available balance and clears the top-up once the pool again covers its
        open contracts. Re-derives the balances, clears the signatures (the content hash
        changes), and re-seals. Raises :class:`SettlementError` on a non-positive amount.
        """
        if amount <= 0.0:
            raise SettlementError(
                f"top-up amount must be positive; got {amount}",
                details={"pool_id": self.id, "amount": amount},
            )
        self.posted_usd = _r6(self.posted_usd + amount)
        self.updated_at = utcnow()
        self.signatures = []
        self._recompute()
        self.seal()
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the pool for the audit chain."""
        return to_jsonable(
            {
                "pool_id": self.id,
                "poster": self.poster,
                "status": self.status,
                "contracts": len(self.contracts),
                "open_contracts": len(self.open_contracts),
                "posted_usd": _r6(self.posted_usd),
                "drawn_usd": _r6(self.drawn_usd),
                "balance_usd": _r6(self.balance_usd),
                "required_open_usd": _r6(self.required_open_usd),
                "available_usd": _r6(self.available_usd),
                "topup_usd": _r6(self.topup_usd),
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> CollateralPool:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the posted stake, the balances, and the cleared contracts."""
        print(
            f"Collateral pool ({self.poster}): ${self.posted_usd:,.2f} posted backing "
            f"{len(self.contracts)} contract(s) — {self.status}"
        )
        print(
            f"  balance ${self.balance_usd:,.2f} | committed ${self.required_open_usd:,.2f} "
            f"| available ${self.available_usd:,.2f} | drawn ${self.drawn_usd:,.2f}"
            + (f" | top-up needed ${self.topup_usd:,.2f}" if self.needs_topup else "")
        )
        for c in self.contracts:
            if c.is_open:
                print(
                    f"  {c.contract_id}: ${c.allocated_usd:,.2f} allocated "
                    f"(needs ${c.required_usd:,.2f}) — open"
                )
            elif c.state == "forfeited":
                pinned = f" ({', '.join(c.breaches)})" if c.breaches else ""
                print(
                    f"  {c.contract_id}: ${c.forfeited_usd:,.2f} forfeited to {c.beneficiary}, "
                    f"${c.released_usd:,.2f} freed — breached{pinned}"
                )
            else:
                print(f"  {c.contract_id}: ${c.released_usd:,.2f} freed — clean")


# -- module-level builders -----------------------------------------------------


def _pool_contract(
    contract: Any,
    *,
    poster: str,
    decision: Any | None,
    fraction: float | None,
    amount: float | None,
    beneficiary: str | None,
) -> PooledContract:
    """Bind one contract into a :class:`PooledContract` under a common poster.

    Resolves the contract's required collateral (via
    :func:`~vincio.settlement.escrow._resolve_collateral`) and pins the beneficiary — the
    counterparty a breach forfeiture is paid out to, the party that is not the poster.
    Raises :class:`SettlementError` when the poster is a party to neither side.
    """
    buyer = contract.buyer
    seller = contract.seller
    if poster not in (buyer, seller):
        raise SettlementError(
            f"poster {poster!r} is not a party to contract {getattr(contract, 'id', None)!r} "
            f"(buyer={buyer!r}, seller={seller!r})",
            details={"poster": poster, "contract_id": getattr(contract, "id", None)},
        )
    resolved_beneficiary = beneficiary or (seller if poster == buyer else buyer)
    price, frac, required, decision_id, decision_hash = _resolve_collateral(
        contract,
        decision=decision,
        fraction=fraction,
        amount=amount,
        source="post_collateral_pool",
    )
    terms = contract.terms
    return PooledContract(
        contract_id=contract.id,
        contract_hash=getattr(contract, "content_hash", "") or "",
        buyer=buyer,
        seller=seller,
        beneficiary=resolved_beneficiary,
        scope=getattr(terms, "scope", ""),
        price_usd=price,
        escrow_fraction=round(frac, 9),
        required_usd=required,
        decision_id=decision_id,
        decision_hash=decision_hash,
    )


def _default_poster(contracts: list[Any]) -> str:
    """The counterparty a pool defaults to backing: the contracts' common seller.

    A pool is one counterparty's margin account, so it defaults to the seller every
    contract shares (the admitted counterparty backing its delivery, as an escrow's poster
    does). Raises :class:`SettlementError` when the sellers differ and no poster is given —
    the pool cannot guess whose collateral it holds.
    """
    sellers = {c.seller for c in contracts}
    if len(sellers) == 1:
        return next(iter(sellers))
    raise SettlementError(
        "post_collateral_pool needs an explicit poster: the backed contracts do not "
        f"share one seller (sellers={sorted(sellers)})",
        details={"sellers": sorted(sellers)},
    )


def post_collateral_pool(
    contracts: Iterable[Any],
    *,
    poster: str | None = None,
    posted: float | None = None,
    decisions: Any | None = None,
    fraction: float | None = None,
    config: EscrowConfig | None = None,
) -> CollateralPool:
    """Post one stake backing many contracts into an (unsigned) :class:`CollateralPool`.

    The pool analogue of :func:`~vincio.settlement.escrow.post_escrow`: resolves each
    contract's admission-required collateral and binds the whole set to a single posted
    stake. Each contract's requirement comes from, in order: a matching
    :class:`~vincio.settlement.admission.AdmissionDecision` in ``decisions`` (a dict keyed
    by contract id, or a single decision applied to all), a uniform ``fraction`` of the
    contract price, or the admission posture stamped onto the contract's terms — exactly as
    :func:`~vincio.settlement.escrow.post_escrow` resolves a single escrow's.

    ``poster`` is the counterparty backing its delivery across the deals (defaulting to the
    seller every contract shares); ``posted`` is the single stake it posts (defaulting to
    the total required collateral, so the pool starts exactly collateralized). Posting less
    than the total required surfaces a bounded top-up obligation. Returns a sealed, unsigned
    pool — sign it with the poster's key (or let a
    :class:`~vincio.settlement.book.SettlementBook` do it on
    :meth:`~vincio.settlement.book.SettlementBook.post_collateral_pool`). Raises
    :class:`SettlementError` when the set is empty, a poster cannot be resolved, or a
    contract carries no collateral source.
    """
    contract_list = list(contracts)
    if not contract_list:
        raise SettlementError(
            "post_collateral_pool needs at least one contract to back",
            details={},
        )
    resolved_poster = poster or _default_poster(contract_list)
    decision_for = _decision_lookup(decisions)
    pooled = [
        _pool_contract(
            contract,
            poster=resolved_poster,
            decision=decision_for(contract),
            fraction=fraction,
            amount=None,
            beneficiary=None,
        )
        for contract in contract_list
    ]
    required_total = _r6(sum(c.required_usd for c in pooled))
    resolved_posted = required_total if posted is None else _r6(posted)
    if resolved_posted < 0.0:
        raise SettlementError(
            f"posted stake must be non-negative; got {resolved_posted}",
            details={"posted": resolved_posted},
        )
    pool = CollateralPool(
        poster=resolved_poster,
        contracts=pooled,
        config=(config or EscrowConfig()).validate_coherent(),
        posted_usd=resolved_posted,
    )
    pool._recompute()
    return pool.seal()


def _decision_lookup(decisions: Any | None) -> Any:
    """A ``contract -> decision`` resolver over ``decisions`` (a dict, a single, or None)."""
    if decisions is None:
        return lambda _contract: None
    if isinstance(decisions, dict):
        return lambda contract: decisions.get(getattr(contract, "id", None))
    return lambda _contract: decisions


def draw_pool(
    pool: CollateralPool, record: Any, *, config: EscrowConfig | None = None
) -> PooledContract:
    """Settle one backed contract against a settlement record (draw or release).

    The module-level convenience over :meth:`CollateralPool.draw`: settles the matching
    contract against its :class:`~vincio.settlement.record.SettlementRecord`, drawing a
    bounded forfeiture from the shared stake on a breach and releasing the rest on a clean
    delivery. Returns the resolved :class:`PooledContract`.
    """
    return pool.draw(record, config=config)
