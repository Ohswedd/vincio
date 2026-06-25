"""Typed, signed, audited negotiation contracts.

A :class:`Contract` is the artifact a :class:`~vincio.negotiation.engine.Negotiation`
converges on: a typed agreement over **price / SLA / scope / quality** that both
parties sign, that verifies **offline** from the bytes alone, and that the
orchestrator can enforce **like any other budget**.

Three guarantees, all dependency-free and deterministic:

* **Typed terms.** :class:`ContractTerms` carries the negotiated dimensions as
  numbers (``price_usd``, ``sla_seconds``, ``quality_floor``) plus the fixed
  ``scope`` string the deal is for — no free-form blob the orchestrator cannot
  reason about.
* **Signed & offline-verifiable.** :meth:`Contract.sign` adds a party's signature
  over the content hash using the same :class:`~vincio.security.audit.ChainSigner`
  the audit chain uses (HMAC by default, Ed25519 for third-party verifiability).
  :meth:`Contract.verify` recomputes the hash from the stored fields and checks
  every signature, so a tampered term or a forged signature is caught without the
  live parties — the contract is a self-contained, content-bound artifact.
* **Enforced like a budget.** :meth:`Contract.to_budget` lowers the agreed price
  and SLA into a :class:`~vincio.core.types.Budget` the runtime already enforces,
  and :meth:`Contract.check` compares delivered cost / latency / quality against
  the terms, returning a :class:`ContractFulfillment` (or raising
  :class:`~vincio.core.errors.ContractError`) — so a contract is a hard cap, not a
  hope.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import ContractError
from ..core.types import Budget
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow

if TYPE_CHECKING:
    from ..security.audit import ChainSigner

__all__ = [
    "ContractTerms",
    "ContractSignature",
    "Contract",
    "ContractVerification",
    "ContractFulfillment",
]


class ContractTerms(BaseModel):
    """The typed, negotiated terms of an agreement.

    The numeric dimensions a :class:`~vincio.negotiation.engine.Negotiation`
    converges on (``price_usd``, ``sla_seconds``, ``quality_floor``) plus the
    fixed ``scope`` the deal covers. Kept deliberately small and typed so the
    orchestrator can enforce it (:meth:`Contract.to_budget` / :meth:`Contract.check`)
    rather than parse a free-form clause.
    """

    scope: str = ""
    price_usd: float = 0.0
    sla_seconds: float = 0.0
    quality_floor: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def canonical(self) -> dict[str, Any]:
        """A stable, hashable projection of the terms (drops mutable metadata)."""
        return {
            "scope": self.scope,
            "price_usd": round(float(self.price_usd), 9),
            "sla_seconds": round(float(self.sla_seconds), 9),
            "quality_floor": round(float(self.quality_floor), 9),
        }


class ContractSignature(BaseModel):
    """One party's signature over a contract's content hash."""

    party: str
    signature: str
    key_id: str = ""


class ContractVerification(BaseModel):
    """The (non-raising) outcome of verifying a contract offline."""

    valid: bool
    hash_ok: bool
    signatures_ok: bool
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class ContractFulfillment(BaseModel):
    """Whether delivered work met the contract's terms, the enforcement verdict.

    A breach line reads ``"<term>: <delivered> > <agreed>"`` (or ``"< "`` for a
    quality floor), so a fulfilment is debuggable, not just a boolean.
    """

    fulfilled: bool
    breaches: list[str] = Field(default_factory=list)
    cost_usd: float | None = None
    latency_ms: float | None = None
    quality: float | None = None


class Contract(BaseModel):
    """A signed, audited, offline-verifiable agreement over typed terms.

    Produced by :meth:`~vincio.negotiation.engine.Negotiation.run` when both
    parties accept. Both sides :meth:`sign` it; :meth:`verify` checks it from the
    bytes alone; :meth:`to_budget` / :meth:`check` enforce it like a budget.
    """

    id: str = Field(default_factory=lambda: new_id("contract"))
    buyer: str
    seller: str
    terms: ContractTerms
    rounds: int = 0
    agreed_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[ContractSignature] = Field(default_factory=list)
    audit_id: str | None = None

    def compute_hash(self) -> str:
        """The content hash binding buyer, seller, terms, rounds, and timestamp."""
        return stable_hash(
            {
                "buyer": self.buyer,
                "seller": self.seller,
                "terms": self.terms.canonical(),
                "rounds": self.rounds,
                "agreed_at": self.agreed_at.isoformat(),
            },
            length=32,
        )

    def seal(self) -> Contract:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    @property
    def fully_signed(self) -> bool:
        """Both buyer and seller have signed."""
        signers = set(self.signed_by)
        return self.buyer in signers and self.seller in signers

    def sign(self, signer: ChainSigner, *, party: str) -> Contract:
        """Add ``party``'s signature over the content hash (sealing first if needed).

        Re-signing for the same party replaces its prior signature, so a contract
        cannot accumulate stale signatures for one identity.
        """
        if not self.content_hash:
            self.seal()
        sig = ContractSignature(
            party=party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != party]
        self.signatures.append(sig)
        return self

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> ContractVerification:
        """Verify the contract offline: the hash recomputes and signatures check.

        ``verifier`` checks each signature against the content hash (use the
        public half of the signing key for Ed25519). ``require`` names the parties
        that must have a verified signature (defaults to both buyer and seller);
        pass ``[]`` to verify the hash binding alone.
        """
        expected = self.compute_hash()
        hash_ok = bool(self.content_hash) and self.content_hash == expected
        if not hash_ok:
            return ContractVerification(
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
                # No verifier: trust the presence of a signature for the binding
                # check, but only a verifier can make ``valid`` mean "authentic".
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
        return ContractVerification(
            valid=valid,
            hash_ok=hash_ok,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(self, verifier: ChainSigner, *, require: list[str] | None = None) -> Contract:
        """Verify and raise :class:`ContractError` if the contract is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise ContractError(
                f"contract {self.id} failed verification: {result.reason}",
                details={"contract_id": self.id, "reason": result.reason},
            )
        return self

    def to_budget(self, base: Budget | None = None) -> Budget:
        """Lower the agreed price/SLA into a :class:`Budget` the runtime enforces.

        ``max_cost_usd`` becomes the agreed price and ``max_latency_ms`` the agreed
        SLA, so a run under this contract is held to the deal by the same hard-cap
        machinery as any other budget. Other limits inherit from ``base``.
        """
        budget = (base or Budget()).model_copy()
        if self.terms.price_usd > 0:
            budget.max_cost_usd = self.terms.price_usd
        if self.terms.sla_seconds > 0:
            budget.max_latency_ms = int(round(self.terms.sla_seconds * 1000))
        return budget

    def check(
        self,
        *,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        quality: float | None = None,
        raise_on_breach: bool = False,
    ) -> ContractFulfillment:
        """Compare delivered work against the terms — the enforcement verdict.

        A delivered cost above the price, a latency above the SLA, or a quality
        below the floor is a breach. With ``raise_on_breach`` a breach raises
        :class:`ContractError` carrying the breach lines; otherwise the
        :class:`ContractFulfillment` reports them.
        """
        breaches: list[str] = []
        if cost_usd is not None and self.terms.price_usd > 0 and cost_usd > self.terms.price_usd + 1e-9:
            breaches.append(f"price: {cost_usd:.6f} > {self.terms.price_usd:.6f}")
        if (
            latency_ms is not None
            and self.terms.sla_seconds > 0
            and latency_ms > self.terms.sla_seconds * 1000 + 1e-6
        ):
            breaches.append(f"sla: {latency_ms:.1f}ms > {self.terms.sla_seconds * 1000:.1f}ms")
        if (
            quality is not None
            and self.terms.quality_floor > 0
            and quality < self.terms.quality_floor - 1e-9
        ):
            breaches.append(f"quality: {quality:.4f} < {self.terms.quality_floor:.4f}")
        fulfillment = ContractFulfillment(
            fulfilled=not breaches,
            breaches=breaches,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality=quality,
        )
        if breaches and raise_on_breach:
            raise ContractError(
                f"contract {self.id} breached: {'; '.join(breaches)}",
                breaches=breaches,
                details={"contract_id": self.id, "seller": self.seller},
            )
        return fulfillment

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the contract for the audit chain."""
        return to_jsonable(
            {
                "contract_id": self.id,
                "buyer": self.buyer,
                "seller": self.seller,
                "terms": self.terms.canonical(),
                "rounds": self.rounds,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )
