"""Cross-org reputation attestation & portability.

Settlement, netting, and arbitration all **close the reputation loop** — but the
standing they earn lives inside one org's own
:class:`~vincio.optimize.reputation.ReputationLedger`. A *new* counterparty, with no
prior history, has no way to trust that standing without a hosted reputation
bureau. This module adds the missing rung: making earned standing **portable** —
a signed, offline-verifiable attestation an org issues over a counterparty's
standing, that a prospective counterparty verifies from the bytes alone and folds
into its negotiation weighting. It is reputation that travels the fabric, never a
central reputation service.

* **Typed attestation.** An org issues a :class:`ReputationAttestation` over a
  counterparty's standing, derived from its own
  :class:`~vincio.settlement.book.SettlementBook` and arbitration
  :class:`~vincio.settlement.arbitration.Resolution`\\ s — the outcomes that earned
  it: a fulfilled settlement as a success, a breached one or a dissent in
  arbitration as a failure. :func:`attest_reputation` (or
  :meth:`~vincio.settlement.book.SettlementBook.attest`) reads only those existing
  signed records and counts what it can recompute. The issuer signs the attestation
  with the same :class:`~vincio.security.audit.ChainSigner` a contract uses.
* **Offline-verifiable.** The attestation is content-bound the way a record is: an
  attestation hash binds the issuer, the subject, the evidence counts, the prior,
  and the source records read, and the issuer co-signs *that* hash.
  :meth:`ReputationAttestation.verify` recomputes it from the bytes alone — the hash
  matches and the attested reputation re-derives from the evidence counts — so a
  tampered score or a forged issuer is caught without the live issuer.
* **Combined into an evidence-weighted prior.** Several issuers' attestations
  :func:`combine_attestations` into a single bounded :class:`PortableReputation` —
  never a single self-asserted number. Because a Beta-Bernoulli posterior is
  conjugate, combining is *pooling the evidence*: each issuer contributes the
  successes and failures it observed, an issuer that vouches for itself is refused,
  a tampered or forged attestation is pinpointed and excluded, and the importer's
  own prior anchors the pooled posterior so a thin attestation barely moves it.
* **Same discipline.** The imported prior exposes ``weight(member_id) -> float``, so
  it drops into the *existing* negotiation / discovery path unchanged: a regressor is
  discounted under the same bounded ``[floor, 1]`` rule a local reputation is — never
  zeroed, never singled out, reversible — and a brand-new counterparty with no
  history falls back to the benefit-of-the-doubt prior. With a local ledger as the
  ``base``, a counterparty the importer already knows keeps its own earned standing
  and only an unknown one leans on the imported attestations.
* **Weighted by trust in the issuer.** Pooling every counted issuer's evidence with
  equal pull lets a clutch of unknown peers out-evidence a few the importer has lived
  through, and an adversary can spin up *Sybil* issuers that all vouch the same way. A
  :class:`TrustModel` (:func:`build_trust_model`) closes that gap: it scales each
  issuer's contributed evidence mass by the importer's **own trust in that issuer** —
  its earned standing as a counterparty in the ``base`` ledger — under a bounded,
  transitive web-of-trust. Trust composes at most ``max_depth`` hops (an issuer the
  importer trusts lends weight to the issuers *it* attests) with a per-hop decay, and
  every multiplier is bounded ``[trust_floor, 1]`` — so corroboration from a trusted
  peer counts for more than volume from an unknown one, an unknown issuer still counts
  (never zeroed), and a cluster of mutually-vouching unknown issuers cannot outvote a
  few corroborating trusted ones because pull follows *earned trust*, not issuer
  count. Issuer-trust weighting is opt-in (``trust`` / ``trust_config``); with neither,
  combining pools evidence with equal pull exactly as before.

:meth:`~vincio.core.app.ContextApp.attest_reputation` issues an attestation from an
app's own settlement book; :meth:`~vincio.core.app.ContextApp.import_reputation`
combines a bundle of attestations into the prior that weights the next negotiation.
Everything is dependency-free, deterministic, and offline — never a hosted
reputation service.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.diagnostics import note_suppressed
from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .record import SettlementRecord, SettlementSignature, _resolve_verifier

if TYPE_CHECKING:
    from ..security.audit import ChainSigner
    from .arbitration import Resolution

__all__ = [
    "AttestationConfig",
    "ReputationAttestation",
    "AttestationVerification",
    "AttestationVerdict",
    "AttestationRevocation",
    "RevocationVerification",
    "SubjectStanding",
    "PortableReputation",
    "TrustConfig",
    "IssuerTrust",
    "TrustModel",
    "build_trust_model",
    "attest_reputation",
    "revoke_attestation",
    "combine_attestations",
]

# The audit action an issued attestation is recorded under.
ATTESTATION_ACTION = "reputation_attestation"
# The audit action an issued revocation is recorded under.
REVOCATION_ACTION = "attestation_revocation"

_TOLERANCE = 1e-9
_SECONDS_PER_DAY = 86400.0


class AttestationConfig(BaseModel):
    """How attested evidence maps to a portable reputation and an aggregation weight.

    A portable reputation is the mean of a Beta-Bernoulli posterior over the
    *pooled* evidence several issuers attest: each fulfilled settlement is a success,
    each breach or arbitration dissent a failure. The prior gives a never-attested
    counterparty a sensible starting reputation (the benefit of the doubt) so an
    unknown party is neither trusted blindly nor frozen out; attested evidence then
    moves the score, and because the posterior is conjugate, combining issuers is
    simply summing their evidence into one posterior.

    * ``prior_success`` / ``prior_failure`` are the Beta prior's pseudo-counts. Their
      ratio is an unknown counterparty's starting reputation
      (``prior_success / (prior_success + prior_failure)``); their sum is how much
      attested evidence it takes to move it, so a thin attestation barely shifts the
      prior and only corroborating evidence from several issuers does.
    * ``weight_floor`` / ``weight_ceiling`` bound the aggregation weight a reputation
      maps to — the *same* ``[floor, 1]`` band a local reputation uses. The floor
      keeps a discounted counterparty's pull positive (it is discounted, never
      singled out or zeroed, and can recover); the ceiling (``1.0``) means an
      attestation only ever *lowers* a counterparty's standing relative to parity,
      never raises it past a party with no adverse history.
    * ``per_issuer_cap`` optionally caps how much evidence any single issuer
      contributes to the pool (scaling its successes and failures down together so
      its attested reputation is preserved but its *mass* is bounded), so no one
      issuer — however much it claims to have seen — can dominate the combined prior.
      ``None`` leaves each issuer's evidence uncapped.
    * ``half_life_days`` optionally **decays** an attestation's evidence by age when a
      combination is evaluated against an as-of clock: an attestation contributes
      ``0.5 ** (age_days / half_life_days)`` of its mass (its successes and failures
      scaled down together, so its attested reputation is preserved but its pull
      shrinks), so an old attestation decays out of the pooled prior toward the
      benefit-of-the-doubt rather than anchoring it forever. ``None`` leaves attested
      evidence undecayed (the default — a combination with no as-of clock never
      decays). Freshness is *importer policy*: the issuer's own ``horizon_days`` is the
      hard validity window, the importer's ``half_life_days`` the soft decay within it.

    The weight is ``weight_floor + (weight_ceiling − weight_floor) · reputation``, a
    monotonic map from reputation ``∈ (0, 1)`` to weight ``∈ [floor, ceiling]`` —
    identical to the local reputation's, so portable and local standing weigh a
    negotiation on the same scale.
    """

    prior_success: float = 2.0
    prior_failure: float = 1.0
    weight_floor: float = 0.1
    weight_ceiling: float = 1.0
    per_issuer_cap: float | None = None
    half_life_days: float | None = None

    def validate_coherent(self) -> AttestationConfig:
        """Raise :class:`SettlementError` unless the configuration is coherent."""
        if self.prior_success <= 0.0 or self.prior_failure <= 0.0:
            raise SettlementError(
                "attestation prior pseudo-counts must be positive; got "
                f"prior_success={self.prior_success}, prior_failure={self.prior_failure}",
                details={"prior_success": self.prior_success, "prior_failure": self.prior_failure},
            )
        if not 0.0 <= self.weight_floor <= self.weight_ceiling <= 1.0:
            raise SettlementError(
                "attestation weights must satisfy 0 ≤ weight_floor ≤ weight_ceiling ≤ 1; got "
                f"floor={self.weight_floor}, ceiling={self.weight_ceiling}",
                details={"weight_floor": self.weight_floor, "weight_ceiling": self.weight_ceiling},
            )
        if self.per_issuer_cap is not None and self.per_issuer_cap <= 0.0:
            raise SettlementError(
                f"per_issuer_cap must be positive when set; got {self.per_issuer_cap}",
                details={"per_issuer_cap": self.per_issuer_cap},
            )
        if self.half_life_days is not None and self.half_life_days <= 0.0:
            raise SettlementError(
                f"half_life_days must be positive when set; got {self.half_life_days}",
                details={"half_life_days": self.half_life_days},
            )
        return self

    def reputation_of(self, successes: float, failures: float) -> float:
        """Posterior-mean reputation for pooled ``successes`` / ``failures``."""
        s = max(0.0, successes)
        f = max(0.0, failures)
        numerator = self.prior_success + s
        denominator = self.prior_success + self.prior_failure + s + f
        return numerator / denominator if denominator > 0.0 else 0.0

    def weight_of(self, reputation: float) -> float:
        """Map a reputation ``∈ [0, 1]`` to an aggregation weight in the band."""
        clamped = min(1.0, max(0.0, reputation))
        span = self.weight_ceiling - self.weight_floor
        return round(self.weight_floor + span * clamped, 9)

    def capped(self, successes: float, failures: float) -> tuple[float, float]:
        """Scale one issuer's evidence down to ``per_issuer_cap`` mass (ratio kept)."""
        cap = self.per_issuer_cap
        evidence = successes + failures
        if cap is None or evidence <= cap or evidence <= 0.0:
            return successes, failures
        scale = cap / evidence
        return successes * scale, failures * scale

    def decay_factor(self, age_days: float) -> float:
        """The fraction of its mass an attestation aged ``age_days`` still contributes.

        ``0.5 ** (age_days / half_life_days)`` — one half-life halves the mass — or
        ``1.0`` when no half-life is set or the attestation is not yet aged (a
        future-dated or as-of-or-earlier attestation is undecayed, never amplified).
        """
        hl = self.half_life_days
        if hl is None or age_days <= 0.0:
            return 1.0
        return 0.5 ** (age_days / hl)

    def decayed(self, successes: float, failures: float, age_days: float) -> tuple[float, float]:
        """Scale one attestation's evidence down by its age-driven decay factor."""
        factor = self.decay_factor(age_days)
        if factor >= 1.0:
            return successes, failures
        return successes * factor, failures * factor


class TrustConfig(BaseModel):
    """How the importer's trust in an issuer scales the evidence it contributes.

    When attestations pool with equal pull, a clutch of unknown issuers can
    out-evidence a few the importer has lived through, and a cluster of *Sybil*
    issuers can manufacture standing by all vouching the same way. A trust model
    (:func:`build_trust_model`) closes that by scaling each issuer's contributed
    evidence *mass* by the importer's **own trust in that issuer** — a bounded,
    transitive web-of-trust over the standing each issuer has *as a counterparty* —
    so corroboration from a trusted peer counts for more than volume from an unknown
    one, without a central trust authority.

    * ``max_depth`` is the hard hop bound on transitivity: trust is anchored at the
      importer (hop ``0`` — issuers it has first-hand evidence for in its ``base``
      ledger), and an issuer the importer trusts can lend weight to the issuers *it*
      attests (hop ``1``), and so on up to ``max_depth`` hops. The bound keeps a long,
      unverifiable chain from manufacturing standing and the computation finite and
      deterministic. ``0`` uses direct (first-hand) trust only.
    * ``hop_decay`` ``∈ (0, 1]`` is the trust lent across **one** hop: a hop-``k``
      issuer's mapped trust is multiplied by ``hop_decay ** k``, so vouched-for trust
      attenuates with distance and a far issuer leans toward the floor.
    * ``trust_floor`` / ``trust_ceiling`` bound every trust multiplier to the same
      ``[floor, 1]`` band a reputation weight uses. The floor keeps an **unknown** (or
      heavily discounted) issuer's evidence counting — discounted, never zeroed or
      singled out, and recoverable — and the ceiling (``1.0``) means trust only ever
      *lowers* an issuer's pull relative to a fully-trusted one, never amplifies it.

    An issuer's earned standing ``r ∈ [0, 1]`` maps to a trust multiplier
    ``trust_floor + (trust_ceiling − trust_floor) · r`` — the *same* monotonic map a
    reputation uses for its aggregation weight — so trust and reputation weigh on one
    scale. Trust scales an issuer's *mass* (its successes and failures together), never
    the reputation it attests, exactly as the per-issuer cap and the age decay do.
    """

    max_depth: int = 1
    hop_decay: float = 0.5
    trust_floor: float = 0.1
    trust_ceiling: float = 1.0

    def validate_coherent(self) -> TrustConfig:
        """Raise :class:`SettlementError` unless the configuration is coherent."""
        if self.max_depth < 0:
            raise SettlementError(
                f"trust max_depth must be non-negative; got {self.max_depth}",
                details={"max_depth": self.max_depth},
            )
        if not 0.0 < self.hop_decay <= 1.0:
            raise SettlementError(
                f"trust hop_decay must satisfy 0 < hop_decay ≤ 1; got {self.hop_decay}",
                details={"hop_decay": self.hop_decay},
            )
        if not 0.0 <= self.trust_floor <= self.trust_ceiling <= 1.0:
            raise SettlementError(
                "trust weights must satisfy 0 ≤ trust_floor ≤ trust_ceiling ≤ 1; got "
                f"floor={self.trust_floor}, ceiling={self.trust_ceiling}",
                details={"trust_floor": self.trust_floor, "trust_ceiling": self.trust_ceiling},
            )
        return self

    def clamp_trust(self, value: float) -> float:
        """Clamp a trust value into the ``[trust_floor, trust_ceiling]`` band."""
        return round(min(self.trust_ceiling, max(self.trust_floor, value)), 9)

    def trust_of(self, reputation: float) -> float:
        """Map an issuer's earned standing ``∈ [0, 1]`` to a trust multiplier in band."""
        clamped = min(1.0, max(0.0, reputation))
        span = self.trust_ceiling - self.trust_floor
        return round(self.trust_floor + span * clamped, 9)


class AttestationVerification(BaseModel):
    """The (non-raising) outcome of verifying a reputation attestation offline."""

    valid: bool
    hash_ok: bool
    evidence_sound: bool
    signatures_ok: bool
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class ReputationAttestation(BaseModel):
    """A signed, offline-verifiable attestation of a counterparty's earned standing.

    Produced by :func:`attest_reputation` (or
    :meth:`~vincio.settlement.book.SettlementBook.attest`) from an issuer's own
    signed :class:`~vincio.settlement.record.SettlementRecord`\\ s and arbitration
    :class:`~vincio.settlement.arbitration.Resolution`\\ s. It carries the evidence
    counts that earned the standing — ``settled`` fulfilments, ``breached``
    settlements, and arbitration ``dissents`` — the Beta prior used to summarize
    them, the resulting :attr:`reputation`, and the reconciliation / resolution
    hashes the evidence was read from.

    The attestation hash (:meth:`compute_hash`) binds the issuer, the subject, the
    evidence, the prior, and the source hashes, so :meth:`verify` recomputes it from
    the bytes alone: the hash matches and the attested reputation re-derives from the
    evidence counts. The issuer co-signs that hash with the same
    :class:`~vincio.security.audit.ChainSigner` a contract or settlement record uses,
    so a tampered score or a forged issuer is caught without the live issuer.
    """

    id: str = Field(default_factory=lambda: new_id("attestation"))
    issuer: str
    subject: str

    # The evidence the issuer observed, read from its own signed records.
    settled: int = 0
    breached: int = 0
    dissents: int = 0

    # The Beta prior the issuer summarized the evidence under (carried so the
    # attested reputation is self-contained and re-derivable by a verifier).
    prior_success: float = 2.0
    prior_failure: float = 1.0
    reputation: float = 0.0

    source_hashes: list[str] = Field(default_factory=list)
    note: str = ""

    # The issuer's validity window: the attestation is fresh for ``horizon_days``
    # after ``issued_at``, after which an as-of-aware combination treats it as stale.
    # ``None`` means the issuer asserts no expiry (the standing holds until revoked).
    horizon_days: float | None = None

    issued_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived figures ----------------------------------------------------

    @property
    def successes(self) -> int:
        """The fulfilled settlements that count toward the standing."""
        return self.settled

    @property
    def failures(self) -> int:
        """The breaches and arbitration dissents that count against the standing."""
        return self.breached + self.dissents

    @property
    def settlements(self) -> int:
        """The settlement records the evidence was drawn from (settled + breached)."""
        return self.settled + self.breached

    @property
    def evidence(self) -> int:
        """The total attested evidence behind the reputation (successes + failures)."""
        return self.successes + self.failures

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    @property
    def is_self_attestation(self) -> bool:
        """The issuer is vouching for itself — refused when combining."""
        return self.issuer == self.subject

    # -- freshness ----------------------------------------------------------

    @property
    def expires_at(self) -> datetime | None:
        """The instant the validity window closes, or ``None`` if it never does."""
        if self.horizon_days is None:
            return None
        # Normalize issued_at: a cross-org attestation deserialized from a tz-naive
        # ISO string would otherwise make the comparison in is_stale raise.
        return _as_utc(self.issued_at) + timedelta(days=float(self.horizon_days))

    def age_days(self, as_of: datetime) -> float:
        """Days between issuance and ``as_of`` (never negative — clamped at ``0``)."""
        delta = (_as_utc(as_of) - _as_utc(self.issued_at)).total_seconds() / _SECONDS_PER_DAY
        return max(0.0, delta)

    def is_stale(self, as_of: datetime) -> bool:
        """Whether the issuer's validity window has closed by ``as_of``.

        An attestation with no declared ``horizon_days`` never goes stale (the issuer
        asserts the standing holds until it is revoked); one with a horizon is stale
        once ``as_of`` passes :attr:`expires_at`.
        """
        expiry = self.expires_at
        return expiry is not None and _as_utc(as_of) > expiry

    # -- hashing & signing --------------------------------------------------

    def attestation_facts(self) -> dict[str, Any]:
        """The facts the attestation hash binds (and the issuer signs).

        Deliberately excludes the attestation id and the timestamp: those are local
        metadata, not the attested standing. The issuer *is* bound — an attestation
        is one issuer's signed claim, not an issuer-independent recomputation — so a
        second issuer attesting the same evidence produces a distinct, separately
        signed attestation that combines beside this one.
        """
        facts: dict[str, Any] = {
            "issuer": self.issuer,
            "subject": self.subject,
            "settled": int(self.settled),
            "breached": int(self.breached),
            "dissents": int(self.dissents),
            "prior_success": round(float(self.prior_success), 9),
            "prior_failure": round(float(self.prior_failure), 9),
            "reputation": round(float(self.reputation), 9),
            "source_hashes": sorted(self.source_hashes),
        }
        # Bind the validity window only when the issuer declares one, so an
        # attestation with no horizon hashes exactly as it did before freshness
        # existed — a pre-3.30 attestation stays offline-verifiable unchanged.
        if self.horizon_days is not None:
            facts["horizon_days"] = round(float(self.horizon_days), 9)
        return facts

    def compute_hash(self) -> str:
        """The attestation hash binding the attested standing (id/time-independent)."""
        return stable_hash(self.attestation_facts(), length=32)

    def seal(self) -> ReputationAttestation:
        """Stamp the attestation hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    def sign(self, signer: ChainSigner, *, party: str | None = None) -> ReputationAttestation:
        """Add the issuer's signature over the attestation hash (sealing first).

        ``party`` defaults to the issuer — an attestation is the issuer's own signed
        claim. Re-signing for the same party replaces its prior signature, so an
        attestation cannot accumulate stale signatures for one identity.
        """
        signer_party = party or self.issuer
        if not self.content_hash:
            self.seal()
        sig = SettlementSignature(
            party=signer_party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != signer_party]
        self.signatures.append(sig)
        return self

    # -- verification -------------------------------------------------------

    def _evidence_sound(self) -> bool:
        """The attested reputation re-derives from the evidence counts and prior.

        Recomputes the posterior-mean reputation from ``settled`` / ``breached`` /
        ``dissents`` under the carried prior and checks it matches what was recorded,
        and that no count is negative — so a tampered score (a high reputation over
        low evidence, or vice versa) is caught even when the hash was recomputed to
        match.
        """
        if self.settled < 0 or self.breached < 0 or self.dissents < 0:
            return False
        if self.prior_success <= 0.0 or self.prior_failure <= 0.0:
            return False
        expected = round(
            AttestationConfig(
                prior_success=self.prior_success, prior_failure=self.prior_failure
            ).reputation_of(self.successes, self.failures),
            9,
        )
        return abs(expected - round(float(self.reputation), 9)) <= _TOLERANCE

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> AttestationVerification:
        """Verify the attestation offline: hash, evidence, and signatures.

        Recomputes the attestation hash from the stored fields, re-derives the
        attested reputation from the evidence counts to confirm it is sound, and —
        with a ``verifier`` — checks the issuer's signature. ``require`` names the
        parties that must have a verified signature (defaults to the issuer, whose
        signed claim the attestation is); pass ``[]`` to check the binding alone. A
        tampered score breaks the hash and, almost always, the evidence too.
        """
        expected = self.compute_hash()
        hash_ok = bool(self.content_hash) and self.content_hash == expected
        evidence_sound = self._evidence_sound()
        if not hash_ok:
            return AttestationVerification(
                valid=False,
                hash_ok=False,
                evidence_sound=evidence_sound,
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
        required = [self.issuer] if require is None else require
        missing = [p for p in required if p not in verified]
        if missing:
            signatures_ok = False
        valid = (
            hash_ok and evidence_sound and signatures_ok and (verifier is not None or not required)
        )
        reason = None
        if not evidence_sound:
            reason = "attested reputation does not re-derive from the evidence counts"
        elif not signatures_ok:
            reason = (
                f"missing/invalid signatures for {missing}" if missing else "signature mismatch"
            )
        elif verifier is None and required:
            reason = "no verifier supplied — signature present but not authenticated"
        return AttestationVerification(
            valid=valid,
            hash_ok=hash_ok,
            evidence_sound=evidence_sound,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner, *, require: list[str] | None = None
    ) -> ReputationAttestation:
        """Verify and raise :class:`SettlementError` if the attestation is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"attestation {self.id} failed verification: {result.reason}",
                details={"attestation_id": self.id, "reason": result.reason},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the attestation for the audit chain."""
        return to_jsonable(
            {
                "attestation_id": self.id,
                "issuer": self.issuer,
                "subject": self.subject,
                "settled": self.settled,
                "breached": self.breached,
                "dissents": self.dissents,
                "evidence": self.evidence,
                "reputation": self.reputation,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> ReputationAttestation:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the attested standing and the evidence behind it."""
        print(
            f"Attestation by {self.issuer} on {self.subject}: "
            f"reputation={self.reputation:.3f} "
            f"(✓{self.settled} settled / ✗{self.breached} breached / "
            f"!{self.dissents} dissents)"
        )


class RevocationVerification(BaseModel):
    """The (non-raising) outcome of verifying a revocation offline."""

    valid: bool
    hash_ok: bool
    signatures_ok: bool
    signed_by: list[str] = Field(default_factory=list)
    reason: str | None = None


class AttestationRevocation(BaseModel):
    """A signed, offline-verifiable withdrawal of a prior attestation by its hash.

    Standing changes: a counterparty reliable a year ago may have regressed, and an
    issuer may need to **withdraw** an attestation it can no longer stand behind. A
    revocation is the issuer's signed statement that a specific attestation — named by
    its :attr:`attestation_hash` (the attestation's ``content_hash``) — no longer
    holds. It optionally carries a :attr:`replacement_hash` (the attestation that
    *supersedes* the withdrawn one), making the revocation a supersession rather than a
    bare withdrawal.

    A revocation is content-bound and verifies the way an attestation does: the
    revocation hash binds the issuer, the subject, the withdrawn hash, and any
    replacement, and the issuer co-signs *that* hash. :meth:`verify` recomputes it from
    the bytes alone, so a forged revocation — one not signed by the very issuer whose
    attestation it withdraws — is refused rather than silently honored. In a
    combination only a revocation that both verifies and matches an attestation's own
    issuer withdraws it (:meth:`revokes`), so no org can cancel another's attestation.
    """

    id: str = Field(default_factory=lambda: new_id("revocation"))
    issuer: str
    subject: str

    # The ``content_hash`` of the attestation being withdrawn.
    attestation_hash: str
    # Optionally, the ``content_hash`` of the attestation that supersedes it.
    replacement_hash: str = ""
    reason: str = ""

    issued_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signatures: list[SettlementSignature] = Field(default_factory=list)
    audit_id: str | None = None

    # -- derived figures ----------------------------------------------------

    @property
    def signed_by(self) -> list[str]:
        """The parties that have signed, in signing order."""
        return [s.party for s in self.signatures]

    @property
    def is_supersession(self) -> bool:
        """Whether the revocation points to a replacement (a supersede, not a withdraw)."""
        return bool(self.replacement_hash)

    def revokes(self, attestation: ReputationAttestation) -> bool:
        """Whether this revocation withdraws ``attestation``.

        A revocation withdraws an attestation only when it names that attestation's
        ``content_hash`` *and* is issued by the same party — an issuer can withdraw
        only its own claim, never another org's.
        """
        return (
            bool(attestation.content_hash)
            and self.attestation_hash == attestation.content_hash
            and self.issuer == attestation.issuer
        )

    # -- hashing & signing --------------------------------------------------

    def revocation_facts(self) -> dict[str, Any]:
        """The facts the revocation hash binds (and the issuer signs).

        Excludes the id, the timestamp, and the human-readable ``reason`` (local
        metadata); binds the issuer, the subject, and the withdrawn / replacement
        hashes — the claim a verifier must be able to recompute.
        """
        return {
            "issuer": self.issuer,
            "subject": self.subject,
            "attestation_hash": self.attestation_hash,
            "replacement_hash": self.replacement_hash,
        }

    def compute_hash(self) -> str:
        """The revocation hash binding the withdrawal (id/time-independent)."""
        return stable_hash(self.revocation_facts(), length=32)

    def seal(self) -> AttestationRevocation:
        """Stamp the revocation hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    def sign(self, signer: ChainSigner, *, party: str | None = None) -> AttestationRevocation:
        """Add the issuer's signature over the revocation hash (sealing first).

        ``party`` defaults to the issuer — a revocation is the issuer's own signed
        statement. Re-signing for the same party replaces its prior signature.
        """
        signer_party = party or self.issuer
        if not self.content_hash:
            self.seal()
        sig = SettlementSignature(
            party=signer_party,
            signature=signer.sign(self.content_hash),
            key_id=getattr(signer, "key_id", ""),
        )
        self.signatures = [s for s in self.signatures if s.party != signer_party]
        self.signatures.append(sig)
        return self

    # -- verification -------------------------------------------------------

    def verify(
        self, verifier: ChainSigner | None = None, *, require: list[str] | None = None
    ) -> RevocationVerification:
        """Verify the revocation offline: hash and the issuer's signature.

        Recomputes the revocation hash from the stored fields and — with a
        ``verifier`` — checks the issuer's signature. ``require`` names the parties that
        must have a verified signature (defaults to the issuer, whose signed statement
        the revocation is); pass ``[]`` to check the binding alone.
        """
        expected = self.compute_hash()
        hash_ok = bool(self.content_hash) and self.content_hash == expected
        if not hash_ok:
            return RevocationVerification(
                valid=False,
                hash_ok=False,
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
        required = [self.issuer] if require is None else require
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
            reason = "no verifier supplied — signature present but not authenticated"
        return RevocationVerification(
            valid=valid,
            hash_ok=hash_ok,
            signatures_ok=signatures_ok,
            signed_by=verified,
            reason=reason,
        )

    def require_valid(
        self, verifier: ChainSigner, *, require: list[str] | None = None
    ) -> AttestationRevocation:
        """Verify and raise :class:`SettlementError` if the revocation is not valid."""
        result = self.verify(verifier, require=require)
        if not result.valid:
            raise SettlementError(
                f"revocation {self.id} failed verification: {result.reason}",
                details={"revocation_id": self.id, "reason": result.reason},
            )
        return self

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the revocation for the audit chain."""
        return to_jsonable(
            {
                "revocation_id": self.id,
                "issuer": self.issuer,
                "subject": self.subject,
                "attestation_hash": self.attestation_hash,
                "replacement_hash": self.replacement_hash,
                "supersession": self.is_supersession,
                "content_hash": self.content_hash,
                "signed_by": self.signed_by,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> AttestationRevocation:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the withdrawal and what it supersedes, if anything."""
        kind = "supersedes" if self.is_supersession else "withdraws"
        print(
            f"Revocation by {self.issuer} on {self.subject}: {kind} "
            f"{self.attestation_hash[:12]}…"
            + (f" → {self.replacement_hash[:12]}…" if self.is_supersession else "")
        )


class AttestationVerdict(BaseModel):
    """One submitted attestation's standing in a combination — pinpointed, not summed.

    An attestation is **admissible** when it verifies as a signed artifact — its
    hash recomputes, its reputation re-derives from its evidence, and (with a
    verifier) the issuer's signature checks. An admissible attestation is **counted**
    toward the pooled prior unless it is a self-attestation (an issuer vouching for
    itself), superseded by the same issuer's larger attestation for the subject,
    **revoked** by its issuer, or **stale** past its validity window against the as-of
    clock; an inadmissible (tampered or forged) one is refused outright. Every
    non-counted attestation's ``reason`` says why — and the ``revoked`` / ``stale``
    flags pinpoint a withdrawn or expired one — so it is named rather than silently
    dropped.
    """

    attestation_id: str
    issuer: str
    subject: str
    evidence: int = 0
    reputation: float = 0.0
    admissible: bool = True
    counted: bool = False
    revoked: bool = False
    stale: bool = False
    # The importer's trust multiplier applied to this issuer's evidence mass when the
    # attestation was counted (``1.0`` when no issuer-trust weighting is in effect).
    trust: float = 1.0
    reason: str | None = None


class SubjectStanding(BaseModel):
    """One subject's pooled, evidence-weighted standing across issuers.

    The combined posterior over every counted issuer's attested evidence for the
    subject: ``successes`` / ``failures`` are the pooled (optionally per-issuer
    capped) pseudo-counts, ``reputation`` is the importer-prior posterior mean over
    them, and ``weight`` is the bounded ``[floor, 1]`` aggregation weight it maps to.
    ``issuers`` names who corroborated the standing, so a prior is traceable to the
    independent attestations behind it.
    """

    subject: str
    successes: float = 0.0
    failures: float = 0.0
    reputation: float = 0.0
    weight: float = 1.0
    issuers: list[str] = Field(default_factory=list)
    attestations: int = 0
    # The trust multiplier each counted issuer's evidence was scaled by (empty when no
    # issuer-trust weighting is in effect — every issuer pooled with equal pull).
    issuer_trust: dict[str, float] = Field(default_factory=dict)

    @property
    def evidence(self) -> float:
        """The pooled evidence behind the standing (successes + failures)."""
        return round(self.successes + self.failures, 9)


class PortableReputation:
    """An imported, evidence-weighted prior combined from several issuers' attestations.

    Produced by :func:`combine_attestations` (or
    :meth:`~vincio.core.app.ContextApp.import_reputation`) from a bundle of signed
    :class:`ReputationAttestation`\\ s. It pools every admissible, non-self,
    non-superseded attestation's evidence per subject into one Beta-Bernoulli
    posterior under the importer's own prior, and exposes ``weight(member_id)`` — so
    it drops into the *existing* negotiation / discovery path (a
    :class:`~vincio.negotiation.engine.LocalParty`,
    :func:`~vincio.negotiation.engine.select_offer`, or a
    :class:`~vincio.choreography.discovery.CapabilityBinder`) exactly where a local
    :class:`~vincio.optimize.reputation.ReputationLedger` would.

    Reputation that travels: a prospective counterparty with no local history is
    weighted by what its past counterparties attest, under the same bounded
    ``[floor, 1]`` rule a local reputation is — discounted without being singled out,
    reversible. A tampered or forged attestation is pinpointed (:attr:`refused`) and
    excluded, and an issuer that vouches for itself is refused — so the prior is
    evidence-weighted across independent issuers, never a single self-asserted
    number. With a local ledger passed as ``base``, a counterparty the importer
    already knows keeps its own earned standing and only an unknown one leans on the
    imported attestations.
    """

    def __init__(
        self,
        standings: dict[str, SubjectStanding],
        verdicts: list[AttestationVerdict],
        config: AttestationConfig,
        *,
        base: Any | None = None,
        as_of: datetime | None = None,
        trust: Any | None = None,
    ) -> None:
        self.config = config
        self._standings = standings
        self.verdicts = verdicts
        self.base = base
        self.as_of = as_of
        self.trust = trust

    # -- reads --------------------------------------------------------------

    def subjects(self) -> list[str]:
        """Every subject an admissible attestation contributed standing for, sorted."""
        return sorted(self._standings)

    def standing(self, member_id: str) -> SubjectStanding | None:
        """The pooled standing for ``member_id``, or ``None`` if none was attested."""
        return self._standings.get(member_id)

    def issuers_for(self, member_id: str) -> list[str]:
        """The issuers that corroborated ``member_id``'s standing, sorted."""
        standing = self._standings.get(member_id)
        return list(standing.issuers) if standing is not None else []

    def reputation(self, member_id: str) -> float:
        """The pooled posterior-mean reputation for ``member_id`` ``∈ (0, 1)``.

        A member no admissible attestation covers returns the prior mean — the
        benefit of the doubt an unknown counterparty is extended — matching how a
        local ledger treats a never-seen member.
        """
        standing = self._standings.get(member_id)
        if standing is None:
            return round(self.config.reputation_of(0.0, 0.0), 9)
        return standing.reputation

    def weight(self, member_id: str) -> float:
        """The aggregation weight for ``member_id`` ``∈ [weight_floor, 1.0]``.

        The drop-in that weights a negotiation: when a local ``base`` ledger already
        has earned evidence for ``member_id`` its local weight wins (the importer
        trusts what it has lived through over what others attest); otherwise the
        imported, pooled prior decides; and a member neither knows falls back to the
        benefit-of-the-doubt prior weight.
        """
        if self.base is not None and _has_local_evidence(self.base, member_id):
            try:
                return float(self.base.weight(member_id))
            except Exception:
                note_suppressed("settlement.attestation.base_weight")
        return self.config.weight_of(self.reputation(member_id))

    def trust_in(self, issuer: str) -> float:
        """The importer's trust multiplier applied to ``issuer``'s attested evidence.

        ``1.0`` (full pull) when no issuer-trust weighting was in effect; otherwise the
        bounded ``[trust_floor, 1]`` multiplier the :class:`TrustModel` resolved for the
        issuer — the floor for one the importer neither knows nor reaches transitively.
        """
        return _trust_multiplier(self.trust, issuer)

    # -- verdicts -----------------------------------------------------------

    @property
    def counted(self) -> list[AttestationVerdict]:
        """The attestations that contributed to the pooled prior."""
        return [v for v in self.verdicts if v.counted]

    @property
    def admitted(self) -> list[AttestationVerdict]:
        """The attestations that verified as artifacts (counted or not)."""
        return [v for v in self.verdicts if v.admissible]

    @property
    def refused(self) -> list[AttestationVerdict]:
        """The attestations refused for tampering or a forged issuer signature."""
        return [v for v in self.verdicts if not v.admissible]

    @property
    def excluded(self) -> list[AttestationVerdict]:
        """Verified attestations not counted — self-vouching, superseded, revoked, or stale."""
        return [v for v in self.verdicts if v.admissible and not v.counted]

    @property
    def revoked(self) -> list[AttestationVerdict]:
        """Verified attestations excluded because their issuer withdrew them."""
        return [v for v in self.verdicts if v.revoked]

    @property
    def stale(self) -> list[AttestationVerdict]:
        """Verified attestations excluded as past their validity window at the as-of clock."""
        return [v for v in self.verdicts if v.stale]

    def verdict_for(self, issuer: str, subject: str) -> AttestationVerdict | None:
        """The verdict for one issuer's attestation about ``subject``, or ``None``."""
        return next((v for v in self.verdicts if v.issuer == issuer and v.subject == subject), None)

    # -- reporting ----------------------------------------------------------

    def standings(self) -> list[SubjectStanding]:
        """Every subject's pooled standing, sorted by subject."""
        return [self._standings[s] for s in sorted(self._standings)]

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print each subject's pooled standing and how the attestations were used."""
        print(
            f"Portable reputation: {len(self.subjects())} subject(s) from "
            f"{len(self.counted)} counted / {len(self.refused)} refused / "
            f"{len(self.excluded)} excluded attestation(s)"
        )
        for standing in self.standings():
            print(
                f"  {standing.subject}: reputation={standing.reputation:.3f} "
                f"weight={standing.weight:.3f} (evidence={standing.evidence:g}, "
                f"issuers={standing.issuers})"
            )
        for verdict in self.refused + self.excluded:
            print(f"  ! {verdict.issuer}→{verdict.subject}: {verdict.reason}")


class IssuerTrust(BaseModel):
    """The importer's resolved trust in one issuer, pinpointed, never silent.

    Produced by :func:`build_trust_model`. ``trust`` is the bounded
    ``[trust_floor, 1]`` multiplier the issuer's attested evidence is scaled by;
    ``depth`` is its hop distance from the importer (``0`` for an issuer the importer
    has first-hand evidence for, ``k`` for one a hop-``k`` chain of trusted issuers
    vouches for, and ``None`` for one neither known nor reached — which falls back to
    the floor). ``vouched_by`` names the trusted issuers that lent transitive trust
    (empty for a direct or unreached issuer), and ``reputation`` is the standing the
    transitive trust was derived from — so an issuer's pull is always traceable to who
    vouched for it, never an opaque number.
    """

    issuer: str
    trust: float
    depth: int | None = None
    direct: bool = False
    vouched_by: list[str] = Field(default_factory=list)
    reputation: float | None = None

    @property
    def transitive(self) -> bool:
        """Whether the trust was lent across at least one hop (not first-hand)."""
        return self.depth is not None and self.depth > 0


class TrustModel:
    """The importer's bounded, transitive trust in each issuer, the Sybil-resistant kernel.

    Produced by :func:`build_trust_model` from the importer's own ``base`` ledger and
    the attestations on hand. It resolves, for each issuer, a trust multiplier in the
    ``[trust_floor, 1]`` band, computed by a bounded web-of-trust:

    * **hop 0 (direct).** An issuer the importer has first-hand evidence for in its
      ``base`` :class:`~vincio.optimize.reputation.ReputationLedger` is trusted exactly
      as much as that ledger weights it (clamped into the trust band).
    * **hops 1..max_depth (transitive).** An issuer the importer does *not* know
      directly, but that an *already-trusted* issuer attests (the trusted issuer
      vouches for it as a counterparty), inherits trust derived from that pooled
      standing, attenuated by ``hop_decay`` per hop. The pool is itself weighted by the
      voucher's trust, so a chain only lends as much as its weakest trusted link.
    * **unreached.** An issuer neither known nor reachable from a trusted root within
      ``max_depth`` hops falls back to ``trust_floor`` — counted, never zeroed.

    The kernel is **Sybil-resistant** because trust is lent only *outward from a
    trusted root*: a cluster of mutually-vouching unknown issuers is never reached from
    the importer's ledger, so every member stays at the floor however much it vouches —
    pull follows earned trust, not issuer count. It is deterministic and offline, and
    quacks like a ledger (:meth:`weight` aliases :meth:`trust_in`) so it drops in as the
    ``trust`` argument to :func:`combine_attestations`.
    """

    def __init__(self, assessments: dict[str, IssuerTrust], config: TrustConfig) -> None:
        self.config = config
        self._assessments = assessments

    def trust_in(self, issuer: str) -> float:
        """The trust multiplier for ``issuer`` — the floor for an unknown one."""
        assessment = self._assessments.get(issuer)
        return assessment.trust if assessment is not None else self.config.trust_floor

    def weight(self, issuer: str) -> float:
        """Alias of :meth:`trust_in` so the model is interchangeable with a ledger."""
        return self.trust_in(issuer)

    def assessment(self, issuer: str) -> IssuerTrust | None:
        """The full :class:`IssuerTrust` for ``issuer``, or ``None`` if unassessed."""
        return self._assessments.get(issuer)

    def issuers(self) -> list[str]:
        """Every issuer the model resolved a non-floor trust for, sorted."""
        return sorted(self._assessments)

    def direct_issuers(self) -> list[str]:
        """The issuers trusted first-hand from the base ledger (hop 0), sorted."""
        return sorted(i for i, a in self._assessments.items() if a.direct)

    def transitive_issuers(self) -> list[str]:
        """The issuers trusted transitively (hop ≥ 1), sorted."""
        return sorted(i for i, a in self._assessments.items() if a.transitive)

    def assessments(self) -> list[IssuerTrust]:
        """Every resolved :class:`IssuerTrust`, sorted by issuer."""
        return [self._assessments[i] for i in sorted(self._assessments)]

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print each issuer's resolved trust and how it was reached."""
        print(
            f"Trust model: {len(self.direct_issuers())} direct / "
            f"{len(self.transitive_issuers())} transitive issuer(s), "
            f"floor={self.config.trust_floor:g}"
        )
        for a in self.assessments():
            how = "direct" if a.direct else f"hop {a.depth} via {a.vouched_by}"
            print(f"  {a.issuer}: trust={a.trust:.3f} ({how})")


def _as_utc(moment: datetime) -> datetime:
    """Normalize a possibly-naive datetime to UTC for a stable age computation.

    :func:`~vincio.core.utils.utcnow` stamps ``issued_at`` tz-aware, so a naive
    ``as_of`` (or revocation time) is assumed to already be UTC rather than rejected.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment


def _has_local_evidence(base: Any, member_id: str) -> bool:
    """Whether a local ledger already has earned evidence for ``member_id``."""
    snapshot = getattr(base, "snapshot", None)
    if callable(snapshot):
        try:
            return int(getattr(snapshot(member_id), "rounds", 0)) > 0
        except Exception:
            note_suppressed("settlement.attestation.membership_snapshot")
    members = getattr(base, "members", None)
    if callable(members):
        try:
            return member_id in members()
        except Exception:
            note_suppressed("settlement.attestation.membership_probe")
            return False
    return False


# -- issuing ------------------------------------------------------------------


def attest_reputation(
    records: Iterable[SettlementRecord],
    subject: str,
    *,
    issuer: str = "",
    resolutions: Iterable[Resolution] | None = None,
    config: AttestationConfig | None = None,
    verifier: ChainSigner | None = None,
    horizon_days: float | None = None,
    note: str = "",
    verify_with: ChainSigner | None = None,
) -> ReputationAttestation:
    """Issue an attestation of ``subject``'s earned standing from signed records.

    Reads the issuer's own signed
    :class:`~vincio.settlement.record.SettlementRecord`\\ s — counting the ones where
    ``subject`` was the **seller** (the delivery reliability a negotiation cares
    about): a fulfilled settlement is a success, a breach a failure — and its
    arbitration :class:`~vincio.settlement.arbitration.Resolution`\\ s — counting a
    dissent against ``subject`` (a claim that did not stand) as a failure. It reads
    only what it can recompute: a record whose reconciliation hash no longer
    recomputes (or, with ``verifier``, whose signature is forged) is skipped, so a
    tampered own record cannot inflate the standing, and the exact source hashes the
    evidence came from are bound into the attestation. ``verify_with`` is a
    deprecated alias for ``verifier`` (since 7.5, removed in 8.0).

    ``horizon_days`` optionally declares a **validity window**: the issuer asserts the
    standing holds for that many days after issuance, after which an as-of-aware
    combination treats the attestation as stale. ``None`` (the default) asserts no
    expiry — the standing holds until the issuer revokes it.

    The returned attestation is sealed but unsigned — sign it with the issuer's key
    (or let :meth:`~vincio.settlement.book.SettlementBook.attest` do it). Raises
    :class:`SettlementError` when there is no admissible history for the subject to
    attest — an attestation asserts evidence, never a bare prior.
    """
    verifier = _resolve_verifier(verifier, verify_with, "attest_reputation")
    cfg = (config or AttestationConfig()).validate_coherent()
    if horizon_days is not None and horizon_days <= 0.0:
        raise SettlementError(
            f"horizon_days must be positive when set; got {horizon_days}",
            details={"subject": subject, "issuer": issuer, "horizon_days": horizon_days},
        )

    counted: list[SettlementRecord] = []
    for record in records:
        if record.seller != subject:
            continue
        if record.content_hash and record.content_hash != record.compute_hash():
            continue  # tampered: its reconciliation hash no longer recomputes
        if verifier is not None and record.signatures:
            if not record.verify(verifier, require=[]).signatures_ok:
                continue  # forged: a signature does not check
        counted.append(record)

    settled = sum(1 for r in counted if r.status == "settled")
    breached = sum(1 for r in counted if r.status == "breached")

    dissents = 0
    resolution_hashes: list[str] = []
    for resolution in resolutions or []:
        if not getattr(resolution, "upheld", False):
            continue
        if subject not in getattr(resolution, "dissenters", []):
            continue
        if not resolution.verify(verifier).decision_sound:
            continue  # a tampered resolution cannot inflate dissents
        dissents += 1
        if resolution.content_hash:
            resolution_hashes.append(resolution.content_hash)

    if not counted and dissents == 0:
        raise SettlementError(
            f"no admissible settled history with {subject!r} to attest a reputation over",
            details={"subject": subject, "issuer": issuer},
        )

    source_hashes = sorted(
        {r.content_hash for r in counted if r.content_hash} | set(resolution_hashes)
    )
    attestation = ReputationAttestation(
        issuer=issuer,
        subject=subject,
        settled=settled,
        breached=breached,
        dissents=dissents,
        prior_success=cfg.prior_success,
        prior_failure=cfg.prior_failure,
        source_hashes=source_hashes,
        horizon_days=horizon_days,
        note=note,
    )
    # Derive the reputation from the attestation's own evidence properties (the same
    # successes / failures :meth:`_evidence_sound` re-derives from), so the mapping
    # has a single source and a freshly-issued attestation always verifies.
    attestation.reputation = round(
        cfg.reputation_of(attestation.successes, attestation.failures), 9
    )
    return attestation.seal()


# -- revoking -----------------------------------------------------------------


def revoke_attestation(
    attestation: ReputationAttestation | str,
    *,
    subject: str = "",
    issuer: str = "",
    replacement: ReputationAttestation | str | None = None,
    reason: str = "",
) -> AttestationRevocation:
    """Issue a revocation withdrawing a prior attestation by its hash.

    Pass the :class:`ReputationAttestation` being withdrawn (its issuer, subject, and
    ``content_hash`` are read from it) or, offline, the attestation's ``content_hash``
    as a string together with the ``subject`` and ``issuer`` it covered. ``replacement``
    optionally names the attestation (or its hash) that *supersedes* the withdrawn one,
    making the revocation a supersession.

    The returned revocation is sealed but unsigned — sign it with the issuer's key (or
    let :meth:`~vincio.settlement.book.SettlementBook.revoke` do it). Raises
    :class:`SettlementError` when the attestation to withdraw has no content hash to
    bind (an unsealed attestation), or when an explicit ``issuer`` / ``subject``
    contradicts the attestation's own.
    """
    if isinstance(attestation, ReputationAttestation):
        att_hash = attestation.content_hash or attestation.seal().content_hash
        att_issuer = attestation.issuer
        att_subject = attestation.subject
        if issuer and issuer != att_issuer:
            raise SettlementError(
                f"issuer {issuer!r} cannot revoke an attestation issued by {att_issuer!r}",
                details={"issuer": issuer, "attestation_issuer": att_issuer},
            )
        if subject and subject != att_subject:
            raise SettlementError(
                f"subject {subject!r} does not match the attestation's subject {att_subject!r}",
                details={"subject": subject, "attestation_subject": att_subject},
            )
        issuer, subject = att_issuer, att_subject
    else:
        att_hash = attestation
        if not att_hash:
            raise SettlementError(
                "a revocation must name the content hash of the attestation it withdraws",
                details={"issuer": issuer, "subject": subject},
            )

    if not att_hash:
        raise SettlementError(
            "the attestation to revoke has no content hash — seal it before revoking",
            details={"issuer": issuer, "subject": subject},
        )

    replacement_hash = ""
    if replacement is not None:
        if isinstance(replacement, ReputationAttestation):
            replacement_hash = replacement.content_hash or replacement.seal().content_hash
        else:
            replacement_hash = replacement

    revocation = AttestationRevocation(
        issuer=issuer,
        subject=subject,
        attestation_hash=att_hash,
        replacement_hash=replacement_hash,
        reason=reason,
    )
    return revocation.seal()


# -- trust --------------------------------------------------------------------


def _admissible(att: ReputationAttestation, verifier: ChainSigner | None) -> bool:
    """Whether an attestation verifies as an artifact (hash, evidence, signature).

    The same admissibility :func:`combine_attestations` applies, minus the policy
    exclusions (self-attestation, revoked, stale) — a tampered or forged attestation
    must not lend trust any more than it pools evidence.
    """
    check = att.verify(verifier, require=[])
    if not check.hash_ok or not check.evidence_sound:
        return False
    if verifier is not None and att.signatures and not check.signatures_ok:
        return False
    return True


def _trust_multiplier(trust: Any | None, issuer: str) -> float:
    """Resolve an issuer's trust multiplier from a model, ledger, or callable.

    Accepts a :class:`TrustModel` (or anything exposing ``trust_in`` / ``weight``) or a
    plain ``issuer -> float`` callable, and clamps the result to ``[0, 1]`` so a custom
    source can only ever *discount* an issuer's pull, never amplify it. ``None`` (no
    weighting) maps every issuer to full pull (``1.0``).
    """
    if trust is None:
        return 1.0
    value: float = 1.0
    resolver = getattr(trust, "trust_in", None) or getattr(trust, "weight", None)
    try:
        if resolver is not None:
            value = float(resolver(issuer))
        elif callable(trust):
            value = float(trust(issuer))
    except Exception:
        note_suppressed("settlement.attestation.trust_source")
        return 1.0
    return min(1.0, max(0.0, value))


def build_trust_model(
    attestations: Iterable[ReputationAttestation],
    *,
    base: Any | None = None,
    config: TrustConfig | None = None,
    attestation_config: AttestationConfig | None = None,
    verifier: ChainSigner | None = None,
    verify_with: ChainSigner | None = None,
) -> TrustModel:
    """Build the importer's bounded, transitive trust over a set of issuers.

    Roots trust at the importer's own ``base``
    :class:`~vincio.optimize.reputation.ReputationLedger` — an issuer the importer has
    first-hand evidence for is trusted as much as that ledger weights it (hop ``0``) —
    then lends trust outward one hop at a time, up to ``config.max_depth``: an
    already-trusted issuer that attests another issuer (vouches for it *as a
    counterparty*) lends it trust derived from that pooled standing, attenuated by
    ``config.hop_decay`` per hop. Only admissible attestations (hash recomputes,
    reputation re-derives, and — with ``verifier`` — the issuer signature checks)
    count toward vouching, and an issuer vouching for itself never bootstraps its own
    trust. Every multiplier is bounded into the ``[trust_floor, 1]`` band; an issuer
    neither known nor reached falls back to the floor.

    Because trust is only ever lent *outward from the trusted root*, a cluster of
    mutually-vouching unknown issuers is never reached and every member stays at the
    floor — the Sybil-resistance property. Pass the resulting :class:`TrustModel` as
    the ``trust`` argument to :func:`combine_attestations`, or let it build one for you
    by passing a ``trust_config``. ``verify_with`` is a deprecated alias for
    ``verifier`` (since 7.5, removed in 8.0).
    """
    verifier = _resolve_verifier(verifier, verify_with, "build_trust_model")
    cfg = (config or TrustConfig()).validate_coherent()
    acfg = (attestation_config or AttestationConfig()).validate_coherent()
    admissible = [a for a in attestations if _admissible(a, verifier)]
    issuers = sorted({a.issuer for a in admissible})

    assessments: dict[str, IssuerTrust] = {}

    # Hop 0: direct, first-hand trust from the importer's own ledger.
    if base is not None:
        for issuer in issuers:
            if not _has_local_evidence(base, issuer):
                continue
            try:
                local_weight = float(base.weight(issuer))
            except Exception:
                note_suppressed("settlement.attestation.issuer_weight")
                continue
            assessments[issuer] = IssuerTrust(
                issuer=issuer,
                trust=cfg.clamp_trust(local_weight),
                depth=0,
                direct=True,
            )

    # Index admissible attestations by the subject they vouch for, so a hop step can
    # find who an already-trusted issuer attests.
    by_subject: dict[str, list[ReputationAttestation]] = {}
    for att in admissible:
        by_subject.setdefault(att.subject, []).append(att)

    # Hops 1..max_depth: lend trust outward from the established (lower-depth) issuers.
    for hop in range(1, cfg.max_depth + 1):
        established = {i: a for i, a in assessments.items() if a.depth is not None and a.depth < hop}
        newly: dict[str, IssuerTrust] = {}
        for issuer in issuers:
            if issuer in assessments:
                continue  # already reached at a shorter (stronger) distance
            vouchers = [
                a for a in by_subject.get(issuer, []) if a.issuer in established and a.issuer != issuer
            ]
            if not vouchers:
                continue
            # Keep each voucher-issuer's largest attestation, then pool its evidence
            # weighted by the voucher's own trust — a chain lends as much as its link.
            best_voucher: dict[str, ReputationAttestation] = {}
            for att in vouchers:
                current = best_voucher.get(att.issuer)
                if current is None or _supersedes(att, current):
                    best_voucher[att.issuer] = att
            successes = failures = 0.0
            for voucher_id, att in best_voucher.items():
                voucher_trust = established[voucher_id].trust
                successes += voucher_trust * att.successes
                failures += voucher_trust * att.failures
            reputation = acfg.reputation_of(successes, failures)
            lent = cfg.trust_of(reputation) * (cfg.hop_decay**hop)
            newly[issuer] = IssuerTrust(
                issuer=issuer,
                trust=cfg.clamp_trust(lent),
                depth=hop,
                direct=False,
                vouched_by=sorted(best_voucher),
                reputation=round(reputation, 9),
            )
        if not newly:
            break  # the frontier is exhausted — no further hop can reach anyone new
        assessments.update(newly)

    return TrustModel(assessments, cfg)


# -- combining ----------------------------------------------------------------


def combine_attestations(
    attestations: Iterable[ReputationAttestation],
    *,
    subject: str | None = None,
    config: AttestationConfig | None = None,
    verifier: ChainSigner | None = None,
    base: Any | None = None,
    allow_self: bool = False,
    revocations: Iterable[AttestationRevocation] | None = None,
    as_of: datetime | None = None,
    trust: Any | None = None,
    trust_config: TrustConfig | None = None,
    verify_with: ChainSigner | None = None,
) -> PortableReputation:
    """Combine several issuers' attestations into one bounded, evidence-weighted prior.

    Verifies each attestation offline (hash recomputes, attested reputation
    re-derives from its evidence, and — with ``verifier`` — the issuer's signature
    checks), refusing and **pinpointing** a tampered or forged one rather than
    silently dropping it. Among the admissible attestations it pools the evidence per
    subject into one Beta-Bernoulli posterior under the importer's ``config`` prior —
    so combining is summing what each issuer observed, never averaging opaque scores.
    An issuer that vouches for itself (``issuer == subject``) is refused unless
    ``allow_self``, and when one issuer submits several attestations for the same
    subject only its largest (most evidence) is counted, so no issuer can stack its
    own pull. ``per_issuer_cap`` on the config caps any single issuer's contributed
    mass.

    **Revocation.** ``revocations`` are signed :class:`AttestationRevocation`\\ s; an
    attestation an *admissible, issuer-matched* revocation withdraws is excluded from
    the combination and pinpointed (``revoked``), never silently honored — a forged
    revocation (one whose signature does not check under ``verifier``, or that
    names another org's attestation) is itself ignored, so no org can cancel another's
    claim.

    **Freshness.** With an ``as_of`` clock, an attestation past its issuer-declared
    validity window (:attr:`ReputationAttestation.horizon_days`) is excluded as stale
    and pinpointed (``stale``); within the window, the config's ``half_life_days``
    **decays** its evidence by age, so an old attestation decays out of the pooled
    prior toward the benefit-of-the-doubt rather than anchoring it forever. Without an
    ``as_of`` clock no attestation expires or decays — the combination is point-in-time.

    **Issuer trust (Sybil resistance).** By default every counted issuer's evidence
    pools with equal pull, weighted only by *how much* it attests. Pass a ``trust``
    source (a :class:`TrustModel`, anything exposing ``trust_in`` / ``weight``, or an
    ``issuer -> float`` callable) — or a ``trust_config`` to build a :class:`TrustModel`
    from ``base`` and these attestations automatically — to scale each issuer's
    contributed mass by the importer's **own trust in that issuer**, bounded
    ``[trust_floor, 1]``. Corroboration from a trusted peer then counts for more than
    volume from an unknown one, an unknown issuer still counts (floored, never zeroed),
    and a cluster of mutually-vouching unknown issuers cannot outvote a few trusted
    ones — pull follows earned trust, not issuer count. Each counted attestation's
    applied multiplier is pinpointed (:attr:`AttestationVerdict.trust`) and the per-
    issuer multipliers surface on :attr:`SubjectStanding.issuer_trust`.

    ``subject`` optionally restricts the combination to one counterparty; ``base`` is
    an optional local :class:`~vincio.optimize.reputation.ReputationLedger` whose
    earned standing wins for a counterparty the importer already knows. Returns a
    :class:`PortableReputation` exposing ``weight(member_id)`` for the negotiation
    path. ``verify_with`` is a deprecated alias for ``verifier`` (since 7.5,
    removed in 8.0).
    """
    verifier = _resolve_verifier(verifier, verify_with, "combine_attestations")
    cfg = (config or AttestationConfig()).validate_coherent()
    all_atts = list(attestations)
    items = [a for a in all_atts if subject is None or a.subject == subject]
    clock = _as_utc(as_of) if as_of is not None else None

    trust_model = _resolve_trust_model(trust, trust_config, all_atts, base, cfg, verifier)
    revoked_keys = _honored_revocations(revocations, subject, verifier)
    verdicts, admissible = _admissibility_verdicts(
        items, revoked_keys, clock, allow_self, verifier
    )
    best = _dedup_best(admissible)
    counted_ids = {att.id for att in best.values()}
    _mark_counted(verdicts, counted_ids, trust_model)
    pooled = _pool_evidence(best, cfg, clock, trust_model)
    standings = _build_standings(pooled, cfg)

    return PortableReputation(
        standings, verdicts, cfg, base=base, as_of=as_of, trust=trust_model
    )


def _supersedes(candidate: ReputationAttestation, current: ReputationAttestation) -> bool:
    """Whether ``candidate`` should replace ``current`` for one (subject, issuer).

    Prefers the attestation that read more evidence (it covers more history); ties
    break to the later issue time, then deterministically by id, so the dedup is
    stable regardless of submission order.
    """
    if candidate.evidence != current.evidence:
        return candidate.evidence > current.evidence
    if candidate.issued_at != current.issued_at:
        return candidate.issued_at > current.issued_at
    return candidate.id > current.id


def _resolve_trust_model(
    trust: Any | None,
    trust_config: TrustConfig | None,
    all_atts: list[ReputationAttestation],
    base: Any | None,
    cfg: AttestationConfig,
    verifier: ChainSigner | None,
) -> Any | None:
    # Issuer-trust weighting (opt-in). An explicit ``trust`` source wins; otherwise a
    # ``trust_config`` builds the bounded transitive model from the *full* attestation
    # set (so a trusted issuer can vouch for another even when ``subject`` is set), the
    # importer's ``base`` ledger rooting it. With neither, no weighting is applied and
    # every issuer pools with equal pull — byte-for-byte the pre-trust behavior.
    trust_model = trust
    if trust_model is None and trust_config is not None:
        trust_model = build_trust_model(
            all_atts,
            base=base,
            config=trust_config,
            attestation_config=cfg,
            verifier=verifier,
        )
    return trust_model


def _honored_revocations(
    revocations: Iterable[AttestationRevocation] | None,
    subject: str | None,
    verifier: ChainSigner | None,
) -> dict[tuple[str, str], AttestationRevocation]:
    # 0. The set of attestation hashes withdrawn by an admissible, issuer-matched
    #    revocation. A revocation is honored only when it verifies as an artifact (and,
    #    with a verifier, the issuer signature checks) — a forged or unsigned-when-
    #    -required revocation is ignored, so no org can cancel another's attestation.
    revoked_keys: dict[tuple[str, str], AttestationRevocation] = {}
    for rev in revocations or []:
        if subject is not None and rev.subject != subject:
            continue
        if not rev.verify(verifier, require=[]).hash_ok:
            continue  # tampered revocation — its hash does not recompute
        if verifier is not None and rev.signatures:
            if not rev.verify(verifier, require=[]).signatures_ok:
                continue  # forged revocation signature
        revoked_keys[(rev.issuer, rev.attestation_hash)] = rev
    return revoked_keys


def _admissibility_verdicts(
    items: list[ReputationAttestation],
    revoked_keys: dict[tuple[str, str], AttestationRevocation],
    clock: datetime | None,
    allow_self: bool,
    verifier: ChainSigner | None,
) -> tuple[list[AttestationVerdict], list[ReputationAttestation]]:
    # 1. Admissibility per attestation, pinpointed not raised. A tampered or forged
    #    attestation is inadmissible; a self-attestation, a revoked one, or a stale one
    #    is a valid artifact that is excluded (not counted) with a pinpointed reason.
    verdicts: list[AttestationVerdict] = []
    admissible: list[ReputationAttestation] = []
    for att in items:
        check = att.verify(verifier, require=[])
        admissible_flag = True
        revoked = stale = False
        reason: str | None = None
        if not check.hash_ok:
            admissible_flag, reason = False, "tampered: attestation hash does not recompute"
        elif not check.evidence_sound:
            admissible_flag = False
            reason = "tampered: attested reputation does not re-derive from the evidence"
        elif verifier is not None and att.signatures and not check.signatures_ok:
            admissible_flag, reason = False, "forged: the issuer signature does not verify"
        elif (match := revoked_keys.get((att.issuer, att.content_hash))) is not None:
            revoked = True
            detail = f" ({match.reason})" if match.reason else ""
            reason = (
                "revoked: superseded by its issuer" + detail
                if match.is_supersession
                else "revoked: withdrawn by its issuer" + detail
            )
        elif clock is not None and att.is_stale(clock):
            stale = True
            reason = f"stale: past its {att.horizon_days:g}-day validity window as of the clock"
        elif not allow_self and att.is_self_attestation:
            reason = "self-attestation: an issuer cannot vouch for itself"
        verdicts.append(
            AttestationVerdict(
                attestation_id=att.id,
                issuer=att.issuer,
                subject=att.subject,
                evidence=att.evidence,
                reputation=round(float(att.reputation), 9),
                admissible=admissible_flag,
                counted=False,
                revoked=revoked,
                stale=stale,
                reason=reason,
            )
        )
        if admissible_flag and reason is None:
            admissible.append(att)
    return verdicts, admissible


def _dedup_best(
    admissible: list[ReputationAttestation],
) -> dict[tuple[str, str], ReputationAttestation]:
    # 2. Per (subject, issuer) dedup: keep each issuer's largest attestation, so an
    #    issuer cannot inflate a subject by stacking several attestations.
    best: dict[tuple[str, str], ReputationAttestation] = {}
    for att in admissible:
        key = (att.subject, att.issuer)
        current = best.get(key)
        if current is None or _supersedes(att, current):
            best[key] = att
    return best


def _mark_counted(
    verdicts: list[AttestationVerdict],
    counted_ids: set[str],
    trust_model: Any | None,
) -> None:
    # 3. Mark the counted verdicts and pinpoint the superseded ones.
    for verdict in verdicts:
        if not verdict.admissible or verdict.reason is not None:
            continue
        if verdict.attestation_id in counted_ids:
            verdict.counted = True
            verdict.trust = round(_trust_multiplier(trust_model, verdict.issuer), 9)
        else:
            verdict.reason = "superseded: a larger attestation from this issuer was counted"


def _pool_evidence(
    best: dict[tuple[str, str], ReputationAttestation],
    cfg: AttestationConfig,
    clock: datetime | None,
    trust_model: Any | None,
) -> dict[str, dict[str, Any]]:
    # 4. Pool the evidence per subject under the prior: decay by age (when an as-of
    #    clock is set), scale by the importer's trust in the issuer, then cap any one
    #    issuer's mass — each step scales successes and failures together, so it changes
    #    how much an issuer's evidence *pulls*, never the reputation it attests.
    pooled: dict[str, dict[str, Any]] = {}
    for (subj, issuer), att in best.items():
        succ, fail = float(att.successes), float(att.failures)
        if clock is not None:
            succ, fail = cfg.decayed(succ, fail, att.age_days(clock))
        tmul = _trust_multiplier(trust_model, issuer)
        if tmul != 1.0:
            succ, fail = succ * tmul, fail * tmul
        succ, fail = cfg.capped(succ, fail)
        bucket = pooled.setdefault(
            subj, {"successes": 0.0, "failures": 0.0, "issuers": [], "trust": {}}
        )
        bucket["successes"] += succ
        bucket["failures"] += fail
        bucket["issuers"].append(issuer)
        if trust_model is not None:
            bucket["trust"][issuer] = round(tmul, 9)
    return pooled


def _build_standings(
    pooled: dict[str, dict[str, Any]],
    cfg: AttestationConfig,
) -> dict[str, SubjectStanding]:
    standings: dict[str, SubjectStanding] = {}
    for subj, bucket in pooled.items():
        successes = round(float(bucket["successes"]), 9)
        failures = round(float(bucket["failures"]), 9)
        reputation = round(cfg.reputation_of(successes, failures), 9)
        standings[subj] = SubjectStanding(
            subject=subj,
            successes=successes,
            failures=failures,
            reputation=reputation,
            weight=cfg.weight_of(reputation),
            issuers=sorted(bucket["issuers"]),
            attestations=len(bucket["issuers"]),
            issuer_trust=dict(bucket["trust"]),
        )
    return standings
