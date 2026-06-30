"""Cross-org reputation-gated admission & progressive exposure.

Reputation is now portable, current, discoverable, and trust-weighted — but it is
still only ever *consulted* as a soft weight on a negotiation; nothing **acts** on a
too-thin or too-low standing to bound how much a new counterparty is trusted with up
front. A brand-new or low-trust counterparty is admitted to a contract on the same
terms as a long-trusted one, with the regression caught only after the fact. This
module turns the weighted standing into a **graduated admission posture** — bounding a
counterparty's exposure to what its earned trust justifies, and ramping it as trust
accrues — so onboarding an unknown org is safe by construction, not by hope.

* **Reputation-gated terms.** An :class:`AdmissionPolicy` reads a counterparty's
  standing from whatever weights the rest of the fabric — a
  :class:`~vincio.settlement.attestation.PortableReputation` imported from other orgs'
  attestations, or a local :class:`~vincio.optimize.reputation.ReputationLedger` — and
  maps it to a bounded :class:`AdmissionDecision`: a maximum contract value (the
  exposure ceiling), a required collateral / escrow fraction, and an SLA-strictness
  factor. A thin or low-trust standing is admitted on *conservative* terms — discounted
  exposure, never a hard gate, never singled out — so it can still close a deal and earn
  its way up, exactly as a discounted negotiation weight lets it.
* **Progressive ramp.** Exposure is the product of two bounded signals: *how good* the
  standing is (its posterior-mean reputation) and *how much corroborated, settled
  history* stands behind it. A brand-new counterparty starts conservative; as it accrues
  settled deliveries its ceiling **ramps** deterministically toward parity, and a
  regression walks it back — so trust earned over real deliveries unlocks exposure the
  way a credit line builds, bounded and reversible at every step.
* **Auditable & offline.** A decision binds the standing it read (the weight,
  reputation, evidence, and corroborating issuers) and the terms it set onto a content
  hash; :meth:`AdmissionDecision.verify` recomputes the terms from that bound standing
  under the policy and checks they match — so a counterparty's exposure is a mechanical,
  reconstructable number, verifiable from the bytes alone, never a hosted underwriting
  service. :meth:`~vincio.core.app.ContextApp.admit` records each decision on the
  hash-chained audit log.

The decision folds into the *existing* negotiation / contracting path without a new code
path through it: :meth:`AdmissionDecision.bound_position` clamps a buyer's negotiating
position to the ceiling (so the bargain converges within the admitted exposure) and
:meth:`AdmissionDecision.apply_to_terms` caps and stamps a contract's terms directly.
Everything is dependency-free, deterministic, and offline — a policy lens over the
standing the fabric already earns, never a hosted underwriting service.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.diagnostics import note_suppressed
from ..core.errors import SettlementError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow

__all__ = [
    "AdmissionConfig",
    "AdmissionDecision",
    "AdmissionVerification",
    "AdmissionPolicy",
    "Standing",
    "admit",
]

# The audit action an admission decision is recorded under.
ADMISSION_ACTION = "reputation_admission"

_TOLERANCE = 1e-9


class AdmissionConfig(BaseModel):
    """How a counterparty's standing maps to a bounded exposure posture.

    Exposure is the product of two independent, bounded signals, each ``∈ [0, 1]``:

    * **standing quality** — the counterparty's posterior-mean reputation, the same
      ``∈ (0, 1)`` standing the negotiation weight is a band-mapping of, so portable and
      local standing gate admission on one scale; and
    * **corroborated history** — how much settled, corroborated evidence stands behind
      that reputation, ramped from ``ramp_floor`` (a brand-new counterparty) to ``1`` at
      ``full_trust_evidence`` settled deliveries.

    Their product is the *trust signal*; the **exposure fraction** of parity is
    ``floor_fraction + (1 − floor_fraction) · trust_signal ∈ [floor_fraction, 1]``. The
    floor keeps a thin or low-trust standing admitted on conservative terms rather than
    refused — discounted exposure, never zeroed, never singled out — and parity (``1``)
    is the ceiling a fully-trusted counterparty reaches, never exceeds.

    * ``parity_exposure_usd`` is the maximum contract value a counterparty at parity is
      admitted to — the exposure ceiling the ramp climbs toward.
    * ``floor_fraction`` is the minimal fraction of parity even a thin or low-trust
      standing earns, so admission is graduated, never a hard gate.
    * ``full_trust_evidence`` is the corroborated, settled evidence at which the ramp
      reaches parity; ``ramp_floor`` is the fraction of the ramp a zero-history
      counterparty starts at, so a brand-new org is admitted conservatively before any
      delivery rather than frozen out.
    * ``max_escrow_fraction`` is the collateral / escrow fraction demanded at minimum
      exposure, falling linearly to ``0`` at parity — a low-trust deal is backed by
      collateral, a fully-trusted one needs none.
    * ``min_sla_factor`` tightens the SLA at minimum exposure (a low-trust seller commits
      to a stricter deadline), relaxing linearly to ``1`` (no tightening) at parity.
    """

    parity_exposure_usd: float = 1000.0
    floor_fraction: float = 0.1
    full_trust_evidence: float = 10.0
    ramp_floor: float = 0.2
    max_escrow_fraction: float = 0.5
    min_sla_factor: float = 0.5

    def validate_coherent(self) -> AdmissionConfig:
        """Raise :class:`SettlementError` unless the configuration is coherent."""
        if self.parity_exposure_usd <= 0.0:
            raise SettlementError(
                f"parity_exposure_usd must be positive; got {self.parity_exposure_usd}",
                details={"parity_exposure_usd": self.parity_exposure_usd},
            )
        if not 0.0 <= self.floor_fraction <= 1.0:
            raise SettlementError(
                f"floor_fraction must be in [0, 1]; got {self.floor_fraction}",
                details={"floor_fraction": self.floor_fraction},
            )
        if self.full_trust_evidence <= 0.0:
            raise SettlementError(
                f"full_trust_evidence must be positive; got {self.full_trust_evidence}",
                details={"full_trust_evidence": self.full_trust_evidence},
            )
        if not 0.0 <= self.ramp_floor <= 1.0:
            raise SettlementError(
                f"ramp_floor must be in [0, 1]; got {self.ramp_floor}",
                details={"ramp_floor": self.ramp_floor},
            )
        if not 0.0 <= self.max_escrow_fraction <= 1.0:
            raise SettlementError(
                f"max_escrow_fraction must be in [0, 1]; got {self.max_escrow_fraction}",
                details={"max_escrow_fraction": self.max_escrow_fraction},
            )
        if not 0.0 < self.min_sla_factor <= 1.0:
            raise SettlementError(
                f"min_sla_factor must be in (0, 1]; got {self.min_sla_factor}",
                details={"min_sla_factor": self.min_sla_factor},
            )
        return self

    # -- the graduated-exposure map ----------------------------------------

    def ramp_progress(self, evidence: float) -> float:
        """The history ramp ``∈ [ramp_floor, 1]`` for ``evidence`` settled deliveries.

        ``ramp_floor`` at zero history, climbing linearly to ``1`` once corroborated
        evidence reaches ``full_trust_evidence`` — and never past it, so accruing
        history beyond the threshold cannot push exposure past parity.
        """
        reach = min(1.0, max(0.0, evidence) / self.full_trust_evidence)
        return self.ramp_floor + (1.0 - self.ramp_floor) * reach

    def exposure_fraction(self, reputation: float, evidence: float) -> float:
        """The fraction of parity ``∈ [floor_fraction, 1]`` a standing is admitted at.

        The product of standing quality (clamped ``reputation``) and the history ramp,
        lifted off ``floor_fraction`` so a thin or low-trust standing is discounted, never
        zeroed.
        """
        quality = min(1.0, max(0.0, reputation))
        signal = quality * self.ramp_progress(evidence)
        return round(self.floor_fraction + (1.0 - self.floor_fraction) * signal, 9)

    def terms_for(self, reputation: float, evidence: float) -> dict[str, float]:
        """The bounded exposure terms a standing earns — the decision's mechanical core.

        Returns the ``exposure_fraction`` and the three terms it sets: the
        ``max_contract_value_usd`` ceiling, the ``escrow_fraction`` collateral, and the
        ``sla_factor`` strictness — each a deterministic function of the bound standing,
        so :meth:`AdmissionDecision.verify` recomputes them from the bytes alone.
        """
        fraction = self.exposure_fraction(reputation, evidence)
        return {
            "exposure_fraction": fraction,
            "max_contract_value_usd": round(self.parity_exposure_usd * fraction, 6),
            "escrow_fraction": round(self.max_escrow_fraction * (1.0 - fraction), 9),
            "sla_factor": round(self.min_sla_factor + (1.0 - self.min_sla_factor) * fraction, 9),
        }

    def canonical(self) -> dict[str, float]:
        """A stable, hashable projection of the policy the decision binds."""
        return {
            "parity_exposure_usd": round(float(self.parity_exposure_usd), 9),
            "floor_fraction": round(float(self.floor_fraction), 9),
            "full_trust_evidence": round(float(self.full_trust_evidence), 9),
            "ramp_floor": round(float(self.ramp_floor), 9),
            "max_escrow_fraction": round(float(self.max_escrow_fraction), 9),
            "min_sla_factor": round(float(self.min_sla_factor), 9),
        }


class Standing(BaseModel):
    """The standing an :class:`AdmissionPolicy` read for a counterparty.

    The inputs the exposure terms are a function of, bound onto the decision so it
    recomputes offline: the negotiation ``weight`` and posterior-mean ``reputation`` the
    fabric weights by, the ``evidence`` (corroborated, settled deliveries) behind them,
    and the ``issuers`` that corroborated it (empty for a local-ledger standing, which is
    first-hand rather than attested).
    """

    weight: float = 1.0
    reputation: float = 0.0
    evidence: float = 0.0
    issuers: list[str] = Field(default_factory=list)

    def canonical(self) -> dict[str, Any]:
        """A stable, hashable projection of the standing the decision binds."""
        return {
            "weight": round(float(self.weight), 9),
            "reputation": round(float(self.reputation), 9),
            "evidence": round(float(self.evidence), 9),
            "issuers": sorted(self.issuers),
        }


class AdmissionVerification(BaseModel):
    """The (non-raising) outcome of verifying an admission decision offline.

    A decision is **valid** when its content hash recomputes (``hash_ok``) and the terms
    it set re-derive from the standing it bound under its policy (``terms_sound``) — so a
    tampered ceiling, escrow, or SLA factor is caught from the bytes alone, without the
    live reputation source.
    """

    valid: bool
    hash_ok: bool
    terms_sound: bool
    reason: str | None = None


class AdmissionDecision(BaseModel):
    """A bounded, offline-verifiable exposure posture for one counterparty.

    Produced by :class:`AdmissionPolicy` (or
    :meth:`~vincio.core.app.ContextApp.admit`) from the counterparty's standing. It binds
    the :class:`Standing` it read and the terms it set — the ``max_contract_value_usd``
    exposure ceiling, the ``escrow_fraction`` collateral, and the ``sla_factor``
    strictness — onto a content hash, so the exposure is a mechanical number anyone
    recomputes: :meth:`verify` re-derives the terms from the bound standing under the
    policy and checks they match.

    It folds into the existing path without a new code path through it:
    :meth:`bound_position` clamps a buyer's
    :class:`~vincio.negotiation.engine.NegotiationPosition` to the ceiling so the bargain
    converges within the admitted exposure, and :meth:`apply_to_terms` caps and stamps a
    :class:`~vincio.negotiation.contract.ContractTerms` directly.
    """

    id: str = Field(default_factory=lambda: new_id("admission"))
    subject: str
    standing: Standing
    config: AdmissionConfig
    exposure_fraction: float = 0.0
    max_contract_value_usd: float = 0.0
    escrow_fraction: float = 0.0
    sla_factor: float = 1.0
    decided_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    audit_id: str | None = None

    # -- derived reads ------------------------------------------------------

    @property
    def at_parity(self) -> bool:
        """Whether the counterparty is admitted at the full parity ceiling."""
        return self.exposure_fraction >= 1.0 - _TOLERANCE

    @property
    def issuers(self) -> list[str]:
        """The issuers that corroborated the standing this decision read."""
        return list(self.standing.issuers)

    # -- hashing ------------------------------------------------------------

    def decision_facts(self) -> dict[str, Any]:
        """The facts the content hash binds — the standing read and the terms set.

        Excludes the id and timestamp (local metadata, not the decision), so the same
        standing under the same policy hashes identically wherever it is recomputed.
        """
        return {
            "subject": self.subject,
            "standing": self.standing.canonical(),
            "config": self.config.canonical(),
            "exposure_fraction": round(float(self.exposure_fraction), 9),
            "max_contract_value_usd": round(float(self.max_contract_value_usd), 6),
            "escrow_fraction": round(float(self.escrow_fraction), 9),
            "sla_factor": round(float(self.sla_factor), 9),
        }

    def compute_hash(self) -> str:
        """The content hash binding the standing read and the terms set."""
        return stable_hash(self.decision_facts(), length=32)

    def seal(self) -> AdmissionDecision:
        """Stamp the content hash from the current fields (idempotent)."""
        self.content_hash = self.compute_hash()
        return self

    # -- verification -------------------------------------------------------

    def _terms_sound(self) -> bool:
        """The bound terms re-derive from the bound standing under the bound policy."""
        try:
            recomputed = self.config.terms_for(self.standing.reputation, self.standing.evidence)
        except SettlementError:
            return False
        return (
            abs(recomputed["exposure_fraction"] - self.exposure_fraction) <= _TOLERANCE
            and abs(recomputed["max_contract_value_usd"] - self.max_contract_value_usd)
            <= _TOLERANCE
            and abs(recomputed["escrow_fraction"] - self.escrow_fraction) <= _TOLERANCE
            and abs(recomputed["sla_factor"] - self.sla_factor) <= _TOLERANCE
        )

    def verify(self) -> AdmissionVerification:
        """Check the decision recomputes from the bytes alone — terms and hash.

        Recomputes the content hash and re-derives the exposure terms from the bound
        standing under the bound policy. A tampered ceiling, escrow, or SLA factor is
        caught (the terms no longer re-derive) even when the hash was recomputed to
        match — so the exposure is a reconstructable number, not a trusted assertion.
        """
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        terms_sound = self._terms_sound()
        valid = hash_ok and terms_sound
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "decision is not sealed (no content hash)"
            elif not hash_ok:
                reason = "content hash does not match the decision facts"
            else:
                reason = "exposure terms do not re-derive from the bound standing"
        return AdmissionVerification(
            valid=valid, hash_ok=hash_ok, terms_sound=terms_sound, reason=reason
        )

    def require_valid(self) -> AdmissionDecision:
        """Verify and raise :class:`SettlementError` if the decision is not valid."""
        result = self.verify()
        if not result.valid:
            raise SettlementError(
                f"admission decision {self.id} failed verification: {result.reason}",
                details={"decision_id": self.id, "reason": result.reason},
            )
        return self

    # -- folding into negotiation / contracting -----------------------------

    def bound_position(self, position: Any) -> Any:
        """Clamp a buyer's negotiating position to the admitted exposure.

        Returns a copy of ``position`` (a
        :class:`~vincio.negotiation.engine.NegotiationPosition`) whose ``price_usd``
        reservation is clamped down to the exposure ceiling and whose ``sla_seconds``
        reservation is tightened by ``sla_factor`` — so the bargain can only converge
        within the admitted exposure, under the *existing* negotiation path. A position
        already inside the ceiling is returned unchanged in effect; the ideal is clamped
        alongside the reservation only when it would otherwise cross it, keeping the
        position coherent.
        """
        bounded = position.model_copy(deep=True)
        for issue in getattr(bounded, "issues", []):
            name = getattr(issue, "name", "")
            if name == "price_usd" and self.max_contract_value_usd > 0.0:
                if issue.reserve > self.max_contract_value_usd:
                    issue.reserve = self.max_contract_value_usd
                if issue.ideal > self.max_contract_value_usd:
                    issue.ideal = self.max_contract_value_usd
            elif name == "sla_seconds" and self.sla_factor < 1.0 and issue.reserve > 0.0:
                issue.reserve = round(issue.reserve * self.sla_factor, 9)
                if issue.ideal > issue.reserve:
                    issue.ideal = issue.reserve
        return bounded

    def apply_to_terms(self, terms: Any) -> Any:
        """Cap and stamp contract terms to the admitted exposure.

        Returns a copy of ``terms`` (a
        :class:`~vincio.negotiation.contract.ContractTerms`) with ``price_usd`` capped at
        the exposure ceiling and ``sla_seconds`` tightened by ``sla_factor``, and the
        admission posture (the ceiling, escrow fraction, and decision id) stamped into the
        terms' ``metadata`` — which is excluded from the contract's canonical hash, so a
        contract minted from the capped terms stays offline-verifiable unchanged while
        carrying the collateral the deal must post.
        """
        capped = terms.model_copy(deep=True)
        if self.max_contract_value_usd > 0.0 and (
            capped.price_usd <= 0.0 or capped.price_usd > self.max_contract_value_usd
        ):
            capped.price_usd = self.max_contract_value_usd
        if self.sla_factor < 1.0 and capped.sla_seconds > 0.0:
            capped.sla_seconds = round(capped.sla_seconds * self.sla_factor, 9)
        metadata = dict(getattr(capped, "metadata", {}) or {})
        metadata["admission"] = {
            "decision_id": self.id,
            "subject": self.subject,
            "max_contract_value_usd": round(float(self.max_contract_value_usd), 6),
            "escrow_fraction": round(float(self.escrow_fraction), 9),
            "sla_factor": round(float(self.sla_factor), 9),
            "exposure_fraction": round(float(self.exposure_fraction), 9),
        }
        capped.metadata = metadata
        return capped

    # -- serialization & reporting -----------------------------------------

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe record of the decision for the audit chain."""
        return to_jsonable(
            {
                "decision_id": self.id,
                "subject": self.subject,
                "weight": round(float(self.standing.weight), 9),
                "reputation": round(float(self.standing.reputation), 9),
                "evidence": round(float(self.standing.evidence), 9),
                "issuers": sorted(self.standing.issuers),
                "exposure_fraction": round(float(self.exposure_fraction), 9),
                "max_contract_value_usd": round(float(self.max_contract_value_usd), 6),
                "escrow_fraction": round(float(self.escrow_fraction), 9),
                "sla_factor": round(float(self.sla_factor), 9),
                "at_parity": self.at_parity,
                "content_hash": self.content_hash,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        """A JSON-safe projection for exchange or persistence."""
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> AdmissionDecision:
        return cls.model_validate(data)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print the admitted exposure and the standing behind it."""
        corroborated = (
            f" from {len(self.standing.issuers)} issuer(s)" if self.standing.issuers else ""
        )
        print(
            f"Admission of {self.subject}: ${self.max_contract_value_usd:,.2f} ceiling "
            f"({self.exposure_fraction:.0%} of parity), {self.escrow_fraction:.0%} escrow, "
            f"SLA ×{self.sla_factor:.2f} — reputation {self.standing.reputation:.3f} over "
            f"{self.standing.evidence:g} settled{corroborated}"
            + (" — at parity" if self.at_parity else "")
        )


class AdmissionPolicy:
    """A graduated-exposure policy over the standing the fabric already earns.

    Wraps an :class:`AdmissionConfig` and resolves a counterparty's standing — from a
    :class:`~vincio.settlement.attestation.PortableReputation`, a local
    :class:`~vincio.optimize.reputation.ReputationLedger`, or anything exposing
    ``weight(member_id)`` / ``reputation(member_id)`` — into a bounded
    :class:`AdmissionDecision`. It is the same standing the negotiation path weights by,
    so admission and negotiation read one reputation, not two.
    """

    def __init__(self, config: AdmissionConfig | None = None) -> None:
        self.config = (config or AdmissionConfig()).validate_coherent()

    def read_standing(self, source: Any, subject: str) -> Standing:
        """Read ``subject``'s standing from a reputation ``source``.

        Reads the negotiation ``weight`` and posterior-mean ``reputation`` the source
        exposes, and the corroborated ``evidence`` / ``issuers`` behind them — from a
        :class:`~vincio.settlement.attestation.SubjectStanding` (a portable prior) or a
        :class:`~vincio.optimize.reputation.MemberReputation` snapshot (a local ledger).

        Local first-hand evidence wins, exactly as
        :meth:`~vincio.settlement.attestation.PortableReputation.weight` resolves it: when
        the source is a portable prior whose ``base`` ledger already has earned evidence
        for the subject, the standing is read from that ledger — so a regression the
        importer lived through walks exposure back even when other orgs still attest a high
        standing, and admission and negotiation read one reputation, not two. A missing
        accessor or an unknown subject falls back to the benefit-of-the-doubt prior rather
        than refusing — never zeroed, never singled out.
        """
        resolved = _resolve_source(source, subject)
        weight = _read_float(resolved, "weight", subject, default=1.0)
        reputation = _read_float(resolved, "reputation", subject, default=0.0)
        evidence, issuers = _read_evidence(resolved, subject)
        return Standing(weight=weight, reputation=reputation, evidence=evidence, issuers=issuers)

    def decide(self, subject: str, standing: Standing) -> AdmissionDecision:
        """Map a resolved ``standing`` to a sealed :class:`AdmissionDecision`."""
        terms = self.config.terms_for(standing.reputation, standing.evidence)
        return AdmissionDecision(
            subject=subject,
            standing=standing,
            config=self.config,
            exposure_fraction=terms["exposure_fraction"],
            max_contract_value_usd=terms["max_contract_value_usd"],
            escrow_fraction=terms["escrow_fraction"],
            sla_factor=terms["sla_factor"],
        ).seal()

    def admit(
        self,
        subject: str,
        *,
        reputation: Any | None = None,
        ledger: Any | None = None,
        standing: Standing | None = None,
    ) -> AdmissionDecision:
        """Decide ``subject``'s admitted exposure from its standing.

        Reads the standing from ``reputation`` (a portable prior), else ``ledger`` (a
        local reputation ledger), else an explicit ``standing`` — and maps it to a bounded
        :class:`AdmissionDecision`. With no source at all the subject is admitted on the
        benefit-of-the-doubt prior (a brand-new counterparty), conservatively but never
        refused.
        """
        if standing is None:
            source = reputation if reputation is not None else ledger
            if source is not None:
                standing = self.read_standing(source, subject)
            else:
                standing = _prior_standing()
        return self.decide(subject, standing)


def admit(
    subject: str,
    *,
    reputation: Any | None = None,
    ledger: Any | None = None,
    standing: Standing | None = None,
    config: AdmissionConfig | None = None,
) -> AdmissionDecision:
    """Decide a counterparty's admitted exposure from its standing.

    The module-level convenience over :class:`AdmissionPolicy`: reads ``subject``'s
    standing from ``reputation`` (a
    :class:`~vincio.settlement.attestation.PortableReputation`), else ``ledger`` (a local
    :class:`~vincio.optimize.reputation.ReputationLedger`), else an explicit
    ``standing``, and maps it to a bounded, offline-verifiable
    :class:`AdmissionDecision` under ``config``::

        decision = admit("vendor", reputation=prior)
        decision.verify().valid  # offline-verifiable
    """
    return AdmissionPolicy(config).admit(
        subject, reputation=reputation, ledger=ledger, standing=standing
    )


# -- standing readers ------------------------------------------------------


def _prior_standing() -> Standing:
    """The benefit-of-the-doubt standing a never-seen counterparty starts at.

    No source at all (a brand-new counterparty): the same prior mean a local ledger or a
    portable prior extends a member it has never seen, mapped to the matching band weight,
    with no evidence and no corroborating issuers — admitted conservatively, never refused.
    """
    from .attestation import AttestationConfig

    config = AttestationConfig()
    reputation = round(config.reputation_of(0.0, 0.0), 9)
    return Standing(weight=config.weight_of(reputation), reputation=reputation)


def _resolve_source(source: Any, subject: str) -> Any:
    """Pick the source admission reads, giving local first-hand evidence precedence.

    Mirrors :meth:`~vincio.settlement.attestation.PortableReputation.weight`: when
    ``source`` is a portable prior carrying a ``base`` ledger that already has earned
    evidence for ``subject``, that ledger is read (the importer trusts what it has lived
    through over what others attest); otherwise the source itself is read.
    """
    base = getattr(source, "base", None)
    if base is None:
        return source
    from .attestation import _has_local_evidence

    try:
        return base if _has_local_evidence(base, subject) else source
    except Exception:
        note_suppressed("settlement.admission.local_evidence_probe")
        return source


def _read_float(source: Any, accessor: str, subject: str, *, default: float) -> float:
    """Call ``source.<accessor>(subject)`` defensively, falling back to ``default``."""
    fn = getattr(source, accessor, None)
    if not callable(fn):
        return default
    try:
        return float(fn(subject))
    except Exception:
        note_suppressed("settlement.admission.source_value")
        return default


def _read_evidence(source: Any, subject: str) -> tuple[float, list[str]]:
    """Read the corroborated, settled evidence and issuers behind a standing.

    From a :class:`~vincio.settlement.attestation.PortableReputation` the evidence is the
    pooled (issuer-capped) attested mass and the issuers are who corroborated it; from a
    local :class:`~vincio.optimize.reputation.ReputationLedger` it is the member's
    first-hand success/failure mass with no attesting issuers. An unknown subject or an
    absent accessor reads as zero evidence — a brand-new counterparty.
    """
    standing_fn = getattr(source, "standing", None)
    if callable(standing_fn):
        try:
            subject_standing = standing_fn(subject)
        except Exception:
            note_suppressed("settlement.admission.standing")
            subject_standing = None
        if subject_standing is not None:
            evidence = float(getattr(subject_standing, "evidence", 0.0) or 0.0)
            issuers = list(getattr(subject_standing, "issuers", []) or [])
            return evidence, sorted(issuers)
    snapshot_fn = getattr(source, "snapshot", None)
    if callable(snapshot_fn):
        try:
            snapshot = snapshot_fn(subject)
        except Exception:
            note_suppressed("settlement.admission.snapshot")
            snapshot = None
        if snapshot is not None:
            successes = float(getattr(snapshot, "successes", 0.0) or 0.0)
            failures = float(getattr(snapshot, "failures", 0.0) or 0.0)
            return round(successes + failures, 9), []
    return 0.0, []
