"""Run-time capability binding for cross-org sagas — discovery, not wiring.

A statically-wired :class:`~vincio.choreography.saga.Saga` names the **org** that
runs each step up front (``participant="warehouse"``). This module adds the next
reach: a step that declares the **capability** it needs and lets the engine
**resolve the participant at dispatch time** from the governed
:class:`~vincio.registry.AgentDirectory` — so a choreography binds the
best-available counterparty for each step rather than a hard-coded org id. It is
discovery, never a hosted matching service: the binding runs in your process,
under the same allow-list, contract, and per-org audit a statically-wired step
runs under. Discovery changes *who* runs a step, never *how* it is governed.

Three guarantees, all dependency-free and deterministic:

* **Governed.** A candidate is considered only after the directory's
  :class:`~vincio.security.access.AllowListGate` allows it — every
  :meth:`~vincio.registry.AgentDirectory.try_resolve` records an ``agent_resolve``
  decision on the audit chain, so an unlisted candidate is never bound and the
  binding itself lands on the coordinator's chain as a ``choreography_bind`` entry.
* **Ranked by capability + reputation + settlement.** Among the allowed,
  reachable candidates that advertise the capability, the binding prefers the one
  whose :class:`~vincio.optimize.reputation.ReputationLedger` standing and prior
  :class:`~vincio.settlement.SettlementBook` record best fit the step's contract
  terms — a reliable, on-budget counterparty wins a close race, a regressing one
  is discounted without being singled out. Ties break deterministically by org id,
  so a binding is reproducible.
* **Same downstream governance.** The resolved org is recorded on the saga journal
  and dispatched, contract-enforced, compensated, and settled exactly as a
  statically-wired one — the binder only chooses; the engine governs.

The binder is pure: :meth:`CapabilityBinder.bind` reads the directory, reputation,
and settlement state and returns a :class:`StepBinding` (the chosen org plus the
full ranked candidate list, for audit) without mutating anything. The engine
records the decision and dispatches.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import ChoreographyError

__all__ = [
    "BIND_ACTION",
    "BindingWeights",
    "BindingCandidate",
    "StepBinding",
    "CapabilityBinder",
]

# The audit action the coordinator records a run-time binding decision under, on
# its own hash-chained chain — beside the ``choreography_step`` handoff.
BIND_ACTION = "choreography_bind"


class BindingWeights(BaseModel):
    """How a candidate's signals combine into one ranking score.

    The score is the weighted mean of three signals, each normalized to
    ``[0, 1]``: the candidate's **reputation** weight (its no-regression /
    contract-fulfilment track record), its **settlement reliability** (the share
    of prior settlements with this coordinator that were honoured rather than
    breached), and its **contract fit** (how well its prior delivered cost sits
    under the step contract's agreed price). Reputation leads by default; settlement
    signals refine a close race. A candidate with no history is neither trusted
    blindly nor frozen out — it scores ``unknown_settlement_score`` on the
    history-derived signals (the benefit of the doubt a newcomer is extended,
    mirroring the reputation ledger's prior).
    """

    reputation: float = 1.0
    settlement: float = 0.5
    contract_fit: float = 0.5
    unknown_settlement_score: float = 0.5

    def validate_coherent(self) -> BindingWeights:
        """Raise :class:`ChoreographyError` unless the weights are usable."""
        if self.reputation < 0 or self.settlement < 0 or self.contract_fit < 0:
            raise ChoreographyError("binding weights must be non-negative")
        if self.reputation + self.settlement + self.contract_fit <= 0:
            raise ChoreographyError("binding weights must sum to a positive value")
        if not 0.0 <= self.unknown_settlement_score <= 1.0:
            raise ChoreographyError("unknown_settlement_score must be in [0, 1]")
        return self

    def score(self, reputation: float, settlement: float, contract_fit: float) -> float:
        """The weighted-mean ranking score for one candidate's signals."""
        total = self.reputation + self.settlement + self.contract_fit
        weighted = (
            self.reputation * reputation
            + self.settlement * settlement
            + self.contract_fit * contract_fit
        )
        return round(weighted / total, 9) if total > 0 else 0.0


class BindingCandidate(BaseModel):
    """One ranked candidate for a capability binding, the decision's evidence.

    Every directory record that advertises the capability becomes a candidate,
    whether or not it is finally eligible, so the :class:`StepBinding` carries a
    complete, auditable picture of who was considered and why each was kept or
    rejected. ``eligible`` is true only when the candidate is both allow-listed
    (``allowed``) and reachable (has a participant binding), the precondition for
    being bound; a rejected candidate scores ``0.0`` and carries a
    ``rejected_reason``.
    """

    org: str
    protocol: str = "a2a"
    allowed: bool = False
    reachable: bool = False
    reputation: float = 1.0
    settlement_reliability: float = 0.0
    settlements: int = 0
    contract_fit: float = 1.0
    score: float = 0.0
    rejected_reason: str | None = None

    @property
    def eligible(self) -> bool:
        """Whether this candidate could be bound (allow-listed and reachable)."""
        return self.allowed and self.reachable


class StepBinding(BaseModel):
    """The resolved run-time binding for one capability step.

    ``org`` is the chosen counterparty; ``score`` / ``reputation`` are its winning
    figures; ``candidates`` is the full ranked list (eligible and rejected) so the
    decision is reconstructable from the record alone. ``considered`` /
    ``eligible`` count the candidates the directory surfaced and how many cleared
    governance, so a thin or empty allowed set is visible rather than silent.
    """

    step: str
    capability: str
    org: str
    score: float = 0.0
    reputation: float = 1.0
    candidates: list[BindingCandidate] = Field(default_factory=list)
    considered: int = 0
    eligible: int = 0
    reason: str = ""

    def audit_details(self) -> dict[str, Any]:
        """A compact, JSON-safe projection of the decision for the audit chain."""
        return {
            "step": self.step,
            "capability": self.capability,
            "bound_org": self.org,
            "score": self.score,
            "reputation": self.reputation,
            "considered": self.considered,
            "eligible": self.eligible,
            "candidates": [
                {
                    "org": c.org,
                    "score": c.score,
                    "eligible": c.eligible,
                    "rejected_reason": c.rejected_reason,
                }
                for c in self.candidates
            ],
        }


class CapabilityBinder:
    """Resolves a capability-declaring saga step to a participant at dispatch time.

    Wraps a governed :class:`~vincio.registry.AgentDirectory` and, optionally, a
    :class:`~vincio.optimize.reputation.ReputationLedger` and a
    :class:`~vincio.settlement.SettlementBook`. :meth:`bind` finds the directory
    records advertising the step's capability, governs each through the directory's
    allow-list (audited), keeps the allowed and reachable ones, ranks them by
    reputation + settlement reliability + contract fit, and returns the highest
    scorer — deterministic, with ties broken by org id. It mutates nothing; the
    engine records the returned :class:`StepBinding` and dispatches.

    ``available`` (passed to :meth:`bind` by the engine) is the set of org ids that
    actually have a participant binding, so a candidate the directory advertises but
    the coordinator cannot reach is rejected as unreachable rather than bound to a
    dead end.
    """

    def __init__(
        self,
        directory: Any,
        *,
        reputation: Any | None = None,
        settlement_book: Any | None = None,
        weights: BindingWeights | None = None,
        principal: Any | None = None,
    ) -> None:
        if directory is None:
            raise ChoreographyError(
                "a CapabilityBinder requires an AgentDirectory to resolve capabilities"
            )
        self.directory = directory
        self.reputation = reputation
        self.settlement_book = settlement_book
        self.weights = (weights or BindingWeights()).validate_coherent()
        self.principal = principal

    # -- public API ---------------------------------------------------------

    def bind(self, step: Any, *, available: set[str] | None = None) -> StepBinding:
        """Resolve a capability step to a :class:`StepBinding`, or raise.

        Raises :class:`~vincio.core.errors.ChoreographyError` when the step does
        not declare a capability, or when no allowed, reachable candidate advertises
        it — the discovery analogue of a statically-wired step naming an
        unregistered participant.
        """
        capability = getattr(step, "capability", "") or ""
        if not capability:
            raise ChoreographyError(
                f"step {getattr(step, 'name', '?')!r} declares no capability to bind"
            )
        contract = getattr(step, "contract", None)
        candidates = self.rank(capability, contract=contract, available=available)
        eligible = [c for c in candidates if c.eligible]
        if not eligible:
            considered = len(candidates)
            raise ChoreographyError(
                f"no allowed, reachable participant advertises capability "
                f"{capability!r} (considered {considered})",
                details={
                    "capability": capability,
                    "step": getattr(step, "name", None),
                    "considered": considered,
                    "rejected": [
                        {"org": c.org, "reason": c.rejected_reason} for c in candidates
                    ],
                },
            )
        best = eligible[0]
        return StepBinding(
            step=getattr(step, "name", ""),
            capability=capability,
            org=best.org,
            score=best.score,
            reputation=best.reputation,
            candidates=candidates,
            considered=len(candidates),
            eligible=len(eligible),
            reason=(
                f"highest-ranked of {len(eligible)} eligible candidate(s) "
                f"for capability {capability!r}"
            ),
        )

    def rank(
        self,
        capability: str,
        *,
        contract: Any | None = None,
        available: set[str] | None = None,
    ) -> list[BindingCandidate]:
        """Score every directory candidate for ``capability``, best first.

        The returned list is deterministic — sorted by descending score, then by
        org id — and includes rejected candidates (score ``0.0``) so the decision
        is fully auditable. Eligible candidates always sort ahead of rejected ones
        because a rejected candidate scores ``0.0``.
        """
        records = self.directory.find(capability=capability)
        scored: list[BindingCandidate] = []
        for record in records:
            scored.append(self._score_candidate(record, contract=contract, available=available))
        scored.sort(key=lambda c: (-c.score, c.org))
        return scored

    # -- scoring ------------------------------------------------------------

    def _score_candidate(
        self, record: Any, *, contract: Any | None, available: set[str] | None
    ) -> BindingCandidate:
        org = record.name
        resolution = self.directory.try_resolve(org, principal=self.principal)
        allowed = bool(resolution.allowed)
        reachable = available is None or org in available
        reputation = self._reputation_weight(org)
        reliability, settlements = self._settlement_reliability(org)
        contract_fit = self._contract_fit(org, contract, settlements)
        rejected_reason: str | None = None
        if not allowed:
            rejected_reason = resolution.decision.reason or "not allow-listed"
        elif not reachable:
            rejected_reason = "no participant binding for this org"
        score = (
            self.weights.score(reputation, reliability, contract_fit)
            if (allowed and reachable)
            else 0.0
        )
        return BindingCandidate(
            org=org,
            protocol=getattr(record, "protocol", "a2a"),
            allowed=allowed,
            reachable=reachable,
            reputation=round(reputation, 9),
            settlement_reliability=round(reliability, 9),
            settlements=settlements,
            contract_fit=round(contract_fit, 9),
            score=score,
            rejected_reason=rejected_reason,
        )

    def _reputation_weight(self, org: str) -> float:
        """The candidate's bounded reputation weight (``1.0`` with no ledger)."""
        if self.reputation is None:
            return 1.0
        try:
            return float(self.reputation.weight(org))
        except Exception:  # noqa: BLE001 - an unknown member is simply unweighted
            return 1.0

    def _settlement_row(self, org: str) -> Any | None:
        if self.settlement_book is None:
            return None
        try:
            report = self.settlement_book.report(org)
        except Exception:  # noqa: BLE001 - a bookless / errored lookup is "no history"
            return None
        rows = getattr(report, "rows", [])
        return rows[0] if rows else None

    def _settlement_reliability(self, org: str) -> tuple[float, int]:
        """Share of prior settlements honoured, and how many there were.

        Returns the configured ``unknown_settlement_score`` (and ``0`` settlements)
        when there is no history, so a newcomer is neither trusted nor punished.
        """
        row = self._settlement_row(org)
        if row is None:
            return self.weights.unknown_settlement_score, 0
        settlements = int(getattr(row, "settlements", 0) or 0)
        if settlements <= 0:
            return self.weights.unknown_settlement_score, 0
        settled = int(getattr(row, "settled", 0) or 0)
        breached = int(getattr(row, "breached", 0) or 0)
        decided = settled + breached
        if decided <= 0:
            return self.weights.unknown_settlement_score, settlements
        return settled / decided, settlements

    def _contract_fit(self, org: str, contract: Any | None, settlements: int) -> float:
        """How well the candidate's prior delivered cost fits the step's price.

        ``1.0`` when there is no contract, no agreed price, or no settlement history
        to judge against (no penalty for the unknown); otherwise ``1.0`` if the
        candidate's average delivered cost sat at or under the agreed price, scaled
        down toward ``0`` the further a historical overrun ran past it.
        """
        if contract is None or settlements <= 0:
            return 1.0
        terms = getattr(contract, "terms", None)
        price = float(getattr(terms, "price_usd", 0.0) or 0.0)
        if price <= 0:
            return 1.0
        row = self._settlement_row(org)
        if row is None:
            return 1.0
        count = int(getattr(row, "settlements", 0) or 0)
        if count <= 0:
            return 1.0
        avg_delivered = float(getattr(row, "total_delivered_usd", 0.0) or 0.0) / count
        if avg_delivered <= price:
            return 1.0
        return round(max(0.0, min(1.0, price / avg_delivered)), 9)
