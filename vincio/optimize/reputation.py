"""Cross-fleet reputation & reliability-weighted federated aggregation.

The federated round (:mod:`vincio.optimize.federated`) merges every member's
contribution with **equal weight**, and the differential-privacy accountant
(:mod:`vincio.governance.privacy`) bounds what each member can *leak* — but the
platform has no notion of a member's **track record**. A member whose
contributions repeatedly fail the no-regression gate still pulls the shared
consensus geometry as hard as a member whose contributions consistently help.

This module adds the missing rung: **an earned, per-member reputation that
discounts an unreliable or adversarial member's pull on the consensus**. The
reputation is accrued purely from how each past contribution fared against the
existing no-regression gate — never from raw traffic — and lives on the same
signed audit chain as every other governance decision, so it is a mechanical,
auditable, replayable number.

Three pieces, all offline-first, deterministic, and additive on the frozen
federated surface:

* :class:`ReputationLedger` keeps a per-member reliability signal as a
  Beta-Bernoulli posterior over gate outcomes (a robust generalization of the
  ``successes / calls`` reliability scoring the tool registry already uses): a
  brand-new member earns the benefit of the doubt from a configurable prior, a
  repeatedly-regressing member's score decays toward a floor, and a reformed
  member recovers. :meth:`~ReputationLedger.record_outcome` composes one round's
  verdict and stamps it on the audit chain; :meth:`~ReputationLedger.weight` maps
  a member's reputation to an aggregation weight in ``[weight_floor, 1.0]``;
  :meth:`~ReputationLedger.replay_from_audit` reconstructs the whole ledger from
  the chain, so reputation is verifiable by anyone holding the audit log.
* :meth:`~ReputationLedger.weight` is consumed by the
  :class:`~vincio.optimize.federated.SecureAggregator`, which weights a member's
  contribution by its reputation before distilling the consensus subspace — so a
  regressing or adversarial member is **discounted without being singled out**.
  Because secure aggregation hides individual updates behind cancelling masks,
  the weight is folded into the contribution *before* masking (the masks still
  cancel exactly); the aggregator enforces that a masked contribution carries the
  weight the ledger assigned, and applies the weight directly only on the
  unmasked path.
* The discount is **bounded and reversible**: a weight never leaves
  ``[weight_floor, 1.0]``, so a bad reputation only ever *lowers* a member's
  pull, never raises it and never zeroes it out; and adoption still clears the
  very same no-regression and canary gates a local promotion does, so reputation
  can never bypass the quality bar — it only changes which geometry the fleet
  converges toward when every candidate already clears the gate.

Attach a ledger with
:meth:`~vincio.core.app.ContextApp.use_reputation_ledger`; the federated round
then weights contributions by reputation and records each round's gate verdict
back to the ledger automatically, and
:meth:`~vincio.core.app.ContextApp.reputation_report` rolls up each member's
score and weight next to the cost and privacy reports. Nothing here is required:
without a ledger the federated round behaves exactly as before (every member
weighted equally).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.errors import OptimizationError
from ..core.utils import utcnow

if TYPE_CHECKING:
    from ..core.events import EventBus
    from ..security.audit import AuditLog
    from ..storage.base import MetadataStore

__all__ = [
    "ReputationError",
    "ReputationConfig",
    "MemberReputation",
    "ReputationWeights",
    "ReputationRow",
    "ReputationReport",
    "ReputationLedger",
]

# The audit action a reputation update is recorded under — the single key
# :meth:`ReputationLedger.replay_from_audit` reads the ledger back from the chain.
REPUTATION_ACTION = "reputation_update"


class ReputationError(OptimizationError):
    """A reputation operation could not proceed.

    Raised on an incoherent :class:`ReputationConfig` (a floor above the ceiling,
    a non-positive prior) or an audit replay that cannot be reconstructed.
    Inherits :class:`~vincio.core.errors.OptimizationError`'s stable ``.code`` so
    it carries the same remediation surface as every other optimization failure
    — no new error-catalog entry is required.
    """


class ReputationConfig(BaseModel):
    """How a member's gate track record maps to an aggregation weight.

    Reputation is the mean of a Beta-Bernoulli posterior over a member's past
    no-regression gate outcomes: a pass is a success, a regression a failure. The
    prior gives a never-seen member a sensible starting reputation (the benefit of
    the doubt) so a newcomer is neither trusted blindly nor frozen out; evidence
    then moves the score.

    * ``prior_success`` / ``prior_failure`` are the Beta prior's pseudo-counts.
      Their ratio is a newcomer's starting reputation
      (``prior_success / (prior_success + prior_failure)``); their sum is how much
      evidence it takes to move it (a small sum lets real outcomes dominate fast).
    * ``decay`` multiplicatively forgets accumulated counts each round
      (``1.0`` keeps full history; ``< 1.0`` weights recent rounds more, so a
      reformed member recovers and a once-reliable member cannot coast forever).
    * ``weight_floor`` / ``weight_ceiling`` bound the aggregation weight a
      reputation maps to. The floor keeps a discounted member's pull positive —
      it is *discounted, never singled out or zeroed*, and can always recover; the
      ceiling (``1.0``) means reputation only ever *lowers* a member's pull
      relative to the unweighted round, never raises it past parity.

    The weight is ``weight_floor + (weight_ceiling − weight_floor) · reputation``,
    a monotonic map from reputation ``∈ (0, 1)`` to weight
    ``∈ [weight_floor, weight_ceiling]``.
    """

    prior_success: float = 2.0
    prior_failure: float = 1.0
    decay: float = 1.0
    weight_floor: float = 0.1
    weight_ceiling: float = 1.0

    def validate_coherent(self) -> ReputationConfig:
        """Raise :class:`ReputationError` unless the configuration is coherent."""
        if self.prior_success <= 0.0 or self.prior_failure <= 0.0:
            raise ReputationError(
                "reputation prior pseudo-counts must be positive; got "
                f"prior_success={self.prior_success}, prior_failure={self.prior_failure}"
            )
        if not 0.0 < self.decay <= 1.0:
            raise ReputationError(f"reputation decay must be in (0, 1]; got {self.decay}")
        if not 0.0 <= self.weight_floor <= self.weight_ceiling <= 1.0:
            raise ReputationError(
                "reputation weights must satisfy 0 ≤ weight_floor ≤ weight_ceiling ≤ 1; got "
                f"floor={self.weight_floor}, ceiling={self.weight_ceiling}"
            )
        return self

    def reputation_of(self, successes: float, failures: float) -> float:
        """Posterior-mean reputation for decayed ``successes`` / ``failures``."""
        numerator = self.prior_success + max(0.0, successes)
        denominator = self.prior_success + self.prior_failure + max(0.0, successes) + max(
            0.0, failures
        )
        return numerator / denominator if denominator > 0.0 else 0.0

    def weight_of(self, reputation: float) -> float:
        """Map a reputation ``∈ [0, 1]`` to an aggregation weight in the band."""
        clamped = min(1.0, max(0.0, reputation))
        span = self.weight_ceiling - self.weight_floor
        return round(self.weight_floor + span * clamped, 9)


class MemberReputation(BaseModel):
    """One member's reputation snapshot, its track record as an auditable number."""

    member_id: str
    successes: float = 0.0
    failures: float = 0.0
    rounds: int = 0
    reputation: float = 0.0
    weight: float = 1.0
    last_round: str | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class ReputationWeights(BaseModel):
    """The per-member aggregation weights assigned for one federated round."""

    round_id: str = "round"
    weights: dict[str, float] = Field(default_factory=dict)
    reputations: dict[str, float] = Field(default_factory=dict)

    def get(self, member_id: str, default: float = 1.0) -> float:
        """The weight assigned to ``member_id`` (``default`` if unweighted)."""
        return self.weights.get(member_id, default)


class ReputationRow(BaseModel):
    """One member's line in a :class:`ReputationReport`."""

    member_id: str
    reputation: float = 0.0
    weight: float = 1.0
    successes: float = 0.0
    failures: float = 0.0
    rounds: int = 0
    last_round: str | None = None


class ReputationReport(BaseModel):
    """Per-member reputation roll-up, alongside the cost and privacy reports.

    Each row is a member's earned reliability score and the aggregation weight it
    currently maps to, with the success / failure tally behind it, so a member's
    standing in the fleet is a mechanical, auditable number.
    """

    rows: list[ReputationRow] = Field(default_factory=list)

    @property
    def mean_reputation(self) -> float:
        """Mean reputation across reported members (a fleet-health indicator)."""
        if not self.rows:
            return 0.0
        return round(sum(r.reputation for r in self.rows) / len(self.rows), 9)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print a compact per-member reputation / weight table."""
        print("Reputation report")
        for row in self.rows:
            print(
                f"  {row.member_id}: reputation={row.reputation:.3f} weight={row.weight:.3f} "
                f"(✓{row.successes:g}/✗{row.failures:g}, {row.rounds} rounds)"
            )


class ReputationLedger:
    """A per-member, gate-earned reputation that weights federated aggregation.

    Tracks each member's reliability as a Beta-Bernoulli posterior over its past
    no-regression gate outcomes, maps that to an aggregation weight in
    ``[weight_floor, 1.0]``, and records every update on the hash-chained audit
    log — so reputation is accrued from how contributions *fared against the
    gate*, never from raw traffic, and is reconstructable from the chain alone.

    Attach one to a :class:`~vincio.core.app.ContextApp` with
    :meth:`~vincio.core.app.ContextApp.use_reputation_ledger`; the federated round
    then weights contributions by reputation and records each round's verdict back
    automatically, and :meth:`~vincio.core.app.ContextApp.reputation_report` rolls
    up each member's standing next to the cost and privacy reports. Used directly,
    :meth:`record_outcome` composes a verdict, :meth:`weight` reads the current
    aggregation weight, and :meth:`assign` produces the weight vector for a round.
    """

    def __init__(
        self,
        config: ReputationConfig | None = None,
        *,
        audit: AuditLog | None = None,
        events: EventBus | None = None,
        store: MetadataStore | None = None,
    ) -> None:
        self.config = (config or ReputationConfig()).validate_coherent()
        self.audit = audit
        self.events = events
        self.store = store
        # Per-member decayed success/failure counts, round tally, and last round.
        self._successes: dict[str, float] = {}
        self._failures: dict[str, float] = {}
        self._rounds: dict[str, int] = {}
        self._last_round: dict[str, str | None] = {}
        if store is not None:
            self._load()

    # -- reads --------------------------------------------------------------

    def members(self) -> list[str]:
        """Every member the ledger has recorded an outcome for, sorted."""
        return sorted(set(self._successes) | set(self._failures) | set(self._rounds))

    def reputation(self, member_id: str) -> float:
        """The member's current posterior-mean reputation ``∈ (0, 1)``.

        A member with no recorded outcome returns the prior mean — the benefit of
        the doubt a newcomer is extended.
        """
        return round(
            self.config.reputation_of(
                self._successes.get(member_id, 0.0), self._failures.get(member_id, 0.0)
            ),
            9,
        )

    def weight(self, member_id: str) -> float:
        """The member's current aggregation weight ``∈ [weight_floor, 1.0]``."""
        return self.config.weight_of(self.reputation(member_id))

    def snapshot(self, member_id: str) -> MemberReputation:
        """A :class:`MemberReputation` capturing the member's current standing."""
        return MemberReputation(
            member_id=member_id,
            successes=round(self._successes.get(member_id, 0.0), 9),
            failures=round(self._failures.get(member_id, 0.0), 9),
            rounds=self._rounds.get(member_id, 0),
            reputation=self.reputation(member_id),
            weight=self.weight(member_id),
            last_round=self._last_round.get(member_id),
        )

    def assign(
        self, members: Iterable[str], *, round_id: str = "round"
    ) -> ReputationWeights:
        """The aggregation weight vector for a round's ``members``.

        Pure — reads the current reputations without recording anything. The
        federated round folds each weight into the matching member's contribution
        before secure-aggregation masking, so the masks still cancel exactly.
        """
        unique = sorted(set(members))
        return ReputationWeights(
            round_id=round_id,
            weights={m: self.weight(m) for m in unique},
            reputations={m: self.reputation(m) for m in unique},
        )

    # -- the update ---------------------------------------------------------

    def record_outcome(
        self,
        member_id: str,
        *,
        passed: bool,
        round_id: str = "round",
        weight_seen: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> MemberReputation:
        """Compose one round's gate verdict onto a member's reputation.

        Decays the member's accumulated counts, credits a success (``passed``) or a
        failure, recomputes the posterior-mean reputation and its aggregation
        weight, and stamps the update on the audit chain (decision ``allow`` on a
        pass, ``deny`` on a regression) and the event bus. Returns the resulting
        :class:`MemberReputation`. ``weight_seen`` records the weight the member's
        contribution actually carried this round, for audit; ``details`` is folded
        into the audit record.
        """
        self._apply(member_id, passed=passed, round_id=round_id)
        snapshot = self.snapshot(member_id)
        self._audit(snapshot, passed=passed, weight_seen=weight_seen, details=details)
        self._persist(snapshot, passed=passed, weight_seen=weight_seen)
        if self.events is not None:
            self.events.emit("reputation.update", snapshot.model_dump(mode="json"))
        return snapshot

    def record_round(
        self,
        members: Iterable[str],
        *,
        passed: bool,
        round_id: str = "round",
        weights: ReputationWeights | None = None,
        details: dict[str, Any] | None = None,
    ) -> list[MemberReputation]:
        """Record one round's verdict for every contributing ``member``.

        The round-level credit the federated round applies after the gate: each
        contributor to a round that cleared the no-regression gate earns a success,
        each contributor to one that regressed earns a failure. Across rounds a
        consistently-helpful member's reputation rises and a persistent regressor's
        decays toward the floor.
        """
        return [
            self.record_outcome(
                member_id,
                passed=passed,
                round_id=round_id,
                weight_seen=weights.get(member_id) if weights is not None else None,
                details=details,
            )
            for member_id in sorted(set(members))
        ]

    def reset(self, member_id: str | None = None) -> None:
        """Clear recorded outcomes (one member, or all) — for tests / new epochs."""
        if member_id is None:
            self._successes.clear()
            self._failures.clear()
            self._rounds.clear()
            self._last_round.clear()
            return
        self._successes.pop(member_id, None)
        self._failures.pop(member_id, None)
        self._rounds.pop(member_id, None)
        self._last_round.pop(member_id, None)

    # -- reporting ----------------------------------------------------------

    def report(self, member_id: str | None = None) -> ReputationReport:
        """Per-member reputation / weight roll-up — beside the cost report."""
        ids = [member_id] if member_id is not None else self.members()
        rows = [
            ReputationRow(
                member_id=mid,
                reputation=self.reputation(mid),
                weight=self.weight(mid),
                successes=round(self._successes.get(mid, 0.0), 9),
                failures=round(self._failures.get(mid, 0.0), 9),
                rounds=self._rounds.get(mid, 0),
                last_round=self._last_round.get(mid),
            )
            for mid in ids
        ]
        return ReputationReport(rows=rows)

    # -- audit replay -------------------------------------------------------

    def replay_from_audit(self, audit: AuditLog) -> ReputationLedger:
        """Reconstruct this ledger purely from ``audit``'s reputation records.

        Reads every ``reputation_update`` entry in order and re-applies it, so the
        in-memory reputation state is provably the one the audit chain attests —
        a member's standing is auditable and tamper-evident, recoverable from the
        signed log without any raw traffic. Replaces the ledger's current state.
        """
        self.reset()
        for entry in audit.query(action=REPUTATION_ACTION, limit=10_000_000):
            details = entry.details or {}
            member_id = entry.resource or details.get("member_id")
            if not member_id:
                continue
            self._apply(
                member_id,
                passed=bool(details.get("passed")),
                round_id=details.get("round_id"),
            )
        return self

    @classmethod
    def from_audit(
        cls, audit: AuditLog, config: ReputationConfig | None = None
    ) -> ReputationLedger:
        """Build a ledger and populate it from an audit chain in one call."""
        ledger = cls(config=config, audit=audit)
        return ledger.replay_from_audit(audit)

    # -- internals ----------------------------------------------------------

    def _apply(self, member_id: str, *, passed: bool, round_id: str | None) -> None:
        """Fold one outcome into the decayed counts (no audit / persist / events)."""
        decay = self.config.decay
        if decay < 1.0:
            self._successes[member_id] = self._successes.get(member_id, 0.0) * decay
            self._failures[member_id] = self._failures.get(member_id, 0.0) * decay
        if passed:
            self._successes[member_id] = self._successes.get(member_id, 0.0) + 1.0
        else:
            self._failures[member_id] = self._failures.get(member_id, 0.0) + 1.0
        self._rounds[member_id] = self._rounds.get(member_id, 0) + 1
        if round_id is not None:
            self._last_round[member_id] = round_id

    def _audit(
        self,
        snapshot: MemberReputation,
        *,
        passed: bool,
        weight_seen: float | None,
        details: dict[str, Any] | None,
    ) -> None:
        if self.audit is None:
            return
        # Caller details are merged first so the canonical fields below — the ones
        # ``replay_from_audit`` reads back — always win and can never be clobbered.
        record = {
            **(details or {}),
            "member_id": snapshot.member_id,
            "round_id": snapshot.last_round,
            "passed": passed,
            "successes": snapshot.successes,
            "failures": snapshot.failures,
            "rounds": snapshot.rounds,
            "reputation": snapshot.reputation,
            "weight": snapshot.weight,
            "weight_seen": weight_seen,
        }
        self.audit.record(
            REPUTATION_ACTION,
            decision="allow" if passed else "deny",
            resource=snapshot.member_id,
            details=record,
        )

    def _persist(
        self, snapshot: MemberReputation, *, passed: bool, weight_seen: float | None
    ) -> None:
        if self.store is None:
            return
        try:
            self.store.save(
                "reputation_outcomes",
                {
                    "member_id": snapshot.member_id,
                    "round_id": snapshot.last_round,
                    "passed": passed,
                    "weight_seen": weight_seen,
                },
            )
        except Exception:  # noqa: BLE001 - persistence is best-effort
            return

    def _load(self) -> None:
        assert self.store is not None  # noqa: S101 - _load runs only when a store is configured
        try:
            rows = self.store.query("reputation_outcomes", limit=10_000_000)
        except Exception:  # noqa: BLE001 - a store without the kind is simply empty
            return
        for row in rows:
            member_id = row.get("member_id")
            if not member_id:
                continue
            self._apply(member_id, passed=bool(row.get("passed")), round_id=row.get("round_id"))
