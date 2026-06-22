"""Differential-privacy accounting: a per-subject privacy budget over learning.

The federated round already bounds a *single member's per-round influence* with
clipping and an optional Gaussian mechanism
(:class:`~vincio.optimize.federated.PrivacyConfig`), but a single bounded round is
not a guarantee about a *subject* whose data is touched again and again — by every
memory consolidation that folds their episodes into a durable summary, by every
federated contribution that learns from their traffic. This module adds the rung
the platform was missing: **a provable, composing, per-subject privacy budget**
over memory consolidation and the whole learning loop.

The accountant is a **Rényi differential privacy (RDP) / moments accountant**.
Each accounted release is modeled as a Gaussian mechanism applied to a clipped,
optionally subsampled statistic; its privacy cost is an RDP curve over a grid of
orders. RDP composes by simple addition, so the cumulative ``(ε, δ)`` a subject's
data has spent across many consolidations and learning rounds is the sum of the
per-step curves converted once at the end — a far tighter bound than naively
adding each step's ``ε``. Three pieces, all deterministic, offline-first, and on
the audit chain:

* :func:`gaussian_rdp` / :func:`rdp_to_epsilon` are the accountant's math: the
  (sub-sampled) Gaussian mechanism's RDP curve, and the standard RDP→``(ε, δ)``
  conversion. :class:`PrivacyMechanism` wraps one mechanism (noise multiplier,
  Poisson sample rate, step count) and reports its own curve and standalone
  ``(ε, δ)``.
* :class:`PrivacyBudget` is a per-subject (or default) ``(ε, δ)`` ceiling with an
  ``on_breach`` policy. :class:`PrivacyAccountant` tracks each subject's composed
  RDP, decides whether a proposed mechanism fits the remaining budget
  (:meth:`~PrivacyAccountant.check`), commits a spend (:meth:`~PrivacyAccountant.record`),
  or does both and refuses over budget (:meth:`~PrivacyAccountant.charge`) — every
  spend and every refusal stamped on the hash-chained audit log.
* :class:`PrivacyReport` sits alongside the cost report: a per-subject roll-up of
  ε spent, ε remaining, operations, and refusals, so the privacy guarantee is a
  mechanical, auditable number — not a policy doc.

A consolidation or a contribution that would exceed a subject's remaining budget
is **refused** (the privacy analogue of a hard cost cap) or **down-weighted**
(clipped harder so its sensitivity — and therefore its privacy cost — drops to fit
the remaining budget). The primitives this builds on — the consent ledger, the
federated clipping + Gaussian mechanism, and the signed audit chain — are already
in the platform; this module composes them into an end-to-end budget that gates a
write the way the cost report gates a dollar.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import GovernanceError
from ..core.utils import new_id, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.audit import AuditLog
    from ..storage.base import MetadataStore

__all__ = [
    "PrivacyBudgetError",
    "RDP_ORDERS",
    "gaussian_rdp",
    "rdp_to_epsilon",
    "PrivacyMechanism",
    "PrivacyBudget",
    "PrivacySpend",
    "PrivacyDecision",
    "PrivacyRow",
    "PrivacyReport",
    "PrivacyAccountant",
]


class PrivacyBudgetError(GovernanceError):
    """A learning step was refused because it would exceed a subject's DP budget.

    Raised by :meth:`PrivacyAccountant.charge` when the proposed mechanism's
    privacy cost, composed onto everything a subject's data has already spent,
    would push the cumulative ``(ε, δ)`` past the subject's
    :class:`PrivacyBudget` and the budget's ``on_breach`` policy is ``"refuse"``
    (or down-weighting cannot bring it under the ceiling). Inherits
    :class:`~vincio.core.errors.GovernanceError`'s remediation surface; the
    spent / remaining budget travels in ``.details``.
    """

    code = "PRIVACY_BUDGET_EXCEEDED"


# ---------------------------------------------------------------------------
# The accountant's math: Rényi DP of the (sub-sampled) Gaussian mechanism
# ---------------------------------------------------------------------------

# The grid of Rényi orders the accountant tracks. Integer orders keep the
# sub-sampled Gaussian's moment a finite binomial sum (Abadi et al.'s moments
# accountant); the spread from tight (2) to loose (64) lets the RDP→(ε, δ)
# conversion pick the order that minimizes ε for whatever regime a subject is in.
RDP_ORDERS: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)


def _log_comb(n: int, k: int) -> float:
    """Natural log of the binomial coefficient ``C(n, k)`` via ``lgamma``."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(terms: list[float]) -> float:
    """Numerically stable ``log(Σ exp(tᵢ))`` (ignoring ``-inf`` terms)."""
    finite = [t for t in terms if t != -math.inf]
    if not finite:
        return -math.inf
    top = max(finite)
    if top == math.inf:
        return math.inf
    return top + math.log(sum(math.exp(t - top) for t in finite))


def gaussian_rdp(
    noise_multiplier: float,
    *,
    sample_rate: float = 1.0,
    steps: int = 1,
    orders: tuple[int, ...] = RDP_ORDERS,
) -> list[float]:
    """Rényi-DP curve of a (Poisson-sub-sampled) Gaussian mechanism.

    ``noise_multiplier`` is the Gaussian noise standard deviation **relative to
    the mechanism's L2 sensitivity** (``z = σ / Δ``); a larger ``z`` is more
    private. ``sample_rate`` (``q``) is the Poisson sub-sampling rate — ``1.0``
    for a full-batch release (a memory consolidation over a subject's episodes),
    ``< 1.0`` for a sub-sampled learning step, which amplifies privacy.
    ``steps`` composes the same mechanism with itself ``steps`` times (RDP adds).

    Returns the RDP value ``ρ(α)`` for each ``α`` in ``orders``. For ``q = 1``
    this is the exact Gaussian RDP ``α / (2 z²)``; for ``q < 1`` it is the
    standard moments-accountant binomial upper bound. A non-positive
    ``noise_multiplier`` is not differentially private, so every order is
    ``+inf`` (any finite budget then refuses it — no noise, no guarantee).
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    q = max(0.0, min(1.0, sample_rate))
    if steps == 0 or q == 0.0:
        return [0.0 for _ in orders]
    if noise_multiplier <= 0.0:
        return [math.inf for _ in orders]
    z2 = noise_multiplier * noise_multiplier
    out: list[float] = []
    for alpha in orders:
        if q >= 1.0:
            # Full-batch Gaussian: exact closed form.
            rho = alpha / (2.0 * z2)
        else:
            # Moments-accountant binomial bound for the sub-sampled Gaussian
            # (Abadi et al. 2016): ρ(α) = (1/(α-1))·log Σ_k C(α,k) (1-q)^{α-k}
            # q^k exp(k(k-1)/(2z²)), evaluated in log space for stability.
            log_terms = [
                _log_comb(alpha, k)
                + (alpha - k) * math.log1p(-q)
                + k * math.log(q)
                + (k * k - k) / (2.0 * z2)
                for k in range(alpha + 1)
            ]
            rho = _logsumexp(log_terms) / (alpha - 1)
        out.append(rho * steps)
    return out


def rdp_to_epsilon(
    rdp: list[float], *, delta: float, orders: tuple[int, ...] = RDP_ORDERS
) -> float:
    """Convert a composed RDP curve to an ``(ε, δ)`` differential-privacy bound.

    Uses the standard tail bound ``ε(δ) = min_α [ ρ(α) + ln(1/δ) / (α − 1) ]``
    (Mironov 2017): every ``(α, ρ(α))``-RDP guarantee implies ``(ε, δ)``-DP, and
    the accountant takes the order that gives the tightest ``ε``. Returns ``0.0``
    for an all-zero curve (nothing spent) and ``+inf`` if no order yields a
    finite bound.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not rdp or all(rho <= 0.0 for rho in rdp):
        # An all-zero RDP curve is the identity mechanism: no privacy is spent, so
        # ε = 0 (the ln(1/δ)/(α−1) tail term applies only to a real release).
        return 0.0
    log_inv_delta = math.log(1.0 / delta)
    best = math.inf
    for alpha, rho in zip(orders, rdp, strict=True):
        if alpha <= 1 or rho == math.inf:
            continue
        eps = rho + log_inv_delta / (alpha - 1)
        if eps < best:
            best = eps
    return max(0.0, best)


def _sum_rdp(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


# ---------------------------------------------------------------------------
# Mechanism, budget, spend, decision, report
# ---------------------------------------------------------------------------


class PrivacyMechanism(BaseModel):
    """One differentially-private release, as accounted against a budget.

    The unit a :class:`PrivacyAccountant` composes: a Gaussian mechanism with a
    ``noise_multiplier`` (noise ``σ`` relative to the clipped L2 sensitivity),
    an optional Poisson ``sample_rate`` (``< 1`` amplifies privacy), and a
    ``steps`` count for a multi-step release. ``label`` names the operation for
    the audit trail (e.g. ``"memory_consolidation"``).
    """

    label: str = "gaussian"
    noise_multiplier: float = 1.0
    sample_rate: float = 1.0
    steps: int = 1

    def rdp(self, orders: tuple[int, ...] = RDP_ORDERS) -> list[float]:
        """This mechanism's RDP curve over ``orders``."""
        return gaussian_rdp(
            self.noise_multiplier,
            sample_rate=self.sample_rate,
            steps=self.steps,
            orders=orders,
        )

    def epsilon(self, *, delta: float, orders: tuple[int, ...] = RDP_ORDERS) -> float:
        """The standalone ``(ε, δ)`` of this single mechanism."""
        return rdp_to_epsilon(self.rdp(orders), delta=delta, orders=orders)

    def scaled(self, weight: float) -> PrivacyMechanism:
        """Down-weight the release: clip harder so its sensitivity drops.

        Scaling a contribution by ``weight ≤ 1`` scales its L2 sensitivity by the
        same factor, so the effective noise multiplier rises to
        ``noise_multiplier / weight`` and the privacy cost falls by ``weight²``.
        Returns a new mechanism representing that down-weighted release.
        """
        w = max(0.0, min(1.0, weight))
        if w <= 0.0:
            return self.model_copy(update={"noise_multiplier": math.inf})
        return self.model_copy(update={"noise_multiplier": self.noise_multiplier / w})


class PrivacyBudget(BaseModel):
    """A per-subject (or default) ``(ε, δ)`` privacy ceiling.

    ``subject_id`` scopes the budget to one data subject; ``None`` is the default
    budget applied to any subject without a specific one. ``epsilon`` is the
    cumulative privacy loss the subject's data may incur across *all* accounted
    consolidations and learning rounds; ``delta`` is the failure probability the
    ``ε`` is reported at. ``on_breach`` decides what happens when a proposed step
    would exceed the ceiling: ``"refuse"`` rejects it outright (a hard cap), while
    ``"downweight"`` admits a clipped-harder version that fits the remaining
    budget when one exists.
    """

    subject_id: str | None = None
    epsilon: float = 1.0
    delta: float = 1e-5
    on_breach: Literal["refuse", "downweight"] = "refuse"


class PrivacySpend(BaseModel):
    """One accounted privacy release for a subject — a row on the audit chain."""

    id: str = Field(default_factory=lambda: new_id("privacy"))
    subject_id: str
    operation: str = ""
    round_id: str = ""
    epsilon: float = 0.0  # this release's standalone (ε, δ)
    delta: float = 1e-5
    noise_multiplier: float = 0.0
    sample_rate: float = 1.0
    steps: int = 1
    downweight: float = 1.0
    cumulative_epsilon: float = 0.0  # composed ε after this release
    at: datetime = Field(default_factory=utcnow)
    details: dict[str, Any] = Field(default_factory=dict)


class PrivacyDecision(BaseModel):
    """An explainable verdict on whether a proposed release fits the budget."""

    action: Literal["allow", "downweight", "refuse"] = "allow"
    subject_id: str = ""
    spent_epsilon: float = 0.0
    projected_epsilon: float = 0.0
    limit_epsilon: float = math.inf
    remaining_epsilon: float = math.inf
    delta: float = 1e-5
    downweight: float = 1.0
    reason: str = ""

    @property
    def allowed(self) -> bool:
        """True when the release may proceed (possibly down-weighted)."""
        return self.action in ("allow", "downweight")


class PrivacyRow(BaseModel):
    """One subject's line in a :class:`PrivacyReport`."""

    subject_id: str
    spent_epsilon: float = 0.0
    limit_epsilon: float | None = None
    remaining_epsilon: float | None = None
    delta: float = 1e-5
    operations: int = 0
    refusals: int = 0
    on_breach: str | None = None


class PrivacyReport(BaseModel):
    """Per-subject DP budget roll-up — the privacy analogue of the cost report.

    Sits alongside :meth:`~vincio.core.app.ContextApp.cost_report`: each row is a
    subject's cumulative ``ε`` spent against its ceiling, with the operation and
    refusal counts, so the spent budget is an auditable number.
    """

    delta: float = 1e-5
    rows: list[PrivacyRow] = Field(default_factory=list)

    @property
    def total_spent_epsilon(self) -> float:
        """Sum of ε spent across all subjects (a fleet-level loss indicator)."""
        return round(sum(r.spent_epsilon for r in self.rows), 9)

    def print_summary(self) -> None:  # pragma: no cover - cosmetic
        """Print a compact per-subject ε-spent / ε-remaining table."""
        print(f"Privacy report (δ={self.delta:g})")
        for row in self.rows:
            limit = "∞" if row.limit_epsilon is None else f"{row.limit_epsilon:g}"
            remaining = (
                "∞" if row.remaining_epsilon is None else f"{row.remaining_epsilon:g}"
            )
            print(
                f"  {row.subject_id}: ε={row.spent_epsilon:g}/{limit} "
                f"(remaining {remaining}), ops={row.operations}, refusals={row.refusals}"
            )


# ---------------------------------------------------------------------------
# The composing accountant
# ---------------------------------------------------------------------------


class PrivacyAccountant:
    """A composing, per-subject differential-privacy budget over the learning loop.

    Tracks the cumulative RDP each subject's data has spent across every accounted
    memory consolidation and learning round, gates a proposed release against the
    subject's remaining budget, and reports the spent budget — every spend and
    refusal on the hash-chained audit log. Attach one to a
    :class:`~vincio.core.app.ContextApp` with
    :meth:`~vincio.core.app.ContextApp.use_privacy_accountant`; it then gates
    memory consolidation and federated contributions automatically and surfaces a
    :class:`PrivacyReport` next to the cost report.

    ``default_budget`` applies to any subject without a specific
    :class:`PrivacyBudget` (set per-subject ones with :meth:`set_budget`).
    ``default_mechanism`` is the privacy posture charged when a caller does not
    pass an explicit :class:`PrivacyMechanism`.
    """

    def __init__(
        self,
        *,
        default_budget: PrivacyBudget | None = None,
        default_mechanism: PrivacyMechanism | None = None,
        orders: tuple[int, ...] = RDP_ORDERS,
        delta: float = 1e-5,
        audit: AuditLog | None = None,
        store: MetadataStore | None = None,
    ) -> None:
        self.orders = orders
        self.default_delta = delta
        self.audit = audit
        self.store = store
        self.default_mechanism = default_mechanism or PrivacyMechanism()
        self._budgets: dict[str | None, PrivacyBudget] = {}
        if default_budget is not None:
            self._budgets[default_budget.subject_id] = default_budget
        # Per-subject composed RDP curve, spend ledger, and refusal tally.
        self._rdp: dict[str, list[float]] = {}
        self._spends: dict[str, list[PrivacySpend]] = {}
        self._refusals: dict[str, int] = {}
        if store is not None:
            self._load()

    # -- budgets ------------------------------------------------------------

    def set_budget(self, budget: PrivacyBudget) -> PrivacyBudget:
        """Register a per-subject (or default, ``subject_id=None``) budget."""
        self._budgets[budget.subject_id] = budget
        return budget

    def budget_for(self, subject_id: str) -> PrivacyBudget | None:
        """The budget governing ``subject_id`` (specific, else the default)."""
        return self._budgets.get(subject_id) or self._budgets.get(None)

    # -- spent / remaining --------------------------------------------------

    def composed_rdp(self, subject_id: str) -> list[float]:
        """The subject's cumulative RDP curve across all recorded spends."""
        return list(self._rdp.get(subject_id, [0.0 for _ in self.orders]))

    def _delta_for(self, subject_id: str, delta: float | None) -> float:
        if delta is not None:
            return delta
        budget = self.budget_for(subject_id)
        return budget.delta if budget is not None else self.default_delta

    def spent(self, subject_id: str, *, delta: float | None = None) -> float:
        """Cumulative ``ε`` the subject's data has spent at ``δ``."""
        d = self._delta_for(subject_id, delta)
        return rdp_to_epsilon(self.composed_rdp(subject_id), delta=d, orders=self.orders)

    def remaining(self, subject_id: str, *, delta: float | None = None) -> float:
        """Remaining ``ε`` headroom before the subject's budget is spent."""
        budget = self.budget_for(subject_id)
        if budget is None:
            return math.inf
        return max(0.0, budget.epsilon - self.spent(subject_id, delta=delta))

    # -- the gate -----------------------------------------------------------

    def check(
        self,
        subject_id: str,
        mechanism: PrivacyMechanism | None = None,
        *,
        delta: float | None = None,
    ) -> PrivacyDecision:
        """Decide whether ``mechanism`` fits ``subject_id``'s remaining budget.

        Projects the mechanism's RDP onto the subject's composed curve, converts
        to ``ε`` at ``δ``, and compares against the budget. Returns ``"allow"``
        when it fits, ``"refuse"`` when it does not (or ``"downweight"`` with the
        largest ``≤ 1`` factor that fits, when the budget's ``on_breach`` is
        ``"downweight"``). Pure — records nothing; pair with :meth:`record`.
        """
        mech = mechanism or self.default_mechanism
        d = self._delta_for(subject_id, delta)
        composed = self.composed_rdp(subject_id)
        spent = rdp_to_epsilon(composed, delta=d, orders=self.orders)
        mech_rdp = mech.rdp(self.orders)
        projected = rdp_to_epsilon(_sum_rdp(composed, mech_rdp), delta=d, orders=self.orders)
        budget = self.budget_for(subject_id)
        if budget is None:
            return PrivacyDecision(
                action="allow",
                subject_id=subject_id,
                spent_epsilon=round(spent, 9),
                projected_epsilon=round(projected, 9),
                delta=d,
                reason="no privacy budget set for subject (unbounded)",
            )
        limit = budget.epsilon
        remaining = max(0.0, limit - spent)
        if projected <= limit + 1e-12:
            return PrivacyDecision(
                action="allow",
                subject_id=subject_id,
                spent_epsilon=round(spent, 9),
                projected_epsilon=round(projected, 9),
                limit_epsilon=limit,
                remaining_epsilon=round(remaining, 9),
                delta=d,
                reason=f"ε {projected:.4g} ≤ budget {limit:g}",
            )
        if budget.on_breach == "downweight":
            weight = self._solve_downweight(composed, mech_rdp, limit, d)
            if weight >= 1.0:  # pragma: no cover - covered by the allow branch
                action, reason = "allow", f"ε {projected:.4g} ≤ budget {limit:g}"
            elif weight <= 0.0:
                action, reason = (
                    "refuse",
                    f"already at budget (ε {spent:.4g} ≥ {limit:g}); cannot down-weight",
                )
            else:
                down_rdp = [weight * weight * r for r in mech_rdp]
                projected = rdp_to_epsilon(
                    _sum_rdp(composed, down_rdp), delta=d, orders=self.orders
                )
                action, reason = (
                    "downweight",
                    f"down-weighted to {weight:.3f} to fit budget {limit:g}",
                )
            return PrivacyDecision(
                action=action,  # type: ignore[arg-type]
                subject_id=subject_id,
                spent_epsilon=round(spent, 9),
                projected_epsilon=round(projected, 9),
                limit_epsilon=limit,
                remaining_epsilon=round(remaining, 9),
                delta=d,
                downweight=round(weight, 6),
                reason=reason,
            )
        return PrivacyDecision(
            action="refuse",
            subject_id=subject_id,
            spent_epsilon=round(spent, 9),
            projected_epsilon=round(projected, 9),
            limit_epsilon=limit,
            remaining_epsilon=round(remaining, 9),
            delta=d,
            reason=f"ε {projected:.4g} would exceed budget {limit:g}",
        )

    def _solve_downweight(
        self, composed: list[float], mech_rdp: list[float], limit: float, delta: float
    ) -> float:
        """Largest weight ``∈ [0, 1]`` whose down-weighted spend fits ``limit``."""
        if rdp_to_epsilon(composed, delta=delta, orders=self.orders) > limit + 1e-9:
            return 0.0
        if rdp_to_epsilon(
            _sum_rdp(composed, mech_rdp), delta=delta, orders=self.orders
        ) <= limit + 1e-12:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            scaled = [mid * mid * r for r in mech_rdp]
            eps = rdp_to_epsilon(_sum_rdp(composed, scaled), delta=delta, orders=self.orders)
            if eps <= limit:
                lo = mid
            else:
                hi = mid
        return lo

    # -- commit -------------------------------------------------------------

    def record(
        self,
        subject_id: str,
        mechanism: PrivacyMechanism | None = None,
        *,
        operation: str = "",
        round_id: str = "",
        downweight: float = 1.0,
        delta: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> PrivacySpend:
        """Commit a privacy spend for ``subject_id`` and stamp it on the audit log.

        Composes the (optionally down-weighted) mechanism's RDP onto the subject's
        running curve, records a :class:`PrivacySpend`, and writes a
        ``privacy_spend`` audit entry. Returns the recorded spend.
        """
        mech = mechanism or self.default_mechanism
        d = self._delta_for(subject_id, delta)
        effective = mech.scaled(downweight) if downweight < 1.0 else mech
        spend_rdp = effective.rdp(self.orders)
        composed = self.composed_rdp(subject_id)
        new_composed = _sum_rdp(composed, spend_rdp)
        self._rdp[subject_id] = new_composed
        cumulative = rdp_to_epsilon(new_composed, delta=d, orders=self.orders)
        spend = PrivacySpend(
            subject_id=subject_id,
            operation=operation,
            round_id=round_id,
            epsilon=round(rdp_to_epsilon(spend_rdp, delta=d, orders=self.orders), 9),
            delta=d,
            noise_multiplier=round(effective.noise_multiplier, 9)
            if effective.noise_multiplier != math.inf
            else math.inf,
            sample_rate=mech.sample_rate,
            steps=mech.steps,
            downweight=round(downweight, 6),
            cumulative_epsilon=round(cumulative, 9),
            details=details or {},
        )
        self._spends.setdefault(subject_id, []).append(spend)
        self._persist(spend)
        self._audit(spend, decision="allow", action=operation or "privacy_spend")
        return spend

    def charge(
        self,
        subject_id: str,
        mechanism: PrivacyMechanism | None = None,
        *,
        operation: str = "",
        round_id: str = "",
        delta: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> PrivacySpend:
        """Gate **and** commit in one call — raise if the budget refuses.

        Runs :meth:`check`; on an allow (or down-weight) it records the spend (at
        the decided weight) and returns it; on a refusal it tallies the refusal,
        writes a ``privacy_refused`` audit entry, and raises
        :class:`PrivacyBudgetError`. This is the privacy analogue of a hard cost
        cap — the call that a consolidation or contribution routes through.
        """
        decision = self.check(subject_id, mechanism, delta=delta)
        if not decision.allowed:
            self._refusals[subject_id] = self._refusals.get(subject_id, 0) + 1
            self._audit_refusal(decision, operation=operation or "privacy_refused")
            raise PrivacyBudgetError(
                f"privacy budget exceeded for subject {subject_id!r}: {decision.reason}",
                details={
                    "subject_id": subject_id,
                    "operation": operation,
                    "spent_epsilon": decision.spent_epsilon,
                    "projected_epsilon": decision.projected_epsilon,
                    "limit_epsilon": decision.limit_epsilon,
                    "remaining_epsilon": decision.remaining_epsilon,
                    "delta": decision.delta,
                },
            )
        merged = dict(details or {})
        if decision.action == "downweight":
            merged["downweighted"] = decision.downweight
        return self.record(
            subject_id,
            mechanism,
            operation=operation,
            round_id=round_id,
            downweight=decision.downweight,
            delta=delta,
            details=merged,
        )

    def note_refusal(self, decision: PrivacyDecision, *, operation: str = "") -> None:
        """Tally and audit a refusal a caller decided on its own.

        For a release a caller **cannot** make more private — a deterministic memory
        consolidation has no noise knob to turn — down-weighting is not honestly
        realizable, so the caller gates on the full cost (:meth:`check`) and routes a
        non-clean fit here. Increments the subject's refusal count and writes a
        ``privacy_refused`` audit entry, mirroring :meth:`charge`'s refusal path.
        """
        self._refusals[decision.subject_id] = self._refusals.get(decision.subject_id, 0) + 1
        self._audit_refusal(decision, operation=operation or "privacy_refused")

    # -- reporting ----------------------------------------------------------

    def spends(self, subject_id: str) -> list[PrivacySpend]:
        """The recorded spend ledger for a subject."""
        return list(self._spends.get(subject_id, []))

    def subjects(self) -> list[str]:
        """Every subject with a spend, a refusal, or a specific budget."""
        named = {s for s in self._budgets if s is not None}
        return sorted(set(self._rdp) | set(self._refusals) | named)

    def report(self, subject_id: str | None = None, *, delta: float | None = None) -> PrivacyReport:
        """Per-subject ε-spent / ε-remaining roll-up — alongside the cost report."""
        ids = [subject_id] if subject_id is not None else self.subjects()
        # Measure every row at one δ so the header, rows, and total are consistent
        # and summable — the report is an "as-of δ" view, like the cost report's
        # currency. Per-subject budget ceilings (their own ε) are reported as-is.
        d = delta if delta is not None else self.default_delta
        rows: list[PrivacyRow] = []
        for sid in ids:
            budget = self.budget_for(sid)
            spent = self.spent(sid, delta=d)
            rows.append(
                PrivacyRow(
                    subject_id=sid,
                    spent_epsilon=round(spent, 9),
                    limit_epsilon=budget.epsilon if budget is not None else None,
                    remaining_epsilon=(
                        round(max(0.0, budget.epsilon - spent), 9)
                        if budget is not None
                        else None
                    ),
                    delta=d,
                    operations=len(self._spends.get(sid, [])),
                    refusals=self._refusals.get(sid, 0),
                    on_breach=budget.on_breach if budget is not None else None,
                )
            )
        return PrivacyReport(delta=d, rows=rows)

    def reset(self, subject_id: str | None = None) -> None:
        """Clear recorded spends (one subject, or all) — for tests / new epochs."""
        if subject_id is None:
            self._rdp.clear()
            self._spends.clear()
            self._refusals.clear()
            return
        self._rdp.pop(subject_id, None)
        self._spends.pop(subject_id, None)
        self._refusals.pop(subject_id, None)

    # -- audit & persistence ------------------------------------------------

    def _audit(self, spend: PrivacySpend, *, decision: str, action: str) -> None:
        if self.audit is None:
            return
        self.audit.record(
            "privacy_spend",
            decision=decision,
            resource=spend.subject_id,
            details={
                "operation": spend.operation or action,
                "round_id": spend.round_id,
                "epsilon": spend.epsilon,
                "delta": spend.delta,
                "noise_multiplier": spend.noise_multiplier,
                "sample_rate": spend.sample_rate,
                "downweight": spend.downweight,
                "cumulative_epsilon": spend.cumulative_epsilon,
            },
        )

    def _audit_refusal(self, decision: PrivacyDecision, *, operation: str) -> None:
        if self.audit is None:
            return
        self.audit.record(
            "privacy_refused",
            decision="deny",
            resource=decision.subject_id,
            details={
                "operation": operation,
                "spent_epsilon": decision.spent_epsilon,
                "projected_epsilon": decision.projected_epsilon,
                "limit_epsilon": decision.limit_epsilon,
                "delta": decision.delta,
                "reason": decision.reason,
            },
        )

    def _persist(self, spend: PrivacySpend) -> None:
        if self.store is None:
            return
        try:
            self.store.save("privacy_spends", spend.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 - persistence is best-effort
            return

    def _load(self) -> None:
        assert self.store is not None
        try:
            rows = self.store.query("privacy_spends", limit=100_000)
        except Exception:  # noqa: BLE001 - a store without the kind is simply empty
            return
        for row in rows:
            spend = PrivacySpend.model_validate(row)
            self._spends.setdefault(spend.subject_id, []).append(spend)
            mech = PrivacyMechanism(
                noise_multiplier=spend.noise_multiplier,
                sample_rate=spend.sample_rate,
                steps=spend.steps,
            )
            composed = self.composed_rdp(spend.subject_id)
            self._rdp[spend.subject_id] = _sum_rdp(composed, mech.rdp(self.orders))
