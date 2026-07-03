"""Formal verification of governance invariants.

The platform already *enforces* its governance invariants at runtime — residency
refuses an out-of-region egress, provable erasure binds a signed proof to the
removed-id set, the budget caps spend, and the injection-containment gate stops
an untrusted-tainted argument reaching a side-effecting tool without a
user-minted capability — and records each decision on the signed audit chain.
This module adds the rung beside that: a **machine-checkable proof that those
invariants hold across the whole input space, ahead of any single run**, rather
than a property merely observed after the fact.

The approach is bounded, exhaustive model checking. Each governance control is a
*deterministic, pure* decision function over a small, well-typed state — a trust
label, a capability's presence, a provider region, an accrued budget, a
removed-id set. An :class:`Invariant` pairs a formal property (the *specification*
the control must satisfy) with the finite, representative state space it
quantifies over; :class:`GovernanceVerifier` enumerates that space in full and
either proves the property holds at every point or returns the **minimal
counterexample** — the concrete assignment (the input, the labels, the
capability gap) that violates it. Over the modeled domain the check is sound and
complete: a ``held=True`` verdict is a proof, not a sample.

Three properties hold by design:

* **Invariants as machine-checked properties** — containment, residency, budget,
  and erasure are stated as predicates over the pipeline's typed state and
  checked by this in-process verifier, binding to the *same* decision functions
  the runtime uses (the containment gate is :func:`~vincio.security.requires_authority`;
  the erasure binding is :func:`~vincio.governance.verify_erasure_proof`).
* **Counterexample, not just a verdict** — a failed property yields a concrete,
  delta-minimized :class:`Counterexample`, so a governance regression is
  debuggable rather than merely flagged.
* **Auditable & offline** — a :class:`VerificationReport` is a deterministic,
  content-hashed artifact recorded on the hash-chained audit log, computed
  in-process with no external prover service.

Everything here is deterministic and dependency-free; it never depends on a model
judgment.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, BaseModel, Field

from ..core.utils import utcnow
from ..observability.finops import within_budget
from ..security.capability import AUTHORIZED, TrustLabel, requires_authority
from .lineage import build_erasure_proof, verify_erasure_proof
from .residency import ResidencyPolicy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.audit import AuditLog

__all__ = [
    "StateVariable",
    "Invariant",
    "Counterexample",
    "InvariantResult",
    "VerificationReport",
    "GovernanceVerifier",
    "containment_invariant",
    "residency_invariant",
    "budget_invariant",
    "erasure_invariant",
    "default_invariants",
    "within_budget",
]


# ---------------------------------------------------------------------------
# The bounded state space
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StateVariable:
    """One typed variable in an invariant's bounded state space.

    ``values`` are the representative points the verifier enumerates for this
    dimension, ordered from the most benign / default (index 0) to the most
    adversarial. The ordering drives counterexample minimization: relaxing a
    variable back toward ``values[0]`` while the violation persists yields a
    simpler, more readable counterexample. Every value must be JSON-serializable
    so a counterexample lands verbatim on the audit chain.
    """

    name: str
    values: tuple[Any, ...]

    @property
    def default(self) -> Any:
        """The benign baseline value (``values[0]``) minimization relaxes toward."""
        return self.values[0]


@dataclass(frozen=True, slots=True)
class Invariant:
    """A formal governance property checked over a bounded, typed state space.

    :attr:`predicate` is the *specification*: given a concrete assignment of every
    :attr:`variable`, it returns whether the property holds at that point. The
    predicate calls into the real governance decision functions, so verifying it
    verifies the shipped machinery — not a re-implementation. :class:`GovernanceVerifier`
    enumerates the full Cartesian product of the variables' values; a property is
    proven exactly when the predicate is true at every point.
    """

    id: str
    statement: str
    category: str  # containment | residency | budget | erasure
    variables: tuple[StateVariable, ...]
    predicate: Callable[[Mapping[str, Any]], bool]
    explain: Callable[[Mapping[str, Any]], str] = field(
        default=lambda assignment: "invariant violated"
    )

    @property
    def domain_size(self) -> int:
        """The number of states the verifier checks (the product of all arities)."""
        size = 1
        for variable in self.variables:
            size *= len(variable.values)
        return size

    def assignments(self) -> Iterator[dict[str, Any]]:
        """Yield every assignment in the bounded state space, in product order."""
        names = [v.name for v in self.variables]
        for combo in itertools.product(*[v.values for v in self.variables]):
            yield dict(zip(names, combo, strict=True))

    def holds(self, assignment: Mapping[str, Any]) -> bool:
        """Whether the property holds at one assignment (a guarded predicate call).

        A predicate that raises is treated as a violation rather than crashing the
        verifier — an exception at a state is itself a failure to satisfy the
        property at that state.
        """
        try:
            return bool(self.predicate(assignment))
        except Exception:  # noqa: BLE001 - a raising predicate is a violation, not a crash
            return False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class Counterexample(BaseModel):
    """A concrete, minimal state that violates an invariant.

    :attr:`assignment` is the exact point in the state space where the property
    failed — the input, the labels, the capability gap — after delta-minimization
    toward each variable's benign default, so it is the simplest witness the
    verifier could find. :attr:`explanation` renders why it violates the property.
    """

    invariant_id: str
    category: str = ""
    assignment: dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""

    def render(self) -> str:
        """A one-line human-readable rendering of the counterexample."""
        state = ", ".join(f"{k}={v!r}" for k, v in sorted(self.assignment.items()))
        return f"[{self.invariant_id}] {self.explanation} | state: {state}"


class InvariantResult(BaseModel):
    """The verdict of checking one :class:`Invariant` over its whole state space."""

    id: str
    statement: str
    category: str
    held: bool
    states_checked: int
    domain_size: int
    counterexample: Counterexample | None = None
    digest: str = ""


class VerificationReport(BaseModel):
    """The verdict of a governance-verification pass over all invariants.

    :attr:`held` is true exactly when every invariant proved across its whole
    state space. :attr:`content_hash` binds the report to the per-invariant
    verdicts (and any counterexamples) and is reproducible — two passes over the
    same invariants produce the same digest — so a governance regression is a
    diff in a mechanical artifact, not a re-run of a flaky check. The timestamp
    and audit linkage are metadata and deliberately excluded from the digest.
    """

    held: bool
    results: list[InvariantResult] = Field(default_factory=list)
    states_checked: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    claim_generator: str = "vincio"
    # Serialized under its field name; old dumps validate via the alias.
    content_hash: str = Field(
        default="", validation_alias=AliasChoices("content_hash", "content_sha256")
    )
    audit_entry_id: str | None = None
    audit_merkle_root: str | None = None

    @property
    def content_sha256(self) -> str:
        """Deprecated since 7.5 (removal in 8.0): read :attr:`content_hash`.

        Read-only alias kept for the rename runway; assignment goes through
        :attr:`content_hash`.
        """
        return self.content_hash

    @property
    def counterexamples(self) -> list[Counterexample]:
        """Every invariant's counterexample (empty when the report holds)."""
        return [r.counterexample for r in self.results if r.counterexample is not None]

    def digest_payload(self) -> str:
        """Canonical, reproducible bytes the content digest covers."""
        return json.dumps(
            {
                "held": self.held,
                "results": [
                    {
                        "id": r.id,
                        "category": r.category,
                        "held": r.held,
                        "domain_size": r.domain_size,
                        "counterexample": (
                            r.counterexample.assignment if r.counterexample else None
                        ),
                    }
                    for r in self.results
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        """SHA-256 content hash over :meth:`digest_payload` (the recorded binding)."""
        return hashlib.sha256(self.digest_payload().encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """Recompute the content digest and check it matches what was recorded."""
        return bool(self.content_hash) and self.content_hash == self.digest()

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dump of the report (for serialization / audit detail)."""
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------


class GovernanceVerifier:
    """Proves governance invariants by exhaustive bounded model checking.

    Construct it with a list of :class:`Invariant`\\ s (defaulting to the four
    platform invariants from :func:`default_invariants`) and, optionally, an
    :class:`~vincio.security.AuditLog` to record the result onto. :meth:`verify`
    enumerates every invariant's full state space, collects per-invariant verdicts
    and any minimal counterexample, and returns a content-hashed
    :class:`VerificationReport`. The pass is deterministic and offline.
    """

    def __init__(
        self,
        invariants: Sequence[Invariant] | None = None,
        *,
        audit_log: AuditLog | None = None,
        claim_generator: str | None = None,
    ) -> None:
        self.invariants: list[Invariant] = (
            list(invariants) if invariants is not None else default_invariants()
        )
        self.audit = audit_log
        self._claim_generator = claim_generator

    def _minimize(self, invariant: Invariant, assignment: Mapping[str, Any]) -> Counterexample:
        """Delta-minimize a violating assignment toward each variable's default.

        Greedily relaxes one variable at a time back to its benign baseline,
        keeping the change only while the violation persists. The result is a
        1-minimal counterexample: no single variable can be simplified without
        the property starting to hold again — the smallest witness the bounded
        search exposes.
        """
        current = dict(assignment)
        for variable in invariant.variables:
            if current[variable.name] == variable.default:
                continue
            trial = dict(current)
            trial[variable.name] = variable.default
            if not invariant.holds(trial):
                current = trial
        return Counterexample(
            invariant_id=invariant.id,
            category=invariant.category,
            assignment=current,
            explanation=invariant.explain(current),
        )

    def check_one(self, invariant: Invariant) -> InvariantResult:
        """Check a single invariant exhaustively over its state space.

        Returns ``held=True`` only after the predicate has been confirmed at every
        point (a proof over the modeled domain); on the first violation it stops,
        minimizes the witness, and reports the counterexample.
        """
        held = True
        checked = 0
        counterexample: Counterexample | None = None
        for assignment in invariant.assignments():
            checked += 1
            if not invariant.holds(assignment):
                held = False
                counterexample = self._minimize(invariant, assignment)
                break
        states_checked = invariant.domain_size if held else checked
        digest = hashlib.sha256(
            json.dumps(
                {
                    "id": invariant.id,
                    "statement": invariant.statement,
                    "category": invariant.category,
                    "domain_size": invariant.domain_size,
                    "held": held,
                    "counterexample": counterexample.assignment if counterexample else None,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:32]
        return InvariantResult(
            id=invariant.id,
            statement=invariant.statement,
            category=invariant.category,
            held=held,
            states_checked=states_checked,
            domain_size=invariant.domain_size,
            counterexample=counterexample,
            digest=digest,
        )

    def verify(self, *, record: bool = True) -> VerificationReport:
        """Run every invariant and return a content-hashed, audited report.

        When ``record`` and an audit log are set, the verdict lands on the
        hash-chained audit log as a ``governance_verification`` entry (``decision``
        ``allow`` when it holds, ``deny`` otherwise) carrying each invariant's
        verdict and any counterexample, so the proof is on the record.
        """
        import vincio

        results = [self.check_one(inv) for inv in self.invariants]
        held = all(r.held for r in results)
        report = VerificationReport(
            held=held,
            results=results,
            states_checked=sum(r.states_checked for r in results),
            claim_generator=self._claim_generator or f"vincio/{vincio.__version__}",
        )
        report.content_hash = report.digest()
        if record and self.audit is not None:
            entry = self.audit.record(
                "governance_verification",
                decision="allow" if held else "deny",
                details={
                    "held": held,
                    "invariants": len(results),
                    "states_checked": report.states_checked,
                    # frozen audit-detail key — external consumers bind to it.
                    "content_sha256": report.content_hash,
                    "verdicts": {r.category: r.held for r in results},
                    "counterexamples": [c.render() for c in report.counterexamples],
                },
            )
            report.audit_entry_id = entry.id
            report.audit_merkle_root = self.audit.merkle_root()
        return report


# ---------------------------------------------------------------------------
# The four platform invariants
# ---------------------------------------------------------------------------

# Representative trust labels, side-effect classes, and authority kinds — the
# complete typed alphabets the containment gate acts over.
_TAINTS = ("trusted", "untrusted", "quarantined")
_SIDE_EFFECTS = ("read", "pure", "write", "external")
_AUTHORITIES = ("trusted", "none", "capability", "approval")


def containment_invariant() -> Invariant:
    """`untrusted ⇒ no unapproved capability`, over the whole gate alphabet.

    Verifies that the injection-containment gate — the *same*
    :func:`~vincio.security.requires_authority` predicate
    :class:`~vincio.security.DualPlaneExecutor` runs — can never admit an
    *escalation*: an untrusted-tainted call to a side-effecting tool executing
    without a user-minted capability or an approval. The verifier proves the gate
    satisfies the :attr:`~vincio.security.ContainmentEvent.is_escalation`
    specification at every point of the (taint × side-effect × authority) space.
    """
    variables = (
        StateVariable("taint", _TAINTS),
        StateVariable("side_effects", _SIDE_EFFECTS),
        StateVariable("authority", _AUTHORITIES),
    )

    def gate_executes(taint: TrustLabel, side_effects: str, authority: str) -> bool:
        # The DualPlaneExecutor decision: a call that requires an authority but
        # carries none is blocked; everything else executes.
        if requires_authority(taint, side_effects) and authority not in AUTHORIZED:
            return False
        return True

    def predicate(state: Mapping[str, Any]) -> bool:
        taint = TrustLabel(state["taint"])
        side_effects = str(state["side_effects"])
        authority = str(state["authority"])
        executed = gate_executes(taint, side_effects, authority)
        # Specification: an executed call must never be an escalation.
        is_escalation = (
            executed
            and side_effects in {"write", "external"}
            and taint.is_tainted
            and authority not in AUTHORIZED
        )
        return not is_escalation

    def explain(state: Mapping[str, Any]) -> str:
        return (
            f"a {state['taint']} argument reached side-effecting tool class "
            f"{state['side_effects']!r} with authority {state['authority']!r} — "
            "an untrusted-tainted side effect executed without a capability or approval"
        )

    return Invariant(
        id="containment_no_unapproved_capability",
        statement=(
            "An untrusted-tainted argument never reaches a side-effecting tool "
            "without a user-minted capability or an explicit approval."
        ),
        category="containment",
        variables=variables,
        predicate=predicate,
        explain=explain,
    )


# Representative provider regions paired with their ground-truth jurisdiction
# (independent of the policy under test), plus the unknown / unresolvable cases.
_REGIONS: dict[str | None, str | None] = {
    None: None,
    "us-east-1": "us",
    "eu-west-1": "eu",
    "europe-west4": "eu",
    "ap-southeast-2": "apac",
    "on_prem": "on_prem",
    "zz-unknown-9": None,
}
_ALLOWED_SETS = ("eu", "us", "eu,on_prem")


def residency_invariant(*, deny_on_unknown: bool = True) -> Invariant:
    """In-jurisdiction egress refusal, over representative regions and policies.

    Verifies that a residency policy admits egress *only* to a provably
    in-jurisdiction region: for every (resolved region × allowed-set) point,
    ``ResidencyPolicy.check`` returning "allowed" implies the region's
    ground-truth jurisdiction is in the allowed set. With the fail-closed default
    (``deny_on_unknown=True``) an unresolvable region is always refused; passing
    ``deny_on_unknown=False`` models a fail-open misconfiguration, for which the
    verifier returns the unknown-region counterexample.
    """
    variables = (
        StateVariable("allowed", _ALLOWED_SETS),
        # ``values[0]`` is a benign in-jurisdiction region so minimization keeps
        # the adversarial region that actually triggers a violation.
        StateVariable("region", ("eu-west-1", None, "us-east-1", "on_prem", "zz-unknown-9")),
    )

    def predicate(state: Mapping[str, Any]) -> bool:
        region = state["region"]
        allowed = set(str(state["allowed"]).split(","))
        policy = ResidencyPolicy(
            allowed_regions=sorted(allowed),
            provider_regions=({"p": region} if region is not None else {}),
            deny_on_unknown=deny_on_unknown,
        )
        admitted = policy.check(provider="p", model=None) is None
        if not admitted:
            return True  # a refusal can never violate "admit ⇒ in-jurisdiction"
        true_jurisdiction = _REGIONS.get(region)
        return true_jurisdiction is not None and true_jurisdiction in allowed

    def explain(state: Mapping[str, Any]) -> str:
        region = state["region"]
        jurisdiction = _REGIONS.get(region)
        return (
            f"egress to region {region!r} (jurisdiction {jurisdiction!r}) was admitted "
            f"under an allowed set of {state['allowed']!r} it is not part of"
        )

    return Invariant(
        id="residency_in_jurisdiction_egress",
        statement=(
            "An enforced residency policy admits egress only to a provider region "
            "whose jurisdiction is in the allowed set; an unknown region is refused."
        ),
        category="residency",
        variables=variables,
        predicate=predicate,
        explain=explain,
    )


_LIMITS = (1.0, 10.0)
_SPENT = (0.0, 0.5, 0.999, 9.999, 100.0)
_PROJECTED = (0.0, 0.001, 0.5, 50.0)


def budget_invariant(admits: Callable[[float, float, float], bool] | None = None) -> Invariant:
    """A budget is a hard cap: an admitted run never overspends.

    Verifies the canonical hard-cap predicate (:func:`within_budget`, the gate
    behind the dollar / energy / carbon budgets) over a grid of (limit × accrued ×
    projected): every admitted run keeps the projected total strictly under the
    limit, and once the scope reaches its limit every further run is refused. Pass
    a different ``admits`` to model a weakened cap (e.g. one that ignores the
    projection); the verifier returns the over-budget counterexample.
    """
    decide = admits if admits is not None else (lambda s, p, lim: within_budget(s, p, lim))
    variables = (
        StateVariable("limit", _LIMITS),
        StateVariable("spent", _SPENT),
        StateVariable("projected", _PROJECTED),
    )

    def predicate(state: Mapping[str, Any]) -> bool:
        limit = float(state["limit"])
        spent = float(state["spent"])
        projected = float(state["projected"])
        admitted = decide(spent, projected, limit)
        if admitted:
            # An admitted run must leave the projected total strictly under the cap.
            return spent + projected < limit
        # A refusal is always sound under a hard cap.
        return True

    def explain(state: Mapping[str, Any]) -> str:
        spent = float(state["spent"])
        projected = float(state["projected"])
        limit = float(state["limit"])
        return (
            f"a run was admitted at spent={spent:g} + projected={projected:g} "
            f"= {spent + projected:g}, which reaches or exceeds the limit {limit:g}"
        )

    return Invariant(
        id="budget_hard_cap_never_overspends",
        statement=(
            "A budget is a hard cap: an admitted run keeps the projected total "
            "under the limit, and a scope at its limit refuses every further run."
        ),
        category="budget",
        variables=variables,
        predicate=predicate,
        explain=explain,
    )


_ID_SETS = ("", "a", "a|b", "a|b|c")
_TAMPERS = ("none", "add", "drop", "swap")


def erasure_invariant() -> Invariant:
    """An erasure proof binds to exactly the removed-id set.

    Verifies that :func:`~vincio.governance.verify_erasure_proof` accepts a proof
    if and only if its recorded removed-id set is intact: over a grid of
    (removed-id set × tamper), an untampered proof verifies and *any* mutation of
    the recorded ids — adding, dropping, or swapping one — breaks verification. A
    proof that still verified after tampering would be a counterexample to the
    content binding.
    """
    variables = (
        StateVariable("ids", _ID_SETS),
        StateVariable("tamper", _TAMPERS),
    )

    def predicate(state: Mapping[str, Any]) -> bool:
        ids = [i for i in str(state["ids"]).split("|") if i]
        tamper = str(state["tamper"])
        proof = build_erasure_proof("subject", {"chunks": list(ids)})
        if tamper == "add":
            proof.removed_ids = {"chunks": [*ids, "injected"]}
        elif tamper == "drop":
            proof.removed_ids = {"chunks": ids[:-1]} if ids else {"chunks": ["phantom"]}
        elif tamper == "swap":
            proof.removed_ids = {"chunks": [*ids[:-1], "swapped"]} if ids else {"chunks": ["swapped"]}
        verified = verify_erasure_proof(proof)
        # Specification: verifies iff untampered.
        return verified == (tamper == "none")

    def explain(state: Mapping[str, Any]) -> str:
        return (
            f"an erasure proof over ids {state['ids']!r} with tamper "
            f"{state['tamper']!r} did not verify exactly when its removed-id set was intact"
        )

    return Invariant(
        id="erasure_proof_content_bound",
        statement=(
            "An erasure proof verifies if and only if its recorded removed-id set "
            "is intact; any added, dropped, or swapped id breaks verification."
        ),
        category="erasure",
        variables=variables,
        predicate=predicate,
        explain=explain,
    )


def default_invariants() -> list[Invariant]:
    """The four platform governance invariants, fail-closed.

    Containment, residency (fail-closed), budget, and erasure — each bound to the
    shipped decision function and proven over its bounded typed state space. This
    is the set :class:`GovernanceVerifier` and
    :meth:`~vincio.core.app.ContextApp.verify_governance` check by default.
    """
    return [
        containment_invariant(),
        residency_invariant(deny_on_unknown=True),
        budget_invariant(),
        erasure_invariant(),
    ]
