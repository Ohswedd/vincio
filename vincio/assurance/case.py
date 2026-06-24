"""The assurance-case argument tree (GSN / CAE) and its continuous re-check.

A :class:`Claim` is an argument node: a top goal (*this app is fit for purpose X
under context Y*) decomposed into sub-claims, each leaf **discharged by evidence
the platform already emits**. An :class:`AssuranceCase` binds the whole tree into a
content hash and signs it; :meth:`AssuranceCase.check` re-evaluates it against the
current evidence into an :class:`AssuranceReport`, pinpointing every claim whose
evidence is missing, stale, or falsified. :func:`assurance_regression_gate` turns a
falsified claim into a build failure — assurance as a living, gated invariant.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import AssuranceError
from ..core.utils import slugify, stable_hash, utcnow
from .evidence import Evidence

__all__ = [
    "Claim",
    "ClaimStatus",
    "AssuranceReport",
    "AssuranceCase",
    "assurance_regression_gate",
]


class Claim(BaseModel):
    """A node in the assurance argument: a statement, its decomposition, its evidence.

    A leaf claim (no ``subclaims``) holds when it carries at least one held
    :class:`~vincio.assurance.Evidence` item and every kind it lists in
    ``required_evidence`` is present and holding. A parent claim holds when its
    ``strategy`` over its children holds (``all`` — every child; ``any`` — at least
    one) and its own evidence, if any, holds.
    """

    id: str
    statement: str
    context: str = ""
    strategy: Literal["all", "any"] = "all"
    subclaims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.subclaims

    def _facts(self) -> dict[str, Any]:
        """The structural facts a case hash binds (evidence by its sealed hash)."""
        return {
            "id": self.id,
            "statement": self.statement,
            "context": self.context,
            "strategy": self.strategy,
            "required_evidence": sorted(self.required_evidence),
            "evidence": sorted(e.evidence_hash for e in self.evidence),
            "subclaims": [c._facts() for c in self.subclaims],
        }

    def find(self, claim_id: str) -> Claim | None:
        """Locate a claim by id anywhere in this subtree."""
        if self.id == claim_id:
            return self
        for child in self.subclaims:
            found = child.find(claim_id)
            if found is not None:
                return found
        return None

    def evaluate(self, *, as_of: datetime | None = None) -> ClaimStatus:
        """Re-derive this claim's discharge status from the current evidence."""
        children = [c.evaluate(as_of=as_of) for c in self.subclaims]

        present_kinds = {e.kind for e in self.evidence}
        missing = [k for k in self.required_evidence if k not in present_kinds]
        discharged_by: list[str] = []
        stale: list[str] = []
        falsified: list[str] = []
        for ev in self.evidence:
            if not ev.verify() or not ev.supports:
                falsified.append(ev.kind)
            elif not ev.is_fresh(as_of=as_of):
                stale.append(ev.kind)
            else:
                discharged_by.append(ev.kind)
        evidence_ok = not missing and not stale and not falsified

        if self.is_leaf:
            # A leaf must rest on at least one held piece of evidence (or, if it
            # only *requires* kinds, on those being present and holding).
            has_support = bool(discharged_by) or (
                bool(self.required_evidence) and not missing and not stale and not falsified
            )
            holds = has_support and evidence_ok
            reason = "" if holds else _leaf_reason(missing, stale, falsified, has_support)
        else:
            if self.strategy == "any":
                children_ok = any(c.holds for c in children)
            else:
                children_ok = all(c.holds for c in children)
            holds = children_ok and evidence_ok
            reason = (
                "" if holds else _parent_reason(self.strategy, children, missing, stale, falsified)
            )

        return ClaimStatus(
            id=self.id,
            statement=self.statement,
            holds=holds,
            reason=reason,
            discharged_by=discharged_by,
            missing=missing,
            stale=stale,
            falsified=falsified,
            children=children,
        )


# Forward-ref resolution for the self-referential ``subclaims`` field.
Claim.model_rebuild()


def _leaf_reason(
    missing: list[str], stale: list[str], falsified: list[str], has_support: bool
) -> str:
    if falsified:
        return f"evidence falsified: {', '.join(sorted(set(falsified)))}"
    if stale:
        return f"evidence stale: {', '.join(sorted(set(stale)))}"
    if missing:
        return f"evidence missing: {', '.join(missing)}"
    if not has_support:
        return "no supporting evidence"
    return "undischarged"


def _parent_reason(
    strategy: str,
    children: list[ClaimStatus],
    missing: list[str],
    stale: list[str],
    falsified: list[str],
) -> str:
    failing = [c.id for c in children if not c.holds]
    parts: list[str] = []
    if failing:
        word = "all of" if strategy == "all" else "every"
        parts.append(f"sub-claim(s) failed ({word}: {', '.join(failing)})")
    if falsified:
        parts.append(f"own evidence falsified: {', '.join(sorted(set(falsified)))}")
    if stale:
        parts.append(f"own evidence stale: {', '.join(sorted(set(stale)))}")
    if missing:
        parts.append(f"own evidence missing: {', '.join(missing)}")
    return "; ".join(parts) or "undischarged"


class ClaimStatus(BaseModel):
    """The re-derived verdict for one claim and its subtree."""

    id: str
    statement: str
    holds: bool
    reason: str = ""
    discharged_by: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    falsified: list[str] = Field(default_factory=list)
    children: list[ClaimStatus] = Field(default_factory=list)

    def walk(self) -> list[ClaimStatus]:
        """This status and every descendant, depth-first."""
        out = [self]
        for child in self.children:
            out.extend(child.walk())
        return out


ClaimStatus.model_rebuild()


class AssuranceReport(BaseModel):
    """The content-bound outcome of re-checking a case against current evidence."""

    subject: str = ""
    statement: str = ""
    holds: bool = False
    as_of: datetime = Field(default_factory=utcnow)
    root: ClaimStatus
    missing: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    falsified: list[str] = Field(default_factory=list)
    failing_claims: list[str] = Field(default_factory=list)
    report_hash: str = ""

    def _facts(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "statement": self.statement,
            "holds": self.holds,
            "as_of": self.as_of.isoformat(),
            "missing": sorted(self.missing),
            "stale": sorted(self.stale),
            "falsified": sorted(self.falsified),
            "failing_claims": sorted(self.failing_claims),
            "root": self.root.model_dump(mode="json"),
        }

    def seal(self) -> AssuranceReport:
        self.report_hash = stable_hash(self._facts(), length=32)
        return self

    def verify(self) -> bool:
        """Recompute the report hash — catches a verdict edited after the fact."""
        return bool(self.report_hash) and self.report_hash == stable_hash(self._facts(), length=32)

    def status_of(self, claim_id: str) -> ClaimStatus | None:
        for status in self.root.walk():
            if status.id == claim_id:
                return status
        return None


class AssuranceCase(BaseModel):
    """A signed, content-bound assurance argument the platform keeps continuously valid."""

    subject: str = ""
    goal: Claim
    created_at: datetime = Field(default_factory=utcnow)
    incidents: list[Any] = Field(default_factory=list)
    case_hash: str = ""
    signature: str = ""
    key_id: str = ""

    def _facts(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "created_at": self.created_at.isoformat(),
            "goal": self.goal._facts(),
            "incidents": sorted(getattr(i, "incident_hash", "") for i in self.incidents),
        }

    def seal(self) -> AssuranceCase:
        """Bind the whole argument tree (and any learned incidents) into a hash."""
        self.case_hash = stable_hash(self._facts(), length=32)
        return self

    def verify(self) -> bool:
        """Recompute the case hash — catches a tampered argument tree."""
        return bool(self.case_hash) and self.case_hash == stable_hash(self._facts(), length=32)

    def sign(self, signer: Any) -> AssuranceCase:
        """Sign the sealed case with a chain signer (binds it to a key / DID)."""
        if not self.case_hash:
            self.seal()
        self.signature = signer.sign(self.case_hash)
        self.key_id = getattr(signer, "key_id", "")
        return self

    def verify_signature(self, verifier: Any) -> bool:
        """Verify the signature over the (re-verified) case hash."""
        if not self.verify() or not self.signature:
            return False
        return bool(verifier.verify(self.case_hash, self.signature))

    def check(self, *, as_of: datetime | None = None) -> AssuranceReport:
        """Re-evaluate the case against the current evidence into a sealed report."""
        if not self.verify():
            raise AssuranceError(
                "assurance case failed integrity check (tampered argument tree); "
                "rebuild or re-seal it before checking"
            )
        now = as_of or utcnow()
        root = self.goal.evaluate(as_of=now)
        statuses = root.walk()
        missing = sorted({f"{s.id}:{k}" for s in statuses for k in s.missing})
        stale = sorted({f"{s.id}:{k}" for s in statuses for k in s.stale})
        falsified = sorted({f"{s.id}:{k}" for s in statuses for k in s.falsified})
        failing = [s.id for s in statuses if not s.holds]
        return AssuranceReport(
            subject=self.subject,
            statement=self.goal.statement,
            holds=root.holds,
            as_of=now,
            root=root,
            missing=missing,
            stale=stale,
            falsified=falsified,
            failing_claims=failing,
        ).seal()

    def holds(self, *, as_of: datetime | None = None) -> bool:
        """Whether the top claim is currently discharged."""
        return self.check(as_of=as_of).holds

    def discharge(self, claim_id: str, *evidence: Evidence) -> AssuranceCase:
        """Attach evidence to a claim (e.g. to close a learned remediation) and re-seal.

        Raises :class:`~vincio.core.errors.AssuranceError` if the claim is unknown.
        """
        target = self.goal.find(claim_id)
        if target is None:
            raise AssuranceError(f"no claim {claim_id!r} in this assurance case")
        target.evidence.extend(e if e.evidence_hash else e.seal() for e in evidence)
        self.signature = ""  # any prior signature no longer binds the changed tree
        self.key_id = ""
        return self.seal()

    def learn_from(self, incident: Any) -> AssuranceCase:
        """Fold an incident in: demand fresh evidence on the claim it falsified.

        The case **learns** — a remediation sub-claim is added under the falsified
        claim, requiring new evidence of the kinds the incident demands before the
        case can re-validate. Raises if the incident names a claim not in the case.
        """
        claim_id = getattr(incident, "falsified_claim", "")
        target = self.goal.find(claim_id)
        if target is None:
            raise AssuranceError(
                f"incident references claim {claim_id!r} which is not in this case"
            )
        required = list(getattr(incident, "required_evidence", []) or ["eval_gate"])
        rid = f"{claim_id}-remediation-{slugify(getattr(incident, 'id', '') or 'incident')}"
        if target.find(rid) is None:
            target.subclaims.append(
                Claim(
                    id=rid,
                    statement=(
                        f"Incident '{getattr(incident, 'id', 'incident')}' is "
                        "remediated and cannot recur"
                    ),
                    context=getattr(incident, "description", ""),
                    required_evidence=required,
                )
            )
        self.incidents.append(incident)
        self.signature = ""
        self.key_id = ""
        return self.seal()


def assurance_regression_gate(before: AssuranceReport, after: AssuranceReport) -> tuple[bool, str]:
    """Block a build when a previously-discharged claim is no longer discharged.

    The assurance analogue of the no-regression gate a prompt or policy deploy
    clears: a claim that **held** in ``before`` but does not in ``after`` — its
    evidence now falsified, stale, or missing — fails the build. Returns
    ``(passed, reason)``; passes only when the after-case holds overall and no
    previously-holding claim regressed.
    """
    before_holds = {s.id: s.holds for s in before.root.walk()}
    regressed: list[str] = []
    for status in after.root.walk():
        if before_holds.get(status.id) and not status.holds:
            why = status.reason or "no longer discharged"
            regressed.append(f"{status.id} ({why})")
    if regressed:
        return False, "assurance regression: " + "; ".join(regressed)
    if not after.holds:
        return False, (
            "assurance case does not hold: "
            + (", ".join(after.failing_claims) or "top claim undischarged")
        )
    return True, "no assurance regression"
