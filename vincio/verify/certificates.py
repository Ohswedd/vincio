"""Proof-carrying answers.

A :class:`Certificate` is a typed, content-bound, offline-verifiable record of
what a deterministic verifier checked about an answer. Each :class:`Check` is a
re-derivable verdict from one kernel — ``verified`` (the kernel recomputed the
claim and it held), ``refuted`` (the kernel found a contradiction), or
``inapplicable`` (the kernel found nothing it could check). A
:class:`VerifiedAnswer` pairs a result with its certificate.

The certificate is **sound by construction**: a kernel may only emit ``verified``
when it actually recomputed the claim and the recomputation matched, so a wrong
answer the relevant kernel can see is *refuted*, never silently passed. The
certificate's status is **re-derived from its checks** on :meth:`Certificate.verify`
and bound into a content hash, so a tampered verdict — a status flipped to
``verified`` after the fact — is caught from the bytes alone, the same discipline
the governance verifier and the cross-org settlement artifacts hold.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..core.types import EvidenceItem
from ..core.utils import stable_hash

__all__ = [
    "CheckStatus",
    "Check",
    "CertificateStatus",
    "Certificate",
    "VerifiedAnswer",
    "VerificationContext",
    "ReasoningVerifier",
    "CompositeVerifier",
    "derive_status",
    "build_certificate",
    "canonical_subject",
]

CheckStatus = Literal["verified", "refuted", "inapplicable"]
CertificateStatus = Literal["verified", "refuted", "inapplicable"]


class Check(BaseModel):
    """One kernel's verdict on an answer.

    ``status`` is ``verified`` only when the kernel **recomputed** the claim and
    the recomputation matched, ``refuted`` when it found a concrete contradiction
    (the recomputation disagreed), and ``inapplicable`` when the kernel found no
    claim of its kind to check. ``detail`` explains the verdict and ``evidence``
    carries the recomputed values, so a refutation is actionable, not opaque.
    """

    name: str
    kind: str
    status: CheckStatus
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when the kernel positively confirmed the claim."""
        return self.status == "verified"

    @property
    def refuted(self) -> bool:
        """True when the kernel found a contradiction."""
        return self.status == "refuted"


class VerificationContext(BaseModel):
    """The grounding a verifier may consult while certifying an answer.

    Every field is optional; a kernel reads only what it needs and returns an
    ``inapplicable`` check when the grounding it requires is absent. ``evidence``
    grounds citation entailment, ``schema`` grounds structural conformance,
    ``constraints`` ground constraint satisfaction, ``statistical_claims`` ground
    the analytical kernels (trend / correlation / interval / forecast), ``now``
    anchors relative temporal claims, and ``facts`` carries named known values a
    kernel can cross-check an extracted claim against.
    """

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    evidence: list[EvidenceItem] = Field(default_factory=list)
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    constraints: list[Any] = Field(default_factory=list)
    statistical_claims: list[Any] = Field(default_factory=list)
    now: Any = None  # datetime | None
    facts: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


def canonical_subject(answer: Any) -> str:
    """Stable string surrogate of an answer for content-hashing.

    A string is taken verbatim; a Pydantic model or mapping is canonicalised via
    its dumped form, so the same answer always binds to the same subject hash.
    """
    if isinstance(answer, str):
        return answer
    if hasattr(answer, "model_dump"):
        return stable_hash(answer.model_dump(mode="json"), length=32)
    return stable_hash(answer, length=32)


def derive_status(checks: list[Check]) -> CertificateStatus:
    """Re-derive a certificate's overall status from its checks.

    ``refuted`` if any kernel found a contradiction (a single refutation sinks the
    answer), ``verified`` if at least one kernel positively confirmed a claim and
    none refuted, ``inapplicable`` if no kernel found anything to check.
    """
    if any(c.refuted for c in checks):
        return "refuted"
    if any(c.passed for c in checks):
        return "verified"
    return "inapplicable"


def build_certificate(
    answer: Any,
    checks: list[Check],
    *,
    kinds: list[str] | None = None,
    issuer: str = "vincio.verify",
) -> Certificate:
    """Assemble a content-bound :class:`Certificate` over ``checks``."""
    subject = canonical_subject(answer)
    cert = Certificate(
        subject_hash=stable_hash(subject, length=32),
        kinds=sorted(set(kinds or [c.kind for c in checks])),
        checks=checks,
        status=derive_status(checks),
        issuer=issuer,
    )
    cert.certificate_hash = cert._compute_hash()
    return cert


class Certificate(BaseModel):
    """A typed, content-bound, offline-verifiable proof over an answer.

    The certificate records exactly which deterministic kernels ran (``kinds``)
    and their verdicts (``checks``), with the overall ``status`` re-derivable from
    those checks. :meth:`verify` recomputes the content hash and re-derives the
    status from the recorded checks, so a tampered verdict is caught from the
    bytes alone — the certificate proves nothing it cannot reconstruct.
    """

    subject_hash: str
    kinds: list[str] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    status: CertificateStatus = "inapplicable"
    issuer: str = "vincio.verify"
    certificate_hash: str = ""

    def _compute_hash(self) -> str:
        return stable_hash(
            {
                "subject": self.subject_hash,
                "kinds": self.kinds,
                "status": self.status,
                "checks": [
                    {"name": c.name, "kind": c.kind, "status": c.status, "detail": c.detail}
                    for c in self.checks
                ],
                "issuer": self.issuer,
            },
            length=32,
        )

    @property
    def holds(self) -> bool:
        """True when the answer carries a positively-verified certificate."""
        return self.status == "verified"

    @property
    def refuted(self) -> bool:
        """True when a kernel positively refuted the answer."""
        return self.status == "refuted"

    @property
    def refutations(self) -> list[Check]:
        """The checks that found a contradiction."""
        return [c for c in self.checks if c.refuted]

    def verify(self) -> bool:
        """Recompute the hash and re-derive the status — offline, from the bytes.

        Returns ``False`` if the content hash no longer matches (an edited check
        or subject) **or** if the stored status disagrees with the status the
        recorded checks imply (a flipped verdict), so a re-sealed tamper is caught.
        """
        if self.status != derive_status(self.checks):
            return False
        return self.certificate_hash == self._compute_hash()

    def render(self) -> str:
        """One-line-per-check human rendering of the certificate."""
        head = f"certificate[{self.status}] over {self.subject_hash[:12]} ({', '.join(self.kinds)})"
        lines = [head]
        for c in self.checks:
            mark = {"verified": "✓", "refuted": "✗", "inapplicable": "·"}[c.status]
            lines.append(f"  {mark} {c.name}: {c.detail or c.status}")
        return "\n".join(lines)


class VerifiedAnswer(BaseModel):
    """An answer paired with the certificate a deterministic verifier produced.

    :attr:`holds` is true only when the certificate positively verified and the
    answer was not refused. When ``app.verify_reasoning`` drives self-correction,
    ``attempts`` counts the cycles and ``refused`` records that the orchestrator
    declined to emit an answer whose certificate did not check.
    """

    model_config = {"arbitrary_types_allowed": True}

    answer: Any = None
    certificate: Certificate
    attempts: int = 1
    refused: bool = False
    stopped_reason: str = ""  # verified | refuted | inapplicable | max_attempts | refused

    @property
    def holds(self) -> bool:
        """True when the answer is verified and was not refused."""
        return self.certificate.holds and not self.refused

    @property
    def text(self) -> str:
        """The answer as text (its ``str`` form for non-string answers)."""
        return self.answer if isinstance(self.answer, str) else str(self.answer)


@runtime_checkable
class ReasoningVerifier(Protocol):
    """A pluggable, deterministic checker that turns an answer into checks.

    A verifier declares a ``kind`` and implements :meth:`check`, returning zero or
    more :class:`Check`\\ s. It must be **sound**: a ``verified`` check is only
    emitted when the kernel recomputed the claim and it matched. A verifier that
    finds no claim of its kind returns an ``inapplicable`` check (or an empty list).
    """

    kind: str

    def check(self, answer: Any, context: VerificationContext) -> list[Check]: ...


class CompositeVerifier:
    """Runs an ordered set of verifiers and folds their checks into one certificate.

    The default verifier set behind ``app.verify_reasoning``; also usable directly.
    Each member sees the same :class:`VerificationContext`, and the resulting
    :class:`Certificate` is ``refuted`` if any member refuted, ``verified`` if any
    member verified and none refuted, and ``inapplicable`` if none found a claim.
    """

    kind = "composite"

    def __init__(self, verifiers: list[ReasoningVerifier]) -> None:
        self.verifiers = list(verifiers)

    def certify(self, answer: Any, context: VerificationContext | None = None) -> Certificate:
        """Produce a content-bound :class:`Certificate` over ``answer``."""
        ctx = context or VerificationContext()
        checks: list[Check] = []
        for verifier in self.verifiers:
            checks.extend(verifier.check(answer, ctx))
        return build_certificate(answer, checks, kinds=[v.kind for v in self.verifiers])
