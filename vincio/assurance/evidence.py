"""Evidence: the platform's existing verdicts, bound by hash to a sub-claim.

An :class:`Evidence` item discharges a :class:`~vincio.assurance.Claim` with a
verdict the platform **already emits** — an eval gate verdict, a
:class:`~vincio.governance.VerificationReport`, a reasoning
:class:`~vincio.verify.Certificate`, an audit-chain segment, an identity or
delegation chain, or an AI-BOM attestation. It never re-implements the check; it
captures whether the artifact currently *supports* the claim, binds the artifact's
own self-hash, and seals the pair into a content hash. So the whole assurance case
verifies offline: a flipped verdict, a tampered support, or a stale proof is caught
from the bytes alone, and a missing piece is pinpointed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import stable_hash, utcnow

__all__ = ["Evidence", "EVIDENCE_KINDS"]

# The evidence kinds the platform's existing verdicts map onto. ``external`` is the
# escape hatch for a verdict produced outside this taxonomy (still bound by hash).
EVIDENCE_KINDS = (
    "eval_gate",
    "governance_proof",
    "reasoning_certificate",
    "audit_segment",
    "identity_chain",
    "sbom",
    "external",
)


def _artifact_hash(artifact: Any) -> str:
    """The artifact's own self-hash when it exposes one, else a content hash.

    Prefers a recorded self-hash (a certificate hash, a report's content digest, an
    audit Merkle root) so the binding tracks the artifact's own identity; falls back
    to hashing the artifact's serialized form so any object can be bound.
    """
    for attr in (
        "certificate_hash",
        "content_sha256",
        "content_hash",
        "evidence_hash",
        "result_hash",
        "library_hash",
        "entry_hash",
    ):
        value = getattr(artifact, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(artifact, BaseModel):
        return stable_hash(artifact.model_dump(mode="json"), length=32)
    return stable_hash(artifact, length=32)


class Evidence(BaseModel):
    """A platform verdict bound by hash to discharge one sub-claim.

    ``supports`` is the verdict captured at bind time (the gate passed, the report
    held, the certificate verified); ``source_hash`` is the artifact's own self-hash;
    ``horizon_days`` is the freshness window after which the proof expires. The pair
    is sealed into :attr:`evidence_hash`, so :meth:`verify` re-derives integrity from
    the bytes and :meth:`holds` adds the freshness and support checks.
    """

    kind: str = "external"
    label: str = ""
    supports: bool = False
    source_hash: str = ""
    detail: str = ""
    horizon_days: float | None = None
    recorded_at: datetime = Field(default_factory=utcnow)
    evidence_hash: str = ""

    def _facts(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "supports": self.supports,
            "source_hash": self.source_hash,
            "detail": self.detail,
            "horizon_days": self.horizon_days,
            "recorded_at": self.recorded_at.isoformat(),
        }

    def seal(self) -> Evidence:
        """Bind the captured verdict and source into a content hash."""
        self.evidence_hash = stable_hash(self._facts(), length=32)
        return self

    def verify(self) -> bool:
        """Recompute the content hash — catches a flipped support or edited source."""
        return bool(self.evidence_hash) and self.evidence_hash == stable_hash(
            self._facts(), length=32
        )

    def is_fresh(self, *, as_of: datetime | None = None) -> bool:
        """Whether the proof is still within its freshness horizon at ``as_of``."""
        if self.horizon_days is None:
            return True
        now = as_of or utcnow()
        return now <= self.recorded_at + timedelta(days=self.horizon_days)

    def holds(self, *, as_of: datetime | None = None) -> bool:
        """A held evidence item is intact, supportive, and fresh."""
        return self.verify() and self.supports and self.is_fresh(as_of=as_of)

    # -- binders: the platform's existing verdicts ------------------------------

    @classmethod
    def _bind(
        cls,
        kind: str,
        *,
        supports: bool,
        source_hash: str,
        label: str,
        detail: str,
        horizon_days: float | None,
        recorded_at: datetime | None,
    ) -> Evidence:
        return cls(
            kind=kind,
            label=label,
            supports=bool(supports),
            source_hash=source_hash,
            detail=detail,
            horizon_days=horizon_days,
            recorded_at=recorded_at or utcnow(),
        ).seal()

    @classmethod
    def from_gate(
        cls,
        verdict: Any,
        *,
        label: str = "eval gate",
        detail: str = "",
        horizon_days: float | None = None,
        recorded_at: datetime | None = None,
    ) -> Evidence:
        """Bind an eval / no-regression gate verdict (a ``CanaryVerdict`` or bool).

        Supports the claim when the gate **passed**.
        """
        passed = bool(getattr(verdict, "passed", verdict))
        reason = getattr(verdict, "reason", "") or ""
        return cls._bind(
            "eval_gate",
            supports=passed,
            source_hash=_artifact_hash(verdict),
            label=label,
            detail=detail or reason,
            horizon_days=horizon_days,
            recorded_at=recorded_at,
        )

    @classmethod
    def from_governance(
        cls,
        report: Any,
        *,
        label: str = "governance verifier",
        detail: str = "",
        horizon_days: float | None = None,
        recorded_at: datetime | None = None,
    ) -> Evidence:
        """Bind a :class:`~vincio.governance.VerificationReport`.

        Supports the claim only when the report **held** and re-verifies offline.
        """
        verify = getattr(report, "verify", None)
        intact = bool(verify()) if callable(verify) else True
        supports = intact and bool(getattr(report, "held", False))
        return cls._bind(
            "governance_proof",
            supports=supports,
            source_hash=_artifact_hash(report),
            label=label,
            detail=detail,
            horizon_days=horizon_days,
            recorded_at=recorded_at,
        )

    @classmethod
    def from_certificate(
        cls,
        certificate: Any,
        *,
        label: str = "reasoning certificate",
        detail: str = "",
        horizon_days: float | None = None,
        recorded_at: datetime | None = None,
    ) -> Evidence:
        """Bind a reasoning :class:`~vincio.verify.Certificate`.

        Supports the claim only when the certificate is ``verified`` and re-derives
        its verdict offline (a refuted or tampered certificate does not support).
        """
        verify = getattr(certificate, "verify", None)
        intact = bool(verify()) if callable(verify) else True
        status = getattr(certificate, "status", "")
        supports = intact and status == "verified"
        return cls._bind(
            "reasoning_certificate",
            supports=supports,
            source_hash=_artifact_hash(certificate),
            label=label,
            detail=detail or f"status={status}",
            horizon_days=horizon_days,
            recorded_at=recorded_at,
        )

    @classmethod
    def from_audit(
        cls,
        log: Any,
        *,
        label: str = "audit chain",
        detail: str = "",
        horizon_days: float | None = None,
        recorded_at: datetime | None = None,
    ) -> Evidence:
        """Bind an audit-chain segment (an ``AuditLog`` or a verdict bool).

        Supports the claim only when the hash-linked chain **verifies**; the source
        is bound to the chain's Merkle root so a later tamper changes the binding.
        """
        verify_chain = getattr(log, "verify_chain", None)
        intact = bool(verify_chain()) if callable(verify_chain) else bool(log)
        root = getattr(log, "merkle_root", None)
        source = root() if callable(root) else _artifact_hash(log)
        return cls._bind(
            "audit_segment",
            supports=intact,
            source_hash=source,
            label=label,
            detail=detail,
            horizon_days=horizon_days,
            recorded_at=recorded_at,
        )

    @classmethod
    def from_identity(
        cls,
        verification: Any,
        *,
        label: str = "identity / delegation chain",
        detail: str = "",
        horizon_days: float | None = None,
        recorded_at: datetime | None = None,
    ) -> Evidence:
        """Bind an identity or delegation-chain verification (has ``.valid``)."""
        supports = bool(getattr(verification, "valid", verification))
        reason = getattr(verification, "reason", "") or ""
        return cls._bind(
            "identity_chain",
            supports=supports,
            source_hash=_artifact_hash(verification),
            label=label,
            detail=detail or reason,
            horizon_days=horizon_days,
            recorded_at=recorded_at,
        )

    @classmethod
    def from_sbom(
        cls,
        aibom: Any,
        *,
        artifacts: dict[str, Any] | None = None,
        label: str = "AI-BOM / provenance",
        detail: str = "",
        horizon_days: float | None = None,
        recorded_at: datetime | None = None,
    ) -> Evidence:
        """Bind an :class:`~vincio.governance.AIBOM` provenance attestation.

        Supports the claim when the bill of materials lists components and every
        component carrying a recorded hash verifies against ``artifacts``.
        """
        components = getattr(aibom, "components", []) or []
        verify_all = getattr(aibom, "verify_all", None)
        checks = verify_all(artifacts) if callable(verify_all) else {}
        supports = bool(components) and all(checks.values())
        return cls._bind(
            "sbom",
            supports=supports,
            source_hash=_artifact_hash(aibom),
            label=label,
            detail=detail or f"{len(components)} component(s)",
            horizon_days=horizon_days,
            recorded_at=recorded_at,
        )

    @classmethod
    def asserted(
        cls,
        supports: bool,
        *,
        label: str,
        detail: str = "",
        source_hash: str = "",
        horizon_days: float | None = None,
        recorded_at: datetime | None = None,
    ) -> Evidence:
        """Bind an external verdict produced outside the platform's taxonomy."""
        return cls._bind(
            "external",
            supports=supports,
            source_hash=source_hash or stable_hash({"label": label, "detail": detail}, length=32),
            label=label,
            detail=detail,
            horizon_days=horizon_days,
            recorded_at=recorded_at,
        )
