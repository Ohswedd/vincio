"""CertificationReport: a portable, offline-verifiable production certification.

``app.certify(case)`` emits a :class:`CertificationReport` — the assurance case,
its discharged evidence verdict, the residual risks, and the build provenance
(``vincio`` version, AI-BOM / SBOM, SLSA note) — signed with the app's identity. A
downstream operator or auditor checks it **from the bytes**: :meth:`verify`
recomputes the report hash and re-runs the case's own check, so a report claiming
``certified`` over a case that does not hold is caught.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import stable_hash, utcnow
from .case import AssuranceCase, AssuranceReport

__all__ = ["CertificationReport", "certify"]


class CertificationReport(BaseModel):
    """The signed, content-bound certificate that an app is fit for production."""

    subject: str = ""
    statement: str = ""
    certified: bool = False
    case: AssuranceCase
    assurance: AssuranceReport
    residual_risks: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    issued_at: datetime = Field(default_factory=utcnow)
    report_hash: str = ""
    signature: str = ""
    key_id: str = ""

    def _facts(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "statement": self.statement,
            "certified": self.certified,
            "case_hash": self.case.case_hash,
            "assurance_hash": self.assurance.report_hash,
            "residual_risks": sorted(self.residual_risks),
            "provenance": self.provenance,
            "issued_at": self.issued_at.isoformat(),
        }

    def seal(self) -> CertificationReport:
        self.report_hash = stable_hash(self._facts(), length=32)
        return self

    def verify(self, *, as_of: datetime | None = None) -> bool:
        """Re-derive the certification from the bytes alone.

        Recomputes the report hash, re-verifies the embedded assurance report and
        the case integrity, re-runs the case's check against its evidence, and
        confirms ``certified`` matches whether the case actually holds — so a
        report certifying a case that does not hold is caught offline. The check
        re-runs as of the time the report was issued (``self.assurance.as_of``)
        unless an explicit ``as_of`` is passed — pass ``as_of`` to test whether the
        certification still holds at a later date (a stale proof then expires).
        """
        if not self.report_hash or self.report_hash != stable_hash(self._facts(), length=32):
            return False
        if not self.assurance.verify() or not self.case.verify():
            return False
        recheck = self.case.check(as_of=as_of or self.assurance.as_of)
        if recheck.holds != self.assurance.holds:
            return False
        return self.certified == recheck.holds

    def verify_signature(self, verifier: Any) -> bool:
        if not self.signature or self.report_hash != stable_hash(self._facts(), length=32):
            return False
        return bool(verifier.verify(self.report_hash, self.signature))

    def sign(self, signer: Any) -> CertificationReport:
        if not self.report_hash:
            self.seal()
        self.signature = signer.sign(self.report_hash)
        self.key_id = getattr(signer, "key_id", "")
        return self

    def to_json(self, *, indent: int = 2) -> str:
        import json

        return json.dumps(self.model_dump(mode="json"), indent=indent, default=str)


def certify(
    case: AssuranceCase,
    *,
    signer: Any | None = None,
    residual_risks: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    as_of: datetime | None = None,
) -> CertificationReport:
    """Build a :class:`CertificationReport` from a checked assurance case.

    The free-function form of :meth:`vincio.ContextApp.certify` for use without an
    app: it checks the case, records the residual risks (the failing claims when the
    case does not hold, plus any passed in), stamps the provenance, and signs the
    report when a ``signer`` is given.
    """
    report = case.check(as_of=as_of)
    risks = list(residual_risks or [])
    if not report.holds:
        risks.extend(f"undischarged claim: {cid}" for cid in report.failing_claims)
    cert = CertificationReport(
        subject=case.subject,
        statement=case.goal.statement,
        certified=report.holds,
        case=case,
        assurance=report,
        residual_risks=sorted(set(risks)),
        provenance=provenance or {},
    ).seal()
    if signer is not None:
        cert.sign(signer)
    return cert
