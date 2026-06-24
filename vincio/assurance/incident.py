"""Incident: a signed production failure tied to the sub-claim it falsified.

An :class:`Incident` closes the loop from a production failure back into a stronger
safety argument. It names the :class:`~vincio.assurance.Claim` the failure
falsified and the evidence kinds that must be supplied before the case can
re-validate; :meth:`vincio.assurance.AssuranceCase.learn_from` folds it in as a
remediation sub-claim. It is content-bound and signable, so an operator verifies
from the bytes which claim a reported incident actually falsified.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import stable_hash, utcnow

__all__ = ["Incident"]


class Incident(BaseModel):
    """A signed observation that a sub-claim no longer holds in production."""

    id: str
    description: str = ""
    falsified_claim: str = ""
    severity: str = "high"
    required_evidence: list[str] = Field(default_factory=list)
    observed_at: datetime = Field(default_factory=utcnow)
    incident_hash: str = ""
    signature: str = ""
    key_id: str = ""

    def _facts(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "falsified_claim": self.falsified_claim,
            "severity": self.severity,
            "required_evidence": sorted(self.required_evidence),
            "observed_at": self.observed_at.isoformat(),
        }

    def seal(self) -> Incident:
        self.incident_hash = stable_hash(self._facts(), length=32)
        return self

    def verify(self) -> bool:
        """Recompute the incident hash — catches a re-pointed or edited report."""
        return bool(self.incident_hash) and self.incident_hash == stable_hash(
            self._facts(), length=32
        )

    def sign(self, signer: Any) -> Incident:
        if not self.incident_hash:
            self.seal()
        self.signature = signer.sign(self.incident_hash)
        self.key_id = getattr(signer, "key_id", "")
        return self

    def verify_signature(self, verifier: Any) -> bool:
        if not self.verify() or not self.signature:
            return False
        return bool(verifier.verify(self.incident_hash, self.signature))
