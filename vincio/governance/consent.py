"""Consent & purpose modeling: bind data to a lawful basis.

Provable erasure answers *"forget this"*; consent modeling answers the prior
question *"were we ever allowed to keep it, and for what?"*. The
:class:`ConsentLedger` records, per data subject, which **purposes** their data
may be processed for and under which GDPR Article 6 **lawful basis**, with grant
and (revocable) expiry times. Access decisions and memory recall consult it, so a
withdrawn consent or a purpose mismatch is enforced in code — not left to a
downstream policy doc.

The ledger is deterministic and in-process. It optionally persists to the
metadata store (so consent survives a restart) and writes every grant / revoke /
denied check to the hash-chained audit log, on the same chain as the erasure
proofs that later honour a withdrawal.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.utils import new_id, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.audit import AuditLog
    from ..storage.base import MetadataStore

__all__ = [
    "Purpose",
    "LawfulBasis",
    "ConsentRecord",
    "ConsentDecision",
    "ConsentLedger",
]


class Purpose(StrEnum):
    """Why personal data is processed (GDPR Art. 5(1)(b) purpose limitation)."""

    SERVICE = "service"  # delivering the feature the user asked for
    PERSONALIZATION = "personalization"  # tailoring responses to the user
    ANALYTICS = "analytics"  # aggregate quality / usage measurement
    TRAINING = "training"  # using the data to improve / fine-tune a model
    MARKETING = "marketing"
    LEGAL = "legal"  # retention for a legal obligation


class LawfulBasis(StrEnum):
    """GDPR Article 6(1) lawful bases for processing."""

    CONSENT = "consent"  # 6(1)(a)
    CONTRACT = "contract"  # 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # 6(1)(c)
    VITAL_INTERESTS = "vital_interests"  # 6(1)(d)
    PUBLIC_TASK = "public_task"  # 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests"  # 6(1)(f)


class ConsentRecord(BaseModel):
    """One subject's consent for a set of purposes under one lawful basis."""

    id: str = Field(default_factory=lambda: new_id("consent"))
    subject_id: str
    purposes: list[Purpose] = Field(default_factory=list)
    lawful_basis: LawfulBasis = LawfulBasis.CONSENT
    granted_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    revoked: bool = False
    revoked_at: datetime | None = None
    source: str = ""  # where consent was captured (ui form, contract, …)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def active(self, *, now: datetime | None = None) -> bool:
        """True when the record is neither revoked nor expired."""
        if self.revoked:
            return False
        if self.expires_at is None:
            return True
        moment = now or utcnow()
        expires = self.expires_at
        if expires.tzinfo is None:
            from datetime import UTC

            expires = expires.replace(tzinfo=UTC)
        if moment.tzinfo is None:
            from datetime import UTC

            moment = moment.replace(tzinfo=UTC)
        return expires > moment


class ConsentDecision(BaseModel):
    """An explainable verdict for a purpose check (suitable for audit)."""

    allowed: bool
    subject_id: str = ""
    purpose: str = ""
    lawful_basis: str | None = None
    consent_id: str | None = None
    reason: str = ""


class ConsentLedger:
    """Records and checks consent, binding data to a purpose + lawful basis."""

    def __init__(
        self,
        *,
        store: MetadataStore | None = None,
        audit: AuditLog | None = None,
        # When no record exists for a subject, deny by default (consent must be
        # affirmative). Bases other than CONSENT (e.g. CONTRACT) are recorded
        # explicitly via :meth:`grant`.
        default_allow: bool = False,
    ) -> None:
        self.store = store
        self.audit = audit
        self.default_allow = default_allow
        self._records: dict[str, ConsentRecord] = {}
        self._by_subject: dict[str, list[str]] = {}
        if store is not None:
            self._load()

    # -- record management ---------------------------------------------------

    def grant(
        self,
        subject_id: str,
        purposes: list[Purpose | str],
        *,
        lawful_basis: LawfulBasis | str = LawfulBasis.CONSENT,
        expires_at: datetime | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ConsentRecord:
        """Record consent for ``subject_id`` over one or more purposes."""
        record = ConsentRecord(
            subject_id=subject_id,
            purposes=[Purpose(p) for p in purposes],
            lawful_basis=LawfulBasis(lawful_basis),
            expires_at=expires_at,
            source=source,
            metadata=metadata or {},
        )
        self._records[record.id] = record
        self._by_subject.setdefault(subject_id, []).append(record.id)
        self._persist(record)
        self._record_audit(
            "consent_grant",
            subject_id,
            details={
                "consent_id": record.id,
                "purposes": [p.value for p in record.purposes],
                "lawful_basis": record.lawful_basis.value,
            },
        )
        return record

    def revoke(self, subject_id: str, *, purpose: Purpose | str | None = None) -> int:
        """Revoke consent for a subject (optionally only for one purpose).

        Returns the number of records affected. A revoked record stays in the
        ledger as an immutable audit fact; :meth:`check` then denies.
        """
        target = Purpose(purpose) if purpose is not None else None
        affected = 0
        for record_id in self._by_subject.get(subject_id, []):
            record = self._records[record_id]
            if record.revoked:
                continue
            if target is None:
                # Revoke the whole record.
                record.revoked = True
                record.revoked_at = utcnow()
            elif target in record.purposes:
                # Withdraw just this purpose; revoke the record only when none
                # remain (other purposes on the same consent stay active).
                record.purposes = [p for p in record.purposes if p != target]
                if not record.purposes:
                    record.revoked = True
                record.revoked_at = utcnow()
            else:
                continue
            self._persist(record)
            affected += 1
        self._record_audit(
            "consent_revoke",
            subject_id,
            details={"purpose": target.value if target else "all", "records": affected},
        )
        return affected

    # -- checks --------------------------------------------------------------

    def check(self, subject_id: str, purpose: Purpose | str) -> ConsentDecision:
        """Decide whether ``subject_id`` may be processed for ``purpose``."""
        want = Purpose(purpose)
        for record_id in self._by_subject.get(subject_id, []):
            record = self._records[record_id]
            if want in record.purposes and record.active():
                return ConsentDecision(
                    allowed=True,
                    subject_id=subject_id,
                    purpose=want.value,
                    lawful_basis=record.lawful_basis.value,
                    consent_id=record.id,
                    reason=f"active {record.lawful_basis.value} consent for {want.value}",
                )
        decision = ConsentDecision(
            allowed=self.default_allow,
            subject_id=subject_id,
            purpose=want.value,
            reason=(
                "no active consent record" if not self.default_allow else "default-allow"
            ),
        )
        if not decision.allowed:
            self._record_audit(
                "consent_denied",
                subject_id,
                decision="deny",
                details={"purpose": want.value},
            )
        return decision

    def allows(self, subject_id: str, purpose: Purpose | str) -> bool:
        return self.check(subject_id, purpose).allowed

    def for_subject(self, subject_id: str) -> list[ConsentRecord]:
        return [self._records[r] for r in self._by_subject.get(subject_id, [])]

    def records(self) -> list[ConsentRecord]:
        return list(self._records.values())

    def to_dict(self) -> dict[str, Any]:
        return {rid: rec.model_dump(mode="json") for rid, rec in self._records.items()}

    # -- persistence / audit -------------------------------------------------

    def _persist(self, record: ConsentRecord) -> None:
        if self.store is None:
            return
        row = record.model_dump(mode="json")
        self.store.save("consent_records", row)

    def _load(self) -> None:
        assert self.store is not None  # noqa: S101 - _load runs only when a store is configured
        try:
            rows = self.store.query("consent_records", limit=10_000)
        except Exception:  # noqa: BLE001 - a store without the kind is simply empty
            return
        for row in rows:
            record = ConsentRecord.model_validate(row)
            self._records[record.id] = record
            self._by_subject.setdefault(record.subject_id, []).append(record.id)

    def _record_audit(
        self, action: str, subject_id: str, *, decision: str = "allow", details: dict[str, Any]
    ) -> None:
        if self.audit is None:
            return
        self.audit.record(action, decision=decision, resource=subject_id, details=details)
