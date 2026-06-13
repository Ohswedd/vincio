"""Audit logging and retention.

Records who requested what, which context/sources/tools were used, what was
returned, and what memory was written. Entries are append-only JSONL with an
integrity hash chain so tampering is detectable.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.utils import new_id, stable_hash, to_jsonable, utcnow

__all__ = [
    "AuditEntry",
    "AuditLog",
    "ChainVerification",
    "RetentionPolicy",
    "apply_retention",
    "verify_audit_file",
]


class AuditEntry(BaseModel):
    id: str = Field(default_factory=lambda: new_id("audit"))
    timestamp: datetime = Field(default_factory=utcnow)
    action: str  # run | retrieval | tool_call | memory_write | access_decision | output
    user_id: str | None = None
    tenant_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    resource: str | None = None
    decision: str | None = None  # allow | deny
    details: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        return stable_hash(
            {
                "id": self.id,
                "timestamp": self.timestamp.isoformat(),
                "action": self.action,
                "user_id": self.user_id,
                "tenant_id": self.tenant_id,
                "run_id": self.run_id,
                "resource": self.resource,
                "decision": self.decision,
                "details": to_jsonable(self.details),
                "prev_hash": self.prev_hash,
            },
            length=32,
        )


class AuditLog:
    """Append-only audit log. ``directory=None`` keeps entries in memory only."""

    def __init__(self, directory: str | Path | None = ".vincio/audit") -> None:
        self.directory = Path(directory) if directory else None
        self.entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._last_hash = ""

    @property
    def path(self) -> Path | None:
        return self.directory / "audit.jsonl" if self.directory else None

    def record(self, action: str, **fields: Any) -> AuditEntry:
        details = fields.pop("details", {})
        entry = AuditEntry(action=action, details=details, **fields)
        with self._lock:
            entry.prev_hash = self._last_hash
            entry.entry_hash = entry.compute_hash()
            self._last_hash = entry.entry_hash
            self.entries.append(entry)
            if self.path is not None:
                self.directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(to_jsonable(entry.model_dump(mode="json"))) + "\n")
        return entry

    def verify_chain(self) -> bool:
        """Validate the integrity hash chain over in-memory entries."""
        previous = ""
        for entry in self.entries:
            if entry.prev_hash != previous or entry.entry_hash != entry.compute_hash():
                return False
            previous = entry.entry_hash
        return True

    def verify_file(self) -> ChainVerification:
        """Re-read the persisted JSONL and verify its hash chain offline.

        Unlike :meth:`verify_chain` (in-memory only), this detects tampering of
        the on-disk log after a process restart — a row edited, inserted, or
        deleted breaks the chain at a pinpointed line. In-memory logs
        (``directory=None``) verify as intact with zero entries.
        """
        if self.path is None:
            return ChainVerification(intact=True, entries=0)
        return verify_audit_file(self.path)

    def query(
        self,
        *,
        action: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        results = [
            entry
            for entry in self.entries
            if (action is None or entry.action == action)
            and (user_id is None or entry.user_id == user_id)
            and (tenant_id is None or entry.tenant_id == tenant_id)
            and (run_id is None or entry.run_id == run_id)
        ]
        return results[-limit:]


class ChainVerification(BaseModel):
    """Result of verifying a persisted audit chain."""

    intact: bool
    entries: int
    broken_at: int | None = None  # 1-based line number where the chain broke
    reason: str | None = None


def verify_audit_file(jsonl_path: str | Path) -> ChainVerification:
    """Verify the integrity hash chain of a persisted audit JSONL file.

    Recomputes each entry's hash from its content and checks that every
    ``prev_hash`` links to the prior ``entry_hash``, so any edit, reorder,
    insert, or delete is detected and localized. A missing file verifies as
    intact with zero entries (nothing has been tampered with).
    """
    path = Path(jsonl_path)
    if not path.is_file():
        return ChainVerification(intact=True, entries=0)
    previous = ""
    count = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = AuditEntry.model_validate_json(line)
            except (ValueError, TypeError) as exc:
                return ChainVerification(
                    intact=False, entries=count, broken_at=lineno, reason=f"unparseable: {exc}"
                )
            count += 1
            if entry.prev_hash != previous:
                return ChainVerification(
                    intact=False, entries=count, broken_at=lineno, reason="prev_hash mismatch"
                )
            if entry.entry_hash != entry.compute_hash():
                return ChainVerification(
                    intact=False, entries=count, broken_at=lineno, reason="entry_hash mismatch"
                )
            previous = entry.entry_hash
    return ChainVerification(intact=True, entries=count)


class RetentionPolicy(BaseModel):
    """Configurable retention per artifact type, in days.
    ``None`` means keep forever."""

    traces: int | None = None
    prompts: int | None = None
    outputs: int | None = None
    evidence: int | None = None
    memory: int | None = None
    eval_results: int | None = None
    audit: int | None = None


def apply_retention(
    jsonl_path: str | Path,
    *,
    max_age_days: int,
    timestamp_field: str = "timestamp",
) -> int:
    """Drop JSONL records older than *max_age_days*. Returns removed count."""
    path = Path(jsonl_path)
    if not path.is_file() or max_age_days is None:
        return 0
    cutoff = utcnow() - timedelta(days=max_age_days)
    kept_lines: list[str] = []
    removed = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                raw_ts = record.get(timestamp_field) or record.get("start_time") or record.get("created_at")
                timestamp = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    from datetime import UTC

                    timestamp = timestamp.replace(tzinfo=UTC)
            except (json.JSONDecodeError, ValueError, TypeError):
                kept_lines.append(line)
                continue
            if timestamp < cutoff:
                removed += 1
            else:
                kept_lines.append(line)
    if removed:
        path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return removed
