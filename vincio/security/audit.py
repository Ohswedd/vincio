"""Audit logging and retention.

Records who requested what, which context/sources/tools were used, what was
returned, and what memory was written. Entries are append-only JSONL with an
integrity hash chain so tampering is detectable.

2.0 makes the chain tamper-*evident against a privileged attacker*. A plain
hash chain detects an outside editor, but anyone who can rewrite the whole file
can recompute every public SHA-256 hash and forge a clean chain. Two additions
close that gap:

* **Per-entry signatures** — when an :class:`AuditLog` is given a
  :class:`ChainSigner` (HMAC with a secret key, or an asymmetric Ed25519 key),
  every entry carries a signature over its ``entry_hash``. Forging history then
  requires the signing key, not just the algorithm.
* **Merkle checkpoints** — :meth:`AuditLog.checkpoint` publishes a Merkle root
  over the entries so far to a sidecar log. A root witnessed externally (or
  signed) pins history at that point, so even a later key compromise cannot
  rewrite already-witnessed entries.

Both are additive: an unsigned log keeps the same hash chain and the same
on-disk format (the new ``signature`` / ``key_id`` fields default to empty), so
1.x audit files still verify.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..core.utils import new_id, stable_hash, to_jsonable, utcnow

__all__ = [
    "AuditEntry",
    "AuditLog",
    "ChainVerification",
    "ChainSigner",
    "HMACSigner",
    "Ed25519Signer",
    "MerkleCheckpoint",
    "merkle_root",
    "merkle_proof",
    "verify_merkle_proof",
    "RetentionPolicy",
    "apply_retention",
    "verify_audit_file",
]


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


@runtime_checkable
class ChainSigner(Protocol):
    """Signs and verifies audit-entry hashes.

    ``key_id`` identifies which key produced a signature so a verifier can
    select the right material; it is recorded on each entry but never secret.
    """

    key_id: str

    def sign(self, message: str) -> str: ...

    def verify(self, message: str, signature: str) -> bool: ...


class HMACSigner:
    """HMAC-SHA256 signer — the zero-dependency default.

    Symmetric: the same secret signs and verifies. Strong tamper-evidence as
    long as the key is held only by the writer and the verifier; for
    third-party verifiability without sharing the secret, use
    :class:`Ed25519Signer`.
    """

    def __init__(self, key: bytes | str, *, key_id: str = "hmac") -> None:
        self._key = key.encode("utf-8") if isinstance(key, str) else key
        if not self._key:
            raise ValueError("HMACSigner requires a non-empty key")
        self.key_id = key_id

    def sign(self, message: str) -> str:
        return hmac.new(self._key, message.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self, message: str, signature: str) -> bool:
        return hmac.compare_digest(self.sign(message), signature or "")


class Ed25519Signer:
    """Asymmetric Ed25519 signer (optional ``cryptography`` dependency).

    Lets an external party verify the chain with only the public key, so the
    audit writer need not share signing material. Falls back with a clear error
    if ``cryptography`` is not installed. Pass a private key for a writer, or a
    public key only for a verify-only instance.
    """

    def __init__(
        self,
        private_key: Any | None = None,
        public_key: Any | None = None,
        *,
        key_id: str = "ed25519",
    ) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "Ed25519Signer requires the 'cryptography' package "
                "(pip install cryptography)"
            ) from exc
        if private_key is None and public_key is None:
            raise ValueError("Ed25519Signer needs a private and/or public key")
        self._private = private_key
        self._public = public_key or (private_key.public_key() if private_key else None)
        self.key_id = key_id

    def sign(self, message: str) -> str:
        if self._private is None:
            raise ValueError("Ed25519Signer has no private key; cannot sign")
        return self._private.sign(message.encode("utf-8")).hex()

    def verify(self, message: str, signature: str) -> bool:
        from cryptography.exceptions import InvalidSignature

        if self._public is None or not signature:
            return False
        try:
            self._public.verify(bytes.fromhex(signature), message.encode("utf-8"))
            return True
        except (InvalidSignature, ValueError):
            return False


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
    # 2.0: signature over ``entry_hash`` and the id of the key that produced it.
    # Empty on an unsigned log; the hash is deliberately excluded from
    # ``compute_hash`` so 1.x entries keep the same hash and still verify.
    signature: str = ""
    key_id: str = ""

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


def _hash_pair(left: str, right: str) -> str:
    return hashlib.sha256((left + right).encode("utf-8")).hexdigest()[:32]


def merkle_root(hashes: list[str]) -> str:
    """Merkle root over a list of entry hashes (odd levels duplicate the last
    node). The empty list hashes to the empty string."""
    if not hashes:
        return ""
    level = list(hashes)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_hash_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(hashes: list[str], index: int) -> list[tuple[str, str]]:
    """Inclusion proof for ``hashes[index]``: a list of (sibling_hash, side)
    pairs (``side`` is ``"L"`` or ``"R"``) walking root-ward."""
    if not 0 <= index < len(hashes):
        raise IndexError("index out of range for merkle proof")
    proof: list[tuple[str, str]] = []
    level = list(hashes)
    idx = index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sibling = idx ^ 1
        side = "R" if sibling > idx else "L"
        proof.append((level[sibling], side))
        level = [_hash_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: list[tuple[str, str]], root: str) -> bool:
    """Check that ``leaf`` is included under ``root`` given its inclusion proof."""
    acc = leaf
    for sibling, side in proof:
        acc = _hash_pair(sibling, acc) if side == "L" else _hash_pair(acc, sibling)
    return acc == root


class MerkleCheckpoint(BaseModel):
    """A periodic, optionally-signed root over the audit chain so far."""

    id: str = Field(default_factory=lambda: new_id("ckpt"))
    timestamp: datetime = Field(default_factory=utcnow)
    root: str
    count: int
    first_entry_id: str | None = None
    last_entry_id: str | None = None
    key_id: str = ""
    signature: str = ""


class AuditLog:
    """Append-only audit log. ``directory=None`` keeps entries in memory only.

    Pass a :class:`ChainSigner` to sign every entry (and Merkle checkpoint),
    making the chain tamper-evident against an attacker who can rewrite the
    whole file. Without a signer the log behaves exactly as in 1.x.
    """

    def __init__(
        self,
        directory: str | Path | None = ".vincio/audit",
        *,
        signer: ChainSigner | None = None,
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._last_hash = ""
        self.signer = signer

    @property
    def path(self) -> Path | None:
        return self.directory / "audit.jsonl" if self.directory else None

    @property
    def merkle_path(self) -> Path | None:
        return self.directory / "audit.merkle.jsonl" if self.directory else None

    def record(self, action: str, **fields: Any) -> AuditEntry:
        details = fields.pop("details", {})
        entry = AuditEntry(action=action, details=details, **fields)
        with self._lock:
            entry.prev_hash = self._last_hash
            entry.entry_hash = entry.compute_hash()
            if self.signer is not None:
                entry.signature = self.signer.sign(entry.entry_hash)
                entry.key_id = self.signer.key_id
            self._last_hash = entry.entry_hash
            self.entries.append(entry)
            if self.path is not None:
                self.directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(to_jsonable(entry.model_dump(mode="json"))) + "\n")
        return entry

    def merkle_root(self) -> str:
        """Merkle root over every recorded entry's hash."""
        with self._lock:
            return merkle_root([e.entry_hash for e in self.entries])

    def checkpoint(self) -> MerkleCheckpoint:
        """Publish a (signed, if a signer is set) Merkle root over the chain so
        far to the sidecar ``audit.merkle.jsonl``. Witness the returned root
        externally to pin history irreversibly at this point."""
        with self._lock:
            root = merkle_root([e.entry_hash for e in self.entries])
            checkpoint = MerkleCheckpoint(
                root=root,
                count=len(self.entries),
                first_entry_id=self.entries[0].id if self.entries else None,
                last_entry_id=self.entries[-1].id if self.entries else None,
            )
            if self.signer is not None and root:
                checkpoint.signature = self.signer.sign(root)
                checkpoint.key_id = self.signer.key_id
            if self.merkle_path is not None:
                self.directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
                with self.merkle_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(to_jsonable(checkpoint.model_dump(mode="json"))) + "\n")
        return checkpoint

    def verify_chain(self, *, verifier: ChainSigner | None = None) -> bool:
        """Validate the integrity hash chain over in-memory entries.

        When ``verifier`` (or the log's own signer) is supplied, every signed
        entry's signature is checked too; an entry whose signature does not
        verify fails the chain even if its hash links correctly.
        """
        check = verifier or self.signer
        previous = ""
        for entry in self.entries:
            if entry.prev_hash != previous or entry.entry_hash != entry.compute_hash():
                return False
            if check is not None and entry.signature:
                if not check.verify(entry.entry_hash, entry.signature):
                    return False
            previous = entry.entry_hash
        return True

    def verify_file(self, *, verifier: ChainSigner | None = None) -> ChainVerification:
        """Re-read the persisted JSONL and verify its hash chain offline.

        Unlike :meth:`verify_chain` (in-memory only), this detects tampering of
        the on-disk log after a process restart — a row edited, inserted, or
        deleted breaks the chain at a pinpointed line. With a ``verifier`` (or
        the log's signer) it also validates per-entry signatures. In-memory logs
        (``directory=None``) verify as intact with zero entries.
        """
        if self.path is None:
            return ChainVerification(intact=True, entries=0)
        return verify_audit_file(self.path, verifier=verifier or self.signer)

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
    # 2.0: how many entries carried a signature, and (when a verifier was
    # supplied) whether all of them verified. ``signatures_ok`` is ``None`` when
    # no verifier was provided — the hash chain was checked but signatures were
    # not.
    signed_entries: int = 0
    signatures_ok: bool | None = None


def verify_audit_file(
    jsonl_path: str | Path, *, verifier: ChainSigner | None = None
) -> ChainVerification:
    """Verify the integrity hash chain of a persisted audit JSONL file.

    Recomputes each entry's hash from its content and checks that every
    ``prev_hash`` links to the prior ``entry_hash``, so any edit, reorder,
    insert, or delete is detected and localized. When ``verifier`` is given,
    every signed entry's signature is validated against its ``entry_hash`` too,
    so an attacker who recomputed the public hashes still fails without the key.
    A missing file verifies as intact with zero entries.
    """
    path = Path(jsonl_path)
    if not path.is_file():
        return ChainVerification(intact=True, entries=0)
    previous = ""
    count = 0
    signed = 0
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
            if entry.signature:
                signed += 1
                if verifier is not None and not verifier.verify(entry.entry_hash, entry.signature):
                    return ChainVerification(
                        intact=False,
                        entries=count,
                        broken_at=lineno,
                        reason="signature mismatch",
                        signed_entries=signed,
                        signatures_ok=False,
                    )
            previous = entry.entry_hash
    return ChainVerification(
        intact=True,
        entries=count,
        signed_entries=signed,
        signatures_ok=(signed > 0) if verifier is not None else None,
    )


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
