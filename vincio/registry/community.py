"""Community pack & skill registry — a governed, signed bundle index.

A :class:`CommunityRegistry` is an index of opt-in domain **packs** and
``SKILL.md`` **skill** bundles. It mirrors the governance of the agent fabric:
every resolution passes the same :class:`~vincio.security.access.AllowListGate`
the :class:`~vincio.registry.directory.AgentDirectory` uses and is recorded as
an access decision on the hash-chained audit log. Each entry is **content-bound**
(SHA-256 over its payload) and may be **signed** with the library's
:class:`~vincio.security.audit.ChainSigner` (HMAC by default, Ed25519 for
third-party verification), so a resolution also verifies integrity and provenance
— a tampered or unsigned-when-required bundle is denied, not silently served.

    from vincio.registry import CommunityRegistry, BundleRecord
    from vincio.security.access import AllowListGate
    from vincio.security.audit import HMACSigner

    signer = HMACSigner("publisher-key")
    registry = CommunityRegistry(
        allow_list=AllowListGate(allow=["support-*"]),
        audit=app.audit,
        signer=signer,
    )
    registry.publish_pack(my_pack, version="1.2.0", publisher="acme")  # signed
    pack = registry.load_pack("support-pro")     # governed + audited + verified
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import AccessDeniedError, VincioError
from ..security.access import AccessDecision, AllowListGate, Principal

__all__ = [
    "BundleKind",
    "BundleRecord",
    "BundleResolution",
    "CommunityRegistry",
]

BundleKind = Literal["pack", "skill"]


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class BundleRecord(BaseModel):
    """One governed, content-bound entry in the community index.

    A ``pack`` carries its :class:`~vincio.packs.Pack` JSON in ``payload``; a
    ``skill`` carries its ``SKILL.md`` text in ``payload_text``. ``digest`` is
    the SHA-256 of that content and is the integrity anchor the signature covers.
    """

    name: str
    kind: BundleKind
    version: str = "0.0.0"
    description: str = ""
    publisher: str = ""
    tags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] | None = None  # pack JSON
    payload_text: str | None = None  # skill SKILL.md
    digest: str = ""  # sha256 of the canonical content
    signature: str = ""  # hex signature over the signing message
    key_id: str = ""  # which key produced the signature
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content(self) -> str:
        """The canonical content string the digest is computed over."""
        if self.kind == "pack":
            return json.dumps(self.payload or {}, sort_keys=True, separators=(",", ":"), default=str)
        return self.payload_text or ""

    def compute_digest(self) -> str:
        return _sha256(self.content())

    def signing_message(self) -> str:
        """Canonical message a signature covers: identity bound to content."""
        return f"{self.name}|{self.kind}|{self.version}|{self.digest}"


class BundleResolution(BaseModel):
    """The (non-raising) outcome of a governed bundle resolution."""

    allowed: bool
    decision: AccessDecision
    verified: bool = False
    record: BundleRecord | None = None


class CommunityRegistry:
    """A governed, signed, audited index of community packs and skills."""

    def __init__(
        self,
        *,
        allow_list: AllowListGate | None = None,
        audit: Any | None = None,
        signer: Any | None = None,
        principal: Principal | None = None,
        require_signature: bool = False,
        index: list[BundleRecord] | None = None,
    ) -> None:
        self.allow_list = allow_list
        self.audit = audit
        self.signer = signer
        self.principal = principal or Principal()
        # When True (or whenever a signer is configured), an unsigned or
        # invalidly-signed bundle is denied at resolve time.
        self.require_signature = require_signature or signer is not None
        self._records: dict[str, BundleRecord] = {}
        for record in index or []:
            self.register(record)

    # -- registration ---------------------------------------------------------

    def register(self, record: BundleRecord) -> BundleRecord:
        """Add (or replace) a record, computing its digest if absent."""
        if not record.digest:
            record = record.model_copy(update={"digest": record.compute_digest()})
        self._records[record.name] = record
        if self.audit is not None:
            self.audit.record(
                "bundle_register",
                resource=record.name,
                decision="registered",
                details={
                    "kind": record.kind,
                    "version": record.version,
                    "publisher": record.publisher,
                    "signed": bool(record.signature),
                },
            )
        return record

    def sign(self, record: BundleRecord) -> BundleRecord:
        """Return a signed copy of ``record`` (requires a configured signer)."""
        if self.signer is None:
            raise VincioError("CommunityRegistry.sign requires a signer")
        digest = record.digest or record.compute_digest()
        signed = record.model_copy(update={"digest": digest})
        signature = self.signer.sign(signed.signing_message())
        return signed.model_copy(
            update={"signature": signature, "key_id": getattr(self.signer, "key_id", "")}
        )

    def publish_pack(
        self, pack: Any, *, version: str = "0.0.0", publisher: str = "", tags: list[str] | None = None
    ) -> BundleRecord:
        """Build, sign (if a signer is set), and register a pack bundle."""
        record = BundleRecord(
            name=pack.name,
            kind="pack",
            version=version,
            description=getattr(pack, "description", ""),
            publisher=publisher,
            tags=list(tags or getattr(pack, "tags", []) or []),
            payload=pack.model_dump(mode="json"),
        )
        record = record.model_copy(update={"digest": record.compute_digest()})
        if self.signer is not None:
            record = self.sign(record)
        return self.register(record)

    def publish_skill(
        self,
        skill_md: str,
        *,
        name: str,
        version: str = "0.0.0",
        description: str = "",
        publisher: str = "",
        tags: list[str] | None = None,
    ) -> BundleRecord:
        """Build, sign (if a signer is set), and register a skill bundle."""
        record = BundleRecord(
            name=name,
            kind="skill",
            version=version,
            description=description,
            publisher=publisher,
            tags=list(tags or []),
            payload_text=skill_md,
        )
        record = record.model_copy(update={"digest": record.compute_digest()})
        if self.signer is not None:
            record = self.sign(record)
        return self.register(record)

    def all(self) -> list[BundleRecord]:
        return list(self._records.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._records)

    # -- discovery ------------------------------------------------------------

    def find(
        self,
        *,
        kind: BundleKind | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> list[BundleRecord]:
        """Discover bundles by kind / tag / free-text (no governance applied)."""
        out: list[BundleRecord] = []
        for record in self._records.values():
            if kind is not None and record.kind != kind:
                continue
            if tag is not None and tag.lower() not in {t.lower() for t in record.tags}:
                continue
            if query is not None:
                haystack = f"{record.name} {record.description} {' '.join(record.tags)}".lower()
                if not any(tok in haystack for tok in query.lower().split()):
                    continue
            out.append(record)
        return sorted(out, key=lambda r: r.name)

    # -- integrity ------------------------------------------------------------

    def verify(self, record: BundleRecord) -> tuple[bool, str]:
        """Check content integrity and (if required) the signature.

        Returns ``(ok, reason)``. The digest must match the payload; when a
        signer is configured (or ``require_signature``), a present signature must
        verify and an absent one is rejected.
        """
        if record.digest != record.compute_digest():
            return False, "content digest mismatch (bundle was modified)"
        if self.signer is not None:
            if not record.signature:
                return False, "bundle is unsigned but a signature is required"
            if not self.signer.verify(record.signing_message(), record.signature):
                return False, "signature verification failed"
            return True, "digest and signature verified"
        if self.require_signature and not record.signature:
            return False, "bundle is unsigned but a signature is required"
        return True, "digest verified"

    # -- governed resolution --------------------------------------------------

    def try_resolve(self, name: str, *, principal: Principal | None = None) -> BundleResolution:
        """Resolve ``name`` under the allow-list + integrity check; never raises.

        The access decision (allow/deny) is recorded on the audit chain either
        way, exactly like :meth:`AgentDirectory.try_resolve`.
        """
        record = self._records.get(name)
        if self.allow_list is None:
            decision = AccessDecision(allowed=True, rule="no_gate", reason="no allow-list configured")
        else:
            decision = self.allow_list.check(name, principal=principal or self.principal)
        if decision.allowed and record is None:
            decision = AccessDecision(
                allowed=False, rule="not_found", reason=f"bundle {name!r} not in registry"
            )
        verified = False
        if decision.allowed and record is not None:
            verified, reason = self.verify(record)
            if not verified:
                decision = AccessDecision(allowed=False, rule="integrity", reason=reason)
        if self.audit is not None:
            self.audit.record(
                "bundle_resolve",
                resource=name,
                decision="allow" if decision.allowed else "deny",
                details={
                    "rule": decision.rule,
                    "reason": decision.reason,
                    "kind": record.kind if record else None,
                    "version": record.version if record else None,
                    "publisher": record.publisher if record else None,
                    "verified": verified,
                },
            )
        return BundleResolution(
            allowed=decision.allowed,
            decision=decision,
            verified=verified,
            record=record if decision.allowed else None,
        )

    def resolve(self, name: str, *, principal: Principal | None = None) -> BundleRecord:
        """Resolve ``name`` under the gate + integrity check; raise if denied."""
        resolution = self.try_resolve(name, principal=principal)
        if not resolution.allowed or resolution.record is None:
            raise AccessDeniedError(
                resolution.decision.reason or f"bundle {name!r} is not reachable",
                details={"rule": resolution.decision.rule, "bundle": name},
            )
        return resolution.record

    # -- materialization ------------------------------------------------------

    def load_pack(
        self, name: str, *, principal: Principal | None = None, register: bool = True
    ) -> Any:
        """Resolve (governed + verified) and materialize a :class:`Pack`."""
        from ..packs import Pack, register_pack

        record = self.resolve(name, principal=principal)
        if record.kind != "pack" or record.payload is None:
            raise VincioError(f"bundle {name!r} is not a pack")
        pack = Pack.model_validate(record.payload)
        if register:
            register_pack(pack)
        return pack

    def load_skill(self, name: str, *, principal: Principal | None = None) -> Any:
        """Resolve (governed + verified) and materialize a :class:`Skill`."""
        from ..skills import skill_from_markdown

        record = self.resolve(name, principal=principal)
        if record.kind != "skill" or record.payload_text is None:
            raise VincioError(f"bundle {name!r} is not a skill")
        return skill_from_markdown(record.payload_text, name=record.name)

    # -- signed index ---------------------------------------------------------

    def index_root(self) -> str:
        """A single digest over the whole index (sorted ``name:digest`` lines)."""
        lines = [f"{r.name}:{r.digest}" for r in sorted(self._records.values(), key=lambda r: r.name)]
        return _sha256("\n".join(lines))

    def sign_index(self) -> str:
        """Sign the index root, so the catalog as a whole is tamper-evident."""
        if self.signer is None:
            raise VincioError("CommunityRegistry.sign_index requires a signer")
        return self.signer.sign(self.index_root())

    def verify_index(self, signature: str) -> bool:
        """Verify a signature produced by :meth:`sign_index`."""
        if self.signer is None:
            raise VincioError("CommunityRegistry.verify_index requires a signer")
        return self.signer.verify(self.index_root(), signature)
