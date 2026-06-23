"""Agent identity, delegation & cryptographic accountability.

Vincio signs contracts, settlements, attestations, audit entries, and engagement
narratives — but *who* a signing key belongs to has been an out-of-band
assumption (a ``key_id`` string the verifier had to trust). For production
multi-org and multi-agent deployments that assumption is the weak link:
accountability is only as good as the answer to *who authorized this action, down
what chain, within what bounds*. This module makes identity itself first-class and
verifiable, the substrate beneath the tool permissions, the agent fabric, and the
whole cross-org trust fabric.

* **Portable, self-certifying identity.** An :class:`AgentIdentity`
  (:meth:`~vincio.core.app.ContextApp.identity`) is built on an Ed25519 key whose
  **DID is derived from the public key** (``did:vincio:ed25519:<hex>``), so the
  identifier *is* the key — anyone resolves the verifying key from the id alone,
  offline, with no registry. Its :class:`IdentityDocument` (keys, advertised
  capabilities, rotation history) is content-bound and signed, and ``verify``\\ s
  from the bytes. A :class:`Keyring` rotates keys along a **signed rotation
  chain** — each new key authorized by the one before it — so a signature is
  validated against the key that was current *at signing time*: a rotated-away or
  revoked key cannot forge new history, while signatures it made while current
  stay valid.
* **Delegation chains & attenuated authority.** A signed :class:`Delegation`
  mints a bounded :class:`Grant` (a subset of capabilities, a budget cap, an
  expiry, an audience) from a principal to an agent — and an agent can
  sub-delegate to a sub-agent. The links compose into a :class:`DelegationChain`
  that ``verify``\\ s **offline** (each issuer's key resolves from its DID) where
  **each link only attenuates, never amplifies**: capabilities only shrink, the
  budget only tightens, the expiry only shortens. So a tool call, a contract
  signature, or a saga handoff carries *provenance of authority*, and an
  over-reaching sub-delegation is refused from the bytes.
* **Verifiable credentials & accountable audit.** An org issues a signed
  :class:`AgentCredential` (a verifiable claim — *this agent is admitted to
  capability X*, *operated by org Y*) an importer verifies offline and folds into
  the existing admission / registry path. Because an :class:`AgentIdentity`
  satisfies the :class:`~vincio.security.audit.ChainSigner` protocol, making it the
  app's signer (:meth:`~vincio.core.app.ContextApp.use_identity`) binds every audit
  entry, contract, and settlement to the **DID** that produced it — so a forged or
  unauthorized signature is refused and pinpointed, never merely logged.

Everything is deterministic, offline, and dependency-free by default: Ed25519 runs
in pure Python (RFC 8032), and the native, constant-time ``cryptography`` backend
is used automatically when ``vincio[crypto]`` is installed — byte-for-byte the
same signatures. Never a hosted identity provider, a CA, or a key-escrow service —
a verifiable identity substrate inside your process.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import IdentityError
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from . import _ed25519 as ed

__all__ = [
    "did_from_public_key",
    "public_key_from_did",
    "is_vincio_did",
    "key_fingerprint",
    "Grant",
    "KeyRecord",
    "KeyAuthorization",
    "SignatureCheck",
    "IdentityVerification",
    "IdentityDocument",
    "Keyring",
    "AgentIdentity",
    "DelegationVerification",
    "Delegation",
    "DelegationChainVerification",
    "DelegationChain",
    "CredentialVerification",
    "AgentCredential",
]

# Audit actions the app facade records identity artifacts under.
IDENTITY_ACTION = "identity"
CREDENTIAL_ACTION = "credential"

# DID method this library mints. The public key is carried verbatim (hex) so the
# verifying key resolves from the identifier alone — self-certifying, no registry.
DID_PREFIX = "did:vincio:ed25519:"

_TOLERANCE = 1e-9


def _r6(value: float) -> float:
    return round(float(value), 6)


# ---------------------------------------------------------------------------
# DID helpers — the identifier is derived from (and resolves to) the public key
# ---------------------------------------------------------------------------


def did_from_public_key(public_key: bytes) -> str:
    """Derive the self-certifying DID for an Ed25519 public key.

    The 32-byte key is carried verbatim as hex after the method prefix, so the DID
    *is* the key: :func:`public_key_from_did` recovers it offline, with no registry
    or network call. This is what makes an :class:`AgentIdentity` portable.
    """
    if len(public_key) != 32:
        raise IdentityError(
            "an Ed25519 public key is 32 bytes",
            details={"length": len(public_key)},
        )
    return DID_PREFIX + public_key.hex()


def public_key_from_did(did: str) -> bytes:
    """Recover the Ed25519 public key embedded in a ``did:vincio:ed25519`` DID.

    The inverse of :func:`did_from_public_key`: a verifier resolves a signer's key
    from its DID string alone — the offline-resolvable property the trust fabric
    rests on. Raises :class:`~vincio.core.errors.IdentityError` on a malformed DID.
    """
    if not is_vincio_did(did):
        raise IdentityError(
            f"not a vincio DID: {did!r}",
            details={"did": did},
        )
    hexpart = did[len(DID_PREFIX) :]
    try:
        key = bytes.fromhex(hexpart)
    except ValueError as exc:
        raise IdentityError(f"malformed DID key material: {did!r}", details={"did": did}) from exc
    if len(key) != 32:
        raise IdentityError(f"DID key material is not 32 bytes: {did!r}", details={"did": did})
    return key


def is_vincio_did(did: str) -> bool:
    """Whether ``did`` is a well-formed ``did:vincio:ed25519`` identifier."""
    return isinstance(did, str) and did.startswith(DID_PREFIX) and len(did) == len(DID_PREFIX) + 64


def key_fingerprint(public_key: bytes) -> str:
    """A short, stable key id (``k<16 hex>``) for a public key — used as ``kid``."""
    return "k" + stable_hash({"pub": public_key.hex()}, length=16)


# ---------------------------------------------------------------------------
# Grant — the bounded authority that only ever attenuates
# ---------------------------------------------------------------------------


class Grant(BaseModel):
    """A bounded grant of authority: the capabilities, budget, expiry, and audience.

    The unit a :class:`Delegation` conveys. Authority is intentionally *negative*
    space — a grant says what is permitted and nothing more — so attenuation is a
    structural check, not a policy: a child grant is valid only if it
    :meth:`attenuates` its parent (every capability still permitted, the budget no
    larger, the expiry no later, the audience no broader, one fewer re-delegation).

    ``capabilities`` lists capability names; the wildcard ``"*"`` permits any
    capability and may appear only where a parent already permits it (a root grant
    a principal issues to itself), so a sub-delegation can never introduce it.
    ``budget_usd`` / ``not_after`` of ``None`` mean *unbounded* / *no expiry* and
    may only be set at a level whose parent is also unbounded — narrowing to a
    bound is attenuation, widening from one is amplification and is refused.
    ``max_delegations`` bounds how many further hops the authority may travel.
    """

    capabilities: list[str] = Field(default_factory=list)
    budget_usd: float | None = None
    not_after: datetime | None = None
    audience: str = ""
    max_delegations: int | None = None

    # -- normalization ------------------------------------------------------

    @property
    def capability_set(self) -> frozenset[str]:
        return frozenset(self.capabilities)

    @property
    def grants_all(self) -> bool:
        """Whether this grant carries the wildcard capability."""
        return "*" in self.capability_set

    def permits_capability(self, capability: str) -> bool:
        """Whether ``capability`` is within this grant (wildcard or explicit)."""
        return self.grants_all or capability in self.capability_set

    # -- attenuation (the core security invariant) --------------------------

    def attenuates(self, parent: Grant) -> bool:
        """Whether this grant is a valid *attenuation* of ``parent`` — never an amplification.

        True only when every dimension narrows or holds: every capability is still
        permitted by ``parent``, the budget is no larger, the expiry no later, the
        audience no broader, and at least one re-delegation hop is consumed. Any
        single widening makes it ``False``, so :meth:`DelegationChain.verify` refuses
        an over-reaching link from the bytes alone.
        """
        # Capabilities: child ⊆ parent (wildcard parent permits anything).
        if not self.grants_all and not parent.grants_all:
            if not self.capability_set <= parent.capability_set:
                return False
        elif self.grants_all and not parent.grants_all:
            return False  # child cannot introduce the wildcard
        # Budget: a bounded parent forbids an unbounded or larger child.
        if not _budget_attenuates(self.budget_usd, parent.budget_usd):
            return False
        # Expiry: a parent with an expiry forbids a later or absent child expiry.
        if not _expiry_attenuates(self.not_after, parent.not_after):
            return False
        # Audience: a fixed parent audience forbids broadening.
        if parent.audience and self.audience != parent.audience:
            return False
        # Re-delegation depth: each hop consumes one, and a depleted parent grants none.
        if not _depth_attenuates(self.max_delegations, parent.max_delegations):
            return False
        return True

    def permits(
        self,
        capability: str | None = None,
        *,
        budget_usd: float | None = None,
        at: datetime | None = None,
        audience: str | None = None,
    ) -> bool:
        """Whether this grant authorizes a concrete request.

        ``capability`` must be within the grant, a requested ``budget_usd`` must fit
        under the cap, ``at`` must be before the expiry, and ``audience`` must match a
        fixed audience. ``None`` arguments are not checked.
        """
        if capability is not None and not self.permits_capability(capability):
            return False
        if (
            budget_usd is not None
            and self.budget_usd is not None
            and budget_usd > self.budget_usd + _TOLERANCE
        ):
            return False
        if at is not None and self.not_after is not None and at > self.not_after:
            return False
        if audience and self.audience and audience != self.audience:
            return False
        return True

    def facts(self) -> dict[str, Any]:
        """The grant facts a content hash binds (capabilities sorted for stability)."""
        return {
            "capabilities": sorted(self.capability_set),
            "budget_usd": None if self.budget_usd is None else _r6(self.budget_usd),
            "not_after": self.not_after.isoformat() if self.not_after else None,
            "audience": self.audience,
            "max_delegations": self.max_delegations,
        }


def _budget_attenuates(child: float | None, parent: float | None) -> bool:
    if parent is None:
        return True
    if child is None:
        return False
    return child <= parent + _TOLERANCE


def _expiry_attenuates(child: datetime | None, parent: datetime | None) -> bool:
    if parent is None:
        return True
    if child is None:
        return False
    return child <= parent


def _depth_attenuates(child: int | None, parent: int | None) -> bool:
    if parent is None:
        return True
    if parent <= 0:
        return False  # a depleted parent permits no further delegation
    if child is None:
        return False
    return child <= parent - 1


# ---------------------------------------------------------------------------
# Keys & the identity document
# ---------------------------------------------------------------------------


class KeyRecord(BaseModel):
    """One public key in an identity's rotation history.

    The genesis key (``prev_kid == ""``) defines the identity's DID; every later
    key carries a ``rotation_sig`` — the predecessor key's signature over this
    record's rotation statement — so the history is a **signed chain**: a key is
    valid only if the key before it authorized it while current. ``status`` /
    ``revoked_at`` retire or revoke a key; a revoked key cannot have authorized any
    rotation dated after its revocation.
    """

    kid: str
    public_key: str  # hex
    algorithm: str = "ed25519"
    created_at: datetime = Field(default_factory=utcnow)
    status: str = "active"  # active | retired | revoked
    not_after: datetime | None = None
    revoked_at: datetime | None = None
    prev_kid: str = ""
    rotation_sig: str = ""

    def public_bytes(self) -> bytes:
        return bytes.fromhex(self.public_key)

    def rotation_message(self, subject: str) -> str:
        """The stable statement a predecessor key signs to authorize this key."""
        return stable_hash(
            {
                "subject": subject,
                "kid": self.kid,
                "public_key": self.public_key,
                "algorithm": self.algorithm,
                "created_at": self.created_at.isoformat(),
                "prev_kid": self.prev_kid,
            },
            length=32,
        )

    def active_at(self, at: datetime) -> bool:
        """Whether this key was usable at ``at`` (created, not expired, not revoked)."""
        if at < self.created_at:
            return False
        if self.not_after is not None and at > self.not_after:
            return False
        if self.revoked_at is not None and at >= self.revoked_at:
            return False
        return self.status != "revoked" or (self.revoked_at is not None and at < self.revoked_at)

    def facts(self) -> dict[str, Any]:
        return {
            "kid": self.kid,
            "public_key": self.public_key,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "not_after": self.not_after.isoformat() if self.not_after else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "prev_kid": self.prev_kid,
            "rotation_sig": self.rotation_sig,
        }


class KeyAuthorization(BaseModel):
    """An offline proof that a signing key descends from an identity's genesis key.

    Carried by an artifact signed with a *rotated* (non-genesis) key, it is the
    minimal slice of the rotation chain from the genesis key to the signer, so the
    artifact stays verifiable from the bytes alone without fetching the signer's
    full :class:`IdentityDocument`. An artifact signed with the genesis key needs no
    authorization (the signer key *is* the DID).
    """

    path: list[KeyRecord] = Field(default_factory=list)

    def facts(self) -> list[dict[str, Any]]:
        return [record.facts() for record in self.path]


class SignatureCheck(BaseModel):
    """Which key verified a signature and whether it was valid at a given time."""

    valid: bool
    kid: str | None = None
    active_at_check: bool = True
    reason: str | None = None


class IdentityVerification(BaseModel):
    """The (non-raising) outcome of verifying an identity document offline."""

    valid: bool
    hash_ok: bool
    did_matches_genesis: bool
    rotation_chain_ok: bool
    signature_ok: bool
    reason: str | None = None


class IdentityDocument(BaseModel):
    """A signed, content-bound description of an agent identity.

    Holds the identity's DID (``subject``), its advertised ``capabilities``, an
    optional operating ``controller`` org, and the full ``keys`` rotation history
    (genesis first). It is signed by the **current active key** and ``verify``\\ s
    from the bytes: the DID re-derives from the genesis key, the rotation chain is
    checked link-by-link (each key authorized by its predecessor), and the document
    signature checks against the active key. A revoked or rotated-away key cannot
    re-sign the document, so it cannot forge new identity history.
    """

    subject: str  # the DID
    name: str = ""
    controller: str = ""
    capabilities: list[str] = Field(default_factory=list)
    keys: list[KeyRecord] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    signature: str = ""
    signer_kid: str = ""
    audit_id: str | None = None

    # -- key lookup ---------------------------------------------------------

    @property
    def genesis_key(self) -> KeyRecord | None:
        for record in self.keys:
            if not record.prev_kid:
                return record
        return self.keys[0] if self.keys else None

    @property
    def active_key(self) -> KeyRecord | None:
        live = [k for k in self.keys if k.status == "active"]
        return live[-1] if live else None

    def resolve(self, kid: str) -> KeyRecord | None:
        for record in self.keys:
            if record.kid == kid:
                return record
        return None

    # -- hashing & signing --------------------------------------------------

    def facts(self) -> dict[str, Any]:
        """The facts the content hash binds — DID, metadata, and the key history."""
        return {
            "subject": self.subject,
            "name": self.name,
            "controller": self.controller,
            "capabilities": sorted(set(self.capabilities)),
            "keys": [record.facts() for record in self.keys],
            "created_at": self.created_at.isoformat(),
        }

    def compute_hash(self) -> str:
        return stable_hash(self.facts(), length=32)

    def seal(self) -> IdentityDocument:
        self.content_hash = self.compute_hash()
        return self

    # -- verification -------------------------------------------------------

    def _verify_rotation_chain(self) -> bool:
        """Each non-genesis key is authorized by its predecessor's signature."""
        by_kid = {record.kid: record for record in self.keys}
        for record in self.keys:
            if not record.prev_kid:
                continue  # genesis
            prev = by_kid.get(record.prev_kid)
            if prev is None:
                return False
            message = record.rotation_message(self.subject)
            if not ed.verify(
                prev.public_bytes(), message.encode("utf-8"), _hexbytes(record.rotation_sig)
            ):
                return False
            # The predecessor must have been current when it signed: a rotation dated
            # strictly after the predecessor's revocation is forged history. Rotating
            # *away* from a key at the instant it is revoked (its last legitimate act)
            # is allowed, so the comparison is strict.
            if prev.revoked_at is not None and record.created_at > prev.revoked_at:
                return False
        return True

    def verify(self) -> IdentityVerification:
        """Verify the document offline: hash, DID↔genesis, rotation chain, and signature."""
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        genesis = self.genesis_key
        did_ok = genesis is not None and self.subject == did_from_public_key(genesis.public_bytes())
        chain_ok = self._verify_rotation_chain()
        sig_ok = True
        if self.signature:
            signer = self.resolve(self.signer_kid) or self.active_key
            sig_ok = signer is not None and ed.verify(
                signer.public_bytes(), self.content_hash.encode("utf-8"), _hexbytes(self.signature)
            )
        valid = hash_ok and bool(did_ok) and chain_ok and sig_ok
        reason: str | None = None
        if not valid:
            if not self.content_hash:
                reason = "document is not sealed"
            elif not hash_ok:
                reason = "content hash does not match the document facts"
            elif not did_ok:
                reason = "subject DID does not match the genesis key"
            elif not chain_ok:
                reason = "rotation chain does not verify"
            else:
                reason = "document signature does not verify"
        return IdentityVerification(
            valid=valid,
            hash_ok=hash_ok,
            did_matches_genesis=bool(did_ok),
            rotation_chain_ok=chain_ok,
            signature_ok=sig_ok,
            reason=reason,
        )

    def verify_signature(
        self, message: str, signature: str, *, kid: str | None = None, at: datetime | None = None
    ) -> SignatureCheck:
        """Verify a signature against this identity's keys (rotation-aware).

        Finds the key that produced ``signature`` over ``message`` (or the named
        ``kid``) and reports it. With ``at`` set, also reports whether that key was
        active at that instant — the check that makes a rotated-away or revoked key
        unable to forge *new* history while its past signatures stay valid.
        """
        candidates = [self.resolve(kid)] if kid else list(self.keys)
        for record in candidates:
            if record is None:
                continue
            if ed.verify(record.public_bytes(), message.encode("utf-8"), _hexbytes(signature)):
                active = record.active_at(at) if at is not None else True
                reason = None
                if not active and at is not None:
                    reason = f"key {record.kid} was not active at {at.isoformat()}"
                return SignatureCheck(
                    valid=True,
                    kid=record.kid,
                    active_at_check=active,
                    reason=reason,
                )
        return SignatureCheck(valid=False, reason="no key in the document produced this signature")

    def audit_details(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "subject": self.subject,
                "name": self.name,
                "controller": self.controller,
                "capabilities": sorted(set(self.capabilities)),
                "keys": len(self.keys),
                "active_kid": self.active_key.kid if self.active_key else None,
                "content_hash": self.content_hash,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> IdentityDocument:
        return cls.model_validate(data)


def _hexbytes(value: str) -> bytes:
    try:
        return bytes.fromhex(value)
    except ValueError:
        return b""


# ---------------------------------------------------------------------------
# Keyring — private key material and the rotation chain
# ---------------------------------------------------------------------------


class Keyring:
    """Holds an identity's private keys and maintains its signed rotation chain.

    Created by :meth:`create` (a fresh random key, or a deterministic one from a
    32-byte ``seed`` for reproducible tests). Private seeds never leave the keyring
    and are never serialized — only the public :class:`IdentityDocument` is. The
    keyring is the only thing that can :meth:`rotate` (mint a new key the current
    one signs into the chain) or :meth:`revoke` a key, so a holder of the document
    alone can verify history but never extend it.
    """

    def __init__(self, document: IdentityDocument, seeds: dict[str, bytes]) -> None:
        self._document = document
        self._seeds = seeds  # kid -> 32-byte seed

    @classmethod
    def create(
        cls,
        *,
        name: str = "",
        controller: str = "",
        capabilities: list[str] | None = None,
        seed: bytes | None = None,
        created_at: datetime | None = None,
    ) -> Keyring:
        """Mint a new identity with a genesis key; return its keyring."""
        genesis_seed = seed if seed is not None else ed.generate_seed()
        if len(genesis_seed) != 32:
            raise IdentityError(
                "an identity seed must be 32 bytes", details={"length": len(genesis_seed)}
            )
        public = ed.public_key_from_seed(genesis_seed)
        did = did_from_public_key(public)
        when = created_at or utcnow()
        genesis = KeyRecord(
            kid=key_fingerprint(public),
            public_key=public.hex(),
            created_at=when,
            status="active",
        )
        document = IdentityDocument(
            subject=did,
            name=name or did,
            controller=controller,
            capabilities=list(capabilities or []),
            keys=[genesis],
            created_at=when,
            updated_at=when,
        )
        keyring = cls(document, {genesis.kid: genesis_seed})
        keyring._reseal()
        return keyring

    # -- state --------------------------------------------------------------

    @property
    def document(self) -> IdentityDocument:
        return self._document

    @property
    def did(self) -> str:
        return self._document.subject

    @property
    def active_kid(self) -> str:
        active = self._document.active_key
        if active is None:
            raise IdentityError("identity has no active key", details={"did": self.did})
        return active.kid

    def _active_seed(self) -> bytes:
        return self._seeds[self.active_kid]

    def _reseal(self) -> None:
        """Re-hash and re-sign the document with the current active key."""
        self._document.updated_at = utcnow()
        self._document.seal()
        kid = self.active_kid
        seed = self._seeds[kid]
        self._document.signer_kid = kid
        self._document.signature = ed.sign(seed, self._document.content_hash.encode("utf-8")).hex()

    # -- signing ------------------------------------------------------------

    def sign(self, message: str) -> str:
        """Sign ``message`` with the current active key; return a hex signature."""
        return ed.sign(self._active_seed(), message.encode("utf-8")).hex()

    def active_public(self) -> bytes:
        active = self._document.active_key
        if active is None:
            raise IdentityError("identity has no active key", details={"did": self.did})
        return active.public_bytes()

    def authorization(self) -> KeyAuthorization | None:
        """The rotation path proving the active key descends from genesis.

        ``None`` when the active key *is* the genesis key (the common, no-rotation
        case): the signer key equals the DID, so no proof is needed.
        """
        active = self._document.active_key
        if active is None or not active.prev_kid:
            return None
        path: list[KeyRecord] = []
        by_kid = {record.kid: record for record in self._document.keys}
        cursor: KeyRecord | None = active
        while cursor is not None:
            path.append(cursor)
            cursor = by_kid.get(cursor.prev_kid) if cursor.prev_kid else None
        path.reverse()
        return KeyAuthorization(path=path)

    # -- rotation & revocation ---------------------------------------------

    def rotate(self, *, seed: bytes | None = None, at: datetime | None = None) -> KeyRecord:
        """Rotate to a fresh key the current key signs into the rotation chain.

        The new key becomes active; the prior key is retired (its past signatures
        stay valid, but it can no longer sign new history). Returns the new key
        record.
        """
        prev_record = self._document.active_key
        if prev_record is None:
            raise IdentityError(
                "cannot rotate an identity with no active key", details={"did": self.did}
            )
        new_seed = seed if seed is not None else ed.generate_seed()
        public = ed.public_key_from_seed(new_seed)
        when = at or utcnow()
        record = KeyRecord(
            kid=key_fingerprint(public),
            public_key=public.hex(),
            created_at=when,
            status="active",
            prev_kid=prev_record.kid,
        )
        # The current key authorizes the new one by signing its rotation statement.
        message = record.rotation_message(self.did)
        record.rotation_sig = ed.sign(self._seeds[prev_record.kid], message.encode("utf-8")).hex()
        prev_record.status = "retired"
        self._seeds[record.kid] = new_seed
        self._document.keys.append(record)
        self._reseal()
        return record

    def revoke(self, kid: str | None = None, *, at: datetime | None = None) -> KeyRecord:
        """Revoke a key (defaults to the active one), then re-sign under a fresh key.

        A revoked key is permanently unusable for new signatures from ``revoked_at``
        on. Revoking the active key first rotates to a new key (so the identity still
        has a signer), then marks the old key revoked — modeling a compromise: the
        attacker's key can no longer forge history, but everything it legitimately
        signed before the compromise remains verifiable.
        """
        when = at or utcnow()
        target_kid = kid or self.active_kid
        target = self._document.resolve(target_kid)
        if target is None:
            raise IdentityError(
                f"unknown key {target_kid!r}", details={"did": self.did, "kid": target_kid}
            )
        if target.kid == self.active_kid:
            # Rotate away from the compromised key first so a signer remains.
            self.rotate(at=when)
        target.status = "revoked"
        target.revoked_at = when
        # Drop the private seed of a revoked key so it can never sign again.
        self._seeds.pop(target.kid, None)
        self._reseal()
        return target

    def signer(self, identity_name: str = "") -> AgentIdentity:
        """Wrap this keyring in an :class:`AgentIdentity` (a usable ChainSigner)."""
        return AgentIdentity(self, name=identity_name or self._document.name)


# ---------------------------------------------------------------------------
# AgentIdentity — the high-level handle and a drop-in ChainSigner
# ---------------------------------------------------------------------------


class AgentIdentity:
    """A portable agent identity: a keyring, its document, and an accountable signer.

    Returned by :meth:`~vincio.core.app.ContextApp.identity`. It exposes the
    identity's :attr:`did` and signed :attr:`document`, mints :class:`Delegation`\\ s
    (:meth:`delegate`) and :class:`AgentCredential`\\ s (:meth:`issue_credential`),
    and **satisfies the** :class:`~vincio.security.audit.ChainSigner` **protocol**
    (``key_id`` is the DID, plus :meth:`sign` / :meth:`verify`) — so it drops into
    every place the platform already takes a signer (the audit chain, contracts,
    settlements). Making it the app signer
    (:meth:`~vincio.core.app.ContextApp.use_identity`) binds every signed artifact to
    this DID, turning accountability from a logged ``key_id`` string into a
    cryptographic fact.
    """

    def __init__(self, keyring: Keyring, *, name: str = "") -> None:
        self._keyring = keyring
        self.name = name or keyring.document.name

    # -- construction -------------------------------------------------------

    @classmethod
    def generate(
        cls,
        name: str = "",
        *,
        controller: str = "",
        capabilities: list[str] | None = None,
        seed: bytes | None = None,
    ) -> AgentIdentity:
        """Mint a brand-new identity (optionally deterministic from a ``seed``)."""
        keyring = Keyring.create(
            name=name, controller=controller, capabilities=capabilities, seed=seed
        )
        return cls(keyring, name=name)

    # -- identity reads -----------------------------------------------------

    @property
    def keyring(self) -> Keyring:
        return self._keyring

    @property
    def did(self) -> str:
        return self._keyring.did

    @property
    def document(self) -> IdentityDocument:
        return self._keyring.document

    @property
    def capabilities(self) -> list[str]:
        return list(self._keyring.document.capabilities)

    # -- ChainSigner protocol ----------------------------------------------

    @property
    def key_id(self) -> str:
        """The DID — recorded as ``key_id`` on every artifact this identity signs."""
        return self.did

    def sign(self, message: str) -> str:
        """Sign ``message`` with the active key (the ChainSigner contract)."""
        return self._keyring.sign(message)

    def verify(self, message: str, signature: str) -> bool:
        """Verify a signature against any key this identity has held.

        Satisfies the ChainSigner contract: a signature this identity produced with
        the key current at the time verifies, so artifacts signed before a rotation
        stay valid. For rotation-aware, time-pinned checking use
        :meth:`IdentityDocument.verify_signature`.
        """
        return self._keyring.document.verify_signature(message, signature).valid

    # -- rotation -----------------------------------------------------------

    def rotate(self, *, seed: bytes | None = None) -> KeyRecord:
        """Rotate the signing key along the signed chain (see :meth:`Keyring.rotate`)."""
        return self._keyring.rotate(seed=seed)

    def revoke(self, kid: str | None = None) -> KeyRecord:
        """Revoke a key (see :meth:`Keyring.revoke`)."""
        return self._keyring.revoke(kid)

    # -- delegation ---------------------------------------------------------

    def delegate(
        self,
        to: str | AgentIdentity,
        *,
        capabilities: list[str] | None = None,
        budget_usd: float | None = None,
        expires_in: timedelta | None = None,
        not_after: datetime | None = None,
        audience: str = "",
        max_delegations: int | None = None,
        grant: Grant | None = None,
        issued_at: datetime | None = None,
    ) -> Delegation:
        """Mint a signed :class:`Delegation` granting bounded authority to ``to``.

        ``to`` is the delegate's DID (or its :class:`AgentIdentity`). The grant is
        built from the explicit ``capabilities`` / ``budget_usd`` / expiry /
        ``audience`` / ``max_delegations`` (or a ready :class:`Grant`). The
        delegation is sealed and signed by *this* identity; sub-delegate it further
        with :meth:`Delegation.delegate`, and compose links into a
        :class:`DelegationChain`.
        """
        subject = to.did if isinstance(to, AgentIdentity) else to
        when = issued_at or utcnow()
        resolved_grant = grant or _grant_from_kwargs(
            capabilities=capabilities,
            budget_usd=budget_usd,
            expires_in=expires_in,
            not_after=not_after,
            audience=audience,
            max_delegations=max_delegations,
            issued_at=when,
        )
        delegation = Delegation(
            issuer=self.did,
            subject=subject,
            grant=resolved_grant,
            issued_at=when,
        )
        return delegation.sign(self)

    # -- credentials --------------------------------------------------------

    def issue_credential(
        self,
        subject: str | AgentIdentity,
        claims: dict[str, str],
        *,
        not_after: datetime | None = None,
        expires_in: timedelta | None = None,
        issued_at: datetime | None = None,
    ) -> AgentCredential:
        """Issue a signed :class:`AgentCredential` asserting ``claims`` about ``subject``.

        The verifiable-claim primitive an org uses to vouch for an agent (*admitted
        to capability X*, *operated by org Y*). Signed by this identity and
        verifiable offline by anyone who resolves this DID.
        """
        subject_did = subject.did if isinstance(subject, AgentIdentity) else subject
        when = issued_at or utcnow()
        if not_after is None and expires_in is not None:
            not_after = when + expires_in
        credential = AgentCredential(
            issuer=self.did,
            subject=subject_did,
            claims=dict(claims),
            issued_at=when,
            not_after=not_after,
        )
        return credential.sign(self)

    # -- low-level signing of artifacts ------------------------------------

    def _sign_artifact(self, content_hash: str) -> tuple[str, str, str, KeyAuthorization | None]:
        """Sign a content hash; return (signature, signer_key_hex, kid, authorization)."""
        signature = self._keyring.sign(content_hash)
        signer_key = self._keyring.active_public().hex()
        kid = self._keyring.active_kid
        return signature, signer_key, kid, self._keyring.authorization()


def _grant_from_kwargs(
    *,
    capabilities: list[str] | None,
    budget_usd: float | None,
    expires_in: timedelta | None,
    not_after: datetime | None,
    audience: str,
    max_delegations: int | None,
    issued_at: datetime,
) -> Grant:
    resolved_not_after = not_after
    if resolved_not_after is None and expires_in is not None:
        resolved_not_after = issued_at + expires_in
    return Grant(
        capabilities=list(capabilities or []),
        budget_usd=budget_usd,
        not_after=resolved_not_after,
        audience=audience,
        max_delegations=max_delegations,
    )


# ---------------------------------------------------------------------------
# Delegation & delegation chains
# ---------------------------------------------------------------------------


class DelegationVerification(BaseModel):
    """The (non-raising) outcome of verifying one delegation offline."""

    valid: bool
    hash_ok: bool
    signature_ok: bool
    authority_ok: bool
    not_expired: bool
    reason: str | None = None


class Delegation(BaseModel):
    """A signed grant of bounded authority from one identity to another.

    ``issuer`` delegates the :class:`Grant` to ``subject`` (both DIDs). The
    delegation binds the issuer, subject, grant, issuance time, and any
    ``parent_hash`` (the link it sub-delegates from) into a content hash the issuer
    signs. Because the issuer is a DID, :meth:`verify` resolves the signing key
    offline; when the issuer signed with a rotated key the embedded
    :class:`KeyAuthorization` proves that key descends from the issuer's genesis
    key — so the delegation verifies from the bytes alone, no registry. Compose
    links into a :class:`DelegationChain` to enforce end-to-end attenuation.
    """

    id: str = Field(default_factory=lambda: new_id("deleg"))
    issuer: str
    subject: str
    grant: Grant
    parent_id: str = ""
    parent_hash: str = ""
    issued_at: datetime = Field(default_factory=utcnow)

    content_hash: str = ""
    signature: str = ""
    signer_key: str = ""  # hex of the key that actually signed
    signer_kid: str = ""
    authority: KeyAuthorization | None = None
    audit_id: str | None = None

    # -- hashing & signing --------------------------------------------------

    def facts(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "subject": self.subject,
            "grant": self.grant.facts(),
            "parent_hash": self.parent_hash,
            "issued_at": self.issued_at.isoformat(),
        }

    def compute_hash(self) -> str:
        return stable_hash(self.facts(), length=32)

    def seal(self) -> Delegation:
        self.content_hash = self.compute_hash()
        return self

    def sign(self, identity: AgentIdentity) -> Delegation:
        """Seal and sign this delegation as ``identity`` (which must be the issuer)."""
        if identity.did != self.issuer:
            raise IdentityError(
                f"a delegation is signed by its issuer {self.issuer!r}, not {identity.did!r}",
                details={"delegation_id": self.id, "issuer": self.issuer, "signer": identity.did},
            )
        self.seal()
        signature, signer_key, kid, authority = identity._sign_artifact(self.content_hash)
        self.signature = signature
        self.signer_key = signer_key
        self.signer_kid = kid
        self.authority = authority
        return self

    # -- sub-delegation -----------------------------------------------------

    def delegate(
        self,
        identity: AgentIdentity,
        to: str | AgentIdentity,
        *,
        capabilities: list[str] | None = None,
        budget_usd: float | None = None,
        expires_in: timedelta | None = None,
        not_after: datetime | None = None,
        audience: str | None = None,
        max_delegations: int | None = None,
        grant: Grant | None = None,
        issued_at: datetime | None = None,
    ) -> Delegation:
        """Sub-delegate from this link, signed by ``identity`` (this link's subject).

        The child grant defaults to *inheriting* this grant and narrowing only the
        dimensions named — so a sub-delegation attenuates by construction. ``identity``
        must be the ``subject`` of this delegation (authority flows only to whom it
        was granted). The child's ``parent_hash`` links back to this link so the
        :class:`DelegationChain` is tamper-evident.
        """
        if identity.did != self.subject:
            raise IdentityError(
                f"only the delegate {self.subject!r} can sub-delegate, not {identity.did!r}",
                details={"delegation_id": self.id, "subject": self.subject, "signer": identity.did},
            )
        when = issued_at or utcnow()
        child_grant = grant or _attenuated_grant(
            self.grant,
            capabilities=capabilities,
            budget_usd=budget_usd,
            expires_in=expires_in,
            not_after=not_after,
            audience=audience,
            max_delegations=max_delegations,
            issued_at=when,
        )
        child = Delegation(
            issuer=identity.did,
            subject=to.did if isinstance(to, AgentIdentity) else to,
            grant=child_grant,
            parent_id=self.id,
            parent_hash=self.content_hash or self.compute_hash(),
            issued_at=when,
        )
        return child.sign(identity)

    # -- verification -------------------------------------------------------

    def _verify_signature_binding(self, at: datetime | None) -> bool:
        """The signature checks against a key provably bound to the issuer DID."""
        if not self.signature or not self.signer_key:
            return False
        signer_bytes = _hexbytes(self.signer_key)
        if not ed.verify(
            signer_bytes, self.content_hash.encode("utf-8"), _hexbytes(self.signature)
        ):
            return False
        genesis = public_key_from_did(self.issuer)
        if signer_bytes == genesis:
            return True  # signed with the identity's genesis key — bound by the DID itself
        # Signed with a rotated key: the embedded authority must chain genesis → signer.
        return _verify_authority(self.authority, self.issuer, signer_bytes, at or self.issued_at)

    def verify(self, *, at: datetime | None = None) -> DelegationVerification:
        """Verify this delegation offline: hash, signature binding, and expiry."""
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        sig_ok = hash_ok and self._verify_signature_binding(at)
        # Distinguish a pure-signature failure from an authority-binding failure.
        authority_ok = sig_ok or not bool(self.signature)
        when = at or utcnow()
        not_expired = self.grant.not_after is None or when <= self.grant.not_after
        valid = hash_ok and sig_ok and not_expired
        reason: str | None = None
        if not valid:
            if not hash_ok:
                reason = "content hash does not match the delegation facts"
            elif not sig_ok:
                reason = "signature does not verify against a key bound to the issuer DID"
            elif not not_expired:
                reason = "delegation has expired"
        return DelegationVerification(
            valid=valid,
            hash_ok=hash_ok,
            signature_ok=sig_ok,
            authority_ok=authority_ok,
            not_expired=not_expired,
            reason=reason,
        )

    def audit_details(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "delegation_id": self.id,
                "issuer": self.issuer,
                "subject": self.subject,
                "capabilities": sorted(self.grant.capability_set),
                "budget_usd": self.grant.budget_usd,
                "not_after": self.grant.not_after.isoformat() if self.grant.not_after else None,
                "parent_hash": self.parent_hash,
                "content_hash": self.content_hash,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Delegation:
        return cls.model_validate(data)


def _attenuated_grant(
    parent: Grant,
    *,
    capabilities: list[str] | None,
    budget_usd: float | None,
    expires_in: timedelta | None,
    not_after: datetime | None,
    audience: str | None,
    max_delegations: int | None,
    issued_at: datetime,
) -> Grant:
    """Build a child grant that inherits the parent and narrows only what is named."""
    resolved_caps = capabilities if capabilities is not None else list(parent.capabilities)
    resolved_budget = budget_usd if budget_usd is not None else parent.budget_usd
    resolved_audience = audience if audience is not None else parent.audience
    resolved_not_after = not_after
    if resolved_not_after is None:
        if expires_in is not None:
            resolved_not_after = issued_at + expires_in
        else:
            resolved_not_after = parent.not_after
    if max_delegations is not None:
        resolved_depth = max_delegations
    elif parent.max_delegations is not None:
        resolved_depth = parent.max_delegations - 1
    else:
        resolved_depth = None
    return Grant(
        capabilities=list(resolved_caps),
        budget_usd=resolved_budget,
        not_after=resolved_not_after,
        audience=resolved_audience,
        max_delegations=resolved_depth,
    )


def _verify_authority(
    authority: KeyAuthorization | None, issuer_did: str, signer_key: bytes, at: datetime
) -> bool:
    """Verify a rotation path proves ``signer_key`` descends from the issuer genesis."""
    if authority is None or not authority.path:
        return False
    path = authority.path
    genesis = path[0]
    if genesis.prev_kid:
        return False
    if genesis.public_bytes() != public_key_from_did(issuer_did):
        return False
    for prev, record in zip(path, path[1:], strict=False):
        if record.prev_kid != prev.kid:
            return False
        message = record.rotation_message(issuer_did)
        if not ed.verify(
            prev.public_bytes(), message.encode("utf-8"), _hexbytes(record.rotation_sig)
        ):
            return False
        if prev.revoked_at is not None and record.created_at > prev.revoked_at:
            return False
    last = path[-1]
    if last.public_bytes() != signer_key:
        return False
    return last.active_at(at)


class DelegationChainVerification(BaseModel):
    """The (non-raising) outcome of verifying a delegation chain offline."""

    valid: bool
    links_valid: bool
    linkage_ok: bool
    attenuation_ok: bool
    not_expired: bool
    principal: str | None = None
    subject: str | None = None
    reason: str | None = None


class DelegationChain(BaseModel):
    """An ordered chain of delegations from a principal down to an acting agent.

    The links run root → leaf: ``links[0]`` is issued by the principal, each later
    link issued by the previous link's subject. :meth:`verify` checks the whole
    chain **offline** — every link's signature, the issuer→subject linkage and
    ``parent_hash``, and the core invariant that **each link only attenuates its
    parent's grant** — so an over-reaching sub-delegation is refused from the bytes.
    :meth:`permits` then answers whether the chain authorizes a concrete action,
    reading the most-attenuated (leaf) grant.
    """

    links: list[Delegation] = Field(default_factory=list)

    @property
    def principal(self) -> str | None:
        """The DID at the root of the chain — the ultimate source of authority."""
        return self.links[0].issuer if self.links else None

    @property
    def subject(self) -> str | None:
        """The DID at the leaf — the agent the authority finally rests with."""
        return self.links[-1].subject if self.links else None

    @property
    def effective_grant(self) -> Grant | None:
        """The leaf grant — the floor of authority after every attenuation."""
        return self.links[-1].grant if self.links else None

    def extend(self, delegation: Delegation) -> DelegationChain:
        """Append a sub-delegation link (does not re-verify; call :meth:`verify`)."""
        return DelegationChain(links=[*self.links, delegation])

    def verify(
        self, *, at: datetime | None = None, root_issuer: str | None = None
    ) -> DelegationChainVerification:
        """Verify the chain offline end-to-end (signatures, linkage, attenuation, expiry)."""
        when = at or utcnow()
        if not self.links:
            return DelegationChainVerification(
                valid=False,
                links_valid=False,
                linkage_ok=False,
                attenuation_ok=False,
                not_expired=False,
                reason="empty delegation chain",
            )
        links_valid = True
        linkage_ok = True
        attenuation_ok = True
        not_expired = True
        for index, link in enumerate(self.links):
            result = link.verify(at=when)
            if not result.hash_ok or not result.signature_ok:
                links_valid = False
            if not result.not_expired:
                not_expired = False
            if index == 0:
                if root_issuer is not None and link.issuer != root_issuer:
                    linkage_ok = False
                if link.parent_hash:
                    linkage_ok = False  # a root link sub-delegates from nothing
            else:
                parent = self.links[index - 1]
                if link.issuer != parent.subject:
                    linkage_ok = False
                if link.parent_hash != (parent.content_hash or parent.compute_hash()):
                    linkage_ok = False
                if not link.grant.attenuates(parent.grant):
                    attenuation_ok = False
        valid = links_valid and linkage_ok and attenuation_ok and not_expired
        reason: str | None = None
        if not valid:
            if not links_valid:
                reason = "a link's hash or signature does not verify"
            elif not linkage_ok:
                reason = "the chain linkage is broken (issuer/subject or parent hash mismatch)"
            elif not attenuation_ok:
                reason = "a sub-delegation amplifies its parent's authority"
            else:
                reason = "a delegation in the chain has expired"
        return DelegationChainVerification(
            valid=valid,
            links_valid=links_valid,
            linkage_ok=linkage_ok,
            attenuation_ok=attenuation_ok,
            not_expired=not_expired,
            principal=self.principal,
            subject=self.subject,
            reason=reason,
        )

    def permits(
        self,
        capability: str | None = None,
        *,
        budget_usd: float | None = None,
        at: datetime | None = None,
        audience: str | None = None,
        root_issuer: str | None = None,
    ) -> bool:
        """Whether a *valid* chain authorizes a concrete action at the leaf grant."""
        when = at or utcnow()
        if not self.verify(at=when, root_issuer=root_issuer).valid:
            return False
        grant = self.effective_grant
        if grant is None:
            return False
        return grant.permits(capability, budget_usd=budget_usd, at=when, audience=audience)

    def require_permits(
        self,
        capability: str | None = None,
        *,
        budget_usd: float | None = None,
        at: datetime | None = None,
        audience: str | None = None,
        root_issuer: str | None = None,
    ) -> DelegationChain:
        """Raise :class:`~vincio.core.errors.IdentityError` unless the chain permits the action."""
        if not self.permits(
            capability, budget_usd=budget_usd, at=at, audience=audience, root_issuer=root_issuer
        ):
            verification = self.verify(at=at, root_issuer=root_issuer)
            raise IdentityError(
                "delegation chain does not authorize the action: "
                + (verification.reason or f"grant does not permit {capability!r}"),
                details={
                    "capability": capability,
                    "principal": self.principal,
                    "subject": self.subject,
                    "reason": verification.reason,
                },
            )
        return self

    def audit_details(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "principal": self.principal,
                "subject": self.subject,
                "links": len(self.links),
                "capabilities": sorted(self.effective_grant.capability_set)
                if self.effective_grant
                else [],
            }
        )

    def to_wire(self) -> dict[str, Any]:
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> DelegationChain:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Verifiable credentials
# ---------------------------------------------------------------------------


class CredentialVerification(BaseModel):
    """The (non-raising) outcome of verifying an agent credential offline."""

    valid: bool
    hash_ok: bool
    signature_ok: bool
    not_expired: bool
    reason: str | None = None


class AgentCredential(BaseModel):
    """A signed, verifiable claim an org makes about an agent.

    The verifiable-credential primitive: ``issuer`` (an org DID) asserts ``claims``
    about ``subject`` (an agent DID) — *admitted to capability X*, *operated by org
    Y* — and signs the binding. An importer :meth:`verify`\\ s it **offline** (the
    issuer's key resolves from its DID) and folds it into the existing admission /
    registry path with :meth:`admits`. Content-bound the way every other artifact
    is: a tampered claim or a forged issuer is caught from the bytes.
    """

    id: str = Field(default_factory=lambda: new_id("cred"))
    issuer: str
    subject: str
    claims: dict[str, str] = Field(default_factory=dict)
    issued_at: datetime = Field(default_factory=utcnow)
    not_after: datetime | None = None

    content_hash: str = ""
    signature: str = ""
    signer_key: str = ""
    signer_kid: str = ""
    authority: KeyAuthorization | None = None
    audit_id: str | None = None

    # -- hashing & signing --------------------------------------------------

    def facts(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "subject": self.subject,
            "claims": {str(k): str(v) for k, v in sorted(self.claims.items())},
            "issued_at": self.issued_at.isoformat(),
            "not_after": self.not_after.isoformat() if self.not_after else None,
        }

    def compute_hash(self) -> str:
        return stable_hash(self.facts(), length=32)

    def seal(self) -> AgentCredential:
        self.content_hash = self.compute_hash()
        return self

    def sign(self, identity: AgentIdentity) -> AgentCredential:
        """Seal and sign this credential as ``identity`` (which must be the issuer)."""
        if identity.did != self.issuer:
            raise IdentityError(
                f"a credential is signed by its issuer {self.issuer!r}, not {identity.did!r}",
                details={"credential_id": self.id, "issuer": self.issuer, "signer": identity.did},
            )
        self.seal()
        signature, signer_key, kid, authority = identity._sign_artifact(self.content_hash)
        self.signature = signature
        self.signer_key = signer_key
        self.signer_kid = kid
        self.authority = authority
        return self

    # -- verification -------------------------------------------------------

    def _verify_signature_binding(self, at: datetime | None) -> bool:
        if not self.signature or not self.signer_key:
            return False
        signer_bytes = _hexbytes(self.signer_key)
        if not ed.verify(
            signer_bytes, self.content_hash.encode("utf-8"), _hexbytes(self.signature)
        ):
            return False
        if signer_bytes == public_key_from_did(self.issuer):
            return True
        return _verify_authority(self.authority, self.issuer, signer_bytes, at or self.issued_at)

    def verify(self, *, at: datetime | None = None) -> CredentialVerification:
        """Verify the credential offline: hash, signature binding, and expiry."""
        hash_ok = bool(self.content_hash) and self.content_hash == self.compute_hash()
        sig_ok = hash_ok and self._verify_signature_binding(at)
        when = at or utcnow()
        not_expired = self.not_after is None or when <= self.not_after
        valid = hash_ok and sig_ok and not_expired
        reason: str | None = None
        if not valid:
            if not hash_ok:
                reason = "content hash does not match the credential facts"
            elif not sig_ok:
                reason = "signature does not verify against a key bound to the issuer DID"
            else:
                reason = "credential has expired"
        return CredentialVerification(
            valid=valid,
            hash_ok=hash_ok,
            signature_ok=sig_ok,
            not_expired=not_expired,
            reason=reason,
        )

    def require_valid(self, *, at: datetime | None = None) -> AgentCredential:
        """Verify and raise :class:`~vincio.core.errors.IdentityError` if invalid."""
        result = self.verify(at=at)
        if not result.valid:
            raise IdentityError(
                f"credential {self.id} failed verification: {result.reason}",
                details={"credential_id": self.id, "reason": result.reason},
            )
        return self

    # -- folding into admission / registry ---------------------------------

    @property
    def admitted_capabilities(self) -> list[str]:
        """Capabilities the credential admits the subject to.

        Reads the conventional ``admitted_capability`` (single) and
        ``admitted_capabilities`` (comma-separated) claims, so a credential drops
        into the capability-gated admission / registry path.
        """
        caps: list[str] = []
        single = self.claims.get("admitted_capability")
        if single:
            caps.append(single)
        many = self.claims.get("admitted_capabilities")
        if many:
            caps.extend(part.strip() for part in many.split(",") if part.strip())
        return caps

    def admits(self, capability: str, *, at: datetime | None = None) -> bool:
        """Whether this credential is valid and admits ``subject`` to ``capability``."""
        if not self.verify(at=at).valid:
            return False
        return capability in self.admitted_capabilities

    def audit_details(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "credential_id": self.id,
                "issuer": self.issuer,
                "subject": self.subject,
                "claims": {str(k): str(v) for k, v in self.claims.items()},
                "not_after": self.not_after.isoformat() if self.not_after else None,
                "content_hash": self.content_hash,
            }
        )

    def to_wire(self) -> dict[str, Any]:
        return to_jsonable(self.model_dump(mode="json"))

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> AgentCredential:
        return cls.model_validate(data)
